"""Train patch-grounded shape and color-shape attribute recognition models.

Run from the project root:
    python -m small_experiments.train_attribute_binding --device cuda --task shape --epochs 50
    python -m small_experiments.train_attribute_binding --device cuda --task pair --epochs 60

Reads train + ID-dev ONLY. No test, OOD, or adaptation split is accessed.
The model receives pixels and question tokens, never scene graphs or labels as inputs.
A separate model is trained per task; this is an engineering gate, not curriculum evidence.
"""
import argparse
import json
import time
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.visual_world.learning import VQADataset, VOCAB, evaluate, environment, set_seed
from src.visual_world.baseline_training import train_epoch


class AttributeBindingViT(nn.Module):
    """Small spatial ViT with joint query-conditioned patch scores.

    Level-1 grammar: [CLS] EXISTS_{COLOR,SHAPE,PAIR} <attribute1> [attribute2]...
    The model learns a joint attribute query rather than using a fixed pooling CLS.
    """

    def __init__(self, dim=48, depth=1):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.pos = nn.Parameter(torch.zeros(1, 256, dim))
        block = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, depth, enable_nested_tensor=False)
        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)
        self.emb = nn.Embedding(len(VOCAB), dim, padding_idx=0)
        self.query = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.score_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x, q):
        if x.shape[1:] != (3, 64, 64):
            raise ValueError("Expected 64x64 RGB images")
        # Position 1 encodes the operator; positions 2 and 3 its arguments.
        e = self.emb(q[:, 1:4])
        query = self.query(e.flatten(1))
        p = self.patch(x).flatten(2).transpose(1, 2)
        v = self.pre_norm(p) + 0.5 * self.post_norm(self.encoder(p + self.pos))
        v = self.score_norm(v)
        s1 = torch.einsum("bnd,bd->bn", v, e[:, 1]) / v.shape[-1] ** .5
        s2 = torch.einsum("bnd,bd->bn", v, e[:, 2]) / v.shape[-1] ** .5
        sj = torch.einsum("bnd,bd->bn", v, query) / v.shape[-1] ** .5
        # Same-location conjunction scores are diagnostic candidates. Their
        # presence doesn't assert that relational binding has been learned.
        prod = torch.tanh(s1) * torch.tanh(s2)
        h = torch.stack([s1.amax(1), s1.mean(1),
                         s2.amax(1), s2.mean(1),
                         sj.amax(1), sj.mean(1),
                         prod.amax(1), prod.mean(1)], dim=-1)
        return self.head(h)


def get_task(ds, task):
    indices = [i for i, r in enumerate(ds.rows) if r["template"] == task]
    if not indices:
        raise RuntimeError(f"No {task!r} records")
    expected = {"color": "EXISTS_COLOR", "shape": "EXISTS_SHAPE", "pair": "EXISTS_PAIR"}[task]
    for i in indices:
        if ds.rows[i]["query_tokens"][0] != expected:
            raise RuntimeError("Question grammar mismatch")
    return Subset(ds, indices)


def make_loader(ds, batch=32, shuffle=False, seed=42):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def paired_counterfactual(model, ds, task, device):
    """Swap images across opposite-label examples with IDENTICAL question tokens.

    All original labels are deliberately retained. Expected swapped-label
    accuracy is LOW if the model is sensitive to relevant visual evidence.
    """
    by_query = defaultdict(lambda: {0: [], 1: []})
    for i, r in enumerate(ds.rows):
        if r["template"] == task:
            by_query[tuple(r["query_tokens"])][int(r["label"])].append(i)
    original, swapped, questions, labels = [], [], [], []
    pairs = 0
    for key in sorted(by_query):
        neg, pos = by_query[key][0], by_query[key][1]
        for i, j in zip(neg, pos):
            xi, qi, yi, _ = ds[i]
            xj, qj, yj, _ = ds[j]
            assert torch.equal(qi, qj) and yi.item() == 0 and yj.item() == 1
            original.extend([xi, xj]); swapped.extend([xj, xi])
            questions.extend([qi, qj]); labels.extend([yi, yj]); pairs += 1
    if pairs == 0:
        raise RuntimeError("No exact-question opposite-label counterfactual pairs")
    q, y = torch.stack(questions).to(device), torch.stack(labels).to(device)
    p0 = model(torch.stack(original).to(device), q).argmax(1)
    p1 = model(torch.stack(swapped).to(device), q).argmax(1)
    return {"pairs": pairs,
            "original_accuracy": round((p0 == y).float().mean().item(), 6),
            "swapped_image_accuracy_against_original_labels": round((p1 == y).float().mean().item(), 6),
            "prediction_flip_rate": round((p0 != p1).float().mean().item(), 6)}


def run_model(task, mode, source_train, source_dev, args, device):
    set_seed(args.seed)
    tr = get_task(VQADataset(source_train, mode=mode, levels=(1,)), task)
    dev = get_task(VQADataset(source_dev, mode=mode, levels=(1,)), task)
    mdl = AttributeBindingViT().to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=0.0)
    ld_tr = make_loader(tr, args.batch_size, True, args.seed)
    ld_eval_tr = make_loader(tr, args.batch_size)
    ld_eval_dev = make_loader(dev, args.batch_size)
    history = []
    best = -1
    for epoch in range(1, args.epochs + 1):
        mdl.train()
        train_epoch(mdl, ld_tr, opt, device)
        a = evaluate(mdl, ld_eval_tr, device)
        b = evaluate(mdl, ld_eval_dev, device)
        row = {"epoch": epoch, "train_accuracy": a["accuracy"],
               "train_loss": a["loss"], "id_dev_accuracy": b["accuracy"],
               "id_dev_loss": b["loss"]}
        history.append(row)
        best = max(best, b["accuracy"])
        if epoch in (1, 5, 10, 20, 30, args.epochs) or epoch % 10 == 0:
            print(f"{task}/{mode} epoch {epoch:3} train={a['accuracy']:.3f} "
                  f"dev={b['accuracy']:.3f} loss={a['loss']:.4f}", flush=True)
    res = {"train_n": len(tr), "dev_n": len(dev),
           "best_id_dev_accuracy": best, "final": history[-1], "history": history}
    if mode == "full":
        orig = VQADataset(source_dev, levels=(1,))
        res["paired_counterfactual"] = paired_counterfactual(mdl, orig, task, device)
        blank = get_task(VQADataset(source_dev, mode="text-only", levels=(1,)), task)
        res["constant_image_dev_accuracy"] = evaluate(mdl, make_loader(blank, args.batch_size), device)["accuracy"]
    return res


def validate_dev(root):
    """Verify synthetic dataset generation audit and only train/ID-dev manifests (never read held-out tests)."""
    meta = json.loads((root / "dataset_metadata.json").read_text(encoding="utf8"))
    audit = json.loads((root / "audit_report.json").read_text(encoding="utf8"))
    if not audit.get("pass"):
        raise RuntimeError("synthetic dataset generation audit must pass")
    for name in ("train.jsonl", "id_dev.jsonl"):
        got = sha256((root / name).read_bytes()).hexdigest()
        if got != meta["manifests_sha256"][name]:
            raise RuntimeError(f"Dataset manifest integrity failed: {name}")
    return {name: meta["manifests_sha256"][name] for name in ("train.jsonl", "id_dev.jsonl")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "outputs/dataset")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/experiments/attribute_binding")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--task", choices=("color", "shape", "pair"), required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-control", action="store_true", help="Also train a matched text-only model")
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    hashes = validate_dev(args.data)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device(args.device)
    train_path, dev_path = args.data / "train.jsonl", args.data / "id_dev.jsonl"
    start = time.perf_counter()
    report = {"status": "diagnostic_only", "task": args.task,
              "dataset_sha256": hashes,
              "architecture": "ViT patch4 depth1 + query-conditioned spatial interaction pooling",
              "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
              "batch_size": args.batch_size, "environment": environment(),
              "runs": {}, "warning": "Only train and ID-dev are accessed; not curriculum or OOD evidence"}
    report["runs"]["full"] = run_model(args.task, "full", train_path, dev_path, args, device)
    if args.text_control:
        report["runs"]["text-only"] = run_model(args.task, "text-only", train_path, dev_path, args, device)
    report["duration_sec"] = round(time.perf_counter() - start, 2)
    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / f"{args.task}_diagnostic.json"
    outfile.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print("Saved", outfile)


if __name__ == "__main__":
    main()
