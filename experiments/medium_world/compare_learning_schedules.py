from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import Counter
from hashlib import sha256
import json
import random
import time
import torch
from torch import nn

from src.medium_visual_world.learning import (
    TASKS, MediumDataset, AttributeBindingViT, augment_images,
    evaluate, environment, set_seed
)

SCHEDULES = ("mixed", "staged", "reverse")


def verify_dataset(directory: Path, include_test: bool = False) -> dict:
    metadata = json.loads((directory / "dataset_metadata.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "audit_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("Medium-world dataset audit has not passed")
    names = ["train.jsonl", "id_dev.jsonl", "ood_dev.jsonl"]
    if include_test:
        names.extend(("id_test.jsonl", "ood_test.jsonl"))
    hashes = {}
    for name in names:
        digest = sha256((directory / name).read_bytes()).hexdigest()
        if metadata["manifests_sha256"][name] != digest:
            raise RuntimeError(f"Dataset manifest differs from its recorded hash: {name}")
        hashes[name] = digest
    return hashes


def repeated(indices: list[int], epochs: int, rng: random.Random) -> list[int]:
    result = []
    for _ in range(epochs):
        order = indices.copy()
        rng.shuffle(order)
        result.extend(order)
    return result


def make_blocks(rows: list[dict], schedule: str, main_epochs: int,
                common_epochs: int, seed: int):
    by_task = {task: [] for task in TASKS}
    for index, row in enumerate(rows):
        by_task[row["template"]].append(index)
    if any(not by_task[task] for task in TASKS):
        raise ValueError("Missing one of the four training tasks")
    all_ids = list(range(len(rows)))
    attributes = by_task["color"] + by_task["shape"]
    groups = {
        "attributes": attributes,
        "conjunction": by_task["conjunction"],
        "binding": by_task["pair"]
    }
    if schedule == "mixed":
        order = [("mixed", all_ids)]
    elif schedule == "staged":
        order = [("attributes", groups["attributes"]),
                 ("conjunction", groups["conjunction"]),
                 ("binding", groups["binding"])]
    elif schedule == "reverse":
        order = [("binding", groups["binding"]),
                 ("conjunction", groups["conjunction"]),
                 ("attributes", groups["attributes"])]
    else:
        raise ValueError(schedule)
    rng = random.Random(seed + 931)
    blocks = [(phase, repeated(indices, main_epochs, rng)) for phase, indices in order]
    if common_epochs:
        blocks.append(("common_mixed", repeated(all_ids, common_epochs,
                                                  random.Random(seed + 8089))))
    exposure = Counter(index for _, indices in blocks for index in indices)
    if len(exposure) != len(rows) or set(exposure.values()) != {main_epochs + common_epochs}:
        raise AssertionError("Each training example must receive identical exposure")
    plan = {"total_presentations": sum(len(indices) for _, indices in blocks),
            "repetitions_per_example": main_epochs + common_epochs,
            "counts_by_template": {
                task: len(indices) * (main_epochs + common_epochs)
                for task, indices in by_task.items()}}
    return blocks, plan


def snapshot(model, train_loss: float | None, dev, ood, device,
             batch_size: int, step: int, phase: str, limit: int | None):
    return {"step": step, "phase": phase, "batch_train_loss": train_loss,
            "id_dev": evaluate(model, dev, device, batch_size, limit),
            "ood_dev": evaluate(model, ood, device, batch_size, limit)}


def run_one(schedule: str, seed: int, train: MediumDataset, dev: MediumDataset,
            ood: MediumDataset, args, device: torch.device) -> dict:
    set_seed(seed)
    dim = 16 if args.smoke else args.dim
    patch_size = 8 if args.smoke else args.patch_size
    model = AttributeBindingViT(dim=dim, patch_size=patch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    blocks, plan = make_blocks(train.rows, schedule, args.main_epochs, args.common_epochs, seed)
    if not args.smoke and any(len(indices) % args.batch_size for _, indices in blocks):
        raise ValueError("Each phase's presentations must divide evenly into full batches")
    plan["planned_updates"] = sum(len(indices) // args.batch_size for _, indices in blocks)
    history = []
    step = 0
    rolling_loss = 0.0
    rolling_batches = 0
    limit = min(64, len(dev), len(ood)) if args.smoke else None
    for phase, indices in blocks:
        steps_in_phase = 0
        for start in range(0, len(indices), args.batch_size):
            if args.smoke and steps_in_phase == 2:
                break
            batch_indices = indices[start:start + args.batch_size]
            if len(batch_indices) != args.batch_size:
                if args.smoke:
                    break
                raise ValueError("Partial optimizer batch")
            x, q, y = train.batch(batch_indices, device)
            model.train()
            if args.augment:
                x = augment_images(x)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x, q), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            steps_in_phase += 1
            rolling_loss += loss.item()
            rolling_batches += 1
            if not args.smoke and step % args.eval_every == 0:
                record = snapshot(model, rolling_loss / rolling_batches, dev, ood,
                                  device, args.batch_size, step, phase, None)
                history.append(record)
                rolling_loss = 0.0
                rolling_batches = 0
                print(f"{schedule} seed={seed} step={step}: "
                      f"ID={record['id_dev']['by_template']} "
                      f"OOD-pair={record['ood_dev']['by_template'].get('pair', 0):.3f}", flush=True)
        if steps_in_phase:
            record = snapshot(model, rolling_loss / rolling_batches if rolling_batches else None,
                              dev, ood, device, args.batch_size, step, phase, limit)
            if not history or history[-1]["step"] != step:
                history.append(record)
            else:
                history[-1] = record
            rolling_loss = 0.0
            rolling_batches = 0
            print(f"{schedule} seed={seed} end={phase} step={step} "
                  f"ID={record['id_dev']['by_template']}", flush=True)
    final = snapshot(model, None, dev, ood, device, args.batch_size, step, "final", limit)
    final["id_dev_constant_image"] = evaluate(
        model, dev, device, args.batch_size, limit, constant_image=True)
    if args.save_models and not args.smoke:
        path = args.out / "checkpoints" / f"{schedule}_seed_{seed}_final.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "seed": seed, "schedule": schedule,
                    "dim": dim, "patch_size": patch_size, "step": step}, path)
    return {"seed": seed, "schedule": schedule,
            "status": "smoke_only" if args.smoke else "completed",
            "steps_run": step, "exposure_plan": plan,
            "history": history, "final": final}


def main():
    parser = argparse.ArgumentParser(description="Medium-world matched-exposure curriculum experiment")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "outputs/medium_world/dataset")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/medium_world/experiments/learning_order")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--schedules", nargs="+", choices=SCHEDULES, default=list(SCHEDULES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--main-epochs", type=int, default=2)
    parser.add_argument("--common-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.main_epochs < 1 or args.common_epochs < 0 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("Invalid epoch, batch-size or evaluation frequency")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    hashes = verify_dataset(args.data)
    train = MediumDataset(args.data / "train.jsonl")
    dev = MediumDataset(args.data / "id_dev.jsonl")
    ood = MediumDataset(args.data / "ood_dev.jsonl")
    device = torch.device(args.device)
    started = time.perf_counter()
    runs = [run_one(schedule, seed, train, dev, ood, args, device)
            for seed in args.seeds for schedule in args.schedules]
    report = {
        "status": "smoke_only" if args.smoke else "exploratory",
        "dataset_hashes": hashes, "environment": environment(),
        "config": {"seeds": args.seeds, "schedules": args.schedules,
                   "main_epochs": args.main_epochs, "common_epochs": args.common_epochs,
                   "batch_size": args.batch_size, "eval_every": args.eval_every,
                   "lr": args.lr, "weight_decay": args.weight_decay,
                   "dim": 16 if args.smoke else args.dim,
                   "patch_size": 8 if args.smoke else args.patch_size,
                   "augmentation": args.augment,
                   "primary_metric": "fixed-final ID-dev pair accuracy",
                   "heldout_metric": "fixed-final OOD-dev pair accuracy"},
        "runs": runs, "duration_sec": time.perf_counter() - started
    }
    args.out.mkdir(parents=True, exist_ok=True)
    name = "smoke.json" if args.smoke else "schedule_comparison.json"
    (args.out / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved", args.out / name)


if __name__ == "__main__":
    main()
