"""Train and validate pixel-grounded color-existence recognition.

Run from the project root:
    python -m small_experiments.train_color_grounding --device cuda --epochs 40

Only train.jsonl and id_dev.jsonl are read. No final test, no curriculum claims.
A successful run diagnoses one Level-1 task; it does not validate all VQA levels.
"""
import argparse
import json
import time
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

from src.visual_world.learning import VQADataset, VOCAB, evaluate, set_seed, environment
from src.visual_world.baseline_training import validate, train_epoch

class GroundedColorViT(nn.Module):
    """Patch-level ViT with query-conditioned spatial scoring for EXISTS_COLOR.

    Uses image pixels and question color token only. No scene graphs, bounding boxes,
    object IDs, labels, or other privileged metadata enter the model.
    The decoder is intentionally tailored to the color-existence diagnostic.
    """
    def __init__(self, dim=48, depth=1):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.pos = nn.Parameter(torch.zeros(1, 256, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.prenorm = nn.LayerNorm(dim)
        self.postnorm = nn.LayerNorm(dim)
        self.emb = nn.Embedding(len(VOCAB), dim)
        self.qnorm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x, q):
        if x.shape[-2:] != (64, 64):
            raise ValueError('This diagnostic requires 64x64 images')
        # Query [CLS] EXISTS_COLOR <color> [PAD]..., so q[:,2] selects color.
        p = self.patch(x).flatten(2).transpose(1, 2)
        v = self.prenorm(p) + 0.5 * self.postnorm(self.encoder(p + self.pos))
        z = self.qnorm(self.emb(q[:, 2]))
        scores = torch.einsum('bnd,bd->bn', v, z) / v.shape[-1] ** 0.5
        pooled = torch.stack((scores.amax(dim=1), scores.mean(dim=1)), dim=-1)
        return self.classifier(pooled)


def loader(ds, batch, shuffle=False, seed=42):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed))


def get_color(ds):
    ids = [i for i, row in enumerate(ds.rows) if row['template'] == 'color']
    if not ids:
        raise RuntimeError('No color examples found')
    for i in ids:
        if len(ds.rows[i]['query_tokens']) != 2 or ds.rows[i]['query_tokens'][0] != 'EXISTS_COLOR':
            raise RuntimeError('Unexpected color query grammar')
    return Subset(ds, ids)


@torch.inference_mode()
def paired_check(model, ds, device):
    """Swap positive and negative images for the SAME question on ID-dev.

    If model uses images properly, performance on swapped images *with original
    labels unchanged* should fall. This is a diagnostic intervention only.
    """
    grouped = {}
    for i, row in enumerate(ds.rows):
        if row['template'] == 'color':
            grouped.setdefault((tuple(row['query_tokens']), int(row['label'])), []).append(i)
    pairs = []
    for q in sorted(set(k[0] for k in grouped)):
        a, b = grouped.get((q, 0), []), grouped.get((q, 1), [])
        for i, j in zip(a, b):
            pairs.append((i, j))
    if not pairs:
        raise RuntimeError('No matched opposite-label query pairs')
    x_original, q_original, y_original = [], [], []
    x_swapped = []
    for neg, pos in pairs:
        x0, q0, y0, _ = ds[neg]
        x1, q1, y1, _ = ds[pos]
        assert torch.equal(q0, q1) and y0.item() == 0 and y1.item() == 1
        x_original += [x0, x1]
        x_swapped += [x1, x0]
        q_original += [q0, q1]
        y_original += [y0, y1]
    q = torch.stack(q_original).to(device)
    y = torch.stack(y_original).to(device)
    original = model(torch.stack(x_original).to(device), q).argmax(1)
    swapped = model(torch.stack(x_swapped).to(device), q).argmax(1)
    return {'pairs': len(pairs),
            'original_accuracy': round((original == y).float().mean().item(), 6),
            'swapped_image_accuracy': round((swapped == y).float().mean().item(), 6),
            'changed_prediction_fraction': round((original != swapped).float().mean().item(), 6)}


@torch.inference_mode()
def constant_image_eval(model, subset, device, batch=64):
    """Replace all images with a constant while preserving each query and label."""
    correct, n = 0, 0
    # Constant training-image baseline from dataset class, not from labels.
    blank = torch.tensor([243 / 255, 246 / 255, 244 / 255], dtype=torch.float32).view(3, 1, 1).expand(3, 64, 64)
    for x, q, y, _ in loader(subset, batch):
        x = blank.unsqueeze(0).expand(x.size(0), -1, -1, -1).to(device)
        logits = model(x, q.to(device))
        correct += (logits.argmax(1) == y.to(device)).sum().item()
        n += y.numel()
    return correct / n


def fit(mode, train, dev, device, epochs, batch, lr, seed):
    set_seed(seed)
    model = GroundedColorViT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    tr = loader(train, batch, True, seed)
    tr_eval = loader(train, batch)
    dev_eval = loader(dev, batch)
    history = []
    best_dev = -1
    for epoch in range(1, epochs + 1):
        train_epoch(model, tr, opt, device)
        a = evaluate(model, tr_eval, device)
        b = evaluate(model, dev_eval, device)
        history.append({'epoch': epoch, 'train_accuracy': a['accuracy'], 'train_loss': a['loss'],
                        'id_dev_accuracy': b['accuracy'], 'id_dev_loss': b['loss']})
        best_dev = max(best_dev, b['accuracy'])
        if epoch in (1, 5, 10, 20, 30, epochs) or epoch % 10 == 0:
            print(f'{mode:9} epoch {epoch:3} train={a["accuracy"]:.3f} dev={b["accuracy"]:.3f} loss={a["loss"]:.4f}', flush=True)
    return model, {'best_id_dev_accuracy': best_dev, 'final': history[-1], 'history': history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=PROJECT_ROOT / 'outputs/dataset')
    p.add_argument('--out', type=Path, default=PROJECT_ROOT / 'outputs/experiments/color_grounding/grounding_diagnostic.json')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs <= 0:
        raise ValueError('epochs must be positive')
    validate(args.data)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device(args.device)
    start = time.perf_counter()
    train_full = VQADataset(args.data / 'train.jsonl', levels=(1,))
    dev_full = VQADataset(args.data / 'id_dev.jsonl', levels=(1,))
    train, dev = get_color(train_full), get_color(dev_full)
    report = {'status': 'diagnostic_only', 'task': 'EXISTS_COLOR', 'environment': environment(),
              'architecture': 'ViT patch4 depth1 + query-conditioned max/mean patch pooling',
              'train_n': len(train), 'id_dev_n': len(dev), 'seed': args.seed, 'epochs': args.epochs,
              'lr': args.lr, 'batch_size': args.batch_size, 'runs': {},
              'warning': 'No test or OOD split read; not evidence of developmental curriculum effects.'}
    model, result = fit('full', train, dev, device, args.epochs, args.batch_size, args.lr, args.seed)
    result['constant_image_dev_accuracy'] = constant_image_eval(model, dev, device)
    result['paired_counterfactual'] = paired_check(model, dev_full, device)
    report['runs']['full'] = result
    # To test question-only shortcuts, repeat the SAME architecture with a constant image.
    train_text = get_color(VQADataset(args.data / 'train.jsonl', mode='text-only', levels=(1,)))
    dev_text = get_color(VQADataset(args.data / 'id_dev.jsonl', mode='text-only', levels=(1,)))
    _, result = fit('text-only', train_text, dev_text, device, args.epochs, args.batch_size, args.lr, args.seed)
    report['runs']['text-only'] = result
    report['duration_sec'] = round(time.perf_counter() - start, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print('Saved', args.out)


if __name__ == '__main__':
    main()
