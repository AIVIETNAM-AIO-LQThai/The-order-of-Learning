"""Compare image augmentation, weight decay, and checkpoint selection for shape/binding generalization.

Requires the shared project package and the train_attribute_binding module.
Accesses ONLY outputs/dataset/train.jsonl and outputs/dataset/id_dev.jsonl.

Examples:
    python -m small_experiments.compare_generalization_controls --task shape --device cuda --variants baseline augment decay --seeds 42
    python -m small_experiments.compare_generalization_controls --task pair --device cuda --variants baseline augment decay --seeds 42
    python -m small_experiments.compare_generalization_controls --task pair --device cuda --variants baseline augment --seeds 42 43 44

Model is unchanged from attribute-binding experiments. 'augment' uses label-preserving horizontal,
vertical and diagonal flips ONLY for level-1 attribute tasks; never use that
augmentation indiscriminately for LEFT/ABOVE/relations, because it changes labels.
This is an engineering diagnostic, NOT a curriculum/generalization claim.
"""
import argparse
import copy
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))
    

import torch
from torch import nn

from small_experiments.train_attribute_binding import AttributeBindingViT, get_task, make_loader, paired_counterfactual, validate_dev
from src.visual_world.learning import VQADataset, evaluate, environment, set_seed


def d4_augment(x: torch.Tensor) -> torch.Tensor:
    """Independent D4 symmetries of square images; preserve color/shape existence."""
    n = x.size(0)
    for dim in (-1, -2):
        mask = torch.rand((n, 1, 1, 1), device=x.device) < 0.5
        x = torch.where(mask, x.flip(dim), x)
    mask = torch.rand((n, 1, 1, 1), device=x.device) < 0.5
    x = torch.where(mask, x.transpose(-1, -2), x)
    return x


def train_one_epoch(model, loader, optimizer, device, augment):
    model.train()
    for x, q, y, _ in loader:
        x, q, y = x.to(device), q.to(device), y.to(device)
        if augment:
            x = d4_augment(x)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x, q), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.inference_mode()
def diagnose_errors(model, loader, device):
    model.eval()
    source = loader.dataset.dataset if hasattr(loader.dataset, 'dataset') else loader.dataset
    buckets = defaultdict(lambda: [0, 0])
    confusion = [[0, 0], [0, 0]]
    errors = []
    for x, q, y, ix in loader:
        logits = model(x.to(device), q.to(device))
        probs = logits.softmax(-1)[:, 1].cpu().tolist()
        predictions = logits.argmax(-1).cpu().tolist()
        for idx, truth, pred, score in zip(ix.tolist(), y.tolist(), predictions, probs):
            r = source.rows[idx]
            query = ' '.join(r['query_tokens'])
            b = buckets[query]
            b[0] += int(truth == pred)
            b[1] += 1
            confusion[truth][pred] += 1
            if truth != pred:
                errors.append({'id': r['id'], 'query': query, 'truth': truth,
                               'pred': pred, 'prob_yes': round(score, 6)})
    return {'confusion_true_rows_pred_cols': confusion,
            'by_query': {k: {'correct': v[0], 'n': v[1], 'accuracy': round(v[0]/v[1], 6)}
                         for k, v in sorted(buckets.items())},
            'errors': errors}


def run_variant(task, variant, seed, args, device):
    set_seed(seed)
    path_tr, path_dev = args.data / 'train.jsonl', args.data / 'id_dev.jsonl'
    # Rows and renderers are unchanged from the original, frozen synthetic dataset generation.
    train_ds = get_task(VQADataset(path_tr, levels=(1,)), task)
    dev_ds = get_task(VQADataset(path_dev, levels=(1,)), task)
    tr = make_loader(train_ds, args.batch_size, True, seed)
    train_eval = make_loader(train_ds, args.batch_size, False, seed)
    dev_eval = make_loader(dev_ds, args.batch_size, False, seed)
    use_aug = variant in ('augment', 'combined')
    wd = 0.05 if variant in ('decay', 'combined') else 0.0
    model = AttributeBindingViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)
    best_loss, best_epoch, best_weights, history = float('inf'), 0, None, []
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, tr, optimizer, device, use_aug)
        tr_metrics = evaluate(model, train_eval, device)
        dev_metrics = evaluate(model, dev_eval, device)
        h = {'epoch': epoch, 'train_accuracy': tr_metrics['accuracy'],
             'train_loss': tr_metrics['loss'], 'id_dev_accuracy': dev_metrics['accuracy'],
             'id_dev_loss': dev_metrics['loss']}
        history.append(h)
        if dev_metrics['loss'] < best_loss:
            best_loss, best_epoch = dev_metrics['loss'], epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch in (1, 10, 20, args.epochs) or epoch % 10 == 0:
            print(f'{task}/{variant}/seed={seed} epoch {epoch:3d}: '
                  f'train={h["train_accuracy"]:.3f} dev={h["id_dev_accuracy"]:.3f} '
                  f'dev_ce={h["id_dev_loss"]:.3f}', flush=True)
    assert best_weights is not None
    model.load_state_dict(best_weights)
    tr_best = evaluate(model, train_eval, device)
    dev_best = evaluate(model, dev_eval, device)
    scene_dev = VQADataset(path_dev, levels=(1,))
    cf = paired_counterfactual(model, scene_dev, task, device)
    const = get_task(VQADataset(path_dev, mode='text-only', levels=(1,)), task)
    const_acc = evaluate(model, make_loader(const, args.batch_size), device)['accuracy']
    breakdown = diagnose_errors(model, dev_eval, device)
    result = {'task': task, 'variant': variant, 'seed': seed,
              'best_epoch_by_id_dev_cross_entropy': best_epoch,
              'best_checkpoint': {'train': tr_best, 'id_dev': dev_best,
                                  'paired_counterfactual': cf,
                                  'constant_image_dev_accuracy': const_acc,
                                  'error_analysis': breakdown},
              'final_epoch': history[-1], 'history': history}
    ckpt = args.out / task / variant / f'seed_{seed}_best.pt'
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_weights, 'task': task, 'variant': variant,
                'seed': seed, 'best_epoch': best_epoch,
                'best_dev_ce': best_loss}, ckpt)
    result['checkpoint'] = str(ckpt)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=PROJECT_ROOT / 'outputs/dataset')
    p.add_argument('--out', type=Path, default=PROJECT_ROOT / 'outputs/experiments/generalization_controls')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--task', choices=('shape', 'pair'), required=True)
    p.add_argument('--variants', nargs='+', choices=('baseline', 'augment', 'decay', 'combined'),
                   default=['baseline', 'augment', 'decay'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.002)
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        p.error('epochs and batch-size must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA was requested but PyTorch cannot find a GPU')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    hashes = validate_dev(args.data)
    t0 = time.perf_counter()
    results = []
    for seed in args.seeds:
        for variant in args.variants:
            results.append(run_variant(args.task, variant, seed, args, torch.device(args.device)))
    by_variant = defaultdict(list)
    for r in results:
        by_variant[r['variant']].append(r['best_checkpoint']['id_dev']['accuracy'])
    summary = {variant: {'n_seeds': len(vals), 'mean_best_dev_accuracy': statistics.mean(vals),
                         'std_best_dev_accuracy': statistics.stdev(vals) if len(vals)>1 else None,
                         'individual': vals} for variant, vals in sorted(by_variant.items())}
    report = {'status': 'engineering_diagnostic_only', 'source': 'attribute-binding experiments',
              'experiment': 'best-checkpoint and inductive-bias evaluation',
              'dataset_hashes': hashes, 'environment': environment(),
              'config': {'task':args.task, 'variants':args.variants, 'seeds':args.seeds,
                         'epochs':args.epochs, 'batch_size':args.batch_size,
                         'lr':args.lr, 'checkpoint_selector':'minimum_id_dev_cross_entropy',
                         'augmentation':'D4 geometric transforms on train images only',
                         'weight_decay_for_decay_or_combined':0.05},
              'summary': summary, 'runs': results,
              'duration_sec':round(time.perf_counter()-t0,2),
              'warning':'No OOD/test/adapt manifests are opened; dev reuse implies exploratory claims only.'}
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / f'{args.task}_generalization.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print('Saved',report_path)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
