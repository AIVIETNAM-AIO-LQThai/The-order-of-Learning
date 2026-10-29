"""Compare matched-exposure mixed, staged and reverse visual-task training schedules.

Requires the shared project package and the train_attribute_binding module.
Uses ONLY synthetic dataset generation train.jsonl and id_dev.jsonl; no OOD, test, or adaptation reads.

All schedules:
* Start from identical weights within each seed.
* Present each Level-1 train example the same number of times.
* Have exactly matched optimizer update counts, batch sizes, and architecture.
* End with an identical amount of mixed-task rehearsal.

Example:
  python -m small_experiments.compare_learning_schedules --device cuda --seeds 42 --main-epochs 20 --common-epochs 4
  python -m small_experiments.compare_learning_schedules --device cuda --seeds 42 43 44 --main-epochs 20 --common-epochs 4
  python -m small_experiments.compare_learning_schedules --device cpu --smoke

This is an exploratory learnability diagnostic, NOT evidence of developmental
abstraction, compositional generalization, plasticity, or a causal mechanism.
"""
import argparse
import json
import math
import random
import time
from collections import Counter
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
from small_experiments.compare_generalization_controls import d4_augment
from src.visual_world.learning import VQADataset, environment, evaluate, set_seed

TASKS = ('color', 'shape', 'pair')


def make_blocks(rows, schedule: str, epochs: int, common: int, seed: int):
    """Build schedules with identical exposures and identical final rehearsal order."""
    ids = {t: [i for i, r in enumerate(rows) if r['template'] == t] for t in TASKS}
    if any(len(ids[t]) != 200 for t in TASKS):
        raise ValueError(f'Expected 200 samples per Level-1 task, got { {t: len(ids[t]) for t in TASKS} }')
    all_ids = sum((ids[t] for t in TASKS), [])
    simple_ids = ids['color'] + ids['shape']
    rng = random.Random(seed + 931)
    def repeated(pool, reps, generator):
        out = []
        for _ in range(reps):
            shuffled = list(pool)
            generator.shuffle(shuffled)
            out += shuffled
        return out
    if schedule == 'mixed':
        blocks = [('mixed', repeated(all_ids, epochs, rng))]
    elif schedule == 'staged':
        blocks = [('attributes', repeated(simple_ids, epochs, rng)), ('binding', repeated(ids['pair'], epochs, rng))]
    elif schedule == 'reverse':
        blocks = [('binding', repeated(ids['pair'], epochs, rng)), ('attributes', repeated(simple_ids, epochs, rng))]
    else:
        raise ValueError(schedule)
    if common:
        blocks += [('common_mixed', repeated(all_ids, common, random.Random(seed + 8089)))]
    flat = [ix for _, indices in blocks for ix in indices]
    cnt = Counter(flat)
    expected = epochs + common
    if len(cnt) != 600 or set(cnt.values()) != {expected}:
        raise AssertionError('Schedules do not have exactly matched example exposure')
    return blocks, {'examples': len(flat), 'updates': None, 'repetitions_per_example': expected,
                    'counts_by_template': {t: len(ids[t]) * expected for t in TASKS}}


def evaluate_snapshot(model, train_loader, dev_loader, device, step, phase):
    tr = evaluate(model, train_loader, device)
    dv = evaluate(model, dev_loader, device)
    return {'step': step, 'phase': phase,
            'train': {'accuracy': tr['accuracy'], 'loss': tr['loss'], 'by_template': tr['by_template']},
            'id_dev': {'accuracy': dv['accuracy'], 'loss': dv['loss'], 'by_template': dv['by_template']}}


def run_schedule(train_ds, dev_ds, schedule, seed, args, device, smoke=False):
    set_seed(seed)  # identical model initial weights / optimizer initial state per seed
    model = AttributeBindingViT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    # Cache rendered pixels and questions, not scene metadata, on CPU or GPU.
    x, q, y = [], [], []
    for i in range(len(train_ds)):
        xi, qi, yi, _ = train_ds[i]
        x.append(xi); q.append(qi); y.append(yi)
    x = torch.stack(x).to(device)
    q = torch.stack(q).to(device)
    y = torch.stack(y).to(device)
    blocks, exposure = make_blocks(train_ds.rows, schedule, args.main_epochs, args.common_epochs, seed)
    for name, idx in blocks:
        if len(idx) % args.batch_size:
            raise ValueError(f'Phase {name} has {len(idx)} examples, not divisible by batch-size {args.batch_size}. '
                             'Use --main-epochs and --common-epochs divisible by 4 with batch-size 32.')
    exposure['updates'] = sum(len(ids)//args.batch_size for _, ids in blocks)
    tr_eval = make_loader(train_ds, batch=args.batch_size)
    dv_eval = make_loader(dev_ds, batch=args.batch_size)
    history, step = [], 0
    best_loss, best_weights, best_step = float('inf'), None, None
    for phase, indices in blocks:
        for start in range(0, len(indices), args.batch_size):
            idx = torch.tensor(indices[start:start+args.batch_size], dtype=torch.long, device=device)
            xb, qb, yb = x[idx], q[idx], y[idx]
            model.train()
            if args.augment:
                xb = d4_augment(xb)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb, qb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if not smoke and step % args.eval_every == 0:
                snap = evaluate_snapshot(model, tr_eval, dv_eval, device, step, phase)
                history.append(snap)
                if snap['id_dev']['loss'] < best_loss:
                    best_loss, best_step = snap['id_dev']['loss'], step
                    best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f'{schedule}/seed={seed} step={step} {phase}: '
                      f'ID-dev={snap["id_dev"]["by_template"]} loss={snap["id_dev"]["loss"]:.4f}', flush=True)
            if smoke and step >= 3:
                break
        if not smoke:
            snap = evaluate_snapshot(model, tr_eval, dv_eval, device, step, phase)
            if not history or history[-1]['step'] != step:
                history.append(snap)
            if snap['id_dev']['loss'] < best_loss:
                best_loss, best_step = snap['id_dev']['loss'], step
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f'{schedule}/seed={seed} phase_end={phase} step={step} '
                  f'ID-dev={snap["id_dev"]["by_template"]}', flush=True)
        if smoke:
            break
    final = evaluate_snapshot(model, tr_eval, dv_eval, device, step, 'final')
    if smoke:
        return {'schedule': schedule, 'seed': seed, 'status': 'smoke_only',
                'steps_run': step, 'final': final, 'exposure_plan': exposure}
    pair_cf = paired_counterfactual(model, dev_ds, 'pair', device)
    blank_ds = VQADataset(args.data / 'id_dev.jsonl', mode='text-only', levels=(1,))
    const_acc = evaluate(model, make_loader(get_task(blank_ds, 'pair'), batch=args.batch_size), device)['accuracy']
    # A best-by-CE checkpoint is retained for engineering diagnostics only. Primary
    # comparison is always fixed final step, with matched training exposure.
    best_metrics = None
    if best_weights is not None:
        final_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_weights)
        best_metrics = evaluate_snapshot(model, tr_eval, dv_eval, device, best_step, 'best_dev_ce')
        ckpt = args.out / 'checkpoints' / f'{schedule}_seed_{seed}_best.pt'
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': best_weights, 'schedule': schedule, 'seed': seed,
                    'best_step': best_step, 'best_dev_ce': best_loss}, ckpt)
        model.load_state_dict(final_weights)
    return {'schedule': schedule, 'seed': seed, 'status': 'exploratory',
            'steps_run': step, 'exposure_plan': exposure, 'history': history,
            'final': final, 'best_dev_ce': best_metrics,
            'paired_counterfactual': pair_cf,
            'constant_image_pair_accuracy': const_acc,
            'best_checkpoint': str(ckpt) if best_weights is not None else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=PROJECT_ROOT / 'outputs/dataset')
    ap.add_argument('--out', type=Path, default=PROJECT_ROOT / 'outputs/experiments/learning_order')
    ap.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    ap.add_argument('--schedules', nargs='+', choices=('mixed','staged','reverse'), default=['mixed','staged','reverse'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--main-epochs', type=int, default=20)
    ap.add_argument('--common-epochs', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.002)
    ap.add_argument('--eval-every', type=int, default=75)
    ap.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but not available')
    if args.main_epochs <= 0 or args.common_epochs < 0 or args.batch_size <= 0 or args.eval_every <= 0:
        ap.error('Expected positive main epochs, batch-size and eval frequency; common epochs may be zero')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    hashes = validate_dev(args.data)
    train_ds = VQADataset(args.data / 'train.jsonl', levels=(1,))
    dev_ds = VQADataset(args.data / 'id_dev.jsonl', levels=(1,))
    assert len(train_ds)==600 and len(dev_ds)==120
    started = time.perf_counter()
    runs = [run_schedule(train_ds, dev_ds, schedule, seed, args, torch.device(args.device), args.smoke)
            for seed in args.seeds for schedule in args.schedules]
    report = {'status': 'smoke_only' if args.smoke else 'exploratory_learnability_only',
              'dataset_hashes': hashes, 'environment': environment(),
              'config': {'schedules': args.schedules, 'seeds': args.seeds,
                         'main_epochs': args.main_epochs, 'common_epochs': args.common_epochs,
                         'batch_size': args.batch_size, 'lr': args.lr,
                         'augmentation': args.augment, 'checkpoint_selector': 'min ID-dev CE, diagnostic only',
                         'primary_metric': 'fixed-final ID-dev by-template accuracy'},
              'runs': runs, 'duration_sec': round(time.perf_counter()-started, 2),
              'warning': 'Dev set is reused; one/few seeds cannot support research conclusions. '
                         'No OOD/test/adapt splits accessed.'}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ('smoke.json' if args.smoke else 'schedule_comparison.json')
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('Saved:',path)
    for r in runs:
        print(f'{r["schedule"]} seed={r["seed"]}: {r["final"]["id_dev"]["by_template"]}', flush=True)

if __name__ == '__main__':
    main()
