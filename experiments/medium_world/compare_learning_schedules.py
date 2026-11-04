"""Matched-exposure medium-world curriculum experiment with plateau LR scheduling.

Run from repository root:
python -m experiments.medium_world.compare_learning_schedules --device cuda --seeds 42 --save-models --out outputs/medium_world/experiments/plateau_seed42

Dataset, model, augmentation, and task labels are unchanged. This file replaces
only experiments/medium_world/compare_learning_schedules.py.
"""
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
    # Stage-local RNG keeps the example order within each named phase
    # identical in staged and reverse; only the stage order changes.
    phase_offsets = {"mixed": 101, "attributes": 201,
                     "conjunction": 301, "binding": 401}
    blocks = [(phase, repeated(indices, main_epochs,
                               random.Random(seed + 931 + phase_offsets[phase])))
              for phase, indices in order]
    if common_epochs:
        blocks.append(("common_mixed", repeated(all_ids, common_epochs,
                                                  random.Random(seed + 8089))))
    exposure = Counter(index for _, indices in blocks for index in indices)
    if len(exposure) != len(rows) or set(exposure.values()) != {main_epochs + common_epochs}:
        raise AssertionError("Every training example must receive equal exposure")
    plan = {
        "total_presentations": sum(len(indices) for _, indices in blocks),
        "repetitions_per_example": main_epochs + common_epochs,
        "counts_by_template": {
            task: len(indices) * (main_epochs + common_epochs)
            for task, indices in by_task.items()
        },
        "phases": [{"name": phase, "presentations": len(indices)}
                   for phase, indices in blocks]
    }
    return blocks, plan


def make_scheduler(optimizer, factor: float, patience: int,
                   threshold: float, min_lr: float):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience,
        threshold=threshold, threshold_mode="abs", min_lr=min_lr
    )


def snapshot(model, train_loss, dev, ood, device,
             batch_size: int, step: int, phase: str, limit=None):
    return {
        "step": step, "phase": phase,
        "batch_train_loss": train_loss,
        "id_dev": evaluate(model, dev, device, batch_size, limit),
        "ood_dev": evaluate(model, ood, device, batch_size, limit)
    }


def checkpoint(path: Path, model, schedule: str, seed: int,
               dim: int, patch_size: int, step: int, metric: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "seed": seed,
        "schedule": schedule, "dim": dim,
        "patch_size": patch_size, "step": step,
        "selection_metric": "id_dev_loss", "selection_value": metric
    }, path)


def binding_area(history: list[dict]) -> float:
    """Trapezoid AULC (accuracy averaged over updates, including step 0)."""
    ordered = sorted(history, key=lambda row: row["step"])
    if len(ordered) < 2 or ordered[0]["step"] != 0:
        raise ValueError("Expected a learning curve with step-0 evaluation")
    score = 0.0
    for previous, current in zip(ordered[:-1], ordered[1:]):
        width = current["step"] - previous["step"]
        if width <= 0:
            raise ValueError("Duplicate or unordered evaluation steps")
        start = previous["id_dev"]["by_template"]["pair"]
        end = current["id_dev"]["by_template"]["pair"]
        score += width * (start + end) / 2
    return score / ordered[-1]["step"]


def run_one(schedule: str, seed: int, train: MediumDataset, dev: MediumDataset,
            ood: MediumDataset, args, device: torch.device) -> dict:
    set_seed(seed)
    dim = 16 if args.smoke else args.dim
    patch_size = 8 if args.smoke else args.patch_size
    model = AttributeBindingViT(dim=dim, patch_size=patch_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.scheduler_factor,
                               args.scheduler_patience,
                               args.scheduler_threshold, args.min_lr)
    blocks, plan = make_blocks(train.rows, schedule, args.main_epochs,
                               args.common_epochs, seed)
    if not args.smoke and any(len(indices) % args.batch_size for _, indices in blocks):
        raise ValueError("Every phase must divide into complete batches")
    plan["planned_updates"] = sum(len(indices) // args.batch_size for _, indices in blocks)
    history = []
    changes = []
    step = 0
    rolling_loss = 0.0
    rolling_batches = 0
    limit = min(64, len(dev), len(ood)) if args.smoke else None
    best_loss = float("inf")
    best_step = None
    best_path = args.out / "checkpoints" / f"{schedule}_seed_{seed}_best_dev.pt"
    final_path = args.out / "checkpoints" / f"{schedule}_seed_{seed}_final.pt"

    def record(phase: str, scheduled: bool):
        nonlocal rolling_loss, rolling_batches, best_loss, best_step
        mean_train = rolling_loss / rolling_batches if rolling_batches else None
        row = snapshot(model, mean_train, dev, ood, device,
                       args.batch_size, step, phase, limit)
        before = optimizer.param_groups[0]["lr"]
        row["lr"] = before
        row["event"] = "interval" if scheduled else "boundary"
        dev_loss = row["id_dev"]["loss"]
        # The checkpoint-selection policy does not depend on the schedule.
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_step = step
            if args.save_models and not args.smoke:
                checkpoint(best_path, model, schedule, seed,
                           dim, patch_size, step, dev_loss)
        # Never step the scheduler at extra phase-boundary evaluations:
        # only identical global evaluation steps may affect learning rate.
        if scheduled and not args.smoke:
            scheduler.step(dev_loss)
        after = optimizer.param_groups[0]["lr"]
        row["lr_after"] = after
        if after < before:
            changes.append({"step": step, "from": before, "to": after})
        history.append(row)
        rolling_loss = 0.0
        rolling_batches = 0
        if not args.smoke:
            pair = row["id_dev"]["by_template"]["pair"]
            print(f"{schedule} seed={seed} step={step} "
                  f"phase={phase} pair={pair:.3f} "
                  f"dev_loss={dev_loss:.4f} lr={after:.2g}", flush=True)

    # Evaluate the common random initialization before any update.
    record("initial", scheduled=False)
    for phase, indices in blocks:
        phase_steps = 0
        for start in range(0, len(indices), args.batch_size):
            if args.smoke and phase_steps == 2:
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
            phase_steps += 1
            rolling_loss += loss.item()
            rolling_batches += 1
            if not args.smoke and step % args.eval_every == 0:
                record(phase, scheduled=True)
        if phase_steps:
            # At non-interval boundaries we log, but do not update the scheduler.
            if history[-1]["step"] != step:
                record(phase, scheduled=False)
    if args.smoke:
        # The smoke mode deliberately executes fewer steps; no scientific claims.
        pass
    final = snapshot(model, None, dev, ood, device,
                     args.batch_size, step, "final", limit)
    final["id_dev_constant_image"] = evaluate(
        model, dev, device, args.batch_size, limit, constant_image=True)
    final["lr"] = optimizer.param_groups[0]["lr"]
    if args.save_models and not args.smoke:
        checkpoint(final_path, model, schedule, seed,
                   dim, patch_size, step, final["id_dev"]["loss"])
    curve = binding_area(history)
    return {
        "seed": seed, "schedule": schedule,
        "status": "smoke_only" if args.smoke else "completed",
        "steps_run": step, "exposure_plan": plan,
        "lr_reductions": changes, "best_id_dev_loss": best_loss,
        "best_id_dev_step": best_step,
        "id_pair_aulc": curve, "history": history, "final": final
    }


def main():
    parser = argparse.ArgumentParser(description="Matched-exposure plateau curriculum")
    parser.add_argument("--data", type=Path,
                        default=PROJECT_ROOT / "outputs/medium_world/dataset")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs/medium_world/experiments/plateau_curriculum")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--schedules", nargs="+", choices=SCHEDULES,
                        default=list(SCHEDULES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--main-epochs", type=int, default=12)
    parser.add_argument("--common-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-threshold", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if (args.main_epochs < 1 or args.common_epochs < 0 or
        args.batch_size < 1 or args.eval_every < 1 or
        args.scheduler_patience < 0 or not 0 < args.scheduler_factor < 1 or
        args.scheduler_threshold < 0 or not 0 < args.min_lr <= args.lr):
        parser.error("Invalid training/scheduler configuration")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    hashes = verify_dataset(args.data)
    train = MediumDataset(args.data / "train.jsonl")
    dev = MediumDataset(args.data / "id_dev.jsonl")
    ood = MediumDataset(args.data / "ood_dev.jsonl")
    device = torch.device(args.device)
    started = time.perf_counter()
    runs = []
    for seed in args.seeds:
        for schedule in args.schedules:
            run = run_one(schedule, seed, train, dev, ood, args, device)
            runs.append(run)
            # Write incrementally so completed runs survive interruptions.
            write_report(args, hashes, runs, started)
    print("Saved", args.out / ("smoke.json" if args.smoke else "schedule_comparison.json"))


def write_report(args, hashes: dict, runs: list[dict], started: float):
    report = {
        "status": "smoke_only" if args.smoke else "exploratory",
        "dataset_hashes": hashes, "environment": environment(),
        "config": {
            "seeds": args.seeds, "schedules": args.schedules,
            "main_epochs": args.main_epochs, "common_epochs": args.common_epochs,
            "batch_size": args.batch_size, "eval_every": args.eval_every,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "scheduler": "ReduceLROnPlateau", "scheduler_factor": args.scheduler_factor,
            "scheduler_patience": args.scheduler_patience,
            "scheduler_threshold": args.scheduler_threshold,
            "scheduler_metric": "fixed ID-dev overall cross-entropy",
            "min_lr": args.min_lr,
            "scheduler_reset": "none (continuous within each run)",
            "dim": 16 if args.smoke else args.dim,
            "patch_size": 8 if args.smoke else args.patch_size,
            "augmentation": args.augment,
            "primary_metric": "fixed-final ID-dev pair accuracy",
            "efficiency_metric": "ID-dev pair AULC (steps 0 to final)",
            "ood_metric": "fixed-final OOD-dev pair accuracy",
            "checkpoint_policy": "best overall ID-dev CE; primary analysis fixed-final"
        },
        "runs": runs, "duration_sec": time.perf_counter() - started
    }
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / ("smoke.json" if args.smoke else "schedule_comparison.json")
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)


if __name__ == "__main__":
    main()
