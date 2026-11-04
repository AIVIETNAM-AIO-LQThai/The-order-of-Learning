"""Matched-exposure medium-world learning-order and rehearsal study.

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

SCHEDULES = ("mixed", "staged", "staged_rehearsal", "reverse")


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


def make_rehearsal_blocks(rows: list[dict], main_epochs: int,
                          common_epochs: int, seed: int,
                          rehearsal_epochs: int = 1):
    """Stage later skills with replay, retaining EXACT per-example exposure.

    For main_epochs=12 and rehearsal_epochs=1:
      attributes: color 10 + shape 10 epochs
      conjunction: conjunction 11 + color 1 + shape 1 epochs
      binding: pair 12 + conjunction 1 + color 1 + shape 1 epochs
      common_mixed: every example 3 epochs (when common_epochs=3)

    Each task's 12 main-epoch permutations are made once, then split across
    phases. Example counts, not just coarse task proportions, are matched.
    """
    if rehearsal_epochs < 1 or main_epochs <= 2 * rehearsal_epochs:
        raise ValueError("Requires main_epochs > 2 * rehearsal_epochs >= 2")
    by_task = {task: [] for task in TASKS}
    for index, row in enumerate(rows):
        by_task[row["template"]].append(index)
    if any(not by_task[task] for task in TASKS):
        raise ValueError("Missing a task")
    all_ids = list(range(len(rows)))
    streams = {
        task: repeated(indices, main_epochs,
                       random.Random(seed + 3103 + 127 * i))
        for i, (task, indices) in enumerate(by_task.items())
    }
    cursor = {task: 0 for task in TASKS}
    r = rehearsal_epochs
    allocations = [
        ("attributes", {"color": main_epochs - 2*r,
                        "shape": main_epochs - 2*r}),
        ("conjunction", {"color": r, "shape": r,
                         "conjunction": main_epochs - r}),
        ("binding", {"color": r, "shape": r,
                     "conjunction": r, "pair": main_epochs})
    ]
    blocks = []
    phase_mix = []
    for phase_num, (phase, epochs_by_task) in enumerate(allocations):
        chunks = {}
        for task, epochs in epochs_by_task.items():
            length = len(by_task[task]) * epochs
            start = cursor[task]
            chunks[task] = streams[task][start:start + length]
            cursor[task] += length
        labels = [task for task, ids in chunks.items() for _ in ids]
        # Shuffle *individual examples*, not whole task blocks, so each
        # minibatch regularly revisits past skills.
        random.Random(seed + 7127 + phase_num).shuffle(labels)
        positions = {task: 0 for task in chunks}
        ids = []
        for task in labels:
            ids.append(chunks[task][positions[task]])
            positions[task] += 1
        blocks.append((phase, ids))
        phase_mix.append({"name": phase, "by_template": {
            task: len(chunk) for task, chunk in chunks.items()}})
    if any(cursor[task] != len(streams[task]) for task in TASKS):
        raise AssertionError("Main-phase exposure differs by task")
    if common_epochs:
        common = repeated(all_ids, common_epochs,
                          random.Random(seed + 8089))
        blocks.append(("common_mixed", common))
        phase_mix.append({"name": "common_mixed", "by_template": {
            task: len(indices) * common_epochs
            for task, indices in by_task.items()}})
    exposure = Counter(index for _, ids in blocks for index in ids)
    if len(exposure) != len(rows) or set(exposure.values()) != {main_epochs + common_epochs}:
        raise AssertionError("Rehearsal plan must preserve exact example exposure")
    plan = {
        "total_presentations": sum(len(ids) for _, ids in blocks),
        "repetitions_per_example": main_epochs + common_epochs,
        "counts_by_template": {
            task: len(indices) * (main_epochs + common_epochs)
            for task, indices in by_task.items()},
        "phases": [{"name": phase, "presentations": len(ids)}
                   for phase, ids in blocks],
        "phase_task_presentations": phase_mix,
        "rehearsal_epochs": rehearsal_epochs
    }
    return blocks, plan


def make_blocks(rows: list[dict], schedule: str, main_epochs: int,
                common_epochs: int, seed: int, rehearsal_epochs: int = 1):
    if schedule == "staged_rehearsal":
        return make_rehearsal_blocks(rows, main_epochs, common_epochs,
                                     seed, rehearsal_epochs)
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


def read_mixed_lr_reference(path: Path, hashes: dict, args) -> dict:
    """Freeze the original mixed-baseline plateau LR trace, by paired seed."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("dataset_hashes") != hashes:
        raise ValueError("LR reference and current dataset hashes differ")
    config = report["config"]
    for key, value in (("main_epochs", args.main_epochs),
                       ("common_epochs", args.common_epochs),
                       ("batch_size", args.batch_size),
                       ("eval_every", args.eval_every),
                       ("lr", args.lr)):
        if config.get(key) != value:
            raise ValueError(f"LR reference differs at {key}")
    if not args.smoke:
        for key, value in (("dim", args.dim), ("patch_size", args.patch_size),
                           ("augmentation", args.augment)):
            if config.get(key) != value:
                raise ValueError(f"LR reference differs at {key}")
    traces = {}
    for run in report["runs"]:
        if run["schedule"] != "mixed" or run["status"] != "completed":
            continue
        seed = run["seed"]
        if seed in traces:
            raise ValueError(f"Duplicate mixed LR reference for seed {seed}")
        previous_lr, previous_step, changes = args.lr, 0, {}
        for entry in run["lr_reductions"]:
            step, initial, final = entry["step"], entry["from"], entry["to"]
            if (step <= previous_step or step % args.eval_every or
                abs(initial - previous_lr) > 1e-12 or
                not args.min_lr <= final < initial):
                raise ValueError(f"Invalid LR reference at seed {seed}, step {step}")
            changes[step] = final
            previous_lr, previous_step = final, step
        traces[seed] = changes
    missing = sorted(set(args.seeds) - set(traces))
    if missing:
        raise ValueError(f"Missing mixed LR reference for seeds: {missing}")
    return traces


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
    scheduler = (make_scheduler(optimizer, args.scheduler_factor,
                                args.scheduler_patience,
                                args.scheduler_threshold, args.min_lr)
                 if args.lr_policy == "plateau" else None)
    frozen_changes = (args.lr_traces[seed]
                      if args.lr_policy == "mixed_reference" else {})
    blocks, plan = make_blocks(train.rows, schedule, args.main_epochs,
                               args.common_epochs, seed, args.rehearsal_epochs)
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
            if scheduler is not None:
                scheduler.step(dev_loss)
            elif step in frozen_changes:
                for group in optimizer.param_groups:
                    group["lr"] = frozen_changes[step]
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
    parser.add_argument("--rehearsal-epochs", type=int, default=1,
                        help="Rehearsal epochs per earlier task across later stages")
    parser.add_argument("--lr-policy", choices=("plateau", "frozen", "mixed_reference"),
                        default="plateau")
    parser.add_argument("--lr-reference", type=Path,
                        help="Previous plateau schedule_comparison.json for mixed_reference")
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
        args.scheduler_threshold < 0 or not 0 < args.min_lr <= args.lr or
        args.rehearsal_epochs < 1 or
        ("staged_rehearsal" in args.schedules and
         args.main_epochs <= 2 * args.rehearsal_epochs)):
        parser.error("Invalid training/scheduler configuration")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    hashes = verify_dataset(args.data)
    args.lr_traces = {}
    if args.lr_policy == "mixed_reference":
        if args.lr_reference is None:
            parser.error("--lr-policy mixed_reference requires --lr-reference")
        args.lr_traces = read_mixed_lr_reference(args.lr_reference, hashes, args)
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
            "rehearsal_epochs": args.rehearsal_epochs,
            "lr_policy": args.lr_policy,
            "lr_reference": str(args.lr_reference) if args.lr_reference else None,
            "batch_size": args.batch_size, "eval_every": args.eval_every,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "scheduler": ("ReduceLROnPlateau" if args.lr_policy == "plateau"
                          else "replay_mixed_reference" if args.lr_policy == "mixed_reference"
                          else "constant"),
            "scheduler_factor": args.scheduler_factor,
            "scheduler_patience": args.scheduler_patience,
            "scheduler_threshold": args.scheduler_threshold,
            "scheduler_metric": ("fixed ID-dev overall cross-entropy" if args.lr_policy == "plateau" else None),
            "min_lr": args.min_lr,
            "scheduler_reset": "none (continuous within each run)",
            "lr_control": "all schedules share the frozen mixed reference within each seed"
                          if args.lr_policy == "mixed_reference"
                          else "independent plateau" if args.lr_policy == "plateau"
                          else "constant LR shared across schedules",
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
