"""Matched-exposure schedules for the Medium World learning-order control.

Keeps the existing AttributeBindingViT and dataset unchanged.
"""
from collections import Counter
import random
from src.medium_visual_world.curriculum import TASKS, repeated

SCHEDULES = (
    "staged_pair_focus",
    "prereq_mixed_pair_focus",
    "mixed_pair_focus",
    "mixed",
)


def make_order_control_blocks(rows: list[dict], schedule: str, seed: int,
                              main_epochs: int = 36, common_epochs: int = 9):
    """Return [(phase, example_indices)], exact exposure plan.

    A task epoch means presenting each of that task's training questions once.
    Per-task example order is fixed for a seed across all schedules.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule: {schedule}")
    if main_epochs < 2 or main_epochs % 2 or common_epochs < 0:
        raise ValueError("main_epochs must be positive/even; common_epochs >= 0")
    by_task = {task: [] for task in TASKS}
    for index, row in enumerate(rows):
        by_task[row["template"]].append(index)
    if any(len(by_task[task]) == 0 for task in TASKS):
        raise ValueError("All four tasks must be present")

    streams = {task: repeated(by_task[task], main_epochs,
                             random.Random(seed + 3103 + 127 * i))
               for i, task in enumerate(TASKS)}
    cursors = {task: 0 for task in TASKS}
    half = main_epochs // 2
    prerequisites = {task: main_epochs for task in TASKS if task != "pair"}
    if schedule == "staged_pair_focus":
        allocations = [
            ("attributes", {"color": main_epochs, "shape": main_epochs}),
            ("conjunction", {"conjunction": main_epochs}),
            ("pair_early", {"pair": half}),
            ("pair_focus", {"pair": half})]
    elif schedule == "prereq_mixed_pair_focus":
        allocations = [
            ("prereq_mixed", prerequisites),
            ("pair_early", {"pair": half}),
            ("pair_focus", {"pair": half})]
    elif schedule == "mixed_pair_focus":
        allocations = [
            ("mixed_prefix", {**prerequisites, "pair": half}),
            ("pair_focus", {"pair": half})]
    else:
        allocations = [("mixed", {task: main_epochs for task in TASKS})]

    blocks, phases = [], []
    for phase_number, (name, per_task_epochs) in enumerate(allocations):
        chunks = {}
        for task, epochs in per_task_epochs.items():
            count = len(by_task[task]) * epochs
            start = cursors[task]
            chunks[task] = streams[task][start:start + count]
            if len(chunks[task]) != count:
                raise AssertionError(f"Exhausted {task} stream")
            cursors[task] += count
        labels = [task for task in TASKS if task in chunks
                  for _ in chunks[task]]
        random.Random(seed + 7127 + phase_number).shuffle(labels)
        used = Counter()
        ids = []
        for task in labels:
            ids.append(chunks[task][used[task]])
            used[task] += 1
        blocks.append((name, ids))
        phases.append({"name": name, "presentations": len(ids),
                       "by_template": {task: len(chunk)
                                       for task, chunk in chunks.items()}})

    if any(cursors[task] != len(streams[task]) for task in TASKS):
        raise AssertionError("Unconsumed main-phase task examples")
    if common_epochs:
        common = repeated(list(range(len(rows))), common_epochs,
                          random.Random(seed + 8089))
        blocks.append(("common_mixed", common))
        phases.append({"name": "common_mixed", "presentations": len(common),
                       "by_template": {task: len(by_task[task]) * common_epochs
                                       for task in TASKS}})

    exposure = Counter(i for _, ids in blocks for i in ids)
    expected = main_epochs + common_epochs
    if len(exposure) != len(rows) or set(exposure.values()) != {expected}:
        raise AssertionError("Exact per-example exposure check failed")
    plan = {"schedule": schedule, "seed": seed,
            "total_presentations": sum(len(ids) for _, ids in blocks),
            "repetitions_per_example": expected,
            "counts_by_template": {task: len(by_task[task]) * expected
                                   for task in TASKS},
            "phases": phases}
    return blocks, plan
