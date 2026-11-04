"""Reusable, exact-exposure curriculum plans for the medium visual world.

Training loops and evaluation live in notebooks. This module only constructs
sample-order schedules, independently of optimizers and devices.

Historical behavior from the staged-rehearsal experiment is preserved for
identical rows, seeds, and per-task epoch counts.
"""
from collections import Counter
import random

TASKS = ("color", "shape", "conjunction", "pair")
SCHEDULES = ("mixed", "staged", "staged_rehearsal", "reverse")

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


