"""Exact-exposure suffix schedules branched from one shared step-6300 checkpoint.

The parent is the existing prereq_mixed_pair_focus curriculum. This module
reorders ONLY its 6300–9000 suffix and does not modify the model or dataset.
"""
from collections import Counter
import random
from src.medium_visual_world.curriculum import TASKS
from src.medium_visual_world.order_control import make_order_control_blocks

SOURCE = "prereq_mixed_pair_focus"
STRATEGIES = ("abrupt", "gradual_bridge", "late_binding")
PREFIX_STEPS, TOTAL_STEPS, BATCH_SIZE = 6300, 9000, 50
EARLIER = ("color", "shape", "conjunction")

# Epochs per task (one epoch = each question of that task once).
# Every strategy uses 27 Binding epochs and 9 epochs of each earlier task.
SUFFIX_ALLOCATIONS = {
    "abrupt": [
        ("pair_focus", {"pair": 18}),
        ("common_mixed", {"color": 9, "shape": 9, "conjunction": 9, "pair": 9})],
    "gradual_bridge": [
        ("pair_initial", {"pair": 6}),
        ("bridge", {"color": 1, "shape": 1, "conjunction": 1, "pair": 9}),
        ("consolidation", {"color": 8, "shape": 8, "conjunction": 8, "pair": 12})],
    "late_binding": [
        ("pair_initial", {"pair": 6}),
        ("early_mix", {"color": 3, "shape": 3, "conjunction": 3, "pair": 3}),
        ("late_pair", {"color": 6, "shape": 6, "conjunction": 6, "pair": 18})],
}


def make_transition_blocks(rows: list[dict], seed: int, strategy: str,
                           main_epochs: int = 36, common_epochs: int = 9):
    """Return (shared_prefix_blocks, suffix_blocks, schedule_plan).

    Per-task ordering during the suffix is identical across strategies; only
    its temporal arrangement changes. The abrupt condition uses the original
    suffix *exactly*, including common-mixed interleaving.
    """
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    if main_epochs != 36 or common_epochs != 9:
        raise ValueError("Frozen protocol requires 36 main and 9 common epochs")
    ref, _ = make_order_control_blocks(rows, SOURCE, seed, main_epochs, common_epochs)
    assert [name for name, _ in ref] == ["prereq_mixed", "pair_early", "pair_focus", "common_mixed"]
    prefix = ref[:2]
    if sum(len(ids) for _, ids in prefix) != PREFIX_STEPS * BATCH_SIZE:
        raise AssertionError("Prefix length changed")
    task_ids = {task: [] for task in TASKS}
    for _, ids in ref[2:]:
        for index in ids:
            task_ids[rows[index]["template"]].append(index)
    expected = {task: (27 if task == "pair" else 9) * 2500 for task in TASKS}
    if {task: len(ids) for task, ids in task_ids.items()} != expected:
        raise AssertionError("Suffix exposure does not match frozen task sizes")
    cursors = {task: 0 for task in TASKS}
    suffix = []
    for phase_idx, (name, allocation) in enumerate(SUFFIX_ALLOCATIONS[strategy]):
        if strategy == "abrupt":
            indices = ref[2 + phase_idx][1].copy()
            for task, epochs in allocation.items():
                cursors[task] += 2500 * epochs
        else:
            chunks = {}
            for task, epochs in allocation.items():
                start, count = cursors[task], epochs * 2500
                chunks[task] = task_ids[task][start:start + count]
                assert len(chunks[task]) == count
                cursors[task] += count
            labels = [task for task in TASKS if task in chunks for _ in chunks[task]]
            random.Random(seed + 8309 + 43 * phase_idx).shuffle(labels)
            used = Counter()
            indices = []
            for task in labels:
                indices.append(chunks[task][used[task]])
                used[task] += 1
        if len(indices) % BATCH_SIZE:
            raise AssertionError("A phase must contain complete training batches")
        suffix.append((name, indices))
    assert all(cursors[task] == len(task_ids[task]) for task in TASKS)
    full = prefix + suffix
    counts = Counter(i for _, ids in full for i in ids)
    if len(counts) != len(rows) or set(counts.values()) != {45}:
        raise AssertionError("Each training question must be presented 45 times")
    assert sum(len(ids) for _, ids in full) == TOTAL_STEPS * BATCH_SIZE
    boundaries, step = [], 0
    for phase, ids in full:
        end = step + len(ids) // BATCH_SIZE
        boundaries.append({"phase": phase, "start_step": step, "end_step": end,
                           "presentations_by_task": dict(Counter(rows[i]["template"] for i in ids))})
        step = end
    assert step == TOTAL_STEPS and boundaries[1]["end_step"] == PREFIX_STEPS
    plan = {"source": SOURCE, "strategy": strategy, "seed": seed,
            "prefix_step": PREFIX_STEPS, "total_steps": TOTAL_STEPS,
            "presentations_per_question": 45,
            "total_presentations_by_task": dict(Counter(rows[i]["template"] for _, ids in full for i in ids)),
            "phases": boundaries}
    return prefix, suffix, plan
