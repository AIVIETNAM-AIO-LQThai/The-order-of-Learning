"""Exact-exposure curricula for scaffolded and recurrent visual learning.

All schedules use the same shuffled, per-task question streams and the same
nine-pass final mixed block. Training and analysis remain in notebooks.
"""
from collections import Counter
import random
from src.medium_visual_world.curriculum import TASKS, repeated

SCHEDULES = (
    "scaffolded_spiral",
    "spiral_late_focus",
    "staged_rehearsal",
    "prereq_mixed_pair_focus",
    "mixed",
)

# Values are full task passes. One pass = all 2,500 questions of that task.
# Each task must receive 36 passes before the nine-pass common consolidation.
PHASE_EPOCHS = {
    "scaffolded_spiral": (
        ("attributes", {"color": 28, "shape": 28}),
        ("conjunction_rehearsal", {"color": 3, "shape": 3, "conjunction": 27}),
        ("binding_specialization", {"color": 2, "shape": 2, "conjunction": 3, "pair": 26}),
        ("rebalancing", {"color": 3, "shape": 3, "conjunction": 6, "pair": 10}),
    ),
    "spiral_late_focus": (
        ("attributes", {"color": 28, "shape": 28}),
        ("conjunction_rehearsal", {"color": 3, "shape": 3, "conjunction": 27}),
        ("early_rebalancing", {"color": 3, "shape": 3, "conjunction": 6, "pair": 10}),
        ("late_binding_focus", {"color": 2, "shape": 2, "conjunction": 3, "pair": 26}),
    ),
    "staged_rehearsal": (
        ("attributes", {"color": 30, "shape": 30}),
        ("conjunction", {"color": 3, "shape": 3, "conjunction": 33}),
        ("binding", {"color": 3, "shape": 3, "conjunction": 3, "pair": 36}),
    ),
    "prereq_mixed_pair_focus": (
        ("prereq_mixed", {"color": 36, "shape": 36, "conjunction": 36}),
        ("pair_early", {"pair": 18}),
        ("pair_focus", {"pair": 18}),
    ),
    "mixed": (("mixed", {task: 36 for task in TASKS}),),
}


def make_spiral_blocks(rows: list[dict], schedule: str, seed: int):
    """Return task-phase indices and a manifest for 9,000 matched updates."""
    by_task = {task: [] for task in TASKS}
    for index, row in enumerate(rows):
        by_task[row["template"]].append(index)
    streams = {task: repeated(by_task[task], 36,
                             random.Random(seed + 3103 + 127 * number))
               for number, task in enumerate(TASKS)}
    cursor = {task: 0 for task in TASKS}
    blocks, phases = [], []
    step = 0
    for phase_number, (name, epochs) in enumerate(PHASE_EPOCHS[schedule]):
        chunks = {}
        for task, count in epochs.items():
            n = len(by_task[task]) * count
            chunks[task] = streams[task][cursor[task]:cursor[task] + n]
            cursor[task] += n
        labels = [task for task, chunk in chunks.items() for _ in chunk]
        random.Random(seed + 7127 + phase_number).shuffle(labels)
        used = Counter()
        indices = []
        for task in labels:
            indices.append(chunks[task][used[task]])
            used[task] += 1
        updates = len(indices) // 50
        phases.append({"name": name, "start_step": step,
                       "end_step": step + updates, "presentations": len(indices),
                       "by_template": {task: len(chunks.get(task, [])) for task in TASKS}})
        blocks.append((name, indices))
        step += updates

    common = repeated(list(range(len(rows))), 9, random.Random(seed + 8089))
    phases.append({"name": "common_mixed", "start_step": step,
                   "end_step": step + len(common) // 50, "presentations": len(common),
                   "by_template": {task: 9 * len(by_task[task]) for task in TASKS}})
    blocks.append(("common_mixed", common))
    plan = {"schedule": schedule, "seed": seed, "phases": phases,
            "total_presentations": sum(len(ids) for _, ids in blocks),
            "repetitions_per_example": 45,
            "counts_by_template": {task: 45 * len(by_task[task]) for task in TASKS}}
    return blocks, plan
