"""Pair-only task-distribution smoothness x recency control, Medium World.

Four 2,700-update suffixes branch from the *same* step-6,300 parent.
Task quotas and per-task example order match exactly. For either recency
condition, smooth and stepwise have the same weighted-mean presentation time,
and identical first/last 300-update task proportions.
"""
from collections import Counter
import random
from src.medium_visual_world.curriculum import TASKS
from src.medium_visual_world.transition_control import make_transition_blocks

STRATEGIES = ("smooth_early", "step_early", "smooth_late", "step_late")
PREFIX_STEPS, TOTAL_STEPS, BATCH_SIZE = 6300, 9000, 50
WINDOW_STEPS, WINDOW_SAMPLES = 300, 15000
EARLIER = ("color", "shape", "conjunction")

# Integer percentages of Binding in nine consecutive 300-step windows.
# Within each recency level, profiles match total count, temporal centroid,
# first-window quota, and last-window quota. They differ in maximal jump.
EARLY_PROFILES = {
    "smooth": (80, 70, 65, 60, 50, 45, 35, 25, 20),
    "step": (80, 74, 74, 46, 46, 46, 32, 32, 20),
}
PROFILES = {
    "smooth_early": EARLY_PROFILES["smooth"],
    "step_early": EARLY_PROFILES["step"],
    "smooth_late": EARLY_PROFILES["smooth"][::-1],
    "step_late": EARLY_PROFILES["step"][::-1],
}


def quotas_for_window(pair_pct: int) -> dict[str, int]:
    pair = WINDOW_SAMPLES * pair_pct // 100
    rest = WINDOW_SAMPLES - pair
    assert WINDOW_SAMPLES * pair_pct % 100 == 0 and rest % 3 == 0
    return {**{task: rest // 3 for task in EARLIER}, "pair": pair}


def symmetric_labels(quotas: dict[str, int], seed: int) -> list[str]:
    """Mirror sampled task labels, fixing each task's within-window centroid.

    A random half and its reverse give each task exactly the same mean
    sample-position (window center), regardless of allocation or seed.
    """
    assert all(n % 2 == 0 for n in quotas.values())
    half = [task for task in TASKS for _ in range(quotas[task] // 2)]
    random.Random(seed).shuffle(half)
    return half + half[::-1]


def profile_stats(profile) -> dict:
    assert len(profile) == 9 and sum(profile) == 450
    centroid = sum((i+1) * p for i, p in enumerate(profile)) / sum(profile)
    max_jump = max(abs(a-b) for a, b in zip(profile[:-1], profile[1:]))
    roughness = sum((a-b)**2 for a, b in zip(profile[:-1], profile[1:]))
    return {"pair_percentage_by_window": list(profile),
            "mean_pair_window": centroid, "mean_pair_optimizer_step":
                PREFIX_STEPS + (centroid - 0.5) * WINDOW_STEPS,
            "max_adjacent_percentage_point_jump": max_jump,
            "squared_jump_score": roughness}


def make_smoothness_blocks(rows: list[dict], seed: int, strategy: str):
    """Return (shared_prefix, scheduled_suffix, exposure_plan).

    Each of the 10,000 training questions is presented exactly 45 times.
    All strategies use identical within-task example streams in their suffix.
    """
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    assert len(rows) == 10000
    prefix, original_suffix, _ = make_transition_blocks(rows, seed, "abrupt")
    assert sum(len(ids) for _, ids in prefix) == PREFIX_STEPS * BATCH_SIZE
    streams = {task: [] for task in TASKS}
    for _, ids in original_suffix:
        for index in ids:
            streams[rows[index]["template"]].append(index)
    cursor = {task: 0 for task in TASKS}
    suffix, phases = [], []
    for window, pair_pct in enumerate(PROFILES[strategy], start=1):
        quotas = quotas_for_window(pair_pct)
        labels = symmetric_labels(quotas, seed + 18511 + window)
        indices = []
        for task in labels:
            indices.append(streams[task][cursor[task]])
            cursor[task] += 1
        assert len(indices) == WINDOW_SAMPLES
        suffix.append((f"window_{window}", indices))
        phases.append({"phase": f"window_{window}",
                       "start_step": PREFIX_STEPS + (window-1)*WINDOW_STEPS,
                       "end_step": PREFIX_STEPS + window*WINDOW_STEPS,
                       "pair_percentage": pair_pct,
                       "presentations_by_task": quotas})
    assert all(cursor[task] == len(streams[task]) for task in TASKS)
    exposure = Counter(i for _, ids in prefix+suffix for i in ids)
    assert len(exposure) == len(rows) and set(exposure.values()) == {45}
    assert sum(len(ids) for _, ids in prefix+suffix) == TOTAL_STEPS * BATCH_SIZE
    stats = profile_stats(PROFILES[strategy])
    plan = {"strategy": strategy, "seed": seed, "prefix_step": PREFIX_STEPS,
            "total_steps": TOTAL_STEPS, "presentations_per_question": 45,
            "suffix_presentations_by_task": {task: len(streams[task]) for task in TASKS},
            "total_presentations_by_task": {task: 112500 for task in TASKS},
            "profile": stats, "phases": phases}
    return prefix, suffix, plan
