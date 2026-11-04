"""Exact per-question curriculum exposure, interleaving, and longer budgets."""
from collections import Counter
import pytest
from src.medium_visual_world.curriculum import TASKS, make_blocks


def test_rehearsal_preserves_exact_per_example_exposure():
    rows = [{"template": task} for task in TASKS for _ in range(2500)]
    all_plans = []
    for schedule in ("mixed", "staged", "staged_rehearsal", "reverse"):
        blocks, plan = make_blocks(rows, schedule, 12, 3, 42)
        seen = Counter(i for _, ids in blocks for i in ids)
        assert seen == Counter({i: 15 for i in range(10000)})
        assert plan["total_presentations"] == 150000
        assert plan["counts_by_template"] == {task: 37500 for task in TASKS}
        assert sum(len(ids) // 50 for _, ids in blocks) == 3000
        assert all(len(ids) % 50 == 0 for _, ids in blocks)
        all_plans.append((blocks, plan))
    assert all_plans[0][0][-1][1] == all_plans[2][0][-1][1]


def test_rehearsal_phase_composition_and_interleaving():
    rows = [{"template": task} for task in TASKS for _ in range(2500)]
    blocks, plan = make_blocks(rows, "staged_rehearsal", 12, 3, 42)
    assert [(name, len(ids) // 50) for name, ids in blocks] == [
        ("attributes", 1000), ("conjunction", 650),
        ("binding", 750), ("common_mixed", 600)]
    assert [item["by_template"] for item in plan["phase_task_presentations"]] == [
        {"color": 25000, "shape": 25000},
        {"color": 2500, "shape": 2500, "conjunction": 27500},
        {"color": 2500, "shape": 2500, "conjunction": 2500, "pair": 30000},
        {task: 7500 for task in TASKS}]
    for name, indices in blocks[1:3]:
        earlier = {"color", "shape"} if name == "conjunction" else {"color", "shape", "conjunction"}
        batches_with_rehearsal = sum(
            any(rows[i]["template"] in earlier for i in indices[j:j+50])
            for j in range(0, len(indices), 50))
        assert batches_with_rehearsal > 0.9 * (len(indices) // 50)
    assert blocks == make_blocks(rows, "staged_rehearsal", 12, 3, 42)[0]


def test_invalid_rehearsal_config():
    rows = [{"template": task} for task in TASKS for _ in range(10)]
    with pytest.raises(ValueError):
        make_blocks(rows, "staged_rehearsal", 2, 1, 42)


def test_long_budget_matched_exposure():
    rows = [{"template": task} for task in TASKS for _ in range(10)]
    for schedule in ("mixed", "staged", "staged_rehearsal", "reverse"):
        blocks, plan = make_blocks(rows, schedule, 36, 9, 42, rehearsal_epochs=3)
        seen = Counter(i for _, ids in blocks for i in ids)
        assert set(seen.values()) == {45}
        assert len(seen) == 40
        assert plan["total_presentations"] == 40 * 45
