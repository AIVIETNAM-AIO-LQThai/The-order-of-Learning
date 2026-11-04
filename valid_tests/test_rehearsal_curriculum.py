"""Run: python -m pytest valid_tests/test_rehearsal_curriculum.py -q"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from experiments.medium_world.compare_learning_schedules import (
    make_blocks, read_mixed_lr_reference
)
from src.medium_visual_world.learning import TASKS


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
        earlier = {"color", "shape"} if name == "conjunction" else {
            "color", "shape", "conjunction"}
        batches_with_rehearsal = sum(
            any(rows[i]["template"] in earlier for i in indices[j:j+50])
            for j in range(0, len(indices), 50))
        assert batches_with_rehearsal > 0.9 * (len(indices) // 50)
    assert blocks == make_blocks(rows, "staged_rehearsal", 12, 3, 42)[0]


def test_invalid_rehearsal_config():
    rows = [{"template": task} for task in TASKS for _ in range(10)]
    with pytest.raises(ValueError):
        make_blocks(rows, "staged_rehearsal", 2, 1, 42)


def test_paired_mixed_lr_reference(tmp_path: Path):
    reference = {
        "dataset_hashes": {"train.jsonl": "same"},
        "config": {"main_epochs": 12, "common_epochs": 3,
                   "batch_size": 50, "eval_every": 100,
                   "lr": 1e-4, "dim": 48, "patch_size": 4,
                   "augmentation": True},
        "runs": [
            {"seed": 42, "schedule": "mixed", "status": "completed",
             "lr_reductions": []},
            {"seed": 43, "schedule": "mixed", "status": "completed",
             "lr_reductions": [{"step": 1000, "from": 1e-4, "to": 5e-5}]}
        ]
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    args = SimpleNamespace(seeds=[42, 43], main_epochs=12, common_epochs=3,
                           batch_size=50, eval_every=100, lr=1e-4,
                           dim=48, patch_size=4, augment=True,
                           smoke=False, min_lr=1e-6)
    traces = read_mixed_lr_reference(path, {"train.jsonl": "same"}, args)
    assert traces == {42: {}, 43: {1000: 5e-5}}
    with pytest.raises(ValueError, match="hashes"):
        read_mixed_lr_reference(path, {"train.jsonl": "different"}, args)
    args.seeds = [42, 44]
    with pytest.raises(ValueError, match="Missing"):
        read_mixed_lr_reference(path, {"train.jsonl": "same"}, args)
