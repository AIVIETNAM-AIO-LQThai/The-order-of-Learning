from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter
import torch
from experiments.medium_world.compare_learning_schedules import (
    make_blocks, make_scheduler, binding_area
)


def test_equal_exposures():
    rows = [{"template": task} for task in
            ("color", "shape", "conjunction", "pair")
            for _ in range(2500)]
    plans = []
    for schedule in ("mixed", "staged", "reverse"):
        blocks, plan = make_blocks(rows, schedule, 12, 3, 42)
        counts = Counter(i for _, indices in blocks for i in indices)
        assert set(counts.values()) == {15}
        assert len(counts) == 10000
        assert plan["total_presentations"] == 150000
        assert plan["counts_by_template"] == {
            task: 37500 for task in
            ("color", "shape", "conjunction", "pair")}
        assert all(len(indices) % 50 == 0 for _, indices in blocks)
        assert sum(len(indices) // 50 for _, indices in blocks) == 3000
        plans.append((blocks, plan))
    assert [len(b[1]) for b in plans[1][0]] == [60000, 30000, 30000, 30000]
    assert [len(b[1]) for b in plans[2][0]] == [30000, 30000, 60000, 30000]
    assert plans[0][1]["total_presentations"] == plans[1][1]["total_presentations"]
    assert plans[0][1]["counts_by_template"] == plans[2][1]["counts_by_template"]
    assert plans[0][0][-1][1] == plans[1][0][-1][1] == plans[2][0][-1][1]
    assert plans[1][0][0][1] == plans[2][0][2][1]
    assert plans[1][0][1][1] == plans[2][0][1][1]
    assert plans[1][0][2][1] == plans[2][0][0][1]


def test_scheduler_reduces_on_plateau():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = make_scheduler(optimizer, 0.5, 3, 0.001, 1e-6)
    for _ in range(5):
        scheduler.step(0.5)
    assert optimizer.param_groups[0]["lr"] == 5e-5


def test_binding_area():
    history = [
        {"step": 0, "id_dev": {"by_template": {"pair": 0.5}}},
        {"step": 100, "id_dev": {"by_template": {"pair": 0.6}}},
        {"step": 200, "id_dev": {"by_template": {"pair": 0.8}}}
    ]
    assert abs(binding_area(history) - 0.625) < 1e-10


def main():
    test_equal_exposures()
    test_scheduler_reduces_on_plateau()
    test_binding_area()
    print("Plateau curriculum tests passed")


if __name__ == "__main__":
    main()
