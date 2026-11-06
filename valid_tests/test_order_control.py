"""Run directly: python -m valid_tests.test_order_control"""
from collections import Counter
from src.medium_visual_world.order_control import (
    SCHEDULES, make_order_control_blocks)


def main():
    rows = [{"template": task, "id": f"{task}:{i}"}
            for task in ("color", "shape", "conjunction", "pair")
            for i in range(7)]
    results = {}
    for schedule in SCHEDULES:
        blocks, plan = make_order_control_blocks(rows, schedule, 42)
        repeat, repeat_plan = make_order_control_blocks(rows, schedule, 42)
        assert blocks == repeat and plan == repeat_plan
        assert plan["repetitions_per_example"] == 45
        assert plan["total_presentations"] == len(rows) * 45
        assert all(n == 7 * 45 for n in plan["counts_by_template"].values())
        assert len([i for _, ids in blocks for i in ids]) == len(rows) * 45
        assert set(Counter(i for _, ids in blocks for i in ids).values()) == {45}
        results[schedule] = blocks
    focus = [dict(results[s])["pair_focus"] for s in SCHEDULES[:3]]
    assert focus[0] == focus[1] == focus[2]
    common = [dict(results[s])["common_mixed"] for s in SCHEDULES]
    assert common[0] == common[1] == common[2] == common[3]
    steps = {s: [len(ids) // 50 for _, ids in blocks]
             for s, blocks in results.items()}
    assert dict(results["staged_pair_focus"])["pair_early"] == dict(results["prereq_mixed_pair_focus"])["pair_early"]
    try:
        make_order_control_blocks(rows, "invalid", 42)
        raise AssertionError("Invalid schedule not rejected")
    except ValueError:
        pass
    print("PASS: deterministic schedule, exact exposure, common consolidation, and identical late Pair block")
    print("Toy stream (not 10k questions):", steps)


if __name__ == "__main__":
    main()
