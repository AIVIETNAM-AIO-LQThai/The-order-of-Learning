"""Run directly: python -m valid_tests.test_smoothness_recency"""
from collections import Counter
from src.medium_visual_world.curriculum import TASKS
from src.medium_visual_world.smoothness_recency import (
    STRATEGIES, PROFILES, PREFIX_STEPS, TOTAL_STEPS,
    profile_stats, make_smoothness_blocks,
)
from src.medium_visual_world.transition_control import make_transition_blocks


def main():
    rows = [{"template": task} for task in TASKS for _ in range(2500)]
    for seed in (42, 43, 44, 45, 46):
        parent, _, _ = make_transition_blocks(rows, seed, "abrupt")
        reference_streams = None
        means = {}
        for strategy in STRATEGIES:
            prefix, suffix, plan = make_smoothness_blocks(rows, seed, strategy)
            assert prefix == parent
            assert len(suffix) == 9 and all(len(ids) == 15000 for _, ids in suffix)
            assert plan["prefix_step"] == PREFIX_STEPS
            assert plan["total_steps"] == TOTAL_STEPS
            assert set(Counter(i for _,ids in prefix+suffix for i in ids).values()) == {45}
            by_task = {t: [i for _, ids in suffix for i in ids if rows[i]["template"] == t]
                       for t in TASKS}
            if reference_streams is None:
                reference_streams = by_task
            assert by_task == reference_streams
            for task in TASKS:
                assert len(by_task[task]) == (67500 if task == "pair" else 22500)
            # Mirror-label construction ensures *exactly* matched temporal first moment,
            # including within-window ordering rather than only expected moment.
            ordered = [i for _, ids in suffix for i in ids]
            means[strategy] = {t: sum(pos for pos, index in enumerate(ordered)
                                      if rows[index]["template"] == t) for t in TASKS}
            stats = profile_stats(PROFILES[strategy])
            assert stats["mean_pair_window"] in (4.0, 6.0)
            assert stats["max_adjacent_percentage_point_jump"] == (10 if "smooth" in strategy else 28)
        assert means["smooth_early"] == means["step_early"]
        assert means["smooth_late"] == means["step_late"]
        for prefix_name in ("early", "late"):
            s, a = PROFILES[f"smooth_{prefix_name}"], PROFILES[f"step_{prefix_name}"]
            assert sum(s) == sum(a) and sum(i*v for i,v in enumerate(s)) == sum(i*v for i,v in enumerate(a))
            assert s[0] == a[0] and s[-1] == a[-1]
        print(f"seed={seed} PASS | exact exposure, task streams, paired recency centroids")
    print("PASS: five seeds, four strategies, shared parent, fixed timing moments")


if __name__ == "__main__":
    main()
