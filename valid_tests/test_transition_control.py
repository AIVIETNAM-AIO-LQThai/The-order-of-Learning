"""Run directly: python -m valid_tests.test_transition_control"""
from collections import Counter
from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medium_visual_world.curriculum import TASKS
from src.medium_visual_world.order_control import make_order_control_blocks
from src.medium_visual_world.transition_control import STRATEGIES, make_transition_blocks


def main():
    rows = [{"template": t} for t in TASKS for _ in range(2500)]
    for seed in (42, 43, 44, 45, 46):
        reference, _ = make_order_control_blocks(rows, "prereq_mixed_pair_focus", seed)
        prefix_ref, suffix_ref, _ = make_transition_blocks(rows, seed, "abrupt")
        assert prefix_ref + suffix_ref == reference, "Abrupt must reproduce original schedule"
        task_sequences = None
        previous_prefix = None
        for strategy in STRATEGIES:
            prefix, suffix, plan = make_transition_blocks(rows, seed, strategy)
            assert prefix == prefix_ref
            assert previous_prefix in (None, prefix)
            previous_prefix = prefix
            assert plan["prefix_step"] == 6300 and plan["total_steps"] == 9000
            full = prefix + suffix
            counts = Counter(i for _, ids in full for i in ids)
            assert len(counts) == 10000 and set(counts.values()) == {45}
            seq = {t: [i for _, ids in suffix for i in ids if rows[i]["template"] == t] for t in TASKS}
            if task_sequences is None:
                task_sequences = seq
            assert seq == task_sequences, "Per-task example order changed across conditions"
            assert plan["phases"][-1]["end_step"] == 9000
            suffix_counts = Counter(rows[i]["template"] for _, ids in suffix for i in ids)
            assert suffix_counts == Counter({"pair":67500, "color":22500, "shape":22500, "conjunction":22500})
            print(seed, strategy, [(p["phase"],p["end_step"]) for p in plan["phases"]])
    print("PASS: shared prefixes, original abrupt baseline, matched task streams/exposure")


if __name__ == "__main__":
    main()
