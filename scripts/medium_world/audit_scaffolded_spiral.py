"""Audit the frozen Medium World question schedule without training."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from collections import Counter
from src.medium_visual_world.learning import TASKS
from src.medium_visual_world.curriculum import make_rehearsal_blocks
from src.medium_visual_world.order_control import make_order_control_blocks
from src.medium_visual_world.scaffolded_spiral import SCHEDULES, make_spiral_blocks

rows = [json.loads(line) for line in (PROJECT_ROOT / "outputs/medium_world/dataset/train.jsonl").read_text(encoding="utf-8").splitlines()]
for seed in (42, 43, 44, 45, 46):
    reference_streams = reference_common = None
    spiral_prefix = None
    for schedule in SCHEDULES:
        blocks, plan = make_spiral_blocks(rows, schedule, seed)
        streams = {task: [i for _, ids in blocks[:-1] for i in ids if rows[i]["template"] == task] for task in TASKS}
        counts = Counter(i for _, ids in blocks for i in ids)
        assert len(counts) == len(rows) and set(counts.values()) == {45}
        assert plan["total_presentations"] == 450000
        assert plan["phases"][-1]["end_step"] == 9000
        assert all(len(ids) % 50 == 0 for _, ids in blocks)
        assert reference_streams is None or streams == reference_streams
        assert reference_common is None or blocks[-1][1] == reference_common
        reference_streams, reference_common = streams, blocks[-1][1]
        if schedule == "scaffolded_spiral":
            spiral_prefix = blocks[:2]
        if schedule == "spiral_late_focus":
            assert blocks[:2] == spiral_prefix, "Spiral variants must share the exact prefix"
        if schedule == "staged_rehearsal":
            historical, _ = make_rehearsal_blocks(rows, 36, 9, seed, 3)
            assert blocks == historical, "Staged rehearsal differs from the existing reference"
        if schedule in ("prereq_mixed_pair_focus", "mixed"):
            historical, _ = make_order_control_blocks(rows, schedule, seed, 36, 9)
            assert blocks == historical, "Reference curriculum differs from existing repo"
        if seed == 42:
            print(schedule, [(p["name"], p["start_step"], p["end_step"]) for p in plan["phases"]])
print("PASS: all 5 seeds, 5 schedules, identical per-task streams, 45 presentations per question")
