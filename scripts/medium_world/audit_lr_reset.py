"""Scientific schedule and LR-policy audit; run from repository root."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from collections import Counter
import torch
from src.medium_visual_world.learning import TASKS
from src.medium_visual_world.scaffolded_spiral import make_spiral_blocks
from src.medium_visual_world.lr_policies import LRPolicy, effective_phases, evaluation_steps
from src.medium_visual_world.curriculum import make_rehearsal_blocks
from src.medium_visual_world.order_control import make_order_control_blocks

SEEDS = (42,43,44,45,46)
SCHEDULES = ("prereq_mixed_pair_focus","staged_rehearsal","scaffolded_spiral","mixed")
EXPECTED_TRANSITIONS = {
    "prereq_mixed_pair_focus": [5400,7200],
    "staged_rehearsal": [3000,4950,7200],
    "scaffolded_spiral": [2800,4450,6100,7200],
    "mixed": [],
}


def audit():
    path = PROJECT_ROOT / "outputs/medium_world/dataset/train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 10000
    assert Counter(row["template"] for row in rows) == Counter({task:2500 for task in TASKS})
    all_counts=0
    for seed in SEEDS:
        common = None
        for schedule in SCHEDULES:
            blocks, plan = make_spiral_blocks(rows,schedule,seed)
            exposure = Counter(i for _,ids in blocks for i in ids)
            assert len(exposure) == len(rows) and set(exposure.values()) == {45}
            assert sum(len(ids) for _,ids in blocks) == 450000
            assert len(blocks[-1][1]) == 90000
            if common is None: common = blocks[-1][1]
            else: assert blocks[-1][1] == common
            segments = effective_phases(plan["phases"])
            transitions = [part["start_step"] for part in segments[1:]]
            assert transitions == EXPECTED_TRANSITIONS[schedule]
            steps = evaluation_steps(plan["phases"],segments)
            assert set(range(0,9001,100)).issubset(steps)
            assert all(t in steps and t+100 in steps for t in transitions)

            if schedule == "staged_rehearsal":
                old,_ = make_rehearsal_blocks(rows,36,9,seed,rehearsal_epochs=3)
                assert old == blocks, "Staged rehearsal differs from the historical builder"
            if schedule in ("prereq_mixed_pair_focus","mixed"):
                old,_ = make_order_control_blocks(rows,schedule,seed,36,9)
                assert old == blocks, "Baseline schedule differs from historical builder"
            all_counts += 1

            parameter = torch.nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.AdamW([parameter],lr=1e-4)
            global_cosine = LRPolicy("global_cosine",optimizer,segments)
            global_rates = [global_cosine.before_update(k) for k in range(9000)]
            reset_cosine = LRPolicy("reset_cosine",optimizer,segments)
            reset_rates=[]
            # Observations are aligned to all required monitoring/transition points.
            observed_steps=set(steps)
            reset_cosine.observe(0,.7)
            for k in range(9000):
                reset_rates.append(reset_cosine.before_update(k))
                if k+1 in observed_steps:reset_cosine.observe(k+1,.7)
            assert [r["step"] for r in reset_cosine.reset_events] == transitions
            assert abs(global_rates[0]-1e-4) < 1e-12
            assert abs(global_rates[-1]-1e-6) < 1e-12
            if schedule=="mixed":
                assert all(abs(a-b)<1e-12 for a,b in zip(global_rates,reset_rates))
            else:
                for boundary in transitions:
                    expected = 5e-5 if boundary==7200 else 1e-4
                    assert abs(reset_rates[boundary]-expected)<1e-12

            # Plateau monitoring remains global or resets only on real transitions.
            for policy in ("global_plateau","reset_plateau"):
                optimizer=torch.optim.AdamW([parameter],lr=1e-4)
                controller=LRPolicy(policy,optimizer,segments)
                controller.observe(0,.7)
                for k in range(9000):
                    controller.before_update(k)
                    if k+1 in observed_steps:controller.observe(k+1,.7)
                expected = transitions if policy=="reset_plateau" else []
                assert [r["step"] for r in controller.reset_events] == expected
    print(f"PASS: {all_counts} exact-exposure plans (5 seeds × 4 curricula)")
    print("PASS: historical control builders, true transition maps, evaluation grids")
    print("PASS: cosine endpoints, phase resets, plateau reset placement")
    print("PASS: Fully Mixed uses the same LR path for Global and Reset Cosine")

if __name__=="__main__":
    audit()
