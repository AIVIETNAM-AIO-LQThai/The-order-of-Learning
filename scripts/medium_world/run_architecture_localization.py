"""Run frozen readout experiments; pretrained ViT weights are never updated."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot",action="store_true",help="One checkpoint: mixed/seed42/step9000")
    group.add_argument("--full",action="store_true",help="All 30 checkpoints, 2 schedules x 5 seeds x 3 steps")
    group.add_argument("--random-visual",action="store_true",help="Five random visual encoders with matched learned queries")
    p.add_argument("--device",default="cuda",choices=["cuda","cpu"])
    p.add_argument("--out",type=Path,default=None)
    a=p.parse_args()
    from src.medium_visual_world.architecture_localization import run_checkpoint,SCHEDULES,SEEDS,STEPS
    if a.random_visual:
        from src.medium_visual_world.architecture_localization import run_random_visual_control
        for seed in SEEDS:
            result=run_random_visual_control(ROOT,seed,device=a.device,out_dir=a.out)
            print(f"random_visual/seed{seed}: {result['status']} -> {result['target']}",flush=True)
        return
    grid=[(s,seed,step) for s in SCHEDULES for seed in SEEDS for step in STEPS] if a.full else [("mixed",42,9000)]
    for schedule,seed,step in grid:
        result=run_checkpoint(ROOT,schedule,seed,step,device=a.device,out_dir=a.out)
        print(f"{schedule}/seed{seed}/step{step}: {result['status']} -> {result['target']}",flush=True)

if __name__=="__main__":main()
