"""Analyze saved readout results; full grid required unless explicitly --partial."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--partial",action="store_true",help="Pilot-only descriptive analysis; no five-seed inference")
    p.add_argument("--input",type=Path,default=ROOT/"outputs/medium_world/experiments/architecture_localization/frozen_heads")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/medium_world/experiments/architecture_localization/analysis")
    p.add_argument("--include-random",action="store_true",help="Require and compare five random-visual controls")
    a=p.parse_args()
    from src.medium_visual_world.architecture_localization_analysis import analyze
    ids,edits,pairs,missing=analyze(a.input,full=not a.partial,output=a.output)
    print("Completed checkpoints:",len(ids[["schedule","seed","step"]].drop_duplicates()),"missing:",len(missing))
    print("Analysis directory:",a.output)
    if a.partial:print("PARTIAL / PILOT ONLY: paired intervals with n < 5 are not inferential")
    print(pairs[(pairs.step==9000)&(pairs.metric.isin(("id_conjunction_missing_shape","strict_swap_id_safe")))].to_string(index=False))
    if a.include_random:
        from src.medium_visual_world.architecture_localization_analysis import random_visual_contrasts
        random_dir=a.input.parent/"random_visual"
        check=random_visual_contrasts(ids,edits,random_dir)
        check.to_csv(a.output/"random_visual_contrasts.csv",index=False)
        print("Random-visual negative controls:\n",check.to_string(index=False))

if __name__=="__main__":main()
