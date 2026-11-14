"""Audit the frozen architecture-localization experiment without training heads.

Run from repository root:
  python scripts/medium_world/audit_architecture_localization.py --pilot
  python scripts/medium_world/audit_architecture_localization.py --full
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full",action="store_true",help="Verify all 30 pretrained checkpoints")
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    import torch
    from src.medium_visual_world.architecture_localization import (SCHEDULES,SEEDS,STEPS,
        audit_param_budget,make_controlled_edits,cache_dataset,audit_feature_parity)
    from src.medium_visual_world.final_diagnostic_followup import validate_frozen_datasets
    from src.medium_visual_world.diagnostic_followup import locate_checkpoint,load_frozen_model
    from src.medium_visual_world.learning import MediumDataset
    hashes=validate_frozen_datasets(ROOT)
    params=audit_param_budget()
    print("Readout parameter counts:",params)
    dataset=MediumDataset(ROOT/"outputs/medium_world/dataset/id_dev.jsonl")
    edits=make_controlled_edits(dataset.rows)
    print("Frozen data hashes verified:",sorted(hashes))
    print("ID-dev rows:",len(dataset.rows),"controlled edit endpoints:",len(edits))
    if len(dataset.rows)!=1024:raise AssertionError("Expected 1024 ID-dev questions")
    manifest=ROOT/"outputs/medium_world/experiments/representation_diagnostics/five_seeds/checkpoint_manifest.csv"
    grid=[(s,seed,step) for s in SCHEDULES for seed in SEEDS for step in STEPS] if a.full else [("mixed",42,9000)]
    for schedule,seed,step in grid:
        path=locate_checkpoint(ROOT,schedule,seed,step,manifest,verify_sha=True)
        print("Verified:",path.name)
    schedule,seed,step=grid[0]
    model,_=load_frozen_model(locate_checkpoint(ROOT,schedule,seed,step,manifest,verify_sha=True),a.device)
    sample=MediumDataset(ROOT/"outputs/medium_world/dataset/id_dev.jsonl",limit=16)
    cache=cache_dataset(model,sample,torch.device(a.device),batch_size=8)
    check=audit_feature_parity(model,sample,cache,torch.device(a.device),tol=0.02)
    print("Exact-original-output parity (float16 cache approximation):",check)
    print("PREFLIGHT PASSED. No decoder training or saved results.")

if __name__=="__main__":main()
