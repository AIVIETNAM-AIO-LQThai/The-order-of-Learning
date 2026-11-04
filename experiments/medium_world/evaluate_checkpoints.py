from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import torch

from experiments.medium_world.compare_learning_schedules import verify_dataset
from src.medium_visual_world.learning import MediumDataset, AttributeBindingViT, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen medium-world checkpoints on ID/OOD test")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "outputs/medium_world/dataset")
    parser.add_argument("--checkpoints", type=Path,
                        default=PROJECT_ROOT / "outputs/medium_world/experiments/learning_order/checkpoints")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs/medium_world/experiments/learning_order/final_test.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    hashes = verify_dataset(args.data, include_test=True)
    id_test = MediumDataset(args.data / "id_test.jsonl")
    ood_test = MediumDataset(args.data / "ood_test.jsonl")
    checkpoints = sorted(args.checkpoints.glob("*_final.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No final checkpoints found in {args.checkpoints}")
    device = torch.device(args.device)
    runs = []
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = AttributeBindingViT(dim=saved["dim"], patch_size=saved["patch_size"]).to(device)
        model.load_state_dict(saved["model"])
        row = {
            "seed": saved["seed"], "schedule": saved["schedule"],
            "step": saved["step"],
            "id_test": evaluate(model, id_test, device, args.batch_size),
            "ood_test": evaluate(model, ood_test, device, args.batch_size)
        }
        runs.append(row)
        print(f"{row['schedule']} seed={row['seed']} "
              f"ID={row['id_test']['by_template']} "
              f"OOD={row['ood_test']['by_template']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"dataset_hashes": hashes, "runs": runs}, indent=2) + "\n",
                        encoding="utf-8")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
