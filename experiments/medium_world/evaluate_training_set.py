from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import torch
from src.medium_visual_world.learning import MediumDataset, AttributeBindingViT, evaluate


def evaluate_checkpoints(checkpoints: Path, data: Path, output: Path,
                         device: torch.device, batch_size: int, limit: int | None):
    files = sorted(checkpoints.glob("*_seed_*_final.pt"))
    if not files:
        raise FileNotFoundError(f"No saved checkpoints in {checkpoints}. Rerun with --save-models.")
    train = MediumDataset(data / "train.jsonl", limit=limit)
    dev = MediumDataset(data / "id_dev.jsonl")
    results = []
    for file in files:
        state = torch.load(file, map_location="cpu", weights_only=True)
        model = AttributeBindingViT(dim=state["dim"], patch_size=state["patch_size"]).to(device)
        model.load_state_dict(state["model"])
        result = {
            "seed": state["seed"], "schedule": state["schedule"],
            "step": state["step"], "train": evaluate(model, train, device, batch_size),
            "id_dev": evaluate(model, dev, device, batch_size)
        }
        results.append(result)
        print(f'{result["schedule"]} seed={result["seed"]}: '
              f'train={result["train"]["accuracy"]:.3f}, '
              f'ID-dev={result["id_dev"]["accuracy"]:.3f}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": results, "train_limit": limit}, indent=2) + "\n", encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default=PROJECT_ROOT / "outputs/medium_world/experiments/learning_order/checkpoints")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "outputs/medium_world/dataset")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/medium_world/experiments/diagnostics/train_vs_dev.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    evaluate_checkpoints(args.checkpoints, args.data, args.out,
                         torch.device(args.device), args.batch_size, args.limit)


if __name__ == "__main__":
    main()
