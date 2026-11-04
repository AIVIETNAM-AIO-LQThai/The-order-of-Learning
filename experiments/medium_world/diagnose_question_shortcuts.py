from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from collections import defaultdict
import torch
from src.medium_visual_world.learning import MediumDataset, AttributeBindingViT


def categorize(row: dict) -> str:
    color, shape = row["question"]["args"]
    objects = row["scene"]["objects"]
    has_color = any(obj["color"] == color for obj in objects)
    has_shape = any(obj["shape"] == shape for obj in objects)
    has_pair = any(obj["color"] == color and obj["shape"] == shape for obj in objects)
    if row["template"] == "conjunction":
        if has_color and has_shape:
            group = "positive_both_present"
        elif not has_color and has_shape:
            group = "negative_missing_color"
        elif has_color and not has_shape:
            group = "negative_missing_shape"
        else:
            group = "negative_both_missing"
        assert int(has_color and has_shape) == row["label"]
        return group
    if row["template"] == "pair":
        if row["label"] == 0 and not (has_color and has_shape):
            raise AssertionError("Pair negative is not hard")
        assert int(has_pair) == row["label"]
        return "positive_pair_present" if has_pair else "negative_hard_pair"
    raise ValueError(row["template"])


def summarize(items: list[tuple[int, int]]) -> dict:
    if not items:
        return {"count": 0, "accuracy": None, "predicted_positive_rate": None}
    return {"count": len(items),
            "accuracy": sum(pred == label for pred, label in items) / len(items),
            "predicted_positive_rate": sum(pred for pred, label in items) / len(items)}


def diagnose(model: AttributeBindingViT, dataset: MediumDataset,
             device: torch.device, batch_size: int = 64) -> dict:
    by_group = defaultdict(list)
    by_target_shape = defaultdict(list)
    color_baseline = defaultdict(list)
    shape_baseline = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            indices = list(range(start, min(start + batch_size, len(dataset))))
            x, q, y = dataset.batch(indices, device)
            predictions = model(x, q).argmax(1).cpu().tolist()
            for index, pred in zip(indices, predictions):
                row = dataset.rows[index]
                if row["template"] not in ("conjunction", "pair"):
                    continue
                label = int(row["label"])
                group = categorize(row)
                shape = row["question"]["args"][1]
                key = row["template"]
                by_group[f"{key}/{group}"].append((pred, label))
                by_target_shape[f"{key}/{shape}"].append((pred, label))
                if key == "conjunction":
                    objects = row["scene"]["objects"]
                    color, shape = row["question"]["args"]
                    color_baseline[key].append((int(any(obj["color"] == color for obj in objects)), label))
                    shape_baseline[key].append((int(any(obj["shape"] == shape for obj in objects)), label))
    return {
        "subgroups": {key: summarize(items) for key, items in sorted(by_group.items())},
        "target_shape": {key: summarize(items) for key, items in sorted(by_target_shape.items())},
        "conjunction_baselines": {
            "color_only": summarize(color_baseline["conjunction"]),
            "shape_only": summarize(shape_baseline["conjunction"])
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "outputs/medium_world/dataset")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/medium_world/experiments/diagnostics/question_shortcuts.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    model = AttributeBindingViT(dim=state["dim"], patch_size=state["patch_size"]).to(device)
    model.load_state_dict(state["model"])
    dev = MediumDataset(args.data / "id_dev.jsonl")
    result = diagnose(model, dev, device)
    result["checkpoint"] = str(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for group, stats in result["subgroups"].items():
        print(f'{group}: n={stats["count"]}, accuracy={stats["accuracy"]:.3f}, '
              f'predicted_positive={stats["predicted_positive_rate"]:.3f}')
    for group, stats in result["conjunction_baselines"].items():
        print(f'{group} conjunction baseline: accuracy={stats["accuracy"]:.3f}')
    print("Saved", args.out)


if __name__ == "__main__":
    main()
