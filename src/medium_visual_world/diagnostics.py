"""Reusable error categories and per-group performance for visual binding."""
from collections import defaultdict


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


def diagnose(model, dataset, device, batch_size: int = 64) -> dict:
    """Return binding/conjunction subgroup metrics; never train or checkpoint."""
    import torch
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
            for index, prediction in zip(indices, predictions):
                row = dataset.rows[index]
                if row["template"] not in ("conjunction", "pair"):
                    continue
                label = int(row["label"])
                group = categorize(row)
                shape = row["question"]["args"][1]
                task = row["template"]
                by_group[f"{task}/{group}"].append((prediction, label))
                by_target_shape[f"{task}/{shape}"].append((prediction, label))
                if task == "conjunction":
                    objects = row["scene"]["objects"]
                    color, shape = row["question"]["args"]
                    color_baseline[task].append((int(any(obj["color"] == color for obj in objects)), label))
                    shape_baseline[task].append((int(any(obj["shape"] == shape for obj in objects)), label))
    return {
        "subgroups": {key: summarize(items) for key, items in sorted(by_group.items())},
        "target_shape": {key: summarize(items) for key, items in sorted(by_target_shape.items())},
        "conjunction_baselines": {
            "color_only": summarize(color_baseline["conjunction"]),
            "shape_only": summarize(shape_baseline["conjunction"])
        }
    }
