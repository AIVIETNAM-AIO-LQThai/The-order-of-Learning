"""Per-example evaluation traces and forgetting/recovery analysis for Medium World.

Keeps the model and dataset unchanged. Uses exact question IDs for alignment.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


@torch.inference_mode()
def evaluate_predictions(model, dataset, device, batch_size=50):
    """Evaluate every question once, returning aggregate metrics and predictions."""
    model.eval()
    n = len(dataset)
    predictions = np.empty(n, dtype=np.uint8)
    probabilities = np.empty(n, dtype=np.float32)
    losses = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        indices = list(range(start, min(start + batch_size, n)))
        x, q, y = dataset.batch(indices, device)
        logits = model(x, q)
        predictions[start:start + len(indices)] = logits.argmax(1).cpu().numpy()
        probabilities[start:start + len(indices)] = logits.softmax(1)[:, 1].cpu().numpy()
        losses[start:start + len(indices)] = F.cross_entropy(logits, y, reduction="none").cpu().numpy()

    labels = np.asarray([int(r["label"]) for r in dataset.rows], dtype=np.uint8)
    tasks = np.asarray([r["template"] for r in dataset.rows], dtype=str)
    accurate = predictions == labels
    per_task, losses_by_task, binary_by_task = {}, {}, {}
    for task in sorted(set(tasks)):
        mask = tasks == task
        per_task[task] = float(accurate[mask].mean())
        losses_by_task[task] = float(losses[mask].mean())
        p, y = predictions[mask], labels[mask]
        tp = int(((p == 1) & (y == 1)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        binary_by_task[task] = {
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "positive_recall": tp / (tp + fn) if tp + fn else None,
            "negative_specificity": tn / (tn + fp) if tn + fp else None,
            "predicted_positive_rate": float((p == 1).mean()),
        }
    summary = {
        "n": n, "accuracy": float(accurate.mean()), "loss": float(losses.mean()),
        "by_template": per_task, "loss_by_template": losses_by_task,
        "binary_by_template": binary_by_task,
    }
    return summary, {"predicted_label": predictions, "probability_yes": probabilities}


class PredictionTrace:
    """Store compact aligned predictions from all evaluation checkpoints of one run."""

    def __init__(self, datasets, metadata=None):
        self.metadata = metadata or {}
        self.data = {}
        for name, dataset in datasets.items():
            self.data[name] = {
                "ids": np.asarray([str(r["id"]) for r in dataset.rows]),
                "tasks": np.asarray([str(r["template"]) for r in dataset.rows]),
                "labels": np.asarray([int(r["label"]) for r in dataset.rows], dtype=np.uint8),
                "steps": [], "predictions": [], "probabilities": [],
            }
            assert len(set(self.data[name]["ids"])) == len(dataset), f"Duplicate IDs in {name}"

    def add(self, split, step, payload):
        block = self.data[split]
        predictions = np.asarray(payload["predicted_label"], dtype=np.uint8)
        probabilities = np.asarray(payload["probability_yes"], dtype=np.float32)
        assert len(predictions) == len(block["labels"]), "Prediction count differs from the dataset"
        assert len(probabilities) == len(predictions)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert not block["steps"] or int(step) > block["steps"][-1], "Steps must increase"
        block["steps"].append(int(step))
        block["predictions"].append(predictions.copy())
        block["probabilities"].append(probabilities.copy())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": np.asarray(1),
                   "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
                   "splits_json": np.asarray(json.dumps(list(self.data)))}
        for name, block in self.data.items():
            if not block["steps"]:
                continue
            payload[f"{name}_ids"] = block["ids"]
            payload[f"{name}_tasks"] = block["tasks"]
            payload[f"{name}_labels"] = block["labels"]
            payload[f"{name}_steps"] = np.asarray(block["steps"], dtype=np.int32)
            payload[f"{name}_predictions"] = np.stack(block["predictions"])
            payload[f"{name}_probabilities"] = np.stack(block["probabilities"])
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        tmp.replace(path)


def load_trace(path):
    """Load only numeric and string arrays; pickle/object arrays are disallowed."""
    with np.load(path, allow_pickle=False) as file:
        if int(file["schema_version"]) != 1:
            raise ValueError("Unsupported trace version")
        out = {"metadata": json.loads(str(file["metadata_json"])), "splits": {}}
        for name in json.loads(str(file["splits_json"])):
            key = f"{name}_steps"
            if key not in file:
                continue
            out["splits"][name] = {
                label: file[f"{name}_{label}"].copy() for label in
                ("ids", "tasks", "labels", "steps", "predictions", "probabilities")
            }
    return out


def step_index(block, step):
    matches = np.flatnonzero(block["steps"] == step)
    if len(matches) != 1:
        raise KeyError(f"Checkpoint {step} missing; available steps: {block['steps'].tolist()}")
    return int(matches[0])


def compare_two(block, before_step, after_step):
    """Return one row per question: retained, forgotten, newly_learned, still_incorrect."""
    a, b = step_index(block, before_step), step_index(block, after_step)
    if before_step >= after_step:
        raise ValueError("before_step must precede after_step")
    labels = block["labels"]
    before, after = block["predictions"][a], block["predictions"][b]
    correct_before, correct_after = before == labels, after == labels
    categories = np.select(
        [correct_before & correct_after,
         correct_before & ~correct_after,
         ~correct_before & correct_after],
        ["retained", "forgotten", "newly_learned"], default="still_incorrect")
    return pd.DataFrame({
        "example_id": block["ids"], "task": block["tasks"], "label": labels,
        "before_step": before_step, "after_step": after_step,
        "before_prediction": before, "after_prediction": after,
        "before_probability_yes": block["probabilities"][a],
        "after_probability_yes": block["probabilities"][b],
        "before_correct": correct_before, "after_correct": correct_after,
        "category": categories,
    })


def compare_three(block, before_step, after_step, final_step):
    """Three-point outcomes; recovered means correct→incorrect→correct."""
    if not before_step < after_step < final_step:
        raise ValueError("Checkpoints must be strictly increasing")
    df = compare_two(block, before_step, after_step)
    last = step_index(block, final_step)
    final = block["predictions"][last]
    df["final_step"] = final_step
    df["final_prediction"] = final
    df["final_probability_yes"] = block["probabilities"][last]
    df["final_correct"] = final == block["labels"]
    df["recovered"] = (df["category"] == "forgotten") & df["final_correct"]
    df["forgotten_unrecovered"] = (df["category"] == "forgotten") & ~df["final_correct"]
    return df


def summarize_changes(df):
    """Summarize per-task changes (plus a row for the entire split)."""
    rows = []
    for task, part in [("all", df), *[(t, x) for t, x in df.groupby("task")]]:
        counts = part["category"].value_counts()
        n = len(part)
        forgotten = int(counts.get("forgotten", 0))
        newly = int(counts.get("newly_learned", 0))
        previous_correct = int(part["before_correct"].sum())
        previous_wrong = n - previous_correct
        row = {
            "task": task, "n": n,
            **{c: int(counts.get(c, 0)) for c in
               ("retained", "forgotten", "newly_learned", "still_incorrect")},
            "accuracy_before": float(part["before_correct"].mean()),
            "accuracy_after": float(part["after_correct"].mean()),
            "forgetting_rate": forgotten / previous_correct if previous_correct else np.nan,
            "newly_learned_rate": newly / previous_wrong if previous_wrong else np.nan,
            "correctness_churn": (forgotten + newly) / n,
        }
        if "recovered" in part:
            recovered = int(part["recovered"].sum())
            row.update({"recovered": recovered,
                        "forgotten_unrecovered": int(part["forgotten_unrecovered"].sum()),
                        "recovery_rate_among_forgotten": recovered / forgotten if forgotten else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def scene_context(df, rows):
    """Join immutable dataset metadata by question ID; no duplicate scene JSON."""
    attributes = []
    for row in rows:
        objs = row["scene"]["objects"]
        attrs = row["question"]["args"]
        attributes.append({"example_id": row["id"], "n_objects": len(objs),
                           "n_colors": len({o["color"] for o in objs}),
                           "n_shapes": len({o["shape"] for o in objs}),
                           "target_color": attrs[0] if row["template"] in ("pair", "conjunction", "color") else "",
                           "target_shape": attrs[-1] if row["template"] in ("pair", "conjunction", "shape") else ""})
    return df.merge(pd.DataFrame(attributes), on="example_id", how="left", validate="one_to_one")
