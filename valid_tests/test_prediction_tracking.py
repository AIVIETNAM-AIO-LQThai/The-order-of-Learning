"""Run directly: python valid_tests/test_prediction_tracking.py"""
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medium_visual_world.prediction_tracking import (
    PredictionTrace, compare_two, compare_three, evaluate_predictions,
    load_trace, scene_context, summarize_changes,
)


class ToyDataset:
    def __init__(self, n=100):
        self.rows = [
            {"id": f"q{i}", "label": 1 if i % 2 else 0,
             "template": "color" if i < 50 else "pair",
             "question": {"args": ["blue", "circle"]},
             "scene": {"objects": [{"color": "blue", "shape": "square"}]}}
            for i in range(n)
        ]
        self.tasks = [r["template"] for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def batch(self, ids, device):
        vals = torch.tensor([self.rows[i]["label"] for i in ids], device=device)
        x = vals.float().view(-1, 1)
        return x, vals, vals


class ToyModel(torch.nn.Module):
    def forward(self, x, q):
        return torch.stack((1 - x[:, 0], x[:, 0]), dim=1) * 4


def main():
    toy = ToyDataset()
    metric, pred = evaluate_predictions(ToyModel(), toy, torch.device("cpu"), batch_size=17)
    assert metric["accuracy"] == 1.0 and metric["by_template"]["pair"] == 1.0
    assert metric["binary_by_template"]["pair"]["tp"] == 25
    assert metric["binary_by_template"]["color"]["tn"] == 25

    labels = np.array([r["label"] for r in toy.rows], dtype=np.uint8)
    first = labels.copy()
    first[80:] = 1 - labels[80:]   # 80 correct, 20 incorrect
    second = first.copy()
    second[0:15] = 1 - second[0:15]   # forget 15
    second[80:95] = 1 - second[80:95]  # learn 15
    final = second.copy()
    final[0:12] = 1 - final[0:12]  # recover 12 of the forgotten 15

    trace = PredictionTrace({"id_dev": toy}, {"seed": 42, "schedule": "toy"})
    for step, labels_pred in [(100, first), (200, second), (300, final)]:
        trace.add("id_dev", step, {"predicted_label": labels_pred,
                                  "probability_yes": np.where(labels_pred, 0.9, 0.1)})
    with tempfile.TemporaryDirectory() as path:
        file = Path(path) / "toy.npz"
        trace.save(file)
        saved = load_trace(file)
        assert saved["metadata"]["seed"] == 42
        block = saved["splits"]["id_dev"]
        two = compare_two(block, 100, 200)
        summary = summarize_changes(two)
        overall = summary[summary.task == "all"].iloc[0]
        assert (overall.retained, overall.forgotten, overall.newly_learned,
                overall.still_incorrect) == (65, 15, 15, 5)
        assert overall.accuracy_before == overall.accuracy_after == .8
        assert overall.correctness_churn == .3
        assert overall.forgetting_rate == 15/80

        three = compare_three(block, 100, 200, 300)
        recovered = summarize_changes(three)
        overall_recovery = recovered[recovered.task == "all"].iloc[0]
        assert overall_recovery.recovered == 12
        assert overall_recovery.forgotten_unrecovered == 3
        assert overall_recovery.recovery_rate_among_forgotten == 12/15
        enriched = scene_context(two, toy.rows)
        assert enriched.n_objects.eq(1).all()
        print(summary.to_string(index=False))
        print("Test passed: binary metrics, save/load, all 4 categories, recovery and context")


if __name__ == "__main__":
    main()
