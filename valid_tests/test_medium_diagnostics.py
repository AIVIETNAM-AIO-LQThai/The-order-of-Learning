from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.medium_world.train_single_task import balanced_indices, evaluate_indices
from experiments.medium_world.diagnose_question_shortcuts import categorize, summarize


class SmallDataset:
    def __init__(self):
        self.rows = ([{"template": "shape", "label": i % 2} for i in range(40)]
                     + [{"template": "pair", "label": i % 2} for i in range(40)])


def test_balanced_indices():
    dataset = SmallDataset()
    indices = balanced_indices(dataset, "shape", 16, 42)
    assert len(indices) == len(set(indices)) == 16
    assert sum(dataset.rows[i]["label"] for i in indices) == 8
    assert all(dataset.rows[i]["template"] == "shape" for i in indices)
    assert balanced_indices(dataset, "shape", 16, 42) == indices


def test_conjunction_subgroups():
    scene = {"objects": [{"color": "red", "shape": "circle"},
                         {"color": "blue", "shape": "square"}]}
    base = {"scene": scene, "template": "conjunction"}
    assert categorize({**base, "question": {"args": ["red", "square"]}, "label": 1}) == "positive_both_present"
    assert categorize({**base, "question": {"args": ["green", "square"]}, "label": 0}) == "negative_missing_color"
    assert categorize({**base, "question": {"args": ["red", "triangle"]}, "label": 0}) == "negative_missing_shape"
    assert categorize({**base, "question": {"args": ["green", "triangle"]}, "label": 0}) == "negative_both_missing"


def test_binding_hard_negative():
    scene = {"objects": [{"color": "red", "shape": "circle"},
                         {"color": "blue", "shape": "square"}]}
    row = {"scene": scene, "template": "pair", "question": {"args": ["red", "square"]}, "label": 0}
    assert categorize(row) == "negative_hard_pair"
    row["question"] = {"args": ["red", "circle"]}
    row["label"] = 1
    assert categorize(row) == "positive_pair_present"


def test_summarize():
    assert summarize([(1, 1), (0, 1), (0, 0), (1, 0)]) == {
        "count": 4, "accuracy": 0.5, "predicted_positive_rate": 0.5}


if __name__ == "__main__":
    test_balanced_indices()
    test_conjunction_subgroups()
    test_binding_hard_negative()
    test_summarize()
    print("Medium diagnostics tests passed")
