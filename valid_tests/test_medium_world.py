from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tempfile import TemporaryDirectory

import random

from src.medium_visual_world.world import Obj, Scene, from_dict, sample_scene, validate_scene
from src.medium_visual_world.grammar import (
    exists_color, exists_shape, exists_pair, has_color_and_shape, from_dict as question_from_dict
)
from src.medium_visual_world.oracle import evaluate
from src.medium_visual_world.generate import generate, read_jsonl, HOLDOUT
from src.medium_visual_world.audit import audit

def test_scene_roundtrip():
    scene = sample_scene(random.Random(42), 5)
    validate_scene(scene)
    restored = from_dict(scene.as_dict())
    assert restored.scene_id() == scene.scene_id()
    assert len(restored.objects) == 5

def test_question_semantics():
    scene = Scene((
        Obj("purple", "circle", 6, 12, 12),
        Obj("blue", "crescent", 6, 35, 12),
        Obj("yellow", "square", 6, 20, 39)
    ))
    validate_scene(scene)
    questions = [
        (exists_color("purple"), True),
        (exists_shape("crescent"), True),
        (has_color_and_shape("purple", "crescent"), True),
        (exists_pair("purple", "crescent"), False),
        (exists_pair("blue", "crescent"), True),
        (exists_color("green"), False)
    ]
    for question, truth in questions:
        assert evaluate(scene, question) == truth
        assert question_from_dict(question.as_dict()).tokens() == question.tokens()

def test_deterministic_small_dataset():
    with TemporaryDirectory() as first, TemporaryDirectory() as second:
        settings = dict(seed=7, train_scenes=32, id_dev_scenes=16,
                        id_test_scenes=16, ood_dev_scenes=8,
                        ood_test_scenes=8, materialize=False)
        meta_a = generate(Path(first), **settings)
        meta_b = generate(Path(second), **settings)
        assert meta_a["manifests_sha256"] == meta_b["manifests_sha256"]
        assert audit(Path(first))["status"] == "passed"
        rows = list(read_jsonl(Path(first) / "train.jsonl"))
        assert len(rows) == 64
        assert all(not ({tuple((o["color"], o["shape"]))
                        for o in row["scene"]["objects"]} & HOLDOUT) for row in rows)


if __name__ == "__main__":
    test_scene_roundtrip()
    test_question_semantics()
    test_deterministic_small_dataset()
    print("Medium world tests passed")