import torch
from src.medium_visual_world.learning import (
    TASKS, VOCAB, encode_query, AttributeBindingViT
)
from src.medium_visual_world.curriculum import make_blocks
from src.medium_visual_world.world import Scene, Obj
from src.medium_visual_world.grammar import exists_color, exists_shape, has_color_and_shape, exists_pair
from src.medium_visual_world.oracle import evaluate as oracle


def test_question_encoding():
    assert len(encode_query(["EXISTS_COLOR", "red"])) == 4
    assert len(encode_query(["EXISTS_PAIR", "purple", "crescent"])) == 4
    assert "purple" in VOCAB and "crescent" in VOCAB


def test_question_semantics():
    scene = Scene((Obj("purple", "circle", 6, 10, 10),
                   Obj("blue", "crescent", 6, 30, 30),
                   Obj("yellow", "square", 6, 50, 50)))
    assert oracle(scene, exists_color("purple")) == 1
    assert oracle(scene, exists_shape("crescent")) == 1
    assert oracle(scene, has_color_and_shape("purple", "crescent")) == 1
    assert oracle(scene, exists_pair("purple", "crescent")) == 0


def test_matched_exposure():
    rows = [{"template": task} for task in TASKS for _ in range(10)]
    plans = []
    for schedule in ("mixed", "staged", "reverse"):
        blocks, plan = make_blocks(rows, schedule, 2, 1, 42)
        flattened = [i for _, ids in blocks for i in ids]
        assert len(flattened) == 120
        assert sorted(flattened) == sorted(list(range(40)) * 3)
        assert len(blocks) == (2 if schedule == "mixed" else 4)
        plans.append(plan)
    for key in ("total_presentations", "repetitions_per_example", "counts_by_template"):
        assert plans[0][key] == plans[1][key] == plans[2][key]
    assert [p["name"] for p in plans[0]["phases"]] == ["mixed", "common_mixed"]
    assert [p["name"] for p in plans[1]["phases"]] == ["attributes", "conjunction", "binding", "common_mixed"]
    assert [p["name"] for p in plans[2]["phases"]] == ["binding", "conjunction", "attributes", "common_mixed"]
    assert make_blocks(rows, "mixed", 2, 1, 42)[0][-1][1] == make_blocks(rows, "staged", 2, 1, 42)[0][-1][1]


def test_model_forward_and_gradients():
    torch.manual_seed(42)
    model = AttributeBindingViT(dim=16, patch_size=8)
    images = torch.rand(4, 3, 64, 64)
    queries = torch.tensor([
        encode_query(["EXISTS_COLOR", "red"]),
        encode_query(["EXISTS_SHAPE", "crescent"]),
        encode_query(["HAS_COLOR_AND_SHAPE", "purple", "circle"]),
        encode_query(["EXISTS_PAIR", "purple", "crescent"])
    ])
    out = model(images, queries)
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert model.patch.weight.grad is not None
