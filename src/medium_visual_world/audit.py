from collections import Counter, defaultdict
from pathlib import Path
import json
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medium_visual_world.generate import HOLDOUT, SPLITS, read_jsonl
from src.medium_visual_world.world import from_dict, validate_scene
from src.medium_visual_world.grammar import from_dict as query_from_dict
from src.medium_visual_world.oracle import evaluate


def audit(out: Path) -> dict:
    out = Path(out)
    metadata = json.loads((out / "dataset_metadata.json").read_text(encoding="utf-8"))
    seen_scenes = set()
    result = {}
    for split in SPLITS:
        records = list(read_jsonl(out / f"{split}.jsonl"))
        grouped = defaultdict(list)
        labels = Counter()
        tasks = Counter()
        hard_negatives = 0
        unbound_conjunctions = 0

        for row in records:
            scene = from_dict(row["scene"])
            validate_scene(scene)
            question = query_from_dict(row["question"])
            scene_id = scene.scene_id()
            if row["scene_id"] != scene_id or row["id"] != f"{scene_id}_{row['template']}":
                raise AssertionError("Incorrect record ID")
            if row["split"] != split or row["query_tokens"] != question.tokens():
                raise AssertionError("Incorrect split or question encoding")
            if int(evaluate(scene, question)) != row["label"]:
                raise AssertionError("Incorrect label")
            grouped[scene_id].append(row)
            labels[(row["template"], row["label"])] += 1
            tasks[row["template"]] += 1
            color_shape_pairs = {obj.pair() for obj in scene.objects}

            if not split.startswith("ood"):
                if color_shape_pairs & HOLDOUT:
                    raise AssertionError("Held-out combination leaked into an ID scene")
                if len(question.args) == 2 and tuple(question.args) in HOLDOUT:
                    raise AssertionError("Held-out combination leaked into an ID question")
            if split.startswith("ood"):
                if row["template"] != "pair" or tuple(question.args) not in HOLDOUT:
                    raise AssertionError("OOD must test held-out bindings")
                if row["label"] == 1 and tuple(question.args) not in color_shape_pairs:
                    raise AssertionError("Missing positive held-out pair")
            if row["template"] == "pair" and row["label"] == 0:
                colors = {obj.color for obj in scene.objects}
                shapes = {obj.shape for obj in scene.objects}
                if question.args[0] not in colors or question.args[1] not in shapes:
                    raise AssertionError("Pair negative is not a hard negative")
                hard_negatives += 1
            if row["template"] == "conjunction" and row["label"] == 1:
                if tuple(question.args) not in color_shape_pairs:
                    unbound_conjunctions += 1
            if metadata["image_files_generated"] and not (out / row["image"]).is_file():
                raise AssertionError("Missing rendered image")

        for scene_id, rows in grouped.items():
            if scene_id in seen_scenes:
                raise AssertionError("Scene shared across data splits")
            if len(rows) != (1 if split.startswith("ood") else 2):
                raise AssertionError("Wrong number of questions for scene")
            if len({row["template"] for row in rows}) != len(rows):
                raise AssertionError("Duplicate question type for scene")
            if any(row["scene"] != rows[0]["scene"] for row in rows):
                raise AssertionError("Scene content differs between questions")
            seen_scenes.add(scene_id)

        if len(grouped) != metadata["splits"][split]["scenes"]:
            raise AssertionError("Incorrect scene count")
        if len(records) != metadata["splits"][split]["questions"]:
            raise AssertionError("Incorrect question count")
        for task in tasks:
            if labels[(task, 0)] != labels[(task, 1)]:
                raise AssertionError(f"Unbalanced labels in {split}: {task}")
        result[split] = {
            "scenes": len(grouped), "questions": len(records),
            "tasks": dict(sorted(tasks.items())),
            "labels": {f"{task}_{label}": n for (task, label), n in sorted(labels.items())},
            "hard_negative_pairs": hard_negatives,
            "positive_conjunction_without_pair": unbound_conjunctions
        }
    report = {"status": "passed", "unique_scenes": len(seen_scenes), "splits": result}
    (out / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report