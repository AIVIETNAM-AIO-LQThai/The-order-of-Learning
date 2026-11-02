from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import random
import shutil
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medium_visual_world.world import COLORS, SHAPES, sample_scene
from src.medium_visual_world.grammar import (
    exists_color, exists_shape, has_color_and_shape, exists_pair
)
from src.medium_visual_world.oracle import evaluate
from src.medium_visual_world.render import render

HOLDOUT = {
    ("red", "crescent"), ("blue", "square"),
    ("green", "circle"), ("purple", "triangle")
}
ALL_PAIRS = [(c, s) for c in COLORS for s in SHAPES]
TRAIN_PAIRS = [pair for pair in ALL_PAIRS if pair not in HOLDOUT]
SPLITS = ("train", "id_dev", "id_test", "ood_dev", "ood_test")


def question_options(scene, task: str, answer: int) -> list[tuple[str, ...]]:
    colors = {obj.color for obj in scene.objects}
    shapes = {obj.shape for obj in scene.objects}
    pairs = {obj.pair() for obj in scene.objects}
    if task == "color":
        return [(c,) for c in COLORS if (c in colors) == bool(answer)]
    if task == "shape":
        return [(s,) for s in SHAPES if (s in shapes) == bool(answer)]
    if task == "conjunction":
        return [(c, s) for c in COLORS for s in SHAPES
                if (c, s) not in HOLDOUT
                and (c in colors and s in shapes) == bool(answer)
                and (answer or (c in colors) != (s in shapes))]
    if task == "pair":
        if answer:
            return sorted(pairs)
        return [(c, s) for c in colors for s in shapes
                if (c, s) not in pairs and (c, s) not in HOLDOUT]
    raise ValueError(task)


def make_question(task: str, target: tuple[str, ...]):
    if task == "color":
        return exists_color(target[0])
    if task == "shape":
        return exists_shape(target[0])
    if task == "conjunction":
        return has_color_and_shape(target[0], target[1])
    if task == "pair":
        return exists_pair(target[0], target[1])
    raise ValueError(task)


def choose_target(rng: random.Random, candidates: list[tuple[str, ...]],
                  task: str, answer: int, counts: Counter):
    rng.shuffle(candidates)
    return min(candidates, key=lambda target: (
        counts[(task, target, answer)] - counts[(task, target, 1-answer)],
        counts[(task, target, answer)]
    ))


def task_plans(n_scenes: int, rng: random.Random):
    if n_scenes % 8:
        raise ValueError("ID split scene count must be divisible by 8")
    task_pairs = (("color", "shape"), ("pair", "conjunction"),
                  ("color", "pair"), ("shape", "conjunction"))
    pairs = [tasks for tasks in task_pairs for _ in range(n_scenes // 4)]
    rng.shuffle(pairs)
    counts = Counter()
    plans = []
    for tasks in pairs:
        planned = []
        for task in tasks:
            planned.append((task, counts[task] % 2))
            counts[task] += 1
        plans.append(tuple(planned))
    return plans


def make_record(scene, split: str, task: str, question, answer: int):
    scene_id = scene.scene_id()
    return {
        "id": f"{scene_id}_{task}",
        "scene_id": scene_id,
        "split": split,
        "template": task,
        "label": int(answer),
        "question": question.as_dict(),
        "query_tokens": question.tokens(),
        "scene": scene.as_dict(),
        "image": f"images/{split}/{scene_id}.png"
    }


def generate_id_split(rng: random.Random, split: str, n_scenes: int,
                      taken: set[str], materialize: bool, out: Path):
    records = []
    counts = Counter()
    for planned in task_plans(n_scenes, rng):
        for attempt in range(2000):
            scene = sample_scene(rng, allowed_pairs=TRAIN_PAIRS)
            scene_id = scene.scene_id()
            if scene_id in taken:
                continue
            options = [question_options(scene, task, answer)
                       for task, answer in planned]
            if all(options):
                break
        else:
            raise RuntimeError(f"Unable to sample a balanced scene: {planned}")

        taken.add(scene_id)
        for (task, answer), choices in zip(planned, options):
            target = choose_target(rng, choices, task, answer, counts)
            question = make_question(task, target)
            if int(evaluate(scene, question)) != answer:
                raise AssertionError("Oracle and requested label disagree")
            counts[(task, target, answer)] += 1
            records.append(make_record(scene, split, task, question, answer))
        if materialize:
            image_path = out / "images" / split / f"{scene_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            render(scene).save(image_path)
    return records


def generate_ood_split(rng: random.Random, split: str, n_scenes: int,
                       taken: set[str], materialize: bool, out: Path):
    if n_scenes % 2:
        raise ValueError("OOD split scene count must be even")
    records = []
    counts = Counter()
    for i in range(n_scenes):
        answer = i % 2
        for attempt in range(3000):
            pairs = ALL_PAIRS if answer else TRAIN_PAIRS
            scene = sample_scene(rng, allowed_pairs=pairs)
            scene_id = scene.scene_id()
            if scene_id in taken:
                continue
            colors = {obj.color for obj in scene.objects}
            shapes = {obj.shape for obj in scene.objects}
            present = {obj.pair() for obj in scene.objects}
            candidates = [pair for pair in HOLDOUT
                          if (pair in present) == bool(answer)
                          and (answer or (pair[0] in colors and pair[1] in shapes))]
            if candidates:
                break
        else:
            raise RuntimeError("Unable to sample an OOD scene")

        target = choose_target(rng, sorted(candidates), "pair", answer, counts)
        question = exists_pair(*target)
        if int(evaluate(scene, question)) != answer:
            raise AssertionError("OOD oracle and requested label disagree")
        counts[("pair", target, answer)] += 1
        taken.add(scene_id)
        records.append(make_record(scene, split, "pair", question, answer))
        if materialize:
            image_path = out / "images" / split / f"{scene_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            render(scene).save(image_path)
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in records:
            file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def generate(out: Path, seed: int = 20261103,
             train_scenes: int = 5000, id_dev_scenes: int = 512,
             id_test_scenes: int = 512, ood_dev_scenes: int = 400,
             ood_test_scenes: int = 400, materialize: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "images").exists():
        shutil.rmtree(out / "images")
    split_sizes = {
        "train": train_scenes, "id_dev": id_dev_scenes,
        "id_test": id_test_scenes, "ood_dev": ood_dev_scenes,
        "ood_test": ood_test_scenes
    }
    taken = set()
    counts = {}
    hashes = {}
    for index, split in enumerate(SPLITS):
        rng = random.Random(seed + index * 1009)
        if split.startswith("ood"):
            records = generate_ood_split(rng, split, split_sizes[split],
                                         taken, materialize, out)
        else:
            records = generate_id_split(rng, split, split_sizes[split],
                                        taken, materialize, out)
        path = out / f"{split}.jsonl"
        write_jsonl(path, records)
        counts[split] = {"scenes": split_sizes[split], "questions": len(records)}
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()

    metadata = {
        "seed": seed,
        "splits": counts,
        "heldout_pairs": [list(pair) for pair in sorted(HOLDOUT)],
        "manifests_sha256": hashes,
        "image_files_generated": materialize,
        "unique_scenes": len(taken)
    }
    (out / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
