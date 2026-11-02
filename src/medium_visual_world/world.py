from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
import random

COLORS = ("red", "blue", "green", "yellow", "purple")
SHAPES = ("circle", "square", "triangle", "crescent")
IMAGE_SIZE = 64
ANGLES = {
    "circle": (0,),
    "square": (0, 45),
    "triangle": tuple(range(0, 360, 45)),
    "crescent": tuple(range(0, 360, 45))
}

class Obj:
    def __init__(self, color: str, shape: str, radius: int, x: int, y: int, angle: int = 0):
        self.color = color
        self.shape = shape
        self.radius = radius
        self.x = x
        self.y = y
        self.angle = angle

    def pair(self) -> tuple[str, str]:
        return self.color, self.shape
    def as_dict(self) -> dict:
        return {
            "color": self.color,
            "shape": self.shape,
            "radius": self.radius,
            "x": self.x,
            "y": self.y,
            "angle": self.angle
        }

class Scene:
    def __init__(self, objects: tuple[Obj, ...]):
        self.objects = objects

    def as_dict(self) -> dict:
        return {"objects": [obj.as_dict() for obj in self.objects]}

    def scene_id(self) -> str:
        objects = [obj.as_dict() for obj in self.objects]
        objects.sort(key=lambda obj: (
            obj["color"], obj["shape"], obj["radius"], obj["x"], obj["y"], obj["angle"]
        ))
        contentn = json.dumps(objects, sort_keys=True)
        return sha256(contentn.encode("utf-8")).hexdigest()

def from_dict(data: dict) -> Scene:
    objects = []
    for item in data["objects"]:
        obj = Obj(
            item["color"],
            item["shape"],
            item["radius"],
            item["x"],
            item["y"],
            item["angle"]
        )
        objects.append(obj)
    return Scene(tuple(objects))

def validate_scene(scene: Scene) -> None:
    objects = scene.objects
    assert 3 <= len(objects) <= 5

    for obj in objects:
        assert obj.color in COLORS
        assert obj.shape in SHAPES
        assert obj.angle in ANGLES[obj.shape]
        assert 5 <= obj.radius <= 8
        assert obj.radius + 3 <= obj.x < IMAGE_SIZE - obj.radius - 3
        assert obj.radius + 3 <= obj.y < IMAGE_SIZE - obj.radius - 3

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            distance = math.hypot(a.x - b.x, a.y - b.y)
            assert distance > a.radius + b.radius + 2

def sample_scene(
    rng: random.Random, n_objects: int | None = None,
    allowed_pairs: list[tuple[str, str]] | None = None
) -> Scene:
    if n_objects is None:
        n_objects = rng.randint(3, 5)
    if not 3 <= n_objects <= 5:
        raise ValueError("n_objects must be between 3 and 5")

    pairs = allowed_pairs
    if pairs is None:
        pairs = [(c, s) for c in COLORS for s in SHAPES]
    if not pairs:
        raise ValueError("allowed_pairs cannot be empty")

    for _ in range(100):
        objects = []
        for _ in range(n_objects):
            color, shape = rng.choice(pairs)
            radius = rng.randint(5, 8)
            angle = rng.choice(ANGLES[shape])
            for _ in range(300):
                x = rng.randint(radius + 3, IMAGE_SIZE - radius - 4)
                y = rng.randint(radius + 3, IMAGE_SIZE - radius - 4)
                obj = Obj(color, shape, radius, x, y, angle)
                if all(
                    math.hypot(x - other.x, y - other.y) > radius + other.radius + 2
                    for other in objects
                ):
                    objects.append(obj)
                    break
            else:
                break

        if len(objects) == n_objects:
            scene = Scene(tuple(objects))
            validate_scene(scene)
            return scene

    raise RuntimeError("Could not generate a non-overlapping scene")