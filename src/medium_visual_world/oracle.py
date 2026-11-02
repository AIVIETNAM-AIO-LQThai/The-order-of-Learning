from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.medium_visual_world.grammar import Query
from src.medium_visual_world.world import Scene


def evaluate(scene: Scene, question: Query) -> bool:
    objects = scene.objects
    op = question.op
    args = question.args

    if op == "EXISTS_COLOR":
        return any(obj.color == args[0] for obj in objects)
    if op == "EXISTS_SHAPE":
        return any(obj.shape == args[0] for obj in objects)
    if op == "HAS_COLOR_AND_SHAPE":
        has_color = any(obj.color == args[0] for obj in objects)
        has_shape = any(obj.shape == args[1] for obj in objects)
        return has_color and has_shape
    if op == "EXISTS_PAIR":
        return any(obj.pair() == args for obj in objects)
    raise ValueError(f"Unknown question type: {op}")
