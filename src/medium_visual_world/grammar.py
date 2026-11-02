from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medium_visual_world.world import COLORS, SHAPES

OPERATIONS = ("EXISTS_COLOR", "EXISTS_SHAPE", "HAS_COLOR_AND_SHAPE", "EXISTS_PAIR")


class Query:
    def __init__(self, op: str, args: tuple[str, ...]):
        if op not in OPERATIONS:
            raise ValueError(f"Unknown question type: {op}")
        expected = 1 if op in ("EXISTS_COLOR", "EXISTS_SHAPE") else 2
        if len(args) != expected:
            raise ValueError(f"{op} expects {expected} attributes")
        if op == "EXISTS_COLOR" and args[0] not in COLORS:
            raise ValueError("Unknown color")
        if op == "EXISTS_SHAPE" and args[0] not in SHAPES:
            raise ValueError("Unknown shape")
        if expected == 2 and (args[0] not in COLORS or args[1] not in SHAPES):
            raise ValueError("Unknown color or shape")
        self.op = op
        self.args = args

    def as_dict(self) -> dict:
        return {"op": self.op, "args": list(self.args)}

    def tokens(self) -> list[str]:
        return [self.op, *self.args]


def from_dict(data: dict) -> Query:
    return Query(data["op"], tuple(data["args"]))


def exists_color(color: str) -> Query:
    return Query("EXISTS_COLOR", (color,))


def exists_shape(shape: str) -> Query:
    return Query("EXISTS_SHAPE", (shape,))


def has_color_and_shape(color: str, shape: str) -> Query:
    return Query("HAS_COLOR_AND_SHAPE", (color, shape))


def exists_pair(color: str, shape: str) -> Query:
    return Query("EXISTS_PAIR", (color, shape))
