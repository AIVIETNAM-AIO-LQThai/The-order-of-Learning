"""Strict model-visible interface: strips scene graph, labels, IDs and roles.

The label is returned separately and MUST never be serialized into the input.
"""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

from src.visual_world.world import from_dict
from src.visual_world.render import render

def model_example(row:dict):
    image=render(from_dict(row['scene']))
    # Only image pixels and symbolic question tokens reach the model.
    return {'image':image,'query_tokens':row['query_tokens']},int(row['label'])
