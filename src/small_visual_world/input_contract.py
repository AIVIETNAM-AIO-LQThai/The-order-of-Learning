"""Strict model-visible interface: strips scene graph, labels, IDs and roles.

The label is returned separately and MUST never be serialized into the input.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.small_visual_world.world import from_dict
from src.small_visual_world.render import render

def model_example(row:dict):
    image=render(from_dict(row['scene']))
    # Only image pixels and symbolic question tokens reach the model.
    return {'image':image,'query_tokens':row['query_tokens']},int(row['label'])
