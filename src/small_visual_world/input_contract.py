"""Strict model-visible interface: strips scene graph, labels, IDs and roles.

The label is returned separately and MUST never be serialized into the input.
"""

from src.small_visual_world.world import from_dict
from src.small_visual_world.render import render

def model_example(row:dict):
    image=render(from_dict(row['scene']))
    # Only image pixels and symbolic question tokens reach the model.
    return {'image':image,'query_tokens':row['query_tokens']},int(row['label'])
