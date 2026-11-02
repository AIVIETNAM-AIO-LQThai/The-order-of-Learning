from pathlib import Path
import sys
from PIL import Image, ImageDraw

PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medium_visual_world.generate import read_jsonl
from src.medium_visual_world.world import from_dict
from src.medium_visual_world.render import render

DATASET_DIR = PROJECT_ROOT / "outputs" / "medium_world" / "dataset"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "medium_world" / "figures"

def main():
    rows = list(read_jsonl(DATASET_DIR / "train.jsonl"))
    width, height = 280, 122
    columns, count = 4, 24
    gallery = Image.new("RGB", (columns * width, 6 * height), "white")
    draw = ImageDraw.Draw(gallery)
    for i, row in enumerate(rows[:count]):
        x, y = (i % columns) * width, (i // columns) * height
        scene = from_dict(row["scene"])
        picture = render(scene).resize((98, 98), Image.Resampling.NEAREST)
        gallery.paste(picture, (x + 6, y + 8))
        tokens = row["query_tokens"]
        draw.text((x + 111, y + 20), tokens[0], fill=(30, 30, 30))
        draw.text((x + 111, y + 39), " ".join(tokens[1:]), fill=(30, 30, 30))
        draw.text((x + 111, y + 60), f"Answer: {bool(row['label'])}", fill=(30, 30, 30))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "medium_questions_gallery.png"
    gallery.save(output)
    print(output)

if __name__ == "__main__":
    main()