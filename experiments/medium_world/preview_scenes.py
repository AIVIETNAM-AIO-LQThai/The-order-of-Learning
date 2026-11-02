from pathlib import Path
import random
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.medium_visual_world.world import IMAGE_SIZE, sample_scene
from src.medium_visual_world.render import render

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "medium_world" / "figures"


def main() -> None:
    rng = random.Random(42)
    columns = 10
    rows = 10
    scale = 2
    padding = 8
    cell_size = IMAGE_SIZE * scale
    gallery = Image.new("RGB", (
        columns * (cell_size + padding) + padding,
        rows * (cell_size + padding) + padding
    ), (255, 255, 255))

    for index in range(columns * rows):
        scene = sample_scene(rng)
        image = render(scene).resize((cell_size, cell_size), Image.Resampling.NEAREST)
        x = padding + (index % columns) * (cell_size + padding)
        y = padding + (index // columns) * (cell_size + padding)
        gallery.paste(image, (x, y))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "medium_scene_gallery.png"
    gallery.save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
