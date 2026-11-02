from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import math
from PIL import Image, ImageDraw, ImageFilter
from src.medium_visual_world.world import IMAGE_SIZE, Scene

PALETTE = {
    "red": (209, 84, 76),
    "blue": (69, 138, 198),
    "green": (73, 156, 110),
    "yellow": (219, 176, 62),
    "purple": (137, 86, 188)
}
BACKGROUND = (243, 246, 244)
OUTLINE = (41, 51, 59)

def render(scene: Scene, size: int = IMAGE_SIZE, scale: int = 4) -> Image.Image:
    image = Image.new("RGB", (size * scale, size * scale), BACKGROUND)

    for obj in scene.objects:
        radius = obj.radius * scale
        margin = 3 * scale
        tile_size = 2 * (radius + margin) + 1
        center = tile_size // 2
        mask = Image.new("L", (tile_size, tile_size), 0)
        draw = ImageDraw.Draw(mask)

        if obj.shape == "circle":
            draw.ellipse((center-radius, center-radius,
                          center+radius, center+radius), fill=255)
        elif obj.shape == "square":
            half = round(radius / math.sqrt(2))
            draw.rectangle((center-half, center-half,
                            center+half, center+half), fill=255)
        elif obj.shape == "triangle":
            points = [
                (center, center-radius),
                (center-round(radius*math.sqrt(3)/2), center+round(radius/2)),
                (center+round(radius*math.sqrt(3)/2), center+round(radius/2))
            ]
            draw.polygon(points, fill=255)
        elif obj.shape == "crescent":
            draw.ellipse((center-radius, center-radius,
                          center+radius, center+radius), fill=255)
            inner = round(radius * 0.82)
            shift = round(radius * 0.42)
            draw.ellipse((center+shift-inner, center-inner,
                          center+shift+inner, center+inner), fill=0)
        else:
            raise ValueError(f"Unknown shape: {obj.shape}")

        if obj.angle:
            mask = mask.rotate(obj.angle, resample=Image.Resampling.BICUBIC)

        position = (obj.x * scale - center, obj.y * scale - center)
        outline = mask.filter(ImageFilter.MaxFilter(scale + 1))
        image.paste(OUTLINE, position, outline)
        image.paste(PALETTE[obj.color], position, mask)

    return image.resize((size, size), Image.Resampling.LANCZOS)
