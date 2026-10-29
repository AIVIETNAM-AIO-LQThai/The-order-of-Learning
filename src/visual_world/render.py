"""Pixel-only rendering. Does not encode labels, scene IDs, roles, or split IDs."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))
    
from PIL import Image,ImageDraw
from src.visual_world.world import Scene

PALETTE={'red':(209,84,76),'blue':(69,138,198),'green':(73,156,110),'yellow':(219,176,62)}
BACKGROUND=(243,246,244)

def render(s:Scene,size=64,scale=4) -> Image.Image:
    im=Image.new('RGB',(size*scale,size*scale),BACKGROUND)
    draw=ImageDraw.Draw(im)
    for o in s.objs():
        x=o.x*scale;y=o.y*scale
        r=(4.7 if o.size=='small' else 6.6)*scale
        box=(round(x-r),round(y-r),round(x+r),round(y+r))
        fill=PALETTE[o.color];outline=(41,51,59)
        if o.shape=='circle':draw.ellipse(box,fill=fill,outline=outline,width=max(1,round(scale*.55)))
        elif o.shape=='square':draw.rectangle(box,fill=fill,outline=outline,width=max(1,round(scale*.55)))
        elif o.shape=='triangle':
            pts=[(round(x),round(y-r)),(round(x-r),round(y+r)),(round(x+r),round(y+r))]
            draw.polygon(pts,fill=fill)
            draw.line(pts+[pts[0]],fill=outline,width=max(1,round(scale*.55)),joint='curve')
        else:raise ValueError(o.shape)
    return im.resize((size,size),Image.Resampling.LANCZOS)
