"""Scene definition and exhaustive feasible role enumeration."""
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

import math
import random

COLORS = ('red', 'blue', 'green', 'yellow')
SHAPES = ('circle', 'square', 'triangle')
SIZES = ('small', 'large')
GRID = (11, 25, 39, 53)
HOLDOUT = {('red', 'triangle'), ('blue', 'square'), ('green', 'circle')}

@dataclass(frozen=True)
class Obj:
    color: str
    shape: str
    size: str
    x: int
    y: int
    def pair(self) -> tuple[str, str]:
        return self.color, self.shape

@dataclass(frozen=True)
class Scene:
    # A/B/C is generation-only metadata: never sent as training input.
    A: Obj
    B: Obj
    C: Obj
    def objs(self) -> tuple[Obj, Obj, Obj]:
        return self.A, self.B, self.C
    def graph(self) -> list[dict]:
        return sorted((asdict(o) for o in self.objs()), key=lambda d: (d['color'],d['shape'],d['size'],d['x'],d['y']))
    def graph_id(self) -> str:
        return sha256(json.dumps(self.graph(),sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
    def as_dict(self) -> dict:
        return {k:asdict(getattr(self,k)) for k in 'ABC'}

def from_dict(d: dict) -> Scene:
    return Scene(*(Obj(**d[r]) for r in 'ABC'))

def role_configs() -> list[tuple[tuple[str,str],tuple[str,str],tuple[str,str],tuple[str,str]]]:
    configs=[]
    for c0 in COLORS:
        for c1 in COLORS:
            if c0==c1: continue
            for s0 in SHAPES:
                for s1 in SHAPES:
                    if s0==s1: continue
                    configs.append(((c0,s0),(c0,s1),(c1,s0),(c1,s1)))
    return configs

def configs_for(kind: str):
    configs=role_configs()
    if kind=='id':
        # The absent fourth combination is allowed too, for hard negatives.
        return [v for v in configs if all(p not in HOLDOUT for p in v)]
    if kind=='id_relaxed':
        return [v for v in configs if all(p not in HOLDOUT for p in v[:3])]
    if kind=='ood_pos':
        return [v for v in configs if v[0] in HOLDOUT and v[1] not in HOLDOUT and v[2] not in HOLDOUT]
    if kind=='ood_neg':
        return [v for v in configs if v[3] in HOLDOUT and all(p not in HOLDOUT for p in v[:3])]
    raise ValueError(kind)

def spatial_truth(s: Scene) -> tuple[int,int]:
    return (int(s.A.x<s.B.x),int(s.A.y<s.C.y))

def closer_margin(s: Scene) -> float:
    dab=math.hypot(s.A.x-s.B.x,s.A.y-s.B.y)
    dac=math.hypot(s.A.x-s.C.x,s.A.y-s.C.y)
    return dac-dab  # positive means B closer to A than C

def sample_scene(rng: random.Random, kind: str, truth: tuple[int,int]|None=None, closer:int|None=None, min_margin:float=4.0) -> Scene:
    configs=configs_for(kind)
    if not configs: raise ValueError(f'no configs for {kind}')
    # Independent attribute/position sampling before oracle-conditional rejection.
    for _ in range(10000):
        a,b,c,_=rng.choice(configs)
        xs=rng.sample(GRID,3);ys=rng.sample(GRID,3)
        coords=[(x+rng.randint(-1,1),y+rng.randint(-1,1)) for x,y in zip(xs,ys)]
        sizes=[rng.choice(SIZES) for _ in range(3)]
        s=Scene(*(Obj(pair[0],pair[1],sz,xy[0],xy[1]) for pair,sz,xy in zip((a,b,c),sizes,coords)))
        if truth is not None and spatial_truth(s)!=truth:continue
        if closer is not None:
            m=closer_margin(s)
            if abs(m)<=min_margin or int(m>0)!=closer:continue
        return s
    raise RuntimeError('scene rejection sampler exhausted; inspect constraints')

def validate_scene(s:Scene) -> None:
    os=s.objs()
    assert len(os)==3 and len({o.pair() for o in os})==3
    assert len({o.x for o in os})==3 and len({o.y for o in os})==3
    assert all(o.color in COLORS and o.shape in SHAPES and o.size in SIZES for o in os)
    assert s.A.color==s.B.color and s.A.shape==s.C.shape
    assert s.A.shape!=s.B.shape and s.A.color!=s.C.color
    assert all(any(abs(v-g)<=1 for g in GRID) for o in os for v in (o.x,o.y))
    assert all(3<=v<=60 for o in os for v in (o.x,o.y))
    # Concentric conservative bounding-circle test, including anti-alias stroke.
    for i in range(3):
        for j in range(i+1,3):
            a,b=os[i],os[j]
            assert math.hypot(a.x-b.x,a.y-b.y) > 15.0
