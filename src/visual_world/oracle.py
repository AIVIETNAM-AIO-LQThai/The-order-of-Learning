"""Independent executable denotational semantics for symbolic queries."""
import math
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

from src.visual_world.grammar import Query
from src.visual_world.world import Scene

def evaluate(s:Scene, q:Query) -> bool|object:
    objs=s.objs(); op=q.op; args=q.args
    if op=='REF':
        matches=[o for o in objs if (o.color,o.shape)==tuple(args)]
        if len(matches)!=1:raise ValueError(f'REF must resolve to exactly one object: {args}, count={len(matches)}')
        return matches[0]
    if op=='EXISTS_COLOR':return any(o.color==args[0] for o in objs)
    if op=='EXISTS_SHAPE':return any(o.shape==args[0] for o in objs)
    if op=='EXISTS_PAIR':return any((o.color,o.shape)==tuple(args) for o in objs)
    if op=='LEFT':return evaluate(s,args[0]).x<evaluate(s,args[1]).x
    if op=='ABOVE':return evaluate(s,args[0]).y<evaluate(s,args[1]).y
    if op=='XOR':return bool(evaluate(s,args[0])) != bool(evaluate(s,args[1]))
    if op=='AND':return bool(evaluate(s,args[0])) and bool(evaluate(s,args[1]))
    if op=='CLOSER':
        a,b,c=(evaluate(s,r) for r in args)
        return math.hypot(a.x-b.x,a.y-b.y)<math.hypot(a.x-c.x,a.y-c.y)
    raise ValueError(f'unknown op: {op}')
