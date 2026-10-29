"""Explicit typed query expression tree and model-visible prefix tokens."""
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Query:
    op: str
    args: tuple[Any,...]

def ref(obj):
    return Query('REF',(obj.color,obj.shape))

def exists_color(color): return Query('EXISTS_COLOR',(color,))
def exists_shape(shape): return Query('EXISTS_SHAPE',(shape,))
def exists_pair(color,shape): return Query('EXISTS_PAIR',(color,shape))
def left(a,b): return Query('LEFT',(ref(a),ref(b)))
def above(a,b): return Query('ABOVE',(ref(a),ref(b)))
def xor(a,b): return Query('XOR',(a,b))
def and_(a,b): return Query('AND',(a,b))
def closer(a,b,c): return Query('CLOSER',(ref(a),ref(b),ref(c)))

def tokens(q:Query) -> list[str]:
    t=[q.op]
    for arg in q.args:
        if isinstance(arg,Query):t.extend(tokens(arg))
        else:t.append(str(arg))
    return t

def encode(q: Query) -> dict:
    return {'op':q.op,'args':[encode(arg) if isinstance(arg,Query) else arg for arg in q.args]}

def decode(data:dict) -> Query:
    return Query(data['op'],tuple(decode(a) if isinstance(a,dict) and 'op' in a else a for a in data['args']))
