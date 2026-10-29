import itertools
from dataclasses import replace
from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.visual_world.world import *
from src.visual_world.grammar import *
from src.visual_world.oracle import evaluate
from src.visual_world.audit import feasible_counts,standalone_label
from src.visual_world.gallery import mutate
from src.visual_world.render import render
from src.visual_world.input_contract import model_example

def fixed_scene():
    return Scene(Obj('red','circle','small',11,11),Obj('red','square','small',39,39),Obj('blue','circle','small',25,53))

def test_role_config_feasibility():
    assert feasible_counts()['all_four_allowed']==12
    assert feasible_counts()['heldout_anchor_configs']==18

def test_reference_oracle_vectors():
    s=fixed_scene();validate_scene(s)
    tests=[(exists_color('red'),1),(exists_shape('triangle'),0),(exists_pair('red','circle'),1),
           (exists_pair('blue','square'),0),(left(s.A,s.B),1),(above(s.A,s.C),1),
           (xor(left(s.A,s.B),above(s.A,s.C)),0),
           (and_(left(s.A,s.B),above(s.A,s.C)),1),(closer(s.A,s.B,s.C),1)]
    for q,val in tests:
        assert int(evaluate(s,q))==val
        assert standalone_label(s.as_dict(),encode(q))==val
        assert decode(encode(q))==q

def test_missing_ref_rejected():
    import pytest
    with pytest.raises(ValueError):evaluate(fixed_scene(),left(Obj('green','triangle','small',11,11),fixed_scene().A))

def test_all_spatial_truths_possible():
    rng=random.Random(12)
    for pq in itertools.product((0,1),repeat=2):
        s=sample_scene(rng,'id',truth=pq)
        assert spatial_truth(s)==pq

def test_counterfactual_interventions():
    s=fixed_scene()
    for name,q in [('pair',exists_pair('red','circle')),
                   ('left',left(s.A,s.B)),('above',above(s.A,s.C)),
                   ('xor',xor(left(s.A,s.B),above(s.A,s.C)))]:
        row={'scene':s.as_dict(),'question':encode(q),'template':name}
        out,fields=mutate(row)
        assert int(evaluate(s,q))!=int(evaluate(out,q))
        assert fields

def test_model_interface_strips_metadata():
    s=fixed_scene();row={'scene':s.as_dict(),'id':'secret','split':'train','label':1,'heldout_pair':['red','triangle'],'query_tokens':['EXISTS_PAIR','red','circle']}
    x,y=model_example(row)
    assert set(x)=={'image','query_tokens'} and y==1
    assert x['image'].size==(64,64)
    assert x['image'].mode=='RGB'

def test_render_deterministic():
    a=render(fixed_scene()).tobytes();b=render(fixed_scene()).tobytes()
    assert a==b
