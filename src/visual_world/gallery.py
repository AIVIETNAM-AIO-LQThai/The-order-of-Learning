"""Renderable inspection sheet and machine-checked counterfactual interventions."""
import argparse
import textwrap
import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

from PIL import Image,ImageDraw,ImageFont
from dataclasses import replace
from src.visual_world.world import Scene,Obj,from_dict,validate_scene,spatial_truth,HOLDOUT
from src.visual_world.generate import read_rows
from src.visual_world.grammar import decode
from src.visual_world.oracle import evaluate
from src.visual_world.render import render

W,H=260,180

def mutate(row):
    s=from_dict(row['scene']);k=row['template'];q=decode(row['question']);orig=int(evaluate(s,q))
    if k=='pair':
        # With the anchor/share-color/share-shape construction, a *single*
        # attribute edit necessarily creates a duplicate. Rebind BOTH anchor
        # attributes to the absent fourth pair, and reinterpret roles.
        changed=replace(s.A,color=s.C.color,shape=s.B.shape)
        out=Scene(changed,s.C,s.B)
        changed_fields=['anchor.color','anchor.shape']
    elif k=='left':
        out=Scene(replace(s.A,x=s.B.x),replace(s.B,x=s.A.x),s.C)
        changed_fields=['anchor.x','same_color_object.x']
    elif k=='above':
        out=Scene(replace(s.A,y=s.C.y),s.B,replace(s.C,y=s.A.y))
        changed_fields=['anchor.y','same_shape_object.y']
    elif k=='xor':
        out=Scene(replace(s.A,x=s.B.x),replace(s.B,x=s.A.x),s.C)
        changed_fields=['anchor.x','same_color_object.x']
        before=spatial_truth(s);after=spatial_truth(out)
        assert before[0]!=after[0] and before[1]==after[1],(before,after)
    else:raise ValueError(k)
    validate_scene(out)
    final=int(evaluate(out,q))
    assert orig!=final
    assert sorted((o.color,o.shape,o.size) for o in s.objs())==sorted((o.color,o.shape,o.size) for o in out.objs()) if k in ('left','above','xor') else True
    return out,changed_fields

def font(sz=13):
    try:return ImageFont.truetype('DejaVuSans.ttf',sz)
    except OSError:return ImageFont.load_default()

def tile(row,size=(W,H)):
    img=Image.new('RGB',size,(250,251,251));d=ImageDraw.Draw(img)
    scene=from_dict(row['scene']);thumb=render(scene).resize((112,112),Image.Resampling.NEAREST)
    img.paste(thumb,(8,8))
    q=' '.join(row['query_tokens']);chunks=textwrap.wrap(q,width=24,break_long_words=False)
    d.text((130,8),str(row['split'])+' / '+str(row['template']),fill=(20,35,45),font=font(12))
    for j,part in enumerate(chunks[:7]):d.text((130,29+j*13),part,fill=(40,57,68),font=font(10))
    d.text((9,126),'answer: '+str(row['label'])+'   '+row['id'][:10],fill=(38,64,79),font=font(12))
    if row.get('heldout_pair'):d.text((9,147),'OOD: '+','.join(row['heldout_pair']),fill=(139,78,45),font=font(11))
    return img

def make_gallery(root:Path):
    root=Path(root);root.mkdir(exist_ok=True,parents=True)
    rng_rows=[]
    for split in ('train','id_test','ood_test'):
        rs=list(read_rows(root/(split+'.jsonl')))
        for level in (1,2,3):
            candidates=[r for r in rs if r['level']==level]
            positives=[r for r in candidates if r['label']==1]
            negatives=[r for r in candidates if r['label']==0]
            rng_rows.extend(positives[:1]+negatives[:1])
    # 18 balanced examples plus 6 extra to reach 24 distinct scenes
    all_rows=list(read_rows(root/'train.jsonl'))
    selected={r['id'] for r in rng_rows}
    rng_rows.extend([r for r in all_rows if r['id'] not in selected][:6])
    sheet=Image.new('RGB',(W*4,H*6),(241,244,244))
    for i,row in enumerate(rng_rows[:24]):sheet.paste(tile(row),((i%4)*W,(i//4)*H))
    sheet.save(root/'gallery_24_scenes.png')
    sources=[]
    for split,k,n in [('train','pair',3),('id_test','left',1),('id_test','above',1),('ood_test','xor',3)]:
        possible=[r for r in read_rows(root/(split+'.jsonl')) if r['template']==k and (r['label']==1 if k=='pair' else True)]
        sources.extend(possible[:n])
    assert len(sources)==8
    pairs=[];pimg=Image.new('RGB',(W*4,H*4),(241,244,244))
    for i,row in enumerate(sources):
        s2,changed=mutate(row)
        after={**row,'id':s2.graph_id(),'scene':s2.as_dict(),'label':int(evaluate(s2,decode(row['question'])))}
        # Scene2 need not obey OOD positive-anchor constraint: it is an intervention, not training data.
        assert after['label']!=row['label']
        pimg.paste(tile(row),((2*i%4)*W,(i//2)*H))
        pimg.paste(tile(after),(((2*i+1)%4)*W,(i//2)*H))
        pairs.append({'pair_id':f'cf_{i:02d}','original_id':row['id'],
            'intervened_graph_id':after['id'],'split_source':row['split'],
            'template':row['template'],'original_label':row['label'],'new_label':after['label'],
            'changed_fields':changed,'query_tokens':row['query_tokens'],
            'before_scene':row['scene'],'after_scene':after['scene']})
    pimg.save(root/'counterfactuals_8_pairs.png')
    (root/'counterfactuals.json').write_text(json.dumps(pairs,indent=2,sort_keys=True)+'\n')
    return {'scenes':len(rng_rows),'counterfactual_pairs':len(pairs)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=PROJECT_ROOT / 'outputs/dataset')
    a=ap.parse_args();print(json.dumps(make_gallery(a.root),indent=2))
