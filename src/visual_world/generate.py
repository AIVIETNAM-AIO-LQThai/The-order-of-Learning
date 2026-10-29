"""Deterministic, label-balanced, graph-disjoint dataset construction."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

import argparse
import json
import random
from src.visual_world.world import COLORS, SHAPES, HOLDOUT, Scene, configs_for, sample_scene, spatial_truth, closer_margin, validate_scene
from src.visual_world.grammar import exists_color, exists_shape, exists_pair, left, above, xor, closer, encode, tokens
from src.visual_world.oracle import evaluate
from src.visual_world.render import render

DATASETS=('train','id_dev','id_test','ood_dev','ood_test','adapt_train','adapt_test')

def task_plan(split:str,level:int|None,n:int):
    """Stratify labels and, in L1, exact query contents to limit query-only cues."""
    if split in ('adapt_train','adapt_test'):
        assert n%2==0
        return [('closer',b,None,None) for b in (0,1) for _ in range(n//2)]
    if level==1:
        types=('pair',) if split.startswith('ood') else ('color','shape','pair')
        assert n%(len(types)*2)==0
        plans=[]
        for typ in types:
            values=(sorted(HOLDOUT) if split.startswith('ood') else
                    list(COLORS) if typ=='color' else list(SHAPES) if typ=='shape' else
                    sorted((c,sh) for c in COLORS for sh in SHAPES if (c,sh) not in HOLDOUT))
            for b in (0,1):
                count=n//(len(types)*2)
                for j in range(count):plans.append((typ,b,None,values[j%len(values)]))
        return plans
    if level==2:
        assert n%4==0
        configs=configs_for('ood_pos' if split.startswith('ood') else 'id_relaxed')
        return [(op,b,None,configs[j%len(configs)][:3])
                for op in ('left','above') for b in (0,1) for j in range(n//4)]
    if level==3:
        assert n%4==0
        configs=configs_for('ood_pos' if split.startswith('ood') else 'id_relaxed')
        return [('xor',(p!=q),(p,q),configs[j%len(configs)][:3])
                for p in (0,1) for q in (0,1) for j in range(n//4)]
    raise ValueError(level)

def make_question(s:Scene,typ:str,truth:int,target=None):
    if typ=='color':return exists_color(target)
    if typ=='shape':return exists_shape(target)
    if typ=='pair':return exists_pair(*target)
    if typ=='left':return left(s.A,s.B)
    if typ=='above':return above(s.A,s.C)
    if typ=='xor':return xor(left(s.A,s.B),above(s.A,s.C))
    if typ=='closer':return closer(s.A,s.B,s.C)
    raise ValueError(typ)

def sample_record(rng:random.Random, split:str, level:int|None, typ:str, requested:int, pq:tuple[int,int]|None, target, taken:set[str]):
    if split.startswith('ood'):
        kind='ood_pos' if typ=='pair' and requested else 'ood_neg' if typ=='pair' else 'ood_pos'
    else:kind='id' if typ=='pair' else 'id_relaxed'
    # Matching the exact L1 query across both labels blocks query-only shortcuts.
    for attempts in range(10000):
        want=pq
        if typ=='left':want=(requested,rng.randint(0,1))
        if typ=='above':want=(rng.randint(0,1),requested)
        s=sample_scene(rng,kind,truth=want,closer=requested if typ=='closer' else None)
        gid=s.graph_id()
        if gid in taken:continue
        if typ=='pair' and (s.A.pair() if requested else (s.C.color,s.B.shape))!=target:continue
        if typ=='color' and (target in {o.color for o in s.objs()})!=bool(requested):continue
        if typ=='shape' and (target in {o.shape for o in s.objs()})!=bool(requested):continue
        if typ in ('left','above','xor') and tuple(o.pair() for o in s.objs())!=target:continue
        q=make_question(s,typ,requested,target)
        result=int(evaluate(s,q))
        if result!=int(requested):raise AssertionError(f'oracle discrepancy {split},{typ},{requested},{result}')
        # OOD positives and negatives always query H; L2/3 anchor is in H.
        if split.startswith('ood'):
            if typ=='pair':assert tuple(q.args) in HOLDOUT
            else:assert s.A.pair() in HOLDOUT
        else:
            assert all(o.pair() not in HOLDOUT for o in s.objs())
            t=tokens(q)
            assert not any((t[i],t[i+1]) in HOLDOUT for i in range(len(t)-1))
        validate_scene(s)
        taken.add(gid)
        return {'id':gid,'split':split,'level':level,'template':typ,'label':result,
                'question':encode(q),'query_tokens':tokens(q),'scene':s.as_dict(),
                'p_q':list(spatial_truth(s)) if typ=='xor' else None,
                'heldout_pair':list(s.A.pair() if typ!='pair' else tuple(q.args)) if split.startswith('ood') else None,
                'closer_margin':round(closer_margin(s),5) if typ=='closer' else None}
    raise RuntimeError('exhausted unique scene graphs')

def rows_for_split(rng:random.Random,split:str,n_level:int,adapt_train:int,adapt_test:int,taken:set[str]):
    if split in ('adapt_train','adapt_test'):
        n=adapt_train if split=='adapt_train' else adapt_test
        plan=task_plan(split,None,n)
    else:
        plan=[(level,typ,truth,pq,target) for level in (1,2,3)
              for typ,truth,pq,target in task_plan(split,level,n_level)]
    rng.shuffle(plan)
    rows=[]
    for item in plan:
        if split in ('adapt_train','adapt_test'):
            typ,truth,pq,target=item;level=None
        else:level,typ,truth,pq,target=item
        rows.append(sample_record(rng,split,level,typ,truth,pq,target,taken))
    return rows

def write_jsonl(path:Path,rows):
    with path.open('w',encoding='utf8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')

def digest(path:Path):return sha256(path.read_bytes()).hexdigest()

def generate(out:Path,seed=20261025,train_per_level=600,eval_per_level=120,adapt_train=128,adapt_test=240,materialize_all=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Old visual/audit artifacts must not survive a regeneration with new seeds.
    for old in ('counterfactuals.json','counterfactuals_8_pairs.png','gallery_24_scenes.png','audit_report.json'):
        (out/old).unlink(missing_ok=True)
    rng=random.Random(seed);taken=set();counts={}; hashes={}
    for split in DATASETS:
        nlev=train_per_level if split=='train' else eval_per_level
        rows=rows_for_split(rng,split,nlev,adapt_train,adapt_test,taken)
        p=out/f'{split}.jsonl';write_jsonl(p,rows)
        counts[split]=len(rows);hashes[p.name]=digest(p)
        if materialize_all:
            imgdir=out/'images'/split;imgdir.mkdir(parents=True,exist_ok=True)
            for row in rows:
                from src.visual_world.world import from_dict
                render(from_dict(row['scene'])).save(imgdir/(row['id']+'.png'))
    config={'seed':seed,'train_per_level':train_per_level,'eval_per_level':eval_per_level,
            'adapt_train':adapt_train,'adapt_test':adapt_test,'holdout':sorted([list(x) for x in HOLDOUT]),
            'materialize_all':materialize_all,'renderer':'PIL LANCZOS supersampling=4',
            'version':'0.1.0'}
    cfg_text=json.dumps(config,sort_keys=True,separators=(',',':'))
    metadata={'config':config,'config_sha256':sha256(cfg_text.encode()).hexdigest(),
              'records':counts,'manifests_sha256':hashes,'unique_scene_graphs':len(taken)}
    (out/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    return metadata

def read_rows(path:Path):
    with open(path,encoding='utf8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=PROJECT_ROOT / 'outputs/dataset')
    ap.add_argument('--seed',type=int,default=20261025)
    ap.add_argument('--train-per-level',type=int,default=600)
    ap.add_argument('--eval-per-level',type=int,default=120)
    ap.add_argument('--adapt-train',type=int,default=128)
    ap.add_argument('--adapt-test',type=int,default=240)
    ap.add_argument('--materialize-all',action='store_true')
    a=ap.parse_args()
    print(json.dumps(generate(a.out,a.seed,a.train_per_level,a.eval_per_level,a.adapt_train,a.adapt_test,a.materialize_all),indent=2))
