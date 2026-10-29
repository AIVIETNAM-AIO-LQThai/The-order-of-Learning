"""Audits for symbolic semantics, holdout leakage, balance and distinct graphs."""
from collections import Counter,defaultdict
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

import argparse
import json
import math
from src.visual_world.world import COLORS,SHAPES,HOLDOUT,role_configs,configs_for,from_dict,validate_scene,spatial_truth,closer_margin
from src.visual_world.grammar import decode
from src.visual_world.oracle import evaluate
from src.visual_world.generate import DATASETS,read_rows,digest

def standalone_label(scene,question):
    # Deliberately independently implemented metadata interpreter for crosschecking.
    objects=list(scene.values()); op=question['op'];args=question['args']
    def obj(ref):
        found=[o for o in objects if (o['color'],o['shape'])==tuple(ref['args'])]
        assert len(found)==1
        return found[0]
    if op=='EXISTS_COLOR':return int(any(o['color']==args[0] for o in objects))
    if op=='EXISTS_SHAPE':return int(any(o['shape']==args[0] for o in objects))
    if op=='EXISTS_PAIR':return int(any((o['color'],o['shape'])==tuple(args) for o in objects))
    if op=='LEFT':return int(obj(args[0])['x']<obj(args[1])['x'])
    if op=='ABOVE':return int(obj(args[0])['y']<obj(args[1])['y'])
    if op=='XOR':return standalone_label(scene,args[0])^standalone_label(scene,args[1])
    if op=='AND':return standalone_label(scene,args[0])&standalone_label(scene,args[1])
    if op=='CLOSER':
        a,b,c=map(obj,args)
        d=lambda x,y:math.hypot(x['x']-y['x'],x['y']-y['y'])
        return int(d(a,b)<d(a,c))
    raise AssertionError(op)

def feasible_counts():
    conf=role_configs()
    id3=[x for x in conf if all(p not in HOLDOUT for p in x[:3])]
    id4=configs_for('id')
    op=configs_for('ood_pos');on=configs_for('ood_neg')
    stats={'all_configs':len(conf),'visible_all_allowed':len(id3),'all_four_allowed':len(id4),
           'heldout_anchor_configs':len(op),'heldout_missing_configs':len(on),
           'covered_allowed_pairs':len({p for row in id4 for p in row[:3]})}
    assert stats=={'all_configs':72,'visible_all_allowed':24,'all_four_allowed':12,
                   'heldout_anchor_configs':18,'heldout_missing_configs':12,'covered_allowed_pairs':9},stats
    assert all(sum(1 for row in op if row[0]==pair)==6 for pair in HOLDOUT)
    assert all(sum(1 for row in on if row[3]==pair)==4 for pair in HOLDOUT)
    return stats

def audit(root:Path):
    root=Path(root);meta=json.loads((root/'dataset_metadata.json').read_text())
    reports={'pass':True,'structural_feasibility':feasible_counts(),'splits':{},'checks':{},'cautions':[]}
    gids=set(); rows_by_split={}
    for split in DATASETS:
        p=root/(split+'.jsonl');rows=list(read_rows(p));rows_by_split[split]=rows
        assert digest(p)==meta['manifests_sha256'][p.name],f'hash mismatch {p}'
        assert len(rows)==meta['records'][split]
        counts=Counter();true_counts=Counter();pair_flags=Counter();seen_pairs=set();pt=Counter();holdout_seen=Counter();q_hash=Counter()
        for row in rows:
            s=from_dict(row['scene']);validate_scene(s)
            assert s.graph_id()==row['id']
            assert row['id'] not in gids,f'graph leakage {split} {row["id"]}'
            gids.add(row['id'])
            q=decode(row['question']);label=int(evaluate(s,q))
            assert label==row['label']==standalone_label(row['scene'],row['question'])
            assert q.op==row['question']['op']
            k=row['template'];counts[k]+=1;true_counts[k]+=label
            q_hash[tuple(row['query_tokens'])]+=1
            for o in s.objs():seen_pairs.add(o.pair())
            if split.startswith('ood'):
                h=tuple(row['heldout_pair']);assert h in HOLDOUT
                holdout_seen[h]+=1
                if k=='pair':
                    assert tuple(q.args)==h
                    if label:assert h in (o.pair() for o in s.objs())
                    else:
                        assert h not in (o.pair() for o in s.objs())
                        assert q.args[0] in {o.color for o in s.objs()}
                        assert q.args[1] in {o.shape for o in s.objs()}
                else:assert s.A.pair()==h
            else:
                assert all(o.pair() not in HOLDOUT for o in s.objs())
                t=row['query_tokens']
                assert not any((t[i],t[i+1]) in HOLDOUT for i in range(len(t)-1))
            if k=='xor':
                pq=spatial_truth(s);assert list(pq)==row['p_q'];pt[str(pq)]+=1
            if k=='closer':assert abs(closer_margin(s))>4
            if k=='pair' and not label:
                assert q.args[0] in {o.color for o in s.objs()}
                assert q.args[1] in {o.shape for o in s.objs()}
        for k,n in counts.items():
            frac=true_counts[k]/n
            assert abs(frac-.5)<=.01+1e-9,f'label imbalance {split}/{k} {frac:.3f}'
        if pt:
            vals=list(pt.values()); assert len(vals)==4
            assert max(vals)-min(vals)<=max(1,sum(vals)*.04),f'xor pattern imbalance {split}: {pt}'
        if split=='train':
            assert len(seen_pairs)==9
            assert all((c,s) in seen_pairs for c in COLORS for s in SHAPES if (c,s) not in HOLDOUT)
            assert counts['left'] and counts['above']
        if split.startswith('ood'):assert len(holdout_seen)==3
        reports['splits'][split]={'records':len(rows),'unique_graphs':len({row['id'] for row in rows}),
            'template_counts':dict(sorted(counts.items())),
            'positive_ratio':{k:round(true_counts[k]/n,4) for k,n in sorted(counts.items())},
            'xor_truth_pattern_counts':dict(sorted(pt.items())),
            'covered_holdout_pairs':{str(k):v for k,v in sorted(holdout_seen.items())},
            'covered_visible_pairs':len(seen_pairs),
            'unique_query_strings':len(q_hash)}
    assert len(gids)==meta['unique_scene_graphs']
    # Empirical *non-neural* text-only baseline. Learns majority class per
    # exact query string on train. This is diagnostic, not a full trained model.
    lookup=defaultdict(Counter)
    for r in rows_by_split['train']:
        lookup[tuple(r['query_tokens'])][r['label']]+=1
    summary={}
    for split in ('id_dev','id_test','ood_dev','ood_test'):
        by_template=defaultdict(list)
        for r in rows_by_split[split]:
            dist=lookup.get(tuple(r['query_tokens']))
            pred=dist.most_common(1)[0][0] if dist else 0
            by_template[r['template']].append(int(pred==r['label']))
        summary[split]={k:round(sum(v)/len(v),4) for k,v in sorted(by_template.items())}
    reports['non_neural_diagnostics']={'exact_query_majority_accuracy':summary,
        'note':'No image pixels used; unseen query strings default to label 0. Finite sample estimates only.'}

    # Paired interventions are verified if their gallery file has been generated.
    cf_path=root/'counterfactuals.json'
    cf_status='NOT_RUN'
    if cf_path.exists():
        pairs=json.loads(cf_path.read_text());assert len(pairs)>=8
        for pair in pairs:
            src=from_dict(pair['before_scene']);dst=from_dict(pair['after_scene'])
            validate_scene(src);validate_scene(dst)
            assert src.graph_id()==pair['original_id']
            assert dst.graph_id()==pair['intervened_graph_id']
            from src.visual_world.grammar import Query
            # Reference query was persisted in original JSONL; resolve by ID.
            src_row=next(r for r in rows_by_split[pair['split_source']] if r['id']==pair['original_id'])
            q=decode(src_row['question'])
            assert int(evaluate(src,q))==pair['original_label']
            assert int(evaluate(dst,q))==pair['new_label']
            assert pair['original_label']!=pair['new_label']
            if pair['template']=='xor':
                p0,q0=spatial_truth(src);p1,q1=spatial_truth(dst)
                assert p0!=p1 and q0==q1
            if pair['template'] in ('left','above','xor'):
                assert sorted((o.color,o.shape,o.size) for o in src.objs())==sorted((o.color,o.shape,o.size) for o in dst.objs())
        cf_status='PASS'
    reports['counterfactual_pairs_status']=cf_status
    reports['checks']={'scene_geometry':'PASS','oracle_crosscheck':'PASS','reference_grounding':'PASS',
         'graph_disjointness':'PASS','holdout_isolation':'PASS','primitive_coverage':'PASS',
         'per_template_answer_balance':'PASS','xor_truth_balance':'PASS',
         'closer_margin':'PASS','manifest_hashes':'PASS','role_feasibility':'PASS','counterfactual_integrity':cf_status}
    reports['cautions']=[
         'Query-only / image-only learned shortcut baselines are not measured until the model preliminary study.',
         'Balanced labels per template do not guarantee every exact query string has a 50:50 label prior.',
         'Matching fine-grained ID/OOD covariate marginals and paired counterfactual error rates requires subsequent audits.',
         'These are data/oracle checks, not evidence about representation learning or human cognition.']
    reports['pass']=all(v=='PASS' for v in reports['checks'].values())
    return reports

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=PROJECT_ROOT / 'outputs/dataset')
    args=ap.parse_args();result=audit(args.root)
    (args.root/'audit_report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
