"""Preflight integrity checks for the full five-seed diagnostic experiment.

No training and no pilot stage. Run from the repository root:
    python scripts/medium_world/audit_representation_probing.py
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from src.medium_visual_world.learning import AttributeBindingViT, MediumDataset
from src.medium_visual_world.world import COLORS, SHAPES, from_dict, validate_scene
from src.medium_visual_world.representation_probing import (
    oracle, distinct_scenes, same_scene_grid, predict_rows,
    consistency_report, unique_question_rows,
    counterfactual_rows, counterfactual_report,
    invariance_rows, invariance_report, image_tensor,
    visual_stages, extract_object_features, collect_dissection,
    foreground_concept_masks, signed_explanations,
    spray_cluster, make_explanation_questions, data_hash_audit,
)

EXPECTED_HASHES={
    'train.jsonl':'69deccacbac6db25f7ffc8165f01545483d856855282ea640997f6660ce27c26',
    'id_dev.jsonl':'ece4b831f1f7fe6efad42b18aca499f12c2d573baa76fc218864bb56fe24b11a',
    'ood_dev.jsonl':'208a7566cc2082fa4438ec0d003430d0927c54bf82c55bb40fa885ef4961e0aa',
}


def main():
    torch.set_num_threads(min(4,torch.get_num_threads()))
    path=ROOT/'outputs/medium_world/dataset'
    hashes=data_hash_audit(path,EXPECTED_HASHES)
    train=MediumDataset(path/'train.jsonl')
    id_dev=MediumDataset(path/'id_dev.jsonl')
    ood_dev=MediumDataset(path/'ood_dev.jsonl')
    assert (len(train),len(id_dev),len(ood_dev))==(10000,1024,400)
    sets=[{r['scene_id'] for r in data.rows} for data in (train,id_dev,ood_dev)]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    for row in id_dev.rows[:24]+ood_dev.rows[:24]:
        validate_scene(from_dict(row['scene']))
        args=row['question']['args']
        task=row['template']
        color=args[0] if task in ('color','conjunction','pair') else COLORS[0]
        shape=args[-1] if task in ('shape','conjunction','pair') else SHAPES[0]
        assert oracle(row['scene'],task,color,shape)==row['label']
    # Verify the scene renderer reproduces the exact pixel data used by the dataset.
    for data in (id_dev,ood_dev):
        for index in range(min(12,len(data))):
            pixels,_,_=data.batch([index],torch.device('cpu'))
            expected=image_tensor(data.rows[index]['scene'])
            assert torch.equal(pixels[0],expected), (
                'Renderer mismatch against existing dataset image',data.path,index)
    model=AttributeBindingViT(dim=48,patch_size=4,depth=1).eval()
    device=torch.device('cpu')
    grid=same_scene_grid(id_dev.rows,limit=5,seed=29)
    pred=predict_rows(model,grid,device,batch_size=20)
    assert len(grid)==5*len(COLORS)*len(SHAPES)*4
    assert len(unique_question_rows(pred))==5*(len(COLORS)+len(SHAPES)+2*len(COLORS)*len(SHAPES))
    assert len(consistency_report(pred))==5*len(COLORS)*len(SHAPES)
    changes=counterfactual_rows(id_dev.rows,limit=15,seed=31)
    pairs=counterfactual_report(predict_rows(model,changes,device,batch_size=20))
    control=invariance_rows(changes)
    invariant=invariance_report(predict_rows(model,control,device,batch_size=20))
    assert len(pairs)>0 and not invariant.empty
    assert (invariant.label_before==invariant.label_after).all()
    scenes=distinct_scenes(id_dev.rows,limit=8,seed=12)
    tensor=torch.stack([image_tensor(r['scene']) for r in scenes[:2]])
    stages=visual_stages(model,tensor)
    assert all(tensor.shape==(2,256,48) for tensor in stages.values())
    info,features=extract_object_features(model,scenes,device,batch_size=4)
    assert all(values.shape==(len(info),48) for values in features.values())
    masks=foreground_concept_masks(scenes[0]['scene'])
    assert len(masks)==len(COLORS)+len(SHAPES)
    dissection=collect_dissection(model,distinct_scenes(train.rows,limit=6),scenes[:5],device,batch_size=2)
    assert len(dissection)==3*48*len(masks) and dissection.iou.between(0,1).all()
    explanations=make_explanation_questions(id_dev.rows,scene_limit=32,seed=80)
    assert sum(r['label'] for r in explanations)==16
    assert all(not r['heldout_target'] for r in explanations)
    assert all(r['logic_case'] in ('present_pair','attributes_separate')
               for r in explanations)
    ood_explanations=make_explanation_questions(
        ood_dev.rows,scene_limit=32,seed=81,split='ood_dev')
    assert len(ood_explanations)==32 and sum(r['label'] for r in ood_explanations)==16
    assert all(r['heldout_target'] for r in ood_explanations)
    maps,_=signed_explanations(model,explanations,device,batch_size=8)
    assert maps.shape==(32,16,16) and np.isfinite(maps).all()
    nonzero=np.linalg.norm(maps.reshape(len(maps),-1),axis=1)>1e-10
    assert nonzero.sum()>=16
    labels,values,gaps,affinity=spray_cluster(maps[nonzero],n_clusters=3,neighbors=5,seed=42)
    assert len(labels)==sum(nonzero) and affinity.shape==(sum(nonzero),sum(nonzero))
    print('PASS: frozen dataset hashes, exact renderer, oracle, scene disjointness')
    print('PASS: repeated-question deduplication, same-scene logic, causal edits and invariance')
    print('PASS: actual model forward, feature shapes, channel IoU, signed maps and spectral clustering')
    print('FULL configuration: 5 seeds x 2 curricula x 3 checkpoints = 30 checkpoint evaluations')


if __name__=='__main__':main()
