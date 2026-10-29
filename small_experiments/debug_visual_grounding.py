"""Diagnose original visual grounding (does not read final test splits).

Run from the project root:
    python -m small_experiments.debug_visual_grounding --device cuda --phase paired
    python -m small_experiments.debug_visual_grounding --device cuda --phase template --template color
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from src.visual_world.learning import TinyViTQuestion,VQADataset,set_seed,evaluate,environment
from src.visual_world.baseline_training import validate,train_epoch
from src.visual_world.render import PALETTE


def get_loader(ds,batch,shuffle=False,seed=42):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=torch.cuda.is_available(),generator=torch.Generator().manual_seed(seed))


def model_new(device):
    return TinyViTQuestion(dim=96,heads=4,vit_depth=3,question_depth=2).to(device)


def balanced_same_question_indices(ds):
    """Build 16 examples: four queries, two correct and two incorrect examples per query."""
    groups=defaultdict(list)
    for i,row in enumerate(ds.rows):
        if row['template']=='color':groups[(tuple(row['query_tokens']),int(row['label']))].append(i)
    chosen=[]
    for color in ('red','blue','green','yellow'):
        query=('EXISTS_COLOR',color)
        for label in (0,1):
            candidates=groups[(query,label)]
            if len(candidates)<2:raise RuntimeError(f'Insufficient paired samples for {query}: {label}')
            chosen.extend(candidates[:2])
    assert len(chosen)==16
    return chosen


def pixel_oracle(ds):
    preds=[]
    for i,row in enumerate(ds.rows):
        if row['template']!='color':continue
        rgb=np.array(PALETTE[row['query_tokens'][1]],dtype=np.int32)
        img=ds.images[i].reshape(-1,3).astype(np.int32)
        yhat=int(np.min(np.sum((img-rgb)**2,axis=1)) < 40**2)
        preds.append(int(yhat==row['label']))
    return {'n':len(preds),'accuracy':sum(preds)/len(preds)} if preds else None


def paired(root,device,steps,lr,seed):
    results={}
    for mode in ('full','text-only'):
        set_seed(seed)
        ds=VQADataset(root/'train.jsonl',mode=mode,levels=(1,))
        ids=balanced_same_question_indices(ds)
        subset=Subset(ds,ids)
        dl=get_loader(subset,len(subset))
        m=model_new(device)
        opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.0)
        trace=[]
        for step in range(1,steps+1):
            train_epoch(m,dl,opt,device)
            if step in {1,3,5,10,20,40,80,120,160,240,steps}:
                metrics=evaluate(m,dl,device)
                row={'step':step,'accuracy':metrics['accuracy'],'loss':metrics['loss']}
                trace.append(row)
                print(f'paired {mode:9} step {step:3d} accuracy={metrics["accuracy"]:.3f} loss={metrics["loss"]:.4f}',flush=True)
        results[mode]={'samples':len(subset),'first16_ids':[ds.rows[i]['id'] for i in ids],'history':trace,'last':trace[-1]}
    return results


def templates(root,device,template,epochs,lr,seed,batch):
    result={}
    for mode in ('full','text-only'):
        set_seed(seed)
        ds=VQADataset(root/'train.jsonl',mode=mode,levels=(1,))
        dv=VQADataset(root/'id_dev.jsonl',mode=mode,levels=(1,))
        train_ids=[i for i,r in enumerate(ds.rows) if r['template']==template]
        dev_ids=[i for i,r in enumerate(dv.rows) if r['template']==template]
        tr=Subset(ds,train_ids); va=Subset(dv,dev_ids)
        dl=get_loader(tr,batch,True,seed); tr_dl=get_loader(tr,batch);dv_dl=get_loader(va,batch)
        m=model_new(device);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.0)
        trace=[]
        for epoch in range(1,epochs+1):
            train_epoch(m,dl,opt,device)
            tr_m=evaluate(m,tr_dl,device);dv_m=evaluate(m,dv_dl,device)
            row={'epoch':epoch,'train_accuracy':tr_m['accuracy'],'train_loss':tr_m['loss'],'id_dev_accuracy':dv_m['accuracy'],'id_dev_loss':dv_m['loss']}
            trace.append(row)
            if epoch in (1,2,5,10) or epoch%10==0 or epoch==epochs:
                print(f'template {mode:9} epoch {epoch:3d} train={tr_m["accuracy"]:.3f} dev={dv_m["accuracy"]:.3f} loss={tr_m["loss"]:.4f}',flush=True)
        result[mode]={'train_n':len(tr),'id_dev_n':len(va),'best_train_accuracy':max(h['train_accuracy'] for h in trace),'best_id_dev_accuracy':max(h['id_dev_accuracy'] for h in trace),'history':trace}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=PROJECT_ROOT / 'outputs/dataset')
    p.add_argument('--out',type=Path,default=PROJECT_ROOT / 'outputs/experiments/visual_grounding_debug/color_diagnostic.json')
    p.add_argument('--phase',choices=('paired','template','both'),default='paired')
    p.add_argument('--template',choices=('color','shape','pair'),default='color')
    p.add_argument('--paired-steps',type=int,default=120)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    a=p.parse_args()
    validate(a.data)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(min(torch.get_num_threads(),4))
    dev=torch.device(a.device)
    start=time.time()
    tr=VQADataset(a.data/'train.jsonl',levels=(1,))
    dv=VQADataset(a.data/'id_dev.jsonl',levels=(1,))
    report={'status':'diagnostic_only','phase':a.phase,'template':a.template,'environment':environment(),
            'pixel_color_oracle_train':pixel_oracle(tr),'pixel_color_oracle_id_dev':pixel_oracle(dv),
            'runs':{},'disclaimer':'Uses train and ID-dev only. Not evidence of scientific curriculum effects.'}
    if a.phase in ('paired','both'):
        report['runs']['paired']=paired(a.data,dev,a.paired_steps,a.lr,a.seed)
    if a.phase in ('template','both'):
        report['runs']['template']=templates(a.data,dev,a.template,a.epochs,a.lr,a.seed,a.batch_size)
    report['duration_sec']=round(time.time()-start,2)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf8')
    print('Saved',a.out)

if __name__=='__main__':main()
