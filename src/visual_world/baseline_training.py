"""Original multimodal Transformer training, smoke tests, baselines, and adaptation.

Scientific boundary: Never read id_test, ood_test or adapt_test from this module.
The preliminary study uses development sets only. No interpretation as final study results.
"""
import argparse
from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.visual_world.generate import digest
from src.visual_world.learning import TinyViTQuestion,VQADataset,make_loader,set_seed,evaluate,environment

REQUIRED = ('train.jsonl','id_dev.jsonl','ood_dev.jsonl','adapt_train.jsonl')
DEFAULT_DATA = PROJECT_ROOT / 'outputs' / 'dataset'
DEFAULT_OUT = PROJECT_ROOT / 'outputs' / 'experiments' / 'transformer_baseline'

def validate(root):
    meta=json.loads((root/'dataset_metadata.json').read_text())
    audit=json.loads((root/'audit_report.json').read_text())
    if not audit.get('pass'):raise RuntimeError('synthetic dataset generation audit must pass before model feasibility experiments')
    for name in REQUIRED:
        if digest(root/name)!=meta['manifests_sha256'][name]:raise RuntimeError(f'Split manifest changed: {name}')
    return meta

def device_for(name):
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but torch.cuda.is_available() is false. Install a compatible PyTorch CUDA wheel and update NVIDIA driver.')
    return torch.device('cuda' if name=='auto' and torch.cuda.is_available() else 'cpu' if name=='auto' else name)

def model_config(args):return {'dim':args.dim,'heads':args.heads,'vit_depth':args.vit_depth,'question_depth':args.question_depth,'dropout':0.0}

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,default=str)+'\n',encoding='utf8')

def train_epoch(model,loader,optimizer,device,amp=False,scaler=None):
    model.train();total_loss=0;correct=0;n=0
    for x,q,y,_ in loader:
        x=x.to(device,non_blocking=True);q=q.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
            logits=model(x,q)
            loss=nn.functional.cross_entropy(logits,y)
        if scaler is not None and amp:
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer);scaler.update()
        else:
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        n+=len(y);total_loss+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item()
    return {'loss':total_loss/n,'accuracy':correct/n}

def build_sets(root,mode='full',batch=64,seed=42,num_workers=0,levels=None):
    train=VQADataset(root/'train.jsonl',mode,levels=levels)
    id_dev=VQADataset(root/'id_dev.jsonl',mode,levels=levels)
    ood_dev=VQADataset(root/'ood_dev.jsonl',mode,levels=levels)
    return train,make_loader(train,batch,True,seed,num_workers),make_loader(id_dev,batch),make_loader(ood_dev,batch)

def train(args,mode='full'):
    root=Path(args.data);tag=mode if args.levels=='all' else f'{mode}-levels-{args.levels}'
    out=Path(args.out)/tag;out.mkdir(parents=True,exist_ok=True)
    meta=validate(root);set_seed(args.seed)
    if args.cpu_threads>0:torch.set_num_threads(args.cpu_threads)
    device=device_for(args.device)
    print('Environment:',json.dumps(environment()))
    config=model_config(args)
    levels=None if args.levels=='all' else tuple(int(x) for x in args.levels.split(','))
    if levels is not None and (not set(levels) or not set(levels).issubset({1,2,3})):raise ValueError('levels must be 1,2,3 or all')
    ds,train_loader,id_loader,ood_loader=build_sets(root,mode,args.batch_size,args.seed,args.num_workers,levels)
    model=TinyViTQuestion(**config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    amp=device.type=='cuda' and args.amp
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    best=-1;history=[];start=time.time()
    for epoch in range(1,args.epochs+1):
        tr=train_epoch(model,train_loader,optimizer,device,amp,scaler)
        id_metrics=evaluate(model,id_loader,device)
        ood_metrics=evaluate(model,ood_loader,device)
        row={'epoch':epoch,'train_loss':tr['loss'],'train_accuracy':tr['accuracy'],'id_dev_accuracy':id_metrics['accuracy'],'ood_dev_accuracy':ood_metrics['accuracy'],'id_dev_by_level':id_metrics['by_level'],'ood_dev_by_level':ood_metrics['by_level']}
        history.append(row)
        print(f'{mode:9} ep {epoch:3d}/{args.epochs} train {tr["accuracy"]:.3f} id_dev {id_metrics["accuracy"]:.3f} ood_dev {ood_metrics["accuracy"]:.3f}',flush=True)
        if id_metrics['accuracy']>best:
            best=id_metrics['accuracy']
            torch.save({'model':model.state_dict(),'model_config':config,'seed':args.seed,'mode':mode,'best_epoch':epoch,'id_dev_accuracy':best,'train_manifest_sha256':meta['manifests_sha256']['train.jsonl']},out/'best.pt')
    result={'status':'completed','mode':mode,'seed':args.seed,'model_config':config,'training':vars(args),'dataset_hashes':{name:meta['manifests_sha256'][name] for name in REQUIRED},'best_id_dev_accuracy':best,'duration_sec':round(time.time()-start,2),'device':str(device),'environment':environment(),'history':history,'IMPORTANT':'preliminary study development results ONLY; no test splits read, no hypothesis claim.'}
    write_json(out/'metrics.json',result)
    print('Saved',out/'metrics.json','and',out/'best.pt')
    return result

def smoke(args):
    root=Path(args.data);validate(root);set_seed(args.seed)
    if args.cpu_threads>0:torch.set_num_threads(args.cpu_threads)
    device=device_for(args.device)
    sample=VQADataset(root/'train.jsonl',limit=8)
    loader=make_loader(sample,4,False)
    model=TinyViTQuestion(dim=32,heads=4,vit_depth=1,question_depth=1).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    x,q,y,_=next(iter(loader))
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        pred=model(x.to(device),q.to(device))
        loss=nn.functional.cross_entropy(pred,y.to(device))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite smoke loss')
        loss.backward();opt.step()
    result={'status':'passed','smoke_loss':round(loss.item(),6),'device':str(device),'environment':environment(),'train_examples_used':8,'steps':2}
    write_json(Path(args.out)/'smoke_result.json',result);print(json.dumps(result,indent=2))
    return result

def overfit(args):
    root=Path(args.data);validate(root);set_seed(args.seed)
    if args.cpu_threads>0:torch.set_num_threads(args.cpu_threads)
    device=device_for(args.device);ds=VQADataset(root/'train.jsonl',limit=args.samples)
    loader=make_loader(ds,args.samples,False)
    model=TinyViTQuestion(**model_config(args)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0)
    history=[]
    for step in range(1,args.steps+1):
        m=train_epoch(model,loader,opt,device)
        if step==1 or step%10==0 or m['accuracy']>=.97:print(f'overfit step={step} accuracy={m["accuracy"]:.3f} loss={m["loss"]:.4f}',flush=True)
        history.append({'step':step,**m})
        if m['accuracy']>=.97:break
    status='passed' if history[-1]['accuracy']>=.97 else 'not_yet_overfit'
    result={'status':status,'samples':args.samples,'steps_run':len(history),'last':history[-1],'history':history,'note':'Engineering learnability check, not a scientific result.'}
    write_json(Path(args.out)/'overfit_result.json',result)
    print('Overfit check:',status)
    return result

def stratified_adaptation_indices(ds,seed=1919,dev_per_class=16):
    import random
    rng=random.Random(seed)
    groups={0:[],1:[]}
    for i,y in enumerate(ds.labels):groups[y].append(i)
    if any(len(v)<=dev_per_class for v in groups.values()):raise ValueError('Too few adaptation rows for stratified split')
    dev=[];train=[]
    for cls in (0,1):
        rng.shuffle(groups[cls]);dev+=groups[cls][:dev_per_class];train+=groups[cls][dev_per_class:]
    return sorted(train),sorted(dev)

def adapt(args):
    root=Path(args.data);meta=validate(root)
    if args.cpu_threads>0:torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed);device=device_for(args.device)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ckpt['mode']!='full':raise RuntimeError('Adaptation requires the full multimodal baseline checkpoint')
    if ckpt['train_manifest_sha256']!=meta['manifests_sha256']['train.jsonl']:raise RuntimeError('Checkpoint is from a different dataset manifest')
    # Our only source for adaptation is adapt_train, internally split 96 train/32 dev.
    ds=VQADataset(root/'adapt_train.jsonl')
    train_idx,dev_idx=stratified_adaptation_indices(ds)
    adapt_dev=DataLoader(Subset(ds,dev_idx),batch_size=args.batch_size)
    id_dev=make_loader(VQADataset(root/'id_dev.jsonl'),args.batch_size)
    results=[]
    for variant in ('frozen-vision','full-finetune'):
        set_seed(args.seed)
        adapt_train=DataLoader(Subset(ds,train_idx),batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        model=TinyViTQuestion(**ckpt['model_config']).to(device)
        model.load_state_dict(ckpt['model'])
        model.reset_new_operator(seed=args.new_operator_seed)
        if variant=='frozen-vision':model.freeze_visual()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=0.0)
        steps=set(int(s) for s in args.steps.split(','))
        if min(steps)<0:raise ValueError('Steps must be >= 0')
        state=iter(adapt_train)
        for step in range(max(steps)+1):
            if step in steps:
                novel=evaluate(model,adapt_dev,device);retained=evaluate(model,id_dev,device)
                row={'variant':variant,'step':step,'adapt_dev_accuracy':novel['accuracy'],'id_dev_accuracy':retained['accuracy'],'adapt_dev_loss':novel['loss']}
                results.append(row);print(f'{variant:15} step {step:4d} adapt_dev={novel["accuracy"]:.3f} old_id_dev={retained["accuracy"]:.3f}',flush=True)
            if step==max(steps):break
            try:x,q,y,_=next(state)
            except StopIteration:state=iter(adapt_train);x,q,y,_=next(state)
            model.train();opt.zero_grad(set_to_none=True)
            logits=model(x.to(device),q.to(device));loss=nn.functional.cross_entropy(logits,y.to(device))
            loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);opt.step()
    output=Path(args.out)/'adaptation';output.mkdir(parents=True,exist_ok=True)
    with (output/'adaptation_curves.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    write_json(output/'adaptation_report.json',{'status':'completed','checkpoint':str(args.checkpoint),'steps':sorted(steps),'adapt_train_n':len(train_idx),'adapt_dev_n':len(dev_idx),'adapt_split_seed':1919,'new_operator_seed':args.new_operator_seed,'results':results,'note':'Not a pure frozen-probe: frozen-vision freezes the image encoder but trains question/fusion/head. CLOSER initialized equally. NEVER read adapt_test.'})
    return results

def arguments():
    p=argparse.ArgumentParser(description='Multimodal baseline (no final-test access)')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('smoke','overfit','train','baselines','adapt'):
        x=sub.add_parser(name)
        x.add_argument('--data',type=Path,default=DEFAULT_DATA)
        x.add_argument('--out',type=Path,default=DEFAULT_OUT)
        x.add_argument('--seed',type=int,default=42)
        x.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
        x.add_argument('--cpu-threads',type=int,default=4)
        x.add_argument('--batch-size',type=int,default=64)
        x.add_argument('--lr',type=float,default=3e-4)
        x.add_argument('--dim',type=int,default=96)
        x.add_argument('--heads',type=int,default=4)
        x.add_argument('--vit-depth',type=int,default=3)
        x.add_argument('--question-depth',type=int,default=2)
        if name in ('train','baselines'):
            x.add_argument('--epochs',type=int,default=40)
            x.add_argument('--levels',type=str,default='all',help='diagnostic subset, e.g. 1 or 1,2; default all')
            x.add_argument('--weight-decay',type=float,default=0.01)
            x.add_argument('--num-workers',type=int,default=0)
            x.add_argument('--amp',action='store_true',help='Use mixed precision with CUDA; optional')
        if name=='overfit':
            x.add_argument('--samples',type=int,default=16)
            x.add_argument('--steps',type=int,default=120)
        if name=='adapt':
            x.add_argument('--checkpoint',type=Path,default=DEFAULT_OUT/'full'/'best.pt')
            x.add_argument('--steps',type=str,default='0,10,50,100,250')
            x.add_argument('--new-operator-seed',type=int,default=777)
    return p.parse_args()

def main():
    args=arguments()
    if args.command=='smoke':smoke(args)
    elif args.command=='overfit':overfit(args)
    elif args.command=='train':train(args,'full')
    elif args.command=='baselines':
        for mode in ('text-only','image-only'):train(args,mode)
    elif args.command=='adapt':adapt(args)
if __name__=='__main__':main()
