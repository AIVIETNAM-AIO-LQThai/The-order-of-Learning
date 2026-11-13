"""Five-seed diagnostic study: original model unchanged; development data only."""
from pathlib import Path
import hashlib, math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SCHEDULES = ("mixed", "prereq_mixed_pair_focus")
SEEDS = (42,43,44,45,46)
STEPS = (5400,7200,9000)


def experiment_grid():
    return [(s,seed,step) for s in SCHEDULES for seed in SEEDS for step in STEPS]


def experiment_tag(schedule,seed,step):
    if schedule not in SCHEDULES or seed not in SEEDS or step not in STEPS:
        raise ValueError(f"Unexpected checkpoint {schedule}/{seed}/{step}")
    return f"{schedule}_seed{seed}_step{step}"


def validate_manifest(root):
    root=Path(root)
    path=root/"outputs/medium_world/experiments/representation_diagnostics/five_seeds/checkpoint_manifest.csv"
    if not path.is_file(): raise FileNotFoundError(f"Missing 30-checkpoint manifest: {path}")
    m=pd.read_csv(path)
    for name in ("schedule","seed","step","checkpoint_sha256"):
        if name not in m: raise ValueError(f"Manifest missing column {name}")
    actual=list(zip(m.schedule,m.seed,m.step))
    if len(actual)!=30 or len(set(actual))!=30 or set(actual)!=set(experiment_grid()):
        raise ValueError("The checkpoint manifest must have exactly 2 curricula × 5 seeds × 3 steps")
    if m.checkpoint_sha256.isna().any() or not m.checkpoint_sha256.astype(str).str.fullmatch(r"[a-fA-F0-9]{64}").all():
        raise ValueError("Manifest has missing or invalid SHA-256 hashes")
    # locate_checkpoint checks SHA-256 as well as existence; done before any model use.
    from src.medium_visual_world.diagnostic_followup import locate_checkpoint
    for schedule,seed,step in experiment_grid():
        locate_checkpoint(root,schedule,seed,step,path,verify_sha=True)
    return path


def verified_sha(root,schedule,seed,step,manifest):
    from src.medium_visual_world.diagnostic_followup import locate_checkpoint
    path=locate_checkpoint(root,schedule,seed,step,manifest,verify_sha=True)
    with open(path,"rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def ensure_run_file(path, required, keyvals, n=None):
    path=Path(path)
    if not path.is_file(): return None
    df=pd.read_csv(path)
    if df.empty: raise ValueError(f"Empty existing output: {path}")
    if any(c not in df.columns for c in required):
        raise ValueError(f"Existing CSV lacks expected columns: {path}")
    for k,v in keyvals.items():
        if k not in df or not df[k].eq(v).all():
            raise ValueError(f"Stale or wrong saved result ({k}): {path}")
    if n is not None and len(df)!=n:
        raise ValueError(f"Unexpected row count in {path}: got {len(df)}, expected {n}")
    return df


def atomic_csv(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    df.to_csv(temp,index=False)
    temp.replace(path)


def mean_ci(values):
    d=np.asarray(values,dtype=float);d=d[np.isfinite(d)]
    n=len(d)
    if not n: return dict(n_seeds=0,mean=np.nan,sd=np.nan,ci95_low=np.nan,ci95_high=np.nan,positive_seeds=0)
    avg=float(d.mean());sd=float(d.std(ddof=1)) if n>1 else np.nan
    half=float(student_t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else np.nan
    return dict(n_seeds=n,mean=avg,sd=sd,ci95_low=avg-half,ci95_high=avg+half,positive_seeds=int((d>0).sum()))


def paired_checkpoint_differences(summary,metrics):
    if not {(s,seed,step) for s,seed,step in zip(summary.schedule,summary.seed,summary.step)}==set(experiment_grid()):
        raise ValueError("Need all 30 checkpoints for paired inference; no partial aggregate")
    out=[]
    for metric in metrics:
        w=summary.pivot(index="seed",columns=["schedule","step"],values=metric)
        for sched in SCHEDULES:
            for a,b in ((5400,7200),(7200,9000),(5400,9000)):
                diff=w[(sched,b)]-w[(sched,a)]
                out.append(dict(metric=metric,contrast=f"{sched}: {a} to {b}",**mean_ci(diff)))
        for step in STEPS:
            diff=w[("prereq_mixed_pair_focus",step)]-w[("mixed",step)]
            out.append(dict(metric=metric,contrast=f"prereq minus mixed at {step}",**mean_ci(diff)))
    return pd.DataFrame(out)


def swap_metrics(df):
    from src.medium_visual_world.diagnostic_followup import binary_metrics
    if not df.label_before.isin((0,1)).all() or not df.label_after.eq(1-df.label_before).all():
        raise AssertionError("Corrupted before/after label flips")
    by={}
    for label in (0,1):
        sub=df[df.label_before==label]
        if sub.empty:raise AssertionError("Swap test needs both original label classes")
        by[label]=sub
    metrics=dict(n=len(df),n_before_no=len(by[0]),n_before_yes=len(by[1]),
                 both_correct=float(df.both_correct.mean()),correct_direction=float(df.correct_direction.mean()),
                 before_accuracy=float(df.before_correct.mean()),after_accuracy=float(df.after_correct.mean()),
                 mean_signed_delta=float(((df.p_yes_after-df.p_yes_before)*(df.label_after-df.label_before)).mean()))
    for label,part in by.items():
        word="before_yes" if label else "before_no"
        metrics[f"both_{word}"]=float(part.both_correct.mean())
        metrics[f"direction_{word}"]=float(part.correct_direction.mean())
        metrics[f"before_accuracy_{word}"]=float(part.before_correct.mean())
        metrics[f"after_accuracy_{word}"]=float(part.after_correct.mean())
    metrics["balanced_both_correct"]=(metrics["both_before_yes"]+metrics["both_before_no"])/2
    metrics["balanced_direction"]=(metrics["direction_before_yes"]+metrics["direction_before_no"])/2
    return metrics


def paired_accuracy_changes(df,split_by):
    """Within-run change table, matched question IDs; descriptive, not seed-independent CI."""
    out=[]
    for keys,part in df.groupby(split_by):
        if not isinstance(keys,tuple):keys=(keys,)
        m=dict(zip(split_by,keys))
        y=part.true_label.to_numpy();pred=part.pred_original_frozen_model.to_numpy();alt=part.pred_spatial_conv_probe.to_numpy()
        out.append(dict(**m,n=len(part),fixed=int(((pred!=y)&(alt==y)).sum()),broken=int(((pred==y)&(alt!=y)).sum())))
    return pd.DataFrame(out)


def task_head_competition_controlled(model,train_dataset,dev_dataset,device="cpu",seed=42,epochs=40,batch_size=120):
    """Frozen exact-8 inputs, with explicitly task-aware and parameter-matched controls.

    Unlike the original pilot: compare shared 8-input, task-routed shared 12-input,
    task-routed width-94 (~1412 params), and 4 separate width-32 (~1416 params).
    A width match does NOT match optimization dynamics or function class.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F
    from src.medium_visual_world.learning import TASKS
    from src.medium_visual_world.diagnostic_followup import original_eight_features,binary_metrics
    def arrays(ds):
        x=original_eight_features(model,ds.images[ds.image_indices].numpy(),ds.queries.numpy(),device)
        task=np.array([TASKS.index(row["template"]) for row in ds.rows],dtype=np.int64)
        return x,task,ds.labels.numpy(),[str(row["id"]) for row in ds.rows]
    x,t,y,ids_train=arrays(train_dataset)
    xd,td,yd,ids_dev=arrays(dev_dataset)
    class Heads(nn.Module):
        def __init__(self,mode):
            super().__init__();self.mode=mode
            if mode=="separate_heads_32":
                self.nets=nn.ModuleList([nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,2)) for _ in range(4)])
            else:
                width=94 if mode=="task_aware_capacity_matched_94" else 32
                self.net=nn.Sequential(nn.Linear(12 if mode.startswith("task_aware") else 8,width),nn.GELU(),nn.Linear(width,2))
        def forward(self,a,task):
            if self.mode=="separate_heads_32":
                out=torch.empty((len(a),2),device=a.device,dtype=a.dtype)
                for i in range(4):
                    mask=task==i
                    if mask.any():out[mask]=self.nets[i](a[mask])
                return out
            if self.mode.startswith("task_aware"):
                a=torch.cat((a,F.one_hot(task,4).to(a.dtype)),dim=1)
            return self.net(a)
    modes=("shared_unaware_32","task_aware_32","task_aware_capacity_matched_94","separate_heads_32")
    pred={}
    with torch.no_grad():
        pred["original_pretrained_head"]=np.concatenate([model.head(torch.as_tensor(xd[i:i+batch_size],device=device)).argmax(1).cpu().numpy() for i in range(0,len(xd),batch_size)])
    parameter_counts={"original_pretrained_head":sum(v.numel() for v in model.head.parameters())}
    for mode in modes:
        # Strictly identical initial seeds, minibatch permutations, epochs, labels.
        torch.manual_seed(seed)
        net=Heads(mode).to(device)
        parameter_counts[mode]=sum(v.numel() for v in net.parameters())
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)
        xx=torch.from_numpy(x);tt=torch.from_numpy(t);yy=torch.from_numpy(y)
        xdev=torch.from_numpy(xd);tdev=torch.from_numpy(td)
        for epoch in range(epochs):
            net.train()
            for chunk in torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+epoch)).split(batch_size):
                opt.zero_grad(set_to_none=True)
                logits=net(xx[chunk].to(device),tt[chunk].to(device))
                loss=F.cross_entropy(logits,yy[chunk].to(device));loss.backward();opt.step()
        net.eval();pieces=[]
        with torch.no_grad():
            for i in range(0,len(xd),batch_size):
                pieces.append(net(xdev[i:i+batch_size].to(device),tdev[i:i+batch_size].to(device)).argmax(1).cpu().numpy())
        pred[mode]=np.concatenate(pieces)
    details=pd.DataFrame({"example_id":ids_dev,"task":[TASKS[i] for i in td],"true_label":yd})
    for mode,p in pred.items():details[f"pred_{mode}"]=p
    summary=[]
    for task_i,name in enumerate(TASKS):
        mask=td==task_i
        for mode,p in pred.items():
            summary.append(dict(readout=mode,task=name,params=parameter_counts[mode],
                                seed=seed,**binary_metrics(yd[mask],p[mask])))
    # Shortcut subgroup, assessed as a specificity-only negative slice.
    missing_shape_indices=[]
    for i,row in enumerate(dev_dataset.rows):
        if row["template"]!="conjunction":continue
        color,shape=row["question"]["args"]
        objs=row["scene"]["objects"]
        color_present=any(o["color"]==color for o in objs)
        shape_present=any(o["shape"]==shape for o in objs)
        if int(row["label"])!=int(color_present and shape_present):
            raise AssertionError("Conjunction oracle mismatch")
        if not row["label"] and color_present and not shape_present:
            missing_shape_indices.append(i)
    if not missing_shape_indices:
        raise ValueError("No Color-present, Shape-missing Conjunction negatives")
    for mode,p in pred.items():
        summary.append(dict(readout=mode,task="conjunction_color_present_shape_missing",
                            params=parameter_counts[mode],seed=seed,
                            **binary_metrics(yd[missing_shape_indices],p[missing_shape_indices])))
    base=pd.DataFrame(summary)
    return base.reset_index(drop=True),details


def validate_frozen_datasets(root):
    """Match training/dev JSONL bytes to the original phase-aware LR run manifest."""
    import json
    root=Path(root)
    report=root/"outputs/medium_world/experiments/phase_aware_lr/schedule_comparison.json"
    if not report.is_file(): raise FileNotFoundError(f"Original run report missing: {report}")
    saved=json.loads(report.read_text(encoding="utf-8"))
    hashes=saved.get("config",{}).get("dataset_hashes")
    if not hashes or not isinstance(hashes,dict):
        raise ValueError("Original LR report lacks the frozen dataset hashes")
    files=("train.jsonl","id_dev.jsonl","ood_dev.jsonl")
    for filename in files:
        expected=hashes.get(filename)
        if not expected:raise ValueError(f"Missing saved dataset hash for {filename}")
        path=root/"outputs/medium_world/dataset"/filename
        if not path.is_file():raise FileNotFoundError(path)
        with path.open("rb") as handle: actual=hashlib.file_digest(handle,"sha256").hexdigest()
        if actual!=expected:raise AssertionError(f"Dataset changed since original LR experiment: {filename}")
    return hashes
