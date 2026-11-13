"""Post-Submission-II diagnostic follow-up. No change to existing model/training.

Integrates with medium_world.learning, medium_world.render, and prediction_tracking.
Run from repository root via notebooks/medium_world. All outputs are diagnostic only.
"""
from pathlib import Path
from collections import Counter
import copy, hashlib, json, math, re
import numpy as np
import pandas as pd


def load_jsonl(path):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f"Dataset missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def question_attributes(row):
    if row["template"] != "pair": raise ValueError("Requires a Pair/Binding question")
    a = row["question"]["args"]
    if len(a) != 2: raise ValueError("Expected queried color and shape")
    return a[0], a[1]


def count_free(row):
    """Exactly one target-color and exactly one target-shape object; *not* shortcut-free."""
    if row["template"] != "pair": return False
    color, shape = question_attributes(row)
    objs = row["scene"]["objects"]
    return sum(o["color"] == color for o in objs) == 1 and sum(o["shape"] == shape for o in objs) == 1


def context_table(rows):
    entries = []
    for row in rows:
        if row["template"] != "pair": continue
        color, shape = question_attributes(row)
        objs = row["scene"]["objects"]
        nc = sum(o["color"] == color for o in objs)
        ns = sum(o["shape"] == shape for o in objs)
        actual = int(any(o["color"] == color and o["shape"] == shape for o in objs))
        if actual != int(row["label"]): raise AssertionError(f"Oracle mismatch {row['id']}")
        entries.append(dict(example_id=str(row["id"]),scene_id=row["scene_id"],label=int(row["label"]),
                            target_color=color,target_shape=shape,n_target_color=nc,n_target_shape=ns,
                            n_objects=len(objs),count_free=(nc == ns == 1)))
    table = pd.DataFrame(entries)
    if table.empty: raise ValueError("No Pair questions in split")
    if table.example_id.duplicated().any(): raise AssertionError("Duplicate question ID")
    return table


def binary_metrics(y, pred):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    if len(y) != len(pred) or not len(y): raise ValueError("Mismatched or empty predictions")
    if not set(np.unique(y)).issubset({0,1}) or not set(np.unique(pred)).issubset({0,1}):
        raise ValueError("Predictions and labels must be binary")
    tp = int(((y==1)&(pred==1)).sum());fn = int(((y==1)&(pred==0)).sum())
    tn = int(((y==0)&(pred==0)).sum());fp = int(((y==0)&(pred==1)).sum())
    tpr = tp/(tp+fn) if tp+fn else float('nan')
    tnr = tn/(tn+fp) if tn+fp else float('nan')
    return dict(n=len(y),yes=int(y.sum()),no=int((y==0).sum()),tp=tp,fn=fn,tn=tn,fp=fp,
                accuracy=float((y==pred).mean()),balanced_accuracy=(tpr+tnr)/2 if np.isfinite(tpr+tnr) else float('nan'),
                tpr=tpr,tnr=tnr,pred_yes=float(pred.mean()),always_no_accuracy=float((y==0).mean()))


def paired_t_summary(df, value="balanced_accuracy", group=("study","policy","schedule"), steps=(5400,7200,9000)):
    """Paired within-seed changes; 95% t intervals descriptive, not multiplicity corrected."""
    from scipy.stats import t
    out=[]
    for keys,part in df.groupby(list(group),dropna=False):
        wide=part.pivot_table(index="seed",columns="step",values=value,aggfunc="first")
        for a,b in zip(steps[:-1],steps[1:]):
            if a not in wide or b not in wide:continue
            diff=(wide[b]-wide[a]).dropna()
            if diff.empty:continue
            sd=float(diff.std(ddof=1)) if len(diff)>1 else float('nan')
            margin=t.ppf(.975,len(diff)-1)*sd/math.sqrt(len(diff)) if len(diff)>1 else float('nan')
            out.append(dict(zip(group,keys if isinstance(keys,tuple) else (keys,)),start=a,end=b,
                            n_seeds=len(diff),mean_change=float(diff.mean()),ci95_low=float(diff.mean()-margin),
                            ci95_high=float(diff.mean()+margin),positive_seeds=int((diff>0).sum())))
    return pd.DataFrame(out)


def _find_trace(run, report_dir):
    rel = run.get("prediction_trace")
    if not rel: return None
    p = Path(rel)
    if p.is_file(): return p
    q = report_dir / p
    return q if q.is_file() else None


def discover_runs(project_root, families=("phase_aware_lr","mixed_pair_control/five_seeds","scaffolded_spiral"),strict=True):
    """Read saved run metadata; do not invent traces when the repository lacks local files."""
    base = Path(project_root)/"outputs/medium_world/experiments"
    found=[];missing=[]
    for family in families:
        report = base/family/"schedule_comparison.json"
        if not report.is_file():
            missing.append(str(report));continue
        d=json.loads(report.read_text(encoding="utf-8"))
        for run in d.get("runs",[]):
            if run.get("status") != "completed":continue
            p=_find_trace(run,report.parent)
            if p is None:
                missing.append(f"Trace unavailable: {family} {run.get('schedule')} {run.get('policy','')} seed{run.get('seed')}")
                continue
            found.append(dict(study=family,policy=str(run.get("policy") or "historical"),
                              schedule=str(run["schedule"]),seed=int(run["seed"]),path=p,
                              report=report,run=run))
    if missing and strict:
        raise FileNotFoundError("Missing saved experiment sources:\n"+"\n".join(missing[:25])+
                                (f"\n... and {len(missing)-25} more" if len(missing)>25 else ""))
    if not found: raise FileNotFoundError("No completed run with local per-question NPZ trace.\n"+"\n".join(missing[:20]))
    return found,missing


def trace_curves(project_root, families=("phase_aware_lr","mixed_pair_control/five_seeds","scaffolded_spiral"),strict=True):
    from src.medium_visual_world.prediction_tracking import load_trace
    base=Path(project_root)/"outputs/medium_world/dataset"
    row_by_split={s:load_jsonl(base/(s+".jsonl")) for s in ("id_dev","ood_dev")}
    meta_by_split={s:context_table(rows) for s,rows in row_by_split.items()}
    reports,missing=discover_runs(project_root,families,strict=strict)
    results=[];seen=set();checks=[]
    for desc in reports:
        key=(desc["study"],desc["policy"],desc["schedule"],desc["seed"])
        if key in seen: raise AssertionError(f"Duplicate run: {key}")
        seen.add(key)
        trace=load_trace(desc["path"])
        for split,metadata in meta_by_split.items():
            if split not in trace["splits"]:raise ValueError(f"Missing {split} in {desc['path']}")
            block=trace["splits"][split]
            ids=np.asarray(block["ids"],dtype=str)
            if len(ids)!=len(set(ids)): raise AssertionError(f"Duplicate trace IDs in {desc['path']}")
            # Exact ID keyed match, never positional inference from unrelated runs.
            joined=pd.DataFrame(dict(example_id=ids,true=block["labels"].astype(int),index=np.arange(len(ids))))
            joined=joined.merge(metadata,on="example_id",how="left",validate="one_to_one",indicator=True)
            pair_rows=joined[joined["_merge"]=="both"]
            pair_ids=set(metadata.example_id)
            traced_tasks=set(ids[(block["tasks"]=="pair")])
            if traced_tasks!=pair_ids:raise AssertionError(f"Pair question ID mismatch in {desc['path']} ({split})")
            if not (pair_rows.true.to_numpy()==pair_rows.label.to_numpy()).all():
                raise AssertionError(f"Trace labels mismatch: {desc['path']} ({split})")
            mask=joined["count_free"].eq(True).to_numpy(dtype=bool)
            all_pair=np.isin(ids,list(pair_ids))
            if not np.any(mask):raise AssertionError(f"No count-free questions in {split}")
            checks.append(dict(study=desc["study"],policy=desc["policy"],schedule=desc["schedule"],
                               seed=desc["seed"],split=split,total=len(ids),pair=int(all_pair.sum()),count_free=int(mask.sum()),
                               count_free_yes=int(block["labels"][mask].sum()),trace_path=str(desc["path"])))
            for i,step in enumerate(block["steps"]):
                for subset,selection in (("count_free",mask),("all_pair",all_pair)):
                    m=binary_metrics(block["labels"][selection],block["predictions"][i][selection])
                    results.append(dict(study=desc["study"],policy=desc["policy"],schedule=desc["schedule"],
                                        seed=desc["seed"],split=split,step=int(step),subset=subset,**m))
    return pd.DataFrame(results),pd.DataFrame(checks),missing


def save_curve_analysis(project_root,output,families=("phase_aware_lr","mixed_pair_control/five_seeds","scaffolded_spiral"),strict=True):
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    curves,checks,missing=trace_curves(project_root,families,strict)
    curves.to_csv(output/"count_free_curves_by_seed.csv",index=False)
    checks.to_csv(output/"trace_integrity.csv",index=False)
    selected=curves[(curves.split=="id_dev") & (curves.subset=="count_free")]
    diffs=paired_t_summary(selected)
    diffs.to_csv(output/"count_free_stage_changes.csv",index=False)
    for (study,policy),group in selected.groupby(["study","policy"]):
        fig,ax=plt.subplots(figsize=(8,4.6))
        for name,sub in group.groupby("schedule"):
            means=sub.groupby("step").balanced_accuracy.mean().sort_index()
            ax.plot(means.index,100*means.values,label=name)
        ax.axhline(50,color="grey",ls="--",lw=1,label="50% balanced chance")
        ax.set(xlabel="Optimizer updates",ylabel="Count-controlled ID Binding balanced accuracy (%)",
               title=f"{study} · {policy} (seed means)")
        ax.set_ylim(0,100);ax.legend(fontsize=7);fig.tight_layout()
        safe=re.sub('[^a-zA-Z0-9_-]','_',study+'__'+policy)
        fig.savefig(output/f"count_free_{safe}.png",dpi=180);plt.close(fig)
    (output/"missing_sources.json").write_text(json.dumps(missing,indent=2),encoding="utf-8")
    return curves,checks,diffs


def locate_checkpoint(project_root,schedule,seed,step,manifest=None,verify_sha=True):
    base=Path(project_root)/"outputs/medium_world/experiments/representation_diagnostics/five_seeds/checkpoints"
    path=base/f"{schedule}_constant_seed_{seed}_step_{step}.pt"
    expected=None
    if manifest is not None:
        m=pd.read_csv(manifest)
        sub=m[(m.schedule==schedule)&(m.seed==seed)&(m.step==step)]
        if len(sub)!=1:raise ValueError(f"Exactly one manifest row expected for {schedule}, {seed}, {step}")
        expected=str(sub.iloc[0].get("checkpoint_sha256",""))
        if not path.is_file():
            # Historical manifest may have absolute Windows paths; only use the safe basename.
            from pathlib import PureWindowsPath
            name=PureWindowsPath(str(sub.iloc[0]["path"])).name
            path=base/name
    if not path.is_file():raise FileNotFoundError(f"Checkpoint missing: {path}. No automatic retraining performed.")
    if verify_sha and expected and expected.lower()!="nan":
        with path.open("rb") as f:
            sha=hashlib.file_digest(f,"sha256").hexdigest()
        if sha!=expected: raise AssertionError(f"Checkpoint SHA mismatch: {path}")
    return path


def load_frozen_model(path,device="cpu"):
    import torch
    from src.medium_visual_world.learning import AttributeBindingViT
    # TorchVersion is the safe subclass present in old model-only metadata.
    safe=[torch.torch_version.TorchVersion] if hasattr(torch,"torch_version") else []
    with torch.serialization.safe_globals(safe):
        state=torch.load(path,map_location="cpu",weights_only=True)
    model_state=state.get("model",state.get("model_state_dict",state.get("state_dict",None)))
    if model_state is None or not isinstance(model_state,dict):
        raise ValueError(f"Expected checkpoint with 'model' state_dict: {path}")
    config=state.get("model_config",{"dim":48,"patch_size":4,"depth":1})
    if config!={"dim":48,"patch_size":4,"depth":1}:
        raise ValueError(f"Model config mismatch: {config}")
    model=AttributeBindingViT(**config)
    model.load_state_dict(model_state,strict=True)
    for p in model.parameters():p.requires_grad_(False)
    return model.to(device).eval(),state


def make_swap(row):
    """Deterministic label-flipping color swap; preserve all color/shape counts and geometry."""
    if not count_free(row):return None
    color,shape=question_attributes(row)
    before=copy.deepcopy(row["scene"])
    objs=before["objects"]
    a=next(i for i,o in enumerate(objs) if o["color"]==color)
    b=next(i for i,o in enumerate(objs) if o["shape"]==shape)
    if a==b:
        # Only one of queried shape; swapping target color to different shape flips Yes→No.
        choices=[i for i,o in enumerate(objs) if i!=a and o["color"]!=color and o["shape"]!=shape]
        if not choices:return None
        b=choices[0]
    # If a!=b, swap target color onto the sole target-shape object to flip No→Yes.
    if objs[a]["color"]==objs[b]["color"]:return None
    after=copy.deepcopy(before)
    after["objects"][a]["color"],after["objects"][b]["color"]=(objs[b]["color"],objs[a]["color"])
    oracle=lambda scene:int(any(o["color"]==color and o["shape"]==shape for o in scene["objects"]))
    y0,y1=oracle(before),oracle(after)
    if y0!=int(row["label"]) or y1!=1-y0:raise AssertionError("Swap did not flip Pair answer")
    assert Counter(o['color'] for o in before['objects'])==Counter(o['color'] for o in after['objects'])
    assert Counter(o['shape'] for o in before['objects'])==Counter(o['shape'] for o in after['objects'])
    return dict(before=before,after=after,label_before=y0,label_after=y1,swap_a=a,swap_b=b,
                example_id=str(row['id']),scene_id=row['scene_id'],target_color=color,target_shape=shape,
                n_objects=len(objs))


def run_swaps(model,rows,device="cpu",batch_size=48):
    import torch
    from src.medium_visual_world.world import from_dict
    from src.medium_visual_world.render import render
    from src.medium_visual_world.learning import encode_query
    swaps=[];x_before=[];x_after=[];queries=[]
    for row in rows:
        item=make_swap(row)
        if item is None:continue
        image_a=np.array(render(from_dict(item["before"])).convert("RGB"),dtype=np.uint8)
        image_b=np.array(render(from_dict(item["after"])).convert("RGB"),dtype=np.uint8)
        if (image_a==image_b).all():raise AssertionError("Swap did not affect pixels")
        x_before.append(image_a.transpose(2,0,1));x_after.append(image_b.transpose(2,0,1))
        queries.append(encode_query(row["query_tokens"]));swaps.append(item)
    if not swaps:raise ValueError("No eligible label-flipping swaps")
    def predict(images):
        logits=[]
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(images),batch_size):
                x=torch.tensor(np.stack(images[start:start+batch_size]),dtype=torch.float32,device=device).div_(255)
                q=torch.tensor(queries[start:start+batch_size],dtype=torch.long,device=device)
                logits.append(model(x,q).softmax(-1)[:,1].cpu().numpy())
        return np.concatenate(logits)
    p0,p1=predict(x_before),predict(x_after)
    result=[]
    for i,item in enumerate(swaps):
        m={k:v for k,v in item.items() if k not in ("before","after")}
        y0,y1=m["label_before"],m["label_after"]
        pr0,pr1=int(p0[i]>=.5),int(p1[i]>=.5)
        result.append(dict(**m,p_yes_before=float(p0[i]),p_yes_after=float(p1[i]),
                           pred_before=pr0,pred_after=pr1,
                           before_correct=int(pr0==y0),after_correct=int(pr1==y1),
                           both_correct=int(pr0==y0 and pr1==y1),
                           correct_direction=int((p1[i]-p0[i])*(y1-y0)>0)))
    return pd.DataFrame(result)


def pooled_features(model,images,device="cpu",batch_size=48):
    """Return trained model's patch features exactly as used by its original head."""
    import torch
    x=np.asarray(images)
    out=[];model.eval()
    with torch.inference_mode():
        for s in range(0,len(x),batch_size):
            batch=torch.as_tensor(x[s:s+batch_size],device=device,dtype=torch.float32).div(255)
            patches=model.patch(batch).flatten(2).transpose(1,2)
            v=model.pre_norm(patches)+0.5*model.post_norm(model.encoder(patches+model.pos))
            out.append(model.score_norm(v).cpu().numpy())
    return np.concatenate(out,axis=0)


def original_eight_features(model,images,queries,device="cpu",batch_size=48):
    """Exact pre-head 8 scalars, keeping learned query parameters frozen."""
    import torch
    result=[];model.eval()
    for start in range(0,len(images),batch_size):
        x=torch.as_tensor(np.asarray(images[start:start+batch_size]),device=device,dtype=torch.float32).div(255)
        q=torch.as_tensor(np.asarray(queries[start:start+batch_size]),device=device,dtype=torch.long)
        with torch.inference_mode():
            e=model.emb(q[:,1:4]);query=model.query(e.flatten(1))
            patches=model.patch(x).flatten(2).transpose(1,2)
            v=model.score_norm(model.pre_norm(patches)+.5*model.post_norm(model.encoder(patches+model.pos)))
            scale=math.sqrt(v.shape[-1])
            s1=torch.einsum("bnd,bd->bn",v,e[:,1])/scale
            s2=torch.einsum("bnd,bd->bn",v,e[:,2])/scale
            sj=torch.einsum("bnd,bd->bn",v,query)/scale
            inter=torch.tanh(s1)*torch.tanh(s2)
            f=torch.stack((s1.amax(1),s1.mean(1),s2.amax(1),s2.mean(1),
                           sj.amax(1),sj.mean(1),inter.amax(1),inter.mean(1)),dim=-1)
        result.append(f.cpu().numpy())
    return np.concatenate(result)


def unique_scene_items(dataset):
    """(scene_id, image[C,H,W], list of objects), once per unique scene."""
    first={}
    for row in dataset.rows:first.setdefault(row["scene_id"],row)
    scene_order=list(first)
    if len(scene_order)!=len(dataset.images):raise AssertionError("Dataset scene-order mismatch")
    return [(sid,dataset.images[i].numpy(),first[sid]["scene"]["objects"]) for i,sid in enumerate(scene_order)]


def extract_object_crops(model,dataset,device="cpu",batch_size=40):
    """Oracle object centers ONLY for diagnostic object-level Shape classification."""
    from src.medium_visual_world.world import SHAPES
    scenes=unique_scene_items(dataset)
    matrices=[];labels=[];details=[]
    for start in range(0,len(scenes),batch_size):
        chunk=scenes[start:start+batch_size]
        features=pooled_features(model,np.stack([item[1] for item in chunk]),device,batch_size).reshape(-1,16,16,48)
        for i,(sid,_,objs) in enumerate(chunk):
            for j,o in enumerate(objs):
                gx,gy=int(o["x"])//4,int(o["y"])//4
                padded=np.pad(features[i],((2,2),(2,2),(0,0)),mode="constant")
                crop=padded[gy:gy+5,gx:gx+5,:]
                if crop.shape!=(5,5,48):raise AssertionError("Incorrect ROI shape")
                matrices.append(crop);labels.append(SHAPES.index(o["shape"]))
                details.append(dict(scene_id=sid,object_index=j,shape=o["shape"],color=o["color"],
                                    radius=o["radius"],angle=o["angle"],object_area_proxy=o["radius"]**2))
    return np.stack(matrices),np.asarray(labels,dtype=np.int64),pd.DataFrame(details)


def shape_object_probes(train_tuple,dev_tuple,seed=42,alpha=10.0,head_epochs=35,batch_size=192):
    """Train separate diagnostic object-label decoders on the *same* oracle ROIs.

    All models share identical image set/labels/splits. 5x5 flatten uses 25x more
    features than mean; per-example shuffled cells has the same 1,200 input dims.
    """
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import torch
    from torch import nn
    from torch.nn import functional as F
    x,y,meta=train_tuple;xdev,ydev,mdev=dev_tuple
    if len(np.unique(y))<4:
        raise ValueError("Object Shape probe training needs all four Shape labels")
    train_means=x.mean((1,2));dev_means=xdev.mean((1,2))
    # Fixed per-sample spatial shuffle, using same seed for train and dev; remove position consistency.
    def shuffle_cells(z,group_seed):
        rng=np.random.default_rng(group_seed)
        perm=np.argsort(rng.random((len(z),25)),axis=1)
        return np.take_along_axis(z.reshape(len(z),25,48),perm[:,:,None],axis=1).reshape(len(z),-1)
    models={"mean48":(train_means,dev_means),"spatial1200":(x.reshape(len(x),-1),xdev.reshape(len(xdev),-1)),
            "shuffled1200":(shuffle_cells(x,seed+111),shuffle_cells(xdev,seed+222))}
    results=[];predictions={}
    for name,(a,b) in models.items():
        # No dev-set hyperparameter tuning. Standardization fitted on train only.
        clf=make_pipeline(StandardScaler(),RidgeClassifier(alpha=alpha))
        clf.fit(a,y);p=clf.predict(b)
        acc=float((p==ydev).mean());predictions[name]=p
        results.append(dict(decoder=name,accuracy=acc,n_train=len(y),n_dev=len(ydev),seed=seed,supervision="oracle_object_label"))
    class HeadLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.score=nn.Linear(48,4)
            self.head=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,4))
        def forward(self,z):
            s=self.score(z.permute(0,2,3,1)).permute(0,3,1,2).flatten(2)
            return self.head(torch.cat((s.amax(2),s.mean(2)),dim=1))
    torch.manual_seed(seed)
    net=HeadLike();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=0)
    xx=torch.tensor(x.transpose(0,3,1,2).copy());yy=torch.tensor(y)
    xdv=torch.tensor(xdev.transpose(0,3,1,2).copy())
    for epoch in range(head_epochs):
        net.train();g=torch.Generator().manual_seed(seed+epoch)
        for ids in torch.randperm(len(y),generator=g).split(batch_size):
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(net(xx[ids]),yy[ids]);loss.backward();opt.step()
    net.eval()
    with torch.no_grad():p=net(xdv).argmax(1).numpy()
    predictions["head_like_object"]=p
    results.append(dict(decoder="head_like_object",accuracy=float((p==ydev).mean()),n_train=len(y),n_dev=len(ydev),
                        seed=seed,supervision="oracle_object_label"))
    detail=mdev.copy();detail["true_shape"]=ydev
    for name,p in predictions.items():detail[f"pred_{name}"]=p
    return pd.DataFrame(results),detail


def scene_shape_head_probe(model,train_dataset,dev_dataset,device="cpu",seed=42,epochs=35,batch_size=100):
    """Scene-level Shape Yes/No. Encoder frozen; compares exact 8 inputs with spatial Conv readout.

    Same scene-level labels, no oracle center; model's original predictions also reported.
    The new spatial readout is more expressive: decoding capacity, not causal mechanism.
    """
    from src.medium_visual_world.world import SHAPES
    from src.medium_visual_world.learning import TOKEN_TO_ID
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import torch
    from torch import nn
    from torch.nn import functional as F
    def shape_arrays(ds):
        idx=[i for i,r in enumerate(ds.rows) if r["template"]=="shape"]
        if not idx: raise ValueError("No Shape task examples")
        images=ds.images[ds.image_indices[idx]].numpy()
        queries=ds.queries[idx].numpy()
        target=np.array([SHAPES.index(ds.rows[i]["question"]["args"][0]) for i in idx])
        labels=ds.labels[idx].numpy()
        # The queried Shape's vocabulary index must agree with question tokens.
        assert np.array_equal(queries[:,2],np.array([TOKEN_TO_ID[SHAPES[j]] for j in target]))
        return images,queries,target,labels
    xtr,qtr,atr,ytr=shape_arrays(train_dataset)
    xdv,qdv,adv,ydv=shape_arrays(dev_dataset)
    if len(np.unique(ytr))<2:
        raise ValueError("Scene Shape probe training requires both Yes and No labels")
    btr=original_eight_features(model,xtr,qtr,device)
    bdv=original_eight_features(model,xdv,qdv,device)
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,random_state=seed))
    clf.fit(btr,ytr);p_8=clf.predict(bdv)
    original_probs=[];model.eval()
    with torch.inference_mode():
        for s in range(0,len(xdv),batch_size):
            xx=torch.as_tensor(xdv[s:s+batch_size],dtype=torch.float32,device=device).div(255)
            qq=torch.as_tensor(qdv[s:s+batch_size],dtype=torch.long,device=device)
            original_probs.extend(model(xx,qq).argmax(1).cpu().numpy())
    ptr=pooled_features(model,xtr,device)
    pdv=pooled_features(model,xdv,device)
    class SpatialShape(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.GELU(),nn.Conv2d(32,4,1))
            self.head=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,2))
        def forward(self,z,idx):
            m=self.spatial(z)
            sel=m.gather(1,idx[:,None,None,None].expand(-1,1,16,16)).squeeze(1)
            f=torch.stack((sel.amax((1,2)),sel.mean((1,2))),dim=1)
            return self.head(f)
    torch.manual_seed(seed)
    net=SpatialShape().to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)
    tr=torch.from_numpy(ptr.transpose(0,2,1).reshape(-1,48,16,16).copy())
    dv=torch.from_numpy(pdv.transpose(0,2,1).reshape(-1,48,16,16).copy())
    ar=torch.from_numpy(atr);yd=torch.from_numpy(ytr)
    for epoch in range(epochs):
        net.train();perm=torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed+epoch))
        for chunk in perm.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            output=net(tr[chunk].to(device),ar[chunk].to(device))
            loss=F.cross_entropy(output,yd[chunk].to(device));loss.backward();optimizer.step()
    net.eval();pred=[]
    with torch.no_grad():
        for s in range(0,len(dv),batch_size):
            batch=dv[s:s+batch_size].to(device);idx=torch.from_numpy(adv[s:s+batch_size]).to(device)
            pred.extend(net(batch,idx).argmax(1).cpu().numpy())
    predictions={"original_frozen_model":np.asarray(original_probs),"retrained_head_on_8":np.asarray(p_8),
                 "spatial_conv_probe":np.asarray(pred)}
    summary=[]
    for name,p in predictions.items():
        m=binary_metrics(ydv,p)
        summary.append(dict(readout=name,seed=seed,supervision="same_scene_shape_yesno",**m))
    details=pd.DataFrame(dict(true_label=ydv,queried_shape=[SHAPES[j] for j in adv]))
    for name,p in predictions.items():details[f"pred_{name}"]=p
    return pd.DataFrame(summary),details


def task_head_competition(model,train_dataset,dev_dataset,device="cpu",seed=42,epochs=40,batch_size=120):
    """Original 8 features, retrained shared-vs-independent MLP heads, encoder/query frozen.

    Four separate heads have ~4x parameter count; diagnostic, not a parameter-matched ablation.
    Same training examples and shared global update sequence. No dev checkpoint selection.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F
    from src.medium_visual_world.learning import TASKS
    def arrays(ds):
        images=ds.images[ds.image_indices].numpy()
        queries=ds.queries.numpy()
        features=original_eight_features(model,images,queries,device)
        task=np.array([TASKS.index(r["template"]) for r in ds.rows])
        return features,task,ds.labels.numpy()
    x,t,y=arrays(train_dataset);xd,td,yd=arrays(dev_dataset)
    class Heads(nn.Module):
        def __init__(self, separate):
            super().__init__();self.separate=separate
            self.net=nn.ModuleList([nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,2))
                                    for _ in range(4 if separate else 1)])
        def forward(self,a,b):
            if not self.separate:return self.net[0](a)
            out=torch.empty(len(a),2,device=a.device)
            for i in range(4):
                mask=b==i
                if mask.any():out[mask]=self.net[i](a[mask])
            return out
    records=[];preds={}
    with torch.inference_mode():
        original=model.head(torch.from_numpy(xd).to(device)).argmax(1).cpu().numpy()
    preds["original_pretrained_head"]=original
    for task_i,name in enumerate(TASKS):
        mask=td==task_i
        records.append(dict(seed=seed,readout="original_pretrained_head",task=name,
                            **binary_metrics(yd[mask],original[mask])))
    for label,separate in (("shared_head",False),("task_specific_heads",True)):
        torch.manual_seed(seed)
        net=Heads(separate).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)
        xx=torch.from_numpy(x);tt=torch.from_numpy(t);yy=torch.from_numpy(y)
        xdv=torch.from_numpy(xd);tdv=torch.from_numpy(td)
        for epoch in range(epochs):
            net.train();perm=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+epoch))
            for indices in perm.split(batch_size):
                opt.zero_grad(set_to_none=True)
                logit=net(xx[indices].to(device),tt[indices].to(device))
                loss=F.cross_entropy(logit,yy[indices].to(device));loss.backward();opt.step()
        net.eval()
        with torch.no_grad():p=np.concatenate([net(xdv[i:i+batch_size].to(device),tdv[i:i+batch_size].to(device))
                            .argmax(1).cpu().numpy() for i in range(0,len(yd),batch_size)])
        preds[label]=p
        for task_i,name in enumerate(TASKS):
            mask=td==task_i
            m=binary_metrics(yd[mask],p[mask]);records.append(dict(seed=seed,readout=label,task=name,**m))
    return pd.DataFrame(records),preds
