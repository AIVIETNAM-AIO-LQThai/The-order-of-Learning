"""Run from repository root: python valid_tests/test_architecture_localization.py

No pretrained checkpoints, dataset, pytest or GPU needed.
"""
from pathlib import Path
import math
import sys
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.medium_visual_world.architecture_localization import (
    build_readout,audit_param_budget,features_from_batch,cache_dataset,
    audit_feature_parity,cached_batch,train_readout,predict_cached,score_edit_predictions,
    summarize_edit_pairs)


class MiniModel(nn.Module):
    """Only the exact modules used by the original AttributeBindingViT forward."""
    def __init__(self):
        super().__init__()
        self.patch=nn.Conv2d(3,48,kernel_size=4,stride=4)
        self.pos=nn.Parameter(torch.zeros(1,256,48))
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(48,4,96,batch_first=True,norm_first=True),1,enable_nested_tensor=False)
        self.pre_norm=nn.LayerNorm(48);self.post_norm=nn.LayerNorm(48);self.score_norm=nn.LayerNorm(48)
        self.emb=nn.Embedding(20,48,padding_idx=0)
        self.query=nn.Sequential(nn.Linear(144,48),nn.GELU(),nn.LayerNorm(48))
        self.head=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,x,q):return self.head(features_from_batch(self,x,q)["eight"])


class TinyDataset:
    def __init__(self):
        self.images=torch.randint(0,255,(2,3,64,64),dtype=torch.uint8)
        self.image_indices=torch.tensor([0,0,1,1],dtype=torch.long)
        self.queries=torch.tensor([[1,2,3,0],[1,4,3,7],[1,2,5,0],[1,4,5,6]])
        self.labels=torch.tensor([0,1,1,0])
        self.rows=[{"id":str(i)} for i in range(4)]
    def batch(self,indices,device):
        x=self.images[self.image_indices[indices]].to(device).float().div(255.)
        return x,self.queries[indices].to(device),self.labels[indices].to(device)


def main():
    torch.manual_seed(12)
    counts=audit_param_budget()
    assert set(counts)=={"scalar","spatial","tokens"}
    model=MiniModel().eval()
    ds=TinyDataset()
    cache=cache_dataset(model,ds,torch.device("cpu"),batch_size=2)
    parity=audit_feature_parity(model,ds,cache,torch.device("cpu"),tol=.02)
    assert cache["tokens"].shape==(2,256,48)
    assert cache["maps"].shape==(4,4,16,16)
    assert cache["eight"].shape==(4,8)
    assert cache["query"].shape==(4,144)
    x=cached_batch(cache,torch.tensor([0,1]),torch.device("cpu"))
    assert torch.allclose(x[2][0],x[2][1]),"Same-scene visual token cache differs by question"
    # Fixed-capacity readers consume the expected inputs, and receive gradients.
    for name in ("scalar","spatial","tokens"):
        net=build_readout(name)
        logits=net(*x)
        assert logits.shape==(2,2) and torch.isfinite(logits).all()
        logits.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.parameters())
    # Readout heads use only saved features; original frozen model receives no grads.
    for p in model.parameters():p.requires_grad_(False)
    head=train_readout(cache,"scalar",run_seed=42,device=torch.device("cpu"),epochs=1,batch_size=4)
    preds=predict_cached(model,{"scalar":head},cache,torch.device("cpu"),batch_size=4)
    assert set(preds)=={"pretrained","scalar"}
    assert len(preds["scalar"])==4
    assert not any(p.grad is not None for p in model.parameters())
    import pandas as pd
    rows=[]
    for state,truth,pa,pb in (("before",0,.2,.1),("after",1,.8,.9)):
        rows.append(dict(pair_id="x",scene_id="scene",readout="notused",group="strict_swap",
                         edit="binding_swap",task="pair",heldout_created=0,state=state,label=truth,
                         p_pretrained=pa,p_scalar=pa,p_spatial=pb,p_tokens=pb))
    pairs=score_edit_predictions(pd.DataFrame(rows))
    assert pairs.both_correct.eq(1).all() and pairs.correct_direction.eq(1).all()
    metrics=summarize_edit_pairs(pairs)
    assert set(metrics.metric)=={"strict_swap_all","strict_swap_id_safe"}
    # Invariance controls include both Shape-present and Shape-absent conditions.
    edit_rows=[]
    for pair_id,label,prob in (("absent",0,.1),("present",1,.9)):
        for state in ("before","after"):
            edit_rows.append(dict(pair_id=pair_id,scene_id="scene",group="shape_recolor_all",
                                  edit="color_insert",task="shape",heldout_created=0,
                                  state=state,label=label,p_pretrained=prob,
                                  p_scalar=prob,p_spatial=prob,p_tokens=prob))
    invariant=score_edit_predictions(pd.DataFrame(edit_rows))
    summary=summarize_edit_pairs(invariant)
    assert set(summary.metric)=={"shape_recolor_all_id_safe","shape_recolor_absent_id_safe","shape_recolor_present_id_safe"}
    assert summary.both_correct.eq(1).all()
    print("PASS: matched parameter budget, spatial features, original logits parity, gradients, frozen encoder, edit metrics")
    print("parameter counts:",counts,"cache parity:",parity)

if __name__=="__main__":main()
