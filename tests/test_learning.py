import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))
    
import pytest
torch=pytest.importorskip('torch')
from src.visual_world.learning import VOCAB,CLOSER_ID,encode_tokens,TinyViTQuestion,VQADataset,make_loader,evaluate
from src.visual_world.baseline_training import stratified_adaptation_indices

ROOT=Path(__file__).resolve().parents[1]/'outputs'/'dataset'

def test_model_contract_and_operator_is_novel():
    tr=VQADataset(ROOT/'train.jsonl',limit=24)
    assert len(tr)==24
    assert all('CLOSER' not in row['query_tokens'] for row in tr.rows)
    assert all('CLOSER' in row['query_tokens'] for row in VQADataset(ROOT/'adapt_train.jsonl',limit=5).rows)
    image,q,label,index=tr[0]
    assert image.shape==(3,64,64) and image.dtype==torch.float32
    assert q.shape==(32,) and q.dtype==torch.long
    assert label.item() in (0,1) and index==0
    assert len(VOCAB)==len(set(VOCAB))

def test_model_forward_backward_and_mask_baselines():
    model=TinyViTQuestion(dim=32,heads=4,vit_depth=1,question_depth=1)
    ds=VQADataset(ROOT/'train.jsonl',limit=4)
    x,q,y,_=next(iter(make_loader(ds,4)))
    logits=model(x,q)
    assert logits.shape==(4,2) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits,y).backward()
    assert model.patch.weight.grad is not None and model.token_embedding.weight.grad is not None
    text=VQADataset(ROOT/'train.jsonl',mode='text-only',limit=4)
    images=torch.stack([text[i][0] for i in range(4)])
    assert torch.all(images[0]==images[1])
    image_only=VQADataset(ROOT/'train.jsonl',mode='image-only',limit=4)
    assert torch.equal(image_only[0][1],image_only[1][1])

def test_frozen_visual_only_freezes_image_encoder():
    model=TinyViTQuestion(dim=32,heads=4,vit_depth=1,question_depth=1)
    model.freeze_visual()
    assert not model.patch.weight.requires_grad
    assert not model.visual_cls.requires_grad
    assert model.token_embedding.weight.requires_grad
    assert model.cross_attn.in_proj_weight.requires_grad

def test_adaptation_internal_split_disjoint_and_balanced():
    ds=VQADataset(ROOT/'adapt_train.jsonl')
    train,dev=stratified_adaptation_indices(ds)
    assert len(train)==96 and len(dev)==32
    assert set(train).isdisjoint(dev)
    assert sorted(train+dev)==list(range(len(ds)))
    assert sum(ds.labels[i] for i in dev)==16
    assert sum(ds.labels[i] for i in train)==48

def test_operator_reset_is_reproducible():
    a=TinyViTQuestion(dim=32,heads=4,vit_depth=1,question_depth=1)
    b=TinyViTQuestion(dim=32,heads=4,vit_depth=1,question_depth=1)
    a.reset_new_operator(777);b.reset_new_operator(777)
    assert torch.equal(a.token_embedding.weight[CLOSER_ID],b.token_embedding.weight[CLOSER_ID])
