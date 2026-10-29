"""Deterministic model inputs, original multimodal ViT/question Transformer, and metrics.

Only pixel arrays and canonical prefix question tokens reach the network. Split,
scene/object roles, labels, metadata and geometry are never network inputs.
"""
from collections import Counter
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Stable imports when launched from VS Code, another working directory, or a notebook.
for IMPORT_ROOT in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(IMPORT_ROOT))

import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from src.visual_world.generate import read_rows
from src.visual_world.input_contract import model_example

VOCAB = ('[PAD]', '[CLS]', '[MASK]', 'EXISTS_COLOR', 'EXISTS_SHAPE', 'EXISTS_PAIR', 'LEFT', 'ABOVE', 'XOR', 'AND', 'CLOSER', 'REF', 'red', 'blue', 'green', 'yellow', 'circle', 'square', 'triangle')
TOKEN_TO_ID = {t:i for i,t in enumerate(VOCAB)}
CLOSER_ID = TOKEN_TO_ID['CLOSER']
MAX_QUERY = 32

def encode_tokens(words):
    unknown = [w for w in words if w not in TOKEN_TO_ID]
    if unknown: raise ValueError(f'Unknown model query token: {unknown}')
    if not (1 <= len(words) <= MAX_QUERY - 1): raise ValueError('Invalid query length')
    ids = [TOKEN_TO_ID['[CLS]']] + [TOKEN_TO_ID[w] for w in words]
    return ids + [0]*(MAX_QUERY-len(ids))

class VQADataset(Dataset):
    def __init__(self, path:Path, mode='full', limit=None, levels=None):
        if mode not in ('full','text-only','image-only'): raise ValueError(mode)
        rows=list(read_rows(Path(path)))
        if levels is not None: rows=[r for r in rows if r["level"] in levels]
        if limit: rows=rows[:limit]
        self.mode=mode
        self.rows=rows
        self.images=[]
        self.queries=[]
        self.labels=[]
        self.levels=[]
        self.templates=[]
        for row in rows:
            inp,y=model_example(row)
            self.images.append(np.asarray(inp['image'],dtype=np.uint8).copy())
            self.queries.append(encode_tokens(inp['query_tokens']))
            self.labels.append(y)
            self.levels.append(str(row['level']) if row['level'] is not None else 'adapt')
            self.templates.append(row['template'])
        self.constant_image=torch.tensor([243/255,246/255,244/255],dtype=torch.float32).view(3,1,1).expand(3,64,64)
        self.mask_query=torch.tensor([TOKEN_TO_ID['[CLS]'],TOKEN_TO_ID['[MASK]']]+[0]*(MAX_QUERY-2),dtype=torch.long)
    def __len__(self):return len(self.rows)
    def __getitem__(self,idx):
        x=torch.from_numpy(self.images[idx]).permute(2,0,1).float().div_(255.0)
        q=torch.tensor(self.queries[idx],dtype=torch.long)
        if self.mode=='text-only': x=self.constant_image
        if self.mode=='image-only': q=self.mask_query
        return x,q,torch.tensor(self.labels[idx],dtype=torch.long),idx

class TinyViTQuestion(nn.Module):
    def __init__(self,dim=96,heads=4,vit_depth=3,question_depth=2,dropout=0.0):
        super().__init__()
        if dim%heads: raise ValueError('dim must be divisible by heads')
        self.dim=dim
        self.patch=nn.Conv2d(3,dim,kernel_size=8,stride=8)
        self.visual_cls=nn.Parameter(torch.zeros(1,1,dim))
        self.visual_pos=nn.Parameter(torch.zeros(1,65,dim))
        self.token_embedding=nn.Embedding(len(VOCAB),dim,padding_idx=0)
        self.question_pos=nn.Parameter(torch.zeros(1,MAX_QUERY,dim))
        self.visual_transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,heads,dim*3,dropout=dropout,batch_first=True,norm_first=True,activation='gelu'),vit_depth,enable_nested_tensor=False)
        self.text_transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,heads,dim*2,dropout=dropout,batch_first=True,norm_first=True,activation='gelu'),question_depth,enable_nested_tensor=False)
        self.vis_norm=nn.LayerNorm(dim)
        self.text_norm=nn.LayerNorm(dim)
        self.cross_attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.cross_norm=nn.LayerNorm(dim)
        self.fusion_transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,heads,dim*2,dropout=dropout,batch_first=True,norm_first=True,activation='gelu'),1,enable_nested_tensor=False)
        self.fusion_norm=nn.LayerNorm(dim)
        self.classifier=nn.Sequential(nn.LayerNorm(dim*3),nn.Linear(dim*3,dim),nn.GELU(),nn.Linear(dim,2))
        nn.init.trunc_normal_(self.visual_cls,std=.02)
        nn.init.trunc_normal_(self.visual_pos,std=.02)
        nn.init.trunc_normal_(self.question_pos,std=.02)
    def forward(self,image,query):
        p=self.patch(image).flatten(2).transpose(1,2)
        v=torch.cat((self.visual_cls.expand(image.shape[0],-1,-1),p),dim=1)+self.visual_pos
        v=self.vis_norm(self.visual_transformer(v))
        q=self.token_embedding(query)+self.question_pos
        q=self.text_norm(self.text_transformer(q,src_key_padding_mask=(query==0)))
        # Every query token separately reads visual patches; an extra fusion
        # Transformer combines evidence for multi-predicate questions.
        a,_=self.cross_attn(query=q,key=v[:,1:],value=v[:,1:],need_weights=False)
        a=self.cross_norm(a)
        fused=self.fusion_norm(self.fusion_transformer(q+a,src_key_padding_mask=(query==0)))
        h=torch.cat((q[:,0],fused[:,0],q[:,0]*fused[:,0]),dim=-1)
        return self.classifier(h)
    def freeze_visual(self):
        for module in (self.patch,self.visual_transformer,self.vis_norm):
            for p in module.parameters():p.requires_grad_(False)
        self.visual_cls.requires_grad_(False)
        self.visual_pos.requires_grad_(False)
    @torch.no_grad()
    def reset_new_operator(self,seed=777):
        # Explicitly reset an operator never included in the source training data.
        gen=torch.Generator(device='cpu').manual_seed(seed)
        vec=torch.randn((self.dim,),generator=gen)*.02
        self.token_embedding.weight[CLOSER_ID].copy_(vec.to(self.token_embedding.weight.device))

def make_loader(dataset,batch_size=64,shuffle=False,seed=42,num_workers=0):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=torch.cuda.is_available(),generator=torch.Generator().manual_seed(seed))

def set_seed(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def evaluate(model,loader,device):
    model.eval();by_template=Counter();by_level=Counter();right_template=Counter();right_level=Counter();n=0;correct=0;loss_sum=0.
    source=loader.dataset.dataset if isinstance(loader.dataset,Subset) else loader.dataset
    with torch.inference_mode():
        for x,q,y,ix in loader:
            pred=model(x.to(device,non_blocking=True),q.to(device,non_blocking=True))
            labels=y.to(device,non_blocking=True)
            loss_sum+=nn.functional.cross_entropy(pred,labels,reduction='sum').item()
            outcomes=(pred.argmax(dim=1)==labels).cpu().tolist()
            for j,ok in zip(ix.tolist(),outcomes):
                template=source.templates[j];level=source.levels[j]
                by_template[template]+=1;by_level[level]+=1
                right_template[template]+=int(ok);right_level[level]+=int(ok)
            correct+=sum(outcomes);n+=len(outcomes)
    if not n:raise ValueError('Empty dataset')
    return {'n':n,'accuracy':correct/n,'loss':loss_sum/n,'by_template':{k:round(right_template[k]/v,5) for k,v in sorted(by_template.items())},'by_level':{k:round(right_level[k]/v,5) for k,v in sorted(by_level.items())}}

def environment():
    info={'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'num_threads':torch.get_num_threads()}
    if torch.cuda.is_available():
        info.update({'device_name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0))})
    return info
