import warnings; warnings.filterwarnings('ignore')
import math, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
N_REPEATS=10; HIDDEN=300; BS=64; LR=1e-4; WD=1e-4; EPOCHS=1000
EMB_DIR="/DATA1/Roxana/Hierarchy/protein_embeddings/stream3_z2_continual"

TASKS={
 "charville":{"file":f"{EMB_DIR}/charville_stream3.pt",
   "labels":{"primary_outcome":[["c002"],["c004"]],"recurrence":[["c002"],["c004"]]}},
 "upmc":{"file":f"{EMB_DIR}/upmc_stream3.pt",
   "labels":{"primary_outcome":[["c003","c005"],["c001","c002"]],"recurred":[["c002","c007"],["c005","c007"]]}},
 "dfci":{"file":f"{EMB_DIR}/dfci_stream3.pt","labels":{"pTR_label":"random"}},
}

class DS(Dataset):
    def __init__(self,X,y): self.X,self.y=X,y
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i],self.y[i]

class MLP(nn.Module):   # matches HEIST num_layers=1 (bare linear)
    def __init__(self,ind,out): super().__init__(); self.f=nn.Linear(ind,out)
    def forward(self,x): return self.f(x)

def aucpr_hinge_loss(y_pred, y_true, margin=1.0):
    # y_pred: scalar score per instance
    pos=y_pred[y_true==1]; neg=y_pred[y_true==0]
    if len(pos)==0 or len(neg)==0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    diff=pos.view(-1,1)-neg.view(1,-1)
    return F.relu(margin-diff).mean()

def score(model,X):           # 2-logit -> scalar ranking score
    lo=model(X); return lo[:,1]-lo[:,0]

def eval_roc(model,loader):
    model.eval(); t,s=[],[]
    with torch.no_grad():
        for X,y in loader:
            t.append(y); s.append(score(model,X))
    t=torch.cat(t).cpu().numpy(); s=torch.cat(s).cpu().numpy()
    if len(np.unique(t))<2: return float('nan')
    return roc_auc_score(t,s)

def train_once(model,tr,va,te):
    bv=eval_roc(model,va); bt=eval_roc(model,te)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    for _ in range(EPOCHS):
        model.train()
        for X,y in tr:
            opt.zero_grad(); loss=aucpr_hinge_loss(score(model,X),y); loss.backward(); opt.step()
        v=eval_roc(model,va)
        if not math.isnan(v) and (math.isnan(bv) or v>=bv): bv=v; bt=eval_roc(model,te)
    return bt

def load_task(file,label):
    d=torch.load(file,weights_only=False); Xa,labs=d["X"],d["labels"]
    X,y,c=[],[],[]
    for i,lab in enumerate(labs):
        v=lab.get(label,float("nan"))
        if v is None or (isinstance(v,float) and math.isnan(v)): continue
        X.append(Xa[i]); y.append(int(v)); c.append(lab["region_id"].split("_")[1])
    return (torch.FloatTensor(np.array(X)).to(DEVICE),
            torch.LongTensor(y).to(DEVICE), np.array(c))

def mk(X,y,idx,sh): return DataLoader(DS(X[idx],y[idx]),batch_size=BS,shuffle=sh)

def run_cohort(X,y,c,split,seed):
    sc=[]
    for fold in range(2):
        torch.manual_seed(seed*100+fold); np.random.seed(seed*100+fold)
        m=MLP(X.shape[1],2).to(DEVICE)
        held=np.array([i for i in range(len(c)) if c[i] in split[fold]])
        tr=torch.LongTensor([i for i in range(len(c)) if c[i] not in split[fold]]).to(DEVICE)
        va,te=train_test_split(held,test_size=0.5,stratify=y.cpu().numpy()[held],random_state=seed*100+fold)
        r=train_once(m,mk(X,y,tr,True),mk(X,y,torch.LongTensor(va).to(DEVICE),False),
                     mk(X,y,torch.LongTensor(te).to(DEVICE),False))
        if not math.isnan(r): sc.append(r)
    return np.mean(sc) if sc else float('nan')

def run_random(X,y,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=MLP(X.shape[1],2).to(DEVICE)
    idx=np.arange(len(y))
    tr,tmp=train_test_split(idx,test_size=0.4,stratify=y.cpu().numpy(),random_state=seed)
    va,te=train_test_split(tmp,test_size=0.5,stratify=y.cpu().numpy()[tmp],random_state=seed)
    return train_once(m,mk(X,y,torch.LongTensor(tr).to(DEVICE),True),
                      mk(X,y,torch.LongTensor(va).to(DEVICE),False),
                      mk(X,y,torch.LongTensor(te).to(DEVICE),False))

if __name__=="__main__":
    for ds,cfg in TASKS.items():
        for label,split in cfg["labels"].items():
            X,y,c=load_task(cfg["file"],label)
            print(f"\n== {ds}/{label} == n={len(y)} classes={np.bincount(y.cpu().numpy())}")
            means=[run_random(X,y,s) if split=="random" else run_cohort(X,y,c,split,s) for s in range(N_REPEATS)]
            means=np.array(means)
            print(f"  MLP 10-seed: mean={np.nanmean(means):.4f} std={np.nanstd(means):.4f} "
                  f"min={np.nanmin(means):.4f} max={np.nanmax(means):.4f}")