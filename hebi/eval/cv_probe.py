#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:01:32 2026

@author: rokny-tower4
"""

import warnings; warnings.filterwarnings('ignore')
import math, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

N_SPLITS=5; N_REPEATS=3

# label columns per dataset
LABELS={"charville":["primary_outcome","recurrence"],
        "upmc":["primary_outcome","recurred"],
        "dfci":["pTR_label"]}

def load_any(path):
    """Return (X [n,d], list-of-label-dicts). Handles HEBI {X,labels} format."""
    d=torch.load(path,weights_only=False)
    if isinstance(d,dict) and "X" in d and "labels" in d:
        return np.asarray(d["X"]), d["labels"]
    # HEIST format: list of graphs with .X (cells) and label attrs
    X,labs=[],[]
    for g in d:
        X.append(np.asarray(g.X.mean(0) if hasattr(g.X,"mean") else g.X))
        labs.append({k:getattr(g,k,float("nan")) for k in
                     ["region_id","primary_outcome","recurrence","recurred","pTR_label"]})
    return np.stack(X), labs

def get_task(X,labs,label):
    keepX,keepy=[],[]
    for i,l in enumerate(labs):
        v=l.get(label,float("nan"))
        if v is None or (isinstance(v,float) and math.isnan(v)): continue
        keepX.append(X[i]); keepy.append(int(v))
    return np.array(keepX), np.array(keepy)

def cv_auc(X,y):
    aucs=[]
    for r in range(N_REPEATS):
        skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=r)
        for tr,te in skf.split(X,y):
            sc=StandardScaler().fit(X[tr])
            clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr]),y[tr])
            p=clf.predict_proba(sc.transform(X[te]))[:,1]
            aucs.append(roc_auc_score(y[te],p))
    return np.array(aucs)

def run(name, files):
    """files: dict dataset->embedding_path"""
    print(f"\n########## {name} ##########")
    for ds,path in files.items():
        X,labs=load_any(path)
        for label in LABELS[ds]:
            Xt,yt=get_task(X,labs,label)
            a=cv_auc(Xt,yt)
            print(f"  {ds}/{label:16s} n={len(yt):3d} "
                  f"5-fold×3 AUC = {a.mean():.4f} ± {a.std():.4f}")

if __name__=="__main__":
    HEBI="/DATA1/Roxana/Hierarchy/protein_embeddings"
    HEIST="/DATA1/Roxana/HEIST/data/space-gm"

    runs={
      "HEIST (pretrained frozen)":{
        "charville":f"{HEIST}/charville_graphs_HEIST.pt",
        "upmc":f"{HEIST}/upmc_graphs_HEIST.pt",
        "dfci":f"{HEIST}/dfci_graphs_HEIST.pt"},
      "HEBI s1 z0-random":{
        "charville":f"{HEBI}/stream1_z0_random/charville_stream1.pt",
        "upmc":f"{HEBI}/stream1_z0_random/upmc_stream1.pt",
        "dfci":f"{HEBI}/stream1_z0_random/dfci_stream1.pt"},
      "HEBI s2 z0-pretrained":{
        "charville":f"{HEBI}/stream2_z0_pretrained/charville_stream2.pt",
        "upmc":f"{HEBI}/stream2_z0_pretrained/upmc_stream2.pt",
        "dfci":f"{HEBI}/stream2_z0_pretrained/dfci_stream2.pt"},
      "HEBI s3 z2-continual":{
        "charville":f"{HEBI}/stream3_z2_continual/charville_stream3.pt",
        "upmc":f"{HEBI}/stream3_z2_continual/upmc_stream3.pt",
        "dfci":f"{HEBI}/stream3_z2_continual/dfci_stream3.pt"},
    }
    for name,files in runs.items():
        run(name,files)