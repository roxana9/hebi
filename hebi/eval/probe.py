#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:56:04 2026

@author: rokny-tower4
"""

import warnings; warnings.filterwarnings('ignore')
import math, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

N_REPEATS = 10
#EMB_DIR = "/DATA1/Roxana/Hierarchy/protein_embeddings/stream1_z0_random"
EMB_DIR="/DATA1/Roxana/Hierarchy/protein_embeddings/stream3_z2_continual"
TASKS = {
    "charville": {"file": f"{EMB_DIR}/charville_stream3.pt",
                  "labels": {"primary_outcome":[["c002"],["c004"]],
                             "recurrence":[["c002"],["c004"]]}},
    "upmc": {"file": f"{EMB_DIR}/upmc_stream3.pt",
             "labels": {"primary_outcome":[["c003","c005"],["c001","c002"]],
                        "recurred":[["c002","c007"],["c005","c007"]]}},
    "dfci": {"file": f"{EMB_DIR}/dfci_stream3.pt",
             "labels": {"pTR_label":"random"}},
}

def load_task(file, label):
    d = torch.load(file, weights_only=False)
    X_all, labels = d["X"], d["labels"]
    keep_X, keep_y, keep_c = [], [], []
    for i, lab in enumerate(labels):
        v = lab.get(label, float("nan"))
        if v is None or (isinstance(v,float) and math.isnan(v)): continue
        keep_X.append(X_all[i]); keep_y.append(int(v))
        keep_c.append(lab["region_id"].split("_")[1])
    return np.array(keep_X), np.array(keep_y), np.array(keep_c)

def probe_random(X, y):
    aucs=[]
    for s in range(N_REPEATS):
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,stratify=y,random_state=s)
        sc=StandardScaler().fit(Xtr)
        clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(Xtr),ytr)
        aucs.append(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:,1]))
    return np.array(aucs)

def probe_cohort(X,y,c,split):
    outs=[]
    for fold in range(2):
        held=np.isin(c,split[fold])
        if len(np.unique(y[held]))<2 or len(np.unique(y[~held]))<2: continue
        sc=StandardScaler().fit(X[~held])
        clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[~held]),y[~held])
        outs.append(roc_auc_score(y[held], clf.predict_proba(sc.transform(X[held]))[:,1]))
    return outs

if __name__=="__main__":
    for ds,cfg in TASKS.items():
        for label,split in cfg["labels"].items():
            X,y,c = load_task(cfg["file"], label)
            print(f"\n== {ds}/{label} ==  n={len(y)}, classes={np.bincount(y)}")
            if split!="random":
                fa=probe_cohort(X,y,c,split)
                if fa: print(f"  cohort prob-AUC: {[f'{a:.3f}' for a in fa]} mean={np.mean(fa):.4f}")
            r=probe_random(X,y)
            print(f"  random 10-seed: mean={r.mean():.4f} std={r.std():.4f}")