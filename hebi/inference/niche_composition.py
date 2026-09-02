#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:04:08 2026

@author: rokny-tower4
"""

import sys
sys.path.insert(0, "/DATA1/Roxana/Hierarchy/Two_levels")
sys.path.insert(0, "/DATA1/Roxana/Hierarchy/Two_levels/clustering")
import os, glob, numpy as np, torch
from tqdm import tqdm
from continual_pretrain import continual_pretrain
from main_clustering import load_trainer_from_checkpoint, encode_slide_to_z2

DEVICE="cuda:0"
VOCAB="/DATA1/Roxana/Hierarchy/Two_levels/default_gene_vocab.json"
CKPT="/DATA1/Roxana/Hierarchy/hebi_random_256.pt"
DATA="/DATA1/Roxana/Hierarchy/protein_data_256"
NUM_EPOCHS=10
NUM_NICHES=20          # z2 prototype count
LR=1e-4

DATASETS={
 "charville":(f"{DATA}/charville",["primary_outcome","recurrence"]),
 "upmc":(f"{DATA}/upmc",["primary_outcome","recurred"]),
 "dfci":(f"{DATA}/dfci",["pTR_label"]),
}
OUT="/DATA1/Roxana/Hierarchy/protein_embeddings/stream_nichecomp"
CKPT_DIR="/DATA1/Roxana/Hierarchy/protein_embeddings/continual_ckpts_nichecomp"

def main():
    os.makedirs(OUT,exist_ok=True); os.makedirs(CKPT_DIR,exist_ok=True)
    for ds,(in_dir,label_cols) in DATASETS.items():
        cont=f"{CKPT_DIR}/{ds}_nichecomp.pt"
        print(f"\n=== {ds}: continual (256) {NUM_EPOCHS} epochs ===")
        continual_pretrain(
            checkpoint_path=CKPT, new_data_dir=in_dir, vocab_path=VOCAB,
            device=DEVICE, lr=LR, num_epochs=NUM_EPOCHS,
            out_checkpoint_path=cont, gene_graphs=None, mode="pretrained",
        )
        print(f"=== {ds}: extracting niche composition ===")
        trainer=load_trainer_from_checkpoint(cont,in_dir,VOCAB,DEVICE,mode="pretrained")
        files=sorted(glob.glob(f"{in_dir}/*.pt"))
        vecs,labs,usage=[],[],np.zeros(NUM_NICHES)
        for f in tqdm(files,desc=ds):
            out=encode_slide_to_z2(trainer,f)
            niche=out[3].cpu().numpy()                       # per-cell niche id
            comp=np.bincount(niche,minlength=NUM_NICHES).astype(float)
            usage+=comp
            comp=comp/comp.sum() if comp.sum()>0 else comp   # proportions
            vecs.append(comp)
            slide=torch.load(f,map_location="cpu",weights_only=False)
            labs.append({"region_id":slide.get("region_id"),
                         **{c:slide.get(c,float("nan")) for c in label_cols}})
        torch.save({"X":np.stack(vecs),"labels":labs},f"{OUT}/{ds}_nichecomp.pt")
        print(f"[{ds}] saved {np.stack(vecs).shape}")
        print(f"[{ds}] niche usage (cells per niche): {usage.astype(int)}")

if __name__=="__main__":
    main()