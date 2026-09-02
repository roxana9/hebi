#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:36:31 2026

@author: tower4
"""

import argparse
from pathlib import Path
import torch
import scanpy as sc
import numpy as np
import json
import warnings
import pandas as pd
from tqdm import tqdm

# --- YOUR MODULES ---
from hebi.data.graph import build_spatial_graph
from hebi.data.coarsening import create_metis_partition 
from hebi.data.vocabulary import harmonise_adata

warnings.filterwarnings('ignore')

def main():
    # ==========================
    # 1. CONFIGURATION
    # ==========================
    ADATA_PATH = "/DATA1/Roxana/Hierarchy/data/Merfish_hebi.h5ad"
    OUT_ROOT = "/DATA1/Roxana/Hierarchy/Two_levels/fine_tune"
    VOCAB_PATH = "/DATA1/Roxana/Hierarchy/Two_levels/default_gene_vocab.json"
    
    # --- CHANGE: DEFINE BOTH TARGET COLUMNS ---
    LABEL_COL_1 = 'class'  # Cell Type
    LABEL_COL_2 = 'niche'  # Tissue Niche
    
    SECTION_COL = 'brain_section_label'
    SPLIT_COL = 'split'

    # ==========================
    # 2. SETUP DIRECTORIES
    # ==========================
    root = Path(OUT_ROOT)
    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {ADATA_PATH}...")
    adata_full = sc.read_h5ad(ADATA_PATH)
    adata_full.layers["counts"] = adata_full.X.copy()
    sc.pp.normalize_total(adata_full, target_sum=1e4, layer="counts")
    sc.pp.log1p(adata_full)
    # ==========================
    # 3. CREATE CONSISTENT MAPPINGS (FOR BOTH)
    # ==========================
    print("🏷️  Creating label dictionaries...")

    # --- MAPPING 1: CLASS ---
    unique_class = sorted(adata_full.obs[LABEL_COL_1].unique().astype(str))
    class_to_id = {label: i for i, label in enumerate(unique_class)}
    
    # --- MAPPING 2: NICHE ---
    unique_niche = sorted(adata_full.obs[LABEL_COL_2].unique().astype(str))
    niche_to_id = {label: i for i, label in enumerate(unique_niche)}
    
    # Save both mappings
    mappings = {
        "class_mapping": class_to_id,
        "niche_mapping": niche_to_id
    }
    with open(root / "label_mappings.json", "w") as f:
        json.dump(mappings, f, indent=4)
        
    print(f"   Saved mappings: {len(unique_class)} classes, {len(unique_niche)} niches.")

    # ==========================
    # 4. PROCESSING LOOP
    # ==========================
    sections = adata_full.obs[SECTION_COL].unique()
    print(f"🚀 Processing {len(sections)} sections...")

    for section_id in tqdm(sections):
        # Subset section
        adata_section = adata_full[adata_full.obs[SECTION_COL] == section_id].copy()
       # print(adata_section)
        # Determine Split
        split_val = adata_section.obs[SPLIT_COL].iloc[0] 
        
        if split_val == 'train':
            save_dir = train_dir
        elif split_val == 'test':
            save_dir = test_dir
        else:
            # Fallback for alternative naming
            is_test = adata_section.obs.get('is_test_section', [False])[0]
            save_dir = test_dir if is_test else train_dir

        out_file = save_dir / f"{section_id}.pt"
        if out_file.exists(): continue

        # --- STANDARD PREPROCESSING ---
        meta = {"species": "Mouse", "tissue": "Brain", "disease": "Normal"} 
        
        adata_section = harmonise_adata(adata_section, VOCAB_PATH, fine_tune=True, species="Mouse")
       # sc.pp.normalize_total(adata_section, target_sum=1e4)
      #  sc.pp.log1p(adata_section)

        if isinstance(adata_section.X, np.ndarray):
            X = torch.tensor(adata_section.X, dtype=torch.float32)
        else:
            X = torch.tensor(adata_section.X.toarray(), dtype=torch.float32)

        # Spatial Coords
        if 'spatial' in adata_section.obsm:
            coords = torch.tensor(adata_section.obsm['spatial'], dtype=torch.float32)
        else:
            coords = torch.tensor(adata_section.obsm['spatial_3d'], dtype=torch.float32)[:, :2] 

        # Graph & METIS
        edge_index, edge_attr, _ = build_spatial_graph(coords.numpy(), k=5)
        
        # Adaptive clusters for small sections
        n_clusters = 1024
        if adata_section.n_obs < 1024: 
            n_clusters = max(1, adata_section.n_obs // 10)
            
        node2comm, super_sp = create_metis_partition(
            edge_index, X.shape[0], n_clusters, seed=42
        )

        gene_ids = torch.tensor(adata_section.var["gene_id"].values, dtype=torch.long)
        
        # --- NEW: ENCODE BOTH LABELS ---
        
        # 1. Encode Class
        raw_class = adata_section.obs[LABEL_COL_1].astype(str).values
        y_class = torch.tensor([class_to_id[l] for l in raw_class], dtype=torch.long)

        # 2. Encode Niche
        raw_niche = adata_section.obs[LABEL_COL_2].astype(str).values
        y_niche = torch.tensor([niche_to_id[l] for l in raw_niche], dtype=torch.long)

        # Save Dict
        data_dict = {
            "X": X.half(),
            "coords": coords,
            "edge_index": edge_index,
            "node2comm": node2comm,
            "super_sp": super_sp,
            "meta": meta,
            "gene_ids": gene_ids,
            
            # SAVE BOTH LABELS
            "y_class": y_class,
            "y_niche": y_niche,
            
            # Optional: Save counts for weighted loss later
            "num_class_types": len(unique_class),
            "num_niche_types": len(unique_niche)
        }
        
        torch.save(data_dict, out_file)

    print("\n✅ Done! Check 'preprocess_finetune' folder.")

if __name__ == "__main__":
    main()