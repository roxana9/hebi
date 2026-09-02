#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:44:37 2026

@author: tower4
"""

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
from build_graph import build_spatial_graph
from coarsening_metis import create_metis_partition 
from preprocess_offline import harmonise_adata

warnings.filterwarnings('ignore')

def main():
    # ==========================
    # 1. CONFIGURATION
    # ==========================
    ADATA_PATH = "/DATA1/Roxana/Data/cancer_liver_cosmax.h5ad"
    OUT_ROOT = "/DATA1/Roxana/Hierarchy/Two_levels/fine_tune/Liver_cosmax_cancer"
    VOCAB_PATH = "/DATA1/Roxana/Hierarchy/Two_levels/default_gene_vocab.json"
    
    # --- CHANGE: DEFINE BOTH TARGET COLUMNS ---
    LABEL_COL_1 = 'cell_type'  # Cell Type
    LABEL_COL_2 = 'niche_type'  # Tissue Niche
    
    SECTION_COL = 'Run_Tissue_name'
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
    adata_full = harmonise_adata(adata_full, VOCAB_PATH, fine_tune=False, species="human")
    sc.pp.normalize_total(adata_full, target_sum=1e4)
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
    S=adata_full.obs["split"].unique().tolist()
    for item in S:
        if item == 'train':
            save_dir = train_dir
        elif item == 'test':
            save_dir = test_dir
        else:
            # Fallback for alternative naming
            is_test = adata_full.obs.get('is_test_section', [False])[0]
            save_dir = test_dir if is_test else train_dir
        data=adata_full[adata_full.obs["split"]==item].copy()
        out_file = save_dir / f"{item}.pt"
         

    
        # --- STANDARD PREPROCESSING ---
        meta = {"species": "Human", "tissue": "Liver", "disease": "Normal"} 
        
    

        if isinstance(data.X, np.ndarray):
            X = torch.tensor(data.X, dtype=torch.float32)
        else:
            X = torch.tensor(data.X.toarray(), dtype=torch.float32)
    
        # Spatial Coords
        if 'spatial' in adata_full.obsm:
            coords = torch.tensor(data.obsm['spatial'], dtype=torch.float32)
        else:
            coords = torch.tensor(data.obsm['spatial_3d'], dtype=torch.float32)[:, :2] 
    
        # Graph & METIS
        edge_index, edge_attr, _ = build_spatial_graph(coords.numpy(), k=5)
        
        # Adaptive clusters for small sections
        n_clusters = 1024
        if adata_full.n_obs < 1024: 
            n_clusters = max(1, adata_full.n_obs // 10)
            
        node2comm, super_sp = create_metis_partition(
            edge_index, X.shape[0], n_clusters, seed=42
        )
    
        gene_ids = torch.tensor(data.var["gene_id"].values, dtype=torch.long)
        
        # --- NEW: ENCODE BOTH LABELS ---
        
        # 1. Encode Class
        raw_class = data.obs[LABEL_COL_1].astype(str).values
        y_class = torch.tensor([class_to_id[l] for l in raw_class], dtype=torch.long)
    
        # 2. Encode Niche
        raw_niche = data.obs[LABEL_COL_2].astype(str).values
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
        del data

    print("\n✅ Done! Check 'preprocess_finetune' folder.")

if __name__ == "__main__":
    main()