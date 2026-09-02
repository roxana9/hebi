#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 9 14:39:15 2025

@author: tower4
"""

import torch
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N_COMPONENTS = 30 # Set the desired number of PCA dimensions

def plot_atlas(subsample_path):
    print(f"Loading {subsample_path}...")
    # Fix 1: Load with weights_only=False (from previous fix)
    viz_list = torch.load(subsample_path, weights_only=False)
    
    # 1. Reconstruct a Giant AnnData object using PCA embeddings
    X_list = []
    obs_list = []
    
    # 2. Pre-process and collect all raw data for optional global scaling (recommended)
    #    NOTE: Since we cannot calculate a common gene set here, we rely entirely 
    #    on PCA to harmonize the dimension (e.g., 30 components).
    
    print("Constructing Giant Matrix via PCA embeddings...")
    for i, item in enumerate(viz_list):
        X_raw = item["X"]
        meta = item["meta"]
        slide_id = item["slide_id"]
        
        print(f"  Processing sample {i+1}: ID={slide_id}, Shape={X_raw.shape}")
        
        # Create metadata dataframe
        n_cells = X_raw.shape[0]
        df = pd.DataFrame({
            "slide_id": [slide_id] * n_cells,
            "species": [meta["species"]] * n_cells,
            "tissue": [meta["tissue"]] * n_cells,
            "disease": [meta["disease"]] * n_cells
        })
        
        # --- FIX 2: Dimensionality Reduction via PCA ---
        
        # Temporary AnnData for processing (Gene names are unknown, using placeholders)
        n_genes = X_raw.shape[1]
        var_names = [f'gene_{j}' for j in range(n_genes)]
        
        # Create a temporary AnnData object
        adata_temp = anndata.AnnData(X=X_raw, obs=df, var=pd.DataFrame(index=var_names))
        
        # Preprocessing before PCA (highly recommended in real analysis)
        # We assume X_raw is already log-transformed or normalized for simplicity here.
        
        # Run PCA on the individual sample
        sc.tl.pca(adata_temp, n_comps=N_COMPONENTS, use_highly_variable=False, svd_solver='arpack')
        
        # Extract the low-dimensional embedding (N_cells x N_COMPONENTS)
        X_pca = adata_temp.obsm['X_pca']
        
        # Append the low-dimensional matrix and metadata
        X_list.append(X_pca)
        obs_list.append(df)
        
    # Concatenate the low-dimensional (PCA) embeddings
    # FIX 3: This concatenation now works because all matrices in X_list are N_i x N_COMPONENTS
    X_all = np.concatenate(X_list, axis=0)
    obs_all = pd.concat(obs_list, axis=0).reset_index(drop=True)
    
    # Create the final AnnData object using the integrated PCA embeddings
    adata = anndata.AnnData(X=X_all, obs=obs_all)
    print(f"Giant Atlas Shape: {adata.shape}")
    
    # 3. Compute UMAP on the integrated PCA space
    print("Computing Neighbors & UMAP...")
    
    # Since X now holds the PCA embedding, we must store it in .obsm to use it for neighbors calculation.
    adata.obsm['X_pca'] = adata.X 
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata, min_dist=0.5) # min_dist=0.5 is a standard default, adjust as needed
    
    # 4. Plotting
    print("Generating Plots...")
    
    # Plot 1: Colored by Tissue
    sc.pl.umap(adata, color=['tissue'], title="Global Atlas: Tissue Type", show=False, frameon=False)
    plt.savefig("global_umap_tissue.png", bbox_inches='tight', dpi=300)
    print("Saved global_umap_tissue.png")
    
    # Plot 2: Colored by Disease Status
    # Note: Assuming 'Normal'/'Diseased'/'unknown' are the actual values in the 'disease' column
    sc.pl.umap(adata, color=['disease'], title="Global Atlas: Disease Status", palette={'Normal': 'g', 'Diseased': 'r', 'unknown': 'grey'}, show=False, frameon=False)
    plt.savefig("global_umap_disease.png", bbox_inches='tight', dpi=300)
    print("Saved global_umap_disease.png")

    # Plot 3: Colored by Species
    sc.pl.umap(adata, color=['species'], title="Global Atlas: Species", show=False, frameon=False)
    plt.savefig("global_umap_species.png", bbox_inches='tight', dpi=300)
    print("Saved global_umap_species.png")

if __name__ == "__main__":
    plot_atlas("/DATA1/Roxana/Hierarchy/processed_train_data/giant_umap_subsample.pt")