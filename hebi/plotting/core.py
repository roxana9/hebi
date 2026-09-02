#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:38:56 2025

@author: tower4
"""

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
    viz_list = torch.load(subsample_path, weights_only=False)
    
    # 1. Reconstruct a Giant AnnData object using PCA embeddings
    X_list = []
    obs_list = []
    
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
        
        # --- DIMENSIONALITY REDUCTION (Initial Embedding) ---
        n_genes = X_raw.shape[1]
        var_names = [f'gene_{j}' for j in range(n_genes)]
        
        adata_temp = anndata.AnnData(X=X_raw, obs=df, var=pd.DataFrame(index=var_names))
        
        # NOTE: We skip normalization/scaling here, assuming it's done in preprocessing. 
        # For better results, ensure X_raw is normalized/log-transformed before PCA.
        
        sc.tl.pca(adata_temp, n_comps=N_COMPONENTS, use_highly_variable=False, svd_solver='arpack')
        X_pca = adata_temp.obsm['X_pca']
        
        X_list.append(X_pca)
        obs_list.append(df)
        
    # Concatenate the low-dimensional (PCA) embeddings
    X_all = np.concatenate(X_list, axis=0)
    obs_all = pd.concat(obs_list, axis=0).reset_index(drop=True)
    
    # Create the final AnnData object using the integrated PCA embeddings
    adata = anndata.AnnData(X=X_all, obs=obs_all)
    print(f"Giant Atlas Shape: {adata.shape}")
    
    # 3. Compute UMAP on the integrated space
    print("Computing Neighbors & UMAP...")
    
    # Store the integrated PCA embedding in the standard .obsm location
    adata.obsm['X_pca'] = adata.X 
    
    # --- FIX 4: Integration/Normalization using Harmony ---
    # Harmony corrects for large batch/species differences (like mouse vs. human)
    print("Applying Harmony for cross-species integration...")
    sc.external.pp.harmony_integrate(adata, key='species')
    
    # 4. Compute Neighbors on the integrated (Harmony) representation
    # The Harmony result is stored in adata.obsm['X_pca_harmony']
    sc.pp.neighbors(adata, use_rep='X_pca_harmony')
    
    # 5. Compute UMAP
    sc.tl.umap(adata, min_dist=0.5) 
    
    # 6. Plotting
    print("Generating Plots...")
    
    # Plot 1: Colored by Tissue (Should now be clustered more purely by tissue/cell type)
    sc.pl.umap(adata, color=['tissue'], title="Global Atlas: Tissue Type (Integrated)", show=False, frameon=False)
    plt.savefig("global_umap_tissue_integrated.png", bbox_inches='tight', dpi=300)
    print("Saved global_umap_tissue_integrated.png")
    
    # Plot 2: Colored by Disease Status
    sc.pl.umap(adata, color=['disease'], title="Global Atlas: Disease Status (Integrated)", palette={'Normal': 'g', 'Diseased': 'r', 'unknown': 'grey'}, show=False, frameon=False)
    plt.savefig("global_umap_disease_integrated.png", bbox_inches='tight', dpi=300)
    print("Saved global_umap_disease_integrated.png")

    # Plot 3: Colored by Species (Crucial Check: Should appear well-mixed/overlapping)
    sc.pl.umap(adata, color=['species'], title="Global Atlas: Species (Integration Check)", show=False, frameon=False)
    plt.savefig("global_umap_species_integrated.png", bbox_inches='tight', dpi=300)
    print("Saved global_umap_species_integrated.png")

if __name__ == "__main__":
    plot_atlas("/DATA1/Roxana/Hierarchy/processed_train_data/giant_umap_subsample.pt")