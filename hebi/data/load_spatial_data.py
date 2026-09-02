#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 15:24:18 2025

@author: roxana
"""
# load_spatial_data.py
import scanpy as sc
import numpy as np
import torch
from typing import Tuple
from Alignment import harmonise_adata
def _find_spatial_key(adata) -> str:
    """
    Automatically detect the correct key in adata.obsm that stores spatial coordinates.
    Handles common variations and typos.
    """
    possible_keys = [
        "spatial", "Spatial", "X_spatial", "spatial_coords",
        "spatial_location", "spatial_locations", "X_spatial_coords",
        "spatial_coordinate", "spatial_coord", "coordinates", "coords"
    ]
    found = []
    for k in adata.obsm.keys():
        for ref in possible_keys:
            if ref.lower() in k.lower():
                found.append(k)

    if len(found) == 0:
        raise KeyError(
            f"No spatial coordinate key found in adata.obsm.\n"
            f"Available keys: {list(adata.obsm.keys())}\n"
            f"Expected one of: {possible_keys}"
        )
    elif len(found) > 1:
        print(f"[⚠️ Warning] Multiple possible spatial keys found: {found}. Using '{found[0]}' by default.")
    return found[0]
def load_spatial_anndata(path,vpath):
    """
    Load spatial transcriptomics data from .h5ad.
    Returns:
        X (torch.FloatTensor): cell × gene matrix (normalised, log1p)
        coords (torch.FloatTensor): cell × 2 spatial coordinates
        gene_names (list): gene names after feature selection
    """

    adata = sc.read_h5ad(path)
    adata = harmonise_adata(adata,vpath)
   # sc.pp.filter_cells(adata, min_counts=40)
   # sc.pp.filter_cells(adata, max_counts=1000)
    # normalise and log1p
    #sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # select highly variable genes
   # sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat", subset=True)

    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    #adata.obsm['spatial'] = adata.obs[['center_x', 'center_y']].values
    spatial_key = _find_spatial_key(adata)
    coords = adata.obsm[spatial_key]

    # convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.float32)

    return X, coords, list(adata.var_names)
