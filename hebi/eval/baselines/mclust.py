#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 15:42:01 2026

@author: tower4
"""

import os, sys, subprocess

# ---- CONFIG: your base R install ----
R_BIN  = "/home/tower4/miniconda3/bin/R"
R_HOME = "/home/tower4/miniconda3/lib/R"
R_LIB  = "/home/tower4/miniconda3/lib/R/lib"

# ---- set env vars BEFORE importing rpy2 ----
os.environ["R_HOME"] = R_HOME
os.environ["RPY2_CFFI_MODE"] = "ABI"
os.environ["PATH"] = "/home/tower4/miniconda3/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = R_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# ---- sanity checks (will raise if broken) ----
print("python:", sys.executable)
print("R RHOME:", subprocess.check_output([R_BIN, "RHOME"]).decode().strip())
print("R version:", subprocess.check_output([R_BIN, "--version"]).decode().splitlines()[0])
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Load R packages
mclust = importr("mclust")

def mclust_cluster_matrix(adata, obsm_key, obs_key, G=7, model="EEE"):
    # 1. Prepare data (ensure float64 for R compatibility)
    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    n, d = X.shape
    
    print(f"Running mclust on {n} x {d} matrix...")

    # 2. Manually create the R matrix
    rX = ro.r.matrix(
        ro.FloatVector(X.ravel(order="F")), 
        nrow=n, 
        ncol=d
    )
    
    # 3. Assign column names (V1, V2...) to prevent 'dimnames' error
    ro.globalenv['rX_temp'] = rX
    ro.r(f'colnames(rX_temp) <- paste0("V", 1:{d})')
    rX = ro.globalenv['rX_temp']

    # 4. FIX: Generate a random sample of indices for initialization
    #    This ensures we use 2000 random points, not just the 2000th point.
    subset_size = 2000
    if n < subset_size:
        subset_size = n # Handle small datasets
        
    print(f"Initializing with a random subset of {subset_size} cells...")
    
    # Create the initialization list with actual indices
    # sample(1:n, size)
    init_cmd = f'list(subset = sample(1:{n}, {subset_size}))'
    
    # 5. Run Mclust
    print("Executing Mclust (this may take a moment)...")
    res = mclust.Mclust(
        rX, 
        G=G, 
        modelNames=ro.StrVector([model]),
        initialization=ro.r(init_cmd) 
    )

    # 6. Extract results
    if 'classification' in res.names:
        r_labels = res.rx2("classification")
        # Convert 1-based (R) to 0-based (Python)
        labels = np.array(r_labels, dtype=int) - 1 
        
        adata.obs[obs_key] = pd.Categorical(labels.astype(str))
        print(f"Success! Labels stored in adata.obs['{obs_key}']")
    else:
        print("Error: Mclust returned no classification.")
        
    # Clean up R environment
    ro.r('rm(rX_temp)')
    
    return res

# --- Run the function ---
res_z0 = mclust_cluster_matrix(adata, "raw", "mclust_raw", G=7, model="EEE")

# --- Evaluate results ---
from sklearn.metrics import adjusted_rand_score

if "domain" in adata.obs:
    gt = adata.obs["domain"].astype(str)
    if "mclust_z0" in adata.obs:
        ari = adjusted_rand_score(gt, adata.obs["mclust_raw"].astype(str))
        print(f"ARI(domain vs mclust_z0): {ari:.4f}")