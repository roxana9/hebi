#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:30:23 2026

@author: tower4
"""

# Plot housekeeping genes in NORMAL vs CANCER liver (side-by-side)
# - Works for spatial datasets with adata.obsm["spatial"]
# - Uses the SAME scaling (0–1) for both datasets so plots are comparable
# - One colourbar only (right side)

import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ----------------------------
# 1) Put your housekeeping genes here (HUMAN liver)
# ----------------------------
housekeeping_genes = [
    "ACTB", "GAPDH", "RPLP0", "RPL13A", "PPIA", "B2M", "HPRT1", "EEF1A1", "RPS18", "TUBA1B"
]

# ----------------------------
# 2) Helpers
# ----------------------------
def _find_gene_index(ad, gene, gene_symbol_col="gene_symbol"):
    g = gene.upper()
    # try var_names
    vn = ad.var_names.astype(str).str.upper()
    hit = np.where(vn.values == g)[0]
    if len(hit) > 0:
        return int(hit[0])

    # try var[gene_symbol_col]
    if gene_symbol_col in ad.var.columns:
        gs = ad.var[gene_symbol_col].astype(str).str.upper().values
        hit = np.where(gs == g)[0]
        if len(hit) > 0:
            return int(hit[0])

    return None

def _get_expr_vec(ad, gene, gene_symbol_col="gene_symbol"):
    idx = _find_gene_index(ad, gene, gene_symbol_col=gene_symbol_col)
    if idx is None:
        return None
    X = ad.X
    v = X[:, idx]
    # sparse -> dense
    try:
        v = v.toarray().ravel()
    except Exception:
        v = np.asarray(v).ravel()
    return v.astype(float)

def _scale_01(v, mode="p1p99", p_low=1, p_high=99):
    v = np.asarray(v, dtype=float)
    if mode == "minmax":
        lo, hi = np.nanmin(v), np.nanmax(v)
    else:
        lo, hi = np.nanpercentile(v, p_low), np.nanpercentile(v, p_high)
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi): hi = lo + 1.0
    if hi <= lo: hi = lo + 1e-12
    v = np.clip(v, lo, hi)
    return (v - lo) / (hi - lo + 1e-12)

def _prep_adata_for_plot(adata, split_key=None, split_value=None, basis="spatial",
                         do_norm_log=True, flip_y=True):
    if split_key is not None and split_value is not None:
        ad = adata[adata.obs[split_key] == split_value].copy()
    else:
        ad = adata.copy()

    if basis not in ad.obsm:
        raise KeyError(f"Missing ad.obsm['{basis}']. Available: {list(ad.obsm_keys())}")

    # flip y once if you want
    if flip_y:
        ad.obsm[basis] = ad.obsm[basis].copy()
        ad.obsm[basis][:, 1] *= -1

    if do_norm_log:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)

    return ad

def plot_housekeeping_normal_vs_cancer(
    adata_normal,
    adata_cancer,
    genes,
    split_key_normal="split",
    split_value_normal="test",
    split_key_cancer="split",
    split_value_cancer="test",
    basis="spatial",
    gene_symbol_col="gene_symbol",
    do_norm_log=True,
    flip_y=True,
    scale_mode="p1p99",
    p_low=1,
    p_high=99,
    cmap="cividis",        # lighter than magma; good option
    point_size=2.0,
    save_path=None,
):
    # subset + preprocess
    ad_n = _prep_adata_for_plot(adata_normal, split_key_normal, split_value_normal, basis, do_norm_log, flip_y)
    ad_c = _prep_adata_for_plot(adata_cancer, split_key_cancer, split_value_cancer, basis, do_norm_log, flip_y)

    # compute a GLOBAL min/max scaling per gene using BOTH datasets (so comparable)
    # store scaled values in .obs for plotting
    genes_ok = []
    for g in genes:
        v_n = _get_expr_vec(ad_n, g, gene_symbol_col)
        v_c = _get_expr_vec(ad_c, g, gene_symbol_col)
        if v_n is None or v_c is None:
            print(f"⚠️ Skipping {g} (not found in one of the datasets)")
            continue

        # concatenate for consistent scaling
        v_all = np.concatenate([v_n, v_c], axis=0)
        v_all01 = _scale_01(v_all, mode=scale_mode, p_low=p_low, p_high=p_high)

        v_n01 = v_all01[: len(v_n)]
        v_c01 = v_all01[len(v_n):]

        k = f"{g}__01"
        ad_n.obs[k] = v_n01
        ad_c.obs[k] = v_c01
        genes_ok.append(g)

    if len(genes_ok) == 0:
        raise RuntimeError("No genes to plot (none found in both datasets).")

    # layout: rows = genes, cols = [Normal, Cancer]
    nrows = len(genes_ok)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(8.5, 2.4 * nrows))
    if nrows == 1:
        axes = np.array([axes])  # make it [1,2]

    # draw plots (no individual colourbars)
    for i, g in enumerate(genes_ok):
        key = f"{g}__01"

        axL = axes[i, 0]
        axR = axes[i, 1]

        sc.pl.embedding(
            ad_n,
            basis=basis,
            color=key,
            cmap=cmap,
            vmin=0, vmax=1,
            size=point_size,
            ax=axL,
            show=False,
            title=g if i == 0 else "",   # keep clean
            colorbar_loc=None
        )
        sc.pl.embedding(
            ad_c,
            basis=basis,
            color=key,
            cmap=cmap,
            vmin=0, vmax=1,
            size=point_size,
            ax=axR,
            show=False,
            title=g if i == 0 else "",
            colorbar_loc=None
        )

        # remove frames / ticks
        for ax in (axL, axR):
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

        # gene label on left margin
        axL.text(
            -0.02, 0.5, g,
            transform=axL.transAxes,
            ha="right", va="center",
            fontsize=12, fontweight="bold"
        )

    # column titles
    axes[0, 0].set_title("Normal", fontsize=14, fontweight="bold", pad=10)
    axes[0, 1].set_title("Cancer", fontsize=14, fontweight="bold", pad=10)

    # one colourbar (right)
    norm = Normalize(vmin=0, vmax=1)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Expression (scaled 0–1)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle("Housekeeping genes (Normal vs Cancer)", fontsize=16, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    return fig

# ----------------------------
# 3) RUN (replace adata_normal / adata_cancer with yours)
# ----------------------------
# Example:
normal = sc.read_h5ad("/DATA1/Roxana/Data/Normal_liver_cosmax.h5ad")
cancer = sc.read_h5ad("/DATA1/Roxana/Data/cancer_liver_cosmax.h5ad")

fig = plot_housekeeping_normal_vs_cancer(
    adata_normal=normal,
    adata_cancer=cancer,
    genes=housekeeping_genes,
    split_key_normal="split", split_value_normal="test",
    split_key_cancer="split", split_value_cancer="test",
    basis="spatial",
    gene_symbol_col="gene_symbol",
    do_norm_log=True,
    flip_y=True,
    cmap="cividis",
    point_size=2.0,
    save_path="housekeeping_normal_vs_cancer.png",
)