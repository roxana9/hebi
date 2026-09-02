#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:04:09 2026

@author: tower4
"""
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
def plot_batch_effect_umap(
    adata_list,           # list of adata per slide
    slide_names,          # list of slide labels e.g. ["Xenium_1", "Xenium_2"]
    embedding_key_before="X_pca",   # raw PCA
    embedding_key_after="z0_embedding",  # HEBI embedding
    cluster_key="kmeans",
    save_path=None,
    
):
    """
    2x2 UMAP showing batch effect before and after HEBI training.
    """
    import scanpy as sc

    # ── Combine all slides into one adata ──
    for i, (adata, name) in enumerate(zip(adata_list, slide_names)):
        adata.obs["slide"] = name
        adata.obs["slide_idx"] = str(i)

    combined = adata_list[0].concatenate(
        adata_list[1:],
        batch_key="slide_batch",
        batch_categories=slide_names,
    )

    # ── BEFORE: raw gene expression PCA ──
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.highly_variable_genes(combined, n_top_genes=2000)
    sc.pp.pca(combined, n_comps=50)
    sc.pp.neighbors(combined, use_rep="X_pca")
    sc.tl.umap(combined)
    combined.obsm["X_umap_before"] = combined.obsm["X_umap"].copy()

    # ── AFTER: HEBI z0 embedding ──
    # z0_embedding already stored in adata.obsm from encoding
    sc.pp.neighbors(combined, use_rep="z0_embedding", key_added="hebi_neighbors")
    sc.tl.umap(combined, neighbors_key="hebi_neighbors")
    combined.obsm["X_umap_after"] = combined.obsm["X_umap"].copy()

    # ── Plot 2x2 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def scatter(ax, umap_coords, color_vals, title, cmap="tab10", legend=True):
        unique_vals = sorted(set(color_vals))
        cmap_obj = plt.cm.get_cmap(cmap, len(unique_vals))
        val_to_idx = {v: i for i, v in enumerate(unique_vals)}
        colors = [cmap_obj(val_to_idx[v]) for v in color_vals]
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                   c=colors, s=0.5, alpha=0.4, rasterized=True)
        if legend:
            handles = [plt.Line2D([0],[0], marker='o', color='w',
                                   markerfacecolor=cmap_obj(val_to_idx[v]),
                                   markersize=6, label=v)
                       for v in unique_vals]
            ax.legend(handles=handles, fontsize=7,
                      loc="upper right", markerscale=2,
                      framealpha=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    umap_before = combined.obsm["X_umap_before"]
    umap_after  = combined.obsm["X_umap_after"]
    slides      = combined.obs["slide"].astype(str).tolist()
    domains     = combined.obs[cluster_key].astype(str).tolist()

    scatter(axes[0,0], umap_before, slides,
            "Before HEBI\n(colored by slide)",
            cmap="Set1")
    scatter(axes[0,1], umap_after, slides,
            "After HEBI\n(colored by slide — batch removed)",
            cmap="Set1")
    scatter(axes[1,0], umap_before, domains,
            "Before HEBI\n(colored by domain)",
            cmap="tab20")
    scatter(axes[1,1], umap_after, domains,
            "After HEBI\n(colored by domain — consistent across slides)",
            cmap="tab20")

    fig.suptitle(
        "HEBI Multi-slide Batch Effect Removal\n"
        "Row 1: slide identity  ·  Row 2: spatial domains",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    if save_path:
        import os
        path=os.path.join(save_path, "multi_slide_spatial.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved UMAP → {save_path}")
    plt.show()
    return combined