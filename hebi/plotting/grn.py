#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:41:18 2026

@author: tower4
"""

#!/usr/bin/env python3
"""
GRN Plotting Functions for HEBI Pipeline
-----------------------------------------
Adds three plot types to grn_report.py:

1. plot_niche_grn_heatmap()       — top GRN edges per discovered niche (kmeans labels)
2. plot_ccc_whole_tissue()        — LigRec cell-cell communication across whole tissue
3. plot_ccc_niche_specific()      — LigRec communication between specific niches

Usage — paste into grn_report.py or import from here, then call from
extract_and_plot_grn_attention() in utils_clustering.py.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import scanpy as sc
import torch
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Colour palette — consistent across all plots
# ─────────────────────────────────────────────────────────────
EDGE_TYPE_COLORS = {
    "pathway":   "#E8B07D",   # apricot
    "spearman":  "#9C9145",   # darker olive  ← changed from #D9CB87
    "ligrec":    "#E0917F",   # salmon
    "tf_target": "#C9A8B4",   # dusty mauve
    "unknown":   "#DAD3C8",   # warm gray
}
HEBI_CMAP = LinearSegmentedColormap.from_list(
    "hebi",
    ["#F7F7F7", "#2D6A9F", "#1A3A5C"],
    N=256,
)

# ─────────────────────────────────────────────────────────────
# Helper: extract LigRec edges from gene_graphs
# ─────────────────────────────────────────────────────────────
def _extract_ligrec_edges(gene_graphs, gene_names, grn_attn_weights,
                           edge_attr_names=None):
    """Return DataFrame of LigRec edges with attention scores."""
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    ligrec_idx = edge_attr_names.index("ligrec") if "ligrec" in edge_attr_names else None
    if ligrec_idx is None:
        return pd.DataFrame()

    g = gene_graphs["global"]
    edge_index = g.edge_index.cpu().numpy()
    edge_attr  = g.edge_attr.cpu().numpy()
    attn = grn_attn_weights.detach().cpu().numpy() \
        if torch.is_tensor(grn_attn_weights) else np.asarray(grn_attn_weights).ravel()

    rows = []
    for e in range(edge_index.shape[1]):
        if edge_attr[e, ligrec_idx] > 0:
            rows.append({
                "ligand":   gene_names[int(edge_index[0, e])],
                "receptor": gene_names[int(edge_index[1, e])],
                "attention": float(attn[e]),
                "pair":     f"{gene_names[int(edge_index[0,e])]} → {gene_names[int(edge_index[1,e])]}",
            })
    return pd.DataFrame(rows).sort_values("attention", ascending=False)


# ─────────────────────────────────────────────────────────────
# Helper: cluster-contextualised scores
# ─────────────────────────────────────────────────────────────
def _cluster_edge_scores(adata, cluster_key, gene_graphs, gene_names,
                          grn_attn_weights, edge_attr_names=None):
    """
    For each cluster, compute mean expression-weighted attention score per edge.
    Returns dict { cluster_label -> pd.DataFrame(edge rows with 'score') }
    """
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    g = gene_graphs["global"]
    edge_index = g.edge_index.cpu().numpy()
    edge_attr  = g.edge_attr.cpu().numpy()
    attn = grn_attn_weights.detach().cpu().numpy() \
        if torch.is_tensor(grn_attn_weights) else np.asarray(grn_attn_weights).ravel()

    gene_to_idx = {name: i for i, name in enumerate(gene_names)}

    import scipy.sparse as sp
    X = adata.X
    is_sparse = sp.issparse(X)

    clusters = adata.obs[cluster_key].astype(str).values
    unique_clusters = sorted(set(clusters))

    # mean expression per cluster
    mean_expr = {}
    for c in unique_clusters:
        mask = clusters == c
        sub = X[mask]
        mean_expr[c] = (np.asarray(sub.mean(axis=0)).ravel()
                        if is_sparse else sub.mean(axis=0))

    var_names = list(adata.var_names)
    var_to_col = {g: i for i, g in enumerate(var_names)}

    results = {}
    for c in unique_clusters:
        rows = []
        expr = mean_expr[c]
        for e in range(edge_index.shape[1]):
            src_gene = gene_names[int(edge_index[0, e])]
            dst_gene = gene_names[int(edge_index[1, e])]
            a = float(attn[e])

            lig_expr = expr[var_to_col[src_gene]] if src_gene in var_to_col else 0.0
            rec_expr = expr[var_to_col[dst_gene]] if dst_gene in var_to_col else 0.0
            score = a * float(lig_expr) * float(rec_expr)

            active = [edge_attr_names[i]
                      for i in range(len(edge_attr_names))
                      if i < edge_attr.shape[1] and edge_attr[e, i] > 0]
            edge_type = active[0] if active else "unknown"

            rows.append({
                "source": src_gene,
                "target": dst_gene,
                "attention": a,
                "score": score,
                "edge_type": edge_type,
                "pair": f"{src_gene} → {dst_gene}",
            })
        results[c] = pd.DataFrame(rows).sort_values("score", ascending=False)

    return results


# ═══════════════════════════════════════════════════════════════
# PLOT 1: Per-niche GRN heatmap
# ═══════════════════════════════════════════════════════════════
def plot_niche_grn_heatmap(
    adata,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    cluster_key="kmeans",          # use kmeans, not annotation
    edge_attr_names=None,
    top_k_edges=15,
    top_k_niches=None,             # None = all
    figsize=None,
    save_path=None,
):
    """
    Heatmap: rows = top GRN edges (by mean attention across niches),
             cols = discovered niches,
             values = cluster-contextualised score.

    Use cluster_key='kmeans' for annotation-free operation.
    """
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    cluster_scores = _cluster_edge_scores(
        adata, cluster_key, gene_graphs, gene_names,
        grn_attn_weights, edge_attr_names
    )

    clusters = sorted(cluster_scores.keys())
    if top_k_niches:
        clusters = clusters[:top_k_niches]

    # Union of top-k edges per niche
    edge_pool = set()
    for c in clusters:
        df = cluster_scores[c]
        edge_pool.update(df.head(top_k_edges)["pair"].tolist())

    # Build matrix
    matrix = pd.DataFrame(index=sorted(edge_pool), columns=clusters, dtype=float)
    for c in clusters:
        df = cluster_scores[c].set_index("pair")
        for edge in edge_pool:
            matrix.loc[edge, c] = df.loc[edge, "score"] if edge in df.index else 0.0

    # Sort rows by variance across niches (most niche-specific first)
    matrix = matrix.fillna(0.0)
    matrix["_var"] = matrix.var(axis=1)
    matrix = matrix.sort_values("_var", ascending=False).drop(columns="_var")

    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (max(10, n_cols * 0.9), max(8, n_rows * 0.45))

    fig, ax = plt.subplots(figsize=figsize)

    vals = matrix.values.astype(float)
    vmax = np.percentile(vals[vals > 0], 95) if (vals > 0).any() else 1.0
    im = ax.imshow(vals, aspect="auto", cmap=HEBI_CMAP, vmin=0, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Expression-weighted attention")
    ax.set_title(
        f"Niche-specific GRN activity\n(cluster_key='{cluster_key}', top {top_k_edges} edges/niche)",
        fontsize=12, pad=12
    )
    ax.set_xlabel("Discovered niches", fontsize=10)
    ax.set_ylabel("Gene interaction", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved niche GRN heatmap → {save_path}")
    plt.show()
    return matrix


# ═══════════════════════════════════════════════════════════════
# PLOT 2: Whole-tissue CCC bubble plot
# ═══════════════════════════════════════════════════════════════
def plot_ccc_whole_tissue(
    adata,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    cluster_key="kmeans",
    edge_attr_names=None,
    top_k_pairs=20,
    figsize=(14, 9),
    save_path=None,
):
    """
    Bubble plot: x = sender niche, y = LigRec pair,
                 bubble size & colour = communication score.

    Sender = niche with highest mean ligand expression.
    """
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    lr_df = _extract_ligrec_edges(gene_graphs, gene_names,
                                   grn_attn_weights, edge_attr_names)
    if lr_df.empty:
        print("  ⚠️  No LigRec edges found — skipping CCC plot.")
        return

    cluster_scores = _cluster_edge_scores(
        adata, cluster_key, gene_graphs, gene_names,
        grn_attn_weights, edge_attr_names
    )

    clusters = sorted(cluster_scores.keys())

    # Top LigRec pairs by mean attention
    top_pairs = lr_df.head(top_k_pairs)["pair"].tolist()

    # Build score matrix for those pairs
    rows_plot = []
    for c in clusters:
        df = cluster_scores[c].set_index("pair")
        for pair in top_pairs:
            score = df.loc[pair, "score"] if pair in df.index else 0.0
            rows_plot.append({"niche": c, "pair": pair, "score": score})

    plot_df = pd.DataFrame(rows_plot)
    plot_df = plot_df[plot_df["score"] > 0]

    if plot_df.empty:
        print("  ⚠️  All CCC scores are zero — skipping.")
        return

    niche_order = clusters
    pair_order  = top_pairs

    niche_map = {n: i for i, n in enumerate(niche_order)}
    pair_map  = {p: i for i, p in enumerate(pair_order)}

    plot_df["x"] = plot_df["niche"].map(niche_map)
    plot_df["y"] = plot_df["pair"].map(pair_map)

    s_vals = plot_df["score"].values
    s_norm = (s_vals - s_vals.min()) / (s_vals.max() - s_vals.min() + 1e-8)
    sizes  = 30 + 400 * s_norm

    fig, ax = plt.subplots(figsize=figsize)
    sc_obj = ax.scatter(
        plot_df["x"], plot_df["y"],
        s=sizes, c=plot_df["score"],
        cmap="YlOrRd", alpha=0.85,
        edgecolors="#333333", linewidths=0.4,
    )
    plt.colorbar(sc_obj, ax=ax, shrink=0.6, label="Communication score\n(attention × ligand expr × receptor expr)")

    ax.set_xticks(range(len(niche_order)))
    ax.set_xticklabels(niche_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pair_order)))
    ax.set_yticklabels(pair_order, fontsize=9)
    ax.set_xlabel("Sender niche", fontsize=10)
    ax.set_ylabel("Ligand → Receptor pair", fontsize=10)
    ax.set_title(
        f"Cell–Cell Communication (whole tissue)\n"
        f"Top {top_k_pairs} LigRec pairs · cluster_key='{cluster_key}'",
        fontsize=12, pad=12
    )
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved whole-tissue CCC → {save_path}")
    plt.show()
    return plot_df


# ═══════════════════════════════════════════════════════════════
# PLOT 3: Niche-specific CCC chord / matrix
# ═══════════════════════════════════════════════════════════════
def plot_ccc_niche_specific(
    adata,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    cluster_key="kmeans",
    edge_attr_names=None,
    top_k_pairs=15,
    figsize=(14, 10),
    save_path=None,
):
    """
    Matrix heatmap: rows = LigRec pair, cols = sender niche,
    cells filled only where score exceeds median threshold.
    Annotates each cell with score value for interpretability.
    """
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    lr_df = _extract_ligrec_edges(gene_graphs, gene_names,
                                   grn_attn_weights, edge_attr_names)
    if lr_df.empty:
        print("  ⚠️  No LigRec edges found.")
        return

    cluster_scores = _cluster_edge_scores(
        adata, cluster_key, gene_graphs, gene_names,
        grn_attn_weights, edge_attr_names
    )

    clusters   = sorted(cluster_scores.keys())
    top_pairs  = lr_df.head(top_k_pairs)["pair"].tolist()

    # Build matrix
    mat = pd.DataFrame(0.0, index=top_pairs, columns=clusters)
    for c in clusters:
        df = cluster_scores[c].set_index("pair")
        for pair in top_pairs:
            if pair in df.index:
                mat.loc[pair, c] = df.loc[pair, "score"]

    threshold = mat.values[mat.values > 0].mean() * 0.3 if (mat.values > 0).any() else 0

    fig, ax = plt.subplots(figsize=figsize)
    vals = mat.values.astype(float)
    vmax = np.percentile(vals[vals > 0], 95) if (vals > 0).any() else 1.0

    im = ax.imshow(vals, aspect="auto",
                   cmap="Blues", vmin=0, vmax=vmax)

    # Annotate cells above threshold
    for i in range(len(top_pairs)):
        for j in range(len(clusters)):
            v = vals[i, j]
            if v > threshold:
                txt = f"{v:.2e}" if v < 0.001 else f"{v:.3f}"
                color = "white" if v > vmax * 0.6 else "#1A3A5C"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(clusters, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_pairs)))
    ax.set_yticklabels(top_pairs, fontsize=8)
    ax.set_xlabel("Sender niche", fontsize=10)
    ax.set_ylabel("Ligand → Receptor pair", fontsize=10)
    ax.set_title(
        f"Niche-specific LigRec communication\n"
        f"(threshold = {threshold:.4f}, cluster_key='{cluster_key}')",
        fontsize=12, pad=12
    )
    plt.colorbar(im, ax=ax, shrink=0.55, label="Communication score")

    # Add LigRec pair source annotation on right side
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(top_pairs)))
    ligrec_attns = []
    for pair in top_pairs:
        row = lr_df[lr_df["pair"] == pair]
        ligrec_attns.append(f"{row['attention'].values[0]:.3f}" if len(row) > 0 else "")
    ax2.set_yticklabels(ligrec_attns, fontsize=7, color="#E8703A")
    ax2.set_ylabel("GRN attention", fontsize=9, color="#E8703A")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved niche-specific CCC → {save_path}")
    plt.show()
    return mat


# ═══════════════════════════════════════════════════════════════
# Master plotting call — add to extract_and_plot_grn_attention()
# ═══════════════════════════════════════════════════════════════
def run_grn_plots(
    adata,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    cluster_key="kmeans",           # ← USE THIS, not "annotation"
    edge_attr_names=None,
    out_dir="./grn_reports",
    top_k_edges=15,
    top_k_pairs=20,
):
    """
    Call all three GRN plots in sequence.
    Add this call at the end of extract_and_plot_grn_attention() in utils_clustering.py.

    Example:
        from grn_plots import run_grn_plots
        run_grn_plots(adata, gene_graphs, gene_names, grn_attn_weights,
                      cluster_key="kmeans", out_dir="./grn_reports")
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    print("\n📊 [1/3] Niche GRN heatmap...")
    plot_niche_grn_heatmap(
        adata=adata,
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        grn_attn_weights=grn_attn_weights,
        cluster_key=cluster_key,
        edge_attr_names=edge_attr_names,
        top_k_edges=top_k_edges,
        save_path=os.path.join(out_dir, "niche_grn_heatmap.png"),
    )

    print("\n📊 [2/3] Whole-tissue CCC bubble plot...")
    plot_ccc_whole_tissue(
        adata=adata,
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        grn_attn_weights=grn_attn_weights,
        cluster_key=cluster_key,
        edge_attr_names=edge_attr_names,
        top_k_pairs=top_k_pairs,
        save_path=os.path.join(out_dir, "ccc_whole_tissue.png"),
    )

    print("\n📊 [3/3] Niche-specific CCC matrix...")
    plot_ccc_niche_specific(
        adata=adata,
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        grn_attn_weights=grn_attn_weights,
        cluster_key=cluster_key,
        edge_attr_names=edge_attr_names,
        top_k_pairs=top_k_pairs,
        save_path=os.path.join(out_dir, "ccc_niche_specific.png"),
    )

    print(f"\n✅ All GRN plots saved to {out_dir}")