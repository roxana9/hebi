#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# ==========================================================
# Core config
# ==========================================================
EDGE_TYPE_NAMES = ["Pathway", "Spearman", "LigRec", "TF-Target"]


# ==========================================================
# Helpers
# ==========================================================
def _get_mean_expr_and_pct_expr(cells):
    """
    Returns:
      mean_expr_scaled : np.ndarray shape [G], min-max scaled to [0, 1] by max
      pct_expr         : np.ndarray shape [G], fraction of cells with expr > 0
    """
    X = cells.X

    if sp.issparse(X):
        mean_expr = np.asarray(X.mean(axis=0)).flatten()
        pct_expr = np.asarray((X > 0).mean(axis=0)).flatten()
    else:
        mean_expr = np.asarray(X.mean(axis=0)).flatten()
        pct_expr = np.asarray((X > 0).mean(axis=0)).flatten()

    mean_expr = np.nan_to_num(mean_expr, nan=0.0, posinf=0.0, neginf=0.0)
    pct_expr = np.nan_to_num(pct_expr, nan=0.0, posinf=0.0, neginf=0.0)

    if mean_expr.max() > 0:
        mean_expr_scaled = mean_expr / mean_expr.max()
    else:
        mean_expr_scaled = mean_expr.copy()

    return mean_expr_scaled, pct_expr


def _gene_activity(mean_expr_scaled, pct_expr, alpha_expr=0.7, alpha_pct=0.3):
    """
    Cluster gene activity score in [0,1]-ish range.

    We combine:
      - scaled mean expression
      - fraction of expressing cells

    Both are square-root transformed to reduce over-penalisation.
    """
    expr_part = np.sqrt(np.clip(mean_expr_scaled, 0.0, None))
    pct_part = np.sqrt(np.clip(pct_expr, 0.0, None))

    activity = alpha_expr * expr_part + alpha_pct * pct_part
    return activity


def _resolve_edge_types(attr_row):
    idx = np.where(attr_row > 0)[0].tolist()
    if len(idx) == 0:
        return "Unknown"
    return "|".join([EDGE_TYPE_NAMES[i] for i in idx])


def _dominant_edge_type(attr_row):
    """
    For display only. If multiple tags exist, priority is:
      TF-Target > LigRec > Pathway > Spearman
    """
    if attr_row[3] == 1:
        return 3
    if attr_row[2] == 1:
        return 2
    if attr_row[0] == 1:
        return 0
    if attr_row[1] == 1:
        return 1
    return None


def _build_target_in_degree(edge_index):
    """
    target_in_degree[v] = number of incoming GRN edges to target gene v
    """
    num_nodes = int(edge_index.max()) + 1 if edge_index.size > 0 else 0
    in_deg = np.zeros(num_nodes, dtype=np.int64)

    for v in edge_index[1]:
        in_deg[v] += 1

    return in_deg


def _edge_competition_factor(target_in_degree, mode="log"):
    """
    Convert target in-degree into a competition factor.

    mode="log" => log2(1 + in_degree)
    """
    target_in_degree = np.asarray(target_in_degree)

    if mode == "log":
        return np.log2(1.0 + target_in_degree)
    elif mode == "sqrt":
        return np.sqrt(np.maximum(target_in_degree, 0))
    elif mode == "none":
        return np.ones_like(target_in_degree, dtype=float)
    else:
        raise ValueError(f"Unknown competition mode: {mode}")


# ==========================================================
# Edge-type-specific scoring
# ==========================================================
def _score_edge_by_type(
    edge_attr_row,
    base_attention,
    src_activity,
    dst_activity,
    competition_factor=1.0,
):
    """
    Cluster-contextualised edge relevance.

    We use global learned attention as base, then modulate by cluster activity.
    Different edge types use different biological weighting rules.

    Priority if edge has multiple tags:
      TF-Target > LigRec > Pathway > Spearman

    Rules:
      TF-Target:
          attention * src_activity * sqrt(dst_activity) * competition
      LigRec:
          attention * sqrt(src_activity) * dst_activity * competition
      Pathway:
          attention * src_activity * sqrt(dst_activity) * competition
      Spearman:
          attention * sqrt(src_activity * dst_activity) * competition
    """
    a = float(base_attention)
    s = float(max(src_activity, 0.0))
    d = float(max(dst_activity, 0.0))
    c = float(max(competition_factor, 1e-8))

    # TF-Target
    if edge_attr_row[3] == 1:
        return a * s * np.sqrt(d) * c

    # LigRec
    if edge_attr_row[2] == 1:
        return a * np.sqrt(s) * d * c

    # Pathway
    if edge_attr_row[0] == 1:
        return a * s * np.sqrt(d) * c

    # Spearman
    if edge_attr_row[1] == 1:
        return a * np.sqrt(s * d) * c

    return a * np.sqrt(s * d) * c


# ==========================================================
# Core computation
# ==========================================================
def compute_cluster_contextual_grn_scores(
    adata,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    cluster_key="z0_cluster",
    min_cells=5,
    alpha_expr=0.7,
    alpha_pct=0.3,
    competition_mode="log",
):
    """
    Main engine.

    Returns:
      result dict with:
        - edge_index
        - edge_attr
        - global_attn
        - clusters
        - cluster_scores        : dict cluster -> np.ndarray [E]
        - cluster_gene_activity : dict cluster -> np.ndarray [G]
        - target_in_degree      : np.ndarray [G]
        - edge_comp_factor      : np.ndarray [E]
    """
    print("\n🧬 Computing cluster-contextualised GRN edge scores...")

    data = gene_graphs["global"]
    edge_index = data.edge_index.detach().cpu().numpy()   # [2, E]
    edge_attr = data.edge_attr.detach().cpu().numpy()     # [E, 4]
    global_attn = grn_attn_weights.detach().cpu().numpy().flatten()  # [E]

    if edge_index.shape[1] != len(global_attn):
        raise ValueError("Mismatch: edge count and grn_attn_weights length differ.")

    clusters_all = sorted(adata.obs[cluster_key].unique())
    clusters = []

    target_in_degree = _build_target_in_degree(edge_index)
    edge_comp_factor = _edge_competition_factor(
        target_in_degree=edge_index[1].copy().astype(int).map if False else target_in_degree[edge_index[1]],
        mode=competition_mode,
    )

    cluster_scores = {}
    cluster_gene_activity = {}

    for cluster_id in clusters_all:
        cells = adata[adata.obs[cluster_key] == cluster_id]

        if len(cells) < min_cells:
            print(f"   Skipping cluster {cluster_id}: too few cells ({len(cells)})")
            continue

        clusters.append(cluster_id)

        mean_expr_scaled, pct_expr = _get_mean_expr_and_pct_expr(cells)
        activity = _gene_activity(
            mean_expr_scaled=mean_expr_scaled,
            pct_expr=pct_expr,
            alpha_expr=alpha_expr,
            alpha_pct=alpha_pct,
        )

        scores = np.zeros(edge_index.shape[1], dtype=np.float64)

        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]

            scores[i] = _score_edge_by_type(
                edge_attr_row=edge_attr[i],
                base_attention=global_attn[i],
                src_activity=activity[src],
                dst_activity=activity[dst],
                competition_factor=edge_comp_factor[i],
            )

        cluster_scores[cluster_id] = scores
        cluster_gene_activity[cluster_id] = activity

    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "global_attn": global_attn,
        "clusters": clusters,
        "cluster_scores": cluster_scores,
        "cluster_gene_activity": cluster_gene_activity,
        "target_in_degree": target_in_degree,
        "edge_comp_factor": edge_comp_factor,
    }


# ==========================================================
# 1) Cluster-wise top edge report
# ==========================================================
def extract_and_plot_grn_attention(
    adata,
    trainer,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    edge_attr_names=None,
    cluster_key="z0_cluster",
    top_k_edges=10,
    min_cells=5,
    alpha_expr=0.7,
    alpha_pct=0.3,
    competition_mode="log",
):
    """
    Replacement for your old function.

    Prints:
      - global mean attention per type
      - cluster-specific mean score per type
      - top-k cluster-contextualised edges per cluster
    """
    if edge_attr_names is None:
        edge_attr_names = EDGE_TYPE_NAMES

    result = compute_cluster_contextual_grn_scores(
        adata=adata,
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        grn_attn_weights=grn_attn_weights,
        cluster_key=cluster_key,
        min_cells=min_cells,
        alpha_expr=alpha_expr,
        alpha_pct=alpha_pct,
        competition_mode=competition_mode,
    )

    edge_index = result["edge_index"]
    edge_attr = result["edge_attr"]
    global_attn = result["global_attn"]
    clusters = result["clusters"]
    cluster_scores = result["cluster_scores"]

    print("\n🧬 --- Cluster-Wise GRN Discovery Analysis ---")

    # ---------------------------------------------------
    # Global mean attention by edge type
    # ---------------------------------------------------
    print("\n🌍 Global Mean Attention per Edge Type:")
    for t in range(len(edge_attr_names)):
        mask = edge_attr[:, t] == 1
        mean_val = global_attn[mask].mean() if mask.sum() > 0 else 0.0
        print(f"   {edge_attr_names[t]}: {mean_val:.4f}")

    # ---------------------------------------------------
    # Per cluster
    # ---------------------------------------------------
    for cluster_id in clusters:
        print(f"\n🟢 Analyzing Cluster: {cluster_id}")

        scores = cluster_scores[cluster_id]

        print("\n   Mean Cluster-Contextualised Score per Type:")
        for t in range(len(edge_attr_names)):
            mask = edge_attr[:, t] == 1
            mean_val = scores[mask].mean() if mask.sum() > 0 else 0.0
            print(f"      {edge_attr_names[t]}: {mean_val:.4f}")

        top_indices = np.argsort(scores)[-top_k_edges:][::-1]

        n_curated = 0
        n_denovo = 0

        print(f"   {'Source':<12} -> {'Target':<12} | {'Score':<8} | {'Edge Type'}")
        print("   " + "-" * 72)

        for idx in top_indices:
            src_gene = gene_names[edge_index[0, idx]]
            dst_gene = gene_names[edge_index[1, idx]]
            score = scores[idx]
            attr = edge_attr[idx]

            edge_type_label = _resolve_edge_types(attr)

            if attr[0] == 1 or attr[2] == 1 or attr[3] == 1:
                n_curated += 1
            elif attr[1] == 1:
                n_denovo += 1

            print(
                f"   {src_gene:<12} -> {dst_gene:<12} | "
                f"{score:>8.4f} | {edge_type_label}"
            )

        print("   " + "-" * 72)
        print(
            f"   Summary: {n_curated} Literature-Backed, "
            f"{n_denovo} Data-Driven.\n"
        )

    return result


# ==========================================================
# 2) Cluster-level edge-type attention summary
# ==========================================================
def compute_cluster_edge_type_attention(
    adata,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    cluster_key="z0_cluster",
    edge_attr_names=None,
    min_cells=5,
    alpha_expr=0.7,
    alpha_pct=0.3,
    competition_mode="log",
):
    """
    Returns:
      dict: cluster -> {edge_type_name: mean cluster-contextualised score}
    """
    if edge_attr_names is None:
        edge_attr_names = EDGE_TYPE_NAMES

    result = compute_cluster_contextual_grn_scores(
        adata=adata,
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        grn_attn_weights=grn_attn_weights,
        cluster_key=cluster_key,
        min_cells=min_cells,
        alpha_expr=alpha_expr,
        alpha_pct=alpha_pct,
        competition_mode=competition_mode,
    )

    edge_attr = result["edge_attr"]
    clusters = result["clusters"]
    cluster_scores = result["cluster_scores"]

    out = {}

    for cluster_id in clusters:
        scores = cluster_scores[cluster_id]
        type_means = {}

        for i, name in enumerate(edge_attr_names):
            mask = edge_attr[:, i] == 1
            if mask.sum() > 0:
                type_means[name] = float(scores[mask].mean())
            else:
                type_means[name] = 0.0

        out[str(cluster_id)] = type_means

    return out


# ==========================================================
# 3) Global heatmap across clusters
# ==========================================================
def plot_global_grn_heatmap(
    adata,
    trainer,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    edge_attr_names=None,
    cluster_key="z0_cluster",
    top_k_edges=20,
    min_cells=5,
    alpha_expr=0.7,
    alpha_pct=0.3,
    competition_mode="log",
    save_path="GRN_Global_Heatmap.png",
):
    """
    One unified heatmap:
      Rows = clusters
      Columns = top global cluster-contextualised edges
      Color family = dominant edge type
      Color intensity = cluster-specific score
    """
    if edge_attr_names is None:
        edge_attr_names = EDGE_TYPE_NAMES

    print("\n🔥 Creating Global Cluster-Contextualised GRN Heatmap")

    result = compute_cluster_contextual_grn_scores(
        adata=adata,
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        grn_attn_weights=grn_attn_weights,
        cluster_key=cluster_key,
        min_cells=min_cells,
        alpha_expr=alpha_expr,
        alpha_pct=alpha_pct,
        competition_mode=competition_mode,
    )

    edge_index = result["edge_index"]
    edge_attr = result["edge_attr"]
    clusters = result["clusters"]
    cluster_scores = result["cluster_scores"]

    if len(clusters) == 0:
        print("⚠ No valid clusters for heatmap.")
        return

    # Mean score across clusters
    mean_scores = np.mean(
        np.vstack([cluster_scores[c] for c in clusters]),
        axis=0
    )

    top_idx = np.argsort(mean_scores)[-top_k_edges:][::-1]

    heatmap_matrix = np.vstack([cluster_scores[c][top_idx] for c in clusters])

    edge_labels = [
        f"{gene_names[edge_index[0, i]]}->{gene_names[edge_index[1, i]]}"
        for i in top_idx
    ]

    # RGB image
    num_clusters = len(clusters)
    num_edges = len(top_idx)

    vmin = np.min(heatmap_matrix)
    vmax = np.max(heatmap_matrix)
    if vmax == vmin:
        vmax = vmin + 1e-6

    norm = Normalize(vmin=vmin, vmax=vmax)
    rgb_image = np.zeros((num_clusters, num_edges, 3), dtype=float)

    # Correct edge-type color map order
    cmap_dict = {
        0: cm.Reds,     # Pathway
        1: cm.Blues,    # Spearman
        2: cm.Purples,  # LigRec
        3: cm.Greens,   # TF-Target
    }

    for j, edge_id in enumerate(top_idx):
        edge_type_idx = _dominant_edge_type(edge_attr[edge_id])
        if edge_type_idx is None:
            edge_type_idx = 1  # fallback to Spearman-like blue

        cmap = cmap_dict[edge_type_idx]

        for i in range(num_clusters):
            score = heatmap_matrix[i, j]
            rgb_image[i, j, :] = cmap(norm(score))[:3]

    plt.figure(figsize=(max(14, 0.55 * num_edges), max(5, 0.45 * num_clusters)))
    plt.imshow(rgb_image, aspect="auto")

    plt.xticks(
        ticks=np.arange(num_edges),
        labels=edge_labels,
        rotation=90
    )
    plt.yticks(
        ticks=np.arange(num_clusters),
        labels=clusters
    )

    legend_elements = [
        Patch(facecolor=cm.Reds(0.7), label="Pathway"),
        Patch(facecolor=cm.Blues(0.7), label="Spearman"),
        Patch(facecolor=cm.Purples(0.7), label="LigRec"),
        Patch(facecolor=cm.Greens(0.7), label="TF-Target"),
    ]

    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.title(
        "Cluster-Contextualised GRN Edge Relevance\n"
        "(Color = Edge Type, Intensity = Score)"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()
    plt.close()

    print("✅ Global GRN Heatmap saved:", save_path)


# ==========================================================
# 4) Balanced grouped heatmap
# ==========================================================
def plot_grouped_balanced_grn_heatmap(
    adata,
    trainer,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    cluster_key="z0_cluster",
    top_k_per_type=3,
    min_cells=5,
    alpha_expr=0.7,
    alpha_pct=0.3,
    competition_mode="log",
    save_path="Grouped_Balanced_GRN_Heatmap.png",
):
    """
    Balanced heatmap:
      top_k_per_type edges per edge class
      columns grouped by type
      rows = clusters
    """
    print("\n🔥 Creating Grouped Balanced Cluster-Contextualised GRN Heatmap")

    result = compute_cluster_contextual_grn_scores(
        adata=adata,
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        grn_attn_weights=grn_attn_weights,
        cluster_key=cluster_key,
        min_cells=min_cells,
        alpha_expr=alpha_expr,
        alpha_pct=alpha_pct,
        competition_mode=competition_mode,
    )

    edge_index = result["edge_index"]
    edge_attr = result["edge_attr"]
    clusters = result["clusters"]
    cluster_scores = result["cluster_scores"]

    if len(clusters) == 0:
        print("⚠ No valid clusters for heatmap.")
        return

    mean_scores_global = np.mean(
        np.vstack([cluster_scores[c] for c in clusters]),
        axis=0
    )

    selected_edges = []
    selected_labels = []

    for edge_type_idx in range(4):
        mask = edge_attr[:, edge_type_idx] == 1
        type_indices = np.where(mask)[0]

        if len(type_indices) == 0:
            continue

        type_scores = mean_scores_global[type_indices]
        top_local = np.argsort(type_scores)[-top_k_per_type:][::-1]

        for j in top_local:
            edge_id = type_indices[j]
            selected_edges.append(edge_id)
            selected_labels.append(
                f"{gene_names[edge_index[0, edge_id]]}->{gene_names[edge_index[1, edge_id]]}"
            )

    if len(selected_edges) == 0:
        print("⚠ No edges selected for grouped heatmap.")
        return

    heatmap_data = np.vstack([cluster_scores[c][selected_edges] for c in clusters])

    df = pd.DataFrame(
        heatmap_data,
        index=clusters,
        columns=selected_labels
    )

    fig, ax = plt.subplots(figsize=(max(14, 0.85 * len(selected_edges)), max(5, 0.45 * len(clusters))))

    sns.heatmap(
        df,
        cmap="magma",
        linewidths=0.5,
        cbar_kws={"label": "Cluster-Contextualised Edge Score"},
        ax=ax
    )

    # separators between groups
    group_size = top_k_per_type
    for i in range(1, 4):
        x = i * group_size
        if x < df.shape[1]:
            ax.axvline(x, color="white", linewidth=4)

    plt.subplots_adjust(top=0.82)

    # group titles
    for i, name in enumerate(EDGE_TYPE_NAMES):
        center = i * group_size + group_size / 2
        if center <= df.shape[1]:
            ax.text(
                center,
                1.05,
                name,
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                transform=ax.get_xaxis_transform()
            )

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(
        f"Balanced Cluster-Contextualised GRN Heatmap\n(Top {top_k_per_type} per Edge Type)",
        pad=40
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()
    plt.close()

    print("✅ Grouped Balanced GRN Heatmap saved:", save_path)