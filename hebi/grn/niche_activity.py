#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Niche-level GRN activity.

An edge's learned attention is global: it says which regulator wins the
competition for a target gene across the whole section. To localise that
to a group of cells, the attention is weighted by how strongly the two
genes are expressed in that group:

    activity(e, c) = a_e * mean_expr(source, c) * mean_expr(target, c)

which is the niche-level GRN activity in the Methods. An edge is active
in a group when the interaction is favoured by the attention *and* both
genes are expressed there.
"""

import os

import numpy as np
import pandas as pd


def niche_edge_activity(adata, gene_graph, gene_names, attention,
                        cluster_key, min_cells: int = 5):
    """Score every edge in every group of cells.

    Returns a DataFrame indexed by edge label, one column per group.
    """
    import torch

    data = gene_graph["global"]
    edge_index = data.edge_index.detach().cpu().numpy()
    attn = (attention.detach().cpu().numpy() if torch.is_tensor(attention)
            else np.asarray(attention)).reshape(-1)

    if edge_index.shape[1] != len(attn):
        raise ValueError(
            f"{edge_index.shape[1]} edges but {len(attn)} attention values — "
            "the graph and the attention come from different runs."
        )

    gene_names = list(gene_names)
    n_genes = len(gene_names)

    # The AnnData may carry more genes than the graph was built on.
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if X.shape[1] < n_genes:
        raise ValueError(
            f"graph has {n_genes} genes but the AnnData has {X.shape[1]}"
        )
    X = X[:, :n_genes]

    groups = adata.obs[cluster_key].astype(str)
    labels = sorted(groups.unique())
    # Drop reverse copies of curated edges (added only for message passing).
    rev = getattr(data, "reverse_only", None)
    if rev is None:
        print("  [warn] graph has no reverse_only mask; reverse copies will be "
              "scored. Rebuild the GRN with build_fixed.py.")
        keep = np.ones(edge_index.shape[1], dtype=bool)
    else:
        keep = ~rev.detach().cpu().numpy().astype(bool).reshape(-1)
    
    src, dst = edge_index[0][keep], edge_index[1][keep]
    attn = attn[keep]
    edge_labels = [f"{gene_names[s]} \u2192 {gene_names[d]}"
                   for s, d in zip(src, dst)]

    columns = {}
    for label in labels:
        mask = (groups == label).to_numpy()
        if mask.sum() < min_cells:
            continue
        mean_expr = X[mask].mean(axis=0)
        columns[label] = attn * mean_expr[src] * mean_expr[dst]

    return pd.DataFrame(columns, index=edge_labels)


def drop_reverse_copies(activity, keep="stronger"):
    """Collapse reciprocal edges to one direction.

    The gene graph is built with reverse copies of curated edges so that
    information can flow both ways during message passing. Both copies are
    scored, so an interaction appears twice: GAD1 -> GABBR1 alongside
    GABBR1 -> GAD1. Only one is biologically meaningful — a receptor does
    not signal back to the enzyme that made its ligand.

    Which direction is canonical is not recoverable from the activity
    matrix alone, so the stronger one is kept. That is a display choice,
    and figures using it should say so.

    Returns the matrix with one row per undirected gene pair.
    """
    totals = activity.sum(axis=1)
    keep_rows, seen = [], {}

    for edge in activity.index:
        parts = [g.strip() for g in str(edge).split("\u2192")]
        if len(parts) != 2:
            keep_rows.append(edge)
            continue
        pair = frozenset(parts)
        if pair not in seen:
            seen[pair] = edge
        else:
            other = seen[pair]
            if totals[edge] > totals[other]:
                seen[pair] = edge

    ordered = [e for e in activity.index if e in set(seen.values())]
    n_dropped = len(activity.index) - len(ordered)
    if n_dropped:
        print(f"  dropped {n_dropped} reverse copies "
              f"({len(ordered)} interactions remain)")
    return activity.loc[ordered]


def top_edges_per_group(activity, top_k: int = 5):
    """The strongest edges in each group, as an ordered unique list.

    Taking the union rather than a fixed set lets a group contribute its
    own top edges even when they are weak elsewhere, which is what makes
    the heatmap show group-specific programmes rather than one global
    ranking repeated.
    """
    seen, ordered = set(), []
    for group in activity.columns:
        for edge in activity[group].sort_values(ascending=False).head(top_k).index:
            if edge not in seen:
                seen.add(edge)
                ordered.append(edge)
    return ordered


def plot_top_edges_heatmap(activity, out_path, top_k: int = 5,
                           title="", formats=("png", "svg"), dpi: int = 200,
                           drop_reverse: bool = False):
    """Heatmap of the top edges per group.

    Values are scaled per row, so the colour shows where an interaction is
    most active rather than how large its raw score is. Without that, a
    few high-expression edges dominate every row and the group structure
    disappears.

    drop_reverse collapses reciprocal edges to a single direction, so the
    same interaction is not shown twice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if drop_reverse:
        activity = drop_reverse_copies(activity)

    edges = top_edges_per_group(activity, top_k=top_k)
    if not edges:
        print("  no edges to plot")
        return None

    matrix = activity.loc[edges]

    row_max = matrix.max(axis=1).replace(0, np.nan)
    scaled = matrix.div(row_max, axis=0).fillna(0.0)

    height = max(4.0, 0.26 * len(edges))
    width = max(6.0, 0.55 * matrix.shape[1] + 3.5)

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(scaled.values, aspect="auto", cmap="magma",
                   vmin=0, vmax=1)

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(len(edges)))
    ax.set_yticklabels(edges, fontsize=8)
    ax.set_title(title or f"Top {top_k} regulatory interactions per group",
                 fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("activity, scaled per interaction", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    written = []
    for fmt in formats:
        path = f"{out_path}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    print(f"  {len(edges)} interactions across {matrix.shape[1]} groups "
          f"-> {out_path}")
    return written
