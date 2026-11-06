#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp
import matplotlib.patches as mpatches
EDGE_TYPE_COLORS = {
    "pathway":   "#B3AED6",   # light lilac
    "spearman":  "#3B3370",   # deep indigo
    "ligrec":    "#E0A21B",   # gold accent (pops)
    "tf_target": "#7B73AE",   # mid purple
    "unknown":   "#D3D0E2",   # pale lilac-gray
}
# =========================================================
# Helpers
# =========================================================
def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _validate_inputs(gene_graphs, gene_names, grn_attn_weights):
    if gene_graphs is None or "global" not in gene_graphs:
        raise ValueError("gene_graphs must contain gene_graphs['global'].")

    g = gene_graphs["global"]

    if not hasattr(g, "edge_index"):
        raise ValueError("gene_graphs['global'] has no edge_index.")

    if not hasattr(g, "edge_attr"):
        raise ValueError("gene_graphs['global'] has no edge_attr.")

    edge_index = _to_numpy(g.edge_index)
    edge_attr = _to_numpy(g.edge_attr)
    attn = _to_numpy(grn_attn_weights)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")

    E = edge_index.shape[1]

    if attn.ndim != 1:
        attn = attn.reshape(-1)

    if len(attn) != E:
        raise ValueError(
            f"Attention length mismatch: got {len(attn)} attention values, but edge_index has {E} edges."
        )

    if edge_attr.shape[0] != E:
        raise ValueError(
            f"edge_attr row mismatch: edge_attr has {edge_attr.shape[0]} rows, edge_index has {E} edges."
        )

    if len(gene_names) != g.num_nodes:
        raise ValueError(
            f"gene_names length mismatch: len(gene_names)={len(gene_names)}, num_nodes={g.num_nodes}"
        )

    return edge_index, edge_attr, attn


def _resolve_edge_types(edge_attr_row, edge_attr_names):
    idx = np.where(edge_attr_row > 0)[0].tolist()
    if len(idx) == 0:
        return "unknown"
    return "|".join([edge_attr_names[i] for i in idx])


def _reverse_only_mask(gene_graphs, n_edges):
    """Boolean mask of edges that exist only as reverse copies of curated
    interactions (added for message passing). Built by build_fixed.py.

    Returns an all-False mask, with a warning, for graphs built without it.
    """
    g = gene_graphs["global"]
    mask = getattr(g, "reverse_only", None)
    if mask is None:
        print("  [warn] graph has no reverse_only mask; reverse copies of "
              "curated edges will be reported. Rebuild the GRN with "
              "build_fixed.py.")
        return np.zeros(n_edges, dtype=bool)
    mask = _to_numpy(mask).astype(bool).reshape(-1)
    if len(mask) != n_edges:
        raise ValueError(f"reverse_only has {len(mask)} entries, graph has "
                         f"{n_edges} edges")
    return mask


# =========================================================
# 1) Edge-level report
# =========================================================
def build_global_edge_report(
    gene_graphs,
    gene_names,
    grn_attn_weights,
    edge_attr_names=None,
    sort_desc=True
):
    """
    Build a dataframe with one row per edge.

    Columns:
      - edge_id
      - source_idx, target_idx
      - source_gene, target_gene
      - attention
      - edge_types
      - curated
      - pathway, spearman, ligrec, tf_target (if present)

    Notes:
      - With your current graph builder, direction may be partially lost because edges are sorted.
      - So source/target should be interpreted cautiously unless you preserve direction in build_hetero_grn.
    """
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    edge_index, edge_attr, attn = _validate_inputs(gene_graphs, gene_names, grn_attn_weights)

    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.shape[1]

    rows = []
    for e in range(E):
        attr_row = edge_attr[e]
        edge_types = _resolve_edge_types(attr_row, edge_attr_names)

        row = {
            "edge_id": e,
            "source_idx": int(src[e]),
            "target_idx": int(dst[e]),
            "source_gene": gene_names[int(src[e])],
            "target_gene": gene_names[int(dst[e])],
            "attention": float(attn[e]),
            "edge_types": edge_types,
            "curated": bool(attr_row[0] > 0 or attr_row[2] > 0 or attr_row[3] > 0),
        }

        # Add one column per edge type if available
        for i, name in enumerate(edge_attr_names):
            if i < len(attr_row):
                row[name] = int(attr_row[i] > 0)
            else:
                row[name] = 0

        rows.append(row)

    df = pd.DataFrame(rows)
    df["reverse_only"] = _reverse_only_mask(gene_graphs, E)[df["edge_id"].values]
    df = df.sort_values("attention", ascending=not sort_desc).reset_index(drop=True)
    df["evidence_tier"] = df.apply(classify_evidence_tier, axis=1)
    # In-degree is taken over ALL incoming edges, reverse copies included,
    # because they all took part in the attention softmax for that target.
    df = add_edge_competition_metrics(df)
    return df


# =========================================================
# 2) Node-level report
# =========================================================
def build_global_node_report(
    gene_graphs,
    gene_names,
    grn_attn_weights,
    edge_attr_names=None
):
    """
    Build a dataframe with one row per gene.

    Metrics:
      - raw_in_degree
      - raw_out_degree
      - raw_total_degree
      - weighted_in_strength
      - weighted_out_strength
      - weighted_total_strength
      - mean_in_attention
      - mean_out_attention
      - mean_total_attention
      - dominant_edge_type_by_attention

    Important:
      Because your build_hetero_grn currently sorts edges, in/out interpretation is limited.
      The safest metric biologically is weighted_total_strength.
    """
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    edge_index, edge_attr, attn = _validate_inputs(gene_graphs, gene_names, grn_attn_weights)

    num_nodes = len(gene_names)
    src = edge_index[0]
    dst = edge_index[1]

    raw_in = np.zeros(num_nodes, dtype=np.int64)
    raw_out = np.zeros(num_nodes, dtype=np.int64)

    w_in = np.zeros(num_nodes, dtype=np.float64)
    w_out = np.zeros(num_nodes, dtype=np.float64)

    # attention mass by edge type per node
    type_mass = np.zeros((num_nodes, len(edge_attr_names)), dtype=np.float64)

    rev = _reverse_only_mask(gene_graphs, edge_index.shape[1])

    for e in range(edge_index.shape[1]):
        if rev[e]:
            continue
        u = int(src[e])
        v = int(dst[e])
        a = float(attn[e])

        raw_out[u] += 1
        raw_in[v] += 1

        w_out[u] += a
        w_in[v] += a

        # allocate edge attention to all active edge-type tags on this edge
        active_types = np.where(edge_attr[e] > 0)[0]
        if len(active_types) > 0:
            share = a / len(active_types)
            for t in active_types:
                if t < len(edge_attr_names):
                    type_mass[u, t] += share
                    type_mass[v, t] += share

    raw_total = raw_in + raw_out
    w_total = w_in + w_out

    mean_in = np.divide(w_in, raw_in, out=np.zeros_like(w_in), where=raw_in > 0)
    mean_out = np.divide(w_out, raw_out, out=np.zeros_like(w_out), where=raw_out > 0)
    mean_total = np.divide(w_total, raw_total, out=np.zeros_like(w_total), where=raw_total > 0)

    dominant_type_idx = np.argmax(type_mass, axis=1)
    dominant_type = [
        edge_attr_names[i] if w_total[g] > 0 else "none"
        for g, i in enumerate(dominant_type_idx)
    ]

    rows = []
    for g in range(num_nodes):
        row = {
            "gene_idx": g,
            "gene": gene_names[g],
            "raw_in_degree": int(raw_in[g]),
            "raw_out_degree": int(raw_out[g]),
            "raw_total_degree": int(raw_total[g]),
            "weighted_in_strength": float(w_in[g]),
            "weighted_out_strength": float(w_out[g]),
            "weighted_total_strength": float(w_total[g]),
            "mean_in_attention": float(mean_in[g]),
            "mean_out_attention": float(mean_out[g]),
            "mean_total_attention": float(mean_total[g]),
            "dominant_edge_type_by_attention": dominant_type[g],
        }

        for i, name in enumerate(edge_attr_names):
            row[f"{name}_attention_mass"] = float(type_mass[g, i])

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("weighted_total_strength", ascending=False).reset_index(drop=True)
    return df


# =========================================================
# 3) Edge-type summary
# =========================================================
def summarize_edge_type_attention(
    gene_graphs,
    grn_attn_weights,
    edge_attr_names=None
):
    """
    Summarize how much total attention is assigned to each edge type.

    Returns dataframe with:
      - edge_type
      - attention_mass
      - attention_fraction
      - edge_count
      - mean_attention_per_edge
    """
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    g = gene_graphs["global"]
    edge_attr = _to_numpy(g.edge_attr)
    attn = _to_numpy(grn_attn_weights).reshape(-1)

    E = edge_attr.shape[0]
    if len(attn) != E:
        raise ValueError("Attention length and edge_attr rows do not match.")

    total_mass = 0.0
    mass = np.zeros(len(edge_attr_names), dtype=np.float64)
    counts = np.zeros(len(edge_attr_names), dtype=np.int64)

    rev = _reverse_only_mask(gene_graphs, E)

    for e in range(E):
        if rev[e]:
            continue
        active_types = np.where(edge_attr[e] > 0)[0]
        if len(active_types) == 0:
            continue

        a = float(attn[e])
        total_mass += a

        # split attention equally across active tags
        share = a / len(active_types)
        for t in active_types:
            if t < len(edge_attr_names):
                mass[t] += share
                counts[t] += 1

    frac = np.divide(mass, total_mass, out=np.zeros_like(mass), where=total_mass > 0)
    mean_per_edge = np.divide(mass, counts, out=np.zeros_like(mass), where=counts > 0)

    df = pd.DataFrame({
        "edge_type": edge_attr_names,
        "attention_mass": mass,
        "attention_fraction": frac,
        "edge_count": counts,
        "mean_attention_per_edge": mean_per_edge,
    })

    df = df.sort_values("attention_mass", ascending=False).reset_index(drop=True)
    return df


# =========================================================
# 4) Print report
# =========================================================
def print_global_grn_report(
    edge_df,
    node_df,
    edge_type_df,
    top_k_edges=15,
    top_k_genes=15
):
    print("\n" + "=" * 70)
    print("GLOBAL PRE-CLUSTERING GRN REPORT")
    print("=" * 70)

    print("\n[1] Top edges by attention")
    cols_edge = ["source_gene", "target_gene", "attention", "edge_types", "curated"]
    print(edge_df[cols_edge].head(top_k_edges).to_string(index=False))

    print("\n[2] Top genes by weighted total strength")
    cols_gene = [
        "gene",
        "weighted_total_strength",
        "weighted_in_strength",
        "weighted_out_strength",
        "raw_total_degree",
        "dominant_edge_type_by_attention",
    ]
    print(node_df[cols_gene].head(top_k_genes).to_string(index=False))

    print("\n[3] Edge-type attention composition")
    print(edge_type_df.to_string(index=False))

    print("\nInterpretation note:")
    print("- Top edges = most emphasised interactions in the learned GRN layer")
    print("- Weighted total strength = safest global hub metric in your current graph")
    print("- In/Out split is less reliable biologically until you preserve edge direction")


# =========================================================
# 5) CSV export
# =========================================================
def save_global_grn_report(
    out_dir,
    edge_df,
    node_df,
    edge_type_df,
    prefix="global_grn"
):
    os.makedirs(out_dir, exist_ok=True)

    edge_path = os.path.join(out_dir, f"{prefix}_edges.csv")
    node_path = os.path.join(out_dir, f"{prefix}_nodes.csv")
    type_path = os.path.join(out_dir, f"{prefix}_edge_types.csv")

    edge_df.to_csv(edge_path, index=False)
    node_df.to_csv(node_path, index=False)
    edge_type_df.to_csv(type_path, index=False)

    print(f"\nSaved:")
    print(f"  {edge_path}")
    print(f"  {node_path}")
    print(f"  {type_path}")


# =========================================================
# 6) Plots
# =========================================================
def plot_top_edges_bar(edge_df, top_k=12, figsize=(5, 6), save_path=None):
    top = edge_df.head(top_k).copy()
    labels = [f"{s}→{t}" for s, t in zip(top["source_gene"], top["target_gene"])]

    # color by dominant edge type (uses edge_types column already in edge_df)
    def first_type(et):
        return et.split("|")[0] if isinstance(et, str) and et else "unknown"
    colors = [EDGE_TYPE_COLORS.get(first_type(et), "#999999")
              for et in top["edge_types"]]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(top))
    ax.barh(y, top["attention"].values, color=colors, alpha=0.9, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Global GRN attention", fontsize=9)
    ax.set_title(f"Top {top_k} GRN-weighted edges (whole tissue)",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    types_present = sorted({first_type(et) for et in top["edge_types"]})
    patches = [mpatches.Patch(color=EDGE_TYPE_COLORS.get(t, "#999999"), label=t)
               for t in types_present]
    ax.legend(handles=patches, fontsize=7, loc="lower left",bbox_to_anchor=(1.02, 0), title="Edge type")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_top_genes_bar(node_df, top_k=20, score_col="weighted_total_strength", figsize=(12, 6),save_path=None):
    top = node_df.head(top_k).copy()

    plt.figure(figsize=figsize)
    plt.bar(range(len(top)), top[score_col].values)
    plt.xticks(range(len(top)), top["gene"].values, rotation=75, ha="right")
    plt.ylabel(score_col)
    plt.title(f"Top {top_k} genes by {score_col}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_edge_type_attention(edge_type_df, figsize=(7, 5),save_path=None):
    plt.figure(figsize=figsize)
    plt.bar(edge_type_df["edge_type"], edge_type_df["attention_fraction"])
    plt.ylabel("Fraction of total attention")
    plt.title("Attention composition by GRN edge type")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_top_edges_bubble(
    edge_df,
    top_n=50,
    figsize=(14, 10),
    cmap="viridis",
    save_path=None
):
    df = edge_df.copy()

    # keep top highest edges directly
    df = df.sort_values("attention", ascending=False).head(top_n).copy()

    # preserve order from ranking
    source_order = list(dict.fromkeys(df["source_gene"]))
    target_order = list(dict.fromkeys(df["target_gene"]))

    source_map = {g: i for i, g in enumerate(source_order)}
    target_map = {g: i for i, g in enumerate(target_order)}

    df["y"] = df["source_gene"].map(source_map)
    df["x"] = df["target_gene"].map(target_map)

    vals = df["attention"].values.astype(float)
    if vals.max() == vals.min():
        sizes = np.full(len(vals), 100.0)
    else:
        sizes = 40 + 260 * (vals - vals.min()) / (vals.max() - vals.min())

    plt.figure(figsize=figsize)
    sc = plt.scatter(
        df["x"],
        df["y"],
        s=sizes,
        c=df["attention"],
        cmap=cmap,
        alpha=0.9
    )

    plt.colorbar(sc, label="Attention")
    plt.xticks(range(len(target_order)), target_order, rotation=90, fontsize=11)
    plt.yticks(range(len(source_order)), source_order, fontsize=11)
    plt.xlabel("Target gene", fontsize=13)
    plt.ylabel("Source gene", fontsize=13)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()

# def plot_source_target_bubble(
#     edge_df,
#     figsize=(16, 12),
#     agg="max",
#     sort_by="sum",
#     top_n_sources=None,
#     top_n_targets=None,
#     min_attention=None,
#     max_labels=60,
#     cmap="viridis",
#     save_path=None
# ):
   
#     df = edge_df.copy()

#     if min_attention is not None:
#         df = df[df["attention"] >= min_attention].copy()

#     if df.empty:
#         raise ValueError("No edges left after filtering. Try lowering min_attention.")

#     mat_df = df.pivot_table(
#         index="source_gene",
#         columns="target_gene",
#         values="attention",
#         aggfunc=agg,
#         fill_value=0
#     )

#     if top_n_sources is not None:
#         keep_rows = mat_df.sum(axis=1).sort_values(ascending=False).head(top_n_sources).index
#         mat_df = mat_df.loc[keep_rows]

#     if top_n_targets is not None:
#         keep_cols = mat_df.sum(axis=0).sort_values(ascending=False).head(top_n_targets).index
#         mat_df = mat_df.loc[:, keep_cols]

#     if sort_by == "sum":
#         row_order = mat_df.sum(axis=1).sort_values(ascending=False).index
#         col_order = mat_df.sum(axis=0).sort_values(ascending=False).index
#         mat_df = mat_df.loc[row_order, col_order]
#     elif sort_by == "alphabetical":
#         mat_df = mat_df.sort_index(axis=0).sort_index(axis=1)

#     plot_df = (
#         mat_df.stack()
#         .reset_index()
#         .rename(columns={"level_0": "source_gene", "level_1": "target_gene", 0: "attention"})
#     )
#     plot_df = plot_df[plot_df["attention"] > 0].copy()

#     if plot_df.empty:
#         raise ValueError("No nonzero interactions available to plot.")

#     row_names = list(mat_df.index)
#     col_names = list(mat_df.columns)

#     row_map = {g: i for i, g in enumerate(row_names)}
#     col_map = {g: i for i, g in enumerate(col_names)}

#     plot_df["y"] = plot_df["source_gene"].map(row_map)
#     plot_df["x"] = plot_df["target_gene"].map(col_map)

#     vals = plot_df["attention"].values.astype(float)
#     if vals.max() == vals.min():
#         sizes = np.full(len(vals), 80.0)
#     else:
#         sizes = 20 + 280 * (vals - vals.min()) / (vals.max() - vals.min())

#     plt.figure(figsize=figsize)
#     sc = plt.scatter(
#         plot_df["x"],
#         plot_df["y"],
#         s=sizes,
#         c=plot_df["attention"],
#         cmap=cmap,
#         alpha=0.85
#     )
#     plt.colorbar(sc, label="Attention")

#     n_rows = len(row_names)
#     n_cols = len(col_names)

#     if n_cols <= max_labels:
#         xticks = np.arange(n_cols)
#     else:
#         xticks = np.linspace(0, n_cols - 1, max_labels, dtype=int)

#     if n_rows <= max_labels:
#         yticks = np.arange(n_rows)
#     else:
#         yticks = np.linspace(0, n_rows - 1, max_labels, dtype=int)

#     plt.xticks(xticks, [col_names[i] for i in xticks], rotation=90,fontsize=12)
#     plt.yticks(yticks, [row_names[i] for i in yticks],fontsize=12)

#     plt.xlabel("Target gene",fontsize=14)
#     plt.ylabel("Source gene",fontsize=14)
#     plt.title("")
#     plt.gca().invert_yaxis()
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=600, bbox_inches="tight")
#     plt.show()

def add_edge_competition_metrics(edge_df):
    df = edge_df.copy()

    # target in-degree from edge table
    target_in_deg = df.groupby("target_idx")["edge_id"].count().to_dict()
    df["target_in_degree"] = df["target_idx"].map(target_in_deg).astype(int)

    # source out-degree too (optional)
    source_out_deg = df.groupby("source_idx")["edge_id"].count().to_dict()
    df["source_out_degree"] = df["source_idx"].map(source_out_deg).astype(int)

    df["is_competitive"] = df["target_in_degree"] >= 2

    # competition-aware score
    df["edge_salience"] = df["attention"] * np.log2(1.0 + df["target_in_degree"])

    return df
def classify_evidence_tier(row):
    curated = bool(row["pathway"] or row["ligrec"] or row["tf_target"])
    spearman = bool(row["spearman"])

    if curated and spearman:
        return "prior+data"
    elif curated and not spearman:
        return "prior_only"
    elif (not curated) and spearman:
        return "data_only"
    else:
        return "unknown"
    
# =========================================================
# 8) Niche × edge matrix — save full matrix to CSV
# =========================================================
def save_niche_grn_matrix(
    adata,
    gene_graphs,
    gene_names,
    grn_attn_weights,
    cluster_key="annotation",
    csv_path="niche_grn_full_matrix.csv",
):
    """
    Compute the full niche × edge expression-weighted attention matrix
    (ALL edges × ALL niches, no top-k filtering) and save to CSV.

    Score for edge (u->v) in niche c:
        score = attention(u,v) * mean_expr_c(u) * mean_expr_c(v)

    Run once while gene_graphs / attention / adata are in memory.
    Afterwards, replot from the CSV with plot_niche_grn_from_csv().
    """
    g = gene_graphs["global"]
    ei = _to_numpy(g.edge_index)
    attn = _to_numpy(grn_attn_weights).reshape(-1)

    var_to_col = {gn: i for i, gn in enumerate(adata.var_names)}
    clusters = adata.obs[cluster_key].astype(str).values
    niches = sorted(set(clusters))

    # per-niche mean expression
    mean_expr = {}
    for c in niches:
        sub = adata.X[clusters == c]
        mean_expr[c] = (np.asarray(sub.mean(0)).ravel()
                        if sp.issparse(sub) else np.asarray(sub).mean(0).ravel())

    rev = _reverse_only_mask(gene_graphs, ei.shape[1])
    keep = np.where(~rev)[0]

    pairs = [f"{gene_names[ei[0, e]]}→{gene_names[ei[1, e]]}" for e in keep]
    mat = pd.DataFrame(index=pairs, columns=niches, dtype=float)

    for c in niches:
        expr = mean_expr[c]
        col = []
        for e in keep:
            sc = var_to_col.get(gene_names[ei[0, e]])
            dc = var_to_col.get(gene_names[ei[1, e]])
            if sc is None or dc is None:
                col.append(0.0)
            else:
                col.append(float(attn[e]) * float(expr[sc]) * float(expr[dc]))
        mat[c] = col

    mat.to_csv(csv_path)
    print(f"✅ Saved full niche×edge matrix "
          f"({mat.shape[0]} edges × {mat.shape[1]} niches) → {csv_path}")
    return mat


# =========================================================
# 9) Replot niche GRN heatmap directly from saved CSV
# =========================================================
def plot_niche_grn_from_csv(
    csv_path,
    top_k_per_niche=5,          
    row_norm=True,
    cmap=None,                  
    group_by_niche=True,        
    save_path=None,
):
    """
    Load the saved niche×edge matrix and plot a high-contrast heatmap 
    consistent with a purple layout but utilizing balanced accent tones.
    """
    from matplotlib.colors import LinearSegmentedColormap
    mat = pd.read_csv(csv_path, index_col=0)

    selected = []           
    seen = set()
    for niche in mat.columns:
        top_edges = mat[niche].sort_values(ascending=False).head(top_k_per_niche).index
        for e in top_edges:
            if e not in seen:
                selected.append(e)
                seen.add(e)

    mat = mat.loc[selected]   

    vals = mat.values.astype(float)
    if row_norm:
        rmax = vals.max(1, keepdims=True); rmax[rmax == 0] = 1
        vals = vals / rmax
        clabel = "Row-normalized expression-weighted attention"
    else:
        vmax = np.percentile(vals[vals > 0], 95) if (vals > 0).any() else 1.0
        vals = np.clip(vals, 0, vmax) / vmax
        clabel = "Expression-weighted attention"

    # 🎨 --- CHOOSE YOUR ACCENT CMAP ---
    if cmap is None:
        # OPTION 1: Periwinkle-to-Champagne Gold (High pop, similar to magma but cooler)
        cmap = LinearSegmentedColormap.from_list(
            "periwinkle_gold", 
            ["#F4F3F7", "#A3B2E0", "#4A3B75", "#D4AF37"]
        )
        
        # OPTION 2: Silver-to-Copper Slate (Elegant, neutral, understated)
        # To try this one, uncomment the lines below:
        # cmap = LinearSegmentedColormap.from_list(
        #     "silver_copper", 
        #     ["#F0EFF2", "#B0AEC2", "#524F5A", "#8C5846"]
        # )

    n_rows, n_cols = mat.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.9),
                                    max(6, n_rows * 0.28)), facecolor="white")
    
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=9, color="#2B2633")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(mat.index, fontsize=8, color="#2B2633")
    
    ax.spines[['top', 'right', 'left', 'bottom']].set_color('#CBC6D6')
    ax.tick_params(colors='#4A4452')

    cb = plt.colorbar(im, ax=ax, shrink=0.6, label=clabel)
    cb.ax.yaxis.label.set_fontsize(10)
    cb.ax.tick_params(labelsize=9)

    ax.set_xlabel("Cell type / niche", fontsize=10, fontweight="bold", color="#2B2633")
    ax.set_ylabel("Gene interaction (top edges per cell type)", fontsize=10, fontweight="bold", color="#2B2633")
    ax.set_title(f"Per-cell-type GRN activity (top {top_k_per_niche} edges per cell type)",
                 fontsize=18, pad=12, fontweight="bold", color="#1A1521")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.show()
    return mat
# =========================================================
# 7) Master function
# =========================================================
def run_global_grn_report(
    gene_graphs,
    gene_names,
    grn_attn_weights,
    edge_attr_names=None,
    out_dir=None,
    prefix="global_grn",
    top_k_edges=15,
    top_k_genes=15,
    make_plots=True,
    adata=None,                       # ← ADDED
    cluster_key="annotation",         # ← ADDED
):
    """
    Full pipeline:
      1) build edge report
      2) build node report
      3) summarize edge types
      4) print summary
      5) optional CSV save (incl. niche×edge matrix if adata given)
      6) optional plots
    Returns:
      edge_df, node_df, edge_type_df
    """
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    edge_df = build_global_edge_report(
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        grn_attn_weights=grn_attn_weights,
        edge_attr_names=edge_attr_names,
    )

    edge_df_full = edge_df.copy()

    n_rev = int(edge_df["reverse_only"].sum())
    if n_rev:
        print(f"  Excluding {n_rev} reverse-only copies from the report")
    edge_df = edge_df[~edge_df["reverse_only"]].reset_index(drop=True)

    edge_df_nontrivial = edge_df[edge_df["is_competitive"]].copy()
    if len(edge_df_nontrivial) == 0:
        edge_df_ranked = edge_df.copy()
    else:
        edge_df_ranked = edge_df_nontrivial.sort_values(
            ["edge_salience", "attention"],
            ascending=False
        ).reset_index(drop=True)

    node_df = build_global_node_report(
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        grn_attn_weights=grn_attn_weights,
        edge_attr_names=edge_attr_names,
    )

    edge_type_df = summarize_edge_type_attention(
        gene_graphs=gene_graphs,
        grn_attn_weights=grn_attn_weights,
        edge_attr_names=edge_attr_names,
    )

    print_global_grn_report(
        edge_df=edge_df_ranked,
        node_df=node_df,
        edge_type_df=edge_type_df,
        top_k_edges=top_k_edges,
        top_k_genes=top_k_genes,
    )

    if out_dir is not None:
        save_global_grn_report(
            out_dir=out_dir,
            edge_df=edge_df,
            node_df=node_df,
            edge_type_df=edge_type_df,
            prefix=prefix,
        )
        edge_df_ranked.to_csv(
            os.path.join(out_dir, f"{prefix}_edges_ranked_competitive.csv"),
            index=False
        )

        # ← ADDED: save full niche×edge matrix (needs adata)
        if adata is not None:
            save_niche_grn_matrix(
                adata=adata,
                gene_graphs=gene_graphs,
                gene_names=gene_names,
                grn_attn_weights=grn_attn_weights,
                cluster_key=cluster_key,
                csv_path=os.path.join(out_dir, f"{prefix}_niche_edge_matrix.csv"),
            )

    if make_plots:
        plot_dir = out_dir if out_dir is not None else "."
        plot_top_edges_bar(edge_df_ranked, top_k=min(top_k_edges, len(edge_df_ranked)),
                           save_path=os.path.join(plot_dir, f"{prefix}_top_edges_bar.png"))
        plot_top_genes_bar(node_df, top_k=min(top_k_genes, len(node_df)),
                           score_col="weighted_total_strength",
                           save_path=os.path.join(plot_dir, f"{prefix}_top_genes_bar.png"))
        plot_edge_type_attention(edge_type_df,
                                 save_path=os.path.join(plot_dir, f"{prefix}_edge_type_attention.png"))
        plot_top_edges_bubble(
            edge_df=edge_df_ranked,
            figsize=(18, 14),
            top_n=50,
            cmap="turbo",
            save_path=os.path.join(out_dir, "Global_buble.png")
        )

        # ← ADDED: plot the clean niche heatmap from the matrix just saved
        if adata is not None and out_dir is not None:
            plot_niche_grn_from_csv(
                csv_path=os.path.join(out_dir, f"{prefix}_niche_edge_matrix.csv"),
                top_k_per_niche=5,
                save_path=os.path.join(out_dir, f"{prefix}_niche_grn_heatmap_clean.png"),
            )

    return edge_df, node_df, edge_type_df