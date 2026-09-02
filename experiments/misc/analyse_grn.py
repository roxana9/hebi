#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:08:12 2026

@author: tower4
"""
import os
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from hebi.training.finetune.finetuner import HebiCellTypeClassifier
from hebi.training.finetune.loader import (
    get_slide_list,
    load_supervised_slide_dict,
    build_local_sampler,
    iterate_supervised_batches,
)

from hebi.grn.build import build_hetero_grn
from hebi.model.grn_layer import GRNRefinementLayer
from hebi.data.vocabulary import harmonise_adata
import re

def default_housekeeping_set():
    """
    Practical housekeeping / ubiquitous list.
    Add/remove genes based on what dominates your panels.
    """
    return {
        # ubiquitous / long non-coding
        "MALAT1", "NEAT1", "XIST",
        # classic housekeeping
        "ACTB", "GAPDH", "EEF1A1", "B2M", "PPIA",
        # stress / heat-shock (often global)
        "HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1",
        # common immune-like ubiquitous in panels
        "HLA-A", "HLA-B", "HLA-C",
    }

def is_housekeeping_gene(g, hk_set=None, hk_regex=True):
    """
    Returns True if gene should be treated as housekeeping/ubiquitous.
    hk_regex catches RPL/RPS/MT-* and HLA-* patterns.
    """
    g = str(g).upper()
    if hk_set is None:
        hk_set = default_housekeeping_set()

    if g in hk_set:
        return True

    if not hk_regex:
        return False

    # ribosomal, mitochondrial, HLA patterns
    if re.match(r"^RPL\d+", g): return True
    if re.match(r"^RPS\d+", g): return True
    if re.match(r"^MT-", g):    return True
    if re.match(r"^HLA-", g):   return True

    return False


def filter_edges_by_housekeeping(unique_edges, hk_set=None, hk_regex=True):
    """
    unique_edges is your list[dict] (from build_unique_top_edges_across_niches_directed).
    Removes any edge where src or dst is housekeeping.
    """
    out = []
    for d in unique_edges:
        if is_housekeeping_gene(d["src_gene"], hk_set, hk_regex): 
            continue
        if is_housekeeping_gene(d["dst_gene"], hk_set, hk_regex): 
            continue
        out.append(d)
    return out

# =========================================================
# 1) Extract GRN attention from ONE batch of the test slide
# =========================================================

def compute_edge_salience(gene_graphs, att):
    """
    att: [E] numpy
    returns: salience [E] numpy
    """
    edge_index = gene_graphs["global"].edge_index.detach().cpu().numpy()  # [2, E]
    dst = edge_index[1]  # target indices

    # target_in_degree[v] = number of edges ending at v
    target_in_degree = np.bincount(dst, minlength=gene_graphs["global"].num_nodes)

    salience = att * np.log2(1.0 + target_in_degree[dst])
    return salience

def edge_score_from_mean_expr(gene_graphs, att, mean_expr, already_log=True):
    """
    mean_expr:
      - shape [G]   -> returns score [E]
      - shape [C,G] -> returns score [C,E]

    If your slide_test["X"] is already log1p-normalised (your .pt writer does this),
    set already_log=True and DO NOT log again.
    """
    edge_index = gene_graphs["global"].edge_index.detach().cpu().numpy()
    src, dst = edge_index[0], edge_index[1]
    att = np.asarray(att).reshape(-1)

    indeg = np.bincount(dst, minlength=gene_graphs["global"].num_nodes)
    salience = att * np.log2(1.0 + indeg[dst])  # [E]

    mean_expr = np.asarray(mean_expr)

    if mean_expr.ndim == 1:
        expr_src = mean_expr[src]
        expr_dst = mean_expr[dst]
        if not already_log:
            expr_src = np.log1p(expr_src)
            expr_dst = np.log1p(expr_dst)
        return salience * expr_src * expr_dst  # [E]

    elif mean_expr.ndim == 2:
        expr_src = mean_expr[:, src]  # [C,E]
        expr_dst = mean_expr[:, dst]  # [C,E]
        if not already_log:
            expr_src = np.log1p(expr_src)
            expr_dst = np.log1p(expr_dst)
        return salience[None, :] * expr_src * expr_dst  # [C,E]

    else:
        raise ValueError("mean_expr must be shape [G] or [C,G]")
def get_global_grn_attention_from_one_batch(model, slide, device, target_key, batch_size=64):
    model.eval()
    sampler = build_local_sampler(slide, batch_size=batch_size, num_workers=0, shuffle=False)

    for (
        x_subgraph_raw,
        edge_index_subgraph,
        edge_attr_b,
        node2comm_batch,
        super_sp_global,
        gene_ids,
        y_seed,
        bs,
    ) in iterate_supervised_batches(slide, sampler, device=device, target_key=target_key):

        with torch.no_grad():
            logits, att, gene_emb = model(
                x_subgraph_raw=x_subgraph_raw,
                edge_index_subgraph=edge_index_subgraph,
                edge_attr_b=edge_attr_b,
                node2comm_batch=node2comm_batch,
                super_sp_global=super_sp_global,
                device=device,
                gene_ids=gene_ids,
                bs=bs,
                return_grn_attention=True,
            )

        att = att.detach().cpu().view(-1).numpy()
        return att

    raise RuntimeError("No batches produced from test slide (sampler yielded nothing).")


# =========================================================
# 2) Mean expression per class on TEST slide (seed nodes only)
# =========================================================
def compute_median_expr_per_class_on_slide(slide, device, target_key, batch_size=64):
    sampler = build_local_sampler(slide, batch_size=batch_size, num_workers=0, shuffle=False)

    X_list, y_list = [], []
    for (
        x_subgraph_raw,
        edge_index_subgraph,
        edge_attr_b,
        node2comm_batch,
        super_sp_global,
        gene_ids,
        y_seed,
        bs,
    ) in iterate_supervised_batches(slide, sampler, device=device, target_key=target_key):
        X_list.append(x_subgraph_raw[:bs].detach().cpu().numpy())   # [bs, G]
        y_list.append(y_seed.detach().cpu().numpy().astype(int))    # [bs]

    X_all = np.vstack(X_list)               # [N, G]
    y_all = np.concatenate(y_list)          # [N]
    C = int(y_all.max() + 1)

    med = np.zeros((C, X_all.shape[1]), dtype=np.float64)
    for c in range(C):
        m = (y_all == c)
        if m.sum() == 0:
            continue
        med[c] = np.median(X_all[m], axis=0)

    # per-class normalisation for comparability
   # med = med / (med.max(axis=1, keepdims=True) + 1e-12)
    return med
def compute_mean_expr_per_class_on_slide(slide, device, target_key, batch_size=64):
    sampler = build_local_sampler(slide, batch_size=batch_size, num_workers=0, shuffle=False)

    X = slide["X"]
    G = X.shape[1]
    y_all = slide[target_key]
    if torch.is_tensor(y_all):
        y_all = y_all.detach().cpu().numpy()
    else:
        y_all = np.asarray(y_all)

    C = int(y_all.max() + 1)

    sum_expr = np.zeros((C, G), dtype=np.float64)
    count = np.zeros((C,), dtype=np.int64)

    for (
        x_subgraph_raw,
        edge_index_subgraph,
        edge_attr_b,
        node2comm_batch,
        super_sp_global,
        gene_ids,
        y_seed,
        bs,
    ) in iterate_supervised_batches(slide, sampler, device=device, target_key=target_key):

        xb = x_subgraph_raw[:bs].detach().cpu().numpy()  # [bs, G]
        y = y_seed.detach().cpu().numpy().astype(int)

        for c in np.unique(y):
            m = (y == c)
            sum_expr[c] += xb[m].sum(axis=0)
            count[c] += int(m.sum())

    mean_expr = sum_expr / (count[:, None] + 1e-12)

    # per-class normalisation for comparability
    mean_expr = mean_expr / (mean_expr.max(axis=1, keepdims=True) + 1e-12)
    return mean_expr  # [C, G]


# =========================================================
# 3) Top edges + top genes per class
#    score_c(edge)=att(edge)*mean_expr_c[src]*mean_expr_c[dst]
# =========================================================
def build_top_edges_and_genes_per_class(
    gene_graphs, gene_names, att, mean_expr,
    top_k_edges=30, top_k_genes=30
):
    edge_index = gene_graphs["global"].edge_index.detach().cpu().numpy()
    src, dst = edge_index[0], edge_index[1]
    E = edge_index.shape[1]
    C = mean_expr.shape[0]
    G = len(gene_names)

    salience = compute_edge_salience(gene_graphs, att)  # attention * log2(1 + target_in_degree)

    # --- edge scores per class (ATTENTION-DRIVEN via log1p) ---
    edge_scores = np.zeros((C, E), dtype=np.float64)
    for c in range(C):
        expr_src = mean_expr[c, src]
        expr_dst = mean_expr[c, dst]
        edge_scores[c] = salience * expr_src * expr_dst

    # --- gene strength per class ---
    gene_strength = np.zeros((C, G), dtype=np.float64)
    for c in range(C):
        np.add.at(gene_strength[c], src, edge_scores[c])
        np.add.at(gene_strength[c], dst, edge_scores[c])

    edge_tables, gene_tables = {}, {}
    for c in range(C):
        eidx = np.argsort(edge_scores[c])[-top_k_edges:][::-1]

        edge_tables[c] = pd.DataFrame({
            "class_id": c,
            "source_gene": [gene_names[i] for i in src[eidx]],
            "target_gene": [gene_names[i] for i in dst[eidx]],
            "attention": np.asarray(att)[eidx],
            "salience": salience[eidx],
            "expr_src": mean_expr[c, src[eidx]],
            "expr_dst": mean_expr[c, dst[eidx]],
        
            "edge_score": edge_scores[c, eidx],
        })

        gidx = np.argsort(gene_strength[c])[-top_k_genes:][::-1]
        gene_tables[c] = pd.DataFrame({
            "class_id": c,
            "gene": [gene_names[i] for i in gidx],
            "gene_strength": gene_strength[c, gidx],
        })

    return edge_tables, gene_tables


def plot_top_genes_per_class(gene_tables, out_dir, id_to_name=None):
    os.makedirs(out_dir, exist_ok=True)

    # one distinct colour per class (repeat-safe)
    C = len(gene_tables)
    cmap = plt.get_cmap("tab20")  # nice categorical palette
    class_colors = [cmap(i % cmap.N) for i in range(C)]

    for c, df in gene_tables.items():
        c_int = int(c)
        niche_label = id_to_name.get(c_int, f"class {c_int}") if id_to_name else f"class {c_int}"
        color = class_colors[c_int % C]

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(df)), df["gene_strength"].values, color=color)
        plt.xticks(range(len(df)), df["gene"].values, rotation=75, ha="right")
        plt.title(f"Top genes by niche-contextual GRN strength – {niche_label} (id={c_int})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"class_{c_int}_top_genes.png"), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


EDGE_ATTR_NAMES = ["pathway", "spearman", "ligrec", "tf_target"]

def edge_type_string(edge_attr_row):
    idx = np.where(edge_attr_row > 0)[0]
    return "unknown" if len(idx) == 0 else "|".join(EDGE_ATTR_NAMES[i] for i in idx)

def top10_edges_for_true_niche(model, slide_test, gene_graphs, gene_names, att, niche_id, device,id_to_name=None, target_key="y_niche",hk_set=None, hk_regex=True, top_k_print=10):
    # 1) collect seed expressions + y_true over the test slide
    niche_name = id_to_name.get(niche_id, str(niche_id)) if id_to_name else str(niche_id)
    print(f"\nNiche: {niche_name} (id={niche_id})")
    sampler = build_local_sampler(slide_test, batch_size=64, num_workers=0, shuffle=False)

    X_list, y_list = [], []
    for (
        x_subgraph_raw,
        edge_index_subgraph,
        edge_attr_b,
        node2comm_batch,
        super_sp_global,
        gene_ids,
        y_seed,
        bs,
    ) in iterate_supervised_batches(slide_test, sampler, device=device, target_key=target_key):

        X_list.append(x_subgraph_raw[:bs].detach().cpu().numpy())   # [bs, G]
        y_list.append(y_seed.detach().cpu().numpy().astype(int))    # [bs]

    X_all = np.vstack(X_list)              # [N, G]
    y_true = np.concatenate(y_list)        # [N]

    # 2) mean expr for this true niche
    mask = (y_true == niche_id)
    mean_expr = np.median(X_all[mask], axis=0)            # [G]  ✅ robust to imbalance
    mean_expr = mean_expr / (mean_expr.max() + 1e-12)

    # 3) edge scores = att * expr_src * expr_dst
    edge_index = gene_graphs["global"].edge_index.detach().cpu().numpy()  # [2, E]
    edge_attr  = gene_graphs["global"].edge_attr.detach().cpu().numpy()   # [E, 4]

    src = edge_index[0]
    dst = edge_index[1]
    #salience = compute_edge_salience(gene_graphs, att)
    score = edge_score_from_mean_expr(gene_graphs, att, mean_expr)        # [E]
    top_idx = np.argsort(score)[-10:][::-1]

    printed = 0
    for e in top_idx:
        g1 = gene_names[int(src[e])]
        g2 = gene_names[int(dst[e])]

        # ✅ skip housekeeping edges
        if is_housekeeping_gene(g1, hk_set, hk_regex) or is_housekeeping_gene(g2, hk_set, hk_regex):
            continue

        et = edge_type_string(edge_attr[e])
        print(f"{g1} - {g2} ({et}) score={score[e]:.4f}")
        printed += 1
        if printed >= top_k_print:
            break



def plot_shared_edges_heatmap_directed_fancy(
    unique_edges,
    niche_names,
    save_path=None,
    cmap="magma",
    figsize=(22, 7),

    # ---- font controls (mentor-friendly) ----
    title_fs=18,
    axis_fs=14,
    ytick_fs=12,
    xtick_fs=11,
    cbar_label_fs=13,
    cbar_tick_fs=11,
    kbadge_label_fs=11,

    # ---- layout / style ----
    zscore_per_edge=False,
    show_values=False,
    max_cols=80,                 # keep readable; increase if you want
    grid_alpha=0.18,
    sep_lw=2.6,
):
    """
    rows = niches
    cols = unique directed edges (shared edges appear once)
    value = score in that niche
    """

    C = len(niche_names)
    K = len(unique_edges)
    if K == 0:
        print("No edges to plot.")
        return

    # sort: shared most + strongest
    unique_edges = sorted(unique_edges, key=lambda d: (d["selected_count"], d["best_score"]), reverse=True)

    if K > max_cols:
        unique_edges = unique_edges[:max_cols]
        K = len(unique_edges)

    H = np.stack([d["scores"] for d in unique_edges], axis=1).astype(float)  # [C,K]

    # optional z-score per column
    if zscore_per_edge:
        mu = H.mean(axis=0, keepdims=True)
        sd = H.std(axis=0, keepdims=True) + 1e-12
        H_plot = (H - mu) / sd
        cbar_label = "Edge score (z-score per edge)"
    else:
        H_plot = H
        cbar_label = "Edge score"

    col_labels = [f"{d['src_gene']}→{d['dst_gene']}" for d in unique_edges]
    k_list = np.array([d["selected_count"] for d in unique_edges], dtype=int)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(H_plot, aspect="auto", cmap=cmap)

    # colourbar with bigger fonts
    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.01)
    cbar.set_label(cbar_label, fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    # axis labels + title (bigger)
    ax.set_title(
        "Niche-contextualised GRN edge scores across niches (top 10 per niche)",
        fontsize=title_fs,
        pad=18,
        fontweight="bold",
    )
    ax.set_ylabel("", fontsize=axis_fs)
    #ax.set_xlabel("Edges (gene→gene)", fontsize=axis_fs)

    # ticks (bigger gene labels)
    ax.set_yticks(np.arange(C))
    ax.set_yticklabels(niche_names, fontsize=ytick_fs)

    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=xtick_fs)

    # subtle grid
    ax.set_xticks(np.arange(-.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-.5, C, 1), minor=True)
    ax.grid(which="minor", linewidth=0.35, alpha=grid_alpha)
    ax.tick_params(which="minor", bottom=False, left=False)

    # separators between k-groups
    boundaries = np.where(np.diff(k_list) != 0)[0]
    for b in boundaries:
        ax.axvline(b + 0.5, linewidth=sep_lw, alpha=0.6)

    # top k-badge row
    #ax_top = ax.inset_axes([0, 1.02, 1, 0.06])  # [x0,y0,w,h] in axes fraction
    
    k_norm = Normalize(vmin=k_list.min(), vmax=(k_list.max() if k_list.max() > k_list.min() else k_list.min() + 1))
    k_colors = cm.get_cmap("viridis")(k_norm(k_list))

    k_img = np.zeros((1, K, 4))
    k_img[0, :, :] = k_colors
    # ax_top.imshow(k_img, aspect="auto")
    # ax_top.set_xticks([])
    # ax_top.set_yticks([])
    

    # optional numeric values
    if show_values:
        for i in range(C):
            for j in range(K):
                ax.text(j, i, f"{H_plot[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

def directed_key(src_gene: str, dst_gene: str):
    return (src_gene, dst_gene)
def build_unique_top_edges_across_niches_directed(
    score_matrix,          # [C, E]
    edge_index,            # [2, E] numpy
    gene_names,            # list length G
    top_k=10,
):
    """
    1) For each niche c: take top_k edges by score_matrix[c]
       - deduplicate within niche by (src_gene, dst_gene)
    2) Merge across niches: same directed edge appears ONCE globally
       - store selected_by niches
       - store scores across all niches for that chosen edge-id

    Returns: list[dict] sorted by (selected_count desc, best_score desc)
    """
    src = edge_index[0]
    dst = edge_index[1]
    C, E = score_matrix.shape

    # collect top edges per niche
    niche_top = []  # (c, key, e)
    for c in range(C):
        eidx = np.argsort(score_matrix[c])[-top_k:][::-1]
        seen = set()
        for e in eidx:
            sg = gene_names[int(src[e])]
            tg = gene_names[int(dst[e])]
            k = directed_key(sg, tg)
            if k in seen:
                continue
            seen.add(k)
            niche_top.append((c, k, int(e)))

    # merge across niches
    merged = {}  # key -> dict
    for c, k, e in niche_top:
        if k not in merged:
            merged[k] = {
                "src_gene": k[0],
                "dst_gene": k[1],
                "edge_id": e,                 # representative edge_id
                "selected_by": set(),
                "scores": score_matrix[:, e].copy(),  # scores across all niches
            }
        merged[k]["selected_by"].add(c)

    unique_edges = []
    for k, d in merged.items():
        d["selected_by"] = sorted(list(d["selected_by"]))
        d["selected_count"] = len(d["selected_by"])
        d["best_score"] = float(np.max(d["scores"]))
        d["best_niche"] = int(np.argmax(d["scores"]))
        unique_edges.append(d)

    unique_edges.sort(key=lambda x: (x["selected_count"], x["best_score"]), reverse=True)
    return unique_edges
 # [C,E]


# top10_edges_for_true_niche(model, slide_test, gene_graphs, gene_names, att, niche_id=1, device=device)
# =========================================================
# 4) MAIN
# =========================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

target_key = "y_niche"
n_classes = 4
species = "human"

data_root = "/DATA1/Roxana/Hierarchy/Two_levels/fine_tune/Liver_cosmax_cancer"
h5ad_path = "/DATA1/Roxana/Data/cancer_liver_cosmax.h5ad"
vocab_path = "/DATA1/Roxana/Hierarchy/Two_levels/default_gene_vocab.json"

foundation_ckpt = "./hebi_foundation_best2.pt"
finetuned_ckpt = "./checkpoints/final_finetuned_model_livercancer.pt"  # your saved finetuned model

out_dir = "./checkpoints/grn_niche_reports/Livercancer"
plot_dir = "./checkpoints/grn_niche_plots/Livercancer"

top_k_edges = 30
top_k_genes = 30
batch_size_for_stats = 64
# =======================

os.makedirs(out_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ---- Load + harmonise adata (defines gene order) ----
print("📦 Loading + harmonising adata...")
adata = sc.read_h5ad(h5ad_path)
adata = harmonise_adata(adata, vocab_path, fine_tune=True, species=species)

gene_names = adata.var["gene_symbol"].astype(str).str.upper().tolist()
print("✅ Harmonised genes:", len(gene_names))

# ---- Build GRN on harmonised genes ----
print("🧬 Building GRN...")
gene_graphs, grn_gene_names, edge_attr_names = build_hetero_grn(
    adata=adata, 
    species="human", 
    spearman_threshold=0.4,
    add_reverse_for_curated=True
)
gene_names = grn_gene_names
print("✅ GRN nodes:", gene_graphs["global"].num_nodes, "| edges:", gene_graphs["global"].edge_index.shape[1])

# ---- Load one TEST slide ----
test_files = get_slide_list(data_root, split="test")
if len(test_files) == 0:
    raise RuntimeError("No test slides found in data_root/test.")

slide_test = load_supervised_slide_dict(test_files[0], target_key=target_key, map_location="cpu")
G_slide = slide_test["X"].shape[1]
print("Slide gene dim:", G_slide, "| GRN nodes:", gene_graphs["global"].num_nodes)

if G_slide != gene_graphs["global"].num_nodes:
    raise RuntimeError(
        f"Gene mismatch: slide X has {G_slide} genes but GRN has {gene_graphs['global'].num_nodes}. "
        "This means your slide gene order is not aligned with harmonised adata gene order."
    )

# ---- Build model ----
print("🧠 Building model skeleton...")
model = HebiCellTypeClassifier(
    n_classes=n_classes,
    dim_model=128,
    l0_encoder_type="transformer",
    global_num_layers=2,
    num_prototypes_lvl1=1024,
    num_prototypes_lvl2=20,
).to(device)

# Build backbone modules (so load_state_dict has targets)
model.load_pretrained_weights(foundation_ckpt, device=device, freeze_backbone=False)

# IMPORTANT: create GRN modules BEFORE loading finetuned state_dict
model.grn_layer = GRNRefinementLayer(dim_model=128, heads=4).to(device)
model.grn_expr_decoder = nn.Linear(128, 1).to(device)
model.gene_graphs = gene_graphs

# Load finetuned weights (contains GRN weights)
print("📥 Loading finetuned checkpoint:", finetuned_ckpt)
state_ft = torch.load(finetuned_ckpt, map_location=device)
model.load_state_dict(state_ft, strict=True)
model.eval()

# ---- Attention ----
print("🔎 Extracting GRN attention from test slide...")
att = get_global_grn_attention_from_one_batch(
    model=model,
    slide=slide_test,
    device=device,
    target_key=target_key,
    batch_size=batch_size_for_stats
)
print("✅ Attention length:", len(att))

# ---- Mean expression per niche label ----
print("📊 Computing mean expression per class...")
mean_expr = compute_median_expr_per_class_on_slide(slide_test, device, target_key, batch_size_for_stats)
# mean_expr = compute_mean_expr_per_class_on_slide(
#     slide=slide_test,
#     device=device,
#     target_key=target_key,
#     batch_size=batch_size_for_stats
# )

# ---- Tables ----
print("🧾 Building per-class top edges / genes...")
edge_tables, gene_tables = build_top_edges_and_genes_per_class(
    gene_graphs=gene_graphs,
    gene_names=gene_names,
    att=att,
    mean_expr=mean_expr,
    top_k_edges=top_k_edges,
    top_k_genes=top_k_genes
)

# ---- Save CSV ----
for c, df in edge_tables.items():
    df.to_csv(os.path.join(out_dir, f"class_{c}_top_edges.csv"), index=False)
for c, df in gene_tables.items():
    df.to_csv(os.path.join(out_dir, f"class_{c}_top_genes.csv"), index=False)

# ---- Plots ----

niche_names = adata.obs["niche_type"].astype("category").cat.categories.tolist()
id_to_name = {i: name for i, name in enumerate(niche_names)}
plot_top_genes_per_class(gene_tables, out_dir=plot_dir,id_to_name=id_to_name)
y_all = slide_test[target_key]
if torch.is_tensor(y_all):
    y_all = y_all.detach().cpu().numpy()
else:
    y_all = np.asarray(y_all)

present_ids = sorted(np.unique(y_all).astype(int).tolist())
print("Niche IDs present in this test slide:", present_ids)

# print top edges for each niche present
for nid in present_ids:
    top10_edges_for_true_niche(
        model=model,
        slide_test=slide_test,
        gene_graphs=gene_graphs,
        gene_names=gene_names,
        att=att,
        niche_id=nid,
        device=device,
        id_to_name=id_to_name,   # from adata.obs["niche"].cat.categories
        target_key=target_key
    )
print("✅ Done.")
print("Saved CSVs:", out_dir)
score_matrix = edge_score_from_mean_expr(gene_graphs, att, mean_expr) 
edge_index_np = gene_graphs["global"].edge_index.detach().cpu().numpy()

unique_edges = build_unique_top_edges_across_niches_directed(
    score_matrix=score_matrix,
    edge_index=edge_index_np,
    gene_names=gene_names,
    top_k=10
)
hk_set = default_housekeeping_set()
unique_edges_filt = filter_edges_by_housekeeping(unique_edges, hk_set=hk_set, hk_regex=True)
plot_shared_edges_heatmap_directed_fancy(
    unique_edges=unique_edges_filt,
    niche_names=niche_names,
    save_path=os.path.join(plot_dir, "shared_edges_heatmap_directed_fancy4.png"),
    cmap="PuBuGn",
    figsize=(24, 7),
    xtick_fs=13,   # gene labels bigger
    ytick_fs=13,
    title_fs=20,
    axis_fs=15,
    cbar_label_fs=14,
    cbar_tick_fs=12,
)
print("Saved plots:", plot_dir)
#########Genes
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os
# import numpy as np
# import scanpy as sc
# import scipy.sparse as sp
# import matplotlib.pyplot as plt


# def _get_gene_column_by_symbol(ad, gene, gene_symbol_col="gene_symbol"):
#     """
#     Returns expression vector for `gene` from ad.X by matching ad.var[gene_symbol_col].
#     Works even if gene is not in ad.var_names.
#     """
#     gene = str(gene).upper()

#     if gene in ad.var_names:
#         j = int(np.where(ad.var_names == gene)[0][0])
#     else:
#         if gene_symbol_col is None or gene_symbol_col not in ad.var.columns:
#             raise KeyError(
#                 f"Gene '{gene}' not found in var_names and gene_symbol_col='{gene_symbol_col}' "
#                 f"not in ad.var.columns."
#             )
#         syms = ad.var[gene_symbol_col].astype(str).str.upper().values
#         hits = np.where(syms == gene)[0]
#         if len(hits) == 0:
#             raise KeyError(f"Gene '{gene}' not found in ad.var['{gene_symbol_col}'].")
#         j = int(hits[0])

#     x = ad.X[:, j]
#     if sp.issparse(x):
#         x = x.toarray().ravel()
#     else:
#         x = np.asarray(x).ravel()
#     return x


# def _scale_01(v, mode="minmax", p_low=1, p_high=99):
#     """
#     Scale vector to [0,1].
#     mode:
#       - "minmax": (v-min)/(max-min)
#       - "p1p99": clip to percentiles then minmax
#     """
#     v = np.asarray(v, dtype=float)

#     if mode == "p1p99":
#         lo = np.percentile(v, p_low)
#         hi = np.percentile(v, p_high)
#         v = np.clip(v, lo, hi)

#     vmin = np.min(v)
#     vmax = np.max(v)
#     if vmax <= vmin + 1e-12:
#         return np.zeros_like(v)
#     return (v - vmin) / (vmax - vmin)
# import textwrap

# def set_niche_gene_title_fixed(
#     ax,
#     niche_name: str,
#     gene_name: str,
#     niche_fs=12,
#     gene_fs=12,
#     niche_color="black",
#     gene_color="black",
#     wrap_width=14,
#     y_niche=1.16,   # ✅ niche always same height
#     y_gene=1.04,    # ✅ gene always same height
#     line_spacing=1.05
# ):
#     ax.set_title("")

#     niche_wrapped = textwrap.fill(niche_name, width=wrap_width)
#     lines = niche_wrapped.split("\n")
#     if len(lines) > 2:
#         niche_wrapped = "\n".join(lines[:2]) + "…"
#         lines = niche_wrapped.split("\n")

#     # ✅ centre the niche block around y_niche (so 1-line and 2-line don't jump)
#     n_lines = len(lines)
#     # vertical offsets so the *block centre* stays at y_niche
#     if n_lines == 1:
#         y0 = y_niche
#     else:
#         # move first line slightly up so overall 2-line block is centred at y_niche
#         y0 = y_niche + 0.5 * 0.06

#     ax.text(
#         0.5, y0, niche_wrapped,
#         transform=ax.transAxes,
#         ha="center", va="bottom",
#         fontsize=niche_fs,
#         color=niche_color,
#         fontweight="bold",
#         linespacing=line_spacing
#     )

#     ax.text(
#         0.5, y_gene, gene_name,
#         transform=ax.transAxes,
#         ha="center", va="bottom",
#         fontsize=gene_fs,
#         color=gene_color
#     )

# from matplotlib.cm import ScalarMappable
# def plot_7cols_two_genes_merfish(
#     adata,
#     gene_pairs,
#     col_titles=None,
#     split_key=None,
#     split_value=None,
#     basis="spatial",           # uses adata.obsm["spatial"]
#     point_size=2.0,
#     cmap="viridis",
#     gene_symbol_col="gene_symbol",
#     save_path=None,
#     do_norm_log=True,          # normalise_total + log1p
#     scale_mode="p1p99",        # "minmax" or "p1p99"
#     p_low=1,
#     p_high=99,
# ):
#     # subset first (important)
#     if split_key is not None and split_value is not None:
#         ad = adata[adata.obs[split_key] == split_value].copy()
#     else:
#         ad = adata.copy()

#     # flip y ONCE (your previous version flipped twice; this keeps it clean)
#     if basis in ad.obsm:
#         ad.obsm[basis] = ad.obsm[basis].copy()
#         ad.obsm[basis][:, 1] *= -1

#     alt = f"X_{basis}"
#     if alt in ad.obsm:
#         ad.obsm[alt] = ad.obsm[alt].copy()
#         ad.obsm[alt][:, 1] *= -1

#     if basis not in ad.obsm_keys():
#         raise KeyError(f"adata.obsm['{basis}'] not found. Available: {list(ad.obsm_keys())}")

#     # normalise/log if you want expression comparable
#     if do_norm_log:
#         sc.pp.normalize_total(ad, target_sum=1e4)
#         sc.pp.log1p(ad)

#     # precompute per-gene scaled vectors into ad.obs (so scanpy can colour by obs)
#     all_genes = sorted({g for pair in gene_pairs for g in pair})
#     gene_to_obskey = {}
#     for g in all_genes:
#         vec = _get_gene_column_by_symbol(ad, g, gene_symbol_col=gene_symbol_col)
#         vec01 = _scale_01(vec, mode=scale_mode, p_low=p_low, p_high=p_high)
#         obskey = f"{g}__01"
#         ad.obs[obskey] = vec01
#         gene_to_obskey[g] = obskey

#     # plot layout: 1 row of columns, each column has 2 plots
#     ncols = len(gene_pairs)
#     fig = plt.figure(figsize=(3.2 * ncols, 6.2))
#     outer = fig.add_gridspec(1, ncols, wspace=0.18)

#     # ---- ONE colourbar only for the LAST column ----
#     norm = Normalize(vmin=0.0, vmax=1.0)
#     mappable = ScalarMappable(norm=norm, cmap=cmap)
#     mappable.set_array([])

#     last_ax_pair = None

#     for j, (g1, g2) in enumerate(gene_pairs):
#         inner = outer[0, j].subgridspec(2, 1, hspace=0.05)
#         ax1 = fig.add_subplot(inner[0, 0])
#         ax2 = fig.add_subplot(inner[1, 0])

#         # top gene (NO scanpy colourbar)
#         sc.pl.embedding(
#             ad,
#             basis=basis,
#             color=gene_to_obskey[g1],   # obs key (scaled 0-1)
#             size=point_size,
#             cmap=cmap,
#             vmin=0, vmax=1,
#             ax=ax1,
#             show=False,
#             title=g1,
#             colorbar_loc=None,          # ✅ important
#         )
#         ax1.invert_yaxis()
        
#         # bottom gene (NO scanpy colourbar)
#         sc.pl.embedding(
#             ad,
#             basis=basis,
#             color=gene_to_obskey[g2],
#             size=point_size,
#             cmap=cmap,
#             vmin=0, vmax=1,
#             ax=ax2,
#             show=False,
#             title=g2,
#             colorbar_loc=None,          # ✅ important
#         )
#         ax2.invert_yaxis()
#         if col_titles is not None:
#             set_niche_gene_title_fixed(
#         ax1,
#         niche_name=col_titles[j],
#         gene_name=g1,
#         niche_fs=12,
#         gene_fs=14,
#         wrap_width=14,
#         y_niche=1.16,
#         y_gene=1.04,
#     )


#             # ax2 (bottom plot): gene only
#             ax2.set_title(g2, fontsize=14)

#         for ax in (ax1, ax2):
#             ax.set_xlabel("")
#             ax.set_ylabel("")
#             ax.set_xticks([])
#             ax.set_yticks([])
#             for spine in ax.spines.values():
#                 spine.set_visible(False)
#             ax.tick_params(left=False, bottom=False)

#         if j == ncols - 1:
#             last_ax_pair = (ax1, ax2)

#     # ✅ Add ONE colourbar attached to the last column only
#     if last_ax_pair is not None:
#         ax1_last, ax2_last = last_ax_pair
#         cbar = fig.colorbar(
#             mappable,
#             ax=[ax1_last, ax2_last],
#             fraction=0.05,
#             pad=0.02
#         )
#         cbar.set_label("Expression (scaled 0–1)", fontsize=12)
#         cbar.ax.tick_params(labelsize=10)

#   #  fig.suptitle("Top-2 genes per niche", fontsize=18, y=1.02)
#     fig.tight_layout()

#     if save_path:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         fig.savefig(save_path, dpi=600, bbox_inches="tight")

#     plt.show()
#     return fig
# pairs_first2 = [
#     ["NFIB", "NFIX"],         # niche 0
#     ["FLI1", "TGFBR2"],       # niche 1
#     ["GJA1", "AQP4"],         # niche 2
#     ["SLC17A6", "NRN1"],      # niche 3
#     ["ADCY2", "EPHA4"],       # niche 4
#     ["SLC32A1", "SPOCK3"],    # niche 5
#     ["CLDN5", "CD34"],        # niche 6
# ]

# niche_names = [
#     "CBX-MOB-other neuronal",
#     "Immune",
#     "Neuroglial",
#     "PAL-sAMY-TH-HY-MB-HB neuronal",
#     "Pallium glutamatergic",
#     "Subpallium GABAergic",
#     "Vascular",
# ]
# plot_7cols_two_genes_merfish(
#     adata_s,
#     gene_pairs=pairs_first2,      # list of 7 tuples (geneA, geneB)
#     col_titles=niche_names,       # list of 7 niche names
#     split_key="split",
#     split_value="test",
#     basis="spatial",
#     point_size=2.0,
#     cmap="turbo",               # try: "cividis", "plasma", "turbo"
#     gene_symbol_col="gene_symbol",
#     save_path="niche_pairs_7cols_2genes_test.png",
#     do_norm_log=True,
#     scale_mode="p1p99",
#     p_low=1, p_high=99
# )