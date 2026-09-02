#!/usr/bin/env python3
"""
Inter-niche Cell-Cell Communication for HEBI Pipeline — v2
===========================================================
Changes from v1:
  - plot_ccc_directed_graph: sender+strongest_receiver both highlighted,
    sender label outside circle, smaller circles, per-niche colours
  - plot_niche_grn_heatmap: one subplot per niche (top-10 edges each)
    instead of one huge heatmap
  - NEW plot_grn_circular_chord: beautiful circular chord diagram showing
    GRN edge-type composition per niche — the surprise plot
  - plot_sender_niche_panel: spatial map shows sender (blue) + top receiver (red)
"""

import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Arc, Wedge
import warnings
warnings.filterwarnings("ignore")

# ── Palette ──────────────────────────────────────────────────────────────────
SENDER_COLOR   = "#1A3A5C"   # dark navy  — sender niche
RECEIVER_COLOR = "#E8703A"   # warm orange — strongest receiver
OTHER_COLOR    = "#B8D4E8"   # light blue  — other receivers
EDGE_COLORS = {
    "pathway":   "#D4A5B8" ,   # Deep Indigo/Plum
    "spearman":  "#7192C9",   # Soft Lavender
    "ligrec":  "#5C3C8B" ,   # Dusty Mauve/Rose
    "tf_target": "#9E86C7",   # Periwinkle Blue
}
NICHE_CMAP = plt.cm.get_cmap("Purples", 20)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _shorten(name, max_len=13):
    return name if len(name) <= max_len else name[:max_len] + "…"

def _mean_expr_per_niche(adata, cluster_key):
    clusters  = adata.obs[cluster_key].astype(str).values
    unique    = sorted(set(clusters))
    var_names = list(adata.var_names)
    mean_expr = {}
    for c in unique:
        mask = clusters == c
        sub  = adata.X[mask]
        mean_expr[c] = (np.asarray(sub.mean(axis=0)).ravel()
                        if sp.issparse(sub) else np.asarray(sub).mean(axis=0).ravel())
    return mean_expr, var_names, unique

def _get_ligrec_edges(gene_graphs, gene_names, grn_attn_weights,
                      edge_attr_names=None):
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]
    ligrec_idx = edge_attr_names.index("ligrec") if "ligrec" in edge_attr_names else None
    if ligrec_idx is None:
        return pd.DataFrame()
    g          = gene_graphs["global"]
    edge_index = g.edge_index.cpu().numpy()
    edge_attr  = g.edge_attr.cpu().numpy()
    attn = (grn_attn_weights.detach().cpu().numpy()
            if torch.is_tensor(grn_attn_weights)
            else np.asarray(grn_attn_weights).ravel())
    rows = []
    for e in range(edge_index.shape[1]):
        if edge_attr[e, ligrec_idx] > 0:
            rows.append({
                "ligand":   gene_names[int(edge_index[0, e])],
                "receptor": gene_names[int(edge_index[1, e])],
                "attention": float(attn[e]),
                "pair": f"{gene_names[int(edge_index[0,e])]} → {gene_names[int(edge_index[1,e])]}",
            })
    return pd.DataFrame(rows).sort_values("attention", ascending=False)

def build_ccc_tensor(adata, gene_graphs, gene_names, grn_attn_weights,
                     cluster_key="annotation", edge_attr_names=None):
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]
    lr_df = _get_ligrec_edges(gene_graphs, gene_names, grn_attn_weights, edge_attr_names)
    if lr_df.empty:
        print("  ⚠️  No LigRec edges found.")
        return pd.DataFrame()
    mean_expr, var_names, niches = _mean_expr_per_niche(adata, cluster_key)
    var_to_col = {g: i for i, g in enumerate(var_names)}
    rows = []
    for _, lr in lr_df.iterrows():
        lig, rec, att = lr["ligand"], lr["receptor"], lr["attention"]
        lig_col = var_to_col.get(lig)
        rec_col = var_to_col.get(rec)
        if lig_col is None or rec_col is None:
            continue
        for sender in niches:
            lig_expr = float(mean_expr[sender][lig_col])
            if lig_expr <= 0: continue
            for receiver in niches:
                rec_expr = float(mean_expr[receiver][rec_col])
                if rec_expr <= 0: continue
                rows.append({
                    "pair": lr["pair"], "ligand": lig, "receptor": rec,
                    "attention": att, "sender": sender, "receiver": receiver,
                    "ligand_expr": lig_expr, "receptor_expr": rec_expr,
                    "score": att * lig_expr * rec_expr,
                })
    df = pd.DataFrame(rows)
    print(f"  ✅ CCC tensor: {len(df)} sender→receiver interactions")
    return df

def find_top_sender_niches(comm_df, top_k=3):
    if comm_df.empty: return []
    df = comm_df[comm_df["sender"] != comm_df["receiver"]].copy()
    sender_scores = df.groupby("sender")["score"].sum().sort_values(ascending=False)
    top = sender_scores.head(top_k).index.tolist()
    print(f"\n🏆 Top {top_k} sender niches:")
    for i, s in enumerate(top):
        print(f"   {i+1}. {s}: score = {sender_scores[s]:.4f}")
    return top


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Communication matrix heatmap
# ═════════════════════════════════════════════════════════════════════════════
def plot_ccc_matrix(comm_df, top_k_pairs=10,entity_label="niche", figsize=None, save_path=None):
    if comm_df.empty: return
    agg = comm_df.groupby(["pair","sender","receiver"])["score"].sum().reset_index()
    agg["comm_axis"] = agg["sender"] + "→" + agg["receiver"]
    top_pairs = (agg.groupby("pair")["score"].sum()
                    .sort_values(ascending=False).head(top_k_pairs).index.tolist())
    agg = agg[agg["pair"].isin(top_pairs)]
    pivot = agg.pivot_table(index="pair", columns="comm_axis",
                            values="score", aggfunc="sum", fill_value=0)
    top_axes = pivot.sum(axis=0).sort_values(ascending=False).head(30).index
    pivot = pivot[top_axes]
    n_rows, n_cols = pivot.shape
    if figsize is None:
        figsize = (max(12, n_cols * 0.35), max(6, n_rows * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    vals = pivot.values.astype(float)
    vmax = np.percentile(vals[vals > 0], 95) if (vals > 0).any() else 1.0
    im = ax.imshow(vals, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.5, label="Communication score")
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(pivot.columns, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel(f"Sender → Receiver {entity_label}", fontsize=10)
    ax.set_ylabel("Ligand → Receptor pair", fontsize=10)
    ax.set_title(f"Inter {entity_label} Cell–Cell Communication (HEBI GRN-weighted)", fontsize=18)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved CCC matrix → {save_path}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Directed graph (sender highlighted + strongest receiver in orange)
# ═════════════════════════════════════════════════════════════════════════════
def plot_ccc_directed_graph(comm_df, sender_niche,entity_label="niche", top_k_pairs=5,
                            figsize=(11, 8), save_path=None):
    if comm_df.empty: return
    df = comm_df[comm_df["sender"] == str(sender_niche)].copy()
    if df.empty:
        print(f"  ⚠️  No outgoing communication from {sender_niche}.")
        return

    agg = df.groupby(["receiver","pair"])["score"].sum().reset_index()
    top_pairs = (agg.groupby("pair")["score"].sum()
                    .sort_values(ascending=False).head(top_k_pairs).index.tolist())
    agg = agg[agg["pair"].isin(top_pairs)]

    recv_score = (agg.groupby("receiver")["score"].sum()
                     .sort_values(ascending=False))
    recv_score = recv_score[recv_score.index != str(sender_niche)].head(8)
    if recv_score.empty: return

    receivers  = recv_score.index.tolist()
    scores     = recv_score.values
    top_recv   = receivers[0]   # strongest receiver → orange

    # Layout
    n_recv = len(receivers)
    angles = np.linspace(np.pi * 0.08, np.pi * 1.92, n_recv)
    radius = 2.8
    pos = {str(sender_niche): np.array([0.0, 0.0])}
    for i, r in enumerate(receivers):
        pos[r] = np.array([radius * np.cos(angles[i]),
                           radius * np.sin(angles[i])])

    cmap_pairs = plt.cm.get_cmap("tab10", len(top_pairs))
    pair_colors = {p: cmap_pairs(i) for i, p in enumerate(top_pairs)}

    s_min, s_max = scores.min(), scores.max()
    def arrow_lw(s):
        return 1.2 + 7.0 * (s - s_min) / (s_max - s_min + 1e-8)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#F8F9FA")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#F8F9FA")

    center = np.array([0.0, 0.0])

    # ── Arrows ──
    for recv, score in zip(receivers, scores):
        src = pos[str(sender_niche)]
        dst = pos[recv]
        unit = (dst - src) / np.linalg.norm(dst - src)
        a_start = src + unit * 0.22
        a_end   = dst - unit * 0.18
        top_pair = agg[agg["receiver"] == recv].sort_values("score", ascending=False).iloc[0]["pair"]
        color = pair_colors.get(top_pair, "#4A90D9")
        lw = arrow_lw(score)
        ax.annotate("", xy=a_end, xytext=a_start,
                    arrowprops=dict(
                        arrowstyle=f"->,head_width={lw*0.07:.2f},head_length={lw*0.045:.2f}",
                        lw=lw, color=color, alpha=0.85,
                        connectionstyle="arc3,rad=0.0"))
        mid = (a_start + a_end) / 2
        ax.text(mid[0], mid[1] + 0.08, f"{score:.2f}",
                ha="center", va="bottom", fontsize=6.5,
                color=color, alpha=0.95,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="#F8F9FA")])

    # ── Receiver nodes ──
    for recv, score in zip(receivers, scores):
        xy  = pos[recv]
        is_top = (recv == top_recv)
        node_r = 0.18 if is_top else 0.15
        node_c = RECEIVER_COLOR if is_top else OTHER_COLOR
        edge_c = "#C0392B" if is_top else "#4A90D9"
        lw_c   = 2.0 if is_top else 1.0
        circ = plt.Circle(xy, node_r, color=node_c, zorder=4,
                          linewidth=lw_c, edgecolor=edge_c)
        ax.add_patch(circ)
        # Label outside node along radial direction
        direction = xy - center
        unit = direction / (np.linalg.norm(direction) + 1e-8)
        lpos = xy + unit * (node_r + 0.32)
        fw = "bold" if is_top else "normal"
        fc = "#C0392B" if is_top else "#333333"
        ax.text(lpos[0], lpos[1], _shorten(recv),
                ha="center", va="center", fontsize=7.5,
                fontweight=fw, color=fc,
                path_effects=[pe.withStroke(linewidth=2, foreground="#F8F9FA")])

    # ── Sender node — smaller, label outside ──
    sender_r = 0.28
    src_circ = plt.Circle(center, sender_r, color=SENDER_COLOR,
                           zorder=5, linewidth=2.0, edgecolor="#0A1A2C")
    ax.add_patch(src_circ)
    # Label above the circle
    ax.text(center[0], center[1] + sender_r + 0.18,
            _shorten(str(sender_niche), max_len=16),
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=SENDER_COLOR,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#F8F9FA")])

    # ── Legend ──
    patches = [mpatches.Patch(color=pair_colors[p], label=p) for p in top_pairs]
    patches += [
        mpatches.Patch(color=SENDER_COLOR,   label="Sender niche"),
        mpatches.Patch(color=RECEIVER_COLOR, label="Strongest receiver"),
        mpatches.Patch(color=OTHER_COLOR,    label="Other receivers"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=7.5,
              title="LigRec pairs & node types", title_fontsize=8,
              framealpha=0.95, edgecolor="#CCCCCC")

    # ── Comm strength scale ──
    for idx, (val, lw) in enumerate([(s_min, arrow_lw(s_min)),
                                      (s_max, arrow_lw(s_max))]):
        y = -2.6 + idx * 0.35
        ax.annotate("", xy=(3.4, y), xytext=(2.7, y),
                    arrowprops=dict(arrowstyle="->", lw=lw, color="gray"))
        ax.text(2.65, y, f"{val:.2f}", ha="right", va="center",
                fontsize=7, color="gray")
    ax.text(3.05, -2.15, "Comm.\nstrength", ha="center",
            fontsize=7, color="gray")

    ax.set_xlim(-4.0, 4.5)
    ax.set_ylim(-3.0, 4.0)
    ax.set_title(f"{entity_label.capitalize()} {_shorten(str(sender_niche),18)} → Receiver niches\n"
                 f"(top {top_k_pairs} LigRec pairs, GRN-weighted)",
                 fontsize=12, pad=12, color="#1A1A2E")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved directed graph → {save_path}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Per-niche GRN heatmaps (one subplot per niche, top-10 edges each)
# ═════════════════════════════════════════════════════════════════════════════
def plot_niche_grn_heatmaps(adata, gene_graphs, gene_names, grn_attn_weights,entity_label="niche",
                             cluster_key="annotation", edge_attr_names=None,
                             top_k_edges=10, ncols=4, save_path=None):
    """
    One small heatmap per niche showing its top-10 GRN edges.
    Rows = top edges, single column = attention score as a bar.
    """
    if edge_attr_names is None:
        edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    g          = gene_graphs["global"]
    edge_index = g.edge_index.cpu().numpy()
    edge_attr  = g.edge_attr.cpu().numpy()
    attn = (grn_attn_weights.detach().cpu().numpy()
            if torch.is_tensor(grn_attn_weights)
            else np.asarray(grn_attn_weights).ravel())

    var_to_col = {gn: i for i, gn in enumerate(adata.var_names)}
    clusters   = adata.obs[cluster_key].astype(str).values
    niches     = sorted(set(clusters))

    # Pre-compute mean expr per niche
    mean_expr = {}
    for c in niches:
        mask = clusters == c
        sub  = adata.X[mask]
        mean_expr[c] = (np.asarray(sub.mean(axis=0)).ravel()
                        if sp.issparse(sub) else np.asarray(sub).mean(axis=0).ravel())

    def edge_type_color(attr_row):
        for i, name in enumerate(edge_attr_names):
            if i < len(attr_row) and attr_row[i] > 0:
                return EDGE_COLORS.get(name, "#999999")
        return "#999999"

    n_niches = len(niches)
    nrows = int(np.ceil(n_niches / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.2, nrows * 3.2),
                              facecolor="white")
    axes = np.array(axes).ravel()

    for ax_i, niche in enumerate(niches):
        ax = axes[ax_i]
        # Score each edge for this niche
        scored = []
        expr = mean_expr[niche]
        for e in range(edge_index.shape[1]):
            src_g = gene_names[edge_index[0, e]]
            dst_g = gene_names[edge_index[1, e]]
            sc = var_to_col.get(src_g); dc = var_to_col.get(dst_g)
            if sc is None or dc is None: continue
            score = float(attn[e]) * float(expr[sc]) * float(expr[dc])
            scored.append((score, float(attn[e]),
                           src_g, dst_g, edge_attr[e]))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k_edges]

        if not top:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.axis("off")
            ax.set_title(_shorten(niche, 18), fontsize=8, pad=3)
            continue

        labels = [f"{r[2]}→{r[3]}" for r in top]
        scores_vals = [r[0] for r in top]
        colors_bar  = [edge_type_color(r[4]) for r in top]
        max_s = max(scores_vals) if max(scores_vals) > 0 else 1

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, scores_vals, color=colors_bar,
                       alpha=0.85, height=0.7, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max_s * 1.15)
        ax.set_xlabel("score", fontsize=6.5)
        ax.tick_params(axis='x', labelsize=6)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        ax.set_title(_shorten(niche, 18), fontsize=8.5, pad=3,
                     fontweight="bold", color="#1A3A5C")

    # Hide unused axes
    for ax_i in range(n_niches, len(axes)):
        axes[ax_i].set_visible(False)

    # Edge type legend
    legend_patches = [mpatches.Patch(color=EDGE_COLORS[t], label=t)
                      for t in edge_attr_names]
    fig.legend(handles=legend_patches, loc="lower right",
               fontsize=8, title="Edge type", ncol=2,
               framealpha=0.9, bbox_to_anchor=(0.99, 0.01))

    fig.suptitle(f"Per-{entity_label} GRN Activity — Top {top_k_edges} edges per {entity_label}",
                 fontsize=14, fontweight="bold", color="#1A1A2E", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"  Saved per-niche GRN heatmaps → {save_path}")
    plt.show()



# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — CCC Chord Diagram (white background, publishable)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_ccc_chord_diagram(
    comm_df,
    entity_label="niche",
    top_k_pairs=6,
    min_score_pct=0.02, # Restored back to your original value so all data displays
    figsize=(13, 13),
    save_path=None,
):
    """
    Vibrant and high-contrast white-background CCC chord diagram.
    Uses a bold, crisp Lavender-to-Indigo palette to prevent visual "freezing".
    """
    if comm_df is None or comm_df.empty:
        return

    agg = (comm_df[comm_df["sender"] != comm_df["receiver"]]
           .groupby(["sender","receiver","pair"])["score"]
           .sum().reset_index())
    if agg.empty:
        return

    top_pairs_list = (agg.groupby("pair")["score"].sum()
                         .sort_values(ascending=False)
                         .head(top_k_pairs).index.tolist())
    agg = agg[agg["pair"].isin(top_pairs_list)].copy()

    sr_agg = (agg.sort_values("score", ascending=False)
                 .groupby(["sender","receiver"])
                 .first()
                 .reset_index())

    max_score = sr_agg["score"].max()
    sr_agg = sr_agg[sr_agg["score"] >= min_score_pct * max_score].copy()

    niches = sorted(set(comm_df["sender"].tolist() + comm_df["receiver"].tolist()))
    niches = [n for n in niches if "unlabeled" not in str(n).lower()]

    niche_score = {}
    for n in niches:
        out = float(sr_agg[sr_agg["sender"]   == n]["score"].sum())
        inc = float(sr_agg[sr_agg["receiver"] == n]["score"].sum())
        niche_score[n] = out + inc + 1e-6

    total_score = sum(niche_score.values())
    GAP_DEG  = 1.8
    usable   = 360.0 - GAP_DEG * len(niches)

    niche_angles = {}
    cursor = 0.0
    for n in niches:
        span = (niche_score[n] / total_score) * usable
        niche_angles[n] = (cursor, cursor + span, cursor + span / 2)
        cursor += span + GAP_DEG

    # 🎨 --- VIBRANT HIGHER-CONTRAST CHORD COLORS ---
    # Bold, rich tones ensuring every line stands out clearly
    vibrant_chords = ["#4A2E80", "#2E66B3", "#8E44AD", "#C25283", "#5B32A6", "#34495E"]
    pair_color  = {p: vibrant_chords[i % len(vibrant_chords)] for i, p in enumerate(top_pairs_list)}
    
    # 🎨 --- DISTINCT OUTER BLOCKS PALETTE ---
    # Alternating vibrant shades so adjacent niches don't blend together
    vibrant_niche_palette = ["#D1C4E9", "#BBDEFB", "#E1BEE7", "#C5CAE9", "#F8BBD0", "#E8EAF6"]
    niche_color = {n: vibrant_niche_palette[i % len(vibrant_niche_palette)] for i, n in enumerate(niches)}

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-7, 7); ax.set_ylim(-7.5, 7.5)

    R_outer = 4.5
    R_inner = 3.9
    R_label = 5.4

    def deg2rad(d): return np.radians(d)
    def arc_xy(r, deg):
        return r * np.cos(deg2rad(deg)), r * np.sin(deg2rad(deg))

    from matplotlib.patches import Wedge
    import matplotlib.patches as mpatch
    import matplotlib.patheffects as pe

    for n in niches:
        s, e, m = niche_angles[n]
        w = Wedge((0,0), R_outer, s, e,
                  width=R_outer - R_inner,
                  facecolor=niche_color[n],
                  alpha=1.0, linewidth=1.2, # Darkened line bounds
                  edgecolor="#4A4452", zorder=3)
        ax.add_patch(w)
        lx, ly = arc_xy(R_label, m)
        rot = m % 360
        if 90 < rot < 270: rot += 180
        short = (n[:14] + "…") if len(n) > 14 else n
        ax.text(lx, ly, short,
                ha="center", va="center",
                fontsize=11.5, fontweight="bold",
                color="#1A1521", rotation=rot % 360,
                rotation_mode="anchor",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    sr_sorted = sr_agg.sort_values("score", ascending=True)
    score_min = float(sr_sorted["score"].min())
    score_max = float(sr_sorted["score"].max())

    for _, row in sr_sorted.iterrows():
        snd = str(row["sender"])
        rcv = str(row["receiver"])
        sc  = float(row["score"])
        pr  = str(row["pair"])
        if snd not in niche_angles or rcv not in niche_angles: continue
        _, _, sm = niche_angles[snd]
        _, _, rm = niche_angles[rcv]
        p1 = np.array(arc_xy(R_inner, sm))
        p2 = np.array(arc_xy(R_inner, rm))
        color = pair_color.get(pr, "#888888")
        
        # Boosted base visibility opacity so lines aren't too faint
        alpha = 0.45 + 0.50 * (sc - score_min) / (score_max - score_min + 1e-8)
        lw    = 0.8  + 6.5 * (sc - score_min) / (score_max - score_min + 1e-8)
        
        patch = mpatch.FancyArrowPatch(
            posA=tuple(p1), posB=tuple(p2),
            connectionstyle="arc3,rad=0.0",
            arrowstyle=f"->,head_width={lw*0.045:.3f},head_length={lw*0.032:.3f}",
            lw=lw, color=color, alpha=alpha, zorder=2)
        ax.add_patch(patch)

    for idx, val in enumerate([score_min, score_max]):
        lw_ex = 0.8 + 6.5 * (val - score_min) / (score_max - score_min + 1e-8)
        y_ex  = 6.5 - idx * 0.5
        ax.plot([4.6, 5.8], [y_ex, y_ex], lw=lw_ex, color="#332C3B", solid_capstyle="round")
        ax.text(5.9, y_ex, f"{val:.1f}", va="center", fontsize=11, color="#1A1521")
    ax.text(5.2, 7.1, "Comm.\nstrength", ha="center", fontsize=11, color="#1A1521", fontweight="bold")

    patches = [mpatches.Patch(color=pair_color[p], label=p, alpha=0.9) for p in top_pairs_list]
    ax.legend(handles=patches, loc="lower right", fontsize=11,
              title="LigRec pairs (chord color)", title_fontsize=13,
              framealpha=0.95, edgecolor="#BCB8C4", bbox_to_anchor=(1.05, -0.02))

    ax.set_title(f"Inter-{entity_label} Cell–Cell Communication", fontsize=18, pad=14, color="#1A1521", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
def run_ccc_analysis(adata, gene_graphs, gene_names, grn_attn_weights,
                     cluster_key="annotation", sender_niche=None,
                     top_k_senders=3, edge_attr_names=None,
                     top_k_pairs=5, out_dir="./grn_reports",entity_label="niche"):

    os.makedirs(out_dir, exist_ok=True)

    print("\n🧬 Building inter-niche CCC tensor...")
    comm_df = build_ccc_tensor(adata, gene_graphs, gene_names,
                                grn_attn_weights, cluster_key, edge_attr_names)
    if comm_df.empty: return comm_df

    if sender_niche is not None:
        top_senders = [str(sender_niche)]
    else:
        # Use every unique sender in the comm_df
        top_senders = sorted(comm_df["sender"].unique().tolist())
        top_senders = [s for s in top_senders if "unlabeled" not in s.lower()]
        print(f"\n📊 Plotting all {len(top_senders)} niches as senders...")

   # print("\n📊 [1] CCC matrix heatmap...")
    #plot_ccc_matrix(comm_df, top_k_pairs=10,entity_label=entity_label,
     #               save_path=os.path.join(out_dir, "ccc_inter_niche_matrix.png"))

   # print("\n📊 [2] Per-niche GRN bar heatmaps...")
   # plot_niche_grn_heatmaps(adata, gene_graphs, gene_names, grn_attn_weights,
                             # cluster_key=cluster_key, entity_label=entity_label,
                             # edge_attr_names=edge_attr_names,
                             # top_k_edges=10, ncols=4,
                             # save_path=os.path.join(out_dir, "niche_grn_heatmaps.png"))

    print("\n✨ [3] CCC chord diagram...")
    plot_ccc_chord_diagram(comm_df,
                            top_k_pairs=top_k_pairs,entity_label=entity_label,
                            save_path=os.path.join(out_dir, "ccc_chord_diagram.png"))

    for sender in top_senders:
        print(f"\n📊 Directed graph + panel for {sender}...")
        plot_ccc_directed_graph(comm_df, sender_niche=sender,
                                top_k_pairs=top_k_pairs,entity_label=entity_label,
                                save_path=os.path.join(out_dir,
                                    f"ccc_directed_niche{sender}.png"))
        plot_sender_niche_panel(adata, comm_df, sender_niche=sender,
                                cluster_key=cluster_key,entity_label=entity_label,
                                top_k_pairs=min(top_k_pairs, 3),
                                save_path=os.path.join(out_dir,
                                    f"ccc_panel_niche{sender}.png"))

    print(f"\n✅ All CCC plots saved to {out_dir}")
    return comm_df


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Full Nature-style panel (sender blue + top receiver orange)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_sender_niche_panel(
    adata,
    comm_df,
    sender_niche,
    entity_label="niche",
    cluster_key="annotation",
    top_k_pairs=3,
    figsize=(18, 10),
    save_path=None,
):
    """
    Full figure:
    Row 1: [Spatial map — sender (navy) + top receiver (orange)] [Directed graph]
    Row 2: ligand + receptor spatial scatter per top LigRec pair
    """
    if comm_df.empty:
        return

    df = comm_df[comm_df["sender"] == str(sender_niche)].copy()
    if df.empty:
        print(f"  No communication from {sender_niche}.")
        return

    top_lr    = (df.groupby("pair")["score"].sum()
                   .sort_values(ascending=False).head(top_k_pairs))
    top_pairs = top_lr.index.tolist()
    ligands   = [p.split(" → ")[0] for p in top_pairs]
    receptors = [p.split(" → ")[1] for p in top_pairs]

    # Strongest receiver
    recv_scores = (df[df["pair"].isin(top_pairs)]
                    .groupby("receiver")["score"].sum()
                    .sort_values(ascending=False))
    recv_scores = recv_scores[recv_scores.index != str(sender_niche)]
    top_recv    = recv_scores.index[0] if len(recv_scores) > 0 else None

    n_genes = len(top_pairs)
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, max(n_genes * 2, 4),
                            height_ratios=[1.6, 1],
                            hspace=0.35, wspace=0.3)

    # ── Row 1 left: spatial map ──
    ax_sp   = fig.add_subplot(gs[0, :max(n_genes * 2, 4) // 2])
    coords  = adata.obsm["spatial"]
    labels  = adata.obs[cluster_key].astype(str).values
    is_send = labels == str(sender_niche)
    is_recv = labels == str(top_recv) if top_recv else np.zeros(len(labels), bool)
    is_other = ~is_send & ~is_recv

    ax_sp.scatter(coords[is_other, 0], coords[is_other, 1],
                  c="#DDDDDD", s=0.8, alpha=0.25, rasterized=True)
    if is_recv.any():
        ax_sp.scatter(coords[is_recv, 0], coords[is_recv, 1],
                      c=RECEIVER_COLOR, s=2.5, alpha=0.8, rasterized=True,
                      label=f"Top receiver: {_shorten(str(top_recv), 14)}")
    ax_sp.scatter(coords[is_send, 0], coords[is_send, 1],
                  c=SENDER_COLOR, s=3.0, alpha=0.9, rasterized=True,
                  label=f"Sender: {_shorten(str(sender_niche), 14)}")
    ax_sp.set_title("Spatial location: sender & top receiver", fontsize=14)
    ax_sp.axis("off")
    ax_sp.legend(loc="upper right", fontsize=8, markerscale=4)

    # ── Row 1 right: simplified directed graph ──
    ax_g = fig.add_subplot(gs[0, max(n_genes * 2, 4) // 2:])
    ax_g.axis("off")
    recv_s2 = recv_scores.head(7)
    if not recv_s2.empty:
        import matplotlib.patches as mpatch_local
        recs2 = recv_s2.index.tolist()
        sc2   = recv_s2.values
        n2    = len(recs2)
        angs  = np.linspace(0.08 * np.pi, 0.92 * np.pi, n2)
        rad   = 2.1
        pg    = {str(sender_niche): np.array([0., 0.])}
        for i, r in enumerate(recs2):
            pg[r] = np.array([rad * np.cos(angs[i]),
                               rad * np.sin(angs[i])])
        smn, smx = sc2.min(), sc2.max()
        for r, s in zip(recs2, sc2):
            unit = ((pg[r] - pg[str(sender_niche)]) /
                    np.linalg.norm(pg[r] - pg[str(sender_niche)]))
            lw = 1.0 + 5.5 * (s - smn) / (smx - smn + 1e-8)
            c  = RECEIVER_COLOR if r == top_recv else "#4A90D9"
            ax_g.annotate("",
                          xy=pg[r] - unit * 0.19,
                          xytext=pg[str(sender_niche)] + unit * 0.19,
                          arrowprops=dict(arrowstyle="->", lw=lw,
                                          color=c, alpha=0.85))
            direction = pg[r] - np.array([0., 0.])
            ulabel = direction / (np.linalg.norm(direction) + 1e-8)
            lp = pg[r] + ulabel * 0.30
            ax_g.text(lp[0], lp[1], _shorten(r, 12),
                      ha="center", va="center", fontsize=6.5, fontweight="bold",
                      color="#C0392B" if r == top_recv else "#333333")
            nc = RECEIVER_COLOR if r == top_recv else OTHER_COLOR
            ax_g.add_patch(plt.Circle(pg[r], 0.16, color=nc, zorder=3,
                                       linewidth=1,
                                       edgecolor="#C0392B" if r == top_recv else "#4A90D9"))
        ax_g.add_patch(plt.Circle([0., 0.], 0.26, color=SENDER_COLOR,
                                   zorder=4, linewidth=1.5, edgecolor="#0A1A2C"))
        ax_g.text(0, 0.34, _shorten(str(sender_niche), 14),
                  ha="center", va="bottom", fontsize=7.5,
                  fontweight="bold", color=SENDER_COLOR)
        ax_g.set_xlim(-3, 3.2); ax_g.set_ylim(-0.8, 3.0)
        ax_g.set_aspect("equal")
        ax_g.set_title(
            f"Outgoing communication\n{_shorten(str(sender_niche), 16)}",
            fontsize=9)

    # ── Row 2: ligand + receptor spatial maps ──
    gene_var = list(adata.var_names)
    for k, (lig, rec) in enumerate(zip(ligands, receptors)):
        for col_k, (gene, cmap_name, label_str) in enumerate([
                (lig, "Reds",  f"Ligand: {lig}"),
                (rec, "Blues", f"Receptor: {rec}")]):
            axi = fig.add_subplot(gs[1, k * 2 + col_k])
            if gene in gene_var:
                gi   = gene_var.index(gene)
                expr = adata.X[:, gi]
                if sp.issparse(expr):
                    expr = np.asarray(expr.todense()).ravel()
                vm = np.percentile(expr[expr > 0], 95) if (expr > 0).any() else 1
                sc_p = axi.scatter(coords[:, 0], coords[:, 1],
                                   c=expr, cmap=cmap_name, s=0.8,
                                   vmin=0, vmax=vm, rasterized=True, alpha=0.7)
                plt.colorbar(sc_p, ax=axi, shrink=0.6)
            axi.set_title(label_str, fontsize=10)
            axi.axis("off")

    fig.suptitle(
        f"{entity_label.capitalize()} {_shorten(str(sender_niche), 20)} — Cell–Cell Communication\n"
        f"(GRN-weighted, cluster_key='{cluster_key}')",
        fontsize=18, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved panel → {save_path}")
    plt.show()