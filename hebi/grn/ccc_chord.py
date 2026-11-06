#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-cell communication export and chord diagram.

The communication tensor keeps every interaction where the ligand is
expressed in the sender and the receptor in the receiver, as defined in
the Methods. Filtering here is a display choice only: the exported table
is unfiltered, so downstream analyses see the full tensor.

Three filters are applied before drawing, all adjustable:

  saturation    drop edges whose attention is exactly 1.0. Attention is
                normalised over each target's incoming edges, so a
                receptor with in-degree 1 saturates by construction and
                its score is purely expression-driven.
  min_score_pct keep links scoring at least this fraction of the strongest
                link.
  top_n_edges   cap on the number of links drawn.
"""

import os

import matplotlib as mpl
import matplotlib.patches as mpatch
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["svg.fonttype"] = "none"

SATURATION_TOL = 1e-6

CHORD_COLORS = [   # distinct hues, one per ligand-receptor pair
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]

NICHE_COLORS = [
    "#D1C4E9", "#BBDEFB", "#E1BEE7", "#C5CAE9",
    "#F8BBD0", "#E8EAF6", "#C8E6C9", "#FFE0B2",
]


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
def write_ccc_excel(comm_dfs, section_labels, out_path: str):
    """Write one sheet per section into a single workbook.

    comm_dfs and section_labels are positional: sheet i holds section i's
    communication table. Sections with no interactions are skipped.
    """
    written = []
    usable = [
        (label, df)
        for label, df in zip(section_labels, comm_dfs)
        if df is not None and not df.empty
    ]

    if not usable:
        print("  No communication tables to export")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for label, df in usable:
            # Excel caps sheet names at 31 characters.
            sheet = str(label)[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            written.append(sheet)
            print(f"  {sheet}: {df.shape[0]} rows, {df.shape[1]} cols")

    print(f"  Saved {out_path}")
    return out_path


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------
def filter_for_chord(
    comm_df,
    top_k_pairs: int = 5,
    min_score_pct: float = 0.15,
    top_n_edges: int = 18,
    drop_saturated: bool = True,
):
    """Reduce a communication table to the links worth drawing.

    Returns (filtered sender-receiver table, list of ligand-receptor pairs
    kept), or (None, []) when nothing survives.
    """
    if comm_df is None or comm_df.empty:
        return None, []

    df = comm_df
    if drop_saturated and "attention" in df.columns:
        df = df[df["attention"] < 1.0 - SATURATION_TOL].copy()

    if df.empty:
        return None, []

    agg = (
        df[df["sender"] != df["receiver"]]
        .groupby(["sender", "receiver", "pair"])["score"]
        .sum()
        .reset_index()
    )
    if agg.empty:
        return None, []

    top_pairs = (
        agg.groupby("pair")["score"]
        .sum()
        .sort_values(ascending=False)
        .head(top_k_pairs)
        .index.tolist()
    )
    agg = agg[agg["pair"].isin(top_pairs)].copy()

    # One row per sender-receiver: its strongest pair.
    sr_agg = (
        agg.sort_values("score", ascending=False)
        .groupby(["sender", "receiver"])
        .first()
        .reset_index()
    )

    max_score = sr_agg["score"].max()
    sr_agg = sr_agg[sr_agg["score"] >= min_score_pct * max_score].copy()

    if top_n_edges and len(sr_agg) > top_n_edges:
        sr_agg = sr_agg.sort_values("score", ascending=False).head(top_n_edges)

    return sr_agg, top_pairs


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def _deg2rad(d):
    return np.radians(d)


def _arc_xy(r, deg):
    return r * np.cos(_deg2rad(deg)), r * np.sin(_deg2rad(deg))


def _draw_chord_panel(ax, sr_agg, pair_colors, niche_colors, title):
    """Draw one section's chord diagram onto an existing axis."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-6.8, 6.8)
    ax.set_ylim(-7.2, 7.2)
    ax.set_title(title, fontsize=20, pad=12, color="#1A1521", fontweight="bold")

    if sr_agg is None or sr_agg.empty:
        ax.text(0, 0, "No interactions\nabove threshold", ha="center",
                va="center", fontsize=13, color="#888888")
        return

    niches = sorted(set(sr_agg["sender"].tolist() + sr_agg["receiver"].tolist()))
    niches = [n for n in niches if "unlabeled" not in str(n).lower()]
    if not niches:
        return

    # Arc width is proportional to a niche's total in + out score.
    niche_score = {}
    for n in niches:
        out = float(sr_agg[sr_agg["sender"] == n]["score"].sum())
        inc = float(sr_agg[sr_agg["receiver"] == n]["score"].sum())
        niche_score[n] = out + inc + 1e-6

    total = sum(niche_score.values())
    gap_deg = 2.5
    usable = 360.0 - gap_deg * len(niches)

    angles, cursor = {}, 0.0
    for n in niches:
        span = (niche_score[n] / total) * usable
        angles[n] = (cursor, cursor + span, cursor + span / 2)
        cursor += span + gap_deg

    r_outer, r_inner, r_label = 4.5, 3.9, 5.25

    for n in niches:
        start, end, mid = angles[n]
        ax.add_patch(Wedge(
            (0, 0), r_outer, start, end, width=r_outer - r_inner,
            facecolor=niche_colors.get(n, "#D1C4E9"), linewidth=1.2,
            edgecolor="#4A4452", zorder=3,
        ))

        lx, ly = _arc_xy(r_label, mid)
        rot = mid % 360
        if 90 < rot < 270:
            rot += 180
        label = (str(n)[:14] + "\u2026") if len(str(n)) > 14 else str(n)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=11,
                fontweight="bold", color="#1A1521", rotation=rot % 360,
                rotation_mode="anchor",
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])

    sr_sorted = sr_agg.sort_values("score", ascending=True)
    lo = float(sr_sorted["score"].min())
    hi = float(sr_sorted["score"].max())
    span = hi - lo + 1e-8

    for _, row in sr_sorted.iterrows():
        snd, rcv = str(row["sender"]), str(row["receiver"])
        if snd not in angles or rcv not in angles:
            continue

        score = float(row["score"])
        p1 = np.array(_arc_xy(r_inner, angles[snd][2]))
        p2 = np.array(_arc_xy(r_inner, angles[rcv][2]))

        alpha = 0.55 + 0.40 * (score - lo) / span
        lw = 1.2 + 5.5 * (score - lo) / span

        ax.add_patch(mpatch.FancyArrowPatch(
            posA=tuple(p1), posB=tuple(p2),
            connectionstyle="arc3,rad=0.0",
            arrowstyle=f"->,head_width={lw*0.040:.3f},"
                       f"head_length={lw*0.030:.3f}",
            lw=lw, color=pair_colors.get(str(row["pair"]), "#888888"),
            alpha=alpha, zorder=2,
        ))

    ax.text(4.9, -5.5, "Comm.\nstrength", ha="center", fontsize=11,
            color="#1A1521", fontweight="bold")
    for idx, val in enumerate([lo, hi]):
        lw_ex = 1.2 + 5.5 * (val - lo) / span
        y = -6.1 - idx * 0.5
        ax.plot([4.4, 5.5], [y, y], lw=lw_ex, color="#332C3B",
                solid_capstyle="round")
        ax.text(5.6, y, f"{val:.1f}", va="center", fontsize=10,
                color="#1A1521")


def select_significant(comm_dfs, fdr_max=0.05, min_enrichment=2.0, shared_pairs=8):
    """Spatial tables only: keep links with fdr < fdr_max and enrichment >=
    min_enrichment, then restrict every section to the same shared_pairs
    ligand-receptor pairs (largest total kept score over all sections), so
    the panels show one common set of pairs."""
    kept = []
    for df in comm_dfs:
        if df is None or df.empty or "fdr" not in df.columns:
            kept.append(df)
            continue
        kept.append(df[(df["fdr"] < fdr_max) & (df["enrichment"] >= min_enrichment)])
    pooled = pd.concat([k for k in kept if k is not None and not k.empty])
    if shared_pairs and not pooled.empty:
        top = (pooled.groupby("pair")["score"].sum()
               .sort_values(ascending=False).head(shared_pairs).index)
        kept = [k[k["pair"].isin(top)] if k is not None else None for k in kept]
    return kept


def plot_ccc_chord(
    comm_dfs,
    section_labels,
    out_path: str,
    top_k_pairs: int = 8,
    min_score_pct: float = 0.0,
    top_n_edges: int = 20,
    drop_saturated: bool = True,
    panel_size=(6.0, 6.5),
    formats=("png", "svg"),
    dpi: int = 300,
    fdr_max: float = 0.05,
    min_enrichment: float = 2.0,
):
    """One figure with a chord panel per section and a shared legend.

    Ligand-receptor pairs and niches keep the same colour across panels,
    so panels can be compared directly. For spatial tables (with 'fdr' and
    'enrichment' columns) only significant, enriched links are drawn, and
    all sections share the same top_k_pairs pairs; the statistical filter
    replaces the display threshold, so min_score_pct defaults to 0.
    """
    if any(df is not None and not df.empty and "fdr" in df.columns for df in comm_dfs):
        comm_dfs = select_significant(comm_dfs, fdr_max=fdr_max,
                                      min_enrichment=min_enrichment,
                                      shared_pairs=top_k_pairs)
    filtered, kept_pairs, all_niches = [], set(), set()

    for df in comm_dfs:
        sr_agg, pairs = filter_for_chord(
            df,
            top_k_pairs=top_k_pairs,
            min_score_pct=min_score_pct,
            top_n_edges=top_n_edges,
            drop_saturated=drop_saturated,
        )
        filtered.append(sr_agg)
        if sr_agg is not None and not sr_agg.empty:
            kept_pairs.update(pairs)
            all_niches.update(sr_agg["sender"].tolist())
            all_niches.update(sr_agg["receiver"].tolist())

    if not kept_pairs:
        print("  No interactions survived filtering — no chord diagram drawn")
        return None

    kept_pairs = sorted(kept_pairs)
    all_niches = sorted(
        n for n in all_niches if "unlabeled" not in str(n).lower()
    )

    pair_colors = {
        p: CHORD_COLORS[i % len(CHORD_COLORS)] for i, p in enumerate(kept_pairs)
    }
    niche_colors = {
        n: NICHE_COLORS[i % len(NICHE_COLORS)] for i, n in enumerate(all_niches)
    }

    n_panels = len(filtered)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(panel_size[0] * n_panels, panel_size[1]),
        facecolor="white",
    )
    if n_panels == 1:
        axes = [axes]

    for ax, sr_agg, label in zip(axes, filtered, section_labels):
        _draw_chord_panel(ax, sr_agg, pair_colors, niche_colors, str(label))

    fig.legend(
        handles=[mpatch.Patch(color=pair_colors[p], label=p, alpha=0.9)
                 for p in kept_pairs],
        loc="center left", bbox_to_anchor=(0.90, 0.5), fontsize=12,
        title="Ligand\u2013receptor pairs", title_fontsize=13,
        frameon=True, framealpha=0.95, edgecolor="#BCB8C4", labelspacing=0.5,
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    written = []
    for fmt in formats:
        path = f"{out_path}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        written.append(path)
    plt.close(fig)

    drawn = sum(0 if f is None else len(f) for f in filtered)
    print(f"  Chord diagram: {drawn} links, {len(kept_pairs)} pairs "
          f"-> {out_path}")
    return written
