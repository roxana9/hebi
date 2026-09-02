#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read both Xenium breast cancer replicates from raw 10x output, harmonise
annotations, plot both ground-truth maps in one figure, and save h5ad files.

Pipeline per replicate:
  1. load expression matrix (cell_feature_matrix.h5)
  2. join spatial coords by POSITION (matrix and cells.parquet share cell order
     but use different id schemes: matrix=barcode strings, cells=integers)
  3. map annotation from the supervised Excel sheet (Barcode -> Cluster)
  4. merge Macrophages_1 + Macrophages_2 -> Macrophages
  5. drop ambiguous hybrid classes (Stromal_&_T_Cell_Hybrid, T_Cell_&_Tumor_Hybrid)
  6. (rep2 only) swap DCIS_1 <-> DCIS_2 to match rep1 biological correspondence
  7. filter cells (min_counts=40), drop "Unlabeled"
Then: plot both replicates side by side in one figure (shared palette),
      print unique labels per rep, and save each as h5ad.
"""

import os
import glob
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REP1_H5 = "/DATA1/Roxana/Data/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs/outs/cell_feature_matrix.h5"
REP2_H5 = "/DATA1/Roxana/Data/xenium_rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_outs/outs/cell_feature_matrix.h5"
ANNOT_XLSX = "/DATA1/Roxana/Data/xenium_rep2/Cell_Barcode_Type_Matrices.xlsx"

OUT_DIR = "/DATA1/Roxana/Data/Xenium_bc"      # where to save h5ad + figure
MIN_COUNTS = 40
#DROP_LABELS = ["Stromal_&_T_Cell_Hybrid", "T_Cell_&_Tumor_Hybrid"]


def find_cells_file(h5_path):
    outs = os.path.dirname(h5_path)
    for name in ["cells.parquet", "cells.csv.gz", "cells.csv"]:
        p = os.path.join(outs, name)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(outs, "*cells*"))
    return hits[0] if hits else None


def load_replicate(h5_path, annot_sheet, swap_dcis=False):
    print("=" * 60, f"\nLoading {annot_sheet}")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)

    # ---- coordinates: join by position (same cell order) ----
    cells_file = find_cells_file(h5_path)
    cells = pd.read_parquet(cells_file) if cells_file.endswith("parquet") else pd.read_csv(cells_file)
    assert len(cells) == adata.n_obs, \
        f"length mismatch cells={len(cells)} vs matrix={adata.n_obs}"
    adata.obs["x_centroid"] = cells["x_centroid"].to_numpy()
    adata.obs["y_centroid"] = cells["y_centroid"].to_numpy()
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy()

    # ---- annotation ----
    annot = pd.read_excel(ANNOT_XLSX, sheet_name=annot_sheet)
    type_col = next((c for c in annot.columns
                     if annot[c].astype(str).str.contains(r"[A-Za-z]").mean() > 0.5),
                    annot.columns[-1])
    cand = [c for c in annot.columns if c != type_col]
    barcode_col = next((c for c in cand
                        if any(k in str(c).lower() for k in ["barcode", "cell", "id"])),
                       cand[0])
    print(f"  barcode_col='{barcode_col}', type_col='{type_col}'")
    annot[barcode_col] = annot[barcode_col].astype(str)
    mapping = annot.set_index(barcode_col)[type_col]

    ann_obs = adata.obs_names.map(mapping)
    ann_cid = pd.Series(cells["cell_id"].astype(str).to_numpy()).map(mapping).to_numpy()
    if pd.notna(ann_cid).sum() >= ann_obs.notna().sum():
        adata.obs["annotation"] = ann_cid
        print(f"  annotation mapped via cell_id ({pd.notna(ann_cid).sum()})")
    else:
        adata.obs["annotation"] = ann_obs.values
        print(f"  annotation mapped via obs_names ({ann_obs.notna().sum()})")
    adata.obs["annotation"] = adata.obs["annotation"].astype("object")

    # ---- merge Macrophages_1 + Macrophages_2 -> Macrophages ----
    # adata.obs["annotation"] = adata.obs["annotation"].replace(
    #     {"Macrophages_1": "Macrophages", "Macrophages_2": "Macrophages"})

    # ---- drop ambiguous hybrid classes ----
    before = adata.n_obs
   # adata = adata[~adata.obs["annotation"].astype(str).isin(DROP_LABELS)].copy()
    print(f"  dropped {before - adata.n_obs} hybrid cells")

    # ---- swap DCIS_1 <-> DCIS_2 for rep2 ----
    if swap_dcis:
        a = adata.obs["annotation"].astype(str)
        a = a.replace({"DCIS_1": "__tmp_dcis__"})
        a = a.replace({"DCIS_2": "DCIS_1"})
        a = a.replace({"__tmp_dcis__": "DCIS_2"})
        adata.obs["annotation"] = a.values
        print("  swapped DCIS_1 <-> DCIS_2")

    # ---- filter cells, drop Unlabeled ----
    #sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
    adata = adata[adata.obs["annotation"].notna()].copy()
    adata = adata[adata.obs["annotation"].astype(str) != "Unlabeled"].copy()

    adata.obs["annotation"] = adata.obs["annotation"].astype("category")
    print(f"  final: {adata.n_obs} cells, {adata.n_vars} genes, "
          f"{adata.obs['annotation'].nunique()} labels")
    return adata


def report_labels(adata, tag):
    vc = adata.obs["annotation"].astype(str).value_counts()
    print(f"\n{tag}: {len(vc)} unique labels")
    for lab, n in vc.items():
        print(f"   {lab:30s} {n}")
    return vc


def plot_one(ax, adata, title, color_of, point_size=2):
    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]
    labels = adata.obs["annotation"].astype(str).to_numpy()
    for t in sorted(set(labels)):
        m = labels == t
        ax.scatter(x[m], y[m], s=point_size, c=[color_of[t]], label=t, linewidths=0)
    ax.set_title(title, fontsize=18)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    rep1 = load_replicate(REP1_H5, "Xenium R1 Fig1-5 (supervised)", swap_dcis=False)
    rep2 = load_replicate(REP2_H5, "Xenium R2 Fig1-5 (supervised)", swap_dcis=True)

    vc1 = report_labels(rep1, "Replicate 1")
    vc2 = report_labels(rep2, "Replicate 2")

    # shared fixed palette keyed by cell-type name
    all_types = sorted(set(vc1.index) | set(vc2.index))
    cmap = plt.get_cmap("nipy_spectral", len(all_types))
    color_of = {t: cmap(i) for i, t in enumerate(all_types)}
    print(f"\nTotal unique labels across both reps: {len(all_types)}")
    print("only in rep1:", sorted(set(vc1.index) - set(vc2.index)))
    print("only in rep2:", sorted(set(vc2.index) - set(vc1.index)))

    # ---- one combined figure, shared legend ----
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    plot_one(axes[0], rep1, "Replicate 1 ", color_of)
    plot_one(axes[1], rep2, "Replicate 2 ", color_of)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                          markerfacecolor=color_of[t], markeredgecolor="none", label=t)
               for t in all_types]
    fig.legend(handles=handles, loc="center right", fontsize=14, frameon=False,
               bbox_to_anchor=(1.0, 0.5))
    fig.subplots_adjust(left=0.02, right=0.82, top=0.93, bottom=0.02, wspace=0.05)

    fig_png = os.path.join(OUT_DIR, "annotation_both_reps.png")
    #fig.savefig(fig_png, dpi=300, bbox_inches="tight")
   # fig.savefig(fig_png.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved figure: {fig_png}")

    # ---- save harmonised h5ad files ----
    out1 = os.path.join(OUT_DIR, "Xenium_breast_cancer_rep1_harmonised1.h5ad")
    out2 = os.path.join(OUT_DIR, "Xenium_breast_cancer_rep2_harmonised2.h5ad")
    rep1.write_h5ad(out1)
    rep2.write_h5ad(out2)
    print(f"saved h5ad: {out1}")
    print(f"saved h5ad: {out2}")