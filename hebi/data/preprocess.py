#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified preprocessing: AnnData -> .pt section ready for HEBI.

One path for every task. What differs between datasets is passed as
arguments rather than living in a separate copy of the script:

    section splitting   --section_col
    train/test split    --split_col     (niche prediction)
    supervised labels   --label_cols    (niche prediction)

The data status is always reported. log1p is applied whenever the matrix
is not already in log space; a second log1p is never applied, because
log-of-log values look plausible and nothing downstream can detect them.
Library-size normalisation is separate and opt-in via --normalise, since
whether it helps depends on the corpus the checkpoint was pretrained on.
The status and whatever was applied are written into the .pt meta.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from hebi.data.graph import build_spatial_graph
from hebi.data.coarsening import create_metis_partition
from hebi.data.graph_diagnostics import neighbour_diagnostics, suggest_k
from hebi.data.vocabulary import harmonise_adata

warnings.filterwarnings("ignore")

# k=30 is the default because it gave the best held-out ARI on every
# section HEBI was developed on, across four platforms. It is not
# universal: what matters is the physical reach of the neighbourhood, so
# each section reports its reach in cell diameters and warns when the
# chosen k falls outside the usual range. See hebi.data.graph_diagnostics.
DEFAULT_K = 30
DEFAULT_METIS_CLUSTERS = 1024
DEFAULT_SEED = 42

# Neighbourhood reach considered local, in cell diameters.
TARGET_REACH = (2.0, 3.0)


# ======================================================================
# Normalisation detection
# ======================================================================
def detect_normalisation(adata, n_sample: int = 5000, seed: int = 0):
    """Infer what has already been done to adata.X.

    Returns a dict with:
        state       one of raw_counts, library_normalised, log_normalised,
                    log_transformed, scaled, unknown
        recommend   {"normalize": bool, "log1p": bool}
        detail      one line explaining the verdict
        stats       the numbers the verdict rests on

    The decisive test for the standard scanpy recipe is that
    expm1(X).sum(axis=1) is near-constant: normalize_total fixes row sums,
    and log1p is invertible, so undoing it recovers them.
    """
    rng = np.random.default_rng(seed)
    n = adata.n_obs
    idx = rng.choice(n, min(n_sample, n), replace=False)

    X = adata[idx].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = np.asarray(X, dtype=np.float64)

    finite = X[np.isfinite(X)]
    if finite.size == 0:
        return {"state": "unknown", "recommend": {"normalize": False, "log1p": False},
                "detail": "matrix is empty or all non-finite", "stats": {}}

    x_min = float(finite.min())
    x_max = float(finite.max())
    integral = bool(np.allclose(finite, np.round(finite), atol=1e-6))

    row_sums = X.sum(axis=1)
    row_cv = float(row_sums.std() / (abs(row_sums.mean()) + 1e-12))

    with np.errstate(over="ignore"):
        expm1_sums = np.expm1(np.clip(X, 0, 50)).sum(axis=1)
    expm1_cv = float(expm1_sums.std() / (abs(expm1_sums.mean()) + 1e-12))

    stats = {
        "min": round(x_min, 4), "max": round(x_max, 4),
        "all_integers": integral,
        "row_sum_cv": round(row_cv, 5),
        "expm1_row_sum_cv": round(expm1_cv, 5),
        "mean_row_sum": round(float(row_sums.mean()), 2),
    }

    # Negative values mean something centred the data; normalising or
    # logging on top of that is meaningless.
    if x_min < -1e-6:
        return {
            "state": "scaled",
            "recommend": {"normalize": False, "log1p": False},
            "detail": (f"contains negative values (min {x_min:.3f}) — looks "
                       f"z-scored or otherwise centred"),
            "stats": stats,
        }

    if integral and x_max > 20:
        return {
            "state": "raw_counts",
            "recommend": {"normalize": True, "log1p": True},
            "detail": f"all values integral, max {x_max:.0f} — raw counts",
            "stats": stats,
        }

    # normalize_total then log1p: undoing the log recovers constant sums.
    if expm1_cv < 0.01 and not integral:
        return {
            "state": "log_normalised",
            "recommend": {"normalize": False, "log1p": False},
            "detail": (f"expm1 row sums are constant (cv {expm1_cv:.4f}, "
                       f"target ~{np.expm1(np.clip(X,0,50)).sum(axis=1).mean():.0f}) "
                       f"— already normalize_total + log1p"),
            "stats": stats,
        }

    # normalize_total without log1p leaves the row sums themselves constant.
    if row_cv < 0.01:
        return {
            "state": "library_normalised",
            "recommend": {"normalize": False, "log1p": True},
            "detail": (f"row sums are constant (cv {row_cv:.4f}, target "
                       f"~{row_sums.mean():.0f}) — normalised but not logged"),
            "stats": stats,
        }

    if x_max < 20 and not integral:
        return {
            "state": "log_transformed",
            "recommend": {"normalize": False, "log1p": False},
            "detail": (f"max {x_max:.2f} with non-integer values — log-transformed, "
                       f"but row sums vary so the normalisation target is unclear"),
            "stats": stats,
        }

    if integral:
        return {
            "state": "raw_counts",
            "recommend": {"normalize": True, "log1p": True},
            "detail": f"all values integral, max {x_max:.0f} — raw counts",
            "stats": stats,
        }

    return {
        "state": "unknown",
        "recommend": {"normalize": False, "log1p": False},
        "detail": (f"could not classify (min {x_min:.3f}, max {x_max:.3f}, "
                   f"row-sum cv {row_cv:.3f}) — inspect before trusting"),
        "stats": stats,
    }


def apply_normalisation(adata, normalise: bool = False,
                        target_sum: float = 1e4, name: str = "section"):
    """Log-transform by default; library-size normalise only if asked.

    log1p is applied whenever the matrix is not already in log space,
    because the input token is a linear contraction of expression against
    the gene embeddings and raw counts make that token scale with library
    size. A second log1p is never applied: log-of-log values look
    plausible and are wrong, and nothing downstream can detect it.

    normalize_total is separate and opt-in, because whether depth
    correction helps depends on the corpus the checkpoint was pretrained
    on. It is only meaningful before log1p, so it cannot be applied to a
    matrix that has already been logged.

    The detected state and whatever was applied are printed and stored in
    the .pt meta.
    """
    verdict = detect_normalisation(adata)
    state = verdict["state"]
    verdict["applied"] = []

    print(f"  [{name}] data status: {state}")
    print(f"           {verdict['detail']}")

    if state == "scaled":
        print("           SKIPPING both: values are centred, so neither "
              "normalize_total nor log1p is meaningful here.")
        print("           Re-export from raw counts or from layers['counts'].")
        return adata, verdict

    if state == "log_normalised":
        print("           SKIPPING both: data is already library-normalised "
              "and log-transformed.")
        return adata, verdict

    if state == "log_transformed":
        print("           SKIPPING log1p: data is already log-transformed.")
        if normalise:
            print("           --normalise ignored: normalize_total only makes "
                  "sense before log1p, and this matrix is already logged.")
        return adata, verdict

    if state == "library_normalised":
        sc.pp.log1p(adata)
        verdict["applied"].append("log1p")
        print("           data is already library-normalised — applying log1p.")
        return adata, verdict

    if state == "raw_counts":
        if normalise:
            sc.pp.normalize_total(adata, target_sum=target_sum)
            verdict["applied"].append("normalize_total")
        sc.pp.log1p(adata)
        verdict["applied"].append("log1p")

        print(f"           applied: {' + '.join(verdict['applied'])}")
        if not normalise:
            print("           library size NOT corrected. Pass --normalise to "
                  "apply normalize_total before log1p.")
        return adata, verdict

    print("           SKIPPING both: could not classify this matrix. Inspect "
          "it before using the output.")
    return adata, verdict


# ======================================================================
# Metadata
# ======================================================================
HUMAN_HINTS = ["human", "hcolon", "hlung", "hpancreas", "hkidney", "htonsil",
               "hskin", "hbonemarrow", "hbone", "hheart", "hlymphnode", "hliver"]
MOUSE_HINTS = ["mouse", "mfemur", "mbrain", "tgcrnd8", "pup", "mouse_colon"]

TISSUE_HINTS = {
    "brain": "Brain", "cortex": "Brain", "pfc": "Brain", "glioblastoma": "Brain",
    "liver": "Liver", "lung": "Lung", "bronchial": "Lung",
    "colon": "Colon", "colorectal": "Colon", "gastrointestinal": "Colon", "crc": "Colon",
    "pancreas": "Pancreas", "skin": "Skin", "melanoma": "Skin", "kidney": "Kidney",
    "femur": "Bone/Marrow", "bonemarrow": "Bone/Marrow", "bone": "Bone/Marrow",
    "heart": "Heart", "uterus": "Uterus", "uterine": "Uterus",
    "prostate": "Prostate", "ovarian": "Ovary",
    "breast": "Breast", "idc": "Breast", "ilc": "Breast",
    "ductal_adenocarcinoma": "Breast",
    "tonsil": "Lymphoid/Immune", "lymphnode": "Lymphoid/Immune",
    "lymph_node": "Lymphoid/Immune",
}

DISEASED_HINTS = ["cancer", "tumor", "melanoma", "glioblastoma", "alzheimers",
                  "leukemia", "tgcrnd8", "idc", "ilc", "crc",
                  "ductal_adenocarcinoma", "hyperplasia", "reactive"]
NORMAL_HINTS = ["normal", "healthy", "nondiseased", "wildtype", "control", "pup"]


def parse_metadata(path_str: str):
    """Infer species, tissue, and disease state from a filename."""
    s = str(path_str).lower()
    meta = {"species": "Unknown", "tissue": "N/A", "disease": "N/A"}

    if any(p in s for p in HUMAN_HINTS):
        meta["species"] = "Human"
    elif any(p in s for p in MOUSE_HINTS):
        meta["species"] = "Mouse"

    for keyword, tissue in TISSUE_HINTS.items():
        if keyword in s:
            meta["tissue"] = tissue
            break

    if any(p in s for p in NORMAL_HINTS):
        meta["disease"] = "Normal"
    elif any(p in s for p in DISEASED_HINTS):
        meta["disease"] = "Diseased"
    elif meta["species"] != "Unknown":
        meta["disease"] = "Normal"

    return meta


def find_spatial_key(adata) -> str:
    candidates = ["spatial", "X_spatial", "spatial_coords", "coordinates", "coords"]
    for key in adata.obsm:
        for ref in candidates:
            if ref.lower() in key.lower():
                return key
    if {"center_x", "center_y"} <= set(adata.obs.columns):
        adata.obsm["spatial"] = adata.obs[["center_x", "center_y"]].values
        return "spatial"
    raise KeyError(f"No spatial coordinates found in obsm: {list(adata.obsm)}")


# ======================================================================
# Neighbourhood reach
# ======================================================================
def check_neighbourhood_reach(coords, k, name="section", n_sample=20000):
    """Report how far the chosen k reaches, and warn if it looks wrong.

    k is a count, but what matters is the physical extent of the
    neighbourhood it produces, and that depends on cell density. The same
    k spans 2.0 cell diameters on one platform and 3.2 on another. Two to
    three diameters covers a cell and its first two rings of contacts.

    Returns the reach in cell diameters, which is stored in the .pt meta
    so sections stay comparable across platforms and coordinate units.
    """
    rows, scale = neighbour_diagnostics(
        coords, k_values=sorted({3, 5, 10, 15, 20, 30, 40, 50, 75, k}),
        n_sample=n_sample,
    )
    reach = next(r["reach_in_cell_diameters"] for r in rows if r["k"] == k)
    lo, hi = TARGET_REACH

    print(f"  [{name}] k={k} spans {reach:.2f} cell diameters "
          f"(median nearest-neighbour distance {scale['nn1_median']})")

    if reach < 1.0:
        suggestion = suggest_k(rows)
        print(f"           WARNING: below one cell diameter — the "
              f"neighbourhood barely reaches")
        print(f"           adjacent cells, so spatial context is largely "
              f"unused. Consider k="
              f"{suggestion['k_min']}-{suggestion['k_max']}.")
    elif reach < lo:
        suggestion = suggest_k(rows)
        print(f"           note: below the usual {lo}-{hi} diameter range. "
              f"Consider k="
              f"{suggestion['k_min']}-{suggestion['k_max']}.")
    elif reach > 4.0:
        suggestion = suggest_k(rows)
        print(f"           note: {reach:.2f} diameters is broad — the "
              f"neighbourhood may span distinct")
        print(f"           tissue structures. Consider k="
              f"{suggestion['k_min']}-{suggestion['k_max']}.")

    return round(float(reach), 3)


# ======================================================================
# Core
# ======================================================================
def build_section_dict(adata, meta, name: str = "section", k: int = DEFAULT_K,
                       n_metis: int = DEFAULT_METIS_CLUSTERS,
                       seed: int = DEFAULT_SEED,
                       label_mappings=None, label_cols=None):
    """Turn one AnnData section into the dict saved as .pt."""
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = torch.tensor(X, dtype=torch.float32)

    coords_full = torch.tensor(
        np.asarray(adata.obsm[find_spatial_key(adata)]), dtype=torch.float32
    )
    coords = coords_full[:, :2] if coords_full.shape[1] > 2 else coords_full

    reach = check_neighbourhood_reach(coords.numpy(), k, name=name)

    edge_index, _, _ = build_spatial_graph(coords.numpy(), k=k)

    n_clusters = n_metis if adata.n_obs >= n_metis else max(1, adata.n_obs // 10)
    node2comm, super_sp = create_metis_partition(
        edge_index, X.shape[0], n_clusters, seed=seed
    )

    # The graph parameters travel with the data: a .pt built with k=30
    # is not interchangeable with one built at k=5.
    meta = dict(meta)
    meta.update({"k_neighbors": k, "n_metis_clusters": n_clusters,
                 "seed": seed, "reach_cell_diameters": reach})

    section = {
        "X": X.half(),
        "coords": coords,
        "edge_index": edge_index,
        "node2comm": node2comm,
        "super_sp": super_sp,
        "meta": meta,
        "gene_ids": torch.tensor(adata.var["gene_id"].values, dtype=torch.long),
    }

    # Always stored: GRN attention is indexed by gene position, so an edge
    # cannot be named without this list.
    names = (adata.var["gene_symbol"] if "gene_symbol" in adata.var.columns
             else adata.var.index)
    section["gene_names"] = list(names.astype(str))

    if label_mappings and label_cols:
        for col, (key, mapping) in zip(label_cols, label_mappings.items()):
            raw = adata.obs[col].astype(str).values
            section[f"y_{key}"] = torch.tensor(
                [mapping.get(v, 0) for v in raw], dtype=torch.long
            )
            section[f"num_{key}_types"] = len(mapping)

    return section


def preprocess_h5ad(h5ad_path, out_root, vocab_path, task="clustering",
                    k=DEFAULT_K, n_metis=DEFAULT_METIS_CLUSTERS,
                    seed=DEFAULT_SEED, normalise=False, target_sum=1e4,
                    min_counts=None, section_col=None, split_col=None,
                    label_cols=None, counts_layer=None,
                    species=None, overwrite=False):
    """Preprocess one .h5ad into one or more .pt sections."""
    h5ad_path = Path(h5ad_path)
    out_root = Path(out_root)
    stem = h5ad_path.stem

    # Nothing below this point is cheap: reading the .h5ad, harmonising
    # the vocabulary and measuring the graph all cost real time. When the
    # output already exists and is not being overwritten, return its path
    # without doing any of it, so re-running a notebook is instant.
    if not overwrite and task == "clustering" and section_col is None:
        existing = out_root / f"{stem}.pt"
        if existing.exists():
            print(f"\n{'=' * 60}\n{h5ad_path.name}: already processed, "
                  f"reusing {existing.name}")
            return [existing]

    print(f"\n{'=' * 60}\nLoading {h5ad_path.name}")
    adata = sc.read_h5ad(h5ad_path)

    if counts_layer:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"--counts_layer '{counts_layer}' is not in this file. "
                f"Available layers: {list(adata.layers)}"
            )
        adata.X = adata.layers[counts_layer].copy()
        print(f"  using adata.layers['{counts_layer}'] as X")
    elif adata.layers:
        # Worth saying: a raw-count layer alongside a processed X is the
        # usual export shape, and silently using the wrong one is easy.
        print(f"  note: this file has layers {list(adata.layers)}. X is used "
              f"as-is; pass --counts_layer to take one of them instead.")

    meta = parse_metadata(h5ad_path)
    print(f"  metadata: {meta}")

    if min_counts:
        before = adata.n_obs
        sc.pp.filter_cells(adata, min_counts=min_counts)
        print(f"  filter_cells(min_counts={min_counts}): "
              f"{before} -> {adata.n_obs} cells")

    # Unknown would take the homolog-mapping branch and query an invalid
    # species, so fall back to Human and say so.
    if species:
        meta["species"] = species
    if meta["species"] == "Unknown":
        print("  WARNING: species not inferable from the filename — assuming "
              "Human. Pass --species Mouse if that is wrong; non-human symbols "
              "are otherwise not mapped to their human homologs.")
        meta["species"] = "Human"

    adata = harmonise_adata(adata, str(vocab_path), species=meta["species"])
    adata, verdict = apply_normalisation(adata, normalise=normalise,
                                         target_sum=target_sum, name=stem)
    # Both states are recorded: "on_input" is how the .h5ad arrived,
    # "state" is what X actually holds in this .pt. They differ whenever
    # --normalise did something, and only the second describes the data.
    meta["normalisation_on_input"] = verdict["state"]
    meta["normalisation_applied"] = verdict.get("applied", [])
    applied = verdict.get("applied", [])
    if "normalize_total" in applied:
        meta["normalisation_state"] = "log_normalised"
    elif "log1p" in applied:
        meta["normalisation_state"] = (
            "log_normalised" if verdict["state"] == "library_normalised"
            else "log_transformed"
        )
    else:
        meta["normalisation_state"] = verdict["state"]

    mappings = None
    if task == "niche_prediction":
        if not label_cols:
            raise ValueError("--label_cols is required for niche_prediction")
        mappings = {}
        for col in label_cols:
            key = "class" if col == label_cols[0] else "niche"
            values = sorted(adata.obs[col].astype(str).unique())
            mappings[key] = {v: i for i, v in enumerate(values)}
        out_root.mkdir(parents=True, exist_ok=True)
        with open(out_root / "label_mappings.json", "w") as f:
            json.dump({f"{k}_mapping": v for k, v in mappings.items()}, f, indent=2)
        print("  label mappings: " +
              ", ".join(f"{k}={len(v)}" for k, v in mappings.items()))

    def save(sub, out_file, tag):
        if out_file.exists() and not overwrite:
            print(f"  exists, skipping: {out_file.name}")
            return
        out_file.parent.mkdir(parents=True, exist_ok=True)
        section = build_section_dict(
            sub, meta, name=out_file.stem, k=k, n_metis=n_metis, seed=seed,
            label_mappings=mappings, label_cols=label_cols,
        )
        torch.save(section, out_file)
        print(f"  saved {out_file.name}  ({section['X'].shape[0]} cells, "
              f"{section['edge_index'].shape[1]} edges, {tag})")

    written = []

    if task == "niche_prediction":
        if split_col is None or split_col not in adata.obs.columns:
            raise ValueError(
                f"--split_col must name a column in obs for niche_prediction. "
                f"Got '{split_col}'. Available: {list(adata.obs.columns)[:20]}"
            )
        for split_value in adata.obs[split_col].unique():
            sub = adata[adata.obs[split_col] == split_value].copy()
            folder = "train" if str(split_value).lower() == "train" else "test"
            out_file = out_root / folder / f"{stem}_{split_value}.pt"
            save(sub, out_file, f"split={split_value}")
            written.append(out_file)

    elif section_col:
        if section_col not in adata.obs.columns:
            raise ValueError(f"--section_col '{section_col}' not in obs")
        for section_value in sorted(adata.obs[section_col].dropna().unique()):
            sub = adata[adata.obs[section_col].astype(str)
                        == str(section_value)].copy()
            out_file = out_root / f"{stem}_section_{section_value}.pt"
            save(sub, out_file, f"section={section_value}")
            written.append(out_file)

    else:
        out_file = out_root / f"{stem}.pt"
        save(adata, out_file, "whole slide")
        written.append(out_file)

    return written


def build_parser():
    p = argparse.ArgumentParser(
        description="Preprocess spatial AnnData into HEBI .pt sections."
    )
    p.add_argument("--h5ad", type=str,
                   help="One .h5ad file.")
    p.add_argument("--h5ad_dir", type=str,
                   help="Directory of .h5ad files, processed in turn.")
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)

    p.add_argument("--task", choices=["clustering", "niche_prediction"],
                   default="clustering")

    p.add_argument("--normalise", action="store_true",
                   help="Also apply normalize_total before log1p, correcting "
                        "for library size. log1p alone is applied by default "
                        "whenever the matrix is not already logged.")
    p.add_argument("--target_sum", type=float, default=1e4,
                   help="Library-size target for normalize_total. Only used "
                        "when the input still needs normalising.")
    p.add_argument("--counts_layer", type=str, default=None,
                   help="Name of the layer holding raw counts, used as X "
                        "before anything else. Commonly 'counts' or 'raw'. "
                        "The available layers are printed if the name is wrong.")
    p.add_argument("--min_counts", type=int, default=None,
                   help="Drop cells below this total count. Off by default; "
                        "whatever you choose is baked into the .pt.")

    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help=f"Spatial kNN degree (default {DEFAULT_K}). Each "
                        f"section reports the physical reach this gives and "
                        f"suggests a range if it looks wrong. Recorded in "
                        f"the .pt meta.")
    p.add_argument("--n_metis", type=int, default=DEFAULT_METIS_CLUSTERS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--section_col", type=str, default=None,
                   help="obs column to split one file into several sections.")
    p.add_argument("--split_col", type=str, default=None,
                   help="obs column holding train/test, for niche_prediction.")
    p.add_argument("--label_cols", nargs="+", default=None,
                   help="Two obs columns for niche_prediction: cell type "
                        "first, niche second.")
    p.add_argument("--species", type=str, default=None,
                   choices=["Human", "Mouse"],
                   help="Override the species inferred from the filename. "
                        "Mouse symbols are mapped to their human homologs.")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if bool(args.h5ad) == bool(args.h5ad_dir):
        raise SystemExit("Give exactly one of --h5ad or --h5ad_dir")

    files = ([Path(args.h5ad)] if args.h5ad
             else sorted(Path(args.h5ad_dir).glob("*.h5ad")))
    if not files:
        raise SystemExit(f"No .h5ad files found in {args.h5ad_dir}")

    print(f"Preprocessing {len(files)} file(s) | task={args.task} | "
          f"k={args.k} | normalise={args.normalise}")

    written = []
    for path in files:
        try:
            written += preprocess_h5ad(
                path, args.out_root, args.vocab_path, task=args.task,
                k=args.k, n_metis=args.n_metis, seed=args.seed,
                normalise=args.normalise,
                target_sum=args.target_sum,
                min_counts=args.min_counts, section_col=args.section_col,
                split_col=args.split_col, label_cols=args.label_cols,
                counts_layer=args.counts_layer,
                species=args.species, overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"  FAILED {path.name}: {exc}")

    print(f"\nDone. {len(written)} section(s) written to {args.out_root}")
    return written


if __name__ == "__main__":
    main()
