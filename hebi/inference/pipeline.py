#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEBI inference pipeline: encode sections, assign prototypes and niches,
score against annotations.

Python API
----------
    from hebi.inference.pipeline import run_hebi

    sections, trainer, grn_attn = run_hebi(
        checkpoint_path="hebi_foundation.pt",
        data_dir="processed/",
        section_files=["section_1.pt", "section_2.pt"],
        h5ad_files=["section_1.h5ad", "section_2.h5ad"],
        vocab_path="default_gene_vocab.json",
        n_cell_clusters=18,
        n_composition_niches=12,
    )

Command line
------------
    python -m hebi.inference.pipeline --config configs/starmap.yaml
    python -m hebi.inference.pipeline --checkpoint_path ... --data_dir ...

Labels written to each section's .obs:
    prototypes           Level-2 attention argmax
    cell_clusters        clustering of the cell-level z0
    composition_niches   k-means on local prototype composition
"""

import argparse
import os
import random
import sys

import numpy as np
import scanpy as sc
import torch

from hebi.data.vocabulary import harmonise_adata
from hebi.grn.analysis import run_grn_analysis
from hebi.grn.ccc_chord import plot_ccc_chord, write_ccc_excel
from hebi.grn.build import build_hetero_grn
from hebi.inference.adapt import continual_adapt, create_random_checkpoint
from hebi.inference.cluster import (
    CLUSTER_METHODS,
    KEY_CELL_CLUSTERS,
    KEY_COMPOSITION_NICHES,
    KEY_PROTOTYPES,
    build_composition_vectors,
    cluster_embedding,
    fide_score,
    score_against_truth,
)
from hebi.inference.encode import encode_section, load_hebi

MODES = ("zero_shot", "continual")
INIT_MODES = ("pretrained", "random", "hybrid")
from hebi.plotting.spatial import (
    get_palette,
    plot_spatial_labels,
    plot_spatial_multi,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def load_sections(
    h5ad_files,
    vocab_path,
    preprocess: bool = True,
    filter_min_counts: int = None,
    drop_unlabelled: bool = False,
    point_size: float = None,
):
    """Read sections and project onto the global gene vocabulary.

    Cell filtering is OFF by default. The .pt sections were built with
    whatever filtering the preprocessing step applied, and their cell
    order is fixed; dropping cells here would silently desynchronise the
    embeddings from the AnnData. Normalisation and log1p do not change
    cell count and are applied when preprocess is True, because the GRN
    step reads expression from the AnnData.
    """
    sections = []
    for path in h5ad_files:
        print(f"Loading {path}")
        adata = sc.read_h5ad(path)
        n_start = adata.n_obs

        if filter_min_counts:
            sc.pp.filter_cells(adata, min_counts=filter_min_counts)

        if preprocess:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        adata = harmonise_adata(adata, vocab_path)

        if drop_unlabelled and "annotation" in adata.obs.columns:
            adata = adata[adata.obs["annotation"] != "Unlabeled"].copy()

        if adata.n_obs != n_start:
            print(f"  {n_start} -> {adata.n_obs} cells after filtering")
        print(f"  {adata.n_obs} cells, {adata.n_vars} genes")
        sections.append(adata)
    return sections


def align_sections(sections, data_dir, section_files, tolerance=1e-6):
    """Make each AnnData hold exactly the cells its .pt encodes, in order.

    The .pt files were written by preprocessing that filtered cells, and
    the thresholds it used are not recorded anywhere the pipeline can
    read. Reconstructing them by replaying filters is guesswork that
    silently produces a wrong answer when it is close but not exact.

    Spatial coordinates identify a cell uniquely, and both files carry
    them, so each .pt cell is matched to its AnnData row by coordinate and
    the AnnData is subset and reordered to match. This runs before any
    training, so a genuine mismatch fails in seconds rather than after a
    full run.
    """
    from scipy.spatial import cKDTree

    aligned, problems = [], []

    for adata, section_file in zip(sections, section_files):
        path = os.path.join(data_dir, section_file)
        pt = torch.load(path, map_location="cpu", weights_only=False)
        pt_coords = np.asarray(pt["coords"], dtype=float)
        n_pt = pt_coords.shape[0]

        ad_coords = np.asarray(adata.obsm["spatial"], dtype=float)

        # Already identical, in order: nothing to do.
        if n_pt == adata.n_obs and np.allclose(pt_coords, ad_coords,
                                               atol=tolerance):
            print(f"  {section_file}: {n_pt} cells, already aligned")
            aligned.append(adata)
            continue

        if adata.n_obs < n_pt:
            problems.append(
                f"  {section_file}: .pt has {n_pt} cells but AnnData has only "
                f"{adata.n_obs} — the AnnData is missing cells the model "
                f"encoded. Drop --filter_min_counts / --drop_unlabelled."
            )
            aligned.append(adata)
            continue

        distances, indices = cKDTree(ad_coords).query(pt_coords, k=1)
        scale = max(np.abs(ad_coords).max(), 1.0)
        unmatched = int((distances > tolerance * scale).sum())

        if unmatched:
            problems.append(
                f"  {section_file}: {unmatched}/{n_pt} .pt cells have no "
                f"matching coordinate in the AnnData (max gap "
                f"{distances.max():.4g})"
            )
            aligned.append(adata)
            continue

        if len(np.unique(indices)) != n_pt:
            problems.append(
                f"  {section_file}: coordinates are not unique — "
                f"{n_pt - len(np.unique(indices))} .pt cells matched a row "
                f"already taken"
            )
            aligned.append(adata)
            continue

        subset = adata[indices].copy()
        print(f"  {section_file}: {adata.n_obs} -> {subset.n_obs} cells "
              f"matched to the .pt by coordinate")
        aligned.append(subset)

    if problems:
        raise SystemExit(
            "Could not align .pt sections to their .h5ad files:\n"
            + "\n".join(problems)
            + "\n\nThe .pt files fix which cells exist and in what order, "
              "so every .pt cell must be findable in the AnnData."
        )

    return aligned


def run_hebi(
    checkpoint_path: str,
    data_dir: str,
    section_files,
    h5ad_files,
    vocab_path: str,
    mode: str = "zero_shot",
    init: str = "pretrained",
    n_cell_clusters: int = 8,
    n_composition_niches: int = None,
    composition_neighbors: int = 200,
    cell_type_key: str = None,
    niche_key: str = None,
    cluster_method: str = "kmeans",
    resolution: float = 1.0,
    grn: bool = False,
    species: str = "Human",
    per_section_grn: bool = False,
    grn_cluster_key: str = None,
    grn_ccc: bool = False,
    section_labels=None,
    ccc_top_k_pairs: int = 5,
    ccc_min_score_pct: float = 0.15,
    ccc_top_n_edges: int = 18,
    ccc_keep_saturated: bool = False,
    lr: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./hebi_results",
    output_checkpoint: str = None,
    device: str = None,
    seed: int = 0,
    preprocess: bool = True,
    save_plots: bool = True,
    filter_min_counts: int = None,
    drop_unlabelled: bool = False,
    point_size: float = None,
):
    """Encode sections and assign prototypes, cell clusters, and
    composition niches.

    Returns (sections, trainer, grn_attention_per_section).
    """
    if cluster_method not in CLUSTER_METHODS:
        raise ValueError(f"cluster_method must be one of {CLUSTER_METHODS}")

    if mode not in MODES:
        raise ValueError(
            f"mode must be one of {MODES}, got '{mode}'. "
            "Values from a config file are not checked by argparse, so a "
            "typo there reaches this point unvalidated."
        )

    if init not in INIT_MODES:
        raise ValueError(f"init must be one of {INIT_MODES}, got '{init}'")

    if grn and mode != "continual":
        raise SystemExit(
            f"--grn requires --mode continual (got '{mode}').\n\n"
            "The GRN refinement layer is only trained during continual "
            "adaptation. In zero-shot mode it is freshly initialised and "
            "never optimised, so its attention carries no signal and every "
            "gene-level result derived from it — hub genes, edge salience, "
            "niche activity, cell-cell communication — would be meaningless.\n\n"
            "Set mode to continual, or drop --grn."
        )

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    if section_labels is None:
        section_labels = [os.path.splitext(f)[0] for f in section_files]
    elif len(section_labels) != len(section_files):
        raise ValueError(
            f"section_labels has {len(section_labels)} entries but "
            f"section_files has {len(section_files)}"
        )

    print(f"HEBI pipeline | mode={mode} | init={init} | cluster={cluster_method}")

    sections = load_sections(
        h5ad_files,
        vocab_path,
        preprocess=preprocess,
        filter_min_counts=filter_min_counts,
        drop_unlabelled=drop_unlabelled,
    )
    sections = align_sections(sections, data_dir, section_files)

    # --- gene-interaction graphs -------------------------------------
    # gene_names travels with its graph: attention is indexed by gene
    # position, so the two must never be separated.
    gene_graph, gene_names = None, None
    gene_graph_list, gene_names_list = None, None

    if grn:
        if per_section_grn:
            gene_graph_list, gene_names_list = [], []
            for adata in sections:
                gg, gn, _ = build_hetero_grn(
                    adata=adata,
                    species=species,
                    spearman_threshold=0.4,
                    add_reverse_for_curated=True,
                )
                gene_graph_list.append(gg)
                gene_names_list.append(gn)
        else:
            print("Building gene-interaction graph")
            gene_graph, gene_names, _ = build_hetero_grn(
                adata=sections[0],
                species=species,
                spearman_threshold=0.4,
                add_reverse_for_curated=True,
            )

    # --- checkpoint --------------------------------------------------
    if init in ("pretrained", "hybrid"):
        if checkpoint_path is None:
            raise ValueError(f"checkpoint_path is required when init='{init}'")
        current_checkpoint = checkpoint_path
    else:
        current_checkpoint = create_random_checkpoint(
            data_dir=data_dir,
            vocab_path=vocab_path,
            output_checkpoint="hebi_random_init_temp.pt",
            device=device,
            seed=seed,
        )
        print("Starting from random initialisation")

    # --- optional continual adaptation -------------------------------
    if mode == "continual":
        adapted = output_checkpoint or os.path.join(output_dir, "hebi_continual.pt")
        continual_adapt(
            checkpoint_path=current_checkpoint,
            data_dir=data_dir,
            vocab_path=vocab_path,
            device=device,
            lr=lr,
            num_epochs=num_epochs,
            output_checkpoint=adapted,
            gene_graph=gene_graph,
            gene_graph_list=gene_graph_list,
            init=init,
        )
        current_checkpoint = adapted

    # --- encode ------------------------------------------------------
    trainer = load_hebi(
        checkpoint_path=current_checkpoint,
        data_dir=data_dir,
        vocab_path=vocab_path,
        device=device,
        init=init,
        use_grn=grn,
    )

    all_z0, all_z2, all_protos, all_grn_attn, cell_counts = [], [], [], [], []

    for i, section_file in enumerate(section_files):
        graph_i = gene_graph_list[i] if gene_graph_list is not None else gene_graph
        out = encode_section(
            trainer=trainer,
            section_path=os.path.join(data_dir, section_file),
            gene_graph=graph_i,
        )
        all_z0.append(out["z0"].cpu().numpy())
        all_z2.append(out["z2"].cpu().numpy())
        all_protos.append(out["prototypes"].numpy())
        all_grn_attn.append(out["grn_attention"])
        cell_counts.append(all_z0[-1].shape[0])
        print(f"  Encoded {section_file}: {cell_counts[-1]} cells")

    joint_z0 = np.concatenate(all_z0, axis=0)
    joint_protos = np.concatenate(all_protos, axis=0)

    n_prototypes = int(joint_protos.max()) + 1
    n_used = len(np.unique(joint_protos))
    print(f"\nTotal cells: {joint_z0.shape[0]}")
    print(f"Prototypes in use: {n_used}/{n_prototypes}")

    # --- cell clusters, fitted jointly across sections ---------------
    print(f"Clustering cell-level z0 with {cluster_method} (k={n_cell_clusters})")
    joint_cell_labels = cluster_embedding(
        joint_z0,
        n_clusters=n_cell_clusters,
        method=cluster_method,
        random_state=seed,
        resolution=resolution,
    )

    # --- composition niches, one joint fit ---------------------------
    joint_comp_niches = None
    if n_composition_niches is not None:
        comp_vectors = [
            build_composition_vectors(
                all_protos[i],
                sections[i].obsm["spatial"],
                k=composition_neighbors,
                n_prototypes=n_prototypes,
            )
            for i in range(len(sections))
        ]
        joint_comp_niches = cluster_embedding(
            np.concatenate(comp_vectors, axis=0),
            n_clusters=n_composition_niches,
            method="kmeans",
            random_state=seed,
        )
        print(f"Composition niches: {len(np.unique(joint_comp_niches))}")

    # Palettes are built over the joint labels so a colour means the same
    # thing in every section's figure.
    palette_clusters = get_palette(joint_cell_labels)
    palette_comp = (
        get_palette(joint_comp_niches) if joint_comp_niches is not None else None
    )

    section_names = [os.path.splitext(f)[0] for f in section_files]
    coords_list = [a.obsm["spatial"] for a in sections]
    offsets, _cursor = [], 0
    for count in cell_counts:
        offsets.append(slice(_cursor, _cursor + count))
        _cursor += count

    comm_dfs = []

    # --- write labels back, plot, score ------------------------------
    start = 0
    for i, (adata, count) in enumerate(zip(sections, cell_counts)):
        sl = slice(start, start + count)

        adata.obsm["z0"] = all_z0[i]
        adata.obsm["z2"] = all_z2[i]
        _set_cat(adata, KEY_CELL_CLUSTERS, joint_cell_labels[sl])
        _set_cat(adata, KEY_PROTOTYPES, all_protos[i])

        if joint_comp_niches is not None:
            _set_cat(adata, KEY_COMPOSITION_NICHES, joint_comp_niches[sl])

        section_name = os.path.splitext(section_files[i])[0]
        section_dir = os.path.join(output_dir, f"{section_name}_plot")

        if save_plots:
            os.makedirs(section_dir, exist_ok=True)
            coords = adata.obsm["spatial"]

            plot_spatial_labels(
                coords,
                joint_cell_labels[sl],
                title=f"{section_name}\ncell clusters ({cluster_method}, k={n_cell_clusters})",
                out_path=os.path.join(section_dir, "cell_clusters_z0"),
                palette=palette_clusters,
                legend_title="Cluster",
                point_size=point_size,
            )

            if joint_comp_niches is not None:
                plot_spatial_labels(
                    coords,
                    joint_comp_niches[sl],
                    title=f"{section_name}\ncomposition niches (k={n_composition_niches})",
                    out_path=os.path.join(section_dir, "composition_niches"),
                    palette=palette_comp,
                    legend_title="Niche",
                    point_size=point_size,
                )

            print(f"  Figures -> {section_dir}")

        if grn:
            grn_dir = os.path.join(section_dir, "grn")
            names_i = (
                gene_names_list[i] if gene_names_list is not None else gene_names
            )
            graph_i = (
                gene_graph_list[i] if gene_graph_list is not None else gene_graph
            )
            key = grn_cluster_key or _default_grn_key(adata)
            print(f"\nGRN analysis for {section_name} (grouped by '{key}')")
            grn_out = run_grn_analysis(
                adata=adata,
                gene_graph=graph_i,
                gene_names=names_i,
                grn_attention=all_grn_attn[i],
                out_dir=grn_dir,
                cluster_key=key,
                prefix=section_name,
                run_ccc=grn_ccc,
            )
            comm_dfs.append(grn_out.get("communication"))

        edge_index = torch.load(
            os.path.join(data_dir, section_files[i]), map_location="cpu"
        )["edge_index"]

        _report(adata, section_files[i], i, edge_index, cell_type_key, niche_key)
        start += count

    # --- combined figures, all sections side by side -----------------
    if save_plots:
        print("\nCombined figures:")
        _plot_combined(
            coords_list=coords_list,
            joint_labels=joint_cell_labels,
            offsets=offsets,
            sections=sections,
            section_names=section_names,
            truth_key=cell_type_key,
            pred_key=KEY_CELL_CLUSTERS,
            palette=palette_clusters,
            suptitle=f"HEBI joint cell clusters on z0 "
                     f"({cluster_method}, k={n_cell_clusters})",
            legend_title="Cluster",
            out_path=os.path.join(output_dir, "joint_cell_clusters_z0"),
            point_size=point_size,
        )

        if joint_comp_niches is not None:
            _plot_combined(
                coords_list=coords_list,
                joint_labels=joint_comp_niches,
                offsets=offsets,
                sections=sections,
                section_names=section_names,
                truth_key=niche_key,
                pred_key=KEY_COMPOSITION_NICHES,
                palette=palette_comp,
                suptitle=f"HEBI composition niches "
                         f"(k={n_composition_niches})",
                legend_title="Niche",
                out_path=os.path.join(output_dir, "joint_composition_niches"),
                point_size=point_size,
            )

    if grn and grn_ccc and comm_dfs:
        print("\nCell-cell communication:")
        write_ccc_excel(
            comm_dfs=comm_dfs,
            section_labels=section_labels,
            out_path=os.path.join(output_dir, "CCC_all_pairs.xlsx"),
        )
        if save_plots:
            plot_ccc_chord(
                comm_dfs=comm_dfs,
                section_labels=section_labels,
                out_path=os.path.join(output_dir, "ccc_chord"),
                top_k_pairs=ccc_top_k_pairs,
                min_score_pct=ccc_min_score_pct,
                top_n_edges=ccc_top_n_edges,
                drop_saturated=not ccc_keep_saturated,
            )

    return sections, trainer, all_grn_attn


def _plot_combined(
    coords_list, joint_labels, offsets, sections, section_names,
    truth_key, pred_key, palette, suptitle, legend_title, out_path,
    point_size=None,
):
    """One side-by-side figure across sections, with per-panel ARI/NMI."""
    subtitles = []
    for adata in sections:
        ari, nmi = score_against_truth(adata, truth_key, pred_key)
        subtitles.append(
            "" if np.isnan(ari) else f"ARI={ari:.3f}  NMI={nmi:.3f}"
        )

    plot_spatial_multi(
        coords_list=coords_list,
        labels_list=[joint_labels[s] for s in offsets],
        section_names=section_names,
        out_path=out_path,
        suptitle=suptitle,
        subtitles=subtitles,
        palette=palette,
        legend_title=legend_title,
        point_size=point_size,
    )


def _default_grn_key(adata):
    """Group GRN activity by niche, per the Methods. Composition niches are
    the niche definition; fall back to cell clusters when they were not
    computed."""
    if KEY_COMPOSITION_NICHES in adata.obs.columns:
        return KEY_COMPOSITION_NICHES
    return KEY_CELL_CLUSTERS


def _set_cat(adata, key, values):
    adata.obs[key] = np.asarray(values).astype(str)
    adata.obs[key] = adata.obs[key].astype("category")


def _report(adata, name, index, edge_index, cell_type_key, niche_key):
    print(f"\nSection {index + 1} [{name}]")

    pairs = [
        (KEY_CELL_CLUSTERS, cell_type_key),
        (KEY_PROTOTYPES, niche_key),
        (KEY_COMPOSITION_NICHES, niche_key),
    ]

    for pred_key, truth_key in pairs:
        if pred_key not in adata.obs.columns:
            continue
        codes = adata.obs[pred_key].cat.codes.to_numpy()
        fide = fide_score(codes, edge_index)
        ari, nmi = score_against_truth(adata, truth_key, pred_key)

        if np.isnan(ari):
            print(f"  {pred_key:<20} FIDE={fide:.4f}")
        else:
            print(f"  {pred_key:<20} ARI={ari:.4f}  NMI={nmi:.4f}  FIDE={fide:.4f}")


def build_parser():
    p = argparse.ArgumentParser(
        description="HEBI inference: encode sections and assign niches."
    )
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file. Command-line flags override it.")

    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Pretrained model checkpoint (.pt). Required unless "
                        "--init random.")
    p.add_argument("--data_dir", type=str,
                   help="Directory of preprocessed .pt sections.")
    p.add_argument("--section_files", nargs="+",
                   help="Names of the .pt sections to encode, in order.")
    p.add_argument("--h5ad_files", nargs="+",
                   help="Matching .h5ad files, same order as --section_files.")
    p.add_argument("--vocab_path", type=str,
                   help="Gene vocabulary JSON used for harmonisation.")

    p.add_argument("--mode", choices=list(MODES),
                   default="zero_shot",
                   help="Encode directly, or adapt to the target sections first.")
    p.add_argument("--init", choices=list(INIT_MODES),
                   default="pretrained",
                   help="Weight initialisation. 'random' is the from-scratch "
                        "baseline; 'hybrid' randomises only the Level-0 encoder.")

    p.add_argument("--cluster_method", choices=list(CLUSTER_METHODS),
                   default="kmeans",
                   help="Algorithm for cell-level clustering. mclust needs R.")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Leiden resolution, used only when no cluster count "
                        "is given.")

    p.add_argument("--n_cell_clusters", type=int, default=8,
                   help="Number of cell-level clusters.")
    p.add_argument("--n_composition_niches", type=int, default=None,
                   help="Number of niches from local prototype composition.")
    p.add_argument("--composition_neighbors", type=int, default=200,
                   help="Spatial neighbours per cell for composition vectors.")

    p.add_argument("--cell_type_key", type=str, default=None,
                   help="adata.obs column with ground-truth cell types, for "
                        "ARI/NMI. Omit if unannotated.")
    p.add_argument("--niche_key", type=str, default=None,
                   help="adata.obs column with ground-truth niches.")

    p.add_argument("--grn", action="store_true",
                   help="Build the gene-interaction graph and enable the GRN "
                        "refinement layer.")
    p.add_argument("--species", type=str, default="Human",
                   help="Species for GRN priors.")
    p.add_argument("--grn_cluster_key", type=str, default=None,
                   help="adata.obs column to group GRN activity and CCC by. "
                        "Defaults to composition_niches, else cell_clusters.")
    p.add_argument("--grn_ccc", action="store_true",
                   help="Also run cell-cell communication. Writes one figure "
                        "per sender niche, so this can be many files.")
    p.add_argument("--section_labels", nargs="+", default=None,
                   help="Display name per section, in the same order as "
                        "--section_files. Used for Excel sheet names and "
                        "figure panel titles. Defaults to the .pt filenames.")
    p.add_argument("--ccc_top_k_pairs", type=int, default=5,
                   help="Ligand-receptor pairs kept in the chord diagram.")
    p.add_argument("--ccc_min_score_pct", type=float, default=0.15,
                   help="Keep links scoring at least this fraction of the "
                        "strongest link.")
    p.add_argument("--ccc_top_n_edges", type=int, default=18,
                   help="Cap on links drawn per panel. 0 for no cap.")
    p.add_argument("--ccc_keep_saturated", action="store_true",
                   help="Keep edges with attention exactly 1.0. These are "
                        "receptors with in-degree 1, whose score is purely "
                        "expression-driven; excluded by default.")
    p.add_argument("--per_section_grn", action="store_true",
                   help="One gene graph per section, for cross-platform runs "
                        "with differing panels.")

    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate for continual adaptation.")
    p.add_argument("--num_epochs", type=int, default=10,
                   help="Epochs of continual adaptation.")

    p.add_argument("--output_dir", type=str, default="./hebi_results")
    p.add_argument("--output_checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--point_size", type=float, default=None,
                   help="Scatter point size. Chosen automatically from cell "
                        "count when omitted.")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip writing figures; only print metrics.")
    p.add_argument("--filter_min_counts", type=int, default=None,
                   help="Drop cells below this total count. Off by default: "
                        "the .pt sections fix the cell order, so filtering "
                        "here desynchronises them.")
    p.add_argument("--drop_unlabelled", action="store_true",
                   help="Drop cells whose 'annotation' is 'Unlabeled'. Same "
                        "alignment caveat as --filter_min_counts.")
    p.add_argument("--no_preprocess", action="store_true",
                   help="Skip normalisation and log1p. Use when the .h5ad is "
                        "already normalised — scanpy warns about this. "
                        "Cell filtering is separate (--filter_min_counts).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.config:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

        # Command-line flags win over the config file.
        tokens = argv if argv is not None else sys.argv[1:]
        given = {t.lstrip("-") for t in tokens if t.startswith("--")}
        for key, value in config.items():
            if key not in given and hasattr(args, key):
                setattr(args, key, value)

    for required in ("data_dir", "section_files", "h5ad_files", "vocab_path"):
        if getattr(args, required, None) is None:
            raise SystemExit(f"--{required} is required (flag or config file)")

    if len(args.section_files) != len(args.h5ad_files):
        raise SystemExit(
            f"--section_files has {len(args.section_files)} entries but "
            f"--h5ad_files has {len(args.h5ad_files)}; they must match."
        )

    return run_hebi(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        section_files=args.section_files,
        h5ad_files=args.h5ad_files,
        vocab_path=args.vocab_path,
        mode=args.mode,
        init=args.init,
        n_cell_clusters=args.n_cell_clusters,
        n_composition_niches=args.n_composition_niches,
        composition_neighbors=args.composition_neighbors,
        cell_type_key=args.cell_type_key,
        niche_key=args.niche_key,
        cluster_method=args.cluster_method,
        resolution=args.resolution,
        grn=args.grn,
        species=args.species,
        per_section_grn=args.per_section_grn,
        grn_cluster_key=args.grn_cluster_key,
        grn_ccc=args.grn_ccc,
        section_labels=args.section_labels,
        ccc_top_k_pairs=args.ccc_top_k_pairs,
        ccc_min_score_pct=args.ccc_min_score_pct,
        ccc_top_n_edges=args.ccc_top_n_edges,
        ccc_keep_saturated=args.ccc_keep_saturated,
        lr=args.lr,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        output_checkpoint=args.output_checkpoint,
        device=args.device,
        seed=args.seed,
        preprocess=not args.no_preprocess,
        save_plots=not args.no_plots,
        filter_min_counts=args.filter_min_counts,
        drop_unlabelled=args.drop_unlabelled,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
