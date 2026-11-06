#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEBI inference, in three stages.

The stages are separate because they cost very different amounts. Encoding
adapts the model and runs it over every cell, which takes minutes to
hours. Clustering is a few k-means fits over embeddings that already
exist, which takes seconds. Keeping them apart means trying a different
number of clusters does not re-encode anything.

    sections, trainer, grn = encode_sections(...)          # slow, once

    cluster_sections(sections, n_cell_clusters=16,
                     n_composition_niches=16)              # seconds

    cluster_sections(sections, n_cell_clusters=24,
                     n_composition_niches=12)              # seconds again

    analyse_grn(sections, grn, ...)                        # GRN + CCC

Labels are named after the setting that produced them — z0_kmeans_16,
comp_niches_12 — so several clusterings coexist on the same AnnData and
their figures do not overwrite each other.

run_hebi() runs all three in order and is what the command line calls.
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
from hebi.grn.niche_activity import niche_edge_activity, plot_top_edges_heatmap
from hebi.grn.build import build_hetero_grn
from hebi.inference.adapt import continual_adapt, create_random_checkpoint
from hebi.inference.cluster import (
    CLUSTER_METHODS,
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
    species: str = "Human",
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

        adata = harmonise_adata(adata, vocab_path,species=species)

        if drop_unlabelled and "annotation" in adata.obs.columns:
            adata = adata[adata.obs["annotation"] != "Unlabeled"].copy()

        if adata.n_obs != n_start:
            print(f"  {n_start} -> {adata.n_obs} cells after filtering")
        print(f"  {adata.n_obs} cells, {adata.n_vars} genes")
        sections.append(adata)
    return sections
def assert_graph_matches_pt(gene_names, data_dir, section_file):
    """Stop if the gene graph's gene order differs from the .pt."""
    pt = torch.load(os.path.join(data_dir, section_file),
                    map_location="cpu", weights_only=False)
    pt_genes = [str(g).upper() for g in pt["gene_names"]]
    graph_genes = [str(g).upper() for g in gene_names]
    if pt_genes != graph_genes:
        first = next((i for i, (a, b) in enumerate(zip(pt_genes, graph_genes))
                      if a != b), min(len(pt_genes), len(graph_genes)))
        raise SystemExit(
            f"Gene graph does not match {section_file}: "
            f"{len(graph_genes)} vs {len(pt_genes)} genes, first difference "
            f"at index {first}. GRN attention would be assigned to the "
            "wrong genes.")
    print(f"  gene graph matches {section_file} ({len(pt_genes)} genes, "
          "same order)")

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


def encode_sections(
    checkpoint_path: str,
    data_dir: str,
    section_files,
    h5ad_files,
    vocab_path: str,
    mode: str = "zero_shot",
    init: str = "pretrained",
    section_labels=None,
    grn: bool = False,
    species: str = "Human",
    per_section_grn: bool = False,
    lr: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./hebi_results",
    output_checkpoint: str = None,
    device: str = None,
    seed: int = 0,
    verbose: int = 0,
    preprocess: bool = True,
    filter_min_counts: int = None,
    drop_unlabelled: bool = False,
):
    """Stage 1: adapt if asked, then encode every section.

    This is the expensive stage and the only one that touches the model.
    It writes the three representations and the prototype assignment into
    each section's AnnData, and returns everything the later stages need.

    Nothing is clustered here and nothing is plotted. The prototype
    assignment is not a clustering: it is the Level-2 attention argmax,
    read straight from the forward pass.

    Returns (sections, context) where context carries the trainer, the
    gene graphs, the GRN attention, and the settings the later stages
    need to stay consistent with this one.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
    if init not in INIT_MODES:
        raise ValueError(f"init must be one of {INIT_MODES}, got '{init}'")
    if grn and mode != "continual":
        raise SystemExit(
            f"grn requires mode='continual' (got '{mode}').\n\n"
            "The GRN refinement layer is only trained during continual "
            "adaptation. In zero-shot mode it is freshly initialised and "
            "never optimised, so its attention carries no signal."
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

    sections = load_sections(
        h5ad_files, vocab_path, preprocess=preprocess,
        filter_min_counts=filter_min_counts, drop_unlabelled=drop_unlabelled,species=species,   
    )
    sections = align_sections(sections, data_dir, section_files)

    # --- gene-interaction graphs -------------------------------------
    # gene_names travels with its graph: attention is indexed by gene
    # position, so the two must never be separated.
    gene_graph, gene_names = None, None
    gene_graph_list, gene_names_list = None, None

    if grn:
        print("\nBuilding gene-interaction graph")
        if per_section_grn:
            gene_graph_list, gene_names_list = [], []
            for adata, section_file in zip(sections, section_files):
                gg, gn, _ = build_hetero_grn(
                    adata=adata, species=species, spearman_threshold=0.4,
                    add_reverse_for_curated=True,
                )
                assert_graph_matches_pt(gn, data_dir, section_file)
                gene_graph_list.append(gg)
                gene_names_list.append(gn)
        else:
            gene_graph, gene_names, _ = build_hetero_grn(
                adata=sections[0], species=species, spearman_threshold=0.4,
                add_reverse_for_curated=True,
            )
            for section_file in section_files:
                assert_graph_matches_pt(gene_names, data_dir, section_file)

    # --- checkpoint --------------------------------------------------
    if init in ("pretrained", "hybrid"):
        if checkpoint_path is None:
            raise ValueError(f"checkpoint_path is required when init='{init}'")
        current_checkpoint = checkpoint_path
    else:
        current_checkpoint = create_random_checkpoint(
            data_dir=data_dir, vocab_path=vocab_path,
            output_checkpoint=os.path.join(output_dir, "hebi_random_init.pt"),
            device=device, seed=seed,
        )

    # --- adaptation --------------------------------------------------
    if mode == "continual":
        adapted = output_checkpoint or os.path.join(output_dir,
                                                    "hebi_continual.pt")
        continual_adapt(
            checkpoint_path=current_checkpoint, data_dir=data_dir,
            vocab_path=vocab_path, device=device, lr=lr,
            num_epochs=num_epochs, output_checkpoint=adapted,
            gene_graph=gene_graph, gene_graph_list=gene_graph_list,
            init=init, verbose=verbose,
        )
        current_checkpoint = adapted

    # --- encode ------------------------------------------------------
    trainer = load_hebi(
        checkpoint_path=current_checkpoint, data_dir=data_dir,
        vocab_path=vocab_path, device=device, init=init, use_grn=grn,
    )

    all_grn_attn, cell_counts = [], []
    print(f"\nEncoding {len(section_files)} section(s)")

    for i, (adata, section_file) in enumerate(zip(sections, section_files)):
        graph_i = gene_graph_list[i] if gene_graph_list is not None else gene_graph
        out = encode_section(
            trainer=trainer,
            section_path=os.path.join(data_dir, section_file),
            gene_graph=graph_i,
        )
        adata.obsm["z0"] = out["z0"].cpu().numpy()
        adata.obsm["z1"] = out["z1"].cpu().numpy()
        adata.obsm["z2"] = out["z2"].cpu().numpy()
        _set_cat(adata, KEY_PROTOTYPES, out["prototypes"].numpy())
        all_grn_attn.append(out["grn_attention"])
        cell_counts.append(adata.n_obs)

    n_prototypes = trainer.num_prototypes_lvl2
    used = len(np.unique(np.concatenate(
        [a.obs[KEY_PROTOTYPES].astype(int).to_numpy() for a in sections])))
    print(f"  {sum(cell_counts):,} cells encoded; "
          f"{used}/{n_prototypes} prototypes in use")

    context = {
        "trainer": trainer,
        "section_files": section_files,
        "section_labels": section_labels,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "cell_counts": cell_counts,
        "n_prototypes": n_prototypes,
        "gene_graph": gene_graph,
        "gene_names": gene_names,
        "gene_graph_list": gene_graph_list,
        "gene_names_list": gene_names_list,
        "grn_attention": all_grn_attn,
        "seed": seed,
    }
    return sections, context


def cluster_sections(
    sections,
    context,
    n_cell_clusters: int = 16,
    n_composition_niches: int = None,
    composition_neighbors: int = 200,
    cluster_method: str = "kmeans",
    resolution: float = 1.0,
    cell_type_key: str = None,
    niche_key: str = None,
    save_plots: bool = True,
    point_size: float = None,
    output_dir: str = None,
):
    """Stage 2: cluster the representations that stage 1 produced.

    Both clusterings are fitted once over all sections together, so a
    label means the same thing everywhere. Running each section
    separately would give labels that cannot be compared.

    Labels are written under names carrying the setting that made them —
    z0_kmeans_16, comp_niches_12 — so calling this again with different
    numbers adds columns rather than replacing them, and the figures do
    not overwrite each other.

    Returns the names of the columns written.
    """
    if cluster_method not in CLUSTER_METHODS:
        raise ValueError(f"cluster_method must be one of {CLUSTER_METHODS}")

    output_dir = output_dir or context["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    seed = context["seed"]
    counts = context["cell_counts"]
    labels = context["section_labels"]
    files = context["section_files"]

    offsets, cursor = [], 0
    for count in counts:
        offsets.append(slice(cursor, cursor + count))
        cursor += count

    written = {}

    # --- cell level --------------------------------------------------
    cell_key = f"z0_{cluster_method}_{n_cell_clusters}"
    print(f"\nCell-level clustering: {cluster_method}, "
          f"k={n_cell_clusters}  ->  obs['{cell_key}']")

    joint_z0 = np.concatenate([a.obsm["z0"] for a in sections], axis=0)
    cell_labels = cluster_embedding(
        joint_z0, n_clusters=n_cell_clusters, method=cluster_method,
        random_state=seed, resolution=resolution,
    )
    for adata, sl in zip(sections, offsets):
        _set_cat(adata, cell_key, cell_labels[sl])
    written["cell_clusters"] = cell_key

    # --- niche level -------------------------------------------------
    niche_key_name = None
    niche_labels = None
    if n_composition_niches is not None:
        niche_key_name = f"comp_niches_{n_composition_niches}"
        print(f"Niche-level clustering: {n_composition_niches} niches from "
              f"{composition_neighbors}-neighbour composition  ->  "
              f"obs['{niche_key_name}']")

        vectors = [
            build_composition_vectors(
                adata.obs[KEY_PROTOTYPES].astype(int).to_numpy(),
                adata.obsm["spatial"],
                k=composition_neighbors,
                n_prototypes=context["n_prototypes"],
            )
            for adata in sections
        ]
        niche_labels = cluster_embedding(
            np.concatenate(vectors, axis=0),
            n_clusters=n_composition_niches, method="kmeans",
            random_state=seed,
        )
        for adata, sl in zip(sections, offsets):
            _set_cat(adata, niche_key_name, niche_labels[sl])
        written["composition_niches"] = niche_key_name

    # --- scores ------------------------------------------------------
    print()
    for i, (adata, name) in enumerate(zip(sections, files)):
        edge_index = torch.load(
            os.path.join(context["data_dir"], name), map_location="cpu",
            weights_only=False,
        )["edge_index"]
        print(f"{labels[i]}")
        for key, truth in ((cell_key, cell_type_key),
                           (KEY_PROTOTYPES, niche_key),
                           (niche_key_name, niche_key)):
            if key is None or key not in adata.obs.columns:
                continue
            codes = adata.obs[key].cat.codes.to_numpy()
            fide = fide_score(codes, edge_index)
            ari, nmi = score_against_truth(adata, truth, key)
            if np.isnan(ari):
                print(f"  {key:<22} FIDE={fide:.4f}")
            else:
                print(f"  {key:<22} ARI={ari:.4f}  NMI={nmi:.4f}  "
                      f"FIDE={fide:.4f}")

    # --- figures -----------------------------------------------------
    if save_plots:
        print()
        _plot_combined(
            coords_list=[a.obsm["spatial"] for a in sections],
            joint_labels=cell_labels, offsets=offsets, sections=sections,
            section_names=labels, truth_key=cell_type_key, pred_key=cell_key,
            palette=get_palette(cell_labels),
            suptitle=f"Cell-level clusters on z0 "
                     f"({cluster_method}, k={n_cell_clusters})",
            legend_title="Cluster",
            out_path=os.path.join(output_dir, cell_key),
            point_size=point_size,
        )
        if niche_labels is not None:
            _plot_combined(
                coords_list=[a.obsm["spatial"] for a in sections],
                joint_labels=niche_labels, offsets=offsets, sections=sections,
                section_names=labels, truth_key=niche_key,
                pred_key=niche_key_name, palette=get_palette(niche_labels),
                suptitle=f"Composition niches (k={n_composition_niches}, "
                         f"{composition_neighbors} neighbours)",
                legend_title="Niche",
                out_path=os.path.join(output_dir, niche_key_name),
                point_size=point_size,
            )

    return written


def analyse_grn(
    sections,
    context,
    cluster_key: str,
    top_k_edges: int = 5,
    run_report: bool = True,
    run_ccc: bool = True,
    ccc_top_k_pairs: int = 8,
    ccc_min_score_pct: float = 0.0,
    ccc_top_n_edges: int = 20,
    ccc_keep_saturated: bool = False,
    ccc_min_edges: int = 100,
    ccc_fdr: float = 0.05,
    ccc_min_enrichment: float = 2.0,
    output_dir: str = None,
):
    """Stage 3: what the learned gene-interaction attention says.

    Three outputs. A heatmap of the strongest regulatory interactions in
    each group of cells, and a chord diagram of ligand-receptor
    communication between those same groups, both written once at the top
    level. Then, per section, the full report: hub genes ranked by total
    attention, edge salience corrected for how contested each target is,
    and the underlying tables as CSV.

    The per-section outputs go to <section>_plot/grn/. Set
    run_report=False for the two summary figures alone.

    cluster_key names the grouping: a discovered clustering from stage 2,
    or any annotation column. The choice matters — interactions between
    annotated cell types are a conventional analysis, while interactions
    between discovered niches are what the model's representation adds.
    """
    attention = context["grn_attention"]
    if not attention or all(a is None for a in attention):
        raise SystemExit(
            "No GRN attention available. Re-run encode_sections with "
            "grn=True and mode='continual'."
        )

    output_dir = output_dir or context["output_dir"]
    labels = context["section_labels"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGRN analysis, grouped by '{cluster_key}'")

    # --- per-section regulatory activity ------------------------------
    for i, adata in enumerate(sections):
        if attention[i] is None:
            print(f"  {labels[i]}: no attention, skipping")
            continue
        if cluster_key not in adata.obs.columns:
            raise KeyError(
                f"'{cluster_key}' not in obs for {labels[i]}. "
                f"Available: {[c for c in adata.obs.columns][:15]}"
            )

        graph = (context["gene_graph_list"][i]
                 if context["gene_graph_list"] is not None
                 else context["gene_graph"])
        names = (context["gene_names_list"][i]
                 if context["gene_names_list"] is not None
                 else context["gene_names"])

        section_name = os.path.splitext(context["section_files"][i])[0]
        grn_dir = os.path.join(output_dir, f"{section_name}_plot", "grn")
        os.makedirs(grn_dir, exist_ok=True)

        activity = niche_edge_activity(
            adata=adata, gene_graph=graph, gene_names=names,
            attention=attention[i], cluster_key=cluster_key,
        )
        activity.to_csv(os.path.join(grn_dir, "niche_edge_activity.csv"))

        plot_top_edges_heatmap(
            activity,
            out_path=os.path.join(output_dir, f"grn_top{top_k_edges}_{labels[i]}"),
            top_k=top_k_edges,
            title=f"{labels[i]}: top {top_k_edges} interactions per "
                  f"{cluster_key}",
        )

        if run_report:
            # Hub genes, edge salience, and the per-cluster tables. These
            # print heavily, so their output is folded into the files
            # rather than the console.
            run_grn_analysis(
                adata=adata,
                gene_graph=graph,
                gene_names=names,
                grn_attention=attention[i],
                out_dir=grn_dir,
                cluster_key=cluster_key,
                prefix=section_name,
                run_report=True,
                run_plots=True,
                run_ccc=False,     # communication is done once, below
            )

        print(f"  {labels[i]} -> {grn_dir}")

    # --- communication ------------------------------------------------
    if run_ccc:
        print("\nCell-cell communication")
        from hebi.grn.ccc import build_spatial_ccc_tensor

        comm_dfs = []
        for i, adata in enumerate(sections):
            if attention[i] is None:
                comm_dfs.append(None)
                continue
            graph = (context["gene_graph_list"][i]
                     if context["gene_graph_list"] is not None
                     else context["gene_graph"])
            names = (context["gene_names_list"][i]
                     if context["gene_names_list"] is not None
                     else context["gene_names"])
            # The spatial graph the model was trained on; align_sections has
            # already put the AnnData in the .pt cell order.
            edge_index = torch.load(
                os.path.join(context["data_dir"], context["section_files"][i]),
                map_location="cpu", weights_only=False)["edge_index"]
            comm_dfs.append(build_spatial_ccc_tensor(
                adata=adata, gene_graphs=graph, gene_names=names,
                grn_attn_weights=attention[i], edge_index=edge_index,
                cluster_key=cluster_key, min_edges=ccc_min_edges,
            ))

        write_ccc_excel(
            comm_dfs=comm_dfs, section_labels=labels,
            out_path=os.path.join(output_dir, "CCC_all_pairs.xlsx"),
        )
        plot_ccc_chord(
            comm_dfs=comm_dfs, section_labels=labels,
            out_path=os.path.join(output_dir, "ccc_chord"),
            top_k_pairs=ccc_top_k_pairs, min_score_pct=ccc_min_score_pct,
            top_n_edges=ccc_top_n_edges,
            drop_saturated=not ccc_keep_saturated,
            fdr_max=ccc_fdr,
            min_enrichment=ccc_min_enrichment,
        )


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
    grn_top_k_edges: int = 5,
    grn_report: bool = True,
    section_labels=None,
    ccc_top_k_pairs: int = 8,
    ccc_min_score_pct: float = 0.0,
    ccc_top_n_edges: int = 20,
    ccc_keep_saturated: bool = False,
    lr: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./hebi_results",
    output_checkpoint: str = None,
    device: str = None,
    seed: int = 0,
    verbose: int = 0,
    preprocess: bool = True,
    save_plots: bool = True,
    filter_min_counts: int = None,
    drop_unlabelled: bool = False,
    point_size: float = None,
):
    """All three stages in order. This is what the command line calls.

    Returns (sections, context). Call cluster_sections or analyse_grn on
    the same objects afterwards to try other settings without re-encoding.
    """
    sections, context = encode_sections(
        checkpoint_path=checkpoint_path, data_dir=data_dir,
        section_files=section_files, h5ad_files=h5ad_files,
        vocab_path=vocab_path, mode=mode, init=init,
        section_labels=section_labels, grn=grn, species=species,
        per_section_grn=per_section_grn, lr=lr, num_epochs=num_epochs,
        output_dir=output_dir, output_checkpoint=output_checkpoint,
        device=device, seed=seed, verbose=verbose, preprocess=preprocess,
        filter_min_counts=filter_min_counts, drop_unlabelled=drop_unlabelled,
    )

    written = cluster_sections(
        sections, context,
        n_cell_clusters=n_cell_clusters,
        n_composition_niches=n_composition_niches,
        composition_neighbors=composition_neighbors,
        cluster_method=cluster_method, resolution=resolution,
        cell_type_key=cell_type_key, niche_key=niche_key,
        save_plots=save_plots, point_size=point_size,
    )

    if grn:
        key = grn_cluster_key or written.get(
            "composition_niches", written["cell_clusters"])
        analyse_grn(
            sections, context, cluster_key=key,
            top_k_edges=grn_top_k_edges, run_report=grn_report,
            run_ccc=grn_ccc,
            ccc_top_k_pairs=ccc_top_k_pairs,
            ccc_min_score_pct=ccc_min_score_pct,
            ccc_top_n_edges=ccc_top_n_edges,
            ccc_keep_saturated=ccc_keep_saturated,
        )

    return sections, context


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


def _set_cat(adata, key, values):
    adata.obs[key] = np.asarray(values).astype(str)
    adata.obs[key] = adata.obs[key].astype("category")


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
    p.add_argument("--no_grn_report", action="store_true",
                   help="Skip the per-section hub-gene and salience report, "
                        "leaving only the heatmap and the chord diagram.")
    p.add_argument("--grn_top_k_edges", type=int, default=5,
                   help="Interactions per group shown in the GRN heatmap.")
    p.add_argument("--grn_ccc", action="store_true",
                   help="Also run cell-cell communication. Writes one figure "
                        "per sender niche, so this can be many files.")
    p.add_argument("--section_labels", nargs="+", default=None,
                   help="Display name per section, in the same order as "
                        "--section_files. Used for Excel sheet names and "
                        "figure panel titles. Defaults to the .pt filenames.")
    p.add_argument("--ccc_top_k_pairs", type=int, default=8,
                   help="Ligand-receptor pairs kept in the chord diagram.")
    p.add_argument("--ccc_min_score_pct", type=float, default=0.0,
                   help="Keep links scoring at least this fraction of the "
                        "strongest link.")
    p.add_argument("--ccc_top_n_edges", type=int, default=20,
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
    p.add_argument("--verbose", action="store_true",
                   help="Print per-slide progress during continual "
                        "adaptation. Off by default: one line per epoch.")
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
        grn_top_k_edges=args.grn_top_k_edges,
        grn_report=not args.no_grn_report,
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
        verbose=1 if args.verbose else 0,
        preprocess=not args.no_preprocess,
        save_plots=not args.no_plots,
        filter_min_counts=args.filter_min_counts,
        drop_unlabelled=args.drop_unlabelled,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
