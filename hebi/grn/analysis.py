#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRN analysis orchestrator.

Consumes the per-edge attention returned by encode_section and runs the
downstream analyses described in the Methods:

  hub importance + edge salience   report.run_global_grn_report
  niche-level GRN activity          plots.run_grn_plots
  cell-cell communication           ccc.run_ccc_analysis

Attention is normalised per target gene, so an edge index only means
something alongside the gene_names list from the same graph. The two are
always passed together here; with per-section graphs each section carries
its own pair.
"""

import contextlib
import os
import traceback

EDGE_TYPE_NAMES = ["pathway", "spearman", "ligrec", "tf_target"]


@contextlib.contextmanager
def non_interactive_plots(enabled: bool = True):
    """Make plt.show() a no-op and close figures as they accumulate.

    The GRN analysis functions call plt.show() after each figure, which
    blocks until the window is dismissed. In a batch run that means one
    click per figure, and a CCC run produces one figure per sender niche.
    Suppressing show lets the same functions save without stopping, and
    closing figures keeps memory flat over a long multi-section run.

    Set enabled=False to get the interactive behaviour back.
    """
    import matplotlib.pyplot as plt

    if not enabled:
        yield
        return

    original_show = plt.show
    original_max_open = plt.rcParams["figure.max_open_warning"]

    def _save_only(*args, **kwargs):
        # Figures have already been written by the caller's savefig.
        plt.close("all")

    plt.show = _save_only
    plt.rcParams["figure.max_open_warning"] = 0
    try:
        yield
    finally:
        plt.show = original_show
        plt.rcParams["figure.max_open_warning"] = original_max_open
        plt.close("all")


def run_grn_analysis(
    adata,
    gene_graph,
    gene_names,
    grn_attention,
    out_dir: str,
    cluster_key: str,
    prefix: str = "grn",
    top_k_edges: int = 15,
    top_k_genes: int = 15,
    top_k_pairs: int = 20,
    run_report: bool = True,
    run_plots: bool = True,
    run_ccc: bool = False,
    entity_label: str = "niche",
    show: bool = False,
):
    """Run the GRN analyses for one section.

    Each stage is independent: a failure in one is reported and the rest
    continue, so a long multi-section run is not lost to a single bad
    section.

    Returns a dict of whatever each stage produced.
    """
    if grn_attention is None:
        print(f"  No GRN attention for this section — skipping")
        return {}

    if gene_names is None:
        raise ValueError(
            "gene_names is required: GRN attention is indexed by gene "
            "position, so edges cannot be named without it."
        )

    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"cluster_key '{cluster_key}' not in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    with non_interactive_plots(enabled=not show):
        _run_stages(
            adata, gene_graph, gene_names, grn_attention, out_dir,
            cluster_key, prefix, top_k_edges, top_k_genes, top_k_pairs,
            run_report, run_plots, run_ccc, entity_label, results,
        )

    return results


def _run_stages(
    adata, gene_graph, gene_names, grn_attention, out_dir, cluster_key,
    prefix, top_k_edges, top_k_genes, top_k_pairs, run_report, run_plots,
    run_ccc, entity_label, results,
):
    if run_report:
        print(f"\n  GRN report -> {out_dir}")
        try:
            from hebi.grn.report import run_global_grn_report

            edge_df, node_df, edge_type_df = run_global_grn_report(
                gene_graphs=gene_graph,
                gene_names=gene_names,
                grn_attn_weights=grn_attention,
                edge_attr_names=EDGE_TYPE_NAMES,
                out_dir=out_dir,
                prefix=prefix,
                top_k_edges=top_k_edges,
                top_k_genes=top_k_genes,
                adata=adata,
                cluster_key=cluster_key,
            )
            results["edges"] = edge_df
            results["nodes"] = node_df
            results["edge_types"] = edge_type_df
        except Exception as exc:
            print(f"  GRN report failed: {exc}")
            traceback.print_exc()

    if run_plots:
        print(f"\n  GRN plots -> {out_dir}")
        try:
            from hebi.grn.plots import run_grn_plots

            run_grn_plots(
                adata=adata,
                gene_graphs=gene_graph,
                gene_names=gene_names,
                grn_attn_weights=grn_attention,
                cluster_key=cluster_key,
                edge_attr_names=EDGE_TYPE_NAMES,
                out_dir=out_dir,
                top_k_edges=top_k_edges,
                top_k_pairs=top_k_pairs,
            )
        except Exception as exc:
            print(f"  GRN plots failed: {exc}")
            traceback.print_exc()

    if run_ccc:
        print("\n  Building communication tensor")
        try:
            from hebi.grn.ccc import build_ccc_tensor

            # The tensor only. Figures are drawn once for all sections
            # together, so nothing per-section is plotted here.
            comm_df = build_ccc_tensor(
                adata=adata,
                gene_graphs=gene_graph,
                gene_names=gene_names,
                grn_attn_weights=grn_attention,
                cluster_key=cluster_key,
                edge_attr_names=EDGE_TYPE_NAMES,
            )
            results["communication"] = comm_df
        except Exception as exc:
            print(f"  CCC tensor failed: {exc}")
            traceback.print_exc()
