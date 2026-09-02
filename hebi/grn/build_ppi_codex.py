#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ppi_codex.py — per-region protein interaction graphs for CODEX panels.

Returns the SAME contract as build_hetero_grn:
    gene_graphs_list, gene_names, edge_attr_names

but one graph PER REGION, so grn attention varies per region
(same mechanism as --cross_platform in main_clustering.py).

Offline by design. Reads curated edges from the local cache written by
omnipath_cache / the omnipath python client. No network calls.

Design decisions specific to protein panels
-------------------------------------------
DROPPED  dorothea — TF -> target encodes TF protein driving target mRNA.
         You measure protein on both sides, so the edge semantics do not
         hold, and a 38-plex antibody panel never contains a TF and its
         transcriptional targets together.
KEPT     ligrecextra — intercellular in reality. Here it acts as a
         co-functional prior on per-cell marker embeddings, not a literal
         binding event. State this in the paper rather than glossing it.
ADDED    isoform edges — CD45 / CD45RA / CD45RO all measure PTPRC. Strongest
         possible relation, present in no database, so added explicitly.
ADDED    marker -> HGNC translation for names that are not valid symbols
         (CD11B, CD15, CD20). Without this they are isolated nodes.

NOT DONE normalize_total / log1p before Spearman. That step assumes counts.
         On 38 arcsinh-transformed CODEX channels it imposes a constant-sum
         constraint that manufactures negative correlation. build_grn.py does
         it; for protein data it is wrong, so it is omitted here.
"""

import os
import glob
import itertools

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.stats import spearmanr

CACHE_DIR = os.environ.get(
    "OMNIPATH_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnipath_cache"))

# Panel names that are NOT valid HGNC symbols. The .pt files are never
# modified — translation happens only here, so curated tables can match.
MARKER_FIX = {
    "CD11B": "ITGAM",
    "CD15":  "FUT4",
    "CD20":  "MS4A1",
}

_SRC_CANDS = ["source_genesymbol", "genesymbol_intercell_source",
              "source_gene_name", "source_symbol"]
_DST_CANDS = ["target_genesymbol", "genesymbol_intercell_target",
              "target_gene_name", "target_symbol"]


# =========================================================
# curated edges from local cache
# =========================================================
def _load_cached(name):
    path = os.path.join(CACHE_DIR, f"{name}.tsv")
    if not os.path.exists(path):
        path_csv = os.path.join(CACHE_DIR, f"{name}.csv")
        if os.path.exists(path_csv):
            path = path_csv
        else:
            print(f"   [cache] {name}: MISSING -> no {name} edges")
            return None
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                print(f"   [cache] {name}: {len(df)} rows")
                return df
        except Exception:
            continue
    print(f"   [cache] {name}: unreadable")
    return None


def _symbol_cols(df, name):
    """Return (src_series, dst_series) of upper-case gene symbols, or None."""
    src_col = next((c for c in _SRC_CANDS
                    if c in df.columns and df[c].notna().any()), None)
    dst_col = next((c for c in _DST_CANDS
                    if c in df.columns and df[c].notna().any()), None)
    if src_col is None or dst_col is None:
        print(f"   [{name}] no gene-symbol columns; "
              f"available: {list(df.columns)[:8]}")
        return None
    if src_col != "source_genesymbol":
        print(f"   [{name}] using {src_col} / {dst_col}")
    s = df[src_col].astype(str).str.strip().str.upper()
    d = df[dst_col].astype(str).str.strip().str.upper()
    return s, d


def load_curated_pairs(panel_symbols, use_ligrec=True, use_pathway=True):
    """
    Returns {edge_type: set of (sym_u, sym_v)} restricted to the panel.
    Only protein-appropriate sources. dorothea is deliberately excluded.
    """
    panel = set(panel_symbols)
    out = {}

    sources = []
    if use_ligrec:
        sources.append(("ligrec", "ligrecextra"))
    if use_pathway:
        sources.append(("pathway", "pathwayextra"))

    for etype, cache_name in sources:
        df = _load_cached(cache_name)
        if df is None or len(df) == 0:
            out[etype] = set()
            continue

        cols = _symbol_cols(df, cache_name)
        if cols is None:
            out[etype] = set()
            continue
        s, d = cols

        if etype == "pathway" and "sources" in df.columns:
            mask = df["sources"].astype(str).str.contains(
                "KEGG|Reactome", case=False, na=False)
            s, d = s[mask], d[mask]

        pairs = {(a, b) for a, b in zip(s, d)
                 if a in panel and b in panel and a != b}
        out[etype] = pairs
        print(f"   [{cache_name}] within panel: {len(pairs)} directed pairs")

    return out


# =========================================================
# panel handling
# =========================================================
def resolve_panel(marker_names):
    """
    marker_names : the gene_names list stored in the .pt
    returns symbols (translated), and family groups for isoform edges
    """
    symbols = [MARKER_FIX.get(str(m).upper(), str(m).upper())
               for m in marker_names]

    fam = {}
    for i, s in enumerate(symbols):
        fam.setdefault(s, []).append(i)
    duplicated = {s: idx for s, idx in fam.items() if len(idx) > 1}

    if duplicated:
        for s, idx in duplicated.items():
            names = [marker_names[i] for i in idx]
            print(f"   isoform family: {names} -> {s}")

    fixed = [(m, MARKER_FIX[str(m).upper()])
             for m in marker_names if str(m).upper() in MARKER_FIX]
    if fixed:
        print(f"   translated: {fixed}")

    return symbols, duplicated


def _spearman_edges(X, threshold):
    """X: [cells, markers] numpy. Returns set of undirected (u,v), u<v."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 3:
        return set()

    # drop constant columns — spearmanr returns nan for them
    std = X.std(0)
    keep = np.where(std > 1e-12)[0]
    if len(keep) < 2:
        return set()

    corr, _ = spearmanr(X[:, keep], axis=0)
    corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    if corr.shape[0] != len(keep):
        return set()

    edges = set()
    iu, ju = np.triu_indices(len(keep), k=1)
    strong = np.abs(corr[iu, ju]) > threshold
    for a, b in zip(iu[strong], ju[strong]):
        edges.add((int(keep[a]), int(keep[b])))
    return edges


# =========================================================
# main builder
# =========================================================
def build_ppi_graphs(
    pt_dir,
    spearman_threshold=0.4,
    max_cells=20000,
    add_isoform_edges=True,
    use_ligrec=True,
    use_pathway=True,
    add_reverse_for_curated=True,
    seed=123,
    verbose=True,
):
    """
    One graph per region .pt file.

    Returns
    -------
    gene_graphs_list : list of {"global": Data}, aligned with sorted(files)
    gene_names       : panel names (from the first file)
    edge_attr_names  : ["ligrec", "pathway", "isoform", "spearman"]
    region_ids       : list of region ids, same order as gene_graphs_list
    """
    files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"no .pt files in {pt_dir}")
    print(f"Building per-region protein graphs from {len(files)} regions")

    d0 = torch.load(files[0], map_location="cpu", weights_only=False)
    marker_names = list(d0["gene_names"])
    n_nodes = len(marker_names)
    print(f"   panel: {n_nodes} markers")

    symbols, families = resolve_panel(marker_names)

    # ---- curated edges: computed ONCE, identical across regions ----
    print("   loading curated priors from cache (offline)...")
    curated_sym = load_curated_pairs(symbols,
                                     use_ligrec=use_ligrec,
                                     use_pathway=use_pathway)

    sym2idx = {}
    for i, s in enumerate(symbols):
        sym2idx.setdefault(s, []).append(i)

    def to_index_edges(pairs):
        e = set()
        for a, b in pairs:
            for u in sym2idx.get(a, []):
                for v in sym2idx.get(b, []):
                    if u != v:
                        e.add((u, v))
                        if add_reverse_for_curated:
                            e.add((v, u))
        return e

    ligrec_e = to_index_edges(curated_sym.get("ligrec", set()))
    pathway_e = to_index_edges(curated_sym.get("pathway", set()))

    # ---- isoform edges (symmetric) ----
    isoform_e = set()
    if add_isoform_edges:
        for s, idx in families.items():
            for u, v in itertools.combinations(idx, 2):
                isoform_e.add((u, v))
                isoform_e.add((v, u))

    curated_all = ligrec_e | pathway_e | isoform_e
    cov = {x for p in curated_all for x in p}
    print(f"   curated (shared by all regions): ligrec={len(ligrec_e)} "
          f"pathway={len(pathway_e)} isoform={len(isoform_e)}")
    print(f"   curated node coverage: {len(cov)}/{n_nodes}")

    # ---- per-region Spearman ----
    rng = np.random.default_rng(seed)
    gene_graphs_list, region_ids, stats = [], [], []

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        rid = d.get("region_id", os.path.basename(f).replace(".pt", ""))

        X = d["X"]
        X = X.float().numpy() if torch.is_tensor(X) else np.asarray(X, float)
        if X.shape[0] > max_cells:
            X = X[rng.choice(X.shape[0], max_cells, replace=False)]

        # NOTE: no normalize_total / log1p — see module docstring
        sp_und = _spearman_edges(X, spearman_threshold)
        spearman_e = set()
        for u, v in sp_und:
            spearman_e.add((u, v))
            spearman_e.add((v, u))

        all_e = curated_all | spearman_e
        ei, ea = [], []
        for (u, v) in sorted(all_e):
            ei.append((u, v))
            ea.append([
                1 if (u, v) in ligrec_e else 0,
                1 if (u, v) in pathway_e else 0,
                1 if (u, v) in isoform_e else 0,
                1 if (u, v) in spearman_e else 0,
            ])

        g = Data()
        if ei:
            g.edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
            g.edge_attr = torch.tensor(ea, dtype=torch.float)
        else:
            g.edge_index = torch.empty((2, 0), dtype=torch.long)
            g.edge_attr = torch.empty((0, 4), dtype=torch.float)
        g.num_nodes = n_nodes

        gene_graphs_list.append({"global": g})
        region_ids.append(rid)
        stats.append((rid, len(spearman_e) // 2, g.edge_index.shape[1]))

    edge_counts = [s[2] for s in stats]
    print(f"\n   per-region edges: min={min(edge_counts)} "
          f"max={max(edge_counts)} mean={np.mean(edge_counts):.0f}")
    if len(set(edge_counts)) == 1:
        print("   ⚠️  every region has the SAME edge count — graphs may be "
              "identical, so attention will not vary. Check X differs.")
    else:
        print("   ✓ graphs differ across regions -> attention will differ")

    if verbose:
        print("\n   first 5 regions:")
        for rid, nsp, tot in stats[:5]:
            print(f"     {rid:<40} spearman={nsp:<5} total={tot}")

    edge_attr_names = ["ligrec", "pathway", "isoform", "spearman"]
    return gene_graphs_list, marker_names, edge_attr_names, region_ids


# =========================================================
# shared curated edge index — for building classifier features
# =========================================================
def shared_curated_edges(gene_graphs_list, gene_names):
    """
    Positions of edges present in EVERY region, with identical (u,v).

    Spearman edges differ per region, so the full attention vectors are not
    comparable across regions. These indices give a fixed-width, aligned
    slice you can safely feed a classifier.

    Returns
    -------
    per_region_positions : list[np.ndarray]  index into that region's attn
    edge_labels          : list[str]         "GENEA->GENEB", same order
    """
    key_sets = []
    for gg in gene_graphs_list:
        ei = gg["global"].edge_index.numpy()
        key_sets.append({(int(ei[0, e]), int(ei[1, e]))
                         for e in range(ei.shape[1])})

    shared = set.intersection(*key_sets) if key_sets else set()
    shared = sorted(shared)

    per_region_positions = []
    for gg in gene_graphs_list:
        ei = gg["global"].edge_index.numpy()
        pos = {(int(ei[0, e]), int(ei[1, e])): e for e in range(ei.shape[1])}
        per_region_positions.append(np.array([pos[k] for k in shared],
                                             dtype=int))

    edge_labels = [f"{gene_names[u]}->{gene_names[v]}" for u, v in shared]
    print(f"shared edges across all regions: {len(shared)}")
    return per_region_positions, edge_labels


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else \
        "/DATA1/Roxana/Hierarchy/protein_data_256/charville"
    ggl, names, eattr, rids = build_ppi_graphs(d)
    shared_curated_edges(ggl, names)