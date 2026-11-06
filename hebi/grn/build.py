#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_grn.py -- FIXED

Root cause of the "7 rows" bug:
  _fetch_dataset always passed genes=gene_names to OmniPath, which chunks
  the gene list into groups of 150 and sends `params["genes"]=...`. That
  parameter is not supported by the current OmniPath API. The server
  replies with HTTP 200 and a plain-text error message in the body
  ("Unknown argument: `genes`"). Because status==200, _get_live() never
  raises, and _parse_response() happily runs pd.read_csv on the error text,
  producing a tiny "dataframe" where each line of the error message becomes
  a one-column row. That garbage then got written to the cache files.

Fix:
  1. Never send the `genes` parameter -- OmniPath does not support
     per-request gene filtering the way this code assumed. Fetch the WHOLE
     dataset (same approach used successfully in build_ppi_codex.py for
     the protein cohorts) and filter to the gene panel locally afterwards.
  2. Add a response-sanity guard: if the parsed dataframe doesn't have the
     expected source/target-ish columns, treat it as a failed fetch (raise)
     rather than silently caching garbage.
"""

import io
import os
import time
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import scanpy as sc
import torch
from torch_geometric.data import Data
from scipy.stats import spearmanr

# =========================================================
# Cache & Fetch Setup
# =========================================================
DEFAULT_CACHE_DIR = "/DATA1/Roxana/Hierarchy/Two_levels/clustering/omnipath_cache"
CACHE_DIR = os.environ.get("OMNIPATH_CACHE_DIR", DEFAULT_CACHE_DIR)

_HOSTS = [
    "https://omnipathdb.org",
    "https://archive.omnipathdb.org",
]

_URL = "https://omnipathdb.org/interactions"
_RETRIES = 2
_TIMEOUT = 60  # whole-dataset fetch is bigger, give it more time


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.tsv")


def _read_cache(name: str):
    p = _cache_path(name)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        try:
            df = pd.read_csv(p, sep="\t", low_memory=False)
            if _looks_valid(df):
                print(f"   [cache-read] loaded '{name}' from local cache ({len(df)} rows)")
                return df
            print(f"   [warn] cached '{name}' looks invalid ({len(df)} rows, "
                  f"cols={list(df.columns)[:3]}) -- ignoring cache")
        except Exception as e:
            print(f"   [warn] failed to read cache for '{name}': {e}")
    else:
        print(f"   [warn] cache file missing/empty for '{name}' at {p}")
    return None


def _write_cache(name: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        df.to_csv(_cache_path(name), sep="\t", index=False)
        print(f"   [cache-write] updated local cache for '{name}': {len(df)} rows")
    except Exception as e:
        print(f"   [warn] could not write cache for '{name}': {e}")


def _looks_valid(df: pd.DataFrame) -> bool:
    """Sanity check: a real OmniPath interactions table has source/target-ish
    columns and more than a handful of rows. An error message parsed as TSV
    will have 1 column and ~5-10 rows -- this catches exactly that."""
    if df is None or df.empty:
        return False
    if len(df.columns) < 3:
        return False
    if len(df) < 20:
        return False
    cols_lower = [c.lower() for c in df.columns]
    has_src_dst = any("source" in c for c in cols_lower) and \
                  any("target" in c for c in cols_lower)
    return has_src_dst


def _parse_response(resp: requests.Response) -> pd.DataFrame:
    txt = resp.text.strip()
    if not txt:
        return pd.DataFrame()
    if txt.startswith(("[", "{")):
        try:
            return pd.DataFrame(resp.json())
        except Exception:
            pass
    return pd.read_csv(io.StringIO(txt), sep="\t", low_memory=False)


def _get_live(params: dict) -> pd.DataFrame:
    """Whole-dataset fetch, no gene filtering (OmniPath doesn't support the
    `genes` param the old code used). Validates the parsed response before
    accepting it, so an HTTP-200 error page can't masquerade as data."""
    last_err = None
    for host in _HOSTS:
        url = _URL.replace("https://omnipathdb.org", host, 1)
        for attempt in range(_RETRIES):
            try:
                r = requests.get(url, params=params, timeout=_TIMEOUT)
                if r.status_code in (500, 502, 503, 504):
                    raise requests.HTTPError(f"Server Error {r.status_code}")
                r.raise_for_status()
                df = _parse_response(r)
                if not _looks_valid(df):
                    raise RuntimeError(
                        f"Response did not look like a valid interactions "
                        f"table (rows={len(df)}, cols={list(df.columns)[:5]}). "
                        f"First 200 chars of body: {r.text[:200]!r}")
                return df
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"   [retry {attempt+1}/{_RETRIES}] {host}: {e} "
                      f"-- retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(f"All OmniPath hosts failed. Last error: {last_err}")


def _fetch_dataset(dataset: str, cache_name: str, species_tax: int = 9606) -> pd.DataFrame:
    """No `genes` param -- fetch the whole dataset, filter locally later."""
    base_params = {
        "datasets": dataset,
        "organisms": str(species_tax),
        "genesymbols": "1",
        "format": "tsv",
    }
    try:
        print(f"   Fetching '{dataset}' from OmniPath web API (whole dataset)...")
        df = _get_live(base_params)
        df = df.drop_duplicates().reset_index(drop=True)
        _write_cache(cache_name, df)
        return df
    except Exception as e:
        print(f"\n   ⚠️ WARNING: failed to fetch '{dataset}' from web API ({e})")
        print(f"   [fallback] reading cache for '{dataset}'...")
        cached_df = _read_cache(cache_name)
        if cached_df is not None:
            return cached_df

    print(f"   ‼️ Could not load '{dataset}'. Proceeding with 0 edges.\n")
    return pd.DataFrame()


# =========================================================
# API Fetch Helpers  (genes param removed -- filtering happens later)
# =========================================================
def fetch_pathwayextra_all(species_tax: int = 9606) -> pd.DataFrame:
    df = _fetch_dataset("pathwayextra", "pathwayextra", species_tax)
    if not df.empty and "sources" in df.columns:
        mask = df["sources"].astype(str).str.contains("KEGG|Reactome", case=False, na=False)
        return df[mask].copy()
    return df


def fetch_ligrecextra_all(species_tax: int = 9606) -> pd.DataFrame:
    return _fetch_dataset("ligrecextra", "ligrecextra", species_tax)


def fetch_dorothea_high_conf(species_tax: int = 9606, conf_levels=("A", "B")) -> pd.DataFrame:
    df = _fetch_dataset("dorothea", "dorothea", species_tax)
    if not df.empty and "confidence" in df.columns:
        return df[df["confidence"].isin(conf_levels)].copy()
    return df


def pick_symbols(df: pd.DataFrame, base: str) -> pd.Series:
    for c in (f"{base}_genesymbol", f"{base}_gene_name", f"{base}_symbol", base):
        if c in df.columns:
            return df[c].astype(str).str.strip().str.upper()
    matches = [c for c in df.columns if base in c.lower() and "symbol" in c.lower()]
    if matches:
        return df[matches[0]].astype(str).str.strip().str.upper()
    matches = [c for c in df.columns if base in c.lower()]
    if matches:
        return df[matches[0]].astype(str).str.strip().str.upper()
    raise KeyError(f"No suitable column found for base='{base}'")


# =========================================================
# Numeric Helper
# =========================================================
def safe_spearman_corr_matrix(X):
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    corr, _ = spearmanr(X, axis=0)
    if isinstance(corr, float):
        return np.zeros((X.shape[1], X.shape[1]), dtype=float)
    if corr.shape[0] != corr.shape[1]:
        n = X.shape[1]
        corr = corr[-n:, -n:]
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


# =========================================================
# Directed GRN Builder
# =========================================================
def build_hetero_grn(
    adata,
    species="Human",
    max_cells=3000,
    spearman_threshold=0.4,
    conf_min=0.2,
    seed=123,
    add_reverse_for_curated=False,
):
    print(f"🧬 Building directed heterogeneous GRN (Species: {species})...")
    species_tax = 9606 if species.lower() == "human" else 10090

    if "gene_symbol" in adata.var.columns:
        gene_names = list(adata.var["gene_symbol"].astype(str).str.upper())
    else:
        gene_names = list(pd.Index(adata.var.index).astype(str).str.upper())
    gene2idx = {g: i for i, g in enumerate(gene_names)}

    print("   Fetching curated priors (Pathway, LigRec, TF-Target)...")
    df_pathway = fetch_pathwayextra_all(species_tax)
    df_ligrec = fetch_ligrecextra_all(species_tax)
    df_doro = fetch_dorothea_high_conf(species_tax)

    def get_directed_edges(df, tf=False, conf_col=None):
        """Return (all_edges, forward_edges).

        forward_edges holds each interaction in the direction recorded by the
        curated source. all_edges additionally holds the reverse copies added
        for message passing when add_reverse_for_curated is set.
        """
        if df is None or len(df) == 0:
            return set(), set()
        try:
            src_list = pick_symbols(df, "source").str.upper()
            dst_list = pick_symbols(df, "target").str.upper()
        except KeyError:
            return set(), set()
        if conf_col is not None and conf_col in df.columns:
            conf_list = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.0).astype(float)
        else:
            conf_list = pd.Series(np.ones(len(df), dtype=float))
        edges = set()
        forward = set()
        for s, d, c in zip(src_list, dst_list, conf_list):
            if s in gene2idx and d in gene2idx and s != d:
                if float(c) >= float(conf_min):
                    edges.add((gene2idx[s], gene2idx[d]))
                    forward.add((gene2idx[s], gene2idx[d]))
                    if tf:
                        continue
                    if add_reverse_for_curated:
                        edges.add((gene2idx[d], gene2idx[s]))
        return edges, forward

    pathway_edges, pathway_forward = get_directed_edges(df_pathway, conf_col="confidence")
    ligrec_edges, ligrec_forward = get_directed_edges(df_ligrec, conf_col=None)
    tf_edges, _ = get_directed_edges(df_doro, tf=True, conf_col=None)

    print("   Computing data-driven Spearman edges...")
    np.random.seed(seed)
    if adata.n_obs > max_cells:
        take = np.random.choice(adata.obs_names, size=max_cells, replace=False)
        adata_sub = adata[take].copy()
    else:
        adata_sub = adata.copy()

    # NOTE: removed sc.pp.normalize_total + sc.pp.log1p here.
    # Input X (from clinical_preprocess.py's de-zscore step) is ALREADY
    # log-normalized (verified: min=0, max~8.3, no negatives, matches
    # log1p(normalize_total(counts)) output shape). Re-applying
    # normalize_total treats log-values as if they were additive counts
    # (wrong), and re-applying log1p on top of an already-log-scale signal
    # (max~8.3 -> log1p gives ~2.2) compresses the dynamic range and flattens
    # real variance between genes -- this was silently weakening every
    # downstream Spearman edge and, transitively, the GRN attention signal.
    # If this function is ever called on TRUE raw counts (protein/CODEX
    # cohorts, or a future dataset), normalize_total+log1p must be restored
    # for that path -- check adata.X.max() as a quick raw-vs-log heuristic
    # (raw counts are typically >>20; log-normalized data is usually <15).
    Xsub = adata_sub.X
    if hasattr(Xsub, "toarray"):
        Xsub = Xsub.toarray()
    Xsub = np.asarray(Xsub, dtype=float)
    looks_like_counts = Xsub.max() > 20 and np.allclose(Xsub, np.round(Xsub), atol=1e-3)
    if looks_like_counts:
        print("   [info] input looks like raw counts -- applying normalize_total+log1p")
        sc.pp.normalize_total(adata_sub, target_sum=1e4)
        sc.pp.log1p(adata_sub)
    else:
        print(f"   [info] input already log-scale (max={Xsub.max():.2f}) -- "
              f"skipping normalize_total/log1p to avoid double-normalizing")
    corr_matrix = safe_spearman_corr_matrix(adata_sub.X)

    triu_i, triu_j = np.triu_indices_from(corr_matrix, k=1)
    abs_corr = np.abs(corr_matrix[triu_i, triu_j])
    keep = abs_corr > spearman_threshold
    spearman_edges = set()
    for u, v in zip(triu_i[keep], triu_j[keep]):
        if u != v:
            spearman_edges.add((int(u), int(v)))

    print("   Constructing PyG graph with directed edge typing...")
    all_edges = pathway_edges | ligrec_edges | tf_edges | spearman_edges

    edge_index_list = []
    edge_attr_list = []
    for (u, v) in all_edges:
        attr = [0, 0, 0, 0]
        if (u, v) in pathway_edges: attr[0] = 1
        if (u, v) in spearman_edges: attr[1] = 1
        if (u, v) in ligrec_edges: attr[2] = 1
        if (u, v) in tf_edges: attr[3] = 1
        edge_index_list.append((u, v))
        edge_attr_list.append(attr)

    if len(edge_index_list) > 0:
        order = np.lexsort((
            np.array([v for (_, v) in edge_index_list]),
            np.array([u for (u, _) in edge_index_list]),
        ))
        edge_index_list = [edge_index_list[i] for i in order]
        edge_attr_list = [edge_attr_list[i] for i in order]

    data = Data()
    if len(edge_index_list) > 0:
        data.edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.edge_attr = torch.empty((0, 4), dtype=torch.float)

    data.num_nodes = len(gene_names)

    # Direction masks, aligned with edge_index.
    #   ligrec_forward  ligand -> receptor as recorded in OmniPath (used by CCC)
    #   reverse_only    edge exists only as a reverse copy added for message
    #                   passing: not a forward curated edge, not a TF edge and
    #                   not a co-expression edge. These carry attention (they
    #                   take part in the softmax) but are not interactions and
    #                   must be excluded from any reported ranking.
    forward_curated = pathway_forward | ligrec_forward | tf_edges
    data.ligrec_forward = torch.tensor(
        [(u, v) in ligrec_forward for (u, v) in edge_index_list],
        dtype=torch.bool,
    )
    data.reverse_only = torch.tensor(
        [((u, v) not in forward_curated) and ((u, v) not in spearman_edges)
         for (u, v) in edge_index_list],
        dtype=torch.bool,
    )
    gene_graphs = {"global": data}
    edge_attr_names = ["pathway", "spearman", "ligrec", "tf_target"]

    print(f"✅ Directed GRN built! Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
    print(f"   Pathway edges:  {len(pathway_edges)}")
    print(f"   Spearman edges: {len(spearman_edges)}")
    print(f"   LigRec edges:   {len(ligrec_edges)}")
    print(f"   TF edges:       {len(tf_edges)}")
    print(f"   Reverse-only copies (excluded from reports): "
          f"{int(data.reverse_only.sum())}")

    return gene_graphs, gene_names, edge_attr_names