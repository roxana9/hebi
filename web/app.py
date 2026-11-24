#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEBI zero-shot explorer: web backend.

One run = one or several sections (one .h5ad file per slide). Several slides
are encoded by the same model and clustered together, so a cluster means the
same thing on every slide. The server runs HEBI exactly as the repository
does, through its own public functions, and never changes them:

    preprocess_h5ad(...)                        h5ad -> .pt (graph, communities)
    encode_sections(init="random", grn=False,
                    mode="zero_shot")           one forward pass, no training
                    mode="continual")           train N epochs on the uploaded
                                                slides first, then encode
    cluster_embedding(z0, ...)                  k-means on the cell embedding

init is "random", or "pretrained" when HEBI_CHECKPOINT points to a checkpoint.
Zero-shot returns z0 and its clusters only: without training on the data the niche
prototypes carry no learned structure. Continual also returns the niches,
the Level-2 prototype each cell is assigned to (argmax of its attention),
learned jointly over all uploaded slides.

Run:
    pip install fastapi uvicorn
    python app.py                # then open http://127.0.0.1:8000

Settings are environment variables (see the block below). The page itself is
index.html, served from the same folder as this file.
"""

import contextlib
import gc
import importlib
import io
import json
import os
import re
import sys
import shutil
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

VOCAB_PATH = os.environ.get(
    "HEBI_VOCAB", "/DATA1/Roxana/Hierarchy/Two_levels/default_gene_vocab.json")
# Pretrained HEBI weights, e.g. hebi_hybrid_oldL0.pt. Leave unset to offer random
# initialisation only.
CHECKPOINT_PATH = os.environ.get("HEBI_CHECKPOINT", "")
INITS = ("random", "pretrained")

# Niche objective for continual training.
#   original  the repository's continual_adapt, unchanged
#   spatial   the same model and losses plus a sharpness term on the
#             community-to-niche assignment and a spatial consistency term between
#             adjacent communities. Uses the repository's
#             encode_sections(niche_objective="spatial") when the installed HEBI
#             has it; otherwise falls back to spatial_niches.py next to app.py.
NICHE_OBJECTIVES = ("original", "spatial")
SPATIAL_SETTINGS = {"spatial_weight": 1.5, "sharp_weight": 2.0,
                    "edge_weighting": "uniform",
                    "loss_weights": {"balance": 2.0, "smooth": 3.0}}
SPATIAL_MODULE = os.environ.get("HEBI_SPATIAL_MODULE", "spatial_niches")
WORK_DIR = Path(os.environ.get("HEBI_WEB_WORKDIR", HERE / "hebi_web_jobs"))
DEVICE = os.environ.get("HEBI_DEVICE") or None        # None: cuda:0 if available
PREPROCESS_MODULE = os.environ.get("HEBI_PREPROCESS_MODULE")  # e.g. hebi.data.preprocess

HOST = os.environ.get("HEBI_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("HEBI_WEB_PORT", "8000"))

MAX_UPLOAD_GB = float(os.environ.get("HEBI_WEB_MAX_UPLOAD_GB", "8"))
MAX_CELLS = int(os.environ.get("HEBI_WEB_MAX_CELLS", "1500000"))
MIN_CELLS = 200
MAX_FILES = int(os.environ.get("HEBI_WEB_MAX_FILES", "12"))
# Optional folder of example .h5ad sections the page offers without an upload.
EXAMPLES_DIR = os.environ.get("HEBI_WEB_EXAMPLES", "")


def _examples():
    if not EXAMPLES_DIR or not Path(EXAMPLES_DIR).is_dir():
        return {}
    return {p.name: p for p in sorted(Path(EXAMPLES_DIR).glob("*.h5ad"))}
KEEP_JOBS = int(os.environ.get("HEBI_WEB_KEEP_JOBS", "3"))    # finished runs kept in memory
EMBED_MAX_CELLS = int(os.environ.get("HEBI_WEB_EMBED_MAX_CELLS", "40000"))

DEFAULT_K_GRAPH = 30
DEFAULT_N_METIS = 1024                   # preprocess.py default
DEFAULT_SEED = 0
DEFAULT_N_CLUSTERS = 8
MIN_CLUSTERS, MAX_CLUSTERS = 2, 24
TOP_MARKERS = 25
DOTPLOT_GENES_PER_CLUSTER = 3
CLUSTER_METHOD = "kmeans"

SPATIAL_HINTS = ("spatial", "x_spatial", "spatial_coords", "coordinates", "coords")
STAGES = ("read", "graph", "adapt", "encode", "niches", "cluster", "embed")
MODES = ("zero_shot", "continual")
DEFAULT_EPOCHS = 10
MAX_EPOCHS = int(os.environ.get("HEBI_WEB_MAX_EPOCHS", "1000"))
DEFAULT_LR = 1e-4
LR_RANGE = (1e-6, 1e-2)


# ----------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------
class Job:
    def __init__(self, job_id, species, k_graph, seed, n_clusters,
                 mode="zero_shot", epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, normalise=False,
                 n_hvg=0, init="random", n_metis=DEFAULT_N_METIS,
                 niche_objective="original"):
        self.id = job_id
        self.niche_objective = niche_objective
        self.n_metis = n_metis
        self.init = init
        self.n_hvg = n_hvg
        self.normalise = normalise
        self.mode, self.epochs, self.lr = mode, epochs, lr
        self.detail = None               # e.g. "epoch 3 of 10"
        self.niches = {}                 # continual only: k (0 = learned prototypes) -> entry
        self.niche_codes = None
        self.niche_res = {}              # tiny policy -> {k: (codes over all cells, {label: path})}
        self.files = []                  # one dict per uploaded slide
        self.species = species
        self.k_graph = k_graph
        self.seed = seed
        self.n_clusters = n_clusters
        self.dir = WORK_DIR / job_id
        self.status = "uploading"        # uploading | queued | running | done | error
        self.stage = None
        self.error = None
        self.log = []
        self.created = time.time()
        self.timings = {}
        self.lock = threading.RLock()   # re-entrant: downloads call the cached builders
        # results
        self.secs = []                   # per slide: {"adata", "X", "n"}
        self.z0 = None
        self.coords = None
        self.sec_of = None
        self.gene_names = None
        self.gene_index = None
        self.clusters = OrderedDict()    # k -> {"codes", "names", "info"}
        self.embed_idx = None
        self.embed_xy = None
        self.meta = {}

    def release(self):
        self.secs = []
        self.z0 = self.coords = self.sec_of = None
        self.clusters.clear()
        self.niches, self.niche_codes, self.niche_res = {}, None, {}
        self.embed_idx = self.embed_xy = None
        shutil.rmtree(self.dir, ignore_errors=True)


JOBS = OrderedDict()
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)     # one model run at a time


def get_job(job_id, need_done=False):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "This run is no longer on the server. Upload the file again.")
    if need_done and job.status != "done":
        raise HTTPException(409, "This run has not finished yet.")
    return job


def evict_old_jobs():
    with JOBS_LOCK:
        for job in [j for j in JOBS.values()
                    if j.status == "uploading" and time.time() - j.created > 12 * 3600]:
            job.release()
            JOBS.pop(job.id, None)
        done = [j for j in JOBS.values() if j.status in ("done", "error")]
        for job in done[:-KEEP_JOBS] if len(done) > KEEP_JOBS else []:
            job.release()
            JOBS.pop(job.id, None)


class _Tee(io.TextIOBase):
    """Send prints from the HEBI functions to the console and to the run log."""

    def __init__(self, job, stream):
        self.job, self.stream, self._buf = job, stream, ""

    def write(self, text):
        self.stream.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.job.log.append(line.rstrip())
                self._watch(line)
        return len(text)

    def _watch(self, line):
        job = self.job
        if job.stage != "adapt":
            return
        m = re.search(r"Continual epoch (\d+)/(\d+)", line)
        m2 = re.match(r"\s*epoch\s+(\d+)\s+loss", line)          # spatial trainer
        if m:
            job.detail = f"epoch {m.group(1)} of {m.group(2)}"
        elif m2:
            job.detail = f"epoch {m2.group(1)} of {job.epochs}"
        elif line.strip().startswith("Encoding ") and "section" in line:
            _stage(job, "encode")
            job.detail = None

    def flush(self):
        self.stream.flush()


# ----------------------------------------------------------------------------
# HEBI calls
# ----------------------------------------------------------------------------
def _spatial_module():
    if str(HERE) not in sys.path:
        sys.path.append(str(HERE))
    try:
        return importlib.import_module(SPATIAL_MODULE)
    except Exception as exc:            # noqa: BLE001
        raise RuntimeError(
            f"The spatially regularised niche objective needs {SPATIAL_MODULE}.py next to "
            f"app.py (or HEBI_SPATIAL_MODULE set to its import path): {exc}")


def _accepts(func, name):
    import inspect
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _repo_spatial():
    """True when the installed HEBI has the spatial niche objective."""
    try:
        from hebi.inference.pipeline import encode_sections
    except Exception:                   # noqa: BLE001
        return False
    return _accepts(encode_sections, "niche_objective")


def _repo_merge_rule():
    """True when the installed HEBI's assign_niches has merge_rule."""
    try:
        from hebi.inference.niches import assign_niches
    except Exception:                   # noqa: BLE001
        return False
    return _accepts(assign_niches, "merge_rule")


@contextlib.contextmanager
def _niche_objective(job):
    """For the spatial objective, route encode_sections' call to continual_adapt
    through continual_adapt_spatial for this run only. Runs are executed one at
    a time, so the swap cannot leak into another job."""
    if job.mode != "continual" or job.niche_objective != "spatial" or _repo_spatial():
        yield
        return
    print("  niche objective: installed HEBI has no niche_objective; using spatial_niches.py")
    import hebi.inference.pipeline as pipeline_mod
    sn = _spatial_module()
    original = pipeline_mod.continual_adapt

    def adapt(**kwargs):
        print(f"  niche objective: spatially regularised {SPATIAL_SETTINGS}")
        return sn.continual_adapt_spatial(seed=job.seed, **SPATIAL_SETTINGS, **kwargs)

    pipeline_mod.continual_adapt = adapt
    try:
        yield
    finally:
        pipeline_mod.continual_adapt = original


def _preprocess_module():
    candidates = [PREPROCESS_MODULE, "hebi.data.preprocess", "hebi.preprocess",
                  "hebi.data.preprocessing", "preprocess"]
    errors = []
    for name in [c for c in candidates if c]:
        try:
            module = importlib.import_module(name)
            if hasattr(module, "preprocess_h5ad"):
                return module
            errors.append(f"{name}: no preprocess_h5ad")
        except Exception as exc:            # noqa: BLE001
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        "Could not import the HEBI preprocessing module (the file that defines "
        "preprocess_h5ad). Set HEBI_PREPROCESS_MODULE to its import path. Tried: "
        + "; ".join(errors))


def _prepare_input(job, entry):
    """Check one file before any heavy work, and make sure the coordinates sit in
    obsm['spatial'] as two columns: preprocessing accepts several key names, but
    the pipeline's alignment step reads obsm['spatial'] only."""
    import anndata as ad

    upload_path, name = entry["path"], entry["filename"]
    try:
        backed = ad.read_h5ad(upload_path, backed="r")
    except Exception as exc:                # noqa: BLE001
        raise ValueError(f"{name} could not be read as .h5ad ({exc}).")
    n_obs, n_vars = backed.shape
    matches = [k for k in backed.obsm.keys()
               if any(h in k.lower() for h in SPATIAL_HINTS)]
    has_xy = {"center_x", "center_y"} <= set(backed.obs.columns)
    n_cols = backed.obsm["spatial"].shape[1] if "spatial" in matches else None
    backed.file.close()

    if n_obs < MIN_CELLS:
        raise ValueError(f"{name} has only {n_obs} cells; at least {MIN_CELLS} are needed.")
    if not matches and not has_xy:
        raise ValueError(f"No spatial coordinates found in {name}. The file needs cell "
                         "positions in obsm['spatial'] (or obs columns center_x and center_y).")

    entry.update(n_cells_file=int(n_obs), n_genes_file=int(n_vars))

    coords_ok = matches == ["spatial"] and n_cols == 2
    if coords_ok and not job.normalise:
        return upload_path

    adata = ad.read_h5ad(upload_path)
    changed = _prepare_normalisation(job, entry, adata) if job.normalise else False
    if coords_ok and not changed:
        del adata
        return upload_path
    if coords_ok:
        fixed_dir = job.dir / "input_fixed"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        fixed = fixed_dir / upload_path.name
        adata.write_h5ad(fixed)
        del adata
        return fixed
    if "spatial" in matches:
        source = "spatial"
    elif matches:
        source = matches[0]
    else:
        source = None
    xy = (np.asarray(adata.obsm[source]) if source
          else adata.obs[["center_x", "center_y"]].to_numpy())
    for key in matches:
        del adata.obsm[key]
    adata.obsm["spatial"] = np.asarray(xy, dtype=np.float64)[:, :2]
    fixed_dir = job.dir / "input_fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    fixed = fixed_dir / upload_path.name
    adata.write_h5ad(fixed)
    print(f"  {name}: coordinates taken from {source or 'obs center_x/center_y'} "
          f"and stored as obsm['spatial'] (2 columns)")
    del adata
    return fixed


def _prepare_normalisation(job, entry, adata):
    """Make library-size normalisation possible when the user asks for it.

    preprocess_h5ad normalises raw counts itself. When X is already
    log-transformed, the counts are recovered first: from a layer holding raw
    counts if the file has one, otherwise by undoing log1p (expm1), and the
    result is normalised and logged here. preprocess then sees log-normalised
    data and leaves it alone. Returns True if adata was changed."""
    import scanpy as sc
    import scipy.sparse as sp

    pre = _preprocess_module()
    name = entry["filename"]
    state = pre.detect_normalisation(adata)["state"]
    entry["norm_action"], entry["norm_warn"] = None, None

    if state in ("raw_counts", "library_normalised"):
        return False                     # preprocess_h5ad handles these
    if state == "log_normalised":
        entry["norm_action"] = "already library-size normalised and log-transformed"
        return False

    # A raw-count layer, if the file carries one.
    candidates = [(f"layers['{k}']", adata.layers[k]) for k in adata.layers]
    if adata.raw is not None and adata.raw.n_vars == adata.n_vars:
        candidates.append(("raw.X", adata.raw.X))
    for label, matrix in candidates:
        probe = ad_like(adata, matrix)
        if pre.detect_normalisation(probe)["state"] == "raw_counts":
            adata.X = matrix.copy()
            entry["norm_action"] = f"raw counts taken from {label}, then normalised"
            print(f"  {name}: X was {state.replace('_', ' ')}; using raw counts "
                  f"from {label} so it can be normalised")
            return True

    if state != "log_transformed":
        entry["norm_warn"] = (f"library-size normalisation was not applied: the values look "
                              f"{state.replace('_', ' ')} and the file has no raw-count layer.")
        return False

    X = adata.X.copy() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float64)
    X = X.expm1() if sp.issparse(X) else np.expm1(X)
    vals = X.data if sp.issparse(X) else X[X > 0]
    sample = vals[:: max(1, vals.size // 200000)] if vals.size else vals
    whole = float(np.mean(np.abs(sample - np.round(sample)) < 1e-2)) if sample.size else 1.0
    adata.X = X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    entry["norm_action"] = "log1p reversed to recover counts, then normalised and log-transformed"
    print(f"  {name}: X was log-transformed; reversed log1p, then normalize_total + log1p "
          f"({whole:.0%} of recovered values are whole numbers)")
    if whole < 0.95:
        entry["norm_warn"] = (f"after reversing log1p only {whole:.0%} of the values were whole "
                              "numbers, so the file may not have been log1p of raw counts. "
                              "It was normalised anyway; check the result, or upload raw counts.")
    return True


def _select_hvg(job, paths):
    """Keep the top job.n_hvg highly variable genes (scanpy, flavor 'seurat', on
    log-normalised values). With several slides, genes are ranked as scanpy does
    with a batch key: first by the number of slides in which they are among the
    top genes, then by their median rank. When normalisation is requested and a
    slide holds counts, it is normalised on all genes before the genes are cut,
    so library size is measured on the whole transcriptome."""
    import anndata as ad
    import scanpy as sc

    from hebi.data.vocabulary import harmonise_adata

    pre = _preprocess_module()
    n = job.n_hvg
    tables, adatas, states = [], [], []
    for entry, path in zip(job.files, paths):
        adata = ad.read_h5ad(path)
        state = pre.detect_normalisation(adata)["state"]
        # Only genes HEBI can use compete: harmonise a small copy with the
        # repository's own function and keep the genes that survive it.
        probe = adata[: min(50, adata.n_obs)].copy()
        probe.var["_web_idx"] = np.arange(adata.n_vars)
        probe = harmonise_adata(probe, VOCAB_PATH, species=job.species)
        if "_web_idx" in probe.var:
            usable = np.zeros(adata.n_vars, bool)
            usable[probe.var["_web_idx"].to_numpy().astype(int)] = True
        else:
            names = {str(g).upper() for g in probe.var_names}
            usable = np.array([str(g).upper() in names for g in adata.var_names])
        print(f"  {entry['filename']}: {int(usable.sum()):,} of {adata.n_vars:,} genes are "
              "in the HEBI vocabulary; highly variable genes are chosen among them")
        entry["n_genes_usable"] = int(usable.sum())
        adata = adata[:, usable].copy()
        tmp = ad.AnnData(X=adata.X.copy(), var=adata.var[[]])
        tmp.var_names = [str(g).upper() for g in adata.var_names]
        tmp.var_names_make_unique()
        if state == "raw_counts":
            sc.pp.normalize_total(tmp, target_sum=1e4)
            sc.pp.log1p(tmp)
        elif state == "library_normalised":
            sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, flavor="seurat", n_top_genes=min(n, tmp.n_vars))
        disp = tmp.var["dispersions_norm"].fillna(-np.inf).to_numpy()
        order = np.argsort(-disp)
        rank = np.empty(len(order)); rank[order] = np.arange(len(order))
        tables.append({"rank": dict(zip(tmp.var_names, rank)),
                       "top": set(tmp.var_names[tmp.var["highly_variable"].to_numpy()])})
        adatas.append((adata, tmp if (job.normalise and state in ("raw_counts", "library_normalised")) else None))
        states.append(state)

    genes = set().union(*[t["rank"].keys() for t in tables])
    big = 10 ** 9
    score = {g: (-sum(g in t["top"] for t in tables),
                 float(np.median([t["rank"].get(g, big) for t in tables]))) for g in genes}
    chosen = set(sorted(genes, key=lambda g: score[g])[:n])

    out_dir = job.dir / "input_hvg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for entry, path, (adata, normed) in zip(job.files, paths, adatas):
        upper = np.array([str(g).upper() for g in adata.var_names])
        keep = np.isin(upper, list(chosen))
        if normed is not None:          # normalised on all genes, then cut
            adata.X = normed.X
            entry["norm_action"] = "library-size normalised on all genes and log-transformed, then cut to HVGs"
        adata = adata[:, keep].copy()
        entry["n_hvg_kept"] = int(adata.n_vars)
        dst = out_dir / Path(path).name
        adata.write_h5ad(dst)
        print(f"  {entry['filename']}: kept {adata.n_vars:,} highly variable genes "
              f"of {len(upper):,}")
        out.append(dst)
    return out


def ad_like(adata, matrix):
    import anndata as ad
    return ad.AnnData(X=matrix, obs=adata.obs[[]], var=adata.var[[]])


def _gene_names(adata):
    names = (adata.var["gene_symbol"] if "gene_symbol" in adata.var.columns
             else adata.var.index)
    return [str(g) for g in names]


def _cluster_stats(secs, codes, k):
    """Per-cluster sum, sum of squares and non-zero count for every shared gene,
    over all slides, in row chunks so a large section never needs a dense copy."""
    import scipy.sparse as sp

    g = secs[0]["X"].shape[1]
    sums = np.zeros((k, g), dtype=np.float64)
    sumsq = np.zeros((k, g), dtype=np.float64)
    nnz = np.zeros((k, g), dtype=np.float64)
    chunk = max(1000, int(2e7 // max(g, 1)))
    offset = 0
    for sec in secs:
        X, n = sec["X"], sec["n"]
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            block = X[start:stop]
            block = block.toarray() if sp.issparse(block) else np.asarray(block)
            block = np.nan_to_num(block.astype(np.float32, copy=False))
            onehot = sp.csr_matrix(
                (np.ones(stop - start, dtype=np.float32),
                 (codes[offset + start:offset + stop], np.arange(stop - start))),
                shape=(k, stop - start))
            sums += onehot @ block
            sumsq += onehot @ (block * block)
            nnz += onehot @ (block > 0).astype(np.float32)
        offset += n
    return sums, sumsq, nnz


def _markers(job, codes, k):
    """Welch t-test, each cluster against all other cells (the test scanpy's
    rank_genes_groups uses by default), on the expression in the AnnData."""
    sums, sumsq, nnz = _cluster_stats(job.secs, codes, k)
    sizes = np.bincount(codes, minlength=k).astype(np.float64)
    n = sizes.sum()
    n_in = np.maximum(sizes[:, None], 1.0)
    n_out = np.maximum(n - sizes[:, None], 1.0)

    mean_in = sums / n_in
    mean_out = (sums.sum(0, keepdims=True) - sums) / n_out
    var_in = np.maximum(sumsq - n_in * mean_in ** 2, 0) / np.maximum(n_in - 1, 1)
    out_sq = sumsq.sum(0, keepdims=True) - sumsq
    var_out = np.maximum(out_sq - n_out * mean_out ** 2, 0) / np.maximum(n_out - 1, 1)
    score = (mean_in - mean_out) / np.sqrt(var_in / n_in + var_out / n_out + 1e-12)
    pct_in = nnz / n_in
    pct_out = (nnz.sum(0, keepdims=True) - nnz) / n_out

    genes = job.gene_names
    markers, dot_genes = [], []
    for c in range(k):
        order = np.argsort(-score[c])[:TOP_MARKERS]
        markers.append([
            {"gene": genes[j], "score": round(float(score[c, j]), 2),
             "diff": round(float(mean_in[c, j] - mean_out[c, j]), 3),
             "pct_in": round(float(pct_in[c, j]), 4),
             "pct_out": round(float(pct_out[c, j]), 4)}
            for j in order if score[c, j] > 0])
        for j in order[:DOTPLOT_GENES_PER_CLUSTER]:
            if score[c, j] > 0 and j not in dot_genes:
                dot_genes.append(int(j))
    dotplot = {
        "genes": [genes[j] for j in dot_genes],
        "mean": np.round(mean_in[:, dot_genes], 4).tolist(),
        "pct": np.round(pct_in[:, dot_genes], 4).tolist(),
    }
    return markers, dotplot


def _cluster(job, k):
    """k-means on z0 of all slides together, with the repository's own function
    (the same joint fit cluster_sections does); cached per k."""
    with job.lock:
        if k in job.clusters:
            job.clusters.move_to_end(k)
            return job.clusters[k]
        from hebi.inference.cluster import cluster_embedding

        raw = np.asarray(cluster_embedding(
            job.z0, n_clusters=k, method=CLUSTER_METHOD,
            random_state=job.seed, resolution=1.0))
        uniq, codes = np.unique(raw, return_inverse=True)
        codes = codes.astype(np.int64)
        n_found = len(uniq)
        markers, dotplot = _markers(job, codes, n_found)
        sizes = np.bincount(codes, minlength=n_found)
        entry = {
            "codes": codes.astype(np.uint8),
            "names": [str(u) for u in uniq],
            "info": {
                "k": int(k),
                "obs_key": f"z0_{CLUSTER_METHOD}_{k}",
                "names": [str(u) for u in uniq],
                "sizes": sizes.astype(int).tolist(),
                "markers": markers,
                "dotplot": dotplot,
            },
        }
        job.clusters[k] = entry
        while len(job.clusters) > 6:
            job.clusters.popitem(last=False)
        return entry


# How prototypes are merged into k niches (k below the number of learned prototypes).
#   keep        repository default: average linkage, tiny prototypes kept as leaves
#   absorb      the same, with tiny_policy="absorb_prototype"
#   size_space  niche_merge.py: the same guards and distances, but the merge cost
#               is scaled by cluster size (Ward-like) and reduced by spatial contact
#               (experimental; niche_merge.py must sit next to app.py)
MERGE_RULES = {"keep": "keep", "absorb_prototype": "absorb_prototype", "size_space": "keep"}
MERGE_SUFFIX = {"keep": "", "absorb_prototype": "_absorb", "size_space": "_sizespace"}
SIZE_SPACE_SETTINGS = {"gamma": 1.0, "lam": 0.5, "knn": 10}
TINY_POLICIES = tuple(MERGE_RULES)


def _niche_entry(job, k=0, policy="keep"):
    """k = 0: the learned Level-2 prototypes (argmax of attention), shown with their
    prototype number. k > 0: niches from assign_niches at that resolution (prototypes
    merged or refined, mapping shared across slides)."""
    if policy not in TINY_POLICIES:
        raise HTTPException(422, f"policy must be one of {TINY_POLICIES}")
    key = (0, "keep") if k == 0 else (k, policy)
    with job.lock:
        if key in job.niches:
            return job.niches[key]
        res = job.niche_res.get(policy, {})
        if k == 0:
            raw, paths = job.niche_codes, None
        elif k in res:
            raw, paths = res[k]
        else:
            raise HTTPException(422, f"No niche assignment at k={k} ({policy}). Available: "
                                f"{sorted(res)}")
        uniq, codes = np.unique(raw, return_inverse=True)
        codes = codes.astype(np.int64)
        markers, dotplot = _markers(job, codes, len(uniq))
        names = [str(u) for u in uniq]
        obs_key = ("hebi_niche" if k == 0 else
                   f"hebi_niche_k{k}" + MERGE_SUFFIX[policy])
        job.niches[key] = {
            "codes": codes.astype(np.uint8), "names": names,
            "info": {"k": len(uniq), "obs_key": obs_key, "policy": policy if k else None,
                     "names": names,
                     "paths": [paths.get(int(u), "") for u in uniq] if paths else None,
                     "sizes": np.bincount(codes, minlength=len(uniq)).astype(int).tolist(),
                     "markers": markers, "dotplot": dotplot},
        }
        return job.niches[key]


def _assign_niche_resolutions(job, sections, ctx, processed, pt_names, rule="keep"):
    """Run the repository's assign_niches for every feasible number of niches,
    from the coarsest the merge rules allow up to a few splits beyond the learned
    prototypes. Each k is tried on its own, so one infeasible k does not stop the
    rest. tiny_policy is passed to assign_niches: "keep" (paper default) or
    "absorb_prototype" (prototypes under 0.5% of cells in every section they occur
    in are attached whole to their nearest compatible prototype before merging).
    rule "size_space" uses niche_merge.size_space_merge for the merge order.
    Returns {k: (codes, {label: prototype path})}."""
    tiny_policy = MERGE_RULES[rule]
    try:
        from hebi.inference.niches import assign_niches
    except Exception as exc:            # noqa: BLE001
        print(f"  niche assignment unavailable ({exc}); showing learned prototypes only")
        return {}
    import io as _io

    names = [e["label"] for e in job.files]
    occupied = len(np.unique(job.niche_codes))
    merge_ctx = contextlib.nullcontext()
    if rule == "size_space" and _repo_merge_rule():
        return _assign_loop(job, sections, ctx, processed, pt_names, names, occupied,
                            tiny_policy, rule, assign_niches,
                            extra={"merge_rule": "size_space",
                                   "merge_gamma": SIZE_SPACE_SETTINGS["gamma"],
                                   "merge_lam": SIZE_SPACE_SETTINGS["lam"],
                                   "merge_knn": SIZE_SPACE_SETTINGS["knn"]})
    if rule == "size_space":
        try:
            import niche_merge
        except Exception as exc:        # noqa: BLE001
            print(f"  size/space-aware merging unavailable ({exc})")
            return {}
        merge_ctx = niche_merge.size_space_merge(sections, **SIZE_SPACE_SETTINGS)
    with merge_ctx:
        return _assign_loop(job, sections, ctx, processed, pt_names, names, occupied,
                            tiny_policy, rule, assign_niches)


def _assign_loop(job, sections, ctx, processed, pt_names, names, occupied,
                 tiny_policy, rule, assign_niches, extra=None):
    import io as _io
    out = {}
    for k in range(2, occupied + 6):
        quiet = _io.StringIO()
        try:
            with contextlib.redirect_stderr(quiet):
                assign_niches(sections, ctx, str(processed), pt_names, n_niches=k,
                              section_names=names, seed=job.seed, tiny_policy=tiny_policy,
                              **(extra or {}))
        except Exception as exc:        # noqa: BLE001
            if k > occupied:
                print(f"  niche assignment stops at k={k - 1}: {exc}")
                break
            continue
        key = f"niche_k{k}"
        if not all(key in a.obs for a in sections):
            continue                    # k below the coarsest feasible resolution
        codes = np.concatenate([a.obs[key].astype(int).to_numpy() for a in sections])
        paths = {}
        for a in sections:
            for lab, path in zip(a.obs[key].astype(int), a.obs[f"{key}_path"].astype(str)):
                paths.setdefault(int(lab), path)
        out[k] = (codes, paths)
        for a in sections:
            del a.obs[key], a.obs[f"{key}_path"]
    if out:
        print(f"  niche assignment ({rule}) at k = {', '.join(map(str, sorted(out)))} "
              f"({occupied} learned prototypes)")
    return out


def _drop_encode_pass(sections):
    """The extra encode pass assign_niches stores; not needed after assignment."""
    for a in sections:
        for key in ("niche_attention", "z2_enc"):
            a.obsm.pop(key, None)
        if "prototypes_enc" in a.obs:
            del a.obs["prototypes_enc"]


def _embed(job):
    """2-D view of z0 for the second panel, on a random subsample of cells."""
    n = job.z0.shape[0]
    rng = np.random.default_rng(job.seed)
    idx = (np.sort(rng.choice(n, EMBED_MAX_CELLS, replace=False))
           if n > EMBED_MAX_CELLS else np.arange(n))
    z = np.asarray(job.z0[idx], dtype=np.float32)
    try:
        import umap

        xy = umap.UMAP(n_neighbors=15, min_dist=0.3,
                       random_state=job.seed).fit_transform(z)
        name = "UMAP"
    except Exception as exc:                # noqa: BLE001
        print(f"  UMAP unavailable ({exc}); using PCA for the embedding panel")
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=job.seed).fit_transform(z)
        name = "PCA"
    job.embed_idx = idx.astype(np.uint32)
    job.embed_xy = np.asarray(xy, dtype=np.float32)
    job.meta.update(embedding=name, n_embedded=int(len(idx)))


def _stage(job, name):
    if not hasattr(job, "_stage_start"):
        job._stage_start = time.time()
    now = time.time()
    if job.stage is not None:
        job.timings[job.stage] = round(now - job._stage_start, 1)
    job.stage, job._stage_start = name, now


def _shared_genes(sections):
    """Genes present on every slide, in the order of the first slide, and for
    each slide the columns that hold them."""
    names = [_gene_names(a) for a in sections]
    upper = [[g.upper() for g in n] for n in names]
    shared = set(upper[0]).intersection(*[set(u) for u in upper[1:]])
    genes = [g for g in dict.fromkeys(upper[0]) if g in shared]
    display = {}
    for g, u in zip(names[0], upper[0]):
        display.setdefault(u, g)
    columns = []
    for u in upper:
        first = {}
        for j, g in enumerate(u):
            first.setdefault(g, j)
        columns.append(np.array([first[g] for g in genes], dtype=np.int64))
    return [display[g] for g in genes], columns


def run_job(job):
    import sys

    job.status = "running"
    tee = _Tee(job, sys.stdout)
    try:
        with contextlib.redirect_stdout(tee):
            import torch

            _stage(job, "read")
            if not Path(VOCAB_PATH).exists():
                raise RuntimeError(f"Gene vocabulary not found at {VOCAB_PATH}. "
                                   "Set HEBI_VOCAB to the vocabulary JSON.")
            h5ad_paths = [_prepare_input(job, entry) for entry in job.files]
            if job.n_hvg:
                h5ad_paths = _select_hvg(job, h5ad_paths)
            total = sum(e["n_cells_file"] for e in job.files)
            if total > MAX_CELLS:
                raise ValueError(f"The upload has {total:,} cells in total; this server "
                                 f"accepts up to {MAX_CELLS:,} per run.")

            _stage(job, "graph")
            processed = job.dir / "processed"
            out_dir = job.dir / "out"
            pre = _preprocess_module()
            pt_names, pt_metas = [], []
            for entry, h5ad_path in zip(job.files, h5ad_paths):
                written = pre.preprocess_h5ad(
                    h5ad_path, processed, VOCAB_PATH, task="clustering",
                    k=job.k_graph, n_metis=job.n_metis, species=job.species, overwrite=True,
                    normalise=job.normalise)
                if not written or not Path(written[0]).exists():
                    raise RuntimeError(f"Preprocessing wrote no section for "
                                       f"{entry['filename']}; see the run log.")
                pt_names.append(Path(written[0]).name)
                pt_metas.append(torch.load(written[0], map_location="cpu",
                                           weights_only=False)["meta"])
            states = [m.get("normalisation_on_input", "unknown") for m in pt_metas]
            count_like = [s in ("raw_counts", "library_normalised") for s in states]
            same_form = len(set(count_like)) == 1

            _stage(job, "adapt" if job.mode == "continual" else "encode")
            from hebi.inference.pipeline import encode_sections

            # The AnnData copies are what markers and gene colouring read. They are
            # normalised + logged only where the file arrived as counts: by the
            # pipeline itself when all slides agree, per slide below when they differ.
            with _niche_objective(job):
                sections, ctx = encode_sections(
                checkpoint_path=CHECKPOINT_PATH if job.init == "pretrained" else None,
                data_dir=str(processed),
                section_files=pt_names, h5ad_files=[str(p) for p in h5ad_paths],
                vocab_path=VOCAB_PATH, mode=job.mode, init=job.init, grn=False,
                num_epochs=job.epochs, lr=job.lr, verbose=1,
                section_labels=[e["label"] for e in job.files],
                species=job.species, output_dir=str(out_dir), device=DEVICE,
                seed=job.seed, preprocess=count_like[0] if same_form else False,
                **({"niche_objective": "spatial",
                    "spatial_weight": SPATIAL_SETTINGS["spatial_weight"],
                    "sharp_weight": SPATIAL_SETTINGS["sharp_weight"]}
                   if job.mode == "continual" and job.niche_objective == "spatial"
                   and _repo_spatial() else {}))
            device_used = str(getattr(ctx.get("trainer"), "device", DEVICE or "auto"))
            if job.mode == "continual":
                from hebi.inference.cluster import KEY_PROTOTYPES as _KP
                job.niche_codes = np.concatenate(
                    [a.obs[_KP].astype(int).to_numpy() for a in sections])
                _stage(job, "niches")
                job.niche_res = {
                    policy: _assign_niche_resolutions(job, sections, ctx, processed,
                                                      pt_names, rule=policy)
                    for policy in TINY_POLICIES}
                _drop_encode_pass(sections)
            ctx.clear()
            del ctx
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not same_form:
                import scanpy as sc
                for adata, counts in zip(sections, count_like):
                    if counts:
                        sc.pp.normalize_total(adata, target_sum=1e4)
                        sc.pp.log1p(adata)

            try:
                from hebi.inference.cluster import KEY_PROTOTYPES
            except Exception:               # noqa: BLE001
                KEY_PROTOTYPES = None
            for adata in sections:
                # z1/z2 are not shown; the prototype column is kept only as niches.
                for key in ("z1", "z2"):
                    adata.obsm.pop(key, None)
                if KEY_PROTOTYPES and KEY_PROTOTYPES in adata.obs:
                    del adata.obs[KEY_PROTOTYPES]

            z0 = np.concatenate([np.asarray(a.obsm["z0"], dtype=np.float32) for a in sections])
            if not np.isfinite(z0).all():
                raise RuntimeError("The encoder returned non-finite values for this upload.")
            genes, columns = _shared_genes(sections)
            if not genes:
                raise ValueError("The slides have no gene in common, so they cannot be "
                                 "analysed together.")
            job.secs = []
            for adata, cols in zip(sections, columns):
                whole = len(cols) == adata.n_vars and np.array_equal(cols, np.arange(adata.n_vars))
                job.secs.append({"adata": adata, "cols": cols, "n": int(adata.n_obs),
                                 "X": adata.X if whole else adata.X[:, cols]})
            job.z0 = z0
            job.coords = np.ascontiguousarray(np.concatenate(
                [np.asarray(a.obsm["spatial"], dtype=np.float32)[:, :2] for a in sections]))
            job.sec_of = np.repeat(np.arange(len(sections)),
                                   [s["n"] for s in job.secs]).astype(np.uint8)
            job.gene_names = genes
            job.gene_index = {g.upper(): j for j, g in enumerate(genes)}

            warnings, sec_meta = [], []
            many = len(sections) > 1
            for entry, adata, state, pm in zip(job.files, sections, states, pt_metas):
                who = f"{entry['label']}: " if many else ""
                if state == "scaled":
                    warnings.append(who + "the expression matrix looks centred or z-scored, "
                                    "so no normalisation was applied. Results are more reliable "
                                    "from raw counts or log-transformed values.")
                elif state == "unknown":
                    warnings.append(who + "the expression values could not be classified as "
                                    "counts or log-transformed, so they were used as they are.")
                if entry.get("norm_warn"):
                    warnings.append(who + entry["norm_warn"])
                if job.n_hvg and entry.get("n_genes_usable") is not None:
                    if entry["n_genes_usable"] < 0.5 * entry["n_genes_file"]:
                        warnings.append(who + f"only {entry['n_genes_usable']:,} of "
                                        f"{entry['n_genes_file']:,} genes in the file are in the HEBI "
                                        "gene vocabulary; the highly variable genes were chosen among "
                                        "those. For mouse data, check that Species is set to Mouse.")
                elif adata.n_vars < 0.5 * entry["n_genes_file"]:
                    warnings.append(who + f"only {adata.n_vars:,} of {entry['n_genes_file']:,} "
                                    "genes in the file are in the HEBI gene vocabulary.")
                sec_meta.append({
                    "name": entry["label"], "filename": entry["filename"],
                    "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
                    "n_genes_file": entry["n_genes_file"],
                    "normalisation_on_input": state,
                    "normalisation_applied": pm.get("normalisation_applied", []),
                    "normalisation_note": entry.get("norm_action"),
                    "n_hvg_kept": entry.get("n_hvg_kept"),
                    "reach_cell_diameters": pm.get("reach_cell_diameters")})
            all_normalised = job.normalise and not any(e.get("norm_warn") for e in job.files)
            if many and not same_form and not all_normalised:
                warnings.append("The slides arrived in different forms (some as counts, some "
                                "already log-transformed). Each was handled in its own form, "
                                "but expression levels may not be comparable between slides.")
            if many and any(len(genes) < s["n_genes"] for s in sec_meta):
                warnings.append(f"The slides have different gene panels. Marker genes and gene "
                                f"colouring use the {len(genes):,} genes they share.")
            n_total = int(z0.shape[0])
            job.meta.update(
                filename=(job.files[0]["filename"] if not many else f"{len(sections)} slides"),
                sections=sec_meta, n_cells=n_total, n_genes=len(genes),
                species=job.species, seed=job.seed, device=device_used,
                mode=job.mode, epochs=job.epochs if job.mode == "continual" else 0,
                lr=job.lr if job.mode == "continual" else None,
                has_niches=job.niche_codes is not None, normalise=job.normalise, n_hvg=job.n_hvg,
                niche_ks={p: sorted(r) for p, r in job.niche_res.items()},
                init=job.init, n_metis=job.n_metis,
                niche_objective=job.niche_objective if job.mode == "continual" else None,
                spatial_settings=SPATIAL_SETTINGS if (job.mode == "continual"
                                                      and job.niche_objective == "spatial") else None,
                n_communities=[int(m.get("n_metis_clusters", 0)) for m in pt_metas],
                checkpoint=Path(CHECKPOINT_PATH).name if job.init == "pretrained" else None,
                n_prototypes=int(len(np.unique(job.niche_codes))) if job.niche_codes is not None else 0,
                k_graph=int(pt_metas[0].get("k_neighbors", job.k_graph)),
                warnings=warnings, default_k=job.n_clusters,
                min_k=MIN_CLUSTERS, max_k=min(MAX_CLUSTERS, max(MIN_CLUSTERS, n_total // 20)),
            )

            # The .pt files, the random checkpoint and the uploads are not needed again.
            for path in (processed, out_dir, job.dir / "input", job.dir / "input_fixed",
                         job.dir / "input_hvg"):
                shutil.rmtree(path, ignore_errors=True)

            _stage(job, "cluster")
            _cluster(job, job.n_clusters)
            if job.niche_codes is not None:
                _niche_entry(job)

            _stage(job, "embed")
            _embed(job)
            _stage(job, "finished")
        job.status = "done"
    except BaseException as exc:            # noqa: BLE001  (the pipeline raises SystemExit)
        job.error = str(exc) or type(exc).__name__
        job.log.append(traceback.format_exc())
        job.status = "error"
        shutil.rmtree(job.dir, ignore_errors=True)
    finally:
        # Give GPU memory back whether the run finished or failed (e.g. out of memory).
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                   # noqa: BLE001
            pass
        evict_old_jobs()


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
app = FastAPI(title="HEBI zero-shot explorer")


def _binary(array):
    return Response(content=np.ascontiguousarray(array).tobytes(),
                    media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    page = HERE / "index.html"
    if not page.exists():
        raise HTTPException(500, "index.html is missing next to app.py")
    return FileResponse(page, media_type="text/html")


def _spatial_available():
    try:
        _spatial_module()
        return True
    except BaseException:               # noqa: BLE001
        return False


@app.get("/api/health")
def health():
    info = {"ok": True, "vocab_found": Path(VOCAB_PATH).exists(),
            "max_upload_gb": MAX_UPLOAD_GB, "max_cells": MAX_CELLS, "max_files": MAX_FILES,
            "examples": [{"name": n, "size": p.stat().st_size} for n, p in _examples().items()],
            "default_k_graph": DEFAULT_K_GRAPH, "default_seed": DEFAULT_SEED,
            "default_n_clusters": DEFAULT_N_CLUSTERS,
            "pretrained_available": bool(CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists()),
            "default_n_metis": DEFAULT_N_METIS,
            "spatial_objective_available": _spatial_available(),
            "default_epochs": DEFAULT_EPOCHS, "max_epochs": MAX_EPOCHS, "default_lr": DEFAULT_LR}
    try:
        import torch
        info["device"] = DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
        importlib.import_module("hebi.inference.pipeline")
        _preprocess_module()
        info["hebi"] = True
    except BaseException as exc:            # noqa: BLE001
        info.update(ok=False, hebi=False, error=str(exc))
    if not info["vocab_found"]:
        info.update(ok=False, error=f"Gene vocabulary not found at {VOCAB_PATH}")
    return info


@app.post("/api/jobs")
def create_job(species: str = "Human", k_graph: int = DEFAULT_K_GRAPH,
               seed: int = DEFAULT_SEED, n_clusters: int = DEFAULT_N_CLUSTERS,
               mode: str = "zero_shot", epochs: int = DEFAULT_EPOCHS,
               lr: float = DEFAULT_LR, normalise: bool = False, n_hvg: int = 0,
               init: str = "random", n_metis: int = DEFAULT_N_METIS,
               niche_objective: str = "original"):
    """Open a run. Slides are then sent one by one to /files, and /start queues it."""
    if niche_objective not in NICHE_OBJECTIVES:
        raise HTTPException(422, f"niche_objective must be one of {NICHE_OBJECTIVES}")
    if niche_objective == "spatial" and mode == "continual":
        try:
            _spatial_module()
        except RuntimeError as exc:
            raise HTTPException(422, str(exc))
    if not 8 <= n_metis <= 8192:
        raise HTTPException(422, "n_metis must be between 8 and 8192")
    if init not in INITS:
        raise HTTPException(422, f"init must be one of {INITS}")
    if init == "pretrained" and not (CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists()):
        raise HTTPException(422, "No pretrained checkpoint on this server. Set HEBI_CHECKPOINT "
                                 "to the .pt file and restart it.")
    if n_hvg and not 100 <= n_hvg <= 20000:
        raise HTTPException(422, "n_hvg must be 0 (all genes) or between 100 and 20000")
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    if mode == "continual" and not 1 <= epochs <= MAX_EPOCHS:
        raise HTTPException(422, f"epochs must be between 1 and {MAX_EPOCHS}")
    if mode == "continual" and not LR_RANGE[0] <= lr <= LR_RANGE[1]:
        raise HTTPException(422, f"lr must be between {LR_RANGE[0]:g} and {LR_RANGE[1]:g}")
    if species not in ("Human", "Mouse"):
        raise HTTPException(422, "species must be Human or Mouse")
    if not 3 <= k_graph <= 100:
        raise HTTPException(422, "k_graph must be between 3 and 100")
    if not MIN_CLUSTERS <= n_clusters <= MAX_CLUSTERS:
        raise HTTPException(422, f"n_clusters must be between {MIN_CLUSTERS} and {MAX_CLUSTERS}")
    job = Job(uuid.uuid4().hex[:12], species, k_graph, seed, n_clusters, mode, epochs, lr,
              normalise, n_hvg, init, n_metis, niche_objective)
    (job.dir / "input").mkdir(parents=True, exist_ok=True)
    with JOBS_LOCK:
        JOBS[job.id] = job
    evict_old_jobs()
    return {"job_id": job.id}


@app.post("/api/jobs/{job_id}/files")
async def add_file(job_id: str, request: Request, filename: str):
    """The request body is one raw .h5ad file (one slide)."""
    job = get_job(job_id)
    if job.status != "uploading":
        raise HTTPException(409, "This run has already started.")
    if not filename.lower().endswith(".h5ad"):
        raise HTTPException(422, f"{filename} is not an .h5ad file")
    if len(job.files) >= MAX_FILES:
        raise HTTPException(422, f"A run takes at most {MAX_FILES} slides.")

    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).stem)[:80] or "section"
    taken = {e["stem"] for e in job.files}
    base, i = stem, 2
    while stem in taken:
        stem, i = f"{base}_{i}", i + 1
    labels = {e["label"] for e in job.files}
    label, i = Path(name).stem[:80] or stem, 2
    while label in labels:
        label, i = f"{Path(name).stem[:80]} ({i})", i + 1
    upload_path = job.dir / "input" / f"{stem}.h5ad"

    limit = int(MAX_UPLOAD_GB * 1024 ** 3)
    received = 0
    try:
        with open(upload_path, "wb") as handle:
            async for chunk in request.stream():
                received += len(chunk)
                if received > limit:
                    raise HTTPException(413, f"{name} is larger than {MAX_UPLOAD_GB:g} GB.")
                handle.write(chunk)
        if received == 0:
            raise HTTPException(422, f"{name} arrived empty.")
    except BaseException:
        upload_path.unlink(missing_ok=True)
        raise
    job.files.append({"filename": name, "label": label, "stem": stem,
                      "path": upload_path, "size": received})
    return {"files": len(job.files)}


@app.post("/api/jobs/{job_id}/example")
def add_example(job_id: str, name: str):
    """Add one of the server's example sections to the run, without an upload."""
    job = get_job(job_id)
    if job.status != "uploading":
        raise HTTPException(409, "This run has already started.")
    src = _examples().get(Path(name).name)
    if src is None:
        raise HTTPException(404, f"No example called {name}")
    if len(job.files) >= MAX_FILES:
        raise HTTPException(422, f"A run takes at most {MAX_FILES} slides.")
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", src.stem)[:80] or "section"
    base, i = stem, 2
    while stem in {e["stem"] for e in job.files}:
        stem, i = f"{base}_{i}", i + 1
    dst = job.dir / "input" / f"{stem}.h5ad"
    shutil.copyfile(src, dst)
    job.files.append({"filename": src.name, "label": src.stem[:80], "stem": stem,
                      "path": dst, "size": dst.stat().st_size})
    return {"files": len(job.files)}


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str):
    job = get_job(job_id)
    if job.status != "uploading":
        raise HTTPException(409, "This run has already started.")
    if not job.files:
        raise HTTPException(422, "No file was uploaded for this run.")
    job.status = "queued"
    job.created = time.time()
    EXECUTOR.submit(run_job, job)
    return {"job_id": job.id, "files": len(job.files)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    ahead = sum(1 for j in JOBS.values()
                if j.status in ("queued", "running") and j.created < job.created)
    stages = [s for s in STAGES if s not in ("adapt", "niches") or job.mode == "continual"]
    return {"status": job.status, "stage": job.stage, "stages": stages, "detail": job.detail,
            "error": job.error, "queue_ahead": ahead if job.status == "queued" else 0,
            "elapsed": round(time.time() - job.created, 1), "log": job.log[-80:]}


@app.get("/api/jobs/{job_id}/meta")
def job_meta(job_id: str):
    job = get_job(job_id, need_done=True)
    return JSONResponse({**job.meta, "genes": job.gene_names, "timings": job.timings})


@app.get("/api/jobs/{job_id}/coords")
def job_coords(job_id: str):
    return _binary(get_job(job_id, need_done=True).coords)


@app.get("/api/jobs/{job_id}/embedding/index")
def job_embedding_index(job_id: str):
    return _binary(get_job(job_id, need_done=True).embed_idx)


@app.get("/api/jobs/{job_id}/embedding/xy")
def job_embedding_xy(job_id: str):
    return _binary(get_job(job_id, need_done=True).embed_xy)


@app.get("/api/jobs/{job_id}/sections")
def job_sections(job_id: str):
    """For every cell, the index of the slide it belongs to."""
    return _binary(get_job(job_id, need_done=True).sec_of)


def _check_k(job, k):
    if not job.meta["min_k"] <= k <= job.meta["max_k"]:
        raise HTTPException(422, f"k must be between {job.meta['min_k']} and {job.meta['max_k']}")


@app.get("/api/jobs/{job_id}/clusters")
def job_clusters(job_id: str, k: int):
    job = get_job(job_id, need_done=True)
    _check_k(job, k)
    return JSONResponse(_cluster(job, k)["info"])


@app.get("/api/jobs/{job_id}/labels")
def job_labels(job_id: str, k: int):
    job = get_job(job_id, need_done=True)
    _check_k(job, k)
    return _binary(_cluster(job, k)["codes"])


def _need_niches(job):
    if job.niche_codes is None:
        raise HTTPException(404, "Niches exist only for runs trained on the data (continual).")


@app.get("/api/jobs/{job_id}/niches")
def job_niches(job_id: str, k: int = 0, policy: str = "keep"):
    job = get_job(job_id, need_done=True)
    _need_niches(job)
    return JSONResponse(_niche_entry(job, k, policy)["info"])


@app.get("/api/jobs/{job_id}/niche_labels")
def job_niche_labels(job_id: str, k: int = 0, policy: str = "keep"):
    job = get_job(job_id, need_done=True)
    _need_niches(job)
    return _binary(_niche_entry(job, k, policy)["codes"])


@app.get("/api/jobs/{job_id}/gene")
def job_gene(job_id: str, name: str):
    import scipy.sparse as sp

    job = get_job(job_id, need_done=True)
    j = job.gene_index.get(name.upper())
    if j is None:
        raise HTTPException(404, f"Gene {name} is not in this dataset")
    parts = []
    for sec in job.secs:
        column = sec["X"][:, j]
        column = column.toarray() if sp.issparse(column) else np.asarray(column)
        parts.append(np.nan_to_num(column.astype(np.float32)).ravel())
    return _binary(np.concatenate(parts))


def _all_niche_keys(job):
    keys = [(0, "keep")]
    for pol in TINY_POLICIES:
        keys += [(k, pol) for k in sorted(job.niche_res.get(pol, {}))]
    return keys


def _safe_stem(job):
    if len(job.files) > 1:
        return f"hebi_{len(job.files)}_slides"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(job.files[0]["filename"]).stem)[:80] or "section"


def _csv_field(text):
    text = str(text)
    return '"' + text.replace('"', '""') + '"' if ("," in text or '"' in text) else text


@app.get("/api/jobs/{job_id}/download.csv")
def download_csv(job_id: str, k: int):
    job = get_job(job_id, need_done=True)
    _check_k(job, k)
    entry = _cluster(job, k)
    names = np.asarray(entry["names"], dtype=object)[entry["codes"]]
    header = "slide,cell_id,x,y," + entry["info"]["obs_key"]
    niche_cols = []
    if job.niche_codes is not None:
        for nk, pol in _all_niche_keys(job):
            ne = _niche_entry(job, nk, pol)
            niche_cols.append(np.asarray(ne["names"], dtype=object)[ne["codes"]])
            header += "," + ne["info"]["obs_key"]
    lines = [header]
    row = 0
    for meta, sec in zip(job.meta["sections"], job.secs):
        slide = _csv_field(meta["name"])
        for cell in sec["adata"].obs_names.astype(str):
            x, y = job.coords[row]
            extra = "".join(f",{col[row]}" for col in niche_cols)
            lines.append(f"{slide},{_csv_field(cell)},{x:.6g},{y:.6g},{names[row]}{extra}")
            row += 1
    return Response("\n".join(lines) + "\n", media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{_safe_stem(job)}_hebi_clusters_k{k}.csv"'})


@app.get("/api/jobs/{job_id}/download.h5ad")
def download_h5ad(job_id: str, k: int):
    """One .h5ad. Several slides are stacked, with obs['slide'] naming each cell's slide
    and only the genes the slides share."""
    import anndata as ad
    import pandas as pd

    job = get_job(job_id, need_done=True)
    _check_k(job, k)
    entry = _cluster(job, k)
    with job.lock:
        key = entry["info"]["obs_key"]
        labels = pd.Categorical(
            np.asarray(entry["names"], dtype=object)[entry["codes"]], categories=entry["names"])
        if len(job.secs) == 1:
            out_adata = job.secs[0]["adata"]
        else:
            parts = []
            for sec in job.secs:
                part = sec["adata"][:, sec["cols"]].copy()
                part.var_names = job.gene_names
                parts.append(part)
            out_adata = ad.concat(parts, join="inner", label="slide", index_unique="-",
                                  keys=[m["name"] for m in job.meta["sections"]])
        out_adata.obs[key] = labels
        if job.niche_codes is not None:
            for nk, pol in _all_niche_keys(job):
                ne = _niche_entry(job, nk, pol)
                out_adata.obs[ne["info"]["obs_key"]] = pd.Categorical(
                    np.asarray(ne["names"], dtype=object)[ne["codes"]], categories=ne["names"])
                if nk:
                    out_adata.uns.setdefault("hebi_niche_paths", {})[ne["info"]["obs_key"]] = {
                        n: p for n, p in zip(ne["names"], ne["info"]["paths"] or [])}
        trained = job.mode == "continual"
        out_adata.uns["hebi_web"] = {
            "mode": job.mode, "init": job.init, "seed": int(job.seed),
            "checkpoint": Path(CHECKPOINT_PATH).name if job.init == "pretrained" else "",
            "epochs": int(job.epochs) if trained else 0,
            "normalise_requested": bool(job.normalise),
            "niche_objective": job.niche_objective if trained else "",
            "n_hvg": int(job.n_hvg),
            "lr": float(job.lr) if trained else 0.0,
            "k_graph": int(job.meta["k_graph"]), "species": job.species,
            "n_metis": int(job.n_metis),
            "cluster_method": CLUSTER_METHOD,
            "slides": [m["name"] for m in job.meta["sections"]],
            "normalisation_on_input": [str(m["normalisation_on_input"]) for m in job.meta["sections"]],
            "note": ("obsm['z0'] is the HEBI cell-level embedding after "
                     f"{job.epochs} epochs of training from "
                     f"{'pretrained' if job.init == 'pretrained' else 'random'} weights on these slides "
                     f"{'(niche level re-initialised before training)' if job.init == 'pretrained' else ''}. "
                     "obs clusters are k-means on z0, fitted over all slides together; "
                     "obs['hebi_niche'] is the learned Level-2 prototype of each cell."
                     if trained else
                     "obsm['z0'] is the HEBI cell-level embedding from one forward pass "
                     f"with {'pretrained' if job.init == 'pretrained' else 'untrained'} weights. "
                     "obs clusters are k-means on z0, fitted over "
                     "all slides together."),
        }
        job.dir.mkdir(parents=True, exist_ok=True)
        out = job.dir / f"{_safe_stem(job)}_hebi_{'continual' if trained else 'zero_shot'}.h5ad"
        out_adata.write_h5ad(out)
    return FileResponse(out, media_type="application/octet-stream", filename=out.name)


@app.exception_handler(HTTPException)
async def http_error(_request, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    import uvicorn

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"HEBI zero-shot explorer on http://{HOST}:{PORT}")
    print(f"  vocabulary: {VOCAB_PATH}")
    print(f"  checkpoint: {CHECKPOINT_PATH or '(none: random initialisation only)'}"
          + ("" if not CHECKPOINT_PATH or Path(CHECKPOINT_PATH).exists() else "  NOT FOUND"))
    print(f"  work dir:   {WORK_DIR}")
    # Import the heavy modules now, so the page's first check answers at once
    # and any import problem shows here in the terminal.
    print("  loading HEBI modules ...", flush=True)
    try:
        importlib.import_module("hebi.inference.pipeline")
        print(f"  HEBI ready (preprocessing from {_preprocess_module().__name__})")
    except BaseException as exc:            # noqa: BLE001
        print(f"  HEBI NOT READY: {exc}")
        print("  The page will open, but uploads stay off until this is fixed.")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
