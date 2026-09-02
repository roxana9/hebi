#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering and niche assignment on HEBI representations.

Three label sets are produced, matching the Methods:

  prototypes          Level-2 attention argmax, straight from the forward
                      pass. No post-hoc clustering.
  cell clusters       k-means / mclust / leiden on the cell-level z0.
  composition niches  k-means on local prototype-composition vectors,
                      fitted jointly across sections so labels are aligned.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

# obs keys written into AnnData.
KEY_CELL_CLUSTERS = "cell_clusters"
KEY_PROTOTYPES = "prototypes"
KEY_COMPOSITION_NICHES = "composition_niches"

CLUSTER_METHODS = ("kmeans", "mclust", "leiden")


# ----------------------------------------------------------------------
# Clustering backends
# ----------------------------------------------------------------------
def cluster_embedding(
    X,
    n_clusters: int,
    method: str = "kmeans",
    random_state: int = 42,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_pca: int = 20,
):
    """Cluster an embedding matrix and return integer labels.

    method:
        "kmeans"  scikit-learn KMeans (default; no extra dependency)
        "mclust"  R mclust via rpy2, EEE model, run on a PCA of the
                  embedding. Requires R and the mclust package.
        "leiden"  scanpy leiden. Leiden is resolution-based rather than
                  k-based, so when n_clusters is given the resolution is
                  found by bisection; the result is not guaranteed to hit
                  n_clusters exactly.
    """
    X = np.asarray(X, dtype=np.float64)

    if method == "kmeans":
        return KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state
        ).fit_predict(X)

    if method == "mclust":
        return _mclust(X, n_clusters=n_clusters, seed=random_state, n_pca=n_pca)

    if method == "leiden":
        return _leiden(
            X,
            n_clusters=n_clusters,
            resolution=resolution,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown cluster method '{method}'. Choose one of {CLUSTER_METHODS}."
    )


def _mclust(X, n_clusters: int, seed: int = 42, n_pca: int = 20):
    """Gaussian mixture clustering with R's mclust, EEE model.

    Raises a clear error if R or rpy2 is unavailable rather than silently
    falling back, so a result reported as mclust is always mclust.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "cluster_method='mclust' needs rpy2 and an R installation with the "
            "mclust package. Install them, or use cluster_method='kmeans'."
        ) from exc

    import pandas as pd

    importr("mclust")

    if n_pca and X.shape[1] > n_pca:
        from sklearn.decomposition import PCA

        X = PCA(n_components=n_pca, random_state=seed).fit_transform(X)

    print(f"[mclust] input shape: {X.shape}")

    with localconverter(default_converter + pandas2ri.converter):
        ro.globalenv["X_df"] = ro.conversion.py2rpy(pd.DataFrame(X))

    ro.globalenv["num_clusters"] = n_clusters
    ro.globalenv["seed"] = seed
    ro.r(
        """
        library(mclust)
        set.seed(seed)
        fit <- Mclust(X_df, G = num_clusters, modelNames = "EEE")
        cluster_labels <- fit$classification
        """
    )

    labels = np.array(ro.r("cluster_labels")).astype(int)
    return labels - labels.min()  # R is 1-indexed


def _leiden(
    X,
    n_clusters: int = None,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    random_state: int = 42,
    max_steps: int = 20,
):
    """Leiden community detection on a kNN graph built from the embedding.

    Leiden takes a resolution, not a cluster count. If n_clusters is given,
    the resolution is bisected until the community count matches (or the
    search budget runs out); the resolution actually used is printed.
    """
    import anndata as ad
    import scanpy as sc

    adata_tmp = ad.AnnData(X=np.asarray(X, dtype=np.float32))
    sc.pp.neighbors(adata_tmp, use_rep="X", n_neighbors=n_neighbors, metric="cosine")

    def run(res):
        sc.tl.leiden(
            adata_tmp,
            resolution=res,
            key_added="leiden",
            random_state=random_state,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        labels = adata_tmp.obs["leiden"].astype(int).to_numpy()
        return labels, len(np.unique(labels))

    if n_clusters is None:
        labels, k = run(resolution)
        print(f"[leiden] resolution={resolution} -> {k} communities")
        return labels

    lo, hi = 0.05, 4.0
    best_labels, best_k, best_res = None, None, None

    for _ in range(max_steps):
        mid = (lo + hi) / 2
        labels, k = run(mid)

        if best_k is None or abs(k - n_clusters) < abs(best_k - n_clusters):
            best_labels, best_k, best_res = labels, k, mid

        if k == n_clusters:
            break
        if k < n_clusters:
            lo = mid
        else:
            hi = mid

    print(
        f"[leiden] resolution={best_res:.3f} -> {best_k} communities "
        f"(target {n_clusters})"
    )
    return best_labels


# ----------------------------------------------------------------------
# Composition niches
# ----------------------------------------------------------------------
def build_composition_vectors(assignment, coords, k: int = 200, n_prototypes=None):
    """Per-cell local prototype-mixture vector.

    For each cell, average the one-hot prototype assignments of its k
    nearest spatial neighbours. Cells in the same kind of microenvironment
    get similar vectors regardless of which single prototype they fall on.

    Neighbour search must be done per section: never pass concatenated
    multi-section coordinates.
    """
    assignment = np.asarray(assignment).astype(int)
    if n_prototypes is None:
        n_prototypes = int(assignment.max()) + 1

    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    onehot = np.eye(n_prototypes)[assignment]
    return onehot[idx].mean(axis=1)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def fide_score(labels, edge_index) -> float:
    """Fraction of spatial edges joining cells with the same label,
    downweighted by the fraction of available labels actually used."""
    labels = np.asarray(labels)
    src = edge_index[0].cpu().numpy() if torch.is_tensor(edge_index) else edge_index[0]
    dst = edge_index[1].cpu().numpy() if torch.is_tensor(edge_index) else edge_index[1]

    n_unique = len(np.unique(labels))
    if n_unique <= 1:
        print(f"  FIDE degenerate: {n_unique} unique label — returning 0.0")
        return 0.0

    usage_ratio = n_unique / (labels.max() + 1)
    same = (labels[src] == labels[dst]).astype(int)
    intra = same.sum()
    if intra == 0:
        return 0.0

    precision = intra / len(same)
    raw_fide = 2 * precision / (precision + 1.0)
    return raw_fide * usage_ratio


def score_against_truth(adata, truth_key, pred_key):
    """ARI and NMI, or (nan, nan) when the ground-truth column is absent."""
    if not truth_key or truth_key not in adata.obs.columns:
        return float("nan"), float("nan")

    df = adata.obs[[truth_key, pred_key]].dropna()
    if len(df) < 2 or df[truth_key].nunique() < 2:
        return float("nan"), float("nan")

    y_true = df[truth_key].astype(str).to_numpy()
    y_pred = df[pred_key].astype(str).to_numpy()
    return (
        adjusted_rand_score(y_true, y_pred),
        normalized_mutual_info_score(y_true, y_pred),
    )
