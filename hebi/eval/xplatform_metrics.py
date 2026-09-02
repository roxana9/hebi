#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:18:02 2026

@author: rokny-tower4
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label-free cross-platform integration metrics for the HEBI Xenium+MERSCOPE run.

Run it on your in-memory `results` (list of per-slide AnnData returned by main()):

    from xplatform_metrics import evaluate_integration
    summary = evaluate_integration(
        results,
        platform_names=["Xenium", "MERSCOPE"],  # order must match results[0], results[1]
        embedding_key="z2_embedding",           # neighbourhood level = correct for cross-platform
        niche_key="comp_niches",                # your 7 joint niches
    )

Then run the SAME call on your Novae results to compare HEBI vs Novae.

Metrics (all annotation-free):
  iLISI          integration / platform mixing        higher = better  (range 1..n_platforms)
  MMD            multi-bandwidth RBF, platforms        lower  = better  (TERRA's 2nd metric)
  batch_ASW      silhouette on platform labels         higher = better mixing (1 - |sil|)
  niche_ASW      silhouette on niches (bio kept)       higher = better separation
Plus:
  platform composition per niche  -> identifies platform-specific niches (e.g. ER+ ~94% Xenium)
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import silhouette_samples, f1_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import jensenshannon


# ----------------------------------------------------------------------- helpers
def build_joint(results, platform_names, embedding_key, niche_key):
    """Stack the per-slide embeddings + attach a platform vector and niche vector."""
    embs, plats, niches = [], [], []
    for ad, pname in zip(results, platform_names):
        Z = np.asarray(ad.obsm[embedding_key], dtype=np.float64)
        embs.append(Z)
        plats.append(np.full(Z.shape[0], pname, dtype=object))
        if niche_key in ad.obs:
            niches.append(ad.obs[niche_key].astype(str).to_numpy())
        else:
            niches.append(np.full(Z.shape[0], "NA", dtype=object))
    Z = np.concatenate(embs, 0)
    platform = np.concatenate(plats, 0)
    niche = np.concatenate(niches, 0)
    # standardise the embedding so kNN distances aren't dominated by a few dims
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    return Z, platform, niche


# ------------------------------------------------------------------------ iLISI
def ilisi_manual(Z, batch, k=90):
    """
    Inverse-Simpson integration LISI, averaged over cells.
    For each cell: among its k nearest neighbours, iLISI = 1 / sum_b p_b^2.
    Approximates scib's iLISI (which adds perplexity weighting).
    Range ~1 (segregated) .. n_batches (perfectly mixed).
    """
    codes, uniq = pd.factorize(batch)
    n_b = len(uniq)
    k = min(k, Z.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]                       # drop self
    neigh_codes = codes[idx]              # [N, k]
    vals = np.empty(Z.shape[0])
    for i in range(Z.shape[0]):
        counts = np.bincount(neigh_codes[i], minlength=n_b) / k
        vals[i] = 1.0 / np.sum(counts ** 2)
    return float(np.mean(vals)), n_b


def ilisi(Z, batch, k=90):
    """Prefer scib-metrics if available; otherwise fall back to ilisi_manual."""
    try:
        from scib_metrics import ilisi_knn
        from sklearn.neighbors import NearestNeighbors as NN
        k = min(k, Z.shape[0] - 1)
        nn = NN(n_neighbors=k + 1).fit(Z)
        dist, idx = nn.kneighbors(Z)
        # scib_metrics wants a (distances, indices) neighbour graph tuple
        score = ilisi_knn((dist[:, 1:], idx[:, 1:]), batch, scale=True)
        return float(np.mean(score)), len(np.unique(batch)), "scib_metrics"
    except Exception:
        v, n_b = ilisi_manual(Z, batch, k=k)
        return v, n_b, "manual"


# -------------------------------------------------------------------------- MMD
def mmd_rbf(X, Y, gammas=None, max_n=2000, seed=0):
    """
    Unbiased MMD^2 between two platform clouds, averaged over several RBF
    bandwidths (TERRA's multi-bandwidth Gaussian). Lower = better mixing.
    """
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_n:
        X = X[rng.choice(X.shape[0], max_n, replace=False)]
    if Y.shape[0] > max_n:
        Y = Y[rng.choice(Y.shape[0], max_n, replace=False)]
    if gammas is None:
        # median heuristic -> a small band of bandwidths around it
        from sklearn.metrics.pairwise import euclidean_distances
        med = np.median(euclidean_distances(X[:500], Y[:500])) + 1e-8
        gammas = [1.0 / (2 * (med * s) ** 2) for s in (0.5, 1.0, 2.0)]
    vals = []
    for g in gammas:
        Kxx = rbf_kernel(X, X, gamma=g)
        Kyy = rbf_kernel(Y, Y, gamma=g)
        Kxy = rbf_kernel(X, Y, gamma=g)
        m, n = X.shape[0], Y.shape[0]
        np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
        mmd2 = (Kxx.sum() / (m * (m - 1))
                + Kyy.sum() / (n * (n - 1))
                - 2 * Kxy.mean())
        vals.append(max(mmd2, 0.0))
    return float(np.mean(vals))


# ------------------------------------------------------------------ silhouettes
def batch_asw(Z, batch, max_n=5000, seed=0):
    """TERRA-style batch ASW: 1 - mean(|silhouette on platform|). Higher = better mixing."""
    rng = np.random.default_rng(seed)
    if Z.shape[0] > max_n:
        sel = rng.choice(Z.shape[0], max_n, replace=False); Z, batch = Z[sel], batch[sel]
    codes = pd.factorize(batch)[0]
    if len(np.unique(codes)) < 2:
        return np.nan
    s = silhouette_samples(Z, codes)
    return float(1.0 - np.mean(np.abs(s)))


def niche_asw(Z, niche, max_n=5000, seed=0):
    """Bio conservation: normalised silhouette on niche labels. Higher = niches better separated."""
    rng = np.random.default_rng(seed)
    mask = niche != "NA"
    Z, niche = Z[mask], niche[mask]
    if Z.shape[0] > max_n:
        sel = rng.choice(Z.shape[0], max_n, replace=False); Z, niche = Z[sel], niche[sel]
    codes = pd.factorize(niche)[0]
    if len(np.unique(codes)) < 2:
        return np.nan
    s = silhouette_samples(Z, codes)
    return float((np.mean(s) + 1) / 2)


# --------------------------------------------------- platform composition / asymmetry
def platform_composition(platform, niche, ref_platform):
    """
    Per-niche fraction of cells from ref_platform. This is the asymmetry panel:
    a niche near 0.5 is shared; a niche near 0/1 is platform-specific
    (e.g. the luminal ER+ niche ~0.94 Xenium).
    """
    df = pd.DataFrame({"platform": platform, "niche": niche})
    df = df[df.niche != "NA"]
    tab = pd.crosstab(df.niche, df.platform, normalize="index")
    frac = tab.get(ref_platform, pd.Series(0.0, index=tab.index)).sort_index()
    out = pd.DataFrame({"frac_" + ref_platform: frac})
    out["type"] = np.where(frac > 0.85, ref_platform + "-specific",
                    np.where(frac < 0.15, "other-specific", "shared"))
    return out


# ----------------------------------------------------- FIDE (spatial coherence)
def _edges_from_coords(coords, k=6):
    """kNN spatial graph -> (i, j) edge endpoint index arrays."""
    G = kneighbors_graph(np.asarray(coords), n_neighbors=k, mode="connectivity")
    i, j = G.nonzero()
    return i, j


def fide_score(labels, coords=None, edge_index=None, k=6):
    """
    Novae's FIDE: macro-F1 between the domain labels of the two endpoints of every
    spatial edge. Spatially coherent domains -> most edges are intra-domain ->
    labels[i] ~= labels[j] -> high F1. Fragmented (salt-and-pepper) -> low F1.
    Per slide, no ground truth. Higher = better.

    Pass edge_index ([2, E] array/tensor, e.g. your slide's .pt graph) to match the
    pipeline exactly; otherwise a kNN graph is built from coords.
    """
    labels = np.asarray(pd.factorize(np.asarray(labels).astype(str))[0])
    if edge_index is not None:
        ei = np.asarray(edge_index.cpu() if hasattr(edge_index, "cpu") else edge_index)
        i, j = ei[0], ei[1]
    elif coords is not None:
        i, j = _edges_from_coords(coords, k=k)
    else:
        raise ValueError("fide_score needs coords or edge_index")
    return float(f1_score(labels[i], labels[j], average="macro"))


# ------------------------------------------- JSD (cross-slide domain homogeneity)
def jsd_cross_slide(results, niche_key):
    """
    Novae's cross-slide homogeneity: build each slide's domain-label distribution
    over the shared vocabulary, take pairwise Jensen-Shannon divergence, average.
    Low = both platforms recovered the same domains in the same proportions.
    (Distributional / niche-level check — tolerates cells staying platform-separated.)
    """
    vocab = sorted(set().union(*[set(ad.obs[niche_key].astype(str)) for ad in results]))
    dists = []
    for ad in results:
        vc = ad.obs[niche_key].astype(str).value_counts()
        p = np.array([vc.get(v, 0) for v in vocab], dtype=float)
        dists.append(p / p.sum())
    jsd_vals = [jensenshannon(a, b, base=2) ** 2          # ^2 -> divergence, not distance
                for a, b in itertools.combinations(dists, 2)]
    return float(np.mean(jsd_vals)), dists, vocab


# ------------------------------------------------------------------------- driver
def evaluate_integration(results, platform_names, embedding_key="z2_embedding",
                         niche_key="comp_niches", k=90, edge_indices=None, fide_k=6):
    Z, platform, niche = build_joint(results, platform_names, embedding_key, niche_key)

    il, n_b, il_src = ilisi(Z, platform, k=k)
    mm = mmd_rbf(Z[platform == platform_names[0]], Z[platform == platform_names[1]])
    b_asw = batch_asw(Z, platform)
    n_asw = niche_asw(Z, niche)
    comp = platform_composition(platform, niche, ref_platform=platform_names[0])

    # --- Novae-comparable pair: FIDE per slide, JSD across slides ---
    fide_per_slide = []
    for s, ad in enumerate(results):
        ei = edge_indices[s] if edge_indices is not None else None
        fide_per_slide.append(
            fide_score(ad.obs[niche_key], coords=ad.obsm.get("spatial"),
                       edge_index=ei, k=fide_k))
    jsd_val, _, _ = jsd_cross_slide(results, niche_key)

    summary = pd.Series({
        "FIDE (mean over slides, ↑)":     round(float(np.mean(fide_per_slide)), 3),
        "JSD (cross-slide, ↓)":           round(jsd_val, 4),
        f"iLISI ({il_src}, 1..{n_b}, ↑)": round(il, 3),
        "MMD (multi-RBF, ↓)":             round(mm, 4),
        "batch_ASW (1-|sil|, ↑)":         round(b_asw, 3),
        "niche_ASW (bio, ↑)":             round(n_asw, 3),
    })
    print("\n=== cross-platform integration (label-free) ===")
    print(summary.to_string())
    print("\n=== platform composition per niche (asymmetry) ===")
    print(comp.to_string())
    return {"summary": summary, "composition": comp,
            "fide_per_slide": dict(zip(platform_names, fide_per_slide)),
            "jsd": jsd_val,
            "embedding": Z, "platform": platform, "niche": niche}


# ------------------------------------------ OPTIONAL: shared-gene transfer ARI
def transfer_ari_shared_genes(results, platform_names, niche_key="comp_niches",
                              layer=None, k=30):
    """
    Strongest annotation-free correspondence number: restrict both slides to their
    SHARED genes, train kNN on platform-A niche labels, predict platform-B, and
    score agreement (ARI) with B's own niches. Symmetric average of both directions.

    layer: name of a log-normalised layer to use; None uses .X.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score

    a, b = results
    shared = sorted(set(a.var_names) & set(b.var_names))
    if len(shared) < 5:
        print(f"Only {len(shared)} shared genes — check harmonisation."); return None
    print(f"Shared genes used for transfer: {len(shared)}")

    def mat(ad):
        sub = ad[:, shared]
        X = sub.layers[layer] if layer else sub.X
        return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)

    Xa, Xb = mat(a), mat(b)
    ya = a.obs[niche_key].astype(str).to_numpy()
    yb = b.obs[niche_key].astype(str).to_numpy()

    def direction(Xtr, ytr, Xte, yte):
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
        return adjusted_rand_score(yte, clf.predict(Xte))

    ari_ab = direction(Xa, ya, Xb, yb)   # train Xenium -> predict MERSCOPE
    ari_ba = direction(Xb, yb, Xa, ya)   # train MERSCOPE -> predict Xenium
    print(f"transfer ARI  {platform_names[0]}->{platform_names[1]}: {ari_ab:.3f}")
    print(f"transfer ARI  {platform_names[1]}->{platform_names[0]}: {ari_ba:.3f}")
    print(f"transfer ARI  (symmetric mean): {(ari_ab + ari_ba) / 2:.3f}")
    return {"ab": ari_ab, "ba": ari_ba, "mean": (ari_ab + ari_ba) / 2}