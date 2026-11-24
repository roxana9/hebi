#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Size- and space-aware merging for HEBI niches (experimental, web page only).

hebi.inference.niches.merge_tree merges by average linkage on the expression /
attention distance d(a, b) alone. Two large, spatially coherent niches that are
transcriptionally close (e.g. two rings of the same tumour region) then merge
early, while a small, scattered but transcriptionally distinct prototype (e.g.
B cells) survives to the coarsest levels.

This module keeps everything in merge_tree (pairwise distance, support and TVD
guards, optional tiny absorption, k = m base, k > m refinement) and changes only
the cost used to pick the next merge:

    cost(a, b) = dbar(a, b) * size(a, b) ** gamma * (1 - lam * contact(a, b))

    dbar     average-linkage distance of the original method
    size     Ward-like: harmonic mean of the two cluster masses,
             2 m_a m_b / (m_a + m_b), divided by the mean leaf mass.
             Mass = cell share per section, averaged over sections.
             Merging something small is cheap; merging two large niches is costly.
    contact  spatial contact on a k-nearest-neighbour cell graph:
             (edges a->b + edges b->a) / 2, divided by min(edges out of a,
             edges out of b). Close to 1 when the smaller cluster sits inside the
             larger one; close to 0 for clusters that do not touch.

gamma = 0 and lam = 0 give the original merge order exactly.

Use:
    with size_space_merge(sections, gamma=1.0, lam=0.5):
        assign_niches(...)
"""
import contextlib
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import hebi.inference.niches as N

GAMMA = 1.0
LAM = 0.5
KNN = 10

_CTX = {}          # set by size_space_merge: knn edges per section, gamma, lam


def knn_edges(coords, k=KNN):
    """Directed kNN edges (i -> j) on the first two coordinate columns."""
    xy = np.asarray(coords, dtype=np.float64)[:, :2]
    k = min(k, len(xy) - 1)
    if k < 1:
        return np.zeros((0, 2), dtype=np.int64)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)
    src = np.repeat(np.arange(len(xy)), k)
    return np.stack([src, idx[:, 1:].ravel()], axis=1)


@contextlib.contextmanager
def size_space_merge(sections, gamma=GAMMA, lam=LAM, knn=KNN, coord_key="spatial"):
    """Route niches.merge_tree through merge_tree_size_space while active."""
    edges = [knn_edges(a.obsm[coord_key], knn) for a in sections]
    original = N.merge_tree
    _CTX.update(edges=edges, gamma=float(gamma), lam=float(lam))
    N.merge_tree = merge_tree_size_space
    try:
        yield
    finally:
        N.merge_tree = original
        _CTX.clear()


def merge_tree_size_space(occupied, support, pi, tiny, prototype_to_col, C0, Cs, attn,
                          section_sizes, section_names, delta_pi=N.DELTA_PI,
                          tiny_policy="keep"):
    """Copy of niches.merge_tree; lines that differ are marked # CHANGED."""
    S = len(section_sizes)
    B = [(np.asarray(attn[s], np.float64).T @ np.asarray(attn[s], np.float64))
         / float(section_sizes[s]) for s in range(S)]

    def compatible(k, l):
        return support[k] == support[l] and N._tvd(pi[k], pi[l]) <= delta_pi + 1e-12

    pairs = [(k, l) for k, l in combinations(occupied, 2) if compatible(k, l)]

    d_sec = {frozenset(p): {} for p in pairs}
    raw_sec = {}
    r_diag = []
    for s in range(S):
        ps = [p for p in pairs if s in (support[p[0]] & support[p[1]])]
        if not ps:
            continue
        V, R = [], []
        for k, l in ps:
            ik, il = prototype_to_col[k], prototype_to_col[l]
            v = N._kappa(C0[ik], C0[il])
            if Cs is not None:
                v = 0.5 * (v + N._kappa(Cs[s][ik], Cs[s][il]))
            r = B[s][ik, il] / (np.sqrt(B[s][ik, ik] * B[s][il, il]) + N.EPS)
            V.append(v)
            R.append(r)
        A_s = 0.5 * (N._ranknorm(V) + N._ranknorm(R))
        for p, a, v, r in zip(ps, A_s, V, R):
            d_sec[frozenset(p)][s] = 1.0 - a
            raw_sec[(frozenset(p), s)] = (v, r)
        R = np.asarray(R)
        r_diag.append({"section": section_names[s], "n_pairs": len(R),
                       "n_distinct_R": int(np.unique(np.round(R, 6)).size),
                       "frac_R_lt_1e-3": round(float((R < 1e-3).mean()), 3),
                       "R_median": float(np.median(R)), "R_max": float(R.max())})

    leaf_distance, worst = {}, {}
    for p in pairs:
        ds = d_sec[frozenset(p)]
        s_w = max(ds, key=lambda s: (ds[s], -s))
        leaf_distance[frozenset(p)] = ds[s_w]
        worst[frozenset(p)] = s_w

    blocked_rows = []
    for k, l in combinations(occupied, 2):
        if compatible(k, l):
            continue
        if not (support[k] & support[l]):
            reason = "disjoint_support"
        elif support[k] != support[l]:
            reason = "different_support_class"
        else:
            reason = f"composition_tvd>{delta_pi:g}"
        blocked_rows.append({
            "a": k, "b": l,
            "support_a": "+".join(section_names[s] for s in sorted(support[k])),
            "support_b": "+".join(section_names[s] for s in sorted(support[l])),
            "tvd_pi": round(N._tvd(pi[k], pi[l]), 4),
            "reason": reason,
            "V_C0": round(N._kappa(C0[prototype_to_col[k]], C0[prototype_to_col[l]]), 6),
        })
    blocked = pd.DataFrame(blocked_rows)
    if not blocked.empty:
        blocked = blocked.sort_values(["V_C0", "a", "b"], ascending=[False, True, True])

    absorbed_rows = []
    target = {}
    if tiny_policy == "absorb_prototype":
        non_tiny = [p for p in occupied if p not in tiny]
        for k in sorted(tiny, key=N.natural_key):
            cands = [l for l in non_tiny if frozenset((k, l)) in leaf_distance]
            if cands:
                t = min(cands, key=lambda l: (leaf_distance[frozenset((k, l))], N.natural_key(l)))
                target[k] = t
                absorbed_rows.append({"tiny": k, "target": t,
                                      "distance": round(leaf_distance[frozenset((k, t))], 6),
                                      "tvd_pi": round(N._tvd(pi[k], pi[t]), 4)})
            else:
                absorbed_rows.append({"tiny": k, "target": None, "distance": None, "tvd_pi": None})
    elif tiny_policy != "keep":
        raise ValueError("tiny_policy must be 'keep' or 'absorb_prototype'")

    # CHANGED: leaf mass and spatial contact counts from argmax(attn) labels + kNN graph
    gamma, lam, edges = _CTX.get("gamma", GAMMA), _CTX.get("lam", LAM), _CTX.get("edges")
    if edges is None or len(edges) != S:
        raise RuntimeError("merge_tree_size_space must run inside size_space_merge(sections)")
    col_to_proto = {c: p for p, c in prototype_to_col.items()}
    occ_index = {p: i for i, p in enumerate(occupied)}
    P = len(occupied)
    leaf_mass = np.zeros(P)
    leaf_edges = np.zeros((P, P))                      # directed edge counts i -> j, pooled
    for s in range(S):
        cols = np.asarray(attn[s]).argmax(axis=1)
        lab = np.array([occ_index.get(col_to_proto[c], -1) for c in range(len(prototype_to_col))])[cols]
        leaf_mass += np.bincount(lab[lab >= 0], minlength=P) / (S * float(section_sizes[s]))
        e = edges[s]
        if len(e):
            a, b = lab[e[:, 0]], lab[e[:, 1]]
            ok = (a >= 0) & (b >= 0)
            np.add.at(leaf_edges, (a[ok], b[ok]), 1.0)
    m_ref = leaf_mass.mean() + N.EPS

    groups = {}
    for p in occupied:
        root = target.get(p, p)
        groups.setdefault(root, []).append(p)
    clusters, cluster_type = {}, {}
    for root, members in groups.items():
        members = sorted(members, key=N.natural_key)
        name = "+".join(members)
        clusters[name] = members
        cluster_type[name] = support[root]

    partitions = {len(clusters): {p: g for g, mem in clusters.items() for p in mem}}
    merge_log = []

    def admissible(a, b):
        return all(frozenset((x, y)) in leaf_distance for x in clusters[a] for y in clusters[b])

    def cluster_distance(a, b):
        return float(np.mean([leaf_distance[frozenset((x, y))]
                              for x in clusters[a] for y in clusters[b]]))

    # CHANGED: size and contact terms
    def _ix(a):
        return [occ_index[p] for p in clusters[a]]

    def size_term(a, b):
        ma, mb = leaf_mass[_ix(a)].sum(), leaf_mass[_ix(b)].sum()
        return (2.0 * ma * mb / (ma + mb + N.EPS)) / m_ref

    def contact(a, b):
        ia, ib = _ix(a), _ix(b)
        ab = 0.5 * (leaf_edges[np.ix_(ia, ib)].sum() + leaf_edges[np.ix_(ib, ia)].sum())
        out_a, out_b = leaf_edges[ia].sum(), leaf_edges[ib].sum()
        return float(ab / (min(out_a, out_b) + N.EPS))

    def merge_cost(a, b):
        d = cluster_distance(a, b)
        sz, ct = size_term(a, b), contact(a, b)
        return d * sz ** gamma * (1.0 - lam * ct), d, sz, ct

    while True:
        best = None
        names = sorted(clusters, key=N.natural_key)
        for a, b in combinations(names, 2):
            if cluster_type[a] != cluster_type[b] or not admissible(a, b):
                continue
            cost, d, sz, ct = merge_cost(a, b)                       # CHANGED
            cand = (cost, N.natural_key(a), N.natural_key(b), a, b, d, sz, ct)
            if best is None or cand[:3] < best[:3]:
                best = cand
        if best is None:
            break
        cost, _, _, a, b, dist, sz, ct = best
        members = sorted(clusters[a] + clusters[b], key=N.natural_key)
        new = "+".join(members)
        pis = [pi[p] for p in members]
        diam = max((N._tvd(x, y) for x, y in combinations(pis, 2)), default=0.0)
        single = len(clusters[a]) == len(clusters[b]) == 1
        fp = frozenset((clusters[a][0], clusters[b][0])) if single else None
        row = {"n_after": len(clusters) - 1, "merged_a": a, "merged_b": b,
               "distance": round(dist, 6),
               "cost": round(cost, 6), "size_term": round(sz, 4),          # CHANGED
               "contact": round(ct, 4),
               "mass_a": round(float(leaf_mass[_ix(a)].sum()), 4),
               "mass_b": round(float(leaf_mass[_ix(b)].sum()), 4),
               "support_type": "+".join(section_names[s] for s in sorted(cluster_type[a])),
               "pi_diameter": round(diam, 4),
               "worst_section": section_names[worst[fp]] if single else None}
        if single:
            for s in sorted(d_sec[fp]):
                v, r = raw_sec[(fp, s)]
                row[f"V_{section_names[s]}"] = round(v, 4)
                row[f"R_{section_names[s]}"] = round(r, 6)
        merge_log.append(row)
        clusters[new] = members
        cluster_type[new] = cluster_type[a]
        for old in (a, b):
            del clusters[old], cluster_type[old]
        partitions[len(clusters)] = {p: g for g, mem in clusters.items() for p in mem}

    return (partitions, pd.DataFrame(merge_log), blocked,
            pd.DataFrame(absorbed_rows), pd.DataFrame(r_diag))
