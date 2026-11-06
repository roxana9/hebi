#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolution-matched niche hierarchy anchored at occupied Level-2 prototypes (v2).

k <= m: constrained average-linkage over the base prototypes.
k  = m: base labels (native argmax(alpha), after optional tiny absorption).
k  > m: prototype-anchored divisive refinement on section-centred z2.

Changes from niche_hierarchy_reviewed.py
  1. Composition guard: clusters A, B merge only if every leaf pair has the
     same support class AND total-variation distance between section-normalised
     compositions TVD(pi_k, pi_l) <= delta_pi (default 0.20), checked on the
     original leaves, never on cluster averages.
  2. Worst-section distance: V^(s) and R^(s) are computed and rank-normalised
     within each section; d_kl = max over common sections of 1 - A^(s)_kl.
     The mapping stays shared across sections.
  3. Split selection is mass-aware: 2-means is fitted with leaf-balanced
     weights, but leaves are ranked by Delta_mass, which weights cells by
     1 / (S * N_s) so a leaf's score scales with its share of each section.
  4. Tiny prototypes (share < tiny_prevalence in every support section) are
     kept and flagged by default. tiny_policy="absorb_prototype" (sensitivity
     only) attaches each tiny prototype, as a whole, to its nearest compatible
     non-tiny prototype; targets are chosen simultaneously, before the tree.

Construction never uses FIDE or JSD.
"""
import sys
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans

MINOR = 0.15
DELTA_PI = 0.20
TINY_PREVALENCE = 0.005
EPS = 1e-12


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def natural_key(value):
    text = str(value)
    return ([int(t) for t in re.findall(r"\d+", text)], text)


def _support(counts, section_sizes, minor=MINOR):
    counts = np.asarray(counts, dtype=float)
    section_sizes = np.asarray(section_sizes, dtype=float)
    if counts.shape != section_sizes.shape or counts.ndim != 1:
        raise ValueError("counts and section_sizes must be 1-D arrays of equal length")
    if np.any(section_sizes <= 0):
        raise ValueError("all sections must contain at least one cell")
    q = counts / section_sizes
    if q.sum() <= 0:
        return frozenset(), np.zeros_like(q)
    pi = q / q.sum()
    return frozenset(int(s) for s in np.where(pi >= minor)[0]), pi


def _tvd(p, q):
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def _kappa(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu <= EPS or nv <= EPS:
        raise ValueError("prototype vectors must have non-zero norm")
    return 0.5 * (1.0 + float((u / nu) @ (v / nv)))


def _ranknorm(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if not np.isfinite(values).all():
        raise ValueError("rank-normalised affinities must be finite")
    return (rankdata(values, method="average") - 0.5) / values.size


def _prepare_attention(matrix, n_cells, n_prototypes, section_name):
    A = np.asarray(matrix)
    if A.ndim != 2 or A.shape != (n_cells, n_prototypes):
        raise ValueError(f"attention for {section_name!r} has shape {A.shape}; "
                         f"expected {(n_cells, n_prototypes)}")
    if not np.isfinite(A).all():
        raise ValueError(f"attention for {section_name!r} contains NaN or inf")
    if np.min(A) < -1e-7:
        raise ValueError(f"attention for {section_name!r} contains negative values")
    A = np.clip(A, 0.0, None).astype(np.float32, copy=False)
    row_sum = A.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError(f"attention for {section_name!r} contains an all-zero row")
    if not np.allclose(row_sum, 1.0, rtol=1e-5, atol=1e-7):
        A = A / row_sum
    return A


def get_obsm(sections, key):
    missing = [i for i, adata in enumerate(sections) if key not in adata.obsm]
    if missing:
        available = list(sections[missing[0]].obsm.keys())
        raise KeyError(f"obsm[{key!r}] missing in sections {missing}; available: {available}")
    return [np.asarray(adata.obsm[key], dtype=np.float32) for adata in sections]


def _effective_number(labels_per_section):
    """exp(entropy) of niche proportions: per section and section-balanced pooled."""
    vocab = sorted(set().union(*map(set, labels_per_section)), key=natural_key)
    per, P = [], []
    for lab in labels_per_section:
        vc = pd.Series(lab).value_counts()
        p = np.array([vc.get(v, 0) for v in vocab], float)
        p = p / p.sum()
        P.append(p)
        nz = p[p > 0]
        per.append(float(np.exp(-(nz * np.log(nz)).sum())))
    pooled = np.mean(P, axis=0)
    nz = pooled[pooled > 0]
    return per, float(np.exp(-(nz * np.log(nz)).sum())), pooled, vocab


# -----------------------------------------------------------------------------
# Per-section prototype vectors
# -----------------------------------------------------------------------------
def section_prototype_vectors(trainer, section_paths):
    """C_k^(s) from the trained Level-2 pooler. Leaves the trainer loaded on
    the final section."""
    import torch
    from hebi.inference.encode import section_community_reps

    output = []
    for module in (trainer.model_lvl0, trainer.model_lvl1, trainer.pooler_lvl2):
        module.eval()
    with torch.no_grad():
        for path in section_paths:
            trainer.data_path = str(path)
            trainer.load_data()
            reps, count = section_community_reps(trainer)
            C, _ = trainer.pooler_lvl2(reps[count > 0])
            output.append(C.detach().cpu().numpy())
    return output


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def attention_diagnostics(attn, section_names, mask_tol=1e-12):
    rows = []
    for matrix, name in zip(attn, section_names):
        A = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, None)
        A = A / A.sum(axis=1, keepdims=True)
        H = -(A * np.log(A + EPS)).sum(axis=1)
        available = int(np.sum(np.any(A > mask_tol, axis=0)))
        Hn = np.clip(H / np.log(max(available, 2)), 0.0, 1.0 + 1e-6)
        K_eff = np.exp(H)
        rows.append({
            "section": name,
            "available": available,
            "occupied": int(np.unique(A.argmax(axis=1)).size),
            "H_norm_q25": np.quantile(Hn, 0.25),
            "H_norm_median": np.median(Hn),
            "H_norm_q75": np.quantile(Hn, 0.75),
            "K_eff_median": np.median(K_eff),
            "K_eff_q90": np.quantile(K_eff, 0.90),
            "frac_max_alpha_gt_0.9": float((A.max(axis=1) > 0.9).mean()),
        })
    return pd.DataFrame(rows).set_index("section").round(3)


def prototype_table(labels, section_names, minor=MINOR, tiny_prevalence=TINY_PREVALENCE):
    section_sizes = np.array([len(v) for v in labels], dtype=float)
    occupied = sorted(set().union(*map(set, labels)), key=natural_key)
    rows, support, pi_all, tiny = [], {}, {}, set()
    for p in occupied:
        counts = np.array([(v == p).sum() for v in labels], dtype=float)
        sup, pi = _support(counts, section_sizes, minor)
        q = counts / section_sizes
        support[p], pi_all[p] = sup, pi
        is_tiny = all(q[s] < tiny_prevalence for s in sup)
        if is_tiny:
            tiny.add(p)
        row = {"prototype": p}
        for s, name in enumerate(section_names):
            row[f"n_{name}"] = int(counts[s])
            row[f"q_{name}"] = round(float(q[s]), 6)
            row[f"pi_{name}"] = round(float(pi[s]), 6)
        row["support"] = "+".join(section_names[s] for s in sorted(sup))
        row["tiny"] = is_tiny
        rows.append(row)
    return pd.DataFrame(rows).set_index("prototype"), occupied, support, pi_all, tiny


# -----------------------------------------------------------------------------
# k <= m: constrained average-linkage tree
# -----------------------------------------------------------------------------
def merge_tree(occupied, support, pi, tiny, prototype_to_col, C0, Cs, attn,
               section_sizes, section_names, delta_pi=DELTA_PI, tiny_policy="keep"):
    S = len(section_sizes)
    B = [(np.asarray(attn[s], np.float64).T @ np.asarray(attn[s], np.float64))
         / float(section_sizes[s]) for s in range(S)]

    def compatible(k, l):
        return support[k] == support[l] and _tvd(pi[k], pi[l]) <= delta_pi + 1e-12

    pairs = [(k, l) for k, l in combinations(occupied, 2) if compatible(k, l)]

    # ---- per-section affinities, rank-normalised within each section
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
            v = _kappa(C0[ik], C0[il])
            if Cs is not None:
                v = 0.5 * (v + _kappa(Cs[s][ik], Cs[s][il]))
            r = B[s][ik, il] / (np.sqrt(B[s][ik, ik] * B[s][il, il]) + EPS)
            V.append(v)
            R.append(r)
        A_s = 0.5 * (_ranknorm(V) + _ranknorm(R))
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

    # ---- blocked pairs (audit)
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
            "tvd_pi": round(_tvd(pi[k], pi[l]), 4),
            "reason": reason,
            "V_C0": round(_kappa(C0[prototype_to_col[k]], C0[prototype_to_col[l]]), 6),
        })
    blocked = pd.DataFrame(blocked_rows)
    if not blocked.empty:
        blocked = blocked.sort_values(["V_C0", "a", "b"], ascending=[False, True, True])

    # ---- initial clusters (optional whole-prototype absorption of tiny prototypes)
    absorbed_rows = []
    target = {}
    if tiny_policy == "absorb_prototype":
        non_tiny = [p for p in occupied if p not in tiny]
        for k in sorted(tiny, key=natural_key):            # simultaneous: targets are non-tiny only
            cands = [l for l in non_tiny if frozenset((k, l)) in leaf_distance]
            if cands:
                t = min(cands, key=lambda l: (leaf_distance[frozenset((k, l))], natural_key(l)))
                target[k] = t
                absorbed_rows.append({"tiny": k, "target": t,
                                      "distance": round(leaf_distance[frozenset((k, t))], 6),
                                      "tvd_pi": round(_tvd(pi[k], pi[t]), 4)})
            else:
                absorbed_rows.append({"tiny": k, "target": None, "distance": None, "tvd_pi": None})
    elif tiny_policy != "keep":
        raise ValueError("tiny_policy must be 'keep' or 'absorb_prototype'")

    groups = {}
    for p in occupied:
        root = target.get(p, p)
        groups.setdefault(root, []).append(p)
    clusters, cluster_type = {}, {}
    for root, members in groups.items():
        members = sorted(members, key=natural_key)
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

    while True:
        best = None
        names = sorted(clusters, key=natural_key)
        for a, b in combinations(names, 2):
            if cluster_type[a] != cluster_type[b] or not admissible(a, b):
                continue
            cand = (cluster_distance(a, b), natural_key(a), natural_key(b), a, b)
            if best is None or cand[:3] < best[:3]:
                best = cand
        if best is None:
            break
        dist, _, _, a, b = best
        members = sorted(clusters[a] + clusters[b], key=natural_key)
        new = "+".join(members)
        pis = [pi[p] for p in members]
        diam = max((_tvd(x, y) for x, y in combinations(pis, 2)), default=0.0)
        single = len(clusters[a]) == len(clusters[b]) == 1
        fp = frozenset((clusters[a][0], clusters[b][0])) if single else None
        row = {"n_after": len(clusters) - 1, "merged_a": a, "merged_b": b,
               "distance": round(dist, 6),
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


# -----------------------------------------------------------------------------
# k > m: divisive refinement
# -----------------------------------------------------------------------------
def _try_split(cells, z2, section_sizes, minor, seed, max_fit):
    S = len(section_sizes)
    counts = np.array([len(ix) for ix in cells], dtype=float)
    if counts.sum() < 2:
        return {"valid": False, "reason": "too small", "delta": -np.inf}
    parent_support, _ = _support(counts, section_sizes, minor)
    if not parent_support:
        return {"valid": False, "reason": "empty support", "delta": -np.inf}

    centred = []
    for s, ix in enumerate(cells):
        if len(ix) == 0:
            centred.append(np.zeros((0, z2[s].shape[1]), dtype=np.float32))
            continue
        r = np.asarray(z2[s][ix], dtype=np.float32)
        r = r / (np.linalg.norm(r, axis=1, keepdims=True) + EPS)
        centred.append(r - r.mean(axis=0, keepdims=True))

    # fit: leaf-balanced weights over support sections
    rng = np.random.default_rng(seed)
    Xf, Wf = [], []
    for s in sorted(parent_support):
        x = centred[s]
        if len(x) > max_fit:
            x = x[rng.choice(len(x), max_fit, replace=False)]
        Xf.append(x)
        Wf.append(np.full(len(x), 1.0 / (len(parent_support) * len(x)), dtype=float))
    Xf, Wf = np.concatenate(Xf), np.concatenate(Wf)
    if len(Xf) < 2:
        return {"valid": False, "reason": "too small", "delta": -np.inf}
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xf, sample_weight=Wf)
    assign = [km.predict(x) if len(x) else np.zeros(0, dtype=int) for x in centred]

    children = [[ix[assign[s] == c] for s, ix in enumerate(cells)] for c in (0, 1)]
    sizes = [int(sum(len(ix) for ix in ch)) for ch in children]
    sup_c = [_support(np.array([len(ix) for ix in ch], float), section_sizes, minor)[0]
             for ch in children]

    # score on ALL cells: mass-aware weights 1/(S*N_s); leaf-normalised kept as diagnostic
    delta_mass, j1_mass, delta_leaf = 0.0, 0.0, 0.0
    for s in range(S):
        x = centred[s]
        if len(x) == 0:
            continue
        w_mass = 1.0 / (S * section_sizes[s])
        j1_mass += w_mass * float(np.square(x).sum())
        between = sum(int((assign[s] == c).sum()) * float(np.square(x[assign[s] == c].mean(0)).sum())
                      for c in (0, 1) if (assign[s] == c).any())
        delta_mass += w_mass * between
        if s in parent_support:
            delta_leaf += between / (len(parent_support) * len(x))

    if min(sizes) == 0:
        reason = "empty child"
    elif not (sup_c[0] & sup_c[1]):
        reason = "pure section split"
    elif delta_mass <= EPS:
        reason = "no within-section gain"
    else:
        reason = ""
    return {"valid": reason == "", "reason": reason, "delta": float(delta_mass),
            "delta_leaf": float(delta_leaf), "rho": float(delta_mass / (j1_mass + EPS)),
            "children": children, "sizes": sizes, "support": parent_support,
            "sup_children": sup_c}


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def build_niche_hierarchy(sections, section_names, C0, Cs=None, attn=None, z2=None,
                          proto_key="prototypes", prototype_ids=None, k_list=(7, 10, 15),
                          minor=MINOR, delta_pi=DELTA_PI, tiny_policy="keep",
                          tiny_prevalence=TINY_PREVALENCE, prefix="niche_k", seed=0,
                          max_fit=100_000, write_obs=True, strict=True, verbose=True):
    n_sections = len(sections)
    if n_sections == 0 or len(section_names) != n_sections:
        raise ValueError("section_names must have one entry per section")
    if attn is None or len(attn) != n_sections:
        raise ValueError("attn must contain one matrix per section")

    C0 = np.asarray(C0)
    n_prototypes = C0.shape[0]
    prototype_ids = ([str(i) for i in range(n_prototypes)] if prototype_ids is None
                     else [str(v) for v in prototype_ids])
    if len(prototype_ids) != n_prototypes or len(set(prototype_ids)) != n_prototypes:
        raise ValueError("prototype_ids must be unique and have length C0.shape[0]")
    prototype_to_col = {p: i for i, p in enumerate(prototype_ids)}
    id_array = np.asarray(prototype_ids, dtype=object)

    section_sizes = np.array([len(a) for a in sections], dtype=float)
    attn = [_prepare_attention(A, int(section_sizes[s]), n_prototypes, section_names[s])
            for s, A in enumerate(attn)]
    labels = [id_array[A.argmax(axis=1)] for A in attn]

    if Cs is not None:
        Cs = [np.asarray(C) for C in Cs]
        if len(Cs) != n_sections or any(C.shape != C0.shape for C in Cs):
            raise ValueError("Cs must contain one [K2, D] matrix per section")

    requested = sorted(set(int(k) for k in k_list))

    agreement = []
    for s, a in enumerate(sections):
        if proto_key not in a.obs:
            agreement.append(np.nan)
            continue
        obs = a.obs[proto_key].astype(str).to_numpy()
        val = float((obs == labels[s]).mean())
        agreement.append(val)
        if strict and val < 1.0:
            raise ValueError(f"argmax(attn) disagrees with obs[{proto_key!r}] for "
                             f"{int((obs != labels[s]).sum())} cells in {section_names[s]!r}")
    if verbose:
        print("argmax(attn) == obs[proto_key]: " + ", ".join(
            f"{n} {v:.4f}" if not np.isnan(v) else f"{n} missing"
            for n, v in zip(section_names, agreement)))

    ptab, occupied, support, pi, tiny = prototype_table(labels, section_names, minor,
                                                        tiny_prevalence)
    partitions, merges, blocked, absorbed, r_diag = merge_tree(
        occupied, support, pi, tiny, prototype_to_col, C0, Cs, attn, section_sizes,
        section_names, delta_pi, tiny_policy)
    m_native = len(occupied)
    m = max(partitions)
    k_min = min(partitions)
    if verbose:
        print(f"occupied prototypes = {m_native}; tiny = {sorted(tiny, key=natural_key)}; "
              f"base leaves m = {m} (tiny_policy={tiny_policy}); "
              f"coarsening feasible for {k_min} <= k <= {m}")

    output_labels, summary = {}, []
    skipped_low = []
    for k in requested:
        if k > m:
            continue
        if k not in partitions:
            skipped_low.append(k)
            print(f"[niches] k={k} is not possible for these sections: merging stops at "
                  f"k_min={k_min}. Choose n_niches >= {k_min} "
                  f"({m} = learned prototypes without merging).", file=sys.stderr)
            continue
        mp = partitions[k]
        output_labels[k] = [np.asarray([mp[v] for v in lab], dtype=object) for lab in labels]
        summary.append({"k": k, "m": m, "merges": m - k, "splits": 0})

    split_log = []
    high = [k for k in requested if k > m]
    if high:
        if z2 is None or len(z2) != n_sections:
            raise ValueError("z2 must contain one embedding matrix per section when any k > m")
        z2 = [np.asarray(Z, dtype=np.float32) for Z in z2]
        for s, Z in enumerate(z2):
            if Z.ndim != 2 or len(Z) != int(section_sizes[s]) or not np.isfinite(Z).all():
                raise ValueError(f"z2[{s}] has invalid shape or values")
        base = partitions[m]
        leaves = {}
        for p, g in base.items():
            leaves.setdefault(g, [np.zeros(0, dtype=int) for _ in range(n_sections)])
        for g in leaves:
            members = [p for p, gg in base.items() if gg == g]
            leaves[g] = [np.where(np.isin(lab, members))[0] for lab in labels]

        cache, n_leaves = {}, m
        while n_leaves < max(high):
            for name, cells in leaves.items():
                if name not in cache:
                    cache[name] = _try_split(cells, z2, section_sizes, minor, seed, max_fit)
            valid = {n: c for n, c in cache.items() if n in leaves and c["valid"]}
            if not valid:
                msg = f"no valid split remains at {n_leaves} leaves"
                if strict:
                    raise RuntimeError(msg)
                if verbose:
                    print("  " + msg + "; stopping")
                break
            name = max(valid, key=lambda n: (valid[n]["delta"], tuple(natural_key(n)[0]), str(n)))
            c = valid[name]
            kids = [f"{name}.1", f"{name}.2"]
            leaves[kids[0]], leaves[kids[1]] = c["children"]
            del leaves[name]
            n_leaves += 1
            split_log.append({
                "n_after": n_leaves, "parent": name,
                "delta_mass": round(c["delta"], 8), "delta_leaf": round(c["delta_leaf"], 6),
                "rho": round(c["rho"], 4),
                "child_1": kids[0], "size_1": c["sizes"][0],
                "support_1": "+".join(section_names[s] for s in sorted(c["sup_children"][0])),
                "child_2": kids[1], "size_2": c["sizes"][1],
                "support_2": "+".join(section_names[s] for s in sorted(c["sup_children"][1])),
                "rejected_leaves": {n: cc["reason"] for n, cc in cache.items()
                                    if n in leaves and not cc["valid"]},
            })
            if n_leaves in high:
                labs = []
                for s in range(n_sections):
                    lab = np.full(int(section_sizes[s]), None, dtype=object)
                    for ln, cells in leaves.items():
                        lab[cells[s]] = ln
                    if pd.isna(lab).any():
                        raise RuntimeError("a split snapshot left cells unassigned")
                    labs.append(lab)
                output_labels[n_leaves] = labs
                summary.append({"k": n_leaves, "m": m, "merges": 0, "splits": n_leaves - m})

    missing = sorted(set(requested) - set(output_labels) - set(skipped_low))
    if missing and strict:
        raise RuntimeError(f"failed to construct the requested resolutions: {missing}")

    # effective resolution per k
    for row in summary:
        per, pooled_eff, pooled, vocab = _effective_number(output_labels[row["k"]])
        for s, n in enumerate(section_names):
            row[f"K_eff_{n}"] = round(per[s], 2)
        row["K_eff_pooled"] = round(pooled_eff, 2)
        row["n_small_niches"] = int((pooled < tiny_prevalence).sum())

    if write_obs:
        for k, labs in output_labels.items():
            names = sorted(set().union(*map(set, labs)), key=natural_key)
            to_int = {n: str(i) for i, n in enumerate(names)}
            for a, values in zip(sections, labs):
                a.obs[f"{prefix}{k}_path"] = pd.Categorical(values, categories=names)
                a.obs[f"{prefix}{k}"] = pd.Categorical(
                    [to_int[v] for v in values], categories=[str(i) for i in range(len(names))])

    sf = pd.DataFrame(summary)
    if not sf.empty:
        sf = sf.sort_values("k").set_index("k")
    return {"prototypes": ptab, "merges": merges, "splits": pd.DataFrame(split_log),
            "blocked_pairs": blocked, "absorbed": absorbed, "R_diagnostics": r_diag,
            "summary": sf, "partitions": partitions, "labels": output_labels,
            "agreement": pd.Series(agreement, index=section_names, name="argmax_agreement"),
            "m_native": m_native, "m": m, "k_min": k_min}


# -----------------------------------------------------------------------------
# Evaluation only
# -----------------------------------------------------------------------------
def _edge_indices(graph):
    if hasattr(graph, "cpu"):
        if hasattr(graph, "detach"):
            graph = graph.detach()
        graph = graph.cpu().numpy()
    if isinstance(graph, np.ndarray) and graph.ndim == 2 and graph.shape[0] == 2:
        source = graph[0].astype(np.int64, copy=False)
        target = graph[1].astype(np.int64, copy=False)
    elif isinstance(graph, tuple) and len(graph) == 2:
        source = np.asarray(graph[0], dtype=np.int64)
        target = np.asarray(graph[1], dtype=np.int64)
    elif hasattr(graph, "nonzero"):
        source, target = graph.nonzero()
        source = np.asarray(source, dtype=np.int64)
        target = np.asarray(target, dtype=np.int64)
    else:
        raise TypeError("each graph must be a [2, E] edge_index, a sparse matrix, "
                        "or an (i, j) edge tuple")
    if source.shape != target.shape:
        raise ValueError("source and target edge arrays must have the same shape")
    return source, target


def evaluate(sections, keys, section_names, graphs=None, graph_key=None, knn=None):
    """FIDE (macro-F1 over spatial edges, per section, averaged) and JSD. Scoring only."""
    from itertools import combinations as pc
    from scipy.spatial.distance import jensenshannon
    from sklearn.metrics import f1_score

    if graphs is not None:
        edge_sets = [_edge_indices(g) for g in graphs]
    elif graph_key is not None:
        edge_sets = [_edge_indices(a.obsp[graph_key]) for a in sections]
    elif knn is not None:
        from sklearn.neighbors import kneighbors_graph
        edge_sets = [_edge_indices(kneighbors_graph(np.asarray(a.obsm["spatial"]),
                                                    n_neighbors=int(knn), include_self=False))
                     for a in sections]
    else:
        raise ValueError("pass the exact benchmark graphs (preferred), graph_key, or knn")

    rows = []
    for key in keys:
        vocab = sorted(set().union(*[set(a.obs[key].astype(str)) for a in sections]),
                       key=natural_key)
        row = {"key": key, "n_niches": len(vocab)}
        fides = []
        for a, (i, j), name in zip(sections, edge_sets, section_names):
            lab = pd.factorize(a.obs[key].astype(str).to_numpy())[0]
            f = f1_score(lab[i], lab[j], average="macro", zero_division=0)
            row[f"FIDE_{name}"] = round(float(f), 6)
            fides.append(float(f))
        row["FIDE_mean"] = round(float(np.mean(fides)), 6)
        P = []
        for a in sections:
            vc = a.obs[key].astype(str).value_counts()
            p = np.array([vc.get(v, 0) for v in vocab], float)
            P.append(p / p.sum())
        row["JSD"] = round(float(np.mean([jensenshannon(x, y, base=2) ** 2
                                          for x, y in pc(P, 2)])), 6)
        rows.append(row)
    return pd.DataFrame(rows).set_index("key")


# -----------------------------------------------------------------------------
# One-call entry point for notebooks
# -----------------------------------------------------------------------------
def assign_niches(sections, ctx, processed_dir, section_files, n_niches=7,
                  section_names=None, tiny_policy="keep", delta_pi=DELTA_PI,
                  minor=MINOR, prefix="niche_k", cache_dir=None, seed=0,
                  verbose=False, return_details=False):
    """Assign HEBI niches at the requested resolution(s), in place.

    sections       list of AnnData returned by encode_sections (same order as section_files)
    ctx            context returned by encode_sections (uses ctx["trainer"])
    n_niches       int or list of ints; total number of niches across all sections
    Writes obs[f"{prefix}{k}"] (integer labels) and obs[f"{prefix}{k}_path"]
    (prototype ancestry) for every k.

    Steps, all automatic:
      1. one encode pass per section for attention, z2 and argmax prototypes,
         stored as obsm["niche_attention"], obsm["z2_enc"], obs["prototypes_enc"]
         (skipped if already present)
      2. per-section prototype vectors C^(s) (cached in ctx, and on disk if cache_dir)
      3. constrained hierarchy (build_niche_hierarchy)
    """
    from pathlib import Path
    import torch
    from hebi.inference.encode import encode_section

    trainer = ctx["trainer"]
    ks = [int(n_niches)] if np.isscalar(n_niches) else [int(k) for k in n_niches]
    names = list(section_names) if section_names is not None else \
        [f"section{i}" for i in range(len(sections))]
    paths = [Path(processed_dir) / f for f in section_files]
    if len(paths) != len(sections):
        raise ValueError("section_files must match sections one-to-one")

    # 1. attention / z2 / prototypes from a single encode pass
    for a, p in zip(sections, paths):
        if all(k in a.obsm for k in ("niche_attention", "z2_enc")) and "prototypes_enc" in a.obs:
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        out = encode_section(trainer, str(p))
        if out["niche_attention"].shape[0] != a.n_obs:
            raise ValueError(f"{p.name}: encoded {out['niche_attention'].shape[0]} cells, "
                             f"AnnData has {a.n_obs}")
        a.obsm["niche_attention"] = out["niche_attention"].cpu().numpy()
        a.obsm["z2_enc"] = out["z2"].cpu().numpy()
        a.obs["prototypes_enc"] = out["prototypes"].numpy().astype(str)

    # 2. per-section prototype vectors (cached)
    Cs = ctx.get("_niche_Cs")
    cache_file = Path(cache_dir) / "niche_section_prototypes.npy" if cache_dir else None
    if Cs is None and cache_file is not None and cache_file.exists():
        Cs = list(np.load(cache_file))
    if Cs is None:
        Cs = section_prototype_vectors(trainer, paths)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, np.stack(Cs))
    ctx["_niche_Cs"] = Cs

    # 3. hierarchy
    C0 = trainer.pooler_lvl2.l2_prototypes.detach().cpu().numpy()
    hier = build_niche_hierarchy(
        sections, names, C0, Cs,
        attn=get_obsm(sections, "niche_attention"),
        z2=get_obsm(sections, "z2_enc"),
        proto_key="prototypes_enc", k_list=ks, minor=minor, delta_pi=delta_pi,
        tiny_policy=tiny_policy, prefix=prefix, seed=seed, verbose=verbose)

    s = hier["summary"]
    for k in [k for k in ks if k in s.index]:
        per = [a.obs[f"{prefix}{k}"].nunique() for a in sections]
        how = (f"{int(s.loc[k, 'merges'])} merges" if s.loc[k, "splits"] == 0
               else f"{int(s.loc[k, 'splits'])} splits")
        print(f"{prefix}{k}: {k} niches ({', '.join(f'{n} {c}' for n, c in zip(names, per))}) | "
              f"{hier['m']} prototypes, {how}")
    return hier if return_details else None


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_niches(sections, n_niches=7, section_names=None, prefix="niche_k",
                spot_size=0.5, out=None, show=True):
    """Spatial niche maps, one panel per section, same colour per niche across
    sections. n_niches: int or list (one figure each). out: path stem; saves
    <out>_k{k}.svg (rasterised points) and .png. Returns the figures."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams["svg.fonttype"] = "none"
    ks = [int(n_niches)] if np.isscalar(n_niches) else [int(k) for k in n_niches]
    names = list(section_names) if section_names is not None else \
        [f"section{i}" for i in range(len(sections))]
    figs = []
    for k in ks:
        key = f"{prefix}{k}"
        missing = [i for i, a in enumerate(sections) if key not in a.obs]
        if missing:
            raise KeyError(f"obs[{key!r}] missing; run assign_niches(..., n_niches={k}) first")
        cats = sorted(set().union(*[set(a.obs[key].astype(str)) for a in sections]),
                      key=natural_key)
        cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
        col = {c: matplotlib.colors.to_hex(cmap(i % cmap.N)) for i, c in enumerate(cats)}

        fig, axes = plt.subplots(1, len(sections), figsize=(7 * len(sections), 7),
                                 facecolor="white", squeeze=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.18, wspace=0.05)
        for ax, a, name in zip(axes[0], sections, names):
            xy = np.asarray(a.obsm["spatial"])
            lab = a.obs[key].astype(str).to_numpy()
            ax.scatter(xy[:, 0], xy[:, 1], c=[col[l] for l in lab], s=spot_size,
                       linewidths=0, rasterized=True)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(name, fontsize=14)
            here = [c for c in cats if c in set(lab)]
            ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=7, mfc=col[c], mew=0, label=c)
                               for c in here],
                      loc="upper center", bbox_to_anchor=(0.5, -0.02),
                      ncol=min(len(here), 8), frameon=False, fontsize=10,
                      handletextpad=0.1, columnspacing=0.6)
        fig.suptitle(f"{k} niches", fontsize=15)
        if out is not None:
            from pathlib import Path
            stem = Path(f"{out}_k{k}")
            stem.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(stem.with_suffix(".svg"), dpi=300)
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        if show:
            try:
                from IPython.display import display
                display(fig)
            except ImportError:
                plt.show()
        plt.close(fig)
        figs.append(fig)
    return figs
