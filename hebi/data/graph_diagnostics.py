#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Choosing k for the spatial graph.

k sets how much of a cell's surroundings enters its representation. Too
small and the neighbourhood misses the microenvironment; too large and it
reaches across tissue boundaries into structures the cell is not part of.
The right value depends on cell density and section geometry, both of
which are measurable from the coordinates alone.

The diagnostic is the physical distance to the k-th nearest neighbour,
expressed in cell diameters. That ratio is comparable across platforms
and unaffected by whether coordinates are in microns, pixels, or
arbitrary units. Two to three diameters covers a cell and its first two
rings of contacts.

No annotations are used. k is chosen from the geometry of the section,
not from the labels it will later be evaluated against.

    python -m hebi.data.graph_diagnostics --h5ad section.h5ad \\
        --out_dir diagnostics/
"""

import argparse
import json
import os

import numpy as np

DEFAULT_K_VALUES = [3, 5, 10, 15, 20, 30, 40, 50, 75, 100]


def _find_spatial(adata):
    for key in adata.obsm:
        if "spatial" in key.lower():
            return np.asarray(adata.obsm[key], dtype=float)[:, :2]
    raise KeyError(f"No spatial coordinates in obsm: {list(adata.obsm)}")


def neighbour_diagnostics(coords, k_values=None, n_sample: int = 20000,
                          seed: int = 0):
    """Measure how far the neighbourhood reaches across a range of k.

    Sampling keeps this fast on large sections: neighbours are searched
    against all cells, but the statistics are computed over a sample of
    query cells, which is enough for a median.

    Returns a list of dicts, one per k, and the distance scale used.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=float)
    k_values = sorted(k_values or DEFAULT_K_VALUES)
    k_max = max(k_values)

    n_cells = len(coords)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(n_cells, n_sample, replace=False)
           if n_cells > n_sample else np.arange(n_cells))

    print(f"  {n_cells:,} cells; querying {len(idx):,} of them up to k={k_max}")

    nbrs = NearestNeighbors(n_neighbors=k_max + 1).fit(coords)
    distances, _ = nbrs.kneighbors(coords[idx])
    distances = distances[:, 1:]        # column 0 is the cell itself

    # A rough cell diameter: twice the median nearest-neighbour distance.
    # Cells that touch sit about one diameter apart, so this is the scale
    # the reach numbers should be read against.
    nn1 = float(np.median(distances[:, 0]))
    cell_scale = 2.0 * nn1

    rows = []
    for k in k_values:
        reach = float(np.median(distances[:, k - 1]))
        rows.append({
            "k": k,
            "reach_median": round(reach, 3),
            "reach_p90": round(float(np.percentile(distances[:, k - 1], 90)), 3),
            "reach_in_cell_diameters": round(reach / (cell_scale + 1e-9), 2),
        })

    return rows, {"n_cells": n_cells, "nn1_median": round(nn1, 3),
                  "cell_scale": round(cell_scale, 3)}


REFERENCE_SECTIONS = [
    # platform, tissue, median NN distance, k used, reach at that k
    ("Xenium",   "human breast",     9.54,  30, 2.05),
    ("MERSCOPE", "human breast",     7.71,  30, 2.28),
    ("CosMx",    "human liver",      None,  30, 2.42),
    ("STARmap",  "mouse CNS",       41.27,  30, 3.21),
]

# Two to three cell diameters covers a cell and its first two rings of
# physical contacts. Below one diameter the neighbourhood barely reaches
# the adjacent cells and spatial context is largely unused; well above
# three it starts spanning distinct tissue structures.
TARGET_DIAMETERS = (2.0, 3.0)


def suggest_k(rows, target_diameters=TARGET_DIAMETERS):
    """Suggest k from the reach curve.

    Returns the k values whose neighbourhood spans the target range of
    cell diameters. Reach is a ratio, so it is comparable across
    platforms and unaffected by whether coordinates are in microns,
    pixels, or arbitrary units.

    This is a starting point, not an optimum: k interacts with tissue
    architecture and with the number of domains being resolved, so it
    should be tuned against held-out annotations where they exist.
    """
    lo_target, hi_target = target_diameters

    in_band = [r["k"] for r in rows
               if lo_target <= r["reach_in_cell_diameters"] <= hi_target]

    if in_band:
        k_min, k_max = min(in_band), max(in_band)
    else:
        # No tested k lands in the band; fall back to the closest one.
        mid = (lo_target + hi_target) / 2
        closest = min(rows, key=lambda r:
                      abs(r["reach_in_cell_diameters"] - mid))
        k_min = k_max = closest["k"]

    return {
        "k_min": int(k_min),
        "k_max": int(k_max),
        "target_diameters": list(target_diameters),
        "reach_at_k_min": next(r["reach_in_cell_diameters"]
                               for r in rows if r["k"] == k_min),
        "reach_at_k_max": next(r["reach_in_cell_diameters"]
                               for r in rows if r["k"] == k_max),
    }


def plot_diagnostics(rows, scale, out_path, title="",
                     target=TARGET_DIAMETERS):
    """Reach against k, with the target band marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in rows]
    reach = [r["reach_in_cell_diameters"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.axhspan(target[0], target[1], color="#2E66B3", alpha=0.10,
               label=f"{target[0]}-{target[1]} cell diameters")
    ax.plot(ks, reach, marker="o", color="#2E66B3")
    ax.axhline(1, ls=":", color="#999", lw=1)
    ax.set_xlabel("k")
    ax.set_ylabel("neighbourhood reach (cell diameters)")
    ax.set_title(title or "How far the neighbourhood extends")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out_path}")


def report(rows, scale, suggestion, name=""):
    """Print the reach table, the suggested range, and the reference set."""
    print(f"\n  median nearest-neighbour distance: {scale['nn1_median']}")
    print(f"  implied cell diameter:             {scale['cell_scale']}")
    print()
    print(f"  {'k':>5}  {'reach':>10}  {'diameters':>10}")
    print("  " + "-" * 29)
    for r in rows:
        print(f"  {r['k']:>5}  {r['reach_median']:>10.2f}  "
              f"{r['reach_in_cell_diameters']:>10.2f}")

    lo, hi = suggestion["target_diameters"]
    print(f"\n  suggested k: {suggestion['k_min']} to {suggestion['k_max']}"
          f"   ({suggestion['reach_at_k_min']:.2f} to "
          f"{suggestion['reach_at_k_max']:.2f} cell diameters)")
    print(f"  target is {lo} to {hi} diameters: a cell plus its first two "
          f"rings of contacts.")

    print("\n  For reference, the sections HEBI was developed on:")
    print(f"    {'platform':<10} {'tissue':<15} {'NN dist':>9} {'k':>4} "
          f"{'reach':>7}")
    for plat, tis, nn, k_ref, reach in REFERENCE_SECTIONS:
        nn_str = f"{nn:.2f}" if nn is not None else "  n/a"
        print(f"    {plat:<10} {tis:<15} {nn_str:>9} {k_ref:>4} {reach:>7.2f}")
    print("  This is a starting point; k also interacts with how many "
          "domains you resolve.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Measure how k should be chosen for a section's graph."
    )
    p.add_argument("--h5ad", nargs="+",
                   help="One or more .h5ad sections.")
    p.add_argument("--pt", nargs="+",
                   help="One or more preprocessed .pt sections. Coordinates "
                        "are read from these.")
    p.add_argument("--k_values", nargs="+", type=int, default=None,
                   help=f"k values to test (default {DEFAULT_K_VALUES}).")
    p.add_argument("--n_sample", type=int, default=20000,
                   help="Query cells sampled per section.")
    p.add_argument("--target_diameters", nargs=2, type=float,
                   default=list(TARGET_DIAMETERS),
                   help="Neighbourhood reach to aim for, in cell diameters "
                        f"(default {TARGET_DIAMETERS[0]} {TARGET_DIAMETERS[1]}).")
    p.add_argument("--out_dir", type=str, default="./graph_diagnostics")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not args.h5ad and not args.pt:
        raise SystemExit("Give --h5ad or --pt")

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    sources = [(path, "h5ad") for path in (args.h5ad or [])]
    sources += [(path, "pt") for path in (args.pt or [])]

    for path, kind in sources:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== {name}")

        if kind == "h5ad":
            import scanpy as sc

            coords = _find_spatial(sc.read_h5ad(path))
        else:
            import torch

            data = torch.load(path, map_location="cpu", weights_only=False)
            coords = np.asarray(data["coords"], dtype=float)[:, :2]

        rows, scale = neighbour_diagnostics(
            coords, k_values=args.k_values,
            n_sample=args.n_sample, seed=args.seed,
        )
        suggestion = suggest_k(
            rows, target_diameters=tuple(args.target_diameters)
        )
        report(rows, scale, suggestion, name=name)
        plot_diagnostics(rows, scale,
                         os.path.join(args.out_dir, f"{name}_k_diagnostics.png"),
                         title=name)

        results[name] = {"scale": scale, "curve": rows,
                         "suggestion": suggestion}

    out_json = os.path.join(args.out_dir, "k_diagnostics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten -> {out_json}")
    return results


if __name__ == "__main__":
    main()
