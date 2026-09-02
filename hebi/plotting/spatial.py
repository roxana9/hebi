#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial plotting of HEBI labels.

One function draws any per-cell label set over the section's spatial
coordinates. Labels are coloured from a palette built over the full label
set across sections, so colours mean the same thing in every figure.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# Fallback palette of 20 visually separable colours. tab20 pairs a light
# and a dark shade of each hue, which can be hard to tell apart in dense
# tissue; pass cmap_name=None to use these instead.
DISTINCT_COLORS = [
    "#E63946", "#2A9D8F", "#E9C46A", "#264653", "#F4A261",
    "#457B9D", "#6A4C93", "#8AC926", "#1982C4", "#FFCA3A",
    "#6A994E", "#F72585", "#7209B7", "#3A86FF", "#FB5607",
    "#BC4749", "#A7C957", "#386641", "#FF595E", "#023E8A",
]

DEFAULT_CMAP = "tab20"


def get_palette(all_labels, cmap_name: str = DEFAULT_CMAP):
    """Map every label present across sections to a fixed colour.

    Pass the concatenated labels from all sections so that the same label
    gets the same colour in each section's figure.

    cmap_name is a matplotlib colormap name, or None for the built-in
    DISTINCT_COLORS list. tab20 is the default because it matches the
    figures produced by the original analysis scripts.
    """
    categories = sorted(np.unique(np.asarray(all_labels)))
    n = len(categories)

    if cmap_name is None:
        if n > len(DISTINCT_COLORS):
            raise ValueError(
                f"{n} categories exceeds the {len(DISTINCT_COLORS)} colours in "
                "DISTINCT_COLORS. Pass a cmap_name instead."
            )
        return {c: DISTINCT_COLORS[i] for i, c in enumerate(categories)}

    # tab20 has exactly 20 entries. Asking it for more repeats colours, so
    # switch to a continuous map when a label set is larger.
    if cmap_name == "tab20" and n > 20:
        print(f"  {n} categories exceeds tab20's 20 colours — using turbo")
        cmap_name = "turbo"

    cmap = _resampled_cmap(cmap_name, n)
    return {c: cmap(i) for i, c in enumerate(categories)}


def _resampled_cmap(name: str, n: int):
    """Colormap lookup that works on both old and new matplotlib.

    plt.cm.get_cmap was deprecated in 3.7 and removed in 3.9.
    """
    try:
        return matplotlib.colormaps[name].resampled(n)
    except (AttributeError, KeyError):
        return plt.cm.get_cmap(name, n)


def plot_spatial_labels(
    coords,
    labels,
    title: str,
    out_path: str,
    palette=None,
    point_size: float = None,
    alpha: float = 0.7,
    legend_title: str = None,
    figsize=(9, 8),
    formats=("png", "svg"),
    dpi: int = 300,
    invert_y: bool = False,
):
    """Scatter one section's cells, coloured by label, and save the figure.

    out_path is a path without extension; one file is written per entry in
    formats. Returns the list of files written.
    """
    coords = np.asarray(coords)
    labels = np.asarray(labels)

    if palette is None:
        palette = get_palette(labels)

    if point_size is None:
        point_size = auto_point_size(len(labels), panel_width=figsize[0])

    fig, ax = plt.subplots(figsize=figsize)

    for label in sorted(np.unique(labels)):
        mask = labels == label
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[palette[label]],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    if invert_y:
        ax.invert_yaxis()

    present = sorted(np.unique(labels))
    if len(present) <= 30:
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=palette[label], markersize=7, label=str(label),
            )
            for label in present
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
            title=legend_title,
        )

    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    written = []
    for fmt in formats:
        path = f"{out_path}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        written.append(path)

    plt.close(fig)
    return written

def auto_point_size(n_cells: int, panel_width: float = 10.0) -> float:
    """Point size that keeps cells visible without merging into solid
    colour. Large sections need small points: 10k cells can take s=20,
    200k cells needs roughly s=4."""
    base = 2000.0 / max(np.sqrt(n_cells), 1.0)
    scaled = base * (panel_width / 10.0)
    return float(np.clip(scaled, 0.5, 20.0))


def plot_spatial_multi(
    coords_list,
    labels_list,
    section_names,
    out_path: str,
    suptitle: str = None,
    subtitles=None,
    palette=None,
    point_size: float = None,
    alpha: float = 0.7,
    legend_title: str = None,
    panel_size=(10, 8),
    formats=("png", "svg"),
    dpi: int = 300,
    invert_y: bool = False,
):
    """One figure with all sections side by side, sharing a palette so the
    same colour means the same label in every panel.

    Point size is chosen from the largest section unless given, so panels
    stay comparable.
    """
    n = len(coords_list)
    if palette is None:
        palette = get_palette(np.concatenate([np.asarray(l) for l in labels_list]))

    if point_size is None:
        point_size = auto_point_size(
            max(len(l) for l in labels_list), panel_width=panel_size[0]
        )

    fig, axes = plt.subplots(
        1, n, figsize=(panel_size[0] * n, panel_size[1])
    )
    if n == 1:
        axes = [axes]

    for i, (coords, labels, name) in enumerate(
        zip(coords_list, labels_list, section_names)
    ):
        coords = np.asarray(coords)
        labels = np.asarray(labels)
        ax = axes[i]

        for label in sorted(np.unique(labels)):
            mask = labels == label
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[palette[label]], s=point_size, alpha=alpha,
                linewidths=0, rasterized=True,
            )

        title = name if subtitles is None else f"{name}\n{subtitles[i]}"
        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        if invert_y:
            ax.invert_yaxis()

    categories = sorted(palette.keys())
    if len(categories) <= 30:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=palette[c], markersize=8, label=str(c))
            for c in categories
        ]
        fig.legend(
            handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
            fontsize=9, frameon=False, title=legend_title,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    written = []
    for fmt in formats:
        path = f"{out_path}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        written.append(path)

    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)} (s={point_size:.1f})")
    return written
