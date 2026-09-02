#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:39:25 2026

@author: rokny-tower4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:39:25 2026

@author: rokny-tower4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
EDGE_TYPE_COLORS = {
    "pathway":   "#D4A5B8" ,   # Deep Indigo/Plum
    "spearman":  "#7192C9",   # Soft Lavender
    "ligrec":  "#5C3C8B" ,   # Dusty Mauve/Rose
    "tf_target": "#9E86C7",   # Periwinkle Blue
    "unknown":   "#C0BCC9",   # Muted Gray-Lavender
}
def plot_all_edges_salience(csv_path, top_k=15, figsize=(9, 6), save_path=None):
    df = pd.read_csv(csv_path).sort_values("edge_salience", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    def first_type(et):
        return str(et).split("|")[0] if pd.notna(et) else "unknown"

    fig, ax = plt.subplots(figsize=figsize)

    # All background edges — clean light slate-lavender cloud
    ax.scatter(df["rank"], df["edge_salience"], s=8, color="#DDDDE6",
               alpha=0.4, edgecolors="none", zorder=1)

    # Top-k prominent nodes
    top = df.head(top_k).reset_index(drop=True)
    top_types = [first_type(t) for t in top["edge_types"]]
    colors = [EDGE_TYPE_COLORS.get(t, "#999999") for t in top_types]
    
    # Render main top-k points without the messy red numbering on top
    ax.scatter(top["rank"], top["edge_salience"], s=100, c=colors,
               edgecolors="#2B2633", linewidths=0.8, zorder=3)

    # --- Table Key Panel on the Left ---
    x0, y0 = 0.05, 0.76      # Table alignment
    line_h = 0.048           # Uniform vertical row spacing
    
    # Styled bounding box for the text key
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0 - 0.03, y0 - line_h * top_k - 0.01), 0.44, line_h * top_k + 0.05,
        transform=ax.transAxes, boxstyle="round,pad=0.01",
        fc="white", ec="#D6D1DF", lw=0.7, zorder=4, mutation_aspect=0.5))
        
    for i, (_, r) in enumerate(top.iterrows()):
        yy = y0 - i * line_h
        
        # Color identity dot
        ax.scatter([x0], [yy], s=75, c=[colors[i]], edgecolors="#2B2633",
                   linewidths=0.6, transform=ax.transAxes, zorder=5, clip_on=False)
        
        # Formatted edge text description
        ax.text(x0 + 0.035, yy, f"{i+1:2d}. {r['source_gene']}→{r['target_gene']}",
                transform=ax.transAxes, fontsize=12, va="center", ha="left",
                family="monospace", fontweight="bold", color="#332C3B", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("edge rank (log scale)", fontsize=13, color="#2B2633")
    ax.set_ylabel("GRN edge salience", fontsize=13, color="#2B2633")

    # Legend Customization
    patches = [mpatches.Patch(color=c, label=t.replace('_', ' ').title())
               for t, c in EDGE_TYPE_COLORS.items() if t != "unknown"]
    ax.legend(handles=patches, fontsize=12, loc="upper right", 
              title="Edge Type", title_fontsize=13, framealpha=0.9)

    # Stylistic chart refinements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBC6D6')
    ax.spines['bottom'].set_color('#CBC6D6')
    ax.tick_params(colors='#4A4452', labelsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.3, color="#BCB8C4")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
plot_all_edges_salience(
    "/DATA1/Roxana/Hierarchy/Two_levels/clustering/Xenium_results/Multi_slide_BC_/grncontinual_nop_15ep_lr1e-4/grn_slide_1/global_grn_slide1_edges_ranked_competitive.csv",
    top_k=15,
    save_path="all_edges_salience_top10_nop_slide1.png"
)   
plot_all_edges_salience(
    "/DATA1/Roxana/Hierarchy/Two_levels/clustering/Xenium_results/Multi_slide_BC_/grncontinual_nop_15ep_lr1e-4/grn_slide_2/global_grn_slide2_edges_ranked_competitive.csv",
    top_k=15,
    save_path="all_edges_salience_top10_nop_slide2.png"
)
plot_all_edges_salience(
    "/DATA1/Roxana/Hierarchy/Two_levels/clustering/Xenium_results/Multi_slide_BC_/grncontinual_pretrained_15ep_lr1e-4/grn_slide_1/global_grn_slide1_edges_ranked_competitive.csv",
    top_k=15,
    save_path="all_edges_salience_top10_pretrained_slide1.png"
)
plot_all_edges_salience(
    "/DATA1/Roxana/Hierarchy/Two_levels/clustering/Xenium_results/Multi_slide_BC_/grncontinual_pretrained_15ep_lr1e-4/grn_slide_2/global_grn_slide2_edges_ranked_competitive.csv",
    top_k=15,
    save_path="all_edges_salience_top10_pretrained_slide2.png"
)

