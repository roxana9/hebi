#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 12:04:52 2026

@author: tower4
"""

import matplotlib.pyplot as plt
import numpy as np

labels = [
    "HEBI — Fine-tuned",
    "HEBI — Linear probe",
    "PCA — 64 PCs",
    "PCA — 30 PCs",
    "PCA — 10 PCs",
    "Nicheformer — Fine-tuned",
    "Nicheformer — Linear probe",
    "Nicheformer (liver only, linear-probing)",
    "scVI "
]

scores = [
    float(np.mean([0.6064, 0.6671, 0.6749, 0.6649, 0.6549])),
    float(np.mean([0.5037, 0.4872, 0.4911, 0.4489, 0.5028])),
    float(np.mean([0.3878, 0.3885, 0.3872, 0.3859, 0.3878])),
    float(np.mean([0.3872, 0.3846, 0.3848, 0.3858, 0.3807])),
    float(np.mean([0.4060, 0.4054, 0.4046, 0.4030, 0.4045])),
    0.53,
    0.1,
    0.53,
    0.49
]

families = [
    "Hebi", "Hebi",
    "PCA", "PCA", "PCA",
    "Nicheformer", "Nicheformer","Nicheformer",
    "scVI" 
]

unique_fams = list(dict.fromkeys(families))

# ----------------------------
# Vibrant, high-contrast palette (colourblind-friendly)
# ----------------------------

vibrant = [
    "#6E8E84",  # Midnight Violet (Deepest)
        "#5F4B8B",  # Royal Amethyst
        "#8E7697",  # Antique Lavender
       # "#A28089",  # Rose-Mauve (The classic pinkish-dusty tone)
      #  "#708ABB",  # Misted Orchid (The 'Bright' limit)
        "#635666"   # Dusty Slate (A bonus 'neutral' purple)
    
]
fam2color = {fam: vibrant[i % len(vibrant)] for i, fam in enumerate(unique_fams)}
bar_colors = [fam2color[f] for f in families]

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(labels))

bars = ax.barh(
    y, scores,
    color=bar_colors,
    edgecolor="white",
    linewidth=1.2
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Macro-F1")
ax.set_xlim(0, 1.05)
ax.set_title("")

# cleaner look
ax.grid(axis="x", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Put value inside bar
for bar, v in zip(bars, scores):
    x = v - 0.02
    x = max(x, 0.02)
    txt_color = "white" if v > 0.0 else "black"
    ax.text(
        x, bar.get_y() + bar.get_height()/2,
        f"{v:.3f}",
        va="center", ha="right",
        color=txt_color, fontsize=10, fontweight="bold"
    )

def add_sig_h(ax, bar1, bar2, text="***", xpad=0.006, tick=0.012, text_dx=0.004):
    y1 = bar1.get_y() + bar1.get_height() / 2
    y2 = bar2.get_y() + bar2.get_height() / 2
    x_right = max(bar1.get_width(), bar2.get_width()) + xpad
    xmin, xmax = ax.get_xlim()
    x_right = min(x_right, xmax - (tick + 0.02))

    ax.plot([x_right, x_right], [y1, y2], color="black", lw=1.2)
    ax.plot([x_right - tick, x_right], [y1, y1], color="black", lw=1.2)
    ax.plot([x_right - tick, x_right], [y2, y2], color="black", lw=1.2)

    ax.text(min(x_right + text_dx, xmax - 0.005), (y1 + y2) / 2, text,
            va="center", ha="left", fontsize=14, fontweight="bold")

idx_hebi_ft = labels.index("HEBI — Fine-tuned")
idx_pca64   = labels.index("Nicheformer — Fine-tuned")
add_sig_h(ax, bars[idx_hebi_ft], bars[idx_pca64], text="***")

# legend (unique families)
handles = []
legend_labels = []
for fam in unique_fams:
    h = plt.Line2D([0], [0], marker='s', linestyle='',
                   markersize=10, color=fam2color[fam])
    handles.append(h)
    legend_labels.append(fam)

ax.legend(handles, legend_labels, title="Method", loc="lower right", frameon=False)

# p-value note
p_one = 0.00018
ax.text(
    0.56, 0.02,
    f"***  p = {p_one:.2e} ",
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
)

plt.tight_layout()
out_png = "niche_model_comparison_livernormal_cosmax.png"
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.show()


##################################################################

import numpy as np
import matplotlib.pyplot as plt

# ---- PCA mean macro-F1 per PCs (from your 5 seeds) ----
pca_by_pcs = {
    10:  np.array([0.3566, 0.3769, 0.3656, 0.3607, 0.3730], dtype=float),
    30:  np.array([0.3872, 0.3846, 0.3848, 0.3858, 0.3807], dtype=float),
    64:  np.array([0.3878, 0.3885, 0.3872, 0.3859, 0.3878], dtype=float),
    128: np.array([0.3990, 0.4017, 0.4009, 0.4005, 0.4013], dtype=float),
    256: np.array([0.4028, 0.4021, 0.4016, 0.4037, 0.4045], dtype=float),
    384: np.array([0.4060, 0.4054, 0.4046, 0.4030, 0.4045], dtype=float),
}

pcs = np.array(sorted(pca_by_pcs.keys()))
pca_mean_f1 = np.array([pca_by_pcs[p].mean() for p in pcs])

# ---- cumulative explained variance you provided ----
cum_var = {
    10:  0.1212,
    30:  0.1739,
    64:  0.2542,
    128:0.3758,
    256: 0.5437,
    384: 0.6628,
}

pca_cum_var = np.array([cum_var[p] for p in pcs])

# ---- choose two distinct colours ----
col_f1  = "#1f77b4"  # blue
col_var = "#ff7f0e"  # orange

fig, ax1 = plt.subplots(figsize=(7.6, 4.8))

# left axis: macro-F1
ax1.plot(pcs, pca_mean_f1, marker="o", color=col_f1, label="PCA macro-F1 (mean)")
ax1.set_xlabel("Number of PCA PCs")
ax1.set_ylabel("Macro-F1 (mean over 5 seeds)", color=col_f1)
ax1.tick_params(axis="y", labelcolor=col_f1)
ax1.set_ylim(0.2, 0.99)
ax1.set_xticks(pcs)

# right axis: cumulative variance
ax2 = ax1.twinx()
ax2.plot(pcs, pca_cum_var, marker="s", linestyle="--", color=col_var, label="Cumulative variance")
ax2.set_ylabel("Cumulative explained variance ratio", color=col_var)
ax2.tick_params(axis="y", labelcolor=col_var)
ax2.set_ylim(0.0, 1.0)

# combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

ax1.set_title("PCA performance vs cumulative variance")

plt.tight_layout()
plt.savefig("supp_line_f1_and_variance_vs_pcs_livernormal.png", dpi=600, bbox_inches="tight")
#plt.savefig("supp_line_f1_and_variance_vs_pcs.pdf", bbox_inches="tight")
plt.show()
#####################################
import numpy as np
import matplotlib.pyplot as plt

# ---- your seed-level results ----
hebi_lp = np.array([0.5037, 0.4872, 0.4911, 0.4489, 0.5028], dtype=float)
hebi_ft = np.array([0.6064, 0.6671, 0.6749, 0.6649, 0.6549], dtype=float)
pca_by_pcs = {
    10:  np.array([0.3566, 0.3769, 0.3656, 0.3607, 0.3730], dtype=float),
    30:  np.array([0.3872, 0.3846, 0.3848, 0.3858, 0.3807], dtype=float),
    64:  np.array([0.3878, 0.3885, 0.3872, 0.3859, 0.3878], dtype=float),
    128: np.array([0.3990, 0.4017, 0.4009, 0.4005, 0.4013], dtype=float),
    256: np.array([0.4028, 0.4021, 0.4016, 0.4037, 0.4045], dtype=float),
    384: np.array([0.4060, 0.4054, 0.4046, 0.4030, 0.4045], dtype=float),
}

# ---- compute mean/std ----
pcs = np.array(sorted(pca_by_pcs.keys()))
pca_means = np.array([pca_by_pcs[p].mean() for p in pcs])
pca_stds  = np.array([pca_by_pcs[p].std()  for p in pcs])

hebi_lp_mean = float(hebi_lp.mean())
hebi_ft_mean = float(hebi_ft.mean())

# ---- colourblind-safe palette (Okabe–Ito) ----
colors = {
    "pca": "#0072B2",  # blue
    "lp":  "#D55E00",  # orange
    "ft":  "#009E73",  # green
}

# ---- plot ----
fig, ax = plt.subplots(figsize=(8, 5))

# PCA curve (mean ± std)
ax.plot(
    pcs, pca_means,
    marker="o",
    linewidth=2.5,
    color=colors["pca"],
    label="PCA (Linear Probe)"
)
ax.fill_between(
    pcs,
    pca_means - pca_stds,
    pca_means + pca_stds,
    color=colors["pca"],
    alpha=0.15
)

# Hebi reference lines
ax.axhline(
    hebi_lp_mean,
    linestyle="--",
    linewidth=2,
    color=colors["lp"],
    label=f"Hebi (LP) — mean={hebi_lp_mean:.3f}"
)
ax.axhline(
    hebi_ft_mean,
    linestyle="-.",
    linewidth=2,
    color=colors["ft"],
    label=f"Hebi (FT) — mean={hebi_ft_mean:.3f}"
)

# annotate PCA points
for x, y in zip(pcs, pca_means):
    ax.text(x, y + 0.0008, f"{y:.3f}", ha="center", va="bottom", fontsize=9)

# axes / labels
ax.set_xlabel("Number of PCA Components")
ax.set_ylabel("Macro-F1 (mean ± std over 5 seeds)")
ax.set_title("PCA Dimensional Sweep vs Hebi")
ax.set_ylim(0.33, 0.70)
ax.set_xticks(pcs)

# optional: log-scale x for nicer spacing
# ax.set_xscale("log")
# ax.set_xticks(pcs)
# ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.legend(frameon=False)
ax.grid(False)

plt.tight_layout()
plt.savefig("supp_pca_sweep_vs_hebi_enhanced_livercosmax.png", dpi=600, bbox_inches="tight")
plt.show()
###########################3
import numpy as np
import matplotlib.pyplot as plt

# ---- your seed-level results ----
hebi_lp = np.array([0.5037, 0.4872, 0.4911, 0.4489, 0.5028], dtype=float)
hebi_ft = np.array([0.6064, 0.6671, 0.6749, 0.6649, 0.6549], dtype=float)
pca_by_pcs = {
    10:  np.array([0.3566, 0.3769, 0.3656, 0.3607, 0.3730], dtype=float),
    30:  np.array([0.3872, 0.3846, 0.3848, 0.3858, 0.3807], dtype=float),
    64:  np.array([0.3878, 0.3885, 0.3872, 0.3859, 0.3878], dtype=float),
    128: np.array([0.3990, 0.4017, 0.4009, 0.4005, 0.4013], dtype=float),
    256: np.array([0.4028, 0.4021, 0.4016, 0.4037, 0.4045], dtype=float),
    384: np.array([0.4060, 0.4054, 0.4046, 0.4030, 0.4045], dtype=float),
}

# ---- means ----
pcs = sorted(pca_by_pcs.keys())
pca_means = [float(pca_by_pcs[p].mean()) for p in pcs]
hebi_lp_mean = float(hebi_lp.mean())
hebi_ft_mean = float(hebi_ft.mean())

labels = [f"PCA-{p}" for p in pcs] + ["Hebi-LP", "Hebi-FT"]
scores = pca_means + [hebi_lp_mean, hebi_ft_mean]

# ---- colours: one for PCA, different for Hebi ----
colors = ["#9e9e9e"] * len(pcs) + ["#1f77b4", "#ff7f0e"]

# ---- bar chart ----
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(labels))
bars = ax.bar(x, scores, color=colors, width=0.75)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel("Macro-F1 (mean over 5 seeds)")
ax.set_title("Supplement — PCA sweep vs Hebi (means)")
ax.set_ylim(0.3, 0.7)

# values on top (clean)
for b, v in zip(bars, scores):
    ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("supp_bar_pca_sweep_vs_hebi_means_livercosmax.png", dpi=600, bbox_inches="tight")
plt.show()
#######################3
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

hebi_lp = np.array([0.9324, 0.9419, 0.9384, 0.9260, 0.9475], dtype=float)
hebi_ft = np.array([0.9782, 0.9758, 0.9742, 0.9747, 0.9809], dtype=float)

pca_by_pcs = {
    10:  np.array([0.8205, 0.8230, 0.8227, 0.8201, 0.8237], dtype=float),
    30:  np.array([0.9428, 0.9428, 0.9429, 0.9428, 0.9432], dtype=float),
    64:  np.array([0.9491, 0.9493, 0.9489, 0.9494, 0.9488], dtype=float),
    128: np.array([0.9535, 0.9537, 0.9538, 0.9535, 0.9546], dtype=float),
    256: np.array([0.9572, 0.9574, 0.9572, 0.9572, 0.9570], dtype=float),
    384: np.array([0.9596, 0.9598, 0.9596, 0.9598, 0.9596], dtype=float),
}
additional_means = {
    "Nicheformer — Fine-tuned": 0.53,
    "Nicheformer — Linear probe": 0.10,
    "Nicheformer (liver only, linear-probing)": 0.53,
    "scVI": 0.49,
}

print("\n===== Hebi (LP) vs PCA =====\n")

for pcs, pca_scores in pca_by_pcs.items():

    # Paired t-test
    t_stat, t_p = ttest_rel(hebi_lp, pca_scores)

    # Wilcoxon
    w_stat, w_p = wilcoxon(hebi_lp, pca_scores)

    print(f"PCA-{pcs:>3}: "
          f"Mean diff = {(hebi_lp - pca_scores).mean():.4f} | "
          f"t-test p = {t_p:.5f} | "
          f"Wilcoxon p = {w_p:.5f}")
    

print("\n===== Hebi (FT) vs PCA =====\n")

for pcs, pca_scores in pca_by_pcs.items():

    t_stat, t_p = ttest_rel(hebi_ft, pca_scores)
    w_stat, w_p = wilcoxon(hebi_ft, pca_scores)

    print(f"PCA-{pcs:>3}: "
          f"Mean diff = {(hebi_ft - pca_scores).mean():.4f} | "
          f"t-test p = {t_p:.5f} | "
          f"Wilcoxon p = {w_p:.5f}")
print("\n===== Additional Methods (mean comparison only) vs PCA =====\n")

for method_name, method_mean in additional_means.items():
    for pcs, pca_scores in pca_by_pcs.items():
        mean_diff = method_mean - pca_scores.mean()

        print(f"{method_name} vs PCA-{pcs:>3}: "
              f"Mean diff = {mean_diff:.4f}")
##########################cancer
import numpy as np
import matplotlib.pyplot as plt

# ---- your seed-level results ----
hebi_lp = np.array([0.6523, 0.6445, 0.6506, 0.6543, 0.6474], dtype=float)
hebi_ft = np.array([0.7900, 0.8005, 0.7974, 0.7899, 0.7956], dtype=float)
pca_by_pcs = {
    10:  np.array([0.4785, 0.4825, 0.4853, 0.4737, 0.4787], dtype=float),
    30:  np.array([0.5402, 0.5402, 0.5389, 0.5381, 0.5365], dtype=float),
    64:  np.array([0.5484, 0.5471, 0.5493, 0.5499, 0.5450], dtype=float),
    128: np.array([0.5522, 0.5521, 0.5504, 0.5473, 0.5509], dtype=float),
    256: np.array([0.5566, 0.5613, 0.5590, 0.5615, 0.5605], dtype=float),
    384: np.array([0.5664, 0.5680, 0.5638, 0.5650, 0.5669], dtype=float),
}

# ---- means ----
pcs = sorted(pca_by_pcs.keys())
pca_means = [float(pca_by_pcs[p].mean()) for p in pcs]
hebi_lp_mean = float(hebi_lp.mean())
hebi_ft_mean = float(hebi_ft.mean())

labels = [f"PCA-{p}" for p in pcs] + ["Hebi-LP", "Hebi-FT"]
scores = pca_means + [hebi_lp_mean, hebi_ft_mean]

# ---- colours: one for PCA, different for Hebi ----
colors = ["#9e9e9e"] * len(pcs) + ["#1f77b4", "#ff7f0e"]

# ---- bar chart ----
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(labels))
bars = ax.bar(x, scores, color=colors, width=0.75)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel("Macro-F1 (mean over 5 seeds)")
ax.set_title("Supplement — PCA sweep vs Hebi (means)")
ax.set_ylim(0.4, 0.9)

# values on top (clean)
for b, v in zip(bars, scores):
    ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("supp_bar_pca_sweep_vs_hebi_means_livercosmax-cancer.png", dpi=600, bbox_inches="tight")
plt.show()
########################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 12:04:52 2026

@author: tower4
"""

import matplotlib.pyplot as plt
import numpy as np

labels = [
    "HEBI — Fine-tuned",
    "HEBI — Linear probe",
    "scVI",
    "PCA — 64 PCs",
    "PCA — 30 PCs",
    "PCA — 10 PCs"    
]

scores = [
    float(np.mean([0.7900, 0.8005, 0.7974, 0.7899, 0.7956])),
    float(np.mean([0.6523, 0.6445, 0.6506, 0.6543, 0.6474])),
    float(np.mean( [0.6683, 0.6696, 0.6688, 0.6684, 0.6689])),
    float(np.mean([0.5484, 0.5471, 0.5493, 0.5499, 0.5450])),
    float(np.mean([0.5402, 0.5402, 0.5389, 0.5381, 0.5365])),
    float(np.mean([0.4785, 0.4825, 0.4853, 0.4737, 0.4787]))
]

families = [
    "Hebi", "Hebi",
    "scVI" ,
    "PCA", "PCA", "PCA",
]

unique_fams = list(dict.fromkeys(families))

# ----------------------------
# Vibrant, high-contrast palette (colourblind-friendly)
# ----------------------------
vibrant = [
    "#6E8E84",  # Midnight Violet (Deepest)
    "#635666" ,
        "#5F4B8B",  # Royal Amethyst
       
         # Dusty Slate (A bonus 'neutral' purple)
]

fam2color = {fam: vibrant[i % len(vibrant)] for i, fam in enumerate(unique_fams)}
bar_colors = [fam2color[f] for f in families]

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(labels))

bars = ax.barh(
    y, scores,
    color=bar_colors,
    edgecolor="white",
    linewidth=1.2
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Macro-F1")
ax.set_xlim(0, 1.05)
ax.set_title("")

# cleaner look
ax.grid(axis="x", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Put value inside bar
for bar, v in zip(bars, scores):
    x = v - 0.02
    x = max(x, 0.02)
    txt_color = "white" if v > 0.18 else "black"
    ax.text(
        x, bar.get_y() + bar.get_height()/2,
        f"{v:.3f}",
        va="center", ha="right",
        color=txt_color, fontsize=10, fontweight="bold"
    )

def add_sig_h(ax, bar1, bar2, text="***", xpad=0.006, tick=0.012, text_dx=0.004):
    y1 = bar1.get_y() + bar1.get_height() / 2
    y2 = bar2.get_y() + bar2.get_height() / 2
    x_right = max(bar1.get_width(), bar2.get_width()) + xpad
    xmin, xmax = ax.get_xlim()
    x_right = min(x_right, xmax - (tick + 0.02))

    ax.plot([x_right, x_right], [y1, y2], color="black", lw=1.2)
    ax.plot([x_right - tick, x_right], [y1, y1], color="black", lw=1.2)
    ax.plot([x_right - tick, x_right], [y2, y2], color="black", lw=1.2)

    ax.text(min(x_right + text_dx, xmax - 0.005), (y1 + y2) / 2, text,
            va="center", ha="left", fontsize=14, fontweight="bold")

idx_hebi_ft = labels.index("HEBI — Fine-tuned")
idx_pca64   = labels.index("scVI")
add_sig_h(ax, bars[idx_hebi_ft], bars[idx_pca64], text="***")

# legend (unique families)
handles = []
legend_labels = []
for fam in unique_fams:
    h = plt.Line2D([0], [0], marker='s', linestyle='',
                   markersize=10, color=fam2color[fam])
    handles.append(h)
    legend_labels.append(fam)

ax.legend(handles, legend_labels, title="Method", loc="lower right", frameon=False)

# p-value note
p_one = 2.9e-07
ax.text(
    0.56, 0.02,
    f"***  p = {p_one:.2e} ",
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
)

plt.tight_layout()
out_png = "niche_model_comparison_livercancer_cosmax.png"
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.show()#####
########################################3
import numpy as np
import matplotlib.pyplot as plt

# ---- PCA mean macro-F1 per PCs (from your 5 seeds) ----
pca_by_pcs = {
    10:  np.array([0.4785, 0.4825, 0.4853, 0.4737, 0.4787], dtype=float),
    30:  np.array([0.5402, 0.5402, 0.5389, 0.5381, 0.5365], dtype=float),
    64:  np.array([0.5484, 0.5471, 0.5493, 0.5499, 0.5450], dtype=float),
    128: np.array([0.5522, 0.5521, 0.5504, 0.5473, 0.5509], dtype=float),
    256: np.array([0.5566, 0.5613, 0.5590, 0.5615, 0.5605], dtype=float),
    384: np.array([0.5664, 0.5680, 0.5638, 0.5650, 0.5669], dtype=float),
}

pcs = np.array(sorted(pca_by_pcs.keys()))
pca_mean_f1 = np.array([pca_by_pcs[p].mean() for p in pcs])

# ---- cumulative explained variance you provided ----
cum_var = {
    10:  0.1319,
    30:  0.1795,
    64:  0.2455,
    128:0.3557,
    256: 0.5319,
    384: 0.6626,
}

pca_cum_var = np.array([cum_var[p] for p in pcs])

# ---- choose two distinct colours ----
col_f1  = "#1f77b4"  # blue
col_var = "#ff7f0e"  # orange

fig, ax1 = plt.subplots(figsize=(7.6, 4.8))

# left axis: macro-F1
ax1.plot(pcs, pca_mean_f1, marker="o", color=col_f1, label="PCA macro-F1 (mean)")
ax1.set_xlabel("Number of PCA PCs")
ax1.set_ylabel("Macro-F1 (mean over 5 seeds)", color=col_f1)
ax1.tick_params(axis="y", labelcolor=col_f1)
ax1.set_ylim(0.2, 0.99)
ax1.set_xticks(pcs)

# right axis: cumulative variance
ax2 = ax1.twinx()
ax2.plot(pcs, pca_cum_var, marker="s", linestyle="--", color=col_var, label="Cumulative variance")
ax2.set_ylabel("Cumulative explained variance ratio", color=col_var)
ax2.tick_params(axis="y", labelcolor=col_var)
ax2.set_ylim(0.0, 1.0)

# combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

ax1.set_title("PCA performance vs cumulative variance")

plt.tight_layout()
plt.savefig("supp_line_f1_and_variance_vs_pcs_livercancer.png", dpi=600, bbox_inches="tight")
#plt.savefig("supp_line_f1_and_variance_vs_pcs.pdf", bbox_inches="tight")
plt.show()
############################Merfish brain
import matplotlib.pyplot as plt
import numpy as np

labels = [
    "HEBI — Fine-tuned",
    "HEBI — Linear probe",
    "PCA — 64 PCs",
    "PCA — 30 PCs",
    "PCA — 10 PCs",
    "Nicheformer — Fine-tuned",
    "Nicheformer — Linear probe",
    "Geneformer",
    "scGPT",
    "scVI "
]

scores = [
    float(np.mean([0.9782, 0.9758, 0.9742, 0.9747, 0.9809])),
    float(np.mean([0.9324, 0.9419, 0.9384, 0.9260, 0.9475])),
    float(np.mean([0.9491, 0.9493, 0.9489, 0.9494, 0.9488])),
    float(np.mean([0.9428, 0.9428, 0.9429, 0.9428, 0.9432])),
    float(np.mean([0.8205, 0.8230, 0.8227, 0.8201, 0.8237])),
    0.89,
    0.69,
    0.45,
    0.43,
    0.33,
]

families = [
    "Hebi", "Hebi",
    "PCA", "PCA", "PCA",
    "Nicheformer", "Nicheformer",
    "Geneformer",
    "scGPT",
    "scVI" 
]

unique_fams = list(dict.fromkeys(families))

# ----------------------------
# Vibrant, high-contrast palette (colourblind-friendly)
# ----------------------------
# vibrant = [
#     "#3B82F6",  # electric blue
#     "#EAB308",  # vivid orange
#     "#22C55E",  # vivid green
#     "#EF4444",  # vivid red
#     "#A855F7",  # vivid purple
#     "#EC4899",  # bright cyan
#     "#EAB308",  # vivid yellow
#     "#EC4899",  # hot pink
#     "#14B8A6",  # teal
#     "#8B5CF6",  # indigo
# ]
vibrant = [
"#6E8E84",  # Midnight Violet (Deepest)
    "#BF40BF",  # Royal Amethyst
    "#8E7697",  # Antique Lavender
    "#A28089",  # Rose-Mauve (The classic pinkish-dusty tone)
    "#708ABB",  # Misted Orchid (The 'Bright' limit)
    "#635666"   # Dusty Slate (A bonus 'neutral' purple)
    
]
fam2color = {fam: vibrant[i % len(vibrant)] for i, fam in enumerate(unique_fams)}
bar_colors = [fam2color[f] for f in families]

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(labels))

bars = ax.barh(
    y, scores,
    color=bar_colors,
    edgecolor="white",
    linewidth=1.2
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Macro-F1")
ax.set_xlim(0, 1.05)
ax.set_title("")

# cleaner look
ax.grid(axis="x", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Put value inside bar
for bar, v in zip(bars, scores):
    x = v - 0.02
    x = max(x, 0.02)
    txt_color = "white" if v > 0.0 else "black"
    ax.text(
        x, bar.get_y() + bar.get_height()/2,
        f"{v:.3f}",
        va="center", ha="right",
        color=txt_color, fontsize=10, fontweight="bold"
    )

def add_sig_h(ax, bar1, bar2, text="***", xpad=0.006, tick=0.012, text_dx=0.004):
    y1 = bar1.get_y() + bar1.get_height() / 2
    y2 = bar2.get_y() + bar2.get_height() / 2
    x_right = max(bar1.get_width(), bar2.get_width()) + xpad
    xmin, xmax = ax.get_xlim()
    x_right = min(x_right, xmax - (tick + 0.02))

    ax.plot([x_right, x_right], [y1, y2], color="black", lw=1.2)
    ax.plot([x_right - tick, x_right], [y1, y1], color="black", lw=1.2)
    ax.plot([x_right - tick, x_right], [y2, y2], color="black", lw=1.2)

    ax.text(min(x_right + text_dx, xmax - 0.005), (y1 + y2) / 2, text,
            va="center", ha="left", fontsize=14, fontweight="bold")

idx_hebi_ft = labels.index("HEBI — Fine-tuned")
idx_pca64   = labels.index("PCA — 64 PCs")
add_sig_h(ax, bars[idx_hebi_ft], bars[idx_pca64], text="***")

# legend (unique families)
handles = []
legend_labels = []
for fam in unique_fams:
    h = plt.Line2D([0], [0], marker='s', linestyle='',
                   markersize=10, color=fam2color[fam])
    handles.append(h)
    legend_labels.append(fam)

ax.legend(handles, legend_labels, title="Method", loc="lower right", frameon=False)

# p-value note
p_one = 0.00018
ax.text(
    0.56, 0.02,
    f"***  p = {p_one:.2e} ",
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
)

plt.tight_layout()
out_png = "niche_model_comparison_brain_merfish.png"
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.show()
######################
import matplotlib.pyplot as plt
import numpy as np

labels = [
    "HEBI — Fine-tuned",
    "HEBI — Linear probe",
    "PCA — 64 PCs",
    "PCA — 30 PCs",
    "PCA — 10 PCs",
    "Nicheformer — Fine-tuned",
    "Nicheformer — Linear probe",
    "Geneformer",
    "scGPT",
    "scVI "
]

scores = [
    float(np.mean([0.9782, 0.9758, 0.9742, 0.9747, 0.9809])),
    float(np.mean([0.9324, 0.9419, 0.9384, 0.9260, 0.9475])),
    float(np.mean([0.9491, 0.9493, 0.9489, 0.9494, 0.9488])),
    float(np.mean([0.9428, 0.9428, 0.9429, 0.9428, 0.9432])),
    float(np.mean([0.8205, 0.8230, 0.8227, 0.8201, 0.8237])),
    0.89,
    0.69,
    0.45,
    0.43,
    0.33,
]

families = [
    "Hebi", "Hebi",
    "PCA", "PCA", "PCA",
    "Nicheformer", "Nicheformer",
    "Geneformer",
    "scGPT",
    "scVI"
]

unique_fams = list(dict.fromkeys(families))

vibrant = [
   "#005F56",  # Signature Berry
    "#3A9BB7",  # Warm Saffron
    "#A1C9F4",  # Deep Fig
    "#D0D1E6",  # Pomegranate Red
    "#E7E1EF",  # Dusty Mauve
    "#F3E8F5",
    
]

fam2color = {fam: vibrant[i % len(vibrant)] for i, fam in enumerate(unique_fams)}
bar_colors = [fam2color[f] for f in families]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(labels))

bars = ax.bar(
    x, scores,
    color=bar_colors,
    edgecolor="white",
    linewidth=1.2
)

# x ticks
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1.05)

# cleaner look
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# value labels (inside/top)
for bar, v in zip(bars, scores):
    y = v - 0.03
    y = max(y, 0.02)
    ax.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f"{v:.3f}",
        va="top", ha="center",
        color="white",
        fontsize=10,
        fontweight="bold"
    )

def add_sig_v(ax, bar1, bar2, text="***", ypad=0.01, tick=0.02, text_dy=0.01):
    """
    Vertical version of significance bracket:
    draws a bracket above bar1 and bar2.
    """
    x1 = bar1.get_x() + bar1.get_width()/2
    x2 = bar2.get_x() + bar2.get_width()/2
    y_top = max(bar1.get_height(), bar2.get_height()) + ypad

    ymin, ymax = ax.get_ylim()
    y_top = min(y_top, ymax - (tick + 0.03))

    ax.plot([x1, x2], [y_top, y_top], color="black", lw=1.2)
    ax.plot([x1, x1], [y_top - tick, y_top], color="black", lw=1.2)
    ax.plot([x2, x2], [y_top - tick, y_top], color="black", lw=1.2)

    ax.text((x1 + x2)/2, min(y_top + text_dy, ymax - 0.01), text,
            va="bottom", ha="center", fontsize=14, fontweight="bold")

idx_hebi_ft = labels.index("HEBI — Fine-tuned")
idx_pca64   = labels.index("PCA — 64 PCs")
add_sig_v(ax, bars[idx_hebi_ft], bars[idx_pca64], text="***")

# legend (unique families)
handles = []
legend_labels = []
for fam in unique_fams:
    h = plt.Line2D([0], [0], marker='s', linestyle='',
                   markersize=10, color=fam2color[fam])
    handles.append(h)
    legend_labels.append(fam)

ax.legend(handles, legend_labels, title="Method", loc="upper right", frameon=False)

# p-value note
p_one = 0.00018
ax.text(
    0.02, 0.02,
    f"***  p = {p_one:.2e}",
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
)

plt.tight_layout()
out_png = "niche_model_comparison_brain_merfish_vertical.png"
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.show()