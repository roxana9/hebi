import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compare_shared_edges_between_conditions(
    normal_csv,
    cancer_csv,
    top_k=50,
    sort_by="max",   # "max", "diff", "cancer", "normal"
    save_path=None
):
    """
    Compare shared GRN edges between normal and cancer.

    Parameters
    ----------
    normal_csv : str
        Path to normal edge CSV
    cancer_csv : str
        Path to cancer edge CSV
    top_k : int
        Number of shared edges to plot
    sort_by : str
        How to rank shared edges:
        - "max"    : by max(normal, cancer)
        - "diff"   : by abs(cancer - normal)
        - "cancer" : by cancer attention
        - "normal" : by normal attention
    save_path : str or None
        Output image path
    """

    normal_df = pd.read_csv(normal_csv)
    cancer_df = pd.read_csv(cancer_csv)

    required_cols = {"source_gene", "target_gene", "attention"}
    for name, df in [("normal", normal_df), ("cancer", cancer_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} csv missing columns: {missing}")

    # make edge key
    normal_df = normal_df.copy()
    cancer_df = cancer_df.copy()

    normal_df["edge"] = normal_df["source_gene"].astype(str) + " -> " + normal_df["target_gene"].astype(str)
    cancer_df["edge"] = cancer_df["source_gene"].astype(str) + " -> " + cancer_df["target_gene"].astype(str)

    # keep one row per edge (highest attention if duplicate)
    normal_df = (
        normal_df.sort_values("attention", ascending=False)
        .drop_duplicates(subset=["edge"])
        [["edge", "source_gene", "target_gene", "attention"]]
        .rename(columns={"attention": "attention_normal"})
    )

    cancer_df = (
        cancer_df.sort_values("attention", ascending=False)
        .drop_duplicates(subset=["edge"])
        [["edge", "source_gene", "target_gene", "attention"]]
        .rename(columns={"attention": "attention_cancer"})
    )

    # shared edges only
    merged = pd.merge(
        normal_df,
        cancer_df[["edge", "attention_cancer"]],
        on="edge",
        how="inner"
    )

    if merged.empty:
        raise ValueError("No shared edges found between the two CSV files.")

    # ranking
    merged["max_attention"] = merged[["attention_normal", "attention_cancer"]].max(axis=1)
    merged["abs_diff"] = (merged["attention_cancer"] - merged["attention_normal"]).abs()

    if sort_by == "max":
        merged = merged.sort_values("max_attention", ascending=False)
    elif sort_by == "diff":
        merged = merged.sort_values("abs_diff", ascending=False)
    elif sort_by == "cancer":
        merged = merged.sort_values("attention_cancer", ascending=False)
    elif sort_by == "normal":
        merged = merged.sort_values("attention_normal", ascending=False)
    else:
        raise ValueError("sort_by must be one of: max, diff, cancer, normal")

    plot_df = merged.head(top_k).copy()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)  # reverse for plotting

    # dumbbell plot
    y = np.arange(len(plot_df))

    plt.figure(figsize=(14, max(6, 0.35 * len(plot_df))))

    for i, row in plot_df.iterrows():
        plt.plot(
            [row["attention_normal"], row["attention_cancer"]],
            [i, i],
            color="lightgray",
            linewidth=1.5,
            zorder=1
        )

    plt.scatter(
        plot_df["attention_normal"],
        y,
        color="blue",
        s=60,
        label="Normal",
        zorder=2
    )

    plt.scatter(
        plot_df["attention_cancer"],
        y,
        color="red",
        s=60,
        label="Cancer",
        zorder=3
    )

    plt.yticks(y, plot_df["edge"], fontsize=10)
    plt.xlabel("Attention score", fontsize=12)
    plt.ylabel("Shared edge", fontsize=12)
    plt.title(f"Shared GRN edges: Normal vs Cancer (top {len(plot_df)})", fontsize=14)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return merged
shared_df = compare_shared_edges_between_conditions(
    normal_csv="/DATA1/Roxana/Hierarchy/Two_levels/clustering/grn_reports/Livernormal/liverNormalcosmax_global_directed_edges_ranked_competitive.csv",
    cancer_csv="/DATA1/Roxana/Hierarchy/Two_levels/clustering/grn_reports/livercancer_1//livercancercosmax_global_directed_edges_ranked_competitive.csv",
    top_k=30,
    sort_by="diff",
    save_path="normal_vs_cancer_shared_edges.png"
)
import pandas as pd

def print_shared_edge_stats(normal_csv, cancer_csv):
    normal_df = pd.read_csv(normal_csv)
    cancer_df = pd.read_csv(cancer_csv)

    required_cols = {"source_gene", "target_gene"}
    for name, df in [("normal", normal_df), ("cancer", cancer_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} csv is missing columns: {missing}")

    # Build edge IDs
    normal_df = normal_df.copy()
    cancer_df = cancer_df.copy()

    normal_df["edge"] = normal_df["source_gene"].astype(str) + " -> " + normal_df["target_gene"].astype(str)
    cancer_df["edge"] = cancer_df["source_gene"].astype(str) + " -> " + cancer_df["target_gene"].astype(str)

    # Unique edge sets
    normal_edges = set(normal_df["edge"])
    cancer_edges = set(cancer_df["edge"])

    shared_edges = normal_edges & cancer_edges
    only_normal = normal_edges - cancer_edges
    only_cancer = cancer_edges - normal_edges

    print("\n===== Shared Edge Statistics =====")
    print(f"Total unique edges in normal : {len(normal_edges)}")
    print(f"Total unique edges in cancer : {len(cancer_edges)}")
    print(f"Shared edges                : {len(shared_edges)}")
    print(f"Only in normal              : {len(only_normal)}")
    print(f"Only in cancer              : {len(only_cancer)}")

    # Optional percentages
    print("\n===== Overlap Percentages =====")
    print(f"Shared / normal : {len(shared_edges) / len(normal_edges):.2%}")
    print(f"Shared / cancer : {len(shared_edges) / len(cancer_edges):.2%}")

    return {
        "normal_edges": normal_edges,
        "cancer_edges": cancer_edges,
        "shared_edges": shared_edges,
        "only_normal": only_normal,
        "only_cancer": only_cancer,
    }
stats = print_shared_edge_stats(
normal_csv="/DATA1/Roxana/Hierarchy/Two_levels/clustering/grn_reports/Livernormal/liverNormalcosmax_global_directed_edges_ranked_competitive.csv",
cancer_csv="/DATA1/Roxana/Hierarchy/Two_levels/clustering/grn_reports/livercancer_1//livercancercosmax_global_directed_edges_ranked_competitive.csv",
)
import pandas as pd

def compare_shared_edge_attention(normal_csv, cancer_csv):
    normal_df = pd.read_csv(normal_csv).copy()
    cancer_df = pd.read_csv(cancer_csv).copy()

    required_cols = {"source_gene", "target_gene", "attention"}
    for name, df in [("normal", normal_df), ("cancer", cancer_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} csv is missing columns: {missing}")

    # edge key
    normal_df["edge"] = normal_df["source_gene"].astype(str) + " -> " + normal_df["target_gene"].astype(str)
    cancer_df["edge"] = cancer_df["source_gene"].astype(str) + " -> " + cancer_df["target_gene"].astype(str)

    # keep one row per edge, using highest attention if duplicated
    normal_df = (
        normal_df.sort_values("attention", ascending=False)
        .drop_duplicates("edge")
        [["edge", "source_gene", "target_gene", "attention"]]
        .rename(columns={"attention": "attention_normal"})
    )

    cancer_df = (
        cancer_df.sort_values("attention", ascending=False)
        .drop_duplicates("edge")
        [["edge", "attention"]]
        .rename(columns={"attention": "attention_cancer"})
    )

    merged = pd.merge(normal_df, cancer_df, on="edge", how="inner")

    merged["delta_cancer_minus_normal"] = (
        merged["attention_cancer"] - merged["attention_normal"]
    )
    merged["abs_delta"] = merged["delta_cancer_minus_normal"].abs()

    merged = merged.sort_values("abs_delta", ascending=False).reset_index(drop=True)

    print("\n===== Shared edges with largest absolute attention change =====")
    print(
        merged[
            ["source_gene", "target_gene", "attention_normal", "attention_cancer",
             "delta_cancer_minus_normal", "abs_delta"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n===== Top edges higher in cancer =====")
    print(
        merged.sort_values("delta_cancer_minus_normal", ascending=False)[
            ["source_gene", "target_gene", "attention_normal", "attention_cancer",
             "delta_cancer_minus_normal"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n===== Top edges higher in normal =====")
    print(
        merged.sort_values("delta_cancer_minus_normal", ascending=True)[
            ["source_gene", "target_gene", "attention_normal", "attention_cancer",
             "delta_cancer_minus_normal"]
        ]
        .head(20)
        .to_string(index=False)
    )

    return merged
shared_df = compare_shared_edge_attention(normal_csv="/DATA1/Roxana/Hierarchy/Two_levels/clustering/grn_reports/Livernormal/liverNormalcosmax_global_directed_edges_ranked_competitive.csv",
cancer_csv="/DATA1/Roxana/Hierarchy/Two_levels/clustering/grn_reports/livercancer_1/livercancercosmax_global_directed_edges_ranked_competitive.csv",
)


def plot_shared_edge_dumbbell(
    shared_df,
    top_k=40,
    mode="abs_delta",   # "abs_delta", "cancer_up", "normal_up"
    save_path=None
):
    df = shared_df.copy()

    if mode == "abs_delta":
        df = df.sort_values("abs_delta", ascending=False).head(top_k)
        title = f"Top {top_k} shared edges with largest attention change"
    elif mode == "cancer_up":
        df = df.sort_values("delta_cancer_minus_normal", ascending=False).head(top_k)
        title = f"Top {top_k} shared edges higher in cancer"
    elif mode == "normal_up":
        df = df.sort_values("delta_cancer_minus_normal", ascending=True).head(top_k)
        title = f"Top {top_k} shared edges higher in normal"
    else:
        raise ValueError("mode must be one of: abs_delta, cancer_up, normal_up")

    df = df.copy()
    df["edge_label"] = df["source_gene"] + " -> " + df["target_gene"]
    df = df.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(df))

    plt.figure(figsize=(14, max(8, 0.35 * len(df))))

    for i, row in df.iterrows():
        plt.plot(
            [row["attention_normal"], row["attention_cancer"]],
            [i, i],
            color="lightgray",
            linewidth=1.5,
            zorder=1
        )

    plt.scatter(
        df["attention_normal"], y,
        color="blue", s=60, label="Normal", zorder=2
    )
    plt.scatter(
        df["attention_cancer"], y,
        color="red", s=60, label="Cancer", zorder=3
    )

    plt.yticks(y, df["edge_label"], fontsize=10)
    plt.xlabel("Attention score", fontsize=12)
    plt.ylabel("Shared edge", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
plot_shared_edge_dumbbell(
    shared_df,
    top_k=40,
    mode="abs_delta",
    save_path="shared_edges_largest_change.png"
) 
plot_shared_edge_dumbbell(
    shared_df,
    top_k=40,
    mode="cancer_up",
    save_path="shared_edges_higher_in_cancer.png"
)
plot_shared_edge_dumbbell(
    shared_df,
    top_k=40,
    mode="normal_up",
    save_path="shared_edges_higher_in_normal.png"
)
shared_df.to_csv("shared_edges_normal_vs_cancer_attention.csv", index=False)

def inspect_unique_edges(normal_csv, cancer_csv):
    normal_df = pd.read_csv(normal_csv).copy()
    cancer_df = pd.read_csv(cancer_csv).copy()

    required_cols = {"source_gene", "target_gene", "attention"}
    for name, df in [("normal", normal_df), ("cancer", cancer_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} csv is missing columns: {missing}")

    # edge key
    normal_df["edge"] = normal_df["source_gene"].astype(str) + " -> " + normal_df["target_gene"].astype(str)
    cancer_df["edge"] = cancer_df["source_gene"].astype(str) + " -> " + cancer_df["target_gene"].astype(str)

    # keep strongest version if duplicates exist
    normal_df = (
        normal_df.sort_values("attention", ascending=False)
        .drop_duplicates("edge")
        .copy()
    )

    cancer_df = (
        cancer_df.sort_values("attention", ascending=False)
        .drop_duplicates("edge")
        .copy()
    )

    normal_edges = set(normal_df["edge"])
    cancer_edges = set(cancer_df["edge"])

    only_normal_edges = normal_edges - cancer_edges
    only_cancer_edges = cancer_edges - normal_edges

    only_normal_df = normal_df[normal_df["edge"].isin(only_normal_edges)].copy()
    only_cancer_df = cancer_df[cancer_df["edge"].isin(only_cancer_edges)].copy()

    only_normal_df = only_normal_df.sort_values("attention", ascending=False).reset_index(drop=True)
    only_cancer_df = only_cancer_df.sort_values("attention", ascending=False).reset_index(drop=True)

    print("\n===== Edges only in cancer =====")
    print(f"Count: {len(only_cancer_df)}")
    if len(only_cancer_df) > 0:
        cols = [c for c in ["source_gene", "target_gene", "attention", "edge_types", "evidence_tier"] if c in only_cancer_df.columns]
        print(only_cancer_df[cols].to_string(index=False))

    print("\n===== Edges only in normal =====")
    print(f"Count: {len(only_normal_df)}")
    if len(only_normal_df) > 0:
        cols = [c for c in ["source_gene", "target_gene", "attention", "edge_types", "evidence_tier"] if c in only_normal_df.columns]
        print(only_normal_df[cols].to_string(index=False))

    return only_normal_df, only_cancer_df

only_normal_df, only_cancer_df = inspect_unique_edges(normal_csv="/DATA1/Roxana/Hierarchy/Two_levels/clustering/grn_reports/Livernormal/liverNormalcosmax_global_directed_edges_ranked_competitive.csv",
cancer_csv="/DATA1/Roxana/Hierarchy/Two_levels/clustering/grn_reports/livercancer_1/livercancercosmax_global_directed_edges_ranked_competitive.csv",
)
def plot_unique_edges_combined(only_normal_df, only_cancer_df, figsize=(16, 12), save_path=None):
    ndf = only_normal_df.copy()
    cdf = only_cancer_df.copy()

    ndf["condition"] = "Normal only"
    cdf["condition"] = "Cancer only"

    df = pd.concat([ndf, cdf], axis=0, ignore_index=True)

    all_sources = sorted(df["source_gene"].unique())
    all_targets = sorted(df["target_gene"].unique())

    ymap = {g: i for i, g in enumerate(all_sources)}
    xmap = {g: i for i, g in enumerate(all_targets)}

    df["x"] = df["target_gene"].map(xmap)
    df["y"] = df["source_gene"].map(ymap)

    vals = df["attention"].values.astype(float)
    if vals.max() == vals.min():
        sizes = np.full(len(vals), 120.0)
    else:
        sizes = 40 + 260 * (vals - vals.min()) / (vals.max() - vals.min())

    plt.figure(figsize=figsize)

    mask_n = df["condition"] == "Normal only"
    mask_c = df["condition"] == "Cancer only"

    plt.scatter(
        df.loc[mask_n, "x"],
        df.loc[mask_n, "y"],
        s=sizes[mask_n],
        color="blue",
        alpha=0.8,
        label="Normal only"
    )

    plt.scatter(
        df.loc[mask_c, "x"],
        df.loc[mask_c, "y"],
        s=sizes[mask_c],
        color="red",
        alpha=0.8,
        label="Cancer only"
    )

    plt.xticks(range(len(all_targets)), all_targets, rotation=90, fontsize=10)
    plt.yticks(range(len(all_sources)), all_sources, fontsize=10)
    plt.xlabel("Target gene", fontsize=12)
    plt.ylabel("Source gene", fontsize=12)
    plt.title("Condition-specific unique GRN edges", fontsize=14)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
plot_unique_edges_combined(
    only_normal_df,
    only_cancer_df,
    figsize=(16, 12),
    save_path="unique_edges_combined.png"
)    