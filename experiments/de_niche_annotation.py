#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:15:34 2026

@author: rokny-tower4
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-platform differential expression on the joint niches, for cross-platform
annotation of the Xenium + MERSCOPE breast pair.

Because each slide keeps its native panel, DE is run on each platform
INDEPENDENTLY (a niche is characterised by that platform's own markers), then
the two marker lists are put side by side so you can name the niche from the
union of evidence.

Usage
-----
from de_niche_annotation import run_de_annotation

res = run_de_annotation(
    results,                                   # [adata_xenium, adata_merscope]
    platform_names=["Xenium", "MERSCOPE"],
    niche_key="comp_niches",
    out_xlsx="niche_markers_k7.xlsx",
)

Writes an .xlsx with one sheet per platform (full ranked DE), a side-by-side
"top_markers" sheet, a "shared_gene_markers" sheet restricted to genes present
in BOTH panels (the evidence that matters for cross-platform correspondence),
and an "annotation" sheet for you to fill in the biological names.
"""
import numpy as np
import pandas as pd
import scanpy as sc


# --------------------------------------------------------------------- DE core
def de_one_platform(adata, niche_key, n_genes=25, layer=None):
    """Wilcoxon rank-sum, one niche vs all others, on this platform's own panel."""
    ad = adata.copy()
    ad.obs[niche_key] = ad.obs[niche_key].astype(str)

    # drop niches too small to test
    vc = ad.obs[niche_key].value_counts()
    keep = vc[vc >= 3].index
    ad = ad[ad.obs[niche_key].isin(keep)].copy()

    sc.tl.rank_genes_groups(
        ad, groupby=niche_key, method="wilcoxon",
        layer=layer, use_raw=False, n_genes=n_genes,
    )

    rows = []
    for niche in ad.obs[niche_key].cat.categories if hasattr(
            ad.obs[niche_key], "cat") else sorted(ad.obs[niche_key].unique()):
        df = sc.get.rank_genes_groups_df(ad, group=niche)
        df.insert(0, "niche", niche)
        df["rank"] = np.arange(1, len(df) + 1)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"names": "gene", "logfoldchanges": "logFC",
                              "pvals": "pval", "pvals_adj": "pval_adj"})
    return out[["niche", "rank", "gene", "logFC", "scores", "pval", "pval_adj"]]


def top_marker_string(de_df, niche, k=8):
    sub = de_df[de_df.niche == niche].nsmallest(k, "rank")
    return ", ".join(sub.gene.tolist())


# ------------------------------------------------------------------- driver
def run_de_annotation(results, platform_names, niche_key="comp_niches",
                      out_xlsx="niche_markers.xlsx", n_genes=25, layer=None):
    de_tables = {}
    for ad, pname in zip(results, platform_names):
        print(f"  DE on {pname} ({ad.n_obs} cells, {ad.n_vars} genes)...")
        de_tables[pname] = de_one_platform(ad, niche_key, n_genes=n_genes,
                                           layer=layer)

    niches = sorted(
        set().union(*[set(t.niche) for t in de_tables.values()]),
        key=lambda s: (len(s), s))

    # side-by-side top markers + platform composition
    comp = []
    for ad, pname in zip(results, platform_names):
        c = ad.obs[niche_key].astype(str).value_counts()
        comp.append(c.rename(pname))
    comp = pd.concat(comp, axis=1).fillna(0)
    frac_first = comp[platform_names[0]] / comp.sum(axis=1)

    top = pd.DataFrame({"niche": niches})
    for pname in platform_names:
        top[f"top_markers_{pname}"] = [
            top_marker_string(de_tables[pname], n) for n in niches]
    for pname in platform_names:
        top[f"n_cells_{pname}"] = [int(comp.loc[n, pname]) if n in comp.index
                                   else 0 for n in niches]
    top[f"frac_{platform_names[0]}"] = [round(frac_first.get(n, np.nan), 3)
                                        for n in niches]
    top["type"] = np.where(top[f"frac_{platform_names[0]}"] > 0.85,
                           f"{platform_names[0]}-specific",
                    np.where(top[f"frac_{platform_names[0]}"] < 0.15,
                             f"{platform_names[1]}-specific", "shared"))

    # markers restricted to genes measured on BOTH panels
    shared_genes = set(results[0].var_names) & set(results[1].var_names)
    print(f"  shared genes across panels: {len(shared_genes)}")
    shared_rows = []
    for pname, t in de_tables.items():
        s = t[t.gene.isin(shared_genes)].copy()
        s.insert(0, "platform", pname)
        shared_rows.append(s)
    shared_tbl = pd.concat(shared_rows, ignore_index=True)
    shared_tbl = shared_tbl.sort_values(["niche", "platform", "rank"])

    # annotation template for you to fill
    annot = pd.DataFrame({
        "niche": niches,
        f"frac_{platform_names[0]}": top[f"frac_{platform_names[0]}"],
        "type": top["type"],
        f"top_markers_{platform_names[0]}": top[f"top_markers_{platform_names[0]}"],
        f"top_markers_{platform_names[1]}": top[f"top_markers_{platform_names[1]}"],
        "annotation_FILL_IN": "",
        "notes": "",
    })

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        annot.to_excel(xw, sheet_name="annotation", index=False)
        top.to_excel(xw, sheet_name="top_markers", index=False)
        shared_tbl.to_excel(xw, sheet_name="shared_gene_markers", index=False)
        for pname, t in de_tables.items():
            t.to_excel(xw, sheet_name=f"DE_{pname}"[:31], index=False)

    _format_xlsx(out_xlsx)
    print(f"saved -> {out_xlsx}")
    return {"de": de_tables, "top": top, "shared": shared_tbl,
            "annotation": annot}


def _format_xlsx(path):
    """Readable column widths, frozen headers, Arial."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(name="Arial", bold=True)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        for col in ws.columns:
            letter = col[0].column_letter
            longest = max((len(str(c.value)) for c in col if c.value is not None),
                          default=10)
            ws.column_dimensions[letter].width = min(max(longest + 2, 10), 60)
            for c in col[1:]:
                if c.font is None or c.font.name != "Arial":
                    c.font = Font(name="Arial")
    wb.save(path)