#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:07:05 2025

@author: roxana
"""
import mygene
import json
import numpy as np
def map_to_human(gene_list, species="mouse"):
    """
    Map gene symbols from mouse, rat, or another species to human HGNC
    symbols via HomoloGene.

    Two lookups are needed. The homologene field gives [taxid, entrez_id]
    pairs, so the human entries yield entrez IDs rather than symbols; a
    second query converts those IDs to symbols.

    Genes with no homolog are absent from the returned mapping, and the
    caller falls back to upper-casing the original symbol.
    """
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        gene_list,
        scopes="symbol",
        fields="homologene",
        species=species,
        verbose=False,
    )

    # query symbol -> human entrez id
    query_to_entrez = {}
    for r in res:
        if r.get("notfound"):
            continue
        homologene = r.get("homologene")
        if not homologene:
            continue
        for entry in homologene.get("genes", []):
            if len(entry) == 2 and entry[0] == 9606:   # human taxon
                query_to_entrez[r["query"]] = entry[1]
                break

    if not query_to_entrez:
        return {}

    # human entrez id -> human symbol
    entrez_ids = sorted(set(query_to_entrez.values()))
    sym_res = mg.querymany(
        entrez_ids,
        scopes="entrezgene",
        fields="symbol",
        species="human",
        verbose=False,
    )

    entrez_to_symbol = {}
    for r in sym_res:
        if r.get("notfound"):
            continue
        symbol = r.get("symbol")
        if isinstance(symbol, str):
            entrez_to_symbol[int(r["query"])] = symbol.upper()

    mapping = {
        query: entrez_to_symbol[entrez]
        for query, entrez in query_to_entrez.items()
        if entrez in entrez_to_symbol
    }

    print(f"  homolog mapping: {len(mapping)}/{len(gene_list)} genes mapped "
          f"to human symbols")
    return mapping
def harmonise_adata(adata,vocab_path=None,  species="human"):
    # pick source gene names
    if "gene_name" in adata.var.columns:
        gene_names = adata.var["gene_name"].astype(str).tolist()
    else:
        gene_names = adata.var_names.astype(str).tolist()
    
    gene_names = [g.strip() for g in gene_names]

    if species.lower() == "human":
        # already human → just uppercase
        human_syms = [g.upper() for g in gene_names]
        adata.var["gene_symbol"] = [g.upper() for g in gene_names]
    else:
        # map to human homologs
        mapping = map_to_human(gene_names, species=species.lower())
        human_syms = [mapping.get(g, g.upper()) for g in gene_names]
    adata.var["gene_symbol"] = human_syms
    
    # spatial coordinates fix as before
    if "spatial" not in adata.obsm.keys():
        for key in adata.obsm.keys():
            if "spatial" in key.lower():
                adata.obsm["spatial"] = adata.obsm[key]
                break
    
    if vocab_path is not None:
        with open(vocab_path, "r") as f:
            vocab = json.load(f)            # name -> id (int)
        vocab_set = set(vocab.keys())

        # mark coverage
        syms = adata.var["gene_symbol"].astype(str)
        in_vocab = syms.isin(vocab_set)

        
        adata = adata[:, in_vocab.values].copy()
        syms = adata.var["gene_symbol"].astype(str)  # refresh after slice
            # assign numeric ids for the kept genes
        gene_ids = np.array([vocab[g] for g in syms], dtype=np.int64)
        adata.var["gene_id"] = gene_ids
        covered = int(in_vocab.sum())
        total = len(human_syms)
        print(f"Covered genes: {covered}/{total} ({covered/total*100:.2f}%)")

    
    return adata
