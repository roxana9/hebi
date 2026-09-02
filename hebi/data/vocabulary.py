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
    Map gene names from mouse/rat/etc. to human HGNC symbols.
    """
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        gene_list,
        scopes="symbol",
        fields="homologene",
        species=species
    )
    
    mapping = {}
    for r in res:
        if "notfound" in r and r["notfound"]:
            continue
        if "homologene" in r:
            # pick human homolog if exists
            homs = r["homologene"]["genes"]
            for sp_id, symbol in homs:
                if sp_id == 9606:   # human NCBI taxon ID
                    mapping[r["query"]] = symbol.upper()
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
