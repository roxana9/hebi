#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised loader consistent with your PRETRAINING:

- Load slide dict from .pt
- Build LocalSampler (same as pretraining)
- Yield batches in the style:
    (edge_block, node_idx_b, bs, y_seed)
  OR, more convenient:
    (x_subgraph_raw, edge_index_subgraph, edge_attr_b, node2comm_batch, super_sp_global, gene_ids, y_seed, bs)

This version yields the convenient form (ready for model forward).
"""

import os
import glob
import torch
from hebi.data.sampler import LocalSampler


def get_slide_list(root_dir, split="train"):
    search_path = os.path.join(root_dir, split, "*.pt")
    files = sorted(glob.glob(search_path))
    if len(files) == 0:
        raise RuntimeError(f"No .pt files found in {search_path}")
    print(f"📁 Found {len(files)} slides for split '{split}'")
    return files


def load_supervised_slide_dict(pt_path, target_key="y_class", map_location="cpu"):
    """
    Loads the .pt slide dict and checks target labels exist.
    Returns the raw dict (like your pretraining load_data output).
    """
    slide = torch.load(pt_path, map_location=map_location)

    if "X" not in slide or "edge_index" not in slide or "node2comm" not in slide or "super_sp" not in slide:
        raise KeyError(f"Slide is missing required keys. Got keys: {list(slide.keys())}")

    if target_key not in slide:
        raise KeyError(
            f"Target '{target_key}' not found in {pt_path}. Available keys: {list(slide.keys())}"
        )

    return slide


def build_local_sampler(slide, batch_size=512, num_workers=4, shuffle=True):
    """
    Builds LocalSampler exactly like pretraining, using slide['edge_index'] and slide['X'].
    """
    edge_index = slide["edge_index"]
    if edge_index.is_cuda:
        edge_index = edge_index.cpu()

    sampler = LocalSampler(
        edge_index=edge_index,
        sizes=[20, 10],                 # same as pretraining
        num_nodes=slide["X"].shape[0],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return sampler


def iterate_supervised_batches(slide, sampler, device, target_key="y_class"):
    """
    Yields batches that match pretraining objects, plus y labels for seed nodes.

    Output tuple:
      x_subgraph_raw      : [N_sub, G_slide] float32 (on device)
      edge_index_subgraph : [2, E_sub] long    (on device)
      edge_attr_b         : (attr0, attr1) each on device (like pretraining)
      node2comm_batch     : [bs] long (on device, global comm id per seed)
      super_sp_global     : slide-level tensor (on device)
      gene_ids            : [G_slide] long (on device)
      y_seed              : [bs] long (on device)
      bs                  : int
    """

    X = slide["X"].float().to(device)
    edge_index_full = slide["edge_index"].long().to(device)
    node2comm_fixed = slide["node2comm"].long().to(device)
    super_sp_fixed = slide["super_sp"].float().to(device)
    gene_ids = slide["gene_ids"].long().to(device)
    y_all = slide[target_key].long().to(device)

    for edge_block, node_idx_b, bs in sampler:
        # node_idx_b is on CPU typically -> move to device
        node_idx_b = node_idx_b.to(device)

        # Subgraph raw genes (before gene projection)
        x_subgraph_raw = X[node_idx_b]  # [N_sub, G_slide]

        # edge_block structure expected like pretraining:
        # edge_block[:2] are edges, [2] and [3] are edge attributes
        # If your LocalSampler yields a different format, print len(edge_block) to debug.
        edge_index_subgraph = edge_block[:2].to(device)

        # This matches your pretraining:
        edge_attr_b = (
            edge_block[2].to(device),
            edge_block[3].to(device),
        )

        # seed nodes = first bs indices in node_idx_b (same as pretraining)
        seed_global = node_idx_b[:bs]

        node2comm_batch = node2comm_fixed[seed_global]  # [bs]
        y_seed = y_all[seed_global]                      # [bs]

        yield (
            x_subgraph_raw,
            edge_index_subgraph,
            edge_attr_b,
            node2comm_batch,
            super_sp_fixed,
            gene_ids,
            y_seed,
            bs,
        )
