#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:07:45 2025

@author: roxana
"""
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path

# --- This import is now required ---
# --- Please install it: pip install nxmetis ---
import pymetis
@torch.no_grad()
def create_metis_partition(edge_index: torch.Tensor, num_nodes: int, num_partitions: int, seed: int = 42):
    """
    Runs METIS partitioning on the full graph *once*.
    This is a CPU-intensive pre-processing step.
    
    Returns:
        - node2comm (torch.LongTensor): [N0] tensor mapping node_idx -> partition_idx
        - super_sp (torch.FloatTensor): [K, K] distance matrix between partitions
    """
    print(f"🔧 [METIS] Creating NetworkX graph for {num_nodes} nodes...")
    src, dst = edge_index
    edge_list = [[] for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        edge_list[u].append(v)
        edge_list[v].append(u)
    
    print(f"🔧 [METIS] Running METIS partitioning to create {num_partitions} partitions...")
    try:
        _, communities = pymetis.part_graph(num_partitions, adjacency=edge_list)
    except ImportError:
        print("\n" + "="*50)
        print("ERROR: `nxmetis` library not found. Cannot perform METIS partitioning.")
        print("Please install it: `pip install nxmetis`")
        print("Using a RANDOM assignment as a fallback.")
        print("="*50 + "\n")
        g = torch.Generator().manual_seed(seed)
        node2comm_tensor = torch.randint(0, num_partitions, (num_nodes,), generator=g, dtype=torch.long)
    else:
        print("✅ [METIS] METIS partitioning complete.")
        # Convert community list to the node2comm tensor
       # node2comm_tensor = torch.zeros(num_nodes, dtype=torch.long)
        # PyMetis returns direct node-to-community mapping
        node2comm_tensor = torch.tensor(communities, dtype=torch.long)

            
    # --- Create Super-Graph Adjacency ---
    print("🔧 [METIS] Computing super-graph adjacency...")
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    cu = node2comm_tensor.cpu().numpy()[row]
    cv = node2comm_tensor.cpu().numpy()[col]
    
    data = np.ones_like(cu, dtype=np.int8)
    super_adj_coo = coo_matrix((data, (cu, cv)), shape=(num_partitions, num_partitions))
    super_adj = (super_adj_coo + super_adj_coo.T)
    super_adj.setdiag(0)
    super_adj.eliminate_zeros()
    super_adj.data[:] = 1
    super_adj_csr = super_adj.tocsr()

    # --- Compute Super-Graph Shortest Paths ---
    print("🔧 [METIS] Computing super-graph shortest paths...")
    sp = shortest_path(csgraph=super_adj_csr, directed=False, unweighted=True)
    sp = np.asarray(sp, dtype=np.float32)
    sp[np.isinf(sp)] = 30.0 # Use 30 as the "infinity" distance
    super_sp_tensor = torch.from_numpy(sp)
    
    print("✅ [METIS] Fixed partition and distance matrix created.")
    
    return node2comm_tensor, super_sp_tensor

