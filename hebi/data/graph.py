# build_graph.py
import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.spatial import distance_matrix

def build_spatial_graph(spatial_coords, k=10, compute_distance_matrix=False):
    """
    Build a memory-efficient spatial kNN graph from coordinates only.

    Args:
        spatial_coords (array-like): (n_cells, 2) or (n_cells, 3)
        k (int): number of nearest neighbours.
        compute_distance_matrix (bool): compute dense pairwise distances (⚠️ heavy).

    Returns:
        edge_index (LongTensor): shape [2, num_edges]
        edge_attr (FloatTensor): pairwise distances for each edge
        dist_mat (FloatTensor or None): optional dense distance matrix
    """
    print(f"🔧 Building {k}-NN spatial graph (coords={spatial_coords.shape}) ...")

    # --- Build sparse kNN adjacency (directly memory-safe) ---
    A_row = kneighbors_graph(spatial_coords, k, mode='distance', include_self=False)
    A_sym = 0.5 * (A_row + A_row.T)  # symmetrise
    edge_index, edge_weight = from_scipy_sparse_matrix(A_sym)

    edge_attr = edge_weight.unsqueeze(1)  # [num_edges, 1]
    print(f"✅ Graph built: {spatial_coords.shape[0]} nodes, {edge_index.shape[1]} edges")

    # --- Optional dense distance matrix (off by default) ---
    dist_mat = None
    if compute_distance_matrix:
        print("⚠️ Computing full distance matrix (can be huge) ...")
        dist_mat = torch.tensor(distance_matrix(spatial_coords, spatial_coords), dtype=torch.float32)

    return edge_index, edge_attr, dist_mat
