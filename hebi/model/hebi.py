#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:24:44 2025

@author: roxana
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:20:25 2025

@author: roxana
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GAT,TransformerConv
from torch_scatter import scatter_mean 

# --- 1. Level 0 GAT Encoder (No Changes) ---
class GATEncoder(nn.Module):
    """
    Level 0 Encoder: Processes local subgraphs using GAT.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout=0.0, **kwargs):
        super(GATEncoder, self).__init__()
        self.gat = GAT(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers,
                       out_channels=out_channels, dropout=dropout, act="elu", heads=heads, v2=True, edge_dim=1)

    def forward(self, x, edge_index, edge_attr=None, pos_enc=None, bs: int = None):
        """ Forward pass for GAT. """
        if edge_attr is not None:
             # GAT expects a single tensor for edge_attr. Let's use edge_dist.
             edge_dist = edge_attr[0] 
             if edge_dist.dim() == 1: edge_dist = edge_dist.unsqueeze(-1) # Ensure shape [num_edges, 1]
             out = self.gat(x=x, edge_index=edge_index, edge_attr=edge_dist)
        else:
             out = self.gat(x=x, edge_index=edge_index)
        
        if bs is not None:
             out = out[:bs] # Slice to return only target node embeddings
        return out
class LocalTransformerEncoder(nn.Module):
    """
    Level 0 Encoder: Processes local subgraphs using TransformerConv.
    This can learn more complex attention than GAT.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout=0.0, **kwargs):
        super(LocalTransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(
            TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=1)
        )
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=1)
            )
        # Output layer
        self.layers.append(
            TransformerConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout, edge_dim=1)
        )
        
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            # We need to norm the multi-head output
            if _ < num_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_channels * heads))
            else:
                self.norms.append(nn.LayerNorm(out_channels)) # Final layer output

    def forward(self, x, edge_index, edge_attr=None, pos_enc=None, bs: int = None):
        """ Forward pass for TransformerConv. """
        
        edge_dist_attr = None
        if edge_attr is not None:
            edge_dist_attr = edge_attr[0] # Select distance
            if edge_dist_attr.dim() == 1: 
                edge_dist_attr = edge_dist_attr.unsqueeze(-1)
            edge_dist_attr = edge_dist_attr.float()
        out = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Pass edge_attr (distance) to the layer.
            out = layer(out, edge_index, edge_attr=edge_dist_attr)
            out = norm(out)
            if i < len(self.layers) - 1: # No ReLU on final layer
                out = F.elu(out)

        if bs is not None:
             out = out[:bs] # Slice to return only target node embeddings
        return out
# --- 2. Level 1 Transformer (Node-to-Prototype) (No Changes) ---
class NodeToPrototypeTransformer(nn.Module):
    """
    Level 1 Module: Refines L0 embeddings (z0) by attending to
    learnable prototypes (C), biased by a fixed distance matrix (Hk).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads, 
                 num_prototypes, num_layers, ff_dropout, attn_dropout, **kwargs):
        super(NodeToPrototypeTransformer, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_prototypes = num_prototypes
        self.num_layers = num_layers
        self.dropout = attn_dropout
        
        norm_func = nn.LayerNorm
        
        # Project input features (from Z0) to hidden dim
        self.fc_in = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            norm_func(hidden_channels),
            nn.ReLU()
        )
        
        # --- Learnable Prototypes (K, V) ---
        prototype_dim = heads * (hidden_channels // heads) # Ensure divisible
        if heads * (hidden_channels // heads) != hidden_channels:
            raise ValueError("hidden_channels must be divisible by heads")
            
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

        # --- HDSE Bias Layer ---
        self.spatial_bias_encoder = nn.Embedding(31, heads)
        torch.nn.init.zeros_(self.spatial_bias_encoder.weight)

        # --- Transformer Layers ---
        self.lin_q_layers = nn.ModuleList()
        self.ffs = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        
        for _ in range(num_layers):
            self.lin_q_layers.append(Linear(hidden_channels, prototype_dim))
            
            self.ffs.append(
                 nn.Sequential(
                     nn.Linear(prototype_dim, hidden_channels * 2), # FF intermediate dim
                     nn.ReLU(),
                     nn.Dropout(ff_dropout),
                     nn.Linear(hidden_channels * 2, prototype_dim), # Back to H*C
                 )
            )
            self.norms_1.append(norm_func(prototype_dim))
            self.norms_2.append(norm_func(prototype_dim))
            
        # Final output layer (maps from H*C back to out_channels)
        self.fc_out = nn.Linear(prototype_dim, out_channels)

    def forward(self, x_lower, node2comm_batch, super_sp_global):
        # x_lower: [bs, D_in] (z0_batch)
        # node2comm_batch: [bs] (fixed map: node_idx -> partition_idx)
        # super_sp_global: [K, K] (fixed distances between partitions)
        
        H, C = self.heads, self.hidden_channels // self.heads
        N_batch = x_lower.size(0)
        N_prototypes = self.num_prototypes

        # 1. Input Projection
        h_lower = self.fc_in(x_lower) # [bs, D_hidden]
        
        # 2. Normalize Prototypes (K, V)
        prototypes_norm = F.normalize(self.prototypes, dim=-1)
        
        k = prototypes_norm.view(N_prototypes, H, C).permute(1, 2, 0) # Shape: [H, C, N_proto]
        v = prototypes_norm.view(N_prototypes, H, C).permute(1, 0, 2) # Shape: [H, N_proto, C]
        
        # 3. Construct HDSE Bias (Hk) for this batch
        Dk_raw = super_sp_global[node2comm_batch.long(), :] # Shape: [bs, N_proto]
        Dk_clipped = torch.clamp(Dk_raw, 0, 30).long()
        
        Hk = self.spatial_bias_encoder(Dk_clipped) # Shape [bs, N_proto, H]
        Hk = Hk.permute(2, 0, 1) # Shape [H, bs, N_proto]

        # 4. Sequential Transformer Blocks
        for i in range(self.num_layers):
            h_residual = h_lower # Save for residual connection [bs, D_hidden]
            
            # 4a. Queries (Q) from nodes
            q = self.lin_q_layers[i](h_lower).view(N_batch, H, C).permute(1, 0, 2) # [H, bs, C]
            
            # 4b. Attention Scores + Bias
            attn_scores = (torch.matmul(q, k) / math.sqrt(C)) + Hk # [H, bs, N_proto]
            attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            
            # 4c. Aggregate Prototypes
            out = torch.matmul(attn_probs, v) # [H, bs, C]
            out = out.permute(1, 0, 2).contiguous().view(N_batch, H * C) # [bs, H*C]
            
            # 4d. First Residual + Norm
            h_lower = self.norms_1[i](h_residual + out)
            
            # 4e. Feed-Forward + Residual
            h_ff = self.ffs[i](h_lower)
            h_lower = self.norms_2[i](h_lower + h_ff)

        # 5. Final output projection
        h_final = self.fc_out(h_lower) # [bs, D_out]
        return h_final

# --- 3. NEW: Level 2 Prototype Pooler (Assignment Generator) ---
class PrototypePoolerL2(nn.Module):
    """
    Learns to cluster L1 prototypes (1024) into L2 prototypes (20).
    Uses a temperature-sharpened softmax + prototype separation.
    """
    def __init__(self, in_channels: int, num_l1_prototypes: int, num_l2_prototypes: int):
        super(PrototypePoolerL2, self).__init__()
        
        self.num_l1_prototypes = num_l1_prototypes
        self.num_l2_prototypes = num_l2_prototypes
        
        self.pool_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_l2_prototypes)
        )
        
        # Learnable temperature — starts sharp, can soften
        # log(temp) parameterization keeps it positive
       # self.log_temperature = nn.Parameter(torch.tensor(-2.0))  # exp(-2) ≈ 0.135
        self.register_buffer('temperature', torch.tensor(0.1))  # fixed, not learned
        # Learnable L2 prototype embeddings — NOT derived from S_matrix
        # These are independent and can be pushed apart by loss
        self.l2_prototypes = nn.Parameter(
            torch.randn(num_l2_prototypes, in_channels) * 0.1
        )

    def forward(self, c_l1_prototypes: Tensor, hard: bool = False):
        # c_l1_prototypes: [1024, 128]
        
        S_logits = self.pool_mlp(c_l1_prototypes)  # [1024, 20]
        
        if self.training:
            # Gumbel-Softmax: encourages discrete assignments during training
            #S_matrix = F.gumbel_softmax(S_logits, tau=0.5, hard=False, dim=-1)
            S_soft = F.softmax(S_logits / 0.5, dim=-1)
            S_hard = torch.zeros_like(S_soft)
            S_hard.scatter_(1, S_soft.argmax(dim=1, keepdim=True), 1.0)
            S_matrix = S_hard - S_soft.detach() + S_soft  # straight-through
        else:
            # Sharp softmax at inference
            S_matrix = F.softmax(S_logits / 0.1, dim=-1)
        
        context = torch.matmul(S_matrix.transpose(0, 1), c_l1_prototypes)  # [20, 128]
        C_L2_prototypes = self.l2_prototypes + 0.1 * context
        
        return C_L2_prototypes, S_matrix
# class PrototypePoolerL2(nn.Module):
#     """
#     Learns to cluster L1 prototypes (1024) into L2 prototypes (30).
#     Generates the S matrix (P) and the L2 prototypes (C_L2).
#     """
#     def __init__(self, in_channels: int, num_l1_prototypes: int, num_l2_prototypes: int):
#         super(PrototypePoolerL2, self).__init__()
        
#         self.num_l1_prototypes = num_l1_prototypes
#         self.num_l2_prototypes = num_l2_prototypes
        
#         # This is your "P matrix" (S) generator MLP.
#         # It looks at L1 prototypes and decides how to group them.
#         self.pool_mlp = nn.Sequential(
#             nn.Linear(in_channels, in_channels),
#             nn.ReLU(),
#             nn.Linear(in_channels, num_l2_prototypes) # [D] -> [30]
#         )

#     def forward(self, c_l1_prototypes: Tensor):
#         # c_l1_prototypes: [1024, D] (from model_lvl1.prototypes)
        
#         # 1. Generate the "P Matrix" (S)
#         # S_logits shape: [1024, 30]
#         S_logits = self.pool_mlp(c_l1_prototypes)
        
#         # S_matrix (soft assignment): [1024, 30]
#         S_matrix = F.softmax(S_logits, dim=-1)
        
#         # 2. Generate the L2 Prototypes (C_L2)
#         # C_L2 = S.T @ C_L1
#         # C_L2 shape: [30, D]
#         C_L2_prototypes = torch.matmul(S_matrix.transpose(0, 1), c_l1_prototypes)
        
#         # Return both: C_L2 for the L2 Transformer, S_matrix for the loss
#         return C_L2_prototypes, S_matrix

# --- 4. NEW: Level 2 Feature Transformer (No Spatial Bias) ---
# class FeatureTransformerL2(nn.Module):
#     """
#     Level 2 Module: Refines L1 embeddings (z1) by attending to
#     the L2 prototypes (C_L2) based on *feature similarity* only.
    
#     This is a simplified version of the L1 Transformer:
#     - No self.prototypes (receives them in forward)
#     - No spatial_bias_encoder (Hk)
#     """
#     def __init__(self, in_channels, hidden_channels, out_channels, heads, 
#                  num_layers, ff_dropout, attn_dropout, **kwargs):
#         super(FeatureTransformerL2, self).__init__()
        
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.heads = heads
#         self.num_layers = num_layers
#         self.dropout = attn_dropout
        
#         norm_func = nn.LayerNorm
        
#         # Project input features (from Z1) to hidden dim
#         self.fc_in = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             norm_func(hidden_channels),
#             nn.ReLU()
#         )
        
#         prototype_dim = heads * (hidden_channels // heads) # Ensure divisible
        
#         # --- Transformer Layers ---
#         self.lin_q_layers = nn.ModuleList()
#         self.ffs = nn.ModuleList()
#         self.norms_1 = nn.ModuleList()
#         self.norms_2 = nn.ModuleList()
        
#         for _ in range(num_layers):
#             self.lin_q_layers.append(Linear(hidden_channels, prototype_dim))
            
#             self.ffs.append(
#                  nn.Sequential(
#                      nn.Linear(prototype_dim, hidden_channels * 2), # FF intermediate dim
#                      nn.ReLU(),
#                      nn.Dropout(ff_dropout),
#                      nn.Linear(hidden_channels * 2, prototype_dim), # Back to H*C
#                  )
#             )
#             self.norms_1.append(norm_func(prototype_dim))
#             self.norms_2.append(norm_func(prototype_dim))
            
#         # Final output layer (maps from H*C back to out_channels)
#         self.fc_out = nn.Linear(prototype_dim, out_channels)

#     def forward(self, x_lower: Tensor, c_l2_prototypes: Tensor):
#         # x_lower: [bs, D_in] (z1_batch)
#         # c_l2_prototypes: [30, D] (from Pooler)
        
#         H, C = self.heads, self.hidden_channels // self.heads
#         N_batch = x_lower.size(0)
#         N_prototypes_L2 = c_l2_prototypes.size(0) # This will be 30

#         # 1. Input Projection
#         h_lower = self.fc_in(x_lower) # [bs, D_hidden]
        
#         # 2. Normalize L2 Prototypes (K, V)
#         prototypes_norm = F.normalize(c_l2_prototypes, dim=-1)
        
#         k = prototypes_norm.view(N_prototypes_L2, H, C).permute(1, 2, 0) # Shape: [H, C, 30]
#         v = prototypes_norm.view(N_prototypes_L2, H, C).permute(1, 0, 2) # Shape: [H, 30, C]

#         # 3. Sequential Transformer Blocks
#         for i in range(self.num_layers):
#             h_residual = h_lower # Save for residual connection [bs, D_hidden]
            
#             # 3a. Queries (Q) from nodes
#             q = self.lin_q_layers[i](h_lower).view(N_batch, H, C).permute(1, 0, 2) # [H, bs, C]
            
#             # 3b. Attention Scores (NO SPATIAL BIAS)
#             attn_scores = (torch.matmul(q, k) / math.sqrt(C)) # [H, bs, 30]
#             attn_probs = attn_scores.softmax(dim=-1)
#             attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            
#             # 3c. Aggregate Prototypes
#             out = torch.matmul(attn_probs, v) # [H, bs, C]
#             out = out.permute(1, 0, 2).contiguous().view(N_batch, H * C) # [bs, H*C]
            
#             # 3d. First Residual + Norm
#             h_lower = self.norms_1[i](h_residual + out)
            
#             # 3e. Feed-Forward + Residual
#             h_ff = self.ffs[i](h_lower)
#             h_lower = self.norms_2[i](h_lower + h_ff)

#         # 4. Final output projection
#         h_final = self.fc_out(h_lower) # [bs, D_out]
#         return h_final
class FeatureTransformerL2(nn.Module):
    """
    Level 2 Module: Refines L1 embeddings (z1) by attending to
    the L2 prototypes (C_L2) based on *feature similarity* AND
    a new *LEARNABLE spatial bias*.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads, 
                 num_layers, ff_dropout, attn_dropout, **kwargs):
        super(FeatureTransformerL2, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = attn_dropout
        
        norm_func = nn.LayerNorm
        
        self.fc_in = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            norm_func(hidden_channels),
            nn.ReLU()
        )
        
        prototype_dim = heads * (hidden_channels // heads)
        
        # --- VVV NEW: L2 Spatial Bias Encoder --- VVV
        # We add a bias encoder, just like in the L1 model.
        # It will operate on the new L2 distances.
        self.spatial_bias_encoder = nn.Embedding(31, heads)
        torch.nn.init.zeros_(self.spatial_bias_encoder.weight)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        # --- ^^^ NEW: L2 Spatial Bias Encoder --- ^^^
        
        self.lin_q_layers = nn.ModuleList()
        self.ffs = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        
        for _ in range(num_layers):
            self.lin_q_layers.append(Linear(hidden_channels, prototype_dim))
            
            self.ffs.append(
                 nn.Sequential(
                     nn.Linear(prototype_dim, hidden_channels * 2),
                     nn.ReLU(),
                     nn.Dropout(ff_dropout),
                     nn.Linear(hidden_channels * 2, prototype_dim),
                 )
            )
            self.norms_1.append(norm_func(prototype_dim))
            self.norms_2.append(norm_func(prototype_dim))
            
        self.fc_out = nn.Linear(prototype_dim, out_channels)

    def forward(self, x_lower: Tensor, c_l2_prototypes: Tensor, hk_l2_distances: Tensor):
        # x_lower: [bs, D_in] (z1_batch)
        # c_l2_prototypes: [30, D] (from Pooler)
        # hk_l2_distances: [bs, 30] (NEW: calculated in train loop)
        
        H, C = self.heads, self.hidden_channels // self.heads
        N_batch = x_lower.size(0)
        N_prototypes_L2 = c_l2_prototypes.size(0) # 30

        h_lower = self.fc_in(x_lower)
        
        prototypes_norm = F.normalize(c_l2_prototypes, dim=-1)
        
        k = prototypes_norm.view(N_prototypes_L2, H, C).permute(1, 2, 0)
        v = prototypes_norm.view(N_prototypes_L2, H, C).permute(1, 0, 2)

        # --- VVV NEW: Construct L2 Bias (Hk_L2) --- VVV
        # We encode the learned distances from the training loop
        Dk_clipped = torch.clamp(hk_l2_distances, 0, 30).long()
        Hk_L2 = self.spatial_bias_encoder(Dk_clipped) # Shape [bs, 30, H]
        Hk_L2 = Hk_L2.permute(2, 0, 1) # Shape [H, bs, 30]
        # --- ^^^ NEW: Construct L2 Bias (Hk_L2) --- ^^^
        final_attn_probs = None 

        for i in range(self.num_layers):
            h_residual = h_lower
            q = self.lin_q_layers[i](h_lower).view(N_batch, H, C).permute(1, 0, 2)
            
            feature_scores = (torch.matmul(q, k) / math.sqrt(C))
            attn_scores = (feature_scores + Hk_L2) * self.temperature.clamp(min=0.1, max=10.0)
            # --- VVV UPDATED: Add L2 Spatial Bias --- VVV
          #  attn_scores = (torch.matmul(q, k) / math.sqrt(C)) + Hk_L2 # [H, bs, 30]
            # --- ^^^ UPDATED: Add L2 Spatial Bias --- ^^^
            
            attn_probs = attn_scores.softmax(dim=-1)
            if i == self.num_layers - 1:
                # Average over heads to get [bs, 30]
                final_attn_probs = attn_probs.mean(dim=0) 
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            
            out = torch.matmul(attn_probs, v)
            out = out.permute(1, 0, 2).contiguous().view(N_batch, H * C)
            
            h_lower = self.norms_1[i](h_residual + out)
            
            h_ff = self.ffs[i](h_lower)
            h_lower = self.norms_2[i](h_lower + h_ff)

        h_final = self.fc_out(h_lower)
        return h_final,final_attn_probs
# =====================================================================
# Community-anchored hierarchy (HDSE-style Levels 1 and 2)
#
# Level 1: keys/values are the METIS communities of the current section,
#   represented by the mean of their cells; logits add log(n_c) and a
#   non-positive distance slope w1 * D[a_i, c] (HDSE, Luo et al. 2024).
# Level 2: communities are assigned to K2 shared niche prototypes by their
#   content; distance cell -> niche k = min distance to a community assigned
#   to k; niches with no community in the batch are masked.
# No community index is ever paired with a prototype index, so outputs do not
# depend on the METIS numbering.
# =====================================================================
FAR_DISTANCE = 30


def nonpositive_slope(theta: Tensor, rate: float) -> Tensor:
    """w = -softplus(rate * theta). `rate` makes Adam move w `rate` times
    faster than the other weights without touching the optimiser."""
    return -F.softplus(theta * rate)


def init_slope_theta(n: int, w_init: float, rate: float) -> nn.Parameter:
    z0 = math.log(math.expm1(-w_init))          # softplus(z0) = -w_init
    return nn.Parameter(torch.full((n,), z0 / rate))


class CommunityTransformerL1(nn.Module):
    """Level 1: cells attend to the communities of their section."""

    def __init__(self, in_channels, hidden_channels, out_channels, heads,
                 num_layers, ff_dropout, attn_dropout,
                 slope_init: float = -0.1, slope_rate: float = 30.0, **kwargs):
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = attn_dropout
        self.slope_rate = slope_rate

        self.fc_in = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                   nn.LayerNorm(hidden_channels), nn.ReLU())
        self.lin_k = nn.Linear(hidden_channels, hidden_channels)
        self.lin_v = nn.Linear(hidden_channels, hidden_channels)
        self.dist_theta = init_slope_theta(heads, slope_init, slope_rate)

        self.lin_q_layers = nn.ModuleList()
        self.ffs = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for _ in range(num_layers):
            self.lin_q_layers.append(Linear(hidden_channels, hidden_channels))
            self.ffs.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2), nn.ReLU(),
                nn.Dropout(ff_dropout), nn.Linear(hidden_channels * 2, hidden_channels)))
            self.norms_1.append(nn.LayerNorm(hidden_channels))
            self.norms_2.append(nn.LayerNorm(hidden_channels))
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    @property
    def slope(self) -> Tensor:
        return nonpositive_slope(self.dist_theta, self.slope_rate)

    def community_reps(self, h: Tensor, node2comm: Tensor, n_comm: int):
        """Mean of h over the cells of each community, and the cell counts."""
        idx = node2comm.long()
        count = torch.bincount(idx, minlength=n_comm).to(h.dtype)
        sums = torch.zeros(n_comm, h.size(1), device=h.device, dtype=h.dtype)
        sums.index_add_(0, idx, h)
        return sums / count.clamp(min=1).unsqueeze(1), count

    def forward(self, x_lower: Tensor, node2comm_batch: Tensor,
                super_sp_global: Tensor, global_reps=None):
        """global_reps: optional (reps, counts) computed over the whole
        section (used at encoding so outputs do not depend on batching).

        Returns z1 [bs, out] and a dict with the community representations
        (reps, present mask) and the cell-to-community distances d [bs, n_comm]
        that Level 2 needs."""
        H, C = self.heads, self.hidden_channels // self.heads
        N = x_lower.size(0)
        n_comm = super_sp_global.size(0)

        h = self.fc_in(x_lower)
        if global_reps is not None:
            reps, count = global_reps
        else:
            reps, count = self.community_reps(h, node2comm_batch, n_comm)

        k = self.lin_k(reps).view(n_comm, H, C).permute(1, 2, 0)        # [H, C, n_comm]
        v = self.lin_v(reps).view(n_comm, H, C).permute(1, 0, 2)        # [H, n_comm, C]
        log_n = torch.log(count).view(1, 1, n_comm)                     # -inf = absent
        d = super_sp_global[node2comm_batch.long(), :].float()
        d = torch.nan_to_num(d, nan=FAR_DISTANCE, posinf=FAR_DISTANCE).clamp(0, FAR_DISTANCE)
        dist_bias = self.slope.view(H, 1, 1) * d.unsqueeze(0)           # [H, bs, n_comm]

        for i in range(self.num_layers):
            res = h
            q = self.lin_q_layers[i](h).view(N, H, C).permute(1, 0, 2)
            attn = (torch.matmul(q, k) / math.sqrt(C) + log_n + dist_bias).softmax(dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v).permute(1, 0, 2).contiguous().view(N, H * C)
            h = self.norms_1[i](res + out)
            h = self.norms_2[i](h + self.ffs[i](h))

        aux = {"reps": reps, "present": count > 0, "d": d}
        return self.fc_out(h), aux


class CommunityPoolerL2(nn.Module):
    """Assigns the communities present in the batch to K2 niche prototypes."""

    def __init__(self, in_channels: int, num_l2_prototypes: int,
                 context_scale: float = 0.5, grad_scale: float = 0.1, **kwargs):
        super().__init__()
        self.num_l2_prototypes = num_l2_prototypes
        self.context_scale = context_scale
        self.grad_scale = grad_scale            # gradient from the pooler into L0/L1
        self.pool_mlp = nn.Sequential(nn.Linear(in_channels, in_channels), nn.ReLU(),
                                      nn.Linear(in_channels, num_l2_prototypes))
        self.l2_prototypes = nn.Parameter(torch.randn(num_l2_prototypes, in_channels) * 0.1)

    def forward(self, community_reps: Tensor):
        """community_reps: [P', D], the communities present in the batch.
        Returns niche prototypes C_L2 [K2, D] and assignment S [P', K2]."""
        c = community_reps
        if self.grad_scale != 1.0:
            c = self.grad_scale * c + (1.0 - self.grad_scale) * c.detach()

        logits = self.pool_mlp(c)
        if self.training:
            s_soft = F.softmax(logits / 0.5, dim=-1)
            s_hard = torch.zeros_like(s_soft).scatter_(1, s_soft.argmax(1, keepdim=True), 1.0)
            S = s_hard - s_soft.detach() + s_soft                     # straight-through
        else:
            S = F.softmax(logits / 0.1, dim=-1)

        # mean of the (normalised) communities assigned to each niche;
        # count clamped at 1 so an empty niche gets zero context, not 0/eps
        context = (S.t() @ F.normalize(c, dim=-1)) / S.sum(0).clamp(min=1.0).unsqueeze(1)
        return self.l2_prototypes + self.context_scale * context, S


def niche_distances(d_present: Tensor, S: Tensor, far: float = FAR_DISTANCE):
    """d_present: [bs, P'] cell-to-community distances for the communities in
    S. Returns [bs, K2] distance to the nearest community assigned to each
    niche, and a [K2] bool mask of niches with no community."""
    niche = S.argmax(dim=1).detach()                                    # [P']
    K2 = S.size(1)
    out = torch.full((d_present.size(0), K2), float(far), device=d_present.device)
    out.scatter_reduce_(1, niche.unsqueeze(0).expand(d_present.size(0), -1).contiguous(),
                        d_present, reduce="amin", include_self=True)
    absent = torch.bincount(niche, minlength=K2) == 0
    return out, absent


class NicheTransformerL2(nn.Module):
    """Level 2: cells attend to the niche prototypes, with a non-positive
    distance slope on the cell-to-niche distance."""

    def __init__(self, in_channels, hidden_channels, out_channels, heads,
                 num_layers, ff_dropout, attn_dropout,
                 slope_init: float = -0.1, slope_rate: float = 10.0, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = attn_dropout
        self.slope_rate = slope_rate

        self.fc_in = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                   nn.LayerNorm(hidden_channels), nn.ReLU())
        self.dist_theta = init_slope_theta(heads, slope_init, slope_rate)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.lin_q_layers = nn.ModuleList()
        self.ffs = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for _ in range(num_layers):
            self.lin_q_layers.append(Linear(hidden_channels, hidden_channels))
            self.ffs.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2), nn.ReLU(),
                nn.Dropout(ff_dropout), nn.Linear(hidden_channels * 2, hidden_channels)))
            self.norms_1.append(nn.LayerNorm(hidden_channels))
            self.norms_2.append(nn.LayerNorm(hidden_channels))
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    @property
    def slope(self) -> Tensor:
        
        return nonpositive_slope(self.dist_theta, self.slope_rate).clamp(min=-1.0)

    def forward(self, x_lower: Tensor, c_l2_prototypes: Tensor,
                niche_dist: Tensor, niche_absent: Tensor):
        H, C = self.heads, self.hidden_channels // self.heads
        N = x_lower.size(0)
        K2 = c_l2_prototypes.size(0)

        h = self.fc_in(x_lower)
        p = F.normalize(c_l2_prototypes, dim=-1)
        k = p.view(K2, H, C).permute(1, 2, 0)
        v = p.view(K2, H, C).permute(1, 0, 2)
        dist_bias = self.slope.view(H, 1, 1) * niche_dist.unsqueeze(0)  # [H, bs, K2]
        absent = niche_absent.view(1, 1, K2)

        final = None
        for i in range(self.num_layers):
            res = h
            q = self.lin_q_layers[i](h).view(N, H, C).permute(1, 0, 2)
            scores = (torch.matmul(q, k) / math.sqrt(C) + dist_bias) \
                * self.temperature.clamp(min=0.1, max=10.0)
            # mask after the temperature product: inside it, d(loss)/dT would
            # contain 0 * (-inf) = NaN
            scores = scores.masked_fill(absent, float("-inf"))
            attn = scores.softmax(dim=-1)
            if i == self.num_layers - 1:
                final = attn.mean(dim=0)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v).permute(1, 0, 2).contiguous().view(N, H * C)
            h = self.norms_1[i](res + out)
            h = self.norms_2[i](h + self.ffs[i](h))
        return self.fc_out(h), final


def hierarchy_forward(model_lvl1, pooler_lvl2, model_lvl2, z0_batch,
                      node2comm_batch, super_sp, global_reps=None):
    """Levels 1 and 2 for one batch, shared by training and encoding.
    Returns z1, z2, niche attention [bs, K2], and S [P', K2]."""
    z1, aux = model_lvl1(z0_batch, node2comm_batch, super_sp, global_reps=global_reps)
    present = aux["present"]
    C_L2, S = pooler_lvl2(aux["reps"][present])
    dist, absent = niche_distances(aux["d"][:, present], S)
    z2, attn = model_lvl2(z1, C_L2, dist, absent)
    return z1, z2, attn, S


# --- 5. Decoders (No Changes) ---
class GlobalMLPDecoder(nn.Module):
    """ Simple MLP decoder to reconstruct X from Z2 """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(GlobalMLPDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class LocalMLPDecoder(nn.Module):
    """ Decoder for local GAT encoder (if used) """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(LocalMLPDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)