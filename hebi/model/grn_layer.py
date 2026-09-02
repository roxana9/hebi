#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class GRNRefinementLayer(nn.Module):
    def __init__(self, dim_model=128, heads=4):
        super().__init__()

        assert dim_model % heads == 0
        out_channels_per_head = dim_model // heads

        self.gat = GATv2Conv(
            in_channels=dim_model,
            out_channels=out_channels_per_head,
            heads=heads,
            concat=True,
            add_self_loops=False,
        )

        self.norm = nn.LayerNorm(dim_model)
        self.activation = nn.GELU()

    def forward(self, gene_emb, edge_index_grn, return_attention=False, return_per_head=False):
        refined_emb, (edge_index_out, attn_weights) = self.gat(
            x=gene_emb,
            edge_index=edge_index_grn,
            return_attention_weights=True
        )

        # attn_weights: [E, H]
        mean_attn = attn_weights.mean(dim=-1)

        refined_emb = self.norm(gene_emb + self.activation(refined_emb))

        if return_attention:
            if return_per_head:
                return refined_emb, mean_attn, attn_weights, edge_index_out
            return refined_emb, mean_attn
        else:
            return refined_emb, None
# class GRNRefinementLayer(nn.Module):
#     def __init__(self, dim_model: int = 128, heads: int = 4, edge_attr_dim: int = 4):
#         super().__init__()

#         assert dim_model % heads == 0, "dim_model must be divisible by heads"
#         out_channels_per_head = dim_model // heads

#         # 🔹 Learnable logits (NOT weights directly)
#         # Start at zeros → uniform softmax
#         self.edge_type_logits = nn.Parameter(torch.zeros(edge_attr_dim))

#         self.gat = GATv2Conv(
#             in_channels=dim_model,
#             out_channels=out_channels_per_head,
#             heads=heads,
#             concat=True,
#             #edge_dim=edge_attr_dim,
#             add_self_loops=False,
#         )

#         self.norm = nn.LayerNorm(dim_model)
#         self.activation = nn.GELU()

#     def forward(self, gene_emb, edge_index_grn, edge_attr_grn, return_attention=False):

#         # 🔥 COMPETITIVE EDGE TYPE WEIGHTS
#         # Now weights sum to 1
#         edge_type_weights = F.softmax(self.edge_type_logits, dim=0)
#         # shape: [4]

#         # Scale edge attributes
#         # edge_attr_grn shape: [E, 4]
#         scaled_edge_attr = edge_attr_grn * edge_type_weights

#         refined_emb, (edge_index_out, attn_weights) = self.gat(
#             x=gene_emb,
#             edge_index=edge_index_grn,
#            # edge_attr=scaled_edge_attr,
#             return_attention_weights=True
#         )

#         # Average attention across heads
#         mean_attn = attn_weights.mean(dim=-1)

#         # Residual + Norm
#         refined_emb = self.norm(
#             gene_emb + self.activation(refined_emb)
#         )

#         if return_attention:
#             return refined_emb, mean_attn
#         else:
#             return refined_emb, None

#     def get_edge_type_importance(self):
#         """
#         Returns normalized importance per edge type.
#         Now interpretable.
#         """
#         return F.softmax(self.edge_type_logits, dim=0).detach().cpu()