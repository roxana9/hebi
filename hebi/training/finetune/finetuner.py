#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from hebi.model.hebi import (
    GATEncoder,
    LocalTransformerEncoder,
    NodeToPrototypeTransformer,
    PrototypePoolerL2,
    FeatureTransformerL2
)


class HebiCellTypeClassifier(nn.Module):
    """
    Backbone = the pretrained hierarchy (gene embedder + L0/L1/L2).
    Head = a single linear classifier on the cell-level z0 of the seed
    nodes, matching the Methods.

    Consistent with pretraining:
      - Uses edge_attr_b (tuple of 2 tensors)
      - L0 called as model_lvl0(x_subgraph, edge_index_subgraph, edge_attr_b, None, bs=bs)
      - L1 forward uses node2comm_batch + super_sp_global
      - L2 uses pooler + hk distances
      - Outputs logits for seed nodes only (first bs rows)
      - Level-2 distances use the HARDENED assignment, as in
        pretraining and encoding
    """

    def __init__(
        self,
        n_classes: int,
        dim_model: int = 128,
        l0_encoder_type: str = "transformer",
        global_num_layers: int = 2,
        num_prototypes_lvl1: int = 1024,
        num_prototypes_lvl2: int = 20,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.dim_model = dim_model
        self.config = {
            "l0_type": l0_encoder_type.lower(),
            "layers": global_num_layers,
            "proto1": num_prototypes_lvl1,
            "proto2": num_prototypes_lvl2,
        }

        # Backbone placeholders
        self.gene_embedder = None
        self.gene_linear = None
        self.model_lvl0 = None
        self.model_lvl1 = None
        self.pooler_lvl2 = None
        self.model_lvl2 = None
        self.classifier_head = nn.Linear(dim_model, n_classes, bias=False)
        self.grn_layer = None
        self.gene_graphs = None
        # Classifier head (probe)
        # self.classifier_head = nn.Sequential(
        #     nn.Linear(dim_model, dim_model),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(dim_model, n_classes),
        # )

    def _build_backbone(self, vocab_size, device):
        self.gene_embedder = nn.Embedding(vocab_size, self.dim_model).to(device)
        self.gene_linear = nn.Linear(self.dim_model, self.dim_model).to(device)

        l0_config = {
            "in_channels": self.dim_model,
            "hidden_channels": 128,
            "out_channels": 128,
            "num_layers": 3,
            "heads": 4,
            "dropout": 0.1,
        }

        if self.config["l0_type"] == "gat":
            self.model_lvl0 = GATEncoder(**l0_config).to(device)
        else:
            self.model_lvl0 = LocalTransformerEncoder(**l0_config).to(device)

        self.model_lvl1 = NodeToPrototypeTransformer(
            in_channels=128,
            hidden_channels=128,
            out_channels=128,
            heads=4,
            num_layers=self.config["layers"],
            ff_dropout=0.1,
            attn_dropout=0.1,
            num_prototypes=self.config["proto1"],
        ).to(device)

        self.pooler_lvl2 = PrototypePoolerL2(
            in_channels=128,
            num_l1_prototypes=self.config["proto1"],
            num_l2_prototypes=self.config["proto2"],
        ).to(device)

        self.model_lvl2 = FeatureTransformerL2(
            in_channels=128,
            hidden_channels=128,
            out_channels=128,
            heads=4,
            num_layers=self.config["layers"],
            ff_dropout=0.1,
            attn_dropout=0.1,
        ).to(device)

    def load_pretrained_weights(self, checkpoint_path, device, freeze_backbone=True):
        """
        Your pretraining checkpoint saved as:
          state = {
             "gene_embedder": state_dict,
             "gene_linear": state_dict,
             "model_lvl0": ...
             ...
          }
        """
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Determine vocab size from checkpoint
        if "gene_embedder" not in ckpt:
            raise KeyError("Checkpoint missing key: 'gene_embedder'")

        vocab_size = ckpt["gene_embedder"]["weight"].shape[0]
        self._build_backbone(vocab_size, device)

        keys = ["gene_embedder", "gene_linear", "model_lvl0", "model_lvl1", "pooler_lvl2", "model_lvl2"]
        for k in keys:
            if k in ckpt:
                getattr(self, k).load_state_dict(ckpt[k])
            else:
                print(f"⚠️ Warning: {k} missing from checkpoint")

        if freeze_backbone:
            print("❄️ Freezing Backbone...")
            for k in keys:
                for p in getattr(self, k).parameters():
                    p.requires_grad = False

    def forward(
        self,
        x_subgraph_raw,        # [N_sub, G_slide]
        edge_index_subgraph,   # [2, E_sub]
        edge_attr_b,           # tuple(attr0, attr1) like pretraining
        node2comm_batch,       # [bs]
        super_sp_global,       # slide tensor
        gene_ids,              # [G_slide]
        device,
        bs: int,
        gene_graphs=None,        # optional override
        return_grn_attention=False,
    ):
        # 1) Gene projection (same as pretraining)
        gene_emb = self.gene_embedder(gene_ids)      # [G, 128]
        gene_emb = self.gene_linear(gene_emb)
        gene_emb = F.normalize(gene_emb, dim=1, p=2)

        grn_attn = None
        if self.grn_layer is not None and self.gene_graphs is not None:
            edge_index_grn = self.gene_graphs["global"].edge_index.to(device)
            
        
            gene_emb,grn_attn = self.grn_layer(
                gene_emb=gene_emb,
                edge_index_grn=edge_index_grn,
                return_attention=return_grn_attention
            )
            
        x_emb = x_subgraph_raw @ gene_emb           # [N_sub, 128]

        # 2) L0 (MUST use edge_attr_b, otherwise TransformerConv asserts)
        z0 = self.model_lvl0(
            x_emb,
            edge_index_subgraph,
            edge_attr_b,
            None,
            bs=bs
        )

        # 3) L1
        z1 = self.model_lvl1.forward(
            x_lower=z0,
            node2comm_batch=node2comm_batch,
            super_sp_global=super_sp_global,
        )

        # 4) L2
        C_L1 = self.model_lvl1.prototypes
        C_L2, S_matrix = self.pooler_lvl2(C_L1)

        # Harden the assignment before projecting distances, matching
        # pretraining and encoding. Using the soft S_matrix here would feed
        # the Level-2 transformer distances it was never trained on.
        Dk_raw_L1 = super_sp_global[node2comm_batch.long(), :]
        S_hard = torch.zeros_like(S_matrix)
        S_hard.scatter_(1, S_matrix.argmax(dim=1, keepdim=True), 1.0)
        Hk_L2_distances = torch.matmul(Dk_raw_L1.float(), S_hard.float())

        z2, _ = self.model_lvl2.forward(
            x_lower=z1,
            c_l2_prototypes=C_L2,
            hk_l2_distances=Hk_L2_distances,
        )

        # IMPORTANT: logits only for seed nodes (first bs)
        # z2_seed = z2[:bs]
        # logits = self.classifier_head(z2_seed)
        z0_seed = z0[:bs]
        logits = self.classifier_head(z0_seed)
        if return_grn_attention:
            return logits, grn_attn,gene_emb
        return logits
