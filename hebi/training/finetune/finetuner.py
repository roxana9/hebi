#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised prediction on top of a pretrained HEBI backbone
(community-based hierarchy).

Two levels:

  level="cell"   linear head on the cell-level embedding z0 of the seed
                 nodes. For labels determined by cell identity
                 (e.g. cell types, MERFISH niche labels).

  level="niche"  each cell's Level-2 attention over the K2 niche prototypes
                 ([bs, K2]) is mapped to the K labels by a learnable,
                 row-softmaxed matrix M [K2, K]:
                     p(label | cell) = attn(cell) @ softmax(W, dim=1)
                 For spatially defined niches (e.g. liver zonation,
                 tumour / interface). M is the prototype -> label mapping
                 (`prototype_label_map`).

The forward returns [bs, K]: logits for level="cell" (CrossEntropyLoss),
log-probabilities for level="niche" (NLLLoss), computed in log space.

Niche level, train/eval consistency:
  - community representations: whole-section `global_reps` in training and
    evaluation alike (built by train_classifier.build_global_cache)
  - pooler: soft inference assignment in every phase (kept in eval mode)

  linear probe : backbone frozen and kept in eval mode, only the head trains
  fine-tune    : backbone unfrozen at a lower learning rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hebi.data.sampler import LocalSampler
from hebi.model.hebi import (
    GATEncoder,
    LocalTransformerEncoder,
    CommunityTransformerL1,
    CommunityPoolerL2,
    NicheTransformerL2,
    hierarchy_forward,
)

POOLER_GRAD_SCALE = 0.1          # same as pretrain.py
CELL_KEYS = ["gene_embedder", "gene_linear", "model_lvl0"]
UPPER_KEYS = ["model_lvl1", "pooler_lvl2", "model_lvl2"]
BACKBONE_KEYS = CELL_KEYS + UPPER_KEYS
LEVELS = ("cell", "niche")


class HebiClassifier(nn.Module):

    def __init__(self, n_classes: int, level: str = "cell",
                 num_prototypes_lvl2: int = 20, dim_model: int = 128,
                 l0_encoder_type: str = "transformer",
                 global_num_layers: int = 2):
        super().__init__()
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.level = level
        self.n_classes = n_classes
        self.k2 = num_prototypes_lvl2
        self.dim_model = dim_model
        self.l0_type = l0_encoder_type.lower()
        self.n_layers = global_num_layers
        self.backbone_frozen = False

        for k in BACKBONE_KEYS:
            setattr(self, k, None)

        if level == "cell":
            self.classifier_head = nn.Linear(dim_model, n_classes, bias=False)
        else:
            # prototype -> label logits; zeros = uniform mapping at start
            self.proto_to_label = nn.Parameter(torch.zeros(self.k2, n_classes))

    # ------------------------------------------------------------------
    @property
    def used_backbone_keys(self):
        """Modules that take part in the forward for this level."""
        return CELL_KEYS if self.level == "cell" else BACKBONE_KEYS

    def head_parameters(self):
        if self.level == "cell":
            return list(self.classifier_head.parameters())
        return [self.proto_to_label]

    # ------------------------------------------------------------------
    def _build_backbone(self, vocab_size, device):
        d = self.dim_model
        self.gene_embedder = nn.Embedding(vocab_size, d).to(device)
        self.gene_linear = nn.Linear(d, d).to(device)

        l0 = dict(in_channels=d, hidden_channels=128, out_channels=128,
                  num_layers=3, heads=4, dropout=0.1)
        self.model_lvl0 = (GATEncoder(**l0) if self.l0_type == "gat"
                           else LocalTransformerEncoder(**l0)).to(device)

        tf = dict(in_channels=128, hidden_channels=128, out_channels=128,
                  num_layers=self.n_layers, heads=4,
                  ff_dropout=0.1, attn_dropout=0.1)
        self.model_lvl1 = CommunityTransformerL1(**tf).to(device)
        self.pooler_lvl2 = CommunityPoolerL2(
            in_channels=128, num_l2_prototypes=self.k2,
            grad_scale=POOLER_GRAD_SCALE).to(device)
        self.model_lvl2 = NicheTransformerL2(**tf).to(device)

    def load_pretrained_weights(self, checkpoint_path, device,
                                freeze_backbone=True):
        """checkpoint_path: community-based HEBI checkpoint
        (e.g. hebi_hybrid_oldL0.pt); all weights are loaded from it."""
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "prototypes" in ckpt["model_lvl1"]:
            raise ValueError("This checkpoint has the old Level-1 prototype "
                             "bank; a community-based checkpoint is needed.")
        k2 = ckpt["pooler_lvl2"]["l2_prototypes"].shape[0]
        if k2 != self.k2:
            raise ValueError(f"checkpoint has {k2} niche prototypes, model "
                             f"was built with {self.k2}")

        self._build_backbone(ckpt["gene_embedder"]["weight"].shape[0], device)
        for k in BACKBONE_KEYS:
            getattr(self, k).load_state_dict(ckpt[k])

        self.set_backbone_frozen(freeze_backbone)

    def set_backbone_frozen(self, frozen: bool):
        self.backbone_frozen = frozen
        for k in BACKBONE_KEYS:
            # modules unused at this level never train
            trainable = (not frozen) and (k in self.used_backbone_keys)
            for p in getattr(self, k).parameters():
                p.requires_grad = trainable
        self.train(self.training)
        print("❄️ Backbone frozen" if frozen else "🔥 Backbone trainable")

    def train(self, mode: bool = True):
        # A frozen backbone stays in eval mode: no dropout, and the pooler
        # uses its inference assignment instead of straight-through.
        # When fine-tuning, the pooler alone stays in eval mode so the soft
        # inference assignment is used in training as at evaluation
        # (eval mode does not stop gradients).
        super().train(mode)
        if self.backbone_frozen:
            for k in BACKBONE_KEYS:
                m = getattr(self, k)
                if m is not None:
                    m.eval()
        elif self.pooler_lvl2 is not None:
            self.pooler_lvl2.eval()
        return self

    # ------------------------------------------------------------------
    def _gene_emb(self, gene_ids):
        g = self.gene_linear(self.gene_embedder(gene_ids))
        return F.normalize(g, dim=1, p=2)

    def prototype_label_map(self):
        """[K2, K] row-stochastic prototype -> label mapping (niche level)."""
        if self.level != "niche":
            raise RuntimeError("prototype_label_map exists only for level='niche'")
        return F.softmax(self.proto_to_label, dim=1)

    def _log_mixture_probs(self, attn):
        """log p(label | cell) = log sum_k attn_k * M_k, computed in log space.
        Masked (absent) prototypes have attn = 0 and contribute -inf."""
        a = attn.float()
        a = a / a.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_a = torch.where(a > 0, torch.log(a), torch.full_like(a, -torch.inf))
        log_m = F.log_softmax(self.proto_to_label.float(), dim=1)    # [K2, K]
        return torch.logsumexp(log_a.unsqueeze(-1) + log_m.unsqueeze(0), dim=1)

    def forward(self, x_subgraph_raw, edge_index_subgraph, edge_attr_b,
                node2comm_batch, super_sp_global, gene_ids, device, bs: int,
                global_reps=None, return_attention=False):
        """Returns logits for the seed nodes, [bs, K]. With
        return_attention=True (niche level) also the [bs, K2] attention."""
        x_emb = x_subgraph_raw @ self._gene_emb(gene_ids)
        z0 = self.model_lvl0(x_emb, edge_index_subgraph, edge_attr_b,
                             None, bs=bs)[:bs]                     # [bs, 128]

        if self.level == "cell":
            return self.classifier_head(z0)

        _, _, attn, _ = hierarchy_forward(
            self.model_lvl1, self.pooler_lvl2, self.model_lvl2,
            z0, node2comm_batch, super_sp_global, global_reps=global_reps)
        log_p = self._log_mixture_probs(attn)                      # [bs, K]
        return (log_p, attn) if return_attention else log_p

    # ------------------------------------------------------------------
    @torch.no_grad()
    def section_global_reps(self, slide, device, batch_size=512,
                            num_workers=4):
        """Niche level only. Whole-section community means of fc_in(z0), as in
        encode.py. Pass as `global_reps` at evaluation so predictions do not
        depend on how cells are batched. Call with the model in eval mode."""
        X = slide["X"].float().to(device)
        node2comm = slide["node2comm"].long().to(device)
        n_comm = slide["super_sp"].size(0)
        gene_emb = self._gene_emb(slide["gene_ids"].long().to(device))
        m1 = self.model_lvl1

        sums = torch.zeros(n_comm, m1.hidden_channels, device=device)
        count = torch.zeros(n_comm, device=device)
        sampler = LocalSampler(slide["edge_index"].cpu(), sizes=[20, 10],
                               num_nodes=X.size(0), batch_size=batch_size,
                               num_workers=num_workers, shuffle=False)
        for edge_block, node_idx_b, bs in sampler:
            node_idx_b = node_idx_b.to(device)
            x_emb = X[node_idx_b] @ gene_emb
            ea = (edge_block[2].to(device), edge_block[3].to(device))
            z0 = self.model_lvl0(x_emb, edge_block[:2].to(device), ea,
                                 None, bs=bs)[:bs]
            idx = node2comm[node_idx_b[:bs]]
            sums.index_add_(0, idx, m1.fc_in(z0))
            count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        return sums / count.clamp(min=1).unsqueeze(1), count
