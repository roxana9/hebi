#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundation pretraining over multiple spatial slides (.pt files)

- 3-level architecture:
  L0: LocalTransformerEncoder (or GAT)
  L1: Node-to-Prototype Transformer
  L2: PrototypePooler + FeatureTransformer
  Decoder: Global MLP from Z2 back to gene space

- Trains across ALL slides every epoch (foundation model style)
- Early stopping + best checkpoint saving
- GPU usage logging
- Loss history stored for plotting

Note: `grn_layer` and `grn_expr_decoder` are not constructed here. They are
attached externally after instantiation; when both are absent the GRN term
stays at zero.

Author: Roxana
"""

import argparse
import os
import sys
import time
import random
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric
import matplotlib.pyplot as plt

from sampler import LocalSampler
from model import (
    GATEncoder,
    LocalTransformerEncoder,
    NodeToPrototypeTransformer,
    PrototypePoolerL2,
    FeatureTransformerL2,
    GlobalMLPDecoder,
)


# Loss weights for the objective in the Methods. lambda_X is 1 by
# construction: the other terms are expressed relative to reconstruction.
DEFAULT_LOSS_WEIGHTS = {
    "X": 1.0,
    "A": 1.0,
    "MAE": 0.5,
    "pool": 0.01,
    "balance": 2.0,
    "smooth": 3.0,
    "coverage": 0.5,
    "GRN": 2.0,
}


class Train:
    def __init__(
        self,
        data_path=None,                 # single slide (optional)
        data_dir=None,                  # directory of .pt slides (preferred)
        vocab_path="./default_gene_vocab.json",
        l0_encoder_type: str = "transformer",
        batch_size=2048,
        num_workers=4,
        num_epochs=30,
        lr=1e-4,
        weight_decay=0.0,
        global_num_layers: int = 2,
        num_prototypes_lvl1: int = 1024,
        num_prototypes_lvl2: int = 20,
        decoder_hidden_dim: int = 256,
        early_stopping_patience: int = 10,
        min_delta: float = 1e-4,
        checkpoint_path: str = "best_model.pt",
        device=None,
        seed: int = 0,
        loss_weights: dict = None,
    ):
        # --- Loss weights ---
        # Defaults are the values the released checkpoint was trained
        # with. Overriding them changes the objective, so any run that
        # does should say so alongside its results.
        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        if loss_weights:
            unknown = set(loss_weights) - set(DEFAULT_LOSS_WEIGHTS)
            if unknown:
                raise ValueError(
                    f"unknown loss weight(s): {sorted(unknown)}. "
                    f"Valid keys: {sorted(DEFAULT_LOSS_WEIGHTS)}"
                )
            self.loss_weights.update(loss_weights)

        # --- Reproducibility ---
        self.seed = seed
        self.set_seed(self.seed)

        # --- Data config ---
        self.data_path = data_path
        self.data_dir = data_dir
        self.data_files = None  # will be list of .pt files

        # --- Hyperparameters ---
        self.vocab_path = vocab_path
        self.l0_encoder_type = l0_encoder_type.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.global_num_layers = global_num_layers
        self.num_prototypes_lvl1 = num_prototypes_lvl1
        self.num_prototypes_lvl2 = num_prototypes_lvl2
        self.decoder_hidden_dim = decoder_hidden_dim

        # --- Early stopping / checkpoint ---
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.epochs_no_improve = 0

        # --- Device ---
        if device is None:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"--- Using device: {self.device} ---")

        # --- Gene vocab / embedding ---
        with open(self.vocab_path, "r") as f:
            self.gene_vocab = json.load(f)  # dict: gene_symbol -> id
        self.vocab_size = len(self.gene_vocab)
        self.embedding_size = 128

        self.gene_embedder = nn.Embedding(self.vocab_size, self.embedding_size).to(
            self.device
        )
        self.gene_linear = nn.Linear(self.embedding_size, self.embedding_size).to(
            self.device
        )
        with torch.no_grad():
            eye = torch.eye(self.embedding_size, device=self.device)
            self.gene_linear.weight.copy_(eye)
            self.gene_linear.bias.zero_()

        # --- Model placeholders ---
        self.model_lvl0 = None
        self.model_lvl1 = None
        self.pooler_lvl2 = None
        self.model_lvl2 = None
        self.global_decoder = None
        self.optimizer = None
        self.grn_layer = None
        self.gene_graphs = None
        self.gene_graphs_list = None

        # --- Per-slide tensors (updated in load_data) ---
        self.X = None
        self.coords = None
        self.edge_index = None
        self.node2comm_fixed = None
        self.super_sp_fixed = None
        self.gene_ids = None
        self.meta = {}

        # --- Loss history across epochs ---
        self.loss_total_history = []
        self.loss_X_history = []
        self.loss_A_history = []
        self.loss_MAE_history = []
        self.loss_pool_history = []
        self.loss_balance_history = []
        self.loss_smooth_map_history = []

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    @staticmethod
    def set_seed(seed: int):
        """Seed every RNG that affects training: slide shuffling, neighbour
        sampling, weight init, masking, and negative-pair sampling."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"🎲 Seed set to {seed}")

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def discover_data_files(self):
        """
        Populate self.data_files with a list of .pt files.
        If data_dir is provided, use all .pt files there.
        Otherwise fall back to single data_path.
        """
        if self.data_dir is not None:
            files = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.lower().endswith(".pt")
            ]
            files = sorted(files)
            if not files:
                raise RuntimeError(f"No .pt files found in {self.data_dir}")
            self.data_files = files
            print(f"📁 Found {len(files)} slides in {self.data_dir}")
        elif self.data_path is not None:
            self.data_files = [self.data_path]
            print(f"📁 Using single slide: {self.data_path}")
        else:
            raise RuntimeError("Either data_dir or data_path must be provided.")

    def load_data(self):
        """Load one preprocessed slide from self.data_path."""
        print(f"📦 Loading preprocessed slide from {self.data_path}")
        data = torch.load(self.data_path, map_location=self.device)

        self.X = data["X"].float().to(self.device)  # [N0, G_slide]
        self.coords = data["coords"].float().to(self.device)  # [N0, 2]
        self.edge_index = data["edge_index"].long().to(self.device)
        self.node2comm_fixed = data["node2comm"].long().to(self.device)
        self.super_sp_fixed = data["super_sp"].float().to(self.device)
        self.gene_ids = data["gene_ids"].long().to(self.device)  # [G_slide]
        self.meta = data.get("meta", {})

        slide_genes = data.get("gene_symbols", None)
        if slide_genes is not None:
            ids = [self.gene_vocab[g] for g in slide_genes]
            self.gene_ids = torch.tensor(ids, dtype=torch.long, device=self.device)

        print(
            f"✅ Loaded data: X={self.X.shape}, coords={self.coords.shape}, "
            f"edge_index={self.edge_index.shape}"
        )

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def gpu_usage_str(self):
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(self.device) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            return f"{mem_alloc:.2f}/{mem_reserved:.2f} GB"
        else:
            return "CPU"

    def _match_grn_graph(self, slide_path):
        """Pick the per-slide GRN graph whose node count matches this slide's
        gene panel. Matches by CONTENT (n_genes), not list order, because
        data_files is shuffled each epoch."""
        data = torch.load(slide_path, map_location="cpu")
        n_genes = data["X"].shape[1]
        for gg in self.gene_graphs_list:
            if gg["global"].num_nodes == n_genes:
                return gg
        raise ValueError(
            f"No GRN graph with {n_genes} nodes for {os.path.basename(slide_path)}; "
            f"available: {[gg['global'].num_nodes for gg in self.gene_graphs_list]}")

    def save_checkpoint(self, epoch: int, avg_loss: float):
        state = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "seed": self.seed,
            "loss_weights": self.loss_weights,
            "gene_embedder": self.gene_embedder.state_dict(),
            "gene_linear": self.gene_linear.state_dict(),
            "model_lvl0": self.model_lvl0.state_dict(),
            "model_lvl1": self.model_lvl1.state_dict(),
            "pooler_lvl2": self.pooler_lvl2.state_dict(),
            "model_lvl2": self.model_lvl2.state_dict(),
            "global_decoder": self.global_decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "vocab": self.gene_vocab,
        }

        # The GRN modules are attached from outside and only exist when the
        # module is active. Without these two lines the refinement layer is
        # trained during continual adaptation and then discarded here, so
        # encoding would rebuild it randomly and every gene-level result —
        # hub genes, edge salience, cell-cell communication — would come
        # from untrained attention.
        if getattr(self, "grn_layer", None) is not None:
            state["grn_layer"] = self.grn_layer.state_dict()
        if getattr(self, "grn_expr_decoder", None) is not None:
            state["grn_expr_decoder"] = self.grn_expr_decoder.state_dict()

        torch.save(state, self.checkpoint_path)
        print(
            f"💾 Saved new best checkpoint (loss={avg_loss:.4f}) → {self.checkpoint_path}"
        )

    # ------------------------------------------------------------------
    # Model + sampler
    # ------------------------------------------------------------------
    def build_models(self, N0, X_shape):
        print("🔧 Initializing models...")

        # L0 encoder
        l0_config = {
            "in_channels": self.embedding_size,
            "hidden_channels": 128,
            "out_channels": 128,
            "num_layers": 3,
            "heads": 4,
            "dropout": 0.1,
        }

        if self.l0_encoder_type == "gat":
            print("  -> Using L0 Encoder: GATEncoder")
            self.model_lvl0 = GATEncoder(**l0_config).to(self.device)
        elif self.l0_encoder_type == "transformer":
            print("  -> Using L0 Encoder: LocalTransformerEncoder")
            self.model_lvl0 = LocalTransformerEncoder(**l0_config).to(self.device)
        else:
            raise ValueError(f"Unknown l0_encoder_type: {self.l0_encoder_type}")

        # L1 transformer
        transformer_l1_config = {
            "in_channels": 128,
            "hidden_channels": 128,
            "out_channels": 128,
            "num_layers": self.global_num_layers,
            "heads": 4,
            "ff_dropout": 0.1,
            "attn_dropout": 0.1,
            "num_prototypes": self.num_prototypes_lvl1,
        }
        self.model_lvl1 = NodeToPrototypeTransformer(**transformer_l1_config).to(
            self.device
        )

        # L2 pooler
        self.pooler_lvl2 = PrototypePoolerL2(
            in_channels=128,
            num_l1_prototypes=self.num_prototypes_lvl1,
            num_l2_prototypes=self.num_prototypes_lvl2,
        ).to(self.device)

        # L2 transformer
        transformer_l2_config = {
            "in_channels": 128,
            "hidden_channels": 128,
            "out_channels": 128,
            "num_layers": self.global_num_layers,
            "heads": 4,
            "ff_dropout": 0.1,
            "attn_dropout": 0.1,
        }
        self.model_lvl2 = FeatureTransformerL2(**transformer_l2_config).to(self.device)

        # Decoder
        self.global_decoder = GlobalMLPDecoder(
            in_dim=128, hidden_dim=self.decoder_hidden_dim, out_dim=self.embedding_size
        ).to(self.device)

        # Optimizer
        print("🔧 Initializing optimizer...")
        params = (
            list(self.gene_embedder.parameters())
            + list(self.gene_linear.parameters())
            + list(self.model_lvl0.parameters())
            + list(self.model_lvl1.parameters())
            + list(self.pooler_lvl2.parameters())
            + list(self.model_lvl2.parameters())
            + list(self.global_decoder.parameters())
        )
        self.optimizer = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

    def build_sampler(self):
        self.sampler = LocalSampler(
            self.edge_index.cpu(),
            sizes=[20, 10],
            num_nodes=self.X.size(0),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        print(f"✅ LocalSampler initialised: batch_size={self.batch_size}")

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def adjacency_reconstruction_loss(self, h, edge_index_batch, num_nodes_batch):
        h = F.normalize(h, p=2, dim=-1)
        src, dst = edge_index_batch
        num_edges = len(src)
        if num_edges == 0:
            return torch.tensor(0.0, device=self.device)

        # Clamp indices to valid range
        src = src.clamp(0, h.shape[0] - 1)
        dst = dst.clamp(0, h.shape[0] - 1)

        pos_scores = torch.sigmoid((h[src] * h[dst]).sum(dim=-1))
        pos_scores = torch.clamp(pos_scores, 1e-4, 1 - 1e-4)
        loss_pos = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))

        neg_src = torch.randint(0, h.shape[0], (num_edges,), device=self.device)
        neg_dst = torch.randint(0, h.shape[0], (num_edges,), device=self.device)
        neg_scores = torch.sigmoid((h[neg_src] * h[neg_dst]).sum(dim=-1))
        neg_scores = torch.clamp(neg_scores, 1e-4, 1 - 1e-4)
        loss_neg = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))

        return 0.5 * (loss_pos + loss_neg)

    def pool_entropy_loss(self, S_matrix):
        S_entropy = -(S_matrix * torch.log(S_matrix + 1e-8)).sum(dim=1)
        return S_entropy.mean()

    def proto_coverage_loss(self, z2_attn_probs):
        # Commitment only — per-cell entropy (soft attn still has gradients)
        per_cell_entropy = -(z2_attn_probs * torch.log(z2_attn_probs + 1e-8)).sum(dim=1)
        loss_commit = per_cell_entropy.mean()
        return loss_commit

    def pool_balancing_loss(self, S_matrix):
        # S_matrix shape: [1024, 20]
        # Normalized usage per L2 prototype
        cluster_usage = S_matrix.mean(dim=0)  # [20]

        # Maximize entropy of usage = minimize negative entropy
        entropy = -(cluster_usage * torch.log(cluster_usage + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(
            float(self.num_prototypes_lvl2), device=self.device
        ))

        # 0 = perfectly balanced, 1 = fully collapsed
        return 1.0 - (entropy / max_entropy)

    def smooth_map_entropy_loss(self, attn_probs):
        if attn_probs is None:
            return torch.tensor(0.0, device=self.device)
        batch_probs = attn_probs.mean(dim=0)
        batch_probs = batch_probs / (batch_probs.sum() + 1e-8)

        entropy = -(batch_probs * torch.log(batch_probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(
            float(self.num_prototypes_lvl2), device=self.device))
        return 1.0 - (entropy / max_entropy)

    # ------------------------------------------------------------------
    # One epoch over ALL slides
    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int):
        self.model_lvl0.train()
        self.model_lvl1.train()
        self.pooler_lvl2.train()
        self.model_lvl2.train()
        self.global_decoder.train()

        total_loss_epoch = 0.0
        total_batches = 0

        total_loss_X_epoch = 0.0
        total_loss_A_epoch = 0.0
        total_loss_MAE_epoch = 0.0
        total_loss_pool_epoch = 0.0
        total_loss_pool_balance_epoch = 0.0
        total_loss_smooth_epoch = 0.0
        total_loss_grn = 0.0
        total_loss_coverage_epoch = 0.0

        start_time = time.time()
        random.shuffle(self.data_files)
        for slide_idx, slide_path in enumerate(self.data_files):
            self.data_path = slide_path
            print(
                f"\n🧬 Epoch {epoch}: Slide {slide_idx+1}/{len(self.data_files)} → {slide_path}"
            )
            if getattr(self, "gene_graphs_list", None) is not None:
                self.gene_graphs = self._match_grn_graph(slide_path)
            # Load slide & sampler
            self.load_data()
            self.build_sampler()

            pbar = tqdm(
                self.sampler,
                desc=f"Epoch {epoch} | slide {slide_idx+1}",
                unit="batch",
            )

            for i, (edge_block, node_idx_b, bs) in enumerate(pbar):
                x_subgraph_raw = self.X[node_idx_b].to(self.device)
                edge_index_subgraph = edge_block[:2].to(self.device)
                edge_attr_b = (
                    edge_block[2].to(self.device),
                    edge_block[3].to(self.device),
                )
                target_node_indices_global = node_idx_b[:bs]
                x_batch_true = x_subgraph_raw[:bs]

                edge_index_batch_local, _ = torch_geometric.utils.subgraph(
                    torch.arange(bs, device=self.device),
                    edge_index_subgraph,
                    relabel_nodes=True,
                )

                # Gene projection
                gene_ids = self.gene_ids.to(self.device)
                gene_emb = self.gene_embedder(gene_ids)
                gene_emb = self.gene_linear(gene_emb)
                gene_emb = F.normalize(gene_emb, dim=1, p=2)
                gene_emb_clean = gene_emb.detach().clone()
                loss_grn = torch.tensor(0.0, device=self.device)
                if self.grn_layer is not None and self.gene_graphs is not None:
                    edge_index_grn = self.gene_graphs["global"].edge_index.to(self.device)
                    gene_emb, _ = self.grn_layer(
                        gene_emb=gene_emb,
                        edge_index_grn=edge_index_grn,
                        return_attention=False
                    )
                    # GRN expression prediction loss
                    if hasattr(self, "grn_expr_decoder"):
                        grn_pred = self.grn_expr_decoder(gene_emb).squeeze(-1)
                        true_gene_expr = x_batch_true.mean(dim=0)
                        mask_ratio = 0.15
                        mask = torch.rand_like(true_gene_expr) < mask_ratio
                        if mask.sum() > 0:
                            loss_grn = F.mse_loss(grn_pred[mask], true_gene_expr[mask])
                x_subgraph = x_subgraph_raw @ gene_emb  # [*, 128]

                # L0
                z0_batch = self.model_lvl0(
                    x_subgraph, edge_index_subgraph, edge_attr_b, None, bs=bs
                )

                # L1
                node2comm_batch = self.node2comm_fixed[target_node_indices_global]
                z1_batch = self.model_lvl1.forward(
                    x_lower=z0_batch,
                    node2comm_batch=node2comm_batch,
                    super_sp_global=self.super_sp_fixed,
                )

                # L2
                C_L1 = self.model_lvl1.prototypes
                C_L2, S_matrix = self.pooler_lvl2(C_L1)

                Dk_raw_L1 = self.super_sp_fixed[node2comm_batch.long(), :]
                S_hard = torch.zeros_like(S_matrix)
                S_hard.scatter_(1, S_matrix.argmax(dim=1, keepdim=True), 1.0)
                Hk_L2_distances = torch.matmul(Dk_raw_L1.float(), S_hard.float())

                z2_batch, z2_attn_probs = self.model_lvl2.forward(
                    x_lower=z1_batch,
                    c_l2_prototypes=C_L2,
                    hk_l2_distances=Hk_L2_distances,
                )
                loss_coverage = self.proto_coverage_loss(z2_attn_probs)

                # Decoder
                z2_decoded = self.global_decoder(z2_batch)
                x_hat_batch = z2_decoded @ gene_emb_clean.T

                # Losses
                loss_X = F.mse_loss(x_hat_batch, x_batch_true)

                loss_A = self.adjacency_reconstruction_loss(
                    h=z0_batch,                    # z0 covers full subgraph [n_sub, 128]
                    edge_index_batch=edge_index_batch_local,
                    num_nodes_batch=edge_index_subgraph.max().item() + 1,  # actual subgraph size
                )
                mask_ratio = 0.15
                mask_missing = (torch.rand_like(x_batch_true) < mask_ratio).float()
                X_hat_missing_parts = x_hat_batch * mask_missing
                X_true_missing_parts = x_batch_true * mask_missing
                loss_mae = F.mse_loss(
                    X_hat_missing_parts,
                    X_true_missing_parts,
                    reduction="sum",
                ) / (mask_missing.sum() + 1e-8)

                loss_pool = self.pool_entropy_loss(S_matrix)
                loss_pool_balance = self.pool_balancing_loss(S_matrix)
                loss_smooth = self.smooth_map_entropy_loss(z2_attn_probs)

                w = self.loss_weights
                loss_base = (
                    w["X"] * loss_X
                    + w["A"] * loss_A
                    + w["MAE"] * loss_mae
                    + w["pool"] * loss_pool
                    + w["balance"] * loss_pool_balance
                    + w["smooth"] * loss_smooth
                    + w["coverage"] * loss_coverage
                )

                if loss_grn is not None:
                    loss_total = loss_base + w["GRN"] * loss_grn
                else:
                    loss_total = loss_base

                # Backprop
                self.optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]["params"], max_norm=1.0
                )
                self.optimizer.step()

                # Accumulate
                total_loss_epoch += loss_total.item()
                total_loss_X_epoch += loss_X.item()
                total_loss_A_epoch += loss_A.item()
                total_loss_MAE_epoch += loss_mae.item()
                total_loss_pool_epoch += loss_pool.item()
                total_loss_pool_balance_epoch += loss_pool_balance.item()
                total_loss_smooth_epoch += loss_smooth.item()
                total_loss_coverage_epoch += loss_coverage.item()
                total_loss_grn += loss_grn.item()
                total_batches += 1

                if (i + 1) % 10 == 0:
                    pbar.set_postfix(
                        Total=f"{loss_total.item():.4f}",
                        X=f"{loss_X.item():.4f}",
                        Pool=f"{loss_pool.item():.4f}",
                        Balance=f"{loss_pool_balance.item():.4f}",
                        grn=f"{loss_grn.item():.4f}",
                        Coverage=f"{loss_coverage.item():.4f}",
                        GPU=self.gpu_usage_str(),
                    )

        if total_batches == 0:
            print(f"⚠ Epoch {epoch} had 0 batches.")
            return float("inf")

        avg_loss_total = total_loss_epoch / total_batches
        avg_loss_X = total_loss_X_epoch / total_batches
        avg_loss_A = total_loss_A_epoch / total_batches
        avg_loss_MAE = total_loss_MAE_epoch / total_batches
        avg_loss_pool = total_loss_pool_epoch / total_batches
        avg_loss_pool_balance = total_loss_pool_balance_epoch / total_batches
        avg_loss_smooth = total_loss_smooth_epoch / total_batches
        avg_loss_grn = total_loss_grn / total_batches
        avg_loss_coverage = total_loss_coverage_epoch / total_batches

        elapsed = time.time() - start_time

        print(f"\n✅ Epoch {epoch} DONE in {elapsed/60:.1f} min")
        print(f"   Avg Total Loss: {avg_loss_total:.4f}")
        print(f"   Avg Loss_X:     {avg_loss_X:.4f}")
        print(f"   Avg Loss_MAE:   {avg_loss_MAE:.4f}")
        print(f"   Avg Loss_A:     {avg_loss_A:.4f}")
        print(f"   Avg Pool:       {avg_loss_pool:.4f}")
        print(f"   Avg Balance:    {avg_loss_pool_balance:.4f}")
        print(f"   Avg Smooth:     {avg_loss_smooth:.4f}")
        print(f"   Avg GRN:        {avg_loss_grn:.4f}")
        print(f"   Avg Coverage:   {avg_loss_coverage:.4f}")
        print(f"   GPU:            {self.gpu_usage_str()}")

        # Store for plotting
        self.loss_total_history.append(avg_loss_total)
        self.loss_X_history.append(avg_loss_X)
        self.loss_A_history.append(avg_loss_A)
        self.loss_MAE_history.append(avg_loss_MAE)
        self.loss_pool_history.append(avg_loss_pool)
        self.loss_balance_history.append(avg_loss_pool_balance)
        self.loss_smooth_map_history.append(avg_loss_smooth)

        return avg_loss_total

    # ------------------------------------------------------------------
    # Run training
    # ------------------------------------------------------------------
    def run(self):
        # 1) Find slides
        self.discover_data_files()

        # 2) Warm-up with first slide to build models
        self.data_path = self.data_files[0]
        self.load_data()
        N0 = self.X.size(0)
        self.build_models(N0, self.X.shape)

        print("\n🚀 Starting foundation pretraining over all slides...")
        for epoch in range(1, self.num_epochs + 1):
            avg_loss = self.train_epoch(epoch)

            if avg_loss + self.min_delta < self.best_loss:
                self.best_loss = avg_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, avg_loss)
            else:
                self.epochs_no_improve += 1
                print(
                    f"⏳ Early stopping counter: {self.epochs_no_improve}/{self.early_stopping_patience}"
                )
                if self.epochs_no_improve >= self.early_stopping_patience:
                    print("⏹ Early stopping triggered. Training stopped.")
                    break

        print(f"\n🏁 Training finished. Best loss = {self.best_loss:.4f}")
        return None

    # ------------------------------------------------------------------
    # Optional: plot losses after training
    # ------------------------------------------------------------------
    def plot_losses(self):
        if not self.loss_total_history:
            print("No loss history to plot.")
            return

        epochs = range(1, len(self.loss_total_history) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.loss_total_history, label="Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Total Training Loss per Epoch")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 7))
        plt.plot(epochs, self.loss_X_history, label="X recon")
        plt.plot(epochs, self.loss_A_history, label="Adjacency")
        plt.plot(epochs, self.loss_MAE_history, label="MAE masked")
        plt.plot(epochs, self.loss_pool_history, label="Pool entropy")
        plt.plot(epochs, self.loss_balance_history, label="Pool balance")
        plt.plot(epochs, self.loss_smooth_map_history, label="Smooth map")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss components per Epoch")
        plt.grid(True)
        plt.legend()
        plt.show()


# =====================================================================
# Main
# =====================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Pretrain HEBI on a corpus of preprocessed sections."
    )
    p.add_argument("--config", type=str, default=None,
                   help="YAML config. Command-line flags override it.")

    p.add_argument("--data_dir", type=str,
                   help="Directory of preprocessed .pt sections.")
    p.add_argument("--vocab_path", type=str,
                   help="Gene vocabulary JSON.")
    p.add_argument("--checkpoint_path", type=str,
                   default="hebi_foundation.pt",
                   help="Where the best checkpoint is written.")

    p.add_argument("--l0_encoder_type", choices=["transformer", "gat"],
                   default="transformer")
    p.add_argument("--global_num_layers", type=int, default=2)
    p.add_argument("--num_prototypes_lvl1", type=int, default=1024,
                   help="METIS communities, the micro-node level.")
    p.add_argument("--num_prototypes_lvl2", type=int, default=20,
                   help="Niche prototypes, the macro level.")
    p.add_argument("--decoder_hidden_dim", type=int, default=256)

    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=1e-4)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot_losses", action="store_true",
                   help="Show the loss curves when training finishes.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    loss_weights = {}

    if args.config:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

        loss_weights = config.pop("loss_weights", {}) or {}

        tokens = argv if argv is not None else sys.argv[1:]
        given = {t.lstrip("-") for t in tokens if t.startswith("--")}
        for key, value in config.items():
            if key not in given and hasattr(args, key):
                setattr(args, key, value)

    for required in ("data_dir", "vocab_path"):
        if getattr(args, required, None) is None:
            raise SystemExit(f"--{required} is required (flag or config file)")

    trainer = Train(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        l0_encoder_type=args.l0_encoder_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        global_num_layers=args.global_num_layers,
        num_prototypes_lvl1=args.num_prototypes_lvl1,
        num_prototypes_lvl2=args.num_prototypes_lvl2,
        decoder_hidden_dim=args.decoder_hidden_dim,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        seed=args.seed,
        loss_weights=loss_weights,
    )

    print("Loss weights: " +
          ", ".join(f"{k}={v}" for k, v in trainer.loss_weights.items()))

    trainer.run()
    if args.plot_losses:
        trainer.plot_losses()
    return trainer


if __name__ == "__main__":
    main()
