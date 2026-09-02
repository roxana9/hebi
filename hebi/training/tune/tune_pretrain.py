#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:10:57 2026

@author: tower4
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric
from tqdm import tqdm

from hebi.training.pretrain import Train


class TuneTrain(Train):
    def __init__(self, *args, train_files=None, val_files=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_files = train_files or []
        self.val_files = val_files or []

    def discover_data_files(self):
        if self.train_files:
            self.data_files = list(self.train_files)
            print(f"📁 Using {len(self.data_files)} train slides")
        else:
            super().discover_data_files()

    @torch.no_grad()
    def evaluate_files(self, files, epoch_tag="val"):
        self.model_lvl0.eval()
        self.model_lvl1.eval()
        self.pooler_lvl2.eval()
        self.model_lvl2.eval()
        self.global_decoder.eval()

        total_loss_epoch = 0.0
        total_batches = 0

        total_loss_X_epoch = 0.0
        total_loss_A_epoch = 0.0
        total_loss_MAE_epoch = 0.0
        total_loss_pool_epoch = 0.0
        total_loss_pool_balance_epoch = 0.0
        total_loss_smooth_epoch = 0.0
        total_loss_grn_epoch = 0.0

        for slide_idx, slide_path in enumerate(files):
            self.data_path = slide_path
            self.load_data()
            self.build_sampler()

            pbar = tqdm(
                self.sampler,
                desc=f"{epoch_tag} | slide {slide_idx+1}/{len(files)}",
                unit="batch",
                leave=False,
            )

            for edge_block, node_idx_b, bs in pbar:
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

                gene_ids = self.gene_ids.to(self.device)
                gene_emb = self.gene_embedder(gene_ids)
                gene_emb = self.gene_linear(gene_emb)
                gene_emb = F.normalize(gene_emb, dim=1, p=2)

                loss_grn = torch.tensor(0.0, device=self.device)
                if self.grn_layer is not None and self.gene_graphs is not None:
                    edge_index_grn = self.gene_graphs["global"].edge_index.to(self.device)
                    gene_emb, _ = self.grn_layer(
                        gene_emb=gene_emb,
                        edge_index_grn=edge_index_grn,
                        return_attention=False
                    )
                    if hasattr(self, "grn_expr_decoder"):
                        grn_pred = self.grn_expr_decoder(gene_emb).squeeze(-1)
                        true_gene_expr = x_batch_true.mean(dim=0)
                        mask = torch.rand_like(true_gene_expr) < 0.15
                        if mask.sum() > 0:
                            loss_grn = F.mse_loss(grn_pred[mask], true_gene_expr[mask])

                x_subgraph = x_subgraph_raw @ gene_emb

                z0_batch = self.model_lvl0(
                    x_subgraph, edge_index_subgraph, edge_attr_b, None, bs=bs
                )

                node2comm_batch = self.node2comm_fixed[target_node_indices_global]
                z1_batch = self.model_lvl1.forward(
                    x_lower=z0_batch,
                    node2comm_batch=node2comm_batch,
                    super_sp_global=self.super_sp_fixed,
                )

                C_L1 = self.model_lvl1.prototypes
                C_L2, S_matrix = self.pooler_lvl2(C_L1)

                Dk_raw_L1 = self.super_sp_fixed[node2comm_batch.long(), :]
                Hk_L2_distances = torch.matmul(Dk_raw_L1.float(), S_matrix.float())

                z2_batch, z2_attn_probs = self.model_lvl2.forward(
                    x_lower=z1_batch,
                    c_l2_prototypes=C_L2,
                    hk_l2_distances=Hk_L2_distances,
                )

                z2_decoded = self.global_decoder(z2_batch)
                x_hat_batch = z2_decoded @ gene_emb.T

                loss_X = F.mse_loss(x_hat_batch, x_batch_true)

                loss_A = self.adjacency_reconstruction_loss(
                    h=z2_batch,
                    edge_index_batch=edge_index_batch_local,
                    num_nodes_batch=bs,
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

                loss_base = (
                    loss_X
                    + loss_A
                    + loss_mae
                    + loss_pool
                    + loss_pool_balance
                    + loss_smooth
                )

                loss_total = loss_base + 2.0 * loss_grn if loss_grn is not None else loss_base

                total_loss_epoch += loss_total.item()
                total_loss_X_epoch += loss_X.item()
                total_loss_A_epoch += loss_A.item()
                total_loss_MAE_epoch += loss_mae.item()
                total_loss_pool_epoch += loss_pool.item()
                total_loss_pool_balance_epoch += loss_pool_balance.item()
                total_loss_smooth_epoch += loss_smooth.item()
                total_loss_grn_epoch += loss_grn.item()
                total_batches += 1

        if total_batches == 0:
            return {"total": float("inf")}

        return {
            "total": total_loss_epoch / total_batches,
            "loss_X": total_loss_X_epoch / total_batches,
            "loss_A": total_loss_A_epoch / total_batches,
            "loss_MAE": total_loss_MAE_epoch / total_batches,
            "loss_pool": total_loss_pool_epoch / total_batches,
            "loss_balance": total_loss_pool_balance_epoch / total_batches,
            "loss_smooth": total_loss_smooth_epoch / total_batches,
            "loss_grn": total_loss_grn_epoch / total_batches,
        }

    def run_with_validation(self):
        self.discover_data_files()
        self.data_path = self.data_files[0]
        self.load_data()
        N0 = self.X.size(0)
        self.build_models(N0, self.X.shape)

        history = []

        print("\n🚀 Starting training with validation...")
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate_files(self.val_files, epoch_tag=f"val-epoch{epoch}")

            print(
                f"\n📊 Epoch {epoch} | train_total={train_loss:.4f} | "
                f"val_total={val_metrics['total']:.4f}"
            )

            history.append({
                "epoch": epoch,
                "train_total": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

            if val_metrics["total"] + self.min_delta < self.best_loss:
                self.best_loss = val_metrics["total"]
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_metrics["total"])
            else:
                self.epochs_no_improve += 1
                print(
                    f"⏳ Early stopping counter: "
                    f"{self.epochs_no_improve}/{self.early_stopping_patience}"
                )
                if self.epochs_no_improve >= self.early_stopping_patience:
                    print("⏹ Early stopping triggered.")
                    break

        return history


def split_files(data_dir, seed=42, train_ratio=0.8, val_ratio=0.2, max_files=None):
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pt")
    ])
    random.Random(seed).shuffle(files)

    if max_files is not None:
        files = files[:max_files]

    n = len(files)
    n_train = max(1, int(n * train_ratio))
    train_files = files[:n_train]
    val_files = files[n_train:]

    if len(val_files) == 0:
        val_files = train_files[-1:]
        train_files = train_files[:-1]

    return train_files, val_files


def run_search():
    data_dir = "/DATA1/Roxana/Hierarchy/processed_train_data"

    # small dev subset of slides for tuning
    train_files, val_files = split_files(
        data_dir=data_dir,
        seed=42,
        train_ratio=0.8,
        val_ratio=0.2,
        max_files=12,   # tune on 12 slides first
    )

    search_space = [
        {"batch_size": 1024, "lr": 1e-4, "weight_decay": 0.0},
        {"batch_size": 2048, "lr": 1e-4, "weight_decay": 0.0},
        {"batch_size": 1024, "lr": 3e-4, "weight_decay": 1e-5},
        {"batch_size": 2048, "lr": 3e-4, "weight_decay": 1e-5},
    ]

    results = []

    for i, cfg in enumerate(search_space, 1):
        print("\n" + "=" * 80)
        print(f"Trial {i}/{len(search_space)} | {cfg}")
        print("=" * 80)

        trainer = TuneTrain(
            data_dir=data_dir,
            train_files=train_files,
            val_files=val_files,
            l0_encoder_type="transformer",
            batch_size=cfg["batch_size"],
            num_workers=4,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            num_epochs=8,
            global_num_layers=2,
            num_prototypes_lvl1=1024,
            num_prototypes_lvl2=20,
            decoder_hidden_dim=256,
            early_stopping_patience=3,
            checkpoint_path=f"trial_{i}_best.pt",
        )

        history = trainer.run_with_validation()
        best_val = min(x["val_total"] for x in history)

        results.append({
            **cfg,
            "best_val_total": best_val,
            "history": history,
        })

    results = sorted(results, key=lambda x: x["best_val_total"])

    print("\n🏆 BEST CONFIG")
    print(results[0])

    with open("tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_search()