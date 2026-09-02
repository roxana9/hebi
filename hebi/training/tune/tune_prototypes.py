#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:01:27 2026

@author: tower4
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import scanpy as sc

from pretraining import Train  # change this import


class PrototypeTuneTrain(Train):
    def __init__(
        self,
        *args,
        train_files=None,
        val_files=None,
        labelled_val_pt=None,
        labelled_val_adata=None,
        eval_n_clusters=24,
        eval_label_key="Subclass",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_files = train_files or []
        self.val_files = val_files or []
        self.labelled_val_pt = labelled_val_pt
        self.labelled_val_adata = labelled_val_adata
        self.eval_n_clusters = eval_n_clusters
        self.eval_label_key = eval_label_key

    def discover_data_files(self):
        if self.train_files:
            self.data_files = list(self.train_files)
            print(f"📁 Using {len(self.data_files)} training slides")
        else:
            super().discover_data_files()

    @torch.no_grad()
    def evaluate_loss_on_files(self, files, tag="val"):
        self.model_lvl0.eval()
        self.model_lvl1.eval()
        self.pooler_lvl2.eval()
        self.model_lvl2.eval()
        self.global_decoder.eval()

        total_loss_epoch = 0.0
        total_batches = 0

        for slide_idx, slide_path in enumerate(files):
            self.data_path = slide_path
            self.load_data()
            self.build_sampler()

            pbar = tqdm(
                self.sampler,
                desc=f"{tag} | slide {slide_idx+1}/{len(files)}",
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
                        return_attention=False,
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
                total_batches += 1

        if total_batches == 0:
            return float("inf")

        return total_loss_epoch / total_batches

    @torch.no_grad()
    def extract_z0_embeddings_from_pt(self, pt_path):
        """
        Extract z0 embeddings for all cells in one .pt slide.
        IMPORTANT:
        This assumes the .pt cell order matches the labelled adata cell order.
        """
        self.model_lvl0.eval()

        self.data_path = pt_path
        self.load_data()

        # Full-graph forward once, then take all cells
        gene_ids = self.gene_ids.to(self.device)
        gene_emb = self.gene_embedder(gene_ids)
        gene_emb = self.gene_linear(gene_emb)
        gene_emb = F.normalize(gene_emb, dim=1, p=2)

        x_all = self.X.to(self.device) @ gene_emb

        # If your edge attributes are required but unavailable here, use zeros
        num_edges = self.edge_index.shape[1]
        dummy_edge_attr = (
            torch.zeros(num_edges, device=self.device),
            torch.zeros(num_edges, device=self.device),
        )

        z0_all = self.model_lvl0(
            x_all,
            self.edge_index,
            dummy_edge_attr,
            None,
            bs=x_all.shape[0],
        )

        return z0_all.detach().cpu().numpy()

    def evaluate_kmeans_on_labelled_slide(self):
        if self.labelled_val_pt is None or self.labelled_val_adata is None:
            raise ValueError("labelled_val_pt and labelled_val_adata must be provided")

        X = self.extract_z0_embeddings_from_pt(self.labelled_val_pt)
        y_true = self.labelled_val_adata.obs[self.eval_label_key].astype(str).values

        if X.shape[0] != len(y_true):
            raise ValueError(
                f"Mismatch between embeddings ({X.shape[0]}) and labels ({len(y_true)}). "
                "Check that the .pt slide and adata have the same cell order and same cells."
            )

        kmeans = KMeans(
            n_clusters=self.eval_n_clusters,
            random_state=42,
            n_init=20,
        )
        y_pred = kmeans.fit_predict(X)

        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        sil = silhouette_score(X, y_pred)

        return {
            "ARI": float(ari),
            "NMI": float(nmi),
            "Silhouette": float(sil),
        }

    def run_with_validation(self):
        self.discover_data_files()

        # Build models using first training slide
        self.data_path = self.data_files[0]
        self.load_data()
        N0 = self.X.size(0)
        self.build_models(N0, self.X.shape)

        history = []

        print("\n🚀 Starting prototype tuning...")
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate_loss_on_files(self.val_files, tag=f"val-epoch{epoch}")
            cluster_metrics = self.evaluate_kmeans_on_labelled_slide()

            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "ARI": cluster_metrics["ARI"],
                "NMI": cluster_metrics["NMI"],
                "Silhouette": cluster_metrics["Silhouette"],
            }
            history.append(row)

            print(
                f"\n📊 Epoch {epoch} | "
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f} | "
                f"ARI={row['ARI']:.4f} | "
                f"NMI={row['NMI']:.4f} | "
                f"Sil={row['Silhouette']:.4f}"
            )

            # still checkpoint by validation loss
            if val_loss + self.min_delta < self.best_loss:
                self.best_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_loss)
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


def split_files(data_dir, seed=42, train_ratio=0.8, max_files=12):
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pt")
    ])
    random.Random(seed).shuffle(files)

    files = files[:max_files]
    n_train = max(1, int(len(files) * train_ratio))

    train_files = files[:n_train]
    val_files = files[n_train:]

    if len(val_files) == 0:
        val_files = train_files[-1:]
        train_files = train_files[:-1]

    return train_files, val_files


def choose_best_epoch(history):
    # primary = ARI, secondary = NMI, tertiary = low val loss
    return sorted(
        history,
        key=lambda x: (-x["ARI"], -x["NMI"], x["val_loss"])
    )[0]


def run_search():
    # ------------------------------------------------------------------
    # CHANGE THESE PATHS
    # ------------------------------------------------------------------
    data_dir = "/DATA1/Roxana/Hierarchy/processed_train_data"
    labelled_val_pt = "/DATA1/Roxana/Hierarchy/processed_seaad31/SEAAD_MTG_MERFISH_31"
    labelled_val_adata_path = "/DATA1/Roxana/Data/SEAAD_MTG_MERFISH_31.h5ad"

    labelled_val_adata = sc.read_h5ad(labelled_val_adata_path)

    # ------------------------------------------------------------------
    # FIX YOUR BEST VALUES HERE
    # replace these 3 with the best values from your first tuning round
    # ------------------------------------------------------------------
    BEST_BATCH_SIZE = 1024
    BEST_LR = 1e-4
    BEST_WEIGHT_DECAY =  0.0

    # ------------------------------------------------------------------
    # Prototype search space
    # ------------------------------------------------------------------
    search_space = [
        {"num_prototypes_lvl1": 512,  "num_prototypes_lvl2": 20},
        {"num_prototypes_lvl1": 1024, "num_prototypes_lvl2": 20},
        {"num_prototypes_lvl1": 1024, "num_prototypes_lvl2": 24},
        {"num_prototypes_lvl1": 1024, "num_prototypes_lvl2": 30},
    ]

    train_files, val_files = split_files(
        data_dir=data_dir,
        seed=42,
        train_ratio=0.8,
        max_files=12,
    )

    results = []

    for i, cfg in enumerate(search_space, 1):
        print("\n" + "=" * 90)
        print(f"Trial {i}/{len(search_space)} | config = {cfg}")
        print("=" * 90)

        trainer = PrototypeTuneTrain(
            data_dir=data_dir,
            train_files=train_files,
            val_files=val_files,
            labelled_val_pt=labelled_val_pt,
            labelled_val_adata=labelled_val_adata,
            eval_n_clusters=24,
            eval_label_key="Subclass",

            # fixed best optimization hyperparameters
            l0_encoder_type="transformer",
            batch_size=BEST_BATCH_SIZE,
            num_workers=4,
            lr=BEST_LR,
            weight_decay=BEST_WEIGHT_DECAY,

            # tune only prototypes
            num_epochs=8,
            global_num_layers=2,
            num_prototypes_lvl1=cfg["num_prototypes_lvl1"],
            num_prototypes_lvl2=cfg["num_prototypes_lvl2"],
            decoder_hidden_dim=256,
            early_stopping_patience=3,
            checkpoint_path=f"prototype_trial_{i}_best.pt",
        )

        history = trainer.run_with_validation()
        best_epoch_result = choose_best_epoch(history)

        result = {
            "num_prototypes_lvl1": cfg["num_prototypes_lvl1"],
            "num_prototypes_lvl2": cfg["num_prototypes_lvl2"],
            "best_epoch": best_epoch_result["epoch"],
            "best_train_loss": best_epoch_result["train_loss"],
            "best_val_loss": best_epoch_result["val_loss"],
            "best_ARI": best_epoch_result["ARI"],
            "best_NMI": best_epoch_result["NMI"],
            "best_Silhouette": best_epoch_result["Silhouette"],
            "history": history,
        }
        results.append(result)

    # rank trials
    results = sorted(
        results,
        key=lambda x: (-x["best_ARI"], -x["best_NMI"], x["best_val_loss"])
    )

    best = results[0]

    print("\n" + "#" * 90)
    print("BEST PROTOTYPE CONFIG")
    print("#" * 90)
    print(json.dumps({
        "num_prototypes_lvl1": best["num_prototypes_lvl1"],
        "num_prototypes_lvl2": best["num_prototypes_lvl2"],
        "best_epoch": best["best_epoch"],
        "best_val_loss": best["best_val_loss"],
        "best_ARI": best["best_ARI"],
        "best_NMI": best["best_NMI"],
        "best_Silhouette": best["best_Silhouette"],
    }, indent=2))

    with open("prototype_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_search()