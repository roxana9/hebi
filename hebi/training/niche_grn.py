#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score

from fine_tune.hebi_finetuner import HebiCellTypeClassifier
from fine_tune.supervised_loader import (
    get_slide_list,
    load_supervised_slide_dict,
    build_local_sampler,
    iterate_supervised_batches,
)

from clustering.build_grn import build_hetero_grn
from grn_layer import GRNRefinementLayer
from preprocess_offline import harmonise_adata


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one_epoch(
    model,
    files,
    target_key,
    device,
    batch_size,
    num_workers,
    criterion,
    do_f,
    do_grn=False,
    optimizer=None,
    train=True,
    desc="",
):
    model.train() if train else model.eval()

    all_true, all_pred = [], []
    total_loss, total_correct, total_samples = 0.0, 0, 0
    last_att = None

    if train:
        random.shuffle(files)

    pbar = tqdm(files, desc=desc, unit="slide")
    for pt_path in pbar:
        try:
            slide = load_supervised_slide_dict(
                pt_path, target_key=target_key, map_location="cpu"
            )
        except Exception as e:
            print("Skipping:", pt_path, "Reason:", e)
            continue

        sampler = build_local_sampler(
            slide, batch_size=batch_size, num_workers=num_workers, shuffle=train
        )

        for (
            x_subgraph_raw,
            edge_index_subgraph,
            edge_attr_b,
            node2comm_batch,
            super_sp_global,
            gene_ids,
            y_seed,
            bs,
        ) in iterate_supervised_batches(
            slide, sampler, device=device, target_key=target_key
        ):
            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                if do_f:
                    if do_grn:
                        logits, att, gene_emb = model(
                            x_subgraph_raw=x_subgraph_raw,
                            edge_index_subgraph=edge_index_subgraph,
                            edge_attr_b=edge_attr_b,
                            node2comm_batch=node2comm_batch,
                            super_sp_global=super_sp_global,
                            device=device,
                            gene_ids=gene_ids,
                            bs=bs,
                            return_grn_attention=True,
                        )
                        last_att = att

                        x_batch_true = x_subgraph_raw[:bs]  # [bs, G_slide]
                        grn_pred = model.grn_expr_decoder(gene_emb).squeeze(-1)  # [G_slide]
                        true_gene_expr = x_batch_true.mean(dim=0)  # [G_slide]

                        mask_ratio = 0.15
                        mask = torch.rand_like(true_gene_expr) < mask_ratio

                        if mask.sum() > 0:
                            loss_grn = F.mse_loss(grn_pred[mask], true_gene_expr[mask])
                        else:
                            loss_grn = torch.tensor(0.0, device=device)
                    else:
                        logits = model(
                            x_subgraph_raw=x_subgraph_raw,
                            edge_index_subgraph=edge_index_subgraph,
                            edge_attr_b=edge_attr_b,
                            node2comm_batch=node2comm_batch,
                            super_sp_global=super_sp_global,
                            device=device,
                            gene_ids=gene_ids,
                            bs=bs,
                            return_grn_attention=False,
                        )
                        loss_grn = torch.tensor(0.0, device=device)
                else:
                    logits = model(
                        x_subgraph_raw=x_subgraph_raw,
                        edge_index_subgraph=edge_index_subgraph,
                        edge_attr_b=edge_attr_b,
                        node2comm_batch=node2comm_batch,
                        super_sp_global=super_sp_global,
                        device=device,
                        gene_ids=gene_ids,
                        bs=bs,
                        return_grn_attention=False,
                    )
                    loss_grn = torch.tensor(0.0, device=device)

                loss_cls = criterion(logits, y_seed)
                loss = loss_cls + loss_grn

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += float(loss.item()) * int(bs)
            preds = torch.argmax(logits, dim=1)

            all_true.append(y_seed.detach().cpu())
            all_pred.append(preds.detach().cpu())

            total_correct += int((preds == y_seed).sum().item())
            total_samples += int(bs)

        if total_samples > 0:
            pbar.set_postfix(
                loss=total_loss / total_samples,
                acc=total_correct / total_samples,
            )

    y_true = torch.cat(all_true).numpy() if len(all_true) else np.array([])
    y_pred = torch.cat(all_pred).numpy() if len(all_pred) else np.array([])

    micro_f1 = f1_score(y_true, y_pred, average="micro") if y_true.size else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") if y_true.size else 0.0
    acc = (total_correct / total_samples) if total_samples > 0 else 0.0
    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0

    if do_f and do_grn:
        return avg_loss, acc, micro_f1, macro_f1, y_true, y_pred, last_att
    else:
        return avg_loss, acc, micro_f1, macro_f1, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(
        description="HEBI niche fine-tuning with optional GRN layer"
    )

    parser.add_argument("--foundation_ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--target_key", type=str, default="y_niche")
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--h5ad_path", type=str, required=True,
                        help="Path to the original .h5ad file.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to default_gene_vocab.json for harmonization.")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--n_epochs_linear", type=int, default=10)
    parser.add_argument("--lr_head", type=float, default=1e-4)

    parser.add_argument("--do_finetune", action="store_true")
    parser.add_argument("--n_epochs_finetune", type=int, default=1)
    parser.add_argument("--lr_backbone", type=float, default=1e-6)

    # GRN options
    parser.add_argument("--do_grn", action="store_true")
    parser.add_argument("--species", type=str, default="Human")
    parser.add_argument("--grn_heads", type=int, default=4)
    parser.add_argument("--grn_max_cells", type=int, default=3000)
    parser.add_argument("--spearman_threshold", type=float, default=0.4)
    parser.add_argument("--lambda_grn", type=float, default=1.0)
    parser.add_argument("--mask_ratio", type=float, default=0.15)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_files = get_slide_list(args.data_root, split="train")
    test_files = get_slide_list(args.data_root, split="test")
    print("Train slides:", len(train_files), "Test slides:", len(test_files))
    print("Task:", args.target_key, "| n_classes:", args.n_classes)

    model = HebiCellTypeClassifier(
        n_classes=args.n_classes,
        dim_model=128,
        l0_encoder_type="transformer",
        global_num_layers=2,
        num_prototypes_lvl1=1024,
        num_prototypes_lvl2=20,
    ).to(device)

    model.load_pretrained_weights(
        args.foundation_ckpt,
        device=device,
        freeze_backbone=True,
    )
    criterion = nn.CrossEntropyLoss()

    # ---- Phase 1: linear probing ----
    print("\n❄️ PHASE 1: Linear probing")
    param_groups = [{"params": model.classifier_head.parameters(), "lr": args.lr_head}]
    optimizer = optim.Adam(param_groups)

    for epoch in range(1, args.n_epochs_linear + 1):
        tr_loss, tr_acc, tr_micro, tr_macro, _, _ = run_one_epoch(
            model=model,
            files=train_files,
            target_key=args.target_key,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            criterion=criterion,
            do_f=False,
            do_grn=False,
            optimizer=optimizer,
            train=True,
            desc=f"P1 Train {epoch}/{args.n_epochs_linear}",
        )
        te_loss, te_acc, te_micro, te_macro, _, _ = run_one_epoch(
            model=model,
            files=test_files,
            target_key=args.target_key,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            criterion=criterion,
            do_f=False,
            do_grn=False,
            optimizer=None,
            train=False,
            desc=f"P1 Test {epoch}/{args.n_epochs_linear}",
        )
        print(
            f"Epoch {epoch:02d} | "
            f"Train acc={tr_acc:.4f} macroF1={tr_macro:.4f} | "
            f"Test acc={te_acc:.4f} macroF1={te_macro:.4f}"
        )

    torch.save(
        model.state_dict(),
        os.path.join(args.output_dir, "final_linear_model3.pt"),
    )

    # ---- Phase 2: full fine-tuning ----
    att_test = None
    if args.do_finetune:
        print("\n🔥 PHASE 2: Fine-tuning")

        if args.do_grn:
            print("\n🧬 Building GRN (once) from representative train slide...")
            adata = sc.read_h5ad(args.h5ad_path)
            adata = harmonise_adata(
                adata,
                args.vocab_path,
                fine_tune=True,
                species=args.species,
            )
            gene_graphs, gene_names, edge_attr_names = build_hetero_grn(
                adata=adata,
                species=args.species,
                spearman_threshold=args.spearman_threshold,
                add_reverse_for_curated=True,
            )
            model.grn_layer = GRNRefinementLayer(
                dim_model=128,
                heads=args.grn_heads,
            ).to(device)
            model.gene_graphs = gene_graphs
            model.grn_expr_decoder = nn.Linear(128, 1).to(device)
            print("✅ GRN attached:", model.gene_graphs["global"].edge_index.shape)

        for p in model.parameters():
            p.requires_grad = True

        lr_grn = args.lr_backbone
        groups = [
            {"params": model.gene_embedder.parameters(), "lr": args.lr_backbone},
            {"params": model.gene_linear.parameters(), "lr": args.lr_backbone},
            {"params": model.model_lvl0.parameters(), "lr": args.lr_backbone},
            {"params": model.model_lvl1.parameters(), "lr": args.lr_backbone},
            {"params": model.pooler_lvl2.parameters(), "lr": args.lr_backbone},
            {"params": model.model_lvl2.parameters(), "lr": args.lr_backbone},
            {"params": model.classifier_head.parameters(), "lr": args.lr_head},
        ]

        if args.do_grn:
            groups.append({"params": model.grn_layer.parameters(), "lr": lr_grn})
            groups.append({"params": model.grn_expr_decoder.parameters(), "lr": lr_grn})

        optimizer = optim.Adam(groups)

        for epoch in range(1, args.n_epochs_finetune + 1):
            if args.do_grn:
                tr_loss, tr_acc, tr_micro, tr_macro, _, _, att_train = run_one_epoch(
                    model=model,
                    files=train_files,
                    target_key=args.target_key,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    criterion=criterion,
                    do_f=True,
                    do_grn=True,
                    optimizer=optimizer,
                    train=True,
                    desc=f"P2 Train {epoch}/{args.n_epochs_finetune}",
                )
            else:
                tr_loss, tr_acc, tr_micro, tr_macro, _, _ = run_one_epoch(
                    model=model,
                    files=train_files,
                    target_key=args.target_key,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    criterion=criterion,
                    do_f=True,
                    do_grn=False,
                    optimizer=optimizer,
                    train=True,
                    desc=f"P2 Train {epoch}/{args.n_epochs_finetune}",
                )

            print(
                f"✅ P2 Ep {epoch} | "
                f"loss={tr_loss:.4f} acc={tr_acc:.4f} macroF1={tr_macro:.4f}"
            )

        if args.do_grn:
            te_loss, te_acc, te_micro, te_macro, _, _, att_test = run_one_epoch(
                model=model,
                files=test_files,
                target_key=args.target_key,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                criterion=criterion,
                do_f=True,
                do_grn=True,
                optimizer=None,
                train=False,
                desc="P2 Final Test",
            )
        else:
            te_loss, te_acc, te_micro, te_macro, _, _ = run_one_epoch(
                model=model,
                files=test_files,
                target_key=args.target_key,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                criterion=criterion,
                do_f=True,
                do_grn=False,
                optimizer=None,
                train=False,
                desc="P2 Final Test",
            )
            att_test = None

        print(
            f"🎯 P2 Final Test | "
            f"loss={te_loss:.4f} acc={te_acc:.4f} macroF1={te_macro:.4f}"
        )

        torch.save(
            model.state_dict(),
            os.path.join(args.output_dir, "final_finetuned_model_livercancer.pt"),
        )

    print("\n✅ Done.")
    return att_test


if __name__ == "__main__":
    import sys

    if any(k.lower().startswith(("ipykernel", "spyder")) for k in sys.modules.keys()):
        print("🖥️ Running in IDE mode... overriding sys.argv")

        sys.argv = [
            "main_train_niche_grn.py",
            "--foundation_ckpt", "./hebi_foundation_best3.pt",
            "--data_root", "/DATA1/Roxana/Hierarchy/Two_levels/fine_tune/Liver_cosmax_cancer",
            "--output_dir", "./checkpoints",
            "--target_key", "y_niche",
            "--n_classes", "4",
            "--device", "cuda:0",
            "--h5ad_path", "/DATA1/Roxana/Data/cancer_liver_cosmax.h5ad",
            "--vocab_path", "/DATA1/Roxana/Hierarchy/Two_levels/default_gene_vocab.json",
            # "--do_grn",
            "--spearman_threshold", "0.4",
            "--grn_max_cells", "3000",
            "--grn_heads", "4",
            "--species", "Human",
            "--n_epochs_linear", "10",
            "--lr_head", "1e-3",
            "--seed", "1",
            "--do_finetune",
            "--n_epochs_finetune", "1",
            "--lr_backbone", "1e-4",
        ]

    att_test = main()