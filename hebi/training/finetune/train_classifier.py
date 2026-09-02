#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised niche-type / cell-type prediction on top of a pretrained HEBI
backbone.

Two phases, following the Methods:
  1. linear probing  — backbone frozen, only the classification head trains
  2. fine-tuning     — backbone unfrozen at a much lower learning rate

Data are split at the section level: whole sections are held out, so the
reported numbers reflect generalisation to unseen sections rather than
memorisation.

Metrics are accuracy plus macro- and micro-averaged F1. Micro-F1 equals
accuracy in the single-label setting and is reported for completeness.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

from hebi.training.finetune.finetuner import HebiCellTypeClassifier
from hebi.training.finetune.loader import (
    build_local_sampler,
    get_slide_list,
    iterate_supervised_batches,
    load_supervised_slide_dict,
)


def _forward(model, batch, device):
    """Unpack one batch from iterate_supervised_batches and run the model.

    The loader yields eight ready-made tensors; nothing is re-derived here.
    edge_attr_b must reach the model, or the Level-0 TransformerConv
    asserts on its missing edge features.
    """
    (x_subgraph_raw, edge_index_subgraph, edge_attr_b, node2comm_batch,
     super_sp_global, gene_ids, y_seed, bs) = batch

    logits = model(
        x_subgraph_raw=x_subgraph_raw,
        edge_index_subgraph=edge_index_subgraph,
        edge_attr_b=edge_attr_b,
        node2comm_batch=node2comm_batch,
        super_sp_global=super_sp_global,
        gene_ids=gene_ids,
        device=device,
        bs=bs,
    )
    return logits, y_seed, bs


def run_one_epoch(model, files, target_key, optimizer, criterion, device,
                  batch_size, desc, num_workers=4):
    model.train()
    total_loss = 0.0
    total_samples = 0
    y_true_all, y_pred_all = [], []

    files = list(files)
    random.shuffle(files)
    pbar = tqdm(files, desc=desc)

    for pt_path in pbar:
        try:
            slide = load_supervised_slide_dict(pt_path, target_key=target_key)
        except Exception as exc:
            print(f"Skipping {pt_path}: {exc}")
            continue

        sampler = build_local_sampler(
            slide, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        for batch in iterate_supervised_batches(
            slide, sampler, device, target_key=target_key
        ):
            optimizer.zero_grad(set_to_none=True)

            logits, y_seed, bs = _forward(model, batch, device)
            loss = criterion(logits, y_seed)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bs
            total_samples += bs
            y_true_all.append(y_seed.detach().cpu().numpy())
            y_pred_all.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

        if total_samples > 0:
            acc = float(
                (np.concatenate(y_pred_all) == np.concatenate(y_true_all)).mean()
            )
            pbar.set_postfix(loss=total_loss / total_samples, acc=acc)

    if total_samples == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "loss": 0.0}

    metrics = _metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
    metrics["loss"] = total_loss / total_samples
    return metrics


@torch.no_grad()
def evaluate(model, test_files, target_key, device, batch_size, num_workers=4):
    model.eval()
    y_true_all, y_pred_all = [], []

    for pt_path in test_files:
        try:
            slide = load_supervised_slide_dict(pt_path, target_key=target_key)
        except Exception as exc:
            print(f"Skipping {pt_path}: {exc}")
            continue

        sampler = build_local_sampler(
            slide, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        for batch in iterate_supervised_batches(
            slide, sampler, device, target_key=target_key
        ):
            logits, y_seed, _ = _forward(model, batch, device)
            y_true_all.append(y_seed.cpu().numpy())
            y_pred_all.append(torch.argmax(logits, dim=1).cpu().numpy())

    if not y_true_all:
        return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

    return _metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))


def _metrics(y_true, y_pred):
    """Accuracy, macro-F1, micro-F1.

    Macro-F1 averages per-class F1 unweighted, so every niche type counts
    equally regardless of size — the number that matters when classes are
    imbalanced. Micro-F1 equals accuracy for single-label prediction.
    """
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro",
                                   zero_division=0)),
    }


def _fmt(m):
    return (f"acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  "
            f"microF1={m['micro_f1']:.4f}")


def _infer_prototype_counts(checkpoint_path, device):
    """Read prototype counts from the checkpoint instead of hardcoding them,
    so a 256-prototype checkpoint loads as readily as a 1024-prototype one."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    n_l1 = ckpt["model_lvl1"]["prototypes"].shape[0]
    try:
        n_l2 = ckpt["pooler_lvl2"]["l2_prototypes"].shape[0]
    except (KeyError, IndexError):
        n_l2 = 20
    del ckpt
    return int(n_l1), int(n_l2)


def train_classifier(
    foundation_ckpt,
    data_root,
    output_dir,
    target_key="y_niche",
    n_epochs_linear=10,
    lr_linear=1e-3,
    do_finetune=False,
    n_epochs_finetune=10,
    lr_finetune=1e-5,
    batch_size=512,
    device="cuda:0",
    num_workers=4,
    seed=1990,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(data_root, "label_mappings.json")) as f:
        mappings = json.load(f)

    mapping_key = "class_mapping" if target_key == "y_class" else "niche_mapping"
    if mapping_key not in mappings:
        raise KeyError(
            f"'{mapping_key}' not in label_mappings.json. "
            f"Available: {list(mappings)}"
        )
    n_classes = len(mappings[mapping_key])
    print(f"Task: {target_key} ({n_classes} classes)")

    n_l1, n_l2 = _infer_prototype_counts(foundation_ckpt, device)
    print(f"Prototypes from checkpoint: L1={n_l1}, L2={n_l2}")

    model = HebiCellTypeClassifier(
        n_classes=n_classes,
        dim_model=128,
        l0_encoder_type="transformer",
        global_num_layers=2,
        num_prototypes_lvl1=n_l1,
        num_prototypes_lvl2=n_l2,
    ).to(device)

    model.load_pretrained_weights(foundation_ckpt, device, freeze_backbone=True)
    criterion = nn.CrossEntropyLoss()

    train_files = get_slide_list(data_root, split="train")
    test_files = get_slide_list(data_root, split="test")

    best_macro_f1 = 0.0
    history = []

    # ------------------------------------------------------------------
    # Phase 1: linear probing, backbone frozen
    # ------------------------------------------------------------------
    print(f"\nPhase 1: linear probing ({n_epochs_linear} epochs)")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_linear
    )

    for epoch in range(1, n_epochs_linear + 1):
        train_m = run_one_epoch(
            model=model, files=train_files, target_key=target_key,
            optimizer=optimizer, criterion=criterion, device=device,
            batch_size=batch_size, desc=f"P1 {epoch}/{n_epochs_linear}",
            num_workers=num_workers,
        )
        test_m = evaluate(
            model=model, test_files=test_files, target_key=target_key,
            device=device, batch_size=batch_size, num_workers=num_workers,
        )
        print(f"  P1 ep {epoch} | train {_fmt(train_m)}")
        print(f"           | test  {_fmt(test_m)}")
        history.append({"phase": 1, "epoch": epoch,
                        "train": train_m, "test": test_m})

        # Selection on macro-F1, not accuracy: with imbalanced niche types
        # accuracy is dominated by the largest class.
        if test_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = test_m["macro_f1"]
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_probe_model.pt"))
            print(f"           | saved (macroF1 {best_macro_f1:.4f})")

    # ------------------------------------------------------------------
    # Phase 2: fine-tuning, backbone unfrozen at a lower learning rate
    # ------------------------------------------------------------------
    if do_finetune:
        print(f"\nPhase 2: fine-tuning ({n_epochs_finetune} epochs, "
              f"lr={lr_finetune})")

        for p in model.parameters():
            p.requires_grad = True

        optimizer = optim.Adam([
            {"params": model.model_lvl0.parameters(), "lr": lr_finetune},
            {"params": model.model_lvl1.parameters(), "lr": lr_finetune},
            {"params": model.pooler_lvl2.parameters(), "lr": lr_finetune},
            {"params": model.model_lvl2.parameters(), "lr": lr_finetune},
            {"params": model.gene_embedder.parameters(), "lr": lr_finetune},
            {"params": model.gene_linear.parameters(), "lr": lr_finetune},
            {"params": model.classifier_head.parameters(), "lr": lr_linear},
        ])

        for epoch in range(1, n_epochs_finetune + 1):
            train_m = run_one_epoch(
                model=model, files=train_files, target_key=target_key,
                optimizer=optimizer, criterion=criterion, device=device,
                batch_size=batch_size, desc=f"P2 {epoch}/{n_epochs_finetune}",
                num_workers=num_workers,
            )
            test_m = evaluate(
                model=model, test_files=test_files, target_key=target_key,
                device=device, batch_size=batch_size, num_workers=num_workers,
            )
            print(f"  P2 ep {epoch} | train {_fmt(train_m)}")
            print(f"           | test  {_fmt(test_m)}")
            history.append({"phase": 2, "epoch": epoch,
                            "train": train_m, "test": test_m})

            if test_m["macro_f1"] > best_macro_f1:
                best_macro_f1 = test_m["macro_f1"]
                torch.save(model.state_dict(),
                           os.path.join(output_dir, "best_finetuned_model.pt"))
                print(f"           | saved (macroF1 {best_macro_f1:.4f})")
    else:
        print("\nSkipping phase 2.")

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump({"target_key": target_key, "n_classes": n_classes,
                   "seed": seed, "history": history}, f, indent=2)

    print(f"\nBest macro-F1: {best_macro_f1:.4f}")
    print(f"Outputs in {output_dir}")
    return history


if __name__ == "__main__":
    train_classifier(
        foundation_ckpt="/DATA1/Roxana/Hierarchy/Two_levels/hebi_foundation_best_new.pt",
        data_root="/DATA1/Roxana/Hierarchy/Two_levels/fine_tune",
        output_dir="./results/niche_prediction_liver_cosmx",
        target_key="y_niche",
        n_epochs_linear=10,
        lr_linear=1e-3,
        do_finetune=False,
        n_epochs_finetune=10,
        lr_finetune=1e-5,
        batch_size=512,
        device="cuda:0",
        num_workers=4,
    )
