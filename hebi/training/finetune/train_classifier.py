#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised niche-type / cell-type prediction on top of a pretrained HEBI
backbone.

level="cell"  : linear head on z0 (labels determined by cell identity)
level="niche" : Level-2 prototype attention -> prototype->label map
                (spatially defined niches)

Two phases, following the Methods:
  1. linear probing  — backbone frozen, only the head trains
  2. fine-tuning     — backbone unfrozen at a much lower learning rate

Niche level: whole-section community representations (global_reps) are used
in training and evaluation alike. They are computed once for the probe
(backbone frozen) and recomputed at the start of every fine-tuning epoch,
detached (stop-gradient through the community means).

Metrics are accuracy plus macro- and micro-averaged F1. Model selection is
on test macro-F1, as in the previous version.
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

from hebi.training.finetune.finetuner import HebiClassifier
from hebi.training.finetune.loader import (
    build_local_sampler,
    get_slide_list,
    iterate_supervised_batches,
    load_supervised_slide_dict,
)

# a prototype whose share of the training attention mass is below this is
# reported as not estimable (its row of the map barely received gradient)
MIN_PROTO_MASS_FRAC = 1e-3


def _forward(model, batch, device, global_reps=None, return_attention=False):
    (x_subgraph_raw, edge_index_subgraph, edge_attr_b, node2comm_batch,
     super_sp_global, gene_ids, y_seed, bs) = batch

    out = model(
        x_subgraph_raw=x_subgraph_raw,
        edge_index_subgraph=edge_index_subgraph,
        edge_attr_b=edge_attr_b,
        node2comm_batch=node2comm_batch,
        super_sp_global=super_sp_global,
        gene_ids=gene_ids,
        device=device,
        bs=bs,
        global_reps=global_reps,
        return_attention=return_attention,
    )
    return out, y_seed, bs


@torch.no_grad()
def build_global_cache(model, files, target_key, device, batch_size,
                       num_workers=4):
    """Whole-section community representations for every file, computed with
    the current weights in eval mode and detached. Empty for level='cell'."""
    if model.level != "niche":
        return {}
    was_training = model.training
    model.eval()
    cache = {}
    for pt_path in tqdm(files, desc="global_reps"):
        slide = load_supervised_slide_dict(pt_path, target_key=target_key)
        reps, count = model.section_global_reps(
            slide, device, batch_size=batch_size, num_workers=num_workers)
        cache[str(pt_path)] = (reps.detach(), count.detach())
    model.train(was_training)
    return cache


def run_one_epoch(model, files, target_key, optimizer, criterion, device,
                  batch_size, desc, num_workers=4, global_cache=None):
    """Returns metrics; for level='niche' also the per-prototype attention
    mass and hard-assignment counts over the training cells."""
    model.train()                 # see HebiClassifier.train for module modes
    niche = model.level == "niche"
    total_loss, total_samples = 0.0, 0
    y_true_all, y_pred_all = [], []
    proto_mass = torch.zeros(model.k2, dtype=torch.float64) if niche else None
    proto_hard = torch.zeros(model.k2, dtype=torch.long) if niche else None

    files = list(files)
    random.shuffle(files)
    pbar = tqdm(files, desc=desc)

    for pt_path in pbar:
        try:
            slide = load_supervised_slide_dict(pt_path, target_key=target_key)
        except Exception as exc:
            print(f"Skipping {pt_path}: {exc}")
            continue

        global_reps = global_cache[str(pt_path)] if niche else None

        sampler = build_local_sampler(
            slide, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        for batch in iterate_supervised_batches(
            slide, sampler, device, target_key=target_key
        ):
            optimizer.zero_grad(set_to_none=True)
            out, y_seed, bs = _forward(model, batch, device,
                                       global_reps=global_reps,
                                       return_attention=niche)
            if niche:
                logits, attn = out
                a = attn.detach()
                proto_mass += a.sum(0).double().cpu()
                proto_hard += torch.bincount(a.argmax(1),
                                             minlength=model.k2).cpu()
            else:
                logits = out

            loss = criterion(logits, y_seed)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bs
            total_samples += bs
            y_true_all.append(y_seed.detach().cpu().numpy())
            y_pred_all.append(logits.argmax(dim=1).detach().cpu().numpy())

        if total_samples > 0:
            acc = float((np.concatenate(y_pred_all)
                         == np.concatenate(y_true_all)).mean())
            pbar.set_postfix(loss=total_loss / total_samples, acc=acc)

    if total_samples == 0:
        m = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "loss": 0.0}
    else:
        m = _metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
        m["loss"] = total_loss / total_samples

    usage = None
    if niche:
        usage = {"mass": proto_mass.numpy(), "hard": proto_hard.numpy(),
                 "n_cells": total_samples}
    return m, usage


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

        global_reps = None
        if model.level == "niche":
            global_reps = model.section_global_reps(
                slide, device, batch_size=batch_size, num_workers=num_workers)

        sampler = build_local_sampler(
            slide, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        for batch in iterate_supervised_batches(
            slide, sampler, device, target_key=target_key
        ):
            logits, y_seed, _ = _forward(model, batch, device,
                                         global_reps=global_reps)
            y_true_all.append(y_seed.cpu().numpy())
            y_pred_all.append(logits.argmax(dim=1).cpu().numpy())

    if not y_true_all:
        return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}
    return _metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))


def _metrics(y_true, y_pred):
    """Macro-F1 averages per-class F1 unweighted, so every niche type counts
    equally regardless of size. Micro-F1 equals accuracy for single-label
    prediction."""
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


def _infer_k2(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    k2 = int(ckpt["pooler_lvl2"]["l2_prototypes"].shape[0])
    del ckpt
    return k2


def _save_checkpoint(model, label_names, output_dir, tag, usage=None):
    torch.save(model.state_dict(),
               os.path.join(output_dir, f"best_{tag}_model.pt"))
    if model.level != "niche":
        return

    M = model.prototype_label_map().detach().cpu().numpy()
    n = max(usage["n_cells"], 1)
    mass_frac = usage["mass"] / n
    estimable = mass_frac >= MIN_PROTO_MASS_FRAC
    row_entropy = -(M * np.log(M + 1e-12)).sum(1)

    prototypes = []
    for k in range(M.shape[0]):
        prototypes.append({
            "prototype": k,
            "label": label_names[int(M[k].argmax())] if estimable[k] else None,
            "estimable": bool(estimable[k]),
            "attention_mass_frac": float(mass_frac[k]),
            "n_hard_assigned": int(usage["hard"][k]),
            "row_entropy": float(row_entropy[k]),
            "label_distribution": {l: float(M[k, j])
                                   for j, l in enumerate(label_names)},
        })
    with open(os.path.join(output_dir,
                           f"prototype_label_map_{tag}.json"), "w") as f:
        json.dump({"labels": label_names,
                   "min_mass_frac": MIN_PROTO_MASS_FRAC,
                   "prototypes": prototypes}, f, indent=2)


def train_classifier(
    foundation_ckpt,
    data_root,
    output_dir,
    level="cell",
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
        raise KeyError(f"'{mapping_key}' not in label_mappings.json. "
                       f"Available: {list(mappings)}")
    label_to_id = mappings[mapping_key]
    label_names = [k for k, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])]
    n_classes = len(label_names)
    k2 = _infer_k2(foundation_ckpt)
    print(f"Task: {target_key} ({n_classes} classes) | level: {level} | "
          f"niche prototypes: {k2}")

    model = HebiClassifier(n_classes=n_classes, level=level,
                           num_prototypes_lvl2=k2).to(device)
    model.load_pretrained_weights(foundation_ckpt, device, freeze_backbone=True)
    # cell: raw logits; niche: log-probabilities from the mixture
    criterion = nn.CrossEntropyLoss() if level == "cell" else nn.NLLLoss()

    train_files = get_slide_list(data_root, split="train")
    test_files = get_slide_list(data_root, split="test")

    best_macro_f1 = 0.0
    history = []

    # ------------------------------------------------------------------
    # Phase 1: linear probing, backbone frozen
    # ------------------------------------------------------------------
    print(f"\nPhase 1: linear probing ({n_epochs_linear} epochs)")
    optimizer = optim.Adam(model.head_parameters(), lr=lr_linear)
    # backbone frozen -> whole-section representations fixed for the phase
    global_cache = build_global_cache(model, train_files, target_key, device,
                                      batch_size, num_workers)

    for epoch in range(1, n_epochs_linear + 1):
        train_m, usage = run_one_epoch(
            model, train_files, target_key, optimizer, criterion, device,
            batch_size, f"P1 {epoch}/{n_epochs_linear}", num_workers,
            global_cache=global_cache)
        test_m = evaluate(model, test_files, target_key, device,
                          batch_size, num_workers)
        print(f"  P1 ep {epoch} | train {_fmt(train_m)}")
        print(f"           | test  {_fmt(test_m)}")
        history.append({"phase": 1, "epoch": epoch,
                        "train": train_m, "test": test_m})

        if test_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = test_m["macro_f1"]
            _save_checkpoint(model, label_names, output_dir, "probe", usage)
            print(f"           | saved (macroF1 {best_macro_f1:.4f})")

    # ------------------------------------------------------------------
    # Phase 2: fine-tuning, backbone unfrozen at a lower learning rate
    # ------------------------------------------------------------------
    if do_finetune:
        print(f"\nPhase 2: fine-tuning ({n_epochs_finetune} epochs, "
              f"lr={lr_finetune})")
        model.set_backbone_frozen(False)
        optimizer = optim.Adam(
            [{"params": getattr(model, k).parameters(), "lr": lr_finetune}
             for k in model.used_backbone_keys]
            + [{"params": model.head_parameters(), "lr": lr_linear}]
        )

        for epoch in range(1, n_epochs_finetune + 1):
            # weights change -> recompute whole-section representations
            global_cache = build_global_cache(model, train_files, target_key,
                                              device, batch_size, num_workers)
            train_m, usage = run_one_epoch(
                model, train_files, target_key, optimizer, criterion, device,
                batch_size, f"P2 {epoch}/{n_epochs_finetune}", num_workers,
                global_cache=global_cache)
            test_m = evaluate(model, test_files, target_key, device,
                              batch_size, num_workers)
            print(f"  P2 ep {epoch} | train {_fmt(train_m)}")
            print(f"           | test  {_fmt(test_m)}")
            history.append({"phase": 2, "epoch": epoch,
                            "train": train_m, "test": test_m})

            if test_m["macro_f1"] > best_macro_f1:
                best_macro_f1 = test_m["macro_f1"]
                _save_checkpoint(model, label_names, output_dir, "finetuned",
                                 usage)
                print(f"           | saved (macroF1 {best_macro_f1:.4f})")
    else:
        print("\nSkipping phase 2.")

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump({"target_key": target_key, "level": level,
                   "n_classes": n_classes, "k2": k2, "seed": seed,
                   "history": history}, f, indent=2)

    print(f"\nBest macro-F1: {best_macro_f1:.4f}")
    print(f"Outputs in {output_dir}")
    return history
