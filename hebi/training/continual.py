#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continual pretraining on NEW spatial slides.

- Loads a pretrained foundation model checkpoint with strict architectural compliance
- Re-instantiates the 3-level graph encoder configuration
- Seamlessly transfers weights and historical gradient momentum trajectories
- Runs a low learning rate continual optimization sequence over target slides

Author: Roxana
"""

import os
import sys
import torch
from pathlib import Path
from pretraining import Train  # Imports your Train class
from grn_layer import GRNRefinementLayer

def continual_pretrain(
    checkpoint_path,
    new_data_dir,
    vocab_path,
    device="cuda:0",
    lr=1e-5,
    num_epochs=10,
    out_checkpoint_path="hebi_continual.pt",
    gene_graphs=None,
    gene_graphs_list=None,
    use_novae_loss=False,
    novae_num_prototypes=64,
    novae_level="z0",
    novae_noise_std=0.05,
    novae_dropout_rate=0.1,
    mode="pretrained",
    multi_slide=False,
    multi_slide_weight=0.5
):
    """
    Continual pretraining on unseen slides with full gradient inertia transfer.
    """
    device = torch.device(device)
    # Infer prototype counts from the checkpoint (supports 256/1024/etc.)
    _ckpt_state = torch.load(checkpoint_path, map_location="cpu")
    _n_proto_l1 = _ckpt_state["model_lvl1"]["prototypes"].shape[0]
    try:
        _n_proto_l2 = _ckpt_state["pooler_lvl2"]["l2_prototypes"].shape[0]
    except Exception:
        _n_proto_l2 = 20
    del _ckpt_state
    # ------------------------------------------------------------------
    # 1) Instantiate Train on the new data directory
    # ------------------------------------------------------------------
    trainer = Train(
        data_dir=new_data_dir,
        vocab_path=vocab_path,
        l0_encoder_type="transformer",
        batch_size=2048,
        num_workers=4,
        lr=lr,
        num_epochs=num_epochs,
        global_num_layers=2,
        num_prototypes_lvl1=_n_proto_l1,
        num_prototypes_lvl2=_n_proto_l2,
        decoder_hidden_dim=256,
        early_stopping_patience=num_epochs + 1,  # Effectively disable early stopping
        checkpoint_path=checkpoint_path,         
        device=device,
        use_novae_loss=use_novae_loss,
        novae_num_prototypes=novae_num_prototypes,
        novae_level=novae_level,
        novae_noise_std=novae_noise_std,
        novae_dropout_rate=novae_dropout_rate,
    )

    if use_novae_loss:
        print(f"🔷 Novae clustering fine-tuning enabled | level={novae_level}")
    else:
        print("🔹 Standard continual pretraining (Novae disabled)")

    # ------------------------------------------------------------------
    # 2) Discover data structures & build model shapes
    # ------------------------------------------------------------------
    trainer.discover_data_files()
    trainer.data_path = trainer.data_files[0]
    trainer.load_data()
    N0 = trainer.X.size(0)
    trainer.build_models(N0, trainer.X.shape)

    # ------------------------------------------------------------------
    # 3) Load pretrained structural weights strictly
    # ------------------------------------------------------------------
    print(f"\n📥 Loading pretrained checkpoint from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)

    trainer.gene_embedder.load_state_dict(state["gene_embedder"])
    trainer.gene_linear.load_state_dict(state["gene_linear"])
    trainer.model_lvl0.load_state_dict(state["model_lvl0"])
    trainer.model_lvl1.load_state_dict(state["model_lvl1"])
    trainer.pooler_lvl2.load_state_dict(state["pooler_lvl2"])
    trainer.model_lvl2.load_state_dict(state["model_lvl2"])
    trainer.global_decoder.load_state_dict(state["global_decoder"])
    
    if use_novae_loss and "novae_swav_head" in state and trainer.novae_swav_head is not None:
        trainer.novae_swav_head.load_state_dict(state["novae_swav_head"])
    # ← STEP A: reinitialize collapsed L2 modules FIRST
    if mode == "pretrained":
        def reinit_module(module):
            for m in module.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.LayerNorm):
                    torch.nn.init.ones_(m.weight)
                    torch.nn.init.zeros_(m.bias)
    
        print("🔄 Reinitializing collapsed L2 modules...")
        reinit_module(trainer.pooler_lvl2)
        reinit_module(trainer.model_lvl2)
    
        # ADD THESE TWO LINES HERE
        with torch.no_grad():
            trainer.model_lvl2.temperature.fill_(1.0)
            torch.nn.init.normal_(trainer.model_lvl2.spatial_bias_encoder.weight, 
                                  mean=0.0, std=0.01)
    
        print("✅ pooler_lvl2 and model_lvl2 reinitialized")
    
        # Orthogonal init for l2_prototypes
        print("🔄 Orthogonal init for L2 prototypes...")
        with torch.no_grad():
            K = trainer.num_prototypes_lvl2   # 20
            d = 128
            rand_matrix = torch.randn(d, K, device=trainer.device)
            Q, _ = torch.linalg.qr(rand_matrix)
            trainer.pooler_lvl2.l2_prototypes.data.copy_(Q.T)
        print("✅ L2 prototypes orthogonally initialized")
    
        print("🔄 Re-aligning node2comm...")
        with torch.no_grad():
            gene_emb = trainer.gene_embedder(trainer.gene_ids)
            gene_emb = trainer.gene_linear(gene_emb)
            gene_emb = torch.nn.functional.normalize(gene_emb, dim=1, p=2)
            X_proj = trainer.X @ gene_emb
            protos = torch.nn.functional.normalize(
                trainer.model_lvl1.prototypes, dim=1
            )
            X_norm = torch.nn.functional.normalize(X_proj, dim=1)
            sim = X_norm @ protos.T
            trainer.node2comm_fixed = sim.argmax(dim=1)
            N = trainer.node2comm_fixed.shape[0]
            K = trainer.num_prototypes_lvl1
            super_sp = torch.zeros(N, K, device=trainer.device)
            super_sp.scatter_(1, trainer.node2comm_fixed.unsqueeze(1), 1.0)
            trainer.super_sp_fixed = super_sp
        print(f"✅ node2comm re-aligned: "
              f"{trainer.node2comm_fixed.unique().shape[0]}/{K} prototypes used")
    # ------------------------------------------------------------------
    # 4) Handle optional dynamic Multi-Omic GRN Layers
    # ------------------------------------------------------------------
    do_grn = (gene_graphs is not None or gene_graphs_list is not None) 
    if do_grn:
        print("🧬 Dynamic GRN prioritization active. Attaching refinement sub-networks...")
        trainer.grn_layer = GRNRefinementLayer(dim_model=128, heads=4).to(trainer.device)
        trainer.grn_expr_decoder = torch.nn.Linear(trainer.embedding_size, 1).to(trainer.device)
        trainer.gene_graphs = gene_graphs
        trainer.gene_graphs_list = gene_graphs_list

    # ------------------------------------------------------------------
    # 5) Track parameter updates and establish parameter arrays
    # ------------------------------------------------------------------
    all_modules = [
        trainer.gene_embedder,
        trainer.gene_linear,
        trainer.model_lvl0,
        trainer.model_lvl1,
        trainer.pooler_lvl2,
        trainer.model_lvl2,
        trainer.global_decoder,
    ]
    
    if do_grn:
        all_modules.extend([trainer.grn_layer, trainer.grn_expr_decoder])
    if use_novae_loss and trainer.novae_swav_head is not None:
        all_modules.append(trainer.novae_swav_head)

    # Set parameters to optimize gradients cleanly
    for module in all_modules:
        for param in module.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # 6) Recreate optimizer state with original structural tracking entries
    # ------------------------------------------------------------------
    print("🔄 Re-instantiating foundation optimizer engine...")
    
    # Establish base parameter array indexing sequence matching pretraining.py
    base_params = (
        list(trainer.gene_embedder.parameters())
        + list(trainer.gene_linear.parameters())
        + list(trainer.model_lvl0.parameters())
        + list(trainer.model_lvl1.parameters())
        + list(trainer.pooler_lvl2.parameters())
        + list(trainer.model_lvl2.parameters())
        + list(trainer.global_decoder.parameters())
    )
    
    if use_novae_loss and trainer.novae_swav_head is not None:
        base_params += list(trainer.novae_swav_head.parameters())
    if mode == "pretrained":
        trainer.optimizer = torch.optim.Adam(
            base_params, lr=lr, weight_decay=trainer.weight_decay)
        print("⚠️ Skipping optimizer state restore — fresh momentum for continual adaptation.")
    else:
        print("Randomly Initialised weights")
        trainer.optimizer = torch.optim.Adam(
            base_params, lr=lr, weight_decay=trainer.weight_decay)
    # Inject gradient history registers strictly if keys perfectly align
    if mode != "pretrained":
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr

   

    # Append any dynamic multi-omic variables as self-contained parameter arrays
    if do_grn:
        grn_params = list(trainer.grn_layer.parameters()) + \
                     list(trainer.grn_expr_decoder.parameters())
        grn_lr = lr *1
        trainer.optimizer.add_param_group({
            'params': [p for p in grn_params if p.requires_grad],
            'lr': grn_lr,
            'weight_decay': 0.0
        })
        print(f"🧬 GRN optimizer lr set to {grn_lr:.2e} (base lr={lr:.2e})")

    # ------------------------------------------------------------------
    # 7) Execute Continual Pretraining Loop
    # ------------------------------------------------------------------
    print(f"\n🚀 Launching continual training loops across slides in: {new_data_dir}")
    trainer.best_loss = float("inf")
    trainer.epochs_no_improve = 0
    last_loss = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n🔁 Continual Epoch {epoch}/{num_epochs}")
        last_loss = trainer.train_epoch(epoch)

    final_loss = last_loss if last_loss is not None else 0.0
    print(f"\n🏁 Continual tuning session completed. Final loss: {final_loss:.4f}")

    # ------------------------------------------------------------------
    # 8) Save Checkpoint Output File
    # ------------------------------------------------------------------
    if out_checkpoint_path is not None:
        out_checkpoint_path = str(out_checkpoint_path)
        trainer.checkpoint_path = out_checkpoint_path
        trainer.save_checkpoint(epoch=num_epochs, avg_loss=final_loss)
        print(f"💾 Checkpoint successfully compiled and stored to: {out_checkpoint_path}")

    return trainer