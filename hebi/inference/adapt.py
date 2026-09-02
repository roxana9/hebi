#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continual adaptation of a pretrained HEBI model to new sections.

Training continues for a small number of epochs on the target section with
exactly the self-supervised objective used in pretraining. This is the only
part of inference that differs between the zero-shot and continual regimes:
both share the same encoding procedure afterwards.
"""

import torch

from hebi.training.pretrain import Train
from hebi.model.grn_layer import GRNRefinementLayer

EMBED_DIM = 128


def continual_adapt(
    checkpoint_path: str,
    data_dir: str,
    vocab_path: str,
    device: str = "cuda:0",
    lr: float = 1e-5,
    num_epochs: int = 10,
    output_checkpoint: str = "hebi_continual.pt",
    gene_graph=None,
    gene_graph_list=None,
    init: str = "pretrained",
):
    """Adapt a checkpoint to the sections in data_dir.

    init:
        "pretrained"  load foundation weights, then re-initialise the
                      niche-level modules (pooler and Level-2 transformer),
                      re-initialise the niche prototypes orthogonally, and
                      re-derive each cell's community assignment from the
                      similarity between its projected embedding and the
                      Level-1 prototypes.
        "random"      all weights start from scratch (from-scratch baseline
                      sharing the architecture but not the pretrained
                      knowledge).
        "hybrid"      handled upstream by load_hebi.
    """
    device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    n_proto_l1 = ckpt["model_lvl1"]["prototypes"].shape[0]
    try:
        n_proto_l2 = ckpt["pooler_lvl2"]["l2_prototypes"].shape[0]
    except (KeyError, IndexError):
        n_proto_l2 = 20
    del ckpt

    trainer = Train(
        data_dir=data_dir,
        vocab_path=vocab_path,
        l0_encoder_type="transformer",
        batch_size=2048,
        num_workers=4,
        lr=lr,
        num_epochs=num_epochs,
        global_num_layers=2,
        num_prototypes_lvl1=n_proto_l1,
        num_prototypes_lvl2=n_proto_l2,
        decoder_hidden_dim=256,
        early_stopping_patience=num_epochs + 1,  # disable early stopping
        checkpoint_path=checkpoint_path,
        device=device,
    )

    trainer.discover_data_files()
    trainer.data_path = trainer.data_files[0]
    trainer.load_data()
    trainer.build_models(trainer.X.size(0), trainer.X.shape)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    trainer.gene_embedder.load_state_dict(state["gene_embedder"])
    trainer.gene_linear.load_state_dict(state["gene_linear"])
    trainer.model_lvl0.load_state_dict(state["model_lvl0"])
    trainer.model_lvl1.load_state_dict(state["model_lvl1"])
    trainer.pooler_lvl2.load_state_dict(state["pooler_lvl2"])
    trainer.model_lvl2.load_state_dict(state["model_lvl2"])
    trainer.global_decoder.load_state_dict(state["global_decoder"])

    if init == "pretrained":
        _reinit_niche_level(trainer)

    do_grn = gene_graph is not None or gene_graph_list is not None
    if do_grn:
        print("Attaching GRN refinement layer")
        trainer.grn_layer = GRNRefinementLayer(dim_model=EMBED_DIM, heads=4).to(
            trainer.device
        )
        trainer.grn_expr_decoder = torch.nn.Linear(trainer.embedding_size, 1).to(
            trainer.device
        )
        trainer.gene_graphs = gene_graph
        trainer.gene_graphs_list = gene_graph_list

    modules = [
        trainer.gene_embedder,
        trainer.gene_linear,
        trainer.model_lvl0,
        trainer.model_lvl1,
        trainer.pooler_lvl2,
        trainer.model_lvl2,
        trainer.global_decoder,
    ]
    if do_grn:
        modules += [trainer.grn_layer, trainer.grn_expr_decoder]

    for module in modules:
        for param in module.parameters():
            param.requires_grad = True

    # Fresh optimiser: momentum from pretraining is not carried over.
    base_params = []
    for module in modules[:7]:
        base_params += list(module.parameters())

    trainer.optimizer = torch.optim.Adam(
        base_params, lr=lr, weight_decay=trainer.weight_decay
    )
    print("Fresh optimiser state for continual adaptation")

    if do_grn:
        grn_params = list(trainer.grn_layer.parameters()) + list(
            trainer.grn_expr_decoder.parameters()
        )
        trainer.optimizer.add_param_group(
            {
                "params": [p for p in grn_params if p.requires_grad],
                "lr": lr,
                "weight_decay": 0.0,
            }
        )
        print(f"GRN parameter group lr={lr:.2e}")

    print(f"\nContinual adaptation over sections in: {data_dir}")
    trainer.best_loss = float("inf")
    trainer.epochs_no_improve = 0
    last_loss = None

    for epoch in range(1, num_epochs + 1):
        print(f"\nContinual epoch {epoch}/{num_epochs}")
        last_loss = trainer.train_epoch(epoch)

    final_loss = last_loss if last_loss is not None else 0.0
    print(f"\nContinual adaptation finished. Final loss: {final_loss:.4f}")

    if output_checkpoint is not None:
        trainer.checkpoint_path = str(output_checkpoint)
        trainer.save_checkpoint(epoch=num_epochs, avg_loss=final_loss)

    return trainer


def _reinit_niche_level(trainer):
    """Re-initialise the niche-level modules and prototypes, then re-derive
    community assignments. Corresponds to the pretrained initialisation mode
    described in the Methods."""

    def reinit(module):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    print("Re-initialising niche-level modules")
    reinit(trainer.pooler_lvl2)
    reinit(trainer.model_lvl2)

    with torch.no_grad():
        trainer.model_lvl2.temperature.fill_(1.0)
        torch.nn.init.normal_(
            trainer.model_lvl2.spatial_bias_encoder.weight, mean=0.0, std=0.01
        )

    print("Orthogonal init for niche prototypes")
    with torch.no_grad():
        K = trainer.num_prototypes_lvl2
        rand_matrix = torch.randn(EMBED_DIM, K, device=trainer.device)
        Q, _ = torch.linalg.qr(rand_matrix)
        trainer.pooler_lvl2.l2_prototypes.data.copy_(Q.T)

    print("Re-deriving community assignments")
    with torch.no_grad():
        gene_emb = trainer.gene_embedder(trainer.gene_ids)
        gene_emb = trainer.gene_linear(gene_emb)
        gene_emb = torch.nn.functional.normalize(gene_emb, dim=1, p=2)

        X_proj = torch.nn.functional.normalize(trainer.X @ gene_emb, dim=1)
        protos = torch.nn.functional.normalize(trainer.model_lvl1.prototypes, dim=1)

        trainer.node2comm_fixed = (X_proj @ protos.T).argmax(dim=1)

        N = trainer.node2comm_fixed.shape[0]
        K = trainer.num_prototypes_lvl1
        super_sp = torch.zeros(N, K, device=trainer.device)
        super_sp.scatter_(1, trainer.node2comm_fixed.unsqueeze(1), 1.0)
        trainer.super_sp_fixed = super_sp

    used = trainer.node2comm_fixed.unique().shape[0]
    print(f"Communities re-aligned: {used}/{K} prototypes used")


def create_random_checkpoint(
    data_dir: str,
    vocab_path: str,
    output_checkpoint: str,
    device: str = "cuda:0",
    n_prototypes_l1: int = 1024,
    n_prototypes_l2: int = 20,
    seed: int = 42,
):
    """Write a randomly initialised checkpoint with the same structure as a
    pretrained one, so the from-scratch baseline runs through the same code
    path as a real checkpoint."""
    print("\nCreating randomly initialised checkpoint")

    trainer = Train(
        data_dir=data_dir,
        vocab_path=vocab_path,
        l0_encoder_type="transformer",
        batch_size=2048,
        num_workers=4,
        lr=1e-4,
        num_epochs=1,
        global_num_layers=2,
        num_prototypes_lvl1=n_prototypes_l1,
        num_prototypes_lvl2=n_prototypes_l2,
        decoder_hidden_dim=256,
        early_stopping_patience=10,
        checkpoint_path=None,
        device=device,
        seed=seed,
    )

    trainer.discover_data_files()
    trainer.data_path = trainer.data_files[0]
    trainer.load_data()

    n_l1 = int(trainer.node2comm_fixed.max().item()) + 1
    trainer.num_prototypes_lvl1 = n_l1
    print(f"Level-1 prototypes from data: {n_l1}")

    trainer.build_models(trainer.X.size(0), trainer.X.shape)

    state = {
        "epoch": 0,
        "seed": seed,
        "gene_embedder": trainer.gene_embedder.state_dict(),
        "gene_linear": trainer.gene_linear.state_dict(),
        "model_lvl0": trainer.model_lvl0.state_dict(),
        "model_lvl1": trainer.model_lvl1.state_dict(),
        "pooler_lvl2": trainer.pooler_lvl2.state_dict(),
        "model_lvl2": trainer.model_lvl2.state_dict(),
        "global_decoder": trainer.global_decoder.state_dict(),
    }
    torch.save(state, output_checkpoint)
    print(f"Random checkpoint saved to: {output_checkpoint}")
    return output_checkpoint
