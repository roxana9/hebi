#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encoding a spatial section with a trained HEBI model.

A single forward pass without gradients produces, for every cell:
    z0  cell-level embedding (local spatial neighbourhood)
    z1  micro-node / prototype-level embedding
    z2  niche-level embedding
    prototype assignment, read directly from the Level-2 attention argmax

No post-hoc clustering is applied to obtain the prototype assignment.
"""

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.utils import subgraph

from hebi.data.sampler import LocalSampler
from hebi.training.pretrain import Train
import hebi.training.pretrain as _pretrain   # GRN_FEEDS_MAIN_PATH, read at call time
from hebi.model.hebi import hierarchy_forward
from hebi.model.grn_layer import GRNRefinementLayer

EMBED_DIM = 128


def load_hebi(
    checkpoint_path: str,
    data_dir: str,
    vocab_path: str,
    device: str = "cuda:0",
    use_grn: bool = False,
    init: str = "pretrained",
    random_init_checkpoint: str = "hebi_random_init_temp.pt",
):
    """Rebuild a HEBI model from a checkpoint.

    Prototype counts are read from the checkpoint itself, so checkpoints
    trained with different numbers of prototypes load without arguments.

    init:
        "pretrained"  load all weights from the checkpoint
        "hybrid"      Level-0 encoder from a random-init checkpoint,
                      Levels 1-2 from the pretrained checkpoint
    """
    state = torch.load(checkpoint_path, map_location=device)

    # Level 1 has no prototype bank any more (keys are the communities of each
    # section); the count is kept only for the Train constructor.
    n_proto_l1 = 1024
    try:
        n_proto_l2 = state["pooler_lvl2"]["l2_prototypes"].shape[0]
    except (KeyError, IndexError):
        n_proto_l2 = 20

    trainer = Train(
        data_dir=data_dir,
        vocab_path=vocab_path,
        l0_encoder_type="transformer",
        batch_size=2048,
        num_workers=4,
        lr=1e-4,
        num_epochs=1,
        global_num_layers=2,
        num_prototypes_lvl1=n_proto_l1,
        num_prototypes_lvl2=n_proto_l2,
        decoder_hidden_dim=256,
        early_stopping_patience=10,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Build module shapes from the first section, then load weights.
    trainer.discover_data_files()
    trainer.data_path = trainer.data_files[0]
    trainer.load_data()
    trainer.build_models(trainer.X.size(0), trainer.X.shape)

    trainer.gene_embedder.load_state_dict(state["gene_embedder"])
    trainer.gene_linear.load_state_dict(state["gene_linear"])
    trainer.model_lvl0.load_state_dict(state["model_lvl0"])
    trainer.model_lvl1.load_state_dict(state["model_lvl1"])
    trainer.pooler_lvl2.load_state_dict(state["pooler_lvl2"])
    trainer.model_lvl2.load_state_dict(state["model_lvl2"])
    trainer.global_decoder.load_state_dict(state["global_decoder"])

    if init == "hybrid":
        random_state = torch.load(random_init_checkpoint, map_location=device)
        trainer.model_lvl0.load_state_dict(random_state["model_lvl0"])
        trainer.optimizer = None
        print("Level-0 encoder randomised; Levels 1-2 kept pretrained")

    if use_grn:
        trainer.grn_layer = GRNRefinementLayer(dim_model=EMBED_DIM, heads=4).to(
            trainer.device
        )

        if "grn_layer" in state:
            trainer.grn_layer.load_state_dict(state["grn_layer"])
            print("Loaded trained GRN weights from the checkpoint")

            if "grn_expr_decoder" in state:
                trainer.grn_expr_decoder = torch.nn.Linear(
                    trainer.embedding_size, 1
                ).to(trainer.device)
                trainer.grn_expr_decoder.load_state_dict(
                    state["grn_expr_decoder"]
                )
        else:
            print("WARNING: this checkpoint contains no GRN weights, so the "
                  "refinement layer is untrained.")
            print("         Attention from it reflects graph structure and "
                  "expression only, not anything learned. Run --mode continual "
                  "--grn to train it, and use the checkpoint that produces.")

    print(f"Loaded checkpoint {checkpoint_path} (epoch {state.get('epoch', '?')})")
    return trainer


def _gene_embedding(trainer, gene_graph, device, return_attention=False):
    gene_emb = trainer.gene_embedder(trainer.gene_ids.to(device))
    gene_emb = trainer.gene_linear(gene_emb)
    gene_emb = torch.nn.functional.normalize(gene_emb, dim=1, p=2)
    attn = None
    if (gene_graph is not None and "global" in gene_graph
            and getattr(trainer, "grn_layer", None) is not None):
        refined, attn = trainer.grn_layer(
            gene_emb=gene_emb,
            edge_index_grn=gene_graph["global"].edge_index.to(device),
            return_attention=True,
        )
        if _pretrain.GRN_FEEDS_MAIN_PATH:     # else: attention only, encoder input unchanged
            gene_emb = refined
    return (gene_emb, attn) if return_attention else gene_emb


@torch.no_grad()
def section_community_reps(trainer, gene_graph=None):
    """First encoding pass: Level-1 input fc_in(z0) for every cell of the
    loaded section, averaged per community. Using these whole-section means
    for every batch makes the output independent of how cells are batched."""
    device = trainer.device
    gene_emb = _gene_embedding(trainer, gene_graph, device)
    m1 = trainer.model_lvl1
    n_comm = trainer.super_sp_fixed.size(0)
    sums = torch.zeros(n_comm, m1.hidden_channels, device=device)
    count = torch.zeros(n_comm, device=device)
    sampler = LocalSampler(trainer.edge_index.cpu(), sizes=[20, 10],
                           num_nodes=trainer.X.size(0), batch_size=trainer.batch_size,
                           num_workers=trainer.num_workers, shuffle=False)
    for edge_block, node_idx_b, bs in sampler:
        x_proj = trainer.X[node_idx_b].to(device) @ gene_emb
        edge_attr_b = (edge_block[2].to(device), edge_block[3].to(device))
        z0 = trainer.model_lvl0(x_proj, edge_block[:2].to(device), edge_attr_b, None, bs=bs)[:bs]
        idx = trainer.node2comm_fixed[node_idx_b[:bs]].long()
        sums.index_add_(0, idx, m1.fc_in(z0))
        count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
    return sums / count.clamp(min=1).unsqueeze(1), count


@torch.no_grad()
def encode_section(trainer, section_path: str, gene_graph=None):
    """Encode one section to its three hierarchical representations.

    Returns a dict with keys:
        z0, z1, z2            [N, 128] float tensors on the trainer device
        prototypes            [N] long tensor, Level-2 attention argmax
        niche_attention       [N, K2] float tensor
        grn_attention         per-edge GRN attention, or None
    """
    trainer.data_path = section_path
    trainer.load_data()

    # Inference mode: dropout off, sharp niche assignment.
    for m in (trainer.model_lvl0, trainer.model_lvl1, trainer.pooler_lvl2,
              trainer.model_lvl2, trainer.global_decoder):
        m.eval()
    if getattr(trainer, "grn_layer", None) is not None:
        trainer.grn_layer.eval()

    global_reps = section_community_reps(trainer, gene_graph)

    X = trainer.X
    n_cells = X.size(0)
    device = trainer.device

    sampler = LocalSampler(
        trainer.edge_index.cpu(),
        sizes=[20, 10],
        num_nodes=n_cells,
        batch_size=trainer.batch_size,
        num_workers=trainer.num_workers,
        shuffle=True,
    )

    z0_full = torch.zeros((n_cells, EMBED_DIM), dtype=torch.float32, device=device)
    z1_full = torch.zeros((n_cells, EMBED_DIM), dtype=torch.float32, device=device)
    z2_full = torch.zeros((n_cells, EMBED_DIM), dtype=torch.float32, device=device)
    attn_full = torch.zeros(
        (n_cells, trainer.num_prototypes_lvl2), dtype=torch.float32, device=device
    )

    grn_attention = None

    for edge_block, node_idx_b, bs in tqdm(
        sampler, desc=f"Encoding {section_path}", unit="batch"
    ):
        x_raw = X[node_idx_b].to(device)
        edge_index_sub = edge_block[:2].to(device)
        edge_attr_b = (edge_block[2].to(device), edge_block[3].to(device))
        target_idx = node_idx_b[:bs]

        subgraph(
            torch.arange(bs, device=device), edge_index_sub, relabel_nodes=True
        )

        # Gene projection, optionally refined over the gene-interaction graph.
        gene_ids = trainer.gene_ids.to(device)
        gene_emb = trainer.gene_embedder(gene_ids)
        gene_emb = trainer.gene_linear(gene_emb)
        gene_emb = torch.nn.functional.normalize(gene_emb, dim=1, p=2)

        if (
            gene_graph is not None
            and "global" in gene_graph
            and getattr(trainer, "grn_layer", None) is not None
        ):
            edge_index_grn = gene_graph["global"].edge_index.to(device)
            refined, attn = trainer.grn_layer(
                gene_emb=gene_emb,
                edge_index_grn=edge_index_grn,
                return_attention=True,
            )
            if _pretrain.GRN_FEEDS_MAIN_PATH:  # else: attention only, encoder input unchanged
                gene_emb = refined
            # The gene graph is global, so attention is identical across
            # batches; keep the first.
            if grn_attention is None:
                grn_attention = attn.cpu()

        x_proj = x_raw @ gene_emb

        z0_batch = trainer.model_lvl0(x_proj, edge_index_sub, edge_attr_b, None, bs=bs)
        z0_targets = z0_batch[:bs] if z0_batch.size(0) > bs else z0_batch

        node2comm_batch = trainer.node2comm_fixed[target_idx]
        z1_batch, z2_batch, attn_probs, _ = hierarchy_forward(
            trainer.model_lvl1, trainer.pooler_lvl2, trainer.model_lvl2,
            z0_targets, node2comm_batch, trainer.super_sp_fixed,
            global_reps=global_reps,
        )
        z1_targets = z1_batch

        z0_full[target_idx] = z0_targets
        z1_full[target_idx] = z1_targets
        z2_full[target_idx] = z2_batch
        attn_full[target_idx] = attn_probs

    return {
        "z0": z0_full,
        "z1": z1_full,
        "z2": z2_full,
        "prototypes": torch.argmax(attn_full, dim=1).cpu(),
        "niche_attention": attn_full,
        "grn_attention": grn_attention,
    }


def embedding_diagnostics(z, name: str):
    """Print variance, mean pairwise cosine similarity, and numerical rank.
    Useful for spotting representational collapse."""
    from sklearn.metrics.pairwise import cosine_similarity

    z_np = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    var = z_np.var(axis=0).mean()

    n_sample = min(500, len(z_np))
    sample = z_np[np.random.choice(len(z_np), n_sample, replace=False)]

    cos = cosine_similarity(sample)
    np.fill_diagonal(cos, np.nan)

    rank = np.linalg.matrix_rank(sample, tol=1e-3)
    print(
        f"[{name}] var={var:.4f} | mean_cosine={np.nanmean(cos):.4f} | "
        f"rank={rank}/{sample.shape[1]}"
    )
