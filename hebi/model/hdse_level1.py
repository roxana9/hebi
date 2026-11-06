#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDSE-style Level 1 for HEBI (experimental; put in hebi/model/).

    rep_c     = mean of h_i over the batch cells i in community c
    key/value = lin_k(rep_c), lin_v(rep_c)
    logits    = q_i . k_c / sqrt(C)  +  log(n_c)  +  w_h * d(a_i, c)

Distance term as in the HDSE code for large graphs (product.py, proteins,
arxiv-year, 100M: fc_dis = nn.Linear(1, 1) applied to the raw distance):
one slope w per head, bias = w_h * d. Negative w = attention decays with
distance. (HDSE shares one w across heads; PER_HEAD=False reproduces that.)

Settings
    INIT        "hdse": w ~ U(-1, 1) (default nn.Linear(1,1) init, as HDSE)
                "zero": w = 0, the model must find the sign itself
    SLOPE_LR_X  w = SLOPE_LR_X * theta. With Adam this makes w move
                SLOPE_LR_X times faster than the other weights, without
                touching the optimiser. 30 ~ HDSE's lr (3e-3) vs HEBI's (1e-4).
    LOG_EVERY   print w every this many training batches (75 = 1 epoch, breast)

Unchanged: self.prototypes (Level 2 still pools them), the trainer, L2.
spatial_bias_encoder still exists (unused at L1) so old tools do not break.
New parameters -> use with init="random" only.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hebi.model.hebi as _m

FAR = 30
INIT = "hdse"          # "hdse" or "zero"
PER_HEAD = True
SLOPE_LR_X = 30.0
LOG_EVERY = 75


class NodeToCommunityTransformer(_m.NodeToPrototypeTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hc = self.hidden_channels
        self.lin_k = nn.Linear(hc, hc)
        self.lin_v = nn.Linear(hc, hc)
        n_w = self.heads if PER_HEAD else 1
        w0 = torch.empty(n_w).uniform_(-1, 1) if INIT == "hdse" else torch.zeros(n_w)
        self.dist_theta = nn.Parameter(w0 / SLOPE_LR_X)
        self.slope_history = []        # w per logged step
        self._calls = 0
        self.reset_locality()

    # ---- helpers -------------------------------------------------------
    @property
    def slope(self):
        """w per head (attention change per unit of community distance)."""
        w = self.dist_theta * SLOPE_LR_X
        return w.expand(self.heads) if w.numel() == 1 else w

    def reset_locality(self):
        self._loc = {"attended": 0.0, "uniform": 0.0, "n": 0}

    def locality(self):
        """Mean community distance attended vs. if attention were uniform
        over the communities present (accumulated while encoding, i.e. under no_grad)."""
        n = max(self._loc["n"], 1)
        return {"attended distance": round(self._loc["attended"] / n, 3),
                "uniform distance": round(self._loc["uniform"] / n, 3)}

    def community_reps(self, h, node2comm_batch, n_comm):
        idx = node2comm_batch.long()
        count = torch.bincount(idx, minlength=n_comm).to(h.dtype)
        s = torch.zeros(n_comm, h.size(1), device=h.device, dtype=h.dtype)
        s.index_add_(0, idx, h)
        return s / count.clamp(min=1).unsqueeze(1), count

    # ---- forward -------------------------------------------------------
    def forward(self, x_lower, node2comm_batch, super_sp_global):
        H, C = self.heads, self.hidden_channels // self.heads
        N = x_lower.size(0)
        n_comm = super_sp_global.size(0)

        h = self.fc_in(x_lower)
        reps, count = self.community_reps(h, node2comm_batch, n_comm)
        k = self.lin_k(reps).view(n_comm, H, C).permute(1, 2, 0)           # [H, C, n_comm]
        v = self.lin_v(reps).view(n_comm, H, C).permute(1, 0, 2)           # [H, n_comm, C]

        log_n = torch.log(count).view(1, 1, n_comm)                        # -inf = absent
        d = super_sp_global[node2comm_batch.long(), :].float()
        d = torch.nan_to_num(d, nan=FAR, posinf=FAR).clamp(0, FAR)         # [bs, n_comm]
        dist_bias = self.slope.view(H, 1, 1) * d.unsqueeze(0)              # [H, bs, n_comm]

        attn = None
        for i in range(self.num_layers):
            res = h
            q = self.lin_q_layers[i](h).view(N, H, C).permute(1, 0, 2)
            logits = torch.matmul(q, k) / math.sqrt(C) + log_n + dist_bias
            attn = logits.softmax(dim=-1)
            attn_d = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn_d, v).permute(1, 0, 2).contiguous().view(N, H * C)
            h = self.norms_1[i](res + out)
            h = self.norms_2[i](h + self.ffs[i](h))

        grad_on = torch.is_grad_enabled()          # training step vs. encoding (no_grad)
        with torch.no_grad():
            if grad_on:
                self._calls += 1
                if self._calls % LOG_EVERY == 0:
                    w = self.slope.detach().cpu().numpy().round(4)
                    self.slope_history.append(w)
                    print(f"    L1 distance slope w (step {self._calls}): {w}")
            else:                                                          # encoding: last layer, mean over heads
                present = (count > 0).float()
                self._loc["attended"] += float((attn.mean(0) * d).sum(1).sum())
                self._loc["uniform"] += float(((d * present).sum(1) / present.sum()).sum())
                self._loc["n"] += N

        return self.fc_out(h)


def enable():
    """Make the trainer build the HDSE-style Level 1. Call before any model is built."""
    import hebi.training.pretrain as pre
    pre.NodeToPrototypeTransformer = NodeToCommunityTransformer
    print(f"HDSE-style Level 1 enabled (keys = community means, bias = w*d, "
          f"init={INIT}, slope lr x{SLOPE_LR_X:g})")


def bias_profile(trainer):
    """L1: slope per head (or the bucket table for the original model). L2: bucket table."""
    out = {}
    m1 = trainer.model_lvl1
    if isinstance(m1, NodeToCommunityTransformer):
        w = m1.slope.detach().cpu()
        out["L1"] = {"slope w per head": w.numpy().round(4),
                     "bias at d=0,5,10,20,30 (mean over heads)":
                         (w.mean() * torch.tensor([0., 5, 10, 20, 30])).numpy().round(3)}
    else:
        w = m1.spatial_bias_encoder.weight.detach().cpu()
        out["L1"] = {"range per head": (w.max(0).values - w.min(0).values).numpy().round(4),
                     "mean over heads, bucket 0..30": w.mean(1).numpy().round(3)}
    w = trainer.model_lvl2.spatial_bias_encoder.weight.detach().cpu()
    out["L2"] = {"range per head": (w.max(0).values - w.min(0).values).numpy().round(4),
                 "mean over heads, bucket 0..30": w.mean(1).numpy().round(3)}
    return out