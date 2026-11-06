#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDSE-style Levels 1 and 2 for HEBI (experimental; put in hebi/model/).
Nothing in the repo is edited: enable() swaps the three classes the trainer
builds. Use with init="random" only (new parameters).

Level 1  (as hdse_level1.py)
    rep_c     = mean of h_i over the batch cells in community c
    key/value = lin_k(rep_c), lin_v(rep_c)
    logits    = q_i . k_c / sqrt(C) + log(n_c) + w1_h * d(a_i, c),
              w1_h = -softplus(theta_h) <= 0

Level 2  (new: no community index is ever paired with a prototype index)
    pooler    S[c, k] = pool_mlp(rep_c)   -> community c assigned to niche k
              (same pool_mlp, same training/inference branches as before;
               input = the communities present in the batch, not the 1024
               free Level-1 prototypes)
    niche key C_L2[k] = l2_prototypes[k] + CONTEXT_SCALE * mean of the
              community reps assigned to k
    distance  d(i, k) = min over communities c assigned to k of D[a_i, c]
              niches with no community in the batch are masked (-inf)
    logits    = (q_i . k_k / sqrt(C) + w2_h * d(i, k)) * temperature,
              w2_h = -softplus(theta_h) <= 0  (attention cannot grow with distance)

How it plugs into the unchanged trainer / encoder
    L1 stores the community reps and distances of the current batch in
    _STATE; the pooler ignores the C_L1 it is given and pools those reps;
    L2 ignores the hk_l2_distances the trainer computes and uses its own.
    The pooler still returns S with one row per community of the section
    (absent communities get a constant uniform row, no gradient) so the
    trainer's own code and losses run unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hebi.model.hebi as _m

FAR = 30
INIT = "hdse"          # L1 slope when L1_NONPOS=False: "hdse" (U(-1,1)) or "zero"
# L1 slope constrained non-positive too: w1 = -softplus(SLOPE_LR_X * theta).
# Unconstrained, it drifted positive once Level 2 took over locality.
L1_NONPOS = True
L1_W_INIT = -0.1
# L2 slope is constrained non-positive: w2 = -softplus(L2_SLOPE_LR_X * theta).
# Unconstrained, it went positive (~ +0.6 to +1.2): every cell then attended
# the niche FARTHEST from it, labels followed geometry instead of content.
L2_NONPOS = True
L2_W_INIT = -0.1       # starting slope (must be < 0 when L2_NONPOS)
L2_SLOPE_LR_X = 10.0
SLOPE_LR_X = 30.0      # slope moves this many times faster than other weights
CONTEXT_SCALE = 0.5    # weight of the community context in the niche keys
LOG_EVERY = 75         # print slopes every N training batches (75 = 1 epoch, breast)
FULL_SLIDE_ENCODE = True   # encoding only: community means over the whole section
NO_DISTANCE = False        # ablation: distance slopes fixed at 0 at both levels
                           # (same keys, pooling and masks; only the distance term removed)
DETACH_POOLER = False
POOLER_GRAD_SCALE = 1.0    # 0 < s < 1: gradient from the pooler into L0/L1 scaled by s
                           # (value unchanged). 1 = full model, 0 = DETACH_POOLER
L2_W_MAX = None            # e.g. 0.5: L2 slope bounded to [-L2_W_MAX, 0]
                           # (w2 = -L2_W_MAX * sigmoid(theta)); None = unbounded -softplus      # True: pooler sees community means without gradient, so the
                           # pooling losses (entropy/balance) cannot reshape L0/L1

_STATE = {}            # shared within one forward pass: L1 -> pooler -> L2


def _slope_param(n, init):
    w0 = torch.empty(n).uniform_(-1, 1) if init == "hdse" else torch.zeros(n)
    return nn.Parameter(w0 / SLOPE_LR_X)


# ============================== Level 1 ==============================
class NodeToCommunityTransformer(_m.NodeToPrototypeTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hc = self.hidden_channels
        self.lin_k = nn.Linear(hc, hc)
        self.lin_v = nn.Linear(hc, hc)
        if L1_NONPOS:
            z0 = math.log(math.expm1(-L1_W_INIT))          # softplus(z0) = -L1_W_INIT
            self.dist_theta = nn.Parameter(torch.full((self.heads,), z0 / SLOPE_LR_X))
        else:
            self.dist_theta = _slope_param(self.heads, INIT)
        self._calls = 0
        self.reset_locality()

    @property
    def slope(self):
        if NO_DISTANCE:
            return torch.zeros_like(self.dist_theta)
        z = self.dist_theta * SLOPE_LR_X
        return -F.softplus(z) if L1_NONPOS else z

    def reset_locality(self):
        self._loc = {"attended": 0.0, "uniform": 0.0, "n": 0}

    def locality(self):
        n = max(self._loc["n"], 1)
        return {"attended distance": round(self._loc["attended"] / n, 3),
                "uniform distance": round(self._loc["uniform"] / n, 3)}

    def forward(self, x_lower, node2comm_batch, super_sp_global):
        H, C = self.heads, self.hidden_channels // self.heads
        N = x_lower.size(0)
        n_comm = super_sp_global.size(0)
        idx = node2comm_batch.long()

        h = self.fc_in(x_lower)                                            # [bs, hc]
        glob = getattr(self, "_global_reps", None)
        if glob is not None and not torch.is_grad_enabled():
            # Encoding with whole-section community means (see encode wrapper):
            # keys no longer depend on which cells share the batch. Counts are
            # whole-section counts; log n_c differs from the batch version by
            # roughly a constant, which softmax ignores.
            reps, count = glob
        else:
            count = torch.bincount(idx, minlength=n_comm).to(h.dtype)      # [n_comm]
            reps = torch.zeros(n_comm, h.size(1), device=h.device, dtype=h.dtype)
            reps.index_add_(0, idx, h)
            reps = reps / count.clamp(min=1).unsqueeze(1)

        k = self.lin_k(reps).view(n_comm, H, C).permute(1, 2, 0)           # [H, C, n_comm]
        v = self.lin_v(reps).view(n_comm, H, C).permute(1, 0, 2)           # [H, n_comm, C]
        log_n = torch.log(count).view(1, 1, n_comm)                        # -inf = absent
        d = super_sp_global[idx, :].float()
        d = torch.nan_to_num(d, nan=FAR, posinf=FAR).clamp(0, FAR)         # [bs, n_comm]
        dist_bias = self.slope.view(H, 1, 1) * d.unsqueeze(0)

        attn = None
        for i in range(self.num_layers):
            res = h
            q = self.lin_q_layers[i](h).view(N, H, C).permute(1, 0, 2)
            attn = (torch.matmul(q, k) / math.sqrt(C) + log_n + dist_bias).softmax(dim=-1)
            attn_d = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn_d, v).permute(1, 0, 2).contiguous().view(N, H * C)
            h = self.norms_1[i](res + out)
            h = self.norms_2[i](h + self.ffs[i](h))

        present = count > 0
        _STATE.update(reps=reps, present=present, d=d, n_comm=n_comm)

        grad_on = torch.is_grad_enabled()
        with torch.no_grad():
            if grad_on:
                self._calls += 1
                if self._calls % LOG_EVERY == 0:
                    print(f"    slopes  L1 w: {self.slope.detach().cpu().numpy().round(4)}"
                          + _l2_slope_str())
            else:
                pf = present.float()
                self._loc["attended"] += float((attn.mean(0) * d).sum(1).sum())
                self._loc["uniform"] += float(((d * pf).sum(1) / pf.sum()).sum())
                self._loc["n"] += N

        return self.fc_out(h)


# ============================== Level 2 pooler ==============================
class CommunityPoolerL2(_m.PrototypePoolerL2):
    """Assigns the communities present in the batch to niches.
    The C_L1 argument (free Level-1 prototypes) is ignored."""

    def forward(self, c_l1_prototypes, hard: bool = False):
        reps, present, n_comm = _STATE["reps"], _STATE["present"], _STATE["n_comm"]
        c = reps[present]                                                  # [P', D]
        if DETACH_POOLER:
            c = c.detach()
        elif POOLER_GRAD_SCALE != 1.0:
            c = POOLER_GRAD_SCALE * c + (1.0 - POOLER_GRAD_SCALE) * c.detach()

        S_logits = self.pool_mlp(c)                                        # same as original
        if self.training:
            S_soft = F.softmax(S_logits / 0.5, dim=-1)
            S_hard = torch.zeros_like(S_soft)
            S_hard.scatter_(1, S_soft.argmax(dim=1, keepdim=True), 1.0)
            S = S_hard - S_soft.detach() + S_soft
        else:
            S = F.softmax(S_logits / 0.1, dim=-1)

        # Mean of the communities assigned to each niche. In training S is
        # straight-through (value = hard 0/1), so an empty niche has column
        # sum exactly 0; dividing by (0 + 1e-6) multiplied its gradient by 1e6,
        # the global grad-norm clip then shrank every other gradient and
        # training stalled. Clamp the count at 1 instead.
        context = (S.t() @ F.normalize(c, dim=-1)) / S.sum(0).clamp(min=1.0).unsqueeze(1)
        C_L2 = self.l2_prototypes + CONTEXT_SCALE * context

        K2 = S.size(1)
        S_full = torch.full((n_comm, K2), 1.0 / K2, device=S.device, dtype=S.dtype)
        S_full = S_full.index_put((present.nonzero(as_tuple=True)[0],), S)
        _STATE.update(niche_of_comm=S.argmax(dim=1).detach())
        return C_L2, S_full


# ============================== Level 2 transformer ==============================
class FeatureTransformerL2Comm(_m.FeatureTransformerL2):
    """hk_l2_distances from the trainer is ignored; distance to niche k is the
    minimum distance to a community actually assigned to k."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if L2_W_MAX is not None:
            p = -L2_W_INIT / L2_W_MAX                       # sigmoid(z0) = p
            z0 = math.log(p / (1 - p))
            self.dist_theta = nn.Parameter(torch.full((self.heads,), z0 / L2_SLOPE_LR_X))
        elif L2_NONPOS:
            z0 = math.log(math.expm1(-L2_W_INIT))          # softplus(z0) = -L2_W_INIT
            self.dist_theta = nn.Parameter(torch.full((self.heads,), z0 / L2_SLOPE_LR_X))
        else:
            self.dist_theta = nn.Parameter(torch.full((self.heads,), L2_W_INIT / L2_SLOPE_LR_X))

    @property
    def slope(self):
        if NO_DISTANCE:
            return torch.zeros_like(self.dist_theta)
        z = self.dist_theta * L2_SLOPE_LR_X
        if L2_W_MAX is not None:
            return -L2_W_MAX * torch.sigmoid(z)
        return -F.softplus(z) if L2_NONPOS else z

    def niche_distance(self, K2):
        d = _STATE["d"][:, _STATE["present"]]                             # [bs, P']
        niche = _STATE["niche_of_comm"]                                    # [P']
        out = torch.full((d.size(0), K2), float(FAR), device=d.device)
        out.scatter_reduce_(1, niche.unsqueeze(0).expand(d.size(0), -1).contiguous(),
                            d, reduce="amin", include_self=True)
        return out

    def forward(self, x_lower, c_l2_prototypes, hk_l2_distances):
        H, C = self.heads, self.hidden_channels // self.heads
        N = x_lower.size(0)
        K2 = c_l2_prototypes.size(0)

        h = self.fc_in(x_lower)
        p = F.normalize(c_l2_prototypes, dim=-1)
        k = p.view(K2, H, C).permute(1, 2, 0)
        v = p.view(K2, H, C).permute(1, 0, 2)
        dist_bias = self.slope.view(H, 1, 1) * self.niche_distance(K2).unsqueeze(0)
        # Niches with no community in the batch are masked (as HDSE masks
        # absent communities with log 0). Giving them distance 30 instead let
        # a positive slope push attention onto them, which lowers the
        # niche-usage losses (smooth/coverage) without meaning anything.
        # The mask is applied AFTER the temperature product: inside it, the
        # gradient of temperature is sum(grad * (scores + bias)), and
        # 0 * (-inf) = NaN there corrupted every weight.
        absent = (torch.bincount(_STATE["niche_of_comm"], minlength=K2) == 0).view(1, 1, K2)

        final = None
        for i in range(self.num_layers):
            res = h
            q = self.lin_q_layers[i](h).view(N, H, C).permute(1, 0, 2)
            scores = (torch.matmul(q, k) / math.sqrt(C) + dist_bias) \
                * self.temperature.clamp(min=0.1, max=10.0)
            scores = scores.masked_fill(absent, float("-inf"))
            attn = scores.softmax(dim=-1)
            if i == self.num_layers - 1:
                final = attn.mean(dim=0)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v).permute(1, 0, 2).contiguous().view(N, H * C)
            h = self.norms_1[i](res + out)
            h = self.norms_2[i](h + self.ffs[i](h))
        return self.fc_out(h), final


_L2_MODEL = []


@torch.no_grad()
def full_slide_reps(trainer, section_path, gene_graph=None):
    """Pass 1 of encoding: Level-1 input h = fc_in(z0) for every cell of the
    section, averaged per community. Mirrors encode_section up to Level 0."""
    from hebi.data.sampler import LocalSampler

    trainer.data_path = section_path
    trainer.load_data()
    dev = trainer.device
    m1 = trainer.model_lvl1
    for m in (trainer.model_lvl0, m1):
        m.eval()
    if getattr(trainer, "grn_layer", None) is not None:
        trainer.grn_layer.eval()

    gene_emb = trainer.gene_linear(trainer.gene_embedder(trainer.gene_ids.to(dev)))
    gene_emb = F.normalize(gene_emb, dim=1, p=2)
    if (gene_graph is not None and "global" in gene_graph
            and getattr(trainer, "grn_layer", None) is not None):
        gene_emb, _ = trainer.grn_layer(gene_emb=gene_emb,
                                        edge_index_grn=gene_graph["global"].edge_index.to(dev),
                                        return_attention=False)

    n_comm = trainer.super_sp_fixed.size(0)
    sums = torch.zeros(n_comm, m1.hidden_channels, device=dev)
    count = torch.zeros(n_comm, device=dev)
    sampler = LocalSampler(trainer.edge_index.cpu(), sizes=[20, 10],
                           num_nodes=trainer.X.size(0), batch_size=trainer.batch_size,
                           num_workers=trainer.num_workers, shuffle=False)
    for edge_block, node_idx_b, bs in sampler:
        x_proj = trainer.X[node_idx_b].to(dev) @ gene_emb
        edge_attr_b = (edge_block[2].to(dev), edge_block[3].to(dev))
        z0 = trainer.model_lvl0(x_proj, edge_block[:2].to(dev), edge_attr_b, None, bs=bs)[:bs]
        idx = trainer.node2comm_fixed[node_idx_b[:bs]].long()
        sums.index_add_(0, idx, m1.fc_in(z0))
        count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
    return sums / count.clamp(min=1).unsqueeze(1), count


def _wrap_encode_full_slide():
    import hebi.inference.encode as enc
    import hebi.inference.pipeline as pip
    original = getattr(enc, "_encode_before_full_slide", enc.encode_section)
    enc._encode_before_full_slide = original

    def encode_full_slide(trainer, section_path, gene_graph=None):
        m1 = trainer.model_lvl1
        m1._global_reps = full_slide_reps(trainer, section_path, gene_graph)
        try:
            return original(trainer, section_path, gene_graph)
        finally:
            m1._global_reps = None

    enc.encode_section = encode_full_slide
    pip.encode_section = encode_full_slide


def _l2_slope_str():
    if not _L2_MODEL:
        return ""
    return f"   L2 w: {_L2_MODEL[0].slope.detach().cpu().numpy().round(4)}"


def enable(level2=True):
    """Call before any model is built."""
    import hebi.training.pretrain as pre
    pre.NodeToPrototypeTransformer = NodeToCommunityTransformer
    if level2:
        pre.PrototypePoolerL2 = CommunityPoolerL2

        class _L2(FeatureTransformerL2Comm):           # registers itself for logging
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                _L2_MODEL[:] = [self]
        pre.FeatureTransformerL2 = _L2
    if FULL_SLIDE_ENCODE:
        _wrap_encode_full_slide()
    print(f"HDSE-style hierarchy enabled: L1 communities + w*d"
          + (", L2 pools communities + w*d" if level2 else "")
          + (", encode with whole-section community means" if FULL_SLIDE_ENCODE else "")
          + (", NO DISTANCE (slopes = 0)" if NO_DISTANCE else "")
          + (f", pooler grad x{POOLER_GRAD_SCALE:g}" if POOLER_GRAD_SCALE != 1.0 else "")
          + (f", L2 |w| <= {L2_W_MAX:g}" if L2_W_MAX is not None else "")
          + f" (L1 w<=0={L1_NONPOS}, init {L1_W_INIT if L1_NONPOS else INIT}, "
             f"lr x{SLOPE_LR_X:g}; L2 w<=0={L2_NONPOS}, "
             f"init {L2_W_INIT}, lr x{L2_SLOPE_LR_X:g})")


def bias_profile(trainer):
    out = {}
    for name, m in [("L1", trainer.model_lvl1), ("L2", trainer.model_lvl2)]:
        if hasattr(m, "dist_theta"):
            w = m.slope.detach().cpu()
            out[name] = {"slope w per head": w.numpy().round(4),
                         "bias at d=0,5,10,20,30 (mean over heads)":
                             (w.mean() * torch.tensor([0., 5, 10, 20, 30])).numpy().round(3)}
        else:
            w = m.spatial_bias_encoder.weight.detach().cpu()
            out[name] = {"range per head": (w.max(0).values - w.min(0).values).numpy().round(4),
                         "mean over heads, bucket 0..30": w.mean(1).numpy().round(3)}
    return out