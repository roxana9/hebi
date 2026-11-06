#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content-anchored distance bias for HEBI (experimental, for testing only).

Problem with the current bias
-----------------------------
Level 1 adds  b(D[a_i, p])  to the attention of cell i on prototype p, where
D is the METIS community distance matrix of the section and p is used as a
community index. METIS labels are arbitrary, so prototype p is anchored to an
unrelated region in every section: the bias is not invariant to relabelling
and has no consistent meaning to learn. Level 2 inherits this through
D[a_i, :] @ S_hard.

What this module does instead
-----------------------------
Each community is anchored to the prototype its cells actually attend to
(a running average of Level-1 attention, kept per section). The distance from
a cell to prototype p is then the shortest-path distance from the cell's
community to the NEAREST community anchored to p:

    d(i, p) = min_{c : anchor(c) = p} D[a_i, c]          (30 if p is absent)

and at Level 2 the distance to niche k is the minimum over its prototypes:

    d(i, k) = min_{p : niche(p) = k} d(i, p)

Both depend on community content, not on community numbering, so relabelling
the METIS communities leaves them unchanged. Scrambling the distances does
not, so a positive control can tell whether the bias is used.

Nothing in the repository is edited. `enable()` swaps the three classes the
trainer builds and wraps encode_section with a warm-up pass (the anchors are
not stored in the checkpoint, so they are rebuilt on the section first).
State dicts are unchanged: checkpoints load either way.
"""

import torch
import torch.nn.functional as F
import math

import hebi.model.hebi as _m

FAR = 30          # bucket used for "prototype not present in this section"
REFRESH = 10      # recompute anchors every this many forward calls
DECAY = 0.95      # forgetting applied to the attention statistics on refresh

# Shared between the three modules within one forward pass
# (L1 -> pooler -> L2, always in that order).
_STATE = {"dist_proto": None, "S": None}


def _fingerprint(D):
    return (D.shape[0], round(float(D.sum()), 3), round(float(D[0].sum()), 3))


class NodeToPrototypeTransformerCB(_m.NodeToPrototypeTransformer):
    """Level 1 with the content-anchored distance bias."""

    def _section_state(self, D):
        if not hasattr(self, "_sections"):
            self._sections, self._fp_cache = {}, {}
        ptr = D.data_ptr()
        fp = self._fp_cache.get(ptr)
        if fp is None:
            fp = _fingerprint(D)
            self._fp_cache = {ptr: fp}           # keep only the current tensor
        st = self._sections.get(fp)
        if st is None:
            n = D.shape[0]
            st = {
                "sum": torch.zeros(n, self.num_prototypes, device=D.device),
                "count": torch.zeros(n, device=D.device),
                "dist": torch.full((n, self.num_prototypes), float(FAR), device=D.device),
                "calls": 0,
            }
            self._sections[fp] = st
        return st

    @torch.no_grad()
    def _refresh(self, st, D):
        valid = st["count"] > 0
        dist = torch.full_like(st["dist"], float(FAR))
        if valid.any():
            anchor = st["sum"][valid].argmax(dim=1)                  # [n_valid]
            Dv = torch.nan_to_num(D[:, valid].float(), posinf=FAR).clamp(0, FAR)
            dist.scatter_reduce_(1, anchor.unsqueeze(0).expand(D.shape[0], -1).contiguous(),
                                 Dv, reduce="amin", include_self=True)
        st["dist"] = dist
        st["sum"] *= DECAY
        st["count"] *= DECAY

    def forward(self, x_lower, node2comm_batch, super_sp_global):
        H, C = self.heads, self.hidden_channels // self.heads
        N_batch = x_lower.size(0)
        P = self.num_prototypes
        D = super_sp_global

        st = self._section_state(D)
        if st["calls"] % REFRESH == 0:
            self._refresh(st, D)
        st["calls"] += 1

        h_lower = self.fc_in(x_lower)
        prototypes_norm = F.normalize(self.prototypes, dim=-1)
        k = prototypes_norm.view(P, H, C).permute(1, 2, 0)
        v = prototypes_norm.view(P, H, C).permute(1, 0, 2)

        dist_b = st["dist"][node2comm_batch.long()]                  # [bs, P]
        _STATE["dist_proto"] = dist_b
        Hk = self.spatial_bias_encoder(dist_b.clamp(0, FAR).long()).permute(2, 0, 1)

        last = None
        for i in range(self.num_layers):
            h_residual = h_lower
            q = self.lin_q_layers[i](h_lower).view(N_batch, H, C).permute(1, 0, 2)
            attn = (torch.matmul(q, k) / math.sqrt(C) + Hk).softmax(dim=-1)
            last = attn
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v).permute(1, 0, 2).contiguous().view(N_batch, H * C)
            h_lower = self.norms_1[i](h_residual + out)
            h_lower = self.norms_2[i](h_lower + self.ffs[i](h_lower))

        # Attention statistics per community, used to anchor communities.
        with torch.no_grad():
            idx = node2comm_batch.long()
            st["sum"].index_add_(0, idx, last.mean(dim=0).detach())
            st["count"].index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))

        return self.fc_out(h_lower)


class PrototypePoolerL2CB(_m.PrototypePoolerL2):
    def forward(self, c_l1_prototypes, hard: bool = False):
        C_L2, S = super().forward(c_l1_prototypes, hard=hard)
        _STATE["S"] = S.detach()
        return C_L2, S


class FeatureTransformerL2CB(_m.FeatureTransformerL2):
    """Level 2: distance to niche k = min distance to any of its prototypes.
    The hk_l2_distances passed in by the trainer is ignored."""

    def forward(self, x_lower, c_l2_prototypes, hk_l2_distances):
        dist_p, S = _STATE["dist_proto"], _STATE["S"]
        K2 = c_l2_prototypes.size(0)
        niche = S.argmax(dim=1)                                      # [P]
        d = torch.full((dist_p.size(0), K2), float(FAR), device=dist_p.device)
        d.scatter_reduce_(1, niche.unsqueeze(0).expand(dist_p.size(0), -1).contiguous(),
                          dist_p, reduce="amin", include_self=True)
        return super().forward(x_lower, c_l2_prototypes, d)


def enable(warmup=True):
    """Build the content-anchored modules wherever the trainer builds models,
    and run a warm-up pass before every encode_section call."""
    import hebi.training.pretrain as pre
    pre.NodeToPrototypeTransformer = NodeToPrototypeTransformerCB
    pre.PrototypePoolerL2 = PrototypePoolerL2CB
    pre.FeatureTransformerL2 = FeatureTransformerL2CB

    if warmup:
        import hebi.inference.encode as enc
        import hebi.inference.pipeline as pip
        original = getattr(enc, "_encode_section_original", enc.encode_section)
        enc._encode_section_original = original

        def encode_with_warmup(trainer, section_path, gene_graph=None):
            original(trainer, section_path, gene_graph)              # builds anchors
            return original(trainer, section_path, gene_graph)

        enc.encode_section = encode_with_warmup
        pip.encode_section = encode_with_warmup
    print("content-anchored distance bias enabled")


def bias_ranges(trainer):
    """Spread of the learned bias across distance buckets, per head."""
    out = {}
    for name, mod in [("L1", trainer.model_lvl1), ("L2", trainer.model_lvl2)]:
        w = mod.spatial_bias_encoder.weight.detach().cpu()
        out[name] = (w.max(0).values - w.min(0).values).numpy().round(4)
    return out
