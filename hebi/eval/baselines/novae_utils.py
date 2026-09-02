#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:43:07 2026

@author: tower4
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
class SimpleNovaeAugment(nn.Module):
    """
    Lightweight augmentation for clustering fine-tuning.
    Creates two different noisy views from the same batch input.
    """
    def __init__(self, noise_std=0.05, dropout_rate=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x_aug = x.clone()

        if self.dropout_rate > 0:
            mask = torch.rand_like(x_aug) < self.dropout_rate
            x_aug = x_aug.masked_fill(mask, 0.0)

        if self.noise_std > 0:
            x_aug = x_aug + self.noise_std * torch.randn_like(x_aug)

        return x_aug


class NovaeSwAVHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_prototypes: int,
        temperature: float = 0.1,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

        self.prototypes = nn.Linear(embedding_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    @torch.no_grad()
    def sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        # scores: [B, K]
        Q = (scores / self.sinkhorn_epsilon).t()   # [K, B]

        # numerical stabilisation
        Q = Q - Q.max()
        Q = torch.exp(Q)

        Q_sum = Q.sum()
        if not torch.isfinite(Q_sum) or Q_sum <= 0:
            Q = torch.ones_like(Q)
            Q_sum = Q.sum()

        Q = Q / Q_sum

        B = Q.shape[1]
        K = Q.shape[0]

        for _ in range(self.sinkhorn_iterations):
            row_sum = Q.sum(dim=1, keepdim=True)
            row_sum = torch.where(row_sum <= 0, torch.ones_like(row_sum), row_sum)
            Q = Q / row_sum
            Q = Q / K

            col_sum = Q.sum(dim=0, keepdim=True)
            col_sum = torch.where(col_sum <= 0, torch.ones_like(col_sum), col_sum)
            Q = Q / col_sum
            Q = Q / B

        Q = Q * B
        return Q.t()
    @torch.no_grad()
    def predict_leaf_ids(self, z: torch.Tensor) -> torch.Tensor:
        self.normalize_prototypes()
        z = F.normalize(z, dim=1)
        logits = self.prototypes(z)
        return torch.argmax(logits, dim=1)
    
    @torch.no_grad()
    def get_prototypes(self) -> torch.Tensor:
        self.normalize_prototypes()
        return self.prototypes.weight.data.detach().clone()
    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        self.normalize_prototypes()

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits1 = self.prototypes(z1) / self.temperature
        logits2 = self.prototypes(z2) / self.temperature

        with torch.no_grad():
            q1 = self.sinkhorn(logits1.detach())
            q2 = self.sinkhorn(logits2.detach())

        loss12 = -torch.mean(torch.sum(q1 * F.log_softmax(logits2, dim=1), dim=1))
        loss21 = -torch.mean(torch.sum(q2 * F.log_softmax(logits1, dim=1), dim=1))
        loss = 0.5 * (loss12 + loss21)

        return loss, logits1, logits2
@torch.no_grad()
def predict_leaf_ids(self, z: torch.Tensor) -> torch.Tensor:
    self.normalize_prototypes()
    z = F.normalize(z, dim=1)
    logits = self.prototypes(z)
    return torch.argmax(logits, dim=1)

@torch.no_grad()
def get_prototypes(self) -> torch.Tensor:
    self.normalize_prototypes()
    return self.prototypes.weight.data.detach().clone()