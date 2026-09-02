#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:04:58 2026

@author: tower4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NovaeSwAVAdapter(nn.Module):
    """
    Minimal Novae-style clustering head for HEBI embeddings.
    This keeps the clustering objective separate from HEBI pretraining.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_prototypes: int = 20,
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

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1, p=2)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=1, p=2)
        return z @ self.prototypes.T

    @torch.no_grad()
    def sinkhorn(self, projections: torch.Tensor) -> torch.Tensor:
        Q = torch.exp(projections / self.sinkhorn_epsilon).T  # [K, B]
        B = Q.shape[1]
        K = Q.shape[0]

        Q /= Q.sum()

        for _ in range(self.sinkhorn_iterations):
            Q /= Q.sum(dim=1, keepdim=True)
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.T  # [B, K]

    def cross_entropy_loss(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.sum(q * F.log_softmax(p / self.temperature, dim=1), dim=1))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        self.normalize_prototypes()

        p1 = self.projection(z1)
        p2 = self.projection(z2)

        with torch.no_grad():
            q1 = self.sinkhorn(p1)
            q2 = self.sinkhorn(p2)

        loss = 0.5 * (
            self.cross_entropy_loss(q1, p2) +
            self.cross_entropy_loss(q2, p1)
        )
        return loss, p1, p2, q1, q2

    @torch.no_grad()
    def predict_clusters(self, z: torch.Tensor) -> torch.Tensor:
        self.normalize_prototypes()
        p = self.projection(z)
        return torch.argmax(p, dim=1)