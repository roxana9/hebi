#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 13:42:24 2025

@author: roxana
"""
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
import torch_sparse
from torch_sparse import SparseTensor

from torch_geometric.utils import to_undirected
import numpy as np



class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class LocalSampler(torch.utils.data.DataLoader):
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        # Save for Pytorch Lightning...
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        batch_size: int = len(batch)

        edge_index, edge_dist = [], []
        for i in range(len(batch)) :
            out = self.sample_one(batch[i:i+1])
            edge_index.append(out[0])
            edge_dist.append(out[1])
        edge_index = torch.cat(edge_index, dim=1)
        edge_dist = torch.cat(edge_dist, dim=1)

        node_idx = torch.unique(edge_index[0]) # source nodes, will include target
        node_idx_flag = torch.tensor([i not in batch for i in node_idx])
        node_idx = node_idx[node_idx_flag]
        node_idx = torch.cat([batch, node_idx])

        # relabel
        node_idx_all = torch.zeros(self.num_nodes, dtype=torch.long)
        node_idx_all[node_idx] = torch.arange(node_idx.size(0))
        edge_index = node_idx_all[edge_index]

        return torch.cat([edge_index, edge_dist], dim=0), node_idx, batch_size

    def sample_one(self, idx):
        assert idx.dim() == 1 and len(idx) == 1

        n_id = idx
        ptrs = []
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            # e_id = adj_t.storage.value()
            total_target = adj_t.sparse_sizes()[::-1] # (total, target)
            total_size = total_target[0]
            ptrs.append(total_size)

        target = torch.tensor([idx.item()] * len(n_id))
        dist = torch.ones(len(n_id))

        for i, ptr in enumerate(reversed(ptrs)) :
            dist[:ptr] = len(self.sizes) - i
        dist[0] = 0
        edge_dist = dist.long()
        # edge_index = torch.stack([target, n_id]) #BUG
        edge_index = torch.stack([n_id, target]) #edge_index[0]:source, edge_index[1]:target
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html

        edge_dist_count = [ptrs[i+1]-ptrs[i] for i in range(len(ptrs)-1)]
        edge_dist_count = [1, ptrs[0]-1] + edge_dist_count
        edge_dist_count = torch.tensor(edge_dist_count)
        edge_dist = torch.stack([edge_dist, edge_dist_count[edge_dist]])

        return edge_index, edge_dist

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'


