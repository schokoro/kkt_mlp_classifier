"""MLP со стволом и тремя головами; первый слой принимает разреженный батч (§4.5).

Первый слой — ``nn.EmbeddingBag(mode="sum")`` с ``per_sample_weights``: для
CSR-батча это ровно ``X @ W + b`` без материализации dense-матрицы
(см. DECISION_LOG, способ подачи разреженных батчей).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from .config import ModelConfig

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}


@dataclass
class SparseBatch:
    indices: torch.Tensor
    offsets: torch.Tensor
    weights: torch.Tensor

    def to(self, device) -> "SparseBatch":
        return SparseBatch(
            self.indices.to(device, non_blocking=True),
            self.offsets.to(device, non_blocking=True),
            self.weights.to(device, non_blocking=True),
        )


def csr_to_batch(X: sp.csr_matrix) -> SparseBatch:
    """CSR-срез → тензоры EmbeddingBag (indices, offsets, per_sample_weights)."""
    return SparseBatch(
        torch.from_numpy(X.indices.astype(np.int64, copy=False)),
        torch.from_numpy(X.indptr[:-1].astype(np.int64, copy=False)),
        torch.from_numpy(X.data.astype(np.float32, copy=False)),
    )


class SparseLinear(nn.Module):
    """y = X @ W + b для разреженного X, через EmbeddingBag."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bag = nn.EmbeddingBag(in_features, out_features, mode="sum")
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Инициализация как у nn.Linear: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        bound = 1.0 / np.sqrt(in_features)
        nn.init.uniform_(self.bag.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, batch: SparseBatch) -> torch.Tensor:
        return self.bag(batch.indices, batch.offsets, per_sample_weights=batch.weights) + self.bias


def _norm(kind: str | None, dim: int) -> nn.Module:
    if kind is None:
        return nn.Identity()
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "batchnorm":
        return nn.BatchNorm1d(dim)
    raise ValueError(kind)


class SparseMLP(nn.Module):
    """Ствол: [SparseLinear|Linear] → norm → activation → dropout; головы L1/L2/L3."""

    def __init__(self, n_features: int, n_classes: tuple[int, int, int], cfg: ModelConfig):
        super().__init__()
        if len(cfg.hidden) != len(cfg.dropout):
            raise ValueError("hidden и dropout должны быть одной длины")
        self.n_features = n_features
        self.n_classes = tuple(n_classes)
        self.input = SparseLinear(n_features, cfg.hidden[0])
        blocks = []
        dims = list(cfg.hidden)
        for i, (h, p) in enumerate(zip(dims, cfg.dropout)):
            layers = [] if i == 0 else [nn.Linear(dims[i - 1], h)]
            layers += [_norm(cfg.norm, h), _ACT[cfg.activation](), nn.Dropout(p)]
            blocks.append(nn.Sequential(*layers))
        self.trunk = nn.ModuleList(blocks)
        last = dims[-1]
        self.head_l1 = nn.Linear(last, n_classes[0])
        self.head_l2 = nn.Linear(last, n_classes[1])
        self.head_l3 = nn.Linear(last, n_classes[2])

    def forward(self, batch: SparseBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input(batch)
        for block in self.trunk:
            h = block(h)
        return self.head_l1(h), self.head_l2(h), self.head_l3(h)


def param_counts(model: nn.Module) -> list[dict]:
    """Число параметров по слоям (для отчёта, §5.4)."""
    rows = []
    for name, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own:
            rows.append({"layer": name, "type": type(module).__name__, "params": int(own)})
    total = sum(r["params"] for r in rows)
    rows.append({"layer": "TOTAL", "type": "", "params": int(total)})
    return rows
