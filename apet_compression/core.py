# Portions derived from llava/model/llava_arch.py:
# Copyright 2023 Haotian Liu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extracted FPS/input compression from llava/model/llava_arch.py.

Preserves legacy basis-slot replacement and FP16 merging. Inputs are not mutated.
Only PyTorch is needed; no Transformers, LLaVA, or Qwen imports occur here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Generator, Tensor


@dataclass(frozen=True)
class CompressionConfig:
    keep_tokens: int = 96
    basis_tokens: int = 10
    epsilon: float = 1e-5
    merge: bool = True

    def __post_init__(self):
        if not isinstance(self.keep_tokens, int) or not isinstance(self.basis_tokens, int):
            raise ValueError("Token budgets must be integers")
        if not 1 <= self.basis_tokens < self.keep_tokens:
            raise ValueError("Require 1 <= basis_tokens < keep_tokens")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")


@dataclass
class CompressionResult:
    full_features: Tensor  # [B, N, D], before the model's projector
    kept_indices: Tensor  # [B, K], ascending spatial indices
    kept_mask: Tensor  # [B, N]
    basis_indices: Tensor | None  # Original FPS locations, not replacement slots
    basis_slots: Tensor | None  # Sorted last M entries of error top-k
    error_indices: Tensor  # Error-ranked top-k, before spatial sorting
    errors: Tensor | None

    @property
    def tokens(self):
        return self.full_features.gather(
            1, self.kept_indices.unsqueeze(-1).expand(-1, -1, self.full_features.shape[-1]),
        )


def fps(features: Tensor, count: int, generator: Generator | None = None):
    import torch

    batch, number, _ = features.shape
    if not 1 <= count <= number:
        raise ValueError("FPS count must be between 1 and the input token count")
    centroids = torch.zeros(batch, count, dtype=torch.long, device=features.device)
    distances = torch.full((batch, number), 1e10, device=features.device)
    farthest = torch.randint(0, number, (batch,), device=features.device, generator=generator)
    batch_indices = torch.arange(batch, device=features.device)
    for index in range(count):
        centroids[:, index] = farthest
        centroid = features[batch_indices, farthest].unsqueeze(1)
        current = ((features - centroid) ** 2).sum(-1)
        distances = torch.min(distances, current)
        farthest = distances.max(1)[1]
    return centroids


def _merge(features, error_indices, basis_tokens):
    import torch
    import torch.nn.functional as F

    batch, number, dim = features.shape
    mask = torch.zeros(batch, number, dtype=torch.bool, device=features.device)
    mask.scatter_(1, error_indices, True)
    destinations = torch.stack([features[b][error_indices[b][:-basis_tokens]] for b in range(batch)])
    discarded = torch.stack([features[b][~mask[b]] for b in range(batch)])
    nearest = F.cosine_similarity(discarded.unsqueeze(2), destinations.unsqueeze(1), dim=3).argmax(2)
    counts = torch.zeros(batch, destinations.shape[1], device=features.device, dtype=torch.int)
    destinations.scatter_add_(1, nearest.unsqueeze(-1).expand(-1, -1, dim), discarded)
    counts.scatter_add_(1, nearest, torch.ones_like(nearest, dtype=counts.dtype))
    destinations /= 1 + counts.unsqueeze(2)
    for b in range(batch):
        features[b, error_indices[b][:-basis_tokens]] = destinations[b]
    return features


def compress_tokens(features: Tensor, config: CompressionConfig,
                    generator: Generator | None = None) -> CompressionResult:
    import torch

    if features.ndim != 3 or min(features.shape) < 1 or not features.is_floating_point():
        raise ValueError("Expected floating-point visual features [B, N, D]")
    batch, number, dim = features.shape
    if config.keep_tokens > number:
        raise ValueError(f"Cannot retain {config.keep_tokens} of {number} tokens")
    if not torch.isfinite(features).all().item():
        raise ValueError("Visual features contain non-finite values")
    if config.keep_tokens == number:
        indices = torch.arange(number, device=features.device).expand(batch, -1)
        return CompressionResult(features, indices, torch.ones(
            batch, number, dtype=torch.bool, device=features.device), None, None, indices, None)

    basis_indices = fps(features, config.basis_tokens, generator)
    basis = features.gather(1, basis_indices.unsqueeze(-1).expand(-1, -1, dim)).float()
    values = features.float()
    gram = basis @ basis.transpose(-1, -2)
    gram.diagonal(dim1=-2, dim2=-1).add_(config.epsilon)
    rhs = basis @ values.transpose(-1, -2)
    coefficients = torch.linalg.solve(gram, rhs).transpose(-1, -2)
    reconstructed = coefficients @ basis
    errors = torch.norm(values - reconstructed, dim=-1)
    if not torch.isfinite(errors).all().item():
        raise RuntimeError("ApET reconstruction produced non-finite errors")
    error_indices = torch.topk(errors, k=config.keep_tokens, dim=1)[1]
    # Boolean assignment in the original places basis vectors in ascending slots,
    # even though the last M top-k entries themselves are in error order.
    basis_slots = error_indices[:, -config.basis_tokens:].sort(dim=1).values
    working = values.half().clone()
    working.scatter_(1, basis_slots.unsqueeze(-1).expand(-1, -1, dim), basis.half())
    if config.merge:
        working = _merge(working, error_indices, config.basis_tokens)
    if not torch.isfinite(working).all().item():
        raise RuntimeError("ApET merging produced non-finite features")
    mask = torch.zeros(batch, number, dtype=torch.bool, device=features.device)
    mask.scatter_(1, error_indices, True)
    return CompressionResult(working, error_indices.sort(dim=1).values, mask,
                             basis_indices, basis_slots, error_indices, errors)
