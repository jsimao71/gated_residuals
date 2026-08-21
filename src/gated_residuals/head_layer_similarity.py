"""Head/layer organization metrics that preserve structural axes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_cosine(vectors: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine matrix for the penultimate item axis."""
    if vectors.ndim < 2:
        raise ValueError("expected [..., item, feature]")
    normalized = F.normalize(vectors.detach().float(), dim=-1)
    return normalized @ normalized.transpose(-1, -2)


def off_diagonal_mean(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-1] != matrix.shape[-2] or matrix.shape[-1] < 2:
        raise ValueError("expected a square matrix with at least two items")
    size = matrix.shape[-1]
    mask = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
    return matrix[..., mask].mean(dim=-1)


def head_layer_organization(representations: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compare head similarity within layers and corresponding heads across layers.

    Expected shape is ``[layer, head, observation, feature]``.
    """
    if representations.ndim != 4:
        raise ValueError("expected [layer, head, observation, feature]")
    layers, heads, _, _ = representations.shape
    if layers < 2 or heads < 2:
        raise ValueError("need at least two layers and two heads")
    summaries = representations.detach().float().flatten(start_dim=2)
    within = off_diagonal_mean(pairwise_cosine(summaries)).mean()
    across_by_head = []
    for head in range(heads):
        across_by_head.append(off_diagonal_mean(pairwise_cosine(summaries[:, head])).mean())
    across = torch.stack(across_by_head).mean()
    return {
        "within_layer_head_similarity": within,
        "across_layer_corresponding_head_similarity": across,
        "within_minus_across_similarity": within - across,
    }
