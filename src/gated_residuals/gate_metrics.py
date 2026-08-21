"""Metrics for explicit computational-selection gates."""

from __future__ import annotations

import torch


DEFAULT_LOW_THRESHOLDS = (0.1, 0.25, 0.5)
DEFAULT_HIGH_THRESHOLDS = (0.75, 0.9)


def validate_gate(gate: torch.Tensor) -> torch.Tensor:
    values = gate.detach().float()
    if values.ndim < 1 or not torch.isfinite(values).all():
        raise ValueError("gate must be a finite tensor")
    if torch.any((values < 0) | (values > 1)):
        raise ValueError("gate values must lie in [0, 1]")
    return values


def gate_summary(gate: torch.Tensor, *, eps: float = 1e-8) -> dict[str, torch.Tensor]:
    """Summarize a gate without discarding the raw layer/head/token tensor."""
    values = validate_gate(gate)
    flat = values.flatten()
    result = {
        "gate_mean": flat.mean(),
        "gate_median": flat.median(),
        "gate_variance": flat.var(unbiased=False),
        "gate_std": flat.std(unbiased=False),
        "gate_bernoulli_entropy": (
            -values.clamp(eps, 1 - eps) * values.clamp(eps, 1 - eps).log()
            - (1 - values).clamp(eps, 1 - eps) * (1 - values).clamp(eps, 1 - eps).log()
        ).mean(),
    }
    for threshold in DEFAULT_LOW_THRESHOLDS:
        result[f"gate_fraction_below_{threshold:g}"] = (flat < threshold).float().mean()
    for threshold in DEFAULT_HIGH_THRESHOLDS:
        result[f"gate_fraction_above_{threshold:g}"] = (flat > threshold).float().mean()
    return result


def token_autocorrelation(gate: torch.Tensor, *, lag: int = 1, token_dim: int = 1) -> torch.Tensor:
    """Pearson autocorrelation over token position for each remaining series."""
    values = validate_gate(gate).movedim(token_dim, -1)
    if lag <= 0 or lag >= values.shape[-1]:
        raise ValueError("lag must be between one and token_count - 1")
    left, right = values[..., :-lag], values[..., lag:]
    left = left - left.mean(dim=-1, keepdim=True)
    right = right - right.mean(dim=-1, keepdim=True)
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    correlation = (left * right).sum(dim=-1) / denominator.clamp_min(1e-8)
    return torch.where(denominator > 1e-8, correlation, torch.zeros_like(correlation))


def correlation_matrix(values: torch.Tensor, *, observation_dim: int) -> torch.Tensor:
    """Correlation matrix between series (heads or layers) over observations."""
    data = validate_gate(values).movedim(observation_dim, -1)
    data = data.reshape(-1, data.shape[-1])
    centered = data - data.mean(dim=-1, keepdim=True)
    normalized = centered / torch.linalg.vector_norm(centered, dim=-1, keepdim=True).clamp_min(1e-8)
    return normalized @ normalized.transpose(0, 1)


def gate_metric_correlations(gate: torch.Tensor, metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute pooled Pearson and rank-based Spearman relations.

    Callers should also report layer/head-stratified results; this helper deliberately
    labels its outputs as pooled.
    """
    gate_flat = validate_gate(gate).flatten()
    output: dict[str, float] = {}
    for name, metric in metrics.items():
        metric_flat = metric.detach().float().flatten()
        if metric_flat.numel() != gate_flat.numel():
            raise ValueError(f"metric {name!r} has {metric_flat.numel()} values; gate has {gate_flat.numel()}")
        output[f"pooled_pearson_gate_{name}"] = float(_pearson(gate_flat, metric_flat))
        output[f"pooled_spearman_gate_{name}"] = float(_pearson(_ranks(gate_flat), _ranks(metric_flat)))
    return output


def _pearson(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return (left @ right) / denominator.clamp_min(1e-8)


def _ranks(values: torch.Tensor) -> torch.Tensor:
    # Average ranks for ties keep Spearman well-defined for thresholded gates.
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks
