"""Attention concentration metrics describing memory selection."""

from __future__ import annotations

import math

import torch


def _probabilities(attention: torch.Tensor, eps: float) -> torch.Tensor:
    if attention.ndim < 2:
        raise ValueError("attention requires query and key dimensions")
    values = attention.float()
    if torch.any(values < 0) or not torch.isfinite(values).all():
        raise ValueError("attention probabilities must be finite and non-negative")
    total = values.sum(dim=-1, keepdim=True)
    if torch.any(total <= eps):
        raise ValueError("attention rows must have positive mass")
    return values / total


def attention_metrics(
    attention: torch.Tensor,
    *,
    topk: int = 5,
    sink_index: int = 0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return per-query entropy, support, concentration, and sink strength."""
    probs = _probabilities(attention, eps)
    keys = probs.shape[-1]
    k = min(max(int(topk), 1), keys)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    top = torch.topk(probs, k=k, dim=-1).values
    sink = probs[..., sink_index]
    non_sink_mean = (probs.sum(dim=-1) - sink) / max(keys - 1, 1)
    return {
        "attention_entropy": entropy,
        "attention_normalized_entropy": (
            entropy / math.log(keys) if keys > 1 else torch.zeros_like(entropy)
        ),
        "attention_effective_support": entropy.exp(),
        "attention_top1_mass": top[..., 0],
        "attention_topk_mass": top.sum(dim=-1),
        "attention_sink_mass": sink,
        "attention_sink_ratio": sink / non_sink_mean.clamp_min(eps),
    }


def evidence_mass(attention: torch.Tensor, evidence_mask: torch.Tensor) -> torch.Tensor:
    """Measure attention allocated to known evidence key positions."""
    probs = _probabilities(attention, 1e-8)
    mask = evidence_mask.to(device=probs.device, dtype=probs.dtype)
    while mask.ndim < probs.ndim:
        mask = mask.unsqueeze(-2)
    try:
        mask = torch.broadcast_to(mask, probs.shape)
    except RuntimeError as exc:
        raise ValueError("evidence mask is not broadcastable to attention") from exc
    return (probs * mask).sum(dim=-1)


def dual_selection_regimes(
    attention_entropy: torch.Tensor,
    gate: torch.Tensor,
    *,
    entropy_threshold: float,
    gate_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Count sharp/diffuse attention crossed with low/high computational selection."""
    entropy = attention_entropy.float()
    gate_values = gate.float()
    try:
        entropy, gate_values = torch.broadcast_tensors(entropy, gate_values)
    except RuntimeError as exc:
        raise ValueError("entropy and gate observations are not alignable") from exc
    diffuse = entropy >= entropy_threshold
    high = gate_values >= gate_threshold
    total = float(entropy.numel())
    return {
        "sharp_low": ((~diffuse) & (~high)).sum() / total,
        "sharp_high": ((~diffuse) & high).sum() / total,
        "diffuse_low": (diffuse & (~high)).sum() / total,
        "diffuse_high": (diffuse & high).sum() / total,
    }
