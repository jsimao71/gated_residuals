"""Conventional task metrics retained alongside internal measurements."""

from __future__ import annotations

import math

import torch


def perplexity(loss: float) -> float:
    return float(math.exp(min(float(loss), 20.0)))


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    if predictions.shape != targets.shape:
        raise ValueError("logit and target token axes do not match")
    valid = targets != ignore_index
    if not valid.any():
        raise ValueError("no valid targets")
    return (predictions[valid] == targets[valid]).float().mean()


def target_log_probability(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logit and target observation axes do not match")
    return logits.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def logit_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Target logit minus the strongest non-target logit."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logit and target observation axes do not match")
    target = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    masked = logits.clone()
    masked.scatter_(-1, targets.unsqueeze(-1), -torch.inf)
    return target - masked.max(dim=-1).values


def output_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.float().softmax(dim=-1)
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
