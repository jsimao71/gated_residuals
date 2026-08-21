"""Typed records shared by model adapters and metric modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ProbeCapture:
    """One layer's computational-selection event.

    Candidate and effective updates must be captured at the same tensor location. For
    gated attention this package uses the output-projection result for residual metrics,
    while preserving the native per-head gate separately.
    """

    layer: int
    residual_input: torch.Tensor
    candidate_update: torch.Tensor
    effective_update: torch.Tensor
    residual_after_update: torch.Tensor
    gate: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None
    candidate_heads: torch.Tensor | None = None
    effective_heads: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, require_finite: bool = True) -> None:
        """Reject mismatched locations, malformed gates, and silent NaN/Inf values."""
        state_shape = self.residual_input.shape
        for name, value in (
            ("candidate_update", self.candidate_update),
            ("effective_update", self.effective_update),
            ("residual_after_update", self.residual_after_update),
        ):
            if value.shape != state_shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} != residual shape {tuple(state_shape)}")
        if self.gate is not None:
            if self.gate.ndim not in (3, 4):
                raise ValueError("gate must preserve batch/token/head dimensions")
            if tuple(self.gate.shape[:2]) != tuple(state_shape[:2]):
                raise ValueError("gate batch/token dimensions do not match the residual state")
            if torch.any((self.gate < 0) | (self.gate > 1)):
                raise ValueError("sigmoid gate values must lie in [0, 1]")
        if require_finite:
            tensors = [
                self.residual_input,
                self.candidate_update,
                self.effective_update,
                self.residual_after_update,
                self.gate,
                self.attention_weights,
            ]
            for value in tensors:
                if value is not None and not torch.isfinite(value).all():
                    raise ValueError("capture contains NaN or Inf")

    def detached(self, *, cpu: bool = True) -> "ProbeCapture":
        """Return a graph-free capture suitable for metric collection."""

        def convert(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            value = value.detach()
            return value.cpu() if cpu else value

        return ProbeCapture(
            layer=self.layer,
            residual_input=convert(self.residual_input),  # type: ignore[arg-type]
            candidate_update=convert(self.candidate_update),  # type: ignore[arg-type]
            effective_update=convert(self.effective_update),  # type: ignore[arg-type]
            residual_after_update=convert(self.residual_after_update),  # type: ignore[arg-type]
            gate=convert(self.gate),
            attention_weights=convert(self.attention_weights),
            candidate_heads=convert(self.candidate_heads),
            effective_heads=convert(self.effective_heads),
            metadata=dict(self.metadata),
        )
