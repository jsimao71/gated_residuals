"""Common model-probe interface and numerical parity checks."""

from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Iterable

import torch

from ..records import ProbeCapture


class ModelProbeAdapter(AbstractContextManager["ModelProbeAdapter"]):
    @abstractmethod
    def iter_layers(self) -> Iterable[torch.nn.Module]:
        """Yield instrumented computational layers in execution order."""

    @abstractmethod
    def captures(self) -> list[ProbeCapture]:
        """Return captures from the most recent forward pass."""


def _logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("cannot locate logits in model output")


@torch.no_grad()
def assert_native_parity(
    model: torch.nn.Module,
    adapter: ModelProbeAdapter,
    model_inputs: dict[str, Any],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> float:
    """Compare untouched and native-instrumented logits, returning max absolute error."""
    model.eval()
    baseline = _logits(model(**model_inputs)).detach()
    with adapter:
        instrumented = _logits(model(**model_inputs)).detach()
    error = float((baseline.float() - instrumented.float()).abs().max())
    if not torch.allclose(baseline, instrumented, atol=atol, rtol=rtol):
        raise AssertionError(f"instrumentation changed logits (max absolute error={error:.6g})")
    return error
