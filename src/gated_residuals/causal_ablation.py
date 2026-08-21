"""Gate and block interventions kept separate from native-forward instrumentation."""

from __future__ import annotations

from contextlib import AbstractContextManager
from enum import Enum

import torch


class GateIntervention(str, Enum):
    NATIVE = "native"
    FORCED_OPEN = "forced_open"
    FORCED_CLOSED = "forced_closed"
    MEAN = "mean"
    SHUFFLED_TOKEN = "shuffled_token"
    THRESHOLDED = "thresholded"


def intervene_gate(
    gate: torch.Tensor,
    mode: GateIntervention | str,
    *,
    threshold: float = 0.5,
    token_dim: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Construct a counterfactual gate while preserving shape and marginal values."""
    mode = GateIntervention(mode)
    values = gate
    if mode is GateIntervention.NATIVE:
        return values
    if mode is GateIntervention.FORCED_OPEN:
        return torch.ones_like(values)
    if mode is GateIntervention.FORCED_CLOSED:
        return torch.zeros_like(values)
    if mode is GateIntervention.MEAN:
        reduce = tuple(index for index in range(values.ndim) if index != values.ndim - 2)
        return values.mean(dim=reduce, keepdim=True).expand_as(values)
    if mode is GateIntervention.THRESHOLDED:
        return (values > threshold).to(values.dtype)
    moved = values.movedim(token_dim, 0)
    permutation = torch.randperm(moved.shape[0], device=values.device, generator=generator)
    return moved[permutation].movedim(0, token_dim)


class SkipBlocks(AbstractContextManager["SkipBlocks"]):
    """Temporarily replace selected residual block outputs by their residual inputs.

    The block is still executed, so this measures functional ablation rather than FLOP or
    latency savings. Tuple outputs retain cache/attention auxiliaries from the original call.
    """

    def __init__(self, layers: list[torch.nn.Module], indices: set[int]):
        self.layers = layers
        self.indices = set(indices)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.inputs: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "SkipBlocks":
        for index in sorted(self.indices):
            layer = self.layers[index]

            def pre_hook(module, args, layer_index=index):
                self.inputs[layer_index] = args[0]

            def post_hook(module, args, output, layer_index=index):
                residual = self.inputs[layer_index]
                if isinstance(output, tuple):
                    return (residual, *output[1:])
                return residual

            self.handles.append(layer.register_forward_pre_hook(pre_hook))
            self.handles.append(layer.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.inputs.clear()
