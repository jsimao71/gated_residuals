"""Architecture-neutral probe adapter for the custom tiny Transformer."""

from __future__ import annotations

from types import MethodType
from typing import Iterable

import torch

from ..records import ProbeCapture
from ..tiny_model import TinyOutput, TinyResidualDecoder
from .base import ModelProbeAdapter


class TinyModelProbeAdapter(ModelProbeAdapter):
    """Enable native capture without changing the tiny model's numerical forward."""

    def __init__(self, model: TinyResidualDecoder):
        self.model = model
        self.output: TinyOutput | None = None
        self._original_forward = None

    def __enter__(self) -> "TinyModelProbeAdapter":
        if self._original_forward is not None:
            raise RuntimeError("adapter is already active")
        self._original_forward = self.model.forward
        adapter = self

        def instrumented_forward(_model, *args, **kwargs):
            kwargs["capture"] = True
            adapter.output = adapter._original_forward(*args, **kwargs)
            return adapter.output

        self.model.forward = MethodType(instrumented_forward, self.model)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None

    def iter_layers(self) -> Iterable[torch.nn.Module]:
        return iter(self.model.blocks)

    def _require_output(self) -> TinyOutput:
        if self.output is None:
            raise RuntimeError("no instrumented forward has completed")
        return self.output

    def captures(self) -> list[ProbeCapture]:
        output = self._require_output()
        captures = []
        for layer in range(self.model.num_layers):
            captures.append(
                ProbeCapture(
                    layer=layer,
                    residual_input=output.states[layer],
                    candidate_update=output.candidates[layer],
                    effective_update=output.effective_updates[layer],
                    residual_after_update=output.states[layer + 1],
                    attention_weights=output.attention[layer],
                    gate=output.gates[layer][:, None, :].expand(
                        -1, output.states[layer].shape[1], -1
                    ),
                    metadata={"layer": layer, "state_location": "full_block_residual"},
                )
            )
        return captures

    def residual_pre(self, layer: int, token: int) -> torch.Tensor:
        return self._require_output().states[layer][:, token]

    def attention_candidate_update(self, layer: int, token: int) -> torch.Tensor:
        return self._require_output().attention_candidates[layer][:, token]

    def residual_after_attention(self, layer: int, token: int) -> torch.Tensor:
        return self._require_output().states_after_attention[layer][:, token]

    def ff_candidate_update(self, layer: int, token: int) -> torch.Tensor:
        return self._require_output().ff_candidates[layer][:, token]

    def residual_post(self, layer: int, token: int) -> torch.Tensor:
        return self._require_output().states[layer + 1][:, token]

    def attention_weights(self, layer: int, head: int, token: int) -> torch.Tensor:
        return self._require_output().attention[layer][:, head, token]

    def gate(self, layer: int, head: int | None, token: int) -> torch.Tensor:
        del head, token
        return self._require_output().gates[layer]

    def logits(self) -> torch.Tensor:
        return self._require_output().logits
