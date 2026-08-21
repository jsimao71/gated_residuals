"""Non-invasive probe for the released Qwen3 gated-attention implementation.

Verified against qiuzh20/gated_attention commit
``f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2`` and Hugging Face revision
``aad415c45ec6b4fa727ef3ff3f4e9f62f958d49b`` (August 2026 audit).

The headwise implementation packs one gate logit per query head into ``q_proj``.
It applies ``sigmoid(gate_score)`` to the per-head SDPA output at shape
``[batch, token, head, head_dim]`` before flattening and ``o_proj``. Native hooks
do not alter that sequence. Counterfactual gate modes act only at the input to
``o_proj`` and are therefore experimental interventions, not parity probes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

from ..causal_ablation import GateIntervention, intervene_gate
from ..records import ProbeCapture
from .base import ModelProbeAdapter


OFFICIAL_CODE_REVISION = "f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2"
OFFICIAL_MODEL_REVISION = "aad415c45ec6b4fa727ef3ff3f4e9f62f958d49b"


def extract_qwen3_headwise_gate(q_projection: torch.Tensor, attention: Any) -> torch.Tensor:
    """Extract native ``g[b,t,h,1]`` from the release's packed query projection."""
    if not bool(getattr(attention, "headwise_attn_output_gate", False)):
        raise ValueError("attention module is not configured for headwise output gating")
    batch, tokens, _ = q_projection.shape
    kv_heads = int(attention.num_key_value_heads)
    groups = int(attention.num_key_value_groups)
    head_dim = int(attention.head_dim)
    packed = q_projection.view(batch, tokens, kv_heads, -1)
    query_width = head_dim * groups
    expected_width = query_width + groups
    if packed.shape[-1] != expected_width:
        raise ValueError(
            f"unexpected packed q_proj width {packed.shape[-1]}; expected {expected_width}"
        )
    _, gate_logits = torch.split(packed, [query_width, groups], dim=-1)
    return torch.sigmoid(gate_logits.reshape(batch, tokens, -1, 1))


class Qwen3AttentionAdapter(ModelProbeAdapter):
    """Capture attention writes from the released baseline or headwise-gated model.

    For the gated variant, raw SDPA head output is reconstructed from the
    pre-``o_proj`` effective output and native sigmoid gate. Reconstruction is
    valid for nonzero finite gates; values below ``minimum_gate`` are rejected
    because division would be unstable. In the baseline, candidate and effective
    updates are identical and the implicit gate is recorded as one.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        intervention: GateIntervention | str = GateIntervention.NATIVE,
        threshold: float = 0.5,
        minimum_gate: float = 0.0,
        detach_to_cpu: bool = True,
        generator: torch.Generator | None = None,
    ):
        self.model = model
        self.intervention = GateIntervention(intervention)
        self.threshold = float(threshold)
        self.minimum_gate = float(minimum_gate)
        self.detach_to_cpu = detach_to_cpu
        self.generator = generator
        self._layers = self._resolve_layers(model)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captures: list[ProbeCapture] = []
        self._state: dict[int, dict[str, torch.Tensor]] = {}

    @staticmethod
    def _resolve_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
        candidates = [
            getattr(getattr(model, "model", None), "layers", None),
            getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
            getattr(model, "layers", None),
        ]
        for candidate in candidates:
            if candidate is not None:
                layers = list(candidate)
                if layers and all(hasattr(layer, "self_attn") for layer in layers):
                    return layers
        raise TypeError("could not locate Qwen3 decoder layers with self_attn modules")

    def iter_layers(self) -> Iterable[torch.nn.Module]:
        return iter(self._layers)

    def captures(self) -> list[ProbeCapture]:
        return list(self._captures)

    def __enter__(self) -> "Qwen3AttentionAdapter":
        if self._handles:
            raise RuntimeError("adapter is already active")
        self._captures.clear()
        self._state.clear()
        for index, layer in enumerate(self._layers):
            attention = layer.self_attn
            is_headwise = bool(getattr(attention, "headwise_attn_output_gate", False))
            is_elementwise = bool(getattr(attention, "elementwise_attn_output_gate", False))
            if is_elementwise:
                raise ValueError("the Paper 1 adapter currently supports baseline and headwise variants")
            if not is_headwise and self.intervention is not GateIntervention.NATIVE:
                raise ValueError("gate interventions require the headwise-gated model")

            def layer_pre(module, args, layer_index=index):
                self._state.setdefault(layer_index, {})["residual_input"] = args[0]

            def q_hook(module, args, output, layer_index=index, attn=attention):
                if bool(getattr(attn, "headwise_attn_output_gate", False)):
                    self._state.setdefault(layer_index, {})["gate"] = extract_qwen3_headwise_gate(
                        output, attn
                    )

            def o_pre(module, args, layer_index=index, attn=attention):
                effective_flat = args[0]
                state = self._state.setdefault(layer_index, {})
                gate = state.get("gate")
                batch, tokens, _ = effective_flat.shape
                heads = int(attn.num_heads)
                head_dim = int(attn.head_dim)
                effective_heads = effective_flat.view(batch, tokens, heads, head_dim)
                if gate is None:
                    if bool(getattr(attn, "headwise_attn_output_gate", False)):
                        raise RuntimeError("q_proj hook did not run before o_proj")
                    gate = torch.ones(
                        batch,
                        tokens,
                        heads,
                        1,
                        dtype=effective_heads.dtype,
                        device=effective_heads.device,
                    )
                    state["gate"] = gate
                if torch.any(gate <= self.minimum_gate):
                    minimum = float(gate.min())
                    raise FloatingPointError(
                        f"cannot stably reconstruct candidate SDPA output; minimum gate={minimum:.3g}"
                    )
                candidate_heads = effective_heads / gate.to(effective_heads.dtype)
                counterfactual_gate = intervene_gate(
                    gate,
                    self.intervention,
                    threshold=self.threshold,
                    generator=self.generator,
                )
                intervened_heads = candidate_heads * counterfactual_gate.to(candidate_heads.dtype)
                intervened_flat = intervened_heads.reshape_as(effective_flat)
                state["candidate_heads"] = candidate_heads
                state["effective_heads"] = intervened_heads
                state["candidate_update"] = F.linear(candidate_heads.reshape_as(effective_flat), module.weight, module.bias)
                if self.intervention is GateIntervention.NATIVE:
                    return None
                return (intervened_flat, *args[1:])

            def o_hook(module, args, output, layer_index=index):
                self._state.setdefault(layer_index, {})["effective_update"] = output

            def attention_hook(module, args, output, layer_index=index):
                state = self._state[layer_index]
                residual = state["residual_input"]
                effective = state["effective_update"]
                attention_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                capture = ProbeCapture(
                    layer=layer_index,
                    residual_input=residual,
                    candidate_update=state["candidate_update"],
                    effective_update=effective,
                    residual_after_update=residual + effective,
                    gate=state["gate"],
                    attention_weights=attention_weights,
                    candidate_heads=state["candidate_heads"],
                    effective_heads=state["effective_heads"],
                    metadata={
                        "architecture": (
                            "Qwen3 headwise gated attention"
                            if bool(getattr(module, "headwise_attn_output_gate", False))
                            else "Qwen3 baseline attention"
                        ),
                        "model_variant": (
                            "headwise_gated"
                            if bool(getattr(module, "headwise_attn_output_gate", False))
                            else "baseline"
                        ),
                        "intervention": self.intervention.value,
                        "gate_location": "post-SDPA, pre-o_proj",
                        "gate_shape": "batch x token x query_head x 1",
                        "candidate_update_location": "post-o_proj reconstruction",
                        "effective_update_location": "post-o_proj attention residual write",
                        "official_code_revision": OFFICIAL_CODE_REVISION,
                        "official_model_revision": OFFICIAL_MODEL_REVISION,
                    },
                )
                capture.validate()
                self._captures.append(
                    capture.detached(cpu=self.detach_to_cpu) if self.detach_to_cpu else capture
                )

            self._handles.extend(
                [
                    layer.register_forward_pre_hook(layer_pre),
                    attention.q_proj.register_forward_hook(q_hook),
                    attention.o_proj.register_forward_pre_hook(o_pre),
                    attention.o_proj.register_forward_hook(o_hook),
                    attention.register_forward_hook(attention_hook),
                ]
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._state.clear()


# Backward-compatible descriptive name for code that only handles the gated variant.
Qwen3GatedAttentionAdapter = Qwen3AttentionAdapter
