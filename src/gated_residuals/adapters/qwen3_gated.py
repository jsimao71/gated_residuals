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
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn.functional as F

from ..causal_ablation import GateIntervention, intervene_gate
from ..records import ProbeCapture
from .base import ModelProbeAdapter


OFFICIAL_CODE_REVISION = "f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2"
OFFICIAL_MODEL_REVISION = "aad415c45ec6b4fa727ef3ff3f4e9f62f958d49b"


def _attention_dimension(attention: Any, attribute: str, config_attribute: str) -> int:
    """Read dimensions from either the released fork or current Transformers."""
    value = getattr(attention, attribute, None)
    if value is None:
        value = getattr(getattr(attention, "config", None), config_attribute, None)
    if value is None:
        raise AttributeError(f"attention exposes neither {attribute} nor config.{config_attribute}")
    return int(value)


def _attention_dimensions(attention: Any) -> tuple[int, int, int, int]:
    heads = _attention_dimension(attention, "num_heads", "num_attention_heads")
    kv_heads = _attention_dimension(attention, "num_key_value_heads", "num_key_value_heads")
    groups = int(getattr(attention, "num_key_value_groups", heads // kv_heads))
    head_dim = _attention_dimension(attention, "head_dim", "head_dim")
    return heads, kv_heads, groups, head_dim


def extract_qwen3_headwise_gate(q_projection: torch.Tensor, attention: Any) -> torch.Tensor:
    """Extract native ``g[b,t,h,1]`` from the release's packed query projection."""
    if not bool(getattr(attention, "headwise_attn_output_gate", False)):
        raise ValueError("attention module is not configured for headwise output gating")
    batch, tokens, _ = q_projection.shape
    _, kv_heads, groups, head_dim = _attention_dimensions(attention)
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
        self._components: dict[int, dict[str, torch.Tensor]] = {}
        self._latest_logits: torch.Tensor | None = None

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
        self._components.clear()
        self._latest_logits = None
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
                heads, _, _, head_dim = _attention_dimensions(attn)
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

            def mlp_hook(module, args, output, layer_index=index):
                self._state.setdefault(layer_index, {})["ff_update"] = output

            def layer_hook(module, args, output, layer_index=index):
                state = self._state[layer_index]
                post = output[0] if isinstance(output, (tuple, list)) else output
                ff_update = state.get("ff_update", torch.zeros_like(state["effective_update"]))
                values = {
                    "residual_pre": state["residual_input"],
                    "attention_update": state["effective_update"],
                    "residual_after_attention": state["residual_input"] + state["effective_update"],
                    "ff_update": ff_update,
                    "residual_post": post,
                    "gate": state["gate"],
                }
                if self.detach_to_cpu:
                    values = {name: value.detach().cpu() for name, value in values.items()}
                self._components[layer_index] = values

            layer_handles = [
                layer.register_forward_pre_hook(layer_pre),
                attention.q_proj.register_forward_hook(q_hook),
                attention.o_proj.register_forward_pre_hook(o_pre),
                attention.o_proj.register_forward_hook(o_hook),
                attention.register_forward_hook(attention_hook),
                layer.register_forward_hook(layer_hook),
            ]
            if hasattr(layer, "mlp"):
                layer_handles.insert(-1, layer.mlp.register_forward_hook(mlp_hook))
            self._handles.extend(layer_handles)
        def model_hook(module, args, output):
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, torch.Tensor):
                logits = output
            elif isinstance(output, (tuple, list)) and output:
                logits = output[0]
            else:
                raise TypeError("cannot locate logits in model output")
            self._latest_logits = logits.detach().cpu() if self.detach_to_cpu else logits

        self._handles.append(self.model.register_forward_hook(model_hook))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._state.clear()

    def _component(self, layer: int, name: str, token: int) -> torch.Tensor:
        if layer not in self._components:
            raise RuntimeError("no instrumented forward has completed")
        return self._components[layer][name][:, token]

    def residual_pre(self, layer: int, token: int) -> torch.Tensor:
        return self._component(layer, "residual_pre", token)

    def attention_candidate_update(self, layer: int, token: int) -> torch.Tensor:
        capture = self._captures[layer]
        return capture.candidate_update[:, token]

    def residual_after_attention(self, layer: int, token: int) -> torch.Tensor:
        return self._component(layer, "residual_after_attention", token)

    def ff_candidate_update(self, layer: int, token: int) -> torch.Tensor:
        return self._component(layer, "ff_update", token)

    def residual_post(self, layer: int, token: int) -> torch.Tensor:
        return self._component(layer, "residual_post", token)

    def attention_weights(self, layer: int, head: int, token: int) -> torch.Tensor:
        weights = self._captures[layer].attention_weights
        if weights is None:
            raise RuntimeError("attention weights were not requested by the model forward")
        return weights[:, head, token]

    def gate(self, layer: int, head: int, token: int) -> torch.Tensor:
        gate = self._captures[layer].gate
        if gate is None:
            raise RuntimeError("no gate was captured")
        return gate[:, token, head]

    def logits(self) -> torch.Tensor:
        if self._latest_logits is None:
            raise RuntimeError("no instrumented forward has completed")
        return self._latest_logits


# Backward-compatible descriptive name for code that only handles the gated variant.
Qwen3GatedAttentionAdapter = Qwen3AttentionAdapter


class Qwen3ResidualIntervention(AbstractContextManager["Qwen3ResidualIntervention"]):
    """Explicitly zero one Qwen attention/FF write or bypass one full block."""

    MODES = {"skip_attention", "skip_ff", "skip_block"}

    def __init__(self, model: torch.nn.Module, *, layer: int, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"unsupported residual intervention: {mode}")
        self.model = model
        self.layer_index = int(layer)
        self.mode = mode
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._block_input: torch.Tensor | None = None

    def __enter__(self) -> "Qwen3ResidualIntervention":
        layers = Qwen3AttentionAdapter._resolve_layers(self.model)
        if not 0 <= self.layer_index < len(layers):
            raise IndexError(f"layer {self.layer_index} outside [0, {len(layers)})")
        layer = layers[self.layer_index]

        def zero_output(module, args, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]), *output[1:])
            if isinstance(output, list):
                return [torch.zeros_like(output[0]), *output[1:]]
            return torch.zeros_like(output)

        if self.mode == "skip_attention":
            self._handles.append(layer.self_attn.register_forward_hook(zero_output))
        elif self.mode == "skip_ff":
            if not hasattr(layer, "mlp"):
                raise TypeError("decoder layer does not expose an mlp sublayer")
            self._handles.append(layer.mlp.register_forward_hook(zero_output))
        else:
            def remember_input(module, args):
                if not args:
                    raise RuntimeError("block intervention requires positional hidden states")
                self._block_input = args[0]

            def bypass_block(module, args, output):
                if self._block_input is None:
                    raise RuntimeError("block input hook did not run")
                if isinstance(output, tuple):
                    return (self._block_input, *output[1:])
                if isinstance(output, list):
                    return [self._block_input, *output[1:]]
                return self._block_input

            self._handles.extend(
                [
                    layer.register_forward_pre_hook(remember_input),
                    layer.register_forward_hook(bypass_block),
                ]
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._block_input = None
