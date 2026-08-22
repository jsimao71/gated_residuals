"""Small causal residual Transformer variants used by experiments E1--E6."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class TinyOutput:
    logits: torch.Tensor
    states: list[torch.Tensor] = field(default_factory=list)
    candidates: list[torch.Tensor] = field(default_factory=list)
    effective_updates: list[torch.Tensor] = field(default_factory=list)
    attention_candidates: list[torch.Tensor] = field(default_factory=list)
    attention_effective_updates: list[torch.Tensor] = field(default_factory=list)
    states_after_attention: list[torch.Tensor] = field(default_factory=list)
    ff_candidates: list[torch.Tensor] = field(default_factory=list)
    ff_effective_updates: list[torch.Tensor] = field(default_factory=list)
    gates: list[torch.Tensor] = field(default_factory=list)
    attention_gates: list[torch.Tensor] = field(default_factory=list)
    ff_gates: list[torch.Tensor] = field(default_factory=list)
    attention: list[torch.Tensor] = field(default_factory=list)
    goal_states: list[torch.Tensor] = field(default_factory=list)
    active: list[torch.Tensor] = field(default_factory=list)


class CausalResidualBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Linear(width * mlp_ratio, width),
        )
        self.dropout = nn.Dropout(dropout)

    def attention_candidate(
        self, state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = state.size(1)
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=state.device), diagonal=1
        )
        normed = self.norm_attn(state)
        attention_output, weights = self.attention(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            key_padding_mask=~attention_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attention_output = self.dropout(attention_output)
        return attention_output, weights

    def ff_candidate(self, state_after_attention: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.mlp(self.norm_mlp(state_after_attention)))

    def components(
        self, state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return SA write, post-SA state, FF write, and attention probabilities."""
        attention_output, weights = self.attention_candidate(state, attention_mask)
        intermediate = state + attention_output
        mlp_output = self.ff_candidate(intermediate)
        return attention_output, intermediate, mlp_output, weights

    def candidate(self, state: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible combined block candidate."""
        attention_output, _, mlp_output, weights = self.components(state, attention_mask)
        return attention_output + mlp_output, weights


class TinyResidualDecoder(nn.Module):
    """Decoder with optional block gates and a shared, unsupervised goal state."""

    def __init__(
        self,
        vocab_size: int,
        *,
        width: int = 64,
        layers: int = 4,
        heads: int = 4,
        mlp_ratio: int = 2,
        max_length: int = 64,
        dropout: float = 0.0,
        variant: str = "baseline",
        goal_width: int = 24,
    ):
        super().__init__()
        if variant not in {
            "baseline",
            "matched_baseline",
            "static_scale",
            "gated",
            "sa_ff_gated",
            "goal",
            "goal_only",
            "goal_gated",
        }:
            raise ValueError(f"unknown tiny-model variant: {variant}")
        self.variant = variant
        self.width = width
        self.goal_width = goal_width
        self.token_embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList(
            [CausalResidualBlock(width, heads, mlp_ratio, dropout) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)
        self.has_goal = variant in {"goal", "goal_only", "goal_gated"}
        self.has_gate = variant in {"gated", "goal_gated"}
        self.has_static_scale = variant == "static_scale"
        if self.has_static_scale:
            # Identity initialization makes this an exact dense-model control at step zero.
            self.layer_scales = nn.Parameter(torch.ones(layers))
        if self.has_goal:
            self.goal_initial = nn.Parameter(torch.zeros(goal_width))
            self.goal_updates = nn.ModuleList(
                [nn.Linear(goal_width + width, goal_width) for _ in range(layers)]
            )
            self.goal_to_width = nn.Linear(goal_width, width)
        if self.has_gate:
            gate_input = goal_width if variant == "goal_gated" else width
            self.gates = nn.ModuleList([nn.Linear(gate_input, 1) for _ in range(layers)])
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, 2.0)
        self.has_component_gates = variant == "sa_ff_gated"
        if self.has_component_gates:
            self.attention_gates = nn.ModuleList([nn.Linear(width, 1) for _ in range(layers)])
            self.ff_gates = nn.ModuleList([nn.Linear(width, 1) for _ in range(layers)])
            for gate in (*self.attention_gates, *self.ff_gates):
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, 2.0)
        if variant == "matched_baseline":
            # Registered capacity control; excluded from forward computation by design.
            self.capacity_control = nn.ParameterList(
                [nn.Parameter(torch.zeros(width + 1)) for _ in range(layers)]
            )

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    @staticmethod
    def _last_token(state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        indices = attention_mask.sum(dim=1).sub(1).clamp_min(0)
        return state[torch.arange(state.size(0), device=state.device), indices]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        skip_layers: set[int] | None = None,
        skip_attention_layers: set[int] | None = None,
        skip_ff_layers: set[int] | None = None,
        random_attention_layers: set[int] | None = None,
        random_ff_layers: set[int] | None = None,
        intervention_seed: int = 0,
        gate_mode: str = "native",
        gate_overrides: list[torch.Tensor] | None = None,
        attention_gate_overrides: list[torch.Tensor] | None = None,
        ff_gate_overrides: list[torch.Tensor] | None = None,
        goal_mode: str = "native",
        hard_threshold: float | None = None,
        capture: bool = False,
    ) -> TinyOutput:
        skip_layers = skip_layers or set()
        skip_attention_layers = skip_attention_layers or set()
        skip_ff_layers = skip_ff_layers or set()
        random_attention_layers = random_attention_layers or set()
        random_ff_layers = random_ff_layers or set()
        if skip_attention_layers & random_attention_layers:
            raise ValueError("an attention write cannot be both skipped and randomly replaced")
        if skip_ff_layers & random_ff_layers:
            raise ValueError("an FF write cannot be both skipped and randomly replaced")
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        state = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        output = TinyOutput(logits=torch.empty(0, device=input_ids.device))
        goal = None
        if self.has_goal:
            goal = self.goal_initial[None].expand(input_ids.size(0), -1)
        if capture:
            output.states.append(state.detach())

        for layer_index, block in enumerate(self.blocks):
            residual_input = state
            summary = self._last_token(state, attention_mask)
            if self.has_goal:
                goal = goal + torch.tanh(self.goal_updates[layer_index](torch.cat([goal, summary], dim=-1)))
                if capture:
                    output.goal_states.append(goal.detach())
            attention_candidate, attention = block.attention_candidate(
                residual_input, attention_mask
            )
            attention_applied = attention_candidate
            if layer_index in skip_attention_layers:
                attention_applied = torch.zeros_like(attention_candidate)
            elif layer_index in random_attention_layers:
                attention_applied = self._norm_matched_random(
                    attention_candidate,
                    seed=int(intervention_seed) + 104729 * (layer_index + 1),
                )
            attention_gate = torch.ones((state.size(0), 1), device=state.device, dtype=state.dtype)
            if self.has_component_gates:
                attention_gate = torch.sigmoid(self.attention_gates[layer_index](summary))
                if gate_mode == "open": attention_gate = torch.ones_like(attention_gate)
                elif gate_mode == "closed": attention_gate = torch.zeros_like(attention_gate)
                elif gate_mode != "native": raise ValueError(f"unsupported gate mode: {gate_mode}")
                if attention_gate_overrides is not None:
                    attention_gate = attention_gate_overrides[layer_index].to(device=state.device, dtype=state.dtype)
                if layer_index in skip_layers: attention_gate = torch.zeros_like(attention_gate)
            attention_effective = attention_gate[:, None, :] * attention_applied
            state_after_attention = residual_input + attention_effective
            ff_candidate = block.ff_candidate(state_after_attention)
            ff_applied = ff_candidate
            if layer_index in skip_ff_layers:
                ff_applied = torch.zeros_like(ff_candidate)
            elif layer_index in random_ff_layers:
                ff_applied = self._norm_matched_random(
                    ff_candidate,
                    seed=int(intervention_seed) + 130363 * (layer_index + 1),
                )
            ff_gate = torch.ones((state.size(0), 1), device=state.device, dtype=state.dtype)
            if self.has_component_gates:
                ff_summary = self._last_token(state_after_attention, attention_mask)
                ff_gate = torch.sigmoid(self.ff_gates[layer_index](ff_summary))
                if gate_mode == "open": ff_gate = torch.ones_like(ff_gate)
                elif gate_mode == "closed": ff_gate = torch.zeros_like(ff_gate)
                elif gate_mode != "native": raise ValueError(f"unsupported gate mode: {gate_mode}")
                if ff_gate_overrides is not None:
                    ff_gate = ff_gate_overrides[layer_index].to(device=state.device, dtype=state.dtype)
                if layer_index in skip_layers: ff_gate = torch.zeros_like(ff_gate)
            ff_effective = ff_gate[:, None, :] * ff_applied
            candidate = attention_applied + ff_applied
            if self.has_gate:
                gate_source = goal if self.variant == "goal_gated" else summary
                if self.variant == "goal_gated" and goal_mode == "shuffled":
                    gate_source = gate_source.roll(1, dims=0)
                elif self.variant == "goal_gated" and goal_mode == "zero":
                    gate_source = torch.zeros_like(gate_source)
                gate = torch.sigmoid(self.gates[layer_index](gate_source))
            elif self.has_static_scale:
                gate = self.layer_scales[layer_index].expand(state.size(0), 1)
            else:
                gate = torch.ones((state.size(0), 1), device=state.device, dtype=state.dtype)
            if gate_mode == "open":
                gate = torch.ones_like(gate)
            elif gate_mode == "closed":
                gate = torch.zeros_like(gate)
            elif gate_mode != "native":
                raise ValueError(f"unsupported gate mode: {gate_mode}")
            if gate_overrides is not None:
                override = gate_overrides[layer_index].to(device=gate.device, dtype=gate.dtype)
                if override.shape != gate.shape:
                    raise ValueError(
                        f"gate override {layer_index} has shape {tuple(override.shape)}; "
                        f"expected {tuple(gate.shape)}"
                    )
                gate = override
            if layer_index in skip_layers:
                gate = torch.zeros_like(gate)
            active = torch.ones_like(gate, dtype=torch.bool)
            if hard_threshold is not None:
                active = gate >= hard_threshold
                # This implementation batches the candidate computation; active FLOPs are
                # theoretical. Realized conditional execution is measured separately.
                gate = active.to(gate.dtype)
            if self.has_component_gates:
                effective = attention_effective + ff_effective
                gate = (attention_gate + ff_gate) / 2
            else:
                effective = gate[:, None, :] * candidate
                attention_effective = gate[:, None, :] * attention_applied
                ff_effective = gate[:, None, :] * ff_applied
            state = residual_input + effective
            if capture:
                output.candidates.append(candidate.detach())
                output.effective_updates.append(effective.detach())
                output.attention_candidates.append(attention_candidate.detach())
                output.attention_effective_updates.append(attention_effective.detach())
                output.states_after_attention.append(state_after_attention.detach())
                output.ff_candidates.append(ff_candidate.detach())
                output.ff_effective_updates.append(ff_effective.detach())
                output.gates.append(gate.detach())
                output.attention_gates.append(attention_gate.detach())
                output.ff_gates.append(ff_gate.detach())
                output.attention.append(attention.detach())
                output.active.append(active.detach())
                output.states.append(state.detach())

        last = self._last_token(state, attention_mask)
        if self.has_goal:
            goal_features = self.goal_to_width(goal)
            if goal_mode == "shuffled":
                goal_features = goal_features.roll(1, dims=0)
            elif goal_mode == "zero":
                goal_features = torch.zeros_like(goal_features)
            elif goal_mode != "native":
                raise ValueError(f"unsupported goal mode: {goal_mode}")
            last = goal_features if self.variant == "goal_only" else last + goal_features
        output.logits = self.output(self.final_norm(last))
        return output

    @staticmethod
    def _norm_matched_random(update: torch.Tensor, *, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=update.device).manual_seed(seed)
        noise = torch.randn(
            update.shape,
            dtype=update.dtype,
            device=update.device,
            generator=generator,
        )
        update_norm = torch.linalg.vector_norm(update.float(), dim=-1, keepdim=True)
        noise_norm = torch.linalg.vector_norm(noise.float(), dim=-1, keepdim=True).clamp_min(1e-8)
        return noise * (update_norm / noise_norm).to(noise.dtype)
