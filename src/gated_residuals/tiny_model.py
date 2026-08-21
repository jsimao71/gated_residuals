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
    gates: list[torch.Tensor] = field(default_factory=list)
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

    def candidate(self, state: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        intermediate = state + attention_output
        mlp_output = self.dropout(self.mlp(self.norm_mlp(intermediate)))
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
        if variant not in {"baseline", "matched_baseline", "gated", "goal", "goal_only", "goal_gated"}:
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
        gate_mode: str = "native",
        hard_threshold: float | None = None,
        capture: bool = False,
    ) -> TinyOutput:
        skip_layers = skip_layers or set()
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
            candidate, attention = block.candidate(residual_input, attention_mask)
            if self.has_gate:
                gate_source = goal if self.variant == "goal_gated" else summary
                gate = torch.sigmoid(self.gates[layer_index](gate_source))
            else:
                gate = torch.ones((state.size(0), 1), device=state.device, dtype=state.dtype)
            if gate_mode == "open":
                gate = torch.ones_like(gate)
            elif gate_mode == "closed":
                gate = torch.zeros_like(gate)
            elif gate_mode != "native":
                raise ValueError(f"unsupported gate mode: {gate_mode}")
            if layer_index in skip_layers:
                gate = torch.zeros_like(gate)
            active = torch.ones_like(gate, dtype=torch.bool)
            if hard_threshold is not None:
                active = gate >= hard_threshold
                # This implementation batches the candidate computation; active FLOPs are
                # theoretical. Realized conditional execution is measured separately.
                gate = active.to(gate.dtype)
            effective = gate[:, None, :] * candidate
            state = residual_input + effective
            if capture:
                output.candidates.append(candidate.detach())
                output.effective_updates.append(effective.detach())
                output.gates.append(gate.detach())
                output.attention.append(attention.detach())
                output.active.append(active.detach())
                output.states.append(state.detach())

        last = self._last_token(state, attention_mask)
        if self.has_goal:
            goal_features = self.goal_to_width(goal)
            last = goal_features if self.variant == "goal_only" else last + goal_features
        output.logits = self.output(self.final_norm(last))
        return output
