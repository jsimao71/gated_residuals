from types import SimpleNamespace

import torch

from gated_residuals.adapters import Qwen3AttentionAdapter, Qwen3GatedAttentionAdapter, assert_native_parity
from gated_residuals.causal_ablation import GateIntervention


class FakeOfficialAttention(torch.nn.Module):
    """Small module with the release's packed-q and pre-o_proj headwise gate sequence."""

    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.num_key_value_heads = 1
        self.num_key_value_groups = 2
        self.head_dim = 2
        self.headwise_attn_output_gate = True
        self.q_proj = torch.nn.Linear(4, 6, bias=False)
        self.o_proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.q_proj.weight.zero_()
            self.o_proj.weight.copy_(torch.eye(4))

    def forward(self, hidden_states):
        batch, tokens, _ = hidden_states.shape
        packed = self.q_proj(hidden_states).view(batch, tokens, 1, 6)
        _, logits = torch.split(packed, [4, 2], dim=-1)
        gate = torch.sigmoid(logits.reshape(batch, tokens, 2, 1))
        candidate = hidden_states.view(batch, tokens, 2, 2)
        effective = candidate * gate
        return self.o_proj(effective.reshape(batch, tokens, 4)), None, None


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeOfficialAttention()

    def forward(self, hidden_states):
        update, _, _ = self.self_attn(hidden_states)
        return hidden_states + update


class FakeQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([FakeLayer(), FakeLayer()]))
        # Register the layers despite the SimpleNamespace model layout.
        self.layers = self.model.layers

    def forward(self, input_ids):
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class FakeBaselineAttention(FakeOfficialAttention):
    def __init__(self):
        super().__init__()
        self.headwise_attn_output_gate = False
        self.q_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states):
        self.q_proj(hidden_states)
        return self.o_proj(hidden_states), None, None


def test_native_adapter_has_exact_parity_and_captures_both_update_locations():
    model = FakeQwen()
    inputs = {"input_ids": torch.tensor([[0, 1, 2]])}
    adapter = Qwen3GatedAttentionAdapter(model)
    assert assert_native_parity(model, adapter, inputs) == 0.0
    captures = adapter.captures()
    assert len(captures) == 2
    for capture in captures:
        assert capture.gate.shape == (1, 3, 2, 1)
        assert torch.all(capture.gate == 0.5)
        assert torch.allclose(capture.effective_update, 0.5 * capture.candidate_update)
        assert capture.metadata["gate_location"] == "post-SDPA, pre-o_proj"


def test_forced_open_is_an_explicit_non_native_intervention():
    model = FakeQwen()
    inputs = {"input_ids": torch.tensor([[0, 1, 2]])}
    native = model(**inputs)
    adapter = Qwen3GatedAttentionAdapter(model, intervention=GateIntervention.FORCED_OPEN)
    with adapter:
        forced_open = model(**inputs)
    assert not torch.equal(native, forced_open)
    assert all(capture.metadata["intervention"] == "forced_open" for capture in adapter.captures())


def test_same_adapter_captures_released_baseline_semantics():
    model = FakeQwen()
    for layer in model.layers:
        layer.self_attn = FakeBaselineAttention()
    inputs = {"input_ids": torch.tensor([[0, 1, 2]])}
    adapter = Qwen3AttentionAdapter(model)
    assert assert_native_parity(model, adapter, inputs) == 0.0
    for capture in adapter.captures():
        assert torch.all(capture.gate == 1)
        assert torch.equal(capture.candidate_update, capture.effective_update)
        assert capture.metadata["model_variant"] == "baseline"
