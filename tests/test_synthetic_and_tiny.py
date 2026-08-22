import torch

from gated_residuals.adapters import TinyModelProbeAdapter, assert_native_parity
from gated_residuals.synthetic import build_splits
from gated_residuals.tiny_model import TinyResidualDecoder


CONFIG = {
    "data": {
        "generation_seed": 9,
        "train_families": 4,
        "val_families": 2,
        "test_families": 2,
        "distractor_probability": 0.5,
    }
}


def test_counterfactual_generation_is_deterministic_and_content_matched():
    first = build_splits(CONFIG)
    second = build_splits(CONFIG)
    assert first == second
    family = first["test"][:3]
    assert len({example.content for example in family}) == 1
    assert len({example.intent for example in family}) == 3
    train_families = {example.family_id for example in first["train"]}
    test_families = {example.family_id for example in first["test"]}
    assert train_families.isdisjoint(test_families)


def test_task_subset_preserves_counterfactual_content_matching():
    config = {"data": {**CONFIG["data"], "intents": ["maximum", "minimum"]}}
    splits = build_splits(config)
    assert {example.intent for example in splits["train"]} == {"maximum", "minimum"}
    assert len(splits["test"]) == 4


def test_tiny_model_capture_and_skip_parity():
    model = TinyResidualDecoder(32, width=16, layers=3, heads=4, max_length=8)
    ids = torch.randint(1, 32, (2, 6))
    mask = torch.ones_like(ids, dtype=torch.bool)
    native = model(ids, mask, capture=True)
    skipped = model(ids, mask, skip_layers={1}, capture=True)
    assert native.logits.shape == (2, 32)
    assert len(native.states) == 4
    assert len(native.attention_candidates) == 3
    assert len(native.ff_candidates) == 3
    for layer in range(model.num_layers):
        torch.testing.assert_close(
            native.states_after_attention[layer],
            native.states[layer] + native.attention_candidates[layer],
        )
        torch.testing.assert_close(
            native.candidates[layer],
            native.attention_candidates[layer] + native.ff_candidates[layer],
        )
        torch.testing.assert_close(
            native.states[layer + 1], native.states[layer] + native.effective_updates[layer]
        )
    assert torch.equal(skipped.states[1], skipped.states[2])
    assert not torch.equal(native.logits, skipped.logits)


def test_tiny_adapter_exact_native_parity_and_semantic_locations():
    model = TinyResidualDecoder(32, width=16, layers=2, heads=4, max_length=8)
    ids = torch.randint(1, 32, (2, 6), generator=torch.Generator().manual_seed(7))
    mask = torch.ones_like(ids, dtype=torch.bool)
    adapter = TinyModelProbeAdapter(model)
    assert assert_native_parity(
        model, adapter, {"input_ids": ids, "attention_mask": mask}, atol=0.0, rtol=0.0
    ) == 0.0
    captures = adapter.captures()
    assert len(captures) == 2
    for capture in captures:
        capture.validate()
    torch.testing.assert_close(
        adapter.residual_after_attention(0, 3),
        adapter.residual_pre(0, 3) + adapter.attention_candidate_update(0, 3),
    )
    torch.testing.assert_close(adapter.logits(), model(ids, mask).logits)


def test_sa_ff_interventions_and_norm_matched_replacements():
    model = TinyResidualDecoder(32, width=16, layers=2, heads=4, max_length=8)
    ids = torch.randint(1, 32, (2, 6), generator=torch.Generator().manual_seed(8))
    mask = torch.ones_like(ids, dtype=torch.bool)
    native = model(ids, mask, capture=True)
    skip_sa = model(ids, mask, skip_attention_layers={0}, capture=True)
    zero_sa = model(ids, mask, skip_attention_layers={0}, capture=True)
    skip_ff = model(ids, mask, skip_ff_layers={0}, capture=True)
    torch.testing.assert_close(skip_sa.logits, zero_sa.logits, rtol=0, atol=0)
    assert torch.count_nonzero(skip_sa.attention_effective_updates[0]) == 0
    assert torch.count_nonzero(skip_ff.ff_effective_updates[0]) == 0
    assert not torch.equal(native.logits, skip_sa.logits)
    assert not torch.equal(native.logits, skip_ff.logits)

    random_sa = model(
        ids, mask, random_attention_layers={0}, intervention_seed=19, capture=True
    )
    random_ff = model(ids, mask, random_ff_layers={0}, intervention_seed=23, capture=True)
    torch.testing.assert_close(
        torch.linalg.vector_norm(random_sa.attention_effective_updates[0], dim=-1),
        torch.linalg.vector_norm(native.attention_candidates[0], dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(random_ff.ff_effective_updates[0], dim=-1),
        torch.linalg.vector_norm(random_ff.ff_candidates[0], dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_forced_open_is_exact_baseline_for_gated_model_at_initialization():
    model = TinyResidualDecoder(32, width=16, layers=2, heads=4, max_length=8, variant="gated")
    ids = torch.randint(1, 32, (2, 6))
    mask = torch.ones_like(ids, dtype=torch.bool)
    first = model(ids, mask, gate_mode="open").logits
    second = model(ids, mask, gate_mode="open", capture=True).logits
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_gate_override_can_close_one_layer_without_changing_other_shapes():
    model = TinyResidualDecoder(32, width=16, layers=2, heads=4, max_length=8, variant="gated")
    ids = torch.randint(1, 32, (2, 6))
    mask = torch.ones_like(ids, dtype=torch.bool)
    native = model(ids, mask, capture=True)
    overrides = [native.gates[0], torch.zeros_like(native.gates[1])]
    closed = model(ids, mask, gate_overrides=overrides, capture=True)
    torch.testing.assert_close(closed.states[1], native.states[1])
    torch.testing.assert_close(closed.states[2], closed.states[1])


def test_goal_shuffle_is_an_explicit_intervention():
    model = TinyResidualDecoder(32, width=16, layers=2, heads=4, max_length=8, variant="goal")
    ids = torch.randint(1, 32, (3, 6))
    mask = torch.ones_like(ids, dtype=torch.bool)
    native = model(ids, mask).logits
    shuffled = model(ids, mask, goal_mode="shuffled").logits
    assert native.shape == shuffled.shape
    assert not torch.equal(native, shuffled)
