import torch

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


def test_tiny_model_capture_and_skip_parity():
    model = TinyResidualDecoder(32, width=16, layers=3, heads=4, max_length=8)
    ids = torch.randint(1, 32, (2, 6))
    mask = torch.ones_like(ids, dtype=torch.bool)
    native = model(ids, mask, capture=True)
    skipped = model(ids, mask, skip_layers={1}, capture=True)
    assert native.logits.shape == (2, 32)
    assert len(native.states) == 4
    assert torch.equal(skipped.states[1], skipped.states[2])
    assert not torch.equal(native.logits, skipped.logits)


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
