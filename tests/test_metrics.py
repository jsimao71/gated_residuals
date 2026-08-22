import math

import pytest
import torch

from gated_residuals.attention_dilution import attention_metrics, dual_selection_regimes
from gated_residuals.causal_ablation import GateIntervention, intervene_gate
from gated_residuals.gate_metrics import gate_metric_correlations, gate_summary, token_autocorrelation
from gated_residuals.head_layer_similarity import head_layer_organization
from gated_residuals.residual_dynamics import (
    amplification_repair,
    cancellation_score,
    causal_block_utility,
    pairwise_layer_matrices,
    sa_ff_geometry,
    update_geometry,
)
from gated_residuals.temporal_stability import (
    autocorrelation,
    covariance_drift,
    eigenspectrum_drift,
    first_second_differences,
    lagged_cross_correlation,
    linear_cka,
    mean_shift,
    principal_subspace_drift,
    rbf_mmd,
    representational_similarity,
    rolling_moments,
    stable_rank,
    wasserstein_1d,
)


def test_residual_geometry_and_utility_are_analytic():
    state = torch.tensor([[[3.0, 4.0], [1.0, 0.0]]])
    update = torch.tensor([[[0.0, 5.0], [-1.0, 0.0]]])
    metrics = update_geometry(state, update)
    assert torch.allclose(metrics["residual_norm"], torch.tensor([[5.0, 1.0]]))
    assert torch.allclose(metrics["update_norm"], torch.tensor([[5.0, 1.0]]))
    assert torch.allclose(metrics["state_update_cosine"], torch.tensor([[0.8, -1.0]]))
    assert torch.allclose(metrics["novelty"], torch.tensor([[0.36, 0.0]]))
    assert torch.allclose(metrics["dominance"], torch.tensor([[0.5, 0.5]]))
    assert metrics["representation_direction_cosine"][0, 1] == pytest.approx(0.0)
    assert torch.allclose(
        causal_block_utility(torch.tensor([0.8, 0.4]), torch.tensor([0.7, 0.6])),
        torch.tensor([0.1, -0.2]),
    )


def test_cancellation_and_full_layer_matrices_are_analytic():
    first = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    assert torch.allclose(cancellation_score(first, first), torch.zeros(3))
    assert torch.allclose(cancellation_score(first, -first), torch.ones(3))
    states = torch.stack([first, first * 2.0])
    updates = torch.stack([first, -first])
    matrices = pairwise_layer_matrices(states, updates)
    assert set(matrices) == {
        "residual_state_cosine",
        "update_cosine",
        "update_cancellation",
        "residual_state_cka",
        "update_cka",
        "residual_state_rsa",
        "update_rsa",
    }
    assert matrices["residual_state_cosine"][0, 1] == pytest.approx(1.0)
    assert matrices["update_cosine"][0, 1] == pytest.approx(-1.0)
    assert matrices["update_cancellation"][0, 1] == pytest.approx(1.0)
    assert matrices["residual_state_cka"][0, 1] == pytest.approx(1.0)


def test_sa_ff_geometry_uses_distinct_pre_norm_locations():
    state = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    attention = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    ff = torch.tensor([[0.0, -1.0], [0.0, 1.0]])
    metrics = sa_ff_geometry(state, attention, ff)
    assert torch.allclose(metrics["attention_relative_update_norm"], torch.tensor([1.0, 0.5]))
    assert torch.allclose(
        metrics["ff_relative_update_norm"],
        torch.tensor([1 / math.sqrt(2), 1.0]),
    )
    assert torch.allclose(metrics["attention_ff_cosine"], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(metrics["attention_ff_cancellation"], torch.ones(2))
    assert torch.allclose(metrics["combined_update_norm"], torch.zeros(2))


def test_amplification_repair_requires_growth_then_recovery():
    event = amplification_repair(torch.tensor([1.0, 2.0, 4.0, 1.5]))
    assert event["detected"] is True
    assert event["peak_layer"] == 2
    assert event["repair_score"] == pytest.approx(0.625)
    assert amplification_repair(torch.tensor([1.0, 1.1, 1.2]))["detected"] is False


def test_attention_metrics_for_uniform_and_one_hot_rows():
    attention = torch.tensor([[[[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]]]])
    metrics = attention_metrics(attention, topk=2)
    assert metrics["attention_entropy"][0, 0, 0] == pytest.approx(math.log(4))
    assert metrics["attention_effective_support"][0, 0, 0] == pytest.approx(4.0)
    assert metrics["attention_topk_mass"][0, 0, 0] == pytest.approx(0.5)
    assert metrics["attention_entropy"][0, 0, 1] == pytest.approx(0.0)
    assert metrics["attention_sink_mass"][0, 0, 1] == pytest.approx(1.0)


def test_dual_selection_regimes_cover_all_observations():
    regimes = dual_selection_regimes(
        torch.tensor([0.1, 0.1, 1.0, 1.0]),
        torch.tensor([0.1, 0.9, 0.1, 0.9]),
        entropy_threshold=0.5,
    )
    assert all(value == pytest.approx(0.25) for value in regimes.values())
    assert sum(float(value) for value in regimes.values()) == pytest.approx(1.0)


def test_gate_metrics_and_interventions_preserve_dimensions():
    gate = torch.tensor([[[[0.1], [0.9]], [[0.2], [0.8]], [[0.3], [0.7]]]])
    summary = gate_summary(gate)
    assert summary["gate_mean"] == pytest.approx(0.5)
    assert summary["gate_fraction_below_0.5"] == pytest.approx(0.5)
    assert token_autocorrelation(gate, lag=1).shape == (1, 2, 1)
    for mode in GateIntervention:
        changed = intervene_gate(gate, mode, generator=torch.Generator().manual_seed(4))
        assert changed.shape == gate.shape
    assert torch.all(intervene_gate(gate, GateIntervention.FORCED_OPEN) == 1)
    assert torch.all(intervene_gate(gate, GateIntervention.FORCED_CLOSED) == 0)


def test_gate_correlations_identify_monotonic_relation():
    gate = torch.tensor([0.1, 0.2, 0.3, 0.4])
    correlations = gate_metric_correlations(gate, {"utility": gate * 2})
    assert correlations["pooled_pearson_gate_utility"] == pytest.approx(1.0)
    assert correlations["pooled_spearman_gate_utility"] == pytest.approx(1.0)


def test_temporal_statistics_known_cases():
    series = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
    moments = rolling_moments(series, 3)
    assert moments["mean"].shape == (1, 3, 2)
    assert torch.allclose(moments["mean"][0, :, 0], torch.tensor([2.0, 4.0, 6.0]))
    acf = autocorrelation(series, 2)
    assert torch.allclose(acf[..., 0], torch.ones_like(acf[..., 0]))
    matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert linear_cka(matrix, matrix) == pytest.approx(1.0)
    assert representational_similarity(matrix, matrix) == pytest.approx(1.0)
    assert covariance_drift(matrix, matrix) == pytest.approx(0.0)
    assert principal_subspace_drift(matrix, matrix, rank=2) == pytest.approx(0.0, abs=1e-4)


def test_extended_stability_metrics_known_cases():
    series = torch.arange(6, dtype=torch.float32)
    differences = first_second_differences(series)
    assert torch.allclose(differences["first_difference"], torch.ones(5))
    assert torch.allclose(differences["second_difference"], torch.zeros(4))
    lags, correlations = lagged_cross_correlation(series, series, 2)
    assert torch.equal(lags, torch.tensor([-2, -1, 0, 1, 2]))
    assert correlations[2] == pytest.approx(1.0)
    first = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    shifted = first + 1
    assert mean_shift(first, first) == pytest.approx(0.0)
    assert wasserstein_1d(series, series + 2) == pytest.approx(2.0)
    assert rbf_mmd(first, first) == pytest.approx(0.0)
    assert eigenspectrum_drift(first, first) == pytest.approx(0.0)
    assert stable_rank(first) == pytest.approx(1.0)
    assert mean_shift(first, shifted) > 0


def test_head_layer_organization_returns_finite_contrast():
    representations = torch.randn(3, 4, 5, 2, generator=torch.Generator().manual_seed(3))
    result = head_layer_organization(representations)
    assert set(result) == {
        "within_layer_head_similarity",
        "across_layer_corresponding_head_similarity",
        "within_minus_across_similarity",
    }
    assert all(torch.isfinite(value) for value in result.values())
