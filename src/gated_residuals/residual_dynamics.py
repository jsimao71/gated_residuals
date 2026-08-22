"""Residual geometry and causal/temporal evidence for update roles."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .records import ProbeCapture


def _require_same_shape(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError(f"tensor locations differ: {tuple(left.shape)} != {tuple(right.shape)}")
    if left.ndim < 1:
        raise ValueError("expected a feature dimension")


def update_geometry(
    state: torch.Tensor,
    update: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute per-observation geometry over the last (feature) dimension."""
    _require_same_shape(state, update)
    state_f = state.float()
    update_f = update.float()
    state_norm = torch.linalg.vector_norm(state_f, dim=-1)
    update_norm = torch.linalg.vector_norm(update_f, dim=-1)
    cosine = F.cosine_similarity(state_f, update_f, dim=-1, eps=eps)
    post_state = state_f + update_f
    return {
        "residual_norm": state_norm,
        "update_norm": update_norm,
        "relative_update_norm": update_norm / state_norm.clamp_min(eps),
        "state_update_cosine": cosine,
        "novelty": (1.0 - cosine.square()).clamp(0.0, 1.0),
        "dominance": update_norm / (state_norm + update_norm).clamp_min(eps),
        "effective_displacement": torch.linalg.vector_norm(post_state - state_f, dim=-1),
        "representation_displacement": torch.linalg.vector_norm(post_state - state_f, dim=-1),
        "representation_direction_cosine": F.cosine_similarity(
            state_f, post_state, dim=-1, eps=eps
        ),
    }


def cancellation_score(
    first_update: torch.Tensor,
    second_update: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalized cancellation of two writes without assigning causal interference.

    Zero indicates exact reinforcement for positively collinear writes, one indicates
    exact cancellation for equal and opposite writes, and intermediate values describe
    geometry only.
    """
    _require_same_shape(first_update, second_update)
    first = first_update.float()
    second = second_update.float()
    numerator = torch.linalg.vector_norm(first + second, dim=-1)
    denominator = (
        torch.linalg.vector_norm(first, dim=-1)
        + torch.linalg.vector_norm(second, dim=-1)
    ).clamp_min(eps)
    return (1.0 - numerator / denominator).clamp(0.0, 1.0)


def sa_ff_geometry(
    residual_pre: torch.Tensor,
    attention_update: torch.Tensor,
    ff_update: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compare pre-norm SA and FF writes at their correct residual locations."""
    _require_same_shape(residual_pre, attention_update)
    _require_same_shape(residual_pre, ff_update)
    state_after_attention = residual_pre.float() + attention_update.float()
    attention = update_geometry(residual_pre, attention_update, eps=eps)
    ff = update_geometry(state_after_attention, ff_update, eps=eps)
    attention_f = attention_update.float()
    ff_f = ff_update.float()
    combined = attention_f + ff_f
    return {
        "attention_update_norm": attention["update_norm"],
        "attention_relative_update_norm": attention["relative_update_norm"],
        "attention_state_update_cosine": attention["state_update_cosine"],
        "attention_novelty": attention["novelty"],
        "attention_dominance": attention["dominance"],
        "ff_update_norm": ff["update_norm"],
        "ff_relative_update_norm": ff["relative_update_norm"],
        "ff_state_update_cosine": ff["state_update_cosine"],
        "ff_novelty": ff["novelty"],
        "ff_dominance": ff["dominance"],
        "attention_ff_cosine": F.cosine_similarity(attention_f, ff_f, dim=-1, eps=eps),
        "attention_ff_cancellation": cancellation_score(attention_f, ff_f, eps=eps),
        "combined_update_norm": torch.linalg.vector_norm(combined, dim=-1),
        "combined_vs_sum_norm": torch.linalg.vector_norm(combined, dim=-1)
        / (
            torch.linalg.vector_norm(attention_f, dim=-1)
            + torch.linalg.vector_norm(ff_f, dim=-1)
        ).clamp_min(eps),
    }


def pairwise_layer_matrices(
    residual_states: torch.Tensor,
    updates: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Full residual/update geometry, CKA, and RSA matrices across layers.

    Inputs use ``[layer, observation, feature]``. CKA and RSA compare matched
    observations and are intentionally kept separate from causal utility.
    """
    if residual_states.ndim != 3 or updates.ndim != 3:
        raise ValueError("layer matrices expect [layer, observation, feature]")
    if residual_states.shape != updates.shape:
        raise ValueError("residual states and updates must have identical shapes")
    if residual_states.shape[0] < 2 or residual_states.shape[1] < 3:
        raise ValueError("need at least two layers and three observations")

    from .temporal_stability import linear_cka, representational_similarity

    layers = residual_states.shape[0]
    matrices = {
        name: torch.empty((layers, layers), dtype=torch.float32)
        for name in (
            "residual_state_cosine",
            "update_cosine",
            "update_cancellation",
            "residual_state_cka",
            "update_cka",
            "residual_state_rsa",
            "update_rsa",
        )
    }
    states = residual_states.detach().float()
    writes = updates.detach().float()
    for left in range(layers):
        for right in range(layers):
            matrices["residual_state_cosine"][left, right] = F.cosine_similarity(
                states[left], states[right], dim=-1, eps=eps
            ).mean()
            matrices["update_cosine"][left, right] = F.cosine_similarity(
                writes[left], writes[right], dim=-1, eps=eps
            ).mean()
            matrices["update_cancellation"][left, right] = cancellation_score(
                writes[left], writes[right], eps=eps
            ).mean()
            matrices["residual_state_cka"][left, right] = linear_cka(
                states[left], states[right]
            )
            matrices["update_cka"][left, right] = linear_cka(
                writes[left], writes[right]
            )
            matrices["residual_state_rsa"][left, right] = representational_similarity(
                states[left], states[right]
            )
            matrices["update_rsa"][left, right] = representational_similarity(
                writes[left], writes[right]
            )
    if not all(torch.isfinite(matrix).all() for matrix in matrices.values()):
        raise RuntimeError("non-finite value in pairwise layer matrices")
    return matrices


def paired_update_geometry(capture: ProbeCapture) -> dict[str, torch.Tensor]:
    """Compare raw candidate and effective updates without assigning mechanism labels."""
    capture.validate()
    candidate = update_geometry(capture.residual_input, capture.candidate_update)
    effective = update_geometry(capture.residual_input, capture.effective_update)
    output: dict[str, torch.Tensor] = {}
    for prefix, values in (("candidate", candidate), ("effective", effective)):
        for name, value in values.items():
            output[f"{prefix}_{name}"] = value
    output["gate_attenuation"] = (
        candidate["update_norm"] - effective["update_norm"]
    ) / candidate["update_norm"].clamp_min(1e-8)
    output["observed_residual_error"] = torch.linalg.vector_norm(
        capture.residual_after_update.float()
        - (capture.residual_input.float() + capture.effective_update.float()),
        dim=-1,
    )
    return output


def causal_block_utility(full_quality: torch.Tensor, skipped_quality: torch.Tensor) -> torch.Tensor:
    """Return U_l = Q(full) - Q(skip_l); negative values are candidate harmful effects."""
    if full_quality.shape != skipped_quality.shape:
        raise ValueError("full and skipped quality observations must be paired")
    return full_quality.float() - skipped_quality.float()


def bootstrap_mean_interval(
    values: torch.Tensor,
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    generator: torch.Generator | None = None,
) -> tuple[float, float, float]:
    """Example-level nonparametric bootstrap interval for a scalar effect."""
    flat = values.detach().float().flatten()
    if not 0 < confidence < 1 or samples <= 0 or flat.numel() < 2:
        raise ValueError("need >=2 values, positive samples, and confidence in (0, 1)")
    indices = torch.randint(flat.numel(), (samples, flat.numel()), generator=generator)
    means = flat[indices].mean(dim=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = torch.quantile(means, torch.tensor([alpha, 1.0 - alpha]))
    return float(flat.mean()), float(low), float(high)


def amplification_repair(
    divergence_by_layer: torch.Tensor,
    *,
    amplification_ratio: float = 1.25,
    repair_fraction: float = 0.25,
) -> dict[str, torch.Tensor | bool | int | float]:
    """Detect whether a trajectory first amplifies and later repairs a perturbation.

    Input is a one-dimensional non-negative distance from a matched reference trajectory.
    This is temporal evidence only; combine it with negative causal utility and replication
    before using the term strong interference.
    """
    distances = divergence_by_layer.detach().float().flatten()
    if distances.numel() < 3 or torch.any(distances < 0) or not torch.isfinite(distances).all():
        raise ValueError("need at least three finite non-negative layer divergences")
    peak_value, peak_index_t = torch.max(distances, dim=0)
    peak_index = int(peak_index_t)
    baseline = float(distances[0])
    final = float(distances[-1])
    peak = float(peak_value)
    amplified = peak >= max(baseline * amplification_ratio, baseline + 1e-8)
    repaired = peak_index < distances.numel() - 1 and final <= peak * (1.0 - repair_fraction)
    return {
        "detected": bool(amplified and repaired),
        "peak_layer": peak_index,
        "initial_divergence": baseline,
        "peak_divergence": peak,
        "final_divergence": final,
        "repair_score": (peak - final) / max(peak, 1e-8),
        "trajectory": distances,
    }


def strong_interference_supported(
    utilities_by_seed: torch.Tensor,
    repair_detected_by_seed: torch.Tensor,
    *,
    minimum_seed_fraction: float = 0.8,
) -> bool:
    """Apply the project's three-part criterion at seed-summary level."""
    utilities = utilities_by_seed.detach().float().flatten()
    repairs = repair_detected_by_seed.detach().bool().flatten()
    if utilities.shape != repairs.shape or utilities.numel() < 2:
        raise ValueError("utility and repair must contain paired values for at least two seeds")
    replicated = (utilities < 0) & repairs
    return bool(replicated.float().mean() >= minimum_seed_fraction)
