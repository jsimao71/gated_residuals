"""Multiscale temporal statistics over token-indexed representations."""

from __future__ import annotations

import torch


def first_second_differences(
    series: torch.Tensor, *, token_dim: int = -1
) -> dict[str, torch.Tensor]:
    """Return first and second differences along an explicit token axis."""
    values = series.detach().float()
    if values.shape[token_dim] < 3:
        raise ValueError("need at least three tokens for second differences")
    return {
        "first_difference": torch.diff(values, dim=token_dim),
        "second_difference": torch.diff(values, n=2, dim=token_dim),
    }


def lagged_cross_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
    max_lag: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pearson cross-correlation for scalar series at lags ``[-max_lag, max_lag]``."""
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("cross-correlation expects matched one-dimensional series")
    if max_lag < 0 or max_lag >= left.numel() - 1:
        raise ValueError("max_lag must leave at least two paired observations")
    values = []
    lags = torch.arange(-max_lag, max_lag + 1)
    for lag in lags.tolist():
        if lag < 0:
            x, y = left[-lag:], right[:lag]
        elif lag > 0:
            x, y = left[:-lag], right[lag:]
        else:
            x, y = left, right
        x = x.float() - x.float().mean()
        y = y.float() - y.float().mean()
        denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
        values.append((x @ y) / denominator.clamp_min(1e-8))
    return lags, torch.stack(values)


def rolling_moments(series: torch.Tensor, window: int, *, token_dim: int = -2) -> dict[str, torch.Tensor]:
    """Rolling mean/variance/skewness/kurtosis and first/second mean differences."""
    values = series.detach().float().movedim(token_dim, -2)
    if window < 2 or window > values.shape[-2]:
        raise ValueError("window must be in [2, token_count]")
    windows = values.unfold(-2, window, 1).movedim(-1, -2)
    mean = windows.mean(dim=-2)
    centered = windows - mean.unsqueeze(-2)
    variance = centered.square().mean(dim=-2)
    std = variance.sqrt().clamp_min(1e-8)
    skewness = (centered.pow(3).mean(dim=-2)) / std.pow(3)
    kurtosis = (centered.pow(4).mean(dim=-2)) / std.pow(4) - 3.0
    first_difference = torch.diff(mean, dim=-2)
    second_difference = torch.diff(mean, n=2, dim=-2)
    return {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "mean_first_difference": first_difference,
        "mean_second_difference": second_difference,
    }


def autocorrelation(series: torch.Tensor, max_lag: int, *, token_dim: int = -2) -> torch.Tensor:
    """Normalized ACF, including lag zero, averaged across feature series."""
    values = series.detach().float().movedim(token_dim, -2)
    token_count = values.shape[-2]
    if max_lag < 0 or max_lag >= token_count:
        raise ValueError("max_lag must be in [0, token_count - 1]")
    centered = values - values.mean(dim=-2, keepdim=True)
    variance = centered.square().mean(dim=-2).clamp_min(1e-8)
    results = []
    for lag in range(max_lag + 1):
        if lag == 0:
            covariance = variance
        else:
            covariance = (centered[..., :-lag, :] * centered[..., lag:, :]).mean(dim=-2)
        results.append((covariance / variance).mean(dim=-1))
    return torch.stack(results, dim=-1)


def correlation_length(acf: torch.Tensor, *, threshold: float | None = None) -> torch.Tensor:
    """First non-zero lag where |ACF| drops below threshold (default 1/e)."""
    if acf.shape[-1] < 2:
        raise ValueError("ACF must contain lag zero and at least one positive lag")
    threshold = float(torch.exp(torch.tensor(-1.0))) if threshold is None else threshold
    below = acf[..., 1:].abs() < threshold
    first = below.float().argmax(dim=-1) + 1
    no_crossing = ~below.any(dim=-1)
    return torch.where(no_crossing, torch.full_like(first, acf.shape[-1] - 1), first)


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Linear centered-kernel alignment between matched observation matrices."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("CKA expects two [observation, feature] matrices")
    x = left.detach().float() - left.detach().float().mean(dim=0, keepdim=True)
    y = right.detach().float() - right.detach().float().mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x.transpose(0, 1) @ y).square()
    denominator = (
        torch.linalg.matrix_norm(x.transpose(0, 1) @ x)
        * torch.linalg.matrix_norm(y.transpose(0, 1) @ y)
    )
    return cross / denominator.clamp_min(1e-8)


def representational_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """RSA as Pearson correlation of upper-triangular cosine-similarity entries."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("RSA expects two [observation, feature] matrices")
    x = torch.nn.functional.normalize(left.detach().float(), dim=-1)
    y = torch.nn.functional.normalize(right.detach().float(), dim=-1)
    indices = torch.triu_indices(x.shape[0], x.shape[0], offset=1)
    if indices.shape[1] < 2:
        raise ValueError("RSA requires at least three observations")
    x_values = (x @ x.T)[indices[0], indices[1]]
    y_values = (y @ y.T)[indices[0], indices[1]]
    x_values -= x_values.mean()
    y_values -= y_values.mean()
    return (x_values @ y_values) / (
        torch.linalg.vector_norm(x_values) * torch.linalg.vector_norm(y_values)
    ).clamp_min(1e-8)


def covariance_drift(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Relative Frobenius change between feature covariance matrices."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("covariance drift expects [observation, feature] matrices")
    first_centered = first.float() - first.float().mean(dim=0)
    second_centered = second.float() - second.float().mean(dim=0)
    cov_first = first_centered.T @ first_centered / max(first.shape[0] - 1, 1)
    cov_second = second_centered.T @ second_centered / max(second.shape[0] - 1, 1)
    return torch.linalg.matrix_norm(cov_second - cov_first) / torch.linalg.matrix_norm(cov_first).clamp_min(1e-8)


def mean_shift(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Feature-mean displacement normalized by the first window's RMS scale."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("mean shift expects [observation, feature] matrices")
    displacement = torch.linalg.vector_norm(second.float().mean(0) - first.float().mean(0))
    scale = first.float().square().mean().sqrt().clamp_min(1e-8)
    return displacement / scale


def wasserstein_1d(first: torch.Tensor, second: torch.Tensor, *, quantiles: int = 101) -> torch.Tensor:
    """Empirical one-dimensional Wasserstein-1 distance on a common quantile grid."""
    if first.numel() < 2 or second.numel() < 2 or quantiles < 2:
        raise ValueError("Wasserstein distance requires two samples per window")
    grid = torch.linspace(0, 1, quantiles, device=first.device)
    return (
        torch.quantile(first.detach().float().flatten(), grid)
        - torch.quantile(second.detach().float().flatten(), grid)
    ).abs().mean()


def rbf_mmd(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    bandwidth: float | None = None,
) -> torch.Tensor:
    """Biased RBF maximum mean discrepancy for two observation matrices."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("MMD expects [observation, feature] matrices")
    joined = torch.cat([first.detach().float(), second.detach().float()])
    distances = torch.cdist(joined, joined).square()
    if bandwidth is None:
        positive = distances[distances > 0]
        bandwidth_t = positive.median() if positive.numel() else torch.tensor(1.0)
    else:
        bandwidth_t = torch.tensor(float(bandwidth), device=joined.device)
    bandwidth_t = bandwidth_t.clamp_min(1e-8)
    kernel = torch.exp(-distances / (2 * bandwidth_t))
    size_first = first.shape[0]
    return (
        kernel[:size_first, :size_first].mean()
        + kernel[size_first:, size_first:].mean()
        - 2 * kernel[:size_first, size_first:].mean()
    )


def eigenspectrum_drift(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """L1 drift between covariance eigenspectra normalized to unit mass."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("eigenspectrum drift expects [observation, feature] matrices")
    spectra = []
    for values in (first, second):
        centered = values.float() - values.float().mean(0)
        covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
        spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0)
        spectra.append(spectrum / spectrum.sum().clamp_min(1e-8))
    return (spectra[0] - spectra[1]).abs().sum()


def stable_rank(values: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm divided by squared spectral norm after centering."""
    if values.ndim != 2:
        raise ValueError("stable rank expects [observation, feature]")
    centered = values.float() - values.float().mean(0)
    singular = torch.linalg.svdvals(centered)
    return singular.square().sum() / singular.square().max().clamp_min(1e-8)


def principal_subspace_drift(first: torch.Tensor, second: torch.Tensor, rank: int) -> torch.Tensor:
    """Mean sine of principal angles between two centered feature subspaces."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("subspace drift expects [observation, feature] matrices")
    if rank <= 0 or rank > min(first.shape[1], first.shape[0], second.shape[0]):
        raise ValueError("invalid subspace rank")
    _, _, vh_first = torch.linalg.svd(first.float() - first.float().mean(dim=0), full_matrices=False)
    _, _, vh_second = torch.linalg.svd(second.float() - second.float().mean(dim=0), full_matrices=False)
    singular_values = torch.linalg.svdvals(vh_first[:rank] @ vh_second[:rank].T).clamp(0, 1)
    return torch.sqrt(1 - singular_values.square()).mean()
