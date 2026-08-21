"""Small systems-measurement helpers for active-compute claims."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class WallClockMeasurement:
    elapsed_seconds: float
    examples_per_second: float
    tokens_per_second: float
    cuda_peak_memory_mib: float


class InferenceTimer:
    """Synchronizing wall-clock timer; use repeated warm runs for paper results."""

    def __init__(self, device: torch.device | str, *, examples: int, tokens: int):
        self.device = torch.device(device)
        self.examples = int(examples)
        self.tokens = int(tokens)
        self.started = 0.0
        self.measurement: WallClockMeasurement | None = None

    def __enter__(self) -> "InferenceTimer":
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        peak = (
            torch.cuda.max_memory_allocated(self.device) / (1024**2)
            if self.device.type == "cuda"
            else 0.0
        )
        self.measurement = WallClockMeasurement(
            elapsed_seconds=elapsed,
            examples_per_second=self.examples / elapsed,
            tokens_per_second=self.tokens / elapsed,
            cuda_peak_memory_mib=float(peak),
        )


def active_fraction(decisions: torch.Tensor) -> torch.Tensor:
    if decisions.dtype != torch.bool:
        decisions = decisions != 0
    return decisions.float().mean()


def active_flops(dense_flops: float, decisions: torch.Tensor) -> float:
    """Logical FLOPs estimate; it is not evidence of realized latency."""
    return float(dense_flops) * float(active_fraction(decisions))
