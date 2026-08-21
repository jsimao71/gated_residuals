"""Model-specific probes behind the architecture-independent capture interface."""

from .base import ModelProbeAdapter, assert_native_parity
from .qwen3_gated import (
    Qwen3AttentionAdapter,
    Qwen3GatedAttentionAdapter,
    extract_qwen3_headwise_gate,
)

__all__ = [
    "ModelProbeAdapter",
    "Qwen3AttentionAdapter",
    "Qwen3GatedAttentionAdapter",
    "assert_native_parity",
    "extract_qwen3_headwise_gate",
]
