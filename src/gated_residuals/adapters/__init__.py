"""Model-specific probes behind the architecture-independent capture interface."""

from .base import ModelProbeAdapter, assert_native_parity
from .qwen3_gated import (
    Qwen3AttentionAdapter,
    Qwen3GatedAttentionAdapter,
    Qwen3ResidualIntervention,
    extract_qwen3_headwise_gate,
)
from .tiny import TinyModelProbeAdapter

__all__ = [
    "ModelProbeAdapter",
    "Qwen3AttentionAdapter",
    "Qwen3GatedAttentionAdapter",
    "Qwen3ResidualIntervention",
    "TinyModelProbeAdapter",
    "assert_native_parity",
    "extract_qwen3_headwise_gate",
]
