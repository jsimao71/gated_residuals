# PRA common compatibility layer

This directory starts from the model-agnostic utilities in
`pdattention/src/common` (Apache-2.0) and keeps their public behavior local so
`gated_residuals` remains reproducible without a sibling-repository import.

Copied modules:

- `config.py`: YAML/config helpers;
- `metrics.py`: running, perplexity, gradient, throughput, and CUDA metrics;
- `recall_sparsity.py`: the exact PRA evidence-recall/materialization metric.

Architecture-specific residual, gate, and attention metrics live one package
level above and do not modify these copied utilities.
