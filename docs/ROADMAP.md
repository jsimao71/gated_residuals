# Residuals / Metrics Paper Line Roadmap

## Purpose

This repository should separate the **research-line position layer** from the **Paper 1 experimental layer**.

The existing repository already has a strong Paper 1 draft and a Paper 1 `AGENTS.md`. What is missing is a Paper 0 that defines the broader conceptual program, common metric ontology, and downstream paper series so that Codex does not accidentally make Paper 1 own all reusable abstractions.

## Current status

- Paper 0 now defines the program-level dual-selection framework and evidence ladder.
- Paper 1 is scoped as the E1--E3 experimental core, with E4--E8 secondary or conditional.
- The pretrained Qwen3 headwise gated-attention comparison is integrated as a controlled E7 probe.
- `docs/METRICS.md` defines the reusable ontology and interpretation limits.
- `src/gated_residuals` implements the initial architecture-neutral metric and adapter layer.

## Proposed series

### Paper 0 — Selective Computation and Residual Dynamics

Position + roadmap paper.

Defines the dual-selection framework:

\[
\text{available information}
\rightarrow
\text{memory selection}
\rightarrow
\text{candidate computation}
\rightarrow
\text{computational selection}
\rightarrow
\text{state update}.
\]

Establishes common metrics and the program-level distinction between memory selection, computational selection, state stability, interference, complementarity, and sufficient activation.

### Paper 1 — Selective Residual Computation

Existing draft.

Mechanism-first experiments on tiny Transformers, same-content/different-goal counterfactuals, residual interference, goal-conditioned gating, shared latent \(z\), and small pretrained gated-attention comparison.

### Paper 2 — Goal-Latent Computational Selection

Focuses on whether a distributed inferred state \(z\) predicts future computational utility.

Core question:

\[
z_t \rightarrow \widehat{U_l(\tau)}
\]

rather than merely classifying task labels.

### Paper 3 — Top-Down and Inter-Layer Modulation

Focuses on functional top-down control: higher/later/recurrent states modulate lower/local candidate transformations.

Core question:

\[
g_l^{(t+1)} = G(x_l^{(t+1)}, h_{>l}^{(t)}).
\]

### Paper 4 — Sufficient Activation and Adaptive Compute

Focuses on hard skipping, active FLOPs, latency, quality-cost Pareto fronts, and the computation-side version of sufficient activation.

Core question:

\[
\exists C_1 \subset C_2:\quad \mathrm{Cost}(C_1)<\mathrm{Cost}(C_2),\quad Q(C_1)>Q(C_2).
\]

### Paper 5 — Memory Selection × Computational Selection

Connects PRA and residual gating.

Core question:

\[
Q = Q(k_{\mathrm{memory}}, k_{\mathrm{compute}}).
\]

Tests whether broad memory activation plus computational gating improves recall without proportional residual interference.

## Immediate implementation priority

1. Complete the deterministic synthetic counterfactual generator and tiny decoder baseline.
2. Run E1 block-utility and repair measurements over at least five training seeds.
3. Freeze confirmatory thresholds and the final counterfactual-family test split.
4. Run E2 and the matched E3 gate/control matrix only after E1 instrumentation parity passes.
5. Treat Paper 2--5 drafts as roadmap placeholders until Paper 1 produces evidence.
