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
- Paper 1's first E1--E5 cycle is complete: learned tasks showed distributed positive block
  utility, while strong interference, competence-controlled goal utility, selective gating, and a
  distinct future-utility-predicting goal state were not supported.
- E6 and E8 stopped at their preregistered evidence gates. E7 also stopped before a real pretrained
  parity forward because the exact checkpoint exceeded available host/device memory.
- Cycle B is active under a separately frozen post-Cycle-A work package. B1 completed the static
  residual atlas over all 30 stored Cycle A checkpoints with validated finite outputs. B2 then
  completed exact-parity SA/FF decomposition: seven of eight competent task--layer cells had
  replicated SA $>$ FF utility, but both sublayers were useful and no repair candidate appeared.

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

1. Run B3's token/depth stability, drift, subspace, and amplification--repair atlas.
2. Train the B4 depth series only after B3 instrumentation passes its analytic tests.
3. Freeze B5's competence-matched task ecology and effect thresholds before its new test split.
4. Reopen gating only if B2--B5 exposes reliable redundancy, negative utility, task-conditioned
   roles, repair, or depth slack.
5. Run the pinned pretrained pair only on hardware that completes real native-forward parity, then
   extend the same adapter semantics to Llama and Gemma if resource use is manageable.
6. Keep Papers 2--5 evidence-conditional; Cycle A through B2 do not support expanding their claims.
