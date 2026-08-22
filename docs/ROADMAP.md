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
- B3 completed the token/depth stability and causal repair atlas. Late FF-norm correctness effects
  replicated exploratorily, but zero of 12,150 perturbations met joint vector and task-gap recovery.
- B4 completed 25 depth/capacity runs. The effective-block fraction fell with depth but failed the
  strict competence-preservation rule; two competent-task FF cells were reliably negative and one
  late SA cell was reliably near zero. This conditionally opens B6 after B5 ecology validation,
  without supporting repair or strong interference.
- B5 completed a five-task, five-seed, competence-matched ecology after documented validation
  pilots. All ten task pairs matched; 13 task-conditioned block/SA utility contrasts replicated,
  and minimum layer 8 had negative block/FF utility. B6 is evidence-enabled, while repair and
  strong interference remain unsupported.
- B6 completed five whole-block and five SA/FF-gated deep runs. Native quality matched baseline;
  forced-open/mean/shuffled interventions showed functional modulation, but magnitude predicted
  gates better than causal utility, the negative cell persisted, and soft gates saved no compute.
- B7 validated the common adapter with exact parity on cached Qwen3-0.6B and ran the full 28-layer
  residual, attention, and SA/FF/block-ablation probe on 24 predictable sequences. Standard-Qwen
  results showed positive mean SA/block utility at every layer and no confidence-supported negative
  FF layer. The exact pinned gated release remained outside measured memory headroom, so no gated
  examples or gate metrics exist and the conditional B8/B9 gate remains closed.
- B8 recorded its conditional Llama-family stop: no checkpoint selection, download, model forward,
  or metric collection was permitted because B7 did not complete the exact gated comparison.
- B9 recorded the downstream Gemma-family stop. With B7 incomplete end to end and B8 containing no
  model result, no Gemma checkpoint or cross-family metric was permitted. Cycle B1--B9 is fully
  accounted for, including its conditional stops.
- Cycle C reframes Paper 1 around learnability and task-conditioned allocation. C1 completed a
  four-variant, five-seed fixed-budget comparison. Static scales and dynamic gates did not reliably
  change learning-curve AUC or saturated final accuracy versus dense residuals, despite clear
  attenuation of effective updates. C2 will test whether the null changes with task difficulty or
  minimum required depth.

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

1. Run Cycle C2 difficulty and minimum-depth sweeps without revising the fixed C1 null.
2. Continue to sparsity pressure and hard depth routing only after their instrumentation tests pass.
3. Retain B7's standard-Qwen atlas as a real-checkpoint adapter validation, not a gated comparison.
4. Run the pinned gated release only on hardware with enough headroom; do not block ordinary Qwen,
   Llama, or Gemma residual science on that resource condition.
5. Retain the no-repair and no-compute-saving boundaries in cross-model interpretation.
6. Keep Papers 2--5 evidence-conditional until their dedicated evidence gates are satisfied.
