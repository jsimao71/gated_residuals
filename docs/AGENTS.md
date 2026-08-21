# AGENTS.md — Paper 1: Selective Residual Computation

## Mission
Implement Paper 1 as a falsifiable mechanism study. Do not optimize for confirming residual inhibition, shared goal state, or sufficient activation. Preserve null and negative results.

## Scientific hierarchy
Mandatory core:
1. E1 baseline residual map on tiny Transformers.
2. E2 same-content/different-goal counterfactuals.
3. E3 gated residual stream.

Secondary:
4. E4 shared distributed goal latent.
5. E5 combined goal-conditioned gating.

Conditional/stretch:
6. E6 hard skipping / sufficient activation.
7. E7 pretrained + third-party architecture transfer.
8. E8 PRA memory × computation factorial.

Do not expand E6–E8 until E1–E4 identify a robust phenomenon.

## Non-negotiable scope
- NLP/token streams only for Paper 1.
- Do NOT add mazes, Sudoku, 2-D games, or virtual embodied environments.
- Primary datasets: controlled synthetic natural-language counterfactual mixtures.
- Continuity datasets: WikiText-2, WikiText-103, HotpotQA, QASPER.
- Reuse PRA instrumentation/datasets where practical, but keep this project scientifically separable.
- Every experiment saves JSON/CSV + config + seed + commit hash + environment metadata.
- Use >=5 seeds for tiny-model claims unless a documented resource constraint prevents it.
- Match parameter/FLOP budgets where the causal question requires it; otherwise report mismatch explicitly.

## Phase A — common instrumentation
Create a reusable interface for embeddings, residual pre/post states, attention outputs/weights, MLP outputs, per-head outputs where feasible, logits/margins, gradients/weights, architecture-specific latents, gates, and active-block decisions.

Metric modules:
- memory_activation.py
- attention_dilution.py
- residual_dynamics.py
- temporal_stability.py
- head_layer_similarity.py
- standard_metrics.py
- systems_metrics.py
- causal_ablation.py

Temporal metrics must include multiscale rolling moments and derivatives, chunk distribution/covariance comparison, ACF and ACF drift, lagged cross-correlation matrices, correlation length, CKA/RSA across windows, subspace/eigenspectrum drift, layer×scale×token maps, and amplification–repair detection.

Unit-test every metric on analytically constructed tensors.

## Phase B — synthetic task ecology
Generate counterfactual families `(content C_i, intents I_1...I_k)`:
- identical content reused across multiple response types
- no explicit MODE/task ID
- paraphrased and implicit goals
- constraints distributed across prompt context
- audience/style/resource/output constraints
- conflicting local affordances
- controlled goal-to-content distance
- distractors / irrelevant constraints
- same-content/different-goal pairs
- same-goal/different-content pairs
- semantically continuous task families
- deterministic generation under seed
- latent task factors stored only for measurement

Add `goal_identifiability`: how strongly local lexical cues reveal the task.

## Phase C — E1 baseline residual map
1. Train tiny decoder-only baselines.
2. Capture metric battery.
3. Exhaustive single-block skip.
4. Selected pair/group skips.
5. Compute `U_l(task)=Q(full)-Q(skip_l)`.
6. Classify useful / redundant / candidate harmful blocks.
7. Search for later repair/reversal of candidate harmful writes.
8. Measure task decodability, logits, attention dilution, temporal stability, head/layer structure.

### Strong interference criterion
Do NOT call anti-alignment or negative cosine "interference".
Require:
- statistically reliable `U_l(task) < 0`;
- later repair/reversal/compensation evidence;
- replication across examples and seeds.

Pre-register statistical method/effect threshold before final test inspection. Prefer example bootstrap confidence intervals plus seed-level replication.

## Phase D — E2 counterfactual goals
Hold content fixed and vary distributed intent. Measure delta residual trajectories, block utility, logits, attention, temporal stability, and relation to `goal_identifiability`.

## Phase E — E3 gated residual stream
Implement scalar/block gates first:
`h[l+1] = h[l] + g_l(z) * F_l(h[l])`.

Controls:
- matched-capacity ungated model
- random skipping
- static task routes
- update/activation magnitude heuristic
- norm heuristic
- early exit where meaningful

Compare quality, active FLOPs, realized latency, gate entropy, harmful-block activation, repair signatures, and distractor robustness.

## Phase F — E4 shared goal latent
Implement:
`z[l+1] = z[l] + G_l(z[l], h[l])`.

Do not supervise `z` with task labels in the main objective.

Primary diagnostic is NOT task classification. Test whether `z` predicts future causal utility `U_l(task)`.

Required controls:
- current residual `h_l`
- pooled residual summaries
- matched-dimensional random vectors with matched first/second moments
- shuffled-goal counterfactuals
- z-only model
- gate-only model

Start with minimal `G_l` variants; do not proliferate architectures before evidence.

## Phase G — E5 combined modulation
Key test: do learned gates suppress blocks independently identified by E1 as harmful/redundant for that task family? If not, call it generic dynamic sparsity rather than mechanistic inhibition.

## Conditional E6–E8
E6: hard skipping; sweep epsilon/lambda; quality vs active FLOPs vs realized latency vs interference Pareto fronts.

E7: transfer only strongest tiny-model signatures. Choose comparator based on finding:
- interference -> mHC/Hyper-Connections
- conditional compute -> MoD/LayerSkip
- iterative refinement -> TRM

E8: PRA memory × active computation factorial; measure Q(k_memory,k_compute), evidence recall, attention dilution, residual interference, cost.

## Paper pivot rules after E1–E4
- Strong harmful + repair effects, weak gating -> residual interference paper.
- Strong goal-state prediction -> distributed task-control paper.
- Strong gating quality improvement -> goal-modulated residual computation paper.
- Quality improves while compute falls -> sufficient/selective activation paper.
- Weak/null effects -> comparative residual-dynamics/falsification paper; remove unsupported claims.

## Primary metrics
Primary:
- task-conditioned causal block utility
- repair/reversal evidence
- counterfactual goal change in block utility
- prediction of future block utility from z
- task quality
- active FLOPs / realized latency for skipping claims

Secondary/exploratory:
- update cosine/geometry
- CKA/RSA/subspaces
- temporal stationarity suite
- head/layer clustering
- attention dilution
- gradients/weights

Exploratory metrics may motivate hypotheses but may not substitute for causal evidence.

## Tests and reproducibility
- exact parity when new mechanisms are disabled
- deterministic synthetic generation
- no silent NaN/Inf
- regression tests for instrumentation
- shape/parity tests
- fixed train/dev/test counterfactual-family split
- no tuning on final test
- machine-readable experiment registry

## Expected repository structure
`src/`, `tests/`, `configs/`, `scripts/`, `notebooks/`, `results/`, `figures/`, `docs/`.

README must document one-command smoke run, E1–E3 core matrix, E4–E8 conditional matrix, expected artifacts, and figure/table regeneration.
