# AGENTS.md — Paper 1 Cycle B: Exhaustive Residual Dynamics and Cross-Model Replication

## Mission

Continue Paper 1 from the completed first experimental cycle (E1–E8), but do **not** overwrite or reinterpret Cycle A null results.

Cycle A established:

- learned tiny-model tasks had positive block utility across all blocks;
- competence-matched goal-dependent utility did not replicate;
- learned gates saturated near open;
- the explicit goal latent did not outperform pooled residual state;
- hard skipping and PRA × computation were correctly stopped at their evidence gates;
- the pretrained gated-Qwen checkpoint was audited but not yet run end-to-end.

Cycle B must answer a deeper question:

> Was Cycle A negative because the broader residual/computational-selection hypothesis is weak, or because the original regime was too shallow, too capacity-constrained, too task-limited, and insufficiently instrumented to expose the relevant dynamics?

The next iteration must therefore prioritize:

1. **exhaustive residual metrics;**
2. **separate self-attention and feed-forward writes;**
3. **static and temporal/stability analyses;**
4. **deeper tiny models with computational slack;**
5. **a larger competence-matched task ecology;**
6. **pretrained Qwen comparison;**
7. **later Llama and Gemma-family replication using the same architecture-neutral probes.**

This is still Paper 1 work.

Do not prematurely move the core residual questions into later papers.

---

# 1. Preserve Cycle A as a fixed empirical baseline

Do not rewrite Cycle A as a success or failure.

Keep it as:

> In a small shallow model where nearly all learned blocks had positive causal utility, learned gates converged near open and did not improve quality.

This supports the revised hypothesis:

\[
\text{computational selection pressure}
\quad\text{may emerge most clearly when}\quad
\text{available computation} > \text{task-conditioned necessary computation}.
\]

Cycle B should test this directly.

Create explicit experiment-generation metadata:

```text
cycle = A | B
experiment = B1 | B2 | ...
revision
model_family
depth
width
task_family
seed
checkpoint
```

Never mix Cycle A and Cycle B results without labels.

---

# 2. Cycle B structure

Use the following execution order.

## B1 — Residual Dynamics Atlas

Run the exhaustive metric suite on existing Cycle A checkpoints first.

Purpose:

- determine what information is already latent in the stored models;
- validate metric implementations cheaply;
- identify promising layer/token/head structures;
- avoid retraining before the measurement stack is trustworthy.

## B2 — SA vs FF decomposition

Separate attention and feed-forward candidate writes.

## B3 — Static + stability / stationarity atlas

Measure depth-wise and token-wise state dynamics at multiple scales.

## B4 — Deep-capacity sweep

Train deeper models while preserving solvable tasks.

## B5 — Expanded competence-matched task ecology

Add more tasks with matched final competence but differing computational demands.

## B6 — Repeat conditional gating only if B2–B5 expose redundancy/interference

Do not force another gating experiment if there is still no selection pressure.

## B7 — Pretrained Qwen replication

Start with the smallest practical pretrained Qwen-family models and the already-audited gated-attention Qwen comparison.

## B8 — Llama replication

Use the same probe API and metric definitions.

## B9 — Gemma replication

Use the same probe API and metric definitions.

B8/B9 are conditional on B7 probe correctness and manageable resource use.

---

# 3. B1 — Exhaustive Residual Dynamics Atlas

The existing metric stack is not yet sufficiently exercised.

Run all static residual metrics on existing Cycle A checkpoints.

For each example \(e\), token \(t\), layer \(l\), and where relevant head \(h\), capture:

\[
x_{l,t}
\]

state entering a sublayer/block,

\[
\Delta_{l,t}
\]

candidate update,

\[
\tilde{\Delta}_{l,t}
\]

effective update if gating/modulation exists,

\[
x_{l+1,t}.
\]

At minimum compute:

### State magnitude

\[
\|x_{l,t}\|
\]

### Candidate update magnitude

\[
\|\Delta_{l,t}\|
\]

### Relative update magnitude

\[
r_{l,t}
=
\frac{\|\Delta_{l,t}\|}{\|x_{l,t}\|+\epsilon}
\]

### State/update alignment

\[
c_{l,t}
=
\cos(x_{l,t},\Delta_{l,t})
\]

### Inter-update alignment

For adjacent or related updates:

\[
\cos(\Delta_{l,t},\Delta_{l+1,t})
\]

and:

\[
\cos(\Delta_i,\Delta_j)
\]

for the layerwise update similarity matrix.

### Cancellation / reinforcement

For consecutive writes:

\[
\|\Delta_i+\Delta_j\|
\]

relative to:

\[
\|\Delta_i\|+\|\Delta_j\|.
\]

Define a normalized cancellation score such as:

\[
C_{ij}
=
1-
\frac{\|\Delta_i+\Delta_j\|}
{\|\Delta_i\|+\|\Delta_j\|+\epsilon}.
\]

Do not call this interference without causal evidence.

### Novelty

Measure the component of \(\Delta_l\) orthogonal to the current residual or recent update subspace.

At minimum:

\[
N_l
=
1-\cos^2(x_l,\Delta_l).
\]

Prefer a subspace-based version when possible.

### Dominance

Measure whether one update overwhelms the pre-existing state:

\[
D_l
=
\frac{\|\Delta_l\|}
{\|x_l\|+\|\Delta_l\|}.
\]

### Effective displacement

\[
d_l
=
\|x_{l+1}-x_l\|.
\]

### Representation direction drift

\[
\cos(x_l,x_{l+1}).
\]

### Pairwise layer similarity matrices

Compute:

- residual-state cosine;
- update cosine;
- CKA;
- RSA where practical.

Save full matrices, not only scalar summaries.

---

# 4. B2 — Separate self-attention and feed-forward writes

This is mandatory.

Do not treat the Transformer block as one indivisible residual update.

For a standard pre-norm block, identify and capture:

\[
x_l
\]

\[
\Delta^{SA}_l
\]

\[
x_l^{SA}
=
x_l+\Delta^{SA}_l
\]

\[
\Delta^{FF}_l
\]

\[
x_{l+1}
=
x_l^{SA}+\Delta^{FF}_l.
\]

Preserve architecture-specific norm placement accurately.

Never compare tensors from mismatched pre/post-norm locations.

Required geometric comparisons:

\[
\cos(x_l,\Delta^{SA}_l)
\]

\[
\cos(x_l^{SA},\Delta^{FF}_l)
\]

\[
\cos(\Delta^{SA}_l,\Delta^{FF}_l)
\]

\[
\frac{\|\Delta^{SA}_l\|}{\|x_l\|}
\]

\[
\frac{\|\Delta^{FF}_l\|}{\|x_l^{SA}\|}
\]

and cancellation/reinforcement of:

\[
\Delta^{SA}_l+\Delta^{FF}_l.
\]

Required causal interventions:

- skip SA only;
- skip FF only;
- skip full block;
- replace SA update with zero;
- replace FF update with zero;
- norm-matched random replacement where feasible.

Define:

\[
U^{SA}_l(\tau)
=
Q_\tau(\mathrm{full})
-
Q_\tau(\mathrm{skip\ SA}_l)
\]

and:

\[
U^{FF}_l(\tau)
=
Q_\tau(\mathrm{full})
-
Q_\tau(\mathrm{skip\ FF}_l).
\]

Key question:

> Do SA and FF play systematically different roles in refinement, complementarity, cancellation, or repair?

Do not assume the answer.

---

# 5. SA–FF interaction hypotheses

Test at least:

## H-B2.1 — Complementary specialization

SA may primarily import/compose contextual information while FF primarily transforms local state.

Prediction:

\[
U^{SA}_l
\]

and

\[
U^{FF}_l
\]

will differ systematically by task and depth.

## H-B2.2 — Intra-block repair

Some FF writes may partially compensate for SA-induced displacement:

\[
\cos(\Delta^{SA}_l,\Delta^{FF}_l)<0
\]

plus positive FF causal utility after harmful/over-large SA updates.

Geometry alone is insufficient.

## H-B2.3 — Redundant intra-block writes

Some SA or FF writes may have near-zero causal utility in over-capacity regimes.

## H-B2.4 — Task-conditioned role inversion

The same SA/FF sublayer may be useful for one task and redundant or harmful for another.

---

# 6. B3 — Static and temporal/stability metrics

The paper line explicitly includes stability/stationarity.

Implement and run these metrics rather than merely listing them.

Use token position as one temporal axis and depth as another computational axis.

## 6.1 Rolling statistics over token position

For residual states and update norms:

- rolling mean;
- rolling variance;
- skewness;
- kurtosis.

Run on:

- selected dimensions;
- projected low-dimensional summaries;
- norms;
- principal components;
- task-relevant probe directions.

## 6.2 First and second differences

For a vector summary \(v_t\):

\[
\Delta v_t=v_{t+1}-v_t
\]

\[
\Delta^2 v_t=\Delta v_{t+1}-\Delta v_t.
\]

Measure magnitude distributions and depth dependence.

## 6.3 Autocorrelation

Compute ACF over token position for:

- residual norm;
- update norm;
- gate values;
- selected principal coordinates;
- task-relevant probe scores.

Estimate effective correlation length.

## 6.4 Cross-correlation

Measure lagged relations between:

- attention entropy and update magnitude;
- SA write and FF write;
- gate value and candidate update utility;
- residual perturbation and later repair;
- task-variable decodability and update magnitude.

## 6.5 Distribution drift

Across token windows or depth bands compute:

- mean shift;
- covariance shift;
- Wasserstein distance where practical;
- MMD where practical;
- KL only for well-defined estimated distributions.

## 6.6 Subspace stability

Compute:

- PCA principal angles;
- eigenspectrum drift;
- CKA;
- RSA;
- stable-rank estimates.

## 6.7 Layer × token stability maps

Create heatmaps for:

\[
M(l,t)
\]

for at least:

- residual norm;
- update norm;
- alignment;
- attention entropy;
- causal utility proxy where available;
- probe decodability;
- gate values.

---

# 7. Static invariants and low-variance directions

Search for state properties that remain comparatively stable under token or layer transitions.

Candidate analyses:

- high-magnitude / low-variance principal directions;
- slowly varying subspaces;
- stable covariance eigenvectors;
- repeated update directions;
- task-invariant vs task-specific directions;
- content-invariant vs goal-specific directions.

Do not label these "symmetries" or "invariants" without a precise operational definition.

Use terminology such as:

- stable direction;
- persistent subspace;
- low-drift coordinate;
- approximately invariant statistic.

This work should remain compatible with later invariants/symmetry papers.

---

# 8. Amplification–repair analysis

Implement explicit detection of candidate amplification–repair events.

For a perturbation or naturally occurring displacement \(\delta_l\), track:

\[
\|\delta_l\|,
\|\delta_{l+1}\|,
\dots
\]

and identify:

### Amplification

\[
\|\delta_{k+1}\|>\|\delta_k\|
\]

for one or more steps.

### Repair

A later stage reduces the perturbation:

\[
\|\delta_{m+1}\|<\|\delta_m\|.
\]

Use both natural and intervention-induced perturbations.

Preferred interventions:

- skip SA;
- skip FF;
- skip block;
- patch state from matched example;
- inject small norm-controlled perturbation.

For strong repair claims require:

- task-relevant recovery;
- not only vector-norm shrinkage;
- replication across examples/seeds.

---

# 9. Correct vs incorrect trajectory analysis

Do not aggregate successful and failed examples together.

Partition examples by outcome:

\[
\mathcal{E}_{correct}
\]

and:

\[
\mathcal{E}_{incorrect}.
\]

Compare:

\[
P(M_l\mid correct)
\]

versus:

\[
P(M_l\mid incorrect)
\]

for:

- update magnitude;
- update alignment;
- SA/FF balance;
- attention entropy;
- stability;
- repair events;
- gate values;
- task-variable decodability.

This is especially important for tasks near the competence boundary.

Use matched difficulty bins where possible.

---

# 10. Representation decomposition: content, goal, interaction

Use controlled tasks to test whether residual representations support an approximate decomposition:

\[
h
=
h_{\mathrm{content}}
+
h_{\mathrm{goal}}
+
h_{\mathrm{interaction}}
+
\epsilon.
\]

Do not assume linearity.

Start with linear probes and variance decomposition because they are easy to falsify.

Measure:

- content decodability;
- goal decodability;
- answer decodability;
- interaction term importance;
- layer where each becomes available;
- relationship to SA/FF writes.

Because Cycle A showed pooled residual outperforming explicit \(z\), test whether the ordinary residual already functions as a distributed control state.

Important question:

> Does an explicit latent \(z\) fail because no goal state exists, or because the residual stream already carries it more effectively?

---

# 11. B4 — Deep-capacity sweep

The original 4-layer model may have had too little slack for computational selection.

Train a controlled depth series:

\[
L\in\{4,8,12,16\}
\]

with at least 5 seeds each where feasible.

Keep other capacity dimensions controlled as much as possible.

Also include a matched-parameter control where depth increases but width decreases, if inexpensive enough.

Purpose:

> Separate depth/computational slack from total parameter count.

Required measurements by depth:

- task quality;
- block utility;
- SA utility;
- FF utility;
- fraction near-zero utility;
- fraction negative utility;
- update magnitude;
- redundancy matrices;
- repair events;
- stability metrics.

Core hypothesis:

\[
L_{\mathrm{available}}
\gg
L_{\mathrm{necessary}}
\]

should increase:

- redundancy;
- conditional utility differences;
- potentially negative utility;
- opportunity for gating/skipping.

Do not claim this in advance.

Test it.

---

# 12. Depth-dependent sufficient-activation test

For each depth \(L\), estimate the minimum useful active subset under intervention.

Define a crude effective depth:

\[
L_{\mathrm{eff}}(\tau)
=
\#\{l: U_l(\tau)>\epsilon\}.
\]

Also estimate:

- SA effective count;
- FF effective count;
- task-specific active fraction.

Key analysis:

\[
\frac{L_{\mathrm{eff}}}{L}
\]

versus:

\[
L.
\]

If this fraction falls as depth rises while quality remains stable, that supports computational slack.

If it stays near 1, the model uses additional depth rather than leaving it redundant.

---

# 13. B5 — Expanded task ecology

The original max/min/sum setup was not competence matched.

Build a larger deterministic controlled task family.

Start with tasks likely to have similar learnability.

Suggested families:

### Selection tasks

- max;
- min;
- first;
- last;
- middle;
- argmax-position;
- argmin-position.

### Comparison tasks

- first > last;
- max > threshold;
- count-above-threshold;
- equality / inequality;
- pairwise relation.

### Transformation tasks

- reverse;
- copy selected position;
- rotate;
- sorted-first / sorted-last;
- local arithmetic with bounded difficulty.

### Aggregate tasks

Only include sum/parity/count if they can be trained to competence comparable with other tasks.

Do not use poorly learned tasks as evidence for task-conditioned interference.

---

# 14. Competence matching

Before comparing task-conditioned block utility, enforce competence matching.

For tasks \(i,j\):

\[
|Q_i-Q_j|<\delta.
\]

Choose and preregister \(\delta\).

If tasks cannot be matched by the same training budget:

- adjust curriculum;
- adjust task difficulty;
- adjust input range;
- or exclude the contrast.

Do not interpret utility differences between a learned and unlearned task as goal-conditioned computation.

---

# 15. Same-content / different-goal counterfactuals

Continue using:

\[
(C,I_1)\rightarrow Y_1
\]

and:

\[
(C,I_2)\rightarrow Y_2.
\]

But expand to many competence-matched goal pairs.

For each pair, measure:

- residual-state similarity;
- SA-update similarity;
- FF-update similarity;
- block utility difference;
- attention redistribution;
- gate redistribution;
- goal-probe separability.

Look for when trajectories diverge.

Important output:

\[
D_{goal}(l,t)
\]

a layer × token map of goal-conditioned divergence.

---

# 16. Gating should be repeated only after selection pressure is demonstrated

Do not retrain gates immediately.

Open the B6 gate experiment only if one or more of the following appears:

- reliable near-zero block utility in competent models;
- reliable negative block/sub-block utility;
- task-conditioned utility differences;
- repeated amplification–repair events;
- depth-dependent redundancy;
- stable residual signatures predicting utility.

If none occur, document that computational selection remains weak in the tested tiny regime.

---

# 17. B6 — Gated tiny models

If evidence gate passes, compare:

### Baseline residual

\[
x_{l+1}=x_l+\Delta_l.
\]

### Input-conditioned gate

\[
x_{l+1}=x_l+\sigma(G_l(x_l))\Delta_l.
\]

### Optional SA/FF-specific gates

\[
x_l^{SA}=x_l+g_l^{SA}\Delta_l^{SA}
\]

\[
x_{l+1}=x_l^{SA}+g_l^{FF}\Delta_l^{FF}.
\]

### Optional latent-conditioned gate

Only if justified by B5.

Key tests:

- gate vs causal utility;
- gate vs redundancy;
- gate vs candidate update magnitude;
- gate vs repair;
- task-conditioned gate differences;
- forced-open/closed/shuffled interventions.

---

# 18. Pretrained replication strategy

The pretrained phase is for external validity.

It should answer:

> Do the same residual-dynamics signatures appear in models trained independently of our hypotheses?

Do not expect exact quantitative matching with tiny models.

Use architecture-neutral metrics first.

---

# 19. B7 — Qwen-family comparison

Qwen is the first pretrained family because the gated-attention comparison already has an audited implementation.

Use:

1. a small standard Qwen/Qwen3-style baseline;
2. the closest available `QwQZh/gated_attention` baseline;
3. the headwise gated-attention checkpoint.

Primary gated checkpoint properties already verified:

- query-derived gate;
- tensor semantics approximately `[batch, token, head, 1]`;
- sigmoid headwise gating;
- applied after SDPA;
- before output projection.

Re-verify against the exact loaded revision before collecting results.

Required Qwen analyses:

- native-forward parity under instrumentation;
- residual atlas;
- SA/FF decomposition;
- gate distributions;
- gate vs causal utility;
- gate vs candidate/effective update metrics;
- attention dilution vs gate;
- forced-open/closed/mean/shuffled interventions;
- token/depth stability maps.

Use the smallest practical checkpoint first.

Scale up only after the probe pipeline is numerically verified.

---

# 20. Resource-aware pretrained policy

Do not download every checkpoint.

Before downloading, record:

- model size;
- on-disk size;
- dtype;
- minimum practical RAM/VRAM;
- expected KV requirements;
- whether CPU inference is acceptable;
- whether quantization invalidates a metric.

Prefer one carefully instrumented small checkpoint over many partially run models.

Quantization is acceptable for exploratory work only if:

- tensor locations remain available;
- the metric is not obviously distorted;
- results are labeled exploratory.

Paper-facing claims should preferably use unquantized or well-controlled precision.

---

# 21. B8 — Llama-family replication

After Qwen works end-to-end, add one small Llama-family pretrained model.

Goals:

- test whether residual/SA/FF signatures generalize across architecture families;
- compare norm placement and GQA effects;
- test whether metric definitions survive adapter changes.

Do not add Llama-specific metrics.

Extend the common adapter.

Required:

- numerical parity;
- same residual atlas;
- same SA/FF metrics;
- same causal ablations where feasible;
- same token/depth stability analysis.

If Llama results diverge, report architecture dependence rather than normalizing it away.

---

# 22. B9 — Gemma-family replication

After Llama, add one small Gemma-family model.

Gemma is especially useful because its architecture differs in attention scheduling/norm details from Qwen/Llama families.

Again:

- use common probes;
- preserve architecture-specific tensor locations;
- validate parity;
- run the same metric definitions;
- document any metric that cannot be made equivalent.

Do not compare raw scalar magnitudes across families without normalization.

Prefer:

- standardized within-model effects;
- ranks;
- layer-relative positions;
- effect sizes;
- causal deltas.

---

# 23. Cross-family comparison

For Qwen, Llama, Gemma, compare normalized signatures:

### Residual geometry

- relative update norm;
- state/update cosine;
- layer similarity;
- SA/FF balance.

### Causal utility

- block;
- SA;
- FF.

### Stability

- depth drift;
- token drift;
- subspace stability;
- ACF/correlation length.

### Organization

- head redundancy;
- layer specialization;
- task-conditioned divergence.

### Computational selection

Where explicit gates exist:

- gate distribution;
- gate–utility relation;
- gate–interference relation.

Avoid claiming universal principles from 2–3 model families.

Use language:

- cross-model regularity;
- architecture-sensitive pattern;
- preliminary common signature.

---

# 24. N-gram / predictable-sequence probe

Add a small clean probe based on predictable token sequences or common n-gram-like patterns.

Purpose:

- provide cases where expected continuation is constrained;
- inspect SA vs FF roles under highly predictable local structure;
- connect residual metrics with the separate n-gram mechanistic line.

Possible analyses:

- prefix length vs SA contribution;
- prefix length vs FF contribution;
- layer of peak next-token decodability;
- stability of update signatures across repeated n-gram families.

Keep this as a probe, not a new paper inside Paper 1.

---

# 25. Attention metrics must be fully exercised

The current code already contains attention-dilution metrics.

Cycle B must actually use them.

For each head/layer/token where attention weights are available, compute:

- entropy;
- top-1 mass;
- top-k mass;
- effective support;
- sink score;
- concentration;
- evidence mass for synthetic tasks;
- head similarity;
- cross-layer similarity.

Relate attention to residual writes:

\[
\text{attention entropy}
\leftrightarrow
\|\Delta^{SA}\|
\]

\[
\text{attention entropy}
\leftrightarrow
U^{SA}
\]

\[
\text{attention concentration}
\leftrightarrow
\text{goal-conditioned divergence}.
\]

In gated-attention models also test:

\[
\text{attention dilution}
\leftrightarrow
g.
\]

---

# 26. Head-level analysis

Do not stop at layer averages.

For each attention head:

- output norm;
- contribution to SA write;
- attention entropy;
- top-k mass;
- sink score;
- task-conditioned divergence;
- redundancy with heads in same layer;
- redundancy across layers;
- optional head-ablation utility.

Where head ablation is feasible, define:

\[
U_{l,h}
=
Q(\mathrm{full})
-
Q(\mathrm{ablate\ head}_{l,h}).
\]

Do not require full head-ablation sweeps for large pretrained models if too expensive.

Use sampled or targeted heads.

---

# 27. Statistical design

For tiny models:

- minimum 5 seeds;
- prefer 10 for final claims if inexpensive;
- fixed train/validation/test split per configuration;
- bootstrap confidence intervals;
- paired example-level comparisons.

For pretrained models:

- deterministic forward passes;
- many examples across task families;
- bootstrap over examples;
- report model/checkpoint as fixed, not random population sampling.

Report:

- mean;
- median;
- standard deviation;
- 95% CI;
- effect size;
- raw \(n\).

Do not rely on \(p\)-values alone.

---

# 28. Artifact requirements

Persist:

```text
artifacts/
  cycle_b/
    b1_residual_atlas/
    b2_sa_ff/
    b3_stability/
    b4_depth_sweep/
    b5_task_ecology/
    b6_gating/
    b7_qwen/
    b8_llama/
    b9_gemma/
```

Each experiment directory should contain:

- config snapshot;
- model metadata;
- git commit;
- summary JSON;
- raw or derived Parquet;
- CSV tables;
- figures;
- run log;
- failure/null-result notes.

Do not commit huge activations.

Persist derived statistics needed for reproducibility.

---

# 29. Common adapter requirements

The architecture-neutral probe API must support:

```python
class ModelProbeAdapter:
    def residual_pre(self, layer, token): ...
    def attention_candidate_update(self, layer, token): ...
    def residual_after_attention(self, layer, token): ...
    def ff_candidate_update(self, layer, token): ...
    def residual_post(self, layer, token): ...
    def attention_weights(self, layer, head, token): ...
    def gate(self, layer, head, token): ...
    def logits(self): ...
```

Exact naming can differ, but semantics must be fixed.

Adapters:

- tiny custom Transformer;
- Qwen;
- gated-Qwen;
- Llama;
- Gemma.

Do not duplicate metric logic per architecture.

---

# 30. Numerical parity tests

Every pretrained adapter must pass instrumentation parity before scientific use.

Procedure:

1. run untouched native forward;
2. run instrumented forward with probes enabled;
3. compare logits and hidden outputs.

Target exact equality where possible.

Otherwise document tolerance.

Do not silently accept parity drift.

Intervention mode must be a separate explicit code path.

---

# 31. Paper updates

Paper 1 should gain a clearly separated section:

## Cycle A — Initial shallow-regime tests

Preserve current results.

Then add:

## Cycle B — Residual Dynamics Atlas and Capacity Expansion

Present:

- rationale from Cycle A;
- exhaustive metrics;
- SA/FF decomposition;
- stability/stationarity;
- depth sweep;
- competence-matched task ecology;
- conditional gating;
- pretrained replication.

Do not rewrite preregistered Cycle A hypotheses after the fact.

Cycle B hypotheses should be explicitly labeled as post-Cycle-A but preregistered before Cycle B results.

---

# 32. Core Cycle B hypotheses

Preregister:

## H-B1 — computational slack

As depth increases beyond what is necessary for competent task performance:

\[
\frac{L_{\mathrm{eff}}}{L}
\]

will decrease.

## H-B2 — sublayer specialization

SA and FF causal utility and residual signatures will differ systematically.

## H-B3 — task-conditioned computation

Competence-matched tasks will show different block/sub-block utility profiles.

## H-B4 — stability signatures

Correct trajectories will show different stability/repair signatures from incorrect trajectories.

## H-B5 — gating pressure

Gates will become more selective only in regimes where measurable redundancy/interference exists.

## H-B6 — pretrained transfer

At least some normalized residual and SA/FF signatures will replicate in pretrained Qwen models.

## H-B7 — cross-family partial regularity

Some signatures will persist in Llama/Gemma, while others may remain architecture-specific.

All are falsifiable.

---

# 33. Important negative outcomes

Preserve:

- deeper models still use every block positively;
- no SA/FF specialization;
- no stable repair signatures;
- competence-matched tasks use indistinguishable trajectories;
- gates remain open even with depth slack;
- pretrained models show no relation to tiny-model signatures;
- Qwen gated-attention gates correlate only with magnitude;
- Llama/Gemma differ qualitatively.

These are useful constraints.

Do not force the theory to survive them.

---

# 34. Execution and commit discipline

Commit and push after each completed stage.

Suggested commits:

```text
B1 residual atlas
B2 SA FF decomposition
B3 stability metrics
B4 depth sweep
B5 task ecology
B6 conditional gating
B7 Qwen replication
B8 Llama replication
B9 Gemma replication
```

Each stage commit must include:

- code;
- tests;
- configs;
- result summaries;
- paper update;
- figures/tables if generated.

Do not proceed to the next stage if current-stage instrumentation is numerically invalid.

---

# 35. Immediate next action

Start with B1 on the already-trained Cycle A checkpoints.

Before new training:

1. enumerate all currently implemented metrics;
2. compare them against `docs/METRICS.md`;
3. implement missing metrics;
4. run the full residual atlas;
5. produce layer × token × metric outputs;
6. update Paper 1 with B1 methods and results;
7. commit and push.

Then proceed to B2.

---

# 36. Final scientific target

Cycle B should answer whether the current null results are specific to a shallow, capacity-constrained regime or reflect a more general property of residual computation.

The most informative possible outcomes are not restricted to "gating helps."

A stronger mechanistic picture would distinguish:

\[
\boxed{\text{memory selection}}
\]

from

\[
\boxed{\text{candidate transformation}}
\]

from

\[
\boxed{\text{effective state update}}
\]

and show how these evolve across:

- SA;
- FF;
- depth;
- token position;
- task/goal;
- model capacity;
- architecture family.

The paper should ultimately characterize **when computational selection is needed**, not merely whether one gating mechanism improves a benchmark.
