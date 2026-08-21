# AGENTS.md — Add Gated-Attention Model to Metrics + Residuals Paper

## Objective

Patch the existing **metrics + residuals / mechanistic cognition** paper and its experiment code to include a pretrained gated-attention Transformer as a first-class comparison architecture.

Use the released Qwen3-based gated-attention implementation and checkpoint family:

- Hugging Face: `QwQZh/gated_attention`
- Official implementation: `qiuzh20/gated_attention`
- Primary experimental variant: **headwise query-dependent sigmoid gating after SDPA**
- Include the closest available ungated/baseline counterpart from the same release or implementation whenever possible.

The purpose is **not merely to benchmark gated attention**.

The scientific purpose is to test whether learned gating decisions correspond to the paper's proposed architecture-independent internal metrics of:

- residual refinement,
- residual complementarity,
- residual interference,
- update magnitude,
- representation stability,
- attention dilution,
- memory activation,
- and downstream usefulness.

The central experimental question is:

> Can independently learned gates predict, or be predicted by, architecture-independent measurements of whether a candidate transformation is useful, complementary, redundant, destabilizing, or interfering?

---

# 1. Why this model must be included

The paper currently studies residual computation as an internal dynamical process.

Gated attention creates an unusually strong test because it introduces an explicit learned variable controlling whether an attention-derived transformation is admitted into the residual stream.

Conceptually:

\[
a_l = \operatorname{Attention}(x_l)
\]

\[
g_l = \sigma(G_l(x_l))
\]

\[
\Delta_l = g_l \odot a_l
\]

\[
x_{l+1}=x_l+\Delta_l.
\]

This exposes a latent decision that is mostly implicit in a classical Transformer:

\[
\boxed{\text{Should this computed update affect the current state?}}
\]

Call this **computational selection**.

The gated model therefore gives the paper a direct empirical probe of the proposed distinction between:

1. **memory / information selection**, primarily implemented by attention or retrieval;
2. **computational / state-update selection**, explicitly implemented by the gate.

---

# 2. Required architecture comparison

At minimum compare:

## A. Standard Transformer baseline

Use the closest available Qwen3-style ungated baseline associated with the gated-attention release.

## B. Headwise gated-attention model

Use the released headwise gating variant.

The preferred comparison is a controlled pair where architecture, scale, tokenizer, training setup, or provenance are as closely matched as the released artifacts permit.

If exact checkpoint equivalence is unavailable, document the mismatch explicitly.

Do not silently compare models with materially different size, tokenizer, context length, or training corpus as though the only difference were gating.

---

# 3. Verify the implementation before instrumentation

Before adding metrics, inspect the official implementation and determine precisely:

- where the gate is computed;
- the tensor shape of the gate;
- whether it is scalar per head, per token × head, or otherwise;
- whether the gate is applied before or after SDPA;
- whether it is applied before or after output projection;
- whether gating modifies attention outputs, residual updates, or both;
- whether gate values are directly accessible without modifying numerical behavior.

Record this in a short implementation note in the experiment metadata.

Do not rely solely on architecture names or README prose.

---

# 4. Required captured tensors

For every instrumented layer and, where applicable, every head and token, capture enough information to reconstruct the computational-selection event.

At minimum capture:

\[
x_l
\]

the residual state entering the relevant attention block;

\[
a_l
\]

the ungated attention output at the gate location;

\[
g_l
\]

the learned sigmoid gate;

\[
\tilde{a}_l = g_l \odot a_l
\]

the gated attention update;

and

\[
x_{l+1}
\]

the residual state after the relevant update.

If the architecture has pre/post normalization or an output projection between these quantities, preserve the actual computational sequence and use precise variable names.

Never compare tensors from mismatched locations as if they represented the same quantity.

---

# 5. Gate metrics

Add explicit gate metrics.

At minimum compute:

- mean gate value;
- median gate value;
- gate variance;
- fraction of gates below thresholds such as 0.1, 0.25, 0.5;
- fraction above 0.75 and 0.9;
- per-layer gate distribution;
- per-head gate distribution;
- per-token gate distribution;
- gate entropy or equivalent dispersion statistic when meaningful;
- temporal/token autocorrelation of gate values;
- cross-head correlation within a layer;
- cross-layer similarity.

For headwise gating compute:

\[
g_{l,h,t}.
\]

Preserve this dimensionality in raw artifacts.

Do not immediately average away layer/head/token structure.

---

# 6. Residual metrics before and after gating

For the **ungated candidate update** \(a_l\), compute the existing residual metrics against the current state:

\[
\cos(x_l,a_l)
\]

\[
\|a_l\|
\]

\[
\frac{\|a_l\|}{\|x_l\|}
\]

and all existing refinement/complementarity/interference metrics already defined by the project.

Then compute the same metrics for the **effective gated update**:

\[
\tilde a_l = g_l \odot a_l.
\]

This creates paired observations:

\[
(x_l,a_l)
\]

versus

\[
(x_l,\tilde a_l).
\]

The key question is whether gating systematically transforms harmful candidate updates into smaller or less disruptive effective updates.

---

# 7. Do not define interference only by cosine

Cosine alignment is useful but insufficient.

Retain or add multiple operationalizations of interference.

Candidates include:

### Geometric alignment

\[
\cos(x_l,a_l)
\]

### Relative update magnitude

\[
r_l = \frac{\|a_l\|}{\|x_l\|}.
\]

### Representation displacement

\[
d_l = \|x_{l+1}-x_l\|.
\]

### Downstream functional effect

Measure the change in task loss or target-token log probability when the update is:

- preserved,
- attenuated,
- removed,
- or counterfactually replaced where feasible.

### Feature preservation

Where practical, measure whether previously predictive directions/features are degraded after the update.

Use the paper's established terminology:

- refinement,
- complementarity,
- interference.

But treat these as hypotheses operationalized by multiple metrics rather than direct labels inferred from cosine sign.

---

# 8. Central gate–metric correlation tests

For each layer/head/token observation, evaluate relations such as:

\[
g_{l,h,t}
\sim
f(
\cos(x_l,a_l),
\|a_l\|,
\|a_l\|/\|x_l\|,
\text{stability},
\text{attention entropy},
\text{downstream utility}
).
\]

Required analyses:

- Pearson correlation where appropriate;
- Spearman correlation;
- rank plots;
- binned gate-versus-metric curves;
- per-layer relationships;
- per-head relationships;
- pooled relationships with architecture/layer/head controls;
- simple predictive models where useful.

Do not report only a pooled correlation.

A globally weak relationship can hide strong layer- or head-specific structure.

---

# 9. Strongest causal-style intervention

Where computationally feasible, perform gate interventions at inference time.

For the gated model compare:

### Native gate

\[
g = g_{\text{learned}}.
\]

### Forced open

\[
g = 1.
\]

### Forced closed

\[
g = 0.
\]

### Mean gate

Replace dynamic values by layer/head means.

### Shuffled gate

Shuffle gate values across tokens or examples while preserving marginal distributions.

### Optional thresholded gate

\[
g' = \mathbb{1}[g>\tau].
\]

These interventions help distinguish:

> gate values correlate with good updates

from

> gate values causally contribute to selecting good updates.

Preserve the original pretrained weights.

Do not retrain unless a later experiment explicitly requires it.

---

# 10. Comparison with ungated baseline

For the ungated model, compute the same residual and attention metrics.

Then compare whether the gated architecture exhibits:

- lower harmful update magnitude;
- reduced interference;
- greater stability;
- different layerwise refinement/complementarity profiles;
- reduced sensitivity to attention dilution;
- different token/head specialization;
- improved downstream task metrics.

Important:

Do not assume the gated model must exhibit "better" values for every internal metric.

A gating mechanism may intentionally permit larger updates when they are useful.

Focus on conditional relationships between gate state and update consequence.

---

# 11. Attention dilution analysis

Because the paper already studies attention dilution, explicitly test whether gating separates:

\[
\text{where attention looks}
\]

from

\[
\text{whether the resulting computation is used}.
\]

Measure existing attention statistics such as:

- entropy;
- concentration;
- top-k mass;
- attention sink behavior if already supported;
- effective number of attended tokens;
- memory activation breadth where defined.

Then relate these to gate values.

Important tests:

\[
\text{high attention entropy} \rightarrow g?
\]

\[
\text{low maximum attention weight} \rightarrow g?
\]

\[
\text{attention sink strength} \rightarrow g?
\]

\[
\text{broad/diffuse retrieval} + g\approx 0?
\]

This directly tests whether computational selection compensates for weak or diffuse memory selection.

---

# 12. Dual-selection analysis

Add a dedicated analysis section using the paper's new terminology.

## Memory selection

Characterize which tokens/KV states become influential through attention.

Candidate measures:

- attention distribution;
- entropy;
- top-k mass;
- memory activation;
- retrieval breadth;
- head specialization.

## Computational selection

Characterize whether the produced attention computation becomes an effective residual update.

Primary observable:

\[
g_{l,h,t}.
\]

The core empirical question is whether these two forms of selection are statistically distinguishable and complementary.

Examples:

- sharp attention + low gate;
- diffuse attention + high gate;
- sharp attention + high gate;
- diffuse attention + low gate.

Quantify these regimes rather than describing isolated examples.

---

# 13. Tasks and datasets

Do not test on only one task.

Use the same task families already planned by the metrics/residuals paper wherever possible.

At minimum include representatives of:

- language modeling / next-token prediction;
- retrieval or QA;
- multi-hop reasoning if supported;
- controlled synthetic tasks useful for interpreting residual dynamics.

Prefer existing datasets already present in the repository.

Do not introduce a large new dataset dependency merely for this model unless required.

---

# 14. Standard performance metrics

Continue to report conventional ML metrics alongside internal measurements.

Depending on task:

- loss;
- perplexity;
- exact match;
- accuracy;
- F1;
- recall;
- precision;
- output entropy.

The paper's point is not that internal metrics replace standard evaluation.

The point is that they reveal mechanisms hidden by aggregate task scores.

---

# 15. Statistical design

Use multiple seeds where stochasticity is involved.

For pretrained inference with deterministic forward passes, vary:

- examples;
- datasets;
- task type;
- prompt/context condition.

Where models are not perfectly matched, avoid invalid paired statistical claims.

Prefer paired comparisons at the **example level** when the same examples can be processed by both models.

Report:

- sample count;
- mean;
- median where useful;
- standard deviation;
- confidence intervals;
- effect sizes.

Avoid significance-only reporting.

---

# 16. Required plots

Add plots that expose internal structure, not merely final performance.

At minimum:

1. **Gate distribution by layer**
2. **Gate distribution by head**
3. **Gate vs residual alignment**
4. **Gate vs relative update magnitude**
5. **Gate vs downstream utility**
6. **Ungated candidate vs gated effective interference**
7. **Attention dilution vs gate**
8. **Layerwise baseline-vs-gated residual dynamics**

Optional high-value plots:

- token-position gate heatmap;
- gate autocorrelation;
- head clustering by gate/interference signature;
- task-conditioned gate profiles;
- gate versus context length.

---

# 17. Raw artifact schema

Persist raw measurements in machine-readable form.

Suggested record fields:

```text
model
model_variant
dataset
task
example_id
token_index
layer
head
gate
candidate_update_norm
effective_update_norm
residual_norm
candidate_cosine
effective_cosine
attention_entropy
attention_top1_mass
attention_topk_mass
loss
target_logprob
intervention
seed
```

Add fields required by existing project metrics.

Prefer Parquet for large tensor-derived tables and CSV summaries for paper-facing tables.

Do not save huge raw activations indefinitely unless required for reproducibility.

---

# 18. Model adapter design

Implement model-specific instrumentation behind a thin adapter.

Do not contaminate architecture-independent metrics with Qwen-specific code.

Preferred conceptual API:

```python
class ModelProbeAdapter:
    def iter_layers(...): ...
    def capture_residual_input(...): ...
    def capture_candidate_update(...): ...
    def capture_effective_update(...): ...
    def capture_gate(...): ...
    def capture_attention_stats(...): ...
```

The metrics layer should consume a common representation regardless of architecture.

This is important because the paper intends to compare:

- standard Transformers;
- gated Transformers;
- PRA/sparse models;
- recurrent or iterative architectures;
- eventually non-Transformer local-rule systems.

---

# 19. Numerical correctness

Instrumentation must not modify model outputs.

Add a parity test:

1. run the untouched model;
2. run the instrumented model with hooks enabled;
3. compare logits.

Target exact equality where implementation permits.

Otherwise require a documented numerical tolerance.

Do this before collecting experimental results.

Gate interventions are separate experimental modes and must never be mixed with native-forward parity tests.

---

# 20. Paper patch

Add a compact subsection introducing the gated-attention comparison.

The subsection should make approximately this argument:

> A pretrained gated-attention Transformer provides a direct test of the proposed computational-selection interpretation of residual dynamics. Unlike standard residual updates, the architecture exposes a learned gate controlling whether an attention-derived candidate transformation is admitted into the residual stream. We therefore compare the learned gate with independently defined measures of refinement, complementarity, interference, stability, and downstream utility.

Do not make the paper primarily about this one architecture.

It is one high-value probe in a broader cross-architecture study.

---

# 21. Main hypotheses

Pre-register or explicitly state hypotheses before interpreting results.

## H1 — Gate/update relation

Low gate values will be enriched among candidate updates that independent metrics classify as destabilizing or interfering.

## H2 — Useful complementarity

High gates need not imply high positive cosine alignment.

Useful near-orthogonal/complementary transformations may receive high gate values.

This is important because otherwise the gate could be reduced to a redundancy/refinement detector.

## H3 — Gating reduces effective interference

The distribution of interference metrics for

\[
g_l \odot a_l
\]

will differ from the distribution for raw candidate updates \(a_l\), with harmful extremes attenuated.

## H4 — Memory and computational selection are distinct

Attention concentration/dilution metrics will not fully determine gate values.

The gate should carry information about update acceptance beyond attention allocation alone.

## H5 — Gate value predicts functional consequence

Gate value should predict task-relevant effects of removing or forcing the corresponding update better than simple update magnitude alone.

Treat all hypotheses as falsifiable.

Report negative results.

---

# 22. Particularly important negative outcomes

The following outcomes are scientifically valuable and must not be hidden:

- gate values correlate mostly with update magnitude but not interference;
- gates are almost constant in many layers;
- gate behavior differs radically by task;
- attention dilution predicts gates almost completely;
- residual cosine fails to predict gates;
- gated and ungated models show similar residual interference;
- forced-open gates barely change performance;
- gate correlations do not generalize across layers or heads.

These results would constrain the computational-selection theory.

---

# 23. Relation to mainstream gated-attention work

In related work, cite the official gated-attention study and implementation.

State accurately that the released work studies query-dependent sigmoid gating applied to softmax-attention outputs and reports effects on performance, sparsity/attention behavior, training stability, and long-context behavior.

Do not claim that the authors endorse the present paper's cognitive interpretation.

Our contribution is different:

> use the pretrained learned gate as an observable internal selection variable and test whether it corresponds to architecture-independent residual-dynamics metrics.

---

# 24. Reproducibility metadata

Record:

- exact HF repository;
- exact revision/commit if available;
- model variant;
- Transformers version;
- PyTorch version;
- dtype;
- device;
- tokenizer;
- context length;
- batch size;
- hook locations;
- gate tensor semantics;
- intervention mode;
- dataset revision;
- random seed.

The experiment must be rerunnable from a manifest/config file.

---

# 25. Acceptance criteria

The patch is complete only when all of the following hold:

- [ ] Gated-attention model loads from a documented revision.
- [ ] Closest available ungated baseline is included.
- [ ] Gate location and tensor semantics are verified from implementation.
- [ ] Native instrumented-forward parity test passes.
- [ ] Gate values are captured per layer/head/token where available.
- [ ] Candidate and effective residual updates are captured separately.
- [ ] Existing residual metrics run on both candidate and effective updates.
- [ ] Attention-dilution metrics are related to gate values.
- [ ] Gate–interference/complementarity correlations are computed.
- [ ] At least forced-open and native-gate intervention conditions run.
- [ ] Standard task metrics are reported.
- [ ] Raw artifacts are persisted.
- [ ] Paper contains a gated-attention methods subsection.
- [ ] Paper contains at least one gated-vs-ungated residual comparison figure.
- [ ] Negative/null findings are retained.
- [ ] Claims distinguish correlation from intervention evidence.
- [ ] Gated attention is framed as computational selection, not as proof of a universal cognitive mechanism.

---

# 26. Final scientific target

The experiment should let the paper answer a question that standard benchmark evaluation cannot:

\[
\boxed{
\text{What does a learned gate actually select in internal dynamical terms?}
}
\]

The strongest possible result would be evidence that gate values systematically track architecture-independent properties of candidate state transitions:

\[
\text{candidate computation}
\rightarrow
\{\text{refinement, complementarity, interference}\}
\rightarrow
\text{learned acceptance/suppression}.
\]

If that pattern generalizes across tasks and later across architectures, it supports the broader thesis that **computational selection** is a measurable principle distinct from, but complementary to, **memory selection**.

If it does not, preserve the negative result and use it to refine the theory.
