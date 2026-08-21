# METRICS.md — Residuals / Selective Computation Metric Ontology

## Rule zero

Metrics are instruments for testing mechanisms, not interpretability decorations.

Do not treat any geometric metric alone as proof of a cognitive or mechanistic category.

In particular:

\[
\cos(x,\Delta)<0
\]

means anti-alignment. It does **not** by itself mean interference.

Strong interference requires causal and temporal evidence.

---

# 1. Canonical quantities

For a model layer or block \(l\), token position \(t\), and example \(e\):

- \(x_{l,t}\): state entering the candidate transformation.
- \(\Delta_{l,t}\): candidate update proposed by the block.
- \(\tilde{\Delta}_{l,t}\): effective update after gating/modulation/skipping.
- \(x_{l+1,t}\): state after the update.
- \(g_{l,t}\) or \(g_{l,h,t}\): gate, if exposed.
- \(A_{l,h,t}\): attention distribution, if exposed.
- \(Q\): task quality metric.
- \(U_l(\tau)=Q_\tau(\mathrm{full})-Q_\tau(\mathrm{skip}\ l)\): causal block utility.

Use exact tensor-location names in implementation metadata. Pre-norm, post-norm, attention-output, MLP-output, residual-pre, and residual-post are not interchangeable.

---

# 2. Memory selection metrics

Memory selection asks:

> Which available information becomes computationally active?

Examples:

- attention entropy;
- top-\(k\) mass;
- effective support;
- evidence/supporting-fact mass;
- attention sink strength;
- retrieved references/chunks/gists;
- active/available KV ratio;
- routing precision/recall/rank;
- materialization breadth;
- memory budget vs quality.

Useful formula:

\[
N_{\mathrm{eff}}=\exp\left(-\sum_i \alpha_i \log \alpha_i\right).
\]

Interpretation limits:

- Attention is interaction structure, not automatically explanation.
- Diffuse attention is not automatically harmful.
- Sharp attention is not automatically useful.
- Retrieval recall and residual usefulness can diverge.

---

# 3. Computational selection metrics

Computational selection asks:

> Which candidate transformation becomes an effective state update?

When gates exist, preserve:

\[
g_{l,h,t}
\]

before averaging.

Minimum metrics:

- gate mean / median / variance;
- fraction below thresholds \(0.1,0.25,0.5\);
- fraction above thresholds \(0.75,0.9\);
- layer/head/token distributions;
- gate autocorrelation over token position;
- cross-head and cross-layer correlation;
- native vs forced-open vs forced-closed gate effects;
- shuffled gate controls.

If no explicit gate exists, infer computational selection only through causal ablation and utility.

---

# 4. Residual update geometry

For candidate update \(\Delta\):

\[
\|\Delta\|,\qquad
\frac{\|\Delta\|}{\|x\|},\qquad
\cos(x,\Delta),\qquad
\|x+\Delta-x\|.
\]

For gated/effective update:

\[
\tilde{\Delta}=g\odot\Delta.
\]

Compute the same metrics for both \(\Delta\) and \(\tilde{\Delta}\).

Interpretation:

- positive alignment can indicate refinement, but is not proof;
- near-orthogonal updates can indicate complementarity, but are not proof;
- negative alignment can indicate conflict, but is not proof.

---

# 5. Refinement, complementarity, interference

## Refinement

A transformation refines when it improves an existing trajectory, representation, or decision.

Candidate evidence:

- increased correct-answer margin;
- improved target log probability;
- increased task-variable decodability;
- positive causal utility;
- stable direction with reduced uncertainty.

## Complementarity

A transformation is complementary when it adds useful new information without destroying previous useful information.

Candidate evidence:

- new decodable factor;
- increased representational rank/subspace coverage;
- low-to-moderate alignment with previous state;
- positive causal utility for both earlier and later transformations.

## Interference

A transformation interferes when it damages useful computation or creates a state that later computation must repair.

Strong criterion:

1. statistically reliable negative task-conditioned utility \(U_l(\tau)<0\);
2. later repair/reversal/compensation evidence;
3. replication across examples and seeds.

Geometry alone is insufficient.

---

# 6. Temporal and stationarity metrics

Token position is a temporal axis.

Required families:

- rolling mean/variance/skewness/kurtosis;
- first and second differences;
- chunk distribution distances;
- covariance/correlation drift;
- ACF and ACF drift;
- lagged cross-correlation;
- effective correlation length;
- CKA/RSA across windows;
- principal-subspace drift;
- eigenspectrum drift;
- layer × scale × token stability maps.

Search specifically for amplification--repair events.

---

# 7. Head and layer organization

For a similarity or utility metric \(M\), compare:

\[
E[M(\mathrm{heads\ within\ layer})]
\]

against

\[
E[M(\mathrm{heads\ across\ layers})].
\]

Report variance decomposition by:

- layer;
- head;
- token position;
- task;
- goal family;
- dataset;
- model variant.

---

# 8. Causal ablation metrics

Required interventions when feasible:

- skip one block;
- skip selected block groups;
- ablate attention output;
- ablate MLP output;
- force gate open;
- force gate closed;
- replace dynamic gate with mean gate;
- shuffle gate across examples/tokens;
- replace update with norm-matched random vector;
- patch states between same-content/different-goal examples.

Always report whether intervention preserves tensor shape, dtype, norm scale, and model numerical stability.

---

# 9. Standard task and system metrics

Internal metrics must connect to task/system effects:

- loss;
- perplexity;
- accuracy;
- exact match;
- F1;
- precision/recall;
- calibration;
- logit margin;
- target log probability;
- output entropy;
- FLOPs;
- active FLOPs;
- realized latency;
- memory use;
- throughput.

---

# 10. Artifact schema

Minimum row fields for derived metric tables:

```text
run_id
model
model_variant
dataset
task_family
example_id
token_index
layer
head
state_location
intervention
seed
quality_metric
loss
target_logprob
gate
candidate_update_norm
effective_update_norm
residual_norm
candidate_cosine
effective_cosine
attention_entropy
attention_top1_mass
attention_topk_mass
block_utility
repair_score
```

Prefer Parquet for large metric tables and CSV for paper summaries.

---

# 11. Naming discipline

Use:

- **memory selection** for selection among inputs/memory/tokens/references;
- **computational selection** for admission/suppression of candidate transformations;
- **candidate update** for raw proposed transformation;
- **effective update** for post-gate/post-modulation update;
- **interference candidate** for geometry-only or ablation-only findings;
- **strong interference** only after the full criterion is met.

Never write that "cosine proves interference."
