# Selective Computation and Residual Dynamics

Mechanism-first research line for studying selective influence over neural state dynamics.

Paper 0 defines the dual-selection framework:

- **memory selection**: which available information becomes computationally active;
- **computational selection**: which candidate transformations become effective state updates.

Paper 1 is the first experimental instantiation. It tests whether Transformer residual blocks
refine, complement, or interfere with task-conditioned trajectories, and whether explicit gates
select updates with independently measurable downstream consequences.

Start with:

- `papers/paper0/paper0_selective_computation_position.tex` — position and measurement framework;
- `papers/paper1/paper1_residual_inhibition.tex` — falsifiable E1–E8 experimental design;
- `METRICS.md` — canonical terms, quantities, and interpretation limits;
- `ROADMAP.md` — evidence-conditional paper sequence.

## Evidence discipline

Descriptive geometry is not a mechanism. In particular:

- negative state–update cosine is anti-alignment, not interference;
- high attention entropy is diffusion, not necessarily distraction;
- a low gate is suppression, not necessarily beneficial inhibition;
- soft gating does not save compute when the candidate update was already evaluated.

Strong interference requires all three:

1. statistically reliable negative task-conditioned causal block utility;
2. later repair, reversal, or compensation;
3. replication across examples and independently trained seeds.

## Experimental priority

### Mandatory core

- **E1** — baseline residual map on tiny decoder-only Transformers;
- **E2** — same-content/different-goal counterfactuals;
- **E3** — matched gated residual stream.

### Secondary

- **E4** — shared distributed goal latent;
- **E5** — combined goal-conditioned gating.

### Conditional

- **E6** — hard skipping and quality–cost Pareto fronts;
- **E7** — pretrained gated-attention and evidence-selected architecture transfer;
- **E8** — PRA memory-selection × computational-selection factorial.

Do not expand E6–E8 until E1–E4 identify a robust phenomenon.

## Python package

The implementation lives in `src/gated_residuals` and keeps model-specific hooks behind adapters.
Its main modules are:

- `records.py` — validated candidate/effective update captures;
- `residual_dynamics.py` — geometry, causal utility, bootstrap intervals, repair detection;
- `attention_dilution.py` and `memory_activation.py` — memory-selection measurements;
- `gate_metrics.py` — gate distributions, autocorrelation, and gate–metric relations;
- `temporal_stability.py` and `head_layer_similarity.py` — multiscale dynamics;
- `causal_ablation.py` — block and gate interventions;
- `artifacts.py` — manifests plus CSV/Parquet outputs;
- `adapters/qwen3_gated.py` — released Qwen3 headwise-gate probe.

The small `gated_residuals/common` compatibility layer is copied from the model-agnostic
`pdattention/src/common` utilities so runs do not depend on a sibling checkout. The exact PRA
recall/materialization metric is reused by `memory_activation.py`.

## One-command smoke test

From the repository root:

```bash
python -m pytest -q
```

The analytic tests cover known residual cosines, causal utility, amplification–repair trajectories,
attention entropy/support, gate interventions, temporal metrics, PRA recall sparsity, artifacts,
and exact native-forward parity for a Qwen-style gated adapter fixture.

## Reproduce staged Paper 1 experiments

Run from the repository root with the package source on the Python path. Each stage writes its
config, run manifests, seed summaries, and derived records beneath `results/`.

```bash
PYTHONPATH=src python scripts/run_e1.py
PYTHONPATH=src python scripts/run_e2.py
PYTHONPATH=src python scripts/run_e3.py
PYTHONPATH=src python scripts/run_e4.py
PYTHONPATH=src python scripts/run_e5.py
```

E1 uses five seeds and a frozen, content-family-disjoint test split. Pass
`--reuse-checkpoints` to regenerate E1 analyses without retraining.
E2 reuses those checkpoints and performs only paired counterfactual analysis.
E3 trains the five-seed matched-capacity ungated/gated matrix; pass `--reuse-checkpoints` to
regenerate its control analyses.
E4 trains shared-goal and z-only variants and fits validation-to-test future-utility probes.
E5 runs the prespecified combined goal-conditioned-gating falsification test.

## Released gated-attention comparison

The pinned experiment configuration is `configs/pretrained_gated_attention.yaml`.

- model repository: `QwQZh/gated_attention`;
- model revision: `aad415c45ec6b4fa727ef3ff3f4e9f62f958d49b`;
- official implementation: `qiuzh20/gated_attention`;
- code revision: `f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2`;
- primary pair: `1B_baseline` versus `1B_gate_headwise`.

Source inspection confirms the headwise gate has shape `[batch, token, query_head, 1]`, is
query-dependent and sigmoid-valued, and multiplies the per-head SDPA output before flattening and
`o_proj`. Candidate and effective updates are recorded separately. Native parity and gate
interventions are separate execution modes.

## Artifact contract

Every experiment must save:

- config, seed, repository commit, environment, package versions, dtype, and device;
- model/tokenizer/dataset revisions and context/batch settings;
- exact hook locations, tensor semantics, and intervention mode;
- derived JSON/Parquet observations and paper-facing CSV summaries.

Raw observations retain run, model, dataset, task, example, token, layer, and head identifiers.
Do not retain large raw activations indefinitely unless a result requires them for reproducibility.

## Evidence-driven pivot

After E1–E4:

- harmful and repaired blocks → residual-interference characterization;
- strong future-utility prediction from goal state → distributed task control;
- modulation with quality gains → goal-modulated residual computation;
- quality improves while hard active compute falls → sufficient/selective activation;
- weak or null effects → comparative residual dynamics and falsification.

The scientific result determines the eventual Paper 1 title and claim.
