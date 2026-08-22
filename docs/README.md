# Selective Computation and Residual Dynamics

Mechanism-first research line for studying selective influence over neural state dynamics.

Paper 0 defines the dual-selection framework:

- **memory selection**: which available information becomes computationally active;
- **computational selection**: which candidate transformations become effective state updates.

Paper 1 is the first experimental instantiation. It tests whether Transformer residual blocks
refine, complement, or interfere with task-conditioned trajectories, and whether explicit gates
select updates with independently measurable downstream consequences.

Cycle B extended that fixed first-cycle baseline. B1 ran the exhaustive residual atlas over all
30 stored Cycle A checkpoints before new training. B2 is also complete: self-attention and
feed-forward writes are captured at distinct pre-norm locations and intervened on separately with
exact native parity. B3's stability and amplification--repair atlas is complete, and B4 completed
the five-condition depth/capacity sweep. B5 now supplies a competence-matched five-task ecology.
Cycle C now tests learnability and conditional allocation. C1 is complete: dense, static-scale,
whole-block gated, and separate SA/FF-gated depth-16 models were trained for identical steps and
tokens across five paired seeds. Dynamic gates changed update amplitude but did not reliably
improve learning-curve AUC or final quality.

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
PYTHONPATH=src python scripts/run_e6.py
HF_HOME=/path/to/hf-cache PYTHONPATH=src python scripts/run_e7.py
PYTHONPATH=src python scripts/run_e8.py
PYTHONPATH=src python scripts/run_b1.py
PYTHONPATH=src python scripts/run_b2.py
PYTHONPATH=src python scripts/run_b3.py
PYTHONPATH=src python scripts/run_b4.py
PYTHONPATH=src python scripts/run_b5.py
PYTHONPATH=src python scripts/run_b6.py
PYTHONPATH=src python scripts/run_c1.py
```

E1 uses five seeds and a frozen, content-family-disjoint test split. Pass
`--reuse-checkpoints` to regenerate E1 analyses without retraining.
E2 reuses those checkpoints and performs only paired counterfactual analysis.
E3 trains the five-seed matched-capacity ungated/gated matrix; pass `--reuse-checkpoints` to
regenerate its control analyses.
E4 trains shared-goal and z-only variants and fits validation-to-test future-utility probes.
E5 runs the prespecified combined goal-conditioned-gating falsification test.
E6 is an evidence-gated audit; it does not run hard-skip sweeps when prior criteria fail.
E7 records scientific and exact-checkpoint resource eligibility. It never emits pretrained metrics
unless a real native-parity forward has completed.
E8 audits both factorial axes and does not modify the external PRA checkout.
B1 reuses all E1/E3/E4/E5 checkpoints and writes compact derived Parquet tables, CSV summaries,
full layer-pair matrices, and a layer-by-token atlas under
`artifacts/cycle_b/b1_residual_atlas`. It computes per-example/token/head quantities but does not
persist raw activation tensors. Pass `--max-checkpoints 1 --output tmp/b1-smoke` for a bounded
instrumentation smoke run.

## Cycle B status

- **B1 complete**: 30 checkpoints, 1,152,840 residual observations, 4,611,360 head-level attention
  observations, and zero non-finite required metrics.
- Residual states were more similar across layers than candidate writes (off-diagonal cosine
  0.710 versus 0.200; CKA 0.652 versus 0.288).
- Baseline final-token state--write cosine progressed from -0.406 at layer 0 to 0.329 at layer 3.
  Because Cycle A learned-task block utility stayed positive, this is descriptive anti-alignment,
  not interference.
- **B2 complete**: exact native-logit parity, Cycle A probability parity within `5.96e-8`,
  192,140 component observations, and 5,400 paired causal observations across five baseline seeds.
  SA was more useful than FF in seven of eight learned-task/layer cells; both were positively useful,
  no intra-block repair candidate appeared, and no maximum/minimum task-conditioned contrast
  replicated. Artifacts are under `artifacts/cycle_b/b2_sa_ff`.
- **B3 complete**: 5,400 stability records and 12,150 intervention trajectories. Twenty-three
  exploratory correct/incorrect cells replicated and concentrated in late FF statistics, but no
  perturbation showed both vector recovery and task-gap recovery. H-B4 remained unconfirmed and the
  B6 gating gate stayed closed at this stage. Artifacts are under `artifacts/cycle_b/b3_stability`.
- **B4 complete**: 25 runs across depths 4/8/12/16 plus a 16-layer width-32 matched-parameter
  control, with 50,400 causal records and exact capture parity. Width-64 accuracy rose from 0.890
  to 0.948 while effective-block fraction fell from 1.000 to 0.775, but the registered strict
  competence gate failed, so this is not claimed as competence-preserving depth slack. Two
  minimum-task FF cells had replicated negative utility, one late SA cell was reliably near zero,
  and joint repair remained absent. This opened a B6 evidence trigger subsequently validated by
  B5. Artifacts are under `artifacts/cycle_b/b4_depth_sweep`.
- **B5 complete**: after validation-only task/difficulty pilots, the confirmatory five-task ecology
  reached 0.98--1.00 accuracy and all ten task pairs were competence matched. Thirteen block/SA
  utility contrasts replicated across seven pair--layer rows. Minimum layer 8 had negative block
  and FF utility, while ordinary-residual goal-probe accuracy peaked at 0.739 mid-depth (chance
  0.20). B6 is now enabled by task-conditioned and negative-utility evidence, not by repair.
  Artifacts are under `artifacts/cycle_b/b5_task_ecology`.
- **B6 complete**: ten deep gated runs compared whole-block and SA/FF-specific gates. Native
  accuracy (0.991/0.993) matched the B5 baseline (0.992); forcing component gates open reduced
  accuracy by 0.024 and shuffling reduced it by 0.028. Mean block-equivalent gates were 0.797 and
  0.757, but gate--candidate-norm correlations exceeded gate--utility correlations. The negative
  minimum layer-8 FF effect persisted, and all soft-gated candidates were still evaluated. B6
  supports functional amplitude modulation, not quality improvement or compute saving. Artifacts
  are under `artifacts/cycle_b/b6_gating`.
- **B7 complete with a bounded resource stop**: cached Qwen3-0.6B completed exact native parity,
  a 24-example predictable-sequence residual/attention atlas, and all 28-layer SA/FF/block
  ablations. Every layer had positive mean SA and block utility; three FF means were negative but
  all three confidence intervals crossed zero. FF update magnitude tracked FF utility more strongly
  than the analogous SA relation, while attention entropy did not track SA utility. The pinned
  headwise gated release still could not fit measured RAM/VRAM headroom, so it has zero evaluated
  examples and no gate result. Artifacts are under `artifacts/cycle_b/b7_qwen`. B8/B9 remain closed
  because the matched Qwen comparison did not complete end to end.
- **B8 conditional stop complete**: the Llama-family stage checked the persisted B7 decision and
  did not select, download, or evaluate a checkpoint because real gated-Qwen parity and the
  reserve-aware resource gate both failed. This is not a Llama result. Decision artifacts are
  under `artifacts/cycle_b/b8_llama`.
- **B9 conditional stop complete**: the Gemma-family stage remained ineligible because neither the
  end-to-end Qwen comparison nor a Llama replication completed. It selected and downloaded no
  checkpoint and produced no Gemma metrics. Decision artifacts are under
  `artifacts/cycle_b/b9_gemma`. Cycle B1--B9 is now fully accounted for; B8/B9 can be resumed only
  after their upstream gates change.

## Cycle C status

- **C1 complete**: 20 runs, each with 456 optimizer steps and 924,168 non-padding training tokens.
  Mean normalized learning-curve AUC was 0.7820 dense, 0.7823 static scale, 0.7796 whole-block
  gated, and 0.7854 separate SA/FF gated. Every paired AUC interval versus dense included zero.
  Frozen-test accuracy was 0.991--0.993 across variants; one whole-block-gate seed did not reach
  99% validation competence. Gates attenuated effective update norms, but did not establish a
  learning-speed or final-quality advantage. Artifacts are under
  `artifacts/cycle_c/c1_learning_curves`.
- **C2 complete**: 120 new runs plus 20 reused C1 cells tested high/medium/low goal-cue
  directness, mixed-cue diversity, and depths 4/8/12/16. Every variant reliably learned all
  single-cue regimes, but none passed the mixed-cue criterion at depth 16. Static scaling alone
  qualified at depth 8 (.901 mean validation; 4/5 seeds), then failed at depths 12/16 and did not
  reliably improve frozen-test accuracy. This is an isolated qualifying cell, not a monotone
  minimum-depth result. Artifacts are under `artifacts/cycle_c/c2_difficulty_depth`.

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
