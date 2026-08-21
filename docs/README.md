# Selective Residual Computation — Paper 1

Mechanism-first research project for studying what residual depth actually does in language models.

## Core question
Do Transformer blocks mainly refine one evolving solution, add complementary computation, or create task-dependent interference that later layers repair? If block utility is task-dependent, can a distributed inferred goal state predict useful transformations and can contextual modulation suppress or skip unnecessary computation?

## Why this is not just interpretability
Internal metrics are instruments for testing competing mechanisms and designing architectures. Strong interference evidence requires:
1. task-conditioned negative causal block utility,
2. later repair/reversal/compensation,
3. replication across examples and seeds.

## Metric families
1. PRA-style memory activation
2. Attention concentration / dilution
3. Residual refinement / complementarity / interference
4. Temporal and multiscale stationarity / stability
5. Head-vs-layer organization
6. Standard ML/DL, optimization, and systems metrics

The instrumentation is broader than the first paper so it can later support alternative architectures, local-rule systems, and neural comparisons.

## Datasets
Primary: controlled synthetic NLP counterfactual mixtures with same-content/different-goal pairs, implicit/distributed goals, conflicting local affordances, controlled goal distance, and `goal_identifiability`.

Continuity: WikiText-2, WikiText-103, HotpotQA, QASPER.

2-D worlds, mazes, Sudoku, and embodied-agent environments are deferred.

## Experimental priority
### Mandatory core
- E1 baseline residual map on tiny Transformers
- E2 counterfactual goal manipulation
- E3 gated residual stream

### Secondary
- E4 shared distributed goal latent
- E5 combined goal-conditioned gating

### Conditional / stretch
- E6 hard skipping and sufficient-activation frontier
- E7 pretrained / third-party architecture transfer
- E8 PRA memory × computation interaction

Do not expand the architecture matrix before E1–E4 identify a robust phenomenon.

## Architecture policy
Tiny Transformers are the discovery environment. Pretrained models initially test only the strongest qualitative transfer effects.

Choose third-party comparators based on results:
- residual interference -> mHC / Hyper-Connections
- conditional compute -> Mixture-of-Depths / LayerSkip
- iterative refinement -> TRM

## Novel interventions
Residual gate:
`h[l+1] = h[l] + g_l(z) * F_l(h[l])`

Shared goal state:
`z[l+1] = z[l] + G_l(z[l], h[l])`

The key test for `z` is not task classification; it should predict future causal block utility.

## Evidence-driven paper pivot
After E1–E4:
- harmful + repaired blocks -> residual interference
- strong utility prediction from goal state -> distributed task control
- modulation quality gains -> goal-modulated residual computation
- quality improves while compute drops -> sufficient/selective activation
- weak/null effects -> comparative residual dynamics / falsification

The title and proposed mechanisms are provisional. The scientific result decides the paper.

## Broader research line
- PRA: what memory becomes active?
- residual modulation: what transformations become active now?
- Neural Modules: what large functional areas are assembled for a task/session?
- Tree of Experts: how the repertoire differentiates developmentally
- local learning rules: how structures and dynamics are acquired

Candidate cross-cutting hypothesis: **Principle of Sufficient Activation** — under contextual constraints, unnecessary activation can be both resource-costly and functionally harmful.
