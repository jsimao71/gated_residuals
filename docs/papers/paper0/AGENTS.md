# AGENTS.md — Paper 0: Selective Computation and Residual Dynamics

## Mission

Create and maintain Paper 0 as the research-line position paper for the residuals / metrics / gated residuals project.

Paper 0 must sit above Paper 1. It should define the conceptual framework, metric ontology, paper-series roadmap, and implementation boundaries so that reusable infrastructure is not accidentally tied to one experimental paper.

## Core thesis

The project studies:

> selective influence over neural state dynamics.

The two central mechanisms are:

1. **Memory selection** — which available information becomes computationally active.
2. **Computational selection** — which candidate transformations become effective state updates.

Use the canonical decomposition:

\[
\mathcal{M}
\xrightarrow{S_M}
M_l
\xrightarrow{F_l}
\Delta_l
\xrightarrow{S_C}
g_l\Delta_l
\rightarrow
x_{l+1}.
\]

## Required Paper 0 content

Paper 0 must include:

- motivation for internal mechanism-first study;
- distinction between memory selection and computational selection;
- residual dynamics framework;
- refinement / complementarity / interference taxonomy;
- strong interference criterion;
- metric ontology;
- relation to PRA and attention dilution;
- relation to gated attention and gated residuals;
- latent/context/goal gating;
- inter-layer and top-down modulation;
- sufficient activation;
- cautious embodied/natural-cognition bridge;
- falsifiability and null-result policy;
- roadmap for Papers 1--5.

## Scope discipline

Paper 0 is a position and roadmap paper.

It should not become:

- a full survey of attention;
- a full survey of gating;
- a PRA paper;
- a neuroscience review;
- a benchmark paper;
- a specific architecture proposal.

Its contribution is the conceptual and measurement framework.

## Relationship to Paper 1

Paper 1 is the first experimental instantiation.

Do not move Paper 1 details into Paper 0 except as examples.

Paper 1 owns:

- tiny Transformer baseline residual map;
- same-content/different-goal counterfactual NLP tasks;
- residual gates;
- shared goal latent \(z\);
- combined modulation;
- small pretrained gated-attention comparison;
- E1--E8 execution plan.

Paper 0 owns:

- terminology;
- reusable metric definitions;
- paper line roadmap;
- distinction between architecture-specific mechanisms and architecture-independent quantities.

## Required repository updates

Create or maintain:

- `docs/papers/paper0/paper0_selective_computation_position.tex`
- `docs/papers/paper0/AGENTS.md`
- `docs/METRICS.md`
- `docs/ROADMAP.md`

Optionally patch README to explain the paper-line structure.

## Metric discipline

Never call a metric result a mechanism without causal support.

In particular:

- negative cosine is anti-alignment, not interference;
- high attention entropy is diffusion, not necessarily distraction;
- low gate is suppression, not necessarily inhibition;
- task classification from \(z\) is not sufficient evidence of computational control.

Strong interference requires:

1. negative causal utility;
2. later repair/reversal/compensation;
3. replication.

## Paper-series roadmap

Maintain this evidence-conditional roadmap:

1. Paper 0 — position and metric program.
2. Paper 1 — residual interference, gated residuals, and first pretrained gated-attention comparison.
3. Paper 2 — goal-latent computational selection.
4. Paper 3 — top-down / inter-layer / recurrent modulation.
5. Paper 4 — sufficient activation and adaptive compute.
6. Paper 5 — PRA memory selection × computational selection.

Do not hard-commit downstream papers before Paper 1 evidence is available. Present the roadmap as a working series.

## Acceptance criteria

- [ ] Paper 0 defines memory selection and computational selection separately.
- [ ] Paper 0 explains how attention/PRA and gates/residual modulation are complementary.
- [ ] Paper 0 defines refinement, complementarity, and interference with causal caution.
- [ ] Paper 0 does not overclaim biological equivalence.
- [ ] `METRICS.md` is consistent with Paper 0.
- [ ] Paper 1 remains experimentally focused.
- [ ] Roadmap identifies what is deferred to later papers.
- [ ] Null results are explicitly allowed to change the theory.
