# AGENTS.md — Patch Paper 1 After Adding Paper 0

## Objective

Align the existing Paper 1 draft with the new Paper 0 position paper.

Paper 1 should no longer carry the entire research-line burden. It should explicitly present itself as the first experimental test of the Paper 0 framework.

## Required changes to Paper 1

1. Add a short paragraph in the introduction:

> Paper 0 defines the broader dual-selection framework. Paper 1 instantiates it in Transformer residual streams and asks whether residual blocks refine, complement, interfere, or can be modulated/skipped under task context.

2. Use Paper 0 terminology consistently:

- memory selection;
- computational selection;
- candidate update;
- effective update;
- causal block utility;
- strong interference.

3. Do not let Paper 1 define the whole paper series. Move broad series language to `ROADMAP.md` / Paper 0.

4. Keep the experimental priority unchanged:

- E1 baseline residual map;
- E2 same-content/different-goal counterfactuals;
- E3 gated residual stream;
- E4 shared goal latent;
- E5 combined modulation;
- E6--E8 conditional.

5. Integrate the existing gated-attention patch as E7 or a dedicated pretrained comparison subsection, without disrupting E1--E3.

## Acceptance criteria

- [ ] Paper 1 clearly references the Paper 0 framework.
- [ ] Paper 1 is not overloaded with full theory-of-cognition claims.
- [ ] Paper 1's main claims remain falsifiable.
- [ ] Gated attention is framed as an external pretrained probe of computational selection.
- [ ] Metrics terminology matches `docs/METRICS.md`.
