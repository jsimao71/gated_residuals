# B1 failure and null-result notes

- No raw activation tensor was persisted; only derived per-example/token/layer/head metrics.
- Cycle A's tiny block capture combines SA and FF writes, so per-head output norms and SA/FF causal roles are intentionally deferred to B2.
- Minimum novelty is the registered $1-\cos^2(x,\Delta)$ statistic; a recent-update subspace version is deferred to B3.
- Cancellation and anti-alignment remain descriptive geometry, not interference claims.
