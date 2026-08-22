# C1 failure and null-result notes

- All variants receive identical data, optimizer, schedule, seed set, steps, and token budget.
- Static scales are unconstrained learned layer constants initialized at 1; dynamic gates are sigmoid multipliers initialized at sigmoid(2).
- Missing competence times are persisted as null and counted explicitly; they are not imputed.
- Wall time includes periodic validation and diagnostics, so cross-variant timing is descriptive rather than a pure training-throughput benchmark.
- Soft gates compute every SA/FF candidate and imply no realized FLOP saving.
