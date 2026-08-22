# C2 failure and null-result notes

- High-identifiability depth-16 cells reuse C1 fixed-final-state runs; their curves are downsampled to C2's 20-step grid.
- Difficulty means goal-cue directness (high/medium/low identifiability). Mixed cues are reported separately because cue diversity is not ordinal directness.
- The mixed-cue depth sweep was selected from the pre-existing B5 validation pilot, not C2 test behavior.
- Reliable competence requires mean validation accuracy at least 0.90 and at least four of five seeds individually at least 0.90.
- A lowest qualifying depth is not called a minimum necessary depth unless every tested deeper model also qualifies; the observed depth relation may be non-monotonic.
- Every listed cell was fixed before C2 training; no depth or cue condition was added after frozen-test inspection.
- Soft gates still evaluate all candidate writes and imply no avoided FLOPs.
