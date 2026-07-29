# CPIA-N1 Decision

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: VERIFIED
- Data Scope: Toys/Beauty training prefixes only
- Validation/Test/Sports Read: `false`

## Fixed decision

- Decision: **`STOP_CPIA_NO_ACTIONABLE_LINK_DEFICIT`**
- Integrity valid: `true`
- Optimizer steps: `0`
- Checkpoint SHA unchanged: `true`

## Frozen gate results

| Dataset | Top-1 accuracy | Median hard margin | Mismatch rate | Mean matched-minus-mismatch 95% CI | Scientific gate |
|---|---:|---:|---:|---:|---|
| Toys | 98.91% | 0.2576 | 1.09% | [0.3229, 0.3416] | FAIL |
| Beauty | 79.53% | 0.0791 | 20.47% | [0.1342, 0.1430] | FAIL |

Both domains exceed the locked 60% top-1 ceiling, exceed the 0.05 median
hard-margin ceiling, and fall below the 30% mismatch-rate floor. The repeated
native lexical IDs already identify their corresponding fine passages too
reliably to support the preregistered weak-link premise.

Only `CPIA_S0_DESIGN_ALLOWED` permits a later correctness-smoke design. CPIA is
closed without threshold changes, post-hoc cohort construction, or a
validation-driven rescue.
