# CET-C2 Initial Attempt — Invalid Engineering Run

- Fixed decision: **`INVALID_RUN_FIX_AND_EXACT_RERUN`**
- Validation read: `false`
- Test read: `false`
- Sports read: `false`
- Cause: the two-sample smoke selected histories with no maskable old fine passage,
  recorded `masked_passages=0`, but the smoke implementation incorrectly labeled the
  finite forward/backward as PASS.
- Scope reached before termination: Toys C0 training completed; Toys C1 was interrupted
  during epoch 1. No candidate was evaluated.
- Corrective action: preserve this directory unchanged; require the replacement smoke
  to select histories with maskable passages, use mask probability 1.0 only for path
  coverage, and hard-fail when zero passages are masked. The registered training
  probability remains 0.25.
- Replacement output root: `artifacts/phase5/cet_c2_run2`.
