# CPIA-N1 Frozen Training-Prefix Audit

## Locked scope

- Datasets: Toys and Beauty.
- Cohort: 128 deterministic unique training users per dataset, each with at
  least five history items; use the latest eligible training-prefix example.
- Per user: audit the five most recent history items only.
- Model: frozen reproduced GRAM checkpoint; optimizer steps must be zero.
- Validation, test, and Sports are forbidden.

## Exact measurements

For every audited history item, locate the filtered native lexical-ID token
sequence exactly once in the coarse prompt and at least once in its
corresponding fine prompt (the first occurrence is the `item:` field). Mean-pool
and L2-normalize the final encoder states on each span. For a user's five items,
form the 5×5 coarse-to-fine cosine matrix.

- `top1_accuracy`: fraction of coarse spans whose highest-cosine fine span is
  the matching item.
- `chance_accuracy`: exactly `1 / 5`.
- `signal_excess`: `top1_accuracy - chance_accuracy`.
- `hard_margin`: matched cosine minus the largest mismatched cosine.
- `mismatch_rate`: fraction with `hard_margin <= 0`.
- `matched_minus_mean_mismatch`: matched cosine minus the mean mismatched
  cosine.

All aggregate uncertainty is a deterministic 1,000-replicate user bootstrap.

## Locked scientific gate

Both datasets must independently satisfy all conditions:

1. `top1_accuracy >= 0.30` and its bootstrap 95% lower bound is strictly above
   chance 0.20, establishing a nontrivial bridge signal;
2. `top1_accuracy <= 0.60`;
3. median `hard_margin <= 0.05`;
4. `mismatch_rate >= 0.30`;
5. mean `matched_minus_mean_mismatch` bootstrap 95% lower bound is strictly
   positive.

The last four constraints establish that the bridge is weak rather than
already reliable. Failure yields `STOP_CPIA_NO_ACTIONABLE_LINK_DEFICIT`.

## Locked integrity gate

- 128 unique users and 640 item spans per dataset;
- exact coarse span mapping rate, exact fine span mapping rate, attention-mask
  validity, and finite-value rate all equal 1.0;
- each audited item maps to its corresponding fine passage by construction;
- checkpoint SHA is unchanged;
- optimizer steps are zero;
- no validation/test/Sports read.

Integrity failure yields `EXECUTION_INVALID`. Passing all gates yields
`CPIA_S0_DESIGN_ALLOWED`.
