# SCDL-N1 Invalid Run Audit

## Attempt 1

- Status: invalid before data loading
- Cause: repository-local Hugging Face cache environment variables were absent, so
  the frozen local tokenizer could not be resolved.
- Scientific output: none
- Resolution: exact command rerun with `HF_HOME` and `TRANSFORMERS_CACHE` pointing
  to the existing repository cache; code/config/data unchanged.

## Attempt 2

- Status: invalid during post-Toys integrity summarization
- Cause: the finite-value integrity check referenced nonexistent set-row field
  `current_margin` instead of the already computed `current_mean_margin`.
- Scientific decision: none; no `summary.json` was produced.
- Partial artifacts: Toys CSV files were written but are invalid-run artifacts and
  will be deterministically overwritten by the exact rerun.
- Resolution: change only that field reference, rerun unit tests, update the
  mandatory code SHA, and execute the same locked audit. No cohort, metric,
  threshold, representation, input, or conclusion changed.
