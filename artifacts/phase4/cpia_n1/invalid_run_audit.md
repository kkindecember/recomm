# CPIA-N1 Invalid Run Audit

## Attempt 1

- Status: invalid before Python execution and before data loading.
- Cause: the sandbox could not communicate with the NVIDIA driver, so the
  launcher's `nvidia-smi` GPU guard exited with code 9.
- Scientific output: none; no log or summary was produced.
- Resolution: verify GPU and disk guards outside the sandbox, then run the
  unchanged frozen command outside the sandbox.

## Attempt 2

- Status: completed forward audit but invalid during integrity summarization.
- Cause: every coarse/fine pair incremented `attention_checked` by two, while
  the joint-valid branch incremented `attention_valid` by one. Because an
  invalid member of the pair raises immediately, the reported rate was
  mechanically 0.5 despite all checked spans being valid.
- Invalid summary SHA-256:
  `39ccc30378e549a5ce816e59b4e0c976ef6165a00ac52bd78cd04a0dc4399277`.
- Invalid item-row SHA-256:
  `f393cb3cc2bc5016d43d89b6cc5bd7d3d9fe4a37a5c05dd25862076599e4486c`.
- Scientific decision: none. The emitted `EXECUTION_INVALID` result is not
  interpreted.
- Resolution: count both spans when the joint validity check passes, add a
  regression test, update only the mandatory code SHA, and execute the same
  locked audit. No cohort, representation, metric, threshold, model input, or
  allowed conclusion changes.
