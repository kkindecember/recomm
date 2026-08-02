# GCGD P0 GPU smoke failed start — 2026-08-02

- Experiment: `GRAM_PHASE7_GCGD_P0_GPU_SMOKE_V1`
- Start: `2026-08-02T13:58:19+08:00`
- Final status: `failed`, scientific exit code `1`
- Failure stage: before model and dataset loading
- Root cause: the scientific workload did not inherit the project-local Hugging Face cache path; `local_files_only=True` therefore could not resolve `t5-small` from the default cache.
- Data integrity: fresh validation, test predictions, and Sports were not read.
- Scientific configuration change: none.
- Automatic retry: none.
- Resource restoration: CodeLlama was restored on physical GPU0 with the 30 GiB reservation.
- Authorized repair: propagate `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` to the workload and add a tokenizer preflight before CodeLlama release.
