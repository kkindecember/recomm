# Phase 17 experiment workspace

Stage 17 is a mechanism-migration portfolio for improving normal-setting GRAM. It is not a 1:1 paper-reproduction track.

Current gate: the `S17-FP0` source/data/fidelity freeze, native adapter, pinned
CUDA 12.6 environment, offline SentenceT5 cache, bounded tokenizer profile, and
full-data tokenizer have passed. Full-data tokenizer `attempt_001` ran once on
the explicitly authorized physical GPU0 and completed in 111.28 seconds with a
984 MiB peak reservation. Whitened PCA and RQ-KMeans fit only the frozen 11,138
train-prefix items; all 11,924 metadata items received three 0--255 codes; 1,337
collisions were reassigned and zero aliases remain. The generated semantic-ID
JSON is byte-identical to the official LATTE `.sem_ids` cache. No FP1/FP2 effect
experiment has run. The immutable tokenizer attempt exported 775 observed
tokens; additive `amendment_001` supplies the one unobserved valid codebook
token (`<s17_sid1_236>`) and freezes the complete 3x256+8 = 776-token G1/G2
inventory without editing the tokenizer attempt or official `.sem_ids` cache.
All five arm-specific resource-profile executors and CPU preflights are now
prepared as `attempt_001`; readiness is
`READY_FOR_ARM_SPECIFIC_PROFILE_AUTHORIZATION`. Their launch gates remain
closed. The 2026-08-31 22:46:57+08:00 read-only resource snapshot found every
assigned card below its frozen free-memory gate, so no GPU profile was started.
The exact deficits and PID-preservation set are frozen in
`artifacts/phase17/fullport/profiles/profile_authorization_request_001.json`.
Formal FP1/FP2 launch remains unauthorized. S17-2R remains closed and is not
rerun. Official test, Sports, D1, and D2 remain sealed.

The active plan is
`plan/第十七阶段/GRAM_第十七阶段_S17-FP完整论文机制迁移与架构级大实验计划v0.1.md`.
Frozen source/data/config artifacts are under `artifacts/phase17/fullport/`.

Safety rules:

- Beauty/Toys official validation/test positions are never exposed to Phase 17 training jobs.
- Sports labels, predictions, and metrics remain sealed until explicit authorization.
- Jobs expected to exceed ten minutes run in a persistent background session and expose a stable JSON status file.
- Stage 17 has no fixed global GPU-count ceiling. Small probes use currently idle eligible cards; before a large experiment, the runner states the requested GPU count and per-card memory and waits for the researcher's allocation. The current planning baseline is usually one or two cards, not a hard cap. Each single-card job is still designed for about 30 GiB usable memory unless a later allocation says otherwise.
- A terminal scientific step produces one consolidated report under `report/第十七阶段/`.
- S17-4 canonical science is closed. Any GPU1 handoff for G0 requires a fresh
  PID/state freeze and explicit authorization; Stage17 never signals an
  existing process automatically. Runtime-maintenance metrics remain excluded
  from architecture-selection evidence.
