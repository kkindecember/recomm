# Phase 17 experiment workspace

Stage 17 is a mechanism-migration portfolio for improving normal-setting GRAM. It is not a 1:1 paper-reproduction track.

Current gate: `S17-FP0` source/data/fidelity freeze, full-data native adapter,
and the first full-port foundation contracts have passed. The isolated LATTE
Python 3.12 environment and commit-pinned SentenceT5 cache run as CPU-only
background preparation with stable status files. A dependent 512-item tokenizer
profile is queued in the background and may use only a genuinely idle non-GPU1
card for at most ten minutes; it blocks instead of taking a busy card. S17-2R remains closed and is
not rerun. The active path is full-data Native-PSID/Native-LATTE plus
GRAM-LATTE-Full, followed by separate SETRec repository-parity and
paper-faithful sparse-attention arms. Official test, Sports, D1, and D2 remain
sealed.

The active plan is
`plan/第十七阶段/GRAM_第十七阶段_S17-FP完整论文机制迁移与架构级大实验计划v0.1.md`.
Frozen source/data/config artifacts are under `artifacts/phase17/fullport/`.

Safety rules:

- Beauty/Toys official validation/test positions are never exposed to Phase 17 training jobs.
- Sports labels, predictions, and metrics remain sealed until explicit authorization.
- Jobs expected to exceed ten minutes run in a persistent background session and expose a stable JSON status file.
- Stage 17 has no fixed global GPU-count ceiling. Small probes use currently idle eligible cards; before a large experiment, the runner states the requested GPU count and per-card memory and waits for the researcher's allocation. The current planning baseline is usually one or two cards, not a hard cap. Each single-card job is still designed for about 30 GiB usable memory unless a later allocation says otherwise.
- A terminal scientific step produces one consolidated report under `report/第十七阶段/`.
- S17-4 canonical science is closed. Physical GPU1 continues the isolated
  run-NNNN post-success workload and is excluded from S17-2R preflight/smoke.
  S17-2R did not stop the GPU1 repeat, and its repeat metrics are excluded from
  the architecture-selection evidence.
