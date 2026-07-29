# GRAM Experiment Artifact Cleanup — 2026-07-29

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: experiment artifact governance
- Verification Status: VERIFIED after deletion
- Source Plans:
  - `plan/GRAM_第三阶段_创新探索与渐进式实验计划.md`
  - `plan/GRAM_第四阶段_方法创新与渐进实验计划.md`
  - `plan/GRAM_第四阶段_续篇_Toys_Beauty非自适应方法创新计划.md`
- Scope: repository-local regenerable experiment artifacts only

## Retention policy

Keep:

- experiment source, unit tests, launchers, and frozen configs;
- `summary.json`, `decision.md`, preregistration, reports, evidence matrices,
  per-user metric tables, and invalid-run audit text;
- the locked original best checkpoints:
  - Toys epoch 30;
  - Beauty epoch 25;
- both GCDH-P0 C0 checkpoints, because the open `NEXT_N0_MECHANISM_REVIEW`
  workflow may reuse this frozen backbone;
- the repository-local Hugging Face cache required for offline execution.

Delete:

- optimizer states after training has ended;
- non-best baseline epoch checkpoints and the phase-2 smoke weights;
- all ten HBTR pilot model checkpoints after `STOP_HBTR`;
- both GCDH-P0 C1 checkpoints after the dual-head/residual family stopped;
- all TCDR S0/P0 trained decoder blocks after
  `STOP_TCDR_MECHANISM_GATE_FAILED`;
- RPCD teacher caches/SASRec weights and FCRD full-catalog caches after that
  family stopped;
- superseded/duplicate Toys prediction TSVs, while retaining the locked final
  validation and test files;
- four explicitly named invalid-run artifact directories.

No source code, configs, summaries, decisions, reports, split manifests, raw
datasets, final baseline checkpoints, GCDH C0 checkpoints, or offline model
cache are deletion targets.

## Pre-deletion inventory

- Repository size: 10,821,505,024 bytes.
- Planned removable bytes: approximately 8,319,000,000 bytes.
- Removal is destructive for untracked binary artifacts. Deleted weights and
  caches can be regenerated from retained code/config/data, but are not
  recoverable from Git.

## Post-deletion verification

- Repository size after cleanup: 2,501,918,720 bytes.
- Repository bytes released: 8,319,586,304 bytes (approximately 7.75 GiB).
- `/home` available space after cleanup: 215,133,548 KiB.
- Confirmed retained:
  - Toys epoch-30 and Beauty epoch-25 baseline checkpoints;
  - Toys/Beauty GCDH-P0 C0 checkpoints;
  - locked Toys/Beauty final validation and test prediction TSVs;
  - all experiment source/config/report/summary/decision artifacts;
  - repository-local Hugging Face model cache.
- Confirmed absent:
  - all deletion-target optimizer/model weights;
  - superseded and duplicate prediction TSVs;
  - the four named invalid-run directories.
- Closed-experiment launchers/configs that refer to deleted binary artifacts are
  retained as provenance. Re-running those closed experiments now requires
  regenerating their intermediate checkpoints/caches.
