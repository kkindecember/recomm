# Stage14 M2：Pseudo-Cold 路径迁移筛查报告

> **日期**：2026-08-21
> **阶段结论**：`M2_STOPPED_AT_STAGE14_1 / FAIL_STOP_PATH_TRANSFER_STAGE14_1`
> **执行边界**：14-1 正式筛查完整结束；14-2 matched smoke、M3/M4 full training 与额外 seed 均不启动

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Verification Status: ANALYZED（正式 workload `rc=0`，输入、输出、Gate、资源与 holder 恢复均已核对；未做重复运行）
- Data Scope: Toys pseudo-cold validation；`test_opened=false`
- Primary Artifact: `artifacts/phase14/m2/pseudo_cold_screen_toys_formal/summary.json`

## 1. 一句话结论

Stage 14-1 不是工程崩溃，而是一次完整、可判定的科学负结果。A2 soft subtree distillation 没有以预注册要求显著优于 A1 top-1 hard cold-path CE：item-level paired-bootstrap 的 A2−A1 exact-path MRR 95% CI 下界为负。因此正式 verdict 为 **`FAIL_STOP_PATH_TRANSFER_STAGE14_1`**。按计划停止 R2PD path-transfer 路线，不用 A3 的较高点估计替换主 Gate，不调参、不换 seed rescue，也不进入 14-2/M3。

## 2. 协议与完整性

- pseudo-cold audit item：1,176；保留 warm item：4,785；从 student-readable train 删除 9,804 条相关 interaction；held evaluation 为 7,660 events / 1,155 unique items。
- 正式筛查固定 512 train transitions、512 held events、351 unique target items；A0–A3 使用同一 fresh `t5-small` clean base。
- A1：top-1 hard path CE；A2：soft subtree distillation；A3：A2 + frozen-v0 retention。
- teacher 为 item-disjoint R² teacher；目标候选与真实 cold item 从训练输入中隔离。
- held ground truth 只在四个 arm checkpoint 全部完成后打开用于 evaluation；没有用于训练。
- `historical_v0_checkpoint_used=false`、`historical_r2_teacher_used=false`、`pseudo_or_real_cold_interactions_used_for_training=false`、`test_opened=false`。
- frozen backbone SHA256：`7cc541f7d54c29243b22f46a861e1834a7b4ed4ea9ad81d7f45bc088ed938002`；全部正式输入另见 `input_file_sha256.json` 与 `open_file_manifest.json`。

## 3. 正式结果

| Arm | exact-path MRR | Recall@50 | mean prefix survival | hit events |
|---|---:|---:|---:|---:|
| A0 frozen reference | 0.000331 | 0.009766 | 0.147824 | 5/512 |
| A1 hard-path CE | 0.001219 | 0.023438 | 0.233436 | 12/512 |
| A2 soft subtree | 0.000915 | 0.021484 | 0.241722 | 11/512 |
| A3 A2 + retention | 0.001736 | 0.035156 | 0.248410 | 18/512 |

预注册 primary comparison 使用 item 作为 paired-bootstrap unit：

```text
A2 − A1 exact-path MRR point = +0.00006930
95% CI = [−0.00077904, +0.00086940]
n_items = 351, resamples = 10,000
```

Gate 判定：

| Gate | 结果 |
|---|---|
| A2−A1 exact-path MRR CI lower > 0 | **FAIL** |
| A2 vs A0 Recall@50 non-degradation | PASS |
| A2 vs A0 beam survival non-degradation | PASS |

主 Gate 失败已经足以裁决路线。A3 的 MRR、Recall@50 与 survival 点估计均最高，说明 retention arm 值得作为机制观察记录；但 A3 不是 A2−A1 primary comparison，也没有预注册为失败后的替代 Gate，不能据此宣称 soft path transfer 成立。

## 4. 结果解释边界

1. A2 相对 A1 的 item-paired point estimate 很小且 CI 明显跨 0，当前样本不能支持 soft subtree 优于 hard path CE。
2. event-level aggregate MRR 中 A2 还低于 A1；item-paired point 略正是因为统计单位和重复 event 权重不同，不能混用两种口径挑选结论。
3. teacher top-50 之外的 tail mass 均值为 0.8732，显示本次 soft target 很分散；这是解释 A2 困难的候选诊断，不是重开 temperature/top-M 调参的授权。
4. 512 events 只有 5–18 个 Recall@50 hits，效应估计仍稀疏；这正是使用 item-level paired bootstrap 和预注册 stop rule 的理由，而不是追加 seed 直到显著。
5. 没有读取 test，没有 Beauty 迁移，也没有多重比较后选择 A3；负结果只适用于本次冻结的 Toys pseudo-cold 协议。

## 5. 资源与运行状态

- 正式 workload runtime：1,315.99 s（21m56s）；后台运行，无自动 retry。
- GPU：RTX A6000 physical GPU5；运行前释放项目自有 holder 后约 28.8 GiB free。
- telemetry peak used：36,038 MiB；扣除同卡外部进程约 19,751 MiB 后，本 workload 峰值增量约 **16,287 MiB**。
- workload `rc=0`，四个 arm checkpoint、prediction、config、provenance 与 summary 均完整生成。
- 结束后同一 `gram_ablation_scan_gpu5` holder 已以原配置 `reserve_mib=18263` 恢复；恢复 PID `2083287`，即时验证实际占用 20,276 MiB，随后稳定为约 20,292 MiB。

由于 14-1 efficacy Gate 已失败，不再用 14-2 去外推 M3 full-update 预算。此前 358–488 GPU-hour active package 与 466–629 GPU-hour contingency ceiling 失效为“未执行历史预算”，不能写成已批准资源。

## 6. 工程试错合并摘要

本阶段只保留这一份报告；以下尝试不各自建立 report：

1. `pseudo_cold_screen_toys_pipeline_smoke`：本地 `t5-small` cache 缺少 PyTorch weights，在任何训练 step 前停止。
2. `..._pipeline_smoke_v2`：dummy normalizer 将真实 token id 配到 size-2 假词表，clean base 完成但 adaptation 结果未接纳。
3. `..._pipeline_smoke_v3`：修正后端到端通过；只有 2 events，明确标记 `scientific_gate_eligible=false`。
4. `..._shape_smoke`：以正式 `top_m=50 / beam=50` 通过 full-shape pipeline smoke；仍不用于 efficacy。
5. `..._formal`：512/512 正式运行完整结束，得到本报告的科学负结果。

每次工程失败均保留原 artifact，没有静默覆盖；正式 run 没有复用失败尝试的 checkpoint。

## 7. 阶段裁决与后续边界

- M2 裁决：**`M2_STOPPED_AT_STAGE14_1`**。
- 14-2 matched smoke：**取消/不执行**；它不能挽救已经失败的 transfer efficacy Gate。
- M3/M4：**禁止启动**；M3 原本也从未获得用户预算批准。
- SpecGR/GenRecEdit：M1 只完成兼容性审计，不能因 R2PD 失败而自动升级成本地 port/reproduction；若要转向它们，需要新计划、明确协议与资源授权。
- 本阶段没有可继续自动执行的实验。下一步只能是关闭第十四阶段当前 R2PD 主线，或由用户另行选择并批准新的研究分支。

## 8. 证据索引

- 正式状态：`artifacts/phase14/m2/pseudo_cold_screen_toys_formal/status.json`
- 正式结果：`artifacts/phase14/m2/pseudo_cold_screen_toys_formal/summary.json`
- 配置：`artifacts/phase14/m2/pseudo_cold_screen_toys_formal/config.json`
- provenance：`artifacts/phase14/m2/pseudo_cold_screen_toys_formal/data_provenance.json`
- 输入 hash：`artifacts/phase14/m2/pseudo_cold_screen_toys_formal/input_file_sha256.json`
- open manifest：`artifacts/phase14/m2/pseudo_cold_screen_toys_formal/open_file_manifest.json`
- GPU telemetry：`artifacts/phase14/m2/pseudo_cold_screen_toys_formal/gpu_telemetry.csv`
- pseudo-cold audit：`artifacts/phase14/m2/pseudo_cold_audit_toys_v2/summary.json`
- item-disjoint teacher：`artifacts/phase14/m2/item_disjoint_r2_teacher_toys/summary.json`
