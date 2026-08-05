# GRAM 第十一阶段 BW3-P1C：Listwise 扩展准入纠偏结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-05
- Verification Status: `P1C_PASSED_P2_ELIGIBLE_NOT_AUTHORIZED`
- Version Label: `phase11_bw3_p1c_listwise_admission_result_v1`
- Experiment ID: `GRAM_PHASE11_BW3_P1C_LISTWISE_ADMISSION_CORRECTION_V1`

## 1. Executive conclusion

BW3-P1C 于 2026-08-05 10:47:47 至 10:49:23 +08:00 完成。Toys、Beauty 的严格 per-user
listwise gate 均通过 integrity 与 calibration scientific gate，聚合终态为
`passed_eligible_for_separate_p2_authorization`。

两域都选中 margin `0.0`，并在冻结 beam50 PCRF 上实现无 regression 的正增益：

- Toys Hit@10 `+0.066406`、NDCG@10 `+0.019294`、tail Hit@10 `+0.073529`；
- Beauty Hit@10 `+0.054688`、NDCG@10 `+0.015832`、tail Hit@10 `+0.036496`；
- Toys/Beauty promotion 分别为 `34/28`，regression 均为 `0`。

这说明 P1 中观察到的 train-prefix admission 信号不是 BCE objective 的假象；在修正为
按用户等权的 listwise objective 后，它仍在两域 calibration 上稳定存在。相比探索性
BCE gate，listwise gate 准入更保守，但没有牺牲 base hit。

P2 `t=-2` validation 仍未读取，本轮也没有自动启动 P2。P1C PASS 只赋予
另行讨论和授权 P2 的资格。

## 2. Frozen protocol

- 只复用成功 P1 recovery 的 8 个已锁 beam TSV，未复用首次中断运行产物；
- fit 使用 `t=-4`，calibration 使用 `t=-3`；
- Toys/Beauty 分域拟合，seed `2023`，Adam 200 epochs，lr `0.05`，L2 `0.001`；
- action set 为 `REJECT_TO_BASE + expansion candidates`，每用户 listwise CE 等权；
- target 在 beam50 时监督 reject，target 只在 expansion 时监督对应候选，union 外事件
  只计 attrition；
- base PCRF 保留 adjusted score 的第二层标准化，base top10 冻结；
- margin 只从 `{0, 0.25, 0.5, 0.75, 1.0}` 选择，最多准入 3 个候选；
- 无 admission 时严格 fallback 至原 base top10。

## 3. Fit and coverage attrition

| Dataset | Fit events | Included in listwise loss | Excluded outside union | Expansion-positive | Initial loss | Final loss |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 1,024 | 743 | 281 | 179 | 5.017281 | 0.854783 |
| Beauty | 1,024 | 658 | 366 | 172 | 5.017330 | 1.042477 |

两域 loss 均 finite 且大幅下降。fit action attrition 为：

| Dataset | Base top10 | Base rank 11–50 | Expansion only | Outside union |
|---|---:|---:|---:|---:|
| Toys | 501 | 63 | 179 | 281 |
| Beauty | 424 | 62 | 172 | 366 |

四类在每域精确加和为 1,024，与 membership 审计对齐。每用户 expansion pool size
为 150，无 empty pool。

## 4. Calibration results

| Dataset | Margin | Base Hit@10 | Final Hit@10 | Hit delta | NDCG delta | Tail Hit delta | Admissions / users | Fallback users | Promotion / regression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Toys | 0.0 | 0.542969 | 0.609375 | +0.066406 | +0.019294 | +0.073529 | 42 / 37 | 475 | 34 / 0 |
| Beauty | 0.0 | 0.457031 | 0.511719 | +0.054688 | +0.015832 | +0.036496 | 34 / 32 | 480 | 28 / 0 |

512 个 calibration 用户/域全部保存 per-user 记录，用户 ID 无重复。按 seed 2023 做
2,000 次 paired bootstrap，Hit@10 delta 的描述性 95% CI 为：

- Toys `[0.046875, 0.087891]`；
- Beauty `[0.035156, 0.074219]`。

这些 CI 来自用于 margin 选择的 calibration split，只表示样本内成对不确定性，不替代
P2 的一次性独立 validation。

## 5. BCE exploratory comparison

| Dataset | BCE Hit delta | Listwise Hit delta | BCE admissions | Listwise admissions | Listwise regressions |
|---|---:|---:|---:|---:|---:|
| Toys | +0.148438 | +0.066406 | 518 | 42 | 0 |
| Beauty | +0.115234 | +0.054688 | 557 | 34 | 0 |

listwise 纠偏后的增益约为 BCE 探索值的一半，准入数则下降约 92–94%。这符合
`REJECT_TO_BASE` 明确进入每用户 action set 后的保守性：当前方法不再大量替换 base
top10 尾部，但仍找到 62 个跨域 target promotions，且无 regression。

## 6. Execution and resource audit

- 正式 runner 在具名 tmux `gram_phase11_bw3_p1c_listwise_admission` 中后台运行；
- 启动前 Phase-11 CPU tests `23 passed`，其中 P1C 专用 tests `9 passed`；
- 26 个 code/input locks 通过，输出从空 `scientific/` 目录生成；
- workload 强制 CPU-only，峰值 RSS 约 610 MiB；
- GPU telemetry 18 行，CodeLlama tmux/controller 全程 running；
- CodeLlama 物理 GPU6 实测 used memory 恒为 `31,206 MiB`，高于 `30,720 MiB`；
- 未启动 sidecar，telemetry 中实验 GPU PID 观测数为 0；
- 终态 `resource_audit=passed`、`resource_status=preserved_running`；
- `validation_target_read=false`、`test_read=false`、`sports_read=false`。

## 7. Decision boundary

P1C 已严格通过，因此下一个最小科学步骤是使用当前两域冻结 gate 和 margin
执行一次 P2 `t=-2` validation。但 P2 会消耗一次性 validation，所以必须先与研究者
确认，再写独立 P2 plan。本轮不自动写入或启动 P2。

## 8. Artifacts

- preregistration：`plan/第十一阶段/GRAM_第十一阶段_BW3-P1C_Listwise准入纠偏计划.md`
- frozen config：`artifacts/phase11/configs/bw3_p1c_listwise_admission_preregistered.json`
- aggregate summary：`artifacts/phase11/bw3_p1c_listwise_admission/scientific/summary.json`
- domain summaries：`scientific/{Toys,Beauty}/summary.json`
- frozen gates：`scientific/{Toys,Beauty}/admission_gate.json`
- per-user records：`scientific/{Toys,Beauty}/calibration_per_user.tsv`
- execution status/log：`artifacts/phase11/bw3_p1c_listwise_admission/status.json`、`run.log`
- telemetry：`artifacts/phase11/bw3_p1c_listwise_admission/gpu_telemetry.csv`、`cpu_telemetry.csv`
