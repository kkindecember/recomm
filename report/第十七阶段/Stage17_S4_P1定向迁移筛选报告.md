# Stage 17 S4：P1 定向迁移筛选报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Step：`S17-4`
- Canonical 结果：`artifacts/phase17/s4_p1_targeted/run-0001`
- 数据：Toys D0 shadow fold；`test_read=false`；`sports_read=false`
- parent：`GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt`
- seed / budget：2023 / 每 arm 1 epoch
- CPU 契约：return code `0`
- P1 smoke：`PASS`

## 1. 结论

至少一个 treatment 的 paired mean NDCG@10 高于 matched control；仅按证据等级冻结候选，独立 S17-5 仍需单独执行。

Matched `GRAM-Continue` 的 Hit@10=`0.303436`、NDCG@10=`0.193899`。正向 treatment：`['pawa_lite']`。所有差值来自同一 D0 用户的 paired prediction；bootstrap 仅用于本折探索证据分级，不构成跨折论文级确认。

## 2. 正式结果

| Arm | 状态 | Hit@10 | NDCG@10 | ΔNDCG@10 | paired 判定 | GPU |
|---|---|---:|---:|---:|---|---|
| `gram_continue` | COMPLETED | 0.303436 | 0.193899 | — | matched control | GPU 1 |
| `pawa_lite` | COMPLETED | 0.303670 | 0.194388 | +0.000234 | WEAK_POSITIVE_CI_CROSSES_ZERO | GPU 0 |
| `latte_sethead` | COMPLETED | 0.276319 | 0.180228 | -0.027118 | NON_POSITIVE | GPU 2 |
| `biflow_s2g` | COMPLETED | 0.303281 | 0.193626 | -0.000156 | NON_POSITIVE | GPU 4 |

详细 paired 95% bootstrap 区间与 history-length / target-frequency 分组保存在 `summary.json -> paired_analysis`。

## 3. P1 迁移与烟测

九个计划内 P1 方向均已建立独立 migration card，并通过 tiny-GRAM 接口契约；GPU smoke 只运行 S3 诊断直接触发的三条候选。

| Smoke | 状态 | GPU | peak reserved MiB |
|---|---|---:|---:|
| `pawa_lite` | COMPLETED | GPU 0 | 21198.0 |
| `latte_sethead` | COMPLETED | GPU 2 | 21458.0 |
| `biflow_s2g` | COMPLETED | GPU 0 | 21388.0 |

未进入正式屏的 LS-FiD/MHM、GraphMAE/DCRec、SPRINT 等方向均在各自卡片记录 `not_triggered` 或完整机制尚未实现的边界，不能把接口烟测写成方法有效性证据。

## 4. 资源与边界

- GPU1 仅在三条 smoke 全通过、正式命令与快照冻结后，从 S17-3 非科学重复轮直接交接给 S17-4。
- 正式科学 arm 可使用当时满足显存准入的额外空闲卡；没有抢占或终止其他用户进程。
- 正式科学结束后，额外 GPU 全部释放；仅 GPU1 进入隔离的 run-NNNN 重复轮。
- 重复轮 `result_selection_eligible=false`、`affects_scientific_result=false`，不得进入本报告数值。

## 5. 下一门槛

`freeze positive candidate configuration for independent S17-5 consideration`。无论本折结果如何，official test 与 Sports 均继续封存。
