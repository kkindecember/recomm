# Stage17 FP2 GRAM-LATTE-Full 正式结果报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：`validate → plan`
- Step：`S17-FP2`
- 日期：2026-09-02
- 状态：`COMPLETED / FP2_NOT_STRONG_PASS / FP2_NO_PROMOTION`
- Verification Status：`ANALYZED`
- Overall Confidence：负向主结论与 Gate 判定为 `SOLID`
- Canonical training：G0/G1/G2 均为 `attempt_004`
- Canonical analysis：`artifacts/phase17/fullport/external_d0/attempt_001/analysis.json`
- Analysis SHA256：`30933210b311253f525b6eb1610c8e9289145b3870a8bed3f4e708645948f620`
- 外部评估用户：12,833
- 数据边界：D0 只物化一次；D1/D2、official test、Sports 均未读取

## 1. 结论

G0 `GRAM-B0-Fresh`、G1 `GRAM-PSID-Full`、G2 `GRAM-LATTE-Full` 已完成 full-data fresh training、checkpoint 冻结和一次性 external D0 评估。结果不是弱正，而是方向明确的负结果：

- G2−G1：`ΔNDCG@10=-0.003872`，95% CI `[-0.005890,-0.001882]`；`ΔHit@10=-0.007948`。
- G2−G0：`ΔNDCG@10=-0.036657`，95% CI `[-0.040028,-0.033205]`；`ΔHit@10=-0.052833`。
- G1−G0：`ΔNDCG@10=-0.032785`，说明主要损失首先来自 lexical-ID → conflict-free PSID 的迁移；G2 的 latent forest 不但没有补回损失，还相对 matched G1 进一步下降。
- G2 的 latent usage、multi-path 与 aggregation 均真实激活，因而当前结果不能归因于“机制没跑起来”；更准确的解释是：在当前 GRAM 输入、训练协议与 identifier linking 边界下，激活的 LATTE 机制没有转化为 item-ranking 增益。

正式判定为 `FP2_NO_PROMOTION`。不进入 D1，不进入 FP4，不用 PCRF/融合救活，不在 D0 上继续调整 LATTE。

## 2. Canonical 合同与主结果

| 项目 | G0 `GRAM-B0-Fresh` | G1 `GRAM-PSID-Full` | G2 `GRAM-LATTE-Full` |
|---|---:|---:|---:|
| canonical training | `attempt_004` | `attempt_004` | `attempt_004` |
| seed / train examples | 2023 / 56,421 | 同左 | 同左 |
| internal-dev users | 1,283 | 1,283 | 1,283 |
| 完成 epochs / best epoch | 35 / 20 | 50 / 35 | 40 / 25 |
| best internal-dev NDCG@10 | 0.062516 | 0.030439 | 0.025351 |
| train microbatch / grad accum | 16 / 8 | 8 / 16 | 8 / 16 |
| effective batch | 128 | 128 | 128 |
| parameters | 60,503,040 | 60,900,352 | 60,900,352 |
| primary inference | beam500 identity | beam500 identity | beam500 `agg_max` |
| external Hit@10 | 0.097561 | 0.052677 | 0.044728 |
| external NDCG@10 | 0.061700 | 0.028915 | 0.025044 |
| external MRR@10 | 0.050685 | 0.021720 | 0.019026 |
| external Hit@50 | 0.180472 | 0.126237 | 0.116652 |
| external NDCG@50 | 0.079821 | 0.044800 | 0.040433 |

G1 与 G2 参数量、SID vocabulary、per-device batch、有效 batch、optimizer 分组和 evaluator 完全匹配，因此 G2−G1 是 LATTE latent/forest/aggregation 的直接 causal control。G0 是 fresh 强绝对基线。

## 3. 配对效应与强 Gate

| Comparison | ΔNDCG@10 | 95% CI | ΔHit@10 | 95% CI | Gain/Loss/Tie |
|---|---:|---|---:|---|---|
| G1−G0 | -0.032785 | [-0.036271,-0.029303] | -0.044884 | [-0.050027,-0.039739] | 415/1,061/11,357 |
| G2−G1 | -0.003872 | [-0.005890,-0.001882] | -0.007948 | [-0.011301,-0.004364] | 347/435/12,051 |
| G2−G0 | -0.036657 | [-0.040028,-0.033205] | -0.052833 | [-0.058287,-0.047612] | 364/1,090/11,379 |

每个比较均使用 12,833 paired users 与 2,000 次 bootstrap。G2−G1 target-rank changed rate 为 14.78%，G2−G0 为 22.16%。

| FP2 预注册条件 | 结果 | 状态 |
|---|---|---|
| G2−G1 `ΔNDCG@10 >= +0.0015` | -0.003872 | **FAIL** |
| G2−G1 CI lower `>0` | -0.005890 | **FAIL** |
| G2−G0 `ΔNDCG@10 >= +0.0015` | -0.036657 | **FAIL** |
| G2−G0 `ΔHit@10 >=0` | -0.052833 | **FAIL** |
| 无大子组 `<=-0.003` 灾难性退化 | 六个预注册大组全部低于 -0.003 | **FAIL** |
| latent usage / multi-path / aggregation | 全部激活 | PASS |
| tree coupling 相对 G1 降低 | correlation 0.003185 vs 0.021173 | PASS |
| legal/valid item 与完整性 | G2 path valid=1；用户严格对齐；受保护数据未读 | PASS |

G2 并非 `WEAK_POSITIVE_FULLPORT`：其两个 primary contrast 均为负，且 CI 不跨 0。

## 4. 子组稳健性

### 4.1 G2−G0

| 维度 | 子组 | users | ΔNDCG@10 | ΔHit@10 |
|---|---|---:|---:|---:|
| history | short ≤3 | 7,610 | -0.038391 | -0.054796 |
| history | medium 4–9 | 3,435 | -0.037744 | -0.051528 |
| history | long ≥10 | 1,788 | -0.027185 | -0.046980 |
| frequency | head | 4,166 | -0.012046 | -0.015122 |
| frequency | mid | 3,921 | -0.041810 | -0.056618 |
| frequency | tail | 4,746 | -0.054002 | -0.082807 |

所有大组方向一致为负；tail 损失最大。不存在“总体为负但某个主要人群稳定救回”的证据。

### 4.2 G2−G1

G2 相对 matched G1 在 short/medium/long history 的 NDCG@10 分别为 `-0.004603/-0.003044/-0.002350`，在 head/mid/tail 分别为 `-0.002818/-0.006547/-0.002587`。latent forest 的额外损失并非只来自单一历史长度或频率组。

## 5. 机制、ablation 与失败归因边界

| G2 beam500 `agg_max` 机制 | 结果 |
|---|---:|
| latent normalized entropy | 0.996721 |
| latent user collapse rate | 0 |
| multi-path item rate | 0.943627 |
| mean duplicate-path rate | 0.847886 |
| mean unique items | 76.06 / 500 paths |
| mean aggregation gain NDCG@10 | +0.012487 |
| target-path survival rate | 0.137146 |
| valid-path rate | 1.0 |
| tree-distance/score correlation | 0.003185 |

聚合显著改善 path-level ranking，但改善后的 item-level NDCG@10 仍只有 0.025044，低于 G1 的 0.028915 和 G0 的 0.061700。也就是说，aggregation 是必要且有效的内部步骤，却不足以形成外部推荐收益。

- `agg_sum`：G2 NDCG@10=0.024687；相对 G1 为 `-0.004229`，CI `[-0.006167,-0.002249]`，比 primary `agg_max` 更差。
- beam 50 → 500 把 G2 NDCG@10 从 0.022368 提高到 0.025044，但仍无法接近 G1/G0；不得继续扩大 beam 追点。
- G1 相对 G0 的大幅下降发生在没有 latent forest 的 PSID control 上，因此“identifier/linking/decoder boundary 不适配当前强 GRAM”是最符合证据的工作解释；它仍是推断，不是已被单独消融验证的唯一原因。

## 6. Attempt、受控恢复与完整性

- G0/G1/G2 `attempt_001`～`attempt_003` 是研究者要求的资源/吞吐调整后封存的 superseded attempts；只有 `attempt_004` 冻结 checkpoint 并可选。
- external `attempt_001` 中，G0/G2 在生成预测前因旧 PyTorch 不接受 `weights_only` 参数失败，G1 未越过 GPU admission；失败证据未覆盖。
- 研究者明确授权后，G0/G2 在 recovery `attempt_002`、G1 在独立 `attempt_003` 完成。三者复用同一 sealed D0 bundle，未重新打开 raw external projection。
- `single_materialization_count=1`、`automatic_retry=false`、`raw_external_projection_reopened=false`。
- prediction SHA256：G0 `f7b2d327...301ea`；G1 `95c3ced8...17d7`；G2 `2f7dcd00...a47d`。
- 五臂用户严格对齐，primary ranking 全部非空；G2 constrained paths 全合法；PSID collision alias=0。
- D1/D2、official test、Sports read flags 均为 false。GPU4/GPU1 的隔离资源守护 `result_selection_eligible=false`，不进入科学结果。
- 当前报告验证冻结产物及统计重算链，但没有独立重训，故 Verification Status 为 `ANALYZED`。

## 7. 统计解释与 11 类谬误扫描

- Coverage：`11/11 checked`

| 类型 | 严重度 | 判断 |
|---|---|---|
| Simpson's paradox | NOTE | overall 与六个大子组均为负，没有观察到方向反转。 |
| Ecological fallacy | NOTE | 报告 user-level gain/loss/tie，不声称每个用户都受损。 |
| Berkson's paradox | NOTE | external D0 cohort 在效果揭盲前冻结，未按结果筛选。 |
| Collider bias | NOTE | 未按 treatment 后变量筛选或调整。 |
| Base-rate neglect | NOTE | 已报告 head/mid/tail 用户数与效果；本任务不属于诊断测试。 |
| Regression to the mean | NOTE | checkpoint 由 internal dev 选择，external D0 为独立的一次性评估。 |
| Survivorship bias | NOTE | 12,833 用户全覆盖，未丢弃失败/空预测用户。 |
| Look-elsewhere effect | NOTE | primary comparison、beam 与 aggregation 预注册；ablation 不参与选模。 |
| Garden of forking paths | CAUTION | 存在工程恢复与早期 microbatch supersession，但科学配置、有效 batch、bundle 和 checkpoint 均冻结；不得继续效果驱动修改。 |
| Correlation != causation | NOTE | matched G2−G1 支持当前实现内的机制干预判断，不支持跨域普遍性结论。 |
| Reverse causality | NOTE | arm 与分析规则在揭盲前确定。 |

primary 三个 contrast 的 CI 均完全低于 0；负向点差也远离 `+0.0015` 门槛。即使对多个主比较采取更保守解释，也不会改变 no-promotion 决策。secondary metrics、subgroup 与 ablation 只用于一致性和归因，不用于寻找可晋级的显著结果。

## 8. 下一步决策

### 8.1 立即冻结

1. 关闭 standalone LATTE full-port 当前配置：不重训 N0/N1/G0/G1/G2，不换 seed，不扫 latent、beam、aggregation 或 loss。
2. D1/D2、official test、Sports 继续锁定；G2 不具备独立 fold 准入资格。
3. FP4 在 v0.1 计划下永久不解锁，因为其前提是 G2 与 S2 都通过强 Gate；后续即使 S2 成功，也只能作为 standalone winner 推进。
4. 不用 PCRF、SETRec 或其他模块掩盖 LATTE 失败；论文中应报告 native 弱正但不确定、GRAM full-port 明确负向的完整证据链。

### 8.2 下一科学主线：FP3 Full SETRec

FP3 是计划内的下一方向，但**现在还不能直接启动正式 efficacy**。仓库当前只有：

- 已冻结的 SETRec source manifest、resolved config 与 clean-room fidelity matrix；
- continuous AE、repo/paper attention、independent query、grounding 的合同级代码与单元测试；
- 旧 S17-2R discrete proxy；该实现明确不是 Full SETRec，不能复用为正式 arm。

尚缺完整的 FP3 training/inference backend、train-only SASRec CF tokenizer、semantic AE 训练产物、S0/S1R/S1P/S2 四臂统一 runner、full-catalog grounding evaluator、机制诊断、资源 profile 与正式 GPU 授权。因此下一步顺序固定为：

1. **FP3 preregistration/implementation freeze**：把 S0/S1R/S1P/S2 的参数、attention 差异、loss、checkpoint rule、D0 one-shot 评估和失败门写成机器可读 config；禁止依据本轮 LATTE D0 结果改变 SETRec 的已冻结默认值。
2. **CPU contract + bounded smoke**：完成 train-only CF/semantic tokenizer、五维连续 identifier、repo-parity/shared-position、paper sparse visibility、independent queries、full-catalog grounding及 GRAM-FiD 接口；验证 forbidden visibility=0、query/token recovery>0、valid item=1。
3. **完整 tokenizer 与逐臂资源 profile**：先训练并冻结 tokenizer，再对四臂做短 profile，确定显存、microbatch/accumulation、wall time 与磁盘；不得削减机制或训练预算换显存。
4. **另行申请 FP3 GPU**：理想四卡一臂一卡；资源不足时分波。获得明确授权后才启动正式训练。
5. **FP3 one-shot external D0**：四臂 checkpoint 全冻结后再一次性评估。只有 S2 通过计划第 7.3 节强 Gate，才允许 S2 standalone 进入 FP5/D1；FP4 仍关闭。

当前最优动作不是再做 LATTE 诊断，而是先完成 FP3 的实现与资源准入包。
