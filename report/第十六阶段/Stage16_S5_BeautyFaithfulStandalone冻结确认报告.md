# Stage16 S5：Beauty Faithful Standalone 冻结确认报告

> 状态：`COMPLETED / PASS_S16_5_BEAUTY_SAUX_COLD_SIGNAL`
>
> 强基线裁决：Beauty 上 S-AUX 相对 F0 复现了方向一致且统计通过的 cold H@50 信号，但在 cold H@50、cold NDCG@10、warm NDCG@10 与 overall NDCG@10 上均低于正确的 `unconditional portfolio@2`；因此本结果不是强基线胜出，也不是全面 Pareto 改善。
>
> 后续边界：原 `S-AUX + G-RIDGE` 组合继续永久停止。结果只解锁“提交 `portfolio@2 default + conditional S-AUX + warm-risk abstention` 新计划修订”的资格，不授权直接开发、运行 validation 或打开 test。
>
> 数据边界：Beauty validation 仅在 Beauty 域内 train-only state 与 comparator 冻结后打开，未用于调参或 state 选择；`test_read=false`。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-01
- Verification Status: ANALYZED（正式 GPU 科学运行未重跑；已完成 artifact contract/SHA 核对、10,655 条 event 全量独立聚合、S16-5 定向 7/7 与 Stage16 全量 166/166 回归测试）
- Version Label: stage16_s5_beauty_saux_frozen_transfer_v0.1

## 1. 执行摘要与正式裁决

正式 attempt `s16_s5_beauty_saux_gpu0_a1` 于 2026-09-01 16:36:47–20:10:54 +08:00 在物理 GPU0 完成，exact command 为：

```text
bash experiment/phase16/run_stage16_s5_beauty_saux_gpu0_a1.sh
```

训练、state/comparator freeze、全量 Beauty validation、统计 finalization 与 artifact contract 共 5/5 pipeline steps 完成，worker exit 0，终态为：

```text
COMPLETED_S16_5_BEAUTY_SAUX_FROZEN_TRANSFER
PASS_S16_5_BEAUTY_SAUX_COLD_SIGNAL
```

主检验使用 5,287 个 cold events。S-AUX 的 cold H@50 从 F0 的 `0.013051` 提高到 `0.021373`，绝对增益 `+0.008322`（`+0.8322` 个百分点；相对 F0 `+63.77%`），event-level paired-bootstrap 95% CI=`[+0.004161,+0.012673]`。精确单侧 paired binary test 的 treatment-only/control-only hits=`86/42`，raw p 与单比较 Holm p 均为 `6.268e-05`，满足预注册 cold-signal Gate。按 target item 聚类的补充 bootstrap CI=`[+0.001516,+0.015751]`，方向也为正。

但是正确的 `unconditional portfolio@2` 在 Beauty 的 cold H@50 为 `0.032533`，比 S-AUX 的 `0.021373` 高 `0.011159`；其 cold NDCG@10、warm NDCG@10 与 overall NDCG@10 也全部更高。因此准确结论是：

1. S-AUX 相对冻结 GRAM F0 的冷命中信号跨 Toys→Beauty 同向复现；
2. 该信号在 Beauty 显著缩小，且没有击败正确的强基线 portfolio@2；
3. 不能把 `PASS_S16_5_BEAUTY_SAUX_COLD_SIGNAL` 改写成“S-AUX 在 Beauty 最优”或“双域全面 Pareto PASS”。

## 2. 冻结协议与防泄漏

### 2.1 方法与比较器冻结

| 项 | 冻结内容 |
|---|---|
| adaptation seed | `1502` |
| Beauty state construction | 只用 train-only interaction/internal-dev 与 pseudo-cold |
| S-AUX inference | draft size 50、threshold `-1.8`、strict `>` acceptance、guided redraft、adaptive exit、beam 50，均继承 Toys 前冻结规则 |
| F0 | 冻结 Beauty GRAM checkpoint |
| R2 | Stage13 `unconditional_portfolio2`；保护 F0 前 8 位，把两个 cold candidates 放入 ranks 9–10，再 stable unique 补齐至 50 |
| statistics | 10,000 次 paired bootstrap，95% CI，seed `20260822`；单侧 exact paired binary test；Holm family=`[S-AUX_vs_F0]` |
| item diagnostic | 10,000 次 target-item cluster bootstrap，seed `20260901` |

Toys 方法配置 SHA256=`a09eda03288eb2d04c38f925f471d061679827958b41063c7dbf98f76740168a`。Beauty state 和 comparator 在 validation 前共同冻结，freeze artifact SHA256=`fd9d436336efab92c7d16e01fb4e7843e8cc4ed6015ef25ab59db166c5c262af`。

### 2.2 数据使用边界

- train transitions=`33,775`；pseudo-cold events/items=`9,229/1,185`；real-cold catalog items=`6,052`；cold interaction label leaks=`0`。
- drafter 在 epoch 14 取得最佳 internal-dev pseudo-cold NDCG@10=`0.019589`，共完成 54 epochs/918 optimizer steps后 early stop。
- content embedding 训练前后 SHA 完全一致，未被训练修改。
- training open-file manifest 只记录 Beauty train-only/internal-dev 资源；validation manifest 明确 `validation_used_for_evaluation_only=true`、`validation_used_for_tuning_or_state_selection=false`。
- `test_read=false`、`test_opened=false`、原始 `user_sequence.txt` 未打开、network 未使用、automatic retry=false。

## 3. Beauty 全量结果

| Arm | Cold H@50 | Cold NDCG@10 | Warm H@50 | Warm NDCG@10 | Overall H@50 | Overall NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 0.013051 | 0.001953 | **0.254657** | **0.075361** | 0.134772 | 0.038936 |
| R2 / portfolio@2 | **0.032533** | **0.009231** | 0.252981 | 0.071397 | **0.143595** | **0.040550** |
| S-AUX | 0.021373 | 0.001147 | 0.214605 | 0.031308 | 0.118724 | 0.016342 |

事件覆盖为 overall `10,655/10,655`、cold `5,287/5,287`、warm `5,368/5,368`；cold/warm unique target items 分别为 `2,708/2,692`。没有 event attrition。

### 3.1 S-AUX 相对 F0

| 指标 | 增益 | Paired-bootstrap 95% CI | 裁决 |
|---|---:|---:|---|
| Cold H@50 | **+0.008322** | **[+0.004161, +0.012673]** | `PASS` |
| Cold NDCG@10 | -0.000806 | [-0.001954, +0.000272] | `INCONCLUSIVE` |
| Warm NDCG@10 | **-0.044053** | **[-0.049968, -0.038299]** | `NEGATIVE` |
| Overall NDCG@10 | **-0.022594** | **[-0.025651, -0.019538]** | `NEGATIVE` |

这说明增益只出现在 top-50 cold reachability；没有证据表明 cold top-10 排序改善，并且 warm/overall 排序质量明确受损。统计显著不等于实际全面优越：绝对 cold H@50 仍只有 `2.137%`。

### 3.2 S-AUX 相对正确 portfolio@2

| 指标 | 增益 | Paired-bootstrap 95% CI | 裁决 |
|---|---:|---:|---|
| Cold H@50 | **-0.011159** | **[-0.016455, -0.006053]** | `NEGATIVE` |
| Cold NDCG@10 | **-0.008083** | **[-0.009716, -0.006507]** | `NEGATIVE` |
| Warm NDCG@10 | **-0.040089** | **[-0.045962, -0.034319]** | `NEGATIVE` |
| Overall NDCG@10 | **-0.024208** | **[-0.027374, -0.021181]** | `NEGATIVE` |

S-AUX 在四个预先关注的质量维度上均低于 portfolio@2。F0/R2 timing 来自 Phase13 冻结 control，未在本次硬件上重测，因此本报告不制造 hardware-normalized latency speedup；但质量差距本身已足以否定“S-AUX 单独替换 R2”。

## 4. 双域迁移判断

| Domain | S-AUX−F0 Cold H@50 | S-AUX−R2 Cold H@50 | S-AUX−F0 Warm NDCG@10 | 双域含义 |
|---|---:|---:|---:|---|
| Toys | +0.049920 | +0.030456 | -0.039105 | 强 cold reachability，形成 S-AUX/R2 trade-off |
| Beauty | +0.008322 | -0.011159 | -0.044053 | F0 信号复现但显著衰减，R2 关系反转 |

Beauty 的 F0-relative cold 增益只有 Toys 的约 `16.7%`；warm cost 没有随增益同步收缩。由此得到两个彼此独立的结论：

- **跨域稳定部分**：S-AUX 相对 F0 的 cold H@50 方向为正，warm NDCG@10 方向为负；
- **不稳定部分**：S-AUX 相对强基线 portfolio@2 的优势未迁移，Toys 的正差在 Beauty 反转为负差。

因此不能把双域结果概括为“faithful SpecGR-Aux 普遍优于 R²”。更窄且被证据支持的表述是：faithful S-AUX 在冻结 GRAM 上提供可迁移但域敏感的 cold reachability 信号，同时带来稳定的 warm/overall 成本。

## 5. 补充互补性诊断（不属于 promotion Gate）

对 5,287 个 Beauty cold events 进行只读逐事件审计：

| Pair | S-AUX-only | Control-only | Both | Oracle-union H@50 |
|---|---:|---:|---:|---:|
| S-AUX vs F0 | 86 | 42 | 27 | 0.029317 |
| S-AUX vs R2 / portfolio@2 | 68 | 127 | 45 | **0.045394** |

S-AUX 虽然整体弱于 R2，仍提供 68 个 R2 未命中的 cold events；R2 则提供 127 个 S-AUX 未命中的 events。两者 oracle union 比单独 R2 的 `0.032533` 高 `0.012862`。这只证明存在 event-level 上限，不证明 train-only selector 能识别这 68 个事件，更不授权在 validation 上搜索路由 threshold。

该诊断支持“可以先提交条件式方法 plan amendment”，但不支持“条件式方法已经成立”。由于这个方向由已经观察到的 Toys/Beauty validation 结果启发，若后续开发，最终效力必须依赖冻结后的 test，而不能把同一 validation 上的提升当成独立确认。

## 6. 机制与资源

| 项 | 实测 |
|---|---:|
| accepted / drafted | 695,151 / 1,362,200（51.03%） |
| redraft rounds / all rounds | 5,934 / 27,244（21.78%） |
| zero-finite draft rounds | 0 |
| draft-capacity shortfall rounds | 0 |
| rankings different from F0 | 10,655 / 10,655 |
| training time | 163.73 s |
| inference time | 12,554.16 s（3.49 h） |
| wall time | 12,847 s（3:34:07） |
| training / validation peak CUDA reserved | 4,326 / 1,848 MiB |
| validation peak CPU RSS | 5,657.59 MiB |
| S-AUX extra state | 34,035,267 bytes（32.46 MiB） |
| attempt artifact size | 122 MiB |

GPU0 admission free=`19,135 MiB`，高于冻结 minimum free=`9,216 MiB`。运行未修改其他进程。日志只有 Hugging Face `resume_download` deprecation FutureWarning，不影响本地冻结模型加载或科学结果。

“10,655/10,655 rankings changed”只说明 S-AUX 干预覆盖完整，不代表每个事件改善；实际 warm/overall 结果说明广泛改写也带来了明显成本。

## 7. 统计谬误扫描与限制

覆盖：`11/11 checked`。

| 谬误 | 裁决 | 证据或限制 |
|---|---|---|
| Simpson's paradox | 未发现 | cold/warm/overall 分层同时报告；没有用 overall 掩盖 cold 正、warm 负的异质性 |
| Ecological fallacy | 未发现 | 主检验使用 event-level paired outcome，不从域聚合值推断单个用户必然受益 |
| Berkson's paradox | `CAUTION` | 结论仅适用于冻结 Beauty cold50 validation universe，不外推其他电商域或线上流量 |
| Collider bias | 未发现 | 没有按 Beauty outcome 选择 state、event 或 comparator |
| Base-rate neglect | 未发现 | 报告 5,287 cold events、独有/共同 hit 绝对计数与总体比例 |
| Regression to the mean | `CAUTION` | 只有 adaptation seed 1502 和一个冻结 GRAM backbone；不能声称多 seed/backbone 稳健性 |
| Survivorship bias | 未发现 | 10,655/10,655 events 完整评估，无失败事件被排除 |
| Look-elsewhere effect | primary 已控制 | primary family 只有预注册 S-AUX−F0，并做 exact test/Holm；S-AUX−R2 与 oracle union 明确为 trade-off/补充诊断 |
| Garden of forking paths | `CAUTION` | Beauty 方法与比较器在 validation 前冻结，但未来条件式方向由已观察 validation 结果启发，必须在新计划中披露为 post-validation hypothesis |
| Correlation→causation | 不适用/边界明确 | 这是冻结离线 arm 干预；不据此声称线上用户留存或商业指标会因 S-AUX 改善 |
| Reverse causation | 不适用 | outcome 未用于反向修改本次 method state |

Overall Confidence：`CAUTION`。S16-5 Gate 由全量 event coverage、paired CI、exact test、item-cluster diagnostic、冻结比较器与 SHA contract 支持；置信度受单 adaptation seed/单 backbone、Beauty validation 并非未触碰外部域、强基线失败和未来方向的 post-validation 动机限制。

## 8. 可复现性、完整性与 artifact

### 8.1 验证结果

- 正式运行内 S16-5 targeted tests：`7/7 PASS`；报告编写时重新执行：`7/7 PASS`。
- 报告编写时 Stage16 full CPU regression：`166/166 PASS`。
- `event_metrics.jsonl`、S-AUX predictions、portfolio@2 predictions 均为 `10,655` 行；独立逐事件聚合与 summary 中全部 cold/warm/overall H@50、NDCG@10 exact match。
- artifact contract 所列 8 个 required artifact 全部存在，所有记录 SHA 与现场重算一致。
- 科学 GPU 重跑：未执行；因此 reproducibility verdict=`CANNOT_VERIFY`（科学 rerun 未做），artifact/统计一致性=`VERIFIED`。

### 8.2 关键 SHA256

| Artifact | SHA256 |
|---|---|
| `summary.json` | `e3699554d2152e363f530760d581306f06afa4711ac28b60474883da4f17b766` |
| `status.json` | `0fbf365eaf10f882060b36229222a238c353c64a3af2e68f544aeb5fe4865eb2` |
| `artifact_contract.json` | `b62974646c1d9bb43049bdf3f0bb75baa1ac37e61b8b02fbc55bbc6a2be9720a` |
| `event_metrics.jsonl` | `70c0633bf61e00a056b8fda9514a65b7434129d0556d3261fb6ef4271501c843` |
| S-AUX predictions | `788318367853a7af61acde9f94eebc36002eec282bd4a1f42f176b281eadcd83` |
| portfolio@2 predictions | `2bd11371ae2e7a08026613c97c88b2d6eff80d14024b7a59f5ce57ed2bcb38c8` |
| Beauty S-AUX checkpoint | `34cbb5804524890d3785afe9f5e0a6f8093a6f99749fe9407cbf2e9aa6714682` |

### 8.3 主要路径

- 正式 attempt：`artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1/`
- 状态：`artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1/status.json`
- 权威 summary：`artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1/summary.json`
- state/comparator freeze：`artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1/state_and_comparator_freeze.json`
- artifact contract：`artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1/artifact_contract.json`

## 9. 后续唯一合法步骤

S16-5 已完成，但没有直接解锁任何 GPU 实验。下一步只能是先提交 Stage16 新 plan amendment，明确：

1. 默认 arm 为正确 `unconditional portfolio@2`，S-AUX 只能条件触发，warm-risk 时 abstain 回默认 arm；
2. selector 只使用预测时可得且 train-only 可构造的 draft confidence、verifier margin、prefix depth、redraft/acceptance 状态等特征，禁止使用 validation/test target、hit label 或 cold outcome 搜 threshold；
3. 方法开发与 stop rule 只在 train-only internal-dev 冻结；Toys/Beauty validation 已经可见，后续在这两份 validation 上的结果不能作为独立确认；
4. 必须预注册相对 R2 的 cold H@50 Gate、warm/overall non-inferiority 或明确 loss cap、增量推理成本与 state 上限；
5. amendment、代码、state、Gate 与一次性 test-open manifest 全部冻结后，仍需用户独立授权才能进入 S16-9 test。

如果 internal-dev selector 无法在冻结 Gate 下超过 R2，S16-8 应写 `NOT_UNLOCKED_STOP` 报告并停止阶段；不得在 validation 上联合搜索 threshold、seed、budget 或 ranking weight。

## 10. 最终状态

```text
S16-5 = COMPLETED
S16-5_GATE = PASS_S16_5_BEAUTY_SAUX_COLD_SIGNAL
BEAUTY_SAUX_VS_F0 = POSITIVE_COLD_H50 / NEGATIVE_WARM_AND_OVERALL_NDCG10
BEAUTY_SAUX_VS_PORTFOLIO2 = LOWER_ON_ALL_REPORTED_QUALITY_METRICS
TOYS_TO_BEAUTY_TRANSFER = DIRECTIONALLY_REPLICATED_BUT_STRONGLY_ATTENUATED
ORIGINAL_GRIDGE_COMPOSITION = STOP_PRESERVED
CONDITIONAL_PORTFOLIO2_PLUS_SAUX = PLAN_AMENDMENT_ELIGIBLE / NOT_AUTHORIZED
NEXT_STEP = WRITE_AND_APPROVE_NEW_PLAN_AMENDMENT_OR_STOP
TEST = SEALED
```
