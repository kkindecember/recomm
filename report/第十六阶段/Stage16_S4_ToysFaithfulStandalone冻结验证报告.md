# Stage16 S4：Toys Faithful Standalone 冻结验证报告

> 状态：`COMPLETED / PASS_S16_4_TOYS_PORTFOLIO2_COMPARATOR_CORRECTION`
>
> 方向裁决：`STOP_GRIDGE_COMPOSITION / CONTINUE_SAUX_FROZEN_TRANSFER`
>
> 科学结论：S-AUX 相对 F0 有可重复的 Toys cold signal；用计划预注册的 `unconditional portfolio@2` 纠正比较器后，S-AUX 不再被 R2 支配，正式晋升为 `PASS_STANDALONE_PARETO`。G-RIDGE 仍为负增益，且对 F0、正确 R2、S-AUX 均无任何独有 cold hit，因此停止 `S-AUX + G-RIDGE` 组合主线；S-AUX 值得按冻结配置进入 Beauty transfer check。
>
> 数据边界：validation 用于本次冻结 efficacy 裁决；`test_read=false`，未使用 validation 调参或选择 state。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: analysis + CPU recovery execution
- Origin Date: 2026-09-01
- Verification Status: VERIFIED（a9 对 8,789 个事件逐用户重建 `portfolio@2`，candidate mismatch=0、F0 metric mismatch=0，与 Phase13 P6 summary 最大绝对误差 `5.55e-17`；独立重算所有派生章节 exact match，9 个本地 SHA、11 个来源 SHA exact，Stage16 159/159 tests PASS）
- Version Label: stage16_s4_toys_portfolio2_comparator_correction_v0.2

## 1. 执行摘要与裁决

GPU4 a7 的四个 formal arm 均已完成 `8,789/8,789` events，但旧 CPU finalizer 把 runtime manifest schema 硬编码为历史 GPU0 a3 名称，导致真实 GPU4 a7 manifest 在科学计算之前被误拒绝，终态为 `FAILED / ARTIFACT_CONTRACT_FAILED`。这是 finalization identity 工程错误，不是 GPU 推理失败。

修复后执行了独立 CPU-only a8 recovery：

```text
bash experiment/phase16/run_stage16_s4_toys_recovery_gpu4_a7_cpu_a8.sh
```

a8 不重跑 GPU 推理，不改写 a7，只读按 SHA 冻结的四臂 prediction/summary/config/runtime/status/log，重做 event metrics、10,000 次 paired bootstrap、精确单侧 McNemar 检验、Holm 校正、Pareto 与 artifact contract。a8 exit 0，终态 `COMPLETED`。

后续完整性审计发现：计划将 `R2` 明确定义为冻结 `R² portfolio@2`，但 S16-4 preflight 实际读取了 Stage13 P0 `r2_top50`。因此 a8 中不涉及 R2 的 primary comparisons 有效，而 R2 metrics、dominance、Pareto label 与 R2 complementarity 不具备预注册比较器身份。独立 CPU-only a9 纠正使用冻结 P0 `v0_top50/resolver_top50`、P6 `portfolio_candidates` 和 cold manifest，按 Phase13 规则逐用户重建正确 `unconditional portfolio@2`；a7/a8 均未修改，GPU 推理未重跑。

方法裁决：

- `G-RIDGE`：停止。对 F0 的 cold H@50 增益为 `-0.01008`，且对 F0、正确 portfolio@2、S-AUX 的 unique cold hits 全是 0，没有组合互补性信号。
- 原定 `S-AUX + G-RIDGE` 四臂组合：建议停止，不进入 S16-6/S16-7 的大实验。
- `S-PLUS`：仅保留为 faithful baseline 证据；它只相对一个明显退化的 matched control 有正增益，绝对质量和成本不支持主线化。
- `S-AUX`：正式标签修正为 `PASS_STANDALONE_PARETO`。它相对 portfolio@2 有更高 cold H@50，但 cold NDCG@10、warm/overall NDCG 和成本更差，形成真实 Pareto trade-off；下一步应先做冻结 Beauty transfer，不应直接在 Toys 开发 selector。

## 2. 运行边界与修复

### 2.1 a7 原失败保留

| 项 | 值 |
|---|---|
| source attempt | `s16_s4_toys_standalone_gpu4_a7` |
| 起止时间 | 2026-08-31 12:11:42–21:45:00 +08:00 |
| GPU | physical GPU4，单卡 |
| formal arm 进度 | 4/4 arms，每臂 8,789/8,789 events |
| 原终态 | `FAILED / ARTIFACT_CONTRACT_FAILED / exit 1` |
| 失败位置 | CPU finalizer runtime-schema identity check |
| test | sealed，`test_read=false` |
| automatic retry | false |
| repeat | 未启动，无重复实验占用 GPU4 |

a7 的 `status.json` 与 `run.log` SHA 在 recovery 前后保持为 `f51bfc1a…f2427` 和 `c353be8f…ef24`，未被覆盖。

### 2.2 修复内容

1. 将旧 finalizer 的 GPU0 a3 schema 字面量改为通用 formal runtime identity 验证：同时检查 config SHA、schema pattern、snapshot root、source repo、write scope、main-worktree visibility 和 manifest 所有 code file SHA。
2. 补上计划已预注册、但旧 finalizer 遗漏的 Holm correction。Cold-signal Gate 现在同时要求 paired-bootstrap CI 下界 `>0` 且 Holm-adjusted exact p `<0.05`。
3. 新增独立 CPU recovery finalizer/runner/config/tests；只接受冻结 a7 SHA，禁止已存在输出覆盖、GPU inference retry 或 automatic resume。

### 2.3 a9 portfolio@2 比较器纠正

| 项 | a8 错误身份 | a9 正确身份 |
|---|---|---|
| R2 来源 | Stage13 P0 `r2_top50` | Stage13 `unconditional_portfolio2` |
| P0/P6 输入 | 未用于 R2 重建 | P0 F0/resolver + P6 candidates，全部 SHA 冻结 |
| 排名规则 | route-and-resolve fusion | 保护 F0 前8位，将两个 cold candidates 放入 ranks 9–10，stable unique 补齐 |
| GPU inference | 无重跑 | 无重跑 |
| test | sealed | sealed |
| 原 artifact | a8 保留 | 独立 write-once a9 |

a9 对 8,789 个用户的 target、cold/warm、F0 event metric、P6 candidate identity 全量核验；candidate mismatch=`0`、F0 metric mismatch=`0`。重建 portfolio@2 后，cold/warm/overall H@50 与 NDCG@10 对 Phase13 P6 summary 的最大绝对误差为 `5.55e-17`。a8 的旧 R2 也被证实逐项等于 Phase13 P0 R2，故错误被明确定位为 comparator identity mismatch，而不是数值计算漂移。

## 3. 冻结数据与统计方法

| 项 | 值 |
|---|---:|
| validation events | 8,789 |
| cold events | 4,367 |
| warm events | 4,422 |
| candidate/ranking width | 50 |
| adaptation seed | 1502 |
| paired bootstrap | 10,000 resamples，95% CI，seed 20260822 |
| primary family | S-AUX−F0、S-PLUS-CTRL−F0、S-PLUS−matched CTRL、G-RIDGE−F0 |
| multiplicity | exact one-sided paired binary test + Holm correction，alpha=0.05 |

Arm 和对照保持预注册语义：S-AUX 与 G-RIDGE 对 F0，S-PLUS 对 matched S-PLUS-CTRL，R2 作为强基线/Pareto 对照。未换 seed，未根据 validation 修改 threshold、request、candidate budget 或 method state。

## 4. 全臂结果

| Arm | Cold H@50 | Cold NDCG@10 | Warm NDCG@10 | Overall NDCG@10 | Standalone label |
|---|---:|---:|---:|---:|---|
| F0 | 0.010305 | 0.002762 | 0.063580 | 0.033361 | reference |
| R2 / portfolio@2 | 0.029769 | **0.008724** | **0.060982** | **0.035016** | frozen Pareto comparator |
| S-AUX | 0.060224 | **0.006101** | 0.024475 | 0.015345 | `PASS_STANDALONE_PARETO` |
| S-PLUS-CTRL | 0.001374 | 0.000000 | 0.030421 | 0.015306 | `FAIL_STANDALONE` |
| S-PLUS | 0.011907 | 0.001557 | 0.027911 | 0.014816 | `PASS_STANDALONE_COLD_SIGNAL` |
| G-RIDGE | 0.000229 | 0.000000 | 0.017681 | 0.008896 | `FAIL_STANDALONE` |

S-AUX 的 cold H@50 显著优于 F0，但 warm NDCG@10 比 F0 低 `0.039105`，overall NDCG@10 低 `0.018016`。相对正确 portfolio@2，S-AUX cold H@50 高 `0.030456`，95% CI=`[+0.022670,+0.038241]`；但 cold NDCG@10 低 `0.002623`、warm NDCG@10 低 `0.036507`、overall NDCG@10 低 `0.019671`，三者 CI 均全负。因此两者互不严格支配：S-AUX 买到更高 top-50 cold reachability，portfolio@2 保留更好的 top-10、warm、overall 与成本。

### 4.1 Primary Gate、CI 与 Holm

| Treatment vs control | Cold H@50 增益 | Paired-bootstrap 95% CI | raw p | Holm p | 支配者 | 裁决 |
|---|---:|---:|---:|---:|---|---|
| S-AUX vs F0 | +0.049920 | [+0.042592, +0.057477] | 2.303e-43 | 9.211e-43 | 无 | cold signal PASS，Pareto PASS |
| S-PLUS vs S-PLUS-CTRL | +0.010534 | [+0.007557, +0.013739] | 1.421e-14 | 4.263e-14 | R2 | cold signal PASS，Pareto FAIL |
| S-PLUS-CTRL vs F0 | -0.008931 | [-0.011907, -0.006183] | 1.000 | 1.000 | F0, R2 | FAIL |
| G-RIDGE vs F0 | -0.010076 | [-0.013281, -0.007328] | 1.000 | 1.000 | F0, R2 | FAIL |

S-PLUS 的正对照效应不能解读成“优于 GRAM/R2”：它的 matched control 本身已从 F0 cold H@50=`0.010305` 退化到 `0.001374`，S-PLUS 只恢复到 `0.011907`，同时 warm/overall 质量仍明显低于 F0。

## 5. 互补性诊断

下表为事后的 frozen-event 诊断，未参与 S16-4 promotion Gate：

| Pair（cold events=4,367） | Treatment-only hit | Control-only hit | Both | Oracle-union H@50 | 含义 |
|---|---:|---:|---:|---:|---|
| G-RIDGE vs F0 | **0** | 44 | 1 | 0.010305 | G-RIDGE 无新 hit，oracle 等于 F0 |
| G-RIDGE vs R2 / portfolio@2 | **0** | 129 | 1 | 0.029769 | G-RIDGE 无新 hit，oracle 等于 portfolio@2 |
| G-RIDGE vs S-AUX | **0** | 262 | 1 | 0.060224 | G-RIDGE 无新 hit，oracle 等于 S-AUX |
| S-AUX vs F0 | 250 | 32 | 13 | 0.067552 | 存在真实 F0 外增量 |
| S-AUX vs R2 / portfolio@2 | 218 | 85 | 45 | **0.079689** | 双方均有独有 hit，存在真实 trade-off/互补 ceiling |
| S-PLUS vs R2 / portfolio@2 | 38 | 116 | 14 | 0.038470 | 互补信号有限，且成本极高 |

这是停止 G-RIDGE 组合方向的关键证据：即使完美 oracle 知道每个 event 该用哪个 arm，加入 G-RIDGE 也无法在 F0、正确 portfolio@2 或 S-AUX 之上多命中一个 cold event。因此没有证据支持为 G-RIDGE 预付 Beauty state reconstruction 或 S16-6/7 四臂成本。

S-AUX vs portfolio@2 则不同：S-AUX-only=`218`、portfolio-only=`85`、both=`45`，oracle union H@50=`0.079689`。这说明二者存在值得跨域确认的互补 ceiling，但仍不证明 train-only selector 能实现该 ceiling；按照预注册顺序，应先完成 S-AUX Beauty frozen transfer，再决定是否修订为 portfolio@2 + S-AUX 条件式方法。

## 6. 机制与成本

| Arm | Update | Inference | Extra state | 机制摘要 |
|---|---:|---:|---:|---|
| R2 | 118 s | 118 s | 4.20 MB | 冻结 portfolio@2 replay |
| S-AUX | 135 s | 8,758 s | 33.77 MB | accepted 558,627 / drafted 1,290,128（43.30%）；redraft 8,225；zero-finite 0 |
| S-PLUS-CTRL | 103,474 s（28.74 h） | 6,330 s | 242.13 MB | matched continued-training control |
| S-PLUS | 416,330 s（115.65 h） | 8,493 s | 242.27 MB | accepted 305,633 / drafted 908,686（33.63%）；redraft 34,908；zero-finite 2,549 |
| G-RIDGE | 58,062 s（16.13 h） | 6,319 s | 33.56 MB | 302,400 train-only requests、6 位置 FP64 solve；8,789/8,789 排序改变 |

这些 runtime 来自来源 artifact，不是所有 arm 在同一硬件/同一次 run 中重测，因此不声称严格 hardware-normalized speedup。但 G-RIDGE 在质量与互补性上均为负证据，不依赖精确 latency Pareto 也可停止；S-PLUS 超过 115 GPU-hour 级 update 也与其有限恢复不匹配。

## 7. 与 Stage15 pilot 的分名对照

| 方法 | Cold H@50 | Warm NDCG@10 | Overall NDCG@10 | 定位 |
|---|---:|---:|---:|---|
| Stage15 `P-SPECGR-LIGHT`/B2 | 0.013510 | 0.033839 | 0.018441 | lightweight pilot，非 faithful |
| Stage16 S-AUX | 0.060224 | 0.024475 | 0.015345 | faithful SpecGR-Aux transplant |
| Stage15 `P-GENRECEDIT-BUDGET`/B3 exploratory | 0.009618 | 0.062121 | 0.032720 | budgeted exploratory，非 faithful |
| Stage16 G-RIDGE | 0.000229 | 0.017681 | 0.008896 | full-target inspired ridge，明确非 faithful |

S-AUX 相对 Stage15 light pilot 明显提高 cold reachability，但付出更大 warm/overall 成本；相对正确 portfolio@2，它在 cold H@50 更高、在 top-10 与 warm/overall 更低，是 Pareto 权衡而非全面胜出。G-RIDGE 增大到全 target/全 request 后没有改善，反而比 Stage15 budgeted exploratory 更差；这对“继续增大 edit 规模可能自然修复效果”构成负证据。

## 8. 谬误扫描与限制

| 项目 | 裁决 | 证据/限制 |
|---|---|---|
| Simpson's paradox | 未发现 | cold/warm/overall 分层报告；S-AUX cold 正、warm/overall 负的 trade-off 未被聚合值隐藏 |
| Ecological fallacy | 未发现 | 主检验和互补诊断均使用 event-level paired outcomes |
| Berkson's paradox | CAUTION | 结论只针对冻结 Toys cold50 validation universe，不自动外推其他域 |
| Collider bias | 未发现 | 未按 validation 中间结果选 event/state |
| Base-rate neglect | 未发现 | 报告 4,367 cold events 及 hit 绝对计数 |
| Regression to the mean | CAUTION | 只有 adaptation seed 1502，不声称多 seed/backbone 稳健性 |
| Survivorship bias | 未发现 | 四臂均 8,789/8,789，无 event attrition；a7 FAIL、a8 recovery 与 a9 comparator correction 同时保留 |
| Look-elsewhere effect | 已处理 primary family | 四个 primary comparisons 做 Holm；其他互补 pair 明确为 diagnostic-only |
| Garden of forking paths | CAUTION | S16-4 有多个工程 attempt，但只有 a7 生成 GPU predictions；a8/a9 均为冻结 SHA 的 CPU 裁决。a9 由 plan/config comparator identity mismatch 触发，不依 efficacy 选择比较器，并保留错误 a8 |
| Correlation→causation | 不适用 | 报告是预注册 arm 对照，不做真实世界用户因果外推 |
| Reverse causation | 不适用 | frozen offline intervention 不存在 outcome 反向改变 method state |

Overall Confidence：`CAUTION`。对 Toys/seed-1502 的纠正裁决由全 event coverage、paired CI、Holm、Phase13 metric reconstruction 和 exact a9 recomputation 支持；但 comparator identity 错误说明 preflight 审计曾失效，Beauty 尚未执行，且当前结果不能证明 S-AUX-only hits 可被 train-only selector 预测。

## 9. 是否继续：正式建议

### 9.1 不建议继续的部分

1. 不运行原计划的 G-RIDGE Beauty full efficacy，除非目标只是用一次冻结迁移试验完成预注册证据链，而非继续押注方法。
2. 不运行 `S-AUX + G-RIDGE` 的 S16-6/S16-7 四臂组合；G-RIDGE 的 oracle 都无法带来一个新 cold hit。
3. 不为 S-PLUS 增加 adaptation seeds 或更大训练搜索；其正信号依赖退化 matched control，并且不具备 R2 竞争力。

### 9.2 建议继续的下一 Gate

S-AUX 已达到 `PASS_STANDALONE_PARETO`，因此建议执行一次严格冻结的 S16-5 Beauty transfer check：

- 沿用 Toys 前已经冻结的 S-AUX method、threshold、draft size、candidate budget、seed 与 evaluator；
- 只允许重建 Beauty domain-local drafter/index，不因 Beauty 结果回调 Toys；
- 同时使用 Beauty 正确的 `unconditional portfolio@2` comparator，先做 comparator identity/SHA preflight；
- 完整报告 cold H@50、cold NDCG@10、warm/overall、成本和 paired CI；
- 若 Beauty 不复现 cold signal，停止 S-AUX；若复现，再讨论条件式方法。

### 9.3 后续可选 pivot

只有 S-AUX Beauty 方向一致时，才值得把后续方法修订为 `portfolio@2 default + S-AUX conditional route + warm-risk abstention`。selector 只能使用 validation/test 前可得的 draft confidence、verifier margin、prefix depth 等 train-only 特征，并先通过 held internal-dev Gate；不得直接用 Toys/Beauty validation 搜 threshold、seed 或 budget。

该 pivot 需先修订 Stage16 plan 和对照/Gate，再由用户授权执行；不应在现有 S16-6 名义下静默替换 G-RIDGE。

## 10. 可复现性与 artifacts

### 10.1 验证

- a8 recovery targeted tests：`24/24 PASS`；a9 correction targeted tests：`29/29 PASS`；
- Stage16 full CPU regression：`159/159 PASS`；
- a9 CPU-only scientific recomputation：全部派生 summary sections exact match；
- a9 local artifact hashes：`9/9 exact match`；
- a9 frozen source hashes：`11/11 exact match`；
- portfolio candidate mismatches=`0`、F0 event metric mismatches=`0`、Phase13 aggregate reconstruction max error=`5.55e-17`；
- GPU scientific inference rerun：false；
- test read：false；
- automatic retry/resume：false。

### 10.2 主要路径

- a7 失败保留：`artifacts/phase16/s4_toys_standalone/formal/toys_seed1502_gpu4_a7/`
- a8 recovery 保留：`artifacts/phase16/s4_toys_standalone/recovery/toys_seed1502_gpu4_a7_recovery_a8_cpu/`
- a9 最终纠正裁决：`artifacts/phase16/s4_toys_standalone/correction/toys_seed1502_portfolio2_a9_cpu/`
- a9 summary SHA256：`f608316c62fb61b89585a9ac82653057bd7b7fa04f737492802590322e062716`
- corrected event metrics SHA256：`52a1d1fe8720326d36debbc21e6701b060430a6c1f9429d0c1177a595f6ee3cc`
- portfolio@2 predictions SHA256：`36fc228f383d86d067ddf5a56e1443d55a669ee4f8bbed33afd8ad6ed1a5c792`
- artifact contract SHA256：`0ab51796ec313a77679d440af25e28a267066f045e65818f739d524789020de7`
- correction status SHA256：`ba65cbaf058be5ce1b9ae7d6d39d48c9a26c0d7765aad4a493b98a4815721996`

### 10.3 原始输入锁定

a7 config/runtime manifest SHA256 分别为 `a09eda03…68a` 和 `95e6ec9c…868a`。四臂 prediction SHA256 分别为 S-AUX `33b99e42…9433`、S-PLUS-CTRL `b8173b72…b154`、S-PLUS `b41c8d8d…116cb`、G-RIDGE `8b28015e…a2bf`。完整值见 a8 `artifact_contract.json`。

正确 R2 的冻结来源为 Phase13 P0 predictions `c8f29872…ed51`、P6 predictions `77923d5b…a000`、P6 summary `ef9b8115…64e73`、portfolio reconstruction code `7b3949a8…5c409` 与 cold manifest `0cf83e3d…24e`。完整值见 a9 `artifact_contract.json`。

## 11. 最终状态

```text
S16-4 = COMPLETED
S16-4_COMPARATOR_CORRECTION = PASS
R2_IDENTITY = STAGE13_UNCONDITIONAL_PORTFOLIO2
S-AUX = PASS_STANDALONE_PARETO
S-PLUS = PASS_STANDALONE_COLD_SIGNAL / FAIL_STANDALONE_PARETO
S-PLUS-CTRL = FAIL_STANDALONE
G-RIDGE = FAIL_STANDALONE
CURRENT_GRIDGE_COMPOSITION_DIRECTION = STOP
NEXT_RECOMMENDED_GATE = S-AUX_FROZEN_BEAUTY_TRANSFER_PENDING_USER_GPU_AUTHORIZATION
OPTIONAL_LATER_PIVOT = PORTFOLIO2_DEFAULT_PLUS_SAUX_CONDITIONAL_ROUTE_PENDING_BEAUTY_AND_PLAN_REVISION
TEST = SEALED
```
