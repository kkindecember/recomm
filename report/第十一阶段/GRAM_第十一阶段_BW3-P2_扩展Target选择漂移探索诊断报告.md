# GRAM 第十一阶段 BW3-P2：扩展 Target 选择漂移探索诊断报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: `ANALYZED_EXPLORATORY`
- Version Label: `phase11_bw3_p2_selection_shift_diagnostic_v1`
- Diagnostic ID: `GRAM_PHASE11_BW3_P2_EXPLORATORY_SELECTION_SHIFT_DIAGNOSTIC_V1`
- Parent Result: `GRAM_PHASE11_BW3_P2_LISTWISE_ADMISSION_ONE_SHOT_VALIDATION_V1`
- Confirmatory P2 Status: `failed`（本诊断不修改该结论）
- Overall Confidence: `CAUTION`（机制证据跨两域一致，但属于 P2 揭示后的探索分析）

## 1. 诊断问题与边界

P2 中 Toys/Beauty 分别有 60/65 个真实 target 位于 `beam200 - beam50`，但 gate 准入的
21/15 个候选中没有一个真实 target。本诊断只回答：

1. target 是因 logit 低于 margin 0 被拒绝，还是过阈值后被其他候选挤出 top3；
2. calibration `t=-3` 到 validation `t=-2` 的 target logit 是否发生特异性漂移；
3. 哪些冻结特征贡献了主要漂移。

诊断原样复现 P1C gate、9 维特征、fit mean/std、margin 0 和最多 3 admissions。没有训练、
调参、改 margin、重新判定 P2 或读取 test/Sports。全部结论均为 post-hoc exploratory，
不得用于把已失败的 P2 改写为成功。

## 2. Target 选择从 calibration 到 validation 的变化

| Dataset | Split | Expansion target users | Target passes margin | Pass rate | Target selected top3 | Selection rate |
|---|---|---:|---:|---:|---:|---:|
| Toys | calibration `t=-3` | 92 | 34 | 36.96% | 34 | 36.96% |
| Toys | validation `t=-2` | 60 | 0 | 0.00% | 0 | 0.00% |
| Beauty | calibration `t=-3` | 81 | 28 | 34.57% | 28 | 34.57% |
| Beauty | validation `t=-2` | 65 | 0 | 0.00% | 0 | 0.00% |

两域 calibration 中，所有通过 margin 的真实 target 都进入 top3；validation 中所有真实
target 都低于 margin，没有“过阈值但被 top3 竞争淘汰”的案例。因此直接失败点是 **target
logit 整体跌到 0 以下**，不是 `max_admissions=3` 的容量限制。

## 3. Target logit 与候选排序漂移

| Dataset | Split | Target logit mean | Median | Maximum | Target rank median | Rank mean |
|---|---|---:|---:|---:|---:|---:|
| Toys | calibration | -0.627 | -0.821 | +3.373 | 1.0 | 2.77 |
| Toys | validation | -5.408 | -5.229 | -0.501 | 23.5 | 45.65 |
| Beauty | calibration | -0.995 | -0.844 | +3.113 | 1.0 | 3.35 |
| Beauty | validation | -5.981 | -5.576 | -3.190 | 46.0 | 56.05 |

Target mean logit 从 calibration 到 validation 的变化为：

- Toys：`-4.781`；
- Beauty：`-4.985`。

与此同时，非 target expansion candidate 的平均 logit 几乎稳定：

- Toys：`-6.896 → -6.930`，变化约 `-0.034`；
- Beauty：`-6.929 → -6.877`，变化约 `+0.052`。

这排除了“所有 candidate 分数一起平移”的简单 calibration drift。漂移集中发生在应被识别的
真实 target 上，即正类可分性消失。Target 的 expansion 内 median rank 也从两域的 1 降至
23.5/46，说明即便后验放宽 margin，排序质量本身也已明显恶化；不能把问题简化为阈值过严。

## 4. 特征贡献分解

### 4.1 Toys

| Feature | Calibration target contribution | Validation target contribution | Shift |
|---|---:|---:|---:|
| `cf_pop_adjusted` | +2.860 | +0.603 | **-2.257** |
| `item_anchor_z` | +2.705 | +0.657 | **-2.048** |
| `item_raw` | +0.639 | +0.255 | -0.384 |
| `popularity_anchor_z` | +0.192 | -0.015 | -0.207 |
| 其余 5 特征合计 | — | — | +0.116 |

### 4.2 Beauty

| Feature | Calibration target contribution | Validation target contribution | Shift |
|---|---:|---:|---:|
| `cf_pop_adjusted` | +2.600 | +0.340 | **-2.260** |
| `item_anchor_z` | +2.522 | +0.336 | **-2.186** |
| `item_raw` | +0.933 | +0.260 | -0.673 |
| `popularity_anchor_z` | +0.103 | +0.007 | -0.095 |
| 其余 5 特征合计 | — | — | +0.229 |

`item_anchor_z` 与 `cf_pop_adjusted = item_anchor_z - 0.5 * popularity_anchor_z` 结构上共享
同一个 item-head anchor 信号，gate 又在两维上都学到接近 1 的正权重。两维合计解释了 target
logit 下降的约 90.1%（Toys）和 89.2%（Beauty）。加上 `item_raw` 后，主要负漂移几乎全部
来自 item-head 路径；其他特征的小幅正变化只抵消了一部分。

原始标准化值也显示同一模式：

- Toys `item_anchor_z`：`3.299 → 0.801`，`cf_pop_adjusted`：`3.449 → 0.727`；
- Beauty `item_anchor_z`：`2.604 → 0.347`，`cf_pop_adjusted`：`2.640 → 0.345`。

Calibration 的正 target 是 item-head anchor 上的极端高分样本，而 validation target 只略高于
平均 expansion candidate。Gate 对这一偶然稳定性不足的正类模式形成了双重依赖。

## 5. 机制判定

证据支持以下分层结论：

1. **直接失败机制：margin rejection。** Validation 中 125 个 expansion target 全部低于 0；
2. **更深层机制：target-specific ranking collapse。** Median rank 从 1 降到 23.5/46，非 target
   分数总体稳定；
3. **主要来源：item-head 正类信号跨 offset 不稳定。** 两个共享 item anchor 的特征承担约
   89%–90% 的 logit 下滑；
4. **不是 beam coverage 失败。** expansion pool 明确包含 60/65 个 target；
5. **不是 top3 容量失败。** 没有 target 通过 margin 后再因竞争被拒；
6. **不是 sequence 信号整体漂移。** sequence feature 的 contribution shift 接近 0，但其权重
   也过小，无法在 item-head 信号消失时提供替代判别力。

因此，当前 P1C 收益更像是对 `t=-3` item-head-positive hotspot 的校准，而不是学到跨
pseudo-future offset 稳定的 expansion relevance。

## 6. 对可能修复方式的约束

以下是证据约束，不是新实验计划：

- 单纯把 margin 从 0 调低不能解决排序崩塌，并会同时放入更多非 target；
- 只增大 `max_admissions` 也无效，因为 target median rank 已远大于 3；
- 继续同时使用 `item_anchor_z` 和 `cf_pop_adjusted` 会保留重复依赖风险；
- 若继续 gate 路线，核心应是跨 offset 稳健性、正类 item-head shift 和特征冗余，而不是再次
  在已消耗 P2 上找阈值；
- 任何利用本诊断选择的新 schema/参数都属于方法开发，必须在新的未使用 holdout 上确认。

## 7. 统计解释与谬误检查

本诊断比较的 calibration/validation target 集不是同一批用户的配对样本，样本数也不同，
因此均值和比例差只作描述，不报告未经预注册的显著性检验。跨 Toys/Beauty 重复出现同方向、
近似量级的 item-head contribution collapse 提高了机制一致性，但不能消除 post-hoc 偏差。

11 类检查：Simpson、ecological、Berkson、collider、base-rate neglect、regression to mean、
survivorship、look-elsewhere、garden of forking paths、correlation/causation、reverse causality
均已检查（`11/11`）。主要 CAUTION 是 look-elsewhere/garden：特征贡献是在看到 P2 失败后分析，
只能作为新假设来源。报告完整保留 9 维贡献，没有只展示最符合解释的特征；不作因果声称。

## 8. 结论与下一步讨论点

当前 admission gate 的问题不是“过于保守所以少放几个”，而是它依赖的 item-head 正类信号
没有从 `t=-3` 泛化到 `t=-2`。我的建议优先级是：

1. 若继续 gate 路线，先研究去冗余、跨 offset 稳健的 scoring，而不是放宽 margin；
2. 若无法提供新的真正 holdout，则不要继续做确认性 gate 优化，可将该路线收束为负结果；
3. 另一条更直接的选择是停止后处理准入，回到候选生成或端到端 item-aware ranking。

下一步仍需与研究者讨论后再写计划。本诊断不授权训练、test/Sports 访问或任何后继实验。

## 9. 诊断产物

- `artifacts/phase11/bw3_p2_exploratory_selection_shift/summary.json`；
- 每域 `summary.json`；
- 每域 calibration/validation expansion target 明细 `target_candidates.tsv`；
- 每域 9 维 `feature_contribution_shift.tsv`；
- 诊断脚本与 4 项合成测试。
