# GRAM 第九阶段：CF0-B4 P9-2D 流行度校准可靠性融合验证计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-04
- Verification Status: `EXECUTED_DEVELOPMENT_GATE_PASSED`
- Experiment ID: `GRAM_PHASE9_CF0_B4_TOYS_RELIABILITY_FUSION_P2D_V1`
- Parent: P9-2C overall BeamFusion positive but tail safety failed
- Scope: Toys validation、5-fold cross-fitted mechanism development、CPU-only
- Excluded: test、Beauty、Sports、重新解码、训练或改写 checkpoint

## 1. Experiment Overview

- **Title**：Popularity-Calibrated Reliability-Aware BeamFusion（PCRF）
- **Objective**：保留 P9-2C 的 overall late-fusion 增益，同时消除固定 lambda 对 tail target 的伤害。
- **Hypothesis H1**：只使用训练频次与当前 50 beams 的可观测结构，对 item-head score 去热门偏置并
  在 tail-heavy candidate set 上收缩 CF 权重，可使 cross-fitted overall Hit@10 提升至少 0.002，
  且 tail Hit@10 不退化。
- **Type**：analysis / mechanism-development。

P9-2C 的 15,316-user holdout aggregate 已经被观察，因此本轮不能被描述为新的独立确认实验。
它使用 out-of-fold prediction 降低逐用户调参泄漏，但仍属于同一 validation 上的开发性证据；
真正的确认需要后续冻结机制后读取 test、换数据域或新建独立 checkpoint/cache，并另行授权。

## 2. Variables 与机制

### 2.1 固定输入

- GRAM epoch-30 cached 50 legal beams 与 sequence scores；
- P9-2A epoch-10 frozen item-head；
- Toys train-prefix item frequency；
- 与 P9-2C 完全相同的 history、catalog、lexical-ID mapping。

### 2.2 PCRF 公式

对用户 `u`、candidate `i`：

```text
pop_z(u,i) = zscore(log(1 + train_frequency(i))) within the 50 beams
cf_pc(u,i) = zscore(cf_z(u,i) - beta * pop_z(u,i))
tail_mass(u) = fraction of original GRAM top-10 candidates with frequency <= q1
reliability(u) = (1 - tail_mass(u)) ** gamma
joint(u,i) = seq_z(u,i) + lambda * reliability(u) * cf_pc(u,i)
```

`q1` 只由 train-prefix frequency 冻结；`tail_mass`、candidate frequency、sequence score、CF score
在推理时均可获得。公式不使用 gold、target frequency、命中与否或 validation label。beta 修正
item-head 的热门偏置，gamma 在 GRAM top-10 呈现 tail-heavy intent 时收缩整条 CF 分支。

### 2.3 冻结网格

- lambda：`[0.5, 0.75, 1.0]`；
- beta：`[0.0, 0.25, 0.5, 1.0, 2.0]`；
- gamma：`[0.0, 1.0, 2.0, 4.0]`；
- 共 60 个候选；另有 `lambda=0` baseline，不参与复杂模型偏好。

不加入神经 gate、target-derived feature、手工 user group label、事后连续优化或网格扩展。

## 3. Five-fold cross-fitting

对 user id 计算 `SHA256("P9-2D:2023:" + user_id)`，排序后 round-robin 分配 fold 0…4。
每个 fold：

1. 其余四折只用于选择一组 `(lambda,beta,gamma)`；
2. 当前折不参与选择，只产生一次 out-of-fold ranks；
3. 五折 OOF ranks 合并后计算主结果。

训练折参数必须先满足：

- overall `ΔHit@10 >= +0.002`；
- overall `ΔNDCG@10 >= 0`；
- tail `ΔHit@10 >= 0`。

可行集合中按 overall Hit@10、tail Hit@10、overall NDCG@10 降序选择；仍并列时按 lambda、beta、
gamma 升序选最简单参数。若某折无可行参数，该折回退 lambda=0，并将
`all_folds_calibration_feasible=false`；不扩大网格。

## 4. Metrics 与科学门

Primary：pooled OOF Hit@10；safety：pooled OOF tail Hit@10。

同时满足才标记 development gate `passed`：

1. 五折均存在 calibration-feasible 非零参数；
2. OOF `ΔHit@10 >= +0.002`；
3. 2,000 次 paired bootstrap 的 OOF `ΔHit@10` 95% CI 下界 > 0；
4. OOF `ΔNDCG@10 >= 0`；
5. OOF tail `ΔHit@10 >= 0`；
6. OOF Hit@50 与 baseline 严格相同，容差 `1e-12`。

辅助指标：Hit/NDCG@1/5/20/50、MRR@50、tail bootstrap CI、middle/head、history length；
对照包括 P9-2C fixed lambda=0.75 和 pure baseline。辅助比较不作新的显著性声明。

## 5. Integrity 与停止规则

- 19,412 users、11,924 items、50 unique legal candidates/user；
- full baseline 与历史 cache 指标 `1e-12` identity；
- frozen input/code SHA256 在正式运行前锁定；
- 不读取 test/Beauty/Sports，不触碰 checkpoint；
- 单测、compile、全量 CPU smoke 通过后才正式运行；
- hard timeout 1,800 秒；非零退出、完整性失败或科学门失败均不自动重试或调参；
- process-alive、status、run.log 为监控面；实验预计小于 2 分钟，无 90 秒输出增长才提示 stall。

## 6. Setup 与 Expected Outputs

- Working directory：`/mnt/18T/jiangtangyunzhi/projects/recomm`；
- Entry：`bash experiment/phase9/run_phase9_cf0_b4_reliability_p2d.sh start`；
- Status：`bash experiment/phase9/run_phase9_cf0_b4_reliability_p2d.sh status`；
- Evaluator：`experiment/phase9/eval_cf0_b4_reliability.py`；
- Tests：`experiment/phase9/test_cf0_b4_reliability.py`；
- Config：`artifacts/phase9/configs/cf0_b4_toys_reliability_p2d_preregistered.json`；
- Output：`artifacts/phase9/cf0_b4_toys_reliability_p2d/`；
- 必需产物：`summary.json`、`fold_assignments.tsv`、`per_user_oof.tsv`、`status.json`、`run.log`。

## 7. Interpretation boundaries

- 通过：支持 PCRF 在同一 validation 内产生 cross-fitted、target-free 的 overall/tail 联合改善，
  只准入后续独立确认；
- overall 正但 tail 负：流行度特征不足以识别可靠性，机制失败；
- tail 安全但 overall 增益消失：gate 过度收缩，不能算成功；
- folds 选择参数不稳定：报告异质性，不挑最好 fold；
- 本轮任何结果均不得称作 test improvement、跨域泛化或独立复现。

## 8. 实际终态（2026-08-04）

- engineering：completed；4/4 tests；CPU wall `12.38 s`；
- 五折均选择 `(lambda=1.0, beta=0.5, gamma=1.0)`；
- OOF Hit@10：`0.119411 → 0.125335`，delta `+0.005924`；
- paired bootstrap 95% CI：`[+0.003864,+0.007985]`；
- OOF NDCG@10 delta：`+0.002441`；
- tail Hit@10：`0.091860 → 0.093023`，delta `+0.001163`，但 tail CI
  `[-0.001744,+0.004264]` 仍跨 0；
- Hit@50 delta：`0`；Hit@1 delta：`-0.000309`；
- development gate：`passed`。

该终态只准入后续独立确认，不升级为 test 或跨域证据。详细结果见
`report/第九阶段/GRAM_第九阶段_CF0-B4_P9-2D流行度校准可靠性融合结果报告.md`。
