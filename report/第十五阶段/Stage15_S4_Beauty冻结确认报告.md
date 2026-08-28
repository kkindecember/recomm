# Stage15 S4：Beauty 冻结确认与阶段收尾报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-24
- Verification Status: ANALYZED（已核对 S15-4 B2 完整 artifact/status/summary/log/hash/资源与 paired-bootstrap，并核对 B3 attempt1–4 的状态、preflight 与 fail-fast 断点；未做独立重复运行）
- Version Label: stage15_s4_beauty_v1_final

## 最终结论

S15-4 已完成，Stage15 主线正式收尾。B2 SpecGR-GRAM attempt2 完成 Beauty validation `10,655/10,655` events，workload exit code=0；B3 GenRecEdit-GRAM exploratory branching recovery 在打开 Beauty validation event 前未能通过冻结的状态构建 Gate。所有运行均保持 `test_read=false`、`automatic_retry=false`，原始完整序列与 test predictions/metrics 未打开。

| Arm | Toys S15-3 | Beauty S15-4 | 双域裁决 |
|---|---|---|---|
| B1 R² portfolio@2 | `PASS_NATIVE_COLD_RECOVERY` | `PASS_NATIVE_COLD_RECOVERY` | 保留为冻结外部干预强基线 |
| B2 SpecGR-GRAM | recovery PASS，Pareto FAIL | recovery FAIL，Pareto FAIL | Toys 弱正信号未在 Beauty 复现，不是双域候选 |
| B3 原正式入口 | edit-state admission FAIL | 不迁移 | 原失败永久保留 |
| B3 exploratory branching recovery | native recovery FAIL | `FAIL_B3_BEAUTY_STATE_CONSTRUCTION_POSITION3` | 无 Beauty efficacy，不具备双域可构建性 |

B2 Beauty cold H@50 对 B0 的差值为 `-0.00018914`，95% paired-bootstrap CI=`[-0.00170229,+0.00113486]`，不满足预注册的 CI 下界 `>0` Gate，且不得将该结果写成“等价”。同时，B2 的 warm NDCG@10 对 B0 下降 `0.05111680`，CI 全负；overall NDCG@10 也明确下降。B2 在 Toys 的小幅 native recovery 因此不能外推为稳健的双域效果。

B3 Beauty attempt4 在 canonical target-token alignment 已 PASS 后，完成 6,052/6,052 train-only pseudo-cold contexts、16/16 probe batches 和 256/256 covariance transitions；但冻结的 4 requests/position、threshold=`0.3`、z steps=`30` 下，positions 0–2 的成功数为 `[1,1,2]/4`，position 3 为 `0/4`，因此依 Gate 在 `0/10,655` validation events 时 fail-fast。这是方法级可构建性失败，不是 OOM、token 对齐或运行器错误；不启动 attempt5，不换 seed/request/layer，不降 threshold，不增加 steps。

依 S15-5 预注册决策表，两个复杂 adaptation arm 都未形成双域 native recovery，且无任何 arm 达到 `PASS_OVER_R2_PARETO`。阶段最终决策为：

> `STOP_GRAM_COLD_ADAPTATION_METHOD_BRANCH`

本计划不进入 S15-5，不自动开发 `R²-guided contextual editing` 或第三个 adaptation 机制。未完成的 Video Games 官方 native sanity 继续保持“非阻塞支线”状态，不影响 Stage15 主表与主线收尾。

## Beauty 正式结果

### 统一指标

| Arm | Overall H@50 | Overall NDCG@10 | Warm H@50 | Warm NDCG@10 | Cold H@50 | Cold NDCG@10 | Cold hit events |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 GRAM v0 | 0.13477241 | 0.03893624 | 0.25465723 | 0.07536116 | 0.01305088 | 0.00195327 | 69/5,287 |
| B1 R² portfolio@2 | 0.14359456 | 0.04055018 | 0.25298063 | 0.07139706 | 0.03253263 | 0.00923071 | 172/5,287 |
| B2 SpecGR-GRAM | 0.12444862 | 0.01236412 | 0.23435171 | 0.02424437 | 0.01286174 | 0.00030186 | 68/5,287 |

事件构成固定为 10,655 total、5,368 warm、5,287 cold；三个 arm 在同一 projected validation target 上评估。B0/B1 来自冻结 Phase13 validation ranking replay，B2 在域内 drafter state 冻结后才打开 validation target。

B2 的 cold hit events 为 68，比 B0 少 1；其 cold H@50 point estimate 相对 B0 约下降 1.45%。B2 warm NDCG@10 仅保留 B0 的约 32.17%，overall NDCG@10 仅保留约 31.75%。B2 虽在 10,621/10,655 个 event 上改变 B0 排序，但大规模排序改变未转化为 cold reachability recovery。

### Paired bootstrap 主比较

10,000 次 event-level paired bootstrap，seed=`20260822`，95% percentile CI：

| 比较 | 指标 | 差值 | 95% CI | 解释 |
|---|---|---:|---:|---|
| B1−B0 | cold H@50 | +0.01948175 | `[+0.01569888,+0.02345375]` | recovery PASS |
| B1−B0 | warm NDCG@10 | −0.00396410 | `[−0.00489292,−0.00309359]` | 有明确 warm cost，但不改变其强基线身份 |
| B1−B0 | overall NDCG@10 | +0.00161394 | `[+0.00084084,+0.00239265]` | overall utility 为正 |
| B2−B0 | cold H@50 | −0.00018914 | `[−0.00170229,+0.00113486]` | inconclusive；recovery FAIL，不得写等价 |
| B2−B0 | cold NDCG@10 | −0.00165142 | `[−0.00259060,−0.00083534]` | cold top-10 placement 明确下降 |
| B2−B0 | warm NDCG@10 | −0.05111680 | `[−0.05670924,−0.04550801]` | warm cost 大且 CI 全负 |
| B2−B0 | overall NDCG@10 | −0.02657213 | `[−0.02941668,−0.02373403]` | overall utility 明确下降 |
| B2−B1 | cold H@50 | −0.01967089 | `[−0.02364290,−0.01569888]` | cold 明确低于 B1 |
| B2−B1 | warm NDCG@10 | −0.04715269 | `[−0.05286189,−0.04166770]` | warm 明确低于 B1 |
| B2−B1 | overall NDCG@10 | −0.02818606 | `[−0.03105598,−0.02528477]` | overall 明确低于 B1 |

B2 被 B1 在 cold/warm/overall 质量轴同时压过。由于 B0/B1 在本次为 replay，跨 arm latency 不具有严格可比性；`FAIL_COST_QUALITY_CANDIDATE` 只需由质量 non-domination 已不成立即可裁决，不需要伪造 latency Pareto 结论。

## B2 执行、状态与成本

- Artifact：`artifacts/phase15/s4_beauty/formal/b0_b1_b2_seed0_attempt2/`
- Status：`COMPLETED_S15_4_BEAUTY_B2_FULL_VALIDATION`
- Runtime：`37,227.83 s`，约 10.34 GPU-hours；其中 B2 full inference=`37,200.69 s`，users/s=`0.2864`
- Resource：peak CUDA allocated=`1,050.29 MiB`，peak CPU RSS=`5,750.79 MiB`
- Drafter：4,096 条 SHA-ranked train-only transitions，2 epochs，loss=`9.32257834→9.13962257`，finite/state changed
- State：1,346,912 trainable parameters，5,395,153 bytes，SHA256=`51f5adcbad3db343a08bcac8cf13dd396fbe834cc13276630f3ef2d437b93f78`
- Budget：draft size 10 × 5 rounds = 50，verifier threshold=`-1.6`，candidate chunk size=10，beam=50
- Forward accounting：532,750 verifier candidates，53,275 encoder histories，184,982 accepted drafts
- Base hash 前后均为 `6309435da398223d15c53664ccdd86163c71281e07713e2d46f4d24083884e59`
- Validation target 未用于 drafter training/state selection；原 `user_sequence.txt`、test predictions 和 test metrics 未打开

| Arm | Offline update | 本次 full inference | users/s | Extra state |
|---|---:|---:|---:|---:|
| B0 | replay | replay | 不可与 B2 严格比较 | 0 |
| B1 | 历史 train+validation 709.54 s | replay | 不可与 B2 严格比较 | 4,202,331 bytes |
| B2 | 12.64 s | 37,200.69 s | 0.2864 | 5,395,153 bytes |

B2 的离线更新很轻，但本次冻结预算下的推理成本高；更关键的是其 Beauty 质量未达 primary Gate。因此无论后续是否补齐 B0/B1 同硬件 latency，都不会改变本阶段的方法 promotion 裁决。

## B3 Beauty 试错与方法级失败

| Attempt | 结果 | 性质 | 处理 |
|---|---|---|---|
| attempt1 | 沙箱 tmux session 创建失败，workload 未启动 | 运行器/环境 | 保留 status，不作科学结论 |
| attempt2 | GPU6 free `<16,384 MiB`，rc=9，0/10,655 | 资源 admission blocked | 用户明确批准后仅将 B3 门槛改为 15,360 MiB |
| attempt3 | 完成 6,052 contexts 与 16/16 probe 后，“probe missed a lexical position” | split-token target alignment 工程错误 | 统一为 canonical catalog token IDs + EOS；55/55 tests 与 preflight attempt4 PASS |
| attempt4 | canonical mismatch=0；16/16 probe、256/256 covariance PASS；position 3 z-success=0/4 | 冻结条件下的方法级 state-construction FAIL | 在 0/10,655 fail-fast，禁止 attempt5 与事后改 Gate |

attempt3 的修复不改 encoder inputs、catalog、request sampler、seed、layer rule、probability threshold、z optimizer 或 efficacy Gate；它只消除 historical split collator 与 constrained generation canonical path 的 teacher-forced label 漂移。preflight attempt4 在 64 条 probe 上得到 historical mismatch=13、canonical mismatch=0，active positions 为 0–7，counts=`[64,64,64,64,64,64,64,16]`。

attempt4 的 selected layers=`[5,5,5,5,5,5,5,4]`。它在修复工程可比性后仍无法为 position 3 构造任何达到冻结 threshold 的 edit request，因此没有生成 B3 Beauty ranking、质量指标或 paired CI。报告只能写“Beauty state 不可构建”，不能写“B3 Beauty 质量差于 B0/B1”。

## 双域证据与研究问题回答

| 研究问题 | 证据 | Stage15 回答 |
|---|---|---|
| RQ1 官方方法能否形成可审计闭环 | 源码/artifact/license 已冻结，但 Video Games native GPU sanity 未执行 | 官方 native 快速复现仍为 non-blocking incomplete；不冒充已复现官方数字 |
| RQ2 能否不改机制语义适配 GRAM | B2 双域可运行；B3 Toys exploratory 可运行，Beauty 无法完成全 position state | SpecGR-GRAM adapter contract 可成立；GenRecEdit-GRAM 的可构建性具有明显域依赖 |
| RQ3 哪种干预位置有效 | B1 双域 cold recovery；B2 Toys 弱正、Beauty 失败；B3 Toys/Beauty 均未形成 native recovery | 当前证据只支持外部 R² portfolio 是稳定的 cold reachability 干预；retrieval-verification 和 parameter editing 未形成双域候选 |
| RQ4 是否存在值得开发的新方法缺口 | 无 arm 达 `PASS_OVER_R2_PARETO`，B2 有大 warm/overall cost，B3 还有 state-construction 瓶颈 | 本计划内没有满足条件式开发门槛的方法缺口；停止而不是继续调参 |

这一证据链支持的边界是：在当前冻结 GRAM backbone、hierarchical lexical ID、cold50 split、beam/candidate budget 和 seed-0 validation 下，复杂的 retrieval-verification 或 parameter editing 并未替代 R² portfolio 成为更好的跨域 cold adaptation 路径。该结论不外推到其他 backbone、其他 identifier、多 seed 或开放世界 cold-start。

## 统计完整性与 11 项 fallacy scan

Overall Confidence：`CAUTION`。B2 primary paired CI、冻结 Gate 与完整 event coverage 足以裁决 S15-4；但只有 seed-0 validation，secondary comparisons 未做 multiplicity correction，B3 没有 efficacy 数字，B0/B1 latency 也非本次同硬件重算。

Coverage：11/11 checked。

| Fallacy | 状态 | 核验结果 |
|---|---|---|
| Simpson's paradox | 未发现 | warm/cold/overall 分层完整报告；B2 各质量轴未隐藏方向反转 |
| Ecological fallacy | 未发现 | 指标与 bootstrap 均以 event/user 为单位，未用域均值推断个体效果 |
| Berkson's paradox | NOTE | 结论仅对冻结 catalog-known cold50 validation 成立，不外推到开放世界 cold-start |
| Collider bias | 未发现 | 未按模型输出或结果变量重新筛选 validation events |
| Base-rate neglect | 未发现 | total/warm/cold events、hit events 和 unique targets 均保留 |
| Regression to the mean | 不适用 | 不是按极端 baseline 表现选组的 pre/post 设计 |
| Survivorship bias | 未发现 | B2 覆盖 10,655/10,655 events；B3 在 validation 前按预注册 Gate fail-fast，未对打开后的 event 做选择性遗弃 |
| Look-elsewhere effect | CAUTION | 多个 secondary metric/arm CI 未校正；路线裁决只使用预注册 primary Gate |
| Garden of forking paths | CAUTION | B3 branching recovery 是 Toys 正式失败后的 exploratory 路径；原 FAIL、工程修复与 Beauty 方法 FAIL 均分开保留，不改写为 confirmatory |
| Correlation ≠ causation | 不适用 | 同一冻结 event 上的算法干预比较，不宣称真实用户因果效应 |
| Reverse causality | 不适用 | 不涉及横截面方向性因果主张 |

## Reproducibility

- Method：未重复运行完整 job；核对冻结 seed、numerical mode、input/code hash、base hash、status/summary/log 与 test guards
- Verdict：`CANNOT_VERIFY`（B2 运行完成且可审计，B3 fail-fast 断点可重建；但未以独立 rerun 达到 ARS 的 VERIFIED 定义）
- Numerical mode：TF32 off、deterministic algorithms on、`CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Safety：`original_user_sequence_opened=false`、`test_read=false`、`automatic_retry=false`，base hash unchanged

## Stage15 收尾状态

- S15-0：source/artifact/protocol freeze 完成
- S15-1：官方 native sanity 保持 non-blocking incomplete，不进入 GRAM 主表
- S15-2：双域输入、防泄漏、adapter 与 contract smoke 完成
- S15-3：Toys B2/B3 exploratory full validation 完成，无 arm 超过 R² Pareto
- S15-4：Beauty B2 full validation 完成，B3 在冻结 state-construction Gate 失败
- S15-5：条件未满足，未启动，也不在本计划授权范围内追加新方法
- Final：`STAGE15_COMPLETE / STOP_GRAM_COLD_ADAPTATION_METHOD_BRANCH / TEST_NOT_OPENED / PRIOR_ATTEMPTS_PRESERVED`

第十五阶段到此收尾，当前无未执行的 Stage15 主线实验。
