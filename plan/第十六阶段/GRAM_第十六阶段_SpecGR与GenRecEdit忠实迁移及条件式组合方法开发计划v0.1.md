# GRAM 第十六阶段：SpecGR 忠实迁移、GenRecEdit-inspired→GRAM、互补性验证及条件式组合方法开发计划 v0.1

> **建立日期**：2026-08-23
> **当前状态**：`S16_0_COMPLETED / S16_1_COMPLETED / S16_2_COMPLETED_SAUX_SPLUS_CTRL_PASS / S16_3F_STRUCTURAL_BLOCKED_PRESERVED / S16_3R_FORMAL_F3_PASS / S16_4_GPU4_A7_FAILED_PRESERVED / S16_4_CPU_RECOVERY_A8_PRESERVED / S16_4_PORTFOLIO2_CORRECTION_A9_COMPLETED / S16_4_SAUX_PARETO / S16_4_GRIDGE_FAILED_NO_COMPLEMENTARITY / STOP_ORIGINAL_GRIDGE_COMPOSITION / NEXT_SAUX_FROZEN_BEAUTY_PENDING_USER_GPU_AUTHORIZATION / TEST_SEALED`
> **阶段定位**：在 GRAM backbone 与 hierarchical lexical ID 上忠实重实现 SpecGR；保留 faithful GenRecEdit 不可行性证据，并以明确非 faithful 的 G-RIDGE 完成 GenRecEdit-inspired→GRAM；分别验证后，再用可归因四臂实验决定是否开发条件式组合方法
> **历史边界**：Stage15 B2/B3 永久保留为 `lightweight/budgeted mechanism pilot`，不覆盖、不改名为 faithful reproduction，不用于否定官方方法

---

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-23
- Verification Status: VERIFIED（S16-0–S16-3 既有证据保持不变；S16-4 GPU4 a7 四臂均完成 8,789/8,789 events，但旧 finalizer 的 GPU0 schema 硬编码导致 `ARTIFACT_CONTRACT_FAILED`，原失败与 SHA 保留。CPU-only a8 恢复了冻结 predictions 的统计裁决；后续审计发现其 `R2` 错读 Stage13 P0 `r2_top50`，而计划预注册的是 `unconditional portfolio@2`。CPU-only a9 对 8,789 个事件逐项重建正确比较器，candidate/F0 mismatch 均为 0、Phase13 P6 aggregate 最大误差 `5.55e-17`，targeted 29/29、Stage16 full 159/159 tests、9 个本地 SHA 与 11 个来源 SHA 全 PASS。S-AUX 纠正为 `PASS_STANDALONE_PARETO`；S-PLUS 仍仅通过 cold-signal Gate且被正确 R2 支配；G-RIDGE FAIL 且相对 F0/正确 R2/S-AUX unique cold hits 均为 0。停止原 G-RIDGE 组合方向，下一 Gate 为 S-AUX 冻结 Beauty transfer；S16-5–S16-9 尚未执行，test sealed）
- Version Label: phase16_faithful_transplant_and_composition_v0.1

---

## 0. 执行摘要

Stage16 回答的不是“论文原始 TIGER 结果能否在本地逐位复现”，而是：

> **在保留 GRAM backbone、冻结 lexical ID 和项目 cold50 协议的前提下，faithful SpecGR 与明确标注的 GenRecEdit-inspired G-RIDGE 能否单独改善 cold recommendation；若能，两者是否存在可重复的互补交互，并足以支撑一个新的条件式方法？**

总顺序固定为：

1. 官方源码→算法组件→GRAM 接口的 fidelity mapping；
2. 冻结 train-only internal-development 协议，继续封存 test；
3. 忠实迁移 SpecGR-Aux 和 SpecGR++，严格分名；
4. 保留 faithful GenRecEdit→GRAM 的失败证据，以相同全量语义和 train-only workload 开发只替换 singular solve 的 GenRecEdit-inspired G-RIDGE；
5. 单方法 contract→admission→Toys validation→Beauty validation；
6. 固定 `2×2` 四臂实验，分离 SpecGR、GenRecEdit 和交互项；
7. 只在存在稳定互补证据时开发条件式方法；
8. 所有设计与 hash 冻结后，经用户再次明确授权才能一次性打开 test。

Stage16 不停止、不修改正在运行的 Stage15 Beauty B2。Stage15 完成后先形成其唯一 S15-4 report；Stage16 的 CPU 审计可并行准备，Stage16 GPU job 不与未完成的 Stage15 大实验争抢资源。

---

## 1. Stage15 交接与科学口径更正

### 1.1 可复用证据

- Stage15 已完成 SpecGR-style draft→verify→redraft 和 GenRecEdit-style edit→trigger 的 GRAM 基础接口、strict item evaluator、variable-length path、candidate budget、test guard 与 artifact schema。
- Toys/Beauty 的 `user_sequence_train_validation.txt`、GRAM checkpoint、lexical path catalog、metadata、cold/warm manifest 与 B0/B1 validation artifact 已审计并可复用；所有 SHA256 必须在 S16-1 重新核对。
- Stage15 已证明 test 封存、canonical token alignment、lexical constrained beam 和 frozen-base hash guard 可执行。

### 1.2 不可沿用为 faithful 结论的部分

| Stage15 对象 | 实际语义 | Stage16 处理 |
|---|---|---|
| B2 | 自写 BGE projection + 2-layer Transformer；4,096 transitions × 2 epochs | 只作 `P-SPECGR-LIGHT`，不作 faithful SpecGR baseline |
| B3 | 4 requests/position；256 train transitions covariance；改写的 z-success Gate | 只作 `P-GENRECEDIT-BUDGET`，不作 faithful GenRecEdit baseline |
| Beauty B3 position-3 FAIL | budgeted port 在该 contract 下无法构建完整 state | 不推广为官方 GenRecEdit 在 Beauty/GRAM 上失败 |

Stage16 不删除、不覆盖 Stage15 任何 artifact/report/plan 历史。新结果不回填 Stage15 主表。

---

## 2. 研究问题与可证伪假设

### RQ1：能否对“忠实迁移”给出可审计定义？

- H1a：SpecGR 官方的 drafting、target-aware verification、guided re-drafting、adaptive exit 可在 GRAM variable-length lexical path 上保留算法语义。
- H1b：GenRecEdit 官方的 full-target z optimization、covariance、valid-z filtering、closed-form delta 和 trigger 可迁移到 GRAM，且每个必要语义变化均可单独标记。

### RQ2：忠实 SpecGR→GRAM 能否改善 cold recommendation？

- H2a：官方 UniSRec drafter + frozen GRAM verifier 相对 GRAM B0 提高 cold H@50。
- H2b：GRAM encoder self-drafting 的 SpecGR++ 两阶段训练相对同训练预算 GRAM control 提高 cold H@50。

### RQ3：忠实 GenRecEdit→GRAM 能否改善 cold recommendation？

- H3：使用全部冻结 cold edit targets 和官方优化/汇总流程后，GenRecEdit-GRAM 相对对应未编辑 GRAM 提高 cold H@50，且 warm 损失可显式定位。

### RQ4：SpecGR 与 GenRecEdit 是否互补？

- H4a：SpecGR 主要提高 cold candidate reachability，GenRecEdit 主要改变 GRAM 对 cold lexical token/path 的验证倾向。
- H4b：组合臂相对两个单方法臂均有正增益，且 event-level interaction 的 paired CI 不跨 0。

### RQ5：组合是工程串联，还是足以导出新方法？

- H5：只有当互补出现在可重复的 prefix depth、candidate confidence、editability 或 warm-risk 子群时，才开发一个事先可定义的 conditional gate。
- 若组合仅是简单加和或只增加计算量，则不冒充方法创新，收束为 faithful transplantation + interaction analysis。

---

## 3. 忠实度合约

### 3.1 官方源码冻结

| 方法 | 官方仓库 | Stage16 冻结 commit | 用途 |
|---|---|---|---|
| SpecGR | `https://github.com/Jamesding000/SpecGR` | `f0ded8884b1df97b5f0599d4ec300bb20b5d1eff` | 算法/默认参数/训练与推理路径映射 |
| GenRecEdit | `https://github.com/Starrylay/GenRecEdit` | `e6878d9c7c6e57479e840ccb8c045b11a2bd69b5` | edit request、z optimizer、covariance、deltaW、trigger 映射 |

两个仓库当前未发现明确 license：只做本地研究审计和独立 adapter 实现，不将第三方源码、checkpoint 或 LFS blob 复制进项目提交。

### 3.2 变更分类

| 级别 | 定义 | 主表资格 |
|---|---|---|
| F0 | 官方算法逻辑/目标/决策规则原样保留 | faithful 必需 |
| F1 | 只因 TIGER/RQ-VAE SID → GRAM/lexical path、device、batching、API 造成的必要接口改动 | 可进 faithful 主表，必须逐项披露 |
| F2 | 样本缩减、步数缩减、替换模型/损失、改 Gate、只选少量 requests | 只能标 `scaled/budgeted diagnostic` |
| F3 | 根据 validation/test 结果改 request、threshold、子集或方法定义 | 禁止；对应 run 不进正式表 |

S16-0 必须生成 function-level `fidelity_matrix.json`，对每个官方组件标记 F0/F1/F2/F3、证据行、GRAM 映射、测试和是否进主表。

### 3.3 SpecGR 必须保留的核心

#### `S-AUX`：SpecGR-Aux → GRAM

- 使用官方 UniSRec 架构与训练目标，不用 Stage15 自写轻量 drafter 替代；
- 使用全部合法 train-only transitions，不固定缩减为 4,096 条；
- cold item 只通过 catalog content representation 进入 retrieval，不作 interaction label；
- 保留 inductive drafting、target-aware verification、guided re-drafting、adaptive exit 和官方 acceptance 规则；
- TIGER verifier 替换为冻结 GRAM 是 F1 改动；参数 hash 前后必须不变。

#### `S-PLUS`：SpecGR++ → GRAM

- 使用 GRAM encoder 作 self-drafter；
- 忠实迁移官方两阶段 contrastive/generative training 和 projection/index 构建；
- 因 GRAM 参数被训练，必须同时训练 `S-PLUS-CTRL`：相同起点、数据、step、batch、optimizer 和 GPU 预算，不使用 SpecGR-specific contrastive/self-drafting 目标；
- `S-PLUS` 的因果对照是 `S-PLUS-CTRL`，不是未继续训练的 B0。

### 3.4 GenRecEdit-inspired G-RIDGE 必须保留的核心

`G-RIDGE` 必须：

- 对全部预冻结 cold catalog edit targets 构造 train-only contexts，不先看 validation/test occurrence；
- 对全部 edit targets 分批执行 z optimization，不再每位置只选 4 条；
- 保留官方 Adam + cosine scheduler、cache-hit probe、active/satisfied lifecycle、norm clipping、valid-z filtering、key extraction 和按 layer 汇总 deltaW；
- covariance primary 使用官方 `mom2_n_samples` 口径：`min(400000, 全部合法 train-only covariance rows)`；更小规模只能作收敛/资源诊断，不替代 primary；
- 官方 `0.3` 仅按官方源码所属 probe/cache 语义映射，不再像 Stage15 一样额外作为所有 z 的全局成功 Gate；
- 不额外加入“edited probability 必须严格高于 baseline”，除非官方路径的对应位置本就有该规则；
- fixed 256-codebook 与 full-vocabulary probability 在 GRAM constrained lexical decoding 中的替代定义，必须在 S16-0 通过 fixed-width emulation 和推理概率一致性测试冻结，禁止依 efficacy 选择定义。
- 唯一允许偏离 faithful `G-FULL` 的位置是 closed-form solve：用预注册的 condition-targeted spectral ridge 形成 `A+mu·I`；`mu` 只依 train-only system spectrum 与固定 condition target，不使用 validation/test 或 outcome 搜索。
- 必须显式记录 `faithful_reproduction=false`，并保留 ridge 前后 spectrum/rank/condition、Cholesky、solve residual；禁止 pinv、额外 jitter fallback、outcome resampling 和自动 retry。

### 3.5 组合主路的固定选择

主四臂组合使用 `S-AUX + G-RIDGE`，不依单方法结果事后在 `S-AUX/S-PLUS` 中挑更好者，理由是：

- `S-AUX` 保持 GRAM base 冻结，与 parameter editing 的干预位置清晰正交；
- `S-PLUS` 已改变 GRAM backbone，若再组合 editing，需要新的 trained-base factorial control，不在主四臂中静默替换。

若 `S-AUX` contract FAIL 而 `S-PLUS` PASS，不自动用 `S-PLUS` 填补组合臂；必须先更新 plan 和对照设计，再获得用户授权。

---

## 4. 数据、防泄漏与评测协议

### 4.1 冻结输入

- GRAM checkpoint、lexical path、metadata、cold/warm manifest、strict evaluator 继承 Stage15 的冻结 SHA，S16-1 重算并生成 Stage16 manifest。
- Stage16 常规 job 只读 Stage15 已审计的 `user_sequence_train_validation.txt`；禁止重新打开原 `user_sequence.txt`。
- `test_read=true`、打开 test predictions/metrics，或从 test target 构造监督，任一情况立即 `KILLED_TARGET_LEAKAGE`。
- cold universe 只来自阶段开始前已冻结 membership；禁止按 validation/test 中的 target occurrence 重新选择编辑或检索对象。

### 4.2 internal-development 切分

Stage16 已知道 Stage15 的部分 Toys/Beauty validation 结果，因此不再把它们称为完全未触碰的开发集。调参必须使用 train-only internal-development：

1. 从每域 `projected_items[:-1]` 的 train 部分构造确定性 interaction train/internal-dev；
2. 使用 train-derived item-disjoint pseudo-cold 评估 inductive cold behavior；
3. split 使用固定 seed/SHA rank，在任何 GPU 训练前生成 manifest 和 target-isolation 测试；
4. 原 validation target 在 model/state/hyperparameter 冻结前不得打开；
5. Toys validation 定位为 primary source-domain validation；Beauty validation 为 frozen transfer check，但二者都不声称独立外部终验。

### 4.3 评测冻结

| 项 | 口径 |
|---|---|
| catalog semantics | catalog-known、metadata-available、zero-interaction cold-start simulation |
| ranking | strict unique item ranking；unknown/ambiguous/collision/duplicate hard-fail |
| budget | primary candidate/beam budget=50；官方方法若有额外 forward 完整记录 |
| primary benefit | cold H@50 |
| primary ranking quality | cold NDCG@10 |
| primary cost | warm NDCG@10 |
| secondary | overall NDCG@10、H@10、latency、state size、update cost |
| uncertainty | event-level paired bootstrap 10,000 resamples, 95% CI |
| multiplicity | 同一 Gate 家族的 primary comparisons 使用 Holm correction；原始 CI 与校正结论同时报告 |
| test | S16-9 前全程封存；一次性打开需用户再次明确授权 |

### 4.4 seed 规则

- contract/smoke/admission 使用 seed `1502`。
- 首次完整 standalone/combination validation 使用 seed `1502`；只有达到预注册 promotion Gate 的方法状态才扩展到 `1502/1503/1504`。
- 这些是 adaptation/training seeds，不得冒充独立 GRAM backbone seeds；论文中必须披露。
- 换 seed 不是工程 retry；未达 promotion Gate 不允许通过多 seed 搜索正结果。

---

## 5. Arm 定义

### 5.1 Standalone arms

| ID | 方法 | 主对照 | 资格 |
|---|---|---|---|
| `F0` | 冻结 GRAM v0，standard beam=50 | — | reference |
| `R2` | 冻结 R² portfolio@2 | F0 | 已有强基线；不重训 |
| `P-SPECGR-LIGHT` | Stage15 B2 | 只作诊断 | 不进 faithful 主结论 |
| `P-GENRECEDIT-BUDGET` | Stage15 B3 | 只作诊断 | 不进 faithful 主结论 |
| `S-AUX` | 官方 UniSRec + frozen GRAM SpecGR loop | F0/R2 | faithful primary SpecGR transplant |
| `S-PLUS-CTRL` | 同预算 continued-training GRAM control | F0 | S-PLUS 因果对照 |
| `S-PLUS` | GRAM encoder self-drafting + SpecGR++ two-stage objective | S-PLUS-CTRL | faithful secondary SpecGR transplant |
| `G-FULL` | faithful full-target GenRecEdit → GRAM | F0/R2 | 历史不可行性证据；不进入后续 efficacy |
| `G-RIDGE` | full-target GenRecEdit-inspired ridge edit → GRAM | F0/R2 | S16-3 新 primary；明确非 faithful |

### 5.2 组合四臂

| Arm | SpecGR-Aux drafting | GenRecEdit editing | 语义 |
|---|---:|---:|---|
| `C00` | 否 | 否 | 未编辑 frozen GRAM |
| `C10` | 是 | 否 | `S-AUX` |
| `C01` | 否 | 是 | `G-RIDGE` |
| `C11` | 是 | 是 | 同一 frozen GRAM 上 drafter + edited verifier/generation |

`C11` 必须重新核对编辑参数 materialization、One-One trigger、candidate verification、guided redrafting、fallback beam 与 base restore；禁止简单把两份 ranking 后处理融合却声称算法组合。

---

## 6. 分阶段执行计划与唯一报告

### S16-0：官方算法与 fidelity contract 冻结（CPU）

**任务**：

- 完成 SpecGR-Aux、SpecGR++、GenRecEdit 的 function-level 映射；
- 记录官方默认参数、实际源码分支、README/实现不一致处；
- 对 GenRecEdit `0.3`、argmax、cache-hit、valid-z 和 delta aggregation 给出精确语义；
- 建立 fixed-width synthetic SID 与 variable lexical trie 的 bridge tests；
- 冻结 F0/F1/F2/F3 矩阵、主实现路径、调参空间和资源待测项。

**Gate**：`PASS_S16_0_FIDELITY_CONTRACT`。任一核心组件无法映射时标 `BLOCKED_<METHOD>_SEMANTICS`，不用自创简化填补。

**唯一 report**：`report/第十六阶段/Stage16_S0_官方算法忠实度与GRAM映射冻结报告.md`

**实测记录（2026-08-23）**：

- exact command：`bash experiment/phase16/run_stage16_s0_fidelity_contract.sh`；
- attempt：`s16_s0_a1`，CPU-only，exit code `0`，无 GPU、无网络、无下载、`test_read=false`、`automatic_retry=false`；
- SpecGR/GenRecEdit 固定 commit 与 clean worktree 均通过；23 个 function-level F0/F1 映射、18/18 bridge checks、8/8 unit tests 通过；
- Gate：`PASS_S16_0_FIDELITY_CONTRACT`；完整证据见该步骤唯一 report 与 `artifacts/phase16/s0_fidelity_contract/`；
- 下一唯一步骤更新为 S16-1；尚未启动任何 Stage16 GPU 作业。

### S16-1：数据、internal-dev、test guard 与资源 preflight（CPU/小 GPU）

**任务**：

- 核对 Stage15 可复用输入 SHA，生成 Stage16 input allowlist/denylist；
- 构造 train-only interaction/internal-dev 和 item-disjoint pseudo-cold；
- 对 S-AUX/S-PLUS/G-FULL/G-RIDGE 统计完整训练条数、edit targets、contexts、path positions 和 covariance rows；G-RIDGE 与 G-FULL 的非 solve workload 必须相同；
- 为每个大实验先做可控小样本的显存/速度试跑，换算 GPU 数、每卡最小空闲显存、wall time、hard timeout 和磁盘；
- 冻结后续 exact commands 和 status schema，但不启动大实验。

**Gate**：`PASS_S16_1_DATA_LEAKAGE_RESOURCE_PREFLIGHT`；任何 test/path/collision 问题在 GPU 前 hard-fail。

**唯一 report**：`report/第十六阶段/Stage16_S1_数据防泄漏InternalDev与资源预检报告.md`

**实测记录（2026-08-23）**：

- exact command：`bash experiment/phase16/run_stage16_s1_data_resource_preflight.sh`；attempt `s16_s1_a1`，exit code `0`，12/12 Stage16 unit tests 通过；
- Toys/Beauty 分别冻结 27,659/33,775 条 train transitions、3,108/3,747 条 internal-dev transitions；pseudo-cold 分别为 1,162/1,185 个；student-readable real-cold/pseudo-cold 泄漏与 train/dev user overlap 均为 0；
- G-FULL/G-RIDGE 的完整计数分别为 5,963/6,052 targets、59,630/60,520 contexts、302,400/425,890 prefix-next-token requests、27,659/33,775 covariance rows；
- 小型资源探针自动选择物理 GPU 7（admission free 15,609 MiB，utilization 23%），18.54 秒完成，最高进程内 allocated 峰值 1,457.58 MiB；无正式训练、完整 editing 或 validation；
- S-AUX 探针因当前缺 RecBole 明确标为 resource proxy；S-PLUS/G-FULL/G-RIDGE 使用真实冻结 Beauty GRAM checkpoint，但仅作 bounded resource smoke；
- Gate：`PASS_S16_1_DATA_LEAKAGE_RESOURCE_PREFLIGHT`；完整证据见该步骤唯一 report 与 `artifacts/phase16/s1_data_resource_preflight/`；
- 第一个大实验 S-AUX 冻结为单 GPU、每卡至少 24,576 MiB 空闲、20,480 MiB 保守显存预留、约 18–48 h、48 h hard timeout、8 GiB 磁盘；启动前由用户指定 GPU。

### S16-2：Faithful SpecGR→GRAM 实现、contract 与 admission

**任务**：

- 实现 `S-AUX`、`S-PLUS`、`S-PLUS-CTRL`；
- 在 synthetic/fixed-width 与真实 lexical path 上验证 draft score、prefix restriction、masked verifier score、acceptance、redraft round、adaptive exit、dedup/fallback；
- 使用 train-only internal-dev 训练/选模，在查看 source-domain validation 前冻结方法 state 和 hyperparameters；
- 先进行小 contract smoke，再进行固定规模 item-disjoint admission；admission 只判路径、finite、完整性和资源，不判 efficacy。

**Gate**：

- `PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION`；
- `PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION`；
- `S-PLUS-CTRL` 训练预算对齐 PASS。

某一 SpecGR 变体 FAIL 不自动取消另一变体，但不得静默替换主四臂的 `S-AUX`。

**唯一 report**：`report/第十六阶段/Stage16_S2_FaithfulSpecGR_GRAM合约与Admission报告.md`

**中间实测记录（2026-08-23–24；S16-2 尚未完成）**：

- 固定 RecBole v1.2.0 / commit `362d31f00af801d7d99bc635c902d1df1405e79d`；固定 SpecGR/RecBole worktree clean，官方 UniSRec 与官方 TransformerEncoder source identity、forward/backward 通过；
- 完成 S-AUX、S-PLUS、S-PLUS-CTRL clean-room GRAM adapters 与全字段预算合约；加入 formal transition/isolation 测试后 26/26 Stage16 tests 通过；
- exact smoke command：`bash experiment/phase16/run_stage16_s2_specgr_contract_smoke.sh`；attempt `s16_s2_contract_a1`，物理 GPU 5，23.74 秒，峰值 3,712.55 MiB，exit code `0`；
- S-AUX official batch-64 one-step loss finite，32 个 pseudo-cold events inductive scores finite，5,963 个 real-cold interaction label leak 为 0；
- S-PLUS joint 与 S-PLUS-CTRL generative one-step loss 均 finite，预算 contract matched，冻结 GRAM checkpoint SHA 不变；
- `PASS_S16_2_SPECGR_IMPLEMENTATION_CONTRACT_SMALL_SMOKE` 已通过；在 small-smoke 时点 S-AUX/S-PLUS formal faithful Gate 与 CTRL formal execution Gate 均保持 PENDING，没有用 one-step smoke 冒充完整 admission；
- S16-2 唯一 report 已建立并保持 `IN_PROGRESS`；其后已按下列 attempt 链完成第一个 S-AUX full train/internal-dev/fixed admission。
- 用户随后指定物理 GPU 2；formal attempt `s16_s2_saux_toys_a1` 在 runner admission 时仅余 23,906 MiB，低于冻结的 24,576 MiB 门槛，以 `GPU_ADMISSION_FAILED` / exit code `9` 结束；workload PID 0、progress 0/4,200、无自动重试，不能据此判算法失败。
- 用户确认后在 GPU 2 独立执行官方 UniSRec 完整 4,799 项训练目录、batch `2048`、单 optimizer step 的资源标定；attempt `s16_s2_saux_batch2048_gpu2_a1` exit code `0`，loss finite，peak allocated/reserved 为 3,440.69/4,314 MiB，总运行 14.24 秒，未读 validation/test。
- 预注册重标规则为 `ceil_to_1024(peak_reserved + max(4096, 0.5×peak_reserved))` 且最低 8,192 MiB；本次得到新 formal attempt 的建议 admission `9,216 MiB`。该结论仅替代资源代理，没有单独晋升科学 Gate；a1 保持不可覆盖，a2 随后独立启动并完成。
- 用户确认启动 formal a2 后，独立 a2 config/runner 已完成且预检通过；启动前 GPU 2 free 从 9,035 MiB 在 30 秒后降至 3,927 MiB，低于 9,216 MiB admission 且低于 4,314 MiB 实测 peak reserved，因此未创建 tmux session、未加载模型、workload PID 0、无自动重试。GPU 2 三个既有外部进程未被修改或终止。
- 用户随后授权改用 GPU 5，并临时释放项目自有 `gram_ablation_scan_gpu5` holder；runner 精确验证初始 PID `2083287`、`reserve_mib=18263`、命令行/state/session 后于 23:22:53 释放，post-release admission free 31,849 MiB。formal a2 workload PID `446318` 已启动并进展至至少 127/4,200 step；Stage15 Beauty B2 与其他进程未被停止或修改。
- holder 恢复是 a2 控制器的 terminal contract：completed/failed/timeout/TERM/INT/HUP 均通过 EXIT trap 以同一 controller/session/state root/reserve 恢复，并验证新 holder 实际占用至少 19,000 MiB；运行期间状态为 pending，最终已按下一条完成。
- formal a2 于 23:25:40 正常完成：50 epochs/700 steps 后 early stop，runtime 135.19 秒；7,435-event/11,924-candidate fixed admission 全部 finite，cold interaction label leak 0，content embedding SHA 不变，artifact contract PASS，最终 `PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION`。
- holder 已在所有 scientific artifact 完成后以原 controller/session/state root 和 `reserve_mib=18263` 恢复；新 PID `464054`，即时实际占用 20,276 MiB、随后稳定 20,292 MiB。Stage15 Beauty B2 和其他用户进程保持运行/未被修改。
- S-PLUS/CTRL objective-complete resource sweep a1 在 GPU 5 holder 保持运行的条件下启动，admission free 11,560 MiB、26/26 tests PASS，但第一个 bf16-mixed S-PLUS pretrain microstep loss non-finite，exit code 1；未进入 CTRL/finetune/item-index、无 summary、无自动重试、test sealed。
- 官方 runner 要求 bf16-mixed，而冻结环境为 PyTorch 1.11.0+cu113；结合既有 FP32 joint smoke finite，将 a1 根因标为 `BF16_RUNTIME_COMPATIBILITY_SUSPECTED_NOT_PROVEN`；a1 已补充当时代码哈希清单，未被覆盖。
- 用户明确确认独立 FP32 resource a2；exact command `bash experiment/phase16/run_stage16_s2_splus_objective_resource_sweep_a2_fp32.sh 5`，admission free 11,552 MiB，26/26 tests 和 S-PLUS/CTRL pretrain/finetune 四条 objective-complete 路径全部 finite，完整 frozen index 为 4,799 items，exit code 0，verdict `PASS_S16_2_SPLUS_OBJECTIVE_RESOURCE_SWEEP`。
- a2 最大 peak allocated/reserved 为 4,731.55/4,978 MiB，建议最低空闲显存 9,216 MiB；holder 未释放且 PID `464054`/实际占用 20,292 MiB 保持不变；checkpoint SHA 前后相同，test/validation 均未读，未生成 efficacy metric。
- 双臂 formal 外推核心 `207.97 GPU·h`、保守 `259.97–415.95 GPU·h`（单卡约 10.83–17.33 天）；建议 1 GPU 至少 9,216 MiB free，工程 admission 保持 10,240 MiB，磁盘预留 8 GiB。正式训练必须等待用户指定 GPU 数/并行策略并明确授权。
- 用户随后确认继续；按连续上下文冻结为物理 GPU5 单卡顺序执行、FP32、保留 holder。formal attempt `s16_s2_splus_ctrl_formal_toys_gpu5_a1_fp32` 于 2026-08-24 01:15:47 启动，tmux `phase16_s2_splus_ctrl_formal_gpu5_a1`，runner/workload PID `1135434/1136093`，admission free 11,560 MiB，初始状态 S-PLUS preprocessing/process alive。
- formal 冻结总计 25,070 optimizer steps；每臂相同 14 天 hard timeout，60 秒 heartbeat/telemetry，pretrain 每 2 epochs、finetune 每 1 epoch 保存可审计恢复 state，但禁止自动 resume/retry。29/29 tests PASS；test/validation sealed。
- GPU5 holder PID `464054`、`reserve_mib=18263` 全程禁止释放/修改，启动核验实际约 20,292 MiB；`holder_released=false`。
- formal a1 于 01:17:51 在 S-PLUS preprocessing 后以 exit code 1 结束，progress 0/25,070；PyTorch 1.11 的 `torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))` 抛出 `Invalid device argument`。模型/optimizer/CTRL 均未启动，无 checkpoint/summary、无自动重试；holder PID 464054 未释放且继续运行。
- 根因已隔离为旧 PyTorch CUDA telemetry API 的 context 初始化时序问题，不是算法或资源 Gate failure。用户确认 a2 后，整数 index-only GPU5 探针仍复现失败，`torch.cuda.init()` 后 reset 的探针 PASS；a2 最小修复据此更新为 context-init-then-reset，a1 保留不覆盖。
- 独立 a2 于 02:07:42 在 tmux `phase16_s2_splus_ctrl_formal_gpu5_a2` 启动，output `toys_seed1502_gpu5_a2_fp32`，runner/workload PID `1412452/1413349`，admission free 11,560 MiB；31/31 tests PASS。
- a2 已越过 a1 failure point并进入 S-PLUS pretrain，首个 optimizer step 完成，paired progress 1/25,070；holder PID `464054` / `reserve_mib=18263` 未释放，test/validation sealed。formal Gate 保持 PENDING，等待 paired 终态。
- 2026-08-28 用户先同意准备、随后明确确认在 GPU7 启动并行 CTRL a4。a4 只运行 `S-PLUS-CTRL`，从 a3 已冻结 `resolved_config.json` 派生，保持 seed/checkpoint/data/epochs/effective batch/optimizer/scheduler/timeout 完全一致；唯一执行差异为同型号 RTX A6000 的物理 GPU7 与隔离 artifact root。
- a4 exact command `bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu7_a4_split_fp32.sh 7` 已于 2026-08-28 10:33:33+08:00 在 tmux `phase16_s2_splus_ctrl_formal_gpu7_a4_split` 启动；输出 `artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu7_a4_ctrl_split_fp32/`，启动 admission free 48,568 MiB，minimum 28,672 MiB、expected peak reserved 17,466 MiB、14 天 hard timeout、8 GiB disk、无 holder 操作、无自动 retry/resume。Stage16 `40/40` tests 与 split preflight PASS；runner/workload PID `1708057/1708947`，初始状态 `RUNNING` / `S-PLUS-CTRL` preprocessing、0/12,535 CTRL optimizer steps。
- 跨 attempt 配对使用只读 finalizer `bash experiment/phase16/run_stage16_s2_splus_ctrl_split_pair_finalize.sh`，只在 GPU5 a3 的 S-PLUS arm 与 GPU7 a4 CTRL arm 各自 PASS 后运行；它比较 scientific config、两阶段预算、起始 checkpoint、finite/admission、防泄漏与 checkpoint 合约，不修改两侧 source artifacts。新增 5 项 split tests 后 Stage16 `40/40` CPU tests PASS；a3 已冻结的 7 个相关代码 SHA 保持原值。
- 为避免 a3 在 S-PLUS PASS 后继续重复 CTRL，2026-08-28 新增 fail-closed one-shot guard；它仅在 S-PLUS summary/admission/checkpoints 完整 PASS、a3 已进入 `S-PLUS-CTRL`、GPU7 a4 健康或已 PASS、runner PID/start-ticks/exact-cmdline 与 CTRL child PPID/cmdline 全部匹配时，向 a3 runner 发送唯一一次 SIGTERM，由 a3 原 EXIT trap 终止子进程并恢复 holder。a4 失败/失联、摘要不完整、身份漂移或任何竞态均不发信号并保留 GPU5 CTRL 后备。dry-run 返回 `WAIT`、`signal_sent=false`，Stage16 全量 `66/66` tests PASS；用户随后确认 exact command，guard 于 2026-08-28 11:04:58+08:00 在 tmux `phase16_s2_splus_ctrl_duplicate_guard_gpu5_a3_gpu7_a4` armed，PID 1807506，当前 `ARMED_RUNNING/WAIT`。
- GPU5 a3 的 S-PLUS arm 随后完成 100+15 epochs、`12,535/12,535` optimizer steps、3,108-event internal-dev 与 7,435-event/11,924-candidate fixed admission，verdict `PASS_S16_2_S_PLUS_FORMAL_EXECUTION`；one-shot guard 在 a3 进入重复 CTRL 后通过全部身份/产物闸门，只终止 a3 runner，并由原 terminal trap 恢复同一 `reserve_mib=18263` holder。guard 终态 `PASS_S16_2_DUPLICATE_CTRL_GUARD`，已完成的 S-PLUS arm artifact 保留且 GPU7 a4 未被修改。
- GPU7 a4 于 2026-08-29 15:19:06+08:00 完成 matched CTRL 全部 `12,535/12,535` optimizer steps，exit 0，peak reserved 4,536 MiB，verdict `PASS_S16_2_S_PLUS_CTRL_FORMAL_EXECUTION`；两臂同一冻结 checkpoint、数据、100+15 epochs、effective/physical batch、optimizer/scheduler/steps/timeout，均未读 validation/test。
- 首次 CPU-only pair finalizer 的 5/5 tests PASS，但对 8 个约 3.7 GiB source checkpoints 做完整 SHA 时恰好触发原 600 秒 timeout；blocked a1 原样保留，无 summary/artifact contract、无 source 修改、无自动重跑。用户确认独立 a2 后，只把 CPU hash timeout `600→1800` 秒，保持 full SHA 与全部 source/scientific contract；定向 8/8、Stage16 全量 `118/118` tests PASS。a2 于 2026-08-29 19:03:47–19:05:41+08:00 exit 0，`PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION`、`PASS_S16_2_SPLUS_CTRL_MATCHED_FORMAL_EXECUTION` 与 `PASS_SPLUS_CTRL_SPLIT_PAIR_ARTIFACT_CONTRACT` 全部通过。S16-2 唯一 report 已更新为 `COMPLETED`。
- S16-3 resource a3 于 2026-08-28 14:44:01+08:00 按用户确认命令启动，物理 GPU4→logical CUDA0，admission free 19,735 MiB，`79/79` CPU tests 与 full train-only `5963/59630/302400` dataset build PASS；三个 z-batch candidate 完成并选择 16。positions 0–4 的 4,096-row covariance 完成，worker 在 position 5 前耗尽 360 秒预注册预算，于 14:50:09+08:00 以 `TIMEOUT / RESOURCE_BLOCKED_BOUNDED_TIMEOUT / exit 124` 终止。partial checkpoint/status/progress/log/telemetry/identity/request shards 完整保留，无 summary、无 Gate 晋升、无自动 retry；validation/test 未读。若继续 a4，因实测超出小实验时间/显存假设，必须使用新 attempt root，重新披露资源并等待用户指定 GPU/授权。
- 用户已指定 S16-3 resource a4 使用物理 GPU4。a4 冻结为独立 config/output/tmux attempt；科学 workload 与 a3 完全一致，candidate cap 仍为 8,192 MiB，整次 attempt 资源边界单独冻结为 expected peak 12,288 MiB、minimum free 18,432 MiB、worker timeout 900s。用户看到 exact command 后最终确认，a4 于 2026-08-28 15:13:22+08:00 在后台 session `phase16_s3_gfull_resource_a4_gpu4` 启动；GPU4 admission free 19,735 MiB，runner/workload PID `2690300/2691233`，execution identity 内 Stage16 `80/80` tests PASS，test/validation 封存、automatic retry=false。
- a4 于 15:20:18 正常结束，worker 400.769 秒，未 timeout；终态 `BLOCKED / RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM / exit 10`。三个 candidate 完成并选择 z-microbatch 16；全六位置 covariance resource/convergence、trigger、generation 与 base-checkpoint parity 完成，peak allocated/reserved `6895.492/8668 MiB`，低于 attempt cap。六个 2,048 维 faithful no-ridge system 的 rank 分别为 `71/1058/1813/1982/2043/1741`，全部不足 2,048，且无 ridge/pinv/jitter/resample fallback；所以没有 completed delta/aggregate/formal projection，S16-3 Gate 未通过，formal G-FULL 与 S16-4 G-FULL arm 不解锁。raw 的非权威 `solve_status` 标签与其 56 valid-z diagnostics 不一致；顶层 verdict/失败 checks/status 正确，后续代码仅修复未来标签并使当前回归达到 `81/81 PASS`，a4 原始 artifact/identity 未改写。
- 用户明确要求不等待 S-PLUS、直接开发 S16-3B。该 diagnostic 使用 full train-only covariance 与全部 `302400` request keys 构造每位置 all-request upper-bound system。GPU4 b1 于 16:01:35 启动并于 17:38:58 完成全部六位置，raw elapsed `5830.412s`、peak reserved `8648 MiB`，不是 timeout/OOM/admission failure。终态因唯一 contract check `positive_semidefinite_evidence` 失败而为 `FAILED / exit 3`：position 5 FP32-finalized covariance 有 2 个超过冻结 tolerance 的负特征值。原 artifact 保留且不自动重跑。用户随后确认 CPU-only recovery c1；c1 于 18:07:35 exit 0，positions 0–4 proof-eligible、position 5 ineligible，positions 0–3 rank `74/1216/1938/2033 < 2048`，正式裁决 `PASS_S16_3B_RECOVERY_ADJUDICATION_COMPLETE / PROVEN_STRUCTURAL_RANK_BLOCKED`。b1 五类 input SHA 前后不变，S16-3 Gate 不晋升。
- 用户决定将 S16-3 后续方向改为 `GenRecEdit-inspired → GRAM`。新方法 `G-RIDGE` 不继承 faithful Gate，只把 A4 的 singular no-ridge solve 替换为预注册的 condition-targeted spectral ridge；其余数据、z/covariance/key/aggregation/trigger、资源 sweep 与 sealed-set contract 保持一致。独立实现/config/runner/finalizer 已完成，Stage16 全量 `104/104` CPU tests PASS；GPU4 resource r1 已准备、尚未启动。

### S16-3：GenRecEdit-inspired G-RIDGE→GRAM 实现、contract 与 admission

**任务**：

- 保留 S16-3F faithful G-FULL 与 S16-3B 的 immutable 失败/结构性阻断证据，不覆盖、不重命名为 PASS；
- 实现 full-target request batching、official z lifecycle/scheduler/cache、covariance、key extraction、valid-z filtering、delta aggregation 和 trigger；
- 保存每位置 request 总数、cache hit、valid/failed z、概率/排名诊断、delta norm/rank/condition、未编辑 parity 和 warm-preservation 证据；
- 保持 A4 的全量工作量和 covariance 数值路径，只以 `condition_targeted_spectral_ridge_v1` 替换 faithful no-ridge solve；固定 condition target `1e6` 和 safety margin `1e-6`，ridge 仅由 train-only system spectrum 决定；
- 保存 ridge 前/后 spectrum、rank/nullity、condition、ridge relative scale、Cholesky 与 solve residual，禁止 pinv/jitter fallback/outcome resampling；
- 在 Toys train-only item-disjoint 上做固定规模 admission，不使用 validation target 判 edit success。

**Gate**：resource 先通过 `PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP`，再以独立 `PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION` 作为 formal Gate。faithful `PASS_S16_3_GFULL_FAITHFUL_CONTRACT_ADMISSION` 保持未通过且不可继承；resource sweep 不能替代 formal G-RIDGE admission。

**faithful 历史终态（2026-08-28）**：resource a4 完成全部冻结组件，但六个 faithful no-ridge resource systems 均秩不足，裁决为 `RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM`；faithful Gate 未通过。

**S16-3B 独立 diagnostic（不改 faithful Gate）**：只验证必要条件 `rank(1000·C_full + K_allᵀK_all)=2048`。b1 已完成全部 `302400` keys，但因 position-5 numerical-PSD evidence 失败而保持 artifact `FAILED`；CPU-only recovery c1 已在不改写 b1 的前提下完成机器可读裁决，positions 0–4 proof-eligible、position 5 ineligible，positions 0–3 结构性秩亏，verdict/classification 为 `PASS_S16_3B_RECOVERY_ADJUDICATION_COMPLETE / PROVEN_STRUCTURAL_RANK_BLOCKED`。该结果关闭当前 faithful no-ridge G-FULL 路径，也是 G-RIDGE 必须显式正则化而不能冒充 faithful 的方法动机。

**S16-3R 终态**：GPU5 resource r1 的 position-0 residual `3.920733e-6 > 1e-6` 工程失败与 artifact 保持不变；独立 r2 已 `COMPLETED / PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP`。formal f1 因运行期共享代码漂移 FAILED，独立 immutable f2 因 allowlisted artifact 父路径未映射而在 GPU 前 FAILED；两者原 artifact 均保留。独立 immutable f3 保持 f2 的 seed/data/method/workload/Gate/resource，只新增失败谱系并从一开始冻结已修复的 parent-link path mapping；主仓与 `.runtime/phase16_s3r_gridge_f3_runtime` 均 `123/123` tests PASS。f3 于 2026-08-30 15:00:28+08:00 在 GPU5 启动，于 2026-08-31 07:08:55+08:00 exit 0：全部 `302400` requests、六位置 FP64 G-RIDGE solve、`7435` 个 item-disjoint admission events 与 `512` 个 warm-preservation pairs 完成，formal Gate 晋升为 `PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION`；elapsed `58061.718s`，peak CUDA reserved `8880 MiB`，validation/test 未读、未生成 efficacy metric。后置非权威 repeat queue 因新增 GPU5 PID `464283` 一直未启动 cycle；用户于 2026-08-31 要求停止后，只向已核验 runner PID `776933` 发送 SIGTERM，队列终态 `INTERRUPTED / exit 143 / process_alive=false`，tmux 已退出，formal f3 与 GPU5 既有进程均未修改。

**唯一 report**：`report/第十六阶段/Stage16_S3_FaithfulGenRecEdit_GRAM合约与Admission报告.md`

### S16-4：Toys standalone frozen validation

**前提**：只运行 S16-2 faithful SpecGR arm 与 S16-3R G-RIDGE 中各自 contract+admission PASS 的 arms；带 F0、R2 和对应 matched control。faithful G-FULL 不进入 efficacy。

**任务**：

- seed-1502 完整 Toys validation；
- 同时报 cold/warm/overall、candidate reachability、acceptance/redraft、edit coverage、update/inference cost；
- 在同一 report 中对比 faithful 与 Stage15 pilot，但不混同名称；
- 按 Gate 决定是否扩展 adaptation seeds；不根据 validation 修改 faithful 定义。

**Gate**：为每个 arm 分配 `PASS_STANDALONE_COLD_SIGNAL`、`PASS_STANDALONE_PARETO`、`FAIL_STANDALONE`；contract-pass 且已冻结的 primary arms 进入 Beauty，不依单个 point estimate 临时调参。

**唯一 report**：`report/第十六阶段/Stage16_S4_ToysFaithfulStandalone冻结验证报告.md`

**完成状态（2026-09-01）**：GPU4 a7 已完成 S-AUX、S-PLUS-CTRL、S-PLUS、G-RIDGE 四臂各 8,789/8,789 events；旧 finalizer 因把 runtime schema 硬编码为 GPU0 a3 名称而在 CPU finalization 阶段 `FAILED / ARTIFACT_CONTRACT_FAILED`，原 a7 status/log/predictions/SHA 全部保留，repeat 未启动。修复通用 runtime identity validator并补齐四比较 Holm correction 后，CPU-only a8 从冻结 a7 predictions 完成统计恢复；未重跑 GPU 推理、未修改 a7、test sealed。后续 comparator audit 发现 a8 的 `R2` 实为 Stage13 P0 `r2_top50`，不符合本计划在 §3 冻结的 `R² portfolio@2`。独立 CPU-only a9 使用冻结 P0 F0/resolver、P6 portfolio candidates 与 cold manifest，按 Phase13 `unconditional portfolio@2` 规则逐用户重建正确 R2；candidate/F0 mismatch 均为 0，与 Phase13 P6 aggregate 最大误差 `5.55e-17`，a7/a8 均保持不变。S-AUX−F0 cold H@50=`+0.049920`，95% CI=`[+0.042592,+0.057477]`、Holm p=`9.211e-43`；相对正确 R2，S-AUX cold H@50 高 `+0.030456`，但 cold NDCG@10、warm/overall NDCG 和成本更差，两者互不严格支配，S-AUX 正式标签为 `PASS_STANDALONE_PARETO`。S-PLUS−matched CTRL=`+0.010534`，CI=`[+0.007557,+0.013739]`、Holm p=`4.263e-14`，仍被正确 R2 支配；G-RIDGE−F0=`−0.010076`，CI=`[−0.013281,−0.007328]`，且相对 F0/正确 R2/S-AUX 的 treatment-only cold hits 均为 0。唯一 report 已更新；停止原 `S-AUX + G-RIDGE` 组合主线，下一 Gate 修订为一次严格冻结的 S-AUX Beauty transfer check。

### S16-5：Beauty standalone frozen transfer check

**a9 后修订边界（2026-09-01）**：S16-4 已证明 G-RIDGE 对 F0、正确 portfolio@2 与 S-AUX 均无独有 cold hit，故 S16-5 不再执行 G-RIDGE efficacy，也不预付原四臂组合。只允许 S-AUX 进入 Beauty frozen transfer；F0 与按同一冻结规则重建的 `unconditional portfolio@2` 是必要对照。

**任务**：

- 使用 Toys 前已冻结的 S-AUX 算法、超参规则、budget、seed 和 evaluator；
- 只重建 Beauty 域内允许的 S-AUX drafter/index，并在打开 validation 前验证 F0 与 `unconditional portfolio@2` 的 comparator identity/SHA；
- 不因 Beauty state construction 或 efficacy 改回 Toys；
- 记录跨域稳定性与方法失效位置。

**Gate**：冻结 S-AUX 双域证据。若 Beauty 的 S-AUX−F0 cold H@50 paired CI 下界与 Holm-adjusted exact p 再次通过 cold-signal Gate，则记录其相对正确 portfolio@2 的 Pareto/trade-off，并允许提出后续条件式方法修订；若 Beauty 不复现 cold signal，则停止 S-AUX 与全部 Stage16 组合方法开发。任何结果都不得回 Toys 调参。

**唯一 report**：`report/第十六阶段/Stage16_S5_BeautyFaithfulStandalone冻结确认报告.md`

### S16-6：原四臂组合 contract 与 internal-dev 互补性诊断（已停止，审计保留）

**执行状态**：`NOT_UNLOCKED_STOP_GRIDGE_NO_COMPLEMENTARITY`。下列原始预注册设计只为审计保留，不得据此启动 C01/C11。若 S16-5 Beauty 复现 S-AUX signal，必须先形成新 plan amendment，将对照改为 portfolio@2 default + S-AUX conditional route，并重新取得用户授权。

**任务**：

- 实现 C00/C10/C01/C11 同路径 evaluator；
- 验证 C10=S-AUX、C01=G-RIDGE 的数值 parity，C00=F0 exact parity；
- 证明 C11 真正在 edited GRAM 上完成 candidate verification/redrafting/generation；
- 在 train-only internal-dev 上估计交互项与互补子群，只允许分析预注册特征：prefix depth、draft confidence、verifier score、edit success/cache status、warm/cold membership；
- 冻结 validation 前的四臂代码与 feature schema。

**Gate**：`PASS_S16_6_FACTORIAL_CONTRACT`。该 Gate 只证明可归因路径和 internal-dev 信号，不是论文 efficacy。

**唯一 report**：`report/第十六阶段/Stage16_S6_四臂组合合约与InternalDev互补诊断报告.md`

### S16-7：原四臂 frozen validation 与 interaction 检验（已停止，审计保留）

**执行状态**：`NOT_UNLOCKED_STOP_GRIDGE_NO_COMPLEMENTARITY`。下列原始 Gate 不再授权任何运行，仅保留为否决前协议记录。

**任务**：

- 使用冻结 C00/C10/C01/C11 在 Toys 运行；按 S16-5 Gate 决定是否执行 Beauty full combination；
- 报告 C11−C10、C11−C01 和 factorial interaction；
- 完整报告冷/暖子群、cost、状态大小和失败事件；
- 不仅展示 C11 最优指标，四臂全部入表。

**Gate**：

- `PASS_COMBINATION_INCREMENT`：C11 对 C10 和 C01 的 cold H@50 paired CI 下界均 > 0；
- `PASS_POSITIVE_INTERACTION`：event-level interaction 的 paired CI 下界 > 0；
- 如果只有特定子群互补，必须在两域方向一致，每域支持量至少为 cold events 的 5% 且不少于 200 events，才允许进入 S16-8。

**唯一 report**：`report/第十六阶段/Stage16_S7_四臂组合冻结验证与交互效应报告.md`

### S16-8：条件式方法开发（非自动）

原 `S-AUX + G-RIDGE` 路径因 S16-4 零互补性已永久停止，不能再由 S16-7 解锁。替代路径仅在 S16-5 Beauty 复现 S-AUX cold signal 后，才允许提交 `portfolio@2 default + S-AUX conditional route + warm-risk abstention` 的新 plan amendment，且需用户明确批准具体方法假设、对照、Gate 与预算。

若上述替代 Gate 未解锁，S16-8 不启动任何方法实验，但仍写该步骤的唯一 report，记录 `NOT_UNLOCKED_STOP` 及其证据，使阶段链条可完整收口。

允许的方向只能来自已冻结失败特征，例如：

- confidence-aware edit activation；
- prefix-depth conditional verification/editing；
- warm-risk abstention；
- 根据 editability 调整 draft/verify 路由。

禁止无假设地联合搜 threshold、seed、requests、candidate budget 和 ranking weight。若解锁，须在执行前将具体 method、loss/gate、对照、有限调参空间和 stop rule 更新到本 plan 新修订版。

**唯一 report**：`report/第十六阶段/Stage16_S8_条件式方法开发与消融报告.md`

### S16-9：最终冻结 test（需用户再次明确授权）

启动前必须同时满足：

- 最终方法、所有 baseline、checkpoint/state、seed、metric code、evaluator、candidate budget 和 report schema SHA 冻结；
- 无未完成的 validation 调参；
- 生成 test-open manifest 与一次性 exact commands；
- 用户在看到 GPU 数、每卡显存、预计时长和总成本后明确同意。

test 打开后禁止任何方法、超参、seed 和 exclusion rule 修改。异常只能做可证明不改数值语义的工程恢复，并完整披露。

**唯一 report**：`report/第十六阶段/Stage16_S9_最终冻结Test与总结报告.md`

---

## 7. 统计 Gate 与互补性定义

### 7.1 Standalone Gate

对每个 method `M` 与其正确 control `B`：

```text
ColdGain(M) = cold_H@50(M) - cold_H@50(B)
WarmDelta(M) = warm_NDCG@10(M) - warm_NDCG@10(B)
```

- `PASS_STANDALONE_COLD_SIGNAL`：ColdGain paired-bootstrap 95% CI 下界 > 0；
- `PASS_STANDALONE_PARETO`：先满足 cold signal，且在 cold H@50、warm NDCG@10、update/inference cost、extra state 上不被 R2 或正确 matched control 严格支配；
- `FAIL_STANDALONE`：contract PASS 但 cold signal 未达标；不把 CI 跨 0 写成等价；
- warm 损失不被 cold 收益自动抵消，始终独立报告。

### 7.2 Factorial interaction

对每个 event `i` 与指标 `m`：

```text
Interaction_i(m) = m_i(C11) - m_i(C10) - m_i(C01) + m_i(C00)
```

对 `Interaction_i` 做 paired bootstrap，同时报告：

- mean interaction 与 95% CI；
- C11−C10 与 C11−C01；
- cold/warm 分层；
- prefix depth、draft confidence、edit success/cache status 的预注册子群；
- cost interaction：组合增益是否只由额外 forward/state 换取。

### 7.3 证据纪律

- 报告全部预注册 primary comparisons，不只报有利的 arm/domain/seed/subgroup。
- point estimate、CI、event count、unique target item count 同时报告。
- 重复 target item 的诊断补充 item-level bootstrap，不取代 event-level primary。
- 对实现忠实度、结果有效性和方法创新性分别下结论；不用任一项代替另一项。

---

## 8. GPU、后台、status 与资源规则（硬约束）

### 8.1 实验大小与启动权

#### 小实验

同时满足以下条件时视为小实验：

- 预计 wall time `≤10 分钟`；
- 单卡；
- 预计新增显存 `≤8,192 MiB`；
- 非 full training、full-target editing、full validation/test。

执行规则：

1. 启动前实时查询 `nvidia-smi`；
2. 在不影响已有进程的卡中，自动选择同时满足显存 admission 且负载最低/空闲显存最多的 GPU；
3. 可在前台实时执行并返回结果；
4. 若无卡满足，不停止他人进程，只报告所需显存并等待用户决定。

#### 大实验

任一下列条件成立即视为大实验：

- 预计 wall time `>10 分钟`；
- full training、full-target editing、full validation/test；
- 预计新增显存 `>8,192 MiB`；
- 需要多卡或可能影响现有任务。

启动前必须先告知用户：

- 所需 GPU 数；
- 每卡最小空闲显存 MiB 和预计 peak；
- 预计 wall time / GPU-hours；
- 预计 CPU RAM / disk；
- hard timeout；
- exact command、artifact 目录和 `status.json` 路径。

必须等用户指定 GPU 或明确批准多卡后才能启动；不自动换卡、降低 admission 或拆分科学 workload。

### 8.2 后台与 status

- 所有预计 `>10 分钟`的实验必须后台运行，禁止使用前台长时占用会话。
- 用户通过 `artifacts/phase16/<stage>/<run>/status.json` 观察进度；不要求助手主动实时监控或循环轮询。只有用户询问状态时才做一次只读核对。
- status 必须由持有 worker 生命周期的 wrapper 原子更新，不依赖外部助手会话存活。
- 运行中至少每 60 秒或每个 progress batch 更新 `updated_at/last_progress_at/progress_current`；如心跳超过 5 分钟未变，status 写 `STALE_HEARTBEAT`，但不自动 kill。
- 长阶段（训练、covariance、z optimization、evaluation）必须分别有 stage 和 progress unit，不能整个数小时始终显示 0/N。

`status.json` 至少包含：

```text
experiment_id, attempt_id,
status, status_code, stage, reason,
started_at, updated_at, last_progress_at,
runner_pid, workload_pid, process_alive,
physical_gpu, visible_gpu, gpu_count,
minimum_free_mib_per_gpu, admission_free_mib_per_gpu,
expected_peak_mib_per_gpu,
progress_current, progress_total, progress_unit,
hard_timeout_seconds,
workload_rc, exit_code, exit_code_pending,
test_read, automatic_retry,
exact_start_command,
output_dir, log_path, summary_path
```

最终状态必须是 `COMPLETED/FAILED/BLOCKED/TIMEOUT/KILLED_TARGET_LEAKAGE` 之一，且 wrapper 在任何 exit path 都要写入 exit code、reason 与 `process_alive=false`。

### 8.3 GPU 安全

- 不停止、迁移、限速或修改任何现有的其他进程；不因实验需求默认释放 holder。
- GPU0 和任何已有 holder 默认不可用，除非用户对该次任务明确授权。
- 如用户授权临时释放项目 holder，无论成功、失败、timeout 或异常退出，都必须通过同一控制器恢复；释放/恢复时间、PID、显存和结果写入 status/resource summary。
- worker 启动时重做 admission；从计划到启动期间显存漂移导致不足时直接 `BLOCKED`，不加载模型，不自动重试。
- 记录 physical GPU→`CUDA_VISIBLE_DEVICES`→进程内 logical device 映射，避免设备编号混淆。

---

## 9. Retry、报告与历史留痕规则

### 9.1 Retry

- `automatic_retry=false` 是全阶段硬约束。
- failed/blocked/timeout 后不自动换卡、换 seed、改 threshold、扩 requests、加 steps 或改数据子集。
- 只有已定位的工程问题才可候选 retry；重试前必须有 root-cause 证据、回归测试、“不改科学参数/数据/Gate”声明和新 attempt artifact 目录。
- 方法级 Gate FAIL 不得以工程 retry 名义 rescue；如要改方法，进入下一计划修订/新 stage。
- 大实验任何 retry/resume 仍需先告知用户资源需求并等待指定 GPU。

### 9.2 每一步只有一份 report

- 每个 `S16-k` 完成、失败或被阻塞时，都必须形成该步骤的唯一正式 report；没有 report 不得宣布该步骤完成。
- 同一步的 smoke、preflight、attempt、retry、blocked 与 recovery 不分别写 report；统一合并到该步骤 report 的“试错与恢复摘要”表。
- 旧 attempt artifact 永久保留且不覆盖；report 只用简表记录 attempt ID、改动类型、结果、资源、test_read 和权威 verdict，不重复粘贴完整日志。
- 每份 report 必须包含 Material Passport、方法/数据版本、exact commands 或 command manifest、artifact 路径、Gate verdict、指标/CI、资源、异常、防泄漏、reproducibility verdict 和下一唯一步骤。
- 每步骤结束时同步更新本 plan 顶部状态、对应小节实测记录、资源与决策记录；不反向篡改原 Gate。

---

## 10. Artifact 与目录合约

```text
experiment/phase16/
  configs/
  protocol/
  tests/
  run_stage16_*.sh

artifacts/phase16/
  s0_fidelity_contract/
  s1_data_resource_preflight/
  s2_specgr/{contract,admission,formal}/
  s3_genrecedit/{contract,admission,formal}/
  s4_toys_standalone/
  s5_beauty_standalone/
  s6_factorial_contract/
  s7_factorial_validation/
  s8_conditional_method/
  s9_final_test/

report/第十六阶段/
  Stage16_S0_官方算法忠实度与GRAM映射冻结报告.md
  Stage16_S1_数据防泄漏InternalDev与资源预检报告.md
  Stage16_S2_FaithfulSpecGR_GRAM合约与Admission报告.md
  Stage16_S3_FaithfulGenRecEdit_GRAM合约与Admission报告.md
  Stage16_S4_ToysFaithfulStandalone冻结验证报告.md
  Stage16_S5_BeautyFaithfulStandalone冻结确认报告.md
  Stage16_S6_四臂组合合约与InternalDev互补诊断报告.md
  Stage16_S7_四臂组合冻结验证与交互效应报告.md
  Stage16_S8_条件式方法开发与消融报告.md
  Stage16_S9_最终冻结Test与总结报告.md
```

每个正式 run 至少包含：

```text
status.json
config.json
summary.json
data_provenance.json
input_file_sha256.json
code_sha256.json
open_file_manifest.json
resource_summary.json
command_manifest.json
```

大实验另必须包含 `run.log`、`gpu_telemetry.csv` 与可分阶段恢复的 progress/checkpoint manifest。checkpoint 只用于确定性 resume，不允许跨配置复用。

不提交第三方源码、官方/本地大权重、raw predictions、per-user dump、logs、telemetry 或数据 cache；只提交实验代码、测试、配置、聚合 summary/provenance、report 和 plan 修订。

---

## 11. 资源分层与授权表

| Stage | 类型 | GPU 决策 | 当前授权 |
|---|---|---|---|
| S16-0 | CPU 审计 | 不用 GPU | 可执行计划/代码审计；不启动科学运行 |
| S16-1 | CPU + 小 GPU 试跑 | 小实验实时选空闲卡 | 可做预检；大实验只估算不启动 |
| S16-2 | SpecGR 训练/admission | 加速 batch `16/4/64` 实测 peak reserved 17,466 MiB；a3/a4 admission 28,672 MiB free | GPU5 S-PLUS a3 与 GPU7 CTRL a4 已获授权并行运行；a3 holder 运行期释放、终态恢复 |
| S16-3 | full-target editing/admission | 大实验；先报 GPU 数/显存/时长 | 需用户指定 GPU |
| S16-4–5 | full standalone validation | 大实验 | 逐次需用户指定 GPU |
| S16-6 | contract/internal-dev | 小实验自选；超 10 分钟则转大实验规则 | 按实测分类 |
| S16-7–8 | full factorial/method development | 大实验 | 需用户指定 GPU；S16-8 还需方法授权 |
| S16-9 | final test | 大实验 | 需独立 test-open + GPU 明确授权 |

计划不在 S16-1 telemetry 前伪精确承诺 GPU-hours。完整 UniSRec/SpecGR++ 训练与 full-target GenRecEdit 的数量级、显存和运行时间必须由真实小试跑外推。

---

## 12. 风险、止损与论文声明边界

| 风险 | 处理 |
|---|---|
| faithful 名称被简化预算稀释 | F0/F1/F2/F3 矩阵；任何 F2 只进 scaled diagnostic |
| SpecGR-Aux 与 SpecGR++ 混称 | 分名、分对照、分训练成本；论文不用“SpecGR-GRAM”覆盖两者 |
| S-PLUS 与 B0 预算不等 | 必须有 S-PLUS-CTRL；无 matched control 不做因果结论 |
| GenRecEdit full targets 成本过高 | 先报真实数量和资源；不用 4-request 结果替代 faithful primary |
| 概率空间与 lexical constraint 语义不确定 | S16-0 fixed-width emulation 在 efficacy 前冻结；多定义只作机制消融 |
| validation 已有 Stage15 暴露 | 调参只用 train internal-dev；validation 作 source-domain evidence；test 仍一次性封存 |
| 同一 validation 上反复微调缝合 | 四臂代码和子群先冻结；S16-8 必须另行授权并更新 plan |
| 组合缺少新颖性 | 若无正 interaction/可复现条件机制，只写 transplantation/analysis，不宣称新方法 |
| warm 伤害被 cold 增益隐藏 | warm NDCG@10 为 primary cost，每域/每臂必报 |
| test 事后修法 | test-open 后禁止修法；工程恢复也必须证明数值语义不变 |
| status 与真实进度脱节 | wrapper 持续心跳、分 stage 进度、最终 exit trap；心跳失效只报 stale，不自动终止科学进程 |

允许声明：

- faithful core-algorithm transplantation to GRAM（仅对通过 fidelity contract 的方法）；
- 同 backbone/ID/protocol 下的 retrieval-verification 与 parameter-editing 干预位置对比；
- 预注册四臂 interaction 证据；
- 条件机制的有限主张（只在 S16-8/9 证据支持时）。

禁止声明：

- 复现了官方 TIGER/官方论文数字（除非另有 native evidence）；
- Stage15 pilots 等于 faithful SpecGR/GenRecEdit；
- 简单 C11 串联自动构成新颖算法；
- Beauty validation 是未触碰的独立第三域；
- 单 seed adaptation 结果等于多 backbone 稳健性。

---

## 13. 时间顺序、完成定义与当前唯一下一步

| 顺序 | Stage | 完成定义 | 报告数 |
|---:|---|---|---:|
| 1 | S16-0 | fidelity matrix、bridge tests、参数/语义冻结、Gate verdict | 1 |
| 2 | S16-1 | data/internal-dev/test guard/resource estimates 冻结 | 1 |
| 3 | S16-2 | S-AUX/S-PLUS/CTRL contract+admission verdict | 1 |
| 4 | S16-3 | G-RIDGE contract+admission verdict；faithful G-FULL 历史证据保留 | 1 |
| 5 | S16-4 | Toys standalone full validation 和 promotion label | 1 |
| 6 | S16-5 | Beauty standalone frozen transfer evidence | 1 |
| 7 | S16-6 | 四臂数值合约与 internal-dev 互补诊断 | 1 |
| 8 | S16-7 | frozen factorial validation 与 interaction verdict | 1 |
| 9 | S16-8 | 条件式方法/消融或带理由 STOP | 1 |
| 10 | S16-9 | 用户授权后的一次性 test 与总结 | 1 |

**当前唯一下一步**：

1. S16-4 已由 GPU4 a7 全量 predictions、CPU-only a8 统计恢复和 CPU-only a9 comparator correction 完成；a7 原 `FAILED`、a8 错误 R2 身份的历史裁决与 a9 最终纠正裁决同时保留，禁止再启动重复 GPU 推理；
2. 当前唯一推荐下一步是一次严格冻结的 S-AUX Beauty transfer check；这是对 Toys `PASS_STANDALONE_PARETO` 的跨域 Gate，需用户另行指定 GPU 后才执行，test 继续封存；
3. 终止 G-RIDGE efficacy/组合主线：其 Toys cold H@50 明确低于 F0，且相对 F0、正确 R2、S-AUX 均没有 treatment-only cold hit，缺少最基本的 oracle complementarity；不把 G-RIDGE 带入 Beauty 或 S16-6/S16-7；
4. Beauty 只允许沿用 Toys 前冻结的 S-AUX 算法、threshold、budget、seed 与 evaluator，并先冻结正确 `unconditional portfolio@2` comparator identity；不得据 Beauty 结果回 Toys 调参；
5. 只有 S-AUX 在 Beauty 方向一致时，才修订 S16-6 之后为 `portfolio@2 default + S-AUX conditional selector + warm-risk abstention` 的 train-only internal-dev 可证伪 screen，冻结特征、对照、Gate 与 stop rule 并再次取得执行授权；不得在原 G-RIDGE 四臂名义下静默替换。

---

## 14. 决策记录

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-08-23 | 建立独立 Stage16，不改写 Stage15 科学协议 | 已查看 Stage15 结果；faithful 迁移是实质方法变更，需要新的预注册边界 |
| 2026-08-23 | Stage15 B2/B3 重分类为 pilot | 轻量 drafter、4 requests/position 和 256 covariance 不足以支撑 faithful baseline 结论 |
| 2026-08-23 | 同时实现 SpecGR-Aux 与 SpecGR++，但分对照 | SpecGR 官方有外部 drafter 与 self-drafting 两路；不事后挑变体冒充唯一方法 |
| 2026-08-23 | 四臂主组合固定为 S-AUX + G-FULL | 冻结 GRAM 上 retrieval-verification 与 parameter editing 干预位置最清晰，便于交互归因 |
| 2026-08-23 | 每个 S16-k 只写一份 report | retry/attempt 全部汇总到该步骤唯一 report，旧 artifact 保留 |
| 2026-08-23 | >10 分钟实验全部后台，用户只看 artifact status | 不做主动实时监控；status heartbeat 由 wrapper 独立维持 |
| 2026-08-23 | 小 GPU 实验自选当前空闲卡；大实验先报资源再等用户指卡 | 继承用户 GPU 执行规则，不触碰其他进程 |
| 2026-08-23 | S16-0 以 23 个 function-level 映射和 18 项 bridge checks 通过 | 官方固定 commit/源码证据、fixed-width→variable lexical 语义与主实现边界均已在 efficacy 前冻结 |
| 2026-08-23 | GenRecEdit probe 冻结为 legal-competitor argmax + full-vocabulary probability；`0.3` 仅用于 cache probe | 与官方源码的比较集合/概率/optimizer satisfied 三种语义分别对应，避免沿用 Stage15 改写 Gate |
| 2026-08-23 | S16-1 以双域零泄漏、完整工作量计数和单卡 bounded telemetry 通过 | 安全投影 SHA、item-disjoint pseudo-cold、user-disjoint internal-dev、路径/分区/资源合约均在大实验前冻结 |
| 2026-08-23 | 第一个大实验固定为 S-AUX，需用户指定单张至少 24 GiB 空闲 GPU | S16-1 只授权小试跑；S-AUX 预计 18–48 h，S-PLUS/G-FULL 尚需更完整的吞吐 sweep |
| 2026-08-23 | 以固定 SpecGR + RecBole v1.2.0 官方源码链取代 S16-1 的 S-AUX 资源代理 | 官方 UniSRec/TransformerEncoder source identity 与真实 forward/backward 已通过；仅内容宽度 768→1024 为冻结 F1 接口改动 |
| 2026-08-23 | S16-2 one-step small smoke 只通过 implementation/contract Gate，不晋升 formal Gate | 24 tests 与三臂 finite/resource/zero-leak 只证明可执行性，不能替代完整训练和固定规模 admission |
| 2026-08-23 | S-AUX formal a1 在 GPU 2 以 `GPU_ADMISSION_FAILED` 保留，禁止自动重试 | runner 二次检查仅 23,906 MiB free，低于冻结 24,576 MiB；workload 未启动，属于资源 admission failure 而非算法 failure |
| 2026-08-23 | 以官方 UniSRec batch-2048 单步实测将后续 S-AUX formal admission 建议修订为 9,216 MiB | 完整 4,799 项训练目录的 peak reserved 为 4,314 MiB；按运行前冻结的 4,096 MiB 安全余量与 1,024 MiB 向上取整得到 9,216 MiB；仅修订新 attempt 的资源门槛，不把 sweep 冒充科学 Gate |
| 2026-08-23 | formal a2 在 GPU 2 启动前资源检查停止，保留 ready 状态而不制造失败 workload | 两次 free 快照为 9,035/3,927 MiB，均低于 9,216 MiB admission，后者也低于实测 4,314 MiB peak；不触碰既有进程、不降阈值、不自动换卡或重试 |
| 2026-08-23 | 用户授权释放 GPU 5 自有 holder 后启动 formal a2，并把同配置恢复设为 terminal contract | 仅释放经 PID/命令行/state/session 四重验证的 `gram_ablation_scan_gpu5`（PID 2083287，reserve 18,263 MiB）；post-release free 31,849 MiB；所有 terminal 路径恢复并验证实际占用至少 19,000 MiB，不触碰 Stage15 Beauty B2 或其他用户进程 |
| 2026-08-23 | S-AUX formal a2 通过，GPU 5 holder 已原配置恢复 | 50 epochs/700 steps early-stop formal、fixed admission 与 artifact contract 全部 PASS；holder 新 PID 464054、reserve 18,263 MiB、稳定实际占用 20,292 MiB；S-AUX Gate 晋升但不把 internal-dev admission 当 efficacy |
| 2026-08-24 | S-PLUS objective resource sweep a1 以 bf16 non-finite 保留，不自动重试 | GPU 5 admission 与 26 tests 均通过，首个 S-PLUS pretrain microstep即 non-finite；holder 未释放、未进入 CTRL/finetune/index；本地 PyTorch 1.11+cu113 与官方 bf16-mixed 路径兼容性高度可疑但尚未由新 attempt 隔离 |
| 2026-08-24 | 用户确认独立 FP32 a2；objective-complete S-PLUS/CTRL resource sweep PASS | 数据/objective/batch/optimizer/step/matched budget 不变；四路径 finite、full 4,799-item index、peak reserved 4,978 MiB，checkpoint 不变且 test sealed；holder PID 464054 全程保留 |
| 2026-08-24 | S-PLUS/CTRL formal 不自动启动，等待大实验 GPU 授权 | 双臂核心 207.97 GPU·h、保守 259.97–415.95 GPU·h；建议每 worker 至少 9,216 MiB free、工程 admission 10,240 MiB、磁盘 8 GiB，超出小试授权范围 |
| 2026-08-24 | 用户确认 GPU5 单卡顺序启动 S-PLUS/CTRL formal a1 | 已披露并接受 259.97–415.95 GPU·h；FP32、25,070 steps、每臂 14 天 hard timeout；admission free 11,560 MiB，29 tests PASS；holder PID 464054 保持且不释放，test/validation sealed |
| 2026-08-24 | formal a1 以 preprocessing CUDA telemetry compatibility failure 保留，不自动重试 | PyTorch 1.11 对 `reset_peak_memory_stats(torch.device("cuda:0"))` 报 Invalid device argument；progress 0/25,070，未加载模型/optimizer/CTRL；最小整数 device-index 修复必须进入独立 a2 并重新确认 |
| 2026-08-24 | 用户确认独立 formal a2；context-init-then-reset patch 后启动 | index-only GPU5 probe 仍失败而 context-init probe PASS；31 tests PASS，a2 已完成 S-PLUS pretrain 首个 optimizer step；科学配置/预算不变，holder 未释放，a1 保留不覆盖 |
| 2026-08-24 | 用户授权中断 a2、完全释放 GPU5 holder 并做更大 microbatch 扫描 | a2 以 `INTERRUPTED`/143 保留 15/25,070 steps；仅停止核验过的 PID 464054 holder，不触碰其他 GPU5 进程 |
| 2026-08-24 | 加速 batch sweep 选择 `16/4/64` | `64/16/16` 与 `32/8/32` 分别在约 30.04/31.12 GiB reserved OOM；`16/4/64` peak reserved 17,466 MiB、joint objective/gradient finite、effective batch 1024/256 不变；sweep 终态恢复 holder |
| 2026-08-24 | 独立 formal a3 以 F1 batching adaptation 启动，运行期全释放 holder | 35 tests 与 9-file SHA freeze PASS；post-release free 33,995 MiB；tail generation/ranking 按样本数加权，optimizer steps 25,070 不变；EXIT controller 在所有受控终态恢复 `reserve_mib=18263` |
| 2026-08-28 | 用户同意准备但未授权启动 GPU7 并行 S-PLUS-CTRL a4 | 只新增隔离 runner/config/finalizer/test，不修改运行中 a3 冻结代码或 artifact；a4 与 a3 scientific core exact match，GPU7/独立输出为唯一执行差异，40/40 tests 与 CPU-only split preflight PASS |
| 2026-08-28 | 用户明确确认在 GPU7 启动 S-PLUS-CTRL a4 | 启动闸门确认 GPU7 free 48,568 MiB、a3 仍为 S-PLUS 且 serial CTRL 无 checkpoint/summary；a4 于 10:33:33+08:00 进入 RUNNING，runner/workload PID 1708057/1708947，不修改 GPU5 a3、holder 或 sealed data |
| 2026-08-28 | 准备但不部署 GPU5 重复 CTRL one-shot guard | fail-closed 条件同时要求 S-PLUS arm PASS、GPU7 a4 健康/完成、a3 runner 与 CTRL child 身份精确匹配；只 SIGTERM runner 以复用其 holder terminal contract。dry-run WAIT、signal_sent=false、66/66 tests PASS；armed 启动仍需用户确认 exact command |
| 2026-08-28 | 用户确认启动 GPU5 重复 CTRL one-shot guard | exact tmux command 于 11:04:58+08:00 执行；guard PID 1807506、armed.lock 与 session 均存在，8/8 启动预检 PASS，跨 20 秒轮询保持 WAIT/signal_sent=false；a3/a4 未受影响并继续推进 |
| 2026-08-28 | S16-3 a4 完成但 faithful no-ridge system 六位置均秩亏 | worker 400.769s、peak reserved 8,668 MiB，无 timeout/OOM；system rank `71/1058/1813/1982/2043/1741 < 2048`，不使用 fallback，S16-3 Gate 不通过且 G-FULL 不解锁 |
| 2026-08-28 | 用户授权不等待 S-PLUS，直接开发独立 S16-3B all-request rank diagnostic | 以 full covariance + 全 train-only request-key Gram 建立 valid-z system 的 PSD 最有利上界，区分 resource subset 不足与结构性 nullspace；不运行 z/solve/ridge/efficacy，不覆盖 A4，89/89 CPU tests PASS，GPU4 启动仍按精确命令确认 |
| 2026-08-28 | 用户确认在 GPU4 启动 S16-3B b1 | exact command 于 16:01:35+08:00 在 tmux `phase16_s3b_gfull_rank_b1_gpu4` 启动；GPU4 admission free 25,525 MiB，identity 与 89/89 tests PASS，runner/workload/Python PID `2843623/2844400/2844403`，test/validation 封存且无自动 retry/resume |
| 2026-08-28 | S16-3B b1 完成计算但 artifact contract FAILED；不自动重跑 | 302,400/302,400 keys 与六位置均完成，elapsed 5,830.412s、peak reserved 8,648 MiB；唯一失败 check 为 position-5 numerical-PSD evidence。positions 0–3 的 proof-eligible full-system rank `74/1216/1938/2033 < 2048` 足以形成 structural blockage 证据；原 b1 FAILED 与 immutable SHA 保留，正式 adjudication 只能另建 CPU-only recovery artifact |
| 2026-08-28 | 用户确认并完成 S16-3B CPU-only recovery c1 | exact command 于 18:07:35+08:00 exit 0；6/6 recovery tests、终态全量 95/95 tests 与 15/15 final checks PASS。positions 0–4 proof-eligible、position 5 明确 ineligible，positions 0–3 rank-blocked，正式分类 `PROVEN_STRUCTURAL_RANK_BLOCKED`；b1 FAILED、无 summary 与五个 frozen input SHA 全部保留，S16-3 Gate 不晋升、S16-4 G-FULL 不解锁 |
| 2026-08-28 | 用户决定将 S16-3 后续主线改为 `GenRecEdit-inspired → GRAM`，方法名冻结为 `G-RIDGE` | faithful G-FULL/S16-3B 历史证据原样保留；只以 train-only、scale-relative condition-targeted spectral ridge 替换 singular no-ridge solve，明确 `faithful_reproduction=false`，建立独立 resource/formal Gates。实现后 Stage16 `104/104` CPU tests PASS，GPU4 resource r1 待精确命令确认 |
| 2026-08-29 | 用户指定并确认在 GPU5 启动 G-RIDGE resource r1；r1 单一 residual Gate BLOCKED 后准备最小 FP64-solve r2 | r1 六位置 full-rank/Cholesky/condition 均通过，但 solver 在 residual/aggregation 前提前 cast FP32，position 0 residual `3.920733e-6 > 1e-6`；合成复现隔离为工程 dtype 错误。r1 artifact/SHA 保留；r2 不改 ridge、数据、threshold 或 Gate，`107/107` tests PASS，等待精确命令确认 |
| 2026-08-29 | 用户确认并完成 GPU5 G-RIDGE resource r2；资源 Gate PASS | r2 elapsed 317.274s、peak reserved 8,668 MiB，六位置 FP64 residual `3.56e-15–5.52e-14`，终态 `PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP`；冻结 formal 单卡 16.859–26.974 GPU·h、minimum free 13,312 MiB、32 GiB disk、7 天 timeout，不把 resource 冒充 formal Gate |
| 2026-08-29 | 用户确认并启动 formal GPU5 f1；完成后 full-compute stability queue 保持独立 | formal 只读 r2 的 47 shards/父证据，执行全量 z/covariance/ridge/aggregation/trigger 与 7,435+512 admission；held events 只在 edit-state freeze 后打开。f1 于 11:50:07+08:00 启动，正式 status 明确 authoritative completion；后续每个 stability cycle 独立重算、独立目录、`affects_scientific_results=false`、失败即停且不 retry。19:06:23 快照为 38,144/59,630 requests，test/validation sealed |
| 2026-08-29 | GPU5 a3 S-PLUS 与 GPU7 a4 matched CTRL 分别完成，duplicate guard PASS | 两臂各完成 12,535 optimizer steps；a3 已完成 S-PLUS arm 保留，guard 仅去重其后续 CTRL 并恢复 holder；a4 exit 0，源 artifact 未改，validation/test 未读 |
| 2026-08-29 | split-pair a1 CPU hash timeout 保留；用户确认独立 a2 后完成 S16-2 | a1 的 5/5 tests PASS，但 8 个 checkpoint 完整 SHA 触发 600 秒 timeout；a2 只把 CPU finalizer 上限提高至 1,800 秒，保持 source/scientific config/full SHA 不变。定向 8/8、Stage16 118/118 tests、paired Gate 与 artifact contract 全 PASS；S16-2 唯一 report 收口为 COMPLETED |
| 2026-08-30 | formal f1 因 code identity drift FAILED；用户确认独立隔离 f2 | f1 的六位置 state 与 7,435+512 admission 已执行，但 Stage17 于运行期修改 `GRAM/src/model/gram.py`，最终 SHA Gate fail-closed；f2 使用独立 runtime、原 GRAM SHA、相同科学配置与新 artifact root，禁止覆盖 f1 |
| 2026-08-30 | f2 在 GPU 前因隔离路径 guard FAILED；按用户授权继续非权威 full-compute repeats | f2 `121/121` tests PASS 后，S16-1 artifact 父目录 symlink 被 `_resolve_repo_path` 拒绝，未加载模型；repeat 1–3 同样在 GPU 前失败。仅对非权威 repeat runtime 修复 allowlisted parent-link 与 snapshot-local config，cycle 4 已真实占用 GPU5；repeat 永不晋升、失败仍新建独立 cycle、检测到新 GPU5 PID 即只让出自身进程 |
| 2026-08-30 | 用户确认继续 formal f3；f2 repeat 让卡后 f3 在 immutable runtime 启动 | 先核验并仅 TERM f2 repeat runner PID `4173017`，cycle 4 以 `INTERRUPTED / exit 143` 封存，GPU5 项目占用释放且外部 PID `1648062` 未修改。f3 与 f2 科学字段完全相同，冻结 f2 status/identity SHA 与修复后的 allowlisted parent-link 映射；主仓/快照 `123/123` tests PASS，config SHA `672739e9…b2ca`。15:00:28 启动后进入 `full_covariance_positions 0/6`，Python PID `70827`、项目显存 `10,634 MiB`；任意终态后再启动隔离 f3 repeats。 |
| 2026-08-31 | G-RIDGE formal f3 PASS；用户要求停止后置 repeat 并开始 S16-4 | f3 于 07:08:55+08:00 exit 0，完成 `302400` requests、六位置 FP64 solve、`7435+512` admission，正式 Gate `PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION`；validation/test sealed。repeat queue 因新增 GPU5 PID `464283` 未启动 cycle，10:02:25 只终止核验过的 runner PID `776933`，终态 `INTERRUPTED/143`，既有 GPU 进程与 formal artifact 均未修改。S16-4 先建立隔离 runtime、统一 evaluator、SHA/资源/命令预检，再等待 GPU 指定与启动确认。 |
| 2026-08-31 | S16-4 CPU input/state/Gate freeze a1 PASS，GPU launch 继续锁定 | a1 校验 8,789-event validation identity、cold/warm/catalog universe、F0/R2 rankings、S-AUX/S-PLUS/CTRL/G-RIDGE 来源 Gate 与所有输入/state SHA；六 arm/control、strict SpecGR 语义、paired-bootstrap 与 Pareto Gate 已冻结。targeted `7/7`、Stage16 full `130/130` tests PASS；无 efficacy metric、无 GPU、test sealed。下一步只实现和冻结 isolated unified evaluator，再给出资源门槛/exact command 请求用户确认。 |
| 2026-08-31 | GPU7 launcher 在 formal root 前因 tmux 权限 exit 3；用户最终改为 GPU0 memory-only admission 并确认启动 | GPU7 未创建 status/formal root，不构成 scientific attempt。GPU0 a1 不检查 utilization，只要求每臂加载前 free≥22,528 MiB；config/manifest SHA 为 `b7a5282f…3d4f` / `5a8ba14f…080e`，定向 11/11 PASS。主机权限下 exact command 成功返回 `STARTED phase16_s4_toys_standalone_gpu0_a1`；初始 free 18,157 MiB，runner 正常等待且不修改外部 PID。 |
| 2026-08-31 | S16-4 GPU4 a7 四臂完成但旧 finalizer artifact contract FAILED | S-AUX、S-PLUS-CTRL、S-PLUS、G-RIDGE 均完成 8,789/8,789 events；CPU finalizer 把预期 runtime schema 硬编码为历史 GPU0 a3 名称，误拒绝有效 GPU4 a7 manifest。a7 终态 `FAILED / ARTIFACT_CONTRACT_FAILED`、四臂 predictions、status/log 与完整 SHA 原样保留；无 automatic retry，后置 repeat 未启动，GPU4 无重复实验占用。 |
| 2026-09-01 | 修复 S16-4 finalizer 并完成独立 CPU-only a8 recovery | 通用 runtime identity validator 取代设备/attempt 字面量；补齐预注册 Holm correction 与 exact paired binary test。a8 只读 12 个冻结 a7 输入 SHA，不重跑 GPU、不覆盖 a7；targeted 24/24、Stage16 154/154 tests 与重算/local/source SHA exact，终态 `PASS_S16_4_TOYS_CPU_RECOVERY_FINALIZATION`。 |
| 2026-09-01 | 审计发现 a8 R2 comparator identity mismatch；完成独立 CPU-only a9 correction | 计划冻结的 R2 是 Stage13 `unconditional portfolio@2`，a8 却沿用了 P0 `r2_top50`。a9 只读冻结 a8/P0/P6/cold-manifest/code SHA，对 8,789 个事件重建 portfolio@2；candidate/F0 mismatch 均为 0，与 P6 aggregate 最大误差 `5.55e-17`，targeted 29/29、Stage16 159/159 tests、9 个本地与 11 个来源 SHA 全 PASS；未重跑 GPU、a7/a8 不覆盖。 |
| 2026-09-01 | 纠正 S16-4 方向：停止 G-RIDGE 组合，S-AUX 进入冻结 Beauty Gate | S-AUX 对 F0 cold H@50 显著提升 `+0.049920`；相对正确 portfolio@2，cold H@50 高 `+0.030456`，但 cold NDCG、warm/overall 与成本更差，双方互不严格支配，故 S-AUX=`PASS_STANDALONE_PARETO`。G-RIDGE 对 F0 明确负增益，并且对 F0/正确 R2/S-AUX unique cold hits 均为 0，原 G-RIDGE 组合主线停止。先验证 S-AUX Beauty 迁移；仅在方向一致时再考虑 portfolio@2-default + S-AUX conditional route。 |
