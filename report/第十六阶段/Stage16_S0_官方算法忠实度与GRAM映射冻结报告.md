# Stage16 S0 官方算法忠实度与 GRAM 映射冻结报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-23
- Verification Status: ANALYZED（确定性审计与 bridge tests 已通过；尚未做独立复跑）
- Version Label: phase16_s0_fidelity_contract_v1

## 1. 阶段结论

S16-0 已完成，唯一 Gate 为：

> `PASS_S16_0_FIDELITY_CONTRACT`

本阶段验证了两份本地官方源码均处于计划冻结 commit、worktree clean，并将 SpecGR-Aux、SpecGR++ 与 GenRecEdit 的 23 个核心函数级组件冻结为 F0/F1 映射。18/18 个语义 bridge checks 和 8/8 个单元测试通过，没有发现必须用自创简化替代的核心组件。

这项 PASS 只表示“算法语义可以映射且映射定义已在 efficacy 前冻结”，不表示 S-AUX、S-PLUS 或 G-FULL 已实现，更不表示方法有效。正式实现仍分别属于 S16-2 与 S16-3。

## 2. 执行与资源

| 项 | 实测 |
|---|---|
| experiment / attempt | `GRAM_PHASE16_S0_FIDELITY_CONTRACT` / `s16_s0_a1` |
| exact command | `bash experiment/phase16/run_stage16_s0_fidelity_contract.sh` |
| working directory | `/mnt/18T/jiangtangyunzhi/projects/recomm` |
| 开始 / 完成 | `2026-08-23T20:07:43+08:00` / `2026-08-23T20:07:46+08:00` |
| exit code | `0` |
| hard timeout | `300 s`，未触发 |
| GPU | `0`；未查询、未占用、未修改任何 GPU 进程 |
| 网络 / 下载 | 未使用网络；下载 `0 bytes` |
| recommendation data | 未打开 |
| test | `test_read=false` |
| automatic retry | `false`；没有 retry |

Stage15 Beauty B2 的既有 status 在本阶段开始前呈现 `status=running` 但 `process_alive=false`、心跳停止的状态不一致。本阶段只读观察，未刷新 status、未终止、未恢复，也未修改 Stage15 任何进程或 artifact。

## 3. 官方源码冻结

| 方法 | remote | commit | worktree | license |
|---|---|---|---|---|
| SpecGR | `https://github.com/Jamesding000/SpecGR.git` | `f0ded8884b1df97b5f0599d4ec300bb20b5d1eff` | clean | `NO_LICENSE_FILE_AT_HEAD` |
| GenRecEdit | `https://github.com/Starrylay/GenRecEdit.git` | `e6878d9c7c6e57479e840ccb8c045b11a2bd69b5` | clean | `NO_LICENSE_FILE_AT_HEAD` |

因此继续沿用 clean-room 边界：第三方仓库只用于本地逐行审计，不复制、提交或再分发第三方源码、checkpoint、LFS blob；Stage16 adapter 在项目代码中独立实现。

## 4. Function-level fidelity matrix 摘要

23 个主路径组件全部为 F0/F1：F0 保留官方逻辑、目标或决策规则；F1 仅承担 TIGER/RQ-VAE fixed-width SID 到 GRAM variable lexical path、模块 API、device 或 batching 的必要接口变化。S16-0 没有把任何 F2/F3 组件放入 faithful 主表。

| 方法 | 冻结组件 | 主要 F0 | 主要 F1 |
|---|---:|---|---|
| SpecGR-Aux / shared loop | 8 | UniSRec 架构、无放回 drafting、严格 `score > threshold` | content catalog 接口、variable-prefix score、trie-guided redraft、adaptive depth、strict-item fallback |
| SpecGR++ | 3 | contrastive+generative pretrain、ranking+generative finetune 的加权目标 | GRAM encoder self-drafting 与冻结 index |
| GenRecEdit | 12 | `0.3` scope、Adam+cosine、active/satisfied、absolute z norm clip、valid-z、closed-form delta、additive aggregation | full-target lexical requests、cache probe competitor set、position covariance/key、variable-position One-One trigger |

完整逐函数证据包含官方文件、匹配行号、F0/F1 分类、GRAM implementation target、语义变化和对应 bridge check，保存在：

- `artifacts/phase16/s0_fidelity_contract/fidelity_matrix.json`

所有条目当前统一标记为 `SEMANTICS_FROZEN_IMPLEMENTATION_DEFERRED_TO_S16_2_OR_S16_3`。这避免把 Stage15 pilot adapter 误写为 faithful implementation。

## 5. SpecGR 冻结语义

### 5.1 S-AUX

- drafter 必须使用官方 UniSRec 的 MoE adaptor、序列编码与 CE 训练目标；Stage15 的两层轻量 Transformer 不得进入 faithful 主路径。
- 训练 label 只能来自合法 train-only warm transitions；cold catalog item 仅通过冻结 content representation 进入 full-catalog retrieval。
- 官方 S-AUX 推理默认冻结为 `draft_size=50`、`threshold=-1.8`、`num_beams=20`。参数只能在 S16-1 冻结的 train-only internal-dev 空间按预注册规则处理，不能由 validation efficacy 选择。

### 5.2 shared draft→verify→redraft→exit

- drafting 为 top-k 后将已选项置为 `-inf`，即同一候选不重复 draft/verify。
- target-aware score 保留“可识别 prefix 上 token log-probability 的均值”；GRAM F1 只把 fixed width 换成每条 lexical path 的 score length。
- acceptance 保留严格大于：`score == threshold` 必须拒绝。
- guided redraft 只允许与当前 verifier beam prefix 一致的 catalog paths；complete path 不再扩展。
- adaptive exit 保留“已接受数量达到 K 或最大深度耗尽”；最大深度从固定 `num_digits` 映射为冻结 lexical trie 的最大深度。
- accepted drafts 不足时使用 GRAM constrained beam fallback；unknown、collision、duplicate 均 hard-fail。

### 5.3 S-PLUS

- self-drafter 使用 GRAM encoder mean-pooling/projection 后的 normalized KNN index。
- 两阶段目标保持官方结构：pretrain 为 `lambda_emb * contrastive_loss + lambda_gen * generative_loss`，finetune 为 `lambda_emb * ranking_loss + lambda_gen * generative_loss`。
- 官方基础配置的 `temperature=0.07`、`lambda_emb=6.0`、`lambda_gen=1.0` 已记录；官方 S-PLUS 推理 CLI 默认 `draft_size=20`、`threshold=-1.8`、`num_beams=50`。
- `S-PLUS-CTRL` 是 Stage16 的项目级因果对照，不伪装成官方组件；其预算对齐将在 S16-1/S16-2 冻结。

## 6. GenRecEdit 冻结语义

### 6.1 request、z 与 cache lifecycle

- 每个冻结 cold target 的每个 lexical token 都形成 prefix→next-token request；EOS/padding 不形成 edit target。
- `z` 使用 Adam，官方 dataclass 参数为 `v_lr=0.5`、`v_num_grad_steps=30`、`v_weight_decay=0.2`，scheduler 为 cosine，`eta_min=0.01`。
- 新优化 z 的 satisfied 判定是 target token argmax；源码没有对新 z 再套 `0.3` global success Gate。
- `0.3` 只在 cached-z probe 中生效，且必须同时满足 target 在竞争集合内为 argmax、full-vocabulary target probability `>0.3`。
- norm clip 按源码实际实现冻结为 `z_vector_max=8000` 的 absolute delta norm clip；不把声明但未用于该路径的 `clamp_norm_factor` 擅自代入。

### 6.2 fixed-width → variable lexical probability bridge

本阶段在任何 efficacy 之前冻结如下定义：

1. fixed-width synthetic SID 使用官方 `256 * position + code + 1` token offset 复核；
2. target 是否为 argmax，只与当前位置的合法竞争 token 比较；
3. GRAM 中该竞争集合替换为 frozen lexical trie 的 legal next children；
4. `0.3` cache threshold 的 target probability 仍来自 full-vocabulary softmax，不改为 legal-set 重新归一化概率；
5. 新 z optimizer success 只使用 argmax，不使用 `0.3`；
6. EOS、padding、complete path 和 constrained beam 的 dead row 不激活 deltaW。

该定义同时保留了官方“position-local competitor set + full-vocabulary probability”的拆分语义，避免沿用 Stage15 的 legal-set probability 或全局 z-success Gate。

### 6.3 covariance、key、deltaW 与 trigger

- covariance 为每个 position/layer 的二阶矩 `E[xxᵀ]`；faithful primary 的 row 数固定为 `min(400000, 全部合法 train-only covariance rows)`。
- key 从目标 decoder FFN 输入的最后 lexical position 抽取。
- 只有非空 z/delta 进入 valid-z solve；failed row 必须保留计数，不按 efficacy 丢弃。
- closed-form update 保留 `ΔW = Δ Kᵀ (K Kᵀ + λC)⁻¹`；实现允许用代数等价的 linear solve 替代显式 inverse，但不能改变方程或 λ。
- 同一参数收到多个 position update 时 additive aggregation，不做 post-hoc weighting。
- generation trigger 按当前 live lexical position 选择 layer/update；complete/EOS/pad/dead rows 不编辑，退出 context 后 frozen base 必须 exact restore。

### 6.4 官方默认值不一致

GenRecEdit dataclass 默认 `cov_lambda=10000`、`number_knowledge=5`，但官方 `Scripts/edit.sh` 实际设置为 `cov_lambda=1000`、`number_knowledge=10`、`pos2layer=[0,1,2,3]`。两套值均已冻结并披露；S16-1 必须根据“官方实际主入口优先、README/代码冲突显式记录”的规则确定 Stage16 exact config，不得依据 efficacy 选择。

SpecGR 两个变体的推理默认也不同：S-AUX 是 `draft_size=50 / beams=20`，S-PLUS 是 `draft_size=20 / beams=50`。后续不得混用或以一个统一的“SpecGR 默认”覆盖两者。

## 7. Bridge tests、异常与可复现性

| 检查族 | 结果 |
|---|---:|
| SpecGR content-only cold access、无放回 draft、fixed-width score parity、strict acceptance、trie redraft、adaptive exit、unique fallback | 7/7 PASS |
| SpecGR++ normalized KNN、weighted two-stage objective | 2/2 PASS |
| GenRecEdit fixed-width probe parity、threshold scope、optimizer、norm clip、second moment、variable position routing、valid-z、closed-form parity、aggregation | 9/9 PASS |
| 独立 `unittest` cases | 8/8 PASS |

运行中没有 crash、timeout、stall、network fallback 或自动 retry。所有源码 assertion、参数 assertion、commit/remote/worktree assertion 都一次通过。

Reproducibility verdict：`ANALYZED / NOT_INDEPENDENTLY_RERUN`。本运行是确定性 CPU 审计，exact command、输入 SHA256、代码 SHA256 和 config SHA256 已保存，但本阶段未进行第二次独立复跑，因此不标 `VERIFIED`。

统计指标、effect size、CI 与多重比较在 S16-0 不适用；本阶段不产生推荐 efficacy 结论。

## 8. Artifact 索引

主 artifact 目录：`artifacts/phase16/s0_fidelity_contract/`

| 文件 | 用途 |
|---|---|
| `status.json` | lifecycle、exit code、test/network/GPU 状态 |
| `summary.json` | Gate 与检查计数 |
| `fidelity_matrix.json` | 23 个 function-level F0/F1 映射与官方证据行 |
| `official_parameters.json` | 19 个官方默认/入口参数及冲突 scope |
| `bridge_test_summary.json` | fixed-width/variable lexical 定义与 18 项结果 |
| `source_manifest.json` | remote、commit、worktree、license |
| `input_file_sha256.json` | 本次实际审计输入 SHA256 |
| `code_sha256.json` | runner、audit、bridge、tests SHA256 |
| `open_file_manifest.json` | 打开文件清单与 `test_read=false` |
| `data_provenance.json` | 无 recommendation data、无第三方代码复制证明 |
| `resource_summary.json` | CPU-only、GPU/network/download 均为 0 |
| `command_manifest.json` | exact command、工作目录、timeout、retry policy |

## 9. Gate 与下一唯一步骤

S16-0 完成定义全部满足：

- pinned source 与官方函数证据 PASS；
- F0/F1/F2/F3 主表资格规则 PASS；
- fixed-width synthetic SID 与 variable lexical trie bridge PASS；
- SpecGR/GenRecEdit 关键默认、实现分支和不一致已冻结；
- test sealed、CPU-only、no network、no Stage15 interference PASS。

下一唯一步骤为 S16-1：数据 SHA、train-only internal-dev、item-disjoint pseudo-cold、test guard、完整样本/target/covariance 计数，以及 CPU/小 GPU resource preflight。S16-1 可以准备小实验，但不会启动任何大实验；首个大实验仍需在资源实测后向用户报告并由用户指定 GPU。
