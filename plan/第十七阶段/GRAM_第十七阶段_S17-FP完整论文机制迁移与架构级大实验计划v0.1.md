# GRAM 第十七阶段 S17-FP：完整论文机制迁移与架构级大实验计划 v0.1

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent (plan mode)`
- Created：2026-08-31
- Status：`PLAN_ACTIVE / T5_ATTEMPT_005_PASS / CUDA_ENV_ATTEMPT_004_PASS / TOKENIZER_ATTEMPT_010_GPU0_PASS / GPU_ALLOCATION_CONFIGURED_NOT_LAUNCHED / NO_EFFECT_EXPERIMENT_STARTED`
- Scope：正常场景 GRAM 推荐提点；不包含 cold-start，不读取或修改 Stage16
- Parent Evidence：S17-0～S17-4、S17-2R 已完成结果；全部保留，不重跑
- Primary Direction：Full LATTE native parity → `GRAM-LATTE-Full`
- Secondary Direction：repo-parity SETRec + paper-faithful SETRec → `GRAM-SETRec-Paper-Full`
- Conditional Synthesis：`Latent-Conditioned Set-GRAM`，仅在两个 standalone 方法均独立提点后进入
- Protected Data：Beauty/Toys official test、Sports 继续封存；D1/D2 在配置冻结前不得读取
- GPU Authorization：本计划允许 CPU 准备和小实验；任何大实验启动前仍须向研究者单独申请 GPU 数量与具体物理卡
- Runtime Policy：预计或实测大于 10 分钟的任务一律后台；用户只通过 `artifacts/phase17/status/` 观察，不要求 agent 实时监看或主动轮询
- GPU1 Policy：GPU1 默认不释放、不用于小实验；若大实验获研究者明确授权使用 GPU1，任意科学终态后必须恢复交接前的重复轮并核验占用

## 0. 决策与历史关系

### 0.1 本计划的核心决策

Stage17 从“同时筛选多个轻量论文 hook”切换为：

> 少选方向、完整移植、充分训练；先验证单方法，再做架构级融合。

本计划不继续修补现有 3k-user 独立缩放模型，不新增轻量 rerank，也不回到第 6/9/10/11/13 阶段已经反复验证的候选后处理路线。

### 0.2 与 S17-2R 的关系

`S17-2R` 已封存为 `COMPLETED_NO_R3_CANDIDATE`。其结果继续有效，但结论边界限定为：

- 3,000-user Toys D0；
- 独立实现；
- 约 5.36M–8.12M 参数；
- 最大 10 epochs，实际 LATTE/PSID 只完成 5 epochs，SETRec 只完成 3/6 epochs；
- train-only internal dev 只有 300 users；
- 不是官方代码运行或完整论文训练协议。

S17-2R 不重跑。已有弱正信号用于选择 full-port 优先级：

| Family | 本地 matched delta | 机制状态 | 本计划处理 |
|---|---:|---|---|
| LATTE beam-200 | `ΔNDCG@10=+0.001018`，3/3 cohort 为正 | multi-path PASS；CI 跨 0 | 第一主线，使用官方实现与完整训练预算重做 |
| SETRec-style | `ΔNDCG@10=+0.001422`，`ΔHit@10=+0.003`，3/3 cohort 为正 | `set_token_recovery=0` | 第二主线；现有 discrete proxy 不再称 Full SETRec |
| Gryphon | 明确负向 | item scorer 排序机制 FAIL | 不进入本计划 |
| DiffGRM | 明确负向 | 速度机制 PASS、accuracy 负向 | 不进入本计划 |

### 0.3 对旧总计划的覆盖范围

本文件是 Stage17 后续 full-port 的权威执行计划，覆盖旧总计划中下列已过期规则：

1. “lite 先行、full 仅作远期 P2”的顺序；
2. 用 3k-user/短 early-stop 结果直接关闭完整论文方法；
3. 所有大实验完成后都在所有获配 GPU 上持续重复的规则。

只有 GPU1 继承强制重复轮恢复要求；其他 GPU 在科学任务终态后正常释放，除非研究者另有明确安排。

历史结果、attempt ledger、数据防泄漏、统一 evaluator、status schema 与 official test/Sports 封存规则不被覆盖。

## 1. 研究问题与假设

### RQ1：Full LATTE 在当前正常推荐数据上是否可复现正增益？

在相同 semantic IDs、数据、训练预算和 item evaluator 下，官方 LATTE 是否优于官方 PSID？

### RQ2：完整 LATTE 机制能否真正提高强 GRAM？

保留 GRAM 的多 passage、文本/协同输入和 FiD encoder，把 identifier 与 decoding tree 完整替换为 conflict-free SID + latent-conditioned forest，是否能同时超过：

- matched `GRAM-PSID-Full`；
- fresh matched `GRAM-B0/Continue` 强基线？

### RQ3：Full SETRec 的完整连续 set-identifier 机制能否提高强 GRAM？

完整引入 CF tokenizer、semantic AE、多维连续 token、sparse history attention、query-guided simultaneous generation 和 full-catalog grounding，是否优于相同 token/capacity 的 ordered control，并超过强 GRAM？

### RQ4：两种机制是否存在可叠加的用户级互补性？

若 `GRAM-LATTE-Full` 与 `GRAM-SETRec-Full` 均独立为正，用户级 gain/loss 是否互补，且 latent-conditioned set generation 是否超过两个 parent 中的最好者？

## 2. “完整迁移”的操作定义

### 2.1 Native faithful arm

只有满足以下条件才可命名为 `Native-Full`：

- 固定官方 repository commit 与 license；
- 直接执行允许复用的官方实现，或在许可不足时建立逐函数 clean-room fidelity matrix；
- 保留 paper/official config 的 tokenizer、identifier、training target、decoder、inference 和 aggregation 主路径；
- 只做数据路径、fold adapter、item evaluator、device/batch 等必要接口变化；
- 所有必要变化在看 D0 efficacy 前冻结。

### 2.2 GRAM full-port arm

只有同时满足以下条件才可命名为 `*-GRAM-Full`：

- GRAM 的正常场景输入语义仍存在：历史 item passages、metadata、协同/相似 item 信息与 FiD 聚合；
- 论文的核心 identifier、training 和 decoding/grounding 机制整体进入模型，而不是只加 auxiliary loss、root token 或末端 rerank；
- 有同 identifier、同训练预算、同 evaluator 的 matched control；
- 方法特有机制指标实际激活；
- 输出是统一 catalog item ranking，可与 GRAM 在同一用户上 paired 比较。

### 2.3 禁止事项

- 禁止把现有 `s2r_architectures.py` / `s2r_parallel_architectures.py` 的缩放实现改名为官方 Full；
- 禁止只增加 deterministic root 后称 LATTE；
- 禁止使用 discrete token permutation loss 代替 SETRec 的连续 token、query generation 和 grounding 后仍称 Full SETRec；
- 禁止只与很弱的 native control 比较而不报告相对强 GRAM 的绝对差；
- 禁止使用 paper 数字替代本地 prediction；
- 禁止在同一 D0 上无限扫描 latent 数、beam、aggregation、loss weight 或 SETRec token 数追点。

## 3. 数据、切分与正常推荐定义

### 3.1 数据漏斗

| 数据视图 | 用途 | 是否可调参 |
|---|---|---|
| Toys D0 full | full-port discovery、完整训练、一次正式外部评估 | 只允许计划内官方默认和一次事先冻结的 fidelity recovery |
| Toys D1 | 独立准入 | 否 |
| Beauty D1 | 跨域独立准入 | 否 |
| Toys/Beauty D2 | 3-seed 稳健性确认 | 否 |
| Beauty/Toys official test | 封存 | 禁止读取 |
| Sports | 封存 | 需研究者另行授权 |

所有数据沿用 Stage17 rolling-origin shadow contract：

- 模型输入只能看到 train prefix；
- shadow validation target 不进入训练、tokenizer fit、graph、CF model、semantic-ID fit 或 negative pool；
- guard item 不参与指标；
- 下游 runner 只读投影后的 shadow 文件，不打开原始 monolithic sequence；
- official `[-2:]` 不进入任何新实验；
- 所有 manifest、用户数和 SHA256 在 FP0 冻结。

### 3.2 正常场景

- 不构造 cold50；
- 不屏蔽目标 item 的历史 train-only transductive 信息；
- overall NDCG@10 为主结果；
- 同时报告 head/mid/tail、短/中/长历史和 memorization/generalization；
- CF/SASRec、semantic tokenizer 和 collision resolution 只使用 fold train 数据。

### 3.3 训练样本

- 使用完整 Toys D0 train prefix 的 rolling next-item transitions；
- `max_history_items=20`；
- early-stop cohort 从 train prefix 内再留最后一个已观察训练 item，不能使用 D0 external target；
- external D0 target 只在所有本 family best checkpoints 冻结后打开一次；
- native treatment/control 共享同一用户、SID cache、batch order、seed 和评估 cohort；
- GRAM G0/G1/G2 共享同一初始 T5 权重、数据顺序和训练机会。

## 4. FP0：来源、许可、配置与 fidelity 冻结（CPU）

### 4.1 LATTE 来源

- Paper：<https://arxiv.org/abs/2605.06331>
- Official repository：<https://github.com/hyp1231/Latte>
- License：MIT；复制或修改时保留 LICENSE 与 attribution
- commit：在 FP0 本地 clone/fetch 后冻结，不以浮动 `main` 作为正式复现身份

必须冻结的官方配置：

| 项 | Official primary |
|---|---|
| semantic extractor | `sentence-transformers/sentence-t5-base`，768 dim，PCA 192 |
| VQ | `rqkmeans`，3 codebooks × 256 |
| conflict policy | PSID conflict-free semantic reassignment；不得使用本地 collision suffix 代替 |
| latent tokens | 8 |
| aggregation | `agg_max` primary；`agg_sum` 只作预注册 inference ablation |
| history | max 20 items，1 hashed user token |
| model | encoder/decoder 4/4 layers，`d_model=128`，`d_ff=1024`，6 heads，`d_kv=64`，dropout 0.1 |
| optimization | lr `3e-3`，weight decay `0.05`，batch 256，eval batch 128，warmup 10,000，clip 1.0 |
| training | max 150 epochs，patience 50；base YAML 每 epoch eval，quick-start wrapper 覆盖为每 3 epoch |
| generation | train/dev beam 50；final method-native beam 500；top-50 item ranking |

项目 seed 固定为 2023，用于与 GRAM 和既有 Stage17 数据一致；这项偏离官方默认 seed 2024，必须标为接口级实验协议变化，不改变方法语义。

SentenceT5 浮动名称在 FP0 进一步固定为 Hugging Face revision
`fc5d4628481afbbaaacd7af6bb07cf9d3865f781`（Apache-2.0）。模型下载、离线
load/encode 验证和逐文件 SHA 均由 CPU-only 后台任务完成；模型权重缓存不构成效果实验。

full Toys adapter 已通过实数审计：12,833 个 train-prefix users、56,421 个 rolling
examples、1,283 个 internal-dev users、11,924 个 catalog items。完整 train prefix 出现
11,182 个 item；把 internal-dev 用户的末位从 supervised train 与 RQ centroid fit 中移除后，
tokenizer-fit mask 为 11,138 个 item。其余 catalog item 只做 target-independent transform/
identifier assignment，不用于 PCA/RQ 参数拟合。

### 4.2 SETRec 来源

- Paper：<https://arxiv.org/abs/2502.10833>
- Official repository：<https://github.com/Linxyhaha/SETRec>
- Repository 仅声明 `NUS © NExT++`，没有标准开源 license；正式项目代码不得复制，默认走 clean-room implementation
- FP0 可只读核对公开论文/代码语义，但必须输出 source manifest 与 function-level fidelity matrix

官方 Toys T5 主配置至少冻结：

- learning rate `1e-3`；
- semantic token count `n_sem=4`；
- AE loss weight `alpha=0.7`；
- pre-trained SASRec CF tokenizer；
- SentenceT5/官方语义表示路径；
- continuous CF + semantic tokens；
- sparse intra-item attention；
- learnable query-guided simultaneous generation；
- per-dimension token corpus grounding 和 full-catalog item score。

其余 epoch、batch、optimizer、scheduler 和 exact attention/grounding 公式在 FP0 从冻结源码/论文提取后写入机器可读 config；不得依据 D0 efficacy 补值。

FP0 已冻结官方 Toys T5 脚本的实际设置：30 epochs、global batch 512、microbatch 128、4-GPU torchrun、FP16、AdamW、cosine、warmup 100、eval/save 每 200 steps、early-stop patience 10、seed 42、history 50。Stage17 matched arm 使用 seed 2023、history 20 和 train-prefix internal dev，并保持有效 batch 512；这些均作为显式接口变化，不伪装成官方原设置。

#### 4.2.1 SETRec paper–repository gap

论文 3.2.2 要求 user-history sparse mask：同一 item identifier 内不同维度 token 互相不可见，但可见之前 items。固定 public T5 commit `2ed9a75ad1ad3784c61bba3c68cbedbe3cfce2d7` 的 encoder 路径只让同一 item 的 token 共享 position id，普通 attention mask 仍为全可见；其显式 identity mask 用在 decoder queries。

因此后续不得把两者混成一个 `Native-SETRec-Full`：

- `Repo-Parity`：复现 public repository 的 shared-position encoder + independent-query mask；
- `Paper-Faithful`：clean-room 实现论文 sparse history visibility + independent-query mask；
- 两臂共享其余 continuous tokenizer、query、grounding、训练预算和 evaluator。

### 4.3 FP0 产物与 Gate

- `artifacts/phase17/fullport/manifests/latte_source_manifest.json`
- `artifacts/phase17/fullport/manifests/latte_fidelity_matrix.json`
- `artifacts/phase17/fullport/manifests/setrec_source_manifest.json`
- `artifacts/phase17/fullport/manifests/setrec_fidelity_matrix.json`
- `artifacts/phase17/fullport/manifests/data_manifest.json`
- `artifacts/phase17/fullport/config/` 下冻结的 resolved configs
- `artifacts/phase17/fullport/fp0/native_data_adapter/attempt_001/summary.json`
- `artifacts/phase17/status/s17_fp0_native_env_setup.status.json`
- `artifacts/phase17/status/s17_fp0_sentence_t5_cache.status.json`
- `artifacts/phase17/status/s17_fp0_tokenizer_bounded_profile.status.json`
- Stage17 regression、leakage、SID collision 和 official-config parity tests

Gate：`PASS_S17_FP0_SOURCE_DATA_FIDELITY_FREEZE`。FP0 不产生效果结论。

## 5. FP1：Full LATTE native parity

### 5.1 Arms

| Arm | 定义 | 正确对照 |
|---|---|---|
| N0 `Native-PSID` | 官方 conflict-free PSID、rqkmeans、官方模型/预算 | — |
| N1 `Native-LATTE` | N0 + 8 latent tokens + 随机 path training + forest decoding + item aggregation | N0 |

### 5.2 训练与推理

- 直接使用固定 commit 的官方实现和当前 Toys D0 adapter；
- 两臂共享 semantic embeddings、PCA、rqkmeans centers、conflict-free SID、train examples 和 seed；
- N0/N1 均从头训练，不加载现有 3k checkpoint；
- 使用第 4.1 节官方 max 150 epochs / patience 50；
- 每个 target exposure 的 latent token 均匀随机采样，不得按 item/user 固定 hash；
- best checkpoint 只由 train-prefix internal dev NDCG@10 选择；
- D0 external target 只评一次；
- primary inference：beam 500、`agg_max`、top-50；
- compute diagnostic：beam 50；
- `agg_sum` 只作 frozen-checkpoint inference ablation，不参与选 checkpoint。

### 5.3 Gate

Native parity 通过需同时满足：

1. `N1-N0 ΔNDCG@10 > 0`；
2. paired 95% CI 下界 `> 0`；
3. `ΔHit@10 >= 0`；
4. multi-path item rate `> 0`、latent usage 不塌缩到单 root；
5. item ranking 使用聚合后概率，valid item rate = 1；
6. 无 target leakage、collision alias 或 evaluator drift。

Native parity 未通过时不自动终止 GRAM port：若 fidelity/训练完全成立，则允许继续 FP2 一次，因为 GRAM 的富输入可能改变机制收益；但必须把 native negative 作为风险披露，不能宣称已复现论文增益。

## 6. FP2：GRAM-LATTE-Full 主实验

### 6.1 Arms

| Arm | 输入/identifier/decoder | 角色 |
|---|---|---|
| G0 `GRAM-B0-Fresh` | 原 GRAM FiD + native lexical ID + Trie AR decoder | 强绝对基线和 matched training control |
| G1 `GRAM-PSID-Full` | 保留 GRAM passages/FiD；历史 ID linking 与 target 全部替换为同一 conflict-free SID；单树 AR | LATTE 的直接 causal control |
| G2 `GRAM-LATTE-Full` | G1 + 8 latent roots + uniform random path training + latent-conditioned forest + item aggregation | 主 treatment |

G0/G1/G2 都从同一 `t5-small` 初始化和相同 seed fresh training。不得将历史 epoch-30 checkpoint 与 fresh new-ID arm 直接混作主因果对照。

### 6.2 GRAM 保留与替换边界

保留：

- GRAM 的历史 item passage 构造；
- title/category/metadata 文本；
- train-only similar-item/CF 信息；
- FiD 多 passage encoder；
- unified item evaluator 与正常场景数据。

替换：

- G1/G2 不再使用 native lexical target path；
- passage 中承担 item identity/linking 的字段改为 frozen conflict-free semantic ID；
- decoder vocabulary、constrained tree 和 item resolver 改为 PSID/latent forest；
- G2 target sequence 为 `<latent_k> + SID(item) + EOS`；
- G2 同一 item 的多条 path 在 item 层聚合，不以 path 当不同 item。

### 6.3 训练设置

| 设置 | G0/G1/G2 |
|---|---|
| backbone | `t5-small`，三臂同初始化 |
| seed | 2023 |
| max history | 20 |
| epochs | max 50，minimum 20 |
| internal eval | 每 5 epochs |
| early stop | 3 次 internal eval 无 `>=0.0001` NDCG@10 改善 |
| train microbatch | 16 |
| gradient accumulation | 8；effective batch 128 |
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `0.01` |
| warmup | 5% total steps |
| gradient clip | 1.0 |
| precision | FP32 primary；若 profile 证明 AMP parity 才可统一切换三臂 |
| checkpoint selection | train-prefix internal dev NDCG@10 |
| external D0 evaluation | best checkpoint 冻结后一次 |

如果 official implementation 与项目 T5 tokenizer 需要新增 SID/latent embeddings，G1/G2 必须共享新 token 初始化 seed；G0 记录新增参数差，并以 G1 作为主 causal control。

### 6.4 推理设置

所有臂同时报告：

- standard：beam 50、top-50；
- compute-matched：beam 500、top-50；
- G2 primary：beam 500、`agg_max`；
- G2 `agg_sum`：仅作预注册 frozen-checkpoint inference ablation；
- G0/G1 beam 500 用于排除“只是更大 beam”；
- 不依据 D0 结果继续增加 beam 或改 aggregation。

### 6.5 FP2 强 Gate

G2 进入独立 fold 必须同时满足：

1. `G2-G1 ΔNDCG@10 >= +0.0015`；
2. `G2-G1` paired 95% CI lower `> 0`；
3. `G2-G0 ΔNDCG@10 >= +0.0015`，比较使用预注册 primary 推理口径；
4. `G2-G0 ΔHit@10 >= 0`；
5. 任一 head/mid/tail 或历史长度大组不出现 NDCG@10 `<= -0.003` 的灾难性退化；
6. latent usage、multi-path coverage、item aggregation gain 与 tree-coupling reduction 全部按预期变化；
7. legal/valid item rate = 1，official test/Sports/D1 未读。

若 `0 < ΔNDCG@10 < 0.0015`，记录为 `WEAK_POSITIVE_FULLPORT`，不进入融合；不得改阈值追认。

## 7. FP3：Repo-Parity 与 Paper-Faithful SETRec 独立大实验

FP3 的实现准备可与 FP1/FP2 并行，但正式 D0 efficacy 只有在 FP0 fidelity、完整 tokenizer 训练和资源 profile PASS 后才能启动。

### 7.1 机制实现

完整 SETRec 必须包含：

1. fold-train-only SASRec item CF embeddings + linear projection；
2. SentenceT5 item semantic representation；
3. AE 生成 `n_sem=4` 个连续 semantic embeddings；
4. 每个 item 的 identifier 为 `{z_CF, z_S1, z_S2, z_S3, z_S4}`；
5. paper-faithful arm 中 item 内 token 互相不可见但可访问之前 items；repo-parity arm 使用同 item shared position id；
6. 五个 learnable query vectors 并行生成对应信息维度；
7. 每维 token corpus 作为 grounding head；
8. 跨维 grounding score 组合为 full-catalog item ranking；
9. 不使用当前 discrete-code permutation proxy 作为 Full SETRec。

### 7.2 Arms

| Arm | 定义 | 角色 |
|---|---|---|
| S0 `SETRec-Ordered-Control` | 相同 CF/semantic tokenizers、容量和 grounding；ordered position + sequential query | 总 causal control |
| S1R `SETRec-Repo-Parity` | public T5 commit 的 continuous identifier、shared item position、independent queries、grounding | repository reproduction |
| S1P `SETRec-Paper-Faithful` | S1R 其余设置不变，history 改为论文 sparse visibility | paper mechanism treatment |
| S2 `GRAM-SETRec-Paper-Full` | GRAM 多 passage/FiD 上下文 + S1P sparse/query/grounding 主路径 | GRAM treatment |

S0/S1R/S1P 使用同一模型容量和训练预算；S1R/S1P 除 history attention contract 外完全匹配。S2 与 fresh G0 共享数据和 item evaluator。若为 S2 增加 GRAM encoder 参数，报告 capacity 和 compute，不能把参数差隐藏成机制增益。

### 7.3 Primary setting 与 Gate

- Toys T5 primary：lr `1e-3`、`n_sem=4`、`alpha=0.7`；
- exact epochs/batches/optimizer/scheduler 由 FP0 官方 config freeze 给出；
- best checkpoint 只由 train-prefix internal dev 选择；
- full-catalog grounding，top-50；不通过 beam 做候选截断；
- S1R/S1P/S2 的 query/token recovery、grounding target rank、valid item 和 latency 全部报告。

S2 晋级要求：

1. S1R 相对 S0、S1P 相对 S1R 的结果与机制诊断全部报告；S1P 相对 S0 `ΔNDCG@10 > 0` 且机制激活；
2. S2 相对 S0 `ΔNDCG@10 >= +0.0015`、paired CI lower `>0`；
3. S2 相对 fresh G0 `ΔNDCG@10 >= +0.0015`；
4. `ΔHit@10 >= 0`；
5. continuous token、sparse attention、query generation、grounding 均非退化路径；
6. 无 subgroup 灾难性退化或数据泄漏。

## 8. FP4：条件式架构融合

只有 G2 与 S2 都通过各自强 Gate 才进入 FP4。一个失败时，不用另一个模块或 PCRF 掩盖其失败。

### 8.1 融合方法

临时工作名：`LCSet-GRAM`（Latent-Conditioned Set-GRAM）。

- 使用 LATTE 的 8 个用户条件化 latent modes；
- 每个 latent mode 条件化 SETRec 的 CF/semantic query vectors；
- 每个 mode 产生完整多维 set representation；
- 在 full catalog grounding 后跨 latent modes 做 `max` primary / logsumexp diagnostic 聚合；
- 输出直接是 item score，不是对两个现成列表做末端 rerank。

### 8.2 2×2 归因

| LATTE latent modes | SETRec set generator | Arm |
|---|---|---|
| off | off | fresh GRAM / common compatible base |
| on | off | G2 |
| off | on | S2 |
| on | on | LCSet-GRAM |

融合晋级必须超过 `max(G2,S2)`：

- `ΔNDCG@10 >= +0.0010`；
- paired 95% CI lower `>0`；
- Hit@10 不下降；
- interaction contrast 为正；
- 至少一个预注册大子组不是由单一 parent 独占贡献；
- 额外成本完整报告。

PCRF 只在最强 standalone/fusion 配置完全冻结后作一个统一后处理对照，不参与救活或选择主架构。

## 9. FP5：独立 fold、跨域与多 seed

### 9.1 D1 准入

冻结 D0 winner 后，只运行：

- Toys D1 seed 2023；
- Beauty D1 seed 2023；
- fresh GRAM matched baseline；
- winner 的直接 parent control；
- 最多一个 standalone winner 和一个 fusion winner。

要求双域 macro NDCG@10 为正，至少一域 paired CI lower `>0`，另一域不得低于 `-0.0005`，双域 Hit@10 均不出现实质下降。

### 9.2 D2 稳健性

只有 D1 PASS 的配置进入 Toys/Beauty D2，seeds `2023/2024/2025`。配置、tokenizer、latent 数、aggregation、beam、query 数和 checkpoint selection rule 全冻结。

official test 与 Sports 仍不读取；Sports 需另立计划和研究者明确授权。

## 10. 指标、统计与完整性

### 10.1 主指标

- Primary：NDCG@10；
- Secondary：Hit@5/10/20/50、NDCG@5/20/50、MRR@10；
- paired user bootstrap：D0/FP 正式实验 2,000 replicates，D1/D2 10,000 replicates；
- 同时报告 point delta、95% CI、gain/loss/tie 用户数和 changed-ranking rate；
- 多 seed 报 dataset-seed unit 和 macro mean，不只报最好 seed。

### 10.2 LATTE 机制指标

- latent-token usage distribution、entropy 和 collapse rate；
- multi-path item rate；
- unique item@50/500；
- duplicate path rate；
- tree distance 与 item probability correlation；
- target prefix/root survival；
- 聚合前后 target rank 和 NDCG；
- PSID collision/reassignment count 与 distortion。

### 10.3 SETRec 机制指标

- CF/semantic token norm、collapse、reconstruction；
- sparse-mask parity 与 forbidden visibility count；
- per-query target grounding rank；
- full-set/item recovery；
- full-catalog valid ranking rate；
- simultaneous vs ordered latency；
- warm/tail 和 history-length 分层。

### 10.4 必须通过的完整性门

- test/Sports/D1 forbidden-read guard；
- target-independent tokenizer/CF/semantic fit；
- parent checkpoint/config/data SHA；
- all new flags off 时 GRAM identity；
- item path/token corpus 唯一性；
- user-level prediction 数与 target 对齐；
- status state machine、immutable runtime、atomic output；
- scientific result 与 GPU1 repeat artifacts 隔离；
- failed attempt 不被覆盖或伪装成新科学结果。

## 11. GPU、并行、后台与 status 强制规则

### 11.1 大于 10 分钟一律后台

1. 预计或实测大于 10 分钟的训练、tokenizer、全目录 inference、bootstrap 或 profile，必须通过 `tmux`/等价不可变后台 runner 启动；
2. 无法判断是否超过 10 分钟时，按超过处理；
3. launcher 完成 preflight、创建 status 和 tmux 后立即返回，不占用 agent 前台等待；
4. agent 不做实时监看、不主动轮询、不持续占用对话；
5. runner 自身负责 heartbeat、epoch/step progress、异常 trap 和 terminal status；
6. 用户只通过稳定 status 文件观察，需要分析时再联系 agent；
7. 小于等于 10 分钟的 CPU tests、静态审计和明确 bounded GPU smoke 可前台运行。

### 11.2 稳定 status 入口

每个实验必须原子更新：

`artifacts/phase17/status/<experiment_id>.status.json`

并更新：

`artifacts/phase17/status/phase17.index.json`

最低字段：

```json
{
  "experiment_id": "s17_fp2_gram_latte_full_toys_d0_seed2023",
  "step_id": "S17-FP2",
  "arm": "G2",
  "scientific_state": "RUNNING",
  "execution_state": "RUNNING_SCIENTIFIC",
  "status_code": "TRAINING",
  "started_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "heartbeat_at": "ISO-8601",
  "launcher_pid": 0,
  "workload_pid": 0,
  "process_alive": true,
  "tmux_session": "phase17_fp2_g2_gpuX",
  "gpu_ids": [],
  "gpu_snapshot": {},
  "stage": "epoch_1",
  "progress": {"current": 1, "total": 50, "unit": "epoch"},
  "best_checkpoint": null,
  "canonical_result_dir": "artifacts/phase17/fullport/fp2/...",
  "log_path": "artifacts/phase17/fullport/fp2/.../run.log",
  "test_read": false,
  "sports_read": false,
  "d1_read": false,
  "result_selection_eligible": true,
  "automatic_retry": false,
  "gpu1_handoff_used": false,
  "gpu1_repeat_restored": null
}
```

heartbeat 建议每 60 秒或每个 optimizer/epoch 边界更新一次；高频更新不得显著拖慢训练。

### 11.3 小实验自行选择 GPU

小实验定义：

- 单卡；
- 预计不超过 4 GPU-hours；
- 不需要停止、迁移或挤压任何现有进程/占位；
- 峰值显存有实测/保守预测且留安全余量；
- 不使用 GPU1。

执行前：

1. 只读检查所有 GPU 的 utilization、free memory、compute PID 和项目状态；
2. 选择真正空闲且最安全的非 GPU1 卡；
3. 将 GPU snapshot、选择理由、预测 peak 写入 status；
4. 不结束或修改任何其他项目/用户进程；
5. 找不到安全空卡时写 `BLOCKED_WAITING_IDLE_GPU`，不自行抢卡；
6. 即使是小实验，只要预计超过 10 分钟仍必须后台运行。

小实验不需要事前询问具体 GPU，但上述安全条件必须全部成立。

### 11.4 大实验必须先申请

满足任一项即为大实验：

- 单 arm 预计超过 4 GPU-hours；
- 需要 2 张或更多 GPU；
- full-data / multi-seed / multi-domain / 150-epoch / full pretraining；
- 需要临时使用研究者已经占用或预留的 GPU；
- 会明显影响其他任务排队；
- 显存接近卡容量或需要 DDP/FSDP。

大实验计划授权不等于 GPU 分配授权。每次正式启动前必须向研究者提交：

```text
实验/步骤：
请求物理 GPU：数量 + 建议 ID（若需已有占用卡，逐卡说明）
每 arm 使用卡数与并行关系：
最低 free memory/卡：
profiled peak allocated/reserved：
预计 wall time 与 GPU-hours：
预计磁盘：
canonical command/config/status 路径：
是否涉及 GPU1：
若涉及 GPU1，交接前重复轮身份与终态恢复命令：
少卡降级/分波方案：
```

未得到明确分配前只可准备代码、CPU tests、bounded small profile 和 status，不得自行释放占位或启动正式 workload。

### 11.5 多卡并行优先

并行原则：优先“一个独立 arm 一张卡”，避免为表面多卡而给小模型上低效 DDP。

FP1+FP2 理想申请 5 张卡并行：

- GPU-A：N0 Native-PSID；
- GPU-B：N1 Native-LATTE；
- GPU-C：G0 GRAM-B0-Fresh；
- GPU-D：G1 GRAM-PSID-Full；
- GPU-E：G2 GRAM-LATTE-Full。

降级方案：

- 3 卡：第一波 G0/G1/G2；第二波 N0/N1；
- 2 卡：先 N0/N1 配对，再 G1/G2 配对，最后 G0；
- 1 卡：不建议；只有研究者明确接受长 wall time 才串行。

FP3 理想申请 4 张卡并行 S0/S1R/S1P/S2。FP4 根据 profile 申请 2–4 张卡；D2 按 dataset×seed 独立 job 并行，理想 6 张卡，资源不足时按 dataset 或 seed 分波。

#### 11.5.1 2026-08-31 当前冻结分配

机器可读配置：`experiment/phase17/config/s17_fp_resource_allocation.json`。

- 当前 8 张 A6000 全部存在 compute PID；30,720 MiB 只是 preferred planning line，不是硬门槛。本次先冻结分配，不启动正式 FP1/FP2，也不停止任何现有进程；
- full-data SentenceT5 tokenizer 固定 GPU0、batch 32，实测峰值 reserved 984 MiB，准入线 5,080 MiB；当前可执行，但仍等待单独启动指令；
- 正式训练前按正式落卡做 arm-specific 短资源 profile：`GPU1:G0 / GPU0:G1 / GPU7:G2 / GPU4:N0,N1`。通过降低 per-device micro/eval batch、保持 effective global batch、beam、top-k、aggregation、模型与训练预算不变，把普通卡 peak reserved 目标压到不超过 20,480 MiB；GPU4 的 N0/N1 profile 以 16,384 MiB 为峰值上限，两臂串行；
- profile 后每个 arm 的正式准入线改为 `实测 peak reserved + 3,072 MiB`，若达到 20 GiB 目标则约需 23,552 MiB free；不得为了适配显存降低 beam、模型容量、epoch 或改变科学协议；
- 若 GPU1 完成资源 profile 后没有立即启动获批的正式 G0，必须用最后一个成功 profile 的冻结命令进入隔离重复轮占卡，直到下一次研究者批准的 handoff；
- FP1+FP2 首选四卡两波：第一波 `GPU1:G0 / GPU0:G1 / GPU7:G2 / GPU4:N0`，第二波在 GPU4 上串行 `N1`；GPU4 只有在当时 free 满足 `N0/N1 各自实测 peak reserved + 3,072 MiB` 时才进入正式运行；
- 若 GPU4 的 Native arm 无法在不改科学协议的前提下压到准入线，回退到三卡两波：第一波 `GPU1:G0 / GPU0:G1 / GPU7:G2`，第二波 `GPU0:N0 / GPU7:N1`；
- 若五卡都满足准入，则升级为 `GPU1:G0 / GPU0:G1 / GPU7:G2 / GPU4:N0 / GPU3:N1`；
- 两卡降级时先用 GPU0/GPU7 跑 N0/N1，再跑 G1/G2，最后 GPU1 跑 G0 并立即进入重复轮；
- 研究者已明确授权 Stage17 共享 GPU4 剩余显存。2026-08-31 18:51 快照为 free 20,425 MiB，Stage16 PID `3680431` 占用 4,448 MiB 且仍在运行；GPU4 上全部既有 PID `3438547/3596503/3680431` 只共存、绝不停止/暂停/修改。初始 Native profile 需 free 至少 19,456 MiB（16,384 + 3,072），正式准入仍使用每臂实测值加 3,072 MiB；
- GPU4 在启动前需两次间隔 5 秒的只读快照；如 free 跌破当前 arm 门槛则等待，不抢占、不换卡、不自动降科学配置。GPU4 的共租 contention 只影响 wall time，所以必须记录但不用耗时作科学比较；
- GPU1 可用于正式大实验和上述资源 profile，但当前 PID `2790130/3862550` 不因本分配自动获得停止授权。启动前必须冻结 handoff 记录并获得精确 PID 处置授权，或等待其自然退出；
- GPU1 科学任务无论成功、失败或中断，均须立即启动隔离重复轮持续占卡；重复轮不可选、指标忽略、不影响科学结论。

同一 family 的 treatment/control 尽量同时间窗并行，减少服务器软件或数据缓存漂移。只有 official implementation 已支持且 profile 证明 wall-time 明显受益时，单 arm 才申请多卡 DDP/FSDP。

### 11.6 研究者已占用 GPU 的交接

- 小实验绝不使用研究者已占用/预留 GPU；
- 只有大实验可申请临时 handoff；
- 申请必须列出精确物理卡、当前项目 PID/session/状态、准备停止的唯一目标和恢复方法；
- 只有研究者明确批准后，才可停止获批的项目进程；
- 未列入批准范围的进程绝不发送信号；
- handoff、科学运行和恢复均写入 status/attempt ledger。

### 11.7 GPU1 不释放与重复轮恢复

1. GPU1 默认维持当前重复轮/资源状态，不用于 CPU preparation、小实验或普通 profile；
2. 只有大实验资源申请明确写出 GPU1 且研究者批准后，才可暂停 GPU1 上获批的重复轮；
3. handoff 前必须冻结：原 experiment ID、tmux/session、runner PID、workload PID、canonical repeat command、state root、显存目标和 status SHA；
4. 新科学 workload 必须在独立 immutable runtime 和 artifact root 中运行；
5. 无论科学任务 `COMPLETED / FAILED / TIMEOUT / INTERRUPTED`，外层 `EXIT`/signal trap 都必须尝试恢复 handoff 前同类重复轮；
6. 恢复后核验新 PID、session、目标显存和 heartbeat；
7. 科学状态与 GPU1 执行状态分离。例如科学成功后：
   - `scientific_state=COMPLETED`
   - `execution_state=RUNNING_GPU1_REPEAT`
   - `gpu1_repeat_restored=true`
8. 若科学失败但重复轮恢复成功，保留 `scientific_state=FAILED`，不得用重复轮掩盖；
9. 若恢复失败，写 `GPU1_REPEAT_RESTORE_FAILED` 并立即向研究者报告；不得静默释放 GPU1；
10. GPU1 重复轮使用隔离目录，`result_selection_eligible=false`、`repeat_metrics_ignored=true`、`affects_scientific_result=false`，不得进入任何效果均值、CI、选模或报告主表。

其他非 GPU1 卡在科学终态后正常释放，不自动持续重复，除非研究者另行明确要求。

### 11.8 失败、timeout 与重试

- 所有正式任务 hard timeout；具体值由 profile 后冻结；
- OOM、NaN、exception、stale heartbeat、identity drift 或 timeout 不自动 retry；
- runner 写 terminal status、failure.json、最后进度与资源快照；
- agent 不因无人实时监看而自动换卡、换 seed、缩模型或改配置；
- 工程恢复需使用新 attempt ID、独立目录并得到研究者对大实验的再次/持续授权；
- 科学失败不得被小实验、重复轮或后续 successful attempt 覆盖。

## 12. Artifact、attempt 与报告结构

```text
artifacts/phase17/
  status/
    phase17.index.json
    s17_fp*.status.json
  attempts/
    S17-FP0.attempts.jsonl
    S17-FP1.attempts.jsonl
    ...
  fullport/
    manifests/
    config/
    fp0/
    fp1_native_latte/
    fp2_gram_latte/
    fp3_setrec/
    fp4_fusion/
    fp5_confirmation/
  runtime/
    <experiment_id>/run-XXXX/    # GPU1 重复轮或非科学 runtime，结果不可选
```

每个正式 attempt 必须保存：

- resolved config、command、source/data/code SHA；
- immutable runtime manifest；
- status、heartbeat、run log、GPU telemetry；
- learning curve、best-checkpoint rule 和 checkpoint SHA；
- user-level predictions、metrics 和 paired bootstrap；
- mechanism diagnostics；
- resource summary；
- failure/terminal summary；
- official test/Sports/D1 read flags。

报告按 FP step 收口，每步最多一份权威汇总：

- `Stage17_FP0_来源数据与Fidelity冻结报告.md`
- `Stage17_FP1_FullLATTE_NativeParity报告.md`
- `Stage17_FP2_GRAM_LATTE_Full正式结果报告.md`
- `Stage17_FP3_FullSETRec正式结果报告.md`
- `Stage17_FP4_LCSet_GRAM融合结果报告.md`
- `Stage17_FP5_独立Fold跨域多Seed确认报告.md`

报告只写 canonical 科学结果。GPU1 重复轮次数、PID 和状态只存在于 status/runtime，不进入科学结果表。

## 13. 授权矩阵与当前下一步

| 动作 | 当前授权 |
|---|---|
| 读取 Stage17 plan/report/artifact 与代码 | 已授权 |
| 修改/新增 Stage17 full-port plan、实现、CPU tests | 已授权 |
| 运行明确 <=10 分钟 CPU/static tests | 已授权 |
| 在安全空闲的非 GPU1 卡运行 bounded 小实验 | 已授权；按第 11.3 节 |
| 启动任何 >10 分钟任务 | 可启动但必须后台；若同时属于大实验仍需单独 GPU 申请 |
| 使用空闲 GPU 跑大实验 | 未自动授权；必须先申请 |
| 暂停/使用研究者已占用 GPU | 未自动授权；仅大实验逐卡申请 |
| 使用 GPU1 | 已明确授权正式大实验使用；启动前必须完成精确 handoff，科学任务结束后必须立即进入隔离重复轮持续占卡；当前 PID 不自动获得停止授权 |
| 使用 GPU0 | 研究者已明确指定 tokenizer `attempt_010` 使用 GPU0；bounded profile 已完成，不扩展到后续大实验 |
| 共享 GPU4 剩余显存 | 已明确授权；优先用于 N0/N1 的 bounded resource profile 及符合实测显存门槛的正式大实验；全部既有 PID（含 Stage16）必须保留，不得发送任何信号 |
| 停止 GPU1 重复轮 | 未授权，除非与获批 GPU1 大实验 handoff 同时发生 |
| 读取 D1/D2、official test、Sports | 未授权或尚未解锁 |

本计划落地后的执行顺序：

1. FP0：冻结 LATTE official commit/config 与 SETRec clean-room fidelity；`COMPLETED`；
2. D0 full adapter、official parity tests、GRAM semantic-ID/latent forest 与 SETRec repo/paper 双合同已通过；`attempt_001` 因快照脚本缺少仓库根 `PYTHONPATH` 在导入期失败，已封存；修复后的 `attempt_002` 成功进入 worker，但 Native LATTE 环境在 `uv` 下载 CPython 3.12.12 时连续三次 TLS EOF，环境任务终止，SentenceT5 cache 与 tokenizer profile 随依赖失败退出；三项状态均由 `artifacts/phase17/status/` 提供；
3. `attempt_003` 已显式复用服务器已有的同版本 uv-managed Python 3.12.12，并通过原生环境 gate；随后 SentenceT5 的 Hugging Face Python 下载在 revision API 处因 TLS EOF 失败，tokenizer profile 随依赖退出。`attempt_004` 改用固定 revision 的 curl 逐文件传输，但 tmux 未继承代理，5 次均在首文件以 TLS code 35 失败且 0 字节落盘。`attempt_005` 给 T5 worker 显式注入经独立 tmux 验证的本机代理，固定 revision 的 13 个文件已完整缓存并通过离线加载；tokenizer profile 因没有满足准入条件的非 GPU1 空卡而 `BLOCKED_WAITING_IDLE_NON_GPU1_GPU`，没有启动 profile；
4. 研究者随后一次性授权 tokenizer `attempt_006` 与 GPU1 原重复轮共享剩余显存，禁止停止、暂停或接管原进程。准入时 GPU1 原 PID `2602227` 存在、空闲约 30.8 GiB，超过预注册的 `10,240 + 4,096 MiB` 门槛；worker 未触碰原进程，但冻结 LATTE 环境的未约束依赖解析到了 `torch 2.13.0+cu130`，服务器驱动仅支持 CUDA 12.6，profile 在任何模型计算前因 CUDA 初始化失败。终态为 `S17_FP0_TOKENIZER_BOUNDED_PROFILE_FAILED`，原 PID 结束核验仍存在，`gpu1_repeat_preserved=true`；没有效果实验、full-data tokenizer 或受保护数据读取；
5. 下一次不得直接重跑 profile。先新建可复现的 CUDA 兼容环境，显式固定支持服务器驱动的 PyTorch wheel，并把 `torch.cuda.is_available()`、设备名和最小 CUDA tensor smoke 纳入环境 gate；环境修复与 `attempt_007` 均需研究者再次确认；
6. 研究者已确认固定官方 `torch 2.7.1+cu126` Python 3.12 wheel（SHA256 `63bce0590bc540fc16139e2be0177847585182b8c5e68d7f9213789d1d96c978`）、GPU1 最小 smoke 与后续 profile。CUDA 环境 `attempt_001` 在创建空 venv 后因新增 execution-state 名称不属于现有 status schema 而退出，torch 安装和 GPU 均未开始；真实终态已封存为 `S17_FP0_CUDA_COMPAT_ENV_RUNNER_SCHEMA_FAILED`。修复后的 `attempt_002` 通过 175 项 tests 后从官方 PyTorch 索引下载，但约 19 分钟后因 `nvidia-nccl-cu12` 触发 uv 30 秒网络超时而失败；tokenizer `attempt_007` 只等待依赖，未运行 profile。研究者确认切换后，实测约快 50 倍的阿里云 `pytorch-wheels/cu126` 直连用于 `attempt_003`，torch 继续校验官方 SHA256，uv timeout 提高到 300 秒；完整环境和 114 个包的依赖检查实际已成功，但 runner 错把 tokenizer profile 的 14,336 MiB 门槛套到仅需极小显存的 CUDA smoke，故环境终态为 `BLOCKED_GPU1_SHARED_HEADROOM_INSUFFICIENT`；tokenizer `attempt_008` 仅因依赖 `BLOCKED` 被错误映射成 `FAILED`，没有执行模型计算；
7. 经研究者再次确认，环境 `attempt_004` 直接复用上述 6 GiB 环境，不联网、不下载、不安装；175 项 Stage17 tests、114 包依赖检查、CPU 离线导入和 GPU1 极小 CUDA smoke 均通过。smoke 峰值保留显存仅 2 MiB，`torch=2.7.1+cu126`、`torch.version.cuda=12.6`，开始前已有 GPU1 PID `2790130/3862550`，结束后均保留。tokenizer `attempt_009` 在独立后台会话等待 GPU1 14,336 MiB 余量，未执行 profile；研究者随后明确要求改用 GPU0，因此该等待 worker 被精确停止并封存为 `S17_FP0_TOKENIZER_PROFILE_STOPPED_FOR_GPU0_SWITCH`，未生成 summary、未使用 GPU；
8. GPU0 专用不可变 `attempt_010` 通过 8 项定向测试与全部 176 项 Stage17 tests 后启动。准入时 GPU0 空闲 28,859 MiB，未停止或修改其已有进程；512-item SentenceT5 profile 于物理 GPU0 完成，输出 shape `[512,768]`、全部 finite、编码 5.798 秒、88.30 items/s、峰值 allocated/reserved 为 932.95/984 MiB，原有 GPU0 PID 均保留。按线性 encoding-only 估计，Toys 全目录约 135.0 秒；终态 `PASS_S17_FP0_TOKENIZER_BOUNDED_PROFILE`；
9. bounded profile 通过后，研究者单独批准物理 GPU0 上的 full-data tokenizer `attempt_001`。该任务以 984 MiB peak reserved 在 111.28 秒内完成：11,138 个 train-prefix 物品拟合 whitened PCA/RQ-KMeans，11,924 个目录物品全部赋码，1,337 个冲突重分配后 collision alias 为 0，官方 `.sem_ids` 与项目导出 byte-identical；终态 `PASS_S17_FP0_FULL_DATA_TOKENIZER`；
10. tokenizer attempt 保持不可变。其 observed vocabulary 为 775 个；`amendment_001` 只补充合法但目录未观察到的 `<s17_sid1_236>`，冻结完整 `3x256 + 8 = 776` 个 G1/G2 新 token，并明确不修改 tokenizer manifest 或官方 `.sem_ids`；
11. G0/G1/G2 leakage-safe GRAM backend、N0/N1 pinned-official backend、primary beam-500 resource workload、两次 GPU snapshot/PID 白名单/逐臂授权 gate 与统一 runner 已实现。五臂 CPU preflight 均 PASS，全部 immutable profile `attempt_001` 已进入 `PREFLIGHT / S17_FP12_RESOURCE_PROFILE_READY_AUTHORIZATION_REQUIRED`；211 项 Stage17 tests 通过（GRAM 环境跳过 1 项 native-only test），native 环境的该项 official test 单独通过；
12. 2026-08-31 22:46:57+08:00 的只读快照显示指定 GPU1/GPU0/GPU7/GPU4 均未达到各自 23,552/19,456 MiB free 准入线，GPU1 同时超过 20% utilization gate；因此没有创建逐臂授权记录、没有查询外部 target、没有启动任何 GPU resource profile 或 FP1/FP2 effect experiment。精确 deficit/PID 集见 `artifacts/phase17/fullport/profiles/profile_authorization_request_001.json`；
13. 下一步仅在指定卡满足门槛，或研究者明确批准重新落卡后，重新冻结 PID/handoff 并生成 attempt-specific profile authorization；先完成五臂 resource profile，再根据实测 peak 提交 FP1+FP2 正式多卡申请。正式大实验仍需新的研究者授权。

## 14. Primary Sources

- Hou et al., [Expressiveness Limits of Autoregressive Semantic ID Generation in Generative Recommendation](https://arxiv.org/abs/2605.06331), 2026；[official Latte repository](https://github.com/hyp1231/Latte)（MIT）。
- Lin et al., [Order-agnostic Identifier for Large Language Model-based Generative Recommendation](https://arxiv.org/abs/2502.10833), SIGIR 2025；[official SETRec repository](https://github.com/Linxyhaha/SETRec)（NUS copyright notice；无标准开源 license 时只作 clean-room reference）。
- Stage17 local evidence：`report/第十七阶段/Stage17_S2R_架构级候选重选与大改筛选报告.md`。
- Stage17 data/runtime contract：`report/第十七阶段/Stage17_S0_证据源码数据与资源审计报告.md`、`report/第十七阶段/Stage17_S1_公共迁移框架与运行合约报告.md`。
