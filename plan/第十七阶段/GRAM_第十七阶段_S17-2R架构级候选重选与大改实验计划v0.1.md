# Stage 17 S17-2R：架构级候选重选与大改实验计划 v0.1

## Material Passport

- Origin Skill：`academic-research-suite / deep-research + experiment-agent`
- Created：2026-08-30
- Status：`AUTHORIZED / STATIC_PREFLIGHT_READY / GPU_NOT_STARTED`
- Supersedes：S17-0～S17-4 完成后直接进入 S17-5 的旧顺序
- Does not invalidate：S17-0～S17-4 的实验事实、失败证据和报告
- Protected data：Toys/Beauty Phase17 shadow folds；official test 与 Sports 继续封存
- Runtime reservation：GPU1 继续只运行 S17-4 成功后的非科学重复轮；本计划准备阶段不得收回

## 1. 决策

研究者于 2026-08-30 明确选择 **S17-2R**，并允许 identifier、tokenizer、decoder、training objective、candidate-generation/ranking architecture 和 backbone 发生大改动。

因此，S17-2R 不再问“哪个轻量 hook 能继续挂到现有 GRAM 上”，而是重新问：

> 在严格保留数据切分、泄漏防护和 item-level 评测的前提下，哪一种生成式推荐架构能真正突破当前 lexical autoregressive GRAM 的效果瓶颈？

S17-5 暂时 `HOLD`。只有 S17-2R 选出至多两个通过严格匹配对照的架构后，才恢复独立 D1 准入。

## 2. 为什么当前路线属于局部迁移

原总计划文字虽然允许替换 backbone/tokenizer，但 S17-2～S17-4 的实际执行一直保留：

- T5-small 与 FiD 输入骨架；
- native lexical identifier 及其合法 Trie/beam；
- 历史 GRAM checkpoint + fresh optimizer + 1 个继续训练 epoch；
- 机制主要以 auxiliary loss、root token、轻量 adapter、gate 或 beam 后处理接入。

具体来说，S17 的 `B1_latte` 只是把确定性 root 放在不变 lexical suffix 前；`P1_sethead` 只是辅助 token-set loss；二者都没有训练 semantic ID、latent-token-conditioned multiple trees 或 order-agnostic simultaneous decoder。因此其负结果只否定当前局部映射，不否定完整 Latte/SETRec。

同时，S17-2 的 1k-user 短 probe 曾把 B0/B1 排在前列，S17-3 全量时却分别变成明显负向和近零，说明“历史诊断基线 + 单个短预算点”的选择协议预测性不足。S17-2R 必须同时重做架构和准入方法。

## 3. 不再保护与继续保护的边界

### 3.1 允许替换

- native lexical ID → semantic/set/differentiable identifier；
- autoregressive beam decoder → masked diffusion、并行 set decoder 或生成后联合 item scorer；
- T5-small/FiD → 方法原生且资源可承受的 encoder-decoder；
- historical checkpoint continuation → 从头训练 tokenizer 和 recommender；
- 单一 token CE → diffusion/set/joint recommendation-ranking objectives；
- Trie-only item mapping → collision-aware SID-to-item resolution 与 item-level ranking。

发生以上替换后，产物称为 `S17 architecture candidate`，不得继续笼统声称是 faithful GRAM；只有保留 GRAM 核心结构的分支才可用 `*-GRAM` 命名。

### 3.2 绝不替换

- Phase17 shadow-fold 因果切分与禁止 future target 进入输入的规则；
- official test、Sports 的封存；
- 同一 item catalog、用户级 prediction 与统一 item-level Hit/NDCG evaluator；
- 每个候选的 matched native control；
- 运行快照、attempt ledger、资源遥测和终态报告合同；
- GPU1 重复轮不参与科学选模。

## 4. 候选架构池

| ID | 架构级变化 | 与当前失败表型的关系 | 代码/许可结论 | 2R 优先级 |
|---|---|---|---|---:|
| `R2A-GRYPHON-ITEM` | semantic-ID generator 旁增加联合训练 item-level scorer；beam 只产候选，最终按 item score 排序 | 直接处理“候选覆盖尚可、sequence likelihood 排序差”和 SID collision | 2026 预印本；未发现官方代码，必须独立实现 | **P0** |
| `R2B-LATTE-FULL` | 重新训练 semantic ID，并用多个 latent token 形成 conditioned trees；完整 item aggregation | 直接处理 tree-induced probability coupling；旧 deterministic root 不代表本方法 | 官方代码 MIT；需写 Phase17 数据适配器 | **P0** |
| `R2C-DIFFGRM` | 以 masked discrete diffusion 替换 AR decoder；PSE/OCN/CPD 全链路 | 绕开 prefix error accumulation 与逐 digit 单向依赖 | WWW 2026 + 官方代码；仓库未见标准 license，默认只独立实现思想；Toys 原配置较重 | **P0 / 先 profile** |
| `R2D-SETREC-FULL` | CF + semantic tokenizers、order-agnostic set identifier、sparse attention、simultaneous generation | 绕开 token order 与 beam local optimum；旧 auxiliary SetHead 不代表本方法 | SIGIR 2025 + Toys/Beauty 官方代码；仅标注 NUS copyright，未见标准开源许可，默认独立实现 | **P0** |
| `R2E-DIGER-JOINTSID` | recommendation gradient 联合优化 RQ-VAE semantic ID，带 Gumbel exploration/decay | 处理 frozen tokenizer 与 recommendation objective mismatch | SIGIR 2026 + 官方仓库；`main` 曾切断到量化器梯度，修复位于 `gradient-fix`；许可未核验 | P1 候补 |
| `R2F-RECOCHAIN` | 同 backbone 先生成再 SIM rank | 与 Gryphon 高度重叠 | 2026 work-in-progress；未发现官方代码 | P2 旁证，不单列 GPU arm |

完整机器可读候选卡位于 `experiment/phase17/registry/architecture_cards/`。

## 5. S17-2R 的分级执行

### R0：静态来源、许可、数据与资源审计（CPU）

对所有候选冻结 paper URL、repository URL、许可状态、原生数据格式、参数量/显存未知项、所需 tokenizer artifact 和 matched control。未明确许可的仓库只能阅读，不复制源码。

通过条件：

- 至少四个不重叠的架构 family；
- 每个 P0 有明确 causal contrast 和 native control；
- Phase17 数据可映射到 `history -> target item` 且无需打开官方 test/Sports；
- 资源未知项被标为 `PROFILE_REQUIRED`，不得凭论文命令直接上全量。

### R1：100-user 端到端契约与过拟合 smoke（GPU，不计效果）

每个 P0 必须完整走通：

`item text/CF -> identifier/tokenizer -> training -> generation -> item resolution -> unified evaluator`

检查：

- target/future 不进入 user encoder；
- 每个生成结果都能解析为 catalog item；
- collision/duplicate 的处理可复现；
- treatment 特有参数获得有限非零梯度；
- 训练 loss 能下降，prediction 文件非空；
- peak memory 和 step time 被记录。

R1 不比较 NDCG，不选方向，不把 100-user 过拟合写成效果证据。GPU1 明确排除。

### R2：3k-user 架构筛选（GPU，首个效果门槛）

- 从 Toys D0 固定 3,000 个 discovery users，冻结成 3 个互斥的 1,000-user evaluation cohorts；
- treatment 与其 native control 使用相同 identifier inputs、训练用户、seed、参数级别、优化机会与 evaluator；
- 训练至预注册 early-stop 或候选对上限，禁止固定“只跑 1 epoch”导致未收敛模型互比；
- 每条路径保存逐用户 predictions、learning curve、candidate recall、ranking gain、collision 和 latency；
- architecture-vs-control 是 causal contrast；跨架构绝对指标只是预算匹配下的 selection evidence，不冒充完全等价训练。

R2 强准入同时满足：

1. mean `ΔNDCG@10 >= +0.0015`；
2. 三个 cohort 至少两个为正；
3. `ΔHit@10 >= 0`，且任何 cohort 不出现 `ΔHit@10 < -0.002`；
4. 目标机制指标按预期变化；
5. 无 leakage、非法 item、OOM 或 evaluator 漂移。

若 mean delta 为正且机制成立但置信区间跨 0，可进入 `BORDERLINE_ONE_REVISION`；每个 family 只允许一次有诊断依据的修订。非正或 control 不完整则拒绝。

### R3：至多两个架构的 full Toys D0 决赛

- 只运行 R2 排名前二；
- 使用完整 Toys D0、paired user predictions 和 frozen configuration；
- 每个 treatment 必须带自己的 matched native control；
- 同时保留现有 `GRAM-Continue` 绝对指标作为 external anchor，但它不是所有新架构的 matched causal control；
- 强通过要求 `ΔNDCG@10 >= +0.0015`、paired 95% CI lower bound `> 0`、`ΔHit@10 >= 0`，且无预定义 subgroup 灾难性回退。

R3 通过后，最多两个架构进入新的 S17-5：Toys D1 + Beauty D1 独立准入。D1 在 2R 期间不得提前查看。

## 6. 每个候选的 matched native control

| Candidate | 必须对照 | 主要机制指标 |
|---|---|---|
| Gryphon-item | 同一 generator、同一 beam candidates，以 sequence likelihood 排序；另报 collision-resolved no-scorer | candidate recall@K、same-candidate item-rank gain、collision split accuracy |
| Latte-full | PSID/单树，使用相同 semantic IDs、backbone、训练步数和 beam | tree coupling、multi-path coverage、duplicate path、item aggregation gain |
| DiffGRM | 相同 PSE identifier 与容量下的 AR/uniform objective control | per-digit entropy/accuracy、denoising steps、valid-item rate、diversity |
| SETRec-full | 相同 CF+semantic tokens 的 ordered AR decoder | simultaneous-generation validity、set recovery、token-order sensitivity、latency |
| DIGER | 相同初始化/容量的 frozen RQ-VAE control | codebook utilization、per-level perplexity、collapse rate、recommendation-gradient reachability |

任何没有 native control 的单模型结果都只能标为 engineering run，不得晋级。

## 7. 资源和并行策略

- CPU R0 与适配器开发可立即并行，不使用 GPU。
- R1/R2 最多并行四个独立 family，每个 job 一张卡；优先申请 GPU1 之外满足 profile 的空闲 A6000。
- GPU1 当前仅供 S17-4 非科学重复轮占卡；不得为 smoke 停掉重复轮。
- 如果没有其他合格卡，先排队，不私自与 GPU1 上现有进程叠跑。只有研究者明确同意科学 handoff 时，才可暂停重复轮让 GPU1 承担科学 job；科学结束后恢复同类重复轮。
- 正式申请在 R1 profile 后给出：候选数、每卡 peak、预计 wall time、建议并发卡数和少卡分波方案。
- DiffGRM 官方 Toys 示例为 1024 hidden、8 heads、batch 1024，不能假设适配后仍在 30 GiB；必须先 profile。
- RQ-VAE/semantic embedding 可缓存复用，但缓存必须带数据 hash、encoder/version、fold 和 forbidden-read manifest。

## 8. 反方审查与止损

1. **公平性不是所有模型强行同 epoch。** 架构需要不同的 tokenizer/pretraining，强行相同 epoch 会偏向旧 GRAM；因此同时报告 family-native matched contrast 与 cross-family fixed-budget ranking。
2. **不把 D0 反复调成 test。** 每个 family 只允许一次修订；D1/Beauty D1 在架构冻结前保持封存。
3. **不把论文增益外推到本仓库。** DiffGRM/Latte/SETRec/DIGER 的论文数字只支持候选合理性，本地结论只来自 Phase17 predictions。
4. **许可不清晰就不复制。** DiffGRM、SETRec、DIGER 暂不导入源码；只能根据论文独立实现或先取得许可。
5. **DIGER 修复分支风险。** `main` 的历史复现结果不能证明修复后的 end-to-end gradient 版本仍有相同效果；未通过 gradient reachability 和小规模 parity 前不升级 P0。
6. **大改不等于无限实验。** P0 最多四个、full D0 最多两个、每个 family 最多一次修订。

## 9. 当前执行边界

本文件落地代表 S17-2R 已授权且静态准备开始，不代表 GPU 科学实验已启动。当前应完成：

1. 冻结候选卡、预算与契约测试；
2. 实现统一数据 adapter 和 native-control 接口；
3. 做 CPU contract tests；
4. 基于 R1 单卡 profile 向研究者提交 GPU 请求；
5. 获得资源后再启动 R1/R2。

## 10. Primary sources（核验于 2026-08-30）

- DiffGRM：[paper](https://arxiv.org/abs/2510.21805)，[official repository](https://github.com/liuzhao09/DiffGRM)
- Gryphon：[paper](https://arxiv.org/abs/2606.08604)
- Latte：[paper](https://arxiv.org/abs/2605.06331)，[official MIT repository](https://github.com/hyp1231/Latte)
- SETRec：[paper](https://arxiv.org/abs/2502.10833)，[official repository](https://github.com/Linxyhaha/SETRec)
- DIGER：[paper](https://arxiv.org/abs/2601.19711)，[official repository](https://github.com/junchen-fu/DIGER)
- RecoChain：[paper](https://arxiv.org/abs/2604.25787)
