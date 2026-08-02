# GRAM 第七阶段：GCGD-v1 图条件生成解码实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-02
- Verification Status: DESIGN_V1_NOT_PREREGISTERED
- Version Label: `phase7_gcgd_v1_graph_conditioned_generation_design_v1`
- Development Domains: Toys、Beauty
- Sports/Test: 封存
- Device: 物理 GPU0；单卡；30 GiB 总显存租约

## 1. 阶段定位

第六阶段 GACR-v3–v7 都冻结 GRAM，只在固定候选上训练有界 residual。该路线适合验证关系信号，
但不会改变 GRAM 的候选生成过程，预期增益受到 beam-50 覆盖率和 residual bound 的共同限制。

第七阶段不再把“换 residual loss”视为主要创新，而是研究一个结构性更大的改动：在 GRAM 的
constrained beam search 每个 token step 内，将只由训练交互图计算的 item 分数投影到当前合法的
lexical-ID prefix，并与 GRAM token logits 融合，使图信号能够改变候选生成和 beam 扩展，而不是
在生成结束后重排。

工作名为 **GCGD（Graph-Conditioned GRAM Decoding）**。AGRec 已证明“GNN logits 引导
autoregressive decoding”是一条已有且有效的基础路线，因此 GCGD 不把“GNN+LLM”本身声称为
新颖性。潜在差异聚焦于：GRAM lexical hierarchical identifier 的 prefix 概率聚合、GRAM
multi-granular semantic evidence 与 graph evidence 的冲突建模，以及历史可见的可靠性回退。

## 2. 研究问题与假设

主问题：**把训练交互图的高阶协同信号注入 GRAM 的 token-level constrained decoding，能否在
Toys 与 Beauty 同时扩大有效候选覆盖，并相对原始 GRAM 和 GACR-v3 获得明显而安全的提升？**

- H1（机制）：GCGD 相对 GRAM 提高 target-in-beam@50 与 unique-prefix coverage；
- H2（效果）：两域 overall NDCG@10 和 Recall@10 均高于 matched GRAM；
- H3（幅度目标）：希望至少一域 mean NDCG@10 相对提升接近 `5%`，但该数值只作为方向目标，
  不作为早期版本的硬停止线；
- H4（安全）：head/tail NDCG@10、Recall@50 不发生预注册阈值以上的下降；
- H5（区别于 residual）：GCGD 新增命中的用户不只来自 GRAM 原 beam-50 内部换位。

H3 是研究目标，不是保留门或承诺；跨 cohort 或论文 Table 2 数值不得用于计算正式提升。

## 3. 方法定义（设计版本）

### 3.1 训练图

- 每域仅用官方 train split 的用户—商品交互构建二部图；validation/test target、Sports 禁读；
- 图编码器第一版使用 LightGCN；输入、采样、层数、维度、训练轮次必须在实现前写入冻结 config；
- 图模型输出每个用户对 catalog item 的分数 `q_u(i)`；所有图产物保存输入 lineage 与 SHA-256；
- 新用户或图证据无效时必须有 null path，严格退回 GRAM。

### 3.2 item-to-prefix 概率投影

设当前生成 prefix 为 `p`，Trie 下一个合法 token 为 `t`，与 `p+t` 相容的 catalog item 集为
`I(p,t)`。先对 catalog item 图分数做归一化，再计算：

`G_u(t|p) = logsumexp_{i in I(p,t)} log softmax_i(q_u(i))`

该定义将 leaf item 概率质量聚合到合法 prefix，不允许把 top-n item 简单复制成 token 分数，也不
允许图分支生成 Trie 外 token。必须测试概率守恒、stable tie-break、空 prefix 与单 leaf 情况。

### 3.3 token-level 融合

第一版目标形式：

`L(t|u,p) = L_GRAM(t|u,p) + alpha * g(u,p) * norm(G_u(t|p))`

- `alpha`、图分数归一化和 gate 特征在预注册前冻结；fresh validation/test 不参与选择；
- `g(u,p)` 只读历史可见信号：图覆盖、图熵、top-1/top-2 margin、有效 leaf 数、GRAM/graph
  prefix agreement、prefix depth；禁止目标商品、未来交互或 validation label；
- gate 无证据时输出 0；保留 `alpha=0` 的精确 GRAM identity test；
- adapter/gate 只能从 train-fit users 学习，calibration users 仅用于 fail-closed 校准和诊断。

### 3.4 训练目标

设计候选为 `next-token CE + listwise item loss + gate reliability loss`。在冻结前必须通过 train-only
pilot 决定是否保留全部三项；不得在 fresh validation 结果后增加 loss。第七阶段不沿用第六阶段的
6 维 residual 或 `bound=0.2`，但 GACR-v3 作为强轻量对照保留。

## 4. 分阶段设计

### P0：CPU/小 GPU 机制与 lineage 验证

- 实现 item-to-prefix 聚合、logit fusion、Trie 合法性、identity、leakage 和 SHA 审计；
- 仅允许 train-fit/calibration 数据；不读取 fresh validation/test/Sports；
- 输出 graph coverage、prefix branching、图/GRAM agreement 和预计显存；不做方法优劣结论。
- 第一拍固定为 CPU-only `GCGD-P0-LINEAGE-V1`：读取 `user_sequence.txt` 后立即剔除每用户最后
  两个 holdout position，只统计 `items[:-2]` 的训练图；不加载 checkpoint、不生成预测、不读取
  validation/test target 值或 Sports。GPU0 CodeLlama 在该 CPU 任务前、中、后持续持有 30 GiB；
  若后续 P0 增加任何 CUDA workload，则必须另立 config 并改用 30,720 MiB sidecar 总租约。
- CPU lineage 通过后增加一次 `GCGD-P0-GPU-SMOKE-V1`：每域仅取一个确定性的 train-only
  sample，加载冻结 C1 parent，执行 baseline、`alpha=0` identity 和非零 graph-prefix 三条生成路径；
  只用于验证真实 generation API、Trie 映射和显存峰值，不读取 development validation，也不允许
  据此作效果结论。该 smoke 使用独立 config/runner，并继承 GPU0、后台 tmux、30 GiB 总租约、
  CodeLlama 前后占位和 no-auto-retry 规则。

### P1：单 seed 开发 pilot

- seed=`2023`，Toys/Beauty 同时执行；所有方法共享用户、checkpoint、beam、Trie 与评测代码；
- arms：`A=GRAM`、`B=GRAM+graph-prefix decoding`、`C=B+trainable adapter/gate`；
- P1 只使用新建且与历史 cohort 隔离的 development validation；用于决定是否值得进入确认性实验，
  不用于反复扫描 alpha、层数、loss 权重或 gate 阈值；
- P1 不设 `+3%/+5%` 一类硬效果门。只要相对 GACR-v3 出现方向一致、可解释的正向信号，且没有
  明显安全退化，就可以进入下一版设计或 P2；overall、tail、graph-covered、new-hit 或
  target-in-beam 的改善均可作为继续依据，但必须同时报告其它指标，不能只挑最好子组；
- P1 的硬门仅限完整性：无泄漏、finite、identity、checkpoint SHA、同 cohort paired comparison、
  GPU/CodeLlama/status 治理全部通过。效果强弱由研究者结合绝对变化、相对变化、CI、覆盖率和
  机制证据判断，不由单一阈值自动终止。
- 修复后的 P1 train-only smoke 实测峰值为 Toys=`4,426 MiB`、Beauty=`1,476 MiB`。正式 P1 按域
  顺序执行并分别留出工程余量：Toys workload budget=`5,120 MiB`、sidecar=`25,600 MiB`；Beauty
  workload budget=`2,048 MiB`、sidecar=`28,672 MiB`。每个域合计均为 `30,720 MiB`，不得继续
  使用 P0 前的 `24,576 MiB` 占位估计；域切换时必须先结束旧 sidecar，再建立新 sidecar。
- P1 development cohort 固定每域 `512` 用户，salt=`phase7-gcgd-p1-development-v1`；排除 GCDH
  train/validation 与 GACR/CET/GACR-v2–v6 全部已用 validation salt。P1 是开发 pilot，不解封 test。
- LightGCN 冻结为 embedding dim=`64`、2 层、BPR、20 epochs、batch=`4096`、lr=`1e-3`、
  L2=`1e-4`、seed=`2023`；只使用 `items[:-2]` 图。B arm 固定 `alpha=0.30`、gate=1；C arm
  固定 `alpha=0.50`，使用 6→16→1 reliability gate 与单一正值 graph-temperature adapter，目标权重
  为 next-token CE=`1.0`、LightGCN BPR/listwise=`0.2`、gate reliability BCE=`0.1`，不扫描参数。

### P2：三 seed 确认性 fresh validation

- seeds=`2023/2024/2025`；P1 后冻结单一配置，另建完全隔离的 fresh cohort；
- matched arms：GRAM、GACR-v3、B、C；主比较为 C vs GRAM，机制比较为 C vs B、C vs GACR-v3；
- 指标：Recall/NDCG@5/10、Recall@50、MRR、target-in-beam@50、new-hit@10 outside GRAM beam、
  changed coverage、broad harm、head/tail 与 graph-covered/uncovered 分组；
- 用户级 paired bootstrap 10,000 次；报告绝对 pp、相对百分比、95% CI 与六 cell seed 稳定性；
- P2 完成前继续封存正式 test。是否解封 test 必须由研究者单独明确授权。

## 5. 冻结项与反泄漏规则

- 原始 GRAM checkpoint、官方 train/validation/test split、Trie、lexical item ID、beam size 与 stable
  tie-break 冻结；所有 arms 共用同一 cohort 与同一逐用户评测；
- 图只读 train interactions；按用户确定性拆分 fit/calibration；目标只作为训练监督，不作为 gate 特征；
- historical GCDH/GACR validation salts 全部排除；P1 与 P2 cohort 互斥；
- Sports/test 默认禁止读取；runner status 与 summary 必须显式记录 `test_read=false`、
  `sports_read=false`；
- 不允许用第一、二阶段 full-test 指标与 fresh validation 指标直接计算正式提升；
- P1 后只允许一次设计冻结；失败不得自动扩大模型、扫描新参数、换 cohort 或重试。

## 6. P2 迭代门与最终 test 前复核

第七阶段允许逐版本改进，不要求 GCGD-v1 一次达到最终形态。P2 后分成两个决定：

1. **是否继续迭代**：six-cell macro NDCG@10 相对 GACR-v3 为正，或存在清晰且可复现的候选覆盖/
   tail/new-hit 机制改善，即可保留路线并设计 GCGD-v2；不要求统计显著，也不要求一次达到 4–5%；
2. **是否进入正式 test**：在准备解封 test 时另行冻结。届时综合两域 overall、tail、Recall@50、
   seed 稳定性、CI、broad harm 与机制增量决定，不预先把 `+5%/+4%` 当作唯一通行证。

所有版本只有完整性门是硬门：lineage、无泄漏、finite、GRAM identity、matched cohort、parent
checkpoint SHA、资源租约和 test/Sports 封存必须全部通过。一次结果后的方法修改必须登记为新的
版本、写清单一改动因素，并使用新的 development cohort 或 train-only calibration；不得在同一
fresh cohort 上反复调参后把最终版本包装成一次确认性实验。

## 7. GPU0、后台与资源治理（继承第六阶段）

- 所有 phase7 GPU workload 固定使用物理 GPU0，且通过 `CUDA_VISIBLE_DEVICES=0` 暴露为逻辑
  `cuda:0`；不自动切换其它 GPU；
- 实验开始前要求 GPU0 上的 CodeLlama reservation 正在运行；runner 在具名 tmux worker 中停止
  CodeLlama，通过 admission gate 后才启动科学 workload；
- 所有实验只允许后台 tmux 运行，禁止前台长任务；runner 提供 `{start|status|worker}`，status 至少
  输出 tmux、status.json 与 run.log tail；
- 每个预计峰值低于 `30,720 MiB` 的 workload，必须持有总计 `30,720 MiB` 显存租约；工作负载自身
  CUDA 占用计入租约，`experiment/gpu_memory_lease.py` sidecar 持有
  `30,720 - expected_workload_peak_mib`；无论 workload 实际只用多少，都不得释放差额；
- config 必须预声明保守的 `expected_workload_peak_mib <= 30,720`；未完成 P0 实测前暂定
  `24,576 MiB`，sidecar=`6,144 MiB`，该数字属于设计占位，不得据此直接启动 P1；
- admission gate 要求 GPU0 free memory `>=30,720 MiB`；每 5 秒写 GPU telemetry；hard timeout、
  no automatic retry；
- workload 成功、失败、timeout、SIGTERM/HUP 的所有退出路径都先释放 sidecar，再在物理 GPU0
  恢复 CodeLlama；恢复失败必须在 status 中显式标记，不能把科学结果状态伪装为完全成功。

## 8. 产物与状态接口

- implementation：`experiment/phase7/gcgd_v1.py`
- tests：`experiment/phase7/test_gcgd_v1.py`
- runner：`experiment/phase7/run_phase7_gcgd_v1.sh`
- P0 implementation/test/runner：`gcgd_p0.py`、`test_gcgd_p0.py`、`run_phase7_gcgd_p0.sh`
- design config：`artifacts/phase7/configs/gcgd_v1_design.json`
- P0 config/output：`artifacts/phase7/configs/gcgd_p0_preregistered.json`、
  `artifacts/phase7/gcgd_p0/`
- P0 GPU smoke：`gcgd_p0_gpu_smoke.py`、`test_gcgd_p0_gpu_smoke.py`、
  `run_phase7_gcgd_p0_gpu_smoke.sh`；config/output 位于
  `artifacts/phase7/configs/gcgd_p0_gpu_smoke_preregistered.json` 与
  `artifacts/phase7/gcgd_p0_gpu_smoke/`
- output：`artifacts/phase7/gcgd_v1/`
- 正式 P1 implementation/test/runner：`gcgd_p1_run.py`、`test_gcgd_p1_run.py`、
  `run_phase7_gcgd_p1.sh`；config/output 位于
  `artifacts/phase7/configs/gcgd_p1_preregistered_draft.json` 与 `artifacts/phase7/gcgd_p1/`
- 正式 P1 查询：`bash experiment/phase7/run_phase7_gcgd_p1.sh status`
- 查询：`bash experiment/phase7/run_phase7_gcgd_v1.sh status`
- P0 查询：`bash experiment/phase7/run_phase7_gcgd_p0.sh status`
- 后续启动（仅冻结后）：`bash experiment/phase7/run_phase7_gcgd_v1.sh start`

正式 P1 当前 `execution_enabled=false`、状态为
`IMPLEMENTED_AWAITING_RESEARCHER_CONFIRMATION`。graph encoder、adapter、loss、P1 cohort、输入
checkpoint/residual SHA、12 个直接代码依赖 SHA、train-only smoke 和按域显存租约均已完成；runner
继续 fail closed，直到研究者明确确认启动，确认前不得停止 CodeLlama 或启动正式 workload。

## 9. 当前设计状态与下一步

P0 lineage、GPU smoke、P1 train-only end-to-end smoke 与正式 P1 实现均已通过完整性检查。正式
P1 的唯一剩余门是研究者确认；确认后只把候选 config 改为
`PREREGISTERED_FROZEN_READY_TO_RUN`/`execution_enabled=true`，重新核验 SHA 与 CodeLlama/GPU0，
再以后台 tmux 启动。未经确认不启动实验。
