# GRAM 第十二阶段 HI-GRAM:层次化跨 item 早融合实验计划(主线设计)

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-05
- Verification Status: `ARCHIVED_CONFIRMATION_TEMPLATE`(2026-08-05 归档;当前使用 v0.2 探索版)
- Version Label: `phase12_hi_gram_main_design_v0.1`
- Superseded By: `phase12_hi_gram_exploration_v0.2` (`GRAM_第十二阶段_HI-GRAM探索计划v0.2.md`)
- Archive Note: 本 v0.1 是投稿版规格(3 seeds/正式 gate/双域全指标 + confirmation),当前**不用**。仅在 v0.2 探索期跑到 CONFIRMATION 门槛后,作为 v1.0 CONFIRMATION plan 的基础骨架
- Experiment ID: `GRAM_PHASE12_HI_GRAM_MAIN_V1`
- Parent Governance: `plan/GRAM_后续结构性方向分阶段实验治理规则.md`(修订版:架构类改动免 P0)
- Development domains: Beauty、Toys
- Confirmation domain: Sports 或 Yelp(方法与主实验冻结后一次性使用)
- Test/Sports: 主实验期间不读取,confirmation 阶段仅一次性读取选定 confirmation 域
- Resource policy: CodeLlama 实验前后占位;实验期间持有 30 GiB 总显存租约

## 1. 阶段定位

第十二阶段是继第十一阶段 BW3-P2 独立验证失败后,首个明确定位为**结构性架构改动**的方向。
不延续 GACR/CF0/PCRF/BW 系列的"事后校准/融合/门控"路线,改为对 GRAM encoder 内部
前向流程做替换级改动。

第一轮的唯一目标是:

> 验证在 GRAM encoder 输出上插入**局部窗口 + 全局**两级 cross-item attention 早融合模块,
> 能否在 Beauty/Toys 上稳定提升 Recall@10 与 NDCG@10。

命名:**HI-GRAM (Hierarchical Interaction GRAM)**

论文 story:

> GRAM 的 "Multi-granular Late Fusion" 效率友好,但 item 之间的语义关系只能在 decoder
> 的 cross-attention 里事后混合,存在 information bottleneck。HI-GRAM 在 encoder 阶段引入
> hierarchical cross-item attention,把 item 间 fusion 提前发生,并通过 residual + 可学习
> scaling 保证 fallback 到 GRAM 的能力。

对照工作(不复现,只在 related work 引用):

- GRAM (ACL'25):late fusion baseline
- SASRec:纯 item-level self-attention,无文本 encoder,与本工作机制不同
- LETTER (CIKM'24) / ETEGRec (SIGIR'25):collaborative tokenization 方向,与本工作正交
- CapsID / QuaSID (2026):variable-length / collision-aware SID,与本工作正交

本阶段允许借鉴已有论文的成熟结构,HI-GRAM v0.1 只是效果原型;
真正的方法定位(local window size、fusion 位置、pooling 方式)可在 W1 smoke 通过后依据
消融再冻结。

## 2. 目标 venue 与时间线

- 目标 venue:CCF-B(SIGIR/CIKM/RecSys 短文或 CCF-B 期刊)
- 投稿时间线:6-8 周完成实验 + 论文初稿;总时长控制在 3-6 个月内
- Constraints:8×A6000 48G 共享,单卡实际可用 20-30G;主要依赖 AI coding 工具实现

## 3. HI-GRAM 最小模型定义

HI-GRAM 在 GRAM encoder 输出之后插入 hierarchical cross-item interaction 路径:

1. 保留 GRAM 原有 T5/FiD 文本 encoder、Trie 约束生成、lexical ID 与 late-fusion decoder;
2. 在 EncoderWrapper 拿到 `last_hidden_states = (B*N, L, D)` 后,先做 masked-mean item pooling,
   得到 `(B, N, D)` 的 item-level 表示;
3. Local Window Attention:每个 item 只 attend 到 `[i-W+1, i]` 范围内的 item,W 默认 5,
   使用可学习 item-position embedding,不复用 T5 relative bias;
4. Global Attention:在 local 输出之上做无限制 self-attention over N items;
5. Residual fusion 回 token level:`hidden = hidden + α · (item_repr_global - item_repr).expand(...)`,
   其中 `α` 为可学习标量,初始化 0.1,通过 sigmoid 或直接实数均可,W1 smoke 时确定;
6. 训练损失完全沿用 GRAM 原 token CE,不增加辅助 loss(第一轮避免多变量);
7. 生成阶段与 GRAM 完全一致,不改 beam、Trie、length penalty。

第一轮**不加入**:temporal decay bias、behavior gate、对比学习、tokenizer 改动、
decoder 端修改、辅助 next-item head。这些方向若 HI-GRAM v1 成功,可作为后续增强;
若 HI-GRAM v1 失败,依失败模式再决定下一步。

## 4. 对照组

| Arm | 模型 | 目的 |
|---|---|---|
| A | 原始 GRAM (repro) | 固定基线,用自己环境跑 3 seeds |
| B | HI-GRAM local-only | 判断局部窗口 attention 是否有效 |
| C | HI-GRAM global-only | 判断全局 attention 是否有效 |
| D | HI-GRAM (full) | 主 claim:local + global 层次化 |
| E | HI-GRAM late-fusion | 消融 fusion timing:同结构但在 decoder 前才 fuse |
| F | HI-GRAM w/o residual | 消融 α:直接替换 hidden 而非残差加法 |

A-F 使用相同数据划分、T5 规模(T5-small)、beam size、Trie、训练样本、评测脚本、
optimizer、seed 集合。

## 5. 实施与训练顺序

### P12-0:代码接口与 CPU smoke

- 在 `GRAM/src/model/gram.py` 的 `EncoderWrapper` 增加 `hi_gram_enabled` 分支;
- 在 `GRAM/src/model/gram_t5_config.py` 与 `GRAM/src/arguments.py` 增加相关 CLI/config 字段;
- 保证 `hi_gram_enabled=False` 时模型输出与原 GRAM 逐 bit 一致(或使用与原 GRAM 完全同一
  评测路径);
- CPU 单测:forward/backward finite;shape 匹配;masked attention 不泄漏 padding;
  当所有 items 都是同一个 item 时 local == global 输出(退化情况);α 有梯度。

### P12-1:Beauty 单 seed smoke

- 数据集 Beauty,seed 2023,从 T5-small 初始化(不用 GRAM 官方 checkpoint 作 warm start,
  避免混淆);
- 1-2 epoch 或 500 steps 小规模,只验证:loss finite、gradient finite、
  α 有更新(不接近 0 或 nan)、GPU 峰值显存 ≤ 25 GiB 单卡;
- 通过后冻结实现,进入 P12-2。

### P12-2:Beauty + Toys × {A, B, C, D} × 3 seeds 主实验

- 从官方 T5-small 初始化(与 GRAM 论文一致);
- 训练超参数与 GRAM `train_gram_beauty_single.sh` / `train_gram_toys_single.sh` 一致,
  仅当 A6000 显存吃紧时允许降 batch(记录实际值,A-D 保持一致);
- Seeds `2023/2024/2025`;
- 每 config × dataset × seed 单独输出目录、单独 checkpoint、单独 metrics summary;
- 全量 validation 每 epoch;test 不读取。

### P12-3:Beauty + Toys × {E, F} 消融

- E 与 F 只跑 seed 2023 单 seed,或按主实验通过 gate 后决定是否扩 3 seeds;
- 与主实验相同的训练配置。

### P12-4:分析实验

- Head/mid/tail item popularity 分层 R@10/N@10;
- 历史长度分组(短/中/长);
- α 的训练轨迹与最终值;
- 抽 3-5 用户可视化 global attention weights;
- (可选)window size ∈ {3, 5, 7, 10} 在 Beauty 单 seed 扫。

### P12-5:Confirmation

- 主实验 + 消融 + 分析全部完成,方法完全冻结后启动;
- 选择 Sports 或 Yelp 之一(启动前研究者授权);
- 3 seeds,与主实验完全相同配置;
- Confirmation 结果作为泛化性证据放论文,不作为方法选择依据。

## 6. 评测与记录

主指标(与 GRAM 论文一致):

- Recall@5、Recall@10、Recall@20;
- NDCG@5、NDCG@10、NDCG@20。

机制记录:

- α 训练轨迹与最终值;
- Local/global attention 层的输出范数,验证不是恒等映射;
- 每个 item pooling 后的 representation norm;
- 分项:token CE loss、gradient norm、参数量、峰值显存、训练 wall-clock、推理 wall-clock;
- Per-user prediction/rank;
- head/mid/tail 分层指标;
- 3 seeds 报告 mean ± std;
- Paired t-test:HI-GRAM (D) vs GRAM (A),`p < 0.05` 视为显著。

## 7. 初始决策规则

以下是主实验通过判定,不是 pilot 方向门:

1. HI-GRAM (D full) 在 Beauty 上 3 seeds 平均 NDCG@10 相对 A `>= +2%`(相对);且
2. HI-GRAM (D full) 在 Toys 上 3 seeds 平均 NDCG@10 相对 A `>= +1%`(相对);且
3. 至少一个 domain 的 paired t-test `p < 0.05`;且
4. 任一 seed 上 R@10 相对 A 不低于 `-2%`(相对);且
5. B 或 C 单独变体 R@10/N@10 不劣于 A(证明 attention 至少无害)。

失败处理:

- 若 gate 未通过:允许**一次**调整 hyperparameters(window size、layer count、α 初始化、
  learning rate schedule 之一)后再跑一次,记录为 `v2` run;
- 若 `v2` 仍未通过:停止 HI-GRAM 方向,不做 v1.1/v1.2/v1.3 邻近救援;
- 事后可做诊断分析(attention pattern、α 轨迹、gradient),但不写成新方案;
- 失败诊断可能触发写下一版结构方向的独立 plan,不在本 plan 内追加。

## 8. 资源、后台运行与状态管理规则

本节继承第九阶段的资源与状态管理协议,是第十二阶段所有 GPU 子实验的默认运行协议。
科学配置、科学退出状态与资源恢复状态必须分开记录。

### 8.1 CodeLlama 占位协议

- 每次正式 GPU 实验开始前,先确认 CodeLlama 在本次目标物理 GPU 上处于占位状态,并把
  实际 GPU 编号写入冻结配置和 `status.json`;
- runner 在 CPU 单测、语法、配置和输入完整性检查通过后,调用
  `tools/run_codellama.sh stop` 释放目标 GPU;
- 不论实验成功、科学门失败、程序非零退出、timeout、手动 `stop` 或预启动阻断,所有
  退出路径都必须尝试在同一物理 GPU 上恢复 CodeLlama;
- 恢复后必须检查 CodeLlama tmux 和运行状态,写入 `restored` 或 `failed_to_restore_resource`;
- CodeLlama 恢复成功不得覆盖实验本身的失败退出码;恢复失败也必须与科学结果独立报告。

### 8.2 30 GiB 显存租约

- CodeLlama 停止后,runner 必须等待目标 GPU 可用显存至少为 `30,720 MiB`;超过预设
  等待时间仍不满足时标记 `blocked`,不启动 workload;
- 正式运行期间,workload 实际占用与 `experiment/gpu_memory_lease.py` sidecar 占位合计
  保持 `30,720 MiB` 总租约,不在 workload 之外另外重复占用 30 GiB;
- workload/sidecar 的具体分配在实现后根据 smoke-test peak 冻结;sidecar 未成功进入
  `holding` 时不得启动正式训练;
- 运行期间每 5 秒记录 GPU index、used/free memory 和 utilization 到 `gpu_telemetry.csv`;
- 若 HI-GRAM 的实测单进程峰值超过 30 GiB(考虑到 A6000 单卡 20-30G 可用限制,超 30 GiB
  已属异常),必须先根据 smoke 结果单独修订租约与 batch size,不允许在正式运行中静默
  取消资源保护。

### 8.3 后台 runner 与用户接口

- 预计超过 20 分钟的实验必须在具名持久 `tmux` 会话中后台运行,不依赖当前终端
  或 Codex 会话存活;
- 第十二阶段 runner 实现后统一提供:

  ```bash
  bash experiment/phase12/run_phase12_hi_gram.sh start
  bash experiment/phase12/run_phase12_hi_gram.sh status
  bash experiment/phase12/run_phase12_hi_gram.sh stop
  ```

- `start` 只启动当前已授权的子实验(P12-1 smoke / P12-2 主实验 / P12-3 消融 /
  P12-4 分析 / P12-5 confirmation 分别独立授权);不自动启动后继 arm、
  读取 Sports/Yelp/test 或进入下一阶段;
- `status` 必须为只读操作,并显示 tmux 是否存活、当前 `status/stage/reason`、物理 GPU、
  runner/workload PID、CodeLlama 占位/恢复状态,以及最新日志摘要;
- **不需要研究者实时监控**:runner 后台跑,研究者通过 `status` 命令查看状态,依据 status
  联系 assistant 决定下一步骤(启动下一 arm、恢复 CodeLlama、诊断、进入下一阶段);
- `stop` 优先向 workload 发送 `TERM` 并走统一清理路径,不直接遗留 sidecar、telemetry 或
  CodeLlama 未恢复状态;
- 日志、`status.json`、telemetry 和结果必须持久化到具名 `artifacts/phase12/hi_gram/...`
  目录,不只写在 tmux pane 内。

### 8.4 启动、超时与失败规则

- 正式启动前通过 CPU 单测、Python compile、Bash syntax、JSON/config 检查,并冻结
  implementation、test、runner、config、输入数据(user_sequence / item ID / prompt)与
  parent checkpoint 的 SHA256;
- hard timeout 在完成小规模 smoke 后按实测耗时写入子实验配置(留 1.5× 余量);
  只有 hard timeout 可自动终止 workload;
- 非零退出、OOM、NaN/Inf、timeout、输出不完整或科学门未通过均**不自动重试**;
- 重试或 recovery 必须先保留原日志和状态,说明失败原因、允许改动的运行参数,并获得研究者
  明确授权;recovery 不改变科学配置、seed 或输入 checkpoint,只修复实现/日志/资源 bug;
- 完成后只保存结果并报告终态。**未经研究者明确要求,不自动实现/启动下一轮**,
  不读取 Sports/Yelp/test。

### 8.5 status.json 字段约定

每个子实验的 `artifacts/phase12/hi_gram/<sub_id>/status.json` 至少包含:

```json
{
  "sub_id": "p12_1_smoke_beauty_seed2023" ,
  "stage": "smoke" | "main" | "ablation" | "analysis" | "confirmation",
  "status": "pending" | "running" | "completed" | "failed" | "blocked",
  "reason": "…",
  "physical_gpu": 6,
  "codellama_state": "occupying" | "stopped" | "restored" | "failed_to_restore_resource",
  "workload_pid": 12345,
  "sidecar_state": "not_started" | "holding" | "released" | "failed",
  "started_at": "...",
  "ended_at": "...",
  "test_read": false,
  "sports_read": false,
  "yelp_read": false,
  "input_sha256": {"user_sequence": "…", "item_ids": "…", "prompt": "…", "config": "…"}
}
```

## 9. 冻结项(v0.1)

### 9.1 Backbone 与训练(与原 GRAM 一致)

- T5-small backbone (d_model=512, num_layers=6/6, num_heads=8);
- 训练超参数:与 GRAM 官方 `train_gram_beauty_single.sh` / `train_gram_toys_single.sh`
  保持一致(learning rate、warmup、weight decay、epochs、beam size、length penalty);
- 分层 lexical ID:使用 `rec_datasets/{Dataset}/item_generative_indexing_hierarchy_v1_*.txt`
  官方文件,不修改;
- Trie 约束 beam search:与 GRAM 一致;
- 数据集划分:官方 leave-last-two,validation = second-last,test = last。

### 9.2 HI-GRAM 新模块超参数(v0.1)

- Item pooling:masked mean over valid tokens
- Local window size:`W = 5`
- Local attention 层数:2 层 TransformerEncoderLayer(d_model=512, heads=4, ff=2048, dropout=0.1)
- Global attention 层数:2 层 TransformerEncoderLayer(d_model=512, heads=4, ff=2048, dropout=0.1)
- Fusion:residual bias + 可学习标量 `α`,初始化 0.1(具体是 sigmoid(param) 还是 raw scalar
  在 W1 实现时冻结)
- Item position embedding:可学习 `nn.Embedding(max_item_num + 1, d_model)`,不复用 T5 relative bias

### 9.3 数据集与 seed

- 主实验:Beauty, Toys
- Confirmation:Sports 或 Yelp(P12-5 授权时选一);Beauty/Toys/main 通过 gate 前一律不读
- Seeds `2023/2024/2025`
- Batch size:与 GRAM 官方脚本一致,若显存吃紧允许降(A-F 保持一致)

### 9.4 SHA256 锁

- 训练脚本、config、tokenizer 文件、item ID 文件、user sequence 文件、prompt 全部 SHA256 记录
- 每次 run 之前打印相关 SHA256 到 `status.json`

## 10. 实现要点

### 10.1 改动文件

| 文件 | 改动性质 | 估算行数 |
|---|---|---|
| `GRAM/src/model/gram.py` | 主要改动,`EncoderWrapper.__init__` 与 `forward` 新增 HI-GRAM 路径 | +300 |
| `GRAM/src/model/gram_t5_config.py` | 增加 `hi_gram_enabled`、`hi_gram_local_window` 等字段 | +30 |
| `GRAM/src/arguments.py` | 增加对应 CLI 参数 | +40 |
| `GRAM/src/main_generative_gram.py` | 透传 config,不改训练逻辑 | +10 |
| `experiment/phase12/run_phase12_hi_gram.sh` | 新 runner 脚本 | +200(参考 phase9) |
| `experiment/phase12/*.py` | 训练/评测调用入口(参考 phase9 结构) | +200 |
| CPU 测试文件 | 单测覆盖 forward/backward、mask 泄漏、degenerate case | +200 |
| **合计** | | **~1000 行** |

### 10.2 关键实现细节(W1 冻结前的开放项)

- `α` 是 raw scalar 还是 sigmoid(param)?
- item pooling 是 masked mean 还是 attention-pool?
- local window 用 causal mask 还是 symmetric window?
- item-position embedding 是否与 GRAM 原 EncoderWrapper `position_embedding` 分离?

以上四项在 W1 CPU 单测通过后、GPU smoke 启动前冻结,写入 v0.2 修订版。

### 10.3 参数量与显存估算

以 T5-small(d_model=512)+ Beauty(N≈21, L≈32)+ B=4 为例:

| 模块 | 参数量 | 前向峰值激活增量 |
|---|---|---|
| Local Attn (2 层) | ~4M | ~20 KB |
| Global Attn (2 层) | ~4M | ~20 KB |
| Item position embedding | ~10K | 忽略 |
| Fusion scale α | 1 | 忽略 |
| **合计** | **~8M**(GRAM ~60M 的 ~13%) | **< 100 MB(含反向)** |

**结论**:20-30G A6000 单卡完全容纳,batch size 与 GRAM 一致。

## 11. 与 11 阶段旧成果的关系

- **不作为 baseline 复现**:CF0/PCRF、GACR-v3 等不在本论文主表出现;
- **作为 motivation 侧证**:在 introduction 或 discussion 章节可引用
  "事后校准与融合方向的局限性"作为"why early fusion"的辅助论据
  (引用你自己的实验结果,注明"our prior experiments in a companion technical report");
- **代码保留**:CF0 分支代码保留在 `gram.py` 中,由 `cf0_enabled=False` 关闭,不影响 HI-GRAM。

## 12. 论文主表与消融(预设结构)

### 12.1 主表(Main Results)

| Method | Beauty R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | Toys R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SASRec | 引用 GRAM 论文 | | | | | | | | | | | |
| BERT4Rec | 引用 | | | | | | | | | | | |
| S3-Rec | 引用 | | | | | | | | | | | |
| P5 | 引用 | | | | | | | | | | | |
| TIGER | 引用 | | | | | | | | | | | |
| LC-Rec | 引用 | | | | | | | | | | | |
| LETTER | 引用 | | | | | | | | | | | |
| IDGenRec | 引用 | | | | | | | | | | | |
| ELMRec | 引用 | | | | | | | | | | | |
| **GRAM (repro)** | 自跑 3 seeds | | | | | | | | | | | |
| **HI-GRAM (Ours)** | 自跑 3 seeds | | | | | | | | | | | |

### 12.2 消融表

见 §4 表格 A-F。每 arm 报告 Beauty + Toys 上 3 seeds(E/F 视情况)的 R@10、N@10。

## 13. 阶段产物(P12-2 完成后至少保留)

- A/B/C/D 的配置、checkpoint 与统一指标 summary;
- per-user prediction/rank 记录;
- 分项 loss 与训练日志;
- α 训练轨迹;
- 完整性检查与结果报告;
- 一份基于实验数据的第十二阶段后续计划(若需要 confirmation 或 v2 tuning)。

## 14. 强制决策模板

```text
阶段:P1 直接完整训练(架构类,免 P0 单 seed calibration-only pilot)
唯一结构假设:encoder 阶段的 hierarchical cross-item attention 能改进推荐效果
固定 seed/cohort:seeds 2023/2024/2025;Beauty/Toys 官方 leave-last-two split
直接机制指标:HI-GRAM (D full) 主表 R/N@k;消融表 6 行(A-F);α 轨迹
最低有效信号:见 §7 gate 条件(Beauty N@10 ≥ +2%, Toys ≥ +1%,一域 p<0.05)
guardrail 非劣界:任一 seed 上 R@10 相对 GRAM (repro) ≥ -2%
通过后唯一下一步:Sports 或 Yelp 一次性 confirmation
失败后停止项:一次超参调整后仍不过,停止该方向,不做救援
禁止的邻近补丁:不允许 v1.1 换 loss、v1.2 加 gate、v1.3 换 pooling 等连续补丁
候选/特征缓存 SHA256:不适用(不复用旧候选)
Sports/Yelp/test read:主实验期 false / false / false;confirmation 阶段仅读一域
```

## 15. 当前状态

- Plan v0.1 已冻结提交,等待研究者审阅;
- P12-0 代码接口尚未实现;
- CodeLlama 状态与 GPU 分配未协商,W1 启动前需研究者指定 CodeLlama 迁移的物理 GPU;
- 实施顺序:研究者审阅并授权 v0.2 → 实现 P12-0 接口 → CPU smoke → GPU smoke → 冻结实现 →
  P12-2 主实验后台启动 → 研究者根据 `status` 决定下一步。

## 16. 讨论与开放问题

以下问题在 v0.1 中未闭环,W1 前可再讨论:

- 是否需要额外加 temporal decay bias(方案 A+T),或保持纯 hierarchical(当前 A+H);
- 是否在 W4 阶段增加 LETTER 或 ETEGRec 的复现作为额外 baseline;
- 论文 method 章节的示意图设计;
- 是否需要专门做一个"HI-GRAM 与 SASRec 差异"的说明段;
- α 是 raw scalar 还是 sigmoid(param)、pooling 是 mean 还是 attention;
- item-position embedding 与 GRAM 原 `position_embedding` 是否共享。
