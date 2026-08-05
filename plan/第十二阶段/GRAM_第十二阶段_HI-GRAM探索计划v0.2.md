# GRAM 第十二阶段 HI-GRAM:层次化跨 item 早融合探索计划(v0.2 探索版)

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan (exploration)
- Created: 2026-08-05
- Verification Status: `EXPLORATION_MODE`
- Version Label: `phase12_hi_gram_exploration_v0.2`
- Experiment ID: `GRAM_PHASE12_HI_GRAM_EXPLORATION_V1`
- Supersedes: `phase12_hi_gram_main_design_v0.1`(v0.1 归档为投稿版规格,当前不用)
- Parent Governance: `plan/GRAM_后续结构性方向分阶段实验治理规则.md`(修订版:探索模式免 3 seeds / 免正式 gate)

## 0. 阶段定位与模式声明

**当前处于探索模式(EXPLORATION),不是投稿实验模式**。

目标:**找到能行的创新点**,不是"跑完整可发论文实验"。允许:

- 单 seed(seed 2023),不跑 3 seeds
- 双域(Beauty + Toys)
- v1 效果不理想 → 改进方案 → v2 再试,允许连续几次改进
- 调整超参、pooling 方式、fusion 位置、层数等
- 不需要 paired t-test / 95% CI / confirmation domain / 严格 preregistered gate

进入投稿实验模式(CONFIRMATION)的门槛(**不是当前门槛**,是未来触发条件):

- 探索模式跑到某个 v 版本,Beauty 或 Toys 至少一域 N@10 相对 GRAM 单 seed baseline 相对提升 ≥ 3%
- 双域都不出现 R@10 相对退化 > 2%
- 达标后再另写 v1.0 CONFIRMATION plan,启动 3 seeds + Sports/Yelp confirmation

## 1. GRAM 对照(直接引用第一/二阶段现成结果)

**不再重跑 GRAM baseline**,直接用你已经复现好的单 seed 结果:

| Dataset | Seed | Best Epoch | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|---|---|
| Beauty | 2023 | 25 | 0.06327 | 0.08890 | 0.12158 | 0.04422 | 0.05246 | 0.06069 |
| Toys | 2023 | 30 | 0.07119 | 0.09530 | (未记录,可补) | 0.05140 | 0.05919 | (未记录,可补) |

来源:
- Beauty:`report/第一阶段/GRAM_第一阶段_Beauty_最佳Checkpoint测试报告.md`
- Toys:`report/第二阶段/GRAM_第二阶段_Toys_阶段C_D与最终结论.md`

**这就是探索期唯一对照**。HI-GRAM 每个 v 版本只需要跑自己,与这两行数字直接比较。

## 2. 核心方法:HI-GRAM(Hierarchical Interaction GRAM)

在 `EncoderWrapper.forward` 拿到 `last_hidden_states = (B*N, L, D)` 之后,插入两级
cross-item attention 早融合:

1. Masked-mean item pooling → `(B, N, D)`
2. Local Window Attention(W=5,只 attend 到 `[i-W+1, i]`)→ `(B, N, D)`
3. Global Attention(所有 N 个 item)→ `(B, N, D)`
4. Residual bias fusion:`hidden = hidden + α · (global - pool).expand(...)`

保留 GRAM 全部其他组件(lexical ID / Trie / late-fusion decoder / token CE loss)。

细节(v1 用这套默认,v2/v3 可调):
- Local/Global 各 2 层 TransformerEncoderLayer,d_model=512,heads=4,ff=2048,dropout=0.1
- α raw scalar,init 0.1(如果 v1 α 掉到 0,v2 换 sigmoid 参数化)
- Item position embedding 独立可学习,不共享 T5 relative bias
- Local window symmetric(不用 causal),v2 可试 causal

## 3. 探索模式实验矩阵

**每个 v 版本 = 一次跑 Beauty + Toys 各一个 seed**。允许连续 v1 → v2 → v3。

### v1(初始尝试)
- Local(2 层,W=5)+ Global(2 层)+ residual α=0.1 init
- 目标:确认方向不完全失效

### 若 v1 效果不佳(N@10 相对 GRAM < +1%),可能的 v2 尝试(选一个):
- v2a:α 换成 sigmoid(param),init sigmoid⁻¹(0.5)
- v2b:去掉 local,只用 global(简化)
- v2c:去掉 residual,直接替换 hidden(fusion_scale=1 fixed)
- v2d:换 pooling(attention pool 代替 mean pool)
- v2e:增大 local/global 层数到 4/4

### 若 v2 依然不行,v3 允许更结构性的改动:
- v3a:引入 temporal decay bias(A+T 补丁)
- v3b:引入 user prompt conditioning(A+B 补丁)
- v3c:改 fusion 位置(在 T5 encoder 中间层就 fuse,而不是最后)
- v3d:决定放弃 HI-GRAM,回到讨论,选下一个方向

**每个 v 版本前**,我会先写一份 ≤ 1 页的 delta 说明(改了什么、期望什么),不写完整 plan。

## 4. 每个 v 版本的最小实验清单

每次 v 版本跑完至少产出:

- Beauty seed 2023:R@5, R@10, N@5, N@10(其他指标可选)
- Toys seed 2023:R@5, R@10, N@5, N@10
- 与 §1 GRAM 表的绝对差与相对差
- α 训练轨迹(最终值)
- 训练 loss 曲线是否正常收敛
- 单卡峰值显存是否 ≤ 25 GiB
- 训练 wall-clock 时间

**不需要**:多 seed、消融、head/tail 分层、显著性、confirmation 数据集。

## 5. 决策规则(轻量化)

每次 v 版本跑完的三种可能:

| 结果 | 决策 |
|---|---|
| Beauty 或 Toys 至少一域 N@10 相对提升 ≥ 3%,且双域 R@10 不劣化 > 2% | **达到 CONFIRMATION 门槛**,另写 v1.0 CONFIRMATION plan,启动 3 seeds |
| 单域小幅提升(0-3%)或双域方向不一 | 记录,写 v(n+1) delta 继续 |
| 单域或双域显著退化 / loss 不收敛 / α 学到 0 | 记录诊断,写 v(n+1) delta 或换方向 |

允许 v1 → v2 → v3 → v4,最多试到 v5;仍不过则换方向讨论。

## 6. 改动文件与工程量

| 文件 | 改动 | 估算行数 |
|---|---|---|
| `GRAM/src/model/gram.py` | `EncoderWrapper.__init__` + `forward` 增加 HI-GRAM 路径 | ~250 |
| `GRAM/src/model/gram_t5_config.py` | 增加 `hi_gram_*` 字段 | ~20 |
| `GRAM/src/arguments.py` | CLI 参数 | ~30 |
| `experiment/phase12/run_phase12_hi_gram.sh` | runner 骨架(可从 phase9 复制改造) | ~150 |
| `experiment/phase12/train_hi_gram.py` | 训练入口(复用 `main_generative_gram.py`) | ~80 |
| CPU 单测 | forward/backward、mask、degenerate case | ~150 |
| **合计** | | **~700 行** |

## 7. 资源、后台与状态协议(继承第九阶段,不放松)

即使是探索模式,以下硬性资源协议**全部保留**:

### 7.1 CodeLlama 占位
- 实验前后 CodeLlama 必须在目标物理 GPU 占位
- runner 通过 `tools/run_codellama.sh stop` 释放,退出后必须恢复
- 恢复状态写入 `restored` / `failed_to_restore_resource`,与实验退出码独立

### 7.2 30 GiB 显存租约
- workload + `experiment/gpu_memory_lease.py` sidecar 合计 30,720 MiB
- 每 5s 记 `gpu_telemetry.csv`
- sidecar 未 `holding` 不启动正式训练

### 7.3 tmux 后台
- 具名 tmux 后台运行,不依赖终端存活
- runner 提供 `start / status / stop`

### 7.4 status.json 与联络接口
- 我(assistant)**不主动 poll**
- 你运行 `status` 查看进度,再联系我下一步
- status.json 字段:sub_id、stage、status、reason、physical_gpu、codellama_state、workload_pid、sidecar_state、started_at、ended_at、test_read、sports_read、yelp_read、input_sha256

### 7.5 禁止自动重试
- 非零退出、OOM、NaN/Inf、timeout 一律不自动 retry
- 需要你授权

### 7.6 SHA256 锁 & 封存
- 训练脚本、config、tokenizer、item ID、user_sequence、prompt 全部记录 SHA256
- Test 全程不读;Sports/Yelp 探索期不读(达到 CONFIRMATION 门槛后另议)

### 7.7 artifacts 目录
- 所有产物写入 `artifacts/phase12/hi_gram/<v_id>_<dataset>_<seed>/`
- 每个 v 版本 × 数据集独立目录

## 8. 时间预算

- v1 实现:2-3 天(参考 §6 工程量)
- v1 Beauty + Toys 训练:每域约 GRAM 原训练时间 × 1.1(因为多了 8M 参数),
  单卡 A6000 上 Beauty 约几小时,Toys 类似。两域顺序或并行由 GPU 情况决定
- v1 分析 + 决策:半天
- 若 v1 通过 CONFIRMATION 门槛:直接进 3 seeds
- 若 v1 未通过:v2 delta 半天写、v2 实现半天到 1 天、v2 训练半天到 1 天、决策半天
- **总体探索期预算**:2-3 周,允许 v1-v4 迭代

## 9. 与旧 plan v0.1 的关系

v0.1 (`GRAM_第十二阶段_HI-GRAM主线设计v0.1.md`) 是**投稿版规格**,包含:
- 3 seeds、双域主表、6 行消融、Sports/Yelp confirmation
- Paired t-test、SHA256 全锁、正式 preregistered gate

**当前不用 v0.1**,归档保留。**只有当探索模式跑到某个 v 版本达到 CONFIRMATION 门槛
(见 §0),才写 v1.0 CONFIRMATION plan,基于 v0.1 骨架修订**。

## 10. 当前状态与下一步

### 已完成(2026-08-05)
- ✅ HI-GRAM v1 代码实现(gram.py EncoderWrapper + arguments.py + main_generative_gram.py)
- ✅ CPU 单测 7/7 过(experiment/phase12/test_hi_gram_encoder.py)
- ✅ Runner 实现(experiment/phase12/run_phase12_hi_gram.sh,GPU6 + recomm CodeLlama 工具)
- ✅ smoke_beauty 完成(2 分钟,pipeline 端到端跑通,peak allocated 8 GiB,CodeLlama 已 restore)

### 正在运行
- ⏳ **beauty_v1**:2026-08-05 19:46:51 启动,Beauty 30 epochs seed 2023,预估 5-8 小时
- Tmux 会话:`gram_phase12_hi_gram_beauty_v1`
- 输出目录:`artifacts/phase12/hi_gram/beauty_v1/`

### 未启动
- toys_v1(等 beauty_v1 完成后再启动,GPU6 单卡顺序)

### 恢复本任务的命令
下次续接时用户可以先跑:

```bash
# 1. 看 beauty_v1 是否完成
bash /mnt/18T/jiangtangyunzhi/projects/recomm/experiment/phase12/run_phase12_hi_gram.sh status beauty_v1

# 2. 看 CodeLlama 状态(应该是 restore 到 GPU6 或者训练中被停了)
cat /mnt/18T/jiangtangyunzhi/projects/recomm/.runtime/codellama/status.txt

# 3. 看 beauty_v1 的最终结果
tail -100 /mnt/18T/jiangtangyunzhi/projects/recomm/artifacts/phase12/hi_gram/beauty_v1/run.log
grep -E "test hit@10|test ndcg@10|alpha|fusion_scale" \
  /mnt/18T/jiangtangyunzhi/projects/recomm/artifacts/phase12/hi_gram/beauty_v1/run.log | tail -30
```

### 关键 baseline(直接引用第一阶段结果对比)
- GRAM Beauty seed 2023 epoch 25: **R@10=0.08890, N@10=0.05246**
- HI-GRAM 目标:N@10 相对提升 ≥ 3% → 目标 N@10 ≥ 0.05403

### 决策规则回顾
- 提升 ≥ 3% → 进入 CONFIRMATION 模式(3 seeds + Sports 或 Yelp confirmation)
- 提升 0-3% → 记录,写 v2 delta(见 §3)
- 退化 / loss NaN → 诊断,写 v2 或换方向

## 11. 实施决策(已冻结,2026-08-05)

- **目标物理 GPU**:**GPU6**(CodeLlama 常驻,单卡顺序跑,不做迁移;2026-08-05 修订自最初的 GPU7)
- **CodeLlama 协议**:实验前 `bash /mnt/18T/jiangtangyunzhi/projects/recomm/tools/run_codellama.sh stop` 释放 GPU6;实验结束后 restore 到 GPU6。使用 **recomm 项目内的** tool(不是 UnitTest 那个),state 写入 `recomm/.runtime/codellama/status.txt`,你可以直接读这个文件看状态
- **T5 初始化**:HuggingFace 官方 `t5-small`,与 GRAM 论文一致
- **Warm start**:**不使用** GRAM 已训 checkpoint;从 T5-small 冷启动,与 §1 baseline 口径一致
- **数据集顺序**:**GPU7 单卡顺序跑**,Beauty 先(与 §1 GRAM Beauty best epoch 25 口径匹配),Toys 后。**不并行**
- **实验前后 CodeLlama**:全程占位/恢复到 GPU7,不迁移

## 12. 开放问题(v1 实现前需要在代码 review 时最终敲定)

- `α` 是 raw scalar 还是 sigmoid(param):v1 用 raw scalar init 0.1,若训练不稳 v2 换 sigmoid
- pooling 是 masked mean 还是 attention pool:v1 用 masked mean(简单)
- local window 是 symmetric `[i-W, i+W]` 还是 causal `[i-W+1, i]`:v1 用 causal(推荐场景更合理)
- item-position embedding 是否共享 GRAM 原 `position_embedding`:v1 独立,不共享
