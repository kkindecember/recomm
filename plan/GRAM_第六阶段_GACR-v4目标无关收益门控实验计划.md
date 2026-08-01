# GRAM 第六阶段：GACR-v4 目标无关收益门控实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run
- Created: 2026-08-01
- Verification Status: PREREGISTERED
- Version Label: `phase6_gacr_v4_target_free_gate_v1`
- Development Domains: Toys、Beauty
- Sports/Test: 封存
- Execution Device: 物理 GPU0
- Post-run Resource: CodeLlama 恢复并继续占用物理 GPU0

## 1. 依据与研究问题

冻结 GACR-v3 相对 GRAM 的 overall NDCG@10 和 Recall@10 在 Toys/Beauty 的 6 个
域-seed cells 全部为正，但并非所有指标均正向。弱点集中在 Toys：seed 2023 的
overall/tail Recall@50 为负，seed 2025 的 tail NDCG@10 与 Recall@10 为负。与此同时，
v3 的 spread attenuation 实际为 identity，CET+GACR 也没有超过单独 GACR。

研究问题：能否仅使用推理时可得的候选、base score 和冻结 residual 的 target-free
统计，预测哪些用户适合启用 residual，从而改善 Toys 的 tail/Recall@50 稳定性，并保留
GACR 的 Beauty 与 overall NDCG 增长？

## 2. 单一改动因素

GACR-v4 只增加一个 seed-specific 的用户级 logistic gate：

- 输入是 8 个 target-free 聚合特征：residual spread、平均绝对 residual、base top-2
  margin、GRAM 候选比例、GRAM/catalog overlap、catalog-only 比例、catalog-z 离散度、
  base-residual alignment；
- 输出是“本用户启用冻结 residual”的概率；
- 通过阈值则完整应用冻结 GACR-v3 residual，否则精确返回 GRAM 排序；
- threshold=`0` 对所有用户启用 residual，是 GACR-v3 的精确 identity control。

不修改 GRAM checkpoint、候选构造、原 6 维 residual 特征、residual 网络或 residual
checkpoint。门控训练标签只在 fit 用户上由冻结 v3 相对 GRAM 的 rank improvement/harm
产生；部署特征不含 target、target group、validation label、Sports 或 test 信息。

## 3. 数据隔离与对照

- Toys、Beauty 使用与 v3 相同的 training-user fit/calibration split；
- fresh validation 各 1024 用户；排除 GCDH training/validation、GACR-P0、v2、v3 和
  CET+GACR 组合 cohort；
- residual seeds：2023/2024/2025；
- 对照：`GRAM`、冻结 `GACR-v3`、`GACR-v4`；
- Sports/test 不读取。

## 4. 阈值选择与安全门

预注册 threshold 候选：`0/0.35/0.45/0.55/0.65/0.75/0.85`。每个域选择一个跨三
seed 共享阈值。

阈值必须在该域三个 calibration seed cells 全部满足：

1. overall Recall@10 不下降；
2. overall Recall@50 不下降；
3. tail NDCG@10 不下降；
4. tail Recall@50 不下降；
5. broad harm ≤ 1%。

在 eligible 阈值中最大化 mean overall NDCG@10；依次以更高 tail NDCG、较低最大
broad harm、较低阈值打破平局。threshold 0 保证至少可以退回冻结 v3。

## 5. Fresh-validation 决策门

只有以下条件同时成立才保留 v4：

1. Toys 三 seed mean overall NDCG@10 严格超过 v3；
2. 双域六 cell 宏平均 NDCG@10 严格超过 v3；
3. Beauty 三 seed mean overall NDCG@10 不低于 v3；
4. Toys mean tail NDCG@10 和 Recall@50 不低于 v3；
5. 所有域-seed 的 Recall@10/50 与 broad-harm 安全门通过；
6. 完整性门全部通过。

否则返回冻结 GACR-v3，只停止当前 gate 配置，不关闭 GACR 方向。CI 用于描述证据强弱，
不取代预注册点估计门。

## 6. 工程与运行协议

- 实现：`experiment/phase6/gacr_v4.py`；
- 实现 SHA256：`eacef4f0780990a911535c11ef5a40dc1b9bff0954d314f066bad529ceda6c96`；
- 配置：`artifacts/phase6/configs/gacr_v4_preregistered.json`；
- runner：`experiment/phase6/run_phase6_gacr_v4.sh`；
- 输出：`artifacts/phase6/gacr_v4/`；
- 运行：`bash experiment/phase6/run_phase6_gacr_v4.sh start`；
- 状态：`bash experiment/phase6/run_phase6_gacr_v4.sh status`；
- 物理 GPU0，启动门要求空闲显存 ≥ 30,720 MiB；GPU gate 最长等待 12 小时；
- 工作负载硬超时 8 小时，tmux 后台运行，每 5 秒记录 GPU0 telemetry；
- 启动前停止 CodeLlama；任何退出路径均恢复 `run_codellama.sh start 0`，恢复状态与实验
  exit code 分开记录；
- 失败不自动重试，结果完成后也不自动分析或启动下一实验。

## 7. 预期产物

- `artifacts/phase6/gacr_v4/status.json`
- `artifacts/phase6/gacr_v4/run.log`
- `artifacts/phase6/gacr_v4/gpu_telemetry.csv`
- `artifacts/phase6/gacr_v4/summary.json`
- `artifacts/phase6/gacr_v4/{Toys,Beauty}/gate_seed*.pt`
- `artifacts/phase6/gacr_v4/{Toys,Beauty}/{gacr_v3,gacr_v4}_seed*_per_user.csv`

## 8. 启动记录（2026-08-01）

- phase6 相关单元测试：`14 passed`；Python compile、Bash syntax、JSON 和 SHA preflight
  全部通过；
- tmux `gram_phase6_gacr_v4` 于 2026-08-01 12:28:07+08:00 启动；
- 启动时 CodeLlama 报告为未运行；runner 仍会在退出路径恢复到物理 GPU4；
- 启动后 GPU4 有三个既有 Python 进程，占用约 42,048 MiB、仅余 6,523 MiB，低于
  30,720 MiB 安全门；当前状态为 `waiting_for_gpu`，未终止未知进程、尚未启动科学
  workload；
- 该 GPU4 等待会话后来按研究者指示停止并迁移；查询命令仍为：
  `bash experiment/phase6/run_phase6_gacr_v4.sh status`。

## 9. GPU0 迁移 amendment（2026-08-01）

- 研究者要求将 GACR-v4 和实验后的 CodeLlama reservation 一并迁移到物理 GPU0；
- 迁移时原 GPU4 tmux 仍处于 `waiting_for_gpu`，`workload_pid=0`，没有科学计算或科学
  产物，因此该操作不构成失败实验的自动重试；
- 科学设计、cohort、checkpoint、residual、gate、阈值、指标、决策门和实现 SHA 均不变；
- GPU0 迁移检查时空闲约 47,798 MiB，满足 30,720 MiB 启动门；
- 新执行设备为物理 GPU0，退出后 CodeLlama 恢复并继续占用物理 GPU0。

## 10. 首次 GPU0 启动失败与工程修复（2026-08-01）

- GPU0 runner 于 2026-08-01 12:31:58+08:00 启动并通过显存门，但在候选生成前
  exit=`1`；没有产生训练、validation 或科学结果；
- 原因是新 standalone runner 漏设既有 phase6 runner 使用的仓库本地 Hugging Face
  cache 环境，离线 `t5-small` 加载失败；
- 退出路径已成功将 CodeLlama 恢复到物理 GPU0，状态为 `running gpu=0`；
- 工程修复仅补回
  `HF_HOME=$ROOT/.cache/huggingface` 与
  `TRANSFORMERS_CACHE=$ROOT/.cache/huggingface`；本地 cache 中已确认存在 `t5-small`
  config、tokenizer 和权重；
- 科学设计和 `gacr_v4.py` SHA 均未变化。按 no-auto-retry 协议，修复后不自动重启，
  等待研究者明确决定。

## 11. 研究者授权重启（2026-08-01）

- 研究者在查看首次 GPU0 无科学结果失败后明确回复“重启”；
- 重启前本地 `t5-small`、实现 SHA、parent checkpoint、6 个 residual、Bash/JSON 和
  CodeLlama GPU0 状态复核通过；
- 本次是显式授权的工程修复后 retry，不改变任何科学设计；仍使用物理 GPU0，退出后
  CodeLlama 恢复到物理 GPU0。
- retry tmux 于 2026-08-01 12:38:23+08:00 启动，CodeLlama 成功释放；科学 workload
  于 12:39:23+08:00 进入 `running`，PID=`3268018`，物理 GPU0；`run.log` 采用追加
  方式，文件前部保留首次无效启动的旧 traceback，不能据此误判当前 retry 状态。

## 12. 正式结果与决策（2026-08-01）

- 有效科学 workload 于 15:25:43+08:00 完成，exit=`0`；两域三 seed、12 份逐用户
  CSV、6 个 gate checkpoint 和 `summary.json` 完整生成；Sports/test 未读取；
- Toys 和 Beauty 都选择 threshold=`0`，gate application rate=100%，因此 v4 与冻结
  v3 在六个 validation cells 的逐用户 rank 和指标上精确一致；
- v4 相对 v3 的 Toys mean NDCG 与双域宏平均增量均为 0，未通过“严格超过 v3”的
  预注册保留门；正式决定为 `RETURN_TO_GACR_V3_STOP_V4_HARD_GATE`；
- 本次新 cohort 上，冻结 GACR 相对 GRAM 的 Toys/Beauty mean NDCG 分别为
  `+1.169%/+3.925%`，6/6 cell 为正；但 Toys mean overall/tail Recall@50 分别为
  `-0.130pp/-0.231pp`，仍需修复；
- 结果后诊断显示 gate probability 对改善/伤害用户仍有中等区分度，但 hard threshold
  删除了太多改善用户。下一主实验计划为 GACR-v5 连续 soft benefit weighting；
- 自动恢复 CodeLlama 曾因继承错误的 T5 cache 路径失败；已显式区分 T5 与 CodeLlama
  的 Hugging Face cache，并在 GPU0 人工恢复成功。该工程事件不影响科学结果。

详细报告见
`report/第六阶段/GRAM_第六阶段_GACR_v4结果与验证报告.md`；下一实验计划见
`plan/GRAM_第六阶段_GACR-v5目标无关软收益加权实验计划.md`。
