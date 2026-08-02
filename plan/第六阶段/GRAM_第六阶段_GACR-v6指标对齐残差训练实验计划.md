# GRAM 第六阶段：GACR-v6 指标对齐残差训练实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-01
- Verification Status: SUPERSEDED_BEFORE_IMPLEMENTATION
- Version Label: `phase6_gacr_v6_metric_aligned_residual_loss_v1`
- Development Domains: Toys、Beauty
- Sports/Test: 封存
- Intended Device: 物理 GPU0
- Post-run Resource: CodeLlama 恢复并继续占用物理 GPU0

> 2026-08-01 研究者确认当前 GACR 仅在小样本 residual 上训练后，决定优先验证训练规模。
> 本指标对齐 loss 方案未实现、未预注册、未运行，后移为备选；当前 v6 改为全量 residual
> 训练规模实验，见 `plan/GRAM_第六阶段_GACR-v6全量残差训练实验计划.md`。

## 1. 决策依据与研究问题

冻结 GACR residual 在 v3/v4/v5 三批互斥 fresh development cohort 上取得 18/18 overall
NDCG 正向点估计；核心方向应继续。与此同时，v3 residual-spread 衰减、v4 hard gate、v5
soft weighting 都选择完整 residual identity，说明继续调 residual 之后的门控或缩放缺少依据。

现有 residual 训练目标是 target 对单个最高分负例的 margin hinge。它不区分交换发生在
NDCG@10 或 Recall@50 截断线内外，也没有利用其余负例。研究问题是：**只把该损失替换为
指标截断敏感的加权 pairwise logistic loss，能否严格超过冻结 GACR-v3，同时保持 Toys
和 Beauty 的安全性？**

## 2. 单一改动因素

只修改 residual 的训练损失。对每个 covered fit record，按冻结 GRAM base score 得到 target
位置 `r_t` 和每个负例位置 `r_j`，定义：

`D10(r) = 1/log2(r+1), if r<=10; otherwise 0`

`R50(r) = 1, if r<=50; otherwise 0`

`w_j = |D10(r_t)-D10(r_j)| + 0.25*|R50(r_t)-R50(r_j)|`

令 `s_i=base_i+residual_i`，单用户损失为：

`L_record = sum_j w_j * softplus(s_j-s_t) / sum_j w_j`

`sum_j w_j=0` 的 record 不进入该步均值。最终 loss 仍为 head records 均值与 tail records
均值的等权平均。权重只由训练 record 的冻结 base rank 计算；`0.25` 在运行前固定，不搜索。

该定义相当于按 target 与负例交换对 NDCG@10 和 Recall@50 的潜在影响加权：截断线附近的
错误得到更大梯度，远离两个评价区间且不影响指标的 pair 不占训练预算。

## 3. 严格冻结项

- parent：GRAM C1 checkpoint 与候选生成器全部冻结
- 候选：generator top-50 与 catalog top-50 的 union，构造和 stable tie-break 不变
- residual 模型：6 维特征、`BoundedResidualRanker(6,16,bound=0.2)` 不变
- 初始化 seeds：2023/2024/2025；zero-output identity initialization 不变
- optimizer：AdamW，lr=`0.01`、weight decay=`0.01`、30 fixed steps、gradient clip=`10`
- fit/calibration 样本量、head/tail 平衡、数据准备、部署 residual scale=`1.0` 不变
- 不使用 v4 gate 或 v5 multiplier；它们只作为已终止因素保留 lineage
- backbone optimizer steps 必须为 0

因此 v6 与原 GACR 的唯一因果差异是 residual loss；不得同时扩大模型、改特征、改候选、
改训练步数或搜索 NDCG/Recall 权重。

## 4. 数据隔离

- fit/calibration：沿用 GACR-v2/v3 已登记 split，二者用户 overlap 必须为 0
- fresh validation：每域 1024 用户，salt 固定为
  `phase6-gacr-v6-development-v1`
- 必须排除 GCDH training/validation、GACR-P0 及 v2/v3/v4/v5 的所有 validation cohort
- training seeds：2023/2024/2025
- Sports/test 禁止读取；fresh validation label 不得参与训练或模型选择

## 5. 校准安全门

v6 没有可调超参数，calibration 只执行 fail-closed 安全检查。每个域-seed cell 必须满足：

1. overall Recall@10 不低于 GRAM；
2. overall Recall@50 不低于 GRAM；
3. tail NDCG@10 不低于 GRAM；
4. tail Recall@50 不低于 GRAM；
5. broad harm ≤ 1%；
6. loss、梯度和 checkpoint 全部 finite。

若任一 cell 失败，终止 v6 fresh validation，正式返回冻结 v3，并报告失败 cell；不得在结果后
调整 `0.25`、步数或 loss。若全部通过，才进入预注册 fresh validation。

## 6. Fresh-validation 对照与保留门

固定对照：`GRAM`、冻结 `GACR-v3`、新训练 `GACR-v6`。只有以下条件全部成立才保留 v6：

1. Toys 三 seed mean overall NDCG@10 严格超过 v3；
2. Beauty 三 seed mean overall NDCG@10 不低于 v3；
3. 六个域-seed cell 宏平均 overall NDCG@10 严格超过 v3；
4. Toys mean tail NDCG@10、overall Recall@50、tail Recall@50 均不低于 v3；
5. Beauty mean tail NDCG@10、overall Recall@50、tail Recall@50 均不低于 v3；
6. 六个 cell 的 overall Recall@10/50 均不低于 GRAM，broad harm ≤ 1%；
7. 至少 5/6 cell 的 v6-v3 overall NDCG 点估计为正，且没有 cell 低于 v3 超过 1%；
8. 完整性门全部通过。

若任一条件失败，决定为 `RETURN_TO_GACR_V3_STOP_V6_METRIC_LOSS_V1`。CI 用于描述不确定性，
不替换上述预注册点估计门，也不得选择性忽略某个域。

## 7. 必须报告的分析

- overall/head/tail 的 NDCG@10、Recall@10、Recall@50
- v6 相对 GRAM、相对冻结 v3 的逐 seed、域均值与六 cell 宏平均
- paired-user bootstrap 95% CI；同时报告绝对 delta，避免只看相对百分比
- changed-user coverage、broad harm、union coverage
- 每域-seed 有效 loss record 数、非零 `w_j` pair 数、NDCG 与 Recall 权重贡献比例
- loss 首末值、gradient norm、checkpoint SHA 和 parent SHA 前后值
- v3/v4/v5 三批历史结果只作上下文，不并入 v6 的模型选择

## 8. 完整性与停止条件

- 运行前生成并冻结 preregistered JSON、实现/test/runner SHA 与所有输入 checkpoint SHA
- 单元测试必须覆盖权重公式、零权重 record、head/tail 平衡、无 target 部署、fresh-cohort
  排除、v3 control 和 Sports/test 禁读
- 任何 NaN/Inf、cohort overlap、checkpoint lineage 不一致或 backbone 更新均使实验无效
- 校准安全门失败、8 小时 timeout 或非零 scientific exit 后不自动重试
- 不读取 Sports/test，不自动开始后继实验

## 9. 计划产物

- `experiment/phase6/gacr_v6.py`
- `experiment/phase6/test_gacr_v6.py`
- `experiment/phase6/run_phase6_gacr_v6.sh`
- `artifacts/phase6/configs/gacr_v6_preregistered.json`
- `artifacts/phase6/gacr_v6/summary.json`
- `artifacts/phase6/gacr_v6/{Toys,Beauty}/{gacr_v3,gacr_v6}_seed*_per_user.csv`

当前仅完成计划，**尚未实现、预注册或启动**。

## 10. 资源协议

- 物理 GPU0；启动前空闲显存门 ≥ 30,720 MiB
- 预计科学 workload 3–4 小时，硬超时 8 小时
- 具名 tmux、持久化日志、status 与 GPU telemetry
- 启动前停止 GPU0 CodeLlama；任何退出路径都恢复到 GPU0
- CodeLlama 恢复显式使用 `/home/jiangtangyunzhi/hf_cache` 下的三个 cache 环境变量
- 科学 exit 与资源恢复状态分开记录

## 11. 结果解释边界

Toys/Beauty 已被用于多轮开发，v6 即使通过也只能成为更强的 development incumbent，不能
宣称最终泛化成立。只有方法冻结后，在未见域或一次性封存 test 上按单一确认协议复现，才可
升级为 confirmatory 结论。
