# GRAM 第六阶段：GACR-v7 全量指标对齐残差训练实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-02
- Verification Status: PLANNED_NOT_IMPLEMENTED
- Version Label: `phase6_gacr_v7_full_fit_metric_aligned_loss_v1`
- Development Domains: Toys、Beauty
- Sports/Test: 封存
- Device: 物理 GPU0，30 GiB 总显存租约

## 1. 决策依据与研究问题

v6 全量 fit 相对 GRAM 的 six-cell macro overall NDCG@10 为 +2.720%，相对 v3 的宏增量为
+0.559%，但 Beauty 的 mean tail NDCG@10、overall Recall@50 和 tail Recall@50 均略低于 v3，
因此未通过保留门。原 hinge loss 只针对 target 与最高分负例，未区分 NDCG@10 和 Recall@50
截断线附近的排序交换。

研究问题：**在 v6 的全量 fit records 与所有其它冻结项不变时，仅以截断敏感的加权 pairwise
logistic loss 替换 hinge loss，能否保留 NDCG 增益并恢复 Beauty 的 tail/Recall@50 安全性？**

## 2. 单一改动因素

仅替换 residual loss。对每个 covered fit record，按冻结 GRAM base score 的 stable rank 得到
target 位置 `r_t` 与每个负例位置 `r_j`：

`D10(r)=1/log2(r+1)`（仅 `r<=10`，否则 0）；`R50(r)=1`（仅 `r<=50`，否则 0）。

`w_j=|D10(r_t)-D10(r_j)|+0.25*|R50(r_t)-R50(r_j)|`。

令 `s_i=base_i+residual_i`，每 record loss 为
`sum_j w_j*softplus(s_j-s_t)/sum_j w_j`；零权重 record 不计入该 group 均值，head/tail 两个
group 的有效 record 均值等权平均。权重只来自冻结训练 record 的 base rank；`0.25` 固定且不
搜索。该定义来自已在 v6 前形成、但因训练规模优先而未实现的备选计划；本次把它作为独立 v7
实验，不与数据规模变化混合。

## 3. 严格冻结项

- GRAM C1 checkpoint、候选构造、stable tie-break、6 维特征与
  `BoundedResidualRanker(6,16,bound=0.2)`；
- v6 的全部 fit-split records、80/20 fit/calibration 用户隔离、calibration 128 head + 128 tail；
- AdamW、lr=`0.01`、weight decay=`0.01`、gradient clip=`10`、30 个固定 full-batch steps；
- residual scale=`1.0`、seeds 2023/2024/2025、GRAM backbone optimizer steps=`0`；
- 只读 v3/v6 checkpoint 作为对照；不使用 v4/v5 gate 或 multiplier，不改模型容量、候选、步数或
  NDCG/Recall 权重。

## 4. 数据与对照

- Toys/Beauty；每域 1024 fresh validation 用户，salt：
  `phase6-gacr-v7-full-fit-metric-loss-development-v1`；
- 排除 GCDH train/validation，及 GACR-P0、v2、v3、v4、v5、v6 的全部 historical fresh cohort；
- 对照：冻结 GRAM、冻结小样本 GACR-v3、全量 hinge GACR-v6、全量指标对齐 GACR-v7；
- Sports/test 禁读，fresh validation label 不得参与训练、loss 权重或配置选择。

## 5. 校准安全门

v7 没有可调参数；calibration 只作 fail-closed 检查。每个域-seed 必须满足：overall Recall@10/
Recall@50、tail NDCG@10/tail Recall@50 均不低于 GRAM，broad harm ≤1%，loss/gradient/checkpoint
均 finite。任一失败即不进入 fresh validation，不调 `0.25`、步数或 loss。

## 6. Fresh-validation 保留门

只有全部满足才以 v7 替换 v3：

1. Toys mean overall NDCG@10 严格高于 v3，Beauty mean 不低于 v3；
2. six-cell macro overall NDCG@10 严格高于 v3，且至少 5/6 cell v7-v3 为正；
3. Toys、Beauty 的 mean tail NDCG@10、overall Recall@50、tail Recall@50 均不低于 v3；
4. 两域四项标准指标均不低于 GRAM；所有 cell overall Recall@10/50 不低于 GRAM，broad harm ≤1%；
5. v7 相对 v6 必须使 Beauty 上述三项 safety 指标均不低于 v6，且至少一项严格提高；
6. cohort/lineage/Sports-test 封存与 30 GiB 租约完整性全部通过。

失败则为 `KEEP_GACR_V3_STOP_FULL_FIT_METRIC_LOSS_V1`；不得结果后调 `0.25` 或开始相邻 loss。

## 7. 产物、资源与停止

- implementation/test/runner/config/output 分别为 `gacr_v7.py`、`test_gacr_v7.py`、
  `run_phase6_gacr_v7.sh`、`gacr_v7_preregistered.json`、`artifacts/phase6/gacr_v7/`；
- 运行前冻结所有 SHA256 与输入 checkpoint SHA；测试至少覆盖权重公式、零权重、head/tail 平衡、
  v3/v6 controls、fresh cohort 排除和 Sports/test 禁读；
- GPU0 总租约=`30,720 MiB`：预声明 workload peak=`24,576 MiB`，sidecar=`6,144 MiB`，使用
  `experiment/gpu_memory_lease.py` 持续持有至 workload 退出；
- 停止 CodeLlama、通过显存门后在具名 tmux 启动；hard timeout=`36`小时；所有退出路径恢复
  CodeLlama；非零 scientific exit 或 timeout 不自动重试；
- 结果完成后仅写结果，未经研究者请求不自动分析、启动后继或读取 Sports/test。
