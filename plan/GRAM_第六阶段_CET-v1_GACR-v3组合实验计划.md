# GRAM 第六阶段：CET-v1 × GACR-v3 组合实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-07-31
- Verification Status: PREREGISTERED
- Version Label: `phase6_cet_v1_x_gacr_v3_v1`
- Development Domains: Toys、Beauty
- Confirmation Domain: Sports（封存）
- Test: 封存
- Execution Device: 物理 GPU6

## 1. 研究问题与假设

研究问题：CET-v1 的证据一致性 backbone 与 GACR-v3 的冻结残差排序是否互补，
使组合方法在 Toys 和 Beauty 上同时超过两个单组件？

预注册假设：若 CET-v1 改善的 backbone 分数与 GACR-v3 的候选内残差信号互补，
`CET-v1+GACR-v3` 的宏平均和每域 NDCG@10 均应严格高于 `CET-v1` 与
`GACR-v3`。

## 2. 四组配对对照

| 方法 | Backbone | Residual |
|---|---|---|
| `GRAM` | 冻结 GCDH-P0 C1 | 无 |
| `CET-v1` | 冻结 CET C2 final epoch | 无 |
| `GACR-v3` | 冻结 GCDH-P0 C1 | 冻结 GACR-v3，budget 0.4 |
| `CET-v1+GACR-v3` | 冻结 CET C2 final epoch | 同一冻结 GACR-v3 |

四组使用完全相同的用户、target、候选构造、特征 schema 和评估代码。不训练
backbone，不重训 residual，不搜索参数。唯一改动是将 GACR-v3 的 parent backbone
从冻结 GRAM/GCDH 替换为冻结 CET-v1。

## 3. Cohort 与封存

- Toys、Beauty 各 1024 个 fresh development validation 用户；
- 排除 GCDH training/validation、GACR-P0、GACR-v2、GACR-v3 以及所有可用的 CET
  development cohort；
- Toys user SHA256：`848c387a184ca2a446e1b0d52138a1135436f7b4690657b83173552ce2c4bf63`；
- Beauty user SHA256：`fc6fdd21b6f4f3e420161abb18473c4714a6ca66c220f8b11f3256227e6f4ef2`；
- cohort 选择只使用 salted user ID 顺序，不使用 target 或模型结果；
- Sports 和 test 不读取。

## 4. 评估和决策门

- 主指标：overall NDCG@10；
- 次指标：Recall@10、Recall@50、tail NDCG@10、changed-user coverage、broad harm；
- 三个冻结 residual seeds：2023/2024/2025；
- 用户级配对 bootstrap 10,000 次，CI 只描述不确定性，不作探索期一票否决。

只有以下条件同时成立才保留组合：

1. 组合的双域宏平均 NDCG@10 严格高于 CET-v1 和 GACR-v3；
2. 组合在 Toys 和 Beauty 各自的三 seed mean NDCG@10 都严格高于两个单组件；
3. 每个域-seed cell 的 Recall@10 相对 GRAM 不低于 `-0.2pp`，broad harm 不高于 1%。

否则决策为 `RETURN_TO_STRONGER_SINGLE_METHOD`，回到更强的冻结单方法，不为保留组合而
事后改参数。

## 5. 完整性与执行

- 启动前校验两域 GRAM/CET-v1 checkpoint 和 6 个 residual SHA256；
- 运行前后 checkpoint SHA 必须不变；optimizer steps = 0；
- 实现 SHA256：`4bb97a52dc73cbe4f86b4191fbc425821db00c36f540258dfd3c3f31116f6313`；
- 配置：`artifacts/phase6/configs/cet_gacr_v1_preregistered.json`；
- 执行：`bash experiment/phase6/run_phase6_cet_gacr_v1.sh start`；
- 查询：`bash experiment/phase6/run_phase6_cet_gacr_v1.sh status`；
- 使用具名 tmux 会话和 GPU6 telemetry；退出路径恢复 CodeLlama；任何失败不自动重试。

## 6. 运行后更新（2026-08-01）

- 本次运行在 `21,600` 秒硬超时处以 exit `124` 终止；Toys 四组和三个 seeds 完整，
  Beauty 只执行到 GRAM arm `192/1024`，总 `summary.json` 未生成；
- Toys 三 seed mean NDCG@10：GRAM=`0.070871`、CET-v1=`0.067475`、
  GACR-v3=`0.071839`、组合=`0.069886`；组合比 GACR-v3 低 `2.718%`，且 3/3 seeds
  均更低；
- 因预注册要求每个域的组合都严格超过两个单组件，Toys 结果已经使保留门不可达，决定为
  **`REJECT_COMBINATION_RETURN_TO_GACR_V3`**；Beauty 缺失意味着不能报告完整双域宏平均；
- 不原样重跑全实验来改变组合决定。若论文完整性需要，可另行预注册 Beauty-only
  completion，只补全描述性结果；
- 详细审计见
  `report/第六阶段/GRAM_第六阶段_CET_v1_GACR_v3组合实验结果与验证报告.md`。
