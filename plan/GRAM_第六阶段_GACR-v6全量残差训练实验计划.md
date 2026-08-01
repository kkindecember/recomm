# GRAM 第六阶段：GACR-v6 全量 Residual 训练实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run
- Created: 2026-08-01
- Verification Status: PREREGISTERED_PENDING_START
- Version Label: `phase6_gacr_v6_full_fit_scale_v1`
- Device: 物理 GPU0
- Sports/Test: 封存

## 1. 研究问题

当前 incumbent GACR-v3 冻结 GRAM，只在每域 1024 个抽样 fit records（有效 covered：
Toys 591、Beauty 510）上训练 residual 30 个 full-batch steps，已经在三批 fresh cohort
取得 18/18 overall NDCG@10 正向点估计。研究问题是：保持方法完全不变，只使用既有
fit split 的全部训练 records，是否能严格超过小样本 GACR-v3？

## 2. 单一改动因素

唯一改动：`fit sampling = 512 head + 512 tail` 改为 `全部 fit-split records`。

严格冻结：

- GRAM C1 checkpoint、候选构造与 stable tie-break；
- 6 维 residual 特征与 `BoundedResidualRanker(6,16,bound=0.2)`；
- target-vs-highest-negative hinge loss，margin=`0.1`；
- AdamW、lr=`0.01`、weight decay=`0.01`、gradient clip=`10`；
- 30 fixed full-batch optimizer steps、seeds 2023/2024/2025；
- residual deployment scale=`1.0`；
- 80/20 fit/calibration 用户隔离与 head/tail loss 等权。

不解冻 GRAM，不改 loss，不调模型容量，不搜索 scale。

## 3. 数据协议

- Toys train users=`4,853`；Beauty train users=`5,591`
- fit/calibration 仍按原 GACR 确定性 80/20 user split；fit 使用所有可构建 record
- calibration 固定抽取 128 head + 128 tail，仅作诊断，不选择配置
- fresh validation：每域 1024 用户，salt=`phase6-gacr-v6-full-fit-development-v1`
- 排除 GCDH train/validation 及 GACR-P0、v2、v3、v4、v5 全部历史 validation cohort
- Sports/test 禁止读取

## 4. 对照与指标

同一 fresh cohort、相同候选同时比较：

1. 原始冻结 GRAM；
2. 冻结小样本 GACR-v3 residual；
3. 全量 fit GACR-v6 residual。

主指标：NDCG@10。标准四项指标：Recall@5、NDCG@5、Recall@10、NDCG@10。
补充报告 Recall@50、tail 指标、changed coverage、broad harm、paired bootstrap 95% CI。

## 5. 预注册保留门

只有全部满足才用 v6 替换 v3：

1. Toys 和 Beauty 四项标准指标均不低于 GRAM；
2. 两域 NDCG@10 相对 GRAM 均至少 `+1%`；
3. 六 cell 宏平均 NDCG@10 相对 GRAM 至少 `+2%`；
4. v6 相对 v3 的六 cell 宏平均 NDCG@10 至少 `+0.5%`；
5. 至少 5/6 域-seed cell 的 v6 NDCG@10 高于 v3；
6. 两域 mean tail NDCG@10、overall/tail Recall@50 均不低于 v3；
7. 每 cell broad harm ≤`1%`，GRAM checkpoint SHA 前后不变；
8. fit/calibration/fresh cohort 隔离、test/sports 禁读等完整性门通过。

失败则决定为 `KEEP_GACR_V3_FULL_FIT_SCALE_NOT_BENEFICIAL`，不得结果后增加 steps 或修改
loss；指标对齐 loss 只能作为独立后继实验重新预注册。

## 6. 产物与执行

- implementation：`experiment/phase6/gacr_v6.py`
- tests：`experiment/phase6/test_gacr_v6.py`
- runner：`experiment/phase6/run_phase6_gacr_v6.sh`
- config：`artifacts/phase6/configs/gacr_v6_preregistered.json`
- output：`artifacts/phase6/gacr_v6/`
- command：`bash experiment/phase6/run_phase6_gacr_v6.sh start`
- success：exit 0、summary 与 12 份逐用户 CSV/6 份新 residual checkpoint 完整
- 根据 48,325 个全量 fit records 的候选生成规模，hard timeout：36 小时；失败不自动重试
- 启动前停止 GPU0 CodeLlama；所有退出路径恢复到 GPU0
- 结果完成后只写结果，不自动分析、不读取 Sports/test、不启动后继实验

## 7. 实现与冻结记录（2026-08-01）

- implementation SHA256：`bbbe1f5102452d88a6ad51cb20f8090e6a371dd8fec6cf8b344ab49a9d3ad14e`
- test SHA256：`0b36b33715010ee1b62df1dfdd2ef9b3cb113bdb4666e1406352dba803e68166`
- runner SHA256：`ba6e0dac5757510e3d638e01eb31368437f563d56cf26bbcd87406fad20d533a`
- frozen config SHA256：`0dd7300cd4edbffed57595d92ca35a786aca1e84e23293005694a078a951418a`
- phase6 v2–v6 合并测试：`22 passed`
- Python compile、Bash syntax、JSON 与 git diff whitespace 检查通过
- 研究者已明确要求开始；上述科学配置在启动前冻结

## 8. 正式启动记录（2026-08-01）

- tmux `gram_phase6_gacr_v6` 于 20:11:00+08:00 启动；
- CodeLlama 从物理 GPU0 正常停止，GPU0 随后释放至约 48.1 GiB 空闲；
- runner 按预注册的 60 秒显存门等待一拍，科学 workload 于 20:12:00+08:00 进入
  `running`，PID=`431746`；
- Toys 实际候选构建总量=`22,095`（全量 fit + 256 calibration），已开始持续写日志；
- 当前未读取 Sports/test，未改变 frozen config，不自动重试。
