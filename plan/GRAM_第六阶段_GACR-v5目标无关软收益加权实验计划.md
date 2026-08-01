# GRAM 第六阶段：GACR-v5 目标无关软收益加权实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-01
- Verification Status: COMPLETED_ANALYZED
- Version Label: `phase6_gacr_v5_target_free_soft_weight_v1`
- Development Domains: Toys、Beauty
- Sports/Test: 封存
- Intended Device: 物理 GPU0
- Post-run Resource: CodeLlama 恢复并继续占用物理 GPU0

## 1. 研究问题与依据

GACR-v4 在新 fresh cohort 上仍复现冻结 GACR 的 6/6 overall NDCG 正向点估计，但
Toys/Beauty 都选择 hard-gate threshold 0，v4 因而与 v3 完全相同。结果后诊断显示，
gate probability 对改善/伤害用户仍有区分度：Toys 三 seed AUC 为 0.588–0.622，Beauty
为 0.691–0.751。hard gate 的问题不是概率完全无信息，而是删除低概率用户时损失的改善
用户多于避免的伤害用户。

研究问题：在不删除任何候选、不重新训练 residual/gate 的前提下，能否把 gate probability
改为连续 residual 强度，使 Toys Recall@50/tail 更安全，同时严格提高 v3 的 NDCG？

## 2. 单一改动因素

冻结每个域和 seed 的 GACR-v3 residual `r_i` 与 GACR-v4 gate probability `p`。v5 只把
v4 的二元应用规则替换为连续 soft multiplier：

`score_i(v5) = base_i + m(p; alpha) * r_i`

`m(p; alpha) = alpha + (1 - alpha) * p`

预注册 `alpha` 候选为 `0/0.25/0.50/0.75/1.00`：

- `alpha=0`：残差强度完全由 `p` 决定；
- `0<alpha<1`：所有用户保留 residual，但低概率用户得到更弱强度；
- `alpha=1`：`m=1`，是冻结 GACR-v3 的精确 identity control。

每个域选择一个跨 2023/2024/2025 三 seed 共享的 alpha。除上述映射外，不修改 gate
特征、gate checkpoint、residual checkpoint、候选构造、GRAM checkpoint 或任何训练数据。

## 3. 冻结项与输入 lineage

- GRAM C1 checkpoints：沿用 v4 已验证 SHA；前后必须不变
- residual：`artifacts/phase6/gacr_v2/{Toys,Beauty}/residual_seed*.pt`，即冻结 v3 identity
- gate：`artifacts/phase6/gacr_v4/{Toys,Beauty}/gate_seed*.pt`
- fit/calibration 用户、8 维 gate 特征及标准化参数全部冻结
- gate 或 backbone optimizer steps 必须为 0
- 实现前必须把上述 12 个 checkpoint SHA、配置 SHA 和实现 SHA 写入预注册 JSON

## 4. 数据隔离

- calibration：只重建 v4 已登记的 calibration records，用于 alpha 选择；不读取任何
  fresh-validation label
- fresh validation：Toys、Beauty 各 1024 用户，salt 固定为
  `phase6-gacr-v5-development-v1`
- 排除 GCDH training/validation、GACR-P0、v2、v3、CET+GACR、v4 的全部开发 cohort
- seeds：2023/2024/2025
- Sports/test 禁止读取

## 5. Calibration 选择规则

对每个域分别评估五个 alpha。非 identity 候选必须在三个 seed cells 全部满足：

1. overall Recall@10 不低于 GRAM；
2. overall Recall@50 不低于 GRAM；
3. tail NDCG@10 不低于 GRAM；
4. tail Recall@50 不低于 GRAM；
5. broad harm ≤ 1%。

在 eligible 候选中最大化三 seed mean overall NDCG@10；依次以更高 mean tail NDCG、
更高 mean Recall@50、较低 maximum broad harm、较大的 alpha 打破平局。若无非 identity
候选满足安全门，或其 mean NDCG 不严格超过 alpha=1，则选择 alpha=1，返回 v3。

## 6. Fresh-validation 对照与保留门

对照固定为 `GRAM`、冻结 `GACR-v3`、`GACR-v5`。只有以下条件全部成立才保留 v5：

1. Toys 三 seed mean overall NDCG@10 严格超过 v3；
2. 双域六 cell 宏平均 NDCG@10 严格超过 v3；
3. Beauty 三 seed mean overall NDCG@10 不低于 v3；
4. Toys mean tail NDCG@10、overall Recall@50、tail Recall@50 均不低于 v3；
5. Toys mean overall Recall@50 不低于 GRAM，明确修复 v4 cohort 暴露的负向信号；
6. 六个域-seed cell 的 broad harm ≤ 1%，且 overall Recall@10 不低于 GRAM；
7. 完整性门全部通过。

CI、逐用户改善/伤害比例和 AUC 只用于描述证据与机制，不取代上述预注册点估计门。若
任一条件失败，保留冻结 v3，并停止继续修改当前 gate/soft-weighting 因素。

## 7. 分析与报告

必须报告：

- overall/head/tail 的 NDCG@10、Recall@10、Recall@50；
- v5 相对 GRAM 与相对 v3 的逐 seed、域均值和六 cell 宏平均；
- paired-user bootstrap 95% CI；
- changed-user coverage、broad harm、mean multiplier 及 multiplier 分位数；
- 改善/伤害用户的 gate probability 和 multiplier 分布；
- alpha=1 与冻结 v3 的逐用户 rank 精确 identity 检查；
- 所有 checkpoint、逐用户 CSV、summary 和配置 SHA。

## 8. 工程计划与预期产物

计划实现：

- `experiment/phase6/gacr_v5.py`
- `experiment/phase6/test_gacr_v5.py`
- `experiment/phase6/run_phase6_gacr_v5.sh`
- `artifacts/phase6/configs/gacr_v5_preregistered.json`
- `artifacts/phase6/gacr_v5/summary.json`
- `artifacts/phase6/gacr_v5/{Toys,Beauty}/{gacr_v3,gacr_v5}_seed*_per_user.csv`

实现完成后先执行单元测试、Python compile、Bash syntax、JSON、checkpoint lineage 与
identity preflight；在这些产物及 SHA 写入预注册配置前不得启动正式实验。

## 9. 运行与资源协议

- 物理 GPU0；启动前空闲显存门 ≥ 30,720 MiB
- 预计科学 workload 约 3–4 小时，硬超时 8 小时
- 使用具名 tmux，状态、日志和 GPU telemetry 持久化
- 启动前停止 GPU0 的 CodeLlama；所有退出路径恢复 CodeLlama 到 GPU0
- 恢复调用必须显式使用：
  `HF_HOME=/home/jiangtangyunzhi/hf_cache`、
  `HF_HUB_CACHE=/home/jiangtangyunzhi/hf_cache/hub`、
  `TRANSFORMERS_CACHE=/home/jiangtangyunzhi/hf_cache/hub`
- 科学 exit 与资源恢复状态分开记录；失败不自动重试
- 结果完成后不自动读取 Sports/test，不自动启动下一实验

## 10. 实现与预注册记录（2026-08-01）

- implementation：`experiment/phase6/gacr_v5.py`
- implementation SHA256：`47b28ca3acf7a36ddbaad67c7c7113dd537379adbb74e8c984409935e6d0b76d`
- test SHA256：`ea8ce9519d22a54ee1c887ac5b91e9bfee6a439854a2d2d24028cc0e765b906b`
- runner SHA256：`70108ac3251f281cd645d1cf01c2dea228ca56147f7e22f045e7df28b4b698be`
- config SHA256：`29eae16f514f14287a7e7dbcd7b9dc493262b119f9fb671ccf2683451606d86a`
- phase6 v2/v3/v4/v5 合并单元测试：`19 passed`
- Python compile、Bash syntax、JSON、implementation SHA、T5 离线 cache、CodeLlama
  离线 cache 与 14 个锁定 checkpoint/state SHA 预检通过
- 启动前 CodeLlama 在物理 GPU0 正常 `running`；runner 将先释放其 CUDA context，再
  等待 GPU0 空闲显存达到 30,720 MiB
- 研究者已于 2026-08-01 明确要求“开始实验”；配置在正式 workload 前冻结，不再修改
  alpha、cohort、指标或保留门。

## 11. 正式启动记录（2026-08-01）

- tmux `gram_phase6_gacr_v5` 于 16:31:57+08:00 启动；
- CodeLlama 已从物理 GPU0 正常停止并释放 CUDA context；
- 首次 GPU gate 检查早于 context 完全释放，runner 按预注册的 60 秒周期等待一次；
- 科学 workload 于 16:32:57+08:00 正式进入 `running`，PID=`3983920`；
- 启动后 GPU0 利用率约 96%，显存占用约 4.4 GiB 并继续进入候选生成；
- 当前未读取 Sports/test，未修改正式配置，未产生需要重试的失败；
- 查询命令：`bash experiment/phase6/run_phase6_gacr_v5.sh status`。

## 12. 完成与分析结论（2026-08-01）

- 科学 workload 于约 18:00:10 完成，exit=`0`；runner 于 18:00:16 记录
  `status=succeeded`，CodeLlama 已恢复到物理 GPU0；
- Toys、Beauty 的校准都选择 `alpha=1`，v5 与冻结 GACR-v3 在全部六个 fresh-validation
  cell 精确一致；
- v5 相对 GRAM：Toys mean overall NDCG `+3.092%`，Beauty `+2.122%`，六 cell
  宏平均 `+2.607%`，6/6 点估计为正；
- v5 相对 v3 的 Toys 和宏平均严格增益均为 0，未通过预注册保留门；
- 正式决定：`RETURN_TO_GACR_V3_STOP_GATE_WEIGHTING_FAMILY`；保留核心 GACR，停止继续
  调 residual 后的 gate/attenuation/soft multiplier；
- 下一实验转向 GACR-v6 指标对齐 residual loss。完整报告见
  `report/第六阶段/GRAM_第六阶段_GACR_v5结果与验证报告.md`。
