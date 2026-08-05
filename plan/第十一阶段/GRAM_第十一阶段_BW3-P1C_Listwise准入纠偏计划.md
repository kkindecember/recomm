# GRAM 第十一阶段 BW3-P1C：Listwise 扩展准入纠偏计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-05
- Verification Status: `PREREGISTERED_CORRECTION_DESIGN`
- Version Label: `phase11_bw3_p1c_listwise_admission_correction_v1`
- Experiment ID: `GRAM_PHASE11_BW3_P1C_LISTWISE_ADMISSION_CORRECTION_V1`
- Parent Experiment: `GRAM_PHASE11_BW3_P1_TRAIN_PREFIX_ADMISSION_RECOVERY_V1`
- Execution Status: `P1C0_IMPLEMENTED_AWAITING_P1C1_AUTHORIZATION`

## 1. 阶段定位与研究问题

BW3-P1 recovery 已在 `t=-4` fit 和 `t=-3` calibration 上生成完整、合法且有
SHA256 锁的 beam50/beam200，但后续 trainer 使用的是候选级加权 BCE，而不是原 BW3
预注册的 per-user listwise cross-entropy；它也没有输出完整的 coverage attrition 和
per-user calibration 记录。

BW3-P1C 只修正这一 objective/审计偏差，不读取新的科学数据，不重新生成 beams，
不修改 GRAM、item-head、PCRF 或用户样本。待检验的问题是：

> 在严格实现 per-user listwise admission objective 后，train-prefix pseudo-future 信号能否仍在
> Toys 和 Beauty 的独立 calibration split 上安全地将 beam200 扩展候选送入 top10？

本轮是 P1 纠偏，不是 P2 validation。计划写入不等于实验启动授权；实现完成、
CPU 测试和冻结配置就绪后，必须由研究者再明确授权 `start`。

## 2. 不可变范围与时序隔离

### 2.1 时序定义

- `t=-4`：fit pseudo-target，history 截止 `[:-4]`；
- `t=-3`：calibration pseudo-target，history 截止 `[:-3]`；
- `t=-2`：一次性 validation，本轮继续封存；
- `t=-1`：test，继续封存。

所有特征标准化、objective、训练、margin 选择、诊断和错误处理只能读取冻结的
`t=-4/-3` 产物和对应 train-prefix 数据。`validation_target_read`、`test_read`、
`sports_read` 在全部 status/summary 中必须为 `false`。

### 2.2 允许改动

仅允许新增 P1C 专用的：

- listwise trainer/evaluator；
- CPU 单测和真实小样本 dry-run；
- 冻结 config、后台 runner 和 status 接口；
- 具名 `artifacts/phase11/bw3_p1c_listwise_admission/` 产物。

不得修改或覆盖原 P1/recovery 的代码、日志、beams、gate 或 summary。原 BCE gate 只作
探索性对照，不参与 P1C 训练、margin 选择或准入判定。

## 3. 冻结输入与复用边界

P1C 只复用成功 recovery 目录中的 8 个 beam 文件，不复用首次
`interrupted_noncompliant` 执行的任何部分产物。

| Dataset | Split | Width | Frozen input | SHA256 |
|---|---|---:|---|---|
| Toys | fit | 50 | `bw3_p1_admission_recovery/Toys/fit/beams_w50.tsv` | `af5f24c1a46ec491f9ae191662e658871c951739b372a6bc83487fdff3719010` |
| Toys | fit | 200 | `bw3_p1_admission_recovery/Toys/fit/beams_w200.tsv` | `da5f49a8fcfde9efca95c71768f3a24e21c40d2904dfb2b8543daa72119fbc12` |
| Toys | calibration | 50 | `bw3_p1_admission_recovery/Toys/calibration/beams_w50.tsv` | `c75407f98fc00908897ac49bf5bed5e7013cb154c24788c1887b16ed2f925b90` |
| Toys | calibration | 200 | `bw3_p1_admission_recovery/Toys/calibration/beams_w200.tsv` | `fd5ce09ddbe89eb7623faad1bf5126129a1d233327e1d21640d57ff9dda9963a` |
| Beauty | fit | 50 | `bw3_p1_admission_recovery/Beauty/fit/beams_w50.tsv` | `0fae8ff1b1f205aba5a807265c9b7c783f3241c55db7b43157910eef1e1fccc1` |
| Beauty | fit | 200 | `bw3_p1_admission_recovery/Beauty/fit/beams_w200.tsv` | `a707a4ccc9d9b11020a606eea04d870946c6c57418453b5c04828252dcb34612` |
| Beauty | calibration | 50 | `bw3_p1_admission_recovery/Beauty/calibration/beams_w50.tsv` | `3e1281d1150f19aa3393478ea721f662ee32123d1278890b1e867db622bf4a00` |
| Beauty | calibration | 200 | `bw3_p1_admission_recovery/Beauty/calibration/beams_w200.tsv` | `ba0f7feefe6dd572d6ce0e99afd9d95b0d0bfc86f907176c09c35ea2e1ffb34e` |

> 注：表中相对路径的根目录为 `artifacts/phase11/`。

实现后的冻结 config 还必须锁定：4 个对应 summary、Toys/Beauty item-head、
item index/catalog、本计划、trainer、tests、runner 与全部实际读取的 dataset 文件。任一
SHA 不符必须在读取科学产物前标记 `blocked_input_lock_mismatch`。

## 4. 候选构造与冻结 base

每个用户独立构造：

1. 用冻结 beam50 候选和 train-prefix 特征重建已确认 PCRF base 排序，公式保持为
   `seq_z + reliability * standardize(item_z - 0.5 * popularity_z)`，不得遗漏 adjusted score
   的第二层标准化；
2. `base_top10` 的顺序和分数不可学习或改动；
3. `expansion_pool = beam200 - beam50`，按 raw beam200 rank 记录；
4. `in_beam50` 只作构造 mask 和审计字段，不进入可学习 feature vector；
5. gate 最多准入 3 个 expansion candidates。若准入 `k` 个，最终列表固定为
   `base_top10[:10-k] + admitted_sorted_by_logit`；
6. 若无候选通过，必须逐用户 byte-for-byte 等价回退到冻结 `base_top10`。

## 5. 严格 listwise admission objective

### 5.1 特征 schema

对 expansion pool 中每个候选固定使用以下 9 维特征：

1. `seq_raw`：GRAM raw sequence score；
2. `seq_anchor_z`：以该用户 beam50 sequence scores 为 anchor 的 z-score；
3. `item_raw`：冻结 item-head raw score；
4. `item_anchor_z`：以该用户 beam50 item-head scores 为 anchor 的 z-score；
5. `popularity_log1p`：只由对应 train prefix 统计的 `log1p(frequency)`；
6. `popularity_anchor_z`：以该用户 beam50 popularity 为 anchor 的 z-score；
7. `beam200_rank_fraction`：`(raw_rank + 1) / 200`；
8. `reliability`：原 PCRF 冻结的用户级 reliability；
9. `cf_pop_adjusted`：`item_anchor_z - 0.5 * popularity_anchor_z`。

raw 特征和 anchor 特征均保留，不新增 base10-gap、user ID、target-derived feature、
validation-derived statistic 或 BCE 输出。候选特征只使用 `t=-4` fit 集的全局 mean/std
做第二层标准化，calibration 直接复用，不重新估计统计量。

### 5.2 每用户 action list

每个可用 fit event 的 listwise action set 为：

- 一个 `REJECT_TO_BASE` action，logit 固定为 `0`；
- 该用户所有 expansion candidates，其 logit 为 `wᵀx + b`。

label 规则在训练前冻结：

- target 在 expansion pool：label 为该 target candidate；
- target 在 beam50：label 为 `REJECT_TO_BASE`；
- target 不在 `beam50 ∪ beam200`：该事件不计 loss，只进入 coverage attrition 报告。

这使每个用户在 loss 中权重相同，避免 150 个左右的 negative candidates 把一个用户
重复计权，也不再使用 BCE、`pos_weight` 或 candidate-level negative sampling。

### 5.3 冻结优化配置

- loss：按用户平均的 listwise cross-entropy + `1e-3 * ||w||²`；
- optimizer：Adam；
- learning rate：`0.05`；
- epochs：`200`，不 early stop；
- seed：`2023`；
- initialization：`w=0`、`b=0`；
- Toys 和 Beauty 分域拟合，不共享权重、标准化统计量或 margin；
- 只跑单 seed，不使用 calibration 搜 seed、epochs、learning rate、L2 或特征子集。

## 6. Coverage attrition 和审计产物

fit 和 calibration 对每域都必须先输出以下四个互斥 action/attrition 分类的
count/fraction，四类之和必须等于全部样本：

- target in base top10；
- target in beam50 rank 11–50；
- target in expansion pool only；
- target outside `beam50 ∪ beam200`。

再单独输出不要求互斥的 membership/overlap 审计表：

- target in beam50；
- target in beam200；
- target in both beam50 and beam200（action label 为 base）；
- target in beam200 but not beam50；
- target in union and included in loss；
- target outside union and excluded from loss；
- empty expansion pool；
- 实际的 expansion pool size 分布。

action/attrition 与 membership/overlap 必须分表输出。每个 calibration 用户还必须保存
per-user TSV，至少包含 user、target、
base rank、wide membership、expansion pool size、选中 margin 下的 admissions、final rank、promotion/
regression/fallback、target frequency group，以及不包含 target label 的可审计特征摘要。

## 7. Calibration 与规则歧义消除

margin 仍仅从事先固定的 grid `{0, 0.25, 0.5, 0.75, 1.0}` 中选择。对每个用户，
按 expansion logit 降序取最多 3 个满足 `logit >= REJECT_TO_BASE_logit + margin` 的候选。

原 BW3 计划中“选择最小安全 margin”与后续字典序条款存在歧义。P1C 在读取
calibration 之前将规则唯一冻结为：

1. 先筛选 Hit@10 delta `>= 0`、NDCG@10 delta `>= -0.001` 且 admissions `> 0`
   的安全 margin；
2. 在安全集合内按 candidate Hit@10 较大、candidate NDCG@10 较大、margin 较大的
   字典序选择；
3. 若没有安全 margin，该域 P1C 失败，不读 validation；
4. 不将原 BCE gate 的 margin 或结果作为 tie-breaker。

## 8. 分阶段实现与执行

### P1C-0 — 实现和 CPU 完整性门

新增专用 trainer/evaluator、tests 和 runner，不修改原 BCE trainer。至少覆盖：

- 每用户 listwise CE 与手算 softmax 一致；
- 每用户等权，不随 expansion pool size 改变；
- base/expansion/outside-union label 三分支；
- 只用 fit mean/std，calibration 无统计重估；
- 特征 schema 与顺序的精确断言；
- base fallback identity、最多 3 admissions 和稳定 tie-breaking；
- margin 安全集与唯一字典序选择；
- attrition count 互斥性、加和一致性与 per-user 输出完整性；
- 输入 SHA 不符、输出已存在、NaN/Inf 和禁读标记的 fail-closed 行为。

然后使用每域 16 个 fit + 16 个 calibration 真实冻结事件做 CPU dry-run。dry-run
只验证解析、特征、loss、输出 schema 和运行时/RSS，不选择正式 margin，不产生
可复用科学 gate。

### P1C-1 — 两域一次 fit/calibration

- 读取冻结的 Toys/Beauty `t=-4/-3` beams；
- 使用 seed 2023 和冻结优化参数分域拟合；
- 输出两域 checkpoint、feature statistics、attrition、margin grid、per-user TSV 和 summary；
- 两域均完成后才计算 P1C gate；
- 任一域 integrity/scientific gate 失败都停止在 P1C，不自动读取 P2。

## 9. P1C 准入门

P1C 只在以下条件全部满足时标记 `passed_eligible_for_separate_p2_authorization`：

1. 8 个 beam 和全部冻结输入 SHA 通过，没有读取首次中断产物；
2. objective 确认为按用户平均的 listwise CE，无 BCE/`pos_weight`；
3. 两域 loss 全程 finite，最终 loss 低于初始 loss；
4. feature schema、fit-only statistics、attrition 和 per-user 审计产物全部完整；
5. Toys、Beauty 都选出唯一安全 margin；
6. 两域 calibration Hit@10 delta 均 `>= 0`，NDCG@10 delta 均 `>= -0.001`，
   tail Hit@10 delta 均 `>= 0`；
7. 两域 admissions 均非零，promotions 均不少于 regressions；
8. 无 admission 用户的 base fallback identity 全部通过，所有分数 finite；
9. `validation_target_read=false`、`test_read=false`、`sports_read=false`；
10. CodeLlama/30 GiB 占位、后台执行、telemetry、status 与退出恢复规则全部通过。

若任一条件失败，P1C 不获得 P2 资格。完成 P1C 只保存结果并报告，即使 PASS 也
不得自动实现、启动或读取一次性 P2 validation。

## 10. CodeLlama、30 GiB 现存占位与 CPU-only 纠偏

P1C 复用已生成 beams，item-head 以 `map_location=cpu` 读取，listwise gate 也仅在 CPU
上拟合。因此本轮冻结为 **CPU-only workload**，避免重复 P1 recovery 的 GPU 总占用
`41,866 MiB` 口径问题。

### 10.1 强制 CPU 路径

- runner 启动 trainer 时必须设置 `CUDA_VISIBLE_DEVICES=""`；
- trainer 启动后必须断言所有 model/tensor 均在 CPU，并写入 `compute_device=cpu`；
- 任何 CUDA context、实验 GPU PID 或实验导致的 GPU used-memory 跃迁都标记资源门失败；
- 不启动 `gpu_memory_lease.py` sidecar，因为 CodeLlama 本身已持有 30 GiB，再启动 sidecar
  会变成重复占位。

### 10.2 CodeLlama 占位协议

- 默认物理 GPU 为上一次恢复的 GPU6；实现后必须在冻结 config 和 `status.json`
  记录最终物理 GPU，若需迁移必须先记录原因并获得研究者授权；
- CPU 预飞检查通过后、正式 workload 开始前，确认 CodeLlama tmux/process 在目标 GPU
  处于 running，且其报告的 reserved memory 至少为 `30,720 MiB`；
- 若 CodeLlama 未运行，runner 先使用 `tools/run_codellama.sh start <GPU>` 建立占位并等待实际
  状态通过；超时则标记 `blocked_codellama_not_ready`，不启动科学 workload；
- P1C 正式运行期间不停止 CodeLlama，使其持续作为 30 GiB 现存占位；
- 不论成功、科学门失败、程序非零退出、timeout 或手动 `stop`，统一清理路径都必须
  再次检查 CodeLlama；若因异常不在，尝试恢复到同一物理 GPU；
- CodeLlama 终态写为 `preserved_running`、`restored` 或 `failed_to_restore_resource`，不得
  覆盖科学退出状态或退出码。

### 10.3 5 秒 telemetry

即使 P1C 为 CPU-only，runner 仍必须每 5 秒记录：

- timestamp、物理 GPU index、used/free memory、utilization；
- CodeLlama tmux/process alive 与 controller state；
- CodeLlama reported allocated/reserved memory；
- 实验 CPU PID 的 RSS 和 elapsed time；
- 是否观测到任何实验 GPU PID。

telemetry 必须从科学 workload 启动前持续到终态资源检查完成。本轮 30 GiB 规则的
通过口径是：CodeLlama 全程保持至少 `30,720 MiB` reserved，实验本身不建立 CUDA context
且不启动重复 sidecar。

## 11. 后台 runner 与用户接口

无论 dry-run 预估耗时是否超过 20 分钟，正式 P1C-1 都必须在具名持久 tmux 会话
中后台运行，并提供：

```bash
bash experiment/phase11/run_phase11_bw3_p1c_listwise_admission.sh start
bash experiment/phase11/run_phase11_bw3_p1c_listwise_admission.sh status
bash experiment/phase11/run_phase11_bw3_p1c_listwise_admission.sh stop
```

- `start` 只能启动已明确授权且 config 中 `execution_enabled=true` 的 P1C-1；
- `start` 必须拒绝覆盖已存科学产物，也不得自动启动 P2；
- `status` 只读，显示 tmux 是否存活、`status/stage/reason`、runner/workload PID、CPU RSS、
  物理 GPU、CodeLlama 存活/占位/恢复状态、最新 telemetry、三个禁读标记和日志尾部；
- `stop` 优先对 workload 发送 `TERM`，等待统一清理，不停止 CodeLlama；若它已异常退出，
  则走恢复路径；
- 日志、`status.json`、`gpu_telemetry.csv`、CPU telemetry、config、checkpoints、per-user TSV
  和 summaries 全部持久化到 `artifacts/phase11/bw3_p1c_listwise_admission/`。

## 12. 启动、超时、失败与 recovery 规则

1. 启动前必须通过 CPU 单测、真实小样本 dry-run、Python compile、Bash syntax、
   JSON/config schema、输入完整性和全部 SHA 检查；
2. dry-run 后根据实测耗时冻结 hard timeout，不得在正式运行中修改；
3. 只有 hard timeout 可以自动终止 workload；资源或科学异常只记录并在安全检查点
   fail closed，不以非预注册的调参“挽救”结果；
4. 非零退出、OOM、NaN/Inf、timeout、输出不完整、输入 SHA 不符或科学门失败均不
   自动重试；
5. 需要 recovery 时，必须保留原日志和状态，使用新的具名输出目录，说明根因、允许改动范围
   和输出复用边界，并获得研究者明确授权；
6. 科学退出状态与 CodeLlama/资源状态分开记录，任一资源恢复成功都不得覆盖
   科学失败退出码；
7. 完成后只保存结果并报告终态。未经研究者明确授权，不得自动实现/启动 P2，
   不得读取 `t=-2`、test 或 Sports。

## 13. 预期产物

| Output | Path | Success criterion |
|---|---|---|
| Frozen config | `artifacts/phase11/configs/bw3_p1c_listwise_admission_preregistered.json` | 含全部 code/input SHA，默认 execution disabled |
| Toys gate | `artifacts/phase11/bw3_p1c_listwise_admission/Toys/admission_gate.json` | feature/order/statistics/margin 完整且 finite |
| Beauty gate | `artifacts/phase11/bw3_p1c_listwise_admission/Beauty/admission_gate.json` | feature/order/statistics/margin 完整且 finite |
| Attrition | `.../{Toys,Beauty}/attrition.json` | 互斥 count 对齐并与 per-user 一致 |
| Per-user | `.../{Toys,Beauty}/calibration_per_user.tsv` | 512 用户、无重复、无缺失 |
| Domain summaries | `.../{Toys,Beauty}/summary.json` | fit/calibration/integrity/scientific gate 齐全 |
| Aggregate summary | `.../summary.json` | 两域结果、P1C 总门和禁读状态齐全 |
| Status/log | `.../status.json`, `.../run.log` | 后台终态与资源终态分离 |
| Telemetry | `.../gpu_telemetry.csv`, `.../cpu_telemetry.csv` | 5 秒级完整覆盖 |

## 14. 完成后的决策边界

- P1C PASS：只说明原 BW3 listwise 假设获得独立 P2 授权资格；另行与研究者讨论并写入
  P2 one-shot validation 执行计划后，才能读取 `t=-2`。
- P1C FAIL：保持 P2 封存，不在 calibration 上改 objective、特征、margin、seed 或权重；
  返回候选表征/生成模型训练方向，并再与研究者讨论新计划。
- 无论 PASS/FAIL，原 BCE 结果都只作探索性对照，不与 P1C 合并为一个预注册结论。

## 15. P1C-0 实现状态（2026-08-05）

- 独立 listwise trainer、专用 tests、后台 runner 和冻结 config 已实现；
- P1C 专用单测 `9 passed`，第十一阶段 CPU 测试合计 `23 passed`；
- 最终真实 16+16 双域 CPU dry-run：
  `artifacts/phase11/bw3_p1c_listwise_admission_dry_run_v5/summary.json`；
- dry-run 两域 integrity 均通过，scientific gate 按计划为 `not_evaluated_dry_run`；
- code/input/evidence SHA 已冻结并验证；
- `execution_enabled=false`，正式 P1C-1 未启动，validation/test/Sports 仍未读。
