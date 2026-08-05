# GRAM 第十一阶段 BW3-P2：Listwise 扩展准入独立一次性验证计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-05
- Verification Status: `PREREGISTERED_ONE_SHOT_VALIDATION_DESIGN`
- Version Label: `phase11_bw3_p2_listwise_admission_one_shot_validation_v1`
- Experiment ID: `GRAM_PHASE11_BW3_P2_LISTWISE_ADMISSION_ONE_SHOT_VALIDATION_V1`
- Parent Experiment: `GRAM_PHASE11_BW3_P1C_LISTWISE_ADMISSION_CORRECTION_V1`
- Execution Status: `P2_1_COMPLETED_FAILED_PREREGISTERED_GAIN_GATE`
- Development domains: Toys、Beauty
- Validation split: 固定 `t=-2`，每域 512 用户
- Test/Sports: 本轮封存，不读取
- Resource policy: CPU-only；CodeLlama 在物理 GPU6 保持现存 30 GiB 占位

## 1. 阶段定位与研究问题

P1C 已在 `t=-4` fit、`t=-3` calibration 上完成严格的 per-user listwise admission
纠偏，并在 Toys、Beauty 两域通过 P1C 准入门。P2 不再训练、调参、选择 margin 或重新解码，
只回答一个确认性问题：

> 完全冻结的 P1C 两域 gate，能否在未参与 gate 拟合和 margin 选择的固定 `t=-2`
> 用户上，安全地利用 beam200 相对 beam50 的候选扩展，在两域保持 base 表现并取得可复现增益？

本计划是经研究者讨论后冻结的下一步设计。**写入计划不等于实现授权，更不等于启动授权**。
当前只允许保存本计划；后续必须先完成 P2-0 实现和合成测试，再由研究者另行明确授权
P2-1 的一次性正式启动。

## 2. “独立验证”的准确含义与局限

本计划中的独立性指：

- `t=-2` 不参与 P1C 的权重拟合、特征统计估计、margin 选择或 P1C 准入判断；
- P2 开始前，模型权重、bias、feature mean/std、margin、最多准入数、base 公式、输入 SHA、
  评价指标和最终判定门全部冻结；
- P2 结果不得反向修改 P1C gate，也不得按 P2 结果选择域、用户、阈值或报告口径。

它不表示 BW1 的固定 `t=-2` 数据从未被历史研究使用：BW1 已报告过 beam50/200 的候选覆盖
和 base 指标。P2 的确认性独立性来自“P1C gate 在未应用于 P2 per-user 结果前已冻结”。
因此 P2 可作为当前路线的独立保留集检验，但不能被描述为全新外部数据集验证。

## 3. 时序隔离与数据访问边界

- `t=-4`：P1C fit，已完成；
- `t=-3`：P1C calibration，已完成；
- `t=-2`：本轮唯一一次 validation；
- `t=-1`：test，继续封存；
- Sports：继续封存。

P2 的每个用户必须使用 `target = sequence[-2]`、`history = sequence[:-2]`。流行度统计也只能
由相应域所有用户的 `sequence[:-2]` train-prefix 计算，严禁包含 `t=-2` target 或 `t=-1` test。
评估器必须核对 beam 文件记录的 gold lexical ID 与 `sequence[-2]` 映射一致。

`test_read=false`、`sports_read=false` 是完整性硬门。任何代码路径、日志或诊断一旦读取
`t=-1` 或 Sports，本轮 P2 立即判为 protocol failure。

## 4. 冻结输入

### 4.1 P1C gate

| Dataset | Frozen gate | SHA256 | Margin |
|---|---|---|---:|
| Toys | `artifacts/phase11/bw3_p1c_listwise_admission/scientific/Toys/admission_gate.json` | `c5256b9f36a51ee3a6d2c44309d3e8f5eb7e92fd7a653f53c17fbb33906ac6a6` | 0.0 |
| Beauty | `artifacts/phase11/bw3_p1c_listwise_admission/scientific/Beauty/admission_gate.json` | `428c811b8955d79c7ab1edf8a9eaf1c9e2636f8a4c9632dd79aab30c9d3a94c6` | 0.0 |

每域必须原样使用 gate 中的 9 维 feature schema、weight、bias、fit-only feature mean/std、
固定 reject logit `0`、selected margin `0.0` 和 `max_admissions=3`。不得重新训练、重估统计量、
校准概率、修改阈值或跨域共享 gate。

### 4.2 固定 validation beams

| Dataset | Width | Frozen input | SHA256 |
|---|---:|---|---|
| Toys | 50 | `artifacts/phase11/bw1_candidate_ceiling/Toys/fresh_beams_w50.tsv` | `62625c06a824c70442272ae7a379e5cf431531876859fb307992436cab5ed6d9` |
| Toys | 200 | `artifacts/phase11/bw1_candidate_ceiling/Toys/fresh_beams_w200.tsv` | `d77392bea8bde078d73b95cff73559425a384f6361f382a3b094dd2409a819d0` |
| Beauty | 50 | `artifacts/phase11/bw1_candidate_ceiling/Beauty/fresh_beams_w50.tsv` | `8270b29d6f6c5b4599dda07e0fb8d639fb44ac0b6c40acb1b51b27963e227945` |
| Beauty | 200 | `artifacts/phase11/bw1_candidate_ceiling/Beauty/fresh_beams_w200.tsv` | `f45b89c875025cc6a4f186a294cb22dbcc565fa8db99df85e2ebebd2bd2e45f0` |

P2 直接复用以上每域固定 512 用户的 fresh beams，不重新解码，不读取或使用 w100。实现后
的冻结 config 还必须锁定：两域 item-head、item index/catalog、sequence dataset、实际读取的
映射文件、本计划、evaluator、tests、runner 以及 config 本身的 SHA256。任一锁不匹配时必须在
语义读取 validation 前标记 `blocked_input_lock_mismatch`。

## 5. 冻结 base、特征与准入算法

### 5.1 PCRF base

每个用户仅在 beam50 上重建冻结 base 排序：

`base_score = seq_z + reliability * standardize(item_z - 0.5 * popularity_z)`

其中必须保留 adjusted item score 的第二层用户内标准化。`base_top10` 的分数和顺序不可学习、
不可重排。评估器必须先核对重建的 base 与 BW1 冻结口径一致；不一致则 fail closed。

### 5.2 Expansion pool 和 9 维特征

`expansion_pool = beam200 - beam50`。每个 expansion candidate 按如下固定顺序构造特征：

1. `seq_raw`；
2. `seq_anchor_z`；
3. `item_raw`；
4. `item_anchor_z`；
5. `popularity_log1p`；
6. `popularity_anchor_z`；
7. `beam200_rank_fraction = (raw_rank + 1) / 200`；
8. `reliability`；
9. `cf_pop_adjusted = item_anchor_z - 0.5 * popularity_anchor_z`。

anchor z-score 只以该用户 beam50 相应特征为 anchor；随后严格使用该域 P1C gate 保存的
fit mean/std 标准化。不得加入 target label、base10 gap、user ID、P2 aggregate statistic 或
任何后验特征。

### 5.3 Gate application

- expansion logit 固定为 `wᵀx + b`，reject logit 固定为 `0`；
- 只接纳 `logit >= 0.0` 的候选；
- 按 logit 降序取最多 3 个，同分时按 candidate lexical ID 稳定排序；
- 若准入 `k` 个，最终列表为 `base_top10[:10-k] + admitted_sorted_by_logit`；
- 若没有候选通过，最终列表必须与 `base_top10` byte-for-byte 等价；
- 不允许基于 P2 target 是否被提升来撤销准入或回退。

## 6. P2-0：只实现、不接触正式验证结果

P2-0 仅允许新增专用 evaluator、tests、runner 和 disabled frozen config。至少覆盖：

- 合成 fixture 上 `t=-2` history/target 截断和 prefix-only popularity；
- beam50 base 公式及 adjusted score 第二层标准化；
- 9 维 schema、顺序、P1C mean/std 和手算 logit 一致；
- expansion difference、最多 3 个 admission、稳定 tie-break 和 fallback identity；
- Hit@10、NDCG@10、tail Hit@10、promotion/regression 的手算一致性；
- 512 用户完整性、两域原子输出、有限数值、输入 SHA 和禁读标记；
- `validation_access_started/validation_consumed/results_revealed` 状态机；
- 非零退出、timeout、已有输出和部分输出的 fail-closed 行为；
- CPU-only、CodeLlama、telemetry、status、TERM cleanup 和资源/科学状态分离。

P2-0 测试只使用合成数据；不得用真实 `t=-2` beam/gold 做 dry-run、抽样调试或性能调优。
允许对冻结文件做字节级 SHA256、大小、存在性和权限预飞检查，但不得在 P2-1 授权前语义解析
用户、候选、gold，或把 gate 应用于任何真实 P2 用户。

实现完成后仅报告代码、测试、runner、config 和输入锁是否就绪。正式 config 默认
`execution_enabled=false`，需研究者单独授权后才能切换并启动 P2-1。

## 7. P2-1：两域一次性原子验证

获得研究者明确启动授权后，一次进程内依次完成 Toys 和 Beauty 的全部 512 用户评估：

1. 完成不语义读取 validation 的资源、代码、config 和 SHA 预飞；
2. 创建不可覆盖的运行目录并记录 frozen execution manifest；
3. 在首次语义解析任一真实 `t=-2` 用户前写入 `validation_access_started=true`；
4. 将同一 frozen evaluator 应用于 Toys 和 Beauty；
5. 两域指标先只保存在进程内或未公开的临时事务目录；
6. 两域全部完成且审计通过后，原子写入最终结果并设置 `results_revealed=true`；
7. 任一域失败时，不公布另一域的临时科学指标，不自动重跑。

不得先看 Toys 再决定是否运行 Beauty，也不得根据一个域的结果修改另一域。P2-1 只运行
单次、单配置、双域完整评价。

## 8. Validation 消耗、失败与禁止自动重跑

状态字段必须至少包含：

- `validation_access_started`；
- `validation_consumed`；
- `results_revealed`；
- `validation_users_expected/processed`；
- `test_read`、`sports_read`。

判定规则：

- 在 `validation_access_started=false` 时发生的 SHA、权限、CodeLlama、CPU 单测或 runner
  预飞失败，不消耗 validation；保留日志后，可在科学配置完全不变的前提下另写具名 recovery，
  并再次取得研究者授权；
- evaluator 一旦开始语义读取任何真实 `t=-2` sequence、gold 或 beam 并应用 gate，必须同时
  设置 `validation_access_started=true`、`validation_consumed=true`；
- 此后任何 crash、非零退出、timeout、NaN/Inf、输出不完整或人为 stop 都视为 validation 已消耗，
  不得自动重跑，也不得把重跑冒充同一个确认性 P2；
- 若已消耗后需要调查，只能先报告 protocol failure，再与研究者讨论新的探索性审计方案；
- 禁止覆盖已有正式输出目录，禁止删除失败证据后重启。

## 9. 指标、诊断与冻结 P2 判定门

每域以冻结 PCRF beam50 `base_top10` 为唯一对照，报告 candidate 与 base 的：

- Hit@10 和 delta；
- NDCG@10 和 delta；
- tail Hit@10 和 delta；
- admissions、admitted users、fallback users；
- target promotions、regressions 和 unchanged；
- target 在 beam50、beam200、expansion-only 和 union 外的 coverage；
- 每用户 base/final rank、admissions、promotion/regression/fallback 和 frequency group。

可附加 seed `2023`、2,000 次 paired user bootstrap 的置信区间作为描述性不确定性分析，
但 bootstrap 不参与 PASS/FAIL，也不得用于更改 gate。

P2 仅在下列条件全部满足时标记
`passed_independent_validation_eligible_for_next_plan_discussion`：

1. Toys、Beauty 的 Hit@10 delta 均 `>= 0`；
2. 至少一个域 Hit@10 delta `>= +0.002`，且两域简单平均 delta `>= +0.001`；
3. Toys、Beauty 的 NDCG@10 delta 均 `>= -0.001`；
4. Toys、Beauty 的 tail Hit@10 delta 均 `>= 0`；
5. 两域 admissions 均非零，且 target promotions 均不少于 regressions；
6. 无 admission 用户的 fallback identity 全部通过，全部 score/metric finite；
7. 两域各恰好处理固定 512 用户，gate/feature/base/input SHA 全部匹配；
8. `test_read=false`、`sports_read=false`，无部分域提前揭示；
9. CodeLlama/30 GiB、CPU-only、后台运行、telemetry、status 和退出恢复规则全部通过。

任一条件不满足即为 P2 未通过；不得通过改 margin、删用户、换 tail 定义、只报单域或重跑来
改变结论。科学 gate 与资源 protocol 分开报告；资源失败不能被科学指标覆盖，科学失败也不能
因资源恢复成功而改写。

## 10. CodeLlama、30 GiB 现存占位与 CPU-only 规则

P2 复用已生成 beams，item-head 使用 `map_location=cpu`，因此冻结为 CPU-only workload。

### 10.1 CodeLlama 现存占位

- 默认保持 CodeLlama 在物理 GPU6 运行；若需迁移，必须先说明原因并取得研究者授权；
- 正式 P2 workload 前确认 CodeLlama tmux/process/controller 均为 running，reported reserved
  memory 至少 `30,720 MiB`；
- 若未运行，必须在 validation access 前使用既有控制器恢复，并等实际状态通过；否则标记
  `blocked_codellama_not_ready`，不得读取 validation；
- P2 全程不停止 CodeLlama，不启动 `gpu_memory_lease.py` sidecar，避免重复 30 GiB 占位；
- success、FAIL、非零退出、timeout 或手动 stop 后均检查 CodeLlama，终态记录为
  `preserved_running`、`restored` 或 `failed_to_restore_resource`。

### 10.2 CPU-only 强制项

- runner 设置 `CUDA_VISIBLE_DEVICES=""`；
- evaluator 断言 model/tensor 均在 CPU，写入 `compute_device=cpu`；
- 实验 GPU PID 必须为 0；任何 CUDA context 或 P2 导致的 GPU memory 异常跃迁均使资源门失败。

### 10.3 5 秒 telemetry

每 5 秒记录：timestamp、物理 GPU index、used/free memory、utilization、CodeLlama tmux/process
alive、controller state、reported allocated/reserved memory、实验 GPU PID、runner/evaluator PID、
CPU RSS 和当前 stage。telemetry 只记录资源状态，不得输出未揭示的单域科学指标。

## 11. 后台 runner、status 与退出协议

实现后统一提供：

```bash
bash experiment/phase11/run_phase11_bw3_p2_one_shot_validation.sh start
bash experiment/phase11/run_phase11_bw3_p2_one_shot_validation.sh status
bash experiment/phase11/run_phase11_bw3_p2_one_shot_validation.sh stop
```

- `start` 只启动当前明确授权的 P2-1，不自动启动 test、Sports 或任何后继实验；
- 使用具名持久 tmux，会话与当前终端/Codex 会话解耦；
- `status` 必须只读，显示 tmux、stage/reason、PID、CodeLlama、资源状态、三个 validation 状态
  字段、用户处理进度和最新日志；在 `results_revealed=false` 时不得显示单域科学指标；
- `stop` 优先向 evaluator 发送 `TERM`，走统一 cleanup，不遗留 telemetry 或错误 CodeLlama 状态；
- 只有预先依据合成/非 validation 性能测试冻结的 hard timeout 可以自动终止 workload；
- 非零退出、OOM、NaN/Inf、timeout、输出不完整均不自动重试；
- runner 必须用锁避免重复 start，并拒绝覆盖任何已存在的正式运行目录。

## 12. 产物与报告

正式产物根目录冻结为：

`artifacts/phase11/bw3_p2_one_shot_validation/`

至少包含：

- frozen config、execution manifest 和全部 SHA256 locks；
- `status.json`、runner/evaluator logs、CPU/GPU telemetry、execution audit；
- 两域 per-user TSV 和 domain summary；
- 双域 aggregate summary 与逐条 P2 gate 判定；
- validation/test/Sports access audit；
- scientific status 与 resource status 的独立终态。

两域科学文件只能在原子 reveal 后进入最终目录。P2 完成后撰写独立结果报告；无论 PASS/FAIL，
只保存和解释结果，不自动写入或启动下一实验。下一步仍须与研究者讨论后再形成新计划。

## 13. 当前状态与授权边界

截至 2026-08-05：

- P1C 已通过，Toys/Beauty gate 和 P2 四个 beam 输入 SHA 已冻结；
- 本 P2 研究问题、算法、一次性访问规则、指标和 PASS/FAIL 门已预注册；
- P2 evaluator、11 项合成 tests、后台 runner 和默认禁用的冻结 config 已完成 P2-0 实现；
- 真实 P2 gate application 已完成，`validation_consumed=true`、`results_revealed=true`；
- 研究者于 2026-08-05 明确发出“启动”指令，P2-1 获得并已使用一次性后台启动授权；该授权不得用于任何重试或后继实验。

### 2026-08-05 P2-1 终态

P2-1 已按一次性协议完成，validation 已消耗并原子揭示。两域各 512 用户，资源与完整性门
全部通过，CodeLlama 在 GPU6 保持 30 GiB 占位，test/Sports 未读。Toys、Beauty 的
Hit@10、NDCG@10 和 tail Hit@10 delta 均为 0；分别发生 21 和 15 次 admissions，但没有
任何真实 target 被准入，promotions/regressions 均为 0。因此安全性条款通过，但“至少一域
Hit@10 `>= +0.002`”和“两域平均 `>= +0.001`”未通过，P2 最终状态为 failed。

Validation 不得重跑，且未授权任何后继实验。完整报告：
`report/第十一阶段/GRAM_第十一阶段_BW3-P2_Listwise扩展准入独立验证结果报告.md`。

后验探索诊断进一步确认：两域 validation expansion target 全部低于 margin，失败不来自
top3 竞争；target logit 相对 calibration 平均下降约 4.8–5.0，约 89%–90% 的负漂移由
`item_anchor_z` 与 `cf_pop_adjusted` 两个共享 item-head anchor 的特征贡献。该结果只作
下一步讨论依据，不改变 P2 failed 终态。诊断报告：
`report/第十一阶段/GRAM_第十一阶段_BW3-P2_扩展Target选择漂移探索诊断报告.md`。

下一步只能先讨论 P2 失败后的研究方向；如需新实验，必须另写计划并重新授权。
不得再次启动 P2-1，也不得读取 test/Sports。
