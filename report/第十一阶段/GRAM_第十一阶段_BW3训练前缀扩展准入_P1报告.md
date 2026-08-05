# GRAM 第十一阶段 BW3：训练前缀扩展准入 P1 报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-05
- Verification Status: `ANALYZED_WITH_PREREGISTRATION_DEVIATION`
- Overall Confidence: `CAUTION`
- Version Label: `phase11_bw3_train_prefix_admission_p1_v1`
- Experiment ID: `GRAM_PHASE11_BW3_P1_TRAIN_PREFIX_ADMISSION_RECOVERY_V1`

## 1. Executive conclusion

BW3-P1 recovery 于 2026-08-05 02:02:09 +08:00 完成，Toys 和 Beauty 的 fit/calibration
pseudo-future beam 均生成完整，冻结 gate 也在两域通过当前实现的 calibration gate。
`validation_target_read=false`、`test_read=false`、`sports_read=false`，因此一次性 P2 validation
尚未被消耗。

当前实现在 calibration 上呈现强信号：Toys Hit@10 从 `0.542969` 升到
`0.691406`（`+0.148438`），Beauty 从 `0.457031` 升到 `0.572266`
（`+0.115234`）；两域 NDCG@10 和 tail Hit@10 同时上升，promotion 也显著多于
regression。这证明 train-prefix 特征确实包含识别有价值扩展候选的信号。

但本轮不能按原计划直接判为严格的 `P1 PASS / P2 ELIGIBLE`：

1. 预注册要求按用户的 listwise cross-entropy，实际代码对所有 expansion candidates
   使用了带 `pos_weight` 的 binary cross-entropy；
2. 预注册要求只对 target 进入 union pool 的事件计 ranking loss 并报告
   coverage attrition，实际训练纳入了全部事件，且 summary 没有事件级 attrition 字段；
3. 30 GiB 租约在 runner 中被声明为 `30,720 MiB`，但 telemetry 观测到 GPU
   总 used memory 峰值 `41,866 MiB`，因此未实际满足“workload + sidecar 合计
   保持 30 GiB”的资源规则。

结论是：**科学方向值得继续，但建议先修复 P1 的预注册一致性与显存租约口径，
不建议立即读取 `t=-2` validation。**

## 2. Execution audit

### 2.1 首次执行

首次 `bw3_p1_admission` 在 Toys offset-4 beam 生成中被研究者授权停止，exit `130`。
该 runner 没有实现第九阶段规定的 CodeLlama 预占/释放/恢复、30 GiB 协作租约、
5 秒 telemetry 和完整 status schema，因此被记为 `interrupted_noncompliant`。部分输出未复用，
未产生科学结果，也未读取 validation/test/Sports。

### 2.2 具名 recovery

- 持久后台会话：`gram_phase11_bw3_p1_admission_recovery`；
- 物理 GPU：6；
- 起止时间：2026-08-05 01:26:13 至 02:02:09 +08:00，约 35 分 56 秒；
- 预飞检查：`5 passed`，Python compile、Bash syntax、冻结 SHA 与离线 tokenizer 检查通过；
- 输出复用：`false`，从空 recovery 目录重新生成；
- 退出状态：`succeeded / finished`；
- 退出时 CodeLlama 恢复状态：`restored_on_gpu6`，科学退出状态与资源恢复状态分开记录。

## 3. Pseudo-future coverage

| Dataset | Split | Users | beam50 target coverage | beam200 target coverage | Width headroom |
|---|---|---:|---:|---:|---:|
| Toys | fit (`t=-4`) | 1,024 | 0.550781 (564) | 0.725586 (743) | +0.174805 (179) |
| Toys | calibration (`t=-3`) | 512 | 0.578125 (296) | 0.757812 (388) | +0.179688 (92) |
| Beauty | fit (`t=-4`) | 1,024 | 0.474609 (486) | 0.642578 (658) | +0.167969 (172) |
| Beauty | calibration (`t=-3`) | 512 | 0.507812 (260) | 0.666016 (341) | +0.158203 (81) |

四个生成单元的 candidates 均为 `legal_fraction=1.0`，checkpoint identity 与
validation/test/Sports 禁读检查均通过。宽 beam 在 fit 和 calibration 上都稳定提供
15.8–18.0pp 的 target coverage headroom，该现象不只存在于 BW1 validation 样本。

## 4. Fit and calibration results

### 4.1 Fit diagnostics

| Dataset | Positive expansion targets | Candidate rows | Initial loss | Final loss | Finite | Selected margin |
|---|---:|---:|---:|---:|---|---:|
| Toys | 179 | 153,600 | 0.788336 | 0.115167 | PASS | 0.25 |
| Beauty | 172 | 153,605 | 0.761953 | 0.150155 | PASS | 0.00 |

loss 大幅下降且 finite，但这是带上限为 100 的 positive class weight 之 BCE 训练损失，
不能解读为预注册 listwise objective 已收敛。实际 feature schema 为 8 维 anchor/gap 特征，
也与计划中同时包含 raw score 和 base-mask 的文字定义不完全相同。

### 4.2 Selected calibration output

| Dataset | Base Hit@10 | Gate Hit@10 | Delta | NDCG@10 delta | Tail Hit@10 delta | Admissions | Promotions / regressions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | 0.542969 | 0.691406 | +0.148438 | +0.044885 | +0.117647 | 518 | 77 / 1 |
| Beauty | 0.457031 | 0.572266 | +0.115234 | +0.034865 | +0.058394 | 557 | 63 / 4 |

Toys 的 5 个预定 margin 全部安全，按实现的 Hit@10、NDCG@10、较大 margin
字典序选中 `0.25`；Beauty 同样全部安全，选中 `0.00`。计划中“选择最小安全
margin”与后文“先最大化 Hit/NDCG，再选较大 margin”存在文字冲突；实现采用了后者。

这些是在同一 calibration split 上选择 margin 后的最优值，不是独立泛化证据。当前产物
也没有保存 per-user calibration 记录或 paired confidence interval，所以报告只将它们定性为
“强探索信号”。

## 5. Integrity and resource assessment

| Check | Result | Assessment |
|---|---|---|
| 两域 fit/calibration 完整 | PASS | 1,024/512 用户全部完成 |
| candidate legal / finite | PASS | 四个生成单元的 gate 均 passed |
| GRAM / item-head identity | PASS | SHA 与冻结输入一致 |
| validation/test/Sports 封存 | PASS | 全部 false |
| 后台 tmux/status 接口 | PASS | 运行中持久化，终态可读 |
| 5 秒 telemetry | PASS | 426 行，平均间隔 5.046s，最大间隔 5.089s |
| 退出时 CodeLlama 恢复 | PASS | `restored_on_gpu6` |
| 预注册训练 objective | **FAIL** | BCE 不等于按用户 listwise CE |
| 预注册 coverage attrition 报告 | **FAIL** | summary 未保存事件级 attrition |
| workload + sidecar = 30,720 MiB | **FAIL** | telemetry 峰值 41,866 MiB，最低 free 6,705 MiB |

各生成单元报告的 PyTorch peak allocated 为 Toys 约 26.81 GiB、Beauty 约
26.88 GiB，但 sidecar 使用 allocated 口径配置租约，telemetry 则反映整卡实际 used memory。
两者口径不一致，导致 nominal lease 通过而实测总占用超标。

## 6. Statistical and methodological validation

- 本轮未报告预注册的显著性检验，也没有独立重跑，因此 Verification Status 为
  `ANALYZED`，不是 `VERIFIED`。
- 五个 margin 为事先固定，但 calibration 指标同时用于选择和报告；其估计对未见数据
  可能乐观。
- 两域全用户完成，无样本脱落；但有用信号主要来自 target 进入 beam200 扩展池的
  事件，应在正式实现中显式报告 coverage/eligibility attrition。

### Fallacy scan

- Coverage: `11/11 checked`
- CAUTION: look-elsewhere / calibration reuse（择优后数值不是无偏 validation 估计）。
- CAUTION: garden of forking paths（objective、特征 schema 与 attrition 口径偏离预注册）。
- NOTE: survivorship bias 未见，四个单元无用户脱落。
- NOTE: Simpson、ecological、Berkson、collider、base-rate neglect、regression-to-mean、
  correlation/causation 与 reverse causality 未在当前描述性设计中观察到可判定证据。

## 7. Decision pending researcher discussion

本报告不自动启动 P2，也不在本轮修改下一步 plan。建议优先讨论以下三个分支：

1. **严格修正 P1（建议）**：保留已合规生成且有 SHA 锁的 pseudo-future beams，另写具名
   correction 计划，实现真正的 per-user listwise loss、预注册特征 schema、coverage attrition
   和 per-user 输出；在不读 validation 的前提下重新冻结 gate/margin。
2. **接受当前 BCE 为新方法**：将当前结果降格为探索性方法发现，重新预注册独立
   P2；但这会接受“不是原 BW3 objective”的方法变更。
3. **直接执行原 P2**：可以最快获得方向性结果，但会消耗一次性 validation，且最终
   只能声称“当前 BCE gate 的探索性验证”，不能声称严格验证了原预注册 BW3。

无论选择哪个分支，后续 plan 都应完整继承：CodeLlama 实验前物理 GPU 占位、受控
释放和所有退出路径恢复；实测口径的 30 GiB 现存租约；具名 tmux 后台 `start/status/stop`
接口；5 秒 telemetry；启动前测试与 SHA 冻结；不自动重试；科学结果与资源恢复
状态分离；以及未授权不读 test/Sports、不自动启动后继实验。

## 8. Artifacts

- preregistration：`plan/第十一阶段/GRAM_第十一阶段_BW3训练前缀扩展准入计划.md`
- recovery config：`artifacts/phase11/configs/bw3_p1_admission_recovery_preregistered.json`
- aggregate summary：`artifacts/phase11/bw3_p1_admission_recovery/admission/summary.json`
- frozen gates：`artifacts/phase11/bw3_p1_admission_recovery/admission/{Toys,Beauty}/admission_gate.json`
- execution status/log：`artifacts/phase11/bw3_p1_admission_recovery/status.json`、`run.log`
- telemetry：`artifacts/phase11/bw3_p1_admission_recovery/gpu_telemetry.csv`
- initial-run audit：`artifacts/phase11/bw3_p1_admission/execution_audit.json`
- trainer：`experiment/phase11/train_bw3_admission_gate.py`
- runner：`experiment/phase11/run_phase11_bw3_p1_admission_recovery.sh`
