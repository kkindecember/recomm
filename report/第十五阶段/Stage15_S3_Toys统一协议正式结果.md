# Stage15 S3：Toys 统一协议正式结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-22
- Verification Status: PARTIALLY_VERIFIED（S15-3A 已完成；S15-3B full validation 尚未授权）
- Version Label: stage15_s3_toys_v1_s3a_complete

## 当前结论

- B2：`PASS_S15_3A_B2_ITEM_DISJOINT_ADMISSION`，允许进入 S15-3B full validation。
- B3：`FAIL_B3_S15_3A_EDIT_STATE_ADMISSION`，不进入 S15-3B；该结论仅表示当前 GenRecEdit-GRAM port 的 edit-state 兼容失败，不等价于 GenRecEdit 方法无效。
- test：未打开。
- automatic retry：false。

S15-3A 只作工程与协议 admission，不作 efficacy 或显著性结论。

## B2 512-event admission

### 执行结果

| 字段 | 结果 |
|---|---:|
| exact command | `bash experiment/phase15/run_stage15_s3a_toys_b2_only_admission.sh start 7` |
| workload exit code | 0 |
| held events | 512/512 |
| runtime | 3,154.16 s（52m34s） |
| peak CUDA allocated | 6,935.35 MiB |
| B2 train transitions | 4,096 |
| B2 loss | `9.36309439 → 9.24616081` |
| B2 trainable parameters | 1,346,912 |
| verifier candidate forwards | 25,600（512×50） |
| accepted drafts | 270 |
| B2 rankings differing from B0 | 164/512 |

全部 512 个事件均完成 B0 beam-50 与 B2 draft→verify→redraft 路径；所有 ranking 均为 50 个唯一、已知 catalog item。B2 loss/score finite、drafter state 改变，冻结 clean base 的运行前后 SHA256 均为 `cadd9eccec616ef85a00c17ef1459cfb46ff34a958c41f43175b67b153072ffd`。

held target 只在 B2 state 完成后用于 evaluation，没有参与训练、状态选择或超参数选择；test 未打开。

### 非推广性指标

| Arm | Hit@50 events | MRR |
|---|---:|---:|
| B0 | 6/512 | 0.0004434164 |
| B2 | 6/512 | 0.0004256444 |

这些指标只证明完整 evaluator 已执行。S15-3A 未设计显著性检验，不能据此宣称 B2 优于、等于或劣于 B0。

### Verdict 修正记录

workload 首次写出的 verdict 为 `FAIL_S15_3A_B2_ITEM_DISJOINT_ADMISSION`，但这是 reducer 的确定性布尔方向错误：`admission_checks` 同时保存了 `held_ground_truth_used_for_training_or_state_selection=false` 与 `test_opened=false`，随后错误执行 `all(admission_checks.values())`，把两个正确的安全事实判为失败。

现已将 Gate 改为正向断言：

- `held_ground_truth_not_used_for_training_or_state_selection=true`
- `test_not_opened=true`

其余 admission checks 原本全部为 true。现有 512-event 完整 artifact 足以确定性重算 verdict，无须重跑模型；summary 同时保留 `original_emitted_verdict` 与 correction reason，回归测试覆盖 PASS/FAIL 两条路径。

runner 日志在完整 summary 与 completed status 写出后出现一次 shell 尾部解析错误，原因是任务运行期间补入 status 观察接口导致长生命周期 shell 读取到变更后的文件尾。该异常未影响 workload、summary、模型 hash 或 512 个预测；当前脚本重新执行 `bash -n` 必须通过，后续禁止在活跃 run 中修改其 runner 文件。

## B3 admission 证据

| Attempt | Layer rule | Position z-success | 结果 |
|---|---|---|---|
| attempt-1 | 复用 v0 probe `[5,5,5,5,5,4]` | positions 0–3=`[2,1,2,1]/4`；position 4=`0/4` | rc=1，held 未打开 |
| attempt-2 | clean-base train-only 6×6 probe；selected=`[5,5,3,5,0,0]` | positions 0–3=`[2,1,2,1]/4`；position 4=`0/4` | rc=1，held 未打开 |

attempt-2 中 positions 4/5 在全部 6 层的 token accuracy 均为 0。按冻结纪律，不放宽 legal probability threshold=`0.3`，不增加 requests、不改 tie-break、不换 seed；B3 不进入当前 S15-3B。若继续 B3，必须另立 recovery 计划并标为 exploratory。

### B3 exploratory branching recovery

后续静态复算发现，失败并非首先由 position-4 layer accuracy=0 导致，而是 request admission 集合含有结构上不可满足成功判据的样本：attempt-2 的 position-4 四个请求 legal branching factor 均为 1。此时 target 在 legal set 中的 baseline probability 恒为 1，而当前成功判据要求 edited probability 严格大于 baseline，因此任何 layer、z residual 或优化步数都不可能成功。

恢复修复只在冻结 catalog trie 上排除 branching factor=1 的请求，然后执行原有 SHA rank 与 distinct-cold selection。真实 artifact 的修复后 branching factors 为：position 3=`[2,2,2,4]`，position 4=`[22,2,3,2]`；其余位置仍为非平凡分支。seed=`1502`、4 requests/position、layer probe、z steps、learning rate、weight decay、max norm、preservation lambda 与 legal probability threshold=`0.3` 均保持不变，也不读取 held/validation outcome。

该恢复标记为 exploratory。它不会把 B3 追加到已运行的 B0/B1/B2 S15-3B；只有独立 recovery 512-event admission 完成全部 One-One path、finite/nonzero delta、unique known top-50、base hash unchanged、held-after-state 和 test sealed Gate 后，才允许另行安排 B3 full validation。

branching recovery attempt-1 已验证 edit-state 根因修复：positions 0–5 的 successful z requests 分别为 `[2,1,2,3,2,1]/4`，六个 covariance/deltaW bundle 全部成功写出。随后在第一个 edited beam 前由 Transformers 4.21 generation kwargs 校验失败：One-One context 的动态 `prepare_inputs_for_generation` wrapper 将原 T5 的显式 `encoder_outputs` 参数折叠进 `**kwargs`，而 GRAM `forward` 也只通过 `**kwargs` 转发，导致旧版校验器将实际使用的 `encoder_outputs` 误报为 unused。该 attempt rc=1、`0/512`，独立 artifact 保留；修复必须恢复原 encoder-decoder generation 参数的显式签名，不允许改 edit-state 超参数。

branching recovery attempt-2 已越过上述 kwargs 校验并再次完成六位置 edit state，但在首个 constrained B3 beam 暴露另一项旧版 Transformers 行为：当某个 frozen trie 层的合法 children 少于 `num_beams=50` 时，beam search 会用 score=`-inf` 的非法 dead rows 补满内部 beam slots。它们不可能成为最终输出，且后续仍受同一 prefix constraint 限制；旧 One-One hook 却在模型前向前将这些 dead rows 当作活跃 lexical prefix 拒绝，故于 20:05:15 rc=1、`0/512`。修复仅对 trie 外 dead rows 禁用 delta 并累计披露其数量，合法活跃 rows 仍按当前 lexical position 应用 position-wise delta，最终输出仍必须通过 exact catalog、unique top-50 和 base-hash checks；未更改 seed、beam budget、catalog、B2/B3 状态或 held/test 边界。对应边界测试与完整 Stage15 tests 47/47 PASS，attempt-2 artifact 原样保留。

## 下一 Gate

S15-3B 只允许 B0、B1、B2 进入 Toys full validation。启动前必须冻结：full validation event 数、paired bootstrap seed/10,000 resamples、cost telemetry、显存与 hard timeout、exact background command 和独立 `status.json`。在该资源合约获得确认前不自动启动。
