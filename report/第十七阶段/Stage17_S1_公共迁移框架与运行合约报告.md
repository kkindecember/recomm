# Stage 17 S1：公共迁移框架与运行合约报告

> 资源口径更正（2026-08-29）：研究者澄清“当前可能没有 4 张卡、通常约 1–2 张可用”不是全阶段最多 2 GPU 的硬上限。S17 大实验此后按实际需要申报 GPU 数量并等待分配；本报告的 S17-1 单卡 smoke、约 30 GiB/单 job 设计和其他运行合同不变。

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：implementation → contract validation → bounded GPU smoke
- Origin Date：2026-08-29
- Verification Status：`VERIFIED_CONTRACT_AND_SMOKE`
- Version Label：`stage17_s1_report_v1`

## 1. 步骤终态

- Step：`S17-1`
- 状态：`COMPLETED`
- 执行代理：Codex
- canonical attempt：`attempt_001`
- canonical experiment：`s17_s1_public_framework`
- GPU 执行窗口：2026-08-29 13:30:08–13:31:39（Asia/Shanghai）
- 结论：`PASS_S17_1_CONTRACT_AND_GPU_SMOKE`，解锁 `S17-2`
- 数据边界：`test_read=false`，`sports_read=false`
- 结果资格：本步只验证框架和运行合约，`scientific_result_eligible=false`，不比较方法效果

S17-1 已把 feature、auxiliary loss、decoder loss、generation score 与 item aggregation 接口统一到一个 module registry，并把 identity runtime 接入现有 GRAM。所有 Stage17 模块关闭时，GRAM 的 logits/loss 保持逐元素完全相等，新增参数为 0。原子 status、attempt ledger、不可变 runner snapshot、十分钟后台门槛、GPU 准入、科学状态/执行状态分离和后续运行隔离均已有可执行实现与测试。

## 2. 实际完成范围

### 2.1 公共迁移接口

新增公共实现位于 `experiment/phase17/core/`：

| 接口 | 文件 | 合约 |
|---|---|---|
| encoder/history feature hook | `feature_hooks.py` | 形状不变、finite、支持参数梯度；空链严格 identity |
| auxiliary/decoder loss hook | `loss_hooks.py` | 零权重不执行模块且返回原 parent loss；活动 loss 必须 finite scalar |
| generation score hook | `generation_hooks.py` | 只在锁定 legal-token mask 内改分数；非法路径 fail-closed |
| item aggregation | `item_aggregation.py` | 全 track 共用唯一 evaluator 路径；支持 max/logsumexp/probability-sum；K=1 严格等价 |
| score/evaluator guard | `metrics.py` | token score 重构；拒绝非 canonical、运行隔离或越界数据结果 |
| data ACL | `leakage_guard.py` | D0-only root/fold/purpose 授权；future position、fold target 与 Sports/test 名称 fail-closed |

`experiment/phase17/registry/module_registry.py` 是唯一 module registry。未知模块、重复注册、冲突 decoder loss 或 aggregation 会直接失败；每条候选轨只注册 hook，不复制 trainer/evaluator。

现有 GRAM 只做三处公共接线：

1. `GRAM/src/arguments.py` 增加 `--s17_modules`，默认空字符串；
2. `GRAM/src/main_generative_gram.py` 把开关写入 config；
3. `GRAM/src/model/gram.py` 在 encoder 输出和 loss 汇合点调用公共 runtime。

默认值不启用任何模块，因此历史 GRAM 命令保持原行为。最终 checkpoint 有 109,876,736 个参数，其中 Stage17 runtime 参数为 0。

### 2.2 运行与状态系统

`status_writer.py` 实现：

- 临时文件 + `os.replace` 的原子 JSON 写入；
- 带文件锁的 `phase17.index.json` 更新；
- `PENDING → PREFLIGHT → RUNNING → terminal` 科学状态机；
- 科学状态与执行状态分离，只有科学成功后才能进入 `RUNNING_OCCUPANCY_REPEAT`；
- heartbeat、PID、GPU snapshot、progress、command hash 和稳定 status 路径；
- 后续运行时强制 `result_selection_eligible=false`、`repeat_metrics_ignored=true`、`affects_scientific_result=false`。

`run_manager.py` 实现：

- runner/config/source snapshot 与逐文件 SHA256；
- 运行前重新核验 snapshot，避免修改正在运行的 launcher；
- 预计大于 600 秒或时长未知时强制 background；
- neutral tmux session 启动且不自动 retry；
- canonical 与 `runtime/<experiment_id>/run-NNNN` 输出树强制不相交；
- experiment/session/log/runtime 名称禁止出现暴露运行保持用途的词。

后续大实验完成后的运行保持状态只进入稳定 status；科学 report、attempt ledger 和 evaluator 都不会把它登记成新科学结果。本次 S17-1 是 84 秒小 smoke，不属于大实验，因此没有启动后续运行。

### 2.3 GPU 资源合同（已更正）

研究者澄清后的资源合同已同步到阶段计划、S17-0 报告、资源摘要和七张 P0 migration card：

- Stage17 不设固定 GPU 数量硬上限；
- 每个单卡 job 按约 30 GiB 可用显存设计；
- 小实验直接选择当时空闲合格卡；大实验必须先说明实际需要的 GPU 数量、每卡显存与少卡降级方案，得到研究者分配后启动；
- 当前可用性通常按 1–2 张估算，但若 3–4 张卡确有并行价值可以如实申请；heavy 模块先 profile，再决定卡数及 micro-batch、accumulation/checkpointing 或 lite 降级。

## 3. Contract tests

冻结 runner 启动时通过 39/39；补齐 background launcher、heartbeat 与 legacy status index rebuild 合约后，当前最终代码通过 43/43，失败 0。

| Gate | 覆盖内容 | 终态 |
|---|---|---|
| parent equivalence | 无 `s17_modules` 与显式空开关的 tiny GRAM logits/loss bitwise equal；0 新参数 | PASS |
| loss/gradient | auxiliary 权重 0 严格退化；活动 feature 参数 finite gradient | PASS |
| lexical/generation | Trie prefix/完整路径合法；非法生成路径拒绝 | PASS |
| item scoring | 手算 logsumexp、排名、K=1、score reconstruction | PASS |
| leakage | D0 ACL、no-future-read、fold isolation、train-only transition、Sports/test fail-closed | PASS |
| status | 合法/非法状态转移、science/execution 分离、heartbeat、原子 index | PASS |
| runtime isolation | snapshot 不随 live source 漂移；canonical/runtime 不相交；evaluator 拒绝非科学结果 | PASS |
| background/resource | 未知或 >10 分钟强制 background；neutral tmux；大实验 GPU 数按需申报并等待分配；当前单 job 约≤30 GiB | PASS（“最多 2 GPU”旧断言已由更正后的动态分配测试取代） |
| report | terminal step 恰好一个步骤报告；非终态不能提前发布 final report | PASS |
| S17-0 regression | shadow fold、lexical path、registry 与 Phase12 forensic tests | PASS |

证据日志：

- 冻结 attempt：`artifacts/phase17/s1_contract/attempt_001/cpu_contract_tests.log`
- 最终当前代码：`artifacts/phase17/s1_contract/postrun_contract_tests.log`
- code manifest：`artifacts/phase17/s1_contract/code_manifest.json`

## 4. 100-user GPU smoke

### 4.1 启动与 GPU 准入

外层 canonical 命令：

```bash
bash experiment/phase17/run_stage17_s1_contract_smoke.sh run
```

launcher 在同一次 preflight 内从满足“预测峰值 21,916 MiB + 4,096 MiB 余量”的卡中选最低利用率者。最终选择 GPU4；冻结快照时 GPU4 利用率 12%、空闲 33,637 MiB。此前人工预选 GPU1 后其空闲显存降至 17.3 GiB，已被排除；随后 GPU0/GPU4 利用率变化导致一次 pre-attempt admission 拒绝。该事件发生在任何 attempt/status/snapshot/训练进程创建前，不是科学失败，也没有形成隐式 retry；最终只存在一个 `attempt_001`。

内层完整 argv、环境无关配置和 command SHA256 均冻结在 config/snapshot 中。关键配置为：Toys D0 target-independent 100-user view、T5-small、GRAM-B0、1 epoch、batch 16、accumulation 8、beam 50、validation-only、`s17_modules=""`、600 秒硬超时。

### 4.2 资源与终态

| 项目 | 观察值 |
|---|---:|
| return code | 0 |
| end-to-end wall time | 84.04 s |
| GPU-hours | 0.02334 |
| training wall / peak allocated / peak reserved | 26.63 s / 15,750.99 MiB / 15,800 MiB |
| validation wall / peak allocated / peak reserved | 33.36 s / 7,845.29 MiB / 21,652 MiB |
| 30 GiB 预算剩余 | 9,068 MiB |
| test/Sports/traceback 证据 | 0 / 0 / 0 |

本次 validation 输出只证明完整训练—合法 validation 路径可运行；没有 matched method arm，且 config 明确 `scientific_result_eligible=false`，因此本报告不使用小样本 Hit/NDCG 做方向判断。

## 5. Attempt 台账与身份哈希

| Attempt | 类型 | 配置差异 | 状态 | 是否计入效果 |
|---|---|---|---|---|
| `attempt_001` | contract + 100-user smoke | 所有 Stage17 方法关闭；GPU4；validation-only | `COMPLETED` | 否 |

- repository base commit：`c12271be3d1090646e1c5e36e17f3f276f0ac821`（工作树内 Stage17 修改由 code manifest 单独冻结）
- command SHA256：`72fcd7d2d13017702ff3ef3f337e975a62c13cde1087fafdcd4934683b535a32`
- config SHA256：`fd4c5336fb07a6acefa05b32563fc87e18008d7bb0f3fb0cc92557b807e0d992`
- D0/D1/D2 data manifest SHA256：`db8673822062534e58060c711116bfed7bdf552c3cbd7c1c7ca13ca25d594f95`
- source registry SHA256：`9ce586e525f695019d47ed955e8d5d37f4618e1438ad838065b27519126b467f`
- run snapshot SHA256：`e114cedbc9a2a403b0562f5d19f03c3404214ecd3865a4d5827260ff93d9e9fe`
- GPU log SHA256：`7e35635e092fcff2f6ae634f5ae75c95c618999a6e7c1e62686e30424f482ef4`
- T5-small parent blob/revision：`dd8c1c79...a4280ffc` / `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`
- item-generator parent blob/revision：`386b29db...953912cd` / `cffe3b8589c2a9521bda72644fb3e18a40ee6ab7`
- smoke output model SHA256：`77cd2c44e345ae534afd7e06b232a477f3ae79d449dae66f2195dc65de2a3eec`

稳定入口：

- status：`artifacts/phase17/status/s17_s1_public_framework.status.json`
- phase index：`artifacts/phase17/status/phase17.index.json`
- attempt ledger：`artifacts/phase17/attempts/S17-1.attempts.jsonl`
- summary：`artifacts/phase17/s1_contract/attempt_001/summary.json`
- immutable snapshot：`artifacts/phase17/snapshots/s17_s1_public_framework/attempt_001/manifest.json`

## 6. 异常、限制与决定

1. GPU 状态变化证明“先人工看卡、再把 GPU id 传给 runner”存在 TOCTOU 竞态。launcher 已改为在同一次 preflight 内读取、选择并冻结 GPU snapshot；不再依赖几分钟前的人工卡号。
2. 当前只验证 identity runtime。A0–E0 的实际 hook 实现、非零梯度规模、专属机制指标与 accuracy 信号属于 S17-2，不在本报告中提前宣称。
3. generation hook 已有合法 mask/路径合约；具体 Hugging Face logits processor 和各 track 的深度状态要随 A0/A1 实现接入并各自测试。
4. 本步的不可变 snapshot 固化了 canonical worker。补齐的 background/heartbeat/index rebuild 代码由最终 code manifest 和 43-test 日志冻结；未以修改 runner 后重跑 GPU 的方式覆盖已完成 attempt。
5. 正式 baseline 仍需在 S17-3 按统一预算重新训练；S17-1 的 100-user checkpoint 不得作为 baseline、parent selection 或效果结论。

决定：`S17-1 ACCEPTED`。公共框架与运行合约已达到进入机制 probe 的最低门槛，下一步执行 `S17-2：P0 七方向固定预算机制 probe`。S17-2 属于小实验组合，逐个查当前空闲单卡；任何单 job 若预计超过 10 分钟仍从 neutral background runner 启动并持续更新 status，不需要申请固定大配额。
