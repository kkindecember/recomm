# Stage 17 S0：证据、源码、数据与资源审计报告

- Step：`S17-0`
- 日期：2026-08-29
- 终态：`COMPLETED`
- 审计结论：`PASS_S17_0`，解锁 `S17-1`
- 本步性质：证据与资源审计，不产生可用于方法优劣选择的正式效果结论
- 数据边界：`test_read=false`，`sports_read=false`

> 资源口径更正（2026-08-29）：本报告原先把“当前可能只能提供约 2 张卡”写成了全阶段硬上限。研究者已澄清：Stage17 不设固定 GPU 数量上限；大实验按实际并行收益申报所需数量并等待分配。本文的显存与 wall-time 实测保持有效，所有“最多 2 GPU”规划文字均由本更正取代。

## 1. 结论摘要

S17-0 已完成计划中的六项任务：候选来源冻结、Phase12 HI-GRAM 取证、Beauty/Toys D0/D1/D2 安全投影、lexical ID/EOS/Trie 合约复核、GRAM 100/1k 用户资源探针，以及 idea/migration/status/attempt 登记结构。

主要结论如下：

1. 已登记 14 个可迁移来源。4 个仓库完成本地 commit、关键文件与根许可证审计；另外 9 个官方仓库冻结了 2026-08-29 的远端 HEAD，但本地文件/许可证审计仍待后续按需完成；OneTrans 未核验到官方完整代码，只迁移论文思想。
2. Phase12 HI-GRAM 的历史产物存在状态、epoch 和 test-read 矛盾，不能作为 Stage17 的确认基线；后续必须重新训练 GRAM-B0/GRAM-Continue。
3. 原计划的直接截断会在 D2 把官方 validation 留在 GRAM loader 的 `[-2]`。现已改为 `train_prefix + shadow target + train-prefix guard`，六个 shadow fold 均锁定 SHA256，并通过泄漏单测。
4. Beauty/Toys lexical path 都是“主体固定深度 + 少量碰撞消歧后缀”的变长路径；没有重复路径、严格前缀路径或 T5 特殊 token 冲突。EOS 不写入路径文件，由 tokenizer/collator 追加。
5. canonical GRAM probe 在 RTX A6000 GPU4 上完成：100 用户 83 秒、1k 用户 677 秒；最大 process-local CUDA reserved memory 为 21,916 MiB。按 1k 探针线性外推，Toys D0 30 epoch + 1 次 validation 约 24.7 小时/arm，只作资源规划，不作正式 runtime 承诺。
6. 当前资源可用性通常按 **1–2 张 GPU、每个单卡 job 约 30 GiB** 估算，但不构成数量硬上限。S17-3/S17-5/S17-7 应按独立 arm 数和并行收益申报实际所需卡数，研究者分配不足时再分波；S17-1/S17-2 的小 probe 继续选择当前空闲合格卡。

## 2. 来源与许可证审计

完整机器可读登记见 `artifacts/phase17/s0_audit/source_manifest.json`，登记源见 `experiment/phase17/registry/source_registry.json`。第三方仓库只做静态阅读与哈希审计，本步没有执行第三方代码。

### 2.1 P0 七方向

| Track | 来源与冻结版本 | 可迁移机制 | 许可证/代码使用边界 |
|---|---|---|---|
| A0 | BEAR `9df41d3a61158c8b4f4266f693831883b906ff63` | top-K/beam survival-aware token loss | 本地核验 MIT；可按许可证保留归属后借鉴代码 |
| A1 | GenRet `a1d80e1506de7e4d5b21afa7c10e08ebff8c104e` | progressive autoregressive ID curriculum | 远端 HEAD 已冻结，未核验根许可证；只独立实现思想 |
| B0 | MINDER `3ea5a0fe9c39e3b9aaca74cc9213c520d26e82b6` | multiple identifier views + item aggregation | 本地核验 GPL-3.0；未做显式 GPL 兼容性决定前不复制进 GRAM |
| B1 | Latte `05e4e6d983225bcb7172f148a076890e80c524d1` | latent root、多路径与 item aggregation | 网页许可证信号为 MIT，但本地 clone/文件核验未完成；当前独立实现 |
| C0 | OneTrans，无官方完整仓库 | 序列/非序列信息逐层双向交换 | 论文思想迁移，不声称代码复现 |
| D0 | MQSA-TED `91c9001e9d3396d2762006772eac3c9833930606` | train-only transition teacher + distillation | 本地未发现根许可证；只独立实现 |
| E0 | LISRec `b4d0b1a0db4d1c9fcb9eae949cde533564cfb6dd` | semantic shortcut 选择与额外 FiD branch | 本地核验 MIT；可按许可证借鉴代码 |

P1/P2 的 Pctx、SETRec、MAERec、DCRec、BlossomRec、SPRINT/GenRec、UGR 也已冻结远端 HEAD。它们在进入实现前仍要补本地关键文件和许可证审计；冻结 commit 不等于允许复制代码。

### 2.2 迁移原则

- 本阶段目标是把可归因机制放进 GRAM，不追求论文系统的 1:1 复现。
- MIT/Apache-2.0 来源可在保留归属与许可证要求后借鉴实现；GPL-3.0 需要单独做兼容性决定；无根许可证或未本地核验的来源只做独立实现。
- 每张 P0 migration card 已明确 `borrowed_mechanism`、`not_reproduced`、GRAM insertion point、controls 和机制指标。

## 3. Phase12 HI-GRAM 历史取证

审计只读取 status、日志、checkpoint 名和 prediction 文件名，没有读取 prediction 内容。总判定为 `PHASE12_HI_GRAM_NOT_ADMISSIBLE_AS_CONFIRMED_BASELINE`。

| Run | 声明状态 | epoch 证据 | 主要矛盾 | 处理 |
|---|---:|---:|---|---|
| `beauty_v1` | succeeded | 完成 19/30，epoch20 开始但未完成 | “成功”与未完成计划冲突 | 仅保留历史信号，必须重训 |
| `beauty_v1_nan_bug_20260805` | failed | 完成 17/30，观察到 NaN | 已明确失败 | 不作为基线 |
| `smoke_beauty` | succeeded | 1/1 | status 声明 `test_read=false`，但存在 test 日志/文件 | 不可作为隔离证据 |
| `toys_v1_light` | running | 30/30 | runner 已死；同时存在 test 证据但 status 声明未读 test | 历史指标不得用于 Stage17 选择 |

`toys_v1_light` 的历史 validation Hit@10/NDCG@10 为 0.11977/0.07626，test 为 0.09710/0.05973；这些数字只用于说明取证对象，不是 Stage17 baseline，也不能参与选方法。

## 4. D0/D1/D2 数据合约

### 4.1 修正后的投影

对每个用户只在一次性 projection job 中打开原始序列：

```text
shadow_seq = train_prefix + [shadow_validation_target] + [guard_item]
guard_item = train_prefix[0]
```

GRAM loader 的 `shadow_seq[:-2]` 恰为训练 prefix，`shadow_seq[-2]` 恰为 shadow target；`shadow_seq[-1]` 是训练内 guard，只占据 loader 的 test 槽且禁止评估。D0/D1/D2 的 target 分别取原序列 `[-5]`、`[-4]`、`[-3]`，官方 `[-2:]` 从未序列化到 shadow 数据。projection job 不输出 heldout item 值，下游 job 禁止打开原始 monolithic sequence。

### 4.2 用户数与哈希

| Domain/Fold | 原用户数 | 输出用户数 | 因 target 前无历史而排除 | `user_sequence.txt` SHA256 |
|---|---:|---:|---:|---|
| Beauty D0 | 22,363 | 15,201 | 7,162 | `aa3301afe220f5625e546667da795ffc38dba23fb0474bd2fbe0c13a6f4e0c80` |
| Beauty D1 | 22,363 | 22,363 | 0 | `ba787ef0eddadf7c357ed7d42aabc62c9266d1f06162a3b49f8edb8d88718d69` |
| Beauty D2 | 22,363 | 22,363 | 0 | `df240e2a5d42630407194775a3780eab8a7f4821437f4d085a551d48e4b59936` |
| Toys D0 | 19,412 | 12,833 | 6,579 | `24e92f46fc21e0192f8f0764c2c79e166c3636c79fb2ef1a4119491dde7be1fa` |
| Toys D1 | 19,412 | 19,412 | 0 | `0831619b4715093ad57c7cb23e474d69e639e3fef747f84f24766952a3de77da` |
| Toys D2 | 19,412 | 19,412 | 0 | `65482cc4cb4e7585cdc4540e2a2f43e77c0bfc6de225afb4ab9ea1fd108c7748` |

原始 Beauty/Toys sequence SHA256 分别为 `47197cbee7bdc1896926e10d96b807dce0aecb9a1f0c2ccc887c21090237c9b7` 与 `9c2f2cce5323c5e5fd840896a68f0274aba4c241c54f0bc77ac9183094004071`。完整字段级 manifest 的 SHA256 为 `db8673822062534e58060c711116bfed7bdf552c3cbd7c1c7ca13ca25d594f95`。

## 5. Lexical ID、EOS 与 Trie 合约

| Domain | items/unique paths | 路径长度分布 | 重复路径 | 严格前缀路径 | 特殊 token 冲突 |
|---|---:|---|---:|---:|---:|
| Beauty | 12,101 / 12,101 | 7: 11,668；8: 433 | 0 | 0 | 0 |
| Toys | 11,924 / 11,924 | 5: 11,062；6: 862 | 0 | 0 | 0 |

结论：`c128_l7` 与 `c32_l5` 是 nominal depth，少量 item 带一层碰撞消歧后缀，因此后续模块必须按真实 `L` 或相对深度 `d/L` 处理，不能硬编码全等长。路径文件不序列化 EOS；`CollatorGRAM.encode_target_split()` 过滤分隔符并保留/补入 tokenizer EOS。当前路径集合满足 Trie 唯一性和无严格前缀的前置条件。

## 6. GRAM 资源探针

### 6.1 配置与隔离

canonical runner：

```bash
S17_PROFILE_ATTEMPT=attempt_003 \
  bash experiment/phase17/run_stage17_s0_resource_profile.sh start 4
```

配置为 T5-small、GRAM-B0、Toys D0、1 epoch、batch size 16、gradient accumulation 8、beam 50、`cf0_phase9=1`、validation-only、`save_predictions=0`。100/1k 用户由 target-independent user-id hash 选取。两份日志均有 `PROFILE_RESULT rc=0`，没有 test 调用或 test prediction 证据。

启动时 GPU4 telemetry 为 15,187 MiB used、33,383 MiB free、20% utilization；服务器当时没有完全空卡，runner 等到该卡满足 ≥27,000 MiB free 且 ≤20% utilization 才加载模型。device-level telemetry 包含既有外部进程，因此显存结论以 PyTorch process-local 指标为准。

### 6.2 结果

| 用户数 | 端到端 wall | 训练 wall | validation wall | peak allocated | peak reserved |
|---:|---:|---:|---:|---:|---:|
| 100 | 83 s | 25.91 s | 30.35 s | 15,750.99 MiB | 21,652 MiB |
| 1,000 | 677 s | 220.27 s | 430.12 s | 15,750.99 MiB | 21,916 MiB |

两点线性外推到 Toys D0 的 12,833 用户：单 epoch 训练约 2,775.7 秒，一次 validation 约 5,700.4 秒；30 epoch + 1 次 validation 约 88,970.8 秒，即 24.7 小时。该估计不含不同迁移模块的额外计算，也没有建模共享服务器争用，只用于 GPU 申请与队列规划。

Phase12 Toys full run 的历史峰值 reserved 约 25,008 MiB。在研究者给定的约 30 GiB/卡预算下，baseline 可容纳但仅余约 5 GiB 增量空间；新增模块必须先 profile，必要时减小 micro-batch、增加 gradient accumulation/checkpointing 或采用 lite 结构。

### 6.3 后续 GPU 规划

| 步骤 | 单 job | 建议申请 | 说明 |
|---|---:|---:|---|
| S17-1 / S17-2 | 1 × A6000 | 不申请固定大配额 | 每次小 probe 选当前空闲单卡 |
| S17-3 | 1 GPU/arm，≤约 30 GiB | 按有效并行 arm 数申报 | 当前通常先按 1–2 卡估算；若更多 standalone arm 并行有价值则如实申请 |
| S17-5 | 1 GPU/arm，≤约 30 GiB | 按双域/arm 数申报 | 获得几张卡就按对应并行度分波 |
| S17-7 | 1 GPU/arm，≤约 30 GiB | 按 seed 数与时限申报 | 资源不足时串行，不预设两路硬上限 |
| S17-8 heavy | 先单卡 profile | profile 后申报 | full 变体先确认每 arm 显存，再决定卡数与 batch/checkpoint/lite 降级 |

本步本身是小型资源探针，不触发正式大实验 GPU 申请。

## 7. 尝试记录与工程修正

本报告汇总整个 S17-0，不为每次试错单独生成 report。

1. 单元测试第一次运行有 2 个错误：合成测试数据位于仓库外，manifest 强制 `relative_to(ROOT)`。修正为仓库内相对路径、仓库外绝对路径后，最终 8/8 通过。泄漏、未知 item、重复 lexical path、status test/sports 常量和 Phase12 矛盾检测均覆盖。
2. `attempt_001` 在训练前失败：绝对 `--prompt_file` 触发 GRAM 旧 `log_name()` 路径切片 bug。恢复默认相对 prompt 后再启动，没有覆盖失败日志。
3. `attempt_002` 暴露当前执行环境对 tmux server 的控制组回收问题；两个 GRAM 子进程后来完成，但 canonical 文件已归档，未接纳该 attempt。
4. `attempt_003` 的两个 probe 与 finalizer 均成功。科学/资源工作完成后，wrapper 因运行期间 shell 源文件被修改而在退出解析时报错；终态依据两个 `PROFILE_RESULT rc=0`、两组终端 `RESOURCE_METRIC`、无 traceback 和 `PASS_VALIDATION_ONLY_RESOURCE_PROFILE` 恢复为 `COMPLETED_WITH_POSTRUN_CONTROL_RECOVERY`。后续不得修改运行中的 runner；S17-1 要把后台控制改成独立不可变 launcher 版本。

所有失败/恢复证据在 `artifacts/phase17/s0_audit/resource_profile/attempts/` 和稳定 status 中保留，没有静默重试或把失败伪装为科学成功。

## 8. 产物清单

- `experiment/phase17/registry/source_registry.json`：来源、commit、许可证策略与关键文件
- `experiment/phase17/registry/idea_registry.yaml`：P0 portfolio 与插入位置
- `experiment/phase17/registry/migration_cards/`：A0/A1/B0/B1/C0/D0/E0 七张迁移卡
- `experiment/phase17/schemas/`：status、attempt、migration card schema
- `artifacts/phase17/s0_audit/source_manifest.json`：来源审计
- `artifacts/phase17/s0_audit/phase12_forensic_audit.json`：Phase12 取证
- `artifacts/phase17/s0_audit/shadow_data_manifest.json`：D0/D1/D2 数据与哈希
- `artifacts/phase17/s0_audit/lexical_contract.json`：lexical/EOS/Trie 合约
- `artifacts/phase17/s0_audit/resource_profile_summary.json`：资源指标与 GPU 规划
- `artifacts/phase17/s0_audit/code_manifest.json`：本步代码哈希
- `artifacts/phase17/status/s17_s0_audit.status.json` 与 `s17_s0_gram_profile_toys_d0.status.json`：稳定状态入口

## 9. 局限与下一步

- 9 个来源当前只冻结远端 HEAD；远端 HEAD 不是永久 tag，后续真正实现对应方向前要按 frozen commit 做本地 clone、关键文件 hash 和 license 复核。
- 资源外推只有两个样本规模，且 GPU4 有外部进程；process-local 显存可信，wall-time 只适合规划。
- 本步没有比较不同方法，也没有读取官方 test 或 Sports；100/1k validation 数字不进入任何方法选择。
- S17-1 应优先实现不可变 run snapshot、原子 status writer、attempt ledger、后台生命周期测试、D0-only dataset ACL 和公共 contract tests。完成这些合约后再进入七个 P0 mechanism probe。

最终判定：`S17-0 COMPLETED`，`S17-1 UNLOCKED`。
