# Stage16 S16-1 数据防泄漏、Internal-Dev 与资源预检报告

> 日期：2026-08-23
> Attempt：`s16_s1_a1`
> 最终 Gate：`PASS_S16_1_DATA_LEAKAGE_RESOURCE_PREFLIGHT`
> 精确命令：`bash experiment/phase16/run_stage16_s1_data_resource_preflight.sh`

## 1. 结论

S16-1 已完成且通过。双域均只读取 Stage15 已审计的 `user_sequence_train_validation.txt` 安全投影；每行最后一个 validation 位在内存中删除，其值未写入任何 Stage16 artifact。原始 `user_sequence.txt`、test prediction、test metric 和 test target 均未打开，`test_read=false`、`network_used=false`。

数据 Gate、路径唯一性、cold/warm 分区、metadata 覆盖、SHA256 和单卡小型资源 Gate 全部通过。没有启动正式训练、完整 editing 或 validation，也没有产出科学效果指标。

## 2. 冻结的数据构造

- 固定 seed 为 `1502`。
- interaction train/internal-dev 采用确定性 SHA user rank，按用户做 90%/10% 隔离。
- pseudo-cold 只从训练投影中有合法历史的 warm item 抽取，比例为 eligible warm 的 20%。抽样按 lexical-path 长度与 catalog 文本词数五分位分层。
- pseudo-cold 从所有 student-readable train/internal-dev 序列中删除；其事件只写入明确标记为 `held_ground_truth_DO_NOT_USE_FOR_TRAINING` 的目录。
- pseudo-cold 的含义仅是“Stage16 adaptation 未见”。继承的 GRAM checkpoint 可能在其历史训练中见过这些原 warm item，因此不能称为 backbone-native unseen cold。

## 3. 双域工作量

| 指标 | Toys_cold50 | Beauty_cold50 |
|---|---:|---:|
| 安全投影用户 | 8,789 | 10,655 |
| 删除 validation 位后的 train items | 49,133 | 60,105 |
| catalog / real-cold items | 11,924 / 5,963 | 12,101 / 6,052 |
| eligible / selected pseudo-cold | 5,810 / 1,162 | 5,926 / 1,185 |
| pseudo-cold held events | 7,435 | 9,229 |
| train / internal-dev users | 7,487 / 832 | 9,147 / 1,016 |
| train / internal-dev transitions | 27,659 / 3,108 | 33,775 / 3,747 |
| S-AUX training examples | 27,659 | 33,775 |
| S-PLUS pretrain/fine-tune examples | 27,659 / 27,659 | 33,775 / 33,775 |
| G-FULL edit targets | 5,963 | 6,052 |
| G-FULL contexts（10/item） | 59,630 | 60,520 |
| G-FULL prefix-next-token requests | 302,400 | 425,890 |
| G-FULL covariance rows | 27,659 | 33,775 |

Toys real-cold lexical path 为 5 位 5,538 个、6 位 425 个；Beauty 为 7 位 5,827 个、8 位 225 个。request 数按“每个 cold item × 10 个 train-derived contexts × 每个非 EOS lexical position”完整计数。

## 4. 泄漏与完整性审计

两个域均满足：

- real-cold 出现在 student-readable items：0；
- pseudo-cold 出现在 student-readable items：0；
- train/internal-dev user overlap：0；
- validation target value 被记录：否；
- test file 被读取：否；
- lexical path collision：0；
- cold/warm 交集：0，且两者并集严格等于 catalog；
- metadata item 集严格等于 lexical catalog。

所有输入均重算并匹配冻结 SHA。完整 manifest 见 `artifacts/phase16/s1_data_resource_preflight/input_file_sha256.json`、`open_file_manifest.json`、`data_provenance.json` 与各域 `split_manifest.json`。

## 5. 小型 GPU 资源探针

Admission 时自动选择物理 GPU 7（NVIDIA RTX A6000），空闲 15,609 MiB、利用率 23%；进程仅暴露一张卡。三项探针总耗时 18.54 秒，最高进程内 allocated 峰值 1,457.58 MiB，低于 8,192 MiB 小实验上限，且远低于 600 秒 hard timeout。

| 探针 | 执行性质 | 批量/步数 | 耗时 | 峰值 allocated |
|---|---|---:|---:|---:|
| S-AUX | `RESOURCE_PROXY_NOT_SCIENTIFIC_EXECUTION` | batch 64 | 10.40 s | 1,015.53 MiB |
| S-PLUS | 真实冻结 GRAM forward/backward | microbatch 1 | 3.21 s | 1,457.58 MiB |
| G-FULL | 真实冻结 GRAM、layer-3 FFN residual hook | 1 request × 3/30 z steps | 3.95 s | 306.91 MiB |

S-AUX 代理复现官方 8-expert `1024→300` MoE 的主要张量代数，以及 2-layer/2-head/300-hidden/256-inner 尺寸；由于当前环境没有 RecBole，Transformer kernel 使用 PyTorch 代理。它只支持资源 admission，不证明官方 UniSRec 已执行，也不进入科学表。S-PLUS 探针使用真实 Beauty checkpoint、21×128 encoder passages 和 9 decoder tokens，但完整 contrastive/index 构建留到 S16-2。G-FULL 探针不物化权重更新、不替代完整 425,890 requests。

## 6. 后续大实验资源冻结与不确定性

S16-1 的 telemetry 足以冻结保守 admission，不足以精确承诺 GPU-hours。后续大实验均要求用户指定 GPU，当前未授权启动。

| Workload | GPU | 每卡最小空闲 | 保守显存预留 | 时间边界 | 单次 hard timeout | 磁盘 |
|---|---:|---:|---:|---|---:|---:|
| S-AUX formal | 1 | 24,576 MiB | 20,480 MiB | 约 18–48 h；S16-2 native sweep 后收窄 | 48 h | 8 GiB |
| S-PLUS formal | 1 | 24,576 MiB | 20,480 MiB | 当前仅能判为数十 GPU-day 级高风险；完整目标/批量 sweep 前不得启动 | 72 h/可恢复 segment | 16 GiB |
| G-FULL formal | 1 | 24,576 MiB | 20,480 MiB | 425,890 requests × 30 steps，约 7–200 GPU-day 宽区间；必须先做分批吞吐与恢复合约 | 7 d/可恢复 segment | 32 GiB |

第一个可进入 S16-2 的大实验是 S-AUX native faithful training/admission。冻结模板命令为：

`CUDA_VISIBLE_DEVICES=<USER_GPU> bash experiment/phase16/run_stage16_s2_saux_formal.sh`

该 runner 尚需在 S16-2 实现并通过 native RecBole/UniSRec contract；本报告中的命令是启动接口冻结，不表示现在已经可运行。S-PLUS 与 G-FULL 必须先完成各自 objective-complete / batching sweep，不能按本次冷启动单步数据直接线性外推后启动。

## 7. 工件与可复现性

- 主摘要：`artifacts/phase16/s1_data_resource_preflight/summary.json`
- 状态：`artifacts/phase16/s1_data_resource_preflight/status.json`
- 工作量：`artifacts/phase16/s1_data_resource_preflight/workload_counts.json`
- 资源：`artifacts/phase16/s1_data_resource_preflight/resource_probe_summary.json`、`resource_summary.json`
- 命令：`artifacts/phase16/s1_data_resource_preflight/command_manifest.json`
- 代码哈希：`artifacts/phase16/s1_data_resource_preflight/code_sha256.json`

最终状态为 `COMPLETED`、exit code 0、4/4 preflight steps；12/12 Stage16 unit tests 通过，`automatic_retry=false`。

## 8. 决策

Gate 固定为 `PASS_S16_1_DATA_LEAKAGE_RESOURCE_PREFLIGHT`。下一阶段是 S16-2 faithful SpecGR→GRAM implementation、contract 与 admission。根据计划，大实验启动前必须由用户指定 GPU；在此之前只可继续 CPU 实现、contract tests 以及符合“小实验”定义的 objective-complete resource smoke。
