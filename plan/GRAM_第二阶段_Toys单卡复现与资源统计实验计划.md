# GRAM 第二阶段实验计划：Toys 单卡复现与资源统计

## 0. 给 Coding AI 的任务说明

在第一阶段 Beauty 单卡复现已成功的基础上，使用同一份 GRAM 官方代码、同一个 `gram-repro` Conda 环境和同一套单卡适配原则，完成 Toys 数据集的 seed 2023 复现。

第二阶段只包含两项核心任务：

1. 复现原始 GRAM 在 Toys 上的结果；
2. 可复核地统计单次正式训练的显存和时间。

本阶段不运行消融实验。官方仓库没有提供可直接执行的消融脚本，不为了扩大实验范围而自行推测其实现。

执行原则：

1. 先完成仓库、Toys 数据、输入路径、环境、磁盘和资源统计方案的静态检查，再消耗 GPU。
2. 不覆盖官方 `command/train_gram_toys.sh`；新增 Toys 单卡、smoke 和最佳 checkpoint 测试脚本。
3. 除单卡有效 batch 适配与“只记录不改变计算”的资源仪表化外，不修改模型、数据划分、优化器、scheduler、评测逻辑或核心超参数。
4. 每次启动 GPU 任务前都请用户指定物理 GPU，不默认沿用上次设备。
5. GPU 资源依照 `/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh` 管理：实验前执行 `stop`，实验退出后立即执行 `start <GPU>` 恢复用户占用，不将 GPU 留给其他任务。
6. 所有 GPU 实验均在后台运行，完整日志、状态、PID、GPU 遥测和磁盘遥测保存到 `experiment/` 与 `artifacts/phase2_toys/`。
7. 训练后按 validation NDCG@10 选择 checkpoint，不使用最后 checkpoint 替代最佳 checkpoint。
8. 遇到 OOM、NaN、磁盘空间不足、其他进程挤占 GPU 或仪表化失效时，保留完整日志，不静默重试。
9. 每次修改记录原因、修改前后、是否可能影响数值，并归档到 `report/第二阶段/` 与 `artifacts/phase2_toys/environment/code_changes.patch`。

官方资源：

- 论文：https://aclanthology.org/2025.acl-long.1596/
- 代码：https://github.com/skleee/GRAM
- 官方 Toys 脚本：`GRAM/command/train_gram_toys.sh`

## 1. 实验目标

完成以下端到端流程：

```text
复核第一阶段环境与代码状态
    -> Toys 数据审计
    -> 单卡与资源仪表化静态检查
    -> Toys smoke test
    -> Toys 30 epoch 正式训练
    -> 按 validation NDCG@10 选择 checkpoint
    -> 最佳 checkpoint full-ranking test
    -> 导出指标、显存、时间和结论
```

第二阶段回答五个问题：

1. 第一阶段建立的单卡流程能否直接迁移到 Toys？
2. 官方 Toys 预处理数据是否完整且与论文统计一致？
3. seed 2023 的 Recall@5、NDCG@5、Recall@10、NDCG@10 是否接近论文？
4. 一次 30 epoch Toys 正式训练的墙钟时间、平均 epoch 时间和最佳 checkpoint 测试时间是多少？
5. 在共享 GPU 上，本 GRAM Python 进程的峰值 allocated/reserved 显存是多少，与整卡总占用如何区分？

## 2. 不在本阶段范围内

- 不运行任何消融实验。
- 不重新运行 NV-Embed-v2、层次聚类或 SASRec 预处理。
- 不修改 Toys 数据、用户序列、商品文本、层次 ID 或相似商品文件。
- 不添加新模型模块、损失函数、注意力、门控或其他创新。
- 不运行 Beauty、Sports、Yelp 或其他 baseline。
- 不执行多随机种子实验。
- 不为了接近论文值而调节超参数或挑选 test 结果。
- 不用整卡 `memory.used` 代替本进程显存指标。

## 3. 论文目标与实验协议

### 3.1 Toys 数据统计

论文 Table 10 与本地文件的预期统计为：

| 数据集 | 用户数 | 商品数 | 交互数 | 密度 |
|---|---:|---:|---:|---:|
| Toys | 19,412 | 11,924 | 167,597 | 0.0724% |

划分与 Beauty 一致：每个用户最后一个交互为 test，倒数第二个为 validation，此前交互为 train。评测必须对全部 11,924 个商品进行 full ranking，不使用负采样。

### 3.2 论文 Toys 主要结果

论文 Table 2 中 GRAM 的 Toys 结果：

| 指标 | 论文值 |
|---|---:|
| Recall@5 | 0.0718 |
| NDCG@5 | 0.0516 |
| Recall@10 | 0.0987 |
| NDCG@10 | 0.0603 |

论文值是多次运行的汇总结果，本阶段只运行官方脚本中的 seed 2023，不要求逐位一致。

### 3.3 官方 Toys 核心配置

| 配置 | 值 |
|---|---|
| dataset | Toys |
| backbone | T5-small |
| seed | 2023 |
| learning rate | 1e-3 |
| epochs | 30 |
| item prompt max length | 128 |
| max user history | 20 |
| ID length | 5 |
| clusters / cluster size | 32 / 32 |
| similar items | 5 |
| CF model | SASRec |
| checkpoint / validation interval | 5 epochs |
| checkpoint selection | validation NDCG@10 最高 |
| inference | constrained beam search，beam size 50 |
| evaluation | full ranking |

除单卡 batch 适配外，上述值必须与 `command/train_gram_toys.sh` 和代码默认值交叉核对。

## 4. 环境、磁盘和仓库策略

### 4.1 复用第一阶段环境

优先直接复用：

```text
Conda: gram-repro
Python: 3.9.25
PyTorch: 1.11.0+cu113
Transformers: 4.26.0
```

正式实验前重新执行 `pip check`、CUDA 可用性和 T5 缓存加载检查，但不重新建环境或升级依赖。

第二阶段环境归档保存到 `artifacts/phase2_toys/environment/`，包含：

- `base_environment_reference.md`：引用第一阶段环境快照并记录差异；
- `git_commit.txt`：GRAM commit 和当前 status；
- `nvidia_smi_before.txt`：用户选定 GPU 的型号、驱动、总显存、其他进程；
- `disk_space_before.txt` 和 `disk_space_after.txt`；
- `code_changes.patch`：相对官方代码的全部差异；
- `instrumentation_note.md`：资源统计的定义、误差和数值影响说明。

### 4.2 磁盘启动门槛

`/home` 是共享磁盘，空间会受其他用户影响。每个 GPU 任务启动前必须记录 `df -h /home`：

- 可用空间 `>= 100 GiB`：正常启动；
- `50–100 GiB`：可启动，但必须在报告中标记风险并加强监控；
- `< 50 GiB`：不启动正式训练，先报告用户。

正式训练期间每 5 分钟记录一次文件系统可用空间。不自动删除 checkpoint；最佳 checkpoint 选择和报告完成后，再由用户授权清理 optimizer、scheduler 和非最佳模型。

## 5. Toys 数据审计

将审计结果写入 `artifacts/phase2_toys/data_audit.md`，至少检查：

- `user_sequence.txt`、`item_plain_text.txt`、层次 ID 和 `similar_item_sasrec.txt` 存在、非空且可读；
- 用户数 19,412、商品数 11,924、交互数 167,597 与密度 0.0724%；
- 每个用户都能产生一个 validation 和一个 test 目标；
- 训练、validation、test 切分符合 leave-one-out；
- 用户序列中的全部商品都存在于文本与层次 ID 映射；
- 层次 ID 是否唯一，是否有空值、重复、越界或缺失映射；
- SASRec 文件是否覆盖全部 11,924 个 anchor，每个 anchor 是否至少有 5 个有效相似商品；
- full-ranking Trie 的候选数是否为 11,924；
- 审计只读数据，不“修复”或重新生成数据。

## 6. 单卡适配

新建：

```text
GRAM/command/train_gram_toys_single.sh
GRAM/command/smoke_test_gram_toys_single.sh
GRAM/command/test_gram_toys_best_single.sh
```

官方两卡配置的近似有效 batch：

```text
32 per GPU × 2 GPUs × 2 accumulation = 128
```

正式单卡初始配置：

```text
CUDA_VISIBLE_DEVICES=<用户选定的物理 GPU>
--distributed 0
--gpu 0
--rec_batch_size 16
--gradient_accumulation_steps 8
```

如发生 OOM，只允许按顺序调整：

| per-device batch | accumulation | 有效 batch |
|---:|---:|---:|
| 16 | 8 | 128 |
| 8 | 16 | 128 |
| 4 | 32 | 128 |

每次 OOM 保留日志，不静默重试。不修改 learning rate、epoch、history length、prompt length、ID 配置、相似商品数量、beam size 或评测方式。

## 7. 共享 GPU 下的显存统计

### 7.1 为什么不直接使用整卡显存

`nvidia-smi --query-gpu=memory.used` 返回整张物理 GPU 上所有进程的总占用。当 GPU 与其他用户共享时，该值不能精确代表 GRAM 的显存。

本阶段使用三层指标：

1. **PyTorch 进程内指标（主指标）**
   - `torch.cuda.max_memory_allocated(device)`：PyTorch tensor 实际分配峰值；
   - `torch.cuda.max_memory_reserved(device)`：PyTorch caching allocator 保留峰值；
   - 它们只统计当前 GRAM Python 进程，不包含其他用户进程，作为最终报告的主显存数值。
2. **NVML / `nvidia-smi` PID 级遥测（辅助指标）**
   - 每 5 秒记录一次与 `workload_pid` 精确匹配的 `used_gpu_memory`；
   - 它包含 CUDA context 等不完全由 PyTorch allocator 统计的内存，但采样可能错过短暂峰值。
3. **整卡遥测（仅背景）**
   - 同时记录 GPU index、`memory.used`、`memory.free`、utilization 和进程列表；
   - 只用于解释共享环境、OOM 和干扰，不将整卡峰值写成“GRAM 峰值显存”。

### 7.2 进程内仪表化

在不改变模型计算的前提下，为单卡入口增加最小资源记录：

1. 在 `runner.train_generator()` 前执行 `torch.cuda.synchronize()` 和 `torch.cuda.reset_peak_memory_stats()`；
2. 使用 `time.perf_counter()` 记录训练阶段开始；
3. `runner.train_generator()` 返回后再执行 `torch.cuda.synchronize()`，记录结束时间、峰值 allocated 和 reserved；
4. 将结果以明确前缀写入日志，例如 `RESOURCE_METRIC training_wall_time_seconds=...`；
5. 最佳 checkpoint 测试使用独立进程，重置峰值后单独记录 test 时间和 test 峰值。

仪表化只在阶段边界同步和读取统计值，不改 forward、backward、optimizer、scheduler、数据顺序或生成结果。必须将其记录为“可观测性修改，预期不影响数值”。

### 7.3 GPU 启动门槛

在用户选定 GPU 后：

1. 执行 `tools/run_codellama.sh stop`；
2. 等待 CodeLlama CUDA context 退出；
3. 记录目标 GPU 上的全部进程和剩余显存；
4. batch 16 配置只在剩余显存至少 40 GiB 时启动；
5. 如剩余显存不足或其他进程在高频波动，不抢占、不盲目启动，报告用户后重新选卡或等待。

其他用户后续加入同一 GPU 不会污染 PyTorch 进程内的 allocated/reserved 指标，但可能降低速度或导致 OOM。如遥测中出现新的外部进程，报告必须标记时间区间，训练时间不得冒充为独占 GPU 性能。

## 8. 时间统计口径

最终 `resource_summary.json` 至少同时报告：

| 时间指标 | 定义 |
|---|---|
| `training_phase_wall_time_seconds` | `runner.train_generator()` 的墙钟时间，包含 30 epoch、周期 validation 和 checkpoint 保存，不包含训练结束后的自动 test |
| `end_to_end_job_time_seconds` | 后台守护脚本从启动 GRAM 到 GRAM 退出的总时间，包含模型/数据加载和最后自动 test |
| `epoch_train_time_seconds` | 内部带时间戳日志中，每个 epoch 从开始到训练 loss 输出的耗时，不含该 epoch 后 validation |
| `mean_epoch_train_time_seconds` | 30 个 `epoch_train_time_seconds` 的平均值，同时报告中位数、最小值和最大值 |
| `periodic_validation_time_seconds` | 6 次 validation 的各自时间与总时间 |
| `best_checkpoint_test_time_seconds` | 独立加载最佳 checkpoint 的 full-ranking test 时间 |
| `average_inference_time_per_user_seconds` | 官方 test 日志输出的每用户平均推理时间 |

“单次训练时间”在最终结论中默认指 `training_phase_wall_time_seconds`，不用包含最终 test 的整个守护任务时间代替。

## 9. 执行阶段

### 9.1 阶段 A：静态检查和数据审计

- 复核 GRAM commit、工作区改动和第一阶段保留产物；
- 检查 Toys 四类输入与数据统计；
- 读取官方 Toys 参数和代码默认值；
- 新脚本执行 `bash -n`，Python 仪表化执行静态导入检查；
- 检查 checkpoint、预测、日志、状态、遥测目录是否可写；
- 检查 `/home` 剩余空间；
- 确认训练结束默认使用最后 checkpoint，所以后续必须独立选择最佳 validation checkpoint。

### 9.2 阶段 B：Toys smoke test

Smoke 脚本使用：

```text
1 epoch
100 条训练样本
100 条 validation/test 样本
batch size 4
gradient accumulation 1
test/save interval 1
```

Smoke 必须验证：

- Toys 数据可加载；
- forward、backward、optimizer step 正常；
- checkpoint 可保存和重载；
- validation/test 和 Trie constrained decoding 可运行；
- 主要资源指标能输出，单位正确，`allocated <= reserved`；
- PID 级遥测匹配的是 GRAM Python workload，不是外层 Bash；
- 退出后 CodeLlama 资源恢复。

Smoke 指标仅用于诊断，不与论文比较。

### 9.3 阶段 C：Toys 正式训练

使用 seed 2023、30 epoch、batch 16 / accumulation 8 启动。日志与遥测分别保存：

```text
artifacts/phase2_toys/logs/train_seed2023.log
experiment/phase2_toys/phase2_toys_train_gpu_board.csv
experiment/phase2_toys/phase2_toys_train_gpu_process.csv
experiment/phase2_toys/phase2_toys_train_disk.csv
experiment/phase2_toys/phase2_toys_status.json
```

记录：

- 守护、workload PID 和实际命令；
- 训练阶段开始/结束时间；
- 30 个 epoch loss 与耗时；
- 6 个 validation 点的 Recall/NDCG 与耗时；
- 6 组 checkpoint 的路径和大小；
- PyTorch peak allocated/reserved；
- PID 级峰值显存与整卡背景峰值；
- 其他 GPU 进程出现/退出的时间；
- 磁盘可用空间；
- NaN、OOM、卡死、数据错误和退出码。

训练软超时 72 小时，硬超时 120 小时。超过软超时不自动杀死，先记录当前 epoch、速度、GPU/磁盘状态和预计剩余时间。

### 9.4 阶段 D：最佳 checkpoint 正式测试

1. 从 epoch 5/10/15/20/25/30 中选择 validation NDCG@10 最高的 checkpoint；
2. 将选择表、checkpoint SHA-256 和选择依据写入报告；
3. 使用 `--train 0 --rec_model_path <best>` 单独运行 test；
4. 确认 19,412 个 test 用户、11,924 个候选商品、beam 50 与 Trie constrained decoding；
5. 将日志保存到 `artifacts/phase2_toys/logs/test_best_checkpoint.log`；
6. 独立记录 test 的 PyTorch allocated/reserved 峰值、PID 显存、总时间和平均每用户时间。

## 10. 指标对齐判定

将最终结果保存为 `artifacts/phase2_toys/metrics_seed2023.json`。对每项指标计算：

```text
绝对误差 = 本地结果 - 论文结果
相对误差 = |本地结果 - 论文结果| / 论文结果 × 100%
```

| 等级 | 标准 | 结论 |
|---|---|---|
| A：基本对齐 | 四项均在论文值相对 ±5% 内 | Toys 单种子复现通过 |
| B：可接受但需复核 | 至少三项在 ±10% 内，且协议一致 | 流程通过，数值待多种子复核 |
| C：未对齐 | 任一核心指标偏差超过 10%，或协议不一致 | 检查数据、checkpoint、评测和环境 |

由于只运行 seed 2023，轻微偏差不能单独视为复现失败。严禁通过修改测试集、候选商品数、beam size、评测公式或 checkpoint 挑选规则提高数值。

## 11. 资源输出格式

`artifacts/phase2_toys/resource_summary.json` 至少包含：

```json
{
  "dataset": "Toys",
  "seed": 2023,
  "gpu_physical_index": null,
  "gpu_name": null,
  "gpu_total_memory_mib": null,
  "shared_gpu": true,
  "measurement_primary": "pytorch_process_allocator",
  "training_phase_wall_time_seconds": null,
  "end_to_end_job_time_seconds": null,
  "mean_epoch_train_time_seconds": null,
  "median_epoch_train_time_seconds": null,
  "periodic_validation_total_time_seconds": null,
  "best_checkpoint_test_time_seconds": null,
  "average_inference_time_per_user_seconds": null,
  "training_peak_allocated_mib": null,
  "training_peak_reserved_mib": null,
  "training_nvml_pid_peak_mib": null,
  "test_peak_allocated_mib": null,
  "test_peak_reserved_mib": null,
  "test_nvml_pid_peak_mib": null,
  "board_peak_used_mib_context_only": null,
  "external_gpu_interference_observed": null,
  "notes": []
}
```

所有 byte 值以 MiB（1 MiB = 1024² bytes）和 GiB（1 GiB = 1024³ bytes）同时报告，不把 MB/GB 与 MiB/GiB 混用。

## 12. 异常排查顺序

指标异常时：

1. 是否加载 Toys 和 `hierarchy_v1_c32_l5_len32768_split`；
2. `NUM_CF=5`、`NUM_CLUSTER=32`、`ID_LEN=5` 是否正确；
3. 是否为 11,924 商品 full ranking；
4. 是否使用 validation NDCG@10 最佳 checkpoint；
5. beam size 是否为 50，Trie 是否启用；
6. `item_prompt=all_text`、`id_linking=1`、`max_his=20` 是否正确；
7. 单卡有效 batch、optimizer step 和 scheduler 步数是否与官方等价；
8. T5 tokenizer/model 版本、seed 和确定性配置是否与第一阶段一致；
9. 是否存在未记录的源码、数据或依赖改动。

资源异常时：

1. 确认记录的 PID 是 GRAM Python workload；
2. 确认 `CUDA_VISIBLE_DEVICES` 下物理 GPU 与逻辑 `cuda:0` 的映射；
3. 区分 PyTorch allocated、PyTorch reserved、PID NVML 和整卡 used；
4. 检查训练期间是否有其他进程加入；
5. 训练时间异常时检查 GPU utilization、外部进程、数据加载和磁盘等待；
6. 仪表化字段缺失时，不用整卡遥测冒充精确进程峰值。

## 13. 预计资源与超时

参考第一阶段 Beauty 在 RTX A6000 上 40.47 小时的端到端正式任务，Toys 用户和交互数略少，但实际时间受共享 GPU 竞争、prompt 长度、validation 和存储 I/O 影响。

计划预算：

- Smoke：小于 15 分钟；
- 30 epoch 正式训练任务：预估 30–45 小时，软超时 72 小时；
- 最佳 checkpoint test：预估 1–2 小时；
- 新增 checkpoint、optimizer、预测、日志和遥测：预留 10–15 GiB；
- 磁盘安全余量：正式训练启动时至少 50 GiB，推荐 100 GiB。

不把预估时间当作实验结果；最终只报告仪表化和日志实测值。

## 14. 最终交付物

```text
artifacts/phase2_toys/
├── REPRODUCTION_REPORT.md
├── data_audit.md
├── metrics_seed2023.json
├── resource_summary.json
├── epoch_timing.csv
├── checkpoint_selection.csv
├── environment/
│   ├── base_environment_reference.md
│   ├── git_commit.txt
│   ├── nvidia_smi_before.txt
│   ├── disk_space_before.txt
│   ├── disk_space_after.txt
│   ├── instrumentation_note.md
│   └── code_changes.patch
└── logs/
    ├── smoke_test.log
    ├── train_seed2023.log
    └── test_best_checkpoint.log
```

仓库新增：

```text
GRAM/command/train_gram_toys_single.sh
GRAM/command/smoke_test_gram_toys_single.sh
GRAM/command/test_gram_toys_best_single.sh
experiment/phase2_toys/run_phase2_toys_smoke.sh
experiment/phase2_toys/run_phase2_toys_train.sh
experiment/phase2_toys/run_phase2_toys_best_test.sh
experiment/phase2_toys/phase2_toys_status.json
experiment/phase2_toys/phase2_toys_*_gpu_board.csv
experiment/phase2_toys/phase2_toys_*_gpu_process.csv
experiment/phase2_toys/phase2_toys_*_disk.csv
report/第二阶段/
```

## 15. `REPRODUCTION_REPORT.md` 必须包含

1. 最终结论：成功、部分成功或失败；
2. A/B/C 对齐等级；
3. 硬件、软件、物理 GPU 与逻辑 GPU 映射；
4. 共享 GPU 情况和显存统计口径；
5. 数据审计结果；
6. 实际执行命令和有效 batch 说明；
7. 对官方代码的全部修改及数值影响；
8. Smoke test 结果；
9. 30 epoch loss、epoch 耗时和 6 个 validation 点；
10. 最佳 checkpoint、SHA-256 和选择依据；
11. 四项正式 test 指标；
12. 与论文 Table 2 的绝对/相对误差；
13. 训练阶段时间、端到端任务时间、平均 epoch 时间和最佳 test 时间；
14. PyTorch allocated/reserved、PID NVML 峰值和整卡背景值；
15. 外部 GPU 干扰是否发生，时间数据是否可视为独占 GPU 性能；
16. 异常、未解决问题、磁盘风险和下一步建议；
17. 从空终端可重新启动同配置实验的完整命令。

## 16. 第二阶段完成条件

- [x] 复用的独立环境已重新验证；
- [x] Toys 数据未修改且统计与论文一致；
- [x] Toys smoke test 通过；
- [ ] 30 epoch seed 2023 单卡训练完成；
- [ ] 6 个周期 checkpoint 与 validation 结果已归档；
- [ ] 最佳 checkpoint 按 validation NDCG@10 选出并重新加载；
- [ ] 11,924 商品 full-ranking test 完成；
- [ ] Recall@5、NDCG@5、Recall@10、NDCG@10 已输出；
- [ ] 与论文 Table 2 的误差和 A/B/C 等级已给出；
- [ ] 训练阶段墙钟时间、平均 epoch 时间与独立 test 时间已给出；
- [ ] PyTorch 进程内 allocated/reserved 峰值已给出；
- [ ] PID 级遥测与共享 GPU 背景已归档；
- [ ] 实验退出后 CodeLlama 已立即重新占用用户指定 GPU；
- [ ] 所有日志、状态、代码修改和环境差异已归档；
- [ ] 从空终端可按报告命令重新启动。

## 17. 给 Coding AI 的最终汇报格式

```markdown
## 执行结论
- 状态：成功 / 部分成功 / 失败
- 对齐等级：A / B / C
- 最佳 checkpoint：...

## 核心指标
| 指标 | 论文 | 本地 | 绝对误差 | 相对误差 |
|---|---:|---:|---:|---:|
| Recall@5 | 0.0718 | ... | ... | ... |
| NDCG@5 | 0.0516 | ... | ... | ... |
| Recall@10 | 0.0987 | ... | ... | ... |
| NDCG@10 | 0.0603 | ... | ... | ... |

## 时间
- 30 epoch 训练阶段（含周期 validation/checkpoint，不含最后 test）：...
- 平均 / 中位 epoch 纯训练时间：...
- 端到端正式任务：...
- 最佳 checkpoint test：...
- 平均每用户推理时间：...

## 显存
- GPU：...
- 是否共享：...
- 训练 peak allocated / reserved：...
- 训练 PID NVML 峰值：...
- Test peak allocated / reserved：...
- 整卡背景峰值（不作为模型峰值）：...
- 是否观察到外部进程干扰：...

## 修改
- 修改文件：...
- 修改原因：...
- 是否可能影响数值：...

## 未解决问题
- ...

## 产物路径
- 最终报告：...
- 日志：...
- 指标 JSON：...
- 资源 JSON：...
- 单卡脚本：...

## 可复现命令
`...`
```
