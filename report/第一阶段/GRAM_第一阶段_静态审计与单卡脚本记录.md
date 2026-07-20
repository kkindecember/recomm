# GRAM 第一阶段静态审计与单卡脚本记录

日期：2026-07-17

## 本次完成内容

在不使用 GPU、不下载模型、不修改数据的前提下完成 Beauty 仓库与数据审计、训练入口静态检查，并新增正式单卡脚本和独立 smoke-test 脚本。

## 代码与环境检查

- GRAM commit：`7ac4d9272a57beed9df35c27ea34221f6e4a8fb1`，官方子仓库修改前工作区干净。
- `src/main_generative_gram.py`、官方 Beauty 脚本和四类 Beauty 输入均存在。
- `gram-repro` 中 `pip check` 通过；Python 静态导入通过。
- 正式脚本和 smoke 脚本均通过 `bash -n`（创建后复查）。
- 当前 `torch.cuda.is_available()` 为 `False`，设备数为 0；`nvidia-smi` 报告驱动未加载，尚不能执行 smoke test。
- 当前工作区所在文件系统约有 47 GiB 可用空间；训练启动前仍需复查。
- Hugging Face 项目缓存目前只有版本标记，`t5-small` 与代码中仍会实例化的 `nandakishormpai/t5-small-machine-articles-tag-generation` 均需后续下载至项目 `.cache/huggingface`。

## 单卡有效 batch 说明

官方分布式 sampler 把数据按两张卡切分，每卡 batch 32、梯度累积 2，因此一次 optimizer step 汇总约 `32 × 2 × 2 = 128` 个样本。单卡正式配置采用 batch 16、梯度累积 8，optimizer step 数与有效 batch 128 均保持一致。学习率、epoch、scheduler 配置和其他核心超参数不变。

## 官方 checkpoint 选择问题

`SingleRunnerGRAM` 定义了 `best_score` 和 `best_epoch`，但未读取或比较 validation NDCG@10；`train_generator()` 中的 `updated` 初始化为 `False` 后从未更新。因此：

1. 每 5 epoch 仍会保存普通 checkpoint，并运行 validation；
2. `model_rec_best.pt` 不会由现有单卡逻辑生成；
3. 训练结束的自动测试使用当前 `cur_model_path`，实际为 epoch 30，而不是 validation NDCG@10 最佳 checkpoint。

当前不修改模型或评测逻辑。正式训练后将从 epoch 5/10/15/20/25/30 的 validation 日志选择 NDCG@10 最高者，再以 `--train 0 --rec_model_path ...` 调用原测试入口。这样保留官方评测协议，并满足实验计划的 checkpoint 选择要求。

## 文件修改记录

| 文件 | 修改前 | 修改后 | 原因 | 可能影响数值 |
|---|---|---|---|---|
| `GRAM/command/train_gram_beauty_single.sh` | 不存在 | 官方配置的单卡版本：GPU 1 张、非分布式、batch 16、累积 8 | 保持有效 batch 128 进行正式单卡复现 | 单卡浮点执行顺序可能带来正常微小差异；核心协议不变 |
| `GRAM/command/smoke_test_gram_beauty_single.sh` | 不存在 | 1 epoch、100 条训练/评测样本、batch 4 | 验证数据、前后向、保存与评测链路 | 不用于正式数值 |
| `artifacts/phase1_beauty/data_audit.md` | 不存在 | 新增只读数据审计结果 | 归档计划要求的审计证据 | 否 |
| 本记录 | 不存在 | 新增静态检查和修改说明 | 满足所有修改均记录的要求 | 否 |

2026-07-18 启动 smoke test 前，两个单卡脚本增加了 `PHYSICAL_GPU` 环境变量（默认仍为 0），用于将用户指定的物理 GPU 映射为进程内逻辑 GPU 0；训练参数和模型代码均未改变，不影响实验协议。另新增 `experiment/run_phase1_beauty_smoke.sh`，负责后台状态、GPU 遥测，以及实验退出后立即恢复用户的 CodeLlama GPU 占用；它不参与模型计算，不影响数值。

2026-07-18 首次 smoke test 暴露两个执行问题并完成修正：单卡入口在训练后将 `debug_test_100` 强制重置为 0，导致 smoke 意外执行 22,363 用户的完整测试；现删除该重置语句，使显式传入的调试限制同时作用于最终测试，正式脚本未传该参数，仍执行完整 full-ranking。后台守护脚本改为直接启动并记录实际 GPU Python `workload_pid`，同时单独记录 `runner_pid`，避免将守护 PID 误认为计算 PID。这些修改只影响 smoke 与可观测性，不改变正式实验协议或数值。

2026-07-18 正式训练启动前，正式单卡脚本增加 `PYTHON_BIN` 支持并以 `exec` 启动 Python，用于让守护状态中的 `workload_pid` 与实际 GPU 计算 PID 一致；训练参数未改变。新增 `experiment/run_phase1_beauty_train.sh`，记录正式训练状态、10 秒间隔 GPU 遥测和完整日志，并在任务退出后恢复 CodeLlama GPU 3 占用。该守护逻辑不参与模型计算，不影响数值。

官方源文件和数据均未修改。

## 下一步

待 NVIDIA 驱动/GPU 可见后，先由用户指定 GPU，再将该物理 GPU 映射为脚本内逻辑 GPU 0，以后台方式启动 smoke test。完整 stdout/stderr 写入 `artifacts/phase1_beauty/logs/smoke_test.log`，状态写入 `experiment/`。
