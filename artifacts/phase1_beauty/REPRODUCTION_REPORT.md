# GRAM Beauty 单卡复现最终报告

完成日期：2026-07-20  
官方 GRAM commit：`7ac4d9272a57beed9df35c27ea34221f6e4a8fb1`  
数据集：Beauty  
随机种子：2023

## 1. 最终结论

- 执行状态：**成功**。静态审计、smoke test、30 epoch 单种子训练、validation checkpoint 选择和最佳 checkpoint full-ranking test 全部完成。
- 对齐等级：**A：基本对齐**。四项核心指标相对论文 Table 2 值均在 ±5% 内。
- 最佳 checkpoint：epoch 25，选择依据是最高 validation NDCG@10 `0.06497370269327131`。
- 数据、划分、模型结构、学习率、有效 batch、生成约束和评测公式均未为对齐数值而修改。
- 运行硬件是 48 GiB RTX A6000，训练遥测峰值约 35.60 GiB。因此本次证明了 48 GiB 单卡流程，**没有实证 24–30 GiB GPU 能以 batch 16 直接运行**。如需在该显存区间部署，应按计划改为 batch 8 / accumulation 16 或 batch 4 / accumulation 32，而不改其他参数。

## 2. 核心指标

Beauty 为 leave-one-out 单目标评测，因此代码输出的 `Hit@K` 等价于计划中的 `Recall@K`。绝对误差为“本地 - 论文”。

| 指标 | 论文 | 本地 | 绝对误差 | 相对误差 |
|---|---:|---:|---:|---:|
| Recall@5 | 0.0641 | 0.0632741582 | -0.0008258418 | 1.2884% |
| NDCG@5 | 0.0451 | 0.0442234679 | -0.0008765321 | 1.9435% |
| Recall@10 | 0.0890 | 0.0888968385 | -0.0001031615 | 0.1159% |
| NDCG@10 | 0.0531 | 0.0524608103 | -0.0006391897 | 1.2037% |

论文数值是三个随机种子的平均；本阶段按计划只运行 seed 2023。本地四项相对误差最大为 1.9435%，满足 A 级“四项指标均在论文值相对 ±5% 内”的定义。

## 3. 硬件与软件环境

| 项目 | 实际值 |
|---|---|
| OS | Ubuntu Linux，kernel 5.4.0-216-generic |
| CPU | 2 × Intel Xeon Gold 6330，共 112 逻辑 CPU |
| 内存 | 755 GiB |
| GPU | 物理 GPU 3，NVIDIA RTX A6000，49,140 MiB |
| NVIDIA driver | 560.35.03 |
| Python | 3.9.25 |
| PyTorch | 1.11.0+cu113 |
| torchvision / torchaudio | 0.12.0+cu113 / 0.11.0+cu113 |
| Transformers | 4.26.0 |
| NumPy | 1.23.1 |
| 环境 | Conda `gram-repro` |

T5 权重缓存固定在项目内 `.cache/huggingface/`。完整 pip 与 Conda 快照见 `environment/`。

## 4. 数据审计

官方 Beauty 预处理数据以只读方式审计，未修改。

| 项目 | 论文 | 本地 | 结果 |
|---|---:|---:|---|
| 用户数 | 22,363 | 22,363 | 一致 |
| 商品数 | 12,101 | 12,101 | 一致 |
| 交互数 | 198,502 | 198,502 | 一致 |
| 密度 | 0.0734% | 0.0734% | 一致 |

每个用户的最后一个交互用于 test，倒数第二个用于 validation，其余用于 train。商品文本、层次 ID 和 SASRec 相似商品映射无缺失、空值或重复 ID。评测使用全部 12,101 个商品建立 Trie，没有负采样。

## 5. 核心配置

| 配置 | 值 |
|---|---:|
| backbone | T5-small |
| seed | 2023 |
| learning rate | 1e-3 |
| epochs | 30 |
| per-device batch | 16 |
| gradient accumulation | 8 |
| 有效 batch | 128 |
| item prompt max length | 128 |
| max user history | 20 |
| ID length | 7 |
| clusters / similar items | 128 / 10 |
| checkpoint 间隔 | 5 epochs |
| checkpoint 选择 | validation NDCG@10 最高 |
| beam size | 50 |
| evaluation | full ranking + constrained Trie |

## 6. Smoke test

2026-07-18 在 GPU 3 上以 100 条训练样本、100 条 validation/test 样本和 1 epoch 完成 smoke test。forward、backward、optimizer step、checkpoint 保存/重载、constrained validation/test、指标和预测输出链路均通过，退出码 0。Smoke 观测峰值显存约 24.35 GiB，其数值不用于论文对比。

Smoke 期间发现并保留了两类失败记录：Hugging Face SOCKS 代理缺少 `PySocks`，以及调试参数被入口覆盖而误进全量 test。前者通过安装 `PySocks==1.7.1` 解决；后者通过保留命令行传入的 `debug_test_100` 解决。正式脚本默认 `debug_test_100=0`，因此正式协议和数值不受影响。

## 7. 正式训练和 checkpoint 选择

训练从 2026-07-18 21:53:55 运行至 2026-07-20 14:21:51，共约 40.47 小时，包含 30 epoch、6 次 validation 和训练结束后的 epoch 30 自动 test。训练 loss 从 `5.7191599059` 平稳下降至 `0.9028391892`，无 NaN、OOM 或 traceback。

| Epoch | Validation Recall@5 | Validation NDCG@5 | Validation Recall@10 | Validation NDCG@10 |
|---:|---:|---:|---:|---:|
| 5 | 0.064571 | 0.045393 | 0.090954 | 0.053902 |
| 10 | 0.070250 | 0.050324 | 0.098645 | 0.059448 |
| 15 | 0.074543 | 0.052727 | 0.104548 | 0.062430 |
| 20 | 0.076376 | 0.054002 | 0.107857 | 0.064177 |
| **25** | **0.077628** | **0.054917** | **0.108751** | **0.064974** |
| 30 | 0.078523 | 0.055269 | 0.108393 | 0.064892 |

官方单卡 runner 没有自动维护 validation 最佳模型；训练结束后自动 test 使用 epoch 30。按预先规定的 validation NDCG@10 最高原则，本实验选择 epoch 25，然后单独使用官方 `--train 0 --rec_model_path` 入口执行最终 test。

最佳 checkpoint：

```text
GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt
```

SHA-256：`0df16263344afec8a40e29a7a33e43e7bbe254e0ab97799c9103ad757e60fa89`。

## 8. 最佳 checkpoint 正式测试

阶段 D 从 2026-07-20 15:02:24 运行至 16:48:09，退出码 0。评测 22,363 个用户的纯推理耗时 5,908.86 秒，平均每用户 0.2642 秒。阶段 D 遥测峰值显存 32,762 MiB（约 31.99 GiB）。

评测结束后已自动执行 `tools/run_codellama.sh start 3`。2026-07-20 16:52:41 复查时，`codellama` tmux 会话为 running，GPU 3 上 CodeLlama 保留目标为 30,720 MiB。

## 9. 资源汇总

| 项目 | 结果 |
|---|---:|
| 正式训练总时间（含周期 validation 和 epoch 30 自动 test） | 40.47 h |
| 最佳 checkpoint 阶段 D 总时间 | 1.76 h |
| 训练期间观测峰值显存 | 36,454 MiB / 35.60 GiB |
| 阶段 D 观测峰值显存 | 32,762 MiB / 31.99 GiB |
| 训练期间平均 GPU 利用率 | 57.97% |
| 阶段 D 平均 GPU 利用率 | 55.43% |

遥测为 10 秒间隔的整卡数值，均值包含数据准备、checkpoint 写入和评测等待。启动前 CodeLlama 保留样本已从峰值和均值统计中排除。

## 10. 对官方代码和环境的修改

| 文件/环境 | 修改 | 原因 | 可能影响数值 |
|---|---|---|---|
| `command/train_gram_beauty_single.sh` | 单 GPU、`distributed=0`、batch 16 / accumulation 8，支持显式 GPU 和 Python | 保持有效 batch 128 的单卡适配 | 单/多卡浮点顺序可有微小差异；核心协议不变 |
| `command/smoke_test_gram_beauty_single.sh` | 1 epoch、100 条训练与评测 | 链路验证 | 否；不用于正式数值 |
| `src/main_generative_gram.py` | 删除对 `debug_test_100` 的强制重置 | 使 smoke 的显式调试限制在最终 test 中仍生效 | 否；正式脚本该值为 0 |
| `command/test_gram_beauty_best_single.sh` | 显式加载 epoch 25 运行 test | 按 validation NDCG@10 选择 checkpoint | 否；使用原评测入口和公式 |
| `experiment/run_phase1_beauty_*.sh` | 后台运行、日志、状态、GPU 遥测和资源恢复 | 可观测与 GPU 占用协调 | 否；不参与计算 |
| Conda 环境 | 增加 `PySocks==1.7.1` | 支持主机 SOCKS 代理下的 Hugging Face 下载 | 否；仅网络传输 |

官方 Beauty 数据和 checkpoint 内容未修改。官方仓库的完整 diff 归档在 `environment/code_changes.patch`。

## 11. 异常与未解决问题

1. Smoke 初次运行缺少 SOCKS 传输依赖，已修复并锁定依赖。
2. Smoke 曾因 PID 识别不准重复启动两个进程，失败日志已保留；守护脚本已分离 runner/workload PID，成功 smoke、训练和阶段 D 均为单一 workload。
3. 官方单卡 runner 不会自动保存 validation 最佳模型；本实验从固定间隔 checkpoint 中按 validation NDCG@10 选择，未修改 runner。
4. 主文件系统在实验完成时仅剩约 4.9 GiB；所需产物已保存，但后续新的大型试验应先规划存储。
5. 本次只运行了 seed 2023，不能代替论文的三种子均值；这是阶段范围限制，不是执行失败。

## 12. 完成条件核对

- [x] 独立环境可通过 Conda 快照或 `requirements_lock.txt` 重建。
- [x] 官方 Beauty 数据未修改。
- [x] 30 epoch 单卡训练完成。
- [x] 峰值显存低于实际 GPU 可用显存。
- [x] checkpoint 可保存并重新加载。
- [x] 12,101 商品 full-ranking test 完成。
- [x] Recall@5、NDCG@5、Recall@10、NDCG@10 已输出。
- [x] 最佳 checkpoint 按 validation NDCG@10 明确选择。
- [x] 已完成与论文 Table 2 的误差比较和 A 级判定。
- [x] 环境、命令、日志、代码修改、指标和 GPU 遥测已归档。
- [x] 报告包含从空终端启动同配置实验的命令。

## 13. 可复现命令

在已按 `environment/conda_environment.yml` 重建 `gram-repro`、数据和 T5 缓存就绪的前提下，从空终端重新启动相同的 30 epoch GPU 3 正式训练：

```bash
cd /home/jiangtangyunzhi/projects/UnitTest
tools/run_codellama.sh stop
tmux new-session -d -s gram_phase1_train \
  'bash /home/jiangtangyunzhi/projects/recomm/experiment/run_phase1_beauty_train.sh 3'
```

已有本次 checkpoint 时，重跑最佳 checkpoint 正式测试：

```bash
cd /home/jiangtangyunzhi/projects/UnitTest
tools/run_codellama.sh stop
tmux new-session -d -s gram_phase1_best_test \
  'bash /home/jiangtangyunzhi/projects/recomm/experiment/run_phase1_beauty_best_test.sh 3'
```

两个守护脚本均在任务退出后自动执行 `tools/run_codellama.sh start 3` 恢复用户的 GPU 占用。

## 14. 产物索引

- 数据审计：`data_audit.md`
- 最终指标：`metrics_seed2023.json`
- 环境快照和代码 diff：`environment/`
- Smoke 日志：`logs/smoke_test.log`
- 30 epoch 训练日志：`logs/train_seed2023.log`
- 最佳 checkpoint 测试日志：`logs/test_best_checkpoint.log`
- 单卡训练脚本：`../../GRAM/command/train_gram_beauty_single.sh`
- Smoke 脚本：`../../GRAM/command/smoke_test_gram_beauty_single.sh`
- 最佳 checkpoint 测试脚本：`../../GRAM/command/test_gram_beauty_best_single.sh`
