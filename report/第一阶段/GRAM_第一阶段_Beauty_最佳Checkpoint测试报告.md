# GRAM 第一阶段 Beauty 最佳 Checkpoint 测试报告

日期：2026-07-20

## 1. 执行结论

- 阶段 D 状态：成功。
- 完整运行区间：2026-07-20 15:02:24 至 16:48:09，约 1 小时 45 分 45 秒。
- 物理设备：GPU 3，NVIDIA RTX A6000（49,140 MiB）；进程内为逻辑 GPU 0。
- 正式加载了由 validation NDCG@10 选出的 epoch 25 checkpoint。
- 在 22,363 个测试用户和 12,101 个全量商品上完成 constrained beam search full-ranking 评测，beam size 为 50。
- 退出码为 0，无 traceback、OOM、NaN 或评测错误。
- 四项核心指标相对论文值的误差均小于 5%，对齐等级为 **A：基本对齐**。
- 评测退出后守护脚本立即执行 `tools/run_codellama.sh start 3`；16:52 复查时 `codellama` tmux 会话为 running，GPU 3 上已重新保留约 30 GiB CUDA cache。

## 2. Checkpoint 选择依据

正式训练在 epoch 5/10/15/20/25/30 分别运行 validation。最高 validation NDCG@10 为 epoch 25 的 `0.06497370269327131`，略高于 epoch 30 的 `0.06489174430982224`。

所用 checkpoint：

```text
/home/jiangtangyunzhi/projects/recomm/GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt
```

SHA-256：

```text
0df16263344afec8a40e29a7a33e43e7bbe254e0ab97799c9103ad757e60fa89
```

选择过程只使用 validation NDCG@10，没有查看或挑选多个 test 结果。

## 3. 评测协议与命令

实际后台启动命令：

```bash
cd /home/jiangtangyunzhi/projects/UnitTest
tools/run_codellama.sh stop
tmux new-session -d -s gram_phase1_best_test \
  'bash /home/jiangtangyunzhi/projects/recomm/experiment/run_phase1_beauty_best_test.sh 3'
```

守护脚本在 `GRAM/command/` 下最终调用：

```bash
PHYSICAL_GPU=3 \
HF_HOME=/home/jiangtangyunzhi/projects/recomm/.cache/huggingface \
TRANSFORMERS_CACHE=/home/jiangtangyunzhi/projects/recomm/.cache/huggingface \
PYTHON_BIN=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
BEST_CHECKPOINT=/home/jiangtangyunzhi/projects/recomm/GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt \
bash test_gram_beauty_best_single.sh
```

关键参数为 `--train 0`、`--rec_model_path <epoch-25>`、`--beam_size 50`、`--max_his 20`、`--item_prompt all_text`、`--item_prompt_max_len 128`、`--id_linking 1`、`--top_k_similar_item 10`以及官方层次 ID。评测候选集是全部 12,101 个商品 ID 构成的 Trie，不是负采样子集。

## 4. 核心指标与论文对比

`Hit@K` 在本 leave-one-out 单目标设定下等价于 `Recall@K`。绝对误差按“本地 - 论文”计算，相对误差取绝对值。

| 指标 | 论文 | 本地 seed 2023 | 绝对误差 | 相对误差 |
|---|---:|---:|---:|---:|
| Recall@5 | 0.0641 | 0.0632741582 | -0.0008258418 | 1.2884% |
| NDCG@5 | 0.0451 | 0.0442234679 | -0.0008765321 | 1.9435% |
| Recall@10 | 0.0890 | 0.0888968385 | -0.0001031615 | 0.1159% |
| NDCG@10 | 0.0531 | 0.0524608103 | -0.0006391897 | 1.2037% |

论文数值是三次运行的平均，本地数值是计划规定的单一 seed 2023。在此限制下，四项均在论文值±5% 内，满足 A 级标准。

## 5. 完整测试输出

| 指标 | 数值 |
|---|---:|
| Recall/Hit@1 | 0.0246836292 |
| Recall/Hit@3 | 0.0470867057 |
| Recall/Hit@5 | 0.0632741582 |
| Recall/Hit@10 | 0.0888968385 |
| Recall/Hit@20 | 0.1215847605 |
| Recall/Hit@50 | 0.1749765237 |
| NDCG@1 | 0.0246836292 |
| NDCG@3 | 0.0375713352 |
| NDCG@5 | 0.0442234679 |
| NDCG@10 | 0.0524608103 |
| NDCG@20 | 0.0606911233 |
| NDCG@50 | 0.0712813667 |

模型纯推理时间为 5,908.86 秒，平均每个用户 0.2642 秒。

## 6. GPU 与资源

- 排除启动前 CodeLlama 保留样本后，阶段 D 遥测 630 个有效样本。
- 观测峰值显存 32,762 MiB（约 31.99 GiB），出现于 2026-07-20 15:59:14+08:00。
- 平均 GPU 利用率 55.43%，最高 99%；均值包含加载与 CPU 数据准备等待时间。
- 峰值未超过 RTX A6000 的 49,140 MiB 物理显存，未发生 OOM。

## 7. 修改与数值影响

| 文件 | 修改原因 | 是否可能影响数值 |
|---|---|---|
| `GRAM/command/test_gram_beauty_best_single.sh` | 通过官方 `train=0` 入口显式加载 epoch 25，固定训练时的数据/模型配置和 beam 50 | 否；只选择预先规定的最佳 checkpoint，不改评测逻辑 |
| `experiment/run_phase1_beauty_best_test.sh` | 后台运行、状态、日志、GPU 遥测和 CodeLlama 资源恢复 | 否；不参与模型计算 |

没有修改 checkpoint、Beauty 数据、候选商品、beam search、Trie、指标公式或预测结果。

## 8. 产物

- 正式测试日志：`artifacts/phase1_beauty/logs/test_best_checkpoint.log`
- 预测明细：`GRAM/preds/20260720_150243_Beauty_sequential_pred_test.tsv`
- GPU 遥测：`experiment/phase1_beauty_best_test_gpu.csv`
- 指标 JSON：`artifacts/phase1_beauty/metrics_seed2023.json`
- 最终报告：`artifacts/phase1_beauty/REPRODUCTION_REPORT.md`
