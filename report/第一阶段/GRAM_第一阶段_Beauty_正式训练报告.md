# GRAM 第一阶段 Beauty 正式单种子训练报告

日期：2026-07-20

## 1. 执行结论

- 阶段 C 状态：成功。
- 数据集与种子：Beauty，seed 2023。
- 物理 GPU：GPU 3，NVIDIA RTX A6000（48 GiB）；进程内映射为逻辑 GPU 0。
- 运行区间：2026-07-18 21:53:55 至 2026-07-20 14:21:51，约 40 小时 28 分（40.47 小时），包含 30 epoch 训练、6 次完整 validation 和 epoch 30 的自动 test。
- 30 个 epoch 全部完成，epoch 5/10/15/20/25/30 的模型、optimizer 和 scheduler checkpoint 均已保存。
- 无 NaN、OOM、traceback 或数据错误；退出码为 0；退出后 CodeLlama GPU 资源占用已恢复。
- validation NDCG@10 最高的 checkpoint 为 epoch 25，数值为 `0.06497370269327131`。
- 训练结束时官方代码自动测试的是 epoch 30，不是 epoch 25。因此本报告不将该次 test 作为最终复现结果；阶段 D 将显式加载 epoch 25 做正式 full-ranking test。

## 2. 实际执行命令与配置

后台守护脚本的启动命令为：

```bash
tmux new-session -d -s gram_phase1_train \
  'bash /home/jiangtangyunzhi/projects/recomm/experiment/run_phase1_beauty_train.sh 3'
```

守护脚本最终调用：

```bash
cd /home/jiangtangyunzhi/projects/recomm/GRAM/command
PHYSICAL_GPU=3 \
HF_HOME=/home/jiangtangyunzhi/projects/recomm/.cache/huggingface \
TRANSFORMERS_CACHE=/home/jiangtangyunzhi/projects/recomm/.cache/huggingface \
PYTHON_BIN=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
bash train_gram_beauty_single.sh
```

| 配置 | 实际值 |
|---|---:|
| backbone | T5-small |
| seed | 2023 |
| learning rate | 1e-3 |
| epochs | 30 |
| per-device batch size | 16 |
| gradient accumulation | 8 |
| 有效 batch size | 128 |
| checkpoint / validation 间隔 | 5 epochs |
| item prompt max length | 128 |
| max user history | 20 |
| ID length / clusters | 7 / 128 |
| similar items | 10 |
| beam size | 50 |
| evaluation | 12,101 个商品 full ranking，Trie 约束 beam search |

## 3. 训练损失

loss 从 epoch 1 的 `5.7191599059` 稳定降至 epoch 30 的 `0.9028391892`，未出现数值发散。

| Epoch | Loss | Epoch | Loss | Epoch | Loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.719160 | 11 | 1.321432 | 21 | 1.003728 |
| 2 | 3.871259 | 12 | 1.265021 | 22 | 0.986897 |
| 3 | 2.842453 | 13 | 1.219509 | 23 | 0.971863 |
| 4 | 2.356674 | 14 | 1.180214 | 24 | 0.958971 |
| 5 | 2.056987 | 15 | 1.142483 | 25 | 0.945762 |
| 6 | 1.846032 | 16 | 1.112705 | 26 | 0.934492 |
| 7 | 1.688513 | 17 | 1.085863 | 27 | 0.925383 |
| 8 | 1.565683 | 18 | 1.063362 | 28 | 0.916416 |
| 9 | 1.468215 | 19 | 1.040855 | 29 | 0.909241 |
| 10 | 1.387160 | 20 | 1.019784 | 30 | 0.902839 |

## 4. Validation 结果与 checkpoint 选择

| Epoch | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---:|---:|---:|---:|---:|
| 5 | 0.064571 | 0.045393 | 0.090954 | 0.053902 |
| 10 | 0.070250 | 0.050324 | 0.098645 | 0.059448 |
| 15 | 0.074543 | 0.052727 | 0.104548 | 0.062430 |
| 20 | 0.076376 | 0.054002 | 0.107857 | 0.064177 |
| **25** | **0.077628** | **0.054917** | **0.108751** | **0.064974** |
| 30 | 0.078523 | 0.055269 | 0.108393 | 0.064892 |

按计划预先规定的 `validation NDCG@10` 选择，epoch 25 以 `0.0649737027` 高于 epoch 30 的 `0.0648917443`，因此最佳 checkpoint 是：

```text
/home/jiangtangyunzhi/projects/recomm/GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt
```

选择仅使用 validation，没有根据 test 结果挑选 checkpoint。

## 5. Epoch 30 自动 test（非最终结果）

官方训练入口在训练结束后自动加载当前 checkpoint，本次为 epoch 30。该次测试确认了 22,363 个测试用户的 full-ranking 评测链路可完整运行：

| 指标 | Epoch 30 test |
|---|---:|
| Recall@5 | 0.0627375576 |
| NDCG@5 | 0.0441467862 |
| Recall@10 | 0.0875553369 |
| NDCG@10 | 0.0521344054 |

推理耗时 5,628.44 秒，平均每用户 0.2517 秒。这些数值不写入最终 `metrics_seed2023.json`，也不用于最终 A/B/C 判定；它们将由阶段 D 的 epoch 25 正式 test 替代。

## 6. GPU 与资源

- GPU 遥测间隔：10 秒，有效样本 14,491 条。
- 排除实验启动前的 CodeLlama 占用样本后，观测到的峰值总显存为 36,454 MiB（约 35.60 GiB），时间为 2026-07-19 19:25:58+08:00。
- 峰值低于 RTX A6000 的 48 GiB 可用上限，未发生 OOM。
- 全过程遥测的平均 GPU 利用率约 57.97%，最高 99%。该均值包含数据加载、checkpoint 写入和 validation/test，不等同于纯训练步利用率。
- 结论限制：本次证明了 48 GiB 单卡可完整运行；35.60 GiB 遥测峰值不能证明 24–30 GiB 卡可直接运行 batch 16 配置。

## 7. 数据、代码与数值影响

- Beauty 官方预处理数据未修改；数据规模与论文完全一致。
- 正式训练使用新增的单卡脚本：batch 16、gradient accumulation 8，保持官方两卡配置的有效 batch 128。
- `PHYSICAL_GPU`、`PYTHON_BIN` 和守护脚本只处理设备映射、环境选择、日志、遥测与资源恢复，不参与模型计算。
- `main_generative_gram.py` 只删除了对 `debug_test_100` 的强制重置；正式脚本没有启用该调试参数，因此正式数值不受影响。
- 单卡与两卡的浮点计算顺序可能造成正常微小差异，但学习率、optimizer step 的有效 batch、epoch、数据与评测协议均未改变。
- 阶段 D 准备新增 `GRAM/command/test_gram_beauty_best_single.sh`：固定加载由 validation NDCG@10 选出的 epoch 25，使用原生 `--train 0 --rec_model_path` 评测入口；不修改模型、数据或评测公式。
- 阶段 D 准备新增 `experiment/run_phase1_beauty_best_test.sh`：负责后台状态、日志、GPU 遥测和退出后资源恢复；不参与模型计算。

## 8. 产物

- 完整训练日志：`artifacts/phase1_beauty/logs/train_seed2023.log`
- GPU 遥测：`experiment/phase1_beauty_train_gpu.csv`
- 阶段状态：`experiment/phase1_beauty_status.json`
- checkpoint 目录：`GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/`
- epoch 30 自动 test 预测：`GRAM/preds/20260720_124018_Beauty_sequential_pred_test.tsv`

## 9. 下一步

使用用户指定的物理 GPU，在后台显式加载 epoch 25 checkpoint，保持 beam size 50、Trie 约束和 12,101 商品 full ranking，完成阶段 D 正式测试。测试完成后再生成 `metrics_seed2023.json`、论文误差表、A/B/C 对齐等级和最终 `REPRODUCTION_REPORT.md`。
