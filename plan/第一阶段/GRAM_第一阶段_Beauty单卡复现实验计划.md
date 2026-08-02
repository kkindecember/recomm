# GRAM 第一阶段实验计划：单卡复现 Beauty

## 0. 给 Coding AI 的任务说明

请在一台可用显存足够的单张 NVIDIA GPU 上，完成 GRAM 官方代码的 Beauty 数据集复现。目标是确认完整流程能够运行，并获得与论文接近的 Recall 和 NDCG。每次选择gpu时请询问我。

本阶段只复现原始 GRAM，不设计或加入任何创新模块，不重新运行 NV-Embed-v2，不重新生成数据集，不运行其他模型的 baseline。

执行原则：

1. 先检查仓库、数据、环境和命令，再运行实验。
2. 保留官方脚本，不直接覆盖 `command/train_gram_beauty.sh`；新增单卡脚本。
3. 除单卡适配和显存适配外，不修改模型结构、数据划分、评测逻辑和核心超参数。
4. 不允许为了接近论文数值而反复试参或修改评测方法。
5. 每次修改必须记录原因、文件、修改前后内容以及是否可能影响结果，记录在recomm/report中。
6. 遇到错误时保留完整日志，不要静默重试或隐瞒失败。
7. 如果必须改变核心依赖、模型代码、数据或评测协议，先停止主实验，在报告中说明阻塞原因。
8. 所有gpu实验均需后台运行，无需实时监控，将实验必要数据、status、log等记录在recomm/experiment中，我会根据status判断实验状态，并主动找你。



官方资源：

- 论文：https://aclanthology.org/2025.acl-long.1596/
- 代码：https://github.com/skleee/GRAM
- 官方 Beauty 脚本：`command/train_gram_beauty.sh`

## 1. 实验目标

完成以下端到端流程：

```text
检查仓库和数据
    -> 建立可复现环境
    -> 单卡 smoke test
    -> Beauty 完整训练
    -> 按验证集 NDCG@10 选择 checkpoint
    -> Beauty 测试集 full-ranking 评测
    -> 导出 Recall/NDCG
    -> 与论文结果比较
```

第一阶段回答四个问题：

1. GRAM 是否能在单张 24GB～30GB GPU 上完整训练和评测？
2. 官方 Beauty 预处理数据是否可以直接使用？
3. 本地运行得到的 Recall@5、NDCG@5、Recall@10、NDCG@10 是否接近论文？
4. 训练入口、checkpoint 选择和评测输出分别在哪里，后续如何稳定重复实验？

## 2. 不在本阶段范围内

- 不运行 NV-Embed-v2 预处理。
- 不重新训练 SASRec 或构造新的协同邻居。
- 不更换 T5-small。
- 不添加注意力、门控、损失函数或其他创新模块。
- 不运行 Toys、Sports、Yelp。
- 不复现 P5、TIGER、IDGenRec 等对比模型。
- 不执行多随机种子正式实验。
- 不以最后一个 checkpoint 代替论文要求的最佳验证集 checkpoint。

## 3. 论文目标值与实验协议

### 3.1 Beauty 数据统计

论文报告的预处理数据规模：

| 数据集 | 用户数 | 商品数 | 交互数 | 密度 |
|---|---:|---:|---:|---:|
| Beauty | 22,363 | 12,101 | 198,502 | 0.0734% |

论文采用 5-core 数据，并对每个用户使用 leave-one-out 划分：

- 最后一次交互：测试集；
- 倒数第二次交互：验证集；
- 此前交互：训练集；
- 评测方式：对全部商品进行 full ranking，不是负采样评测。

Coding AI 需要确认仓库数据与上述统计和划分逻辑一致。如果文件结构导致无法直接统计，也要说明实际检查方法和结果。

### 3.2 论文主要结果

论文 Table 2 中 GRAM 在 Beauty 上的三次运行平均结果为：

| 指标 | 论文目标值 |
|---|---:|
| Recall@5 | 0.0641 |
| NDCG@5 | 0.0451 |
| Recall@10 | 0.0890 |
| NDCG@10 | 0.0531 |

注意：论文结果是三个随机种子的平均值；本阶段只运行官方脚本中的 `seed=2023`，因此不要求与表格逐位相同。

### 3.3 核心配置

需要保持以下论文/官方脚本配置：

| 配置 | 值 |
|---|---|
| 数据集 | Beauty |
| 主干模型 | T5-small |
| seed | 2023 |
| learning rate | 1e-3 |
| epochs | 30 |
| item prompt max length | 128 |
| max user history | 20 |
| ID length | 7 |
| clusters / cluster size | 128 / 128 |
| similar items | 10 |
| checkpoint 间隔 | 每 5 epoch |
| checkpoint 选择 | 验证集 NDCG@10 最高 |
| inference | constrained beam search，beam size 50 |
| evaluation | full ranking |

如果某个值由代码默认提供而没有出现在 shell 脚本中，需要在代码中定位默认值并记录证据。

## 4. 环境配置

### 4.1 首选环境

优先按官方版本建立独立环境：

```text
Python 3.9
PyTorch 1.11.0 + CUDA 11.3
Transformers 4.26.0
其余依赖使用 requirements.txt
```

建议环境名：`gram-repro`

参考命令：

```bash
conda create -n gram-repro python=3.9 -y
conda activate gram-repro
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

执行时必须检查 `requirements.txt` 是否会覆盖 PyTorch/Transformers 版本。安装完成后验证：

```bash
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果旧版 PyTorch wheel 因操作系统、Python、驱动或镜像问题无法安装：

1. 保留完整错误日志；
2. 尽量使用容器或兼容的 Conda 环境解决；
3. 只有确实无法使用官方环境时，才建立一个最小现代兼容环境；
4. 不得在没有记录的情况下升级依赖；
5. 现代化修改必须单独提交，并在报告中标为“环境兼容修改”，说明是否可能影响数值。

### 4.2 必须记录的环境信息

保存到 `artifacts/phase1_beauty/environment/`：

- `git_commit.txt`：GRAM 仓库 commit hash 和当前 git status；
- `system_info.txt`：OS、CPU、内存；
- `nvidia_smi.txt`：GPU、驱动、总显存和实验开始时可用显存；
- `pip_freeze.txt`：完整 Python 依赖；
- `conda_environment.yml`：Conda 环境导出；
- `disk_space.txt`：实验前可用磁盘空间；
- `code_changes.patch`：相对官方仓库的全部改动。

## 5. 仓库与数据审计

在训练前完成并记录以下检查：

### 5.1 仓库检查

- 当前 commit hash；
- 工作区是否干净；
- `src/main_generative_gram.py` 是否存在；
- `command/train_gram_beauty.sh` 是否存在；
- `rec_datasets/Beauty/` 是否存在；
- 官方脚本引用的层次ID文件是否存在；
- 官方脚本引用的 SASRec 相似商品文件是否存在；
- T5-small 是否需要联网下载，下载后缓存位置在哪里；
- 代码的 checkpoint、日志、预测和指标默认输出位置。

### 5.2 数据检查

至少检查：

- 用户序列文件非空；
- 商品文本文件非空；
- 层次语义 ID 文件非空；
- SASRec 相似商品文件非空；
- 用户数、商品数、交互数是否接近论文统计；
- 每个测试用户是否只有一个目标商品；
- 商品 ID 能否在层次 ID 和商品文本映射中找到；
- 是否存在明显重复 ID、空文本、越界 ID 或缺失映射。

将结果写入 `artifacts/phase1_beauty/data_audit.md`。不要修改数据来“修复”统计差异；如果发现问题，先报告。

## 6. 单卡适配

### 6.1 新建单卡脚本

复制官方脚本为：

```text
command/train_gram_beauty_single.sh
```

只允许进行以下单卡改动：

```text
CUDA_VISIBLE_DEVICES=0
--distributed 0
--gpu 0
```

官方两卡配置的近似有效 batch size 是：

```text
32 × 2 GPUs × 2 gradient accumulation = 128
```

建议单卡初始配置：

```text
--rec_batch_size 16
--gradient_accumulation_steps 8
```

仍保持有效 batch size 128。如果发生 CUDA OOM，按以下顺序调整，且每次只调整 batch 和梯度累积：

| 单卡 batch size | gradient accumulation | 有效 batch size |
|---:|---:|---:|
| 16 | 8 | 128 |
| 8 | 16 | 128 |
| 4 | 32 | 128 |

不得因为单卡适配改变 learning rate、epoch、最大历史长度、item prompt长度、ID配置、相似商品数量或评测方式。

如果代码中 `rec_batch_size` 的实际语义与上述计算不同，需要阅读训练循环和分布式实现，给出正确计算，再设置等价配置。

### 6.2 路径注意事项

官方脚本使用 `../src/main_generative_gram.py`，因此通常需要从 `command/` 目录执行：

```bash
cd command
bash train_gram_beauty_single.sh
```

如果 Coding AI 改为从仓库根目录执行，必须相应修正路径并在报告中写明，不得留下依赖当前目录的隐式错误。

## 7. 执行阶段

### 7.1 阶段 A：静态检查

在消耗 GPU 前：

- 运行 Python import 检查；
- 运行 shell 语法检查；
- 检查所有输入文件路径；
- 检查输出目录是否可写；
- 阅读参数解析，确认 `--distributed 0` 和 `--gpu 0` 的单卡分支；
- 确认训练、验证、测试由哪些参数触发；
- 确认指标名称和输出格式；
- 确认最佳 checkpoint 的选择逻辑确实基于 validation NDCG@10。

### 7.2 阶段 B：smoke test

新增独立的 smoke-test 脚本，不覆盖正式脚本。smoke test 可以临时使用：

```text
1 epoch
较小 batch size
test/save interval = 1
```

smoke test 只验证：

- 数据能够读取；
- T5-small 能够加载；
- forward、backward 和 optimizer step 正常；
- checkpoint 能保存；
- validation/test 能运行；
- 能输出至少一组 Recall/NDCG；
- GPU 峰值显存不超过可用上限。

smoke-test 数值不能作为复现结果。日志保存到：

```text
artifacts/phase1_beauty/logs/smoke_test.log
```

### 7.3 阶段 C：正式单种子训练

smoke test 成功后，使用完整配置运行 30 epochs：

```bash
cd command
bash train_gram_beauty_single.sh
```

完整标准输出和错误输出保存到：

```text
artifacts/phase1_beauty/logs/train_seed2023.log
```

建议同时记录：

- 开始和结束时间；
- 每个 epoch 的训练 loss；
- 每个验证点的 Recall/NDCG；
- 当前最佳 validation NDCG@10；
- GPU 峰值显存；
- 平均 GPU 利用率；
- checkpoint 路径；
- 是否发生 NaN、OOM、卡死或数据错误。

训练软超时设为 72 小时：超过后不要直接杀死进程，先记录当前 epoch、速度、GPU状态和预计剩余时间。只有确定无进展或用户授权后再终止。硬超时可设为 120 小时。

### 7.4 阶段 D：最佳 checkpoint 测试

论文使用 validation NDCG@10 最高的 checkpoint 做测试。Coding AI 必须确认官方代码是否自动完成这一选择。

如果官方脚本只是每5个epoch测试一次，或默认使用最后checkpoint，需要：

1. 不修改评测公式；
2. 从已保存checkpoint中找到 validation NDCG@10 最高者；
3. 使用该checkpoint运行正式测试；
4. 记录选择依据和checkpoint epoch；
5. 确认使用 full ranking 和 constrained beam search（beam=50）。

最终测试日志保存到：

```text
artifacts/phase1_beauty/logs/test_best_checkpoint.log
```

## 8. 指标整理与“对齐”判定

将结果保存为 `artifacts/phase1_beauty/metrics_seed2023.json`：

```json
{
  "dataset": "Beauty",
  "seed": 2023,
  "checkpoint_epoch": null,
  "selection_metric": "validation_NDCG@10",
  "evaluation": "full-ranking",
  "beam_size": 50,
  "recall@5": null,
  "ndcg@5": null,
  "recall@10": null,
  "ndcg@10": null,
  "peak_gpu_memory_gb": null,
  "training_time_hours": null
}
```

对每个指标计算：

```text
绝对误差 = 本地结果 - 论文结果
相对误差 = |本地结果 - 论文结果| / 论文结果 × 100%
```

判定标准：

| 等级 | 标准 | 结论 |
|---|---|---|
| A：基本对齐 | 四项指标均在论文值相对 ±5% 内 | 第一阶段通过 |
| B：可接受但需复核 | 至少三项在 ±10% 内，流程和协议确认一致 | 流程通过，数值待多种子复核 |
| C：未对齐 | 任一核心指标偏差超过 10%，或评测协议不一致 | 检查环境、数据、checkpoint和评测 |

由于本阶段只运行 seed=2023，而论文报告三个种子的平均值，不能仅凭轻微差距判断复现失败。严禁通过改测试集、降低候选商品数、使用负采样或挑选测试结果来获得更高指标。

## 9. 异常排查顺序

指标明显低于论文时，按以下顺序检查，不要立即调参：

1. 是否加载了正确的 Beauty 数据和层次 ID 文件；
2. 是否为 full-ranking，而不是 sampled ranking；
3. 是否使用最佳 validation NDCG@10 checkpoint；
4. beam size 是否为50，是否启用 constrained decoding / Trie；
5. `ID_LEN=7`、`NUM_CLUSTER=128`、`NUM_CF=10` 是否正确；
6. `item_prompt=all_text`、`id_linking=1`、`max_his=20` 是否正确；
7. 单卡改写是否改变了有效 batch size、scheduler或optimizer step数量；
8. T5 tokenizer/model版本是否与官方兼容；
9. 是否存在未记录的代码或依赖修改；
10. seed 和确定性设置是否正确。

常见异常处理：

- CUDA OOM：只降低 per-device batch，并提高梯度累积保持有效 batch；
- DataLoader 卡死：先将 worker 数降为0验证，不修改样本；
- 下载失败：记录所需模型及版本，手动缓存后重试；
- FP16/数值异常：先确认官方是否默认使用混合精度，不要擅自启用；
- 指标为0：优先检查生成ID、tokenizer、Trie、商品ID映射和checkpoint加载；
- 指标异常高：检查是否有训练/验证/测试泄漏以及是否错误缩小了候选集合。

## 10. 最终交付物

Coding AI 完成后应提供：

```text
artifacts/phase1_beauty/
├── REPRODUCTION_REPORT.md
├── data_audit.md
├── metrics_seed2023.json
├── environment/
│   ├── git_commit.txt
│   ├── system_info.txt
│   ├── nvidia_smi.txt
│   ├── pip_freeze.txt
│   ├── conda_environment.yml
│   ├── disk_space.txt
│   └── code_changes.patch
└── logs/
    ├── environment_setup.log
    ├── smoke_test.log
    ├── train_seed2023.log
    └── test_best_checkpoint.log
```

仓库中新增：

```text
command/train_gram_beauty_single.sh
command/smoke_test_gram_beauty_single.sh
```

`REPRODUCTION_REPORT.md` 至少包含：

1. 最终结论：成功、部分成功或失败；
2. 硬件和软件环境；
3. 实际执行命令；
4. 数据审计结果；
5. 对官方代码的全部修改；
6. smoke test结果；
7. 训练时间和峰值显存；
8. 最佳checkpoint及选择依据；
9. 四项测试指标；
10. 与论文结果的绝对误差和相对误差；
11. A/B/C对齐等级；
12. 异常、未解决问题和下一步建议；
13. 一条可以重新运行相同实验的完整命令。

## 11. 第一阶段完成条件

必须同时满足：

- [ ] 独立环境可重建；
- [ ] 官方 Beauty 数据未经修改；
- [ ] 单卡训练能完成；
- [ ] 峰值显存不超过可用显存；
- [ ] 能保存并重新加载 checkpoint；
- [ ] 能完成 full-ranking 测试；
- [ ] 输出 Recall@5、NDCG@5、Recall@10、NDCG@10；
- [ ] 明确最佳 checkpoint 的选择逻辑；
- [ ] 完成与论文 Table 2 的误差比较；
- [ ] 所有环境、命令、日志和代码修改均已归档；
- [ ] 从空终端能够按照报告中的命令再次启动实验。

如果训练流程完整但指标只达到B级，本阶段仍可视为“工程流程已打通”，但不能声称已经成功复现论文结果；需要在第二次运行或多随机种子实验后再下结论。

## 12. 给 Coding AI 的最终汇报格式

完成后请直接按以下格式回复：

```markdown
## 执行结论
- 状态：成功 / 部分成功 / 失败
- 对齐等级：A / B / C
- 最佳checkpoint：...

## 核心指标
| 指标 | 论文 | 本地 | 绝对误差 | 相对误差 |
|---|---:|---:|---:|---:|
| Recall@5 | 0.0641 | ... | ... | ... |
| NDCG@5 | 0.0451 | ... | ... | ... |
| Recall@10 | 0.0890 | ... | ... | ... |
| NDCG@10 | 0.0531 | ... | ... | ... |

## 资源
- GPU：...
- 峰值显存：...
- 训练时间：...

## 修改
- 修改文件：...
- 修改原因：...
- 是否可能影响数值：...

## 未解决问题
- ...

## 产物路径
- 完整报告：...
- 日志：...
- 指标JSON：...
- 单卡脚本：...

## 可复现命令
`...`
```

