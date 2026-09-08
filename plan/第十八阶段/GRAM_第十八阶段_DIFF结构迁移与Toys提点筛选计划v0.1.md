# 第十八阶段：DIFF 结构迁移与 Toys 提点筛选计划 v0.1

## Material Passport

- 日期：2026-09-08
- 来源：academic-research-suite / experiment-agent；inline
- 用户要求：继续 DIFF 候选并写 plan；接受大改；借鉴第十七阶段失败；优先使用已有 GRAM / PCRF 数据；方法确定后统一重跑。
- 当前任务：实现、检查并运行第一轮候选筛选；PCPS 暂停。
- 本计划是机制迁移实验，不是完整原生 DIFF 基准复现，也不提前声明有效。

**2026-09-08 并行修订：** 用户要求 Beauty 同时跑。执行顺序更新为 Toys / Beauty 独立并行筛选，详见同目录《GRAM_第十八阶段_DIFF_Beauty并行筛选补充计划v0.1.md》。下文“等 Toys 后再检查 Beauty”的顺序由补充计划覆盖；Toys 已启动配置与源码快照继续固定。

## 1. 第一轮要回答的问题

在已训练的 Toys GRAM 上，加入完整 DIFF 历史序列计算模块，由原词法解码器读取其逐商品表示，能否在同一 official validation 上超过已有 GRAM？只有新结构有希望后，才检查 PCRF 组合与 Beauty。

第一轮保留词法输出以避免再次同时承担 identifier 替换的损失；encoder / decoder / identifier 都不是后续研究的不可变约束。当前增加的是完整可学习分支与解码记忆接口，不是 loss-only 改动。

## 2. 来源与实际默认值

- 论文：[DIFF，SIGIR 2025](https://arxiv.org/abs/2505.13974)
- 作者代码：[HyeYoung1218/DIFF](https://github.com/HyeYoung1218/DIFF)
- 核对 commit：`ef6283a3bf9ed242a9b867fcf63368f0fb36eb33`
- 原始核心：`recbole/model/sequential_recommender/diff.py`
- 原始配置：`configs/Amazon_Toys_and_Games_diff.yaml`、`run_diff.sh`
- Toys 的实际启动脚本覆盖 YAML：**fusion=concat、alpha=0.5、alignment lambda=100**；不能只读 YAML 而误用 gate。
- 层数 2、heads 2、hidden / attributes / FFN 256，dropout 0.5、cutoff c=9、类别与品牌、属性 sum pooling。
- 作者原训练上限 400 epoch；本迁移采用已有 GRAM checkpoint 适配，训练日程不同，不能称为按原论文训练预算复现。

## 3. 来源机制到本项目的映射

| 来源机制 | 本轮实现 |
| --- | --- |
| Item embedding 与属性 embedding 分开建模 | 现有商品目录的 item embedding，以及现有 metadata 中 categories / brand embedding |
| Frequency filtering | 固定历史宽度上的 rFFT，低频分量 + beta² × 高频分量，dropout / residual / LayerNorm |
| Intermediate fusion | item、position、各属性分别计算 QK，再 concat 投影成注意力分数；value 使用 item 表示 |
| Early fusion | item、attributes、position concat 投影后，经独立 self-attention |
| 双路、多层更新 | 每层过滤两路，保留两套 attention 与共享 FFN；最终 alpha 混合 |
| 表示对齐 | 历史 item / 属性表示的双向相似分布损失；保留原计算主干与 lambda=100 |
| Item recommendation CE | 保留为新分支的训练目标，形成 DIFF 完整学习路径；不参与最终重排 |
| GRAM 接口（新增迁移） | 混合后的逐商品表示投影到 T5 hidden，作为额外 encoder memory，由原 decoder cross-attention 读取 |

GRAM 输入历史为最近商品在前；DIFF 内部翻为时间正序、固定 pad 到 20。所有分支只读取当前样本历史；目标 item 只进入训练 loss。频域过滤可以混合整个已知历史，不将中间位置当作独立未来预测来宣称严格逐位置因果。

必要适配明确记录：原 DIFF max history 50 → 本地 GRAM 20；属性字典来源为本地现有预处理文本；每层屏蔽 padding，防止动态 batch / 无效槽影响表示；最后一个序列状态用于 item CE，所有有效历史状态供 GRAM decoder 读取。原 item-scoring CE 加权进入生成训练，不直接替代词法解码。

总损失：`L = L_generation + 0.1 × (L_DIFF_item_CE + 100 × L_alignment)`。0.1 是本次生成任务的适配权重，不是作者默认值；独立记录三项原始 loss，避免只看到 total loss。

## 4. 与旧失败的区别

- CF0：旧实现是普通 item Transformer + 将结果加到各 passage；这里使用属性参与的双路注意力与频域过滤，并为 decoder 追加独立记忆。
- BiFlow：旧实现主要是 pooled global / history 向量之间的标量门控；这里保留逐商品序列与两条多头注意力计算。
- LATTE / SETRec：本轮不同时替换原词法输出，因此首先测试内部表示贡献。
- PCPS：本轮不以 beam 代理损失为核心，也不继续开展梯度方向 / TF32 诊断链。

## 5. 数据与已有参照

- 原 Toys 目录：`GRAM/rec_datasets/Toys`；商品文本、词法 ID、SASRec 相似项原样复用。
- 与原 GRAM 相同的 leave-one-out：最后一项 test，倒数第二项 validation，之前的全部 rolling transitions 训练。skip empty history，history 20，passage 128，beam 50。
- 本次只生成 train / validation 视图，不构造 test 样本，不读取 test predictions，不使用 D1/D2 或 Sports。
- 属性 vocabulary 用 train 出现商品拟合；目录其余商品只按同一映射转换，未知 / 缺失值显式处理。
- 已有 validation 参照：第二阶段 epoch 30 的 `NDCG@10=0.07627451426000033`、`Hit@10=0.11941067381001443`（已由已有 checkpoint_selection.csv 核对；不重跑基线）。
- 原 checkpoint：`GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt`。
- checkpoint SHA：`b0d76ea4da9a40b1be43c55c1c7e20cdca4e1eff0194acbbe1b3cc15f471fd82`。
- 历史 test NDCG@10=0.0591926919 仅作背景，不能与候选 validation 相减；PCRF 结果同理按 split 匹配。

## 6. 实施与训练预算

1. **prepare / CPU 检查**：来源哈希、metadata 映射、训练 / 验证样本、native 输入一致性；测试 padding、梯度到达两条 attention 与属性、训练与生成共用 memory、checkpoint 重载。
2. **真实模型 bounded smoke / profile**：少量 train 样本，前向 / 反向 / 更新 / Trie 生成；检查有限 loss、可达梯度、保存重载。它不承担效果判断。
3. **Toys 候选训练**：已有 parent 初始化。第一 epoch 冻结 GRAM，训练新分支与 memory 投影；后续最多 10 epoch 联合适配。新参数 lr=1e-3，GRAM lr=1e-5；AdamW，weight decay=0.01，clip=1，线性 warmup / decay。有效 batch=128，通过 microbatch / accumulation 适配显存；默认 FP32，不更换优化精度作为提点变量。
4. **选模与停止**：warmup 后及每 2 个 joint epoch 做完整 validation；至少完成 4 个 joint epoch，之后连续 3 个评估点无改善则停止，上限 10 joint epoch。以 validation NDCG@10 选择 checkpoint。所有评估点、loss 和耗时保留，停止规则不因效果临时放宽。
5. **探索判断**：明显负向则结束本候选；小幅正向且 Hit 无明显损失可进入一次有限适配 / PCRF 检查；不要求第一轮即完成最终显著性、多 seed 与消融。

当前只训练候选，不重跑 GRAM / PCRF 基线。已有 checkpoint 续训相对历史分数的差异是探索信号，最终方法确定后统一训练预算并重跑完整对照。

## 7. 资源与产物

- 单 GPU，实测 profile 后选择显存充足的卡；不停止或占用其他任务的既有进程，不运行占卡循环。
- 先完成 CPU / bounded smoke，依据峰值与吞吐确定 microbatch 和可执行日程。最大训练预算按上节固定，长训练用独立进程与持久状态文件。
- 模型代码、数据 adapter、配置、runner、tests 放在 `experiment/phase18/` 独立文件；尽量不修改原 GRAM 和旧实验代码。
- 输出目录：`artifacts/phase18/diff_gram/`；每次运行独立目录，保存配置、源码 / 输入哈希、学习曲线、checkpoint、逐用户 item predictions 与 split 明确的比较表。
- 状态必须区别 `PREPARED`、`SMOKE_PASSED`、`TRAINING`、`COMPLETED`、`FAILED`；工程完成不等于提点成功。

## 8. 当前执行状态

2026-09-08：方案与来源 commit 已记录；现有 metadata 含 categories / brand，准备实现。当前尚无本地 DIFF 效果结果。后续实测状态与任何必要适配追加在本节。

### 执行修订 A：属性可用性核查（训练前）

第一次 prepare 发现 11,924 件商品的 brand **全部为 na**。字段存在不等于属性有效。保留 `prepared_v1` 作为核查记录，正式首轮使用 `prepared_v2_categories`，只输入 categories，品牌分支不进入模型。类别 vocabulary 在 train 商品上拟合得到 466 个真实值，加 PAD / UNK 共 468 个；23 件商品类别缺失或未知。

本修订保留双路融合、频率过滤、共享 FFN、对齐与 item CE，但属性源由作者的 categories + brand 改为本地有效 categories。前文品牌方案由此条覆盖。不得把本轮称为原 DIFF 完整复现，也不从该轮负结果推出原论文机制无效。

数据实测：train 109,361 条、official validation 19,412 条、目录 11,924 件；573 个训练样本和 7 个验证样本与原生 GRAM 样本构造逐字段一致，另核对原生分词张量；parent SHA 与历史记录一致。未生成 test 样本。

### 实现与 GPU 检查完成（2026-09-08）

- 独立代码：`experiment/phase18/core/diff_sequence.py`、`diff_gram.py`、`diff_data.py`；runner：`experiment/phase18/protocol/s18_diff_gram.py`；配置：`experiment/phase18/config/s18_diff_gram_toys.json`。
- 新增 5,358,478 参数，总参数 65,875,854；原 GRAM 源码未为本候选修改。原 checkpoint 使用 32,128 行词表，已修正复用初始化工具默认缩为 32,100 行的兼容问题，严格加载全部 parent 参数。
- 7 项 CPU 行为测试通过，包括仅生成损失也能到达两路 / 属性 / memory、固定 padding、目标不进入编码输入、遮蔽追加 memory 时恢复原解码 logits、checkpoint 重载、freeze→joint 与尾部 accumulation / 选模。
- `smoke_v1_mb2`：沙箱内无 CUDA；`smoke_v2_mb2`：发现词表兼容问题；`smoke_v3_mb2`：训练通过，GPU 7 共享显存下降导致 beam 50 OOM。均是工程检查失败，不能作为模型效果负证据。
- `smoke_v4_gpu3_mb2` 与 `smoke_v5_gpu3_mb4` 均完成真实模型前向 / 反向 / 更新、50 个合法且唯一的商品候选生成、保存与重载一致性检查。
- 正式选择 microbatch=4、accumulation=32，有效 batch=128；GPU 3 UUID：`GPU-7952ad6e-2db7-fb2e-581b-34be56287640`。最长历史 20 的联合更新 peak allocated 4,322.92 MiB / reserved 4,610 MiB；beam 50 peak reserved 7,068 MiB，另有 CUDA 运行时与 optimizer 占用。共享卡速度波动大，完成时间以正式状态中的实测吞吐为准。
- Smoke 更新不进入正式模型。正式训练重新严格加载原 epoch 30 checkpoint 与同 seed 新分支；只训练候选。

### 正式运行与查看方式

运行目录：`artifacts/phase18/diff_gram/train_v1`。会话：`s18_diff_gram_toys`。控制台：`artifacts/phase18/diff_gram/train_v1.console.log`。

```bash
bash experiment/phase18/run_stage18_diff_gram.sh status --output artifacts/phase18/diff_gram/train_v1
tail -n 5 artifacts/phase18/diff_gram/train_v1/events.jsonl
```

`TRAINING` 表示正在训练（第一轮 phase=warmup，之后 phase=joint）；`VALIDATING` 表示正在跑完整验证集；只有 `COMPLETED` 并生成 `result.json` 才是这一轮结束。`FAILED` 需读 error / traceback，不能默认是效果失败。状态有更新时间，异常退出后也须结合进程检查是否仍存活。

计划评估在总 epoch 1、3、5、7、9、11 后进行，最多 warmup 1 + joint 10；保留全部验证分数，选择 NDCG@10 最好的候选 checkpoint。比较结果自动写入 `validation_epoch_XX.json`，其中 `historical_gram` 与 `delta` 直接引用已有第二阶段 validation 分数；完成后看 `result.json`。当前 GPU smoke 通过仅说明工程可运行，尚无提点结论。

正式启动已确认：2026-09-08 12:37（Asia/Shanghai），PID `1911709`，tmux 会话 `s18_diff_gram_toys`。12:38 状态为 `TRAINING / warmup / epoch 1`，已处理 1,292 / 109,361 个训练样本、完成 10 次 optimizer update；loss 有限，进度正常增长。第一分钟吞吐约 21.5 样本/秒，当前训练 epoch 预计还需约 84 分钟（不含随后验证；共享 GPU 波动，不是整轮完成时间承诺）。无须手动启动后续 joint 或 validation，runner 会按冻结日程自动执行。
