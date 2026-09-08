# 第十八阶段：耗时估计与 DiffGRM 原生并行计划 v0.1

## Material Passport

- 日期：2026-09-08；academic-research-suite / experiment-agent，inline。
- 用户要求：估算当前 Toys / Beauty 耗时；有其他值得跑的候选可以并行启动；持续遵守已有基线优先、大改允许、借鉴第十七阶段失败、最终统一重跑。
- 当前已有两条 DIFF→GRAM 任务，新增候选是 **DiffGRM**。二者名称接近，但机制不同。

同日后续：用户明确要求再启动 DiffGRM Toys，按 [Toys 并行补充计划](GRAM_第十八阶段_DiffGRM_Toys并行补充计划v0.1.md) 执行；下文“先只跑 Beauty”的历史资源决策据此更新。用户只通过四个 status 文件观察，不添加总览工具。

## 1. 当前两条任务的时间估计

2026-09-08 12:53 左右，Toys 适配 epoch 1 完成约 24%，当前 epoch 剩余约 52 分钟；Beauty 完成约 4%，剩余约 76 分钟。均实际完成 optimizer updates，状态 TRAINING。

- 当前 epoch ETA **不含随后验证和 10 个 joint epoch**。
- 结合现有训练吞吐和历史完整验证用时，第一份完整 validation 结果暂估还需 2–4 小时。不是保证，GPU 为共享资源。
- 跑满计划先按约 1–2 天预算。联合更新尚未开始，不能直接以冻结 backbone 时速度乘 11；GPU smoke 的最长历史 joint/frozen 耗时比例为 Toys 约 3.7、Beauty 约 1.7，不能当作全数据的稳定比例。
- 后续以首个完整 validation 与首个 joint epoch 的真实耗时修订；早停可能缩短，总 ETA 当前精度低于单 epoch ETA。

## 2. 新增候选为什么选 DiffGRM

[DiffGRM 论文](https://arxiv.org/abs/2510.21805)、[作者代码](https://github.com/liuzhao09/DiffGRM)，WWW 2026。核对 commit：`238911435b6ca4f576d0b4d943359b98c865c172`。当前仓库已含 CC BY-NC 4.0 与第三方归属说明，旧第十七阶段卡片“未见标准许可证”的观察不再代表当前仓库。

与 DIFF 属性融合相比，这是一条替换 tokenizer / encoder / decoder 的独立大改路线：

1. Sentence-T5 → train-fit whitened PCA → OPQ/PQ 并行语义编码（PSE）。
2. 双向 masked discrete diffusion decoder，保留作者 guided least / MSP 的连贯多视图训练（OCN）。
3. confidence-guided 任意位置展开（CPD），使用作者生成实现。

第十七阶段 DiffGRM 是 3,000 用户、5 epoch 的独立缩放实现，且 tokenizer 与来源不同。此次直接调用固定作者代码，使用本项目完整 Beauty 训练前缀与完整 validation；不把旧缩放实现的负结果冒充官方完整路径失败。LATTE / SETRec 已有完整迁移负证据，当前不重跑。HSTU 保留备选，但不同时再扩一个相近的历史分支实验。

这一步检验完整原生路线在同任务上能否超过已有强 GRAM，不提前称为自己的方法。若有明显绝对差距，及时降低该路线优先级；若有正信号，再决定迁移组合和方法设计。PCRF 的 native score 接口需要候选入围后明确，不能直接把 SID 顺序或任意分数硬套进去。

## 3. 首轮运行域与数据适配

先运行 **Beauty**：作者公开 Beauty 模型为 hidden=256、heads=4、encoder 1 层、decoder 4 层、FFN 1024；Toys 官方 hidden=1024 的成本更高，先不额外启动第四条训练。

- 复用 `GRAM/rec_datasets/Beauty` 的商品目录、交互序列和现有 item_plain_text，不重新下载 / 清洗数据；metadata 来源与作者重新处理 raw meta 的文本不完全相同，明确为本项目数据适配。
- 最后两项分别留作 validation / test；仅准备 train / official validation。目标只进入训练标签和指标计算。
- 为与已有 GRAM 训练信号一致，完整 rolling transitions 保留最短历史 1（作者例子 min_hist_len=2）；history max 50 沿用来源模型，GRAM 为 20，记录上下文预算差异。
- Sentence-T5 权重复用已缓存的 `fc5d4628481afbbaaacd7af6bb07cf9d3865f781`；原生环境复用 phase17 隔离环境，不升级现有 GRAM 环境。软件版本与作者 requirements 不逐项相同，记录实际环境。
- PCA / OPQ 仅在 train 出现商品上拟合；全目录只用同一变换编码；记录 SID 碰撞。不得用 validation 目标指导编码器或量化器训练。
- 主参照直接读取第一阶段 epoch 25 validation：NDCG@10=0.06497370269327131、Hit@10=0.10875106202209006，不重跑 GRAM 或 PCRF。

## 4. 完整学习预算与评估

- 保留作者 Beauty 架构、四位 SID / 每位 256、guided least / MSP / refresh=false、label smoothing=0.2、AdamW lr=0.01 / weight decay=0、cosine scheduler / warmup 10,000 updates、FP32。
- effective batch=1024；microbatch 由真实模型 profile 确定并 accumulation，不缩小 hidden 或 decoder 层数。
- 最多 200 epoch；从 epoch 20 开始每 10 epoch 完整 validation，至少 100 epoch 后允许连续 5 个评估点无改善早停，按 item NDCG@10 选模。最低训练预算覆盖 10,000-step warmup，避免再次拿 5 epoch 判定原生路线无效。
- 不运行 native AR 控制、GRAM 基线、多 seed / 完整消融；先用已有强基线找绝对收益。
- CPD beam=256，输出有序 raw SID candidates 后去重 / 过滤无效 SID 并展开成真实 item，最多 50 个，不用重复项填充。SID 碰撞按 train 出现次数与固定 item ID 排序，不看验证 gold。记录覆盖率与不足 50 的用户比例。
- beam 256 与作者 test 档相同，但本轮只跑 validation，且与历史 GRAM beam 50 预算不同；只作为探索比较，不宣称匹配算力归因。
- 改为 item 级评估，避免 SID 命中把碰撞商品误算为正确；不使用来源 evaluator 的 SID 命中作为本地提点证据。

## 5. 执行与产物

独立源码：`artifacts/phase18/diffgrm_native/sources/238911435b6ca4f576d0b4d943359b98c865c172`，保留来源与许可证。新 pipeline 实现本地数据边界、状态、训练日程与 item 指标；来源 model / tokenizer / beam 文件不修改。

自动流水：CPU contracts → 实际模型 GPU profile → Sentence-T5 与 OPQ 准备 → 真实数据训练 / 生成 smoke → 候选训练和 validation。Smoke 不作提点证据；正式从头初始化，不使用 smoke 更新后的权重。

输出目录：`artifacts/phase18/diffgrm_native/beauty/run_v1`；tmux 会话 `s18_diffgrm_native_beauty`。状态区分 `PREPARING`、`PROFILE_PASSED`、`SMOKE_PASSED`、`TRAINING`、`VALIDATING`、`COMPLETED`、`FAILED`。只有参数更新才记为训练，不能把 tokenizer 准备称为训练完成。

### 2026-09-08 13:14 启动记录

- CPU 4 项检查通过：原始时序切分、SID padding / history-only 推理、碰撞商品落地与无重复填充、作者 guided 四视图梯度与 CPD checkpoint roundtrip。记录：`artifacts/phase18/diffgrm_native/beauty/cpu_contracts.json`。
- 原先用于导入检查的 `latte_05e4e6d98322` 装的是 torch 2.13 / CUDA 13，与当前驱动不匹配；正式使用已有 `latte_05e4e6d98322_torch_2_7_1_cu126`，torch 2.7.1+cu126，不安装或改动其他环境。
- tmux `s18_diffgrm_native_beauty` 已启动，PID `2335375`，GPU 1 / `GPU-2afcc1df-7df8-1f3f-8970-a252a06c6160`。GPU 3 / 7 的两个 DIFF→GRAM 继续运行。
- 完整 Beauty 架构 GPU profile 通过：5,602,048 参数，microbatch 128；训练峰值 reserved 1,068 MiB，beam 256 / batch 8 生成峰值 3,856 MiB，保存恢复后生成相同。显存数字不含全部 CUDA 上下文与其他进程。两步 profile 含初始化与梯度检查，不直接外推完整训练耗时。
- 当前已进入 Sentence-T5 商品编码，随后自动 PCA / OPQ、真实数据 smoke、全量训练。该准备状态尚不算正式训练。
- 13:13 最新现有任务：Toys warmup 完成约 59%，当前 epoch 剩余约 25 分钟；Beauty 约 30%，剩余约 55 分钟。完整验证及后续 joint 训练尚未完成。

### 2026-09-08 13:18 已进入全量训练

- 13:17:32 开始正式训练；13:18:02 已处理 43,520 / 131,413 个训练样本，完成 42 optimizer steps，平均 loss 5.53497、learning rate 0.000042，未出现非有限值。第一 epoch 估计约 91 秒；这次是真实参数更新，不是仅启动会话。
- 准备完成：训练样本 131,413，完整 validation 22,363；12,101 商品对应 10,594 个不同 SID，784 个碰撞桶，最大桶 37 商品。实际 item 指标会按上述固定 train-frequency 排序展开碰撞，不以 SID 命中冒充 item 命中。
- 真实数据 smoke 通过：最长历史 microbatch 128，完整生成 `[8, 256, 4]`，训练峰值 reserved 1,506 MiB、生成 4,252 MiB，checkpoint roundtrip 相同。相比合成 profile 显存增加，仍在当前 GPU 余量内。
- 新增 DiffGRM Beauty 的粗估：约 1.5 分钟 / epoch；epoch 20 训练约 30 分钟，完整 validation 按实测 0.39 秒 / 8 用户约 18 分钟外推，含负载波动后第一份 validation 先按 **1–2 小时**，最多 200 epoch + 定期 validation 整轮先按 **6–12 小时**。尚未完成首个完整 validation，不能把该估计当作已测完整时间。
- 13:17 时原 DIFF→GRAM：Toys 当前 warmup 剩余约 22 分钟；Beauty 剩余约 52 分钟。整体仍暂按 1–2 天，第一份 validation 暂按今天下午，优先根据第一份 validation / joint epoch 实测更新。

只读查看三条状态：

```bash
cat artifacts/phase18/diff_gram/train_v1/status.json
cat artifacts/phase18/diff_gram/beauty/train_v1/status.json
cat artifacts/phase18/diffgrm_native/beauty/run_v1/status.json
```

`TRAINING` 看 epoch / seen / optimizer_steps 是否增长；`VALIDATING` 看 evaluated；`PREPARING` 看 phase / encoded_items 与 console。`COMPLETED` 且存在 `result.json` 才是整轮结束，`FAILED` 读取 error 和 traceback。
