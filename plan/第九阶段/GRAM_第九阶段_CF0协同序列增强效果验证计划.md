# GRAM 第九阶段：CF0 协同序列增强效果验证计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-03
- Verification Status: `PLANNED`
- Version Label: `phase9_gram_cf0_effect_pilot_v2`
- Development domains: Toys、Beauty
- Test/Sports: 本轮不使用
- Resource policy: CodeLlama 实验前后占位；实验期间持有 30 GiB 总显存租约

## 1. 阶段定位

第九阶段不延续 TIPA、teacher、adapter 或解码后处理路线，改为对 GRAM 模型内部进行
结构级改造。第一轮的唯一目标是：

> 验证原始商品 ID 的协同序列信号能否在保留 GRAM 文本生成能力的同时，稳定提升
> Recall@10 和 NDCG@10。

本阶段允许借鉴已有论文的成熟结构，`CF0` 只是效果原型，不视为最终论文创新。
真正的方法创新点将在 CF0 出现可复现增益后，依据消融和错误分析再确定。

## 2. CF0 最小模型定义

CF0 在 GRAM 内部增加一条商品 ID 协同序列路径：

1. dataset/collator 额外返回 `history_item_ids`、`history_item_mask` 和 `target_item_id`；
2. 使用可训练 item embedding 和两层 causal Transformer 编码历史商品 ID；
3. 保留 GRAM 原有 T5/FiD 文本编码和 Trie 约束生成；
4. 使用动态门控将 item-level 协同表示注入对应历史商品的 token 表示；
5. 训练损失为原 GRAM token CE 与 next-item 分类损失的加权和；
6. 完整版在 Trie 合法 beams 内联合生成分数和 item-level 分数排序。

第一轮不加入双记忆 decoder、图神经网络、新 Semantic ID、prefix survival loss 或复杂
对比学习，避免同时引入太多变量。

## 3. 对照组

| Arm | 模型 | 目的 |
|---|---|---|
| A | 原始 GRAM | 固定基线 |
| B | GRAM + ID Transformer + next-item loss | 判断协同监督本身是否有效 |
| C | B + 动态门控注入 + beam 联合评分 | 验证协同信号能否转化为最终生成排序收益 |

A/B/C 使用相同的数据划分、T5 规模、beam size、Trie、训练样本和评测脚本。主实验
先使用 T5-small，不用 backbone 参数量解释模型提升。

## 4. 实施与训练顺序

### P9-0：数据与模型接口

- 建立稳定的 raw item ID 到连续 embedding index 映射，保留 padding/OOV；
- 验证 history 顺序、target 截断、padding mask 和 item–lexical ID 映射；
- 保证关闭 CF0 时，模型输出与原 GRAM 一致或复用同一基线评测路径。

### P9-1：单 seed 效果 pilot

- 从已有 GRAM checkpoint 初始化文本生成路径；
- 先预热 item embedding、ID Transformer 和 next-item head；
- 再联合微调新分支、融合层与 GRAM 顶层；
- 先在 Toys 跑通 A/B/C，无训练或评测异常后再扩展到 Beauty。

学习率、损失权重、预热轮数和解冻范围在实现时先选一组保守默认值，不在本计划中
预先锁死；若 pilot 显示优化不稳定，可根据 loss/gradient 记录另写 recovery 计划。

## 5. 评测与记录

主指标：

- Recall@5、Recall@10、Recall@50；
- NDCG@5、NDCG@10。

机制记录：

- 目标商品是否已在原 GRAM beam@50 中；
- B/C 对目标商品 rank 的变化；
- item head 独立的 Recall@10/50；
- 生成分数与 item 分数的尺度、相关性与排序贡献；
- head/tail、历史长度和原 GRAM hit/miss 分层；
- 分项 loss、gradient norm、参数量、显存、训练和推理时间。

## 6. 初始决策规则

以下只是 pilot 的方向判断，不是最终论文门槛：

- 若 B 相对 A 在 Toys/Beauty 的平均 Recall@10 或 NDCG@10 有正增益，则保留协同序列分支；
- 若 C 进一步超过 B，则进入融合/生成对齐方向；
- 若 item head 强但目标经常不在 beam 内，下一阶段研究协同信号参与逐步解码；
- 若目标已在 beam 内但排名低，下一阶段研究端到端 item-aware scoring；
- 若 B/C 两域均无收益，先检查 ID branch 单独预测能力和融合梯度，不直接叠加更多模块；
- 若只提升 head 商品且明显伤害 tail，则将后续问题收窄为冷门商品的语义—协同自适应融合。

## 7. 阶段产物

P9-1 完成后至少保留：

- A/B/C 的配置、checkpoint 与统一指标 summary；
- per-user prediction/rank 记录；
- 分项 loss 和训练日志；
- 完整性检查与结果报告；
- 一份基于实验数据的第九阶段后续计划。

## 8. 资源、后台运行与状态管理规则

本节为第九阶段所有 GPU 子实验的默认运行协议。科学配置、科学退出状态与资源恢复状态
必须分开记录。

### 8.1 CodeLlama 占位协议

- 每次正式 GPU 实验开始前，先确认 CodeLlama 在本次目标物理 GPU 上处于占位状态，并把
  实际 GPU 编号写入冻结配置和 `status.json`；
- runner 在 CPU 单测、语法、配置和输入完整性检查通过后，调用
  `tools/run_codellama.sh stop` 释放目标 GPU；
- 不论实验成功、科学门失败、程序非零退出、timeout、手动 `stop` 或预启动阻断，所有
  退出路径都必须尝试在同一物理 GPU 上恢复 CodeLlama；
- 恢复后必须检查 CodeLlama tmux 和运行状态，写入
  `restored` 或 `failed_to_restore_resource`；
- CodeLlama 恢复成功不得覆盖实验本身的失败退出码；恢复失败也必须与科学结果独立报告。

### 8.2 30 GiB 显存租约

- CodeLlama 停止后，runner 必须等待目标 GPU 可用显存至少为 `30,720 MiB`；超过预设
  等待时间仍不满足时标记 `blocked`，不启动 workload；
- 正式运行期间，workload 实际占用与 `experiment/gpu_memory_lease.py` sidecar 占位合计
  保持 `30,720 MiB` 总租约，不在 workload 之外另外重复占用 30 GiB；
- workload/sidecar 的具体分配在实现后根据 smoke-test peak 冻结；sidecar 未成功进入
  `holding` 时不得启动正式训练；
- 运行期间每 5 秒记录 GPU index、used/free memory 和 utilization 到 `gpu_telemetry.csv`；
- 若 CF0 的实测单进程峰值超过 30 GiB，必须先根据 smoke 结果单独修订租约，不允许在
  正式运行中静默取消资源保护。

### 8.3 后台 runner 与用户接口

- 预计超过 20 分钟的实验必须在具名持久 `tmux` 会话中后台运行，不依赖当前终端
  或 Codex 会话存活；
- 第九阶段 runner 实现后统一提供：

  ```bash
  bash experiment/phase9/run_phase9_gram_cf0.sh start
  bash experiment/phase9/run_phase9_gram_cf0.sh status
  bash experiment/phase9/run_phase9_gram_cf0.sh stop
  ```

- `start` 只启动当前已授权的子实验；不自动启动后继 arm、读取 Sports/test 或进入下一阶段；
- `status` 必须为只读操作，并显示 tmux 是否存活、当前 `status/stage/reason`、物理 GPU、
  runner/workload PID、CodeLlama 占位/恢复状态，以及最新日志；
- `stop` 优先向 workload 发送 `TERM` 并走统一清理路径，不直接遗留 sidecar、telemetry 或
  CodeLlama 未恢复状态；
- 日志、`status.json`、telemetry 和结果必须持久化到具名 `artifacts/phase9/...` 目录，不只写在
  tmux pane 内。

### 8.4 启动、超时与失败规则

- 正式启动前通过 CPU 单测、Python compile、Bash syntax、JSON/config 检查，并冻结
  implementation、test、runner、config、输入数据与 parent checkpoint 的 SHA256；
- hard timeout 在完成小规模 smoke 后按实测耗时写入子实验配置；只有 hard timeout 可自动
  终止 workload；
- 非零退出、OOM、NaN/Inf、timeout、输出不完整或科学门未通过均不自动重试；
- 重试或 recovery 必须先保留原日志和状态，说明失败原因、允许改动的运行参数，并获得研究者
  明确授权；
- 完成后只保存结果并报告终态。未经研究者明确要求，不自动实现/启动下一轮，不读取
  Sports/test。

## 9. 当前状态

P9-0 的 dataset/collator、CF0-B/C 模型路径、联合损失、单卡 runner 和 CPU 测试已实现。
CPU 单测 4/4 通过，真实 Toys batch 的 item/text 对齐检查通过，使用真实 T5-small 和
原 GRAM checkpoint 的 CF0-B forward/backward 通过。

P9-0 GPU smoke 首次运行完成训练与 validation 后，因 `save_predictions=0` 时收尾日志引用了
未定义的 `pred_fname` 而非零退出；原日志和产物保留。研究者授权只修复该日志 guard 的
具名 recovery。recovery 于 2026-08-03 在物理 GPU4 成功完成：100 个 train/validation 调试样本、
1 epoch、30 GiB 总租约，checkpoint/metrics 完整，test/Sports 未读，CodeLlama 已恢复到 GPU4。
该指标只作 smoke 诊断，不作收益结论。规范摘要：
`artifacts/phase9/cf0_b_toys_smoke_recovery/summary.json`。研究者已授权下一步单 seed 全量 P9-1：
Toys、arm B、seed 2023、5 epochs（1 epoch CF0 pretrain + 4 epochs top-layer joint tuning）、全量
validation 和 30 GiB 总租约；启动前发现物理 GPU4 已被既有任务占用，仅将资源位置迁移至空闲的
物理 GPU6，模型、数据、seed、训练及评测参数均不变；当前已重新冻结并准备后台启动。

### 2026-08-04 P9-1 与后验诊断更新

Toys、arm B、seed 2023 的 5-epoch P9-1 已成功完成。validation Recall@10 为
`0.084947`、NDCG@10 为 `0.055441`，相对原 GRAM epoch-30 validation 分别下降
28.86% 和 27.31%。研究者随后授权 item-head、融合和梯度后验诊断。

诊断已完成：item-head Recall@10/50 为 `0.009376/0.032403`，均低于 popularity
baseline `0.012312/0.034927`；tail/middle target 的 item-head Recall@50 均为 0。
512 样本 teacher-forced 消融中，完整 CF0-B NLL 为 `4.8134`，bypass CF0 路径为
`1.6013`；完整路径在 95.31% 样本上更差。joint 阶段 CF Transformer 的 generation
gradient L2 约为加权 item gradient 的 26 倍，且解冻 tied LM head 实际同时解冻了
16.45M 参数的 shared embedding。

当前判定为“CF0-B v1 设计未达到机制验证条件”，不是“协同序列机制无效”。未经研究者
进一步授权，不启动 arm C、Beauty 或 test。完整报告：
`report/第九阶段/GRAM_第九阶段_CF0-B_ItemHead融合与梯度诊断报告.md`。
