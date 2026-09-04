# GRAM 第十八阶段：S18-1 可作用性诊断执行补遗 v0.1

## Material Passport

- Parent Plan：`GRAM_第十八阶段_PCPS-GRAM词法锚定协同前缀生存与低风险验证计划v0.2.md`
- Scope：只补足 S18-1 中未冻结的 fold-local parent/teacher 构建与诊断口径；不改变 S18-1 Gate
- Created：2026-09-03
- Status：`PREREGISTERED / AWAITING_RESOURCE_AUTHORIZATION`
- Scientific attempt：尚未启动；本补遗与 CPU preflight 均不读取任何效果结果

## 1. 为什么需要补遗

S18-1 要求 `G_parent` 与 `H_cf` 只能看相应 fold cutoff 之前的数据，但主计划没有冻结 parent
的初始化、训练轮数与 checkpoint 选择。历史 Toys/Beauty GRAM checkpoint 已经看过 I-1/I0 cutoff
之后的 transition，不能复用；否则 first-drop 与 beam headroom 会发生未来泄漏。另一方面，临时缩短
parent 训练又可能把“模型尚未学会”误判成“没有可作用前缀瓶颈”。因此在首次 S18-1 scientific
attempt 前冻结以下口径。

## 2. Fold-local parent 与 item-head

- 每个 `domain × fold` 从本机冻结的通用 `t5-small` snapshot 独立初始化，不加载任何历史
  domain-specific GRAM checkpoint；四个 parent 不跨 fold 续训或共享 optimizer。
- parent fit population 为该 domain/fold 的全部 eligible D0-train-only 用户，而不是只用 1,024
  诊断用户。每行只把 visible prefix 用作增强 next-item transition；fold target 不进入 parent fit、
  validation、early stopping 或 checkpoint selection。
- parent 固定训练 10 epochs；沿用原 GRAM `rec_batch_size=16`、gradient accumulation `8`、
  `rec_lr=1e-3`、warmup `0.05`、max history `20`。只保存 epoch 10；不依据 target 指标挑 checkpoint。
- 选择 10 epochs 的依据仅来自已存在的历史训练曲线：Toys epoch 10 的 validation Hit@50 已达到
  历史 epoch 30 的约 95%，可降低欠训练导致假阴性的风险；这个决定在新 target effect 未读时冻结。
- `H_cf` 使用 Phase9 的 `CF0B2ItemHead` 结构，按同一 fold visible transitions 独立训练 10 epochs；
  固定最后一个 epoch，不做 target-based early stopping。
- frequency 与 `q1` 只由 fold visible occurrence 计算；`q1` 是所有 positive-frequency catalog
  items 的 train-frequency 第 25 百分位（lower empirical quantile）。

## 3. Cohort 与诊断口径

- 每域从同时满足 I-1/I0 的用户中按
  `sha256("S18-1|2023|<domain>|<user>")` 排序取前 1,024 名；两个 fold 使用同一 cohort。
- Toys cohort SHA256：`3872e5545d3f410452fdf3396c57bb42bed77575b1e4a1b4a9093a29ed919248`。
- Beauty cohort SHA256：`91c13e190600322f0f3c3d572a312a9ce14e006e00401ad15b896caf069152fd`。
- first-drop depth：beam50 解码时，target prefix 第一次不再出现在 active beam prefixes 的 token
  深度；decoder start token 不计深度，EOS 计入合法 path。
- legal actual pruner：first-drop depth 上，与 target 共享上一层 prefix、选择另一合法 child，且能
  映射到 parent on-policy returned legal path 的错误 item。全局竞争导致没有同节点错误 child 时记空，
  不用全词表 token 补 denominator。
- `K=8` hard negatives：先在 beam200 同节点实际错误 paths 内按冻结 PCRF joint score 排序；不足
  8 时，从合法 sibling descendants 中按 `z(cf)-0.5*z(log1p(freq))` 补足。Recall 的 denominator
  仅为该 event 的 legal actual-pruner item set。
- parent full-path score 用 normalized transition log probability；同时保存 raw cumulative log
  probability、path length 与 target-minus-negative paired gap。
- CF stability score 对每个用户按完整 catalog logits 标准化：
  `(target_logit - catalog_mean) / catalog_std`，比较同一 cohort 的 I-1/I0 均值漂移；不得在 fold
  内再次把 target-score 向量整体 z-score（否则均值机械为 0）。
- I1/I2 不构造 view、不训练模型、不生成 cohort、不写 target artifact。

## 4. 执行与资源

- S18-1 是大于 10 分钟的后台任务，必须使用具名 tmux；agent 不实时监看。
- 研究者只通过 `artifacts/phase18/status/s18_s1_actionability.status.json` 查看状态。
- 推荐资源：4 张不冲突的 GPU，单卡可用显存至少 30 GiB，四个 domain-fold 独立并行，预计
  3.5–4.5 小时。
- 最小资源：2 张同规格 GPU，分两波运行，预计 7–9 小时。
- 排除 Stage17 occupancy guard 所占物理 GPU4；GPU1 继续按主计划默认不用于 Stage18。
- 未得到研究者资源授权前，只允许 CPU preflight、合同测试与代码 smoke，不启动 scientific attempt。
- 不自动重试，不自动启动 S18-2；S18-1 结束后只产出 Gate 结论和报告。
