# GRAM 第十四阶段

**阶段主题**：从 R² 的外部 cold reachability，走向 GRAM 原生生成路径的冷路径支持。

**当前状态（2026-08-21）**：M1 已完成；M2 Stage 14-1 正式筛查完整运行但科学 Gate 失败，verdict=`FAIL_STOP_PATH_TRANSFER_STAGE14_1`。R2PD 当前主线停止，14-2/M3/M4 与额外 seed 均不启动；GPU5 项目 holder 已恢复约 20 GiB。

## 先读什么

| 文件 | 用途 |
|---|---|
| [`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.4.md`](./GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.4.md) | **当前执行主计划**：双结构口径、staged seed promotion、Toys-only ablation 上限与 19–27 顺序 GPU-days 训练预算包络 |
| [`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.3.md`](./GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.3.md) | 方法算子与科学口径修正版；末尾保留另一专家回评，供版本审计 |
| [`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.2.md`](./GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.2.md) | 专家审阅后的战略收缩版，保留用于版本审计 |
| [`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.1.md`](./GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.1.md) | 初版完整论证与严格 Gate，保留用于版本审计 |
| [`../../report/第十四阶段/Stage14_M2_PseudoCold迁移筛查报告.md`](../../report/第十四阶段/Stage14_M2_PseudoCold迁移筛查报告.md) | M2 唯一阶段报告：工程试错合并摘要、正式负结果、资源与 stop 决策 |

## 一句话决策

第十四阶段保留第十三阶段的 **R²（domain-local content resolver + warm-anchored portfolio）** 作为有效历史贡献点、teacher 与强基线。R2PD 已完成 item-disjoint pseudo-cold 机制筛查，但 soft subtree distillation 没有显著优于 hard-path CE；因此按预注册规则停止，不继续 14-2、full training、Beauty 或 seed expansion。SpecGR/GenRecEdit 仍只是兼容性边界，若要本地 port 必须另立计划并重新授权资源。

## 阶段边界

- 第十三阶段原 v1 Semantic Bridge 的 raw gain 已被 collision audit 否定；不得复活该结论。
- R² 的双域正向结果没有被否定，但只是单 split、单 resolver seed、validation-level 证据。
- T1-4 多兴趣 resolver 若以后补跑，只能作为第十三阶段封口实验或 resolver ablation，不是第十四阶段主创新。
- SpecGR 已覆盖“inductive drafter + GR verifier”，GenRecEdit 已覆盖 cold SID model editing，USIM 已覆盖“想象用户序列”，AGRec 已覆盖“辅助模型增强 token logits”。R2PD 只能主张 user-conditioned prefix distribution transfer + 标准 native beam 的组合差异。
- Toys/Beauty 都已用于开发；test 只允许冻结后批量开封一次。Sports/第三域延期到双域 `PASS_R2_TRANSFER` 以后，不得在方法尚未成立时提前消耗。

## 目录约定

当前实现与证据目录：

```text
experiment/phase14/
artifacts/phase14/
report/第十四阶段/
```

任何实现都不得原地覆盖第十三阶段 frozen artifacts，也不得直接修改原始 GRAM 文件后让旧实验不可复现。
