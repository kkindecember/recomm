# GRAM 第十四阶段

**阶段主题**：从 R² 的外部 cold reachability，走向 GRAM 原生生成路径的冷路径支持。

**当前状态（2026-08-20）**：M1 Stage 14-0A 已 `PASS`；14-0B probe 已实现且 18 tests OK。工具原生 session 下仍在 GPU3 首次 CUDA 分配前终止，最小 PyTorch CUDA 探针复现；当前需要一张可正常分配、至少 12 GiB 空闲（建议 16 GiB）的其他 GPU。尚未得到 14-0B 模型结果。

## 先读什么

| 文件 | 用途 |
|---|---|
| [`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.4.md`](./GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.4.md) | **当前执行主计划**：双结构口径、staged seed promotion、Toys-only ablation 上限与 19–27 顺序 GPU-days 训练预算包络 |
| [`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.3.md`](./GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.3.md) | 方法算子与科学口径修正版；末尾保留另一专家回评，供版本审计 |
| [`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.2.md`](./GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.2.md) | 专家审阅后的战略收缩版，保留用于版本审计 |
| [`GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.1.md`](./GRAM_第十四阶段_R2突破与ColdPath蒸馏探索计划v0.1.md) | 初版完整论证与严格 Gate，保留用于版本审计 |

## 一句话决策

第十四阶段保留第十三阶段的 **R²（domain-local content resolver + warm-anchored portfolio）** 作为有效贡献点、teacher 与强基线，但不再继续堆 resolver 训练轮数、hard negative、gating 或 slate allocator。预期主线 **R²-to-Path Distillation（R2PD）** 显式构造 teacher candidate path 的 synthetic decoder prefixes，以 absolute prefix mass 加权 soft next-token distillation，并用冻结 v0 retention 抑制 warm forgetting；是否进入训练由真实 NLL/rank/beam 诊断决定。预算采用 seed-0 双域筛选后再条件扩到 3 seeds，不在方法成立前预付全部 GPU 成本。

## 阶段边界

- 第十三阶段原 v1 Semantic Bridge 的 raw gain 已被 collision audit 否定；不得复活该结论。
- R² 的双域正向结果没有被否定，但只是单 split、单 resolver seed、validation-level 证据。
- T1-4 多兴趣 resolver 若以后补跑，只能作为第十三阶段封口实验或 resolver ablation，不是第十四阶段主创新。
- SpecGR 已覆盖“inductive drafter + GR verifier”，GenRecEdit 已覆盖 cold SID model editing，USIM 已覆盖“想象用户序列”，AGRec 已覆盖“辅助模型增强 token logits”。R2PD 只能主张 user-conditioned prefix distribution transfer + 标准 native beam 的组合差异。
- Toys/Beauty 都已用于开发；test 只允许冻结后批量开封一次。Sports/第三域延期到双域 `PASS_R2_TRANSFER` 以后，不得在方法尚未成立时提前消耗。

## 目录约定

计划通过后再建立以下实现目录；当前不预建空代码：

```text
experiment/phase14/
artifacts/phase14/
report/第十四阶段/
```

任何实现都不得原地覆盖第十三阶段 frozen artifacts，也不得直接修改原始 GRAM 文件后让旧实验不可复现。
