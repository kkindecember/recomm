# GRAM 第十阶段：CF1-B0 生成分数身份验证与资源 Pilot 计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_B0_TOYS_SCORE_IDENTITY_V1`
- Preregistration Status: `FROZEN_BEFORE_PILOT`
- Scope: 64 deterministic Toys validation users，cached G50 only

## 1. Static score-definition audit

冻结运行环境为 Transformers `4.26.0`。其 beam search 先对全词表 logits 做 `log_softmax`，再由
`PrefixConstrainedLogitsProcessor` 将 trie 外 token 置为 `-inf`，没有对合法 token 子集重新
归一化。`BeamHypotheses` 在 length penalty=1 时用累计 token log-prob 除以生成长度。

因此 identity scorer 固定为：teacher-forced 全词表 log-prob（包含 EOS）求和，再除以包含 EOS 的
预测 token 数。合法子集重归一化不是 cache score identity，本轮不运行也不用于 gate。该定义在
读取 B0 输出前由本地源代码审计确定。

## 2. Frozen sample and inputs

- 从 19,412 validation users 按 `sha256("2023:" + user_id)` 排序取前 64；
- 每用户只重算 cache 中已有的 G50，合计 3,200 user-candidate pairs；
- checkpoint：Toys epoch-30 frozen GRAM；
- prompt/history/collator 与原 validation 配置一致；
- GPU5，float32，candidate micro-batch=10；不读取 Toys test/Beauty/Sports。

## 3. Frozen gates

1. 3,200 个分数全部 finite；
2. pooled Pearson `>=0.995`；
3. pooled Spearman `>=0.995`；
4. mean per-user top-10 set overlap `>=0.98`；
5. recomputed 与 cached pilot Hit@10 absolute delta `<=0.001`；
6. peak allocated GPU memory `<=12000 MiB`；
7. wall time `<=1800 s`。

失败不自动 retry、不更换 scorer 定义。通过后才建立 512-user `fill_cf_only_40` arbitrary-candidate
resource pilot。

