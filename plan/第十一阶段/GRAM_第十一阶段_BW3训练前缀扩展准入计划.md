# GRAM 第十一阶段 BW3：Train-Prefix 扩展候选准入计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-05
- Verification Status: PREREGISTERED_DESIGN
- Experiment ID: `GRAM_PHASE11_BW3_TRAIN_PREFIX_EXPANSION_ADMISSION_V1`

## 1. Research question

BW1 证明 beam200 在 Toys / Beauty 各提供约 12–13pp candidate-recall headroom，但冻结 PCRF
不能把新增候选送入 top10。BW2 又证明独立 beam width 会改变约 9% 的 top50 搜索路径，单纯 anchor
normalization 不能被干净识别。

BW3 检验：能否只用 validation 之前的 train-prefix pseudo-future，学习一个 expansion admission
gate，在保留冻结 beam50 PCRF top10 的前提下，识别 beam200 中值得进入 top10 的扩展候选？

## 2. Temporal isolation

对每个原始序列定义：

- `t=-4`：fit pseudo-target，history 截止 `[:-4]`；
- `t=-3`：calibration pseudo-target，history 截止 `[:-3]`；
- `t=-2`：一次性 validation target，仅在方法与 threshold 冻结后读取；
- `t=-1`：test，继续封存。

任何 fit/calibration 特征、标签、阈值或 early stopping 均不得使用 `-2/-1`。Toys、Beauty 分域训练，
不共享 item embedding 或阈值；模型结构和阈值选择规则共享。

## 3. Candidate construction

- 每个 pseudo-event 和 validation event 独立生成 beam50 与 beam200；
- beam50 使用已确认 PCRF，作为不可修改的 base list；
- expansion pool 仅取 beam200 中不在 beam50 的候选；
- base top50 的相对顺序和分数冻结，gate 只能决定扩展候选是否越过 top10 boundary；
- 若无扩展候选通过 gate，逐用户严格 fallback 到原 beam50 PCRF。

## 4. Admission model

使用小型、可审计的线性 listwise residual，不输入 user ID 或 validation-derived statistics。候选特征固定为：

1. GRAM raw sequence score 及相对 beam50 anchor 的 z-score；
2. 冻结 item-head score 及 anchor z-score；
3. train-prefix popularity 与 anchor z-score；
4. beam200 rank / 200；
5. PCRF reliability、CF-pop adjusted score；
6. candidate 是否同时出现在 beam50（仅作 base mask，不作为可学习捷径）。

在 `t=-4` 以每用户 listwise cross-entropy 拟合；只对目标进入 union candidate pool 的事件计 ranking
loss，同时报告 coverage attrition。模型 seed 固定为 2023，不做 validation seed 搜索。

## 5. Calibration

在 `t=-3` 从预先固定的 admission margin grid
`{0, 0.25, 0.5, 0.75, 1.0}` 中选择最小安全 margin：

1. Hit@10 不低于 beam50 PCRF；
2. NDCG@10 delta ≥ `-0.001`；
3. 至少一个 expansion candidate 被准入；
4. 满足安全条件后按 Hit@10、NDCG@10、较大 margin 的字典序选择。

若没有 margin 通过，域内 gate 失败，validation 不读取。

## 6. Staged execution

### P0 — data/decode smoke

- 每域 16 个 `t=-4` + 16 个 `t=-3` 用户；
- 验证 target/history offset、gold lexical ID、beam50/200 合法性、无 `-2/-1` 读取；
- 检查显存、运行时间、checkpoint identity；不训练、不选择 margin。

### P1 — fit/calibration

- 每域用确定性 hash 选择最多 1,024 个 `t=-4` fit 用户和 512 个 `t=-3` calibration 用户；
- 先完成两域 fit/calibration 并冻结 checkpoint、feature schema 与 margin；
- 任一域失败则停止，不读取 validation。

### P2 — one-shot validation

- 复用 BW1 的固定 512 用户 fresh beam50/200；
- 每域只执行一次冻结 admission gate；不补跑、不改 margin、不调权重。

## 7. Gates

P2 跨域科学门：

1. Toys、Beauty Hit@10 delta 均 `>=0`；
2. 至少一域 Hit@10 delta `>=+0.002`，两域 mean delta `>=+0.001`；
3. 两域 NDCG@10 delta `>=-0.001`；
4. 两域 tail Hit@10 均不退化；
5. expansion admissions 非零且 target promotion 数不少于 regression；
6. base fallback identity、checkpoint SHA、finite、test/Sports 禁读全部通过。

PASS 才考虑正式宽候选方法；FAIL 则说明现有 GRAM/item-head 特征不足以利用覆盖 headroom，下一方向应
回到生成模型训练或候选表征，而不是继续增加 beam 或在 validation 上调 gate。
