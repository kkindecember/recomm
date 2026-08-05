# GRAM 第十一阶段 BW2：锚定扩展候选校准计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run + validate
- Origin Date: 2026-08-04
- Verification Status: PREREGISTERED
- Experiment ID: `GRAM_PHASE11_BW2_ANCHORED_EXPANSION_VALIDATION_V1`

## 假设

BW1 中 beam200 的候选召回比 beam50 高约 12–13pp，但 top10 没有改善。一个可检验原因是：
PCRF 对整个候选组分别标准化，width 变化会改变 seq / CF / popularity 的尺度，导致扩展候选无法与
原 top50 稳定比较，并在 Beauty 产生轻微退化。

## 冻结方法

直接复用 BW1 的 fresh beam50 / beam200、相同 512 validation 用户和冻结 item-head。不得重新解码、
训练或修改 PCRF 参数。

对每个 beam200 用户：

1. 以该 beam 的前 50 个候选为 anchor；
2. seq、CF、log-popularity 均只用 anchor50 的均值和标准差进行变换，但将该变换应用到全部 200；
3. `adjusted = z_anchor(CF) - 0.5*z_anchor(log-pop)`，其二次标准化同样只由 anchor50 决定；
4. reliability 仍仅由 sequence top10 的 tail mass 计算，`gamma=1`；
5. `joint = z_anchor(seq) + reliability*z_anchor(adjusted)`，stable descending sort。

## 完整性门控

- beam200 前 50 与独立 beam50 的平均候选集合 overlap 至少 `0.98`；
- 两域全部用户候选合法、分数 finite，checkpoint SHA 与 BW1 一致；
- anchored 公式在前 50 候选上的分数/排序与普通 width50 PCRF 数学一致；
- test / Sports 不读取。

## 科学门控

相对各域独立 fresh beam50 PCRF：

1. Toys、Beauty anchored beam200 Hit@10 均不下降；
2. 至少一个域 Hit@10 提升至少 `+0.002`；
3. 两域 NDCG@10 delta 均至少 `-0.001`；
4. 至少一个扩展候选进入 top10，证明方法实际使用扩展集合。

全部通过才进入正式宽候选校准；否则判定无参数 anchored normalization 不足，下一步转向使用
train-prefix pseudo-future 训练 expansion admission gate，不在 validation 上调 margin 或权重。
