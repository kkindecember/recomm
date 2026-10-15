# SASRec与ItemCF保留价值对照

状态：COMPLETED；两组、两域真实生成及排名核验完成。

Origin Skill：academic-research-suite。共用已核验的完整组合基线；上游冻结、每组融合器独立拟合4轮。

| 域 | 组 | 真实生成 N10 | H10 | H50 | 后重排 N10 |
|---|---|---:|---:|---:|---:|
| Toys | 完整组合（共享） | 0.069174 | 0.121094 | 0.187500 | 0.069347 |
| Toys | without_sasrec | 0.066979 | 0.107422 | 0.179688 | 0.067912 |
| Toys | without_itemcf | 0.066745 | 0.115234 | 0.185547 | 0.067557 |
| Beauty | 完整组合（共享） | 0.081197 | 0.125000 | 0.244141 | 0.078102 |
| Beauty | without_sasrec | 0.075199 | 0.123047 | 0.244141 | 0.074236 |
| Beauty | without_itemcf | 0.080655 | 0.126953 | 0.248047 | 0.076663 |

| 域 | 完整组减去删减组 | ΔN10 | 描述性95%区间 | 前十新增/丢失 |
|---|---|---:|---|---|
| Toys | without_sasrec | +0.002195 | [-0.003412, +0.008097] | 10/3 |
| Toys | without_itemcf | +0.002429 | [+0.000185, +0.005246] | 3/0 |
| Beauty | without_sasrec | +0.005998 | [-0.002243, +0.015134] | 9/8 |
| Beauty | without_itemcf | +0.000542 | [-0.002067, +0.003280] | 1/2 |

预设逐域方向判断：

```json
{
  "without_sasrec": {
    "Toys": "RETENTION_DIRECTION",
    "Beauty": "RETENTION_DIRECTION"
  },
  "without_itemcf": {
    "Toys": "RETENTION_DIRECTION",
    "Beauty": "INCONCLUSIVE_OR_TRADEOFF"
  }
}
```

RETENTION_DIRECTION：该域支持保留方向；SIMPLIFICATION_DIRECTION：该域支持进一步验证删减；INCONCLUSIVE_OR_TRADEOFF：证据不足或指标取舍。它们不是显著性或最终方法判定。

去SASRec只删除外部评分；GRAM内部既有协同近邻仍保留。两组均屏蔽移除专家的评分和四个统计特征，并重新学习其余融合权重。

移除结果是相对其余组件的条件贡献；不能从两个单独删减实验推断可同时删掉两个专家。当前流程复用旧评分缓存，计时不是最终部署节省。

一个上游seed，开发池多次使用；配对区间不包括训练随机性和研究选择。无新机制、无调参搜索、无test读取。

[冻结协议](../../plan/第二十一阶段/21.3C_SASRec与ItemCF保留价值对照_20260923.md)；[完整结果](../../artifacts/phase21/other_expert_contribution_20260923_v1/summary.json)。
