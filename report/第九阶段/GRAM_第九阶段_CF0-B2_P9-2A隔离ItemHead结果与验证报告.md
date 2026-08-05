# GRAM 第九阶段：CF0-B2 P9-2A 隔离 Item-head 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run → analyze
- Origin Date: 2026-08-04
- Verification Status: `COMPLETED_GATE_PASSED`
- Experiment ID: `GRAM_PHASE9_CF0_B2_TOYS_ITEM_P2A_V1`
- Scope: Toys validation, seed 2023, isolated collaborative item branch
- Excluded: test, Sports, Beauty, GRAM loading, generation/fusion training

## 1. Executive conclusion

P9-2A 完成 10/10 epochs，预注册科学门三项全部通过。最佳 epoch 10 的 Recall@10 为
`0.090150`，是 popularity baseline 的 7.32 倍；Recall@50 为 `0.174634`，是 baseline
的 5.00 倍。tail 和 middle 也学到非零信号，合并 non-head Recall@50 为 `0.140049`。

因此，当前证据否定“原始 item 序列本身没有可用协同信号”这个解释。CF0-B v1 的负结果
更像是联合优化和融合设计问题，而不是机制原理失败。

## 2. 执行与完整性

| 项目 | 实际值 |
|---|---:|
| train samples | 109,361 |
| validation samples | 19,412 |
| catalog size | 11,924 |
| epochs | 10/10 |
| best epoch | 10 |
| wall time | 76.22 s |
| peak allocated | 1,109.84 MiB |
| peak reserved | 1,564 MiB |
| finite loss/gradient | 是 |
| test / Sports read | false / false |
| checkpoint / ranks / summary | 完整 |
| CodeLlama restoration | 已恢复 |

GPU smoke 先以与正式实验相同 batch 完成，随后才启动全量训练。正式 summary 、best checkpoint
和 19,412 条 validation ranks 均已落盘。

## 3. 预注册科学门

| 检查 | 门槛 | 观测 | 结果 |
|---|---:|---:|---|
| Recall@10 | 0.014774 | 0.090150 | PASS |
| Recall@50 | 0.041912 | 0.174634 | PASS |
| non-head Recall@50 | 0.005000 | 0.140049 | PASS |

non-head 按 tail 与 middle 的 validation 样本数加权合并，不包括 head。门槛和 baseline 在正式运行前
已写入冻结配置，未根据结果调整。

## 4. 主要排序结果

| 指标 | Popularity | CF0-B2 isolated | 倍率 |
|---|---:|---:|---:|
| Recall@10 | 0.012312 | 0.090150 | 7.32× |
| Recall@50 | 0.034927 | 0.174634 | 5.00× |
| NDCG@10 | 0.006427 | 0.055361 | 8.61× |

其他 best-validation 指标：MRR `0.050271`，Recall@5 `0.066454`，Recall@20 `0.121574`，
NDCG@50 `0.073678`，median rank `1353`。

CF0-B v1 item-head 的 Recall@10/50 分别为 `0.009376/0.032403`，均低于 popularity。
B2 在去除 generation loss、隔离共享参数并使用 normalized cosine objective 后，超过了这两个对照。

## 5. 分层结果

### 5.1 Target popularity

| 分层 | count | Recall@10 | Recall@50 | NDCG@10 |
|---|---:|---:|---:|---:|
| tail | 5,160 | 0.035659 | 0.076357 | 0.020652 |
| middle | 9,235 | 0.088035 | 0.175636 | 0.055415 |
| head | 5,017 | 0.150090 | 0.273869 | 0.090963 |

tail 明显弱于 head，但已不再是 v1 的 Recall@50=0。这表明 item 分支并非只复制一个纯热度排序器。

### 5.2 History length

| 历史长度 | count | Recall@10 | Recall@50 | NDCG@10 |
|---|---:|---:|---:|---:|
| 1–5 | 12,673 | 0.093585 | 0.167206 | 0.058790 |
| 6–10 | 4,319 | 0.084279 | 0.167631 | 0.050159 |
| 11–20 | 2,420 | 0.082645 | 0.226033 | 0.046691 |

长历史组的 Recall@10 略低，但 Recall@50 更高；它不构成完全冷启动或仅短历史有效的迹象。

## 6. 训练轨迹与解释

Recall@10 从 epoch 1 的 `0.012518` 持续提升到 epoch 8 的 `0.089893`，并在 epoch 10
达到 `0.090150`。Recall@50 在 epoch 6–10 稳定于约 `0.174–0.176`。训练末期没有数值发散，
最佳 epoch 是事先约定的 lexicographic Recall@10/NDCG@10 准则所选。

与 v1 诊断联合解读：

1. 协同序列信号本身可学；
2. v1 中 generation gradient 对 CF Transformer 约为 item gradient 的 26 倍，足以改写协同表示；
3. tied lm-head/shared embedding 导致“只解冻 lm-head”实际污染共享 embedding；
4. post-hoc LayerNorm 不是恒等融合，在 v1 中已对 teacher-forced NLL 造成明显伤害。

因此，最合理的下一个问题是“一个已学好且固定的 item state 能否以初始严格恒等的方式安全注入
GRAM”，而不是继续否定协同机制。

## 7. 决策与边界

P9-2A 结论：`eligible_for_separately_authorized_safe_fusion`。

这不等于已证明融合能提高生成指标，也不等于多 seed 结论。下一阶段应单独预注册 P9-2B：
zero-init residual，冻结全部 GRAM/shared/lm-head 和 item checkpoint，先过 identity 与 teacher-forced NLL
非劣化门，再做 beam validation。本轮没有启动 P9-2B。

## 8. 可复核产物

- 冻结计划：`plan/第九阶段/GRAM_第九阶段_CF0-B2协同分支隔离训练与安全融合计划.md`
- 冻结配置：`artifacts/phase9/configs/cf0_b2_toys_item_p2a_preregistered.json`
- 正式汇总：`artifacts/phase9/cf0_b2_toys_item_p2a/summary.json`
- best checkpoint：`artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt`
- best ranks：`artifacts/phase9/cf0_b2_toys_item_p2a/best_validation_ranks.tsv`
- 运行日志/遥测：`artifacts/phase9/cf0_b2_toys_item_p2a/run.log` 与 `gpu_telemetry.csv`
