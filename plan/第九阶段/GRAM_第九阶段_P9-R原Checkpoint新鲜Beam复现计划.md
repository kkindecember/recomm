# GRAM 第九阶段 P9-R：原 checkpoint 新鲜 Beam 复现计划

## 目的

第九阶段已经在 Toys 独立 test 上确认冻结 PCRF，但其候选来自历史缓存。本实验只回答一个工程与机制复现问题：在不训练、不调参、不读取 test 的条件下，从原 Toys epoch-30 checkpoint 重新生成 beam，冻结 PCRF 的排序结果是否保持一致。

本实验不是新的提分搜索，也不改变第九阶段独立 test 结论。第十阶段 CF1 的失败只说明候选扩展没有超过 PCRF，不构成本实验的否定证据。

## 冻结项

- checkpoint：`GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt`
- 历史 validation cache：`GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv`
- item head：`artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt`
- split：Toys validation；`test_read=false`
- 用户：按 `sha256("2023:" + user_id)` 排序后的前 512 个用户
- constrained decoding：全商品 trie、50 beams、50 return sequences、length penalty 1.0
- PCRF：`lambda=1.0, beta=0.5, gamma=1.0`，参数不再选择
- seed：2023；单卡；不训练

## 主门控

同时满足才允许进入 Beauty 外部确认：

1. 512/512 用户均生成 50 个唯一、可映射、有限分数的合法候选；
2. 历史分数与 fresh 分数在共同候选上的 Pearson、Spearman 均不低于 0.995；
3. 历史与 fresh 的 PCRF top-10 集合平均重合率不低于 0.98；
4. fresh baseline Hit@10 与历史 subset baseline Hit@10 的绝对差不超过 0.001；
5. fresh PCRF Hit@10 与历史 subset PCRF Hit@10 的绝对差不超过 0.001。

候选集合重合率、sequence top-10 重合率、NDCG@10 和 PCRF 相对 baseline 的增量作为诊断输出，不用于事后改门。

## 产物

- `artifacts/phase9/p9r_toys_fresh_beam_512/fresh_beams.tsv`
- `artifacts/phase9/p9r_toys_fresh_beam_512/per_user.tsv`
- `artifacts/phase9/p9r_toys_fresh_beam_512/summary.json`
- `artifacts/phase9/p9r_toys_fresh_beam_512/status.json`
- `artifacts/phase9/p9r_toys_fresh_beam_512/run.log`

## 决策

- PASS：第九阶段机制与原解码链路可复现，下一步只做 Beauty validation 校准与一次冻结 test 确认。
- FAIL：保留第九阶段已完成的 Toys test 结论，但暂不外推；先按候选集合、分数相关性和排序差异定位代码或环境漂移，禁止自动重跑和改阈值。
