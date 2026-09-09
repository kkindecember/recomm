# 第十八阶段：DIFF→GRAM 双域 full_r2 结果与方向结论

**2026-09-09 13:28 后续执行更新：** 用户已授权 Beauty 单域固定日程复核，见 [新执行计划](../../plan/第十八阶段/GRAM_第十八阶段_DIFF_Beauty固定日程复核计划v0.1.md)。新任务从原始 GRAM parent 重新初始化，固定 seed 2023、1 frozen + 10 joint，已在 GPU 4 启动，状态为 `artifacts/phase18/diff_gram/beauty/confirm_v1/status.json`。这是后续投入决定，覆盖下文“当前不追加 Beauty”的先前建议；本报告的 full_r2 数值和未确认稳定收益的结论继续保留，新复核尚无效果结果。

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / validate，inline。
- Origin Date：2026-09-09（Asia/Shanghai）；Version Label：s18_diff_gram_final_v1。
- Verification Status：**ANALYZED**。本次重算保存的完整 validation 预测并核对同用户差分；未重跑训练或模型推理，不声明模型复现实验 VERIFIED。
- 结果范围：seed 2023，Toys / Beauty 的 `full_r2` 最终选中 checkpoint；两域均 COMPLETED，Beauty 于 09-09 12:53:47 完成。
- 本轮新增：CPU 分析；训练 0、模型推理 0、test 读取 0；历史 GRAM / PCRF 缓存复用。

## 1. 方向结论

**Beauty 得到小幅正向点估计，Toys 接近持平且略降；目前没有确认稳定的双域提点。** Beauty NDCG@10 为 **0.065354591**，相比 GRAM 的 0.064973703，绝对增加 **0.000380888**，相对 **+0.5862%**；Hit@10 净多命中 **23** 人，Hit@50 净多 **14** 人。不能再将两域一概写成“完全没有提升”。

但 Beauty 的 NDCG@10 配对 95% 区间为 **[-0.000133315, +0.000910499]**，包含 0；Hit@10 / Hit@50 的区间也包含 0。单 seed、子集选 checkpoint 和缺少匹配 GRAM 续训对照，使这个结果尚不足以证明稳定收益，更不能单独归因于 DIFF 结构。区间含 0 同样不等于证明真实增益为零。

**建议将 Beauty 保留为弱正信号备选，Toys 记为未取得提点；当前不追加 DIFF→GRAM 长训，主线继续已有 ETEGRec 筛选。** 若之后需要检验与 PCRF 的组合，Beauty 可以作为有限成本复查候选；本报告没有执行组合，不能据此宣称叠加有效或无效。

此路线是 DIFF 的属性 / 序列融合机制迁入 GRAM，与扩散生成模型 DiffGRM 不同。后者结论见 [DiffGRM 最终报告](Stage18_DiffGRM_双域full_r2结果与方向结论报告.md)。

## 2. 实际实现与比较边界

本地 DIFF 来源固定为作者代码 commit `ef6283a3bf9ed242a9b867fcf63368f0fb36eb33`。迁移保留频域过滤、item / 属性的 intermediate fusion、early fusion 双路径和表示对齐；新分支输出有效历史状态，投影到 T5 hidden 后追加到 GRAM encoder memory，由原词法 decoder 的 cross-attention 读取。

训练目标为 `L_generation + 0.1 × (L_item_CE + 100 × L_alignment)`；item CE 是辅助训练项，最终推荐仍由 GRAM 词法生成得到，不是将辅助分类分数作为重排结果。保留原 GRAM / FiD 和词法 ID。这是结构迁移，不能称为作者 DIFF 原系统完整复现，也不能与作者从头训练轮数直接等价比较。

| 设置 | Toys | Beauty |
| --- | --- | --- |
| 训练转移 / validation 用户 | 109,361 / 19,412 | 131,413 / 22,363 |
| 目录商品数 | 11,924 | 12,101 |
| 属性 | categories；brand 全缺失，未使用 | categories + brand |
| 新分支 hidden / FFN / layers / heads | 256 / 256 / 2 / 2 | 256 / 256 / 2 / 2 |
| 历史长度 / dropout | 20 / 0.5 | 20 / 0.5 |
| 频率 cutoff / 双路径 alpha | 9 / 0.5 | 5 / 0.3 |
| 父 GRAM checkpoint | 原 Toys epoch 30 | 原 Beauty epoch 25 |
| 新增参数 / 总参数 | 5,358,478 / 65,875,854 | 6,410,672 / 66,928,048 |
| 新参数 / GRAM 学习率 | 0.001 / 0.00001 | 0.001 / 0.00001 |
| 有效 batch / 精度 / beam | 128 / FP32 / 50 | 128 / FP32 / 50 |

Beauty 属性更丰富，且 cutoff / alpha 与 Toys 不同。因此两域表现差别与属性利用的解释相容，但没有属性消融支持“Beauty 的收益就是 brand 带来的”这一因果判断。

代码：[diff_sequence.py](../../experiment/phase18/core/diff_sequence.py)、[diff_gram.py](../../experiment/phase18/core/diff_gram.py)。运行冻结配置、父 checkpoint SHA 和实现文件 SHA 见 [Toys manifest](../../artifacts/phase18/diff_gram/train_v1/manifest.json)、[Beauty manifest](../../artifacts/phase18/diff_gram/beauty/train_v1/manifest.json)。

## 3. 最终完整 validation 结果

| 数据集 / 模型 | NDCG@5 | NDCG@10 | NDCG@50 | Hit@5 | Hit@10 | Hit@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toys GRAM | 0.067135 | 0.076275 | 0.096532 | 0.090923 | 0.119411 | 0.211931 |
| Toys GRAM + PCRF | 0.068508 | 0.078716 | 0.098071 | 0.093653 | 0.125335 | 0.211931 |
| Toys DIFF→GRAM，epoch 11 | 0.066874 | 0.076092 | 0.096315 | 0.090460 | 0.119153 | 0.211622 |
| Beauty GRAM | 0.054917 | 0.064974 | 0.086701 | 0.077628 | 0.108751 | 0.208112 |
| Beauty GRAM + PCRF | 0.057534 | 0.067908 | 0.088760 | 0.081608 | 0.113849 | 0.208112 |
| Beauty DIFF→GRAM，epoch 9 | 0.055264 | 0.065355 | 0.086916 | 0.078388 | 0.109780 | 0.208738 |

相对同用户 GRAM 的差分与配对区间：

| 数据集 | 指标 | 绝对增量 | 95% 区间 | 命中用户净变化 |
| --- | --- | ---: | --- | ---: |
| Toys | NDCG@10 | -0.000182258 | [-0.000589228, +0.000213625] | — |
| Toys | Hit@10 | -0.000257573 | [-0.001236349, +0.000772718] | -5 |
| Toys | Hit@50 | -0.000309087 | [-0.001442407, +0.000824232] | -6 |
| Beauty | NDCG@10 | +0.000380888 | [-0.000133315, +0.000910499] | — |
| Beauty | Hit@10 | +0.001028485 | [-0.000268300, +0.002280553] | +23 |
| Beauty | Hit@50 | +0.000626034 | [-0.000983768, +0.002101686] | +14 |

Toys NDCG@10 相对变化 **-0.2389%**；Hit@10 从 2,318 变为 2,313 人，新命中 49、丢失 54。Beauty Hit@10 从 2,432 变为 2,455 人，新命中 110、丢失 87；Hit@50 从 4,654 变为 4,668 人，新命中 159、丢失 145。两域所有用户都返回 50 个合法且唯一的 item，没有通过丢弃短列表用户抬高分数。

Beauty 六项汇总指标均为正增量；Toys 六项均略负。Beauty NDCG@10 绝对变化折成百分数分数是 **+0.03809 个百分点**，相对提升才是 **+0.5862%**，二者不要混写。

当前 Beauty DIFF→GRAM 仍低于历史 GRAM + PCRF：NDCG@10 差 **-0.002553091**。这是候选独立模型相对已有组合系统的差距；**它不能回答 DIFF→GRAM + PCRF 能否超过 GRAM + PCRF**。本轮没有运行前者，结果应记为“未评估”。

## 4. 子集与完整集合是否一致

模型按固定 2,000 用户子集选择，随后只对选中 checkpoint 做本轮完整 validation。以下子集拆分从最终完整预测按相同用户重算，不额外推理模型。

| 数据集 / 人群 | 用户数 | DIFF→GRAM NDCG@10 | 相对同组 GRAM 的增量 | Hit@10 净变化 |
| --- | ---: | ---: | ---: | ---: |
| Toys 选模子集 | 2,000 | 0.069738526 | +0.000367961 | +1 |
| Toys 子集之外 | 17,412 | 0.076822067 | -0.000245457 | -6 |
| Beauty 选模子集 | 2,000 | 0.076745231 | +0.000393730 | +6 |
| Beauty 子集之外 | 20,363 | 0.064235832 | +0.000379627 | +17 |

Toys 的子集小幅上涨没有延续到完整人群；不能拿子集值宣称总体提点。Beauty 的小幅正增量在子集之外也存在，**不是只有选模子集上涨**。但子集之外依然属于已有开发 validation，不能改称新鲜独立 test 或另一次训练复现。

NDCG@10 相对 GRAM 的分组增量如下。分组阈值沿用历史缓存，Toys tail frequency ≤ 5，Beauty ≤ 6；分组是解释性描述，没有按结果重新调阈值。

| 分组 | Toys 人数 / 增量 | Beauty 人数 / 增量 |
| --- | --- | --- |
| 历史 1–5 | 12,673 / +0.000106681 | 14,063 / +0.000337161 |
| 历史 6–10 | 4,319 / -0.000736574 | 5,311 / +0.000282023 |
| 历史 11–20 | 2,420 / -0.000706069 | 2,989 / +0.000762290 |
| Tail | 5,160 / +0.000390336 | 6,049 / +0.000410799 |
| Non-tail | 14,252 / -0.000389568 | 16,314 / +0.000369798 |

Beauty 在上述分组中点估计均正，值得如实保留；这些微小组内差异未做组间显著性检验，不能据此声称长历史用户一定受益更多。Toys 有正负取舍，整体略负，不能只报告 tail 的正值。

## 5. 训练预算与是否应继续跑

| 数据集 | 接续源 epoch | full_r2 新增 epoch | 总训练 | 选中 checkpoint | optimizer steps | 停止原因 | full_r2 墙钟 / 其中最终完整验证 |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| Toys | 2 | 9 | 1 frozen + 10 joint = 11 | epoch 11 / stage 9 | 9,405 | epoch_limit | 18.20 / 2.91 小时 |
| Beauty | 3 | 8 | 1 frozen + 10 joint = 11 | epoch 9 / stage 6 | 11,297 | epoch_limit | 19.28 / 4.14 小时 |

两条均跑到本地批准上限，最后两组学习率均为 0。Toys 于 09-09 10:30 完成，Beauty 于 12:53 完成。上述时间是 `full_r2` 本身的墙钟，包含接续训练、子集和完整验证，不包含此前所有准备 / 短分支，也不是纯 GPU 计算时长。

接续保留原先短日程模型、Adam 与 RNG，一个 epoch 学习率桥接后回到长曲线。由此得到的是“保留短训练前缀的完整本地预算”，不是从头连续运行作者原始日程，也不是与 GRAM 纯续训完全匹配的结构归因对照。

| 绝对 epoch | Toys 子集 NDCG@10 | Beauty 子集 NDCG@10 |
| --- | ---: | ---: |
| 2 | 0.069196810 | — |
| 3 | 0.068403626 | 0.076190903 |
| 5 | 0.068704315 | 0.076309922 |
| 7 | 0.069260604 | 0.076361322 |
| 9 | 0.069649344 | **0.076745231** |
| 11 | **0.069738526** | 0.076396171 |

Toys 后半程子集仍略涨，不能声称所有更长训练都不会改善；但完整验证没有显示收益。Beauty 在 epoch 9 选优后 epoch 11 回落，不能用“还在持续上涨”作为继续加预算的理由。**当前缺的是可重复、可归因的提点证据，不能把它简化为“只要多跑几轮就能解决”。**

## 6. 统计解释与完整性

配对 bootstrap 以用户为单位，2,000 次有放回重采样，seed 2023，取 percentile 95% 区间。共享复核有 4 候选 × 2 参照 × 3 指标，共 **24 个未作多重比较校正的区间**；本报告对 GRAM 的六个区间全部含 0，不作显著性发现声明。它们只描述固定 seed / checkpoint 下用户差分的不确定性，不覆盖模型选择、训练 seed 和共享商品依赖。

完整性检查已通过：四条最终预测的 SHA 与 metadata / result 一致；validation 用户集合、当前 gold、目录 item 合法性及去重正确；六项指标与保存汇总在 1e-12 内一致。历史 GRAM / PCRF rank 缓存 SHA、基线指标、历史长度及数据来源链相符。历史 rank TSV 本身不含 gold，因此历史目标身份依赖先前冻结的缓存验证和来源记录，本轮不冒称重新独立核验历史原始目标。

统计误读检查 **11 / 11**：

| 检查项 | 本次处理与剩余边界 |
| --- | --- |
| Simpson 悖论 | 报告整体与分组；Beauty 所列组点估计均正，Toys 存在真实正负取舍。 |
| 生态谬误 | 均值改善不等于每人改善；明确新命中与丢失人数。 |
| Berkson / 样本选择 | 结论限于现有目录和验证人群，不外推其他域或用户。 |
| Collider bias | 所有用户均纳入，不按候选是否命中选择分析样本。 |
| 基率忽视 | 同时报原始分数、相对增量、绝对增量及命中人数。 |
| 回归均值 | 披露子集挑最佳及子集外结果，承认全量与选模重叠。 |
| 幸存者偏差 | 两域全部用户计分，正负域完整报告。 |
| 多处寻找效应 | 报告全部主指标；区间未校正，不能按个别正值宣布稳定有效。 |
| 分析路径自由度 | 保留短日程前缀、桥接与 checkpoint 选择记录；为探索筛选。 |
| 相关与因果 | 缺少匹配 GRAM 续训、属性消融和多 seed，不将小增量归因于 DIFF 算子。 |
| 反向因果 | 按既有训练来源与 validation 分析，不用未来效果反推训练输入的因果作用。 |

## 7. 文件与后续处理

- 最终结果：[Toys full_r2](../../artifacts/phase18/diff_gram/train_v1/full_r2/result.json)、[Beauty full_r2](../../artifacts/phase18/diff_gram/beauty/train_v1/full_r2/result.json)。根目录及 `screen_r1` 的早期结果只作历史记录。
- 完整预测：[Toys epoch 11](../../artifacts/phase18/diff_gram/train_v1/full_r2/full_validation_stage_epoch_09.jsonl)、[Beauty epoch 9](../../artifacts/phase18/diff_gram/beauty/train_v1/full_r2/full_validation_stage_epoch_06.jsonl)。保存了候选 item 和 sequence scores；未来组合复查仍须核对 PCRF 所需评分口径，不能未经运行直接相加历史收益。
- CPU 复核：[完整快照](../../artifacts/phase18/final_diff_review/review_20260909.json)、[可重算脚本](../../experiment/phase18/analysis/s18_diff_final_review.py)。快照含完整精度指标、配对区间、输入哈希、分组和轨迹；脚本拒绝覆盖已有快照。
- 执行历史：[Toys 结构计划](../../plan/第十八阶段/GRAM_第十八阶段_DIFF结构迁移与Toys提点筛选计划v0.1.md)、[Beauty 补充计划](../../plan/第十八阶段/GRAM_第十八阶段_DIFF_Beauty并行筛选补充计划v0.1.md)、[完整预算补遗](../../plan/第十八阶段/GRAM_第十八阶段_四实验完整预算与自动接续执行补遗v0.1.md)。

**当前处理：Beauty 标记“弱正信号、未确认”，Toys 标记“未提点”；保留模型与候选缓存，停止追加本路线训练预算。** 已运行 ETEGRec 按自己的筛选计划推进。若回头检查组合，先固定 Beauty 当前 epoch 9 与既有 PCRF 规则，明确比较 `DIFF→GRAM + PCRF` 和 `GRAM + PCRF`，再决定是否值得做匹配续训及多 seed；本报告不将这些未执行步骤记为已完成。
