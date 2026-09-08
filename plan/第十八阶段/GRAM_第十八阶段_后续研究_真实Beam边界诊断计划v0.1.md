# 第十八阶段后续研究：真实 Beam 边界与继续训练退化诊断计划 v0.1

## Material Passport

- 日期：2026-09-08。
- 状态：`RESEARCH_DESIGN_READY_NOT_EXECUTED`。CPU 回顾性审计已经完成；本文件所述新追踪与梯度诊断尚未实现或运行。
- 依据：研究者“那你继续研究吧”；[本轮研究报告](../../report/第十八阶段/Stage18_S18-2_失败后机制复核与文献研究.md)。
- 性质：独立机制诊断，不是 S18-2 correction/retry，不是 S18-3，不改变原失败判定。
- 本阶段产出应是机制可辨别性和测量可信度的结论，不能是 accuracy success 或新方法确认。

## 1. 为什么先做这一步

已有 CPU 证据：Toys I0 的 frozen parent Hit@50=0.166，而 C0/M0=0.128/0.133；旧 actual-pruner 指标只覆盖最终 beam 的同父 sibling。继续用原指标证明“可作用”然后训练新 loss，可能继续混淆全局剪枝与局部排名。

本次只回答两个问题：

1. 原 PCPS 关注的 sibling/path 与真实剪枝当步边界之间，是否存在可测量的缺口？
2. 在 frozen parent 上，既有 CE 和 PCPS 的更新方向是否有改善下一 offset 真实边界的迹象；共同 continuation 退化是否需要优先处理？

备选训练思想是同深度的跨父前缀累计分数比较。它继承 BSO、BEAR §3.1 与 APAO 的相关思想，本计划不声称这一思想具有独立新意，也不立即构造新的 CF 组合方法。

## 2. 固定数据与模型边界

- 使用 S18-2 已冻结的 Toys/Beauty I0 parent 与 item-head，读取前验证原 SHA。
- 每域使用原 S18-2 cohort 固定前 100 用户的 training transition，不按命中或 first-drop 筛人；两域共 200 个训练输入。
- 使用同一前 100 名 Toys 用户的已消费 I0 evaluation 输入做只读迁移诊断。Beauty 不补跑 treatment probe，也不打开新的 confirmation fold。
- user/transition/cohort/config/source SHA 在新追踪前保存。所有 100 用户都进入分母，失效样本给出明确原因。
- `model.eval()`，无 optimizer、无参数更新、无新 checkpoint。权重在前后核验一致。
- beam 固定 50；lexical ID、Trie、EOS、length penalty、输入构造与原生 parent 一致。无需新生成 beam200，负例复用既有缓存。
- I1/I2、D1/D2、official validation/test、Sports 继续封存。

## 3. 诊断 A：记录真实剪枝过程

实现被动 tracer，记录原生 generation 的 logits processor 与 BeamSearchScorer.process 输入/输出，计算诊断后原样返回，不改 logits、候选或排序。严禁先根据最终 beam item 倒推一个集合，然后称其为真实 frontier。

每用户/每解码深度保存：

- 输入 active prefixes、对应累计分数、有效/填充 beam 标识；
- 目标父前缀是否仍在 active beam；
- 目标 token 在其父前缀合法 child 内的 rank 与合法 child 数；
- 扩展目标的原生累计分数；所有合法扩展中严格高于目标及与目标同分的数量；
- top2B 候选及原生 scorer 保留的 B 个 non-EOS active prefixes、EOS 接收记录；
- 目标是否实际保留，首次非 EOS 丢失深度，最末保留 non-EOS 前缀及其父分支；
- 旧 final-sibling proxy 对上述真实竞争前缀的覆盖；先按 prefix 去重，item 数另报，不能混合分母。

主要诊断量：局部 top50 已满足但实际被淘汰的比例；同父/跨父边界候选比例；目标相对真实保留边界的累计分数差；旧代理集合的覆盖率与漏掉的竞争类型。Toys/Beauty 分开报告，训练 transition 与 Toys I0 evaluation 分开报告。

实际 scorer 的保留结果为判断真值。遇到 EOS、相同分数、beam 填充或候选不足，必须单列；不能用简化的“第 50 名分数”公式覆盖所有情况。目标父前缀已丢失后的深度不能再次作为独立 first-drop 事件。

## 4. 诊断 B：继续训练与 CF 增量的梯度方向

先做静态样本/配置审计：报告 parent 的全部可见 transition 训练与 S18-2 每用户最后一个 transition 的差异，以及有效 batch 128→16、AdamW/调度重置。检查 native train batch 与 S18-2 batch 在同一个 history/target 上的 identity；训练样本分布改变和输入拼装错误需要分开处理。

在 frozen parent 上分别计算同一 100 用户训练集合的平均 CE、generic、真实 CF、shuffled CF 梯度。辅助权重只使用原冻结 alpha=0.1，不新增 alpha 候选，不执行 optimizer step。

若诊断 A 的分数复核可信，针对固定 parent frontier 定义边界差值 `m = score(target_prefix) - score(boundary_prefix)`，报告：

```text
g_L = mean training gradient of the frozen objective
directional_change = - dot(gradient_of_margin_on_evaluation_input, g_L)
```

对 Toys 已消费 I0 evaluation 输入测量此方向；它与构造训练梯度的 transition 不同。共同报告原始方向导数和单位范数方向，区分 CE 主方向与辅助增量，保留 generic/shuffled 对照。

这是一阶局部 SGD 方向诊断，不是实际 AdamW 更新效果，不是有限步长收益保证，也不是跨域泛化确认。若拿 `-m` 自身的梯度证明同一个 m 会改善，结论是数学恒等式，不能作为机制通过；必须使用独立输入或对照揭示额外信息。

teacher-forcing 与原生生成分数需独立报告残差；已见缓存残差最大约 0.001015，不能宣称 exact。计算梯度所用可微分数与原生分数在 near-tie 附近不一致时，保留原生剪枝真值并把方向诊断标为不确定，不靠改 epsilon 改变 Gate。

## 5. 必须先通过的工程验证

这些是新 tracer 的验收条件，不是已完成测试的描述：

1. 无 tracer 与被动 tracer：固定输入的 beam50 item 顺序、序列分数和参数 SHA 完全一致；若不同先修 tracer，不能解释成机制变化。
2. 用已知跨父竞争的 synthetic beam 验证：局部第 1 名仍会被别的父分支挤出；测试同时覆盖 EOS、同分、单 child、变长路径和初始无效 beam。
3. 记录的 scorer 输出可逐步重建 actual active set；最终候选与原生生成一致。
4. 原有训练/评估 cutoff 合同、source SHA、negative legality、alpha-zero、CF shuffle identity 持续成立。
5. 任一方向导数存在 NaN/Inf 或评分残差无法解释，梯度部分记为 `MEASUREMENT_UNRESOLVED`，不得启动训练。

## 6. 决策规则与停点

本诊断不设 NDCG 晋级线，不替换旧 S18-2 阈值。

| 观察 | 结论与下一行动 |
|---|---|
| tracer identity 失败或真实事件无法重建 | 修测量实现；不做科学判断 |
| 每域有效 first-drop 事件不足 30 | 只作描述，记证据不足；不按结果挑更多用户补足 |
| 真实边界与旧 proxy 基本一致，CF 与对照也无有意义差异 | 缺少不同机制依据，关闭该重设计方向 |
| 存在跨父竞争缺口，但 CE/PCPS 对下一 offset 方向共同不利 | 优先另立 continuation 稳定性协议；不把 frontier loss 当作现成解法 |
| 存在缺口且真实 CF 在独立输入上有一致额外方向信息 | 可以设计新的小训练实验；仍需 frozen parent、matched CE、generic、CF、shuffled 五个参照及独立确认 |
| 只有 generic 方向更好或 CF 不增加信息 | 只能作为 generic 方法迁移的依据，不包装成新的 CF 方法 |

“一致”“有意义”在这里是研究讨论用语，不是自动晋级数值 Gate。若决定进入训练，需要在任何新 treatment 结果前另行冻结具体目标、预算、数值门槛和确认样本。只有恢复到 frozen parent 水平不能作为 accuracy success。

## 7. 资源与执行方式

预计单卡可以完成，但尚未对新 tracer 实测。参考旧 S18-2 smoke：workload reserved 峰值约 6.41 GiB；新实现先以最长历史样本做 smoke，使用实测 peak 加 1 GiB 判断启动余量，不把旧数值当成新实验保证。

沿用单卡小诊断规则，排除物理 GPU1/GPU4 及已有预约/占位卡；不抢占现有进程。若预计超过 10 分钟，使用独立具名 tmux、status/heartbeat、日志及 2 小时 hard timeout；不实时盯跑，不自动重试，不启动 occupancy。

建议新产物目录 `artifacts/phase18/beam_boundary_diagnostic/run-0001/`，与 canonical S18-2 完全分开。本文件不包含虚构的启动命令：tracer/runner 尚待实现、合同与 smoke 验证。当前已完成的是 CPU 研究与设计，不是该诊断的 GPU 执行。
