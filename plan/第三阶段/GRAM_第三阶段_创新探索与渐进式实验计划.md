# GRAM 第三阶段：创新探索与渐进式实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-22
- Verification Status: ANALYZED（既有实验结果、MARC L0）；STOPPED（MARC L1 及后续）
- Version Label: code_plan_v36_marc_l0_stopped
- Upstream Dependencies: `artifacts/phase1_beauty/REPRODUCTION_REPORT.md`; `artifacts/phase2_toys/REPRODUCTION_REPORT.md`

### Amendment Record

- 2026-07-24：MARC L0 已按修复后预注册配置单次完成。首次 scoring 错把 K20
  作为 source-utility full reference；128-token 截断因此机械移除 metadata，
  semantic utility 恒为 0。该次记
  `EXECUTION_INVALID_SOURCE_REFERENCE` 并保留，不进入科学结论。修复后 source
  reference 与 matched baseline 对齐（Toys K5、Beauty K10），K20 只作为动态预算
  action；未修改 cohort、features 或 gates。有效运行中两数据集各 512
  training-prefix users，全部 lineage/target-exclusion/current-replay/Trie/finite/
  no-update/critic-convergence gate 通过。Toys collaborative negative utility rate
  为 14.0625%，低于预注册 15%，L0-A 固定失败；L0-B 又因两数据集 semantic
  corruption direction 错误、Beauty semantic active coverage/utility CI 和 budget
  regret 失败。固定决定为 **`STOP_MARC_NO_UTILITY_HETEROGENEITY`**，L1、完整
  MARC、reflection、RL 和 validation/test 均不解锁。
- 2026-07-24：用户提出扩大方法改动，探索 LLM reflection、语义/协同可信度网络、
  动态邻居数、逐生成层融合与强化学习。新增方向 L：
  **Marginal-utility Aware Reflective Controller（MARC）**。MARC 不把“反思”
  实现为昂贵的外部 LLM chain-of-thought，而以轻量 critic 预测语义/协同证据对
  当前 Trie 决策的反事实边际效用，用同一效用统一控制 source trust、CF neighbor
  budget、layer-wise injection 与可选二次 refinement。R4ec 已覆盖 LLM
  reasoning-reflection-refinement，RRCM 已覆盖 ranking-reward memory policy，
  DiscRec 已覆盖 semantic/collaborative dual branch gating，PRORec 已覆盖渐进融合；
  上述均只作为借鉴边界。AGPTD 尚未执行，现记
  `PAUSED_AGPTD_SUPERSEDED_BEFORE_EXECUTION`；当前只解锁 MARC L0 的
  training-only utility-predictability audit，不解锁 full training、validation/test
  或 RL。
- 2026-07-24：在 FFNF 因统一字段配额不可行而停止后，新增方向 K：
  **Advantage-Gated Privileged Trie Distillation（AGPTD）**。训练时 teacher 读取
  256-token 长证据，student 与部署时完全一致、只读取 current 128-token 输入；
  蒸馏仅发生在当前 textual-ID prefix 的 Trie 合法 children 上，并且只转移 teacher
  对 gold child 相比 student 确有正优势的 token。增量检索确认 privileged/context
  distillation、selective/hard-gated KD、token-wise KD、推荐 KD 和训练期 Trie scope
  mask 均已有先例，因此这些组件本身不得声称首创；只保留“GRAM 截断证据转移 +
  gold-advantage 防负迁移 + catalog-prefix 局部条件分布”的窄组合创新。当前只解锁
  training-only、0-training 的 K0-T teacher-informativeness probe；不解锁优化器更新、
  validation/test 或大规模实验。
- 2026-07-24：FFNF J0-N/J0-S 已完成。J0-N 固定为
  `TRANSFER_INNOVATION_ALLOWED_WITH_STRONG_NARROWING`，明确 two-stream、unimodal
  auxiliary loss、modality dropout、residual interaction 与 gradient-conflict
  control 均为借鉴组件。J0-S 对 Toys 11,924、Beauty 12,101 个 catalog items 完成
  exact tokenizer census；所有 catalog/parse/finite/component/128-width/no-link-
  duplication/current-replay integrity gate 通过，未加载 checkpoint、未使用 GPU、
  未读取 validation/test。Toys 的 CF coverage 为 1.0，但 META64 相对 current
  aggregate 少 234,052 tokens、gain-positive item rate 为 0；Beauty metadata
  gain 为 +345,311，但 CF coverage 仅 0.6320。固定决策为
  **`STOP_FFNF_BUDGET_INFEASIBLE`**，J1/J2 不解锁，禁止事后搜索或使用
  dataset-specific quota。
- 2026-07-24：用户确认仍以方法创新为目标。新增方向 J：
  **Field-Factorized Non-Degrading Fusion（FFNF）**。简单 two-stream 已被 EAGER，
  collaborative/metadata adaptive memory retrieval 已被 RRCM，多粒度 guided FiD
  已被 MGFiD 覆盖；因此本方向不以“双流”本身为贡献，而把新增部分固定为：
  (i) GRAM fine item passage 内 collaborative/metadata 的固定总预算字段因子化；
  (ii) 两个来源共享历史 item position、增加 field-source identity；
  (iii) 以 paired source ablation 构造 non-degradation objective，直接惩罚加入另一
  来源后的 gold-path negative interference。当前只建立 J0-N/J0-S 设计，不解锁训练
  或既有 Beauty/Toys validation。
- 2026-07-24：SMBR I0-D 已按 `smbr_i0_d_preregistered_v1` 单次完成。Toys/Beauty
  各 1200 fit + 400 calibration + 400 audit training-prefix；用户 split 零重叠，
  held-out fields 未读取，target feature inclusion 为 0，current repeat error 为 0，
  fixed-budget/raw-component/finite checks 全部为 1。两个数据集 calibration probability
  均低于冻结网格最小值 0.50，固定触发
  **`STOP_SMBR_NO_CALIBRATED_SUBSET`**。事后仅作不救援的 ranking audit：Toys audit
  AUROC 0.5172（95% CI [0.4286, 0.6037]），Beauty 0.5614
  （[0.4858, 0.6355]）；两者均不满足 point ≥ 0.60/lower > 0.50，且概率 top
  10%–40% 的 active mean benefit 在两个数据集全部为负。因此 I1/I2 不解锁，
  不因 absolute-threshold 设计缺点重开同一假设。
- 2026-07-24：完成 SMBR I0-N 差异审计，并修正 novelty gate 的解释：已有工作覆盖
  adaptive compression、decision-aware selection 或 abstention 不构成停止条件，
  允许显式借鉴并做 recommendation-specific transfer innovation。SMBR 的必要新增
  部分固定为 GRAM-native displacement 问题定义、training-only paired benefit
  supervision、target-free decision 与 current-layout identity fallback，以及新数据集
  上的机制/风险验证。固定决策为
  **`TRANSFER_INNOVATION_ALLOWED_I0_D_DESIGN_UNLOCKED`**；只解锁 I0-D 预注册设计，
  不直接解锁评分、训练、GPU 或 test。
- 2026-07-24：在 PENS 停止后完成跨 A–H 的失败模式复盘、GRAM 训练/解码代码核对和
  2026 年近邻增量检索。APAO、scope-mask loss/RecLM-cgen、TrieRec、PRO 与刚发布的
  BONSAI 已分别覆盖 prefix-aware optimization、训练期 Trie mask、Trie-aware
  representation、prefix retention 与 branching-optimized identifier；因此不再把
  “普通 Trie-aware loss / prefix survival / 降低浅层 branching factor”作为方向 I。
  新建 **方向 I / Selective Metadata Budget Recovery（SMBR）** 草案：不推翻
  CPBD 的 `STOP_CPBD_NO_NET_VALUE`，而把 D2 暴露的异质性作为新的 post-hoc 问题，
  研究能否仅用 training-only counterfactual labels 与 target-free features 学到带
  abstention 的序列化策略。当前只解锁 I0-N 原文级差异审计；I0-D、训练、GPU、
  Beauty/Toys 新 validation 分析和 test 均未解锁。若进入效果确认，Sports（或另一个
  未参与 A–H 方向生成的数据集）必须成为主要确认边界，Beauty/Toys 只能作为
  repeated-validation exploratory evidence。
- 2026-07-24：修正 median 实现后的 PENS H0-D 从头完成，CPU 单元测试 4/4
  通过；Toys/Beauty 各 256 tail-miss + 256 tail-hit，cohort、checkpoint、共享
  position table、direction preservation、equal norm、position-0 identity、current
  repeat/restore、finite 与 held-out exclusion 全部通过，未训练、未生成 beam、未读
  test，GPU3/CodeLlama 已恢复。结构 gate 再次强通过：exposure–norm Pearson 为
  -0.9740/-0.9522，`||P20||/||P1||` 为 5.0423/5.3910。但 Toys tail-miss
  norm-only gain 仅 0.00455，95% CI [-0.06709, 0.07968]、positive rate 0.4492；
  Beauty 为 -0.85199，95% CI [-0.99823, -0.70063]、positive rate 0.1484。
  tail-hit mean 亦分别为 -0.34012/-0.62216。按串行 gate 固定决策为
  **`STOP_PENS_NO_CAUSAL_BENEFIT`**；H1 及后续不解锁，不得在同一 validation 上
  改 norm target、只改后段 positions、扫缩放/clip 或用 zero-position 挽救。
- 2026-07-24：PENS H0-D 首次实现完成评分后，在结果校验中发现偶数样本 median
  语义不一致：预注册要求 positions 1–20 的通常样本中位数，即排序后第 10/11 个值
  的平均，但实现使用 `torch.median` 取较低的第 10 个值。Toys 实际 intervention
  target 为 1.7133，而正确预注册值应为 1.7624；Beauty 分别为 1.5520/1.5522。
  因此该次执行固定记为 **`EXECUTION_INVALID_MEDIAN_IMPLEMENTATION`**，其
  `STOP_PENS_NO_CAUSAL_BENEFIT` 不作科学结论。产物保留；唯一修复为用 0.5 quantile
  实现通常中位数并增加精确单元测试，随后保持 config、cohort、seed、条件、endpoint
  与 gate 不变从头重跑。
- 2026-07-24：在 CPBD `STOP_CPBD_NO_NET_VALUE` 后建立方向 H：Post-Encoder Norm
  Shortcut（PENS）。只读 checkpoint census 显示，Toys/Beauty 的 fine-passage
  position embedding 训练曝光量与范数 Pearson 相关分别为 -0.974/-0.952，位置 1
  到位置 20 的范数分别约从 0.45/0.38 增至 2.29/2.06。GRAM 在 T5 encoder 输出后
  才把该向量广播加到 passage 的每个 token，随后直接交给 decoder cross-attention。
  原文级近邻审计发现位置表示分析、additive PE 解耦、FiD passage guidance 与
  embedding-norm bias 均有先例，但在本轮固定检索簇中未检出对生成式推荐
  post-encoder passage-position embedding 做“训练曝光—范数分层 + 保方向仅改范数”
  因果诊断的工作，故固定为
  **`NOVELTY_SCOPE_PASS_WITH_STRONG_MECHANISTIC_NARROWING`**。H0-D 已在读取其
  新反事实分数前冻结：保持 checkpoint、输入、位置方向与 coarse position 不变，仅把
  fine positions 1–20 统一为 checkpoint 内中位范数，并以 zero-position 作描述性
  control；只有双数据集 tail-miss 改善且不伤 tail-hit 才允许设计 H1。
- 2026-07-24：CPBD G0-D2 按冻结协议完成，8/8 CPU 单元测试通过；Toys/Beauty
  tail-miss/tail-hit 各 256 人，cohort、coarse prompt、raw components、128-token
  budget、mask localization、matched eligibility、finite 与 current score repeat
  全部通过，重算误差为 0，未训练、未读 test，GPU3/CodeLlama 资源已恢复。
  Toys tail-miss net 为 0.06625，但 95% CI [-0.02673, 0.15351] 且 positive rate
  0.546875，未过 net gate；Beauty net 为 -0.07070，95% CI
  [-0.27081, 0.08917]，亦失败。更重要的是 tail-hit net 分别为 -0.85501 与
  -2.18000，说明固定 metadata-first 会严重破坏已有正确路径。虽然两数据集
  recovered-all contribution 的 CI 均为正（0.01575/0.02128），其量级不足以抵消
  layout/CF visibility 损失；slice8 也未双数据集达标。按串行规则固定决策为
  **`STOP_CPBD_NO_NET_VALUE`**，G1 及后续不解锁，不得在同一 validation 上改
  threshold、只挑 tail miss、扫字段顺序或做条件 gate 挽救。
- 2026-07-24：CPBD G0-D1 完成，CPU 单元测试 5/5 通过；Toys/Beauty 分别完整
  census 11,924/12,101 个 catalog items，catalog intersection、parse、finite、
  component identity 与 exact Collator replay 均为 1.0，未加载 checkpoint、未评分、
  未训练、未用 GPU、未读 validation/test 效果。Toys 可恢复至少 8 个 metadata
  token 的 item 比例为 0.7642，中位恢复 33，current metadata retention 中位数
  0.6562；Beauty 分别为 0.9998、83 与 0.2742，双数据集全部机制 gate 通过，固定
  决策为 **`G0_D2_DESIGN_ALLOWED`**。随后在读取新 checkpoint 分数前冻结 G0-D2：
  用 current、metadata-first、minus-all-recovered、recovered-slice8 与 matched
  visible-slice8，在固定 checkpoint/cohort/budget/CF identity 下检验恢复内容是否
  真正提高 gold lexical path。
- 2026-07-24：G0-D1 的首次测试使用 base Python 3.13，因环境无 PyTorch 在导入
  GRAM Collator 时退出；第二次使用正确 `gram-repro` 环境，但项目内 Hugging Face
  cache 未显式指定而在加载 tokenizer 前退出。显式使用既有项目 cache 后 5/5 测试
  通过。第一次全量 census 在写任何结果前因把 `encode_texts_split` 的 tuple 返回值
  当作 dict 而退出；只修正返回值解包与动态宽度比较后从头重跑，未改变数据、构造、
  指标、seed 或门槛。
- 2026-07-24：CPBD G0-N 在读取任何新 raw/visible/lost token census 前完成。GRAM、
  LBR、LLMLingua 系列、RECOMP、RFiD/MGFiD/FiDO、ReLLa/ReLLaX、MSL、
  I-LLMRec、VarLenRec 与 dynamic retrieval-budget 近邻均已纳入。通用 prompt
  compression、length bias、reordering、token-efficient representation、动态预算和
  top-k 均已有先例，但未检出固定 128-token budget、固定 CF identity 下对 GRAM
  collaborative-prefix metadata displacement 的 paired reserialization 诊断。三项
  gate 均通过，固定决策为
  **`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_GRAM_NARROWING`**。最强反例是该问题
  可能只是本地 serialization defect，且 metadata-first 仅以 CF evidence 换位置优势；
  因而只解锁已另行预注册的 CPU-only G0-D1 static truncation census，不解锁效果
  诊断、训练、GPU 或 test。
- 2026-07-24：在 LEI `STOP_LEI_NO_RAW_ECHO` 后建立方向 G：Collaborative Prefix
  Budget Displacement（CPBD）。这是 result-informed、post-hoc 新周期：LEI 的现有
  span audit 显示，Toys fine passage 平均约有 25.41 个可见 CF token 与 66.81 个
  metadata token；Beauty 则为 70.35 与 26.48，CF 在 CF+metadata 可见预算中的平均
  占比分别约 28.2% 与 72.7%。GRAM 把 `similar items:` 放在 metadata 前并对每个
  passage 右截断到 128 token，故固定 collaborative prefix 可能把有益 metadata
  直接挤出模型输入。该问题不把可见 CF token 称为有害，也不重跑 LEI/CGI mask：
  被截断内容无法靠 attention mask 恢复。当前只详细锁定 G0-N 原文级差异审计；
  不运行新的数据诊断、不加载 checkpoint、不训练、不使用 GPU、不读取 test。
- 2026-07-24：LEI F0-D 按冻结协议完成，4/4 CPU 单元测试与双数据集真实 tokenizer
  span 集成检查通过；Toys/Beauty tail miss/hit 各 256 人，cohort 与 CGI E0 完全
  一致，full 重算误差为 0，role localization、matched-control eligibility、finite、
  validation lineage 与资源恢复均通过，未训练、未生成 beam、未读取 test 或
  `sequence[-1]`。Toys 的 raw link harm 为 0.00561，95% CI
  [-0.00885, 0.02002]、positive rate 0.53125，raw 与 adjusted gate 均失败；
  Beauty raw 均值 0.02095 且 CI 为正，但 positive rate 0.50000 <0.55，亦失败。
  两数据集 metadata benefit 与 miss–hit association 均通过，Beauty adjusted gate
  通过，但不能越过串行 raw gate；CF-ID 仅为预注册 secondary descriptive，不得
  挽救方向。固定决策为 **`STOP_LEI_NO_RAW_ECHO`**，F1 及后续不解锁。
- 2026-07-24：LEI F0-D 第一次启动在 checkpoint 加载前因 runner 未执行 plan 要求的
  CUDA context 最多 120 秒释放轮询，显存门槛检查过早而以 exit 4 退出；未产生任何
  科学分数，CodeLlama 已恢复。唯一修复是在相同 30 GiB 门槛下增加最多 24 次、每次
  5 秒的显存轮询，未改变 cohort、mask、seed、endpoint 或 gate；随后从头执行成功。
- 2026-07-24：LEI F0-N 在读取任何新 checkpoint 分数前完成，三项必要 gate 均通过，
  固定决策为 **`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_GRAM_NARROWING`**。
  CopyNet/RepeatNet 已覆盖通用 copy 与 repeat，DECOR/ActionPiece 已覆盖 contextual
  token representation，Token-Weighted/Ghost 已覆盖 token/popularity optimization，
  Term-ID 工作已覆盖 native-vocabulary identifier；因此只保留 GRAM 原生
  link-anchor/CF-attribute/decoder-symbol 的位置保持 span 归因及经机制证据驱动的
  target-free role disambiguation。最强反例是 lexical repetition 本来就是有效 linking，
  故另行预注册 F0-D：以 link span 为唯一主角色，要求 raw harm、matched metadata
  specificity、metadata 正贡献和 tail failure association 在双数据集全部通过。当前只
  解锁该 frozen diagnosis；仍不解锁训练、模型修改或 test。
- 2026-07-24：在 CGI `STOP_CGI_NO_INTERFERENCE` 后建立方向 F：Lexical Echo
  Interference（LEI）。该方向不把 E0 的 whole-passage 负干扰假设改名重试，而是研究
  GRAM information linking 特有的 token-role 耦合：同一原生 lexical ID 同时在
  coarse prompt、fine `item:` anchor、fine `similar items:` 属性和 decoder target
  中出现。当前只详细锁定 F0-N 原文级差异审计；通过后才允许预注册 span-factorized
  冻结诊断。当前不加载 checkpoint、不读 validation 效果或 test、不训练、不用 GPU。
- 2026-07-24：CGI E0-D 按冻结协议完成，CPU 单元测试 4/4 通过；Toys/Beauty
  四层各 256 人，full 重算最大绝对误差均为 0，finite、顺序 identity、validation
  lineage 与资源恢复审计均通过，未训练、未读取 test 或 `sequence[-1]`。两数据集
  tail-miss `G_all` 均为负（Toys -0.3246，Beauty -0.1546），说明总体上 fine
  passages 提高而非降低 gold lexical path 得分；累计干扰、old-passage 与 temporal
  specificity 三项 gate 均失败。虽然 tail miss-hit association 为正，但它只表示
  fine evidence 对 hit 用户帮助更强，不能挽救不存在的负干扰主机制。固定决策为
  **`STOP_CGI_NO_INTERFERENCE`**；E1 及后续不解锁。
- 2026-07-24：CGI E0-N 在加载 checkpoint 或计算 E0-D 分数前完成。三项必要 gate
  均通过，固定决策为 **`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_NARROWING`**。
  LWGR 已覆盖 semantic knowledge 与 behavioral signal 冲突及 selective beneficial
  fusion，RFiD/MGFiD 已覆盖 FiD evidence guidance/pruning，CFT/MHL 已覆盖行为反事实
  与 history masking。因此只保留 GRAM 原生 coarse/fine history passage 的结构化负
  贡献审计及推荐特异 target-free gate 空间；只解锁预注册 E0-D。
- 2026-07-24：在 NLPL `STOP_NLPL_NO_EXPOSURE` 后建立方向 E：
  Counterfactual Granularity Interference（CGI）。该方向允许迁移 RAG 的
  counterfactual passage attribution 作为诊断工具，但研究问题收窄为 GRAM 特有的
  coarse lexical-history prompt 与逐 item fine-text passages 在 FiD late fusion 中是否
  产生负贡献。当前详细锁定 E0-N 与条件执行的 E0-D；E1 以后只保留轮廓。
- 2026-07-24：曾基于静态代码怀疑 coarse/fine 历史顺序相反；继续核对运行配置后确认
  当前 Toys/Beauty checkpoint 均为 `reverse_history=1`，coarse lexical history 与
  `history_input` 实际都按新→旧排列。该时间错位假设在写入预注册前被否定，不作为
  E0 假设、结果或论文主张。
- 2026-07-24：NLPL D0-D 按预注册完成，CPU 单元测试 6/6 通过，双数据集
  Recall@10/50 与 S0 精确一致，未加载 GRAM checkpoint、未读取 test、未训练、未使用
  GPU。Toys/Beauty matched-sibling concordance 分别为 0.4438/0.5376，bootstrap
  下界为 0.3995/0.4607，permutation p 为 0.9959/0.1890；tail miss OR 分别为
  0.5498/0.5426。两数据集均只通过 support，固定决策为
  **`STOP_NLPL_NO_EXPOSURE`**，D1 及后续不解锁。
- 2026-07-24：D0-D 首次运行完成 Toys 计算后，在写诊断 JSON 时因 NumPy `bool_`
  不可序列化而退出；该次为执行失败，不作科学结论。唯一修复是把 gate 值显式转换为
  Python `bool`，未改变输入、公式、seed、门槛或分析顺序；随后从头重跑双数据集。
- 2026-07-24：NLPL D0-N 在读取 D0-D 诊断结果前完成，三项必要 gate 均通过，固定
  决策为 **`NOVELTY_SCOPE_PASS_WITH_NARROWING`**。Decoding Matters 已覆盖
  length-normalization/ghost-token amplification，Calibrate Before Use 已覆盖通用
  content-free answer-prior calibration，因此未来只能主张 GRAM 类 lexical-ID 的
  frozen-base prior exposure 机制与推荐特异结构化干预，不能声称通用 token bias 或
  prior subtraction 首创。按串行 gate 只解锁预注册 D0-D CPU 诊断。
- 2026-07-24：在 CAMI `STOP_CAMI_NOVELTY` 后建立全新方向 D：Native Lexical
  Prior Leakage（NLPL）。该方向不再修改 identifier 数量或做 beam 内 margin 重排，
  而是检验 GRAM 的 semantic-to-lexical translation 是否把冻结 T5 的原生词元条件
  先验带入 item/path 概率和 beam-50 候选曝光。当前详细锁定 D0-N 原文级差异审计；
  只有 D0-N 通过，才允许实现已预注册的双数据集 D0-D 0-GPU 诊断。D1 以后只保留
  进入条件和阶段目标，不预先选择实现。
- 2026-07-24：CAMI C0-N 按预注册完成原文级差异审计。新增检出的近邻工作 Pctx
  已在单个自回归生成推荐器中使用同一 item 的多个上下文相关 Semantic ID，并在推理时
  把不同 SID 路径概率聚合为 item probability；PIT 与 MTGRec 进一步挤压多 identifier
  空间。因此 C0-N 的 `single-decoder item marginalization` 必要门槛失败，固定决策为
  **STOP_CAMI_NOVELTY**。按串行 gate 不实现、不运行 C0-D，不训练、不使用 GPU、
  不读取 test；方向 C/CAMI 当前表述终止。
- 2026-07-24：在 HBTR 路径整体 `STOP_HBTR` 后建立全新方向 C：
  Context-Adaptive Multi-View Identifiers（CAMI）渐进式周期。当前只详细锁定
  C0 的文献差异审计、Semantic-ID 候选饥饿诊断和 training-only 行为视图互补性探针；
  C1 以后只记录阶段目标与进入条件，必须依据上一阶段结果另行修订和预注册。CAMI
  当前不解锁模型实现、correctness smoke、GPU、test 或效果声明。
- 2026-07-24：HBTR-v2 F0 按唯一锁定 quantile-tail 公式完成，CPU 单元测试 4/4
  通过，未读取 validation 效果或 test，未使用 GPU。Toys 全部门槛通过；Beauty 的
  tail 非平凡行 15.79% <20%、joint 非平凡行 8.91% <10%、C4-v2=C2 pair 率
  84.21% >80%，因此固定决策为 **STOP_HBTR**。按预注册不尝试第二套公式、不做
  显存修复、不启动 correctness smoke；HBTR 路径终止。
- 2026-07-24：在 `V2_DESIGN_ALLOWED` 后建立 HBTR-v2 F0 独立预注册。F0 只检验
  一个与 training-only head 20% / tail 80% 定义严格对齐的 popularity-rank quantile
  权重能否让 tail/joint margin 获得最低可辨识激活率；不训练、不读 validation 效果、
  不读 test、不扫描第二套公式。F0 通过也只解锁显存等价修复设计，不解锁 GPU。
- 2026-07-24：HBTR-v2 failure-autopsy 按锁定命令完成，CPU 单元测试 4/4 通过，
  未使用 GPU、未读取 test。两数据集 eligible/prefix 支持门槛通过，tail/joint 可辨识性
  门槛失败；generic signal 仅 Toys 通过。固定决策为 **V2_DESIGN_ALLOWED**，只解锁
  新 HBTR-v2 的设计与预注册，不解锁 correctness smoke 或任何 GPU。
- 2026-07-24：HBTR 10% pilot 完成，协议完整但 8 个预注册 gate 失败，科学决策为
  **STOP**。Toys C4 vs C0 NDCG@10 为 +0.64%，Beauty 为 -0.06%；Beauty overall
  Recall@10 与 tail Recall/NDCG、至少一个数据集 +2% 实际效应、C4 超越 C2 以及两
  数据集 peak-reserved 门槛均失败。HBTR-v1 不进入 25%、全量、更多 seed 或 test。
- 2026-07-24：基于 pilot 结果建立 HBTR-v2 failure-autopsy 独立诊断周期。该周期明确
  标记为 **result-informed、post-hoc exploratory**，只读取既有 training-only cache、
  validation per-user 结果、训练摘要和锁定配置，不读取 test、不训练、不修改 HBTR-v1
  决策。0-GPU 诊断只决定是否允许设计 HBTR-v2，不直接解锁任何 GPU pilot。
- 2026-07-23：HBTR pilot 首次执行在 Toys/C0 完成训练及 2,048 用户验证后，因
  history subgroup 使用截断后的 `max_his=20` 长度，导致预注册 `21+` 分组为空并在
  汇总时触发除零。该次状态记为执行失败、科学结论不可判定；修复仅保留截断前历史长度
  用于分组、增加空组保护与断点续跑，不改变 split、模型输入、C0–C4、训练预算、终点或
  晋级门槛。已完成的 Toys cache/C0 checkpoint/逐用户验证保留，不静默删除。
- 2026-07-22：S0 双数据集 validation 离线诊断完成后修订。
- 2026-07-22：S0b 16 配置可靠性拒绝探针完成后再次修订；科学决策为 STOP。
- 2026-07-22：用户选择方向 A；新增 Learned Reliability-Calibrated UCRF（LRC-UCRF）独立预注册周期。
- 2026-07-22：LRC-F0 CPU 可学习性探针完成，两数据集均未通过全部必要条件，科学决策为 STOP；方向 A 终止。
- 2026-07-22：按原转向规则建立方向 B（HBTR）独立周期；当前只允许 B0 文献差异审计与 0-GPU 诊断，不允许训练。
- 2026-07-22：HBTR-B0 完成，Beauty/Toys 四项数据门槛均通过；文献决策为 GO WITH NOVELTY NARROWING，只解锁 B1 设计/正确性 smoke，不解锁 pilot/全量。
- 2026-07-22：HBTR-B1 CPU 测试与 Toys/Beauty GPU3 正确性 smoke 通过；首次权重分支覆盖不完整的产物已保留，repair smoke 完成 joint/prefix/tail 真实 gradient 覆盖。只解锁 10% pilot 设计，不构成效果 GO。
- 修订性质：**result-informed、post-hoc amendment**；S0b 必须与原 S0 分开报告，不能把结果后提出的门槛或设计写成事前预注册。
- 证据来源：`report/第三阶段/GRAM_第三阶段_S0离线诊断报告.md`、`artifacts/phase3/s0/{Toys,Beauty}/validation/summary.json`、`artifacts/phase3/promotion_decisions.md`。

> S0/S0b 已执行并完成 lineage 审计，因此标为 `ANALYZED`；它们尚未做独立复跑，不标为 `VERIFIED`。原 UCRF-v1 路径已停止，任何后续 learned-gate 或优先级 2 实验必须建立新的预注册边界。

## Experiment Overview

- **Title**：方向 I / SMBR：带 abstention 的 training-only metadata budget
  recovery policy（当前仅为 I0-N 草案）。
- **Objective**：检验 CPBD 中“恢复内容有小幅正价值、全局换序却产生大范围净伤害”
  的异质性，能否在不使用目标、validation 成败或 test 的条件下被 training-only
  结构特征可靠预测；不能预测时保持 current serialization。
- **Type**：I0-N 有界原文审计；I0-D 及以后尚未预注册、未解锁。
- **当前 endpoint**：仅文献差异 gate，不读取新的模型分数。
- **Confirmatory boundary**：Sports 或另一个未参与 A–H 方向生成的数据集；现有
  Beauty/Toys validation 只允许作为 repeated-validation exploratory evidence。
- **Current gate**：UCRF、HBTR、CAMI、NLPL、CGI、LEI、CPBD 与 PENS 均已停止；
  当前只解锁 SMBR I0-N，不解锁诊断、训练、模型修改、GPU pilot 或 test。

### 当前方向 G 的阶段性假设

| 编号 | 假设 | 操作化判据 | 性质 |
|---|---|---|---|
| G-H1 | 固定 CF prefix 在 128-token 右截断下造成 metadata displacement | G0-N 先确认该 GRAM-native field-order × hard-budget 机制未被实质覆盖；数据判据须在 G0-N 后独立预注册 | 主机制 |
| G-H2 | displacement 可与“CF token 本身是噪声”区分 | 未来诊断必须恢复被截断 metadata，并用固定总预算/固定 CF identity 的 paired construction 区分内容替换与 token deletion | 因果特异性 |
| G-H3 | 机制区别于一般 length bias、prompt compression 和 dynamic top-k | 近邻矩阵必须覆盖 LBR、LLMLingua 系列、RECOMP、FiD pruning、ReLLa/X、VarLenRec 与 GRAM top-k analysis | 新颖性前提 |

方向 G 不声称首次研究长文本、截断、prompt compression、dynamic top-k 或 length
bias。只有未来证明当前 serialization 确实丢失 metadata，且在固定预算下用
target-free field reallocation 恢复这些 token 能稳定改善 gold path，才允许称为
collaborative-prefix budget displacement。当前只有 post-hoc 结构动机，没有机制结论。

## 0. 阶段定位

第三阶段不是立刻开展 30 epoch 全量实验，而是建立一条可证伪、可止损的创新筛选漏斗：

```text
S0 双数据集离线诊断（已完成：整体 MODIFY）
    -> S0b 可靠性拒绝探针（已完成：STOP，0/16 通过）
    -> UCRF-v1 offline path 终止，原 S1 不启动
    -> LRC-F0 可学习性探针（已完成：STOP）
    -> 方向 B / HBTR 新周期：B0 差异审计 + 0-GPU 诊断
    -> B0 GO 后才建立 B1 小样本 smoke
    -> 10% 用户、固定验证子集微型实验（已完成：STOP）
    -> HBTR-v1 终止，不进入 25%/全量/test
    -> HBTR-v2 failure-autopsy（0 GPU、post-hoc，只决定是否允许新设计）
    -> 若 V2_DESIGN_ALLOWED：建立独立预注册周期；否则结束 HBTR
    -> HBTR-v2 F0（已完成：Beauty FAIL，整体 STOP_HBTR）
    -> 方向 C / CAMI C0-N（已完成：Pctx 实质覆盖必要核心，STOP_CAMI_NOVELTY）
    -> C0-D 未解锁，CAMI 不进入诊断实现、模型、GPU 或 test
    -> 方向 D / NLPL D0-N 原文级差异审计（已完成：PASS_WITH_NARROWING）
    -> D0-D 双数据集 0-GPU native-prior 曝光诊断（已完成：STOP_NLPL_NO_EXPOSURE）
    -> NLPL 终止，不进入 D1、模型实现、GPU 或 test
    -> 方向 E / CGI E0-N（已完成：PASS_WITH_TRANSFER_AND_NARROWING）
    -> 冻结 checkpoint 的 E0-D 反事实粒度诊断（已完成：STOP_CGI_NO_INTERFERENCE）
    -> CGI 终止，不进入 E1、训练、模型效果实验或 test
    -> 方向 F / LEI F0-N 原文级差异审计（已完成：PASS WITH TRANSFER AND NARROWING）
    -> span-factorized frozen F0-D（已完成：STOP_LEI_NO_RAW_ECHO）
    -> LEI 终止，不进入 F1、训练、更多 validation 分析或 test
    -> 方向 G / CPBD G0-N 原文级差异审计（当前唯一解锁）
    -> 仅当 G0-N 通过：另行预注册 truncation census 与 frozen budget-recovery diagnosis
```

本阶段的第一目标是找到有数据证据、跨 Beauty/Toys 有一致趋势、且能清楚区别于已有工作的创新点。全量指标只是最后的确认手段，不是早期试错工具。

当前证据不是“再换一个排序权重就可能成功”，而是“**Beauty 的主要限制已经发生在
beam-50 候选支持集之前或之中**”。CAMI 的多 identifier 论文点又已被 Pctx 覆盖。
方向 D 的 frozen-base lexical prior 解释没有得到跨数据集支持。方向 E 未修改
identifier、native prior 或 beam 内 margin，而是回到 GRAM 的另一个特有结构：
coarse history 与逐 item fine text 被独立编码后直接在 decoder 融合；E0 检验“更多
正确内容是否会对特定失败用户产生可定位的负贡献”。E0-D 已否定该主机制：fine
passages 在 tail miss 上的累计贡献总体为正，且不存在 oldest-specific 负贡献。
方向 F 据此不删除有益 passage，而是审计 passage 内部不同 token role 是否相互抵消。

第三阶段默认使用用户已指定的物理 **GPU3**，除非用户之后明确改卡；不再要求用户为每个子任务重复指定。默认启动门槛为 30 GiB（30,720 MiB）。所有超过几分钟的任务必须在持久 `tmux` 会话后台运行，保存 PID、状态 JSON、日志和资源遥测，并提供稳定的 `status` 查询入口。GPU 任务启动前必须调用 `/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh stop` 释放 CodeLlama reservation；任务无论成功、失败、收到信号或后处理失败，都必须通过 trap 调用同一脚本的 `start 3` 恢复 reservation。不得让用户重复提醒该协议。

### 0.1 第三阶段固定运行协议

1. 默认物理卡：GPU3；通过 `CUDA_VISIBLE_DEVICES=3` 映射为进程内 `cuda:0`。
2. 启动顺序：检查无同名实验会话 → 停止 CodeLlama reservation → 最多等待 120 秒让 CUDA context 完全释放 → 确认 GPU3 空闲显存不少于 30,720 MiB → 启动实验。
3. 后台要求：预计超过几分钟的训练、全量推理或离线分析一律使用具名 `tmux` 会话，不使用会随调用结束而被回收的普通 `nohup` 子进程。
4. 状态要求：每个任务保存 runner/workload PID、started/updated time、stage、status、reason、GPU、checkpoint、日志、遥测和 reservation 状态；至少支持 `running / succeeded / partial / failed / blocked / restoring_resource`。
5. 查询要求：在 `experiment/phase3/` 提供 `start/status` 入口；启动后立即向用户返回 status 命令，不在对话中持续轮询。
6. 资源记录：GPU board 与 workload PID 每 5 秒采样，磁盘每 5 分钟采样；日志和状态使用原子更新。
7. 退出恢复：EXIT/INT/TERM/HUP trap 先停止监控和残留 workload，再最多尝试 3 次恢复 CodeLlama；实验失败不得被 reservation 恢复成功所掩盖。
8. 禁止静默重试：OOM、NaN、超时、数据错配或外部中断均保留原始日志和状态，等待用户决定。

## 1. 已有基础

### 1.1 可复现基线

| 数据集 | 最佳 epoch | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 全量任务耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Beauty | 25 | 0.063274 | 0.044223 | 0.088897 | 0.052461 | 约 40.47 h |
| Toys | 30 | 0.071193 | 0.051401 | 0.095302 | 0.059193 | 约 33.52 h |

两套基线均完成 full ranking、最佳 validation checkpoint 选择、预测保存和资源归档，可作为配对误差分析与后续公平对照。

### 1.2 双数据集错误分析证据

以下统计来自已经锁定的最佳 checkpoint test 预测，仅用于发现问题；后续创新参数不得在 test 上调节。

#### 目标商品流行度

| 数据集 | 分组 | 用户数 | Recall@5 | Recall@10 | Recall@50 |
|---|---|---:|---:|---:|---:|
| Beauty | 头部 20% 商品 | 9,590 | 0.0991 | 0.1421 | 0.3033 |
| Beauty | 尾部 80% 商品 | 12,773 | 0.0364 | 0.0489 | 0.0786 |
| Toys | 头部 20% 商品 | 7,339 | 0.0849 | 0.1166 | 0.2341 |
| Toys | 尾部 80% 商品 | 12,073 | 0.0629 | 0.0823 | 0.1357 |

Beauty 的长尾退化尤其严重，Toys 也存在稳定差距。主创新必须同时报告 overall、head 和 tail，而不能用头部增益掩盖长尾损失。

#### 目标是否被历史商品的 CF 邻居覆盖

| 数据集 | 分组 | 覆盖率 | Recall@5 | Recall@10 | Recall@50 |
|---|---|---:|---:|---:|---:|
| Beauty | 目标在最近商品 top-10 CF 邻居 | 6.37% | 0.6489 | 0.7514 | 0.8862 |
| Beauty | 目标在任一最近 20 个历史商品 top-10 邻居 | 11.53% | 0.4261 | 0.5304 | 0.7158 |
| Beauty | 无上述覆盖 | 88.47% | 0.0160 | 0.0313 | 0.1045 |
| Toys | 目标在最近商品 top-5 CF 邻居 | 4.49% | 0.6881 | 0.7787 | 0.8784 |
| Toys | 目标在任一最近 20 个历史商品 top-5 邻居 | 9.24% | 0.4278 | 0.5137 | 0.6748 |
| Toys | 无上述覆盖 | 90.76% | 0.0349 | 0.0527 | 0.1218 |

协同关系一旦覆盖目标，准确率极高，但当前覆盖率很低。这是两个数据集共同、最有力的创新动机。

#### 扩大 CF 邻居池的覆盖—噪声矛盾

| 数据集 | 每个历史商品 k | 最近商品覆盖率 | 最近 20 个历史的并集覆盖率 | 平均并集候选数 |
|---|---:|---:|---:|---:|
| Beauty | 5 | 4.41% | 7.79% | 31.20 |
| Beauty | 10（当前） | 6.37% | 11.53% | 59.74 |
| Beauty | 20 | 8.80% | 15.95% | 112.91 |
| Toys | 5（当前） | 4.49% | 9.24% | 30.42 |
| Toys | 10 | 5.83% | 12.43% | 58.32 |
| Toys | 20 | 7.36% | 16.05% | 111.02 |

直接增大 k 可以提高覆盖，但会把平均候选集合扩展到约 113 个，产生明显噪声与输入成本。因此“只调 top-k”既不是充分方案，也不足以成为论文创新；需要用户条件化选择和可靠性控制。

### 1.3 S0 已完成结果与证据更新

S0 使用锁定 best checkpoint 的完整 validation 预测，所有 semantic ID、用户和目标 lineage 均为零错配；没有查看 test 调参。

| 数据集 | 用户数 | k=20 并集覆盖率 | k=20 平均候选数 | Baseline NDCG@10 | Rerank NDCG@10 | 相对变化 | Recall@10 绝对变化 | S0 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Toys | 19,412 | 18.550% | 97.27 | 0.076275 | 0.077585 | +1.718% | +0.003245 | GO |
| Beauty | 22,363 | 16.366% | 98.02 | 0.064974 | 0.065280 | +0.472% | +0.000447 | STOP_OR_MODIFY |

困难子组揭示了比 overall 更重要的机制差异：

| 数据集 | Tail NDCG@10 相对变化 | Covered NDCG@10 相对变化 | Uncovered Recall@10 相对变化 | Uncovered NDCG@10 相对变化 |
|---|---:|---:|---:|---:|
| Toys | +6.575% | +9.447% | -10.507% | -17.271% |
| Beauty | +1.878% | +2.156% | -5.244% | -8.158% |

证据、推断与决策必须分开：

- **证据**：扩大到 k=20 后 validation relation coverage 提升到 16–19%，关系重排明显帮助 covered 用户，但伤害占多数的 uncovered 用户；Toys overall 达标，Beauty 未达标。
- **推断**：关系分数包含强信号，但固定正向加权把无关关系候选推到前面；问题更接近 reliability calibration / selective fusion，而不是 candidate recall 完全不足。
- **尚未证明**：离线 hand-crafted gate 能否识别何时应拒绝关系；可学习 gate 能否在训练后跨数据集提升；任何论文级因果或显著性结论。
- **决策**：S0 整体为 **MODIFY**。不得直接进入 S1；先执行一次边界锁定的 S0b。

## 2. 核心研究问题

第三阶段围绕一个主问题展开：

> 如何在不重新构造语义 ID、不引入目标泄漏的条件下，把静态、低覆盖、等权使用的协同关系，变成用户条件化、覆盖更高且能够拒绝噪声的动态证据？

拆成四个可验证问题：

1. 现有 beam-50 候选中，协同关系分数能否在不训练时改善 top-5/top-10 排序？**S0 部分回答：Toys 可以，Beauty 效应不足。**
2. 从完整历史的邻居并集检索，能否提升目标覆盖？**S0 已回答：k=20 可把 validation 覆盖提高到 Toys 18.55%、Beauty 16.37%，但平均候选约 97–98 个，不能直接全部使用。**
3. 仅凭推理时可得的置信特征，能否选择性启用关系重排并把 uncovered 退化控制在 1% 内？**由 S0b 回答。**
4. 可学习的 signed reliability gate 是否优于固定 top-k、固定位置编码、等权 late fusion 和 hand-crafted abstention？**通过 S0b 后由 S2 回答。**
5. 增益能否同时存在于 Beauty/Toys，并且改善长尾而不牺牲 no-CF-covered 多数用户？**当前尚未满足。**

### 2.1 变量、实验单位与控制

| 类别 | 定义 |
|---|---|
| 自变量（IV） | retrieval 类型（静态/动态）、relation passage、reliability gate、auxiliary loss |
| 主因变量（DV） | validation NDCG@10 |
| 次因变量 | Recall@5、NDCG@5、Recall@10；tail/head、CF-covered/uncovered、history bins 指标；GPU 时间、显存和推理延迟 |
| 评测单位 | 用户；同一用户在 baseline/candidate 间做 paired comparison |
| 独立重复单位 | 完整训练 run/seed，而不是用户数。不能把 20,000 个用户错误当成 20,000 次独立模型重复 |
| 固定控制 | 数据、leave-one-out split、semantic ID、T5-small、候选商品全集、beam 50、训练预算、batch、scheduler、checkpoint 选择规则 |
| matched controls | 原始 GRAM、固定扩大 k、更多文本但无动态检索、parameter-matched MLP |

### 2.2 主要混杂因素与处理

| 混杂/偏差 | 风险 | 控制方法 |
|---|---|---|
| 参数量增加 | 效果可能来自容量而非机制 | parameter-matched MLP control；报告 trainable/total parameters |
| 输入 token 增加 | 效果或成本来自更多文本 | 固定 passage/token budget；A1 控制“更多文本” |
| 训练预算不同 | 更多 step 可能虚增效果 | baseline/candidate 使用相同用户、epoch、optimizer step 和 validation 次数 |
| Pilot 子集选择 | 可形成 Berkson/selection bias | 一次性分层抽样、保存用户 ID 与 SHA-256，禁止换子集 |
| 多次试参 | look-elsewhere / garden of forking paths | 实验 registry 记录全部配置；每阶段限制候选数；区分 exploratory/confirmatory |
| Test 反复查看 | test leakage | 所有检索/门控/超参数只用 validation；方案锁定后 test 一次 |
| Target leakage | 动态关系构造可能误用下一商品 | relation passage 只读历史；加入单元测试和数据 lineage 审计 |
| Beauty/Toys 原始 k 不同 | 跨域比较可能混淆 | 保留各自官方 baseline，并用共同的候选预算 M 做机制比较 |
| GPU 外部竞争 | 污染时间、可能改变失败率 | PID allocator/NVML 分离；效果指标可比较，时间只在无外部干扰时作独占结论 |
| Checkpoint cherry-picking | 选择性汇报 | 仅按预注册 validation NDCG@10；固定 ties 规则为更早 epoch 优先 |
| 随机性 | 单 seed 偶然增益 | S2 只筛选，S3 至少 2 seeds，确认阶段种子数由功效分析确定且不得少于 3 |

## 3. 推荐主创新：UCRF

暂定名称：**User-Conditioned Collaborative Retrieval and Reliability-Gated Fusion**，简称 **UCRF**。

### 3.1 模块一：用户条件化协同检索（UCR）

对用户最近 `H <= 20` 个历史商品，读取现有 SASRec top-20 邻居，不重新训练 SASRec，不使用 validation/test 目标。构造候选并集：

```text
C(u) = union(N_K(i_j)),  i_j in recent user history
```

对候选关系商品 `c` 计算只依赖历史的检索分数：

```text
relation_score(u,c)
  = max_j recency_weight(j) * cf_rank_weight(i_j,c)
  + consensus_weight * number_of_supporting_history_items(c)
```

首版只使用可解释的三项：

- 历史位置/recency；
- SASRec 邻居排名；
- 被多少个历史商品共同召回的 consensus。

从约 30–113 个候选中只保留 `M` 个，形成一个独立 collaborative relation passage。`M` 在 validation 上从 `{5, 10, 20}` 中选择，不在 test 上调节。

### 3.2 模块二：可靠性门控 late fusion（RGF）

GRAM 当前为 coarse user prompt 和每个 item prompt 分别编码，之后将所有 token 拼接给 decoder cross-attention；除静态 passage position embedding 外，没有用户条件化的 passage 可靠性控制。

UCRF 为 relation passage 计算候选级门值 `g_j`，并为整个关系分支计算用户级 abstention 值 `a_u`：

```text
g_j = sigmoid(MLP([user_summary; passage_summary; relation_features_j]))
a_u = sigmoid(MLP([user_summary; relation_pool_summary; confidence_features_u]))
```

其中 relation/confidence features 只包含推理时可用的信息，例如历史位置、CF rank、consensus、候选池大小、最高关系分、top-1/top-2 margin 和有效 support 数。S0 已证明“只增强不拒绝”会伤害 uncovered 用户，因此第一版改为可保持 baseline 初始行为、同时允许增强与抑制的 signed residual gate：

```text
h'_relation,j = h_relation,j * (1 + alpha * a_u * (2*g_j - 1))
alpha initialized to 0
```

`alpha=0` 时严格回到 baseline；`2g_j-1` 允许正负调制，`a_u` 允许整个关系分支 abstain。必须保留 null/no-relation 路径、relation dropout，并监控 `a_u/g_j` 是否塌缩为常数。原来的单向 `h'=h*(1+alpha*g)` 不再作为主设计，只保留为“amplify-only”反例消融。

### 3.3 模块三：关系可靠性辅助监督（可选第二步）

训练样本已知下一个目标，可以构造 passage-level 标签：某个历史商品的邻居集合是否覆盖目标。该标签只用于训练门控器；推理仍只输入历史和关系特征，不输入目标。

```text
L = L_token_CE + lambda_rel * L_reliability
```

由于正样本稀疏，`L_reliability` 使用 class-balanced BCE 或 focal loss。S0 后它从“可选装饰”提升为核心机制候选，但实验仍必须逐项增加：S2 先比较无 gate、无辅助监督 signed gate、带辅助监督 signed gate，不能在首次 run 同时改变检索、门控、损失和容量而失去归因。

### 3.4 预期贡献表达

如果实验成立，论文贡献可表述为：

1. 发现并量化静态 CF verbalization 的“高价值、低覆盖、高噪声”瓶颈，证据跨 Beauty/Toys 一致；
2. 提出用户条件化的历史关系检索，在不重新构造 semantic ID 的情况下扩大有效协同证据覆盖；
3. 提出带显式可靠性监督的 gated late fusion，使生成器能够利用强关系并拒绝噪声；
4. 在 overall、head/tail、CF-covered/uncovered、history-length 分组下验证收益来源。

## 4. 备选创新与优先级

| 优先级 | 方向 | 价值 | 风险/说明 |
|---:|---|---|---|
| 1 | UCRF：动态关系检索 + 可靠性门控 | 与双数据集证据最直接对应，机制和分析可形成完整故事 | 需防止与普通 RAG/门控融合表述过于接近；必须强调历史关系检索、可靠性监督和生成式 late fusion 的组合 |
| 2 | Hierarchical hard-negative objective | 从相同 ID 前缀或 beam 候选中选择难负例，补充 token CE 对商品级排序监督 | LOHRec 等工作已研究 order/hierarchy；需严格新颖性审计，暂作为 UCRF 后续增强而非首选 |
| 3 | Tail-aware prefix calibration | 对尾部目标或深层 ID token 加权，改善 Beauty 长尾 | 容易退化成 loss reweighting，单独论文创新性偏弱；可作为分析/消融 |
| 4 | User-adaptive history passage gate | 替代纯静态 position embedding，对不同历史长度动态选 passage | 与 UCRF 的门控自然兼容，但单独使用可能难形成足够强的贡献 |
| 5 | Relation-aware beam reranking | 零训练、实现快，可验证关系信号和提供强轻量 baseline | Recall 上限受原 beam-50 限制，且已有 trend-aware inference 类工作；只作为探针/对照，不作为唯一创新 |

本阶段禁止同时实现多个主方向。只有 UCRF 被明确证伪后，才切换到优先级 2。

## 5. 与已有工作的边界

- GRAM 已提出 semantic-to-lexical translation、静态 CF verbalization 和 multi-granular late fusion；因此“加入 CF 文本”“加入 late fusion”不是新贡献。
- PRORec/UNGER 类方法研究 collaborative/semantic knowledge 在 item code 中的融合；UCRF 必须强调在线用户历史条件化检索和 passage reliability，而不是重新做统一 code。
- LOHRec 已指出单一 next-item likelihood 忽略顺序、层次和多种可能项；普通 hierarchical loss 不能直接宣称新颖。
- GRUT 已使用 training-free trend-aware inference；关系重排只能作为诊断或对照，不能仅换一种外部分数就作为主贡献。
- GRID 的系统研究说明 semantic-ID 生成推荐中许多架构细节会显著影响性能；所有 UCRF 对比必须参数量、训练预算和数据划分匹配。

正式写论文前必须针对 2024–2026 年工作做完整相关工作矩阵，至少记录：任务、ID 类型、协同信号位置、是否动态检索、是否用户条件化、是否有可靠性门、训练目标、数据集和公开代码。仅完成当前初筛不等于确认绝对新颖。

## 6. 渐进式实验漏斗

### S0：离线诊断与候选上限（0 GPU，已完成）

执行状态：`ANALYZED`。脚本、完整结果和报告分别位于：

- `experiment/phase3/s0_offline_diagnostics.py`
- `artifacts/phase3/s0/{Toys,Beauty}/validation/`
- `report/第三阶段/GRAM_第三阶段_S0离线诊断报告.md`

原任务均已完成：双数据集 validation 分层、k=`1/3/5/10/15/20` coverage、候选并集、beam-50 oracle、relation-aware reranking 和 lineage 审计；test 未用于调参。

结果：Toys 达到原门槛，Beauty 未达到；整体决定为 **MODIFY**。S0 的模型分数加固定正关系分只验证了信号存在，不足以支持无条件关系增强。

### S0b：可靠性拒绝探针（0 GPU，已完成，post-hoc exploratory）

执行入口：`bash experiment/phase3/run_phase3_s0b.sh start`；查询入口：`bash experiment/phase3/run_phase3_s0b.sh status`。完整任务使用持久 `tmux`，报告固定写入 `report/第三阶段/GRAM_第三阶段_S0b可靠性拒绝探针报告.md`。

目标：在不训练、不读取 test、不使用目标商品的前提下，验证推理时置信特征能否识别“应该启用关系”与“应该保持 baseline”。这是结果后提出的新探索，不能改写为原始预注册。

固定输入：

- Beauty/Toys 已锁定的 validation beam-50 预测；
- 最近 20 个历史商品及现有 SASRec top-20 邻居；
- S0 已生成的 baseline、coverage 和 subgroup 结果；
- 不生成新模型预测，不查看 test。

对 beam 候选 `c` 定义：

```text
q(u,c) = max_j 0.9^age(j) / log2(cf_rank(j,c)+1)
         + beta * support_count(u,c) / max(1, history_length)

active(u) = [q_top1 >= tau]
            AND [q_top1 - q_top2 >= 0.05 OR max_support >= 2]

score'(u,c) = model_score(u,c)
              + active(u) * lambda * q(u,c) * [support_count(u,c) >= s]
```

运行前锁定的唯一网格为 16 个配置：

- `k=20`、`recency_decay=0.9`、`margin=0.05` 固定；
- `beta ∈ {0, 0.25}`；
- `lambda ∈ {0.05, 0.20}`；
- `tau ∈ {0.50, 0.75}`；
- `s ∈ {1, 2}`。

禁止增加更多阈值、按数据集使用不同公式、根据 test 选配置或在看到结果后替换主指标。选择一个 **Beauty/Toys 共同配置**，按两数据集 NDCG@10 相对变化的宏平均排序；若并列，依次选择更小 `lambda`、更大 `tau`、更大 `s`，偏向保守拒绝。

S0b → S1 的全部必要条件：

1. 同一配置在 Toys、Beauty 的 validation NDCG@10 均相对 baseline 提升至少 1%；
2. 两数据集 Recall@10 均不下降超过 0.5 个百分点；
3. 两数据集 uncovered Recall@10 和 NDCG@10 相对下降均不超过 1%；
4. 两数据集 tail NDCG@10 均不下降；
5. `active(u)` 比例在每个数据集均介于 5%–60%，避免退化为 always-on 或 always-off；
6. lineage 零错配、所有 16 个配置完整保留、结果写入 registry 和 promotion record。

若无配置通过，结论为 **STOP UCRF-v1 offline path**，不得通过继续扩网格挽救。之后只能二选一：重新预注册“可学习 gate”研究周期，或切换优先级 2；不能直接把失败的 S0b 当作 S1 放行。

#### S0b 实际结果

- 执行状态：`succeeded`；科学决策：**STOP**；耗时 252.23 秒；GPU 使用为 0。
- 完整性：16/16 个共同配置、32 个 dataset-config 行均已保存；Toys 19,412、Beauty 22,363 个用户；gold/prediction 映射错配均为 0。
- 通过配置：`0 / 16`。
- 诊断最优共同配置：`b0_l0.2_t0.75_s2`，宏平均 NDCG@10 相对变化 `+0.504%`。

| 数据集 | Active rate | NDCG@10 相对变化 | Recall@10 绝对变化 | Tail NDCG@10 | Uncovered Recall@10 | Uncovered NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 83.598% | +0.942% | +0.001494 | +2.569% | -1.208% | -2.881% |
| Beauty | 90.498% | +0.067% | -0.000402 | +2.632% | -6.329% | -9.502% |

16 配置总体范围进一步说明失败原因：Toys active rate 为 83.60%–96.25%，Beauty 为 90.50%–98.73%，所有配置都违反 5%–60% 的 selective gate 要求。Beauty 最佳单数据集 NDCG@10 也只有 `+0.491%`，同时 uncovered NDCG@10 至少下降 `2.702%`。因此失败不是某一个共同配置选择规则造成，而是当前 hand-crafted confidence proxy 缺少区分力。

证据支持的结论是：固定阈值/支持数不能可靠判断何时启用关系；它不等价于“所有 learned gate 必然失败”。若继续 learned gate，必须利用训练期可靠性标签、校准损失和明确的 abstention regularization 建立**新的预注册周期**，不能沿用 UCRF-v1 的 S1 名义。

### S1：实现正确性 smoke（单次 < 15 分钟）

当前状态：**STOPPED for UCRF-v1**。S0b 已判定 STOP，原 S1 永久不放行。只有用户选定新方向、写出新假设/配置/成功门槛并生成新的 Material Passport 后，才创建新周期的 smoke；不得直接复用本节作为启动授权。

使用 100–500 条训练样本和 100 条 validation 样本，只验证：

- 无目标泄漏；
- relation passage 只由历史构造；
- gate shape/mask 与 passage 对齐；
- baseline 权重能非严格加载，新模块初始化为近似 identity；
- forward/backward、checkpoint、full-ranking constrained generation 可运行；
- `alpha=0` 时输出与 baseline 在容差内一致。
- signed gate 能产生增强和抑制两个方向，user-level abstention 与 candidate-level gate 均不是常数；
- amplify-only 旧门控只作为测试/消融，不作为默认实现。

Smoke 指标不得用于创新结论。

### S2：微型筛选（每个配置预计 0.5–1.5 GPU 小时）

为每个数据集建立一次性、固定且版本化的 pilot split：

- 训练：按 target popularity × history length 分层抽取 10% 用户；
- validation：固定 2,048 用户，保持相同分层；
- 候选集：仍为全部商品，保持 full ranking；
- epochs：5；只在 epoch 5 validation；
- seed：2023；
- baseline 与 candidate 使用完全相同的 split、epoch、batch、scheduler 和评测用户。

首轮最多比较四个配置：

```text
P0: matched-budget GRAM baseline
P1: baseline + relation passage（固定打分，无 gate）
P2: baseline + relation passage + unsupervised signed residual gate + user abstention
P3: P2 + reliability auxiliary loss
```

一次只改变一个因素。禁止在同一轮扫描大量 M、lambda、层数和 hidden size。

晋级条件：

- candidate 相对 P0 的 validation NDCG@10 提升至少 2%；
- Recall@10 不下降；
- tail Recall@10 不下降超过 1%；
- uncovered Recall@10 和 NDCG@10 均不下降超过 1%；
- paired bootstrap 的提升方向稳定；
- Beauty/Toys 至少一个显著提升，另一个不为负。

由于 S0 已观察到固定关系增强伤害 uncovered 用户，P1 是预期可能失败但不可删除的机制对照；P2/P3 是否消除该退化是 S2 的核心判据，而不是只看 overall。

### S3：中型确认（每个配置预计 2–5 GPU 小时）

只允许 S2 最优的一个配置和 matched baseline 进入：

- 分层训练用户 25%；
- 固定 validation 用户 4,096；
- 10 epochs，在 epoch 5/10 验证；
- 运行 2 个训练 seed；
- 报告均值、标准差、paired bootstrap CI、训练时间和峰值显存。

晋级全量条件：

1. 两个数据集平均 NDCG@10 均相对提升至少 2%；
2. 两个 seed 提升方向一致；
3. Recall@5/10 至少三项不下降；
4. tail 和 no-CF-covered 至少一个困难子组有明确收益；
5. 增益不能仅来自参数量增加：需有 parameter-matched MLP/control；
6. 新增显存不超过 20%，推理时间不超过 25%，否则必须证明更强的效果—成本收益。

不满足则不跑 30 epoch。

### S4：单数据集全量验证（约 34 小时）

先只跑 Toys seed 2023，原因是基线全量任务较 Beauty 短。使用完整训练集、完整 validation、30 epochs 和原 full-ranking 协议。

早停/止损检查点：

- epoch 5 和 10 与基线同 epoch validation 对比；
- 若两个点 NDCG@10 均低于 baseline，且没有困难子组改善证据，则在 epoch 10 后停止，不继续浪费约 20 小时；
- 若趋势为正，继续至 epoch 30，并按 validation NDCG@10 选最佳 checkpoint。

只有模型、超参数和 checkpoint 选择全部锁定后，才运行一次 test。

晋级条件：最佳 validation NDCG@10 相对 baseline 提升至少 2%，正式 test 的 Recall/NDCG 至少三项提升，且 NDCG@10 不下降。

### S5：跨数据集全量验证（约 41 小时）

Toys 通过后，才在 Beauty seed 2023 完成同配置 30 epoch。不得针对 Beauty 单独改变核心结构；只允许数据集原有的 ID length、cluster 和 CF k 差异。

Beauty 尤其关注 tail Recall/NDCG。若 overall 小幅提升但 tail 明显下降，不视为成功。

### S6：论文级确认（后续阶段）

当 Beauty/Toys 单种子均成功后，再规划：

- 由 S3 方差和 power analysis 锁定 seed 数，且不少于 3 seeds；
- Sports/Yelp 或至少一个额外数据集；
- 完整消融 `UCR / gate / reliability loss / relation dropout`；
- 与 reranking、增大固定 k、parameter-matched gate、tail reweighting 的对照；
- 显著性检验、效率和复杂度；
- case study 与 gate/retrieval 可解释性。

第三阶段探索期不提前承诺这些昂贵实验。

## 7. 公平性与防止小实验误导

1. Pilot split 一次生成后写入用户 ID 文件和 SHA-256，禁止根据结果换子集。
2. Pilot 仍使用完整商品 Trie，不能通过缩小候选集制造虚高指标。
3. 每个 candidate 必须有同训练预算 baseline，不能拿 10-epoch 新模型对 5-epoch baseline。
4. 只在 validation 做超参数选择；test 在方案完全锁定后运行一次。
5. 小样本结果只用于排序候选，不作为最终论文结论。
6. 使用 paired user-level bootstrap，而不是只比较一个四舍五入后的均值。
7. 同时报告 Recall@5/10、NDCG@5/10、head/tail、CF-covered/uncovered 和 history bins。
8. 记录参数量、训练 GPU 小时、peak allocated/reserved、PID NVML 和每用户推理时间。
9. 任何 OOM、NaN、目标泄漏、候选集合错误或 checkpoint 选择错误均保留日志，不静默重试。

### 7.1 探索性与确认性分析边界

| 阶段 | 统计性质 | 可以得出的结论 |
|---|---|---|
| S0–S2（含 post-hoc S0b） | 探索性 | 筛掉明显无效方案、估计效应方向和方差；不能宣称论文方法显著优于 baseline；S0b 必须显式标记为结果后修正 |
| S3 | 探索性确认 | 检查跨 split/seed 稳定性，为功效分析提供方差；2 seeds 仍不足以支持正式显著性结论 |
| S4–S5 | 单种子全量机制确认 | 检查增益在完整数据是否存在；不能代替多种子统计推断 |
| S6 | 确认性 | 使用预注册 seed 数、主终点和统计检验形成论文级结论 |

### 7.2 统计分析计划

1. **主比较**：每数据集 UCRF vs matched GRAM 的 NDCG@10；先报告绝对差、相对差、seed 均值/标准差和 95% CI，再报告 p-value。
2. **用户级配对不确定性**：对固定 checkpoint 的 baseline/candidate 用户级指标差做 paired bootstrap，至少 10,000 次重采样；该 CI 描述评测用户抽样不确定性，不替代训练 seed 方差。
3. **Seed 级不确定性**：论文确认阶段以训练 seed 为独立重复单位。S3 得到方差后做保守 power analysis，目标 power 0.80、双侧 alpha 0.05、最小相关效应为 NDCG@10 相对 2%；最终不少于 3 seeds。若估计所需 seed 超出资源预算，必须把研究定位为探索性并降低结论强度，不能擅自缩小效应阈值。
4. **次指标**：Recall@5、NDCG@5、Recall@10 使用 Holm correction；head/tail、covered/uncovered、history bins 属于异质性探索，使用 Benjamini–Hochberg FDR 并同时报告未校正/校正结果。
5. **实际意义**：即使统计显著，若 NDCG@10 相对增益低于 2% 或推理成本增加超过 25%，不满足本阶段工程晋级标准。
6. **稳健性**：报告每 seed 方向、均值、标准差、CI；禁止只报告最好 seed。若 overall 与 subgroup 方向相反，显式检查 Simpson's paradox。
7. **缺失/失败 run**：OOM、NaN、超时或外部 GPU 挤占的 run 不得静默删除；记录原因并按预先规则决定是否重跑。不得只保留成功 run 造成 survivorship bias。

### 7.3 统计假设与检查

- 不默认假设用户级指标差服从正态，因此主用 paired bootstrap。
- Seed 数较小时不依赖正态性 t-test 单独下结论；同时给出每 seed 原始值。
- 明确总测试数量并执行校正，防止 look-elsewhere effect。
- 不把 validation 上发现的相关模式表述为模型机制因果证据；机制主张必须由预注册消融支持。
- validate 阶段按 ARS 11 类 fallacy scan 全量检查：Simpson、ecological、Berkson、collider、base-rate neglect、regression-to-mean、survivorship、look-elsewhere、forking paths、correlation/causation、reverse causality。

### 7.4 判定顺序

```text
数据/协议完整性
  -> primary endpoint 实际效应是否过 2%
  -> Recall@10 与困难子组是否无不可接受退化
  -> seed 方向与 CI 是否稳定
  -> 多重比较后结论是否仍成立
  -> 参数/时间/显存代价是否可接受
  -> 决定晋级、修改或停止
```

不能用单一 `p < .05` 覆盖实际效应小、假设不满足、重复试验或成本过高的问题。

## 8. 最小消融矩阵

全量之前只做以下必要消融，不做组合爆炸：

| 编号 | Dynamic retrieval | Relation passage | Reliability gate | Auxiliary loss | 用途 |
|---|---:|---:|---:|---:|---|
| A0 | 0 | 0 | 0 | 0 | 原始 GRAM |
| A1 | 0 | 1（固定 k） | 0 | 0 | 控制“只是更多文本” |
| A2 | 1 | 1 | 0 | 0 | 检验用户条件化检索 |
| A3 | 1 | 1 | 1 | 0 | 检验门控 |
| A4 | 1 | 1 | 1 | 1 | 完整 UCRF |
| A5 | 0 | 0 | 参数量匹配 MLP | 0 | 排除参数量解释 |

`M`、relation dropout 和 loss weight 先通过 S2/S3 固定；全量不重新扫描。

## 9. 失败判据与转向规则

以下任一发生时，UCRF 不进入全量：

- S0b 的 16 个锁定配置无一通过，但仍需要扩大网格或按数据集改规则才能制造正结果；
- S2/S3 的动态检索只提高 covered 子组，对占多数的 uncovered 子组继续产生超过 1% 的相对退化；
- 增益完全可由固定增大 k 解释；
- gate/abstention 退化为几乎常数、always-on 或 always-off；
- pilot 增益只在一个任意子集/seed 出现；
- 推理成本增加超过 25% 而 NDCG@10 增益低于 2%。

已观察到的 uncovered 退化本身不再重复计作“新发现”；S0b/S2 必须直接检验能否消除它。若 UCRF-v1 被证伪，优先转向“beam 内 hierarchical hard-negative + tail-aware ranking objective”，但需先完成 LOHRec、MERGE、popularity-aware contrastive recommendation 等工作的差异审计。

## 10. 预计资源预算

| 阶段 | GPU 预算 | 是否允许当前立即执行 |
|---|---:|---|
| S0 离线诊断/重排 | 已完成 | 整体 MODIFY |
| S0b 可靠性拒绝探针 | 已完成，0 GPU | STOP（0/16 通过） |
| UCRF-v1 S1 smoke | 0 GPU h（未启动） | **永久停止，不得启动** |
| UCRF-v1 S2 | 0 GPU h（未启动） | **永久停止，不得启动** |
| 新方向 feasibility/smoke | 未锁定 | 需先选择方向并建立新预注册周期 |
| S3 25% 中型确认 | 2–5 GPU h / 配置 / 数据集 | 仅一个胜出配置 |
| S4 Toys 全量 | 约 34 GPU h | 必须通过 S3 |
| S5 Beauty 全量 | 约 41 GPU h | 必须通过 Toys 全量 |
| S6 多种子/更多数据集 | 另行预算 | 不属于当前探索期 |

因此，不应直接全量开展实验。严格执行晋级门槛后，大多数失败想法会在 0–5 GPU 小时内被淘汰。

### 10.1 功效分析与总预算锁定

当前没有 UCRF 的跨 seed 方差，不能诚实地预先声称 3 seeds 已有足够 power。预算流程为：

1. S3 两个 seeds 只估计效应与方差；
2. 使用更保守的 baseline/UCRF 方差上界做 simulation-based power analysis；
3. 在 S4 之前锁定 S6 所需 seed 数、最大 GPU 小时和停止规则；
4. 若确认性预算不可承受，停止在探索性研究，不把单种子结果包装成稳定提升。

## 11. 代码与产物隔离

不直接覆盖 GRAM baseline。建议新增：

```text
GRAM/src/model/gram_ucrf.py
GRAM/command/train_gram_ucrf_{toys,beauty}_pilot.sh
GRAM/command/train_gram_ucrf_{toys,beauty}_full.sh
experiment/phase3/
artifacts/phase3/
report/第三阶段/
```

至少保存：

```text
artifacts/phase3/
├── s0/
├── s0b/
├── RELATED_WORK_MATRIX.md
├── pilot_splits/
├── experiment_registry.csv
├── promotion_decisions.md
├── configs/
├── metrics/
├── resources/
└── logs/
```

`experiment_registry.csv` 每行记录 hypothesis、唯一 config ID、commit/diff、dataset、split hash、seed、预算、状态、validation 指标、资源和晋级决定，防止选择性汇报。

### 11.1 Setup

| 项目 | 固定设置 |
|---|---|
| Working directory | `/home/jiangtangyunzhi/projects/recomm` |
| Language/framework | Python 3.9.25；PyTorch 1.11.0+cu113；Transformers 4.26.0 |
| Conda environment | `gram-repro`；不得为创新实验顺手升级依赖 |
| Backbone | T5-small |
| GPU | 默认物理 GPU3，直到用户明确改卡；通过 `CUDA_VISIBLE_DEVICES=3` 映射为逻辑 `cuda:0` |
| 默认显存门槛 | 30,720 MiB；创新模块如实测峰值更高，按 smoke 结果上调，不降低安全余量 |
| Baseline entry | `GRAM/command/train_gram_{toys,beauty}_single.sh` |
| Candidate entry | UCRF-v1 不实现；新方向需使用新入口，完整命令必须写入 registry/config，不使用口头配置 |

### 11.2 Inputs

| Input | Path | 用途/约束 |
|---|---|---|
| Beauty data | `GRAM/rec_datasets/Beauty/` | 只读；不重建 semantic ID/SASRec |
| Toys data | `GRAM/rec_datasets/Toys/` | 只读；不重建 semantic ID/SASRec |
| Beauty baseline report | `artifacts/phase1_beauty/REPRODUCTION_REPORT.md` | 锁定正式 baseline |
| Toys baseline report | `artifacts/phase2_toys/REPRODUCTION_REPORT.md` | 锁定正式 baseline |
| Best-checkpoint predictions | `GRAM/preds/*_{Beauty,Toys}_sequential_pred_{validation,test}.tsv` | S0；test 仅诊断，调参只用 validation |
| S0 summaries | `artifacts/phase3/s0/{Beauty,Toys}/validation/summary.json` | S0b 的锁定上游；不得修改或覆盖 |
| Pilot split manifests | `artifacts/phase3/pilot_splits/` | 固定用户列表、分层统计、生成脚本版本和 SHA-256 |

### 11.3 Expected Outputs 与成功条件

| Output | Path | Format | Success Criterion |
|---|---|---|---|
| S0 实际报告 | `report/第三阶段/GRAM_第三阶段_S0离线诊断报告.md` | Markdown | 已完成；双数据集 lineage、coverage、rerank、subgroup 和整体 MODIFY 决策完整 |
| S0b 结果 | `artifacts/phase3/s0b/`、`report/第三阶段/GRAM_第三阶段_S0b可靠性拒绝探针报告.md` | JSON/CSV/Markdown | 16 个锁定配置完整、同一配置跨数据集选择、无 test、明确 GO/STOP |
| 新颖性矩阵 | `artifacts/phase3/RELATED_WORK_MATRIX.md` | Markdown/CSV | 记录检索边界、日期、查询式、至少覆盖 GRAM/UNGER/GRID/LOHRec/GRUT |
| Split manifests | `artifacts/phase3/pilot_splits/` | TXT/JSON | 用户不重叠、分层统计完整、SHA-256 固定 |
| Registry | `artifacts/phase3/experiment_registry.csv` | CSV | 每个尝试一行，失败也保留，无缺失 config/seed/split/status |
| Config snapshots | `artifacts/phase3/configs/` | JSON/YAML | 能从空终端重建命令；包含代码版本和环境版本 |
| Metrics | `artifacts/phase3/metrics/` | JSON/CSV | primary/secondary/subgroup、绝对/相对差和 CI 完整 |
| Resources | `artifacts/phase3/resources/` | JSON/CSV | wall time、allocator、PID NVML、整卡背景和干扰完整 |
| Promotion record | `artifacts/phase3/promotion_decisions.md` | Markdown | 每阶段按预注册门槛明确 GO / MODIFY / STOP 及证据 |

### 11.4 Monitoring Configuration

- **软超时**：S1 15 分钟；S2 每配置 2 小时；S3 每配置 6 小时；S4/S5 72 小时。
- **硬超时**：仅用于明显失控任务，分别为软超时的 1.5–2 倍；除硬超时外不自动 kill。
- **Process monitoring**：runner PID、workload PID、进程存活、exit code、最后日志更新时间。
- **Metric monitoring**：epoch、train loss、validation NDCG@10、Recall@10；NaN/Inf/异常回退显式告警。
- **Resource monitoring**：每 5 秒 board/PID GPU，每 5 分钟磁盘；区分 PyTorch allocator、PID NVML 和 whole board。
- **Progress files**：`experiment/phase3/*status.json`、`artifacts/phase3/logs/*.log`、`experiment/phase3/*gpu*.csv`、`experiment/phase3/*disk*.csv`。
- **异常规则**：不静默 retry；先保留日志和 config，再由用户决定是否重跑。不得因指标暂时不好自动终止，只有预注册硬错误或用户确认才停止。
- **后台与 reservation**：所有预计超过几分钟的任务遵循第 0.1 节；必须使用持久 `tmux`、固定 `status` 入口，并在任何退出路径恢复 GPU3 的 CodeLlama reservation。

### 11.5 Reproducibility Classification

- 模型训练属于 **stochastic + environment-sensitive**；同 seed 也可能因 CUDA 算子和共享环境产生微差。
- 指标复跑默认使用对称相对差 `< 5%` 作为 stochastic reproducibility 参考，但 primary endpoint 的方向和晋级阈值必须单独满足；不能因为“在 5% 容差内”就宣称方法增益复现。
- 时间指标不做精确复现比较；只比较数量级、结构和资源异常，并记录外部 GPU 干扰。
- Split、config、用户数、候选数、CSV schema、checkpoint SHA-256 等确定性产物要求精确匹配。
- 每个晋级 candidate 至少保留一次从空终端命令的独立复跑记录；未复跑的结果保持 `UNVERIFIED`，统计分析但未复跑的结果最多标记 `ANALYZED`。

## 12. 第三阶段立即执行顺序（CGI E0 立项后修订）

当前按顺序执行：

1. **S0/S0b 已完成**：冻结输入哈希、32 行 registry、报告和 STOP 决策；不得扩大 S0b 网格、按数据集改规则或查看 test 挽救结果。
2. **LRC-F0 已完成**：整体 STOP；不实现 LRC-S1，不扩大 tabular 模型/特征/阈值网格。
3. **B0 已完成**：数据门槛全部通过；创新边界已收缩，不得宣称 hierarchy、ranking loss、beam hard negatives 或 popularity weighting 任一单点首创。
4. **B1 已完成**：预注册、CPU 测试与 Toys/Beauty GPU3 correctness smoke 通过；smoke 权重已丢弃，test 未读取。
5. **HBTR 10% pilot 已完成**：协议完整，8 个预注册 gate 失败，科学决策为 STOP；
   HBTR-v1 不进入 25%、全量、更多 seed 或 test。
6. **HBTR-v2 failure-autopsy 已完成**：决策为 `V2_DESIGN_ALLOWED`，HBTR-v1 STOP
   不变，不解锁 GPU。
7. **HBTR-v2 F0 已完成**：Toys PASS、Beauty FAIL，整体 `STOP_HBTR`；不得尝试
   第二套公式、显存修复或 GPU smoke。
8. **方向 C / CAMI C0-N 已完成**：Pctx 实质覆盖“上下文多 SID、单生成器、SID
   概率聚合到 item”必要核心，固定决策为 `STOP_CAMI_NOVELTY`。
9. **C0-D 未解锁且未执行**：不得实现 `cami_c0.py`、运行 `B40+A10`、扫描行为代理
   或以数据效果挽救已经失败的新颖性 gate。
10. **方向 D / NLPL D0-N 已完成**：三项新颖性必要 gate 全部通过，决策为
    `NOVELTY_SCOPE_PASS_WITH_NARROWING`。
11. **D0-D 已完成**：两数据集均仅 support gate 通过，固定决策为
    `STOP_NLPL_NO_EXPOSURE`；D1 不解锁，不尝试替代 prompt、平滑、匹配或阈值。
12. **方向 E / CGI E0-N 已完成**：决策为
    `NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_NARROWING`。
13. **E0-D 已完成**：完整性通过，但两数据集 cumulative、old-passage 与 temporal
    gate 均失败，固定决策为 `STOP_CGI_NO_INTERFERENCE`；不得运行替代
    cohort/mask/阈值，也不得进入 E1。
14. **方向 F / LEI F0-N 已完成**：三项 gate 通过，但只保留 GRAM-native
    span attribution 与 role-disambiguation 的极窄组合空间。
15. **LEI F0-D 已完成**：完整性全部通过，但双数据集 raw link harm gate 失败，
    固定决策为 `STOP_LEI_NO_RAW_ECHO`；不得改 cohort/span/阈值、把 CF-ID 升为主
    endpoint 或进入 F1。
16. **方向 G / CPBD G0-N 当前唯一解锁**：只做原文级新颖性审计；不得利用
    post-hoc span token count 直接声称机制成立，不运行 G0-D、不加载 checkpoint、
    不训练、不使用 GPU、不读取 test。
17. 任何未来新方向/GPU 任务仍需先建立独立预注册边界，并继续遵循第 0.1 节的 GPU3、
   30 GiB、tmux、status 和 CodeLlama 释放/恢复协议。

当前只解锁 G0-N 文献审计；诊断、训练、模型修改、GPU 和 test 均阻塞。

## 13. 新周期 A：Learned Reliability-Calibrated UCRF

### 13.1 Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-22
- Verification Status: ANALYZED
- Version Label: lrc_ucrf_f0_v1
- Design Status: COMPLETED WITH SCIENTIFIC DECISION STOP
- Upstream: S0/S0b 只作为问题证据，不提供本周期 GO 状态。

暂定简称 **LRC-UCRF**。核心变化不是继续调 hand-crafted threshold，而是利用训练期可得的 coverage label 学习可校准的 reliability probability；推理仍只读历史和检索统计。

### 13.2 F0 研究问题与假设（0 GPU）

研究问题：只使用历史和 relation-pool 统计，能否预测 top-20 动态关系 passage 是否覆盖下一商品，并形成非退化、可校准的 abstention 信号？

| 编号 | 假设 | F0 操作化判据 |
|---|---|---|
| F0-H1 | Coverage label 对历史检索统计是可学习的 | Beauty/Toys validation AUROC 均 ≥ 0.60，AUPRC 均 ≥ 各自 prevalence 的 1.5 倍 |
| F0-H2 | 输出概率优于常数基率且可校准 | 两数据集 Brier score 均比 constant-prevalence baseline 至少降低 5%，ECE ≤ 0.05 |
| F0-H3 | 概率能形成真正 selective gate | calibration 锁定阈值后，validation active rate 10%–40%、precision lift ≥ 1.5、positive recall ≥ 25% |

任一数据集不满足任一必要条件，F0 整体为 STOP；不能用另一数据集的强结果平均掉失败。

### 13.3 数据 lineage 与防泄漏

对每个用户只构造两个时序样本：

```text
F0 train/calibration sample:
  history = sequence[:-3][-20:]
  label_target = sequence[-3]

F0 validation sample:
  history = sequence[:-2][-20:]
  label_target = sequence[-2]

test item = sequence[-1]  # 完全不读取、不构造标签、不评估
```

训练/校准按用户 ID 的 SHA-256 确定性分成 80%/20%；不得按结果换 split。目标商品只用于生成二元标签 `y = [target in top-20 retrieved relations]`，绝不进入特征。validation 只评估一次，不用于增加模型、特征或阈值。

### 13.4 固定特征、模型与校准

特征 schema 在运行前锁定，只含：history length、union size、top relation scores、top1-top2 gap、top-20 score mean/std/min、max/mean support、multi-support fraction、anchor overlap ratio、最近 anchor agreement、pool-per-history 和 score entropy。所有量都能在推理时从历史与 SASRec top-20 邻居计算。

关系池固定为每个历史商品的 SASRec top-20 并集，候选排序固定使用 `max_j 0.9^age/log2(rank+1) + 0.25*support/history_len`，取前 20 个生成 coverage label；F0 不扫描 k、M、recency 或 beta。

仅比较三个固定 control/model，不做超参数扫描：

1. `C0`：constant prevalence；
2. `C1`：StandardScaler + class-balanced LogisticRegression（C=1，max_iter=1000，seed=2023）；
3. `C2`：class-balanced HistGradientBoosting（learning_rate=0.05，max_iter=100，max_leaf_nodes=15，L2=1，seed=2023）。

C1/C2 在 80% train users 上拟合，在 20% calibration users 上做 isotonic calibration；按 calibration Brier score 选择模型，平局优先 C1。阈值只从 calibration 预测的 active-rate `{10%,20%,30%,40%}` 四个分位点中选择：先满足 positive recall ≥25%，再最大化 precision lift；平局选择更低 active rate。每数据集允许独立校准阈值，但 feature schema、候选模型和选择规则必须相同。

### 13.5 F0 输出、门槛与后续

输出固定为：

- `artifacts/phase3/lrc_ucrf_f0/{Toys,Beauty}/dataset_summary.json`
- `artifacts/phase3/lrc_ucrf_f0/model_metrics.csv`
- `artifacts/phase3/lrc_ucrf_f0/feature_schema.json`
- `artifacts/phase3/lrc_ucrf_f0/summary.json`
- `report/第三阶段/GRAM_第三阶段_LRC-UCRF_F0可学习性报告.md`
- `artifacts/phase3/promotion_decisions.md` 新增 `LRC-F0 → LRC-S1`。

F0 只回答“reliability 是否可学习”，不回答推荐效果是否提升。只有 F0 三个假设在 Beauty/Toys 全部通过，才允许设计 LRC-S1：identity-initialized signed gate、coverage auxiliary loss、calibration regularizer 和 100–500 样本 smoke。F0 STOP 时，不训练 T5、不扩大 tabular 模型/特征网格，转向方向 B。

执行入口：`bash experiment/phase3/run_phase3_lrc_f0.sh start`；查询入口：`bash experiment/phase3/run_phase3_lrc_f0.sh status`。完整任务使用持久 `tmux`，不占 GPU。

### 13.6 F0 实际结果与终止决策

F0 于 2026-07-22 完成，耗时 17.82 秒，GPU 使用为 0，科学决策为 **STOP**。

| 数据集 | AUROC | AUPRC lift | Brier 改善 | ECE | 锁定阈值 Active rate | Precision lift | Positive recall | 未通过项 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Toys | 0.6133 | 1.725× | 1.776% | 0.0466 | 26.180% | 1.572× | 41.156% | Brier 改善 < 5% |
| Beauty | 0.7064 | 3.045× | 12.992% | 0.1235 | 16.845% | 2.656× | 44.737% | ECE > 0.05 |

证据表明两数据集都有一定 discrimination，但跨数据集的概率质量不稳定：Toys 对常数基率的改善不足，Beauty 的校准误差过高。这只支持“信号存在但尚不能作为可靠门控”，不支持启动 LRC-S1。

产物：`artifacts/phase3/lrc_ucrf_f0/summary.json`、`report/第三阶段/GRAM_第三阶段_LRC-UCRF_F0可学习性报告.md`。

## 14. 新周期 B：Hierarchical Beam-hard-negative + Tail-aware Ranking（HBTR）

### 14.1 Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-22
- Verification Status: UNVERIFIED
- Version Label: hbtr_b0_v1
- Design Status: RESULT-INFORMED NEW CYCLE AFTER LRC-F0 STOP
- Upstream: S0 的 beam-50 上界与 tail gap 只用于问题定位；UCRF/LRC 不提供 GO 状态。

### 14.2 核心问题与暂定设计

新方向不再向输入中增加关系 passage，而是直接优化“正确商品已进入 constrained beam，但被相似 semantic-ID 商品压在 top-10 之外”的排序错误。B0 文献审计后，暂定方法名仍为 **HBTR**，但候选差异必须收缩为“student beam error × lexical-prefix confusion × training-only popularity”的联合 margin：

1. 从当前模型 beam 内选取高分错误商品作为 hard negatives，而不是随机负样本；
2. 用 hierarchical semantic ID 的最长公共前缀深度表示混淆程度，越接近正样本分支的负样本权重/间隔越高；
3. 对正样本使用由训练交互频次得到的 capped tail weight，禁止读取 validation/test 流行度；
4. 候选训练目标为原 generative CE 加一个 beam-aware pairwise/listwise ranking loss；不改 semantic ID、Trie 或 full-ranking 评测协议。

此结构目前是**研究假设**，不是已确认的论文创新。LOHRec 已覆盖 hierarchy + ranking loss，OneRec/WPAUC 已覆盖 beam hard negatives，Token-Weighted Multi-Target Learning 已覆盖 prefix/frequency weighting；因此不得宣称这些单独元素首创。后续只能通过组件对照证明联合 margin 有非叠加价值。

### 14.3 B0：差异审计与 0-GPU 可行性诊断

B0 不训练，分为两个必须部分：

- **B0-Lit**：建立 2024–2026 工作矩阵，至少比较 LOHRec、MERGE、popularity-aware contrastive recommendation 及相关生成式推荐方法的 negative source、semantic hierarchy、beam awareness、tail objective、full-ranking 协议与代码可用性。
- **B0-Diag**：仅读取已锁定 Beauty/Toys validation beam-50 预测、历史训练流行度和 semantic ID；报告 target 在 rank 11–50 的比例、head/tail beam recall@10/50 gap、错误 top-10 与 target 的前缀深度、可能的 oracle NDCG@10 上界。不扫描损失权重，不查看 test。

B0 的锁定 GO 条件为：

1. 差异审计未发现与“beam hard negatives + semantic-prefix weighting + tail-aware objective”实质相同的方法；若已有，必须能明确写出非文字性的新差异，否则 STOP。
2. Beauty 和 Toys 的 beam-50 target recall 都至少比 Recall@10 高 5 个绝对百分点，且 tail 用户中 target 在 rank 11–50 的用户数均不少于 200。
3. 两数据集在 miss@10/hit@50 子集中，至少 25% 的 target 与一个错误 top-10 候选共享非空 semantic-ID 前缀，证明 hierarchical hard negative 不是空集机制。
4. 两数据集的 oracle 重排 NDCG@10 相对 baseline 都有至少 5% 的上界空间。

任一条件失败则 B0 STOP，不实现 B1，不用 GPU “试试看”。B0 GO 后才预注册 B1 的唯一个 loss 配置、负样本数、tail-weight cap 与 <15 分钟正确性 smoke；然后才进入 10% 用户微型对照，不直接全量。

### 14.4 B0 实际结果

B0 于 2026-07-22 完成，诊断耗时 6.64 秒，GPU 使用为 0，整体决策为 **GO WITH NOVELTY NARROWING**。

| 数据集 | Recall@10→50 gap | miss@10/hit@50 | Tail 样本 | 共享前缀率 | Oracle NDCG@10 相对空间 | 数据门槛 |
|---|---:|---:|---:|---:|---:|---|
| Toys | +9.252 pp | 1,796 | 683 | 81.180% | +177.853% | PASS |
| Beauty | +9.936 pp | 2,222 | 338 | 58.911% | +220.301% | PASS |

Oracle 是将 beam-50 中 target 人为提至第 1 位的不可达理想上界，不是效果预测。Beauty tail 的 Recall@10→50 gap 仅 +2.994 pp，后续即使排序成功也不能宣称解决了全部长尾召回问题。

产物：`artifacts/phase3/hbtr_b0/diagnostic_summary.json`、`artifacts/phase3/hbtr_b0/literature_matrix.csv`、`report/第三阶段/GRAM_第三阶段_HBTR_B0可行性与差异审计报告.md`。

### 14.5 B1：联合 margin 预注册与正确性 smoke

B1 于 2026-07-22 在任何 GPU smoke 之前锁定。完整机器可读配置为
`artifacts/phase3/configs/hbtr_b1_preregistered.json`，本节为对应的人类可读边界。

#### 14.5.1 锁定假设与得分

B1 只检验如下狭化假设：在不增加独立 ranker/RL、不改 ID/Trie 的情况下，
对 matched baseline 的 `miss@10/hit@50` 训练错误使用联合 margin，能否修正
lexical-prefix 混淆且不忽略尾部目标。序列得分固定为包含 EOS、排除 padding 后的
teacher-forced mean log probability。

```text
wp = 1 + min(LCP(y+, y-), 3) / 3
wt = 1 + min(1, max(0, log((median_train_frequency + 1) /
                          (positive_train_frequency + 1))))
m  = 0.1 * wp * wt
Lrank = mean softplus(m + score(y-) - score(y+))
L = L_token_CE + 0.1 * Lrank
```

`wp` 和 `wt` 均封顶为 2，因此 margin 封顶为 0.4。不扫描 margin、lambda、
prefix cap 或 popularity cap。

#### 14.5.2 负样本与缓存

- 每个有效样本固定 `K=4`，从 matched baseline constrained beam-50 的错误 top-10 中按得分选取。
- 只有 target rank 在 11–50 时启用 `Lrank`；target 已进 top-10 或 miss@50 时不构造排序伪标签。
- 排除 target、重复商品、用户已知历史商品和 Trie 外商品。
- 缓存由 matched baseline checkpoint 生成一次，训练期间不刷新；保存 checkpoint/split/input/cache SHA-256。
- popularity 只由每个用户 `sequence[:-2]` 统计，不允许读取 validation/test target。

#### 14.5.3 组件对照与 B1 边界

后续 pilot 的锁定组件对照为 `C0 token CE`、`C1 unweighted beam pairwise`、
`C2 prefix-only`、`C3 popularity-only`、`C4 joint margin`。B1 smoke 只运行 C4，
但 CPU 测试必须覆盖各分支的边界与 `lambda=0` 回退。

B1 最多使用每数据集 100 个样本、GPU3 各 15 分钟。smoke 只验证 cache、
forward/backward、checkpoint reload 和 constrained full-ranking 链路；smoke 权重不得用于 pilot，
smoke 指标不得用于效果、机制或论文主张。

#### 14.5.4 B1 → 10% pilot 必要门槛

1. CPU 测试全部通过，包括泄漏、封顶、单调性、空负样本和缓存哈希。
2. `lambda=0` 时 loss/logits/gradient 在锁定容差内回到 baseline。
3. Toys/Beauty 均完成 forward/backward，无 NaN/Inf，且有非零有效 ranking pair。
4. checkpoint 保存/重载后一致，beam-50 输出均是合法 Trie 商品。
5. 不读 test，不生成 10% pilot split，不根据 smoke 指标改变配置。
6. GPU3/tmux/status/telemetry/CodeLlama 释放与恢复协议完整。

任一必要门槛失败时，B1 为 **BLOCKED/REPAIR**，不静默重试，不进入 pilot。

### 14.6 B1 实际结果与晋级决策

B1 于 2026-07-22 完成，正确性决策为 **PASS FOR PILOT DESIGN**。第三阶段
16/16 项 CPU 单元测试通过；Toys/Beauty 各用 100 个 training-only 样本挖掘静态 beam-50 cache。

| 数据集 | 有效 cache 行 | 优化步 | wall time | Peak reserved | Checkpoint reload diff | 权重分支覆盖 |
|---|---:|---:|---:|---:|---:|---|
| Toys | 12 | 2 | 30.51 s | 15,020 MiB | 0.0 | joint + prefix |
| Beauty | 21 | 2 | 34.87 s | 17,982 MiB | 0.0 | prefix + tail |

首次 smoke 链路通过但 Beauty 的前两行 cache 未覆盖非平凡权重；该产物已保留。
repair smoke 只改为确定性优先选择 joint/prefix/tail correctness rows，不改预注册配置或 cache。

两数据集的 loss/gradient 均有限且非零，临时 checkpoint 重载精确一致；smoke 权重已丢弃，
test 未读，pilot split 未生成。这只证明实现可运行，不证明 NDCG/Recall/tail 改善。

完整报告：`report/第三阶段/GRAM_第三阶段_HBTR_B1正确性Smoke报告.md`。

### 14.7 HBTR 10% pilot 预注册

pilot 在生成 split 与启动 GPU 之前锁定。机器可读配置为
`artifacts/phase3/configs/hbtr_pilot_preregistered.json`。它是从锁定全量 baseline checkpoint
继续训练的**探索性机制筛选**，不是从零训练或论文级确认。

#### 14.7.1 Split

- 训练用户：全部可用用户的 10%，seed 2023；
- validation：从其余用户中固定 2,048 人，与训练用户零重叠；
- 分层：只使用 `target=sequence[-3]` 的 training-only head/tail 与
  `history=sequence[:-3]` 长度档 `1–5/6–10/11–20/21+`；
- popularity 只由 `sequence[:-2]` 统计；split 不读 validation/test target；
- 用户文件、strata CSV 和输入均保存 SHA-256，生成后禁止更换。

#### 14.7.2 C0–C4 和统一预算

| 配置 | 目标 |
|---|---|
| C0 | token CE only |
| C1 | CE + unweighted beam pairwise margin |
| C2 | CE + prefix-only margin |
| C3 | CE + popularity-only margin |
| C4 | CE + joint prefix×popularity margin |

所有配置使用同一锁定 baseline checkpoint、split、seed 2023、5 epochs、
AdamW `lr=1e-5`、batch 16、gradient accumulation 8，只在 epoch 5 validation。
C1–C4 共享由 baseline 生成一次的静态 beam-50 cache，`K=4`；C0 忽略 cache。
评测保持全商品 Trie 和 beam-50，不读 test。

#### 14.7.3 主终点与晋级

主终点为 validation NDCG@10；次终点为 Recall@5/NDCG@5/Recall@10，
另报告 head/tail/history bins 和 10,000 次 user-level paired bootstrap。

GO 的全部必要条件：

1. C4 vs C0 在两数据集 NDCG@10 方向均为正，且至少一个相对增益≥2%；
2. 两数据集 Recall@10 均不下降；tail Recall@10/NDCG@10 相对下降均不超过 1%；
3. C4 的双数据集 macro NDCG@10 高于 C1/C2/C3，且每数据集不低于最优单组件超过 0.5%；
4. 相对 C0，peak reserved 增加≤25%，训练 wall time 增加≤100%，validation latency 增加≤5%；
5. lineage、cache、split、checkpoint 和输出完整，无 test 泄漏。

只有全部通过才为 GO；数据完整且只有一项可修复的近失为 MODIFY，
其余为 STOP。pilot 不允许任何效果或论文主张。

### 14.8 HBTR 10% pilot 实际结果与终止决策

HBTR 10% pilot 于 2026-07-23 至 2026-07-24 完成。Toys/Beauty 的 C0–C4 均从同一
锁定 baseline checkpoint、同一 split 和 seed 继续训练 5 epochs，并在各自固定 2,048
validation 用户上做 full-item Trie、beam-50 评测。数据、split、cache、checkpoint 与
输出完整，未读取 test；科学决策为 **STOP**。

| 数据集 | C0 NDCG@10 | C4 NDCG@10 | 相对变化 | C0 Recall@10 | C4 Recall@10 | C4 vs C0 bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Toys | 0.079372 | 0.079882 | +0.643% | 0.123047 | 0.124512 | [+0.000039, +0.001096] |
| Beauty | 0.067215 | 0.067174 | -0.061% | 0.114746 | 0.114258 | [-0.000967, +0.000871] |

失败项共 8 个，分为四类：

1. **实际效应**：两数据集均未达到 2%；Beauty NDCG@10 方向为负。
2. **安全性**：Beauty Recall@10 下降，tail Recall@10/NDCG@10 分别相对下降
   1.67%/1.20%，超过 1% 界限。
3. **组件证据**：C4 双数据集 macro NDCG@10 未超过 C2，联合 margin 的非叠加价值
   未获支持。
4. **资源**：C4 peak reserved 相对 C0 增加 Toys 85.60%、Beauty 61.31%，超过 25%；
   训练 wall time 与 validation latency 门槛通过。

因此 HBTR-v1 不允许进入 25% 中型、Toys/Beauty 全量、更多 seed 或 test。Toys 的
小幅正向结果只保留为“generic beam pairwise 可能存在弱信号”的探索性证据，不构成
效果、稳定性或联合机制主张。完整报告与机器结果分别为
`report/第三阶段/GRAM_第三阶段_HBTR_10%Pilot报告.md` 和
`artifacts/phase3/hbtr_pilot/summary.json`。

### 14.9 HBTR-v2 failure-autopsy 预注册（0 GPU，post-hoc exploratory）

#### 14.9.1 边界与研究问题

本周期在看到 HBTR-v1 pilot 结果后建立，不能追溯写成事前假设。它不寻找使当前结果
转正的超参数，而回答两个更窄的问题：

1. HBTR-v1 的 `prefix × popularity` 联合 margin 是否因非平凡激活率过低而实际上
   无法与 C1/C2/C3 区分？
2. 现有结果是否同时保留足够的 beam-hard-negative 支持与至少一个数据集的弱正向
   排名信号，使设计一个新周期仍有最低合理性？

只读取以下既有产物：

- `artifacts/phase3/hbtr_pilot/{Toys,Beauty}/cache/negative_cache.json`
- `artifacts/phase3/hbtr_pilot/{Toys,Beauty}/C0/validation_per_user.csv`
- `artifacts/phase3/hbtr_pilot/{Toys,Beauty}/C1/validation_per_user.csv`
- `artifacts/phase3/hbtr_pilot/{Toys,Beauty}/C4/validation_per_user.csv`
- `artifacts/phase3/hbtr_pilot/{Toys,Beauty}/C*/training_summary.json`
- `artifacts/phase3/hbtr_pilot/summary.json`
- `artifacts/phase3/configs/hbtr_pilot_preregistered.json`
- `GRAM/rec_datasets/{Toys,Beauty}/user_sequence.txt`，仅以 `sequence[:-2]`
  重建 training-only popularity。

禁止读取任何 test prediction/target，禁止加载模型 checkpoint，禁止训练、重新生成
beam、扫描 margin/lambda/threshold 或改写现有 pilot 文件。

#### 14.9.2 固定诊断量

每个数据集固定计算：

- `eligible_all_rate`：有效 rank11–50 cache 行 / 全部 pilot 训练样本；
- `prefix_nontrivial_pair_rate` 与 `prefix_nontrivial_row_rate`；
- `tail_nontrivial_row_rate`：按现有实现 `frequency < training popularity median`
  会产生 `tail_weight > 1` 的 cache 行比例；
- `joint_nontrivial_row_rate`：同一 cache 行同时具有非零 prefix 和非平凡 tail weight；
- 上述三类非平凡行相对全部训练样本的覆盖；
- C1–C4 margin 的均值、标准差、分位数、相等率和绝对差；
- C0→C1、C0→C4、C1→C4 的用户 rank 上升/下降、进入/退出 top-10 与 subgroup 转移；
- 既有 NDCG@10、Recall@10、bootstrap CI 和资源比，只读取、不重算 gate。

#### 14.9.3 诊断阈值与唯一决策

机制支持下限在运行脚本前锁定：

| 条件 | 两数据集要求 | 含义 |
|---|---:|---|
| `eligible_all_rate` | ≥15% | 有足够 beam miss@10/hit@50 训练支持 |
| `prefix_nontrivial_row_rate` | ≥25% | hierarchy 分支不是稀有事件 |
| `tail_nontrivial_row_rate` | ≥20% | tail 分支能影响足够有效行 |
| `joint_nontrivial_row_rate` | ≥10% | C4 能与单组件形成可辨识差异 |
| generic signal | 至少一个数据集 C1 或 C4 vs C0 NDCG@10 >0，且 top-10 净迁入 >0 | 仅证明存在弱方向性信号 |

决策规则：

- `V2_DESIGN_ALLOWED`：两数据集均通过 eligible/prefix 支持；当前 tail 或 joint
  可辨识性门槛失败；同时 generic signal 通过。只允许设计新的 HBTR-v2，不解锁 GPU。
- `STOP_HBTR`：任一数据集 eligible/prefix 支持失败，或 generic signal 失败；或者
  tail/joint 已有充分激活但效果仍未过 HBTR-v1 gate，说明不能再用“未激活”解释失败。
- 不设置 `GO` 或 `MODIFY`，不允许根据输出追加第三种有利判定。

执行命令锁定为：

```bash
python3 experiment/phase3/hbtr_v2_failure_autopsy.py
```

预期输出：

- `artifacts/phase3/configs/hbtr_v2_autopsy_preregistered.json`
- `artifacts/phase3/hbtr_v2_autopsy/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/hbtr_v2_autopsy/activation_metrics.csv`
- `artifacts/phase3/hbtr_v2_autopsy/rank_transitions.csv`
- `artifacts/phase3/hbtr_v2_autopsy/summary.json`
- `report/第三阶段/GRAM_第三阶段_HBTR_v2失效解剖报告.md`

确定性成功条件为：输入 SHA-256 完整、两个数据集行数/用户 lineage 一致、输出 schema
通过单元测试、`test_data_read=false`、退出码为 0。该分析预计低于 1 分钟，硬超时 10
分钟，不需要 tmux/GPU/CodeLlama reservation 操作。

#### 14.9.4 实际结果

failure-autopsy 于 2026-07-24 按锁定命令完成，CPU 单元测试 4/4 通过，退出码为 0；
两个数据集 validation 均为 2,048 用户、lineage mismatch 为 0，输入 SHA-256 完整，
`test_data_read=false`。

| 数据集 | eligible/all | prefix 非平凡/有效行 | tail 非平凡/有效行 | joint 非平凡/有效行 | C4 vs C0 NDCG@10 | top-10 净迁入 |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 21.78% | 44.01% | 8.77% | 5.46% | +0.643% | +3 |
| Beauty | 18.11% | 32.98% | 3.84% | 2.41% | -0.061% | -1 |

两数据集均通过 eligible ≥15% 与 prefix-row ≥25%；均未通过 tail-row ≥20% 和
joint-row ≥10%。Toys 的 generic signal 通过，Beauty 失败，因此固定决策为
**V2_DESIGN_ALLOWED**。

进一步的可辨识性证据为：C4 与 C1 的 pair margin 完全相等率在 Toys 为 67.22%，
Beauty 为 81.99%；C4 与 C2 的完全相等率分别为 91.23% 和 96.16%。validation 排名上，
C1→C4 仅改变 Toys 8/2,048 与 Beauty 6/2,048 个用户，两个数据集均没有产生 top-10
净迁移。这支持“现有联合权重多数时候没有形成可观测差异”，但不证明提高激活率一定会
提升推荐效果。

产物：

- `artifacts/phase3/hbtr_v2_autopsy/summary.json`
- `artifacts/phase3/hbtr_v2_autopsy/activation_metrics.csv`
- `artifacts/phase3/hbtr_v2_autopsy/rank_transitions.csv`
- `report/第三阶段/GRAM_第三阶段_HBTR_v2失效解剖报告.md`

### 14.10 HBTR-v2 后续边界

若 failure-autopsy 为 `V2_DESIGN_ALLOWED`，下一轮仍须先建立独立机器可读预注册，至少：

1. 用 training-only popularity quantile 定义与 head 20% / tail 80% 评测一致的连续
   tail 权重，并在任何 GPU 前证明 tail/joint 激活率过门槛；
2. 先做数值等价的 negative decoder micro-batching，单独验证 loss/gradient 与原实现
   一致及 peak reserved ≤25%，不能把资源修复与效果改法混在同一个 smoke；
3. 只允许一个固定 HBTR-v2 loss、一个 seed 和一次小型 validation；不得沿用 HBTR-v1
   的 GO 状态，也不得查看 test。

若为 `STOP_HBTR`，第三阶段结束 HBTR 路径，后续必须重新选择方向 C 并建立全新研究
问题、差异审计和预注册周期。

### 14.11 HBTR-v2 F0：Quantile-tail margin 可辨识性预注册

#### 14.11.1 Material Passport 与研究边界

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-24
- Verification Status: UNVERIFIED
- Version Label: hbtr_v2_f0_preregistered_v1
- Design Status: RESULT-INFORMED NEW CYCLE LOCKED BEFORE F0 EXECUTION

F0 只回答：一个与论文评测 head 20% / tail 80% 定义一致的 training-only quantile
权重，能否在不改变 beam cache、prefix 权重、base margin 或负样本的情况下，使
tail/joint 分支达到预先规定的非平凡激活率并与 prefix-only margin 可区分。

F0 不回答推荐效果，不加载 checkpoint，不训练，不生成 beam，不读取任何 validation
per-user/summary 或 test 文件，也不允许根据 F0 输出尝试第二套公式。

#### 14.11.2 唯一锁定公式

对每个数据集，仅用全体用户 `sequence[:-2]` 统计商品频次。商品按
`(-frequency, item_id)` 确定性排序，设商品数为 `N`，
`H = ceil(0.2 * N)`，1-based popularity rank 为 `r_i`：

```text
q_tail(i) = 0                                      if r_i <= H
            (r_i - H) / (N - H)                    if r_i > H

wt_v2(i) = 1 + q_tail(i)
wp(i+, i-) = 1 + min(LCP(i+, i-), 3) / 3

C1: m = 0.1
C2: m = 0.1 * wp
C3-v2: m = 0.1 * wt_v2
C4-v2: m = 0.1 * wp * wt_v2
```

因此 head 权重严格为 1，tail 权重位于 `(1, 2]`，margin 位于 `[0.1, 0.4]`。
排序中的 item-ID tie break 只用于使 top 20% item 集合与既有 `head_items` 评测实现
完全一致；不得按结果调整 head 比例、变换函数或 cap。

#### 14.11.3 输入、输出与完整性

只允许读取：

- `artifacts/phase3/hbtr_pilot/{Toys,Beauty}/cache/negative_cache.json`
- `GRAM/rec_datasets/{Toys,Beauty}/user_sequence.txt`，且仅使用 `sequence[:-2]`
- `artifacts/phase3/configs/hbtr_v2_f0_preregistered.json`
- `artifacts/phase3/hbtr_v2_autopsy/summary.json`，只验证上游决策确为
  `V2_DESIGN_ALLOWED`，不得读取其中 validation 效果字段选择公式。

执行命令锁定为：

```bash
python3 experiment/phase3/hbtr_v2_f0.py
```

预期输出：

- `artifacts/phase3/hbtr_v2_f0/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/hbtr_v2_f0/activation_metrics.csv`
- `artifacts/phase3/hbtr_v2_f0/margin_distributions.csv`
- `artifacts/phase3/hbtr_v2_f0/summary.json`
- `report/第三阶段/GRAM_第三阶段_HBTR_v2_F0可辨识性报告.md`

完整性要求：cache 行与 training-only popularity 一致、公式单调性/cap/head-tail 对齐
单元测试通过、输入 SHA-256 完整、`validation_effect_data_read=false`、
`test_data_read=false`、退出码为 0。

#### 14.11.4 固定 gate 与后续

两数据集必须同时满足：

1. `tail_nontrivial_row_rate >= 20%`；
2. `joint_nontrivial_row_rate >= 10%`；
3. `C4-v2 vs C2 pair-margin exact-equality rate <= 80%`；
4. head 行 `wt_v2=1`、tail 行 `wt_v2>1`、最稀有商品 `wt_v2=2`；
5. `wt_v2` 随 popularity rank 单调不减，所有 margin `<=0.4`。

全部通过才为 `PASS_FOR_RESOURCE_REPAIR_DESIGN`，只允许预注册并实现数值等价的
negative-decoder micro-batching。任一失败为 `STOP_HBTR`，不得尝试第二套 quantile
公式或启动 GPU。F0 不设置 MODIFY。

#### 14.11.5 F0 实际结果与 HBTR 终止

HBTR-v2 F0 于 2026-07-24 按锁定命令完成。纯 CPU 单元测试 4/4 通过，cache 与
training-only popularity mismatch 为 0，输入 SHA-256 完整；
`validation_effect_data_read=false`、`test_data_read=false`。

| 数据集 | tail 非平凡/有效行 | joint 非平凡/有效行 | C4-v2=C2 pair 率 | 最大 tail 权重 | 最大 margin | 数据集 gate |
|---|---:|---:|---:|---:|---:|---|
| Toys | 32.61% | 18.25% | 67.39% | 2.000 | 0.380 | PASS |
| Beauty | 15.79% | 8.91% | 84.21% | 2.000 | 0.332 | FAIL |

Beauty 同时失败三个必要可辨识门槛：

1. tail 非平凡行率 15.79%，低于 20%；
2. joint 非平凡行率 8.91%，低于 10%；
3. C4-v2 与 C2 pair margin 完全相等率 84.21%，高于 80%。

公式自身的 head/tail 对齐、单调性、tail cap、最稀有商品到达 cap、margin cap 和
cache lineage 均通过。失败含义不是公式实现错误，而是 Beauty 的有效 beam-hard-negative
cache 目标过度集中于 popularity head；即使把 item-level tail 80% 全部赋予连续权重，
联合分支在真正 eligible 样本上仍达不到锁定的最低可辨识覆盖。

整体固定决策为 **STOP_HBTR**。不得尝试第二套 quantile 公式，不实现 negative-decoder
micro-batching，不启动 correctness smoke/GPU，不进入新 pilot、更多 seed、全量或 test。
如继续第三阶段，必须转向方向 C 并建立全新周期。

产物：

- `artifacts/phase3/hbtr_v2_f0/summary.json`
- `artifacts/phase3/hbtr_v2_f0/activation_metrics.csv`
- `artifacts/phase3/hbtr_v2_f0/margin_distributions.csv`
- `report/第三阶段/GRAM_第三阶段_HBTR_v2_F0可辨识性报告.md`

## 15. 参考工作

- GRAM：<https://aclanthology.org/2025.acl-long.1596/>
- TIGER：<https://papers.nips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf>
- MINDER：<https://aclanthology.org/2023.acl-long.366/>
- ActionPiece：<https://proceedings.mlr.press/v267/hou25f.html>
- EAGER：<https://arxiv.org/abs/2406.14017>
- LETTER：<https://arxiv.org/abs/2405.07314>
- CoST：<https://doi.org/10.1145/3640457.3688178>
- RPG：<https://arxiv.org/abs/2506.05781>
- MTGRec：<https://arxiv.org/abs/2504.04400>
- Pctx：<https://arxiv.org/abs/2510.21276>
- PIT：<https://arxiv.org/abs/2602.08530>
- Progressive Collaborative and Semantic Knowledge Fusion / UNGER：<https://arxiv.org/abs/2502.06269>
- GRID / Generative Recommendation with Semantic IDs: A Practitioner's Handbook：<https://arxiv.org/abs/2507.22224>
- LOHRec：<https://aclanthology.org/2025.findings-emnlp.977/>
- GRUT / Enhancing Time Awareness in Generative Recommendation：<https://aclanthology.org/2025.findings-emnlp.1300/>
- APAO / Aligned Prefix-Aware Optimization：<https://arxiv.org/abs/2603.02730>
- Expressiveness Limits of Generative Recommendation / Latte：<https://arxiv.org/abs/2605.06331>
- Token-Weighted Multi-Target Learning：<https://arxiv.org/abs/2601.17787>
- Echoes in the Filter Bubble / Ghost Tokens：<https://arxiv.org/abs/2605.16825>
- Decoding Matters：<https://aclanthology.org/2024.emnlp-main.589/>
- Calibrate Before Use：<https://proceedings.mlr.press/v139/zhao21c.html>
- Generative Retrieval with Structured Term IDs / GRLM：<https://aclanthology.org/2026.findings-acl.984/>
- Causality-Enhanced Behavior Sequence Modeling / CFT：<https://arxiv.org/abs/2410.22809>
- From Past to Path / Masked History Learning：<https://arxiv.org/abs/2509.23649>
- RAGONITE counterfactual passage attribution：<https://arxiv.org/abs/2412.10571>
- Multi-Granularity Guided Fusion-in-Decoder：<https://aclanthology.org/2024.findings-naacl.142/>
- Beyond Unimodal Boundaries / MGR-LF++：<https://arxiv.org/abs/2503.23333>
- LWGR personalized world-knowledge fusion：<https://arxiv.org/abs/2605.18771>

这些文献只完成了首轮定位。任何“论文创新”声明都必须以正式相关工作矩阵和更全面检索为准。

## 16. 新周期 C：Context-Adaptive Multi-View Identifiers（CAMI）

### 16.1 Material Passport 与当前边界

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-24
- Verification Status: ANALYZED（C0-N）；UNVERIFIED（C0-D 未执行）
- Version Label: cami_c0_novelty_stopped_v2
- Design Status: STOP_CAMI_NOVELTY; C0-D NOT UNLOCKED

CAMI 的暂定研究问题为：

> 在总解码候选预算固定为 50 时，为 candidate-starved item 提供稀疏、互补且可回收到
> 同一 item 的上下文行为 alias，是否可能缓解生成式推荐中的 long-tail candidate
> starvation？

当前只把它当作**待证伪的研究方向**，不把下列设计当作已经成立的贡献。C0 通过前：

1. 不实现多 alias decoder、item-level marginal likelihood 或新 Trie；
2. 不训练、不加载模型 checkpoint 做新推理、不使用 GPU；
3. 不读取任何 test prediction/test metric，不在任何统计、候选或标签中使用
   `sequence[-1]`；
4. 不宣称 CAMI 优于 GRAM、MINDER、EAGER、ActionPiece、LETTER、CoST 或 RPG；
5. 不根据 C0 的中间输出改公式、换 quota 或增加第二个行为代理。

CAMI 与已终止 HBTR 的边界是：HBTR 调整已经进入 beam 的候选之间的训练排序；CAMI
研究的是 target 为什么没有进入 beam，以及一个 item 是否应有不止一条合法生成入口。

### 16.2 暂定机制（C0 通过前不实施）

方向 C 的最小构想是让 item `i` 拥有一个或少量合法 identifier：

```text
A(i) = {原 GRAM semantic lexical ID}
       ∪ {仅对候选饥饿/前缀拥堵 item 分配的 contextual behavioral alias}

L_alias(i | h) = -log sum_{a in A(i)} p(a | h)
item_score(i)  = logsumexp_{a in A(i)} path_score(a)
```

暂定约束为单 decoder、单 constrained-decoding Trie、多 alias 到单 item 的确定性回收，
且比较时总候选数固定；不采用两个 decoder 的置信融合，不以增加 beam 数量替代机制。
这些只是后续可能设计，不是 C0 的被试模型，也不因写入本计划而解锁。

### 16.3 C0 总览：先检查“值得设计”，再检查“能否实现”

C0 分成两个串行 gate，均为 0 GPU：

```text
C0-N：正式差异/新颖性审计
  -> PASS 才进入 C0-D
C0-D1：数据 lineage、基线复算与 Semantic-ID 拥堵诊断
  -> PASS 才读取行为代理结果
C0-D2：training-only 行为视图互补性与固定预算候选测试
  -> 全部 PASS 才为 C0_DESIGN_ALLOWED
  -> 任一必要项失败即 STOP_CAMI
```

C0-N 与 C0-D 可以由同一实现周期准备，但判定必须按上述顺序输出。C0-D 的目的只是
检验候选支持集是否存在足够空间；即使通过，也不证明 alias 能被 decoder 学会。

### 16.4 C0-N：文献差异审计（详细锁定）

#### 16.4.1 审计范围

至少建立以下工作的 claim-by-component 矩阵：GRAM、TIGER、MINDER、ActionPiece、
EAGER、LETTER、CoST、RPG、LOHRec，以及 Better Generalization with Semantic IDs。
只接受论文原文、正式 proceedings 或 DOI 页面；搜索摘要只能发现文献，不能作为
“尚无人做过”的证据。

每篇至少记录：

| 字段 | 内容 |
|---|---|
| representation unit | 单一 ID、多 ID、sub-piece、set/action token 或长 ID |
| context dependence | identifier 是否随用户历史/上下文变化 |
| view construction | semantic、collaborative、behavioral 或 textual |
| allocation | 所有 item 统一分配，还是只对拥堵/饥饿 item 自适应分配 |
| training objective | 单路径 CE、多路径 marginalization、contrastive 或双流目标 |
| inference | 单/双 decoder、Trie 约束、alias-to-item aggregation、beam budget |
| closest overlap | 与 CAMI 最接近的已发表 claim |
| remaining distinction | 有原文支持的差异；禁止只写措辞差异 |

#### 16.4.2 C0-N gate

以下三条必须在原文级审计后仍同时成立：

1. **adaptive allocation**：已有工作未同时按 item candidate starvation / prefix
   congestion 稀疏分配多视图 alias；
2. **single-decoder item marginalization**：已有工作未在单 decoder 生成推荐中同时
   使用多条合法路径训练并在 item 层聚合；
3. **fixed-budget mechanism claim**：研究主张明确是固定候选预算下的 tail candidate
   recovery，而不是更长 ID、更大 beam、双 decoder 或 beam 内 reranking。

全部成立记为 `NOVELTY_SCOPE_PASS`。任一条被已有工作实质覆盖，记为
`STOP_CAMI_NOVELTY`：停止当前 CAMI 表述，先重新定义研究问题，不进入 C0-D 数据
结果，也不得把措辞改写当作新颖性修复。

预期产物：

- `artifacts/phase3/cami_c0/novelty_matrix.csv`
- `artifacts/phase3/cami_c0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_CAMI_C0差异审计.md`

### 16.5 C0-D 输入、lineage 与禁止项（详细锁定）

#### 16.5.1 允许输入

每个数据集只允许：

- 锁定的 baseline validation prediction：
  `GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv` 与
  `GRAM/preds/20260722_125916_Beauty_sequential_pred_validation.tsv`；
- `GRAM/rec_datasets/{Toys,Beauty}/user_sequence.txt`；
- 唯一匹配的
  `GRAM/rec_datasets/{dataset}/item_generative_indexing_hierarchy_*.txt`；
- `artifacts/phase3/s0/{dataset}/validation/summary.json`，只用于复算一致性；
- 执行前新建并锁定的
  `artifacts/phase3/configs/cami_c0_preregistered.json`。

实现必须记录所有输入 SHA-256，并验证 validation gold 等于 `sequence[-2]`。构造
popularity、transition count、行为候选和 alias 资格时只能读取每条
`sequence[:-2]`；`sequence[-2]` 只作为 validation label。数据解析器虽需打开完整
sequence 行，但必须在进入任何下游统计前丢弃 `sequence[-1]`，并记录
`test_target_used=false`。test prediction 文件不得出现在打开文件清单中。

#### 16.5.2 固定定义

1. **head/tail**：只用所有 `sequence[:-2]` 的 item 频次，按
   `(-frequency, item_id)` 排序；top 20% item 为 head，其余 80% 为 tail。
2. **baseline candidate set `B50`**：锁定 validation prediction 的去重后前 50 个
   合法 item；`B40` 同理取前 40。不得重新生成 beam。
3. **一级 Semantic-ID prefix**：index 文件中 item 的第一个 lexical token。
4. **prefix congestion**：同一一级 prefix 下的 item 数。按全部 item 的该数值中位数
   固定划分 high/low congestion；等于中位数归 high。
5. **candidate starvation**：validation target 不在 `B50`。
6. **行为转移统计**：对每个 `sequence[:-2]` 的相邻有向 item pair 计数；不跨用户，
   不加入 validation target。
7. **行为候选分数**：对 validation history 最近至多 20 个 item `x_l` 与候选 `i`，
   使用唯一公式

   ```text
   score_beh(i | h) =
       sum_l 0.9^l * count(x_l -> i)
                     / sqrt(out_count(x_l) * in_count(i))
   ```

   其中 `l=0` 为最近 item；分母为 0 的项贡献 0。排除 history item 和训练中未出现
   的 item，按 `(-score_beh, item_id)` 确定性排序。不做参数扫描。
8. **行为集合 `A50`/`A10`**：分别取行为排序前 50/10；不足则保持实际长度。
9. **固定预算集合 `F50`**：先保留 `B40`，再加入不在 `B40` 的 `A10`；若去重后不足
   50，按原 baseline rank 从 `B50` 补足。最终必须 `|F50| <= 50`，不允许 `B50∪A50`
   作为效果比较。
10. **alias 资格代理**：只给 training-only 定义下同时属于 tail、且 prefix
    congestion 位于全部 item 上四分位数（`>= Q75`）的 item 记为 eligible。资格计算
    不得使用 validation target 的 hit/miss；candidate starvation 只作为评估结果。

#### 16.5.3 完整性与单元测试

执行前至少覆盖：

1. `sequence[-2:]` 的扰动不会改变任何 training-only 统计或行为排序；
2. transition 不跨用户边界；当前用户的 validation target 不被注入 transition 或
   行为分数，但它可以因其他用户的 training-only transition 自然成为候选；
3. score、tie break 和输入顺序扰动下结果确定；
4. `B50`、`B40`、`A50`、`A10` 去重正确且 `|F50| <= 50`；
5. head/tail 与既有评测定义完全一致；
6. baseline Recall@10/50 精确复算为 S0 summary，误差不超过 `1e-12`；
7. 两数据集均无 unknown gold、target mismatch 或缺失用户；
8. 输出显式记录 `test_prediction_read=false`、`test_target_used=false`、
   `gpu_used=false`。

任一完整性项失败记为 `EXECUTION_INVALID`，不产生科学 GO/STOP；只能修复同一实现
错误后重新执行，不得改研究公式或门槛。

### 16.6 C0-D 分析顺序与固定 gate（详细锁定）

#### 16.6.1 D1：Semantic-ID 拥堵与候选饥饿

主要分析单位为 validation 用户，分别在 Beauty/Toys 的 tail target 上计算：

```text
OR_miss =
  odds(target miss@50 | high congestion)
  / odds(target miss@50 | low congestion)
```

列联表如有零格，固定使用 Haldane–Anscombe `+0.5` 修正。使用用户级 10,000 次
bootstrap、固定 seed `20260724` 给出 95% percentile CI。
以下两项必须在两个数据集都成立：

1. `OR_miss >= 1.25`；
2. bootstrap 95% CI 下界 `> 1.0`。

一级 prefix 是唯一 gated 深度。其他 prefix 深度、bucket size 分布和 head-share
只作为 descriptive analysis，不得替代主判据。任一数据集失败即
`STOP_CAMI_PREMISE`，不读取 D2 的科学判定。

#### 16.6.2 D2a：不限预算的互补性上限

`A50` 只用来回答行为视图是否含有 baseline 没有的 target，不作为最终比较。分别计算：

- `unique_tail_recovery@50 = P(target in A50 and target not in B50 | tail)`；
- `behavior_unique_fraction =
  count(tail target in A50 and not in B50) / count(tail target in A50)`。

两数据集必须同时满足：

1. `unique_tail_recovery@50 >= 5` 个百分点；
2. `behavior_unique_fraction >= 30%`。

任一失败即 `STOP_CAMI_NO_COMPLEMENTARITY`。

#### 16.6.3 D2b：固定 50 候选预算的净收益

primary comparison 为同一 validation 用户上的 `F50` 对 `B50`。两数据集必须同时满足：

1. tail Recall@50 绝对提升 `>= 2` 个百分点；
2. overall Recall@50 不下降；
3. head Recall@50 绝对下降不超过 `1` 个百分点；
4. `|F50| <= 50` 的用户比例为 100%；
5. alias 资格代理覆盖的 unique item 不超过全体 item 的 30%。

同时报告 paired 用户的 recovered/lost 数量和净迁移，但 C0 不对 NDCG 做效果结论，
因为行为代理没有经过生成模型校准。任一必要项失败即
`STOP_CAMI_FIXED_BUDGET`。

#### 16.6.4 C0 唯一决策

| 决策 | 条件 | 解锁范围 |
|---|---|---|
| `C0_DESIGN_ALLOWED` | C0-N、完整性、D1、D2a、D2b 在两数据集全部通过 | 只允许依据 C0 结果具体化 C1 设计和预注册 |
| `STOP_CAMI_NOVELTY` | C0-N 任一必要差异被覆盖 | 停止当前论文点 |
| `STOP_CAMI_PREMISE` | D1 任一数据集失败 | 停止 CAMI 机制假设 |
| `STOP_CAMI_NO_COMPLEMENTARITY` | D2a 任一数据集失败 | 停止当前行为视图 |
| `STOP_CAMI_FIXED_BUDGET` | D2b 任一数据集失败 | 不进入模型实现 |
| `EXECUTION_INVALID` | lineage、复算或单元测试失败 | 只修实现，不作科学结论 |

C0 不设置结果导向的 `MODIFY`。科学 STOP 后不得在同一 validation 上扫描第二个行为
公式、0.8/0.95 recency、不同归一化、`B45+A5`、`B35+A15` 或其他 prefix gate。
如果未来有理论上不同的新方向，必须建立新名称、新差异审计和独立数据边界。

#### 16.6.5 C0 预期实现与产物

计划中的唯一入口（尚未实现）：

```bash
python3 experiment/phase3/cami_c0.py \
  --config artifacts/phase3/configs/cami_c0_preregistered.json
```

预期产物：

- `artifacts/phase3/cami_c0/{Toys,Beauty}/lineage.json`
- `artifacts/phase3/cami_c0/{Toys,Beauty}/prefix_starvation.csv`
- `artifacts/phase3/cami_c0/{Toys,Beauty}/candidate_transitions.csv`
- `artifacts/phase3/cami_c0/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/cami_c0/paired_metrics.csv`
- `artifacts/phase3/cami_c0/summary.json`
- `report/第三阶段/GRAM_第三阶段_CAMI_C0候选可行性报告.md`

资源边界：纯 CPU；建议 timeout 30 分钟；峰值内存和 wall time 必须记录。超时只记执行
失败，不自动放宽采样范围或门槛。

### 16.7 C1 以后：只保留轮廓，不提前锁死

以下阶段均**尚未解锁且尚未完整预注册**。上一阶段通过后，必须先把真实结果、失败风险
和下一阶段唯一方案写回本计划，用户确认后才执行。

#### C1：identifier 与目标函数设计

进入条件：`C0_DESIGN_ALLOWED`。

只回答如何把 C0 的互补信号变成合法 behavioral alias、如何稀疏分配 alias、如何在
单 decoder 中做多路径训练和 item-level 概率回收。此阶段先完成设计、泄漏审计、
新颖性复核和机器可读预注册；是否需要写代码、采用何种 alias 结构和门槛，待 C0
结果后具体化。C1 本身不解锁效果 pilot。

#### C2：正确性与极小 smoke

进入条件：C1 设计通过独立审计。

暂定只验证 alias-to-item 映射、Trie legality、marginal loss 数值、非零梯度、固定
beam 预算和资源可行性；样本量、GPU/CPU、具体门槛待 C1 后锁定。smoke 权重不得用于
效果结论。

#### C3：小型效果 pilot

进入条件：C2 correctness 全部通过。

暂定使用与 HBTR pilot 不同、一次性冻结的 validation cohort，只运行一个 CAMI 主方案、
一个 matched baseline、一个 seed 和固定 beam；主终点、样本比例、实际效应门槛、
资源门槛和 STOP 规则必须依据 C0/C2 可观测量另行预注册。不得沿用 HBTR 的 GO 状态。

#### C4：中型/跨种子确认

进入条件：C3 达到其未来预注册的全部门槛。

确认规模、数据集顺序、seed 数和统计分析由 C3 效应与方差决定；至少保持 paired
evaluation，并防止把用户数当作独立训练重复。此时仍不自动读取 test。

#### C5：全量与一次性 test

只有 Beauty/Toys validation 均完成确认、方案与 checkpoint 选择规则完全冻结后，才
另行决定是否进行全量训练和一次性 test。论文级消融、效率和泛化实验也必须在前一阶段
存活后再具体化，不在 C0 阶段预支预算。

### 16.8 C0-N 实际结果与 CAMI 终止决策

C0-N 于 2026-07-24 按串行 gate 完成。审计覆盖计划指定的 GRAM、TIGER、MINDER、
ActionPiece、EAGER、LETTER、CoST、RPG、LOHRec、Better Generalization with
Semantic IDs，并纳入检索中新发现、与当前主张更接近的 MTGRec、Pctx 与 PIT。

最关键的新证据是 Pctx：

1. 同一 item 根据完整用户历史拥有多个 personalized Semantic ID；
2. 训练时可用同一 item 的替代 SID 做合法序列增强；
3. 单个自回归 encoder-decoder 生成不同 SID 路径；
4. 推理时把映射到同一 item 的 SID 路径概率聚合为 next-item probability。

因此三项必要 gate 的结果为：

| Gate | 结果 | 说明 |
|---|---|---|
| adaptive allocation | PASS（差异很窄） | 尚未发现专门按 candidate starvation / prefix congestion 稀疏分配 alias 的工作 |
| single-decoder item marginalization | **FAIL** | Pctx 已覆盖多合法 SID、单生成器和 SID probability → item probability aggregation |
| fixed-budget tail recovery | PASS | 近邻工作未把固定总候选预算下的 tail recovery 作为主要机制问题 |

三项均为必要条件，不能由另外两项抵消。固定决策为
**`STOP_CAMI_NOVELTY`**。C0-D 未实现、未运行，也没有读取其 validation 数据结果；
未读取 test、未加载 checkpoint、未训练、未使用 GPU。不得通过改名、只强调
`B40+A10` 或把 path sum 换一种写法继续当前 CAMI。

产物：

- `artifacts/phase3/cami_c0/novelty_matrix.csv`
- `artifacts/phase3/cami_c0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_CAMI_C0差异审计.md`

该结论只否定当前 CAMI 论文点的新颖性，不否定 Semantic-ID candidate starvation
可能存在，也不否定 multi-identifier 在本数据上可能有效。继续研究前必须先提出一个
不以“上下文多 identifier + item 概率聚合”为核心的新研究问题，并重新建立 C0-N。

## 17. 新周期 D：Native Lexical Prior Leakage（NLPL）

### 17.1 Material Passport 与研究边界

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-24
- Verification Status: PREREGISTERED（D0-N）；UNVERIFIED（D0-D 及后续）
- Version Label: `nlpl_d0_preregistered_v1`
- 研究对象：GRAM 的 semantic-to-lexical translation 所复用的 T5 原生词元概率。
- 不属于本方向：多 identifier/item marginalization、beam 内 hard-negative margin、
  单纯 popularity weighting、扩大 beam、换 tokenizer 或重新构造 Semantic ID。
- 当前授权：只执行 D0-N。D0-N 通过后，才按本节原样实现 D0-D；D0 通过也只解锁
  D1 的设计与预注册，不自动解锁训练、GPU 或 test。

核心研究问题是：在同一 lexical parent、近似相同训练频次的商品之间，原始 T5-small
在没有用户语义输入时对下一原生词元的偏好，是否仍系统性地改变 GRAM beam-50 的商品
曝光，并使 native prior 较低的 tail target 更容易完全不进入候选集。

### 17.2 D0 总览与串行 gate

D0 严格按下列顺序执行：

1. **D0-N（novelty）**：先做原文级差异审计，不查看新诊断结果；
2. 若且仅若 D0-N 为 `NOVELTY_SCOPE_PASS` 或
   `NOVELTY_SCOPE_PASS_WITH_NARROWING`，实现 D0-D；
3. **D0-D（diagnosis）**：用冻结 prediction、training-only 频次、lexical mapping
   和本地原始 T5-small 做确定性 CPU 分析；
4. D0-N 或 D0-D 任一必要项失败即 STOP，不用替代 prompt、平滑常数、匹配深度、
   频次比或数据集特定阈值补救。

### 17.3 D0-N：原文级差异审计（详细锁定）

审计至少覆盖四类最接近工作：

1. GRAM、GRLM/structured term IDs 等 semantic-to-lexical 或原生 textual identifier；
2. APAO、Latte、Decoding Matters 等 prefix/decoding/tree probability 偏差；
3. Token-Weighted Multi-Target Learning、Ghost Tokens 等训练频率与 popularity 偏差；
4. Calibrate Before Use、verbalizer/label-word calibration 等预训练 label prior。

三项均为必要 gate：

| Gate | PASS 条件 |
|---|---|
| mechanism isolation | 未发现工作在 generative recommendation 中把“冻结预训练 LM 的 native lexical conditional prior”与训练流行度、树耦合和长度效应明确分离 |
| GRAM-specific audit | 未发现工作对 semantic-to-lexical item path 做 parent-matched、popularity-matched 的 beam exposure 审计 |
| intervention space | 未发现工作已在受 Trie 约束的 lexical-ID recommendation 中以 frozen-base conditional prior 为对象做 path-level neutralization；通用 contextual calibration 不计作本项首创 |

允许的通过结论必须收窄为：**GRAM 类 lexical identifier 的 native-prior exposure
机制与结构化控制/干预**。不得声称“首次发现语言模型有 token bias”“首次做概率校准”
或“首次发现生成推荐有 decoding bias”。若通用 prior subtraction 已知但尚未用于该
机制，可记 `PASS_WITH_NARROWING`，但未来论文贡献必须包含机制证据和推荐特异设计，
不能把简单减分公式单独包装为算法创新。

固定产物：

- `artifacts/phase3/nlpl_d0/novelty_matrix.csv`
- `artifacts/phase3/nlpl_d0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_NLPL_D0差异审计.md`

### 17.4 D0-D：输入、定义与防泄漏（D0-N 通过后执行）

#### 17.4.1 固定输入

- 冻结 prediction：
  `GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv` 与
  `GRAM/preds/20260722_125916_Beauty_sequential_pred_validation.tsv`；
- 对应 `user_sequence.txt`，仅 `sequence[:-2]` 统计训练频次，`sequence[-2]`
  作为 validation target，绝不读取 `sequence[-1]`；
- 对应 `item_generative_indexing_hierarchy_*.txt` lexical ID mapping；
- 本地原始 `t5-small` snapshot
  `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`；
- S0 validation summary 只用于复算 Recall@10/50 完整性。

禁止加载 GRAM checkpoint、使用 finetuned logits、读取 test、训练、访问 GPU、下载
新模型或根据输出更换输入。所有输入、代码、config 和本地模型文件写 SHA-256。

#### 17.4.2 Native prior

对 item \(i\) 的 lexical token path \(t_{i,1:L}\)，使用冻结原始 T5-small、neutral
encoder `[pad, eos]` 和 teacher-forced 真值前缀，定义：

\[
LP_0(i)=L^{-1}\sum_{\ell=1}^{L}\log p_{\mathrm{T5base}}
(t_{i,\ell}\mid [pad,eos],t_{i,<\ell}).
\]

不计 EOS；decoder start 为 T5 `pad_token_id`。同时保存末词元条件 log-prob
`LP_last(i)`。主检验只使用 `LP_last`，因为 matched sibling 共享此前全部 lexical
prefix；全路径 `LP_0` 仅作描述性稳健性结果，不用于晋级。

#### 17.4.3 Exposure amplification 与 matched sibling

将锁定 beam-50 surface candidate 按现有 S0 decoder 精确映射为 item。令
`beam_count_i` 为 item 在所有 validation 用户 top-50 中出现次数，训练频次为
`train_freq_i`，商品数为 \(I\)，总用户数为 \(N\)，固定：

\[
A_i=\log\frac{beam\_count_i+0.5}{50N+0.5I}
-\log\frac{train\_freq_i+0.5}{\sum_j train\_freq_j+0.5I}.
\]

eligible pair 必须共享除最后词元外的完整 path、两者 `train_freq>0`，且
`max(train_freq)/min(train_freq) <= 2`。预检查只使用 mapping 与 training-only
频次，已确认 Toys/Beauty 分别至少有 1,131/266 个 eligible pair，故最低样本门槛
事前固定为每数据集 200 对。每个无序 item pair 只计一次。

主统计量是 pair 内 `ΔLP_last` 与 `ΔA` 同号比例；tie 任一侧为 0 时剔除并单独报告。
不把 item pair 当独立训练重复：以 parent path 为 cluster 做 10,000 次 bootstrap，
并在 parent 内对 `ΔA` 做 10,000 次 sign-flip permutation。固定 seed `20260724`。

#### 17.4.4 Tail miss 关联

tail 定义沿用 training-only item 频次的 bottom 80%。先在 parent 内中心化 `LP_last`，
再仅在实际作为 validation tail target 的事件上按数据集 median 分为 low/high prior。
比较 `miss@50` odds：

\[
OR_{miss}=\frac{miss_{low}/hit_{low}}{miss_{high}/hit_{high}},
\]

四格均加固定 Haldane–Anscombe 0.5。该项是机制落到推荐失败的必要关联，但不解释为
因果效应。

### 17.5 D0-D 完整性、固定 gate 与唯一决策

实现前必须有最小单元测试覆盖：teacher forcing shift、EOS 排除、pair 去重、
training-only 截断、候选映射和 bootstrap determinism。运行时另须满足：

- S0 Recall@10/50 在两数据集逐项复算绝对误差 ≤`1e-12`；
- prediction 每行恰有 50 个可映射候选，无 unknown/duplicate item；
- 修改 `sequence[-2:]` 不改变任何 training frequency；
- 全部 native log-prob finite，local model hash 与 config 记录完整。

D0-D 的全部必要通过条件为：

| Gate | Toys 与 Beauty 均须满足 |
|---|---|
| support | eligible non-tie matched pairs ≥200 |
| concordance | `concordance >= 0.55` |
| uncertainty | parent-cluster bootstrap 95% CI lower bound `>0.50` |
| randomization | parent-level sign-flip permutation `p<=0.01` |
| recommendation link | tail target `OR_miss>=1.10` |

唯一决策：

| 决策 | 条件 |
|---|---|
| `D0_MECHANISM_ALLOWED` | D0-N 与完整性通过，且上表五项在两数据集全部通过 |
| `STOP_NLPL_NOVELTY` | D0-N 任一必要 gate 失败 |
| `STOP_NLPL_NO_EXPOSURE` | support/concordance/CI/permutation 任一数据集失败 |
| `STOP_NLPL_NO_TAIL_LINK` | 曝光 gate 通过但 OR 任一数据集失败 |
| `EXECUTION_INVALID` | lineage、复算、映射、finite 或单元测试失败 |

D0 不允许 `MODIFY`。不得在同一 validation 上换 neutral prompt、改 0.5 smoothing、
改 sibling depth、改频次比、删难例、改单侧统计或为两个数据集设不同阈值。

计划入口：

```bash
CUDA_VISIBLE_DEVICES="" \
/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9 \
  experiment/phase3/nlpl_d0.py \
  --config artifacts/phase3/configs/nlpl_d0_preregistered.json
```

预期产物：

- `artifacts/phase3/nlpl_d0/{Toys,Beauty}/item_native_prior.csv`
- `artifacts/phase3/nlpl_d0/{Toys,Beauty}/matched_pairs.csv`
- `artifacts/phase3/nlpl_d0/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/nlpl_d0/summary.json`
- `report/第三阶段/GRAM_第三阶段_NLPL_D0诊断报告.md`

## 18. 新周期 E：Counterfactual Granularity Interference（CGI）

### 18.1 Material Passport 与边界

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-24
- Verification Status: ANALYZED（E0-N/E0-D）；UNVERIFIED（E1 及后续，未解锁）
- Version Label: `cgi_e0_preregistered_v1`
- 核心对象：GRAM FiD 中一个 coarse lexical-history passage 与多个 fine item-text
  passages 的条件贡献。
- 允许迁移：RAG counterfactual passage deletion 作为诊断；CFT、MHL、MGFiD 作为
  必须比较的近邻和基线思想。
- 不允许声称：首次 counterfactual attribution、首次 history masking、首次 adaptive
  fusion、首次发现长 context 有噪声。
- E0-N/E0-D 已按串行协议完成；E0 未训练、未读 test，E1 未解锁。

静态代码与真实配置审计确认：两个 checkpoint 均为 `reverse_history=1`，coarse 与
fine history 都是新→旧。E0 明确不研究“两个粒度顺序相反”，也不把它作为失败解释。

### 18.2 E0-N：原文级新颖性审计

至少覆盖：GRAM、CFT、MHL、RAGONITE、MGFiD、MGR-LF++、LWGR、FiD/RFiD、
long-context/RAG context dilution 与 adaptive multimodal sequential recommendation。

三项必要 gate：

| Gate | PASS 条件 |
|---|---|
| mechanism specificity | 未发现 generative recommendation 工作把正确的逐 item fine-text passage 作为可能降低正确 lexical target path 概率的干扰源，并与 coarse-only 做配对反事实分解 |
| structured attribution | 未发现工作在 GRAM 类 multi-passage FiD recommendation 中区分累计 fine contribution、oldest/newest passage contribution 与 tail miss/hit |
| intervention room | 未发现工作已用 training-only passage contribution supervision 学习 target-free granularity gate；通用 RAG pruning/adaptive fusion 不计首创 |

允许 `NOVELTY_SCOPE_PASS_WITH_TRANSFER`：明确承认反事实删除、masking、adaptive fusion
来自既有领域；未来贡献必须是推荐特异机制、结构化诊断和由该机制约束的 target-free
干预三者组合。任一必要 gate 被覆盖则 `STOP_CGI_NOVELTY`，不运行 E0-D。

固定产物：

- `artifacts/phase3/cgi_e0/novelty_matrix.csv`
- `artifacts/phase3/cgi_e0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_CGI_E0差异审计.md`

### 18.3 E0-D：输入、cohort 与干预

#### 18.3.1 固定输入

- Toys checkpoint：
  `GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt`；
- Beauty checkpoint：
  `GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt`；
- 对应 run `config.json`、validation `user_sequence.txt`、item text/mapping、冻结
  prediction TSV 与 S0 summary；
- 只使用 `sequence[:-2]` 构造 history，`sequence[-2]` 为 validation target；
  `sequence[-1]` 禁止进入任何统计、prompt 或 cohort 特征。

所有输入、checkpoint、代码和 config 写 SHA-256。必须复算 cohort 中 full-condition
gold-path score 两次且绝对误差 `<=1e-7`；模型保持 `eval()`、`no_grad()`。

#### 18.3.2 固定 cohort

只纳入截断后 history length `>=2` 且 prediction/target 可完整映射的用户。tail/head
沿用 training-only item frequency bottom 80% / top 20%；hit/miss 由冻结 full-model
beam-50 prediction 定义。对 `tail_miss`、`tail_hit`、`head_miss`、`head_hit` 四层，
按 `SHA256("20260724|dataset|stratum|user")` 升序各取最多 256 人。样本不足不从其他
层补齐；任一数据集 `tail_miss<256` 或 `tail_hit<256` 为 `EXECUTION_INVALID`。

这是 outcome-stratified、result-informed 机制 cohort，不用于估计总体 Recall，也不把
四层直接合并。E0 不生成新 beam，不做效果提升声明。

#### 18.3.3 四个锁定条件

对相同 gold lexical target path 做 teacher-forced mean log-prob/token。主分数只聚合
lexical target token，明确排除 tokenizer 自动附加的 EOS 与 padding；decoder start
仍为 T5 `pad_token_id`，四个条件必须使用完全相同的 labels：

1. `full`：原始 coarse + 全部 fine passages；
2. `coarse_only`：保留 coarse passage，把所有 fine passage attention mask 置零；
3. `minus_oldest`：保留 coarse 和其余 fine，只 mask 最旧 fine passage；
4. `minus_newest`：保留 coarse 和其余 fine，只 mask 最新 fine passage。

mask 只改变 encoder attention mask，不删除、移动或替换 token；至少保留 coarse
passage，因此不存在全 mask 行。最新 fine passage 在当前 `reverse_history=1` 配置中
固定为 passage index 1，最旧为最后一个有效 fine passage。实现必须以 item identity
检查 coarse/fine 顺序一致，而不是只相信 index。

定义：

\[
G_{all}=LP_{coarse\_only}-LP_{full},
\quad G_{old}=LP_{minus\_oldest}-LP_{full},
\quad G_{new}=LP_{minus\_newest}-LP_{full}.
\]

正值表示移除相应 fine evidence 后正确 target path 得分上升，即该 evidence 在该执行
轨迹中有负贡献。它是模型反事实敏感度，不解释为用户行为因果效应。

### 18.4 E0-D 统计、完整性与唯一 gate

所有均以用户为配对单位。每个数据集、每个 stratum 独立报告均值、中位数、正值率和
10,000 次用户 bootstrap percentile CI；seed `20260724`。主分析只使用
`tail_miss`，tail-hit 只作锁定对照。

全部必要门槛，两数据集均须通过：

| Gate | 固定条件 |
|---|---|
| cohort/integrity | tail_miss 与 tail_hit 各 256；full 重算误差 ≤`1e-7`；无 NaN/Inf；顺序 identity 审计通过 |
| cumulative interference | tail_miss `mean(G_all)>=0.03` nats/token，95% CI lower `>0`，且 `P(G_all>0)>=0.55` |
| old-passage interference | tail_miss `mean(G_old)>=0.01`，95% CI lower `>0` |
| temporal specificity | tail_miss `mean(G_old-G_new)>=0.01`，95% CI lower `>0` |
| failure association | `mean(G_all_tail_miss)-mean(G_all_tail_hit)>=0.02`，分层 bootstrap 95% CI lower `>0` |

固定决策：

| 决策 | 条件 |
|---|---|
| `E0_MECHANISM_ALLOWED` | E0-N、完整性及五项 E0-D gate 在两数据集全部通过 |
| `STOP_CGI_NOVELTY` | E0-N 任一必要项失败 |
| `STOP_CGI_NO_INTERFERENCE` | cumulative/old/temporal 任一数据集失败 |
| `STOP_CGI_NO_FAILURE_LINK` | interference 通过但 failure association 任一数据集失败 |
| `EXECUTION_INVALID` | cohort、lineage、重算、finite、顺序或单元测试失败 |

E0 不设 `MODIFY`。不得看结果后改 cohort 大小、0.03/0.01/0.02、history 最低长度、
tail 定义、mask 方式、target 聚合、checkpoint 或改做 “any harmful passage” 最大值。

预期入口：

```bash
CUDA_VISIBLE_DEVICES=3 \
/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9 \
  experiment/phase3/cgi_e0.py \
  --config artifacts/phase3/configs/cgi_e0_preregistered.json
```

预期产物：

- `artifacts/phase3/cgi_e0/{Toys,Beauty}/cohort.csv`
- `artifacts/phase3/cgi_e0/{Toys,Beauty}/counterfactual_scores.csv`
- `artifacts/phase3/cgi_e0/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/cgi_e0/summary.json`
- `report/第三阶段/GRAM_第三阶段_CGI_E0诊断报告.md`

### 18.5 E1 以后：只保留轮廓

- **E1：target-free gate 设计。** 仅在 `E0_MECHANISM_ALLOWED` 后，使用 training-only
  counterfactual contribution 监督轻量 passage gate；不得在推理时访问 target。
- **E2：correctness smoke。** 验证 mask legality、贡献监督、非零梯度、显存和延迟。
- **E3：一次性小 pilot。** 一个 CGI 主方案、一个 matched baseline、一个 seed 与固定
  cohort；终点和门槛依据 E0–E2 另行预注册。
- **E4/E5：跨种子确认与全量/test。** 只有前一步通过后才具体化。

### 18.6 E0-N 实际结果

E0-N 于 2026-07-24 在加载 checkpoint 和计算 E0-D 之前完成。三项 gate 均通过，
固定决策为 **`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_NARROWING`**。

| Gate | 结果 |
|---|---|
| mechanism specificity | PASS |
| structured attribution | PASS |
| intervention room | PASS WITH TRANSFER AND NARROWING |

LWGR 已覆盖不受控 semantic knowledge fusion 与 behavioral signal 冲突以及 selective
beneficial fusion；RFiD/MGFiD 已覆盖 FiD spurious evidence、guidance 与 pruning；
CFT/MHL 已覆盖 behavior counterfactual 与 history masking。剩余主张只能是 plan 锁定
的 GRAM 原生 coarse/fine history passage 结构化负贡献机制。

产物：

- `artifacts/phase3/cgi_e0/novelty_matrix.csv`
- `artifacts/phase3/cgi_e0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_CGI_E0差异审计.md`

### 18.7 E0-D 实际结果与停止决定

E0-D 于 2026-07-24 按 `cgi_e0_preregistered_v1` 一次完成。CPU 单元测试 4/4
通过；Toys/Beauty 的 `tail_miss`、`tail_hit`、`head_miss`、`head_hit` 均锁定为
256 人。两数据集 full-condition 重算最大绝对误差均为 0；所有分数 finite，
`reverse_history=1` 与 newest/oldest item identity 审计通过；未训练、未读取 test
或 `sequence[-1]`。GPU3 运行结束后 CodeLlama reservation 已恢复。

| Dataset | tail-miss mean `G_all` (95% CI) | P(`G_all>0`) | mean `G_old` (95% CI) | mean `G_old-G_new` (95% CI) | miss-hit `G_all` (95% CI) |
|---|---:|---:|---:|---:|---:|
| Toys | -0.3246 [-0.4388, -0.2149] | 0.3047 | -0.0321 [-0.0938, 0.0095] | -0.0035 [-0.0880, 0.0738] | 1.2654 [1.0107, 1.5246] |
| Beauty | -0.1546 [-0.3343, 0.0084] | 0.4141 | -0.0073 [-0.0196, 0.0050] | -0.0078 [-0.1213, 0.1391] | 2.6379 [2.1780, 3.1154] |

两数据集均通过 cohort/integrity 与 failure-association gate，但 cumulative
interference、old-passage interference 和 temporal specificity 均失败。负
`G_all=LP_coarse_only-LP_full` 表明，平均而言删除全部 fine passages 会降低 gold
path 得分；因此实际机制是“fine evidence 对 tail miss 仍有帮助，但帮助弱于 tail
hit”，不是预注册的“fine evidence 导致 tail miss”。association 不能在主机制缺失时
单独晋级。

固定科学决策为 **`STOP_CGI_NO_INTERFERENCE`**。E1–E5 不解锁；不得改成 any
harmful passage、另选 cohort、阈值、mask、checkpoint 或训练 gate 来挽救方向 E。

产物：

- `artifacts/phase3/configs/cgi_e0_preregistered.json`
- `artifacts/phase3/cgi_e0/{Toys,Beauty}/cohort.csv`
- `artifacts/phase3/cgi_e0/{Toys,Beauty}/counterfactual_scores.csv`
- `artifacts/phase3/cgi_e0/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/cgi_e0/summary.json`
- `report/第三阶段/GRAM_第三阶段_CGI_E0诊断报告.md`

## 19. 新周期 F：Lexical Echo Interference（LEI）

### 19.1 Material Passport、动机与非重试边界

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-24
- Verification Status: ANALYZED（F0-N/F0-D）；UNVERIFIED（F1 及后续，未解锁）
- Version Label: `lei_f0_d_stopped_v2`
- 当前状态：LEI 已停止；不训练、不修改模型、不读更多 validation 或 test。

GRAM 的 coarse prompt 按历史列出 lexical IDs；每个 fine prompt 又以
`item: <history lexical ID>` 开头，并在 `similar items:` 中继续使用 lexical IDs；
decoder target 仍是同一 T5 原生词表中的 lexical ID。`item:` anchor 是论文定义的
information linking，论文总体 ablation 显示移除 linking 最多降低约 1.8% NDCG@5，
但这不回答 link span、CF-ID span 与 metadata span 在 tail miss 上是否相互抵消。

E0-D 已证明整个 fine passage 的净贡献为正，因此 LEI **不得**声称 fine passage
整体有害，也不得重新运行 E0 的 oldest/whole-passage mask。方向 F 的最窄问题是：
同一 lexical token 同时作为输入中的 link/CF symbol 与输出 identifier 时，是否产生
超出合理序列/语义相似性的 echo；有益 metadata 是否掩盖了这个负分量。

### 19.2 F0-N：先于任何新诊断的原文级差异审计

F0-N 至少覆盖以下近邻簇，并阅读方法与实验而不只看摘要：

1. GRAM 的 semantic-to-lexical translation、information linking、CF verbalization
   与 `w/o linking` ablation；
2. lexical identifier / generative retrieval：GLEN、PAG，以及 lexical-index
   ambiguity/collision 工作；
3. generative recommendation token representation：DECOR、LETTER、DIGER、
   purely semantic indexing；
4. token/popularity optimization：Token-Weighted Multi-Target Learning、Ghost、
   variable-length tokenization；
5. history/repetition：repeat–explore sequential recommendation、MHL、
   OneRec-Think/ReaRec；
6. 通用 copy mechanism、input–output lexical overlap 或 copying bias 中能实质覆盖
   “同词表 identifier 兼任跨 passage anchor 与 decoder symbol”的工作。

三项必要 gate：

| Gate | PASS 条件 |
|---|---|
| GRAM-role specificity | 未发现工作已在 GRAM 类 multi-passage generative recommendation 中把原生 lexical ID 的 link-anchor/CF-attribute/decoder-symbol 三重角色定义为可检验机制 |
| span-factorized attribution | 未发现工作已保持 passage 与 token 位置，分别估计 `item:` anchor、`similar items:` IDs 和 metadata 对 gold identifier path 的反事实贡献 |
| intervention room | 未发现工作已针对上述角色耦合设计 target-free role-disambiguated linking；通用 token weighting、copy suppression、embedding decomposition 或 identifier redesign 不算本项首创 |

允许的通过结论只能是
**`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_GRAM_NARROWING`**：copy diagnostics、
counterfactual masking、role embeddings 等均承认来自既有领域；潜在贡献只保留
“GRAM-native lexical linking role attribution + recommendation-specific
role-disambiguated intervention”的组合。任一必要 gate 被覆盖则固定
`STOP_LEI_NOVELTY`，F0-D 不解锁。

固定产物：

- `artifacts/phase3/lei_f0/novelty_matrix.csv`
- `artifacts/phase3/lei_f0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_LEI_F0差异审计.md`

### 19.3 F0-N 实际结果

F0-N 固定决策为
**`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_GRAM_NARROWING`**，三项 gate 均通过。
详细原文边界、来源矩阵和最强反对意见见本节固定产物。

通过只表示在截至 2026-07-24 的预注册近邻簇中，未检出对 GRAM-native 三重 token
role 做位置保持 span attribution 的实质覆盖。通用 copy、repeat、overlap、role
embedding、contextual representation、token weighting、identifier redesign 均为
已有工作。任何未来“first”表述必须是检索范围限定的组合式主张。

### 19.4 F0-D：span-factorized frozen diagnosis（已完成）

固定配置：`artifacts/phase3/configs/lei_f0_d_preregistered.json`。配置在读取任何新的
LEI checkpoint 分数前写入。

#### 19.4.1 数据与冻结边界

- 复用 CGI E0 的 checkpoint、validation prediction lineage、seed 与四层 deterministic
  cohort；完整核对四层各 256 人，但只对 `tail_miss`/`tail_hit` 各 256 人计分。
- 不读 `sequence[-1]`、test、test target；不训练、不更新参数、不生成 beam。
- gold score 为 target lexical tokens 的 mean log-prob，排除 EOS/pad，与 E0-D 一致。
- 输入 ID、passage 数量、token 位置全部不变；干预只把预先定位的 attention-mask
  entry 置零。

#### 19.4.2 唯一 span 定义与条件

主角色是每个 fine passage 开头 `item:` 后的 history lexical ID。它既在 coarse
history 出现，又承担 GRAM information-link anchor，最直接对应 LEI。`similar items:`
中的 CF IDs 必须单独定位和报告，但因为它们不保证与该用户 history 重复，只作为
secondary descriptive，不参与晋级门槛。

固定条件为 `full`、`coarse_only`、`minus_link_ids`、`minus_cf_ids`、
`minus_all_fine_ids`，以及 8 个 `matched_link_metadata_r`。matched control 在同一
passage 的 CF list 之后、EOS 之前，从 active 且含字母或数字的 metadata token 中
无放回选择与 link span 等量的位置；位置顺序由
`SHA256(seed|dataset|user|passage|replicate|position)` 唯一决定。role localization
与 control selection 不得读取 target、rank 或任何效果分数。

#### 19.4.3 固定 estimand

对用户 \(u\)：

- `R_link = LP(minus_link_ids) - LP(full)`；
- `C_link = mean_r LP(matched_link_metadata_r) - LP(full)`；
- `A_link = R_link - C_link`；
- `M_meta = LP(minus_all_fine_ids) - LP(coarse_only)`；
- `R_cf` 与 `R_all` 只作 secondary descriptive。

正的 `R_link` 表示删除 link IDs 后 gold path 变好；正的 `A_link` 表示该变化超过
同 token 数 metadata deletion；正的 `M_meta` 表示移除所有 fine IDs 后，剩余
metadata 相对 coarse-only 仍有帮助。

#### 19.4.4 双数据集必要 gate 与决策

每个均值都做 10,000 次 bootstrap；涉及正向机制的 95% CI 下界必须严格大于 0。

| Gate | Toys 与 Beauty 各自必须满足 |
|---|---|
| integrity | tail miss/hit 各 256；full 重算误差 ≤1e-7；finite；role 定位率与 matched-control eligibility 均为 100%；复用 cohort 完全一致 |
| raw link harm | tail-miss mean `R_link` ≥0.01，CI 下界 >0，positive rate ≥0.55 |
| role specificity | tail-miss mean `A_link` ≥0.02，CI 下界 >0 |
| separable metadata benefit | tail-miss mean `M_meta` ≥0.05，CI 下界 >0 |
| failure association | mean_tail-miss(`A_link`) − mean_tail-hit(`A_link`) ≥0.02，CI 下界 >0 |

串行固定决策：

1. 任一 integrity 失败：`EXECUTION_INVALID`，只允许修复执行错误后原协议重跑；
2. raw link harm 失败：`STOP_LEI_NO_RAW_ECHO`；
3. matched specificity 失败：`STOP_LEI_NO_ROLE_SPECIFICITY`；
4. metadata benefit 失败：`STOP_LEI_METADATA_NOT_SEPARABLE`；
5. failure association 失败：`STOP_LEI_NO_FAILURE_LINK`；
6. 双数据集全部通过：`F0_MECHANISM_ALLOWED`，仅解锁 F1 的设计与另行预注册。

这是一次性机制检验；失败后不得改 cohort、扫描另一套 span、把 CF descriptive
改成主 endpoint，或用单数据集结果继续。

固定预期产物：

- `artifacts/phase3/lei_f0/{Toys,Beauty}/cohort.csv`
- `artifacts/phase3/lei_f0/{Toys,Beauty}/span_audit.csv`
- `artifacts/phase3/lei_f0/{Toys,Beauty}/counterfactual_scores.csv`
- `artifacts/phase3/lei_f0/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/lei_f0/summary.json`
- `report/第三阶段/GRAM_第三阶段_LEI_F0诊断报告.md`

### 19.5 F0-D 实际结果与 LEI 终止决策

F0-D 按第 19.4 节配置一次性完成。第一次启动在 checkpoint 加载前因资源脚本漏掉
CUDA context 释放等待而以 exit 4 退出，没有生成科学分数；唯一修复为实现 plan 已
规定的最多 120 秒显存轮询，随后同协议从头执行。CPU 单元测试 4/4 通过，Toys 与
Beauty 的真实 tokenizer/passage span 定位集成检查通过。

双数据集 tail miss/hit 各 256 人；cohort 与 CGI E0 完全一致，full 重算最大绝对误差
均为 0，role localization 与 matched-control eligibility 均为 100%，finite、顺序、
prediction lineage 和资源恢复审计通过。未训练、未生成 beam、未读取 test 或
`sequence[-1]`。

| 数据集 | raw `R_link` mean [95% CI] / P(>0) | adjusted `A_link` mean [95% CI] | metadata `M_meta` mean [95% CI] | miss-hit adjusted mean [95% CI] |
|---|---|---|---|---|
| Toys | 0.005607 [-0.008848, 0.020022] / 0.53125 | 0.007718 [-0.008556, 0.023076] | 0.413219 [0.342700, 0.486318] | 0.031164 [0.008914, 0.053482] |
| Beauty | 0.020948 [0.002046, 0.046830] / 0.50000 | 0.025195 [0.006720, 0.050103] | 0.110914 [0.003189, 0.216563] | 0.034683 [0.013275, 0.060939] |

| 数据集 | integrity | raw link harm | role specificity | metadata benefit | failure association |
|---|---:|---:|---:|---:|---:|
| Toys | PASS | FAIL | FAIL | PASS | PASS |
| Beauty | PASS | FAIL | PASS | PASS | PASS |

Toys 的 raw mean、CI 与 positive rate 均未达到门槛；Beauty 虽 raw mean/CI 达标，
positive rate 仅 0.50，低于锁定的 0.55。两数据集都在串行第一项机制 gate 失败。
metadata 的正贡献和 miss–hit association 说明 passage 内仍有结构差异，但不能证明
稳定、跨数据集的 raw link echo。secondary CF 结果方向也不一致（Toys
`R_cf=0.08636`，Beauty `R_cf=-0.04484`），且按预注册不得升格为主 endpoint。

固定决策为 **`STOP_LEI_NO_RAW_ECHO`**。F1–F5 均不解锁；不得增加 control replicate、
改 positive-rate 门槛、只保留 Beauty、扫描 CF 子集或重新定义 echo。

产物：

- `artifacts/phase3/lei_f0/{Toys,Beauty}/cohort.csv`
- `artifacts/phase3/lei_f0/{Toys,Beauty}/span_audit.csv`
- `artifacts/phase3/lei_f0/{Toys,Beauty}/counterfactual_scores.csv`
- `artifacts/phase3/lei_f0/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/lei_f0/summary.json`
- `report/第三阶段/GRAM_第三阶段_LEI_F0诊断报告.md`

### 19.6 F0-D 后的条件式路线（未解锁）

- **F1：role-disambiguated linking。** 仅当双数据集 F0-D 证明 identifier span 有
  负贡献而 metadata 有正贡献时设计；不得修改 item identity、Trie 或读取 target。
- **F2：correctness smoke。** 只验证 role mask/embedding、梯度、合法解码、显存与
  延迟，不做效果结论。
- **F3：一次性小 pilot。** 进入条件、样本量、唯一主方案、matched baseline、终点
  与门槛依据 F0–F2 另行预注册。
- **F4/F5：跨种子确认与全量/test。** 只有前一步通过后才具体化。

F0-N 通过不代表机制存在，也不解锁训练或 GPU；F0-D 失败则方向 F 立即停止，不得
改成一般 copy bias、换 cohort 或扫描 span 定义。

## 20. 新周期 G：Collaborative Prefix Budget Displacement（CPBD）

### 20.1 Material Passport、动机与非重试边界

- Origin Skill: academic-research-suite / deep-research
- Origin Mode: plan
- Origin Date: 2026-07-24
- Verification Status: PREREGISTERED（G0-N）；UNVERIFIED（G0-D 及后续）
- Version Label: `cpbd_g0_n_preregistered_v1`
- 当前唯一允许动作：原文级新颖性审计，不读取新的实验效果。

GRAM 当前把 fine passage 序列化为：

```text
item: <history lexical ID>;
similar items: <CF lexical ID 1>, ..., <CF lexical ID k>;
<title/category/brand/description metadata>
```

`CollatorGRAM.encode_texts_split` 再从左到右保留最多 128 个 token。于是 CF prefix
拥有先占预算权；后置 metadata 一旦被截断，在 encoder 输入中根本不存在。

LEI 已有 span audit 只作为 **result-informed、post-hoc motivation**：Toys 的
3,002 个可见 fine passages 中，CF/metadata token 均值约为 25.41/66.81；Beauty 的
3,042 个 passages 则约为 70.35/26.48。Beauty 的 CF token 在 CF+metadata 可见预算中
平均约占 72.7%，Toys 约为 28.2%。这说明 field-budget imbalance 值得审计，但无法
证明 metadata 确实被截断，更不能证明恢复它会改善推荐。

CPBD 与既有失败方向的边界：

- 不重试 CGI：CGI 删除整个 fine passage；CPBD 保持 passage，只研究 passage 内
  serialization 与 hard budget。
- 不重试 LEI：LEI 对**已经可见**的 link/CF token 改 attention mask；CPBD 研究的是
  被 prefix 挤出、从未进入模型的 metadata，mask 无法恢复。
- 不把 CF 称为噪声：GRAM 原论文已显示 collaborative semantics 整体有益，且 LEI
  的 CF secondary effect 跨数据集方向不一致。
- 不做简单 `top_k` sweep：GRAM 已报告 dataset-level 最优 `k` 及“过多 similar
  items 引入噪声”。CPBD 必须区分直接 CF noise 与 metadata displacement，并以固定
  总预算、固定 CF identity 的 paired construction 识别机制。

### 20.2 G0-N：先于任何新诊断的原文级差异审计

G0-N 至少覆盖以下近邻簇，并阅读方法、实验与关键 appendix，而不只看摘要：

1. GRAM 的 CF verbalization、field order、per-passage truncation、top-k sensitivity
   与 multi-granular late fusion；
2. 推荐 length bias：LBR 的 input attention length bias、feature ablation、equal-length
   truncation/padding，以及 D3 的 effective/ghost token 讨论；
3. 推荐 token/information budget：VarLenRec/PIBA、Token-Weighted Multi-Target、
   I-LLMRec、长行为建模与 ReLLa/ReLLaX；
4. prompt/context compression：LLMLingua、LongLLMLingua、LLMLingua-2、RECOMP、
   rate–distortion prompt compression；
5. RAG/FiD evidence allocation：RFiD、MGFiD、FiDO、passage pruning、dynamic
   retrieval budget；
6. 任何在生成推荐中已针对 collaborative field 与 metadata field 做 source-aware
   hard-budget allocation、paired reserialization 或 target-free adaptive CF budget
   的工作。

三项必要 gate：

| Gate | PASS 条件 |
|---|---|
| GRAM structural specificity | 未发现工作已把 GRAM 类 fine passage 中“前置 collaborative lexical field 造成后置 metadata 右截断”定义为可检验推荐机制 |
| displacement identifiability | 未发现工作已在固定 passage/token budget 与固定 CF identity 下，通过 reserialization/budget recovery 区分 CF noise 与 metadata displacement |
| intervention room | 未发现工作已在 GRAM 类生成推荐中，用 training-only/static、target-free 信号对 collaborative 与 metadata field 做 per-item budget allocation；通用 compression、length calibration 或 dataset-level top-k 不算本项首创 |

允许的通过结论只能是
**`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_GRAM_NARROWING`**。Dynamic budget、
prompt compression、field selection、counterfactual reserialization 与 top-k selection
均必须承认来自既有领域；潜在贡献只保留：

> GRAM-native collaborative-prefix metadata displacement diagnosis，以及由该机制
> 证据驱动的 recommendation-specific、target-free field-budget allocation。

任一必要 gate 被实质覆盖则固定 `STOP_CPBD_NOVELTY`，G0-D 不解锁。检索结论必须带
“截至 2026-07-24、在固定近邻簇中未检出”的范围限定，不得使用绝对 first claim。

固定预期产物：

- `artifacts/phase3/cpbd_g0/novelty_matrix.csv`
- `artifacts/phase3/cpbd_g0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_CPBD_G0差异审计.md`

### 20.3 G0-N 后的条件式路线

- **G0-D1：static truncation census。** 仅在 G0-N 通过后、读取新的诊断结果前
  另行预注册。必须精确重放 `CollatorGRAM` tokenizer/filter/truncation，按 item、
  popularity 与 dataset 统计各 field 的原始/可见/lost token；当前不预锁门槛。
- **G0-D2：frozen budget-recovery diagnosis。** 仅当双数据集 D1 证明真实而非
  偶发的 metadata loss 后另行预注册。必须固定 128-token budget、passage 数和 CF
  identity，至少包含原顺序、metadata-first/reallocated、matched within-metadata
  control；不允许用简单扩长 context 或删除全部 CF 冒充机制证据。
- **G1：target-free field-budget allocator。** 仅当 D2 证明恢复被挤出的 metadata
  对 gold path 有跨数据集收益，且收益不能由 position/control 解释时设计。输入只能
  使用 training-only CF statistics、metadata length/field presence 与 item frequency，
  不得读取 target 或 validation success。
- **G2：correctness smoke。** 只验证 serialization、budget conservation、梯度、
  合法解码、显存与延迟。
- **G3：一次性小 pilot。** 一个主方案、一个固定-budget baseline、固定 seed/cohort；
  样本量、终点和门槛由 G0–G2 后另行锁定。
- **G4/G5：跨种子确认与全量/test。** 只有前一步通过后才具体化。

G0-N 通过不代表 displacement 存在，也不解锁诊断、训练或 GPU。任何后续失败都不得
改成一般 prompt compression、简单 top-k tuning、扩长 max token 或单数据集故事。

### 20.4 G0-N 实际结果

G0-N 于 2026-07-24 在任何新 truncation census 前完成。三项必要 gate 的固定结果为：

| Gate | 结果 |
|---|---|
| GRAM structural specificity | PASS |
| displacement identifiability | PASS |
| intervention room | PASS WITH STRONG NARROWING |

固定决策为
**`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_GRAM_NARROWING`**。这不表示“token budget”
或“reordering”本身新颖；只保留 GRAM-native collaborative-prefix metadata
displacement、固定预算/固定 CF 的 paired diagnosis，以及在该机制成立后才可能设计的
recommendation-specific target-free field allocator。

产物：

- `artifacts/phase3/cpbd_g0/novelty_matrix.csv`
- `artifacts/phase3/cpbd_g0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_CPBD_G0差异审计.md`

### 20.5 G0-D1：static truncation census 预注册

本节在读取新的 raw/visible/lost token 统计前冻结，配置文件为
`artifacts/phase3/configs/cpbd_g0_d1_preregistered.json`。

**目的。** 只回答当前生产 serialization 是否在 Toys 与 Beauty 中形成广泛而实质的
metadata displacement；不加载 checkpoint、不评分、不训练、不用 GPU、不读 test。

**总体与 lineage。**

- 对各锁定数据集的完整 catalog intersection 做 census：item 必须同时存在于
  `item_plain_text`、lexical index 与 `similar_item_sasrec`。
- Toys 使用当前 `top_k=5`，Beauty 使用当前 `top_k=10`；两者均固定
  `id_linking=1`、`item_prompt=all_text`、`cf_model=sasrec` 和
  `item_prompt_max_len=128`。
- popularity 只由每个用户的 `sequence[:-2]` 计数；`sequence[-2]` validation 与
  `sequence[-1]` test 均排除。zero、nonzero bottom-50%、top-50% 仅作描述。

**精确 tokenizer replay。** 使用本地 `t5-small` fast tokenizer，先按 Collator 的
`max_length=999` 编码，再移除 token IDs `1820/9175`，从左保留 128；若 EOS 不在
保留序列，则以 EOS 替换最后一个 token，随后补齐。通过 offset mapping 统计
link、collaborative、metadata 及 title/brand/categories/description/price/salesrank
字段。`raw` 指 999-token ceiling 内、delimiter filter 后、128 截断前的非
padding/EOS field token；`visible` 指完整 128-token 规则后仍可见者；`lost=raw-visible`。

**唯一 paired construction。**

```text
current:
item: <link>; similar items: <fixed ordered CF IDs>; <metadata>

metadata_first:
item: <link>; <same metadata>; similar items: <same fixed ordered CF IDs>
```

两者固定 item/link、CF identities 与顺序、metadata string、passage 数和 128-token
budget，只交换 collaborative field 与 metadata field 顺序。主量为
`recoverable_metadata_tokens = visible_meta(metadata_first) -
visible_meta(current)`；同时报告交换所付出的 `displaced_cf_tokens`，但后者不作为
挽救 gate。

**完整性 gate。** catalog intersection coverage、parse、finite、fixed-component
identity 与 exact Collator replay 必须全部为 1.0；运行配置还必须精确匹配上述锁定值。
任一失败记 `EXECUTION_INVALID`。

**双数据集机制 gate。** Toys 与 Beauty 各自必须同时满足：

1. `recoverable_metadata_tokens >= 8` 的 item 比例 `>= 0.50`；
2. `recoverable_metadata_tokens` 中位数 `>= 8`；
3. current serialization 的 item-level metadata retention ratio 中位数 `<= 0.90`。

任一数据集失败则固定 **`STOP_CPBD_NO_STRUCTURAL_DISPLACEMENT`**；全部通过才记
**`G0_D2_DESIGN_ALLOWED`**。通过只证明结构性 displacement 与后续诊断空间，不证明
lost metadata 有推荐价值；metadata 子字段与 popularity 分层都是 secondary
descriptive，不得挽救主 gate，也不得在结果后扫描阈值。

### 20.6 G0-D1 实际结果

G0-D1 于 2026-07-24 按 20.5 的冻结协议完成。CPU 单元测试 5/5 通过，完整性 gate
全部为 1.0。

| 数据集 | catalog items | recoverable>=8 | recoverable median | current metadata retention median | displaced CF median |
|---|---:|---:|---:|---:|---:|
| Toys | 11,924 | 0.7642 | 33 | 0.6562 | 29 |
| Beauty | 12,101 | 0.9998 | 83 | 0.2742 | 79 |

双数据集三项机制 gate 全部通过，固定决策为 **`G0_D2_DESIGN_ALLOWED`**。Beauty 的
结构效应尤其强，但本结果仍只是 tokenizer-level mechanism evidence：它不能说明被
恢复的 description/price/salesrank 对推荐有用，也不能说明牺牲 CF visibility 后净值
为正。

产物：

- `artifacts/phase3/cpbd_g0/{Toys,Beauty}/item_truncation_census.csv`
- `artifacts/phase3/cpbd_g0/{Toys,Beauty}/summary.json`
- `artifacts/phase3/cpbd_g0/summary.json`
- `report/第三阶段/GRAM_第三阶段_CPBD_G0_D1诊断报告.md`

### 20.7 G0-D2：frozen budget-recovery outcome diagnosis 预注册

本节在读取任何新的 checkpoint score 前冻结，完整配置为
`artifacts/phase3/configs/cpbd_g0_d2_preregistered.json`。

**总体与固定项。** 使用 S0 已锁定的 Toys epoch-30、Beauty epoch-25 checkpoint，
以及 CGI 已冻结的双数据集各 256 tail-miss + 256 tail-hit validation cohort。保持
模型权重、gold lexical target、coarse prompt、history passage 数、每 passage
128-token budget、link ID、ordered CF identities 和 metadata string 不变；不训练、
不读 test。唯一完整重排是把每个 fine passage 的 metadata 放到 collaborative field
之前。

**五个条件。**

1. `current`：当前生产 serialization；
2. `metadata_first_full`：固定内容的 metadata-first；
3. `metadata_first_minus_all_recovered`：在条件 2 的相同 IDs/positions/layout 上，
   只 mask 超出 current metadata visible-rank cutoff 的全部新可见 metadata tokens；
4. `metadata_first_minus_recovered_slice8`：对 recoverable metadata 至少 8 个的
   fine passage，mask 最先恢复的 8 个；低于 8 的 passage 在两个 slice 条件都不 mask；
5. `metadata_first_minus_matched_visible_slice8`：对完全相同的 eligible passages，
   在同一 metadata-first passage 中 mask current cutoff 前紧邻的 8 个原本可见
   metadata tokens，作为逐 passage token-count-matched secondary control。

条件 3 把“新恢复内容”从同一 metadata-first layout 中拿掉，因此
`full - minus_all_recovered` 直接检验 recovered content contribution；条件 4/5
比较相同 8-token metadata slice 的局部价值，但 matched control 仅作 descriptive，
不得挽救主 gate。

**主量。**

- `net_reallocation = lp_metadata_first_full - lp_current`
- `recovered_all_contribution = lp_metadata_first_full -
  lp_metadata_first_minus_all_recovered`
- `residual_layout_effect = lp_metadata_first_minus_all_recovered - lp_current`
- `recovered_slice8_contribution = lp_metadata_first_full -
  lp_metadata_first_minus_recovered_slice8`
- `matched_visible_slice8_contribution = lp_metadata_first_full -
  lp_metadata_first_minus_matched_visible_slice8`

**完整性 gate。** cohort identity、coarse prompt identity、raw component identity、
fixed-budget、recovered mask localization、matched-slice eligibility 与 finite rate
均须为 1.0；current 重算误差须 `<=1e-7`；tail miss/hit 各须 256 人。任一失败记
`EXECUTION_INVALID`。

**双数据集串行机制 gate。** 对 Toys 与 Beauty 的 tail miss 均要求：

1. net mean `>=0.02`、cluster bootstrap 95% CI 下界 `>0`、positive rate `>=0.55`；
2. recovered-all mean `>=0.02`、95% CI 下界 `>0`、positive rate `>=0.55`；
3. recovered-slice8 mean `>=0.005` 且 95% CI 下界 `>0`；
4. recovered-all mean 至少占 net mean 的 50%；
5. tail-hit net mean 不得低于 `-0.01`。

依次失败记 `STOP_CPBD_NO_NET_VALUE`、`STOP_CPBD_NO_RECOVERED_VALUE` 或
`STOP_CPBD_BROAD_HARM`；全部通过才记 **`G1_DESIGN_ALLOWED`**。failure association、
residual layout 和 matched slice 均只作 secondary descriptive。不得在同一 validation
上改 serialization、mask size、cohort、top-k、context length 或阈值寻找通过结果。

### 20.8 G0-D2 实际结果与 CPBD 终止决策

G0-D2 于 2026-07-24 按 20.7 的冻结协议一次性完成。8/8 CPU 单元测试通过；
Toys/Beauty 各 256 tail miss + 256 tail hit。cohort identity、coarse prompt、
raw component、fixed budget、mask localization、matched-slice eligibility 和
finite gate 均为 1.0，current score 重算误差为 0。未训练、未生成 beam、未读取
test，GPU3 的 CodeLlama 资源在运行后恢复。

| 数据集/stratum | net mean | net 95% CI | net P(>0) | recovered-all mean | recovered-all 95% CI | slice8 mean |
|---|---:|---:|---:|---:|---:|---:|
| Toys tail miss | 0.066255 | [-0.026730, 0.153510] | 0.546875 | 0.015748 | [0.004449, 0.027150] | 0.002663 |
| Toys tail hit | -0.855011 | [-1.066960, -0.660584] | 0.238281 | -0.011434 | [-0.029773, 0.004947] | -0.006722 |
| Beauty tail miss | -0.070700 | [-0.270813, 0.089167] | 0.585938 | 0.021275 | [0.003927, 0.039174] | 0.004733 |
| Beauty tail hit | -2.180000 | [-2.668435, -1.721842] | 0.144531 | -0.024237 | [-0.059468, 0.009293] | -0.009538 |

两数据集的第一道 `net_value` gate 均失败，因此固定决策为
**`STOP_CPBD_NO_NET_VALUE`**。Toys 的 mean 虽为正，但 CI 跨 0 且 positive rate
比 0.55 少 1/256；Beauty mean 为负。按预注册串行规则，到此已经停止，不能用后续
gate 挽救。作为解释而非晋级依据，recovered-all 在 tail miss 上确有小幅正贡献且
双数据集 CI 均为正，但固定 metadata-first 的 residual layout/CF visibility 代价更大；
对原本 tail-hit 用户的净损失尤其严重。

因此 D1 的结论应限定为“结构性 metadata displacement 真实存在”，不能升级为
“恢复越多 metadata 越好”。G1–G5 均不解锁；不得在同一 validation 上改为只服务
tail miss、扫描 confidence threshold、字段排列、top-k 或 context length，也不得把
recovered-all 的 secondary 正值单独包装成成功。

产物：

- `artifacts/phase3/cpbd_g0_d2/{Toys,Beauty}/cohort.csv`
- `artifacts/phase3/cpbd_g0_d2/{Toys,Beauty}/span_audit.csv`
- `artifacts/phase3/cpbd_g0_d2/{Toys,Beauty}/counterfactual_scores.csv`
- `artifacts/phase3/cpbd_g0_d2/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/cpbd_g0_d2/summary.json`
- `report/第三阶段/GRAM_第三阶段_CPBD_G0_D2诊断报告.md`

### 17.6 D1 以后：只保留轮廓

以下阶段均未解锁。每一步必须先写回上一阶段实际结果，再由结果具体化下一步。

- **D1：干预设计与反事实验证。** 仅在 `D0_MECHANISM_ALLOWED` 后，比较
  frozen-base prior neutralization 与保持 Trie/item legality 的实现选择；先证明它确实
  改变被识别的机制而非仅改 score scale，再完整预注册。
- **D2：correctness 与极小 smoke。** 只验证数值、合法解码、梯度/无梯度边界、
  延迟和显存；不做效果结论。
- **D3：一次性小型 pilot。** 一个主方案、一个 matched baseline、一个 seed、固定
  cohort 与 beam；样本量、终点和门槛由 D0–D2 后另行锁定。
- **D4：中型跨种子确认。** 仅在 D3 通过后依据效应与方差设计。
- **D5：全量与 test。** 仅在双数据集 validation 确认、方案和 checkpoint 规则冻结后
  才决定是否执行一次性 test。

### 17.7 D0-N 实际结果

D0-N 于 2026-07-24 在任何 D0-D 结果读取前完成。GRAM/GRLM 使用原生 lexical
identifier，但未审计冻结预训练先验；APAO 与 Latte 分别研究训练—推理 prefix gap 和
tree probability coupling；Token-Weighted/Ghost 研究推荐训练频率与 popularity。
最接近的 Decoding Matters 已研究 length normalization 对 near-one ghost token 的
放大并使用 text-free assistant，Calibrate Before Use 已研究 content-free answer-prior
calibration。它们要求 NLPL 收窄主张，但没有覆盖本节 locked mechanism。

| Gate | 结果 |
|---|---|
| mechanism isolation | PASS |
| GRAM-specific audit | PASS |
| intervention space | PASS WITH NARROWING |

固定决策为 **`NOVELTY_SCOPE_PASS_WITH_NARROWING`**，只解锁 D0-D。产物为：

- `artifacts/phase3/nlpl_d0/novelty_matrix.csv`
- `artifacts/phase3/nlpl_d0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_NLPL_D0差异审计.md`

### 17.8 D0-D 实际结果与 NLPL 终止决策

D0-D 于 2026-07-24 按预注册从头完成。首次执行在完成 Toys 计算后因 NumPy
`bool_` 无法 JSON 序列化而退出；只做类型转换修复后从头重跑，公式、输入、seed 与
门槛均未改变。最终 CPU 单元测试 6/6 通过，Toys/Beauty 的 Recall@10/50 均与 S0
精确一致，所有 prediction 每行 50 个候选均可映射且无重复，修改
`sequence[-2:]` 不改变 training-only frequency。未加载 GRAM checkpoint、未读取
test、未训练、未使用 GPU。

| 数据集 | non-tie pairs | concordance | cluster bootstrap 95% CI | permutation p | tail miss OR | 通过 gate |
|---|---:|---:|---:|---:|---:|---|
| Toys | 1,129 | 0.443756 | [0.399544, 0.483810] | 0.995900 | 0.549751 | support |
| Beauty | 266 | 0.537594 | [0.460673, 0.616862] | 0.188981 | 0.542580 | support |

两个数据集的 concordance、uncertainty、randomization 和 recommendation-link 均失败。
Toys 的主关系方向还与 D-H1 相反；两个 OR 也都低于 1，而不是预注册要求的
`>=1.10`。因此固定决策为 **`STOP_NLPL_NO_EXPOSURE`**。

这意味着现有证据不支持把 frozen T5 native lexical prior 作为 GRAM beam-50 tail
失败的跨数据集机制。D1–D5 均不解锁；不得在同一 validation 上改 neutral prompt、
0.5 smoothing、sibling depth、frequency ratio 或阈值寻找显著结果。

产物：

- `artifacts/phase3/nlpl_d0/{Toys,Beauty}/item_native_prior.csv`
- `artifacts/phase3/nlpl_d0/{Toys,Beauty}/matched_pairs.csv`
- `artifacts/phase3/nlpl_d0/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/nlpl_d0/summary.json`
- `report/第三阶段/GRAM_第三阶段_NLPL_D0诊断报告.md`

## 21. 新周期 H：Post-Encoder Norm Shortcut（PENS）

### 21.1 Material Passport、研究问题与非重试边界

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-24
- Verification Status: ANALYZED（H0-N）；PREREGISTERED（H0-D）；UNVERIFIED（H1 及后续）
- Version Label: `pens_h0_d_preregistered_v1`
- 当前允许动作：冻结 checkpoint 的 H0-D 诊断；不得训练、生成 beam、读取 test 或扫描范数目标。

GRAM 对每个 passage 独立完成 T5 encoder 后，执行：

```text
H'_p,t = H_p,t + P_p
```

其中同一个 learned passage-position vector `P_p` 被广播到该 passage 的所有 token，
且相加后不再经过 encoder normalization，直接成为 decoder cross-attention 的
key/value 来源。方向 H 检验的不是“GRAM 不应使用顺序”，而是：

> `P_p` 的方向与范数是否承担不同作用；训练曝光不均衡是否让范数形成与 passage
> 内容无关的 ordinal magnitude shortcut，并对 tail failure 产生可消除的负效应。

只读 checkpoint census 作为 result-informed、post-hoc 动机：

| 数据集 | `||P_1||` | `||P_20||` | training exposure–norm Pearson |
|---|---:|---:|---:|
| Toys | 0.4539 | 2.2888 | -0.9740 |
| Beauty | 0.3821 | 2.0601 | -0.9522 |

该相关性不是缺陷结论。GRAM 原论文的 position-embedding ablation 以及当前 validation
按历史长度统计都允许一个更强反解释：范数增长可能是有效的 recency/age code，长历史
样本也没有普遍更差。因此 H0-D 必须证明**保留方向、只消除范数日程**有跨数据集净收益；
仅有范数相关、score 改变或 zero-position 结果均不能晋级。

PENS 与 A–G 的边界：

- 不改变 identifier、Trie、beam loss 或 candidate exposure，不重试 HBTR/CAMI/NLPL；
- 不删除 passage、token span 或 fine evidence，不重试 CGI/LEI；
- 不改变字段顺序、CF identity、metadata 或 token budget，不重试 CPBD；
- 不声称首次研究 positional encoding、additive PE、embedding normalization 或
  norm bias；潜在贡献只限 GRAM 类 post-encoder passage-position 的机制诊断。

### 21.2 H0-N：原文级差异审计及实际结论

本轮固定近邻簇覆盖：

1. GRAM 的 post-encoder passage-position addition 与原论文 ablation；
2. learned absolute position embedding 的几何、低维结构、位置脆弱性；
3. sequential recommendation 的 positional attention、multi-temporal encoding 与
   additive position/semantics disentanglement；
4. FiD/RFiD/MGFiD 的 passage guidance、reranking、anchor/rationale embedding；
5. item/session embedding norm、popularity bias 与 normalization。

必要 gate：

| Gate | 结果 |
|---|---|
| post-encoder structural specificity | PASS：未检出工作直接审计 GRAM 类 post-encoder、broadcast-to-all-token 的 passage-position 范数 |
| exposure–norm mechanism | PASS WITH NARROWING：通用 PE geometry/norm 已有先例，但未检出该结构中的训练曝光—范数分层 |
| norm-only causal identifiability | PASS：未检出保持每个 passage-position 方向、只统一范数的 frozen causal counterfactual |

截至 2026-07-24，在上述有界检索中固定结论为
**`NOVELTY_SCOPE_PASS_WITH_STRONG_MECHANISTIC_NARROWING`**。允许的潜在主张只能是：

> 生成式序列推荐中 post-encoder passage-position embedding 的曝光相关范数分层，
> 以及 direction-preserving norm-only 因果诊断。

不得使用绝对“首次”表述；PARec/FPARec、DIET、absolute-PE robustness 与 embedding
norm bias 必须列为直接近邻。产物：

- `artifacts/phase3/pens_h0/novelty_matrix.csv`
- `artifacts/phase3/pens_h0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_PENS_H0差异审计.md`

### 21.3 H0-D：direction-preserving frozen norm diagnosis 预注册

本节在读取任何 H0-D 新反事实 score 前冻结，完整机器可读配置为
`artifacts/phase3/configs/pens_h0_d_preregistered.json`。

**总体与 lineage。** 使用 S0 锁定的 Toys epoch-30、Beauty epoch-25 checkpoint，
以及 CGI E0 已冻结的双数据集各 256 tail-miss + 256 tail-hit validation cohort。
该 cohort 已被既有方向使用，故 H 明确标记为 repeated-validation exploratory
mechanism screen，不能直接提供无偏确认效果；若 H0-D 通过，后续必须使用训练期设计
和新的确认边界。保持输入文本、passage 顺序/数量、attention mask、gold lexical
target、decoder、checkpoint 其余权重全部不变；不生成 beam、不读 test、不训练。

**训练曝光量。** 对每个用户只读取 `sequence[:-2]`。精确重放
`MultiTaskDatasetGRAM.load_train`：跳过空历史，对每个 prefix `i=1..len(items)-1`
取最后 20 个历史 item；fine position `p` 的 exposure 是所有训练 prefix 中
`min(i,20) >= p` 的次数。不得读取 `sequence[-2]` 或 `sequence[-1]` 来构造 exposure。

**三个固定条件。**

1. `current`：原 checkpoint 的 position table；
2. `equal_fine_norm`：对 `p=1..20` 写成
   `median(||P_1||,...,||P_20||) * P_p / ||P_p||`；保留每个方向、position 0、
   其余权重和所有输入不变；
3. `zero_all_position`：整个 position table 置零，只作描述性 sanity control，
   不得用于晋级或选择其他 target norm。

唯一主 estimand 为：

```text
norm_only_gain = lp_equal_fine_norm - lp_current
```

其中 `lp` 是排除 EOS/pad 后 gold lexical target token 的 mean log-prob。
`lp_zero_all_position - lp_current`、按 history length 的分层、position cosine/PCA
均为 secondary descriptive，不得挽救主 gate。

**完整性 gate。**

1. tail-miss/tail-hit 各 256，cohort identity 与 CGI E0 完全一致；
2. checkpoint position table 与模型内 duplicate reference 一致，current 恢复误差
   和 current 重算误差均 `<=1e-7`；
3. equal-fine-norm 的 fine norm 最大偏差 `<=1e-6`，fine direction cosine 最小值
   `>=0.999999`，position 0 与所有非 position 参数不变；
4. exposure 精确排除最后两个 item，所有分数 finite；
5. 未读 test、未生成 beam、未训练，GPU 资源按第三阶段统一协议恢复。

**双数据集串行机制 gate。**

1. 结构 replication：每个数据集 exposure–norm Pearson `<=-0.90`，且
   `||P_20|| / ||P_1|| >=3.0`；
2. 对 tail-miss，`norm_only_gain` mean `>=0.02`、cluster bootstrap 95% CI 下界
   `>0`、positive rate `>=0.55`；
3. 对 tail-hit，mean `>=-0.01`；
4. 以上所有条件必须同时在 Toys 与 Beauty 成立。

依次失败固定为 `EXECUTION_INVALID`、`STOP_PENS_NO_STRUCTURAL_REPLICATION`、
**`STOP_PENS_NO_CAUSAL_BENEFIT`** 或 `STOP_PENS_BROAD_HARM`；全部通过才记
`H1_DESIGN_ALLOWED`。不得在相同 validation 上改成 mean norm、position-specific
clip、只改后半段、扫描缩放系数、选择单一数据集或依赖 zero-position 挽救。

### 21.4 H0-D 之后：只保留轮廓

后续均未解锁，必须先回写 H0-D 实际结果再具体化下一步：

- **H1：training-only norm/direction disentanglement。** 仅在
  `H1_DESIGN_ALLOWED` 后比较一个 target-free 参数化或正则化方案；不得读取
  validation outcome 选择 position 或范数。
- **H2：correctness smoke。** 验证方向保持、梯度、checkpoint、合法生成、显存和
  延迟，不作效果结论。
- **H3：一次性小 pilot。** 一个主方案、一个 matched baseline、固定 seed/cohort；
  样本量、终点和门槛由 H0–H2 后另行预注册。
- **H4/H5：跨种子确认与全量/test。** 仅在前一步通过后逐步具体化。

H0-D 失败即终止 PENS；不得把结构相关性、zero-position ablation 或单数据集 score
变化包装为成功。

### 21.5 H0-D 实际结果与 PENS 终止决策

首次实现因偶数样本 median 取值错误固定记为
`EXECUTION_INVALID_MEDIAN_IMPLEMENTATION`，无效产物保存在
`artifacts/phase3/pens_h0_d_execution_invalid_lower_median/`。唯一修复为把 target
改成排序后第 10/11 个 fine norm 的平均，并增加精确测试；没有改变 cohort、seed、
条件、endpoint 或 gate。

修正后的 H0-D 于 2026-07-24 从头完成。CPU 单元测试 4/4 通过；双数据集各
256 tail-miss + 256 tail-hit。完整性 gate 全部通过：

- CGI E0 cohort 完全一致，所有分数 finite；
- equal-fine-norm 最大误差均为 `1.19e-7`，最小方向 cosine 均为
  `0.99999988`，position 0 误差为 0；
- current repeat 与干预后 restore 最大误差均为 0，模型和 encoder position table
  共享 storage；
- 未训练、未生成 beam、exposure 排除 `sequence[-2:]`、未使用 test；
- GPU3 的 CodeLlama reservation 已恢复。

| 数据集 | exposure–norm Pearson | `||P20||/||P1||` | tail-miss gain mean | 95% CI | P(>0) | tail-hit gain mean |
|---|---:|---:|---:|---:|---:|---:|
| Toys | -0.973954 | 5.042281 | 0.004552 | [-0.067095, 0.079676] | 0.449219 | -0.340117 |
| Beauty | -0.952188 | 5.390997 | -0.851986 | [-0.998229, -0.700632] | 0.148438 | -0.622155 |

结构 replication 在双数据集强通过，但 causal-benefit gate 在双数据集失败：
Toys 的 mean、CI 与 positive rate 均不达标；Beauty 则是大幅、置信区间严格为负的
退化。按预注册串行规则，固定决策为 **`STOP_PENS_NO_CAUSAL_BENEFIT`**。tail-hit
也存在明显 broad harm，但它是主 gate 已失败后的进一步反证，不改变停止类型。

作为 secondary descriptive，zero-position 在 Toys tail-miss 上为正
（0.09781，95% CI [0.05037, 0.14810]），但在 Beauty tail-miss 近零且 CI 跨 0，
并在两数据集 tail-hit 上为负。它既不跨数据集，也删除方向和范数两种信息，按计划
不得用于挽救或派生新的 position ablation sweep。

因此当前证据支持的结论只有“曝光相关范数分层真实存在”，不支持它是一个可通过统一
范数修复的 harmful shortcut。H1–H5 均不解锁；不得在相同 validation 上尝试 mean
norm、只处理 positions 10–20、norm clipping、全局缩放、per-position gate 或只讲
Toys。

有效产物：

- `artifacts/phase3/pens_h0_d/{Toys,Beauty}/cohort.csv`
- `artifacts/phase3/pens_h0_d/{Toys,Beauty}/position_census.csv`
- `artifacts/phase3/pens_h0_d/{Toys,Beauty}/counterfactual_scores.csv`
- `artifacts/phase3/pens_h0_d/{Toys,Beauty}/diagnostic_summary.json`
- `artifacts/phase3/pens_h0_d/summary.json`
- `report/第三阶段/GRAM_第三阶段_PENS_H0_D诊断报告.md`

## 22. 新周期 I：Selective Metadata Budget Recovery（SMBR，草案）

### 22.1 为什么不是第九个同类 frozen intervention

方向 A–H 已经形成清楚的证据链：

1. 关系信号、层次 hard negative、fine evidence、metadata 和位置结构都不是“完全没
   信号”；
2. 失败反复发生在把局部或子组信号升级为全局干预时；
3. CPBD D2 尤其明确：tail-miss 的 `recovered_all_contribution` 在 Toys/Beauty
   均为正且 95% CI 下界大于 0，但 `metadata_first-current` 的净效应不稳定，并对
   tail-hit 造成大幅伤害；
4. 因此下一问题不应是再找一个固定字段顺序、norm target、mask 或 loss weight，而是
   先问“净受益的情形是否能在看不到目标时被可靠识别”。若不能识别，保持 current
   serialization 就是正确结论。

SMBR 不修改 CPBD 的停止决定，也不把 CPBD 的 secondary positive result 升格为成功。
它是看到 D2 后建立的独立、result-informed、post-hoc 周期。

### 22.2 暂定研究问题与最小机制

暂定研究问题：

> 对 GRAM 当前会发生 metadata displacement 的历史输入，能否只用 training-only
> counterfactual benefit labels 和推理时可得的 target-free 结构特征，学习一个带
> abstention 的 serialization policy：只有在高置信度净受益时恢复 metadata，其余
> 样本严格保持 current serialization？

最小动作空间只能是：

```text
KEEP_CURRENT
RECOVER_WITH_FIXED_BUDGET
```

当前不决定第二个动作的最终实现是 paired reserialization、overflow passage 还是
source-aware compression；I0-N 必须先判断哪一种仍有非文字性新颖空间。策略默认
`KEEP_CURRENT`，不得用 validation tail-miss/hit 身份、gold target、beam rank 或
当前 checkpoint 的 validation outcome 作为输入。

### 22.3 I0-N：唯一当前解锁的原文级差异审计

I0-N 至少覆盖并逐项记录以下近邻：

- GRAM 与本计划 CPBD；
- LLMLingua、LongLLMLingua、LLMLingua-2、RECOMP 与 rate–distortion prompt
  compression；
- RFiD、MGFiD、FiDO、dynamic retrieval/token budget 与 source-aware pruning；
- selective prediction、learning to defer、conformal risk control 和 conservative
  policy improvement；
- RecLM-cgen/scope-mask loss、TrieRec、APAO、PRO 与 BONSAI，防止把训练—解码
  对齐或 Trie 结构优化误写成本方向贡献；
- 已有 LLM/生成式推荐中按用户或 item 动态选择 collaborative/metadata source 的工作。

每篇至少记录：决策单位、动作空间、监督来源、是否需要 gold/validation outcome、
是否有 abstention/default policy、预算是否严格守恒、是否区分 content value 与
layout cost、确认数据是否独立。

以下三项是“借鉴后新增价值”的必要 gate，而不是“别人做过相似组件就停止”：

| Gate | PASS 条件 |
|---|---|
| mechanism-conditioned selectivity | 借鉴的 selector/compressor 被实质改造成 GRAM-native collaborative-prefix displacement 的逐样本净收益决策，并胜过原方法与简单 threshold |
| target-free conservative policy | 新增 training-only paired benefit supervision、target-free features 与 current-layout identity fallback，并能控制 broad harm |
| independent-boundary contribution | 新增部分的价值可在未参与 A–H 方向生成的数据集确认，而不是依赖 Beauty/Toys repeated validation |

I0-N 的固定结论为
`TRANSFER_INNOVATION_ALLOWED_I0_D_DESIGN_UNLOCKED`。允许借鉴 compression、
selective prediction、conservative policy learning 与 counterfactual supervision；
论文贡献只能落在 GRAM-native displacement、training-only benefit learnability、
keep-current identity fallback 及其机制验证上。I0-D 后若新增部分不能胜过 borrowed
baseline、固定 recovery、单一 lost-token threshold 与 matched random activation，
再固定为 `STOP_SMBR_NO_ADDED_VALUE`。

固定预期产物：

- `artifacts/phase3/smbr_i0/novelty_matrix.csv`
- `artifacts/phase3/smbr_i0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_SMBR_I0差异审计.md`

### 22.4 I0-N 通过后的条件式路线（均未解锁）

#### I0-D：training-only benefit learnability（已执行并停止）

预注册文件固定为
`artifacts/phase3/configs/smbr_i0_d_preregistered.json`，版本
`smbr_i0_d_preregistered_v1`。本阶段只检验 policy label 是否可学习，不更新 GRAM。

**样本与隔离**

1. 每个用户只取一个 training prefix：target 固定为 `sequence[-3]`，history 固定为
   `sequence[:-3][-20:]`；两者均位于 `sequence[:-2]` 内。
2. history 长度限制为 2–20；target/history 必须存在于锁定 GRAM index，history item
   还必须存在于冻结 CPBD census。
3. `sha256(seed|dataset|user) mod 100` 固定用户 split：0–59 为 fit、60–79 为
   calibration、80–99 为 audit；上限分别为 1200/400/400。split 内再按独立 hash
   排序截取，禁止按 label、target、validation outcome 或可恢复 token 筛样本。
4. 特征构造前移除 target；严禁读取 `sequence[-2]`、`sequence[-1]`、validation/test
   target/prediction、hit/miss、beam rank 或 target text。

**标签与动作**

- `KEEP_CURRENT` 的 utility 固定为 0；
- `RECOVER_WITH_FIXED_BUDGET` 对全部 fine passages 使用 CPBD 已验证的
  metadata-first paired reserialization；总宽度仍为 128，passage count、CF identity
  和 raw components 不变；
- 连续 benefit 为 `lp_recover - lp_current`；
- 二分类正例预先固定为 `benefit > 0.02`。

**固定特征**

只使用 history item 的冻结 census 聚合量：history length；recoverable token 的
sum/mean/max、positive fraction、`>=8` fraction；displaced-CF 的 sum/mean；
metadata lost 的 sum/mean；metadata retention 的 mean/min；current visible
metadata/CF sum；`log1p(popularity)` mean/min；tail-item fraction；以及 mean
top-k。共 18 项，不加入 target、文本 embedding、checkpoint score 或事后交互项。

**固定模型与 calibration**

- 唯一主模型为 `StandardScaler + L2 LogisticRegression(C=1, lbfgs)`；
- fit 只拟合参数；calibration 只从
  `{0.50,0.55,...,0.90}` 选择 policy threshold；
- 候选必须同时满足 active rate 0.10–0.40、precision ≥ 0.60；先最大化 calibration
  active mean benefit，再按 precision、threshold 降序打破平局；
- 无候选即 `STOP_SMBR_NO_CALIBRATED_SUBSET`，不得改网格。

**audit 主 gate（Toys 与 Beauty 分别全部通过）**

1. AUROC 点估计 ≥ 0.60 且 bootstrap 95% CI 下界严格大于 0.50；
2. Brier 严格优于 audit prevalence constant，ECE ≤ 0.10；
3. active rate 在 0.10–0.40，precision Wilson 95% CI 下界严格大于 0.50；
4. active mean benefit ≥ 0.02 且 bootstrap 下界严格大于 0；
5. 全体 policy mean gain（inactive 记 0）bootstrap 下界严格大于 0；
6. AUROC 至少高于单一 `recoverable_sum` baseline 0.02；
7. policy gain 点估计高于同 active count 的 recoverable threshold、displacement
   ratio、1000 次 matched-random expectation 和 always-recover。oracle 只报告上界。

任一 integrity gate 失败记 `EXECUTION_INVALID`；任一数据集的 learnability gate
失败记 `STOP_SMBR_NO_BENEFIT_LEARNABILITY`。只有双数据集全部通过才解锁 I1 设计。
本阶段不得用一个数据集挽救另一个，也不得在读分数后修改模型、特征、阈值、样本量
或 gate。

**执行结果（2026-07-24）**

- 两个数据集各 2000 样本全部完成；所有 integrity gates 通过；
- Toys/Beauty calibration probability range 分别为 0.0349–0.3466 和
  0.0931–0.3434，冻结的 0.50–0.90 threshold grid 无候选，主决策固定为
  `STOP_SMBR_NO_CALIBRATED_SUBSET`；
- absolute threshold 与 12.5%/16.0% audit base rate 不匹配是设计缺点，但不能事后
  改规则救援；
- secondary ranking audit 仍失败：audit AUROC 为 0.5172/0.5614，bootstrap lower
  为 0.4286/0.4858；top 10%–40% active mean benefit 全为负；
- 因此结论是“当前 target-free 特征与固定 recovery action 不可形成安全子集”，而非
  “所有借鉴式 selector 永远不可行”。I1/I2 不解锁。

固定产物：

- `artifacts/phase3/smbr_i0_d/summary.json`
- `artifacts/phase3/smbr_i0_d/posthoc_descriptive.json`
- `report/第三阶段/GRAM_第三阶段_SMBR_I0_D诊断报告.md`

#### I1：conservative policy correctness

只有 I0-D 双数据集通过才设计。必须验证：

- abstain 时输入 token、mask、score 与 current 完全一致；
- recovery 时总预算守恒，且 content recovery 与 layout/CF cost 可分别归因；
- 不使用 gold target 或 outcome-conditioned cohort；
- matched more-token / layout control、延迟和显存边界完整。

I1 只做 correctness，不作效果结论。

#### I2：独立确认边界

SMBR 不允许直接在既有 Beauty/Toys validation 上完成主要效果筛选。优先顺序为：

1. 先在 Sports（或另一个未参与 A–H 方向生成的数据集）复现 matched GRAM baseline；
2. 在该数据集上冻结一次性 pilot split、主终点、实际效应、broad-harm 和成本门槛；
3. 只运行一个 SMBR 主方案、一个 current-layout matched baseline 和必要机制对照；
4. Beauty/Toys 只能在方案完全冻结后作为 repeated-validation exploratory
   transport check，不能反向改变设计；
5. test 仍只在跨数据集 validation 确认和 checkpoint 规则锁定后读取一次。

### 22.5 停止规则与备选论文形态

以下任一发生即停止 SMBR：

- I0-N 被已有工作实质覆盖；
- training-only benefit label 无法跨用户稳定学习或校准；
- 高精度 active subset 只能靠极低覆盖、target proxy 或 dataset-specific threshold
  获得；
- independent dataset 上不能消除 broad harm；
- 收益可由更多 token、额外参数或额外计算完全解释。

若 SMBR 停止，不再自动建立 J/K/L 机制周期。A–I 的预注册失败链应转为独立的
**falsification-first empirical study**：系统报告“结构异常、局部贡献、oracle
headroom 与真正可训练的跨数据集净收益为何不同”，以方法学/负结果贡献为目标，而
不是继续在同一 validation 上派生新干预。

### 22.6 本轮增量文献定位

- APAO：<https://arxiv.org/abs/2603.02730>
- RecLM-cgen / scope-mask loss：
  <https://aclanthology.org/2026.findings-acl.310/>
- TrieRec：<https://arxiv.org/abs/2602.21677>
- Prefix Retention Optimization：
  <https://arxiv.org/abs/2606.09241>
- BONSAI：<https://arxiv.org/abs/2607.16633>
- Sequential Data Augmentation / GenPAS：
  <https://arxiv.org/abs/2509.13648>

这些来源用于划定可借鉴组件与不可重复声称的贡献。补充近邻为 RECOMP、ACC-RAG、
SARA、Decision-Aware Memory Cards、Learning to Defer 与 AdaptRec。完整审计见：

- `artifacts/phase3/smbr_i0/novelty_matrix.csv`
- `artifacts/phase3/smbr_i0/claim_evidence.json`
- `report/第三阶段/GRAM_第三阶段_SMBR_I0差异审计.md`

当前固定状态为 **`STOP_SMBR_NO_CALIBRATED_SUBSET`**。

## 23. 新周期 J：Field-Factorized Non-Degrading Fusion（FFNF，草案）

### 23.1 换的是机制层级，不是换 selector

SMBR 说明：用静态 target-free census features 预测“谁应接受 metadata-first”不可行；
但它没有检验 end-to-end 模型能否在表示与训练目标层面避免 source interference。
CPBD 又说明单一 passage 内存在真实的 CF/metadata budget competition，而全局换顺序
会把一种来源的恢复变成另一种来源的损失。

FFNF 因而不再做逐样本 recovery policy。它把每个历史 item 的 fine input 从一个
混合 passage 改为两个字段因子化 micro-passage：

```text
CF stream:   item: <same lexical link>; similar items: <same CF identities>
META stream: item: <same lexical link>; <same metadata>
```

两流使用共享 T5 encoder。它们继承相同的 history item-position embedding，但增加
不同的 field-source embedding。编码后在同一 item group 内拼接，再交给原 GRAM
decoder；输出 textual ID、Trie 和 constrained decoding 不变。

### 23.2 固定预算原则

首个方案只允许 `64 CF tokens + 64 META tokens`，合计仍为每个历史 item 128 tokens。
不增加 decoder 可见 token 数、history item 数、CF identity 或 metadata raw text。
两个 64-token encoder self-attention 的理论二次项小于一个 128-token encoder，但必须
实测 wall time、显存和 decoder cross-attention，不能只凭理论声称更高效。

以下均暂时禁止：

- dataset-specific quota；
- 根据 validation outcome 动态分配 64/64；
- 增加总 token、额外 pretrained encoder 或第二个 decoder；
- 修改 textual identifier、Trie、beam size 或 top-k CF；
- 把 Beauty/Toys 已观察 outcome 用于选择 quota。

### 23.3 Non-degrading fusion objective

对同一个 training prefix，在共享参数下计算：

- `CE_full`：CF 与 META 两流均可见；
- `CE_cf`：只保留 CF 流；
- `CE_meta`：只保留 META 流。

暂定目标为：

```text
L = CE_full
  + alpha * (CE_cf + CE_meta) / 2
  + lambda * [
      relu(CE_full - stopgrad(CE_cf) + margin)
    + relu(CE_full - stopgrad(CE_meta) + margin)
    ]
```

其中 auxiliary branch CE 防止通过故意破坏单流来满足约束，`stopgrad` 防止 margin
项直接推高参照分支。核心 estimand 不是 attention balance，而是：

```text
ND_cf   = LP_full - LP_cf
ND_meta = LP_full - LP_meta
```

即完整融合是否至少不劣于任一单来源。`alpha/lambda/margin` 在 J0 结束前必须只锁定
一个主配置；不得在 Beauty/Toys validation 上搜索。

### 23.4 与近邻工作的差异边界

- [GRAM](https://aclanthology.org/2025.acl-long.1596/) 已做 coarse/fine
  multi-granular late fusion；FFNF 只主张进一步拆解 fine item passage 内的
  collaborative/metadata fields。
- [EAGER](https://arxiv.org/abs/2406.14017) 已做 behavior/semantic two-stream、
  shared encoder、separate decoders、contrastive/transfer tasks；FFNF 不主张首次
  two-stream，而是使用单一 GRAM textual-ID decoder、固定 input budget 和
  source-non-degradation constraint。
- [RRCM](https://arxiv.org/abs/2605.07129) 已在 collaborative/meta memories 之间
  用 ranking reward 学习 retrieve/interleave policy；FFNF 不做 agentic retrieval、
  RL 或变长 context，而是确定性的 end-to-end field fusion。
- [MGFiD](https://aclanthology.org/2024.findings-naacl.142/) 已做 passage reranking、
  sentence classification、anchor guidance 与 pruning；FFNF 不声称首次 source-aware
  FiD，只保留 recommendation-specific within-item factorization 和 non-degradation
  objective。
- CEMG 已做 collaborative-guided multimodal fusion；FFNF 不涉及图像、RQ-VAE 或
  multimodal tokenization。

允许的论文表述只能是**有机制证据支撑的 transfer innovation**，不能写“首次双流”
“首次异构融合”或“首次 collaborative/metadata 自适应使用”。

### 23.5 分阶段 gate

#### J0-N：原文级差异审计（已完成）

补读 EAGER、RRCM、GRAM、MGFiD、CEMG，以及 multi-source generation、
source-dropout、modality non-degradation/negative-transfer 与 robust multi-view
learning。必要条件不是所有组件无人使用，而是未检出工作已同时定义：

1. GRAM-native within-item CF/metadata factorization；
2. 固定 128-token grouped budget 与共享 item position；
3. 对单一 textual-ID decoder 的 paired source non-degradation loss。

固定结果为 `TRANSFER_INNOVATION_ALLOWED_WITH_STRONG_NARROWING`。two-stream、
source/unimodal auxiliary loss、modality dropout、residual multimodal interaction
和 Pareto gradient-conflict control 都只能作为借鉴组件。未在固定近邻簇中检出
“GRAM-native displacement + grouped 128-token capacity + shared item position +
single textual-ID decoder paired non-degradation”的完整组合，但该结论不是绝对
first。结构化产物：

- `artifacts/phase3/ffnf_j0/novelty_matrix.csv`
- `artifacts/phase3/ffnf_j0/claim_evidence.json`

#### J0-S：静态预算可行性（CPU only，已执行并停止）

不读取 checkpoint outcome，精确走 tokenizer/filter/truncation 路径，比较 current
128 与 FFNF 64+64：

- CF identity coverage 与 metadata field coverage；
- 每 item/group 的 decoder-visible token 恒等；
- link duplication 成本；
- title/brand/category 等关键字段是否因 64-token META cap 反而广泛下降；
- Toys/Beauty 均只作结构 feasibility，不作效果筛选。

只有两数据集都满足：

- total visible token = 128；
- CF identity coverage 不低于 current 的 95%；
- title/brand/category aggregate coverage 不低于 current；
- metadata recoverable gain 严格为正；
- passage/item-position/source identity 可无歧义恢复；

才解锁 J1 correctness smoke。否则记 `STOP_FFNF_BUDGET_INFEASIBLE`。

执行时将“total visible token = 128”精确定义为 padded tensor width / decoder
capacity `64+64=128`；短文本不要求存在 128 个 active non-padding tokens。META
stream 不重复 link，active-token delta 与第二 EOS 只作 J1 confound 描述。

固定结果：

| Dataset | CF visible ratio | Metadata aggregate gain | Gain-positive items | Title ratio | Brand ratio | Categories ratio |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 1.0000 | -234,052 | 0.0000 | 0.9995 | 0.9930 | 0.9969 |
| Beauty | 0.6320 | +345,311 | 1.0000 | 1.0196 | 1.0970 | 1.8918 |

Toys 的短 CF 让 current 原本可留给 metadata 的空间超过 64，META64 因而普遍删减
metadata；Beauty 的长 CF 则使 CF64 丢失 36.8% collaborative tokens。统一 64+64
把 dataset-dependent competition 固化成相反瓶颈，双数据集必要 gate 失败。固定
产物：

- `artifacts/phase3/ffnf_j0_s/{Toys,Beauty}/item_budget_census.csv`
- `artifacts/phase3/ffnf_j0_s/summary.json`
- `report/第三阶段/GRAM_第三阶段_FFNF_J0_S可行性报告.md`

#### J1：training-only correctness smoke

只使用 `sequence[:-2]`；验证 full/CF-only/META-only mask、共享 item position、
source embedding、loss/gradient 有限、checkpoint reload 和 constrained decoding
兼容。只运行一个锁定的 `alpha/lambda/margin`，不看 validation 效果。

#### J2：fresh-dataset method pilot

主要效果确认必须优先在 Sports（或另一个未参与 A–I 生成的数据集）完成：

1. 先复现 matched GRAM baseline；
2. 冻结 FFNF 主配置、current baseline、factorization-only ablation、
   non-degradation-only ablation；
3. 同时报告 Recall/NDCG、`ND_cf/ND_meta`、broad harm、训练/推理成本；
4. 只有新数据集通过后，才能把完全冻结的配置带回 Beauty/Toys 作 transport check；
5. test 仍只允许在 checkpoint selection 规则冻结后读取一次。

### 23.6 最强反对意见与停止规则

最强反对意见是：FFNF 只是 EAGER/multi-source FiD 的 GRAM 版本；64/64 split 可能用
人为 quota 替代原来的右截断，non-degradation hinge 也可能只增加三个 forward 的
成本而不改善 top-k recommendation。

以下任一发生即停止：

- J0-N 证明方法差异只剩字段名变化；
- 64+64 不能同时保存 CF identity 和关键 metadata；
- branch auxiliary/hinge 通过恶化单流或扩大 logit norm 满足；
- factorization-only 已解释全部收益，non-degradation objective 无增量；
- 收益来自更多 decoder-visible token、额外参数或不可接受计算；
- fresh dataset 上不改善主终点或出现 broad harm。

当前固定状态为 **`STOP_FFNF_BUDGET_INFEASIBLE`**。

## 24. 新周期 K：Advantage-Gated Privileged Trie Distillation（AGPTD，草案）

### 24.1 转向理由与核心假设

J 的失败来自一个部署期约束：把固定 128-token 容量预先切成 CF/META 两份，会在
不同数据集上制造相反瓶颈。K 不再改变部署输入布局，而把更长证据严格限制在训练期：

- **student view**：现有 GRAM 序列化与 128-token 截断，推理路径完全不变；
- **privileged teacher view**：同一历史、同一字段顺序、同一 tokenizer，但 fine
  passage 上限增至 256；不新增字段、不读未来交互；
- **output space**：teacher/student 共用 textual identifier、tokenizer 和 catalog
  Trie；
- **deployment**：只保留 student；参数量、128-token 输入、beam、Trie 与 decoder
  调用次数均与 matched GRAM baseline 相同。

核心假设不是“更长上下文必然更好”，而是：

> 被 current 128-token 截断的证据，至少在一部分 training prefix 上能让 teacher
> 更准确地区分当前 Trie 节点的合法 children；若只蒸馏这些正优势节点，student
> 可以学习跨样本可迁移的偏好规律，而无需在推理时看到 privileged suffix。

### 24.2 方法定义

对目标 textual ID 的第 \(t\) 个 token，令 \(C_t\) 为给定 gold prefix 后 catalog
Trie 的合法 child token 集。teacher 与 student 的 logits 只在 \(C_t\) 内重新归一化：

```text
q_t = softmax(z_teacher[C_t] / tau)
p_t = softmax(z_student[C_t] / tau)
a_t = log q_t(y_t) - log p_t(y_t)
w_t = stopgrad(clip(relu(a_t - delta), 0, w_max))

L = CE_student + lambda * sum_t w_t * KL(q_t || p_t)
```

其中 `y_t` 只用于 training-only advantage gate；推理时不存在 teacher、gold target
或 gate。首个实现固定使用 detached teacher distribution，禁止 teacher/student
互相追逐。若 K0-T 证明 frozen checkpoint 不会利用 128 之后的证据，再考虑联合训练
属于改题，必须停止本周期而不是临时加入。

该设计有三个不可删除的部分：

1. **privileged evidence asymmetry**：teacher 256、student 128；
2. **gold-advantage gate**：只转移 teacher 在正确 child 上优于 student 的节点；
3. **Trie-local KD**：匹配合法 catalog continuation 的条件分布，而非全词表概率。

只做长上下文 teacher、普通 KD、confidence gate 或全词表 token KD，均只能作为
baseline/ablation，不能称为 AGPTD。

### 24.3 与已有工作的边界

- [Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942)
  已提出共享参数的 privileged teacher / unconditioned student 联合训练；K 不声称
  首次使用训练期特权信息，且首轮采用 frozen teacher 而非其 agentic RL 设定。
- [DiSC](https://arxiv.org/abs/2602.16093) 已用不同 context segments 构造
  teacher/student 分布并在共享 token 上做 context distillation；K 不声称首次把
  长 context 压入短 context。
- [Hard Gate KD](https://aclanthology.org/2022.emnlp-main.665/)、
  [ATKD](https://aclanthology.org/2024.acl-long.587/) 与
  [Selective KD](https://arxiv.org/abs/2602.01395) 已覆盖 token/sample 选择、
  teacher/student 状态驱动的门控及自适应教学；K 不声称首次 selective KD。
- [C2KD](https://aclanthology.org/2025.findings-acl.917/) 与
  [ConKD](https://aclanthology.org/2023.emnlp-main.840/) 已覆盖推荐场景的
  teacher-student transfer 和 contextual teacher gate；K 不声称首次推荐蒸馏。
- [RecLM-cgen / scope-mask loss](https://aclanthology.org/2026.findings-acl.310/)
  已在训练中排除 domain-extrinsic tokens；K 不声称首次训练期合法 token mask。

因此允许的贡献表述仅为：

> 面向 constrained generative recommendation 中的 evidence truncation，研究
> teacher 的额外证据何时能改善 gold Trie branch，并以正优势门控的
> catalog-prefix conditional KD 避免将无效或有害的长证据信号传给部署等价 student。

这是借鉴既有组件后的 recommendation-specific transfer innovation，不作绝对
“首次”声明。若完整原文审计发现已有方法同时满足 privileged long evidence、
gold-relative token gate 与 prefix-valid candidate KD，则记
`STOP_AGPTD_NOVELTY`。

### 24.4 K0-T：teacher-informativeness probe（当前唯一解锁）

目的只回答“现有 checkpoint 能否利用 privileged suffix”，不训练模型、不评价
最终推荐效果。

数据边界：

- Toys 与 Beauty 各取由固定 seed 决定的 512 个用户；
- history 截止 `sequence[-4]`、target 固定为 `sequence[-3]`，因此只使用训练
  split；`sequence[-2:]` 的 validation/test item 及其派生 outcome 均不得读取；
- student 与 teacher 使用同一 frozen matched checkpoint；
- 仅纳入至少一个 fine passage 在 current 128-token 后仍有非 padding token 的样本；
  cohort 规则在评分前固定，不按结果筛样本。

必须输出：

- 128/256 两种 view 的 exact token census 与 raw component provenance；
- 每个 gold Trie node 的 `a_t`、teacher/student gold rank、合法 child 数和 entropy；
- sample-level mean gold-path LP difference；
- positive-advantage node/sample coverage；
- 按 suffix 的 CF token、metadata token及二者混合分层，但不得据结果改变输入顺序。

K0-T 同时满足以下条件才解锁 K1：

1. 两数据集 sample-level mean privileged gain 均严格为正；
2. 两数据集 bootstrap 95% CI lower bound 均大于 0；
3. 两数据集 positive-gain sample rate 均至少 0.55；
4. teacher 在 gold rank 上改善的节点数多于恶化节点数；
5. provenance、checkpoint、target exclusion、current replay、finite、Trie-child
   membership 与 128/256 width checks 全部通过。

任一失败即记 `STOP_AGPTD_NO_TEACHER_ADVANTAGE`，不得改成 192/384/512、交换字段
顺序、只保留正样本或换 checkpoint 挽救。

### 24.5 K1/K2：通过 K0-T 后才允许

#### K1：training-only correctness smoke

- 32–64 个 training examples，单一预注册 `tau/lambda/delta/w_max`；
- 对照 `CE only`、`ungated privileged KD`、`full-vocabulary gated KD` 与完整 AGPTD；
- 验证 detached teacher、零权重 identity、singleton Trie child、EOS、mixed
  precision、checkpoint reload 和 constrained decoding；
- 只验证梯度与实现，不据 smoke outcome 搜索超参数。

#### K2：fresh-dataset method pilot

主要效果确认优先放在未参与 A–J 方向生成的 Sports：

1. matched GRAM baseline 与完全冻结的 AGPTD；
2. 三个必要消融：无 privileged view、无 advantage gate、无 Trie-local restriction；
3. 同报 Recall/NDCG、broad harm、teacher-positive coverage、不同 branching depth
   的收益以及训练/推理成本；
4. 推理参数、输入、beam 和 Trie 必须与 baseline 完全相同；
5. Sports 通过后才可把冻结配置带回 Beauty/Toys 作 transport check；test 仍只在
   checkpoint selection 规则锁定后读取一次。

### 24.6 最强反对意见与停止规则

最强反对意见有三项：

1. student 看不到被截断的 item-specific 信息，teacher soft targets 可能不可学习；
2. gold-advantage 是训练期 oracle，可能只筛出容易节点而不能产生可泛化信号；
3. 方法可能只是 privileged distillation、hard-gate KD 与 scope mask 的工程拼接。

以下任一发生即停止：

- K0-T 没有跨数据集 teacher advantage；
- 完整近邻已覆盖三项不可删除机制的同一问题定义；
- ungated KD 或全词表 KD 已解释全部收益；
- gate 只在 singleton/低 branching 节点激活，或有效权重趋近于零；
- student 训练 loss 改善但 fresh-dataset ranking 无增益或出现 broad harm；
- 收益依赖更多推理 token、额外 serving 参数、validation-derived gate 或
  dataset-specific 超参。

当前固定状态为 **`PAUSED_AGPTD_SUPERSEDED_BEFORE_EXECUTION`**；尚未训练、未读取
validation/test。

## 25. 新周期 L：Marginal-utility Aware Reflective Controller（MARC，已停止）

### 25.1 为什么需要比“小门控”更大的改动

方向 A 的 LRC-UCRF 已证明：仅靠 history/retrieval census 预测“top-20 是否覆盖
target”，虽然有一定 discrimination，但概率质量不能跨 Toys/Beauty 同时通过；
SMBR 又证明静态 target-free 特征不能可靠决定序列化策略。因此 MARC 不重复训练一个
输入前的二元 gate，而是改变 GRAM 的内部决策结构：

```text
                      ┌─ semantic expert/probe ─┐
history + item fields ┤                         ├─ utility critic ─ routing action
                      └─ collaborative expert ──┘        │
                                                        ├─ source trust
                                                        ├─ neighbor budget
                                                        ├─ decoder-layer injection
                                                        └─ reflect / do-not-reflect
                                                                  │
                                                        constrained refinement
```

这里的“reflection”不是生成自然语言理由，而是先形成 modality-specific draft
belief，再根据两者分歧、各自不确定性和当前 Trie branching 判断是否重算/修正。

### 25.2 单一核心量：反事实边际效用

对训练样本、gold textual-ID prefix \(y_{<t}\)，定义合法 Trie child 集 \(C_t\)。
在相同 checkpoint 和 target prefix 下计算：

```text
U_sem(t) = CE_without_sem(t) - CE_full(t)
U_cf(t)  = CE_without_cf(t)  - CE_full(t)
U_k(t)   = CE_with_Kprev(t)  - CE_with_K(t), K ∈ {5, 10, 20}
```

正值表示加入该证据降低 gold-child loss，负值表示造成干扰。tiny critic 不直接看
gold token，只读取推理时可得状态：

- semantic/collaborative probe 的 Trie-local entropy、margin 与分歧；
- encoder pooled state、decoder hidden state 与 cross-attention concentration；
- neighbor score/support/age 分布、history sparsity 与 metadata missingness；
- 当前 generation depth、合法 child 数和 prefix ambiguity。

critic 输出 \(\hat U_{sem},\hat U_{cf},\hat U_5,\hat U_{10},\hat U_{20}\) 及 uncertainty。
所有动态行为均由同一预测效用导出：

```text
source trust:    positive U receives weight; negative U is suppressed
neighbor budget: add the next nested neighbor block only if predicted marginal U > cost
layer injection: gate semantic/CF adapters by layer- and prefix-conditioned U
reflection:      only if expected correction utility > second-pass cost
```

这避免分别训练四套互不一致的 heuristics。`abstain` 的含义也固定：某来源预测效用
非正或不确定性超过上界时，该来源 residual gate 回到 identity。

### 25.3 建议模型形态

首版 **MARC-lite** 使用 T5-small 共享主干，不引入第二个大模型：

1. 在 fine-passage encoder output 上增加 semantic/CF source mask 与两个低秩
   expert adapters；
2. 增加两个很小的 auxiliary Trie-local logit probes，形成 semantic/CF draft
   belief；probe 只作状态估计，不单独生成 beam；
3. critic 为两层 MLP 或单层小 Transformer，参数上限锁定为 GRAM 的 2%；
4. decoder 六层各有 identity-initialized residual gate
   \(g^{sem}_{l,t},g^{cf}_{l,t}\)，而不是只在输入端乘一个权重；
5. CF neighbors 使用固定 nested list `top5 ⊂ top10 ⊂ top20`，通过 block mask
   选择预算，不重新训练/检索邻居；
6. 默认单遍解码；只有 critic 预测高 correction utility 时才允许一次 refinement，
   禁止无上限循环。

首版不使用外部闭源 LLM，也不让 LLM生成“可信度标签”。如果后续需要语言反思，
只允许 offline teacher 生成可审计的 preference critique，再蒸馏进 critic；不能把
API 模型留在正式推理链中。

### 25.4 训练目标

```text
L = L_trie_ce
  + alpha * L_source_probe
  + beta  * L_utility_regression
  + gamma * L_corruption_order
  + eta   * L_identity_non_degradation
  + rho   * expected_compute_cost
```

- `L_source_probe`：让两个 probe 分别具备独立预测能力，防止 router 读到空信号；
- `L_utility_regression`：Huber/ordinal loss 拟合 detached counterfactual utility；
- `L_corruption_order`：随机打乱 CF neighbors、遮蔽/替换 metadata 后，对应可信度
  必须下降；这只提供方向监督，不把合成噪声当真实 reliability；
- `L_identity_non_degradation`：当 critic 不确定或预测非正效用时，routing residual
  回到 matched GRAM，而不是强制选一个来源；
- `expected_compute_cost`：约束 neighbor blocks 与第二遍 refinement 的平均成本。

utility labels、corruption 和 gold target 只用于训练；正式推理全部 target-free。

### 25.5 Reflection 与 RL 的位置

#### 不推荐的主方案：外部 LLM 反思

[R4ec](https://arxiv.org/abs/2507.17249) 已采用 actor + reflection model + iterative
refinement，再把知识注入 recommendation backbone。直接复用会带来较大的训练/推理
成本，也很难让自然语言 critic 判断 GRAM 内部 CF neighbor 是否可靠。因此它适合做
相关工作或 offline teacher baseline，不适合作为本项目主创新。

#### 推荐的主方案：内部数值反思

MARC 的 draft 是两个 source probe 对当前 Trie children 的条件分布，critique 是
“哪一来源能减少多少 gold-path loss”的 target-free 预测，refinement 是受成本约束
的一次 residual rerouting。这保留 reflection 的 actor–critic–refine 结构，但对象是
推荐证据与生成路径，不是自然语言推理链。

#### RL 只作为后期扩展

[RRCM](https://arxiv.org/abs/2605.07129) 已用 outcome-only ranking reward 和 GRPO
学习直接推荐/取 collaborative memory/取 metadata/interleave 的策略。因此“前面加
RL 动态选证据”本身不新。且当前 Amazon leave-one-out 数据只有离线 next-item
反馈，没有真实交互环境；直接 PPO/REINFORCE 容易产生高方差和 off-policy 偏差。

若 MARC-lite 已通过监督式 pilot，才允许增加 contextual-bandit fine-tuning：

```text
action = (K, reflect_or_not)
reward = gold Trie log-prob or differentiable NDCG surrogate
         - c1 * encoded_tokens - c2 * second_pass
```

禁止把 RL 同 source expert、critic、layer router 一起从零训练。RL 必须证明在相同
architecture 上相对 supervised/Gumbel routing 有独立增益，否则删除。

### 25.6 近邻边界与允许主张

- [DiscRec](https://arxiv.org/abs/2506.15576) 已做 semantic/collaborative 双分支、
  localized attention 与 gating；MARC 不声称首次解耦或门控。
- [PRORec/UNGER](https://arxiv.org/abs/2502.06269) 已指出 semantic domination，
  并做 cross-modality alignment 与 distillation；MARC 不声称首次处理两类信号冲突。
- [MSCGRec](https://arxiv.org/abs/2602.03713) 已把 collaborative features 作为
  modality，并用 permissible-token training 与 modality Shapley 分析；MARC 不声称
  首次多模态 generative recommendation 或贡献分析。
- [HSUGA](https://aclanthology.org/2026.findings-acl.726/) 已根据用户活跃度调整
  semantic utilization；MARC 不声称首次按用户动态使用语义。
- R4ec 与 RRCM 分别覆盖 LLM reflection 和 RL evidence acquisition。

当前只允许把创新表述为：

> 在 constrained generative recommendation 中，以 prefix-level counterfactual
> marginal utility 作为统一控制量，让一个带 uncertainty/abstention 的轻量 critic
> 同时决定 semantic–collaborative trust、nested neighbor budget、decoder-layer
> residual injection 与是否进行有成本上限的一次 refinement。

若完整原文审计发现已有工作同时覆盖这个统一控制原则与四个下游动作，则停止；不能靠
换名维持。

### 25.7 L0：先验证“critic 有东西可学”

L0 不训练 GRAM 主干、不读 validation/test，只使用 `sequence[-3]` training target。

#### L0-A：counterfactual utility census

Toys/Beauty 各固定 512 users，使用同一 frozen checkpoint，计算：

- semantic-only、CF-only、full；
- CF `K ∈ {0,5,10,20}` nested masks；
- 每个 generation depth 的 `U_sem/U_cf/U_k`；
- 正负 utility 比例、跨 depth/branching 的异质性、oracle action headroom；
- current replay、mask restoration、target exclusion 与 finite checks。

必要条件：

1. 两数据集 `U_sem` 与 `U_cf` 都必须同时存在足量正/负样本（各至少 15%），否则
   动态 routing 没有意义；
2. oracle source/budget policy 相比 fixed full 至少降低 5% mean Trie-local CE；
3. 至少两个 generation depths 的最佳 source/action 分布显著不同；
4. `K=20` 不得在超过 90% 样本上单调支配较小 K，否则动态邻居数没有必要。

#### L0-B：target-free predictability

按用户切分 fit/calibration/audit，tiny critic 只读第 25.2 节 target-free features，
冻结 feature schema、模型与阈值后一次性审计：

- utility sign AUROC、Spearman 与 calibration；
- learned action 对 oracle action 的 regret；
- active/abstain coverage；
- 跨 Toys/Beauty 使用同一结构与选择规则，不用 dataset-specific feature/threshold。

两数据集都要求：

- `U_sem`、`U_cf` sign AUROC ≥ 0.65；
- budget action normalized regret ≤ fixed-full regret 的 0.75；
- predicted-positive bucket 的 actual mean utility > 0，bootstrap 95% CI lower > 0；
- abstention 后 active coverage ≥ 30%；
- corruption sanity 中对应 source 的 predicted utility 有正确下降方向。

任一关键 gate 失败即记 `STOP_MARC_UTILITY_NOT_LEARNABLE`，不得直接用全模型/RL
绕过。L0 允许 frozen-checkpoint forward，但不做 optimizer update；资源与输入 lineage
必须完整记录。

#### L0 实际结果（2026-07-24）

首次尝试因 K20 source reference 在 128-token 截断下机械移除 metadata 而判定
`EXECUTION_INVALID_SOURCE_REFERENCE`；它不进入科学结论。修复后只把 source
reference 对齐 matched baseline（Toys K5、Beauty K10），动态预算候选
`{0,5,10,20}`、cohort、target-free feature schema 和门槛均未改变。

| Dataset | Integrity | Sem + / - | CF + / - | Oracle CE reduction | K20 dominance | L0-A | L0-B |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | PASS | 74.22% / 25.78% | 85.94% / **14.06%** | 35.24% | 16.56% | **FAIL** | **FAIL** |
| Beauty | PASS | 51.76% / 48.24% | 84.57% / 15.43% | 14.52% | 13.62% | PASS | **FAIL** |

Toys 的 CF negative rate 比 15% 门槛少 0.9375 个百分点，仍必须按预注册规则停止。
L0-B 的额外失败为：

- semantic corruption predicted-utility drop：Toys -0.0353、Beauty -0.0026，
  都是错误方向；
- Beauty semantic active coverage 15.48% < 30%，active utility 95% CI
  `[-0.0462, 0.0632]` 跨 0；
- Beauty learned/fixed budget regret ratio 0.9406 > 0.75。

正面诊断是两数据集 collaborative critic 均通过 AUROC、active utility 与 corruption
门槛，但不能据此事后把 unified MARC 改成 CF-only 并晋级。固定决定：
**`STOP_MARC_NO_UTILITY_HETEROGENEITY`**。

### 25.8 L1–L3 渐进实验

以下阶段因 L0 固定失败而全部锁定，未执行：

- **L1 correctness smoke**：实现 source masks、two probes、critic、identity gates；
  32–64 training examples，验证 zero-gate 精确复现、梯度有限、单来源退化、nested K、
  Trie decoding、checkpoint reload；不看 validation 指标。
- **L2 10% pilot**：先只训练 MARC-lite 的 probes/critic/gates，主干冻结；比较
  GRAM、dual-branch fixed gate、source-trust only、`+dynamic K`、`+layer routing`。
  reflection 与 RL 仍关闭，防止一次加入过多因素。
- **L3 fresh-dataset confirmation**：在 Sports 冻结配置后比较
  `MARC-lite`、`+one-reflection`、`+bandit`；只有前一项通过才增加后一项。
  同报 Recall/NDCG、broad harm、calibration/regret、平均 K、各层 gate、二次执行率、
  latency、显存和参数量。

### 25.9 最强反对意见与停止规则

最强反对意见是：这可能是 DiscRec gate + RRCM policy + R4ec reflection 的复杂拼装；
counterfactual utility 使用 gold target，推理时 critic 可能无法恢复；逐层 gate 和
二次 refinement 也可能只是用更多参数/计算换效果。

以下任一发生即停止或删减：

- utility 没有足够异质性或 target-free 不可预测；
- source-trust-only 已解释全部收益，则删除 K/layer/reflection；
- fixed dual-branch gate 已解释全部收益，则不主张 MARC；
- critic collapse 到固定来源/固定 K/固定层；
- synthetic corruption calibration 不迁移到自然噪声；
- fresh dataset 无增益、出现 broad harm，或收益可由相同参数/计算的 dense control
  解释；
- reflection 触发率接近 100%，或平均 latency 超过 baseline 1.5×；
- bandit 相对监督式 routing 无独立增益。

当前固定状态为 **`STOP_MARC_NO_UTILITY_HETEROGENEITY`**。L0 已完成；GRAM 未训练，
validation/test 未读取，L1/L2/L3、reflection 与 bandit 均未执行。
