# GRAM 第八阶段：TIPA-P0 商品到词法路径对齐审计实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-03
- Verification Status: `ANALYZED_STOPPED`
- Version Label: `phase8_tipa_p0_item_to_lexical_path_alignment_audit_v1`
- Parent results: `STOP_CANDIDATE_DRAFTING` and `STOP_BEFORE_SEALED_TEST`
- Governance: `plan/GRAM_后续结构性方向分阶段实验治理规则.md`
- Development domains: Toys、Beauty
- Sports/Test/Fresh validation: 封存

## 1. 证据链与唯一结构假设

现有证据不支持继续增加 candidate source：F0-T 中 SASRec 的独占 target users 只有
Toys 6 人、Beauty 7 人，且 Beauty tail 为 0。现有证据也不支持把 item score 用全局
scalar 直接加到 token logits：ST-GCGD-v2.1 的 transition/session 模型能改善 train-only catalog
ranking，但在 P1 中的固定 prefix fusion 两域均伤害，且没有新 top-10 hit。

因此 TIPA（**Token–Item Path Alignment**）P0 只检验一个新假设：

> item-level teacher 的相对排序信号不能被无条件 scalar prefix mass 稳定传递，但可能通过
> 共享参数、prefix-conditioned 且 path-consistent 的轻量适配器，在不破坏 GRAM identity
> 回退的前提下转化为 target-free token 偏移。

P0 不重训 GRAM backbone，不引入新 candidate source，不训 verifier，不使用 F0-T 外部
calibration 用户。

## 2. 阶段定位

- 阶段：`P0-A`，单 seed、train-only pseudo-future 对齐 pilot。
- 目的：回答“对齐接口是否存在可训练且可安全回退的机制”，不回答正式方法是否有效。
- seed：`2023`。
- 数据：每用户继续封存 `[-2]` development target 和 `[-1]` test target；只从
  `items[:-2]` 内构造 prefix→pseudo-future 记录。
- 不创建 fresh cohort，不读取 Sports/test，不在第六/七阶段已用 development cohort 上选择。

## 3. 固定对照与教师信号

每域只使用以下四个 nested arms：

- A：冻结 GRAM，不注入任何 item-level 信号。
- B：ST-GCGD-v2.1 已否决的固定 scalar prefix fusion，只作负对照，不调 alpha。
- C：TIPA path-consistent adapter，训练目标只对齐 teacher item distribution 与 prefix subtree
  distribution。
- C0：C 的 zero-adapter identity，必须与 A 逐 token 完全相同。

teacher 固定为 ST-GCGD-v2.1 已锁定的 transition/session catalog scorer，不重训、不使用
validation label。如果现有 teacher checkpoint 或输入 lineage 无法完整复用，P0-A 必须标记
`BLOCKED_MISSING_TEACHER_LINEAGE`，不得默认重训。

## 4. TIPA 接口的最小定义

对当前 legal prefix `p` 和每个合法 next token `t`，先从冻结 teacher item logits 计算 Trie
subtree log-mass `M_u(t|p)`。与旧 scalar 融合不同，适配器输出：

`delta_u(t,p) = 0.3 * tanh(center(f_theta(z_GRAM(t|u,p), M_u(t|p), depth, entropy, margin, leaf_fraction)))`

其中：

- `f_theta` 固定为 `Linear(6,64)-LayerNorm-GELU-Linear(64,64)-GELU-Linear(64,1)`，
  对同一 prefix 的所有 legal children 共享，不对单 item 设独立参数；
- 六个输入为 legal-child GRAM logit z-score、teacher child log-probability、normalized depth、
  normalized teacher entropy、teacher top margin 和 compatible leaf fraction；
- 最后一层零初始化，同 prefix 内输出先去均值，再用 `0.3*tanh`有界化。无
  teacher coverage、非 finite 或单 leaf 时精确返回 0；
- 所有非法 Trie token 始终不可见；所有 leaf item 的 path score 仅由共享 token 参数产生；
- 禁止输入 pseudo-future item identity、target rank、hit/miss 标志或 development 指标。

第一版固定 hidden=`64`、bound=`0.3`、AdamW lr=`0.001`、weight decay=`0.0001`、
100 steps、每 step 32 prefixes、identity penalty=`0.05`、gradient clip=`10`。每域 fit/calibration
各抽取 128 head + 128 tail records，fit 至少保留 192 个非 null prefix records。不允许从
P0 结果中搜索多个变体。

## 5. 训练目标与数据隔离

- fit/calibration 用户按新 salt `phase8-tipa-p0a-v1` 固定分离，用户不重叠。
- fit loss 只包含：teacher→prefix subtree KL、path-mass consistency 和 identity regularization。
- pseudo-future label 只用于在隔离 calibration 上评估 teacher advantage 是否被兑现，不进入 adapter
  feature；若需要 supervised rank loss，必须视为另一方向并重写计划。
- GRAM backbone 和 teacher optimizer steps 均必须为 0；只有 adapter 可更新。
- 不使用 epoch 网格；固定训练步数，仅允许 finite/NaN 的 fail-closed 退出。

## 6. 强制机制指标

P0-A 必须逐域、逐用户输出：

1. teacher item pair 与 TIPA path pair 的 Kendall agreement，以及与 B 的配对差；
2. teacher top-50 target coverage 可兑现为 beam@50 的比例，及 `new_hit@10 outside A beam`；
3. A/B/C 的 Recall@10、NDCG@10、Recall@50、changed-user rate 和 broad harm；
4. head/tail、transition-covered/uncovered、短/长 history、高/低 teacher margin 分层；
5. prefix depth 下的 subtree-mass 守恒误差、Trie legality、identity max-abs diff 和 null-path rate；
6. adapter 参数、显存、latency、每阶段耗时、checkpoint/candidate/cohort SHA256。

所有集合 coverage 与固定预算 Recall 分开命名。任何 `union@K` 都必须在计划/config
中显式定义总预算、source quota 与 target-free ordering，禁止再使用“追加后直接截断”的
模糊口径。

## 7. P0-A 预注册资格门

只有 Toys 和 Beauty 同时满足以下条件，才允许设计 P1：

- 完整性：identity max-abs diff `<=1e-7`，Trie legality 100%，subtree-mass 最大误差
  `<=1e-6`，所有值 finite，fit/calibration overlap=0；
- 对齐：C 的 teacher→path Kendall agreement 均值相对 B 每域至少 `+0.10`；
- 兑现：每域至少 5 个 teacher 独占 target users 被带入 C beam@50，且不少于 B；
- 排序：C 相对 A 的 Recall@10 与 NDCG@10 不得同时下降，且至少一项每域为正；
- 安全：broad harm `<=1%`，tail Recall@50 绝对差 `>=-0.5pp`，null-path 精确回退 A。

任一域失败即决定 `STOP_TIPA_NO_PATH_REALIZATION`，不得在同 cohort 上搜索 bound、
adapter 层数、teacher 或 loss 权重。P0 通过只解锁一份新 P1 计划，不解锁 test/Sports
或 full-backbone 训练。

## 8. 实现前必须冻结的内容

1. teacher checkpoint、GRAM checkpoint、Trie/item mapping、train-only records 和 cohort SHA256；
2. adapter 精确结构、参数上限、bound、训练 steps、batch size 与单一 seed；
3. A/B/C/C0 的统一解码程序、固定 beam size=50 和 stable tie-break；
4. 单元测试：数据截断、target-free feature、mass conservation、identity、null path、
   illegal token、item/path mapping、同 cohort arm alignment、Sports/test guard；
5. 输出 schema：summary、per-user、per-prefix、strata、integrity、timing、telemetry、status、
   manifest。runner 必须在标记 succeeded 前验证这些文件均存在且可复算。

GPU 位置、显存租约和 CodeLlama 恢复位置在实现授权后依当时实际设备空闲状态
另行冻结；不沿用历史 GPU0/GPU6 假设。

## 9. 强制决策记录

```text
阶段：P0-A
唯一结构假设：prefix-conditioned path adapter 可将 item-level teacher 排序信号转化为安全的 lexical-path 偏移。
固定 seed/cohort：2023；train-only items[:-2] pseudo-future；新 salt。
直接机制指标：teacher→path Kendall agreement、teacher 独占 target 的 beam@50 兑现、mass conservation。
最低有效信号：两域 Kendall +0.10；每域至少 5 个独占 target 兑现；top-10 不同降且至少一项为正。
guardrail 非劣界：broad harm <=1%；tail Recall@50 >=-0.5pp；identity/null-path exact。
通过后唯一下一步：另写并冻结 TIPA-P1 三 seeds fresh-development 计划。
失败后停止项：不训 full backbone；不增加 teacher/source；不训 verifier。
禁止的邻近补丁：bound/alpha/层数/loss 网格、target-conditioned gate、同 cohort 重试。
候选/特征缓存 SHA256：实现授权后、启动前写入。
Sports/test read：false / false
```

## 10. 当前状态

研究者已于 2026-08-03 授权继续实验。teacher lineage 已审计：Toys/Beauty 的
`transition_graph.pt` 可在 CPU 上 strict load，图规模分别为 19,412/11,924 与
22,363/12,101 users/items。workload、测试、runner 和 config 已创建，CPU 测试
5/5 通过。首次启动后的状态与 recovery 规则见第 11 节。

## 11. P0-A 首次启动失败与具名 recovery

P0-A 于 2026-08-03 启动后，Toys 完成 256/256 个 fit sample 的 teacher path 构造，但
“在 teacher top-item path 的所有深度中随机选一个”会频繁选到已变为单 child 的深层
prefix，可用多 child records 少于冻结下限 192。程序因此在 adapter 训练前以
`insufficient non-null prefix records` 退出；Beauty、external development、Sports/test 均未读取。

研究者随后明确授权继续。唯一 recovery 改动为：仍先 target-free 选定 teacher top item，
但只在该 item path 上 `legal_children > 1` 的 depths 中，用冻结 SHA256 次序确定性选择
一个 prefix。这不读取 pseudo-future target，不改 teacher、adapter、loss、steps、cohort、seed、
gate 或解码协议。可用 records 下限改为 240/256，以要求至少 93.75% 的样本
存在可识别 child 对齐问题。

recovery 还必须在任何可用性门退出前写入 `prefix_availability_audit.json` 和
`per_prefix.csv`，明确报告 attempted/usable/null-or-single-child 计数。原输出目录、日志与 SHA
保留；recovery 使用独立 config、runner、tmux 和输出目录，不得覆盖首次失败。

## 12. P0-A recovery 实际结果与阶段冻结

具名 recovery 于 2026-08-03 在物理 GPU6 完成。Toys、Beauty 的 branching-prefix
可用记录均为 256/256，说明首次失败对应的样本构造问题已经修复；CPU 测试 6/6 通过，
GRAM/teacher optimizer steps 均为 0，fit/calibration overlap=0，Sports/test/external
development 均未读取。

科学门控在两域均失败：

- Toys：Kendall(C-B)=`-0.008157`，teacher-exclusive users=6，B/C 兑现数=0/0，
  Recall@10=`-0.78125pp`，NDCG@10=`-0.007506`，broad harm=`3.125%`；
- Beauty：Kendall(C-B)=`-0.018769`，teacher-exclusive users=14，B/C 兑现数=2/1，
  Recall@10=`+1.5625pp`，NDCG@10=`+0.007203`，broad harm=`0%`。

因此固定决定为 **`STOP_TIPA_NO_PATH_REALIZATION`**。`TIPA-P1`、fresh development、
Sports/test、full-backbone training 均不解锁；不得在同 cohort 上搜索 bound、层数、teacher、
loss weight 或 seed。后续只允许另立 analysis-only 的失败归因审计，且该审计不能推翻本
停止决定或成为 TIPA-P1 的晋级门。

规范结果：`artifacts/phase8/tipa_p0_branching_recovery/summary.json`；正式报告：
`report/第八阶段/GRAM_第八阶段_TIPA-P0A分支前缀恢复结果与验证报告.md`。
