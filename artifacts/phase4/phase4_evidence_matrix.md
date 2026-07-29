# GRAM Phase 4 Evidence Matrix

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate + plan
- Created: 2026-07-28
- Verification Status: ANALYZED（本地 summary/config/code）
- Scope: Toys、Beauty development evidence；Sports 未读取
- Source Plan: `plan/GRAM_第四阶段_方法创新与渐进实验计划.md`

## 1. 逐方向证据

| 方向 / 阶段 | 完整性 | 双域主要证据 | 固定决定 | 排除的解释或机制 | 保留的正证据 |
|---|---|---|---|---|---|
| CF-SAT C0 | 通过；0 update、等预算 corruption、split/Trie/SHA 全通过 | clean–corrupt user margin CI 两域均 >0；helpful-node rate 仅 Toys 48.15%、Beauty 31.91%，Beauty deficit rate 16.02% | `STOP_CFSAT_NO_CLEAN_CORRUPT_SIGNAL` | “多数 lexical nodes 都能从真实 CF 获益”不成立 | 模型整体能区分真实/错误 CF，但效用高度 node-dependent |
| RPCD T0 | 通过；共享 epoch=8、weight=0.2、test 未读 | union R@50 +3.17/+3.03pp；miss recovery 5.70%/6.30%；NDCG +0.34%/+0.20%；tail −2.48%/−3.38% | `STOP_RPCD_NO_TEACHER_COMPLEMENTARITY` | raw SASRec distribution/global fusion 不能安全兑现 top-10 | user-conditioned catalog proposal 在两域稳定补充约 3pp 候选 |
| PRPD R0 | 通过；55 个共享配置、gamma=0 identity | 无非零 residual 配置同时满足 calibration broad/tail；锁定 weight=0，audit 增益=0 | `STOP_PRPD_NO_DEBIASED_EFFECT` | post-top50 popularity arithmetic 不是充分修复 | tail harm 与 popularity 有关，但不是简单标量校正问题 |
| CPGV V0 | 通过；51,850 candidate paths、0 update | exact rescore R@10：Toys 25.96%→22.11%，Beauty 20.77%→9.93% | `STOP_CPGV_GRAM_CANNOT_VERIFY_PROPOSALS` | frozen GRAM exact likelihood 不能作为可靠 proposal verifier | GRAM 对 proposal 的 pairwise signal在 Toys 略高于随机，但 Beauty 近随机 |
| FCRD F0 | 通过；full catalog、gamma=0 identity、0 update | gamma=0 保 overall +3pp 但 tail 仅 +0.14/+0.05pp；gamma=1 tail +1.83/+1.63pp 但 overall 仅 +2.22/+1.88pp | `STOP_FCRD_NO_FULL_CATALOG_RESIDUAL_EFFECT` | top50 截断不是 PRPD 失败的唯一原因；统一 gamma 存在结构性 trade-off | full-catalog residual 确实能移动 tail coverage |
| CCRR R0 | 通过；candidate identity、fit/audit 隔离、test 未读 | calibration overall NDCG +6.31%/+11.15%；Toys tail +9.19%，Beauty tail −3.80%；audit 未读 | `STOP_CCRR_NO_CANDIDATE_CONDITIONAL_EFFECT` | 轻量 candidate-level conditional model 仍不能跨域保 tail | union 内存在很强的可学习排序信号 |
| GCDH P0 | 通过；matched C0/C1、finite/gradient/resource | catalog-primary NDCG −82.95%/−86.41%；union +2.54/+1.68pp | `STOP_GCDH_NO_DUAL_HEAD_EFFECT` | coarse pooling + flat catalog head + catalog-primary ranking 不可用 | catalog auxiliary CE 可学习，且仍补充少量候选 |
| GCDH D0 | 通过；0 update、SHA identity、fresh diagnostics | user state non-collapse；C1 MRR/R@50 两域均高于 C0；generator-only coverage 16.99%/18.38% 远大于 catalog-only 2.54%/1.68% | `GCDH_D0_READOUT_RANKING_MISMATCH` | P0 失败不是简单 user-state collapse 或 popularity collapse | 主要问题位于 readout/ranking 对齐 |
| GACR S0 | 通过；两域全部 correctness gates | zero-residual identity=100%，loss 下降，bounded residual、reload、tail pairs 全通过 | `GACR_S0_CORRECTNESS_PASS` | bounded generator-anchored residual 在工程上可实现 | 可以在不破坏 base identity 的前提下优化候选排序 |
| GACR P0 | 通过；fit/calibration user-disjoint，新 validation 与旧 cohort 零重合 | Toys overall/tail +0.79%/+0.70%；Beauty +2.30%/+1.62%；两域 Recall 不降、harm 0.098% | `STOP_GACR_NO_RESIDUAL_RANK_EFFECT` | 冻结 generator 上的小 residual 仍缺少跨域稳定性 | Beauty 给出真实 positive effect；安全锚定有效避免灾难性下降 |
| IALC N1 | 通过；两域各 512 个 unique training-prefix users，0 update，validation/test/Sports 未读 | mean illegal mass 2.26%/4.53%；mean full-vocab–legal loss gap 0.0238/0.0513；large-mass sample rate 17.19%/48.44% | `STOP_IALC_NO_SUPPORT_MISMATCH` | full-vocabulary CE 与 constrained inference 的支持集差异太小，不足以支撑 legal-child-only 训练 | tail loss gap 相对更大，但两域整体 mismatch 均低于冻结门 |
| LNDR N1 | 通过；两域各 1,024 个 unique training-prefix users，0 update，checkpoint SHA 不变，validation/test/Sports 未读 | same-token/same-depth eligible node centroid distance 中位数仅 0.00864/0.00978；冻结高多义阈值 0.10 下两域均为 0 个 audit steps | `STOP_LNDR_NO_NODE_POLYSEMY_DEFICIT` | lexical token 的 catalog 复用不等于 node semantic polysemy；prefix-specific residual readout 缺少前提 | catalog 中 token reuse 极广，但同词同深度子树 metadata centroid 高度相近 |
| SCDL N1 | 通过；catalog text/lexical IDs only，checkpoint 与 interaction targets 未读，validation/test/Sports 未读；两次 invalid attempt 已审计后精确重跑 | current nonpositive sibling margin 10.17%/8.67%；joint assignment 改善 set rate 70.89%/64.10%、mean margin +0.0222/+0.0185，但 positive-margin coverage 仅 +6.09/+4.05pp | `STOP_SCDL_NO_SIBLING_LEXICALIZATION_DEFICIT` | GRAM independent lexicalization 已使约 90% child 具有正 sibling margin；联合 native-token assignment 的可兑现覆盖增量不足 | joint assignment 100% feasible，且不牺牲平均 representativeness，说明离线 lexicalization objective 可优化但当前不是主要瓶颈 |
| FPUG N1 | 通过；两域各 512 unique training-prefix users，0 update；coarse prompt 恒定、每次仅 mask 一个 detailed passage，validation/test/Sports 未读 | harmful-passage sample rate 69.14%/67.77%；best removal legal CE +0.104/+0.122 nats；fixed-oldest 为 −0.0418/−0.0198；oracle advantage +0.146/+0.142，四个 recency quartiles 全覆盖 | `FPUG_S0_DESIGN_ALLOWED` | harmful detail signal 不能由固定删除最旧 history 解释 | GRAM FiD detailed passages 存在稳定、user-dependent 的条件效用差异，双域 premise 首次完整通过 |
| FPUG S0 | 通过；两域各 8 training-prefix users，backbone frozen，gate-only 20 steps，validation/test/Sports 未读 | zero-init logits/coarse identity max diff=0；CE −18.60%/−14.28%；finite nonzero gradients、bounded gates、reload diff=0 | `FPUG_S0_CORRECTNESS_PASS` | 排除 gate 无梯度、identity 破坏、coarse passage 被误门控与 checkpoint mutation | bounded detail-passage gate 工程可实现且在冻结 decoder 下可优化 |
| FPUG P0 | 通过；共享 epoch=2，fit/calibration user-disjoint，训练前缀烟测与 validation mapping/finite=100%，checkpoint SHA 不变，test/Sports 未读 | Toys overall NDCG +2.37%（CI −1.33%, +6.32%）、tail +5.69%；Beauty overall −5.87%（CI −11.36%, −1.25%）、Recall −0.78pp、harm 0.78% | `STOP_FPUG_EFFECT_GATE_FAILED` | training-prefix passage-removal utility 与 CE 可优化性不能稳定转化为双域 top-10 recommendation effect；固定 gate 在 Beauty 产生显著整体伤害 | Toys tail 给出正向且 CI>0 的局部证据，但不足以通过预注册双域合取门 |
| TCDR N1 | 通过；两域各 128 unique training-prefix users、64 close/frequency-matched-far pairs，0 update，mapping/Trie/finite/bin-match=100%，validation/test/Sports 未读 | close vs far score-correlation median：Toys 0.537 vs 0.384，Beauty 0.935 vs 0.385；paired excess median +0.219/+0.495；mean excess CI lower +0.045/+0.417 | `TCDR_S0_DESIGN_ALLOWED` | 低 collaborative cosine 与 endpoint popularity 不能解释 tree-near items 的跨用户 exact-score coupling | GRAM 单 lexical Trie 在双域存在可复现的 tree-structure response coupling，Beauty 尤强 |
| TCDR S0 | 通过；每域 8 training-prefix users、8 frozen pairs，仅 decoder 最后一层更新 5 步；参数量/checkpoint/inference structure 不变，validation/test/Sports 未读 | TCDR loss −81.53%/−51.74%；lexical CE −0.30%/+0.25%；zero-lambda identity=0，finite nonzero gradients 与参数变化全部通过 | `TCDR_S0_CORRECTNESS_PASS` | 排除 legal-child score 不可微、跨用户 correlation 无梯度、短程优化无效和新增 inference 参数等工程失败 | 原 generator decoder 能直接降低 tree-coupling excess，且微型 smoke 未造成明显 lexical CE 伤害 |
| TCDR P0 | 通过；C0/C1 每域 256 fit、128 calibration unique users，fit/calibration disjoint，32 matched steps，checkpoint/mapping/Trie/finite 全通过；机制门前停止，validation/test/Sports 未读 | calibration excess C0→C1：Toys 0.08653→0.08359（−3.40%），Beauty 0.36127→0.35701（−1.18%）；CE −0.0046%/+0.0021% | `STOP_TCDR_MECHANISM_GATE_FAILED` | S0 微型 cohort 的可优化性不能扩展为独立 calibration 上至少 10% 的结构解耦；固定 `λ=0.1`/两 epoch 正式方案机制效应不足 | 正则在不伤 CE 的情况下使 excess 方向一致地小幅下降，但不足以授权 validation effect read |
| CPIA N1 | 通过；两域各 128 deterministic unique training users、640 coarse/fine spans，mapping/attention/finite/checkpoint identity 全通过，0 update，validation/test/Sports 未读；完整性计数错误已审计后精确重跑 | top-1 matched-passage retrieval 98.91%/79.53%；median hard margin 0.2576/0.0791；mismatch rate 1.09%/20.47%；matched-minus-mismatch CI lower 0.3229/0.1342 | `STOP_CPIA_NO_ACTIONABLE_LINK_DEFICIT` | repeated native lexical ID 在 coarse/fine passages 间并不缺乏 contextual linking；两域均远强于冻结的 weak-link deficit 范围 | GRAM 的 information-linking bridge 在 frozen encoder states 中高度可辨识，Toys 尤强 |

## 2. 跨实验稳定事实

### Evidence

1. **候选互补真实存在。** SASRec/catalog 分支在多个独立诊断中为 GRAM beam-50
   增加约 1.7–3.2pp coverage；这不是单次偶然。
2. **最终排序对齐是主要瓶颈。** raw fusion、popularity residual、frozen exact
   verifier、catalog-primary 与 bounded residual 都没有稳定跨域兑现候选增益。
3. **问题不是简单的用户表示塌缩。** GCDH-D0 的 pooled-state 方差、cosine distance
   与 effective rank 均明显通过 non-collapse 门。
4. **可学习信号存在但跨域/tail 不稳定。** CCRR 在 overall 上很强，GACR 在 Beauty
   上过门，但 Beauty tail 或 Toys 整体反复成为 conjunctive failure。
5. **继续在已冻结候选上增加融合容量缺少机制依据。** GBDT/MLP、更多 feature、
   dataset-specific weight 或放宽门槛都只是已失败家族的容量扩张。

### Inference

现有 GRAM 的 lexical decoder 只接受单一 gold path CE，并未直接学习“在一个
target-free collaborative/catalog hard-negative 与 gold item 竞争时，哪个 lexical
child 必须在最早分叉处胜出”。CPGV 证明 frozen exact score 做不到这件事；GACR 则
说明只在冻结输出后修补仍不够稳定。尚未被直接检验的是：**把该竞争约束训练进原
generator logits，同时保持 inference 完全不变。**

## 3. 新候选假设：CHPR

**Collaborative Hard-negative Prefix Ranking**
（协同硬负例前缀排序）

对 training history \(h\)，用 target-free SASRec/catalog/current-beam proposal
产生 hard-negative item \(j\)。令 gold lexical ID 为 \(y\)，negative ID 为 \(z_j\)，
二者最早分叉前缀为 \(p_j\)，对应合法 children 为 \(y^+_j,z^-_j\)：

```text
L = L_lexical_CE
  + lambda * mean_j max(0,
      margin
      - log p(y+_j | p_j, h)
      + log p(z-_j | p_j, h))
```

只在训练期使用 proposer；正式推理仍是原 GRAM input、decoder、Trie 与 beam。
Proposal 只充当 negative，不作为伪 positive，因此不继承 RPCD 的 teacher-quality
假设。Loss 直接作用于原 generator 的合法-child logits，而不是新增 catalog readout
或冻结候选后的 reranker。

### 与既有失败的边界

- 不同于 CF-SAT：不修改输入 CF，不学习 clean/corrupt sensitivity；
- 不同于 RPCD/PRPD/FCRD：不蒸馏 teacher distribution，不做全局融合；
- 不同于 CPGV：不是用 frozen GRAM 验证 proposal，而是把验证缺口作为训练信号；
- 不同于 CCRR/GACR：不在冻结 candidate scores 后拟合 reranker；
- 不同于 GCDH：不增加 catalog-primary inference head。

### 新颖性边界

Hard-negative mining、pairwise margin、prefix supervision 和 Trie-constrained
generation 各自都不是新贡献。候选差异只能表述为：以 training-only collaborative
proposal 定位 gold/negative lexical IDs 的最早 Trie 分叉，并在原生成器合法-child
logits 上做 inference-free prefix ranking。该差异尚未完成全文级 novelty review，
不能声称首次提出。

## 4. 建议第一步：CHPR-A0 premise audit

A0 只使用 training-prefix、frozen GRAM 与既有 proposer，不训练、不读
validation/test，回答：

1. gold child 相对 hard-negative child 的 margin deficit 是否在两域普遍存在；
2. deficit 是否覆盖 tail，而非只由 head items 或 depth-0 分叉驱动；
3. deficit 是否与 gold beam rank/beam miss 有方向一致关系。

建议预注册门：

- 每域固定 512 training-prefix samples，head/tail 各 256；
- proposal 为 target-free SASRec/catalog top-50 与 current GRAM beam-50 的并集，
  排除 history、gold 与重复 item；
- 每个 sample 取 proposal 中 generator exact path score 最高的 8 个 negatives；
- 保存最早分叉 depth、gold/negative child log-prob、margin、beam rank 与 source；
- mapping、Trie membership、finite、target/history exclusion、0 update、parameter SHA
  与 test exclusion 必须为 100%；
- 两域均要求：有效 proposal coverage ≥90%，margin<0.10 deficit sample rate ≥30%，
  tail deficit rate ≥25%，deficit user/sample coverage ≥30%，且 depth>0 至少两个
  depths 各有 ≥50 个 deficits；
- 任一完整性失败为 `EXECUTION_INVALID`；科学门失败为
  `STOP_CHPR_NO_PREFIX_RANKING_DEFICIT`；全部通过才是
  `CHPR_S0_DESIGN_ALLOWED`。

A0 是 premise audit，不因已有 CPGV/GACR 结果预设必然通过，也不用于报告最终
Recall/NDCG。

## 5. CHPR-A0 正式结果

A0 已按冻结配置完成，完整性全部通过：Toys/Beauty 各 512 个 unique training
users，proposal、mapping、Trie、finite、history/gold exclusion、0 update、checkpoint
SHA 与 validation/test exclusion 均为 100%。

| 数据集 | deficit sample rate | tail deficit rate | mean minimum margin | beam hit / miss deficit | supported depth>0 |
|---|---:|---:|---:|---:|---|
| Toys | 86.33% | 83.59% | -1.600 | 77.10% / 99.07% | depth 1: 401；depth 2: 172 |
| Beauty | 90.23% | 90.63% | -2.082 | 82.01% / 100% | depth 1: 222 |

Beauty depth 2 只有 34 个 deficit pairs，低于冻结的每 depth 50；故双域 conjunctive
gate 失败，固定决定为 **`STOP_CHPR_NO_PREFIX_RANKING_DEFICIT`**。这里更准确的解释
是“缺少跨域 depth-diverse deficit”，不是完全没有 deficit。

只读 source/depth 归因进一步显示：

- deficit pairs 集中在 depth 0：Toys 1,987/2,580（77.02%），Beauty
  2,363/2,648（89.24%）；
- 被选为 top-8 exact hard negatives 的来源几乎全是当前 GRAM beam：
  Toys 2,420/2,580、Beauty 2,511/2,648；其余为 beam/catalog overlap，
  **catalog-only 为 0**；
- 因而冻结的“选 exact score 最高 negatives”规则实际上没有保留 collaborative-only
  proposals。A0 的强 deficit 主要是 generator self-competition，不足以支持
  collaborative hard-negative 方法表述。

不得事后强制 catalog-only quota、把 Beauty depth-2 门槛从 50 降到 34，或把方法
改名后沿用本次结果。若研究 self-beam hard-negative ranking，需要先做新颖性复核；
该机制本身接近常规 hard-negative/pairwise generation training，当前没有足够证据
作为第四阶段的新方法主线。
