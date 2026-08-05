# GRAM 第十阶段：CF1-C0 特征与基线审计结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_C0_TOYS_FEATURE_AUDIT_V1`
- Verification Status: `PASSED`
- Evidence Class: validation development audit, no fusion fitting
- Test/Beauty/Sports Read: false

## 1. Executive conclusion

CF1-C0 在 40.21 秒内通过全部 11 项冻结审计。1,698,905 条候选均与 B2 逐行匹配，SHA256
身份一致，无重复 `(user,candidate)`，全部特征 finite；19,412 用户被稳定分入
`[3883,3883,3882,3882,3882]` 五折，target/gold/label 未进入 inference feature schema。

朴素 source-agnostic sum 并不能利用候选互补性：相对 frozen PCRF，Hit@10 从 `0.125335`
下降到 `0.116423`，Hit@50 仅从 `0.211931` 升到 `0.221255`。与此同时预算化 union oracle
为 `0.264733`，仍有 `+0.052802` Hit@50 候选上界。证据支持“候选有价值但分数尺度/来源校准
不成熟”，而不是“CF-only 候选或 arbitrary GRAM score 无效”。

因此授权执行 CF1-C1 五折 cross-fitted monotone listwise calibrator；不得把朴素 sum 当作正式融合。

## 2. Frozen audit results

| check | observed | status |
|---|---:|---|
| users exact | 19,412 | PASS |
| candidates exact | 1,698,905 | PASS |
| CF-only exact | 728,305 | PASS |
| union size 50--90 | 100% | PASS |
| duplicate user-candidate | 0 | PASS |
| finite features | 100% | PASS |
| B2 row identity | 100% | PASS |
| B2 SHA256 identity | true | PASS |
| cached G50 footer identity | true | PASS |
| five-fold integrity | true | PASS |
| target absent from inference schema | true | PASS |

## 3. Source composition

| source | candidates | fraction |
|---|---:|---:|
| GRAM only | 791,057 | 46.56% |
| both | 179,543 | 10.57% |
| CF only | 728,305 | 42.87% |
| total | 1,698,905 | 100% |

CF-only 占比很高，因此 source bias、rank agreement 与 reliability calibration 不是边缘修正，而是
决定 union top-k 行为的核心部分。

## 4. Frozen baseline profile

| method | Hit@1 | Hit@10 | NDCG@10 | Hit@50 |
|---|---:|---:|---:|---:|
| GRAM G50 | 0.041675 | 0.119411 | 0.076275 | 0.211931 |
| frozen PCRF `(1,.5,1)` | 0.041366 | 0.125335 | 0.078716 | 0.211931 |
| pure CF50 | 0.028591 | 0.090150 | 0.055367 | 0.174531 |
| source-agnostic sum union | 0.037606 | 0.116423 | 0.072366 | 0.221255 |
| budgeted union oracle | 0.264733 | 0.264733 | 0.264733 | 0.264733 |

Oracle 把命中目标理想化置于 rank 1，只用于 coverage upper bound，不能与实际排序 NDCG 作模型性能
比较。实际可解释的信号是：朴素 sum 只兑现约 `0.009324` Hit@50，远低于 union 对 G50 的
`0.052802` coverage 增量，而且牺牲了 Hit@1/10。

## 5. Interpretation and decision

1. 数据联接、score identity、feature completeness 和 fold isolation 均不是当前瓶颈；
2. PCRF 仍是必须击败的安全基线，不能以 GRAM 或 naive sum 替代；
3. candidate source 的 score scale 明显不兼容，必须学习 source-aware calibration；
4. C1 使用固定的单调线性 listwise 规格，五折训练折拟合、留出折产出 OOF rank；
5. 若 C1 只提升 Hit@50、不提升 Hit@10，则进入另行预注册的 C2 非线性/gating 设计；
6. Toys test、Beauty、Sports 继续关闭。

## 6. Reproducibility pointers

- plan：`plan/第十阶段/GRAM_第十阶段_CF1-C跨折校准融合实验计划.md`
- preregistration：`artifacts/phase10/configs/cf1_c0_toys_feature_audit_preregistered.json`
- summary：`artifacts/phase10/cf1_c0_toys_feature_audit/summary.json`
- feature table：`artifacts/phase10/cf1_c0_toys_feature_audit/feature_table.npz`
- folds：`artifacts/phase10/cf1_c0_toys_feature_audit/fold_assignments.tsv`
- evaluator：`experiment/phase10/eval_cf1_c0_feature_audit.py`
- runner：`experiment/phase10/run_phase10_cf1_c0_feature_audit.sh`

