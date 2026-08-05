# GRAM 第十阶段：CF1-A2 预算化候选融合结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_A2_TOYS_BUDGETED_UNION_V1`
- Verification Status: `PASSED`
- Dataset: Toys validation only
- Test/Beauty/Sports Read: false

## 1. Executive conclusion

CF1-A2 通过全部冻结 gate。主策略 `fill_cf_only_40` 将每个用户的候选数严格限制在 90 以内，
同时把 G50 coverage 从 `0.211931` 提升到 `0.264733`，保留原始 U50 候选增益的
`96.43%`。tail complementary coverage 为 `0.022093`，保留原始 U50 tail 互补增益的
`94.21%`。

这解决了 CF1-A 的核心矛盾：无需放弃 item-head 的 beam 外补充价值，也无需接受接近 100 个候选
的无界 50+50 并集。CF1-B 可以按最多 90 candidates/user 的冻结预算恢复 constrained GRAM
score identity 和 resource pilot。

## 2. Frozen gate results

| check | requirement | observed | status |
|---|---:|---:|---|
| G50 identity | exact | 0.2119307645 | PASS |
| U50 identity | exact | 0.2666907068 | PASS |
| users with union size <=90 | 100% | 100% | PASS |
| U50 coverage-gain retention | >=80% | 96.43% | PASS |
| U50 tail-complement retention | >=80% | 94.21% | PASS |

正式运行 CPU wall time `8.93 s`，5 个单元测试通过；无自动 retry、无 checkpoint mutation、
无 GPU 占用。

## 3. Primary result

| candidate set | coverage | gain vs G50 | mean size | max size | CF-only scoring total |
|---|---:|---:|---:|---:|---:|
| G50 | 0.211931 | — | 50 | 50 | 0 |
| raw U50 | 0.266691 | +0.054760 | 90.75 | 100 | 791,057 |
| fill_cf_only_40 | 0.264733 | +0.052802 | 87.52 | 90 | 728,305 |

预算化仅损失 `0.001958` absolute oracle coverage，却将所有用户置于硬上限内，并减少 62,752 个
待补充 GRAM scoring 的 CF-only candidate。这里的 coverage 仍是 candidate oracle，不是最终
Hit@10。

## 4. Slot diagnostics

| policy | coverage | tail complement | mean size | scoring total |
|---|---:|---:|---:|---:|
| fixed_top_20 | 0.237121 | 0.009884 | 64.56 | 282,654 |
| fixed_top_30 | 0.247012 | 0.012791 | 73.04 | 447,253 |
| fixed_top_40 | 0.256851 | 0.017636 | 81.80 | 617,331 |
| fill_cf_only_20 | 0.247373 | 0.016085 | 69.96 | 387,485 |
| fill_cf_only_30 | 0.257676 | 0.019574 | 79.50 | 572,615 |
| **fill_cf_only_40** | **0.264733** | **0.022093** | **87.52** | **728,305** |
| adaptive_history 25/30/40 | 0.256851 | 0.019380 | 77.37 | 531,319 |

去重后回填比普通 top-k union 更有效，尤其对 tail。history-adaptive 方案以约少 27% 的 scoring
量保留 `82.03%` 的 U50 coverage gain，但它不是本轮 primary，不能在看到结果后替换冻结方案。
它可作为 CF1-B resource pilot 的低成本诊断，不参与 A2 pass 判定。

## 5. Interpretation and next gate

证据支持“先扩候选、再做生成精排”的路线，也说明 PCRF 的局限主要来自 beam-only support，而非
协同信号失效。下一步不再调整候选 slot：

1. 先在历史 G50 上重算 constrained token-level GRAM score；
2. 要求与 cache score 高相关、top-10 set 和 baseline Hit@10 identity 达标；
3. 再对 512 users 的 `fill_cf_only_40` 做最多 90 candidates/user resource pilot；
4. 只有 score identity 与资源 gate 均通过，才对全 validation 生成 CF-only GRAM scores；
5. 最终 CF1-C 以 frozen PCRF 为主 baseline 做 calibrated union reranking。

## 6. Reproducibility pointers

- plan：`plan/第十阶段/GRAM_第十阶段_CF1-A2预算化自适应候选融合实验计划.md`
- preregistration：`artifacts/phase10/configs/cf1_a2_toys_budgeted_union_preregistered.json`
- summary：`artifacts/phase10/cf1_a2_toys_budgeted_union/summary.json`
- per-user evidence：`artifacts/phase10/cf1_a2_toys_budgeted_union/per_user_budget.tsv`
- evaluator：`experiment/phase10/eval_cf1_a2_budgeted_union.py`
- runner：`experiment/phase10/run_phase10_cf1_a2_budgeted_union.sh`
