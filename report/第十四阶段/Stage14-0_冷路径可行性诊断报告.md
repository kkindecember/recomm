# Stage 14-0：冷路径可行性诊断报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-20
- Verification Status: VERIFIED
- Version Label: phase14_stage14_0b_dual_domain_v1

## 结论

`PASS_PATH_TRANSFER_GATE`。冻结 GRAM 在双域 validation 上均存在明确且可定位的 cold path failure；tie-aware 配对统计同时表明，R² 对 cold target prefix 的 mass 与 rank 在两个域、四个 normalized-depth quartile 中均显著优于最强固定 prior。Stage14-0B 因而支持继续探索 path transfer，但不授权直接训练：仍须完成 14-0C competitor/interface audit 与 14-0D 预算锁定。

## 正式运行完整性

| 域 | Users | Cold / Warm | Frozen parity mismatch | Runtime | Peak CUDA |
|---|---:|---:|---:|---:|---:|
| Toys | 8,789 | 4,367 / 4,422 | 0 | 25.93 min | 7.08 GiB |
| Beauty | 10,655 | 5,287 / 5,368 | 0 | 40.02 min | 7.34 GiB |

两域均为 validation-only，`test_predictions_opened=false`，无模型训练。GPU0 仅使用用户既有进程之外的剩余显存。

## Cold failure profile

| 域 | Cold H@50 | Warm H@50 | Cold 首次跌出 beam 中位深度 | 主要 learned 断崖 |
|---|---:|---:|---:|---|
| Toys | 1.03% | 19.04% | d=2，d/L=0.40 | d3：cold legal NLL 12.02 / rank 17.89；warm 3.38 / 6.30 |
| Beauty | 1.31% | 25.47% | d=2，d/L=0.286 | d2：cold legal NLL 8.70 / rank 37.64；warm 4.27 / 14.87 |

结果定位为 early-to-middle path acquisition failure，而不是统一的 nominal terminal-layer failure；两域断崖的 normalized position 也不同，因此后续方法必须按相对深度建模，不能硬编码同一个 raw layer。

## R² teacher Gate

下表为 cold users 上 R² 相对每个域/区间最强固定 baseline 的配对均值；mass 正值、rank advantage 正值均代表 R² 更好。所有 95% CI 下界均大于 0。

| 域 | q1 | q2 | q3 | q4 |
|---|---|---|---|---|
| Toys mass Δ | +0.05346 | +0.01135 | +0.001087 | +0.000483 |
| Toys rank advantage | +0.65 | +13.76 | +286.53 | +527.29 |
| Beauty mass Δ | +0.01337 | +0.001014 | +0.000301 | +0.000245 |
| Beauty rank advantage | +2.15 | +167.44 | +298.88 | +333.79 |

最弱项为 Beauty q1 rank advantage `+2.15`，95% CI `[+0.03, +4.28]`，仍过 Gate。cold exact-item 层面，R² median rank 为 Toys 1,752、Beauty 1,645，优于 catalog-text 的 2,437、2,045；H@50 仅小幅提高（11.40% vs 10.90%；11.03% vs 10.38%），说明后续价值主要来自 prefix-level teacher signal，而非直接把 R² 当最终推荐器。

## 工程校正与裁决

- Constrained beam 的 filler rows 可能带 `-inf/-1e9` 累计分数；prefix survival 必须由约束后的 score-aware observer 统计，不能按 callback presence 统计。
- uniform scores 全相等，严格 `>` rank 会把所有 item 误记为 rank 1；正式 synthesis 已改用 one-based tie-aware midrank，并以 teacher-only 双域校正复算。校正前的机械 FAIL 不作为科学结果。
- 路线裁决：进入 M1 剩余的 14-0C/14-0D；通过竞争边界和预算复核后，才可进入 M2 pseudo-cold screen/matched smoke。当前不启动 full training。

## 产出

- 正式状态：`artifacts/phase14/diagnostics/oracle_prefix_probe_formal_dual_domain_score_aware_recovery/status.json`
- 双域 synthesis：`artifacts/phase14/diagnostics/oracle_prefix_probe_formal_dual_domain_score_aware_recovery/dual_domain_synthesis.json`
- Tie-aware 校正：`artifacts/phase14/diagnostics/oracle_prefix_probe_tie_aware_teacher_correction/`
- 代码：`experiment/phase14/protocol/{oracle_prefix_probe.py,synthesize_oracle_prefix_probe.py}`
