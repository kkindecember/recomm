# GRAM 第十阶段：CF1-C1 跨折校准融合结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_C1_TOYS_CROSSFIT_CALIBRATOR_V1`
- Verification Status: `ANALYZED`
- Evidence Class: cross-fitted validation development, not independent confirmation
- Development Gate: `FAILED`
- Toys Test/Beauty/Sports Read: false

## 1. Executive conclusion

CF1-C1 工程运行完整有效，但未通过冻结 development gate，不能进入 CF1-D Beauty external
confirmation。五折均收敛、OOF score 全部 finite、训练折标准化隔离成立；失败来自排序效果而不是
数据身份、数值异常或训练崩溃。

相对 frozen PCRF，C1 的 Hit@10 delta 为 `-0.000309`，paired bootstrap 95% CI
`[-0.002113,+0.001597]`，只有 1/5 折为正；tail Hit@10 下降 `-0.004845`。Hit@50 虽在五折
全部为正并净增加 284 个命中，但总体 delta `+0.014630` 仍低于冻结门槛 `+0.020`。

只读错误分解进一步显示：C1 对 `both` target 的 Hit@10 净增 107、对 CF-only target 净增 21，
却对 GRAM-only target 净损失 134。结论是候选互补已被部分兑现，但当前无锚定 listwise calibrator
过度改写了 frozen PCRF 的安全排序，且收益偏向 middle/head。下一步允许另行预注册一个
PCRF-anchored、source-asymmetric 的 CF1-C2；不允许重跑 C1、降低门槛或读取 Toys test。

## 2. Frozen primary comparison

| metric | frozen PCRF | C1 OOF | delta | frozen gate | status |
|---|---:|---:|---:|---:|---|
| Hit@1 | 0.041366 | 0.041727 | +0.000361 | >= -0.001 | PASS |
| Hit@10 | 0.125335 | 0.125026 | -0.000309 | >= +0.003 | FAIL |
| NDCG@10 | 0.078716 | 0.078719 | +0.000003 | diagnostic | neutral |
| Hit@50 | 0.211931 | 0.226561 | +0.014630 | >= +0.020 | FAIL |
| NDCG@50 | 0.098071 | 0.101009 | +0.002939 | diagnostic | positive |

Hit@10 有 162 个 gain、168 个 loss，净损失 6；Hit@50 有 534 个 gain、250 个 loss，净增
284。C1 兑现了 PCRF 到 union oracle `+0.052802` Hit@50 gap 的 `27.71%`，但没有把这些候选
收益稳定送入 top-10。

## 3. Gate audit

| check | observed | status |
|---|---:|---|
| Hit@10 delta >= +0.003 | -0.000309 | FAIL |
| Hit@50 delta >= +0.020 | +0.014630 | FAIL |
| tail Hit@10 non-degradation | -0.004845 | FAIL |
| Hit@1 delta >= -0.001 | +0.000361 | PASS |
| Hit@10 bootstrap lower > 0 | -0.002113 | FAIL |
| positive Hit@10 folds >= 4 | 1/5 | FAIL |
| all folds converged | 5/5 | PASS |
| all OOF scores finite | 100% | PASS |
| train-only scaling | true | PASS |

五折 Hit@50 delta 分别为 `+0.015194/+0.017255/+0.013138/+0.011077/+0.016486`；方向一致
但没有任何一折达到总体冻结门槛的证据替代权。Hit@10 五折 delta 分别为
`-0.000773/-0.000258/+0.001030/-0.000515/-0.001030`。

## 4. Subgroup behavior

| subgroup | users | Hit@10 delta | Hit@50 delta | interpretation |
|---|---:|---:|---:|---|
| target tail | 5,160 | -0.004845 | +0.006202 | top-10 safety failure |
| target middle | 9,235 | +0.000866 | +0.019491 | mainly top-50 gain |
| target head | 5,017 | +0.002193 | +0.014351 | strongest top-10 direction |
| history 1--5 | 12,673 | -0.000237 | +0.011836 | top-10 neutral/negative |
| history 6--10 | 4,319 | -0.001158 | +0.013197 | top-10 negative |
| history 11--20 | 2,420 | +0.000826 | +0.031818 | clearest top-50 benefit |

总体 Hit@10 接近零掩盖了 target-popularity 异质性；尤其 tail 下降不能由 head/middle 的改善抵消，
因为 tail safety 是冻结的联合 gate。

## 5. Source-level mechanism diagnosis

| gold source | users | Hit@10 gain/loss/net | Hit@10 delta | Hit@50 gain/loss/net |
|---|---:|---:|---:|---:|
| GRAM-only | 1,789 | 14 / 148 / -134 | -0.074902 | 0 / 250 / -250 |
| both | 2,325 | 127 / 20 / +107 | +0.046022 | 0 / 0 / 0 |
| CF-only | 1,025 | 21 / 0 / +21 | +0.020488 | 534 / 0 / +534 |
| union miss | 14,273 | 0 / 0 / 0 | 0 | 0 / 0 / 0 |

这表明完全冻结 G50 内部顺序也不是合适的 C2：它会保护 GRAM-only，却同时删除 `both` target
的 107 个净增益。C2 应保留 PCRF anchor，但允许 source-aware residual 在证据充分时重排 `both`
或插入 CF-only，并对挤出原 GRAM-only top-10 施加非对称安全代价。

五折系数本身较稳定：`source_both` 全为正，均值 `+0.1587`；`source_cf_only` 全为负，均值
`-0.1819`；`item_log_frequency` 五折全为正，均值 `+0.0288`。后者与 tail/head 分化方向一致，
但这里只能视为机制线索，不能单独证明因果，也不能通过事后删特征把 C1 改判为通过。

## 6. Statistical validation and fallacy scan

- paired bootstrap 使用冻结的 2,000 replicates、seed 2023；区间跨 0，不支持 Hit@10 正增益；
- primary comparison 与联合 gate 已预注册，NDCG、subgroup 和 source decomposition 只作解释；
- 11/11 statistical fallacy types checked；未发现严格 Simpson、ecological、Berkson、collider、
  base-rate neglect、regression-to-mean、survivorship、causal-language 或 reverse-causality 问题；
- look-elsewhere / garden-of-forking-paths 风险由冻结主比较、SHA256、禁止 retry 和不以诊断替代 gate
  控制；C2 必须明确标为 post-C1 development；
- 需要保留的 caution 是 aggregation masking：总体 Hit@10 接近零掩盖 tail/head 与 source 的反向行为。

未进行独立复跑，因此 Verification Status 保持 `ANALYZED`，不写成 `VERIFIED`。

## 7. Decision

1. CF1-C1：`FAILED_DEVELOPMENT_GATE`；
2. CF1-D Beauty：未授权；
3. C1 自动 retry、换 seed、降门槛、读取 Toys test：禁止；
4. CF1-C2：允许预注册一个 PCRF-anchored source-asymmetric primary；
5. C2 仍沿用原 C1 gate；只有全部通过才冻结规格并进入 Beauty；
6. 若 C2 再次出现 top-50 gain 但 Hit@10/tail failure，则停止 Toys CF1 calibration，保留 frozen PCRF。

## 8. Reproducibility pointers

- plan：`plan/第十阶段/GRAM_第十阶段_CF1-C跨折校准融合实验计划.md`
- C1 preregistration：`artifacts/phase10/configs/cf1_c1_toys_crossfit_calibrator_preregistered.json`
- C1 summary：`artifacts/phase10/cf1_c1_toys_crossfit_calibrator/summary.json`
- fold models：`artifacts/phase10/cf1_c1_toys_crossfit_calibrator/fold_models.json`
- per-user OOF：`artifacts/phase10/cf1_c1_toys_crossfit_calibrator/per_user_oof.tsv`
- read-only diagnostic：`artifacts/phase10/cf1_c1_error_decomposition/summary.json`
- diagnostic transitions：`artifacts/phase10/cf1_c1_error_decomposition/hit10_transitions.tsv`
- diagnostic evaluator：`experiment/phase10/analyze_cf1_c1_error_decomposition.py`

