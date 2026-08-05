# GRAM 第十阶段：CF1-B2 全量 Validation 候选生成打分结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_B2_TOYS_FULL_SCORE_V1`
- Verification Status: `PASSED`
- Scope: 19,412 Toys validation users
- Test/Beauty/Sports Read: false

## 1. Executive conclusion

CF1-B2 单次正式运行通过全部预注册工程与科学门槛。冻结的 `fill_cf_only_40` 候选策略在
19,412 个 validation 用户上产生 1,698,905 条候选，其中 728,305 条为 CF-only；每用户候选数
均在 50--90，所有 lexical path 合法，所有 GRAM score finite。

G50 identity sentinel 在全量数据上继续成立：Pearson `0.9999996652`、Spearman
`0.9999993113`、mean top-10 set overlap `0.999542`。cached/recomputed Hit@10 分别为
`0.119411` 和 `0.119308`，绝对差 `0.000103`，低于冻结上限 `0.001`。

正式评分用时 `2,797.03 s`（46.62 min），吞吐 `607.40 candidates/s`；PyTorch peak
allocated/reserved 为 `1,578.83/4,286 MiB`。实验结束后 CodeLlama 已恢复到物理 GPU 6。
因此 CF1-B 阶段完成，正式授权进入 CF1-C cross-fitted calibrated union reranking。

## 2. Frozen gate results

| check | requirement | observed | status |
|---|---:|---:|---|
| users | exactly 19,412 | 19,412 | PASS |
| total candidates | exactly 1,698,905 | 1,698,905 | PASS |
| CF-only candidates | exactly 728,305 | 728,305 | PASS |
| valid union budget | 100% | 100% | PASS |
| legal lexical paths | 100% | 100% | PASS |
| finite GRAM scores | 100% | 100% | PASS |
| G50 Pearson | >=0.995 | 0.9999996652 | PASS |
| G50 Spearman | >=0.995 | 0.9999993113 | PASS |
| G50 top-10 overlap | >=0.98 | 0.999542 | PASS |
| G50 Hit@10 absolute delta | <=0.001 | 0.000103 | PASS |
| peak allocated memory | <=12,000 MiB | 1,578.83 MiB | PASS |
| wall time | <=4 h | 0.777 h | PASS |

## 3. Candidate and resource profile

| field | result |
|---|---:|
| users | 19,412 |
| total candidates | 1,698,905 |
| cached G50 pairs | 970,600 |
| CF-only candidates | 728,305 |
| mean / max union size | 87.518 / 90 |
| candidates/s | 607.40 |
| peak allocated / reserved | 1,578.83 / 4,286 MiB |
| wall time | 2,797.03 s |
| telemetry samples | 556 |
| observed board memory range | 22,798--30,146 MiB |
| mean / max board utilization | 32.72% / 100% |

本轮全量吞吐比 B1 共享资源 pilot 的 `421.73 candidates/s` 更高，说明 B1 的 1.1--2 h
估计是保守的。资源数据仅支持工程可行性，不作为排序效果证据。

## 4. Artifact integrity

- `candidate_scores.tsv`：1,698,906 行（含表头），126,344,570 bytes；
- SHA256：`6a7ce546ada91fd8e87534af54706b7ad78a5d9517e7b5d49e0cb1b3c7c4941c`；
- summary 内登记的 SHA256 与独立复算完全一致；
- 字段为 `user_id, union_rank, candidate, source, gram_score`；
- 未读取 Toys test、Beauty 或 Sports。

## 5. Interpretation boundary

B2 证明的是任意合法候选的 constrained GRAM path score 可以可靠、完整且低成本地产生，且对原
G50 cached score 基本保持身份一致。B2 没有训练融合器，也没有证明新增的 CF-only 候选能在 top-k
中被正确排序。候选互补价值已由 CF1-A/A2 的 coverage oracle 支持；实际排序增益必须由 CF1-C 的
user-level cross-fitting 独立检验。

## 6. Decision

1. CF1-B 完成，不再调 candidate batch、候选 slot 或 score 定义；
2. 下一步先做 CF1-C0 feature-table 与 baseline identity audit；
3. C0 通过后执行 CF1-C1 五折 cross-fitted 单调线性/listwise calibration；
4. primary comparison 仍是 frozen PCRF `(lambda,beta,gamma)=(1.0,0.5,1.0)`；
5. Toys test 继续关闭，Beauty/Sports 不读取；
6. 若 coverage 价值存在但 C1 ranking gate 失败，结论是 calibrator 尚不成熟，不回退改 B2 分数。

## 7. Reproducibility pointers

- plan：`plan/第十阶段/GRAM_第十阶段_CF1-B2全量Validation候选生成打分计划.md`
- preregistration：`artifacts/phase10/configs/cf1_b2_toys_full_scores_preregistered.json`
- summary：`artifacts/phase10/cf1_b2_toys_full_scores/summary.json`
- candidate scores：`artifacts/phase10/cf1_b2_toys_full_scores/candidate_scores.tsv`
- evaluator：`experiment/phase10/eval_cf1_b2_full_scores.py`
- runner：`experiment/phase10/run_phase10_cf1_b2_full_scores.sh`

