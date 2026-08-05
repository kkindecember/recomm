# GRAM 第十阶段：CF1-B1 任意候选生成打分资源 Pilot 结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_B1_TOYS_ARBITRARY_SCORE_PILOT_V1`
- Verification Status: `PASSED_WITH_RESOURCE_CONTENTION_NOTE`
- Scope: 512 deterministic Toys validation users
- Test/Beauty/Sports Read: false

## 1. Executive conclusion

CF1-B1 通过全部预注册 gate。冻结的 `fill_cf_only_40` 在 512 users 上形成 44,730 个候选，
其中 19,130 个是需要新 GRAM scoring 的 CF-only candidates；所有候选均合法、所有分数均 finite，
每用户最多 90 candidates。

G50 identity sentinel 继续成立：Pearson `0.9999996714`、Spearman `0.9999992863`、mean
top-10 set overlap `0.999805`，cached/recomputed Hit@10 均为 `0.121094`。正式 pilot 用时
`106.06 s`、吞吐 `421.73 candidates/s`，线性外推完整 19,412-user validation 约 `1.12 h`。
因此可以进入 full validation arbitrary-candidate scoring。

## 2. Frozen gate results

| check | requirement | observed | status |
|---|---:|---:|---|
| valid union budget | 100% | 100% | PASS |
| legal lexical paths | 100% | 100% | PASS |
| finite GRAM scores | 100% | 100% | PASS |
| G50 Pearson | >=0.995 | 0.9999996714 | PASS |
| G50 Spearman | >=0.995 | 0.9999992863 | PASS |
| G50 top-10 overlap | >=0.98 | 0.999805 | PASS |
| G50 Hit@10 absolute delta | <=0.001 | 0 | PASS |
| peak allocated memory | <=12,000 MiB | 1,578.83 MiB | PASS |
| pilot wall time | <=600 s | 106.06 s | PASS |
| projected full validation | <=4 h | 1.117 h | PASS |

## 3. Candidate and resource profile

| field | result |
|---|---:|
| users | 512 |
| total candidates | 44,730 |
| cached G50 paths | 25,600 |
| CF-only paths | 19,130 |
| mean union size | 87.36 |
| max union size | 90 |
| candidates/s | 421.73 |
| peak allocated / reserved | 1,578.83 / 4,284 MiB |
| wall time | 106.06 s |

按该样本比例，full validation 预计约 169.6 万条 candidate scores，其中约 72.5 万条 CF-only。
实际数目必须由 full run 输出确认，不能以线性估计替代正式产物。

## 4. Resource contention note

启动审批期间，原本空闲的 GPU5 被另一用户新增约 28 GiB 占用；runner 启动时仍满足预注册的
`>=12 GiB` free-memory lease，因此没有自动取消。正式阶段运行期间 GPU 总利用率长期较高，本进程
显存稳定在约 6.2 GiB process-visible usage，PyTorch peak allocated 仍仅 1.58 GiB。

本轮 gate 在该共享条件下仍全部通过，因此资源可行性结论是保守成立的；但 `421.73 candidates/s`
不能表述为独占 GPU 峰值吞吐。full run 应重新检查开跑时租约，并记录实际竞争状态。

## 5. Smoke accounting

runner 先做固定 2-user smoke。其合法性、finite、identity 均通过；由于 15 秒模型/数据固定加载
成本除以仅 179 candidates，naive full ETA 为 41.2 h，故 smoke summary 自身显示 resource gate
失败。该 gate 从未用于正式判定，runner 也没有据此修改策略；512-user 正式结果才是 B1 的冻结
科学判定。

## 6. Decision

1. B1 已完成，不需要调 candidate micro-batch 或改用 adaptive slots；
2. 下一步执行一次 full Toys validation `fill_cf_only_40` scoring，预计约 1.1–2 h；
3. full run 保留 G50 identity sentinel、100% legal/finite、候选硬上限和 4 h timeout gate；
4. 产物按 user/candidate/source 保存，随后才进入 CF1-C 5-fold cross-fitted calibration；
5. Toys test 继续关闭。

## 7. Reproducibility pointers

- plan：`plan/第十阶段/GRAM_第十阶段_CF1-B1任意候选生成打分资源Pilot计划.md`
- preregistration：`artifacts/phase10/configs/cf1_b1_toys_arbitrary_score_pilot_preregistered.json`
- summary：`artifacts/phase10/cf1_b1_toys_arbitrary_score_pilot/summary.json`
- candidate scores：`artifacts/phase10/cf1_b1_toys_arbitrary_score_pilot/candidate_scores.tsv`
- evaluator：`experiment/phase10/eval_cf1_b1_arbitrary_score_pilot.py`
- runner：`experiment/phase10/run_phase10_cf1_b1_arbitrary_score_pilot.sh`

