# GRAM 第十一阶段 BW1：Beam Width 候选覆盖上限验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Verification Status: `CONFIRMED_CANDIDATE_RANKING_GAP`
- Version Label: phase11_bw1_candidate_ceiling_v1
- Experiment ID: `GRAM_PHASE11_BW1_CANDIDATE_CEILING_VALIDATION_V1`

## 1. Executive conclusion

Toys、Beauty 各 512 个固定 validation 用户的 fresh constrained beam width `50/100/200`
全部通过完整性门控。beam200 相对 beam50 显著增加目标候选覆盖：Toys `+0.117188`，Beauty
`+0.126953`；但原始 GRAM Hit@10 在两个域均完全不变，冻结 PCRF 在 Toys 不变、在 Beauty
下降 `-0.001953`。

预注册决策为：`coverage_not_converted_by_frozen_pcrf`。

这不是“没有提升空间”。恰恰相反，beam200 暴露出约 12–13pp 的候选召回余量；问题是当前序列分数
与按整组候选标准化的 PCRF 无法把新增候选送入 top10。因此不能直接扩大 beam，下一步应研究宽候选集
专用的 anchored normalization / 分层候选准入，而不是继续调 beam50 的 PCRF 参数。

## 2. Frozen protocol

- Toys / Beauty 各按 `sha256("2023:<user>")` 固定 512 validation 用户；
- 每个 width 均独立执行 constrained beam search，不从 beam200 截断构造较小 beam；
- GRAM checkpoint、Trie、lexical IDs、item-head 和 PCRF
  `(lambda=1.0,beta=0.5,gamma=1.0)` 全部冻结；
- test / Sports 未读，模型无训练、checkpoint SHA 前后不变；
- 每个 dataset-width 均要求 512/512 候选数量正确、唯一、合法且分数 finite。

## 3. Results

| Dataset | Width | Candidate recall | GRAM Hit@10 | PCRF Hit@10 | PCRF NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Toys | 50 | 0.208984 | 0.121094 | 0.123047 | 0.084175 |
| Toys | 100 | 0.267578 | 0.121094 | 0.123047 | 0.085342 |
| Toys | 200 | 0.326172 | 0.121094 | 0.123047 | 0.085681 |
| Beauty | 50 | 0.207031 | 0.101562 | 0.103516 | 0.064186 |
| Beauty | 100 | 0.263672 | 0.101562 | 0.103516 | 0.063843 |
| Beauty | 200 | 0.333984 | 0.101562 | 0.101562 | 0.063352 |

Toys width200 相对 width50 的 PCRF Hit@10 paired bootstrap 95% CI 为
`[-0.005859,+0.005859]`；Beauty 为 `[-0.005859,0]`。本轮是 512 用户方向性 pilot，CI
较离散，但两个域“覆盖大增而 top10 不增”的结构模式一致。

## 4. Integrity and resources

两个数据集均通过：

- 512/512 × 3 widths 候选合法、唯一、finite；
- candidate recall 随 width 单调不降；
- fresh beam50 与冻结 cache 的 baseline Hit@10 差异在 `0.002` 内；
- GRAM 与 item-head checkpoint SHA256 运行前后相同；
- `test_read=false`、`sports_read=false`。

Toys 总运行约 `485.2s`、峰值分配显存约 `26.80 GiB`；Beauty 总运行约 `619.0s`、峰值约
`26.88 GiB`。正式 runner 于 `2026-08-04T23:48:58+08:00` 正常退出。

## 5. Mechanistic interpretation

三种 width 的 GRAM Hit@10 在每个数据集内完全相同，说明扩大 beam 主要把新的正确目标加入较低排名，
却没有改变序列模型的 top10。PCRF 在 Toys 保持 Hit@10、仅轻微改善 NDCG；Beauty 的 width200
出现小幅退化，说明对整个 200 候选重新标准化会产生候选集尺度漂移。

因此接下来的最小可证伪假设是：以原 top50 为统计锚点，对 51–200 的扩展候选计算同尺度 PCRF 分数，
只允许真正超过 top10 边界的候选准入。若 anchored expansion 在两个域都不能兑现候选覆盖，则应转向
训练式宽候选 reranker；若能安全兑现，再决定是否正式采用更宽 beam。

## 6. Artifacts

- preregistration：`plan/第十一阶段/GRAM_第十一阶段_BW1候选覆盖上限验证计划.md`
- aggregate：`artifacts/phase11/bw1_candidate_ceiling/summary.json`
- per-dataset summaries：`artifacts/phase11/bw1_candidate_ceiling/Toys/summary.json`、
  `artifacts/phase11/bw1_candidate_ceiling/Beauty/summary.json`
- fresh beams：各数据集目录下 `fresh_beams_w50.tsv`、`w100.tsv`、`w200.tsv`
- runner：`experiment/phase11/run_phase11_bw1_candidate_ceiling.sh`
- evaluator：`experiment/phase11/eval_bw1_candidate_ceiling.py`
