# GRAM 第十阶段：CF1-B0 生成分数身份验证结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_B0_TOYS_SCORE_IDENTITY_V1`
- Verification Status: `PASSED`
- Scope: 64 deterministic Toys validation users × cached G50
- Test/Beauty/Sports Read: false

## 1. Executive conclusion

CF1-B0 通过全部预注册 identity 与 resource gates。对 3,200 个 cached GRAM beam candidates，
teacher-forced scorer 与原 `sequences_scores` 的 pooled Pearson 为 `0.9999996844`、Spearman 为
`0.9999986674`；每用户 top-10 set overlap 为 `1.0000`，pilot Hit@10 均为 `0.09375`。

因此已经具备对 beam 外 CF-only lexical paths 计算与原 GRAM beam score 同口径分数的工程基础。
下一步可以进入 512-user、最多 90 candidates/user 的 arbitrary-candidate resource pilot。

## 2. Score-definition correction

本轮在读取结果前审计了 Transformers `4.26.0` 本地实现：beam search 先做全词表
`log_softmax`，prefix constraint 只把 trie 外 token 置为 `-inf`，不会在合法 token 子集上重新
归一化；`BeamHypotheses` 再按生成长度做 length penalty。

identity scorer 因此固定为“包含 EOS 的全词表 teacher-forced token log-prob 求和，除以包含
EOS 的预测 token 数”。这项修正替代了总计划中原先的“allowed-token renormalization”表述；后者
如未来使用，只能作为新评分特征另行验证，不能冒充历史 cache identity。

## 3. Frozen gate results

| check | threshold | observed | status |
|---|---:|---:|---|
| finite scores | 100% | 100% | PASS |
| pooled Pearson | >=0.995 | 0.9999996844 | PASS |
| pooled Spearman | >=0.995 | 0.9999986674 | PASS |
| mean top-10 set overlap | >=0.98 | 1.0000 | PASS |
| Hit@10 absolute delta | <=0.001 | 0 | PASS |
| peak allocated memory | <=12,000 MiB | 1,578.83 MiB | PASS |
| wall time | <=1,800 s | 15.49 s | PASS |

平均绝对分数误差为 `0.0001043`，最大绝对误差为 `0.0009183`。这些微小误差没有改变任何
pilot top-10 集合或 Hit@10，符合 GPU generation 与 batched teacher-forcing 的数值差异范围。

## 4. Integrity and execution notes

- sample 由 `sha256("2023:" + user_id)` 确定，不按结果选择；
- 64 users、50 candidates/user，合计 3,200 paths；
- 原 epoch-30 GRAM checkpoint、validation cache、user sequence、lexical mapping、item text、
  SASRec similar-item file、prompt、scorer code 与 T5 tokenizer 均已 SHA256 锁定；
- GPU5、float32、candidate micro-batch=10；
- 4 个 unit tests 通过；正式运行单次完成，无 retry；
- 两次前置 smoke 工程退出分别发生在 CUDA 初始化和 tokenizer dependency 阶段，均未加载/读取
  科学分数；修复后 1-user smoke 才首次完成 scorer 链路；
- 为保持 Transformers 4.26 环境，只从官方 `t5-small` snapshot 下载 tokenizer/config，未下载
  或替换模型权重。

## 5. Decision

CF1-B0 score identity 已关闭。CF1-B1 应冻结：

1. 使用 CF1-A2 `fill_cf_only_40`，最多 90 candidates/user；
2. 选取 512 个 deterministic validation users；
3. 同时重算 G50 与 CF-only paths，验证全部 legal/finite；
4. 记录吞吐、peak allocated/reserved memory、候选长度和 ETA；
5. 以 B0 cached G50 identity 作为同一运行内 sentinel；
6. 资源 gate 通过后，才允许 full validation arbitrary-candidate scoring。

## 6. Reproducibility pointers

- plan：`plan/第十阶段/GRAM_第十阶段_CF1-B0生成分数身份验证与资源Pilot计划.md`
- preregistration：`artifacts/phase10/configs/cf1_b0_toys_score_identity_preregistered.json`
- summary：`artifacts/phase10/cf1_b0_toys_score_identity/summary.json`
- score evidence：`artifacts/phase10/cf1_b0_toys_score_identity/score_pairs.tsv`
- evaluator：`experiment/phase10/eval_cf1_b0_score_identity.py`
- runner：`experiment/phase10/run_phase10_cf1_b0_score_identity.sh`

