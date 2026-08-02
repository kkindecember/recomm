# GRAM 第七阶段：ST-GCGD-v2.1 P1 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-02
- Verification Status: COMPLETE
- Version Label: `phase7_st_gcgd_v21_p1_analysis_v1`
- Scientific Result: `STOP_BEFORE_SEALED_TEST`
- Governance Result: `PASS`

## 1. 结论

新鲜开发队列上的五臂 P1 已完成。Toys、Beauty 各 512 人，分别排除 Phase 4 用户及
8,192 名既往开发用户；每域 A/V3/B/D/E 各 512 行，sample key 完全对齐且无非有限指标。

预注册候选 D（深层 transition/session-GRU 固定融合）在两域均降低 Recall@10 与 NDCG@10；
候选 E 的训练期优势门控在固定阈值 0.70 下对两域均选择零干预，因而精确回退到 A。
两者均不满足 test nomination 门槛，结论为不读取密封测试、不读取 Sports。

## 2. 总体结果

| 域 | 臂 | Recall@10 | NDCG@10 | 相对 A 的 NDCG@10 | Recall@50 | broad harm |
|---|---|---:|---:|---:|---:|---:|
| Toys | A | 0.128906 | 0.095812 | 0.00% | 0.210938 | 0.000000 |
| Toys | V3 | 0.132812 | 0.097411 | +1.67% | 0.210938 | 0.003906 |
| Toys | B | 0.125000 | 0.089382 | -6.71% | 0.208984 | 0.005859 |
| Toys | D | 0.121094 | 0.089613 | -6.47% | 0.208984 | 0.009766 |
| Toys | E | 0.128906 | 0.095812 | 0.00% | 0.210938 | 0.000000 |
| Beauty | A | 0.107422 | 0.063261 | 0.00% | 0.210938 | 0.000000 |
| Beauty | V3 | 0.111328 | 0.065473 | +3.50% | 0.214844 | 0.001953 |
| Beauty | B | 0.101562 | 0.060078 | -5.03% | 0.207031 | 0.005859 |
| Beauty | D | 0.099609 | 0.061010 | -3.56% | 0.205078 | 0.007812 |
| Beauty | E | 0.107422 | 0.063261 | 0.00% | 0.210938 | 0.000000 |

D 的 paired-bootstrap NDCG@10 相对增益 95% CI 为 Toys `[-11.18%, -2.27%]`、Beauty
`[-8.75%, +0.82%]`；Toys 明确为负，Beauty 的 Recall@10 绝对增益 CI
`[-0.015625, -0.001953]` 也明确为负。V3 虽在两域点估计均为正，但总体 CI 均跨零，且不属于
本轮 ST-GCGD 候选，不能据此开启测试。

## 3. 优势门控诊断

Toys/Beauty 的训练期真实改善标签率分别为 10.55%/10.94%，独立校准标签率均为 7.81%。门控
平均概率约为 0.50，未有开发请求超过预冻结阈值 0.70，因此 E 的干预率均为 0。约 90% 的
阈值准确率主要来自类别不平衡，不能解释为有效的优势识别器；fail-closed 行为本身按设计正确，
避免了 D 已观察到的伤害。

## 4. 机制解释

P0-G2 证明深层有向转移模型能改善 train-only catalog ranking，但 P1 表明把其 catalog 分数以
固定标量加到生成 token logit 上不能保留该收益。D 在每个用户上都改变 beam，却没有产生
`new_hit@10 outside A beam`，反而增加 broad harm。这定位到“图空间到生成前缀空间的对齐”而非
模型容量不足：继续增大图 encoder、仍使用同一标量 prefix fusion，缺乏科学依据。

下一版应大改接口：学习 item-path/token-level 的条件适配器或 cross-attention，并用冻结 GRAM
teacher 在 train-only prefix 上做排序蒸馏与安全回退；不再沿用 catalog score 的无条件固定 alpha。

## 5. 资源与完整性

用户授权取消 `30720+256 MiB` 硬上限后，运行未使用人工 sidecar。域内 PyTorch peak reserved
为 Toys 10,142 MiB、Beauty 10,922 MiB；遥测整段最大 GPU used 为 12,922 MiB，共 603 行。
运行状态成功，CodeLlama 已恢复。parent checkpoint SHA 前后相同；test/Sports 均为 false。

收尾复测的 24 项测试全部执行通过，但 pytest 在写可再生 `.pytest_cache` 元数据时因共享
`/dev/sdb` 满盘返回 `Errno 28`；这发生在测试完成之后，不影响科学产物或结果审计。

## 6. 关键产物

- 跨域 summary SHA-256: `57ced84536d8190a594fb9d1d5f0b167b2c4adf88353b096542f2052fa7e5398`
- Toys summary SHA-256: `c70ac015809f310c213bde98155f30cce342025c987393921edf834cad1935cf`
- Beauty summary SHA-256: `c39e938e8bcb130571cc6495b91be22f108023659f9304e4183f56d1ff6c3aef`
- telemetry SHA-256: `4f1d117754f23bbb9e5ba61985133dd7cdc41f8c1dbf9093b619a1e16b3048f8`
- decision audit SHA-256: `ae9caedcb3b4909bbe952cfcbaab5ce12283e71012495974f2296409945b4277`
