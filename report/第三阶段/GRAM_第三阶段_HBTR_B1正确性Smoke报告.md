# GRAM 第三阶段 HBTR-B1 正确性 Smoke 报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-22
- Verification Status: ANALYZED
- Version Label: hbtr_b1_smoke_v1_repair_complete
- Design Status: CORRECTNESS PASS; EFFECT UNVERIFIED
- Upstream: HBTR-B0 GO WITH NOVELTY NARROWING

## 1. 结论

HBTR-B1 正确性决策：**PASS，只解锁 10% pilot 的预注册与实现，不解锁中型或全量。**

Toys/Beauty 均完成锁定 baseline 的 constrained beam-50 训练负样本挖掘、
joint-margin forward/backward、临时 checkpoint 保存/重载和合法 Trie 链路检查。
smoke 权重已丢弃，没有读取 test，没有生成 pilot split，不允许从本报告得出效果结论。

## 2. 锁定实现

机器可读预注册为 `artifacts/phase3/configs/hbtr_b1_preregistered.json`：

- `K=4` 的静态 baseline beam negatives；只使用 target rank 11–50 样本；
- 序列得分为包含 EOS 的 mean teacher-forced log probability；
- `margin=0.1*prefix_weight*tail_weight`，prefix/tail weight 均封顶为 2；
- `L=L_CE+0.1*L_rank`；
- popularity 仅由 `sequence[:-2]` 统计；
- 负样本排除 target、重复项、已知历史与 Trie 外商品。

## 3. CPU 验证

第三阶段全部 16 项单元测试通过，其中 B1 新增 6 项，覆盖：

1. prefix/tail 权重与 margin 封顶；
2. padding mask 与 EOS 序列得分；
3. pairwise loss 对正样本得分的单调性；
4. 空 ranking pair 返回零；
5. `lambda=0` 的 loss/gradient 精确回退；
6. `sequence[:-2]` 防泄漏与 cache hash/行级校验。

Toys 12 行、Beauty 21 行有效 cache 通过独立重载校验。

## 4. GPU3 Smoke 结果

| 数据集 | 挖掘样本 | 有效 cache 行 | 有效率 | 优化步 | wall time | Peak allocated | Peak reserved | reload max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Toys | 100 | 12 | 12% | 2 | 30.51 s | 6,921.76 MiB | 15,020 MiB | 0.0 |
| Beauty | 100 | 21 | 21% | 2 | 34.87 s | 6,952.82 MiB | 17,982 MiB | 0.0 |

修复后的确定性步骤覆盖了非平凡权重：

- Toys 第一步同时覆盖 prefix/tail，margin 为 `0.13365–0.17820`；
- Toys 第二步覆盖 prefix-only；
- Beauty 第一步覆盖 prefix-only；
- Beauty 第二步覆盖 tail-only，margin 为 `0.12877`。

所有步骤的 token CE、ranking loss、total loss 和 gradient norm 均为有限且非零。

## 5. 首次执行与修复记录

首次 smoke 的链路和数值检查均通过，但按 cache 顺序取前两行使 Beauty 的
GPU backward 未覆盖非平凡 prefix/tail 权重。该不完整尝试保存在
`artifacts/phase3/hbtr_b1_smoke_attempt1/`，没有删除或覆盖。

修复只改变 correctness sample selection：确定性优先选择 joint、prefix 和 tail 行；
没有改 margin、lambda、K、cache 或数据。两次 cache SHA-256 完全一致：

- Toys: `dff96550d819a94de64319ef199f527cfa39dd8646b941e9b43bf06342b81a92`
- Beauty: `a8c449fb4fe9acee9f199fb6cf402cffc819d238e48236fa5e30e3f070043904`

## 6. 边界与下一门

**证据**：HBTR 实现链路可运行，静态 beam cache 在训练样本上能产生足够的
miss@10/hit@50 pairs，联合权重可参与真实 GPU gradient。

**未证明**：任何 NDCG/Recall 提升、tail 改善、跨数据集效果、组合非叠加价值或论文新颖性。

**决策**：允许下一步编写 10% pilot 预注册、生成一次性分层 split 和实现 C0–C4 对照。
在 pilot 配置、预算、主终点与晋级门槛锁定前，不启动 pilot GPU。

## 7. 产物

- `artifacts/phase3/configs/hbtr_b1_preregistered.json`
- `experiment/phase3/hbtr_b1_objective.py`
- `experiment/phase3/hbtr_b1_smoke.py`
- `experiment/phase3/test_hbtr_b1_objective.py`
- `experiment/phase3/run_phase3_hbtr_b1.sh`
- `artifacts/phase3/hbtr_b1_smoke/{Toys,Beauty}/summary.json`
- `artifacts/phase3/hbtr_b1_smoke/{Toys,Beauty}/negative_cache.json`
- `artifacts/phase3/hbtr_b1_smoke_attempt1/`
- `experiment/phase3/hbtr_b1_{gpu_board,gpu_process,disk}.csv`

