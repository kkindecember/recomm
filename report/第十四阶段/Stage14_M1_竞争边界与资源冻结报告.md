# Stage14 M1 竞争边界与资源冻结报告

> **结论（2026-08-20）**：14-0C/0D 初版完成，裁决为 `PASS_INTERFACE_CONTROL_COMPLETE_PATH_TRANSFER_STILL_NEEDED`。冻结的推理期 verifier 能继承 R² top-50 的 cold reachability，但在双域均显著恶化 cold NDCG@10，不能替代训练期 path transfer。SpecGR/GenRecEdit 均未通过“原实现直接同协议运行”兼容门，本阶段不占本地 GPU arm。M3 仍未授权。

## 1. 同-backbone control

统一口径：validation-only、strict item evaluator、candidate budget=50、beam K=50、冻结 v0/R²，无阈值调参。verifier 对 R² top-50 逐候选计算完整 lexical path 的 mean raw token log-likelihood，再排序。

| 域 / arm | overall NDCG@10 | warm NDCG@10 | cold NDCG@10 | cold H@50 |
|---|---:|---:|---:|---:|
| Toys v0 / GRAM likelihood only | 0.03336 | 0.06358 | 0.00276 | 0.01030 |
| Toys R² score only | 0.03128 | 0.05757 | 0.00465 | 0.08267 |
| Toys R²+GRAM verifier | 0.03320 | 0.06326 | 0.00276 | 0.08267 |
| Toys portfolio@2 | **0.03502** | 0.06098 | **0.00872** | 0.02977 |
| Beauty v0 / GRAM likelihood only | 0.03894 | 0.07536 | 0.00195 | 0.01305 |
| Beauty R² score only | 0.03853 | 0.07047 | 0.00609 | 0.08095 |
| Beauty R²+GRAM verifier | 0.03902 | 0.07552 | 0.00195 | 0.08095 |
| Beauty portfolio@2 | **0.04055** | 0.07140 | **0.00923** | 0.03253 |

verifier 相对 R² score-only 的 cold NDCG@10：Toys `−0.001895`，95% CI `[−0.002953, −0.000877]`；Beauty `−0.004142`，CI `[−0.005311, −0.003036]`。因此结论不是“verifier 无用”：它恢复了 native GRAM 的 top-rank/warm 排序，并把 R² cold H@50 完整带入候选池；但冻结 v0 likelihood 正是 cold path failure 的来源，直接拿它重排会把冷目标压回去。R2PD 仍需回答训练期 path acquisition 能否同时提升 native cold placement 与控制 warm cost。

正式 artifact：`artifacts/phase14/controls/same_backbone_verifier_formal_dual_domain_gpu5_recovery/`。Toys 8,789、Beauty 10,655 用户；未读 test、未训练。GPU5 增量峰值 6.06 GiB，双域墙钟 19m28s。

## 2. 竞争者兼容性

### SpecGR

[官方论文](https://ojs.aaai.org/index.php/AAAI/article/view/38486)与[官方仓库](https://github.com/Jamesding000/SpecGR)（审计 commit `f0ded888...`）实现的是 TIGER + UniSRec、自有 Amazon Reviews 2023 temporal cold pipeline、固定长度 RQ-VAE SID 与 semantic-sequence evaluator。它不能直接复用本项目的 GRAM multi-passage encoder、hierarchical lexical ID、frequency cold50 split 和 collision-hard-fail item evaluator；官方 `run.py` 还需修正硬编码 test 路由才能服从本项目 validation freeze。

裁决：SpecGR 是必须保留的论文边界，但当前兼容性 **FAIL_PORT_REQUIRED**，不升为第 5 个本地 arm。若后续重开，必须先完成同 split/catalog/budget/K/evaluator/cold definition 的显式 port，不引用官方数字冒充本地 baseline。

### GenRecEdit

[论文](https://arxiv.org/abs/2603.14259)与[官方仓库](https://github.com/Starrylay/GenRecEdit)（审计 commit `e6878d9...`）基于 TIGER 固定 SID position，对 decoder FFN 权重做 position-wise edit，并维护 covariance、deltaW、edit request/trigger 等额外状态。本地 clone 的 Git-LFS 大文件不完整；迁移到 GRAM variable-length lexical path 还需重新定义 position map、触发和 strict evaluator。

裁决：只作机制/update-cost 边界，不升为自跑 arm。论文报告的约 9.5% retraining time 仅标为外部数字，不触发本地自动 kill。

R2PD 的可检验差异仍成立：它迁移的是 `visible history → user-conditioned soft item distribution → batch prefix acquisition`，目标是训练后使用标准 GRAM beam；GenRecEdit 注入的是 item-centric SID pattern，并保留独立 edit state/trigger。成本优劣尚未得到本地证据，不能预称 R2PD 更便宜。

## 3. arm 与资源初锁

seed-0 core 冻结为：`v0`、`R² portfolio@2`、`same-backbone verifier`、`R2PD`。promotion set 同样使用这四项；portfolio 必须区分 resolver seed 与 backbone seed，verifier 不产生训练 seed。SpecGR/GenRecEdit 当前均不 promotion。

| 项目 | GPU-hours | 30G lease 下 wall-clock |
|---|---:|---:|
| M3/M4 seed-0 core4 | 144–188 | 6.0–9.8 d |
| 通过 Gate 后 v0+R2PD seed-1/2 | +144–188 | +6.0–9.8 d |
| Toys seed-0 ablation，最多 7 runs | +70–112 | +2.9–5.8 d |
| 当前 active package | **358–488** | **14.9–25.4 d** |
| trainable competitor 重开时的全局 contingency ceiling | **466–629** | **19.4–32.8 d** |

历史 v0 seed-0 实测：Toys 10.76h、Beauty 26.65h，双域 37.41h。本次 shared GPU5 control 的 runner wall / domain runtime 为 `1.006×`，但这只是该 inference job 的 orchestration 实测，不冒充 full-training contention 倍率。预算冻结：30G lease 规划倍率暂取 `1.0–1.25×`；无 lease 共享环境保守区间仍为 `1–20×`，不可作为 M3 执行方案。M2 smoke 必须用实测 step time 替换暂定倍率。

## 4. Gate

- 14-0C：**PASS**，interface control 完整；结果支持继续检验训练期 path transfer。
- competitor story：**PASS**，差异不只剩 lookup latency，但成本优势未证明。
- SpecGR/GenRecEdit local arm：**NO-GO（需 port）**。
- 14-0D 初锁：**PASS_WITH_M2_PENDING**。
- M3：**未授权**。M2 完成、预算按实测定版并取得用户明确批准前禁止启动。

失败恢复只保留一条经验：在已有大进程的卡上，候选 verifier 必须分块；本次冻结 `batch=4, candidate_chunk=10`。未停止或释放 GPU0/GPU5 的既有进程，GPU3 未使用。
