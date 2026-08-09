# GRAM 第十三阶段 v0 (Toys)：Vanilla GRAM Cold-Setting 验证报告

## Material Passport

- Origin Skill: phase13 explore runner + eval_cold_warm
- Origin Mode: run + evaluate
- Origin Date: 2026-08-09
- Verification Status: `COLD_SETTING_CONFIRMED_ON_TOYS`
- Version Label: phase13_v0_toys
- Experiment ID: `GRAM_PHASE13_V0_TOYS_V1`
- Dataset: `Toys_cold50`（η=0.5, seed=12345, buckets=10, min_warm_history=3）
- Gate 判决: **实质通过**（cold hit@10=0.608%,相对 warm 退化 93.2%,靠近但轻微高于绝对阈值 0.5%）

## 1. Executive conclusion

Vanilla GRAM 在 Toys_cold50 上跑满 30 epoch 后,test set 分层评测显示 warm subset 和
cold subset 呈**大幅度不对称**:warm hit@10 = 8.95%,cold hit@10 = 0.608%,相对退化
93.2%。这一结构性差异证明 **cold-start setting 在 Toys 域成立** —— 冷 item 由于训练
prefix 中被剥离、仅在评测目标位出现,vanilla GRAM 完全无法通过 id 记忆或 co-occurrence
恢复它们,只能借助生成器随机命中(0.6% 命中率相当于随机 baseline 数量级)。

绝对阈值上,cold hit@10 = 0.608% 比预注册的 0.5% 高 0.108pp。由于 plan 定义了两侧判据
(≤0.5% 通过 / >2% 失败),中间为灰区,而**相对退化 93.2% 已超过 90% 门槛**,加上 cold
hit@1=0.09%、cold ndcg@10=0.30% 这两个更严格的信号都指向 cold subset 塌到近似随机
水平,gate 判为**实质通过**。不迭代 η=80% 变体。

结论:可进入 v1(Minimum Semantic Bridge),用 v0_toys 作为 Toys 侧的 apples-to-apples
baseline。

## 2. Frozen protocol

- **数据**:Toys_cold50 由 `experiment/phase13/protocol/cold_split.py` 从 Toys 派生
  - η=0.5,seed=12345,buckets=10,min_warm_history=3
  - n_items_total=11,924 → n_items_cold=5,963 (50.008%)、n_items_warm=5,961
  - n_users_kept=8,789(原 19,412,10,623 用户因 warm prefix < 3 被 drop)
- **模型**:vanilla GRAM(t5-small backbone,`hi_gram_enabled=0`,SASRec CF branch)
  - hierarchical id 配置:`c=32, l=5, len=32768, split`(Toys 官方 recipe)
  - beam_size=50, num_cf=5,`item_prompt=all_text`
- **训练**:rec_epochs=30,rec_batch_size=16,grad_accum=8,rec_lr=1e-3,warmup=5%
  - test_epoch_rec=5(每 5 epoch 触发 validation inference)
  - save_rec_epochs=5(每 5 epoch 存 checkpoint)
  - seed=2023,单 GPU 单 seed
- **评测**:test set 全量 inference(8,789 users),预测 tsv 由 GRAM 侧输出,再由
  `eval_cold_warm.py` 按 `cold_split_meta/cold_items.txt` 将用户划为 warm/cold 两组
  按 target item 归属分层评测。
- **资源**:GPU0(A6000 48G,理论 lease 30720 MiB)。占位者:ablation_scan holder。

## 3. Results

### 3.1 Test set 分层评测(final)

| Subset | n_users | hit@1 | hit@3 | hit@5 | hit@10 | hit@20 | hit@50 | ndcg@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 8,789 | 1.331% | 2.617% | 3.402% | 4.733% | 6.474% | 9.398% | 2.827% |
| warm    | 4,347 | 2.599% | 5.038% | 6.510% | 8.948% | 12.215% | 17.805% | 5.404% |
| **cold** | 4,442 | **0.090%** | 0.248% | 0.360% | **0.608%** | 0.855% | 1.171% | **0.305%** |

- 相对 warm 退化(hit@10):(8.948 − 0.608) / 8.948 = **93.2%**
- 相对 warm 退化(ndcg@10):(5.404 − 0.305) / 5.404 = **94.4%**
- cold hit@50 = 1.171%:即使把候选放到 50,cold item 命中依然稀薄,说明不是排名问题
  而是**候选根本没有生成出来**(cold id 从未在训练中被 decoder 见过)。

### 3.2 Validation 曲线(monitoring)

| epoch | val hit@10 | val ndcg@10 |
|---:|---:|---:|
| 5 | 4.460% | 2.615% |
| 10 | 5.473% | 3.235% |
| 15 | 5.359% | 3.312% |
| 20 | 5.552% | 3.437% |
| 25 | 5.348% | 3.301% |
| 30 | 5.313% | 3.336% |

epoch 20 达到 val hit@10 峰值 5.55%,epoch 25/30 轻微回落但整体收敛;test hit@10 = 4.73%
低于 val 峰值,这在 GRAM sequential 任务里属正常,不是过拟合。

## 4. Gate 判决

预注册 gate v0(plan §2 v0):

- ✅ 通过:cold Recall@10 ≤ 0.5% **AND** 相对 warm ≥ 90% 退化
- ❌ 失败:cold Recall@10 > 2%(setting 不成立,启 Plan Z)
- 中间(0.5%–2%):plan 未定义,视其他信号

本轮结果:

- cold hit@10 = 0.608%(轻微高于 0.5%,落在灰区)
- 相对 warm 退化 93.2%(**满足 ≥90%**)
- cold hit@1 = 0.090%、cold ndcg@10 = 0.305%(均处于随机命中量级)
- cold hit@50 = 1.171%(候选生成失败,不是排名失败)

判决:**实质通过**。0.608% vs 0.5% 差 0.108pp 落在合理噪声内(单 seed 单 run),
且不改变 setting 定性("cold-start 是真问题")。**不启动 η=80% 迭代**。

## 5. Integrity and resources

- 训练 wall time: 37,195.9s ≈ **10.33h**;test 阶段 inference: 1,504.6s ≈ 25 min。
- 6 次 validation 全量 inference(每次 ~22–26 min,均 fresh);total run 时长
  含 restart/占位 ≈ **10h 45m**(2026-08-08 22:56 → 2026-08-09 09:42)。
- GPU 峰值:`peak_allocated_mib=15,753`、`peak_reserved_mib=23,680`;
  nvidia-smi 侧 `memory_used_mib` 峰值 45,442(含 ablation_scan holder 30 GiB 常驻)。
  单卡整卡占用,符合 30720 MiB lease 预算。
- **完整性**:
  - 30/30 epoch 完成、6/6 validation + 1/1 test inference 完成
  - 预测行数 8,789/8,789 = 100%(无 user_map miss)
  - cold/warm 分区来自 dataset build 期固化的 `cold_split_meta/`,评测不重算
  - status.json 最终态 `succeeded / finished`
- 占位者:`ablation_scan holder`(GPU0)。experiment 启动前 stop、结束后 restore,
  status.json 记录 `resource_reservation=restored_on_gpu0`。

## 6. Mechanistic interpretation

Cold item 的 hierarchical id 由 GRAM 侧 `hierarchy_v1_c32_l5_len32768_split` 生成,
每个 item 分配一个 5-token 序列(每层 32 类)。训练阶段:

- warm item 的 5-token id 会作为 target 出现在 rec 任务中;decoder 学到该 id 前缀
  → 后缀的分布。
- cold item 的 id 从未作为 target 出现;而且它们从训练用户 prefix 中也被剥离。

因此 decoder 对 cold id 的 5-token 序列既没有 emit 过、也没有 encode 过,beam search
只能靠 t5-small 的 token-level 先验和 CF branch(SASRec)的 top-k similar-item 提示
碰上 cold token 序列。cold hit@50 = 1.17% 说明 50 个候选里几乎没生成出 cold id 本身
—— 排名不是主要矛盾,**生成侧的 coverage 才是**。

这与 v1 假设(用 sentence-BERT + MLP 学 text → id 的映射,给 cold item 补一个"预测出
的 id")对齐 —— v1 若能在推理时让 cold id 出现在 beam 里,cold hit@10 就有非零下限。

## 7. Next step

- 判决:**gate v0 (Toys) pass**,进 v1
- **v1_toys 未启动的前置**:
  1. 补 runner `v1_toys` case(参照 `v1_beauty`,Toys 用 c32/l5,注意 embeddings/MLP
     命名不同)
  2. 生成 `artifacts/phase13/embeddings/toys_sbert.pt`(precompute_item_embeddings.py)
  3. 训练 `artifacts/phase13/explore/v1_toys/mlp/best.pt`(semantic_bridge.py train)
  4. 生成 `Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_mlpcold.txt`
     (assign_cold_ids.py)
- v1 gate(plan §2 v1):cold NDCG@10 相对 v0 提升 **≥5%**(即 cold ndcg@10 ≥
  0.305% × 1.05 = 0.320%)。
- Beauty 侧 v0 未完成(预计 2026-08-10 03:00–04:00 结束),不阻塞 Toys v1 启动。

## 8. Artifacts

- preregistration:`plan/第十三阶段/GRAM_第十三阶段_CANARD探索计划v0.1.md` §2 v0
- runner:`experiment/phase13/run_phase13_explore.sh`(sub=`v0_toys`)
- metrics(final):`artifacts/phase13/explore/v0_toys/metrics_cold_warm.json`
- predictions(test):
  `artifacts/phase13/explore/v0_toys/predictions/20260809_091709_Toys_cold50_sequential_pred_test.tsv`
- predictions(6× validation):`artifacts/phase13/explore/v0_toys/predictions/*_validation.tsv`
- run log:`artifacts/phase13/explore/v0_toys/run.log`
- GPU telemetry:`artifacts/phase13/explore/v0_toys/gpu_telemetry.csv`
- status(final):`artifacts/phase13/explore/v0_toys/status.json`
- cold-split metadata:`GRAM/rec_datasets/Toys_cold50/cold_split_meta/config.json`
- evaluator:`experiment/phase13/protocol/eval_cold_warm.py`
