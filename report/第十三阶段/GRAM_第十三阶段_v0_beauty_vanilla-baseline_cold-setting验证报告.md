# GRAM 第十三阶段 v0 (Beauty)：Vanilla Baseline Cold-Setting 验证报告

## Material Passport

- Origin Skill: phase13 explore runner + eval_cold_warm
- Origin Mode: run + evaluate（端到端自动完成）
- Origin Date: 2026-08-11（训练 2026-08-10 10:04 → 2026-08-11 12:07；test 2026-08-11 12:07 → 12:43）
- Verification Status: `V0_BEAUTY_BASELINE_CONFIRMED`
- Version Label: phase13_v0_beauty
- Experiment ID: `GRAM_PHASE13_V0_BEAUTY_V1`
- Dataset: `Beauty_cold50`（η=0.5, seed=12345, buckets=10, min_warm_history=3）
- Hierarchical ID: `hierarchy_v1_c128_l7_len32768_split`（原始 GRAM 7层 hierarchical ID）
- Gate 判决: **PASS**（cold hit@10=0.306%，远低于 warm 11.62%，退化 >97%，cold-start 问题确认）

## 1. 实验目的

确认 vanilla GRAM 在 Beauty η=50% cold-start protocol 下确实崩溃，为后续 v1/v2 改进建立对照基线。

## 2. 配置

- **数据集**: Beauty_cold50（n_items=12,101, cold=6,052, warm=6,049, n_users=10,655）
- **Cold ratio**: η=0.5（实际 50.01%）
- **Seed**: 12345（split seed），2023（training seed）
- **模型**: vanilla GRAM（t5-small backbone, SASRec CF branch）
  - `hierarchical_id_type=hierarchy_v1_c128_l7_len32768_split`
  - beam_size=50, `item_prompt=all_text`
- **训练**: rec_epochs=30, rec_batch_size=16, rec_lr=1e-3, 单 GPU
- **评测**: test set 全量 inference + eval_cold_warm.py 分层

## 3. 核心数字

| Subset | n | hit@1 | hit@5 | hit@10 | hit@20 | hit@50 | ndcg@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 10,655 | 1.680% | 4.458% | 6.063% | 8.156% | 11.694% | 3.608% |
| warm | 5,421 | 3.228% | 8.541% | 11.621% | 15.606% | 22.062% | 6.919% |
| **cold** | 5,234 | **0.076%** | 0.229% | **0.306%** | 0.439% | 0.955% | **0.179%** |

## 4. 对比

### 4.1 Beauty cold 退化程度

| 指标 | warm | cold | 退化(相对) |
|---|---:|---:|---:|
| hit@10 | 11.621% | 0.306% | **-97.4%** |
| ndcg@10 | 6.919% | 0.179% | **-97.4%** |
| hit@50 | 22.062% | 0.955% | **-95.7%** |

Cold 几乎完全崩溃，退化超过 97%。

### 4.2 Beauty vs Toys baseline 对比

| 指标 | Toys v0 | Beauty v0 |
|---|---:|---:|
| overall hit@10 | 4.733% | 6.063% |
| warm hit@10 | 8.948% | 11.621% |
| cold hit@10 | 0.608% | 0.306% |
| cold ndcg@10 | 0.305% | 0.179% |
| cold 退化 | -93.2% | -97.4% |

Beauty 的 cold 退化比 Toys 更严重（-97.4% vs -93.2%），可能因为 Beauty 的 7 层
hierarchical ID 空间更大（c128_l7 vs c32_l5），cold item 被"命中"的概率更低。

### 4.3 Validation 曲线

| epoch | val hit@10 | val ndcg@10 |
|---:|---:|---:|
| 5 | 5.575% | 3.339% |
| 10 | 6.138% | 3.652% |
| 15 | 6.673% | 3.919% |
| 20 | 6.504% | 3.771% |
| **25** | **6.832%** | **3.967%** |
| 30 | 6.710% | 3.894% |

Epoch 25 达到 val 峰值，epoch 30 轻微回落。Test 使用 epoch 30 checkpoint。

## 5. Gate 判决

预注册 gate v0:
- ✅ 通过：vanilla GRAM 在 cold subset 上 Recall@10 ≤ 0.5%（相对 warm ≥90% 退化）
- ❌ 失败：GRAM 在 cold 上依然 OK（>2% Recall@10）

本轮结果：
- cold hit@10 = 0.306%（**≤ 0.5% ✓**）
- 相对 warm 退化 = -97.4%（**≥ 90% ✓**）
- 结论：**v0 gate PASS**，cold-start setting 在 Beauty 上确认有效

## 6. 资源使用

- GPU: GPU6（A6000 48G），CodeLlama 保护
- 训练 wall time: 93,763.3s ≈ **26.05h**
- Test inference: 2,131.9s ≈ 35.5 min
- GPU 峰值: `peak_allocated_mib=15,812.58`；`peak_reserved_mib=18,376`
- 30/30 epoch 完成，6 次 validation + 1 次 test inference
- GPU 保护恢复: ✅ `resource_reservation: "restored_on_gpu6"`

## 7. Artifacts

- runner: `experiment/phase13/run_phase13_explore.sh`（sub=`v0_beauty`）
- metrics: `artifacts/phase13/explore/v0_beauty/metrics_cold_warm.json`
- predictions(test): `artifacts/phase13/explore/v0_beauty/predictions/20260811_120744_Beauty_cold50_sequential_pred_test.tsv`
- predictions(6× val): `artifacts/phase13/explore/v0_beauty/predictions/*_validation.tsv`
- run log: `artifacts/phase13/explore/v0_beauty/run.log`
- status: `artifacts/phase13/explore/v0_beauty/status.json`
- cold-split metadata: `GRAM/rec_datasets/Beauty_cold50/cold_split_meta/config.json`

## 8. 下一步

- Beauty v0 baseline 确认：cold 退化 97.4%，问题严重
- **Beauty v1**: 可以启动，验证 MLP semantic bridge 在 Beauty 域的 transfer
- Toys v2 (LLM prior) 当前正在进行中
