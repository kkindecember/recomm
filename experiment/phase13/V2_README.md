# Phase 13 v2: LLM Prior Regularization

V2 在 v1 基础上添加 LLM prior作为正则化项,使用 DeepSeek API 单次 first-pass 预测 cold items 的 hierarchical tokens,并用 KL divergence loss 将 MLP 的输出分布向 LLM prior 对齐。

## 新增文件

### 核心脚本
- `llm_cache.py`: SQLite-based LLM API 响应缓存(线程安全,避免重复调用)
- `deepseek_client.py`: DeepSeek Chat API 的最小化 OpenAI-compatible 客户端
- `generate_llm_priors.py`: 为 cold items 生成 LLM prior 预测(5-shot few-shot prompting)
- `semantic_bridge_v2.py`: MLP 训练,添加 λ_llm · KL(MLP || LLM) 正则项

### 辅助脚本
- `prep_v2_toys.sh`: 端到端 v2 准备 pipeline(embeddings + LLM priors + MLP v2 + assign cold IDs)

## V2 Pipeline

```bash
# Step 0: 设置 API key
export DEEPSEEK_API_KEY=sk-...

# Step 1-4: 自动化 pipeline
bash experiment/phase13/prep_v2_toys.sh

# 内部步骤:
# 1. precompute_item_embeddings.py (复用 v1 的 Toys_sbert.pt,如已存在)
# 2. generate_llm_priors.py → artifacts/phase13/explore/v2_toys/llm_priors.jsonl
#    - 对每个 cold item,从 warm pool 随机采样 5-shot examples
#    - 调用 DeepSeek API 预测 5 个 hierarchical tokens
#    - 结果缓存到 artifacts/phase13/llm_cache.db(SQLite)
# 3. semantic_bridge_v2.py train → artifacts/phase13/explore/v2_toys/mlp/best.pt
#    - 训练 loss = L_CE + λ_llm · L_KL
#    - L_CE: supervised cross-entropy (ground truth tokens)
#    - L_KL: KL(MLP || LLM prior)
# 4. assign_cold_ids.py → Toys_cold50/item_generative_indexing_..._v2_mlpcold_llmprior.txt
#    - Warm items: 保留原 ID
#    - Cold items: MLP argmax 预测
```

## API Key配置

DeepSeek API key 需要设置在环境变量:
```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

或者在 prep_v2_toys.sh 运行前设置。

## Hyperparameters

- `λ_llm`: LLM prior 权重,默认 0.5 (plan 推荐起点)
  - 可通过环境变量覆盖: `LAMBDA_LLM=0.3 bash prep_v2_toys.sh`
- MLP 训练: epochs=200, lr=1e-3, batch_size=512 (同 v1)

## 与 v1 对比

| 组件 | v1 | v2 |
|---|---|---|
| Text embedding | all-MiniLM-L6-v2 | 同 v1 |
| MLP 架构 | Per-level linear heads | 同 v1 |
| 训练 loss | L_CE(ground truth) | L_CE + λ_llm · KL(MLP ∥ LLM) |
| LLM prior | 无 | DeepSeek API 5-shot first-pass |
| API 调用 | 0 | ~5963 items (cached) |
| 预期成本 | $0 | ~$3-5 (首次) |

## 输出文件

- `artifacts/phase13/explore/v2_toys/llm_priors.jsonl`: 每行一个 item 的 LLM prediction
- `artifacts/phase13/explore/v2_toys/mlp/best.pt`: 训练好的 MLP v2 模型
- `artifacts/phase13/explore/v2_toys/mlp/training_history.json`: 训练 loss/acc 曲线
- `GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v2_mlpcold_llmprior.txt`: 最终 hierarchical ID 文件(GRAM 训练用)

## GRAM 训练配置

V2 训练时指定新的 hierarchical_id_type:

```bash
# 在 run_phase13_explore.sh 中配置 v2_toys:
hierarchical_id_type="hierarchy_v1_c32_l5_len32768_split_v2_mlpcold_llmprior"
```

其余训练参数与 v1_toys 完全一致(apples-to-apples comparison)。

## Gate v2

Plan gate: cold NDCG@10 相对 v1 提升 ≥ 3%

- ✅ 通过: cold ndcg@10 相对 v1 提升 ≥ 3%
- ❌ 失败: 提升 < 3% 或退化 → LLM prior 无效,iteration 调 λ_llm / 换 prompt / 换模型

## Notes

- LLM prior generation 是 **CPU-bound**(API 调用),不需要 GPU lease
- MLP v2 training 是 **GPU-bound**,需要在 GPU 保护环境下运行(占位者hold住)
- Cache 命中后,重跑 v2 pipeline 几乎零成本(只有 MLP 训练时间)
- 如需清除 cache: `rm artifacts/phase13/llm_cache.db`
