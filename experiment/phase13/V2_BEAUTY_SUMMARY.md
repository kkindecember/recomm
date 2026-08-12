# Phase 13 v2_beauty 准备完成文档

## 概述

v2_beauty 已完成准备，参考 v2_toys 的修复（warm+cold LLM priors）。

## 核心改动（v1 → v2）

### 技术变更
1. **LLM Prior 引入**: 使用 DeepSeek Chat API 生成语义预测
   - 5-shot few-shot prompting
   - 为 warm+cold items 都生成 priors（v2_toys 修复）

2. **MLP 训练目标增强**:
   ```
   L_total = L_CE(MLP, ground_truth) + λ_llm · KL(MLP || LLM_prior)
   ```
   - λ_llm = 0.5 (起点，可调)
   - KL 散度将 MLP 输出拉向 LLM 的语义预测

3. **Hierarchical ID 后缀变更**:
   - v1: `..._split_v1_mlpcold`
   - v2: `..._split_v2_mlpcold_llmprior`

### 关键修复（从 v2_toys 学到的）

**问题**: 最初 v2 只为 cold items 生成 LLM priors
- 训练时 warm items 没有 LLM prior 约束
- MLP 学习不稳定，warm/cold 语义空间割裂

**修复**: 为 warm+cold 所有 items 生成 LLM priors
- prep_v2_beauty.sh 自动处理（Step 2: cold, Step 3: warm, Merge: all）
- MLP 训练时对所有 items 都有 LLM 指导
- 更好的语义一致性

## 准备文件清单

### 脚本
- ✅ `experiment/phase13/prep_v2_beauty.sh` - 完整准备 pipeline
- ✅ `experiment/phase13/start_v2_beauty.sh` - 快速启动脚本
- ✅ `experiment/phase13/run_phase13_explore.sh` - 已添加 v2_beauty 配置

### 协议代码（复用）
- `experiment/phase13/protocol/generate_llm_priors.py` - LLM prior 生成
- `experiment/phase13/protocol/semantic_bridge_v2.py` - MLP v2 训练
- `experiment/phase13/protocol/assign_cold_ids.py` - ID 分配
- `experiment/phase13/protocol/llm_cache.py` - API 响应缓存
- `experiment/phase13/protocol/deepseek_client.py` - DeepSeek API 客户端

## 执行步骤

### 方案 A: 一键启动（推荐）
```bash
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export DEEPSEEK_API_KEY=sk-...
bash experiment/phase13/start_v2_beauty.sh both
# 会自动运行 prep + 提示选择 GPU + 启动训练
```

### 方案 B: 分步执行
```bash
# Step 1: 准备阶段（LLM priors + MLP v2 training）
export DEEPSEEK_API_KEY=sk-...
bash experiment/phase13/prep_v2_beauty.sh

# Step 2: GRAM 训练
bash experiment/phase13/run_phase13_explore.sh start v2_beauty 6
```

## 预期输出

### 准备阶段产物
```
artifacts/phase13/explore/v2_beauty/
├── llm_priors_cold.jsonl         # 6,052 lines (cold items)
├── llm_priors_warm.jsonl         # 6,049 lines (warm items)
├── llm_priors_all.jsonl          # 12,101 lines (merged)
└── mlp/
    ├── best.pt                   # MLP v2 模型
    ├── training_history.json     # 训练曲线
    └── vocab.json

GRAM/rec_datasets/Beauty_cold50/
└── item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v2_mlpcold_llmprior.txt
```

### 训练阶段产物
```
artifacts/phase13/explore/v2_beauty/
├── run.log                       # GRAM 训练日志
├── status.json                   # 运行状态
├── metrics_cold_warm.json        # 最终指标 ← 关键
├── predictions/                  # 预测文件
└── gram_logs/                    # GRAM 训练 checkpoints
```

## 监控与验证

```bash
# 查看状态
bash experiment/phase13/run_phase13_explore.sh status v2_beauty

# 实时日志
tail -f artifacts/phase13/explore/v2_beauty/run.log

# GPU 占用
watch -n 5 nvidia-smi

# 检查是否完成
cat artifacts/phase13/explore/v2_beauty/status.json | jq '.status'
```

## Gate v2 标准

**目标**: cold NDCG@10 相对 v1_beauty 提升 ≥ 3%

| 指标 | v1_beauty | v2_beauty 目标 | 提升要求 |
|------|-----------|----------------|----------|
| cold ndcg@10 | 0.00418 | ≥ 0.00431 | +3.0% |
| cold hit@10 | 0.00802 | ≥ 0.00826 | +3.0% |

**判定**:
- ✅ 通过: 任一指标达到 +3% → LLM prior 有效
- ❌ 失败: 两者都 < +3% → 需调整 λ_llm 或换 prompt

## 时间估算

| 阶段 | 时间 | 说明 |
|------|------|------|
| LLM priors 生成 | ~40-60 min | 12k items, DeepSeek API, 有 cache 加速 |
| MLP v2 训练 | ~40-60 min | 200 epochs, GPU |
| GRAM 训练 | ~16-20 hours | 30 epochs, Beauty 较大 |
| **总计** | **~18-22 hours** | 主要是 GRAM 训练 |

## 对比表：v0 / v1 / v2

| 维度 | v0_beauty | v1_beauty | v2_beauty |
|------|-----------|-----------|-----------|
| Cold item ID | Random/原始 | MLP 预测 | MLP + LLM prior |
| Warm item ID | 原始 | 原始 | 原始 |
| MLP 训练数据 | N/A | Warm only | Warm+Cold |
| LLM prior | 无 | 无 | DeepSeek 5-shot |
| 训练 loss | N/A | L_CE | L_CE + λ·KL |
| Cold hit@10 | 0.306% | 0.802% (+162%) | ? (目标 +3% vs v1) |
| Cold ndcg@10 | 0.179% | 0.418% (+133%) | ? (目标 ≥ 0.431%) |

## 后续计划（如果 v2 成功）

1. **多 seed 验证**: v2_beauty 和 v2_toys 各跑 3 seeds
2. **消融实验**: 
   - λ_llm ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
   - 不同 LLM (DeepSeek vs GPT-4o vs Qwen)
   - Few-shot 数量 {3, 5, 10}
3. **投稿准备**: CCF-B 级会议，2 域 × 3 seeds × 3-5 方法

## 参考文档

- V2 设计: `experiment/phase13/V2_README.md`
- v2_toys 实例: `artifacts/phase13/explore/v2_toys/`
- v1_beauty 报告: `report/第十三阶段/GRAM_第十三阶段_v1_beauty_MLP-semantic-bridge_验证报告.md`
- API key 设置: `experiment/phase13/DEEPSEEK_API_KEY_SETUP.md`

---

**文档版本**: 2026-08-12  
**状态**: v2_beauty 准备就绪，等待执行