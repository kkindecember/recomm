# V2 Beauty 准备流程

## 前置条件检查

```bash
# 1. 检查 API key
echo $DEEPSEEK_API_KEY

# 2. 检查 conda 环境
conda activate gram-repro

# 3. 检查 GPU 可用性（准备阶段需要 GPU for MLP training）
nvidia-smi
```

## 准备步骤（参考 v2_toys 的修复）

### Step 1-5: 运行完整 prep 脚本
```bash
# 确保在项目根目录
cd /mnt/18T/jiangtangyunzhi/projects/recomm

# 设置 API key
export DEEPSEEK_API_KEY=sk-...

# 运行 v2 准备流程（自动生成 warm+cold priors）
bash experiment/phase13/prep_v2_beauty.sh
```

脚本会自动执行：
1. 预计算 embeddings (reuse if exists)
2. 生成 COLD items 的 LLM priors (~6052 items)
3. 生成 WARM items 的 LLM priors (~5943 items) **← v2_toys 修复的关键**
4. 合并为 llm_priors_all.jsonl (~11995 items)
5. 训练 MLP v2 (使用 warm+cold priors, λ_llm=0.5, 200 epochs)
6. Assign cold IDs (生成最终 hierarchical ID file)

### 预期输出文件

```
artifacts/phase13/explore/v2_beauty/
├── llm_priors_cold.jsonl         # Cold items LLM predictions (~6052 lines)
├── llm_priors_warm.jsonl         # Warm items LLM predictions (~5943 lines)
├── llm_priors_all.jsonl          # Merged (~11995 lines)
├── mlp/
│   ├── best.pt                   # MLP v2 model
│   ├── training_history.json     # Loss/acc curves
│   └── vocab.json                # Token vocabulary
└── ...

GRAM/rec_datasets/Beauty_cold50/
└── item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v2_mlpcold_llmprior.txt
    # Final hierarchical ID file for GRAM training
```

## v2_toys 修复说明（重要！）

**问题**: v1 只为 cold items 生成了 LLM priors，但训练时 warm items 没有 LLM prior 指导

**修复**: v2 为 warm+cold items 都生成 LLM priors
- 好处: MLP 在训练时对所有 items 都有 LLM prior 约束
- 结果: 更稳定的语义-ID 映射学习

**v2_beauty 已应用此修复**（prep_v2_beauty.sh 自动处理）

## GRAM 训练启动

```bash
# 等待 prep 完成后，启动 GRAM 训练
bash experiment/phase13/run_phase13_explore.sh start v2_beauty [gpu_id]

# 例如：使用 GPU 6
bash experiment/phase13/run_phase13_explore.sh start v2_beauty 6
```

## 监控

```bash
# 查看状态
bash experiment/phase13/run_phase13_explore.sh status v2_beauty

# 查看日志
tail -f artifacts/phase13/explore/v2_beauty/run.log

# 检查 GPU 占用
watch -n 5 nvidia-smi
```

## 关键差异：v1 vs v2

| 组件 | v1_beauty | v2_beauty |
|------|-----------|-----------|
| Text embedding | SBERT (MiniLM-L6) | Same |
| MLP 架构 | Per-level linear | Same |
| 训练 loss | L_CE only | L_CE + 0.5·KL(MLP∥LLM) |
| LLM prior | 无 | DeepSeek 5-shot |
| Prior 覆盖 | N/A | Warm+Cold all items |
| ID file suffix | v1_mlpcold | v2_mlpcold_llmprior |

## Gate v2

**成功标准**: cold NDCG@10 相对 v1_beauty 提升 ≥ 3%

- v1_beauty cold ndcg@10 = 0.418%
- v2_beauty 目标: ≥ 0.431% (相对提升 3%)

## 预期时间成本

- LLM priors 生成: ~30-60 min (12k items, API 调用, 有 cache)
- MLP v2 训练: ~30-60 min (200 epochs, GPU)
- GRAM 训练: ~16-20 hours (30 epochs, Beauty 数据集较大)

**总计**: ~17-21 hours