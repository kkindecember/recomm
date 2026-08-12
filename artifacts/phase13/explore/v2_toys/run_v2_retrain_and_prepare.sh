#!/bin/bash
# V2 Toys 完整流程：重新训练 MLP v2（使用 warm+cold LLM priors）并启动 GRAM 训练
#
# 前置条件：
#   - llm_priors_all.jsonl 已经合并完成（cold + warm）
#
# 步骤：
#   1. 重新训练 MLP v2（使用完整 priors）
#   2. 重新 assign cold IDs
#   3. 启动 GRAM 训练（使用 run_phase13_explore.sh）

set -euo pipefail

OUTPUT_DIR="artifacts/phase13/explore/v2_toys"
MERGED_PRIORS="${OUTPUT_DIR}/llm_priors_all.jsonl"
MLP_DIR="${OUTPUT_DIR}/mlp"

echo "=========================================="
echo "V2 Toys: 重新训练 MLP v2 + 启动 GRAM"
echo "=========================================="

# Step 1: 等待 merged priors
echo "[1/4] 检查 merged priors..."
if [[ ! -f "$MERGED_PRIORS" ]]; then
    echo "  ERROR: $MERGED_PRIORS 不存在"
    echo "  请等待 warm priors 生成完成并合并"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$MERGED_PRIORS")
if [[ $TOTAL_LINES -lt 11900 ]]; then
    echo "  ERROR: merged priors 不完整 ($TOTAL_LINES lines, expected ~11924)"
    exit 1
fi
echo "  ✓ Merged priors ready: $TOTAL_LINES items"
echo ""

# Step 2: 重新训练 MLP v2
echo "[2/4] 重新训练 MLP v2 (with warm+cold LLM priors)..."
echo "  Log: ${OUTPUT_DIR}/mlp_v2_retrain.log"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gram-repro

rm -f "${MLP_DIR}/best.pt"

CUDA_VISIBLE_DEVICES=7 python3 experiment/phase13/protocol/semantic_bridge_v2.py train \
    --embeddings artifacts/phase13/embeddings/Toys_sbert.pt \
    --id-file GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    --cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
    --llm-priors "$MERGED_PRIORS" \
    --output-dir "$MLP_DIR" \
    --lambda-llm 0.5 \
    --epochs 200 \
    --lr 1e-3 \
    --batch-size 512 \
    --device cuda:0 \
    --seed 12345 \
    > "${OUTPUT_DIR}/mlp_v2_retrain.log" 2>&1

echo "  ✓ MLP v2 retrained"
echo ""

# Step 3: 重新 assign cold IDs
echo "[3/4] 重新 assign cold IDs..."
CUDA_VISIBLE_DEVICES=7 python3 experiment/phase13/protocol/assign_cold_ids.py \
    --embeddings artifacts/phase13/embeddings/Toys_sbert.pt \
    --mlp "${MLP_DIR}/best.pt" \
    --source-id-file GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    --cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
    --output-id-file GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v2_mlpcold_llmprior.txt \
    --device cuda:0

echo "  ✓ Cold IDs assigned"
echo ""

# Step 4: 检查 v2_toys 配置
echo "[4/4] 准备启动 GRAM 训练..."

# 检查 runner 是否支持 v2_toys
if ! grep -q "v2_toys)" experiment/phase13/run_phase13_explore.sh; then
    echo "  ERROR: run_phase13_explore.sh 不支持 v2_toys"
    echo "  需要在 runner 中添加 v2_toys 配置"
    exit 1
fi

echo "  ✓ v2_toys 配置就绪"
echo ""

echo "=========================================="
echo "✅ V2 Toys MLP 准备完成"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  - MLP v2 model: ${MLP_DIR}/best.pt"
echo "  - Final ID file: GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v2_mlpcold_llmprior.txt"
echo ""
echo "下一步: 启动 GRAM 训练"
echo "  bash experiment/phase13/run_phase13_explore.sh start v2_toys [gpu]"
