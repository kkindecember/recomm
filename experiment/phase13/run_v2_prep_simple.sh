#!/bin/bash
# Phase 13 v2 准备脚本（无需 30G GPU lease，任意空闲 GPU 即可）
# 自动完成：embeddings → LLM priors → MLP v2 → assign IDs
#
# 使用方法：
#   export DEEPSEEK_API_KEY=sk-xxx
#   bash experiment/phase13/run_v2_prep_simple.sh
#
# 或指定 GPU：
#   bash experiment/phase13/run_v2_prep_simple.sh 5

set -euo pipefail

# ========== 配置 ==========
DATASET_NAME="Toys_cold50"
DATASET_DIR="GRAM/rec_datasets/${DATASET_NAME}"
SOURCE_DATASET_DIR="GRAM/rec_datasets/Toys"
OUTPUT_BASE="artifacts/phase13/explore/v2_toys"

EMBEDDINGS_FILE="artifacts/phase13/embeddings/Toys_sbert.pt"
LLM_PRIORS_FILE="${OUTPUT_BASE}/llm_priors.jsonl"
MLP_DIR="${OUTPUT_BASE}/mlp"
FINAL_ID_FILE="${DATASET_DIR}/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v2_mlpcold_llmprior.txt"

ITEM_TEXT="${SOURCE_DATASET_DIR}/item_plain_text.txt"
SOURCE_ID_FILE="${SOURCE_DATASET_DIR}/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
COLD_ITEMS="${DATASET_DIR}/cold_split_meta/cold_items.txt"
WARM_ITEMS="${DATASET_DIR}/cold_split_meta/warm_items.txt"

LAMBDA_LLM="${LAMBDA_LLM:-0.5}"

# GPU 选择（默认使用 GPU5，可通过参数覆盖）
GPU_ID="${1:-5}"

echo "=========================================="
echo "Phase 13 v2 准备脚本"
echo "=========================================="
echo "Dataset: ${DATASET_NAME}"
echo "GPU: ${GPU_ID}"
echo "λ_llm: ${LAMBDA_LLM}"
echo ""

# ========== Step 0: 检查环境 ==========
echo "[Step 0/4] 检查环境..."

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "❌ 错误: DEEPSEEK_API_KEY 未设置"
    echo ""
    echo "请先设置 API key:"
    echo "  export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    echo ""
    echo "获取 API key: https://platform.deepseek.com/"
    exit 1
fi

echo "✓ DEEPSEEK_API_KEY 已设置: ${DEEPSEEK_API_KEY:0:20}..."

if ! python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    echo "❌ 错误: PyTorch 未安装或不可用"
    exit 1
fi

if ! nvidia-smi -i ${GPU_ID} &>/dev/null; then
    echo "❌ 错误: GPU ${GPU_ID} 不可用"
    exit 1
fi

GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU_ID})
echo "✓ GPU${GPU_ID} 可用空闲: ${GPU_FREE} MiB"

if [[ ${GPU_FREE} -lt 5000 ]]; then
    echo "⚠️  警告: GPU 空闲内存较少 (<5GB)，可能不足"
fi

# 设置环境变量，让 PyTorch 只看到指定 GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "✓ 环境变量设置: CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo ""

# ========== Step 1: 预计算 embeddings ==========
echo "[Step 1/4] 预计算 item embeddings..."

if [[ -f "${EMBEDDINGS_FILE}" ]]; then
    echo "✓ 跳过: embeddings 已存在 (${EMBEDDINGS_FILE})"
    echo "  如需重新生成，请先删除该文件"
else
    echo "→ 开始生成 embeddings (预计 5-10 分钟)..."
    python3 experiment/phase13/protocol/precompute_item_embeddings.py \
        --item-text "${ITEM_TEXT}" \
        --output "${EMBEDDINGS_FILE}" \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --batch-size 32 \
        --max-seq-len 256
    echo "✓ Embeddings 已保存: ${EMBEDDINGS_FILE}"
fi
echo ""

# ========== Step 2: 生成 LLM priors ==========
echo "[Step 2/4] 生成 LLM priors (DeepSeek API)..."
echo "→ 调用 API 为 cold items 生成 hierarchical token 预测"
echo "  预计耗时: 20-30 分钟（首次）；有 cache 后 <1 分钟"
echo ""

mkdir -p "$(dirname "${LLM_PRIORS_FILE}")"

python3 experiment/phase13/protocol/generate_llm_priors.py \
    --cold-items "${COLD_ITEMS}" \
    --warm-items "${WARM_ITEMS}" \
    --item-text "${ITEM_TEXT}" \
    --source-id-file "${SOURCE_ID_FILE}" \
    --output-jsonl "${LLM_PRIORS_FILE}" \
    --cache-db artifacts/phase13/llm_cache.db \
    --model deepseek-chat \
    --num-shots 5 \
    --seed 42

echo ""
echo "✓ LLM priors 已保存: ${LLM_PRIORS_FILE}"
echo ""

# ========== Step 3: 训练 MLP v2 ==========
echo "[Step 3/4] 训练 MLP v2 (with LLM prior regularization)..."
echo "→ 训练 loss = L_CE + ${LAMBDA_LLM} × L_KL(MLP || LLM)"
echo "  预计耗时: 1-2 小时（200 epochs）"
echo ""

mkdir -p "${MLP_DIR}"

python3 experiment/phase13/protocol/semantic_bridge_v2.py train \
    --embeddings "${EMBEDDINGS_FILE}" \
    --id-file "${SOURCE_ID_FILE}" \
    --cold-items "${COLD_ITEMS}" \
    --llm-priors "${LLM_PRIORS_FILE}" \
    --output-dir "${MLP_DIR}" \
    --lambda-llm "${LAMBDA_LLM}" \
    --epochs 200 \
    --lr 1e-3 \
    --batch-size 512 \
    --device cuda:0 \
    --seed 12345

echo ""
echo "✓ MLP v2 已保存: ${MLP_DIR}/best.pt"
echo ""

# ========== Step 4: 分配 cold IDs ==========
echo "[Step 4/4] 分配 cold item hierarchical IDs..."
echo "→ Warm items: 保留原 ID"
echo "→ Cold items: MLP v2 argmax 预测"
echo ""

python3 experiment/phase13/protocol/assign_cold_ids.py \
    --embeddings "${EMBEDDINGS_FILE}" \
    --mlp "${MLP_DIR}/best.pt" \
    --source-id-file "${SOURCE_ID_FILE}" \
    --cold-items "${COLD_ITEMS}" \
    --output-id-file "${FINAL_ID_FILE}" \
    --device cuda:0

echo ""
echo "✓ 最终 hierarchical ID 文件: ${FINAL_ID_FILE}"
echo ""

# ========== 完成 ==========
echo "=========================================="
echo "✅ V2 准备完成！"
echo "=========================================="
echo ""
echo "输出产物:"
echo "  - Embeddings:     ${EMBEDDINGS_FILE}"
echo "  - LLM priors:     ${LLM_PRIORS_FILE}"
echo "  - MLP v2 model:   ${MLP_DIR}/best.pt"
echo "  - Training hist:  ${MLP_DIR}/training_history.json"
echo "  - Final ID file:  ${FINAL_ID_FILE}"
echo ""
echo "下一步:"
echo "  在 run_phase13_explore.sh 中添加 v2_toys 子实验配置，使用:"
echo "  hierarchical_id_type=\"hierarchy_v1_c32_l5_len32768_split_v2_mlpcold_llmprior\""
echo ""
