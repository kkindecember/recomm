#!/bin/bash
# Phase 13 v2 preparation pipeline for Toys dataset
# Runs: 1) precompute embeddings 2) generate LLM priors 3) train MLP v2 4) assign cold IDs
#
# Usage: bash experiment/phase13/prep_v2_toys.sh

set -euo pipefail

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

LAMBDA_LLM=0.5  # Default from plan; can override with env var
LAMBDA_LLM="${LAMBDA_LLM:-0.5}"

echo "[prep_v2_toys] Starting v2 preparation pipeline for ${DATASET_NAME}"
echo "[prep_v2_toys] λ_llm=${LAMBDA_LLM}"

# Step 0: Check API key
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[ERROR] DEEPSEEK_API_KEY not set in environment"
    echo "Please set it:"
    echo "  export DEEPSEEK_API_KEY=sk-..."
    exit 1
fi

# Step 1: Precompute embeddings (reuse v1 if already exists)
if [[ -f "${EMBEDDINGS_FILE}" ]]; then
    echo "[prep_v2_toys] Reusing existing embeddings: ${EMBEDDINGS_FILE}"
else
    echo "[prep_v2_toys] Step 1/4: Precomputing embeddings..."
    python3 experiment/phase13/protocol/precompute_item_embeddings.py \
        --item-text "${ITEM_TEXT}" \
        --output "${EMBEDDINGS_FILE}" \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --batch-size 32 \
        --max-seq-len 256
    echo "[prep_v2_toys] Embeddings saved to ${EMBEDDINGS_FILE}"
fi

# Step 2: Generate LLM priors for cold items
echo "[prep_v2_toys] Step 2/4: Generating LLM priors (DeepSeek API)..."
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

echo "[prep_v2_toys] LLM priors saved to ${LLM_PRIORS_FILE}"

# Step 3: Train MLP v2 with LLM prior regularization
echo "[prep_v2_toys] Step 3/4: Training MLP v2 (with LLM prior KL loss)..."
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

echo "[prep_v2_toys] MLP v2 saved to ${MLP_DIR}/best.pt"

# Step 4: Assign cold IDs (merge warm original + MLP cold)
echo "[prep_v2_toys] Step 4/4: Assigning cold IDs..."
python3 experiment/phase13/protocol/assign_cold_ids.py \
    --embeddings "${EMBEDDINGS_FILE}" \
    --mlp "${MLP_DIR}/best.pt" \
    --source-id-file "${SOURCE_ID_FILE}" \
    --cold-items "${COLD_ITEMS}" \
    --output-id-file "${FINAL_ID_FILE}" \
    --device cuda:0

echo "[prep_v2_toys] Final hierarchical ID file: ${FINAL_ID_FILE}"
echo "[prep_v2_toys] ✓ v2 preparation complete"
echo ""
echo "Next step: Run GRAM training with hierarchical_id_type=hierarchy_v1_c32_l5_len32768_split_v2_mlpcold_llmprior"
