#!/bin/bash
# Phase 13 v2_iter2 preparation pipeline
# Fixes from v2_iter1:
#   1. LLM prompt now includes per-level vocab constraint (Direction D)
#   2. semantic_bridge_v2.py: OOV → mask instead of uniform (Direction A)
#   3. lambda_llm reduced from 0.5 → 0.2 to let L_CE dominate
#
# Usage: bash experiment/phase13/prep_v2iter2.sh <toys|beauty>

set -euo pipefail

DOMAIN="${1:-toys}"

case "$DOMAIN" in
    toys)
        DATASET_NAME="Toys_cold50"
        SOURCE_DATASET_DIR="GRAM/rec_datasets/Toys"
        OUTPUT_BASE="artifacts/phase13/explore/v2_toys_iter2"
        EMBEDDINGS_FILE="artifacts/phase13/embeddings/Toys_sbert.pt"
        SOURCE_ID_SUFFIX="hierarchy_v1_c32_l5_len32768_split"
        ;;
    beauty)
        DATASET_NAME="Beauty_cold50"
        SOURCE_DATASET_DIR="GRAM/rec_datasets/Beauty"
        OUTPUT_BASE="artifacts/phase13/explore/v2_beauty_iter2"
        EMBEDDINGS_FILE="artifacts/phase13/embeddings/Beauty_sbert.pt"
        SOURCE_ID_SUFFIX="hierarchy_v1_c128_l7_len32768_split"
        ;;
    *)
        echo "Usage: $0 <toys|beauty>"
        exit 1
        ;;
esac

DATASET_DIR="GRAM/rec_datasets/${DATASET_NAME}"
LLM_PRIORS_COLD="${OUTPUT_BASE}/llm_priors_cold.jsonl"
LLM_PRIORS_WARM="${OUTPUT_BASE}/llm_priors_warm.jsonl"
LLM_PRIORS_ALL="${OUTPUT_BASE}/llm_priors_all.jsonl"
MLP_DIR="${OUTPUT_BASE}/mlp"
FINAL_ID_FILE="${DATASET_DIR}/item_generative_indexing_${SOURCE_ID_SUFFIX}_v2iter2_mlpcold_llmprior.txt"

ITEM_TEXT="${SOURCE_DATASET_DIR}/item_plain_text.txt"
SOURCE_ID_FILE="${SOURCE_DATASET_DIR}/item_generative_indexing_${SOURCE_ID_SUFFIX}.txt"
COLD_ITEMS="${DATASET_DIR}/cold_split_meta/cold_items.txt"
WARM_ITEMS="${DATASET_DIR}/cold_split_meta/warm_items.txt"

LAMBDA_LLM="${LAMBDA_LLM:-0.2}"   # iter2 default: 0.2 (was 0.5 in iter1)
TOP_N_PER_LEVEL="${TOP_N_PER_LEVEL:-500}"

mkdir -p "${OUTPUT_BASE}"

echo "=========================================="
echo "v2_iter2 prep pipeline: ${DOMAIN}"
echo "=========================================="
echo "  λ_llm = ${LAMBDA_LLM}"
echo "  top_n_per_level = ${TOP_N_PER_LEVEL}"
echo ""

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[ERROR] DEEPSEEK_API_KEY not set"
    exit 1
fi

# Step 1: Embeddings (reuse from v1/v2_iter1)
if [[ -f "${EMBEDDINGS_FILE}" ]]; then
    echo "[1/5] Embeddings exist: ${EMBEDDINGS_FILE}"
else
    echo "[1/5] Computing embeddings..."
    python3 experiment/phase13/protocol/precompute_item_embeddings.py \
        --item-text "${ITEM_TEXT}" \
        --output "${EMBEDDINGS_FILE}" \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --batch-size 32 --max-seq-len 256
fi

# Step 2: LLM priors for COLD items (vocab-constrained)
echo "[2/5] Generating LLM priors for COLD items (vocab-constrained)..."
python3 experiment/phase13/protocol/generate_llm_priors_v2iter2.py \
    --target-items "${COLD_ITEMS}" \
    --warm-items "${WARM_ITEMS}" \
    --item-text "${ITEM_TEXT}" \
    --source-id-file "${SOURCE_ID_FILE}" \
    --output-jsonl "${LLM_PRIORS_COLD}" \
    --cache-db artifacts/phase13/llm_cache.db \
    --model deepseek-chat --num-shots 5 \
    --top-n-per-level "${TOP_N_PER_LEVEL}" --seed 42

# Step 3: LLM priors for WARM items (vocab-constrained)
echo "[3/5] Generating LLM priors for WARM items (vocab-constrained)..."
python3 experiment/phase13/protocol/generate_llm_priors_v2iter2.py \
    --target-items "${WARM_ITEMS}" \
    --warm-items "${WARM_ITEMS}" \
    --item-text "${ITEM_TEXT}" \
    --source-id-file "${SOURCE_ID_FILE}" \
    --output-jsonl "${LLM_PRIORS_WARM}" \
    --cache-db artifacts/phase13/llm_cache.db \
    --model deepseek-chat --num-shots 5 \
    --top-n-per-level "${TOP_N_PER_LEVEL}" --seed 42

# Merge
cat "${LLM_PRIORS_COLD}" "${LLM_PRIORS_WARM}" > "${LLM_PRIORS_ALL}"
echo "  Merged: $(wc -l < "${LLM_PRIORS_ALL}") lines"

# Step 4: MLP v2_iter2 training with OOV mask
echo "[4/5] Training MLP v2_iter2 (OOV-masked KL, λ_llm=${LAMBDA_LLM})..."
mkdir -p "${MLP_DIR}"
python3 experiment/phase13/protocol/semantic_bridge_v2.py train \
    --embeddings "${EMBEDDINGS_FILE}" \
    --id-file "${SOURCE_ID_FILE}" \
    --cold-items "${COLD_ITEMS}" \
    --llm-priors "${LLM_PRIORS_ALL}" \
    --output-dir "${MLP_DIR}" \
    --lambda-llm "${LAMBDA_LLM}" \
    --epochs 200 --lr 1e-3 --batch-size 512 \
    --device cuda:0 --seed 12345

# Step 5: Assign cold IDs
echo "[5/5] Assigning cold IDs..."
python3 experiment/phase13/protocol/assign_cold_ids.py \
    --embeddings "${EMBEDDINGS_FILE}" \
    --mlp "${MLP_DIR}/best.pt" \
    --source-id-file "${SOURCE_ID_FILE}" \
    --cold-items "${COLD_ITEMS}" \
    --output-id-file "${FINAL_ID_FILE}" \
    --device cuda:0

echo ""
echo "=========================================="
echo "✅ v2_iter2 prep complete for ${DOMAIN}"
echo "=========================================="
echo "  Final ID file: ${FINAL_ID_FILE}"
echo "  Next: bash experiment/phase13/run_phase13_explore.sh start v2_${DOMAIN}_iter2 <gpu>"
