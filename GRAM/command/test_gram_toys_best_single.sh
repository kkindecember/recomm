#!/usr/bin/env bash

# GRAM Toys test of a checkpoint selected by validation NDCG@10.
# Run from this directory with BEST_CHECKPOINT set.
set -euo pipefail

SEED=2023
PHYSICAL_GPU=${PHYSICAL_GPU:-0}
BEST_CHECKPOINT=${BEST_CHECKPOINT:?set BEST_CHECKPOINT to the selected Toys model}

ITEM_ID_TYPE=split
ID_LEN=5
NUM_CF=5
NUM_CLUSTER=32

ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len32768_split
if [[ ! -s "$BEST_CHECKPOINT" ]]; then
  echo "best checkpoint is missing or empty: $BEST_CHECKPOINT" >&2
  exit 1
fi

echo ">>>>>>>>>>>>>>>>>>>>> Toys best-checkpoint test SEED: ${SEED} ITEM_ID: ${ITEM_ID}"
echo "checkpoint: ${BEST_CHECKPOINT}"

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" exec "${PYTHON_BIN:-python}" ../src/main_generative_gram.py --datasets Toys \
  --distributed 0 \
  --master_port 2443 \
  --gpu 0 \
  --seed ${SEED} \
  --train 0 \
  --resource_metrics 1 \
  --rec_model_path "${BEST_CHECKPOINT}" \
  --item_prompt_max_len 128 \
  --item_prompt all_text \
  --cf_model sasrec \
  --id_linking 1 \
  --max_his 20 \
  --save_predictions 1 \
  --beam_size 50 \
  --top_k_similar_item ${NUM_CF} \
  --item_id_type ${ITEM_ID_TYPE} \
  --hierarchical_id_type ${ITEM_ID}
