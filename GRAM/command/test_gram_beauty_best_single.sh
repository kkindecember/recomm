#!/usr/bin/env bash

# GRAM Beauty phase-D test of the checkpoint selected by validation NDCG@10.
# Run from this directory: bash test_gram_beauty_best_single.sh
set -euo pipefail

SEED=2023
PHYSICAL_GPU=${PHYSICAL_GPU:-0}

ITEM_ID_TYPE=split
ID_LEN=7
NUM_CF=10
NUM_CLUSTER=128

ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len32768_split
BEST_CHECKPOINT=${BEST_CHECKPOINT:-../log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt}

if [[ ! -s "$BEST_CHECKPOINT" ]]; then
  echo "best checkpoint is missing or empty: $BEST_CHECKPOINT" >&2
  exit 1
fi

echo ">>>>>>>>>>>>>>>>>>>>> Beauty best-checkpoint test SEED: ${SEED} ITEM_ID: ${ITEM_ID}"
echo "checkpoint: ${BEST_CHECKPOINT}"

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" exec "${PYTHON_BIN:-python}" ../src/main_generative_gram.py --datasets Beauty \
  --distributed 0 \
  --master_port 2341 \
  --gpu 0 \
  --seed ${SEED} \
  --train 0 \
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
  --hierarchical_id_type ${ITEM_ID} \
  --lexical_id_type_user idgenrec
