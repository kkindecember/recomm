#!/usr/bin/env bash

# GRAM Toys single-GPU reproduction (seed 2023).
# Run from this directory: bash train_gram_toys_single.sh
set -euo pipefail

SEED=2023
PHYSICAL_GPU=${PHYSICAL_GPU:-0}

ITEM_ID_TYPE=split
ID_LEN=5
NUM_CF=5
NUM_CLUSTER=32

ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len32768_split
echo ">>>>>>>>>>>>>>>>>>>>> Toys single GPU SEED: ${SEED} ITEM_ID: ${ITEM_ID}"

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" exec "${PYTHON_BIN:-python}" ../src/main_generative_gram.py --datasets Toys \
  --distributed 0 \
  --master_port 2443 \
  --gpu 0 \
  --seed ${SEED} \
  --train 1 \
  --resource_metrics 1 \
  --item_prompt_max_len 128 \
  --item_prompt all_text \
  --cf_model sasrec \
  --id_linking 1 \
  --max_his 20 \
  --rec_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --rec_lr 1e-3 \
  --rec_epochs 30 \
  --test_epoch_rec 5 \
  --save_rec_epochs 5 \
  --save_predictions 1 \
  --beam_size 50 \
  --top_k_similar_item ${NUM_CF} \
  --item_id_type ${ITEM_ID_TYPE} \
  --hierarchical_id_type ${ITEM_ID}
