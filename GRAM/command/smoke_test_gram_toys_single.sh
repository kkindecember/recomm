#!/usr/bin/env bash

# One-epoch, 100-train/100-evaluation-sample Toys smoke test.
# Its metrics are diagnostic only and must not be reported as reproduction results.
# Run from this directory: bash smoke_test_gram_toys_single.sh
set -euo pipefail

SEED=2023
PHYSICAL_GPU=${PHYSICAL_GPU:-0}

ITEM_ID_TYPE=split
ID_LEN=5
NUM_CF=5
NUM_CLUSTER=32

ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len32768_split
echo ">>>>>>>>>>>>>>>>>>>>> Toys smoke test single GPU SEED: ${SEED} ITEM_ID: ${ITEM_ID}"

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
  --rec_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --rec_lr 1e-3 \
  --rec_epochs 1 \
  --test_epoch_rec 1 \
  --save_rec_epochs 1 \
  --save_predictions 1 \
  --debug_train_100 1 \
  --debug_test_100 1 \
  --beam_size 50 \
  --top_k_similar_item ${NUM_CF} \
  --item_id_type ${ITEM_ID_TYPE} \
  --hierarchical_id_type ${ITEM_ID}
