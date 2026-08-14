#!/bin/bash
# Phase 13 v3 Stage-1 screen: (A) seed variance + (D) PK-sampler comparison.
# Pure-MLP, ~1 min per run. Runs on a spare GPU — never touches GPU0/GPU5.
set -uo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm

P=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
OUT=artifacts/phase13/explore/v3_screen
GPU=${SCREEN_GPU:-4}
mkdir -p "$OUT/stage1"

run() {  # run <domain> <tag> <weights> <seed> [extra flags...]
  local dom=$1 tag=$2 w=$3 seed=$4; shift 4
  local log="$OUT/stage1/${dom}_${tag}_s${seed}.log"
  if [[ "$dom" == toys ]]; then
    local emb=artifacts/phase13/embeddings/Toys_sbert.pt
    local idf=GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt
    local cold=GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt
  else
    local emb=artifacts/phase13/embeddings/Beauty_sbert.pt
    local idf=GRAM/rec_datasets/Beauty/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt
    local cold=GRAM/rec_datasets/Beauty_cold50/cold_split_meta/cold_items.txt
  fi
  echo "[screen] START ${dom}_${tag}_s${seed}"
  CUDA_VISIBLE_DEVICES=$GPU $P -u experiment/phase13/protocol/semantic_bridge_v3.py train \
    --embeddings "$emb" --id-file "$idf" --cold-items "$cold" \
    --output-dir "$OUT/stage1/${dom}_${tag}_s${seed}" \
    --align-weights "$w" --epochs 200 --device cuda:0 --seed "$seed" "$@" \
    > "$log" 2>&1
  echo "[screen] DONE  ${dom}_${tag}_s${seed}: $(grep -o 'Best val_acc=[0-9.]*' "$log" | tail -1)"
}

TOYS_W=1.0,0.45,0.27,0.14,0.11
TOYS_Z=0,0,0,0,0
BEAU_W=1.0,0.51,0.23,0.13,0.12,0.08,0.09
BEAU_Z=0,0,0,0,0,0,0

for seed in 12345 777 2024; do
  # (A) seed variance: control vs weighted, random-batch alignment
  run toys   ctrl     "$TOYS_Z" "$seed"
  run toys   weighted "$TOYS_W" "$seed"
  run beauty ctrl     "$BEAU_Z" "$seed"
  run beauty weighted "$BEAU_W" "$seed"
  # (D) same weights, but alignment gets same-parent PK batches
  run toys   weightedpk "$TOYS_W" "$seed" --pk-sampler
  run beauty weightedpk "$BEAU_W" "$seed" --pk-sampler
done

echo "[screen] ALL_DONE"
