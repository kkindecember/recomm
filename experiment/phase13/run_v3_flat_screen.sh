#!/bin/bash
# Phase 13 v3 iter1 option-1: flat alignment (plan §2 v3 iteration option 3).
# Single InfoNCE over the full leaf id, no hierarchy. 3 seeds x both domains,
# plus the ctrl arm re-used from the earlier screen for comparison.
# Pure-MLP, ~1 min per run. Spare GPU only — never GPU0/GPU5.
set -uo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm

P=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
OUT=artifacts/phase13/explore/v3_screen/stage1
GPU=${SCREEN_GPU:-4}
mkdir -p "$OUT"

run() {  # run <domain> <flat_weight> <seed>
  local dom=$1 fw=$2 seed=$3
  local tag="flat${fw}"
  local log="$OUT/${dom}_${tag}_s${seed}.log"
  if [[ "$dom" == toys ]]; then
    local emb=artifacts/phase13/embeddings/Toys_sbert.pt
    local idf=GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt
    local cold=GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt
    local zeros=0,0,0,0,0
  else
    local emb=artifacts/phase13/embeddings/Beauty_sbert.pt
    local idf=GRAM/rec_datasets/Beauty/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt
    local cold=GRAM/rec_datasets/Beauty_cold50/cold_split_meta/cold_items.txt
    local zeros=0,0,0,0,0,0,0
  fi
  echo "[flat] START ${dom}_${tag}_s${seed}"
  CUDA_VISIBLE_DEVICES=$GPU $P -u experiment/phase13/protocol/semantic_bridge_v3.py train \
    --embeddings "$emb" --id-file "$idf" --cold-items "$cold" \
    --output-dir "$OUT/${dom}_${tag}_s${seed}" \
    --align-weights "$zeros" --flat-align "$fw" \
    --epochs 200 --device cuda:0 --seed "$seed" > "$log" 2>&1
  echo "[flat] DONE  ${dom}_${tag}_s${seed}: $(grep -o 'Best val_acc=[0-9.]*' "$log" | tail -1)"
}

# Two strengths: 0.5 (comparable to per-level scale) and 1.0 (strong).
for seed in 12345 777 2024; do
  for fw in 0.5 1.0; do
    run toys   "$fw" "$seed"
    run beauty "$fw" "$seed"
  done
done

echo "[flat] ALL_DONE"
