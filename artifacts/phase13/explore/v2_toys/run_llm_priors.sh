#!/bin/bash
# LLM priors generation with status tracking

set -euo pipefail

STATUS_FILE="artifacts/phase13/explore/v2_toys/llm_priors_status.json"
LOG_FILE="artifacts/phase13/explore/v2_toys/llm_priors.log"
OUTPUT_FILE="artifacts/phase13/explore/v2_toys/llm_priors.jsonl"

# Write initial status
cat > "$STATUS_FILE" <<EOF
{
  "step": "generate_llm_priors",
  "status": "running",
  "started_at": "$(date -Iseconds)",
  "pid": $$,
  "output_file": "$OUTPUT_FILE",
  "log_file": "$LOG_FILE"
}
EOF

# Activate conda and run
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gram-repro

python3 experiment/phase13/protocol/generate_llm_priors.py \
    --cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
    --warm-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/warm_items.txt \
    --item-text GRAM/rec_datasets/Toys/item_plain_text.txt \
    --source-id-file GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    --output-jsonl "$OUTPUT_FILE" \
    --cache-db artifacts/phase13/llm_cache.db \
    --model deepseek-chat \
    --num-shots 5 \
    --seed 42 2>&1

# Update status on success
cat > "$STATUS_FILE" <<EOF
{
  "step": "generate_llm_priors",
  "status": "completed",
  "started_at": "$(date -Iseconds)",
  "completed_at": "$(date -Iseconds)",
  "output_file": "$OUTPUT_FILE",
  "log_file": "$LOG_FILE"
}
EOF
