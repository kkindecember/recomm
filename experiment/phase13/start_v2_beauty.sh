#!/bin/bash
# Quick start script for v2_beauty preparation and training
# Usage: bash start_v2_beauty.sh [prep|train|both]

set -euo pipefail

ACTION="${1:-both}"
PROJECT_ROOT="/mnt/18T/jiangtangyunzhi/projects/recomm"

cd "$PROJECT_ROOT"

case "$ACTION" in
  prep)
    echo "=========================================="
    echo "Starting v2_beauty preparation pipeline"
    echo "=========================================="

    # Check API key
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
      echo "[ERROR] DEEPSEEK_API_KEY not set"
      echo "Please run: export DEEPSEEK_API_KEY=sk-..."
      exit 1
    fi

    # Activate conda
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate gram-repro

    # Run prep pipeline
    bash experiment/phase13/prep_v2_beauty.sh

    echo ""
    echo "✅ v2_beauty preparation complete"
    echo "Next: bash $0 train"
    ;;

  train)
    echo "=========================================="
    echo "Starting v2_beauty GRAM training"
    echo "=========================================="

    # Check prep artifacts
    REQUIRED_FILES=(
      "artifacts/phase13/embeddings/Beauty_sbert.pt"
      "artifacts/phase13/explore/v2_beauty/mlp/best.pt"
      "artifacts/phase13/explore/v2_beauty/llm_priors_all.jsonl"
      "GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v2_mlpcold_llmprior.txt"
    )

    for f in "${REQUIRED_FILES[@]}"; do
      if [[ ! -f "$f" ]]; then
        echo "[ERROR] Missing required file: $f"
        echo "Please run: bash $0 prep"
        exit 1
      fi
    done

    echo "✓ All prep artifacts found"
    echo ""

    # Prompt for GPU selection
    read -p "Select GPU [0-7, default 6]: " GPU_ID
    GPU_ID="${GPU_ID:-6}"

    echo "Starting GRAM training on GPU ${GPU_ID}..."
    bash experiment/phase13/run_phase13_explore.sh start v2_beauty "$GPU_ID"

    echo ""
    echo "✅ Training started"
    echo "Monitor: bash experiment/phase13/run_phase13_explore.sh status v2_beauty"
    ;;

  both)
    bash "$0" prep && bash "$0" train
    ;;

  *)
    echo "Usage: $0 [prep|train|both]"
    echo ""
    echo "  prep  - Run v2 preparation (LLM priors + MLP training)"
    echo "  train - Start GRAM training (requires prep artifacts)"
    echo "  both  - Run prep then train (default)"
    exit 1
    ;;
esac
