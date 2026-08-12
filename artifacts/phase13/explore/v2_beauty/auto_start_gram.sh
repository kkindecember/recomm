#!/bin/bash
# v2_beauty 自动启动脚本：监控 prep 完成后自动启动 GRAM 训练
# Usage: bash artifacts/phase13/explore/v2_beauty/auto_start_gram.sh [gpu_id]

GPU_ID="${1:-6}"
OUTPUT_DIR="artifacts/phase13/explore/v2_beauty"
PREP_COMPLETE_MARKER="${OUTPUT_DIR}/.prep_complete"

echo "[auto_start] v2_beauty GRAM 自动启动守护进程"
echo "[auto_start] 目标 GPU: ${GPU_ID}"
echo "[auto_start] 监控准备阶段完成..."
echo ""

# 等待准备完成的标志
check_prep_complete() {
    # 检查所有必需文件是否存在
    [[ -f "${OUTPUT_DIR}/mlp/best.pt" ]] && \
    [[ -f "GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v2_mlpcold_llmprior.txt" ]] && \
    return 0
    return 1
}

while true; do
    if check_prep_complete; then
        echo "[auto_start] ✓ 准备阶段完成！"
        echo "[auto_start] 启动 GRAM 训练..."

        bash experiment/phase13/run_phase13_explore.sh start v2_beauty "${GPU_ID}"

        echo "[auto_start] GRAM 训练已启动"
        echo "[auto_start] 监控命令: bash experiment/phase13/run_phase13_explore.sh status v2_beauty"
        exit 0
    fi

    # 显示进度
    if [[ -f "${OUTPUT_DIR}/llm_priors_cold.jsonl" ]]; then
        COLD_COUNT=$(wc -l < "${OUTPUT_DIR}/llm_priors_cold.jsonl" 2>/dev/null || echo 0)
        echo "[auto_start] $(date '+%H:%M:%S') - Cold priors: ${COLD_COUNT}/6052"
    fi

    if [[ -f "${OUTPUT_DIR}/llm_priors_warm.jsonl" ]]; then
        WARM_COUNT=$(wc -l < "${OUTPUT_DIR}/llm_priors_warm.jsonl" 2>/dev/null || echo 0)
        echo "[auto_start] $(date '+%H:%M:%S') - Warm priors: ${WARM_COUNT}/6049"
    fi

    if [[ -f "${OUTPUT_DIR}/mlp/training_history.json" ]]; then
        echo "[auto_start] $(date '+%H:%M:%S') - MLP 训练中..."
    fi

    sleep 60  # 每分钟检查一次
done
