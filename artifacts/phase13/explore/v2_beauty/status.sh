#!/bin/bash
# 快速查看 v2_beauty 状态
# Usage: bash artifacts/phase13/explore/v2_beauty/status.sh

OUTPUT_DIR="artifacts/phase13/explore/v2_beauty"

echo "============================================"
echo "v2_beauty 实验状态"
echo "============================================"
echo ""

# Phase 1: 准备阶段
echo "📋 准备阶段 (LLM priors + MLP)"
echo "────────────────────────────────────────────"

if [[ -f "${OUTPUT_DIR}/llm_priors_cold.jsonl" ]]; then
    COLD=$(wc -l < "${OUTPUT_DIR}/llm_priors_cold.jsonl")
    echo "  Cold priors: ${COLD}/6052 ($(( COLD * 100 / 6052 ))%)"
else
    echo "  Cold priors: 待开始"
fi

if [[ -f "${OUTPUT_DIR}/llm_priors_warm.jsonl" ]]; then
    WARM=$(wc -l < "${OUTPUT_DIR}/llm_priors_warm.jsonl")
    echo "  Warm priors: ${WARM}/6049 ($(( WARM * 100 / 6049 ))%)"
else
    echo "  Warm priors: 待开始"
fi

if [[ -f "${OUTPUT_DIR}/mlp/best.pt" ]]; then
    echo "  MLP v2 训练: ✅ 完成"
elif [[ -f "${OUTPUT_DIR}/mlp/training_history.json" ]]; then
    echo "  MLP v2 训练: 🔄 进行中"
else
    echo "  MLP v2 训练: ⏳ 待开始"
fi

if [[ -f "GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v2_mlpcold_llmprior.txt" ]]; then
    echo "  Cold ID 分配: ✅ 完成"
else
    echo "  Cold ID 分配: ⏳ 待开始"
fi

echo ""

# Phase 2: GRAM 训练
echo "🚀 GRAM 训练阶段"
echo "────────────────────────────────────────────"

if [[ -f "${OUTPUT_DIR}/status.json" ]]; then
    STATUS=$(grep -o '"status":"[^"]*"' "${OUTPUT_DIR}/status.json" | cut -d'"' -f4)
    STAGE=$(grep -o '"stage":"[^"]*"' "${OUTPUT_DIR}/status.json" | cut -d'"' -f4)
    echo "  状态: ${STATUS}"
    echo "  阶段: ${STAGE}"

    if [[ -f "${OUTPUT_DIR}/metrics_cold_warm.json" ]]; then
        echo "  结果: ✅ 训练完成，指标已生成"
    fi
else
    echo "  状态: ⏳ 等待准备完成"
fi

echo ""
echo "============================================"
echo "监控命令:"
echo "  tail -f ${OUTPUT_DIR}/auto_start.log    # 自动启动日志"
echo "  tail -f ${OUTPUT_DIR}/run.log           # GRAM 训练日志"
echo "  watch -n 10 bash ${OUTPUT_DIR}/status.sh  # 自动刷新状态"
echo "============================================"
