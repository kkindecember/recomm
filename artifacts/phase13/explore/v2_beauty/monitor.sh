#!/bin/bash
# v2_beauty 实验监控脚本
# Usage: bash artifacts/phase13/explore/v2_beauty/monitor.sh

OUTPUT_DIR="artifacts/phase13/explore/v2_beauty"

echo "=========================================="
echo "v2_beauty 实验状态监控"
echo "=========================================="
echo ""

# 1. 准备阶段状态
echo "## 准备阶段 (LLM priors + MLP v2)"
echo ""

if [[ -f "${OUTPUT_DIR}/prep.log" ]]; then
    echo "准备日志最后 15 行:"
    tail -15 "${OUTPUT_DIR}/prep.log"
    echo ""
fi

# 检查关键文件
echo "关键文件检查:"
FILES=(
    "llm_priors_cold.jsonl"
    "llm_priors_warm.jsonl"
    "llm_priors_all.jsonl"
    "mlp/best.pt"
)

for f in "${FILES[@]}"; do
    if [[ -f "${OUTPUT_DIR}/${f}" ]]; then
        SIZE=$(du -h "${OUTPUT_DIR}/${f}" | cut -f1)
        echo "  ✓ ${f} (${SIZE})"
    else
        echo "  ⏳ ${f} (待生成)"
    fi
done
echo ""

# 2. GRAM 训练状态
echo "## GRAM 训练阶段"
echo ""

if [[ -f "${OUTPUT_DIR}/status.json" ]]; then
    echo "训练状态:"
    cat "${OUTPUT_DIR}/status.json"
    echo ""

    if [[ -f "${OUTPUT_DIR}/run.log" ]]; then
        echo "训练日志最后 10 行:"
        tail -10 "${OUTPUT_DIR}/run.log"
    fi
else
    echo "  ⏳ GRAM 训练尚未启动（等待准备完成）"
fi

echo ""
echo "=========================================="
echo "实时监控命令:"
echo "  tail -f ${OUTPUT_DIR}/prep.log        # 准备日志"
echo "  tail -f ${OUTPUT_DIR}/run.log         # 训练日志"
echo "  watch -n 5 bash ${OUTPUT_DIR}/monitor.sh  # 自动刷新"
echo "=========================================="
