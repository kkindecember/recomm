#!/bin/bash
# v2_iter2 双域实验状态监控
# Usage: bash experiment/phase13/status_v2iter2.sh

echo "======================================================"
echo "v2_iter2 双域 prep 进度 ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "======================================================"

for domain in toys beauty; do
    if [[ "$domain" == "toys" ]]; then
        COLD_TOTAL=5963
        WARM_TOTAL=5961
    else
        COLD_TOTAL=6052
        WARM_TOTAL=6049
    fi

    OUT="artifacts/phase13/explore/v2_${domain}_iter2"

    echo ""
    echo "── ${domain} ──"

    # Cold priors 进度
    if [[ -f "${OUT}/llm_priors_cold.jsonl" ]]; then
        C=$(wc -l < "${OUT}/llm_priors_cold.jsonl")
        PCT=$(( C * 100 / COLD_TOTAL ))
        echo "  Cold priors: ${C}/${COLD_TOTAL} (${PCT}%)"
    else
        echo "  Cold priors: 待开始"
    fi

    # Warm priors 进度
    if [[ -f "${OUT}/llm_priors_warm.jsonl" ]]; then
        W=$(wc -l < "${OUT}/llm_priors_warm.jsonl")
        PCT=$(( W * 100 / WARM_TOTAL ))
        echo "  Warm priors: ${W}/${WARM_TOTAL} (${PCT}%)"
    else
        echo "  Warm priors: 待开始"
    fi

    # Merged priors
    if [[ -f "${OUT}/llm_priors_all.jsonl" ]]; then
        A=$(wc -l < "${OUT}/llm_priors_all.jsonl")
        echo "  Merged: ${A} lines"
    fi

    # MLP 训练
    if [[ -f "${OUT}/mlp/best.pt" ]]; then
        echo "  MLP v2_iter2: ✅ 完成"
        if [[ -f "${OUT}/mlp/training_history.json" ]]; then
            LAST_ACC=$(python3 -c "import json; h=json.load(open('${OUT}/mlp/training_history.json')); print(f\"{h[-1]['val_avg_acc']:.4f}\")" 2>/dev/null || echo "N/A")
            echo "    val_avg_acc (final): ${LAST_ACC}"
        fi
    elif [[ -f "${OUT}/mlp/training_history.json" ]]; then
        LAST_EPOCH=$(python3 -c "import json; h=json.load(open('${OUT}/mlp/training_history.json')); print(h[-1]['epoch'])" 2>/dev/null || echo 0)
        echo "  MLP v2_iter2: 🔄 训练中 (epoch ${LAST_EPOCH}/200)"
    else
        echo "  MLP v2_iter2: ⏳ 未开始"
    fi

    # 最终 ID 文件
    if [[ "$domain" == "toys" ]]; then
        FINAL_ID="GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v2iter2_mlpcold_llmprior.txt"
    else
        FINAL_ID="GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v2iter2_mlpcold_llmprior.txt"
    fi

    if [[ -f "$FINAL_ID" ]]; then
        echo "  Final ID: ✅ 已生成 ($(wc -l < "$FINAL_ID") lines)"
    else
        echo "  Final ID: ⏳ 待生成"
    fi
done

echo ""
echo "======================================================"
echo "启动 GRAM 训练命令 (prep 完成后):"
echo "  bash experiment/phase13/run_phase13_explore.sh start v2_toys_iter2 <gpu>"
echo "  bash experiment/phase13/run_phase13_explore.sh start v2_beauty_iter2 <gpu>"
echo ""
echo "实时监控日志:"
echo "  tail -f artifacts/phase13/explore/v2_toys_iter2/prep.log"
echo "  tail -f artifacts/phase13/explore/v2_beauty_iter2/prep.log"
echo ""
echo "自动刷新: watch -n 30 bash experiment/phase13/status_v2iter2.sh"
echo "======================================================"
