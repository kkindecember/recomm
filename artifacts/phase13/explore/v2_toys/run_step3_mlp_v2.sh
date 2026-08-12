#!/bin/bash
# V2 Toys Step 3: MLP v2 training with 30G GPU protection
# 执行步骤：
#   1. Stop ablation_scan holder (if running on GPU3)
#   2. Start GPU memory lease (30G protection)
#   3. Train MLP v2 (background)
#   4. After completion: restart ablation_scan holder

set -euo pipefail

GPU=3
OUTPUT_DIR="artifacts/phase13/explore/v2_toys"
STATUS_FILE="${OUTPUT_DIR}/mlp_v2_training_status.json"
LOG_FILE="${OUTPUT_DIR}/mlp_v2_training.log"
MLP_DIR="${OUTPUT_DIR}/mlp"

LEASE_MIB=30720
EXPECTED_PEAK_MIB=3000  # MLP v2 只需 ~2-3GB

ABLATION_SCAN_TOOL="tools/gram_ablation_scan.sh"
LEASE_HELPER="experiment/gpu_memory_lease.py"

echo "========================================"
echo "V2 Toys Step 3: MLP v2 Training"
echo "========================================"
echo "GPU: ${GPU}"
echo "Total lease: ${LEASE_MIB} MiB"
echo ""

# Step 1: Stop ablation_scan holder
echo "[1/5] Stopping ablation_scan holder on GPU${GPU}..."
bash "$ABLATION_SCAN_TOOL" stop || echo "  (already stopped or not running)"
sleep 2
echo ""

# Step 2: Start GPU memory lease sidecar
echo "[2/5] Starting GPU memory lease sidecar..."
LEASE_STATUS="${OUTPUT_DIR}/mlp_v2_lease.json"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gram-repro

nohup python3 "$LEASE_HELPER" \
    --gpu "$GPU" \
    --total-lease-mib "$LEASE_MIB" \
    --expected-workload-peak-mib "$EXPECTED_PEAK_MIB" \
    --status-path "$LEASE_STATUS" \
    > "${OUTPUT_DIR}/mlp_v2_lease.log" 2>&1 &

LEASE_PID=$!
echo "  Lease sidecar started: PID=${LEASE_PID}"
sleep 3

# Verify lease is holding
LEASE_STATE=$(python3 -c "import json; d=json.load(open('${LEASE_STATUS}')); print(d.get('state', 'unknown'))" 2>/dev/null || echo "unknown")
if [[ "$LEASE_STATE" != "holding" ]]; then
    echo "  ERROR: Lease sidecar failed to start (state=${LEASE_STATE})"
    exit 1
fi
echo "  Lease confirmed: state=holding"
echo ""

# Step 3: Write initial status
cat > "$STATUS_FILE" <<EOF
{
  "step": "train_mlp_v2",
  "status": "running",
  "started_at": "$(date -Iseconds)",
  "gpu": ${GPU},
  "lease_pid": ${LEASE_PID},
  "log_file": "$LOG_FILE"
}
EOF

# Step 4: Train MLP v2
echo "[3/5] Training MLP v2 (200 epochs, ~1-2 hours)..."
echo "  Log: ${LOG_FILE}"
echo ""

mkdir -p "$MLP_DIR"

CUDA_VISIBLE_DEVICES=${GPU} python3 experiment/phase13/protocol/semantic_bridge_v2.py train \
    --embeddings artifacts/phase13/embeddings/Toys_sbert.pt \
    --id-file GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    --cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
    --llm-priors "${OUTPUT_DIR}/llm_priors.jsonl" \
    --output-dir "$MLP_DIR" \
    --lambda-llm 0.5 \
    --epochs 200 \
    --lr 1e-3 \
    --batch-size 512 \
    --device cuda:0 \
    --seed 12345 \
    > "$LOG_FILE" 2>&1

TRAIN_RC=$?

# Step 5: Update status
if [[ $TRAIN_RC -eq 0 ]]; then
    cat > "$STATUS_FILE" <<EOF
{
  "step": "train_mlp_v2",
  "status": "completed",
  "started_at": "$(date -Iseconds)",
  "completed_at": "$(date -Iseconds)",
  "gpu": ${GPU},
  "output_dir": "$MLP_DIR",
  "log_file": "$LOG_FILE"
}
EOF
    echo "[4/5] ✓ MLP v2 training completed"
else
    cat > "$STATUS_FILE" <<EOF
{
  "step": "train_mlp_v2",
  "status": "failed",
  "started_at": "$(date -Iseconds)",
  "failed_at": "$(date -Iseconds)",
  "exit_code": ${TRAIN_RC},
  "log_file": "$LOG_FILE"
}
EOF
    echo "[4/5] ✗ MLP v2 training FAILED (exit code=${TRAIN_RC})"
fi
echo ""

# Step 6: Stop lease sidecar
echo "[5/5] Stopping GPU memory lease..."
kill $LEASE_PID 2>/dev/null || echo "  (lease already stopped)"
sleep 2
echo ""

# Step 7: Restart ablation_scan holder
echo "[6/6] Restarting ablation_scan holder on GPU${GPU}..."
bash "$ABLATION_SCAN_TOOL" start ${GPU}
sleep 3

HOLDER_STATUS=$(bash "$ABLATION_SCAN_TOOL" status | grep "=== status.json ===" -A 1 | tail -1 | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('state', 'unknown'))" 2>/dev/null || echo "unknown")
if [[ "$HOLDER_STATUS" == "holding" ]]; then
    echo "  ✓ ablation_scan holder restarted successfully"
else
    echo "  ⚠ ablation_scan holder may not be running (state=${HOLDER_STATUS})"
fi

echo ""
echo "========================================"
if [[ $TRAIN_RC -eq 0 ]]; then
    echo "✅ V2 Toys Step 3 COMPLETED"
    echo "========================================"
    echo ""
    echo "Output:"
    echo "  - MLP v2 model: ${MLP_DIR}/best.pt"
    echo "  - Training history: ${MLP_DIR}/training_history.json"
    echo "  - Log: ${LOG_FILE}"
    echo ""
    echo "Next: Run Step 4 (assign_cold_ids.py)"
else
    echo "❌ V2 Toys Step 3 FAILED"
    echo "========================================"
    echo "Check log: ${LOG_FILE}"
    exit $TRAIN_RC
fi
