#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
ORDER_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ORDER_OUTPUT=artifacts/phase20/item_order_toys_chunk16_v3
case "${1:-status}" in
  start)
    test -f "$ORDER_OUTPUT/design.json"
    if [ -e "$ORDER_OUTPUT/controller_started.json" ]; then
      echo 'Existing item-order attempt; refusing overwrite or implicit retry.'
      exit 1
    fi
    tmux new-session -d -s s20_item_order_toys_chunk16_v3 "$ORDER_PYTHON -u -m experiment.phase20.item_order_transfer_chunk16_v3 --mode controller > $ORDER_OUTPUT/controller.log 2>&1"
    echo 'Started bounded two-arm item-order controller.'
    ;;
  status) cat "$ORDER_OUTPUT/status.json" ;;
  *) echo 'Usage: run_item_order_transfer.sh start|status'; exit 2 ;;
esac
