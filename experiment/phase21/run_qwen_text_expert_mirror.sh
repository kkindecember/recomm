#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*' HF_ENDPOINT=https://hf-mirror.com
.runtime/qwen_embedding_20261015/env/bin/python -u -m experiment.phase21.qwen_download_accelerated
exec bash experiment/phase21/run_qwen_text_expert.sh
