#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
  -m experiment.phase18.protocol.s18_beam_boundary_diagnostic "$@"
