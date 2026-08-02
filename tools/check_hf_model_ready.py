#!/usr/bin/env python3
"""Exit successfully when a cached Hugging Face safetensors model is complete."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} MODEL_ID", file=sys.stderr)
        return 2
    model_id = sys.argv[1]
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / "hf_cache"))
    hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    repo = hub / f"models--{model_id.replace('/', '--')}"
    ref = repo / "refs" / "main"
    if not ref.is_file():
        return 1
    snapshot = repo / "snapshots" / ref.read_text().strip()
    required = [snapshot / "config.json", snapshot / "model.safetensors.index.json"]
    if not all(path.is_file() for path in required):
        return 1
    try:
        index = json.loads(required[1].read_text())
    except (OSError, json.JSONDecodeError):
        return 1
    shards = {snapshot / name for name in index.get("weight_map", {}).values()}
    if not shards or not all(path.is_file() and path.stat().st_size > 0 for path in shards):
        return 1
    if not (snapshot / "tokenizer_config.json").is_file():
        return 1
    if not any((snapshot / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        return 1
    print(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
