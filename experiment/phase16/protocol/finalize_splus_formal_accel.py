#!/usr/bin/env python3
"""Finalize accelerated formal artifacts and correct holder/recovery metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import finalize_splus_formal as base


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sys.argv = [sys.argv[0], "--config", str(config_path)]
    rc = base.main()
    if rc != 0:
        return rc
    output = ROOT / config["output_dir"]
    recovery = json.loads((output / "recovery_manifest.json").read_text(encoding="utf-8"))
    recovery["resume_command_template"] = config["execution"]["resume_command_template"]
    base.write_json(output / "recovery_manifest.json", recovery)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary["holder_released"] = True
    summary["holder_release_scope"] = "released for accelerated formal runtime only"
    summary["holder_terminal_restoration_required"] = True
    summary["batching_adaptation"] = config["batching_adaptation"]
    base.write_json(output / "summary.json", summary)
    command = json.loads((output / "command_manifest.json").read_text(encoding="utf-8"))
    command["holder_released_during_runtime"] = True
    command["holder_restored_on_every_terminal_path"] = True
    base.write_json(output / "command_manifest.json", command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
