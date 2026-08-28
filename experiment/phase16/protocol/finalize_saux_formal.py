#!/usr/bin/env python3
"""Finalize formal S-AUX reproducibility and artifact contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["verdict"] != "PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION":
        raise SystemExit("Formal S-AUX summary is not PASS")
    runner_path = ROOT / config.get("execution", {}).get(
        "runner_path", "experiment/phase16/run_stage16_s2_saux_formal.sh"
    )
    code_paths = [
        ROOT / "experiment/phase16/protocol/official_specgr_runtime.py",
        ROOT / "experiment/phase16/protocol/specgr_faithful.py",
        ROOT / "experiment/phase16/protocol/saux_formal_train.py",
        ROOT / "experiment/phase16/protocol/finalize_saux_formal.py",
        runner_path,
    ]
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_read": False,
    }
    write_json(
        output / "code_sha256.json",
        {**common, "files": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}},
    )
    write_json(
        output / "command_manifest.json",
        {
            **common,
            "exact_start_command": config.get("execution", {}).get(
                "exact_start_command", "bash experiment/phase16/run_stage16_s2_saux_formal.sh 2"
            ),
            "background_start_command": config.get("execution", {}).get(
                "background_start_command",
                "tmux new-session -d -s phase16_s2_saux_gpu2 'cd /mnt/18T/jiangtangyunzhi/projects/recomm && bash experiment/phase16/run_stage16_s2_saux_formal.sh 2'",
            ),
            "working_directory": str(ROOT),
            "physical_gpu": config["resources"]["physical_gpu"],
            "visible_gpu": 0,
            "hard_timeout_seconds": config["resources"]["hard_timeout_seconds"],
            "automatic_retry": False,
        },
    )
    required = [
        "summary.json",
        "status.json",
        "config.json",
        "source_manifest.json",
        "input_file_sha256.json",
        "code_sha256.json",
        "open_file_manifest.json",
        "data_provenance.json",
        "command_manifest.json",
        "metrics.jsonl",
        "gpu_telemetry.csv",
        "run.log",
        "checkpoints/best_model.pt",
        "checkpoints/last_state.pt",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"Formal S-AUX artifact contract missing: {missing}")
    write_json(
        output / "artifact_contract.json",
        {**common, "verdict": "PASS_SAUX_FORMAL_ARTIFACT_CONTRACT", "required": required},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
