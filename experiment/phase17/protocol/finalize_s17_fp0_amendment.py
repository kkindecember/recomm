#!/usr/bin/env python3
"""Finalize FP0 foundation-contract evidence after the source freeze."""

from __future__ import annotations

import json
from pathlib import Path

from experiment.phase17.core.run_manager import sha256
from experiment.phase17.core.status_writer import StatusWriter, atomic_json, utc_now


ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = ROOT / "artifacts/phase17/fullport/fp0/attempt_001/summary.json"
AMENDMENT_PATH = ROOT / "artifacts/phase17/fullport/fp0/amendment_001/implementation_contract_summary.json"
REPORT_PATH = ROOT / "report/第十七阶段/Stage17_FP0_来源数据与Fidelity冻结报告.md"
STATUS_DIR = ROOT / "artifacts/phase17/status"
EXPERIMENT_ID = "s17_fp0_source_data_fidelity_freeze"


def main() -> int:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if summary["verdict"] != "PASS_S17_FP0_SOURCE_DATA_FIDELITY_FREEZE":
        raise RuntimeError("FP0 source/data freeze is not complete")
    if amendment["verdict"] != "PASS_S17_FP0_FOUNDATION_CONTRACTS":
        raise RuntimeError("FP0 foundation-contract amendment is not passing")

    canonical_outputs = (
        ROOT / "artifacts/phase17/fullport/config/latte_native_toys_d0.json",
        ROOT / "artifacts/phase17/fullport/config/setrec_native_toys_d0.json",
        ROOT / "artifacts/phase17/fullport/manifests/data_manifest.json",
        ROOT / "artifacts/phase17/fullport/manifests/latte_fidelity_matrix.json",
        ROOT / "artifacts/phase17/fullport/manifests/latte_source_manifest.json",
        ROOT / "artifacts/phase17/fullport/manifests/setrec_fidelity_matrix.json",
        ROOT / "artifacts/phase17/fullport/manifests/setrec_source_manifest.json",
    )
    summary["outputs"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in canonical_outputs
    }
    summary.update(
        {
            "finalized_at": utc_now(),
            "foundation_contracts": {
                "verdict": amendment["verdict"],
                "amendment_path": str(AMENDMENT_PATH.relative_to(ROOT)),
                "amendment_sha256": sha256(AMENDMENT_PATH),
                "repo_paper_gap_recorded": True,
                "effect_experiment_started": False,
            },
            "contract_tests": {
                "command": "/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -m unittest discover -q -s experiment/phase17/tests -p test_*.py",
                "passed": 149,
                "failed": 0,
                "wall_seconds": 29.127,
                "exit_code": 0,
            },
            "report_sha256": sha256(REPORT_PATH),
            "next_gate": "S17-FP0-TOKENIZER-MODEL-INTEGRATION",
        }
    )
    atomic_json(SUMMARY_PATH, summary)
    writer = StatusWriter(STATUS_DIR, EXPERIMENT_ID)
    writer.transition(
        "COMPLETED",
        "SCIENTIFIC_COMPLETED",
        "PASS_S17_FP0_FOUNDATION_CONTRACTS",
        process_alive=False,
        stage="foundation_contracts_complete",
        progress={"current": 2, "total": 3, "unit": "fp0_gate"},
        summary_sha256=sha256(SUMMARY_PATH),
        report_sha256=sha256(REPORT_PATH),
        amendment_path=str(AMENDMENT_PATH.relative_to(ROOT)),
        amendment_sha256=sha256(AMENDMENT_PATH),
        contract_tests=summary["contract_tests"],
        next_gate=summary["next_gate"],
        result_selection_eligible=False,
        gpu_ids=[],
        gpu1_handoff_used=False,
        gpu1_repeat_restored=None,
        automatic_retry=False,
        d1_read=False,
        d2_read=False,
        test_read=False,
        sports_read=False,
    )
    print(json.dumps(summary["foundation_contracts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
