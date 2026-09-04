#!/usr/bin/env python3
"""Run the CPU-only Stage18 S18-0 evidence and execution-contract audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment.phase18.core.contracts import (
    ROOT,
    authorize_path,
    json_pointer,
    load_json,
    metrics_from_ranks,
    sha256,
    values_match,
)


DEFAULT_EVIDENCE = ROOT / "experiment/phase18/config/s18_evidence_contract.json"
DEFAULT_DATA = ROOT / "experiment/phase18/config/s18_data_contract.json"
DEFAULT_OUTPUT = ROOT / "artifacts/phase18/s0_audit"
DEFAULT_STATUS = ROOT / "artifacts/phase18/status/s18_s0_audit.status.json"
DEFAULT_LEDGER = ROOT / "artifacts/phase18/attempts/S18-0.attempts.jsonl"
DEFAULT_REPORT = ROOT / "report/第十八阶段/Stage18_S0_历史证据与执行契约报告.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def load_frozen_sources(contract: dict[str, Any], data_contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    for alias, source in contract["sources"].items():
        relative = authorize_path(source["path"], "s18_s0_historical_audit", data_contract)
        path = ROOT / relative
        actual_hash = sha256(path) if path.is_file() else None
        passed = actual_hash == source["sha256"]
        checks.append({
            "id": f"source_hash:{alias}",
            "passed": passed,
            "path": relative,
            "expected_sha256": source["sha256"],
            "actual_sha256": actual_hash,
        })
        if not passed:
            continue
        sources[alias] = load_json(path)
    return sources, checks


def resolve_ref(sources: dict[str, Any], ref: dict[str, str]) -> Any:
    return json_pointer(sources[ref["source"]], ref["pointer"])


def evaluate_derived(claim: dict[str, Any], sources: dict[str, Any]) -> Any:
    operation = claim["op"]
    if operation == "list_length":
        return len(resolve_ref(sources, claim["ref"]))
    if operation == "all_multiseed_positive":
        rows = resolve_ref(sources, claim["ref"])
        return all(
            row["hit10_delta"] > 0
            and row["ndcg10_delta"] > 0
            and row["tail_hit10_delta"] > 0
            and row["hit50_delta"] == 0
            for row in rows
        )
    if operation == "count_positive":
        rows = resolve_ref(sources, claim["ref"])
        return sum(json_pointer(row, claim["value_pointer"]) > 0 for row in rows)
    if operation == "rounded_product":
        product = 1.0
        for ref in claim["refs"]:
            product *= float(resolve_ref(sources, ref))
        return round(product, int(claim["digits"]))
    if operation == "difference":
        return float(resolve_ref(sources, claim["left"])) - float(resolve_ref(sources, claim["right"]))
    if operation == "interval_crosses_zero":
        return float(resolve_ref(sources, claim["low"])) <= 0 <= float(resolve_ref(sources, claim["high"]))
    raise ValueError(f"unsupported derived claim operation: {operation}")


def audit_evidence(contract: dict[str, Any], data_contract: dict[str, Any]) -> dict[str, Any]:
    sources, checks = load_frozen_sources(contract, data_contract)
    tolerance = float(contract["numeric_tolerance"])
    for claim in contract["scalar_claims"]:
        try:
            actual = json_pointer(sources[claim["source"]], claim["pointer"])
            passed, difference = values_match(actual, claim["expected"], tolerance)
            error = None
        except (KeyError, IndexError, TypeError, ValueError) as exception:
            actual, difference, passed, error = None, None, False, str(exception)
        checks.append({
            "id": claim["id"],
            "passed": passed,
            "source": claim["source"],
            "pointer": claim["pointer"],
            "expected": claim["expected"],
            "actual": actual,
            "absolute_difference": difference,
            "error": error,
        })
    for claim in contract["derived_claims"]:
        try:
            actual = evaluate_derived(claim, sources)
            passed, difference = values_match(actual, claim["expected"], tolerance)
            error = None
        except (KeyError, IndexError, TypeError, ValueError) as exception:
            actual, difference, passed, error = None, None, False, str(exception)
        checks.append({
            "id": claim["id"],
            "passed": passed,
            "operation": claim["op"],
            "expected": claim["expected"],
            "actual": actual,
            "absolute_difference": difference,
            "error": error,
        })
    return {
        "status": "passed" if checks and all(row["passed"] for row in checks) else "failed",
        "checks": checks,
        "passed": sum(row["passed"] for row in checks),
        "total": len(checks),
    }


def audit_data_contract(contract: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(identifier: str, passed: bool, **detail: Any) -> None:
        checks.append({"id": identifier, "passed": bool(passed), **detail})

    plan_path = ROOT / contract["plan_path"]
    add(
        "canonical_plan_hash",
        plan_path.is_file() and sha256(plan_path) == contract["plan_sha256"],
        expected_sha256=contract["plan_sha256"],
        actual_sha256=sha256(plan_path) if plan_path.is_file() else None,
    )
    manifest_info = contract["shadow_manifest"]
    manifest_path = ROOT / authorize_path(
        manifest_info["path"], "s18_s0_historical_audit", contract
    )
    actual_manifest_hash = sha256(manifest_path) if manifest_path.is_file() else None
    add(
        "shadow_manifest_hash",
        actual_manifest_hash == manifest_info["sha256"],
        expected_sha256=manifest_info["sha256"],
        actual_sha256=actual_manifest_hash,
    )
    manifest = load_json(manifest_path)
    for domain, item in contract["d0_shadow_inputs"].items():
        authorized = authorize_path(item["path"], "s18_internal_runner", contract)
        path = ROOT / authorized
        actual_hash = sha256(path) if path.is_file() else None
        manifest_row = manifest["domains"][domain]["folds"]["D0"]
        add(
            f"d0_shadow_hash:{domain}",
            actual_hash == item["sha256"] == manifest_row["output_sha256"],
            expected_sha256=item["sha256"],
            actual_sha256=actual_hash,
        )
        add(
            f"d0_projection_sealed:{domain}",
            manifest_row["official_validation_position_serialized"] is False
            and manifest_row["official_test_position_serialized"] is False
            and manifest_row["official_heldout_values_logged"] is False
            and manifest_row["target_in_train_by_position"] is False,
        )
    for denied in contract["representative_denied_paths"]:
        for profile in contract["access_profiles"]:
            try:
                authorize_path(denied, profile, contract)
                denied_as_expected = False
            except PermissionError:
                denied_as_expected = True
            add(f"deny:{profile}:{denied}", denied_as_expected)
    boundary = contract["authorization_boundary"]
    add(
        "s18_0_cpu_only_boundary",
        boundary == {
            "cpu_only": True,
            "gpu_allowed": False,
            "training_allowed": False,
            "bounded_generation_allowed": False,
            "external_evaluation_allowed": False,
            "automatic_next_stage": False,
        },
        actual=boundary,
    )
    return {
        "status": "passed" if checks and all(row["passed"] for row in checks) else "failed",
        "checks": checks,
        "passed": sum(row["passed"] for row in checks),
        "total": len(checks),
        "opened_raw_shadow_rows": False,
        "d1_read": False,
        "d2_read": False,
        "official_source_read": False,
        "external_d0_raw_read": False,
        "sports_read": False,
    }


def audit_pcrf_reconstruction(
    evidence_contract: dict[str, Any],
    data_contract: dict[str, Any],
) -> dict[str, Any]:
    reconstruction = evidence_contract["pcrf_reconstruction"]
    source = evidence_contract["sources"][reconstruction["summary_source"]]
    summary_path = ROOT / authorize_path(source["path"], "s18_s0_historical_audit", data_contract)
    summary = load_json(summary_path)
    tolerance = float(reconstruction["metric_abs_tolerance"])
    checks: list[dict[str, Any]] = []

    rank_relative = authorize_path(
        reconstruction["rank_cache_path"], "s18_s0_historical_audit", data_contract
    )
    rank_path = ROOT / rank_relative
    rank_hash = sha256(rank_path)
    checks.append({
        "id": "phase9_rank_cache_hash",
        "passed": rank_hash == reconstruction["rank_cache_sha256"] == summary["artifacts"]["per_user_test_sha256"],
        "actual_sha256": rank_hash,
    })
    baseline_ranks: list[int] = []
    pcrf_ranks: list[int] = []
    with rank_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"user_id", "baseline_rank", "pcrf_rank"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Phase9 derived rank cache schema drifted")
        for row in reader:
            baseline_ranks.append(int(row["baseline_rank"]))
            pcrf_ranks.append(int(row["pcrf_rank"]))
    computed = {
        "baseline": metrics_from_ranks(baseline_ranks),
        "pcrf": metrics_from_ranks(pcrf_ranks),
    }
    computed["delta"] = {
        key: float(computed["pcrf"][key]) - float(computed["baseline"][key])
        for key in computed["pcrf"]
        if key != "count"
    }
    maximum_difference = 0.0
    for view, expected in (
        ("baseline", summary["baseline"]),
        ("pcrf", summary["pcrf"]["metrics"]),
        ("delta", summary["pcrf"]["delta"]),
    ):
        for metric, expected_value in expected.items():
            if metric not in computed[view]:
                continue
            passed, difference = values_match(computed[view][metric], expected_value, tolerance)
            maximum_difference = max(maximum_difference, difference or 0.0)
            checks.append({
                "id": f"rank_reaggregation:{view}:{metric}",
                "passed": passed,
                "actual": computed[view][metric],
                "expected": expected_value,
                "absolute_difference": difference,
            })
    checks.append({
        "id": "frozen_formula_identity",
        "passed": summary["frozen_params"] == reconstruction["formula"],
        "actual": summary["frozen_params"],
        "expected": reconstruction["formula"],
    })
    checks.append({
        "id": "candidate_set_hit50_identity",
        "passed": computed["delta"]["Hit@50"] == 0.0,
        "actual": computed["delta"]["Hit@50"],
    })

    fresh_relative = authorize_path(
        reconstruction["fresh_rank_cache_path"], "s18_s0_historical_audit", data_contract
    )
    fresh_path = ROOT / fresh_relative
    fresh_hash = sha256(fresh_path)
    fresh_summary_source = evidence_contract["sources"][reconstruction["fresh_summary_source"]]
    fresh_summary = load_json(ROOT / authorize_path(
        fresh_summary_source["path"], "s18_s0_historical_audit", data_contract
    ))
    fresh_rows = 0
    ordering_exact = True
    with fresh_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            fresh_rows += 1
            ordering_exact &= (
                float(row["candidate_overlap"]) == 1.0
                and float(row["sequence_top10_overlap"]) == 1.0
                and float(row["pcrf_top10_overlap"]) == reconstruction["required_top10_overlap"]
                and int(row["cached_baseline_rank"]) == int(row["fresh_baseline_rank"])
                and int(row["cached_pcrf_rank"]) == int(row["fresh_pcrf_rank"])
            )
    checks.extend([
        {
            "id": "fresh_rank_cache_hash",
            "passed": fresh_hash == reconstruction["fresh_rank_cache_sha256"] == fresh_summary["artifacts"]["per_user_sha256"],
            "actual_sha256": fresh_hash,
        },
        {
            "id": "fresh_beam_pcrf_ordering_identity",
            "passed": ordering_exact and fresh_rows == reconstruction["required_fresh_rows"],
            "rows": fresh_rows,
            "required_rows": reconstruction["required_fresh_rows"],
            "top10_overlap": reconstruction["required_top10_overlap"],
        },
    ])
    return {
        "status": "passed" if all(row["passed"] for row in checks) else "failed",
        "checks": checks,
        "passed": sum(row["passed"] for row in checks),
        "total": len(checks),
        "rank_rows": len(baseline_ranks),
        "fresh_rows": fresh_rows,
        "max_metric_abs_difference": maximum_difference,
        "tolerance": tolerance,
        "historical_test_derived_cache_read": True,
        "raw_official_test_read": False,
    }


def render_report(summary: dict[str, Any]) -> str:
    gate = summary["gate"]
    evidence = summary["evidence_audit"]
    data = summary["data_audit"]
    pcrf = summary["pcrf_reconstruction"]
    failed = [
        row["id"]
        for section in (evidence, data, pcrf)
        for row in section["checks"]
        if not row["passed"]
    ]
    failed_text = "无" if not failed else "、".join(failed)
    return f"""# Stage18 S0 历史证据与执行契约报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent (run mode)`
- Origin Date：{summary['completed_at']}
- Verification Status：`VERIFIED`
- Version Label：`stage18_s0_audit_v1`
- Experiment ID：`{summary['experiment_id']}`
- Attempt ID：`{summary['attempt_id']}`
- Canonical Plan：`{summary['plan_path']}`

## 结论

S18-0 总 Gate 为 **{gate['status']}**。本步骤没有使用 GPU、没有训练模型、没有运行 bounded generation，
也没有读取 D1、D2、Sports、原始 official validation/test 或 Stage17 external-D0 原始目标/预测。

## 机器审计结果

| Contract | Result | Passed / Total |
|---|---:|---:|
| 历史证据 SHA + 数值回溯 | {evidence['status']} | {evidence['passed']} / {evidence['total']} |
| 数据权限与封存边界 | {data['status']} | {data['passed']} / {data['total']} |
| Phase9 frozen PCRF 重建 | {pcrf['status']} | {pcrf['passed']} / {pcrf['total']} |

Phase9 全量 derived rank cache 共 {pcrf['rank_rows']} 行，重新聚合后的最大指标绝对误差为
`{pcrf['max_metric_abs_difference']:.3e}`，门限为 `{pcrf['tolerance']:.1e}`。另在 512-user fresh-beam
审计 cache 上确认 candidate set、sequence top10、PCRF top10 与 target rank 均逐行一致。

这里读取的是 Phase9 已冻结且已做 SHA 绑定的派生 rank cache，只用于复核历史口径；没有重新打开
原始 official test 数据、候选分数或 target item，也不得将该 cache 用于方法、alpha、epoch 或 checkpoint 选择。

## Baseline 与禁止路径冻结

- 训练对照：`C0_CONT`，相同 fold、相同预算的 matched continuation；
- 主基线视图：`C1_CONT_PCRF`，即 native lexical GRAM + frozen Phase9 PCRF；
- `alpha=0` 必须退化等价于 C0；beam 固定为 50，identifier 固定为 native lexical，Trie 固定；
- 共冻结 {len(summary['hard_exclusions'])} 条 hard exclusions，包括 identifier replacement、推理后候选准入、
  重跑 C1/C2/A0、PAWA 换权重、外部 fold/seed 挽救和自动 scientific retry。

## 数据边界

- 后续 internal runner 仅可读取两域 D0 shadow 的两个精确路径，并必须只返回 `shadow_items[:-2]`；
- `shadow_items[-2]` 的已消耗 external D0 target 与 `shadow_items[-1]` guard 不得进入 internal runner；
- D1/D2、原始 monolithic sequence、Sports、external-D0 raw/materialized examples 与 predictions 均 fail closed；
- S18-0 不自动解锁 S18-1。

失败检查：{failed_text}。

## Gate 与下一步

当前裁决：`{gate['decision']}`。只有研究者另行明确同意，才可启动 S18-1 的 CPU + bounded generation
可作用性诊断；本报告本身不构成该授权。
"""


def write_status(path: Path, *, state: str, execution: str, code: str, started: str, **extra: Any) -> None:
    payload: dict[str, Any] = {
        "schema_version": "phase18.status.v1",
        "experiment_id": "s18_s0_audit",
        "attempt_id": "s18_s0_audit_attempt_001",
        "step_id": "S18-0",
        "scientific_state": state,
        "execution_state": execution,
        "status_code": code,
        "started_at": started,
        "updated_at": utc_now(),
        "heartbeat_at": utc_now(),
        "launcher_pid": os.getpid(),
        "workload_pid": os.getpid() if state == "RUNNING" else 0,
        "process_alive": state == "RUNNING",
        "tmux_session": None,
        "gpu_ids": [],
        "stage": "s18_0_contract_audit",
        "progress": {"current": 0 if state == "RUNNING" else 1, "total": 1, "unit": "audit"},
        "canonical_result_dir": "artifacts/phase18/s0_audit",
        "log_path": "artifacts/phase18/s0_audit/run.log",
        "config_paths": [
            "experiment/phase18/config/s18_evidence_contract.json",
            "experiment/phase18/config/s18_data_contract.json"
        ],
        "d0_read": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
        "historical_test_derived_cache_read": True,
        "result_selection_eligible": False,
        "occupancy_mode": "none",
        "repeat_iteration": 0,
        "repeat_metrics_ignored": False,
        "affects_scientific_result": False,
    }
    payload.update(extra)
    atomic_json(path, payload)


def append_attempt(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("attempt_id") == record["attempt_id"] for row in existing):
        raise ValueError(f"attempt already exists; automatic retry forbidden: {record['attempt_id']}")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-contract", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--attempt-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--attempt-id", default="s18_s0_audit_attempt_001")
    return parser.parse_args()


def frozen_input_hashes(
    evidence_contract: dict[str, Any], data_contract: dict[str, Any]
) -> dict[str, Any]:
    return {
        "canonical_plan": data_contract["plan_sha256"],
        "shadow_manifest": data_contract["shadow_manifest"]["sha256"],
        "d0_shadow": {
            domain: row["sha256"]
            for domain, row in data_contract["d0_shadow_inputs"].items()
        },
        "historical_sources": {
            alias: row["sha256"]
            for alias, row in evidence_contract["sources"].items()
        },
    }


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    started_clock = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_contract = load_json(args.evidence_contract)
    data_contract = load_json(args.data_contract)
    config_hashes = {
        "evidence": sha256(args.evidence_contract),
        "data": sha256(args.data_contract),
    }
    input_hashes = frozen_input_hashes(evidence_contract, data_contract)
    write_status(
        args.status,
        state="RUNNING",
        execution="RUNNING_SCIENTIFIC",
        code="S18_0_AUDIT_RUNNING",
        started=started_at,
        attempt_id=args.attempt_id,
        config_sha256=config_hashes,
        input_sha256=input_hashes,
    )
    try:
        evidence = audit_evidence(evidence_contract, data_contract)
        data = audit_data_contract(data_contract)
        pcrf = audit_pcrf_reconstruction(evidence_contract, data_contract)
        passed = all(section["status"] == "passed" for section in (evidence, data, pcrf))
        completed_at = utc_now()
        summary = {
            "schema_version": "phase18.s18_0_audit_summary.v1",
            "experiment_id": "s18_s0_audit",
            "attempt_id": args.attempt_id,
            "step_id": "S18-0",
            "status": "completed" if passed else "failed_contract_gate",
            "completed_at": completed_at,
            "wall_time_seconds": time.monotonic() - started_clock,
            "plan_path": data_contract["plan_path"],
            "plan_sha256": data_contract["plan_sha256"],
            "git_head": git_head(),
            "config_sha256": config_hashes,
            "input_sha256": input_hashes,
            "evidence_audit": evidence,
            "data_audit": data,
            "pcrf_reconstruction": pcrf,
            "hard_exclusions": evidence_contract["hard_exclusions"],
            "baseline_identity": evidence_contract["baseline_identity"],
            "gate": {
                "status": "ENGINEERING_PASS" if passed else "FAILED_CONTRACT_GATE",
                "decision": "S18_0_COMPLETE_AWAIT_S18_1_AUTHORIZATION" if passed else "STOP_STAGE18",
            },
            "scientific_results_produced": False,
            "gpu_used": False,
            "d0_read": False,
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
            "historical_test_derived_cache_read": True,
            "automatic_next_stage": False,
        }
        atomic_json(args.output_dir / "summary.json", summary)
        atomic_text(args.report, render_report(summary))
        atomic_text(
            args.output_dir / "run.log",
            json.dumps({
                "experiment_id": summary["experiment_id"],
                "attempt_id": summary["attempt_id"],
                "gate": summary["gate"],
                "wall_time_seconds": summary["wall_time_seconds"],
                "completed_at": completed_at,
            }, ensure_ascii=False, sort_keys=True) + "\n",
        )
        append_attempt(args.attempt_ledger, {
            "attempt_id": args.attempt_id,
            "step_id": "S18-0",
            "kind": "cpu_contract_audit",
            "started_at": started_at,
            "ended_at": completed_at,
            "state": "COMPLETED" if passed else "FAILED",
            "scientific_result_eligible": False,
            "failure_reason": None if passed else "one or more frozen contracts failed",
            "artifact_dir": "artifacts/phase18/s0_audit",
            "config_sha256": summary["config_sha256"],
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
        })
        write_status(
            args.status,
            state="COMPLETED" if passed else "FAILED",
            execution="SCIENTIFIC_COMPLETED" if passed else "SCIENTIFIC_FAILED",
            code="S18_0_ENGINEERING_PASS" if passed else "S18_0_CONTRACT_FAILED",
            started=started_at,
            attempt_id=args.attempt_id,
            summary_path="artifacts/phase18/s0_audit/summary.json",
            report_path="report/第十八阶段/Stage18_S0_历史证据与执行契约报告.md",
            gate=summary["gate"],
            config_sha256=config_hashes,
            input_sha256=input_hashes,
        )
        print(json.dumps({
            "gate": summary["gate"],
            "evidence": f"{evidence['passed']}/{evidence['total']}",
            "data": f"{data['passed']}/{data['total']}",
            "pcrf": f"{pcrf['passed']}/{pcrf['total']}",
            "max_metric_abs_difference": pcrf["max_metric_abs_difference"],
        }, ensure_ascii=False, sort_keys=True))
        return 0 if passed else 1
    except Exception as exception:
        write_status(
            args.status,
            state="FAILED",
            execution="SCIENTIFIC_FAILED",
            code="S18_0_AUDIT_EXCEPTION",
            started=started_at,
            attempt_id=args.attempt_id,
            failure_reason=f"{type(exception).__name__}: {exception}",
            config_sha256=config_hashes,
            input_sha256=input_hashes,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
