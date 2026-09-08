#!/usr/bin/env python3
"""Audit the S18-1 K=8 recall ceiling using frozen derived diagnostics only."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment.phase18.core.contracts import ROOT, load_json, normalize_repo_relative, sha256
from experiment.phase18.core.s1_contracts import capped_recall_metrics, evaluate_capacity_gate


CONFIG = ROOT / "experiment/phase18/config/s18_s1r_capacity_audit.json"
OUTPUT = ROOT / "artifacts/phase18/s1r_capacity_audit"
STATUS = ROOT / "artifacts/phase18/status/s18_s1r_capacity_audit.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/S18-1R.attempts.jsonl"
REPORT = ROOT / "report/第十八阶段/Stage18_S18-1R_容量校正Gate审计报告.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def append_ledger(payload: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verified_path(source: dict[str, str]) -> Path:
    relative = normalize_repo_relative(source["path"])
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    actual = sha256(path)
    if actual != source["sha256"]:
        raise ValueError(
            f"source SHA mismatch for {relative}: expected {source['sha256']}, got {actual}"
        )
    return path


def load_events(path: Path, k: int) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            actual = set(row["actual_pruner_items"])
            selected = set(row["hard_negative_items"])
            denominator = len(actual)
            intersection = len(actual & selected)
            if denominator != int(row["hard_negative_actual_denominator"]):
                raise ValueError(f"denominator mismatch at {path}:{line_number}")
            if intersection != int(row["hard_negative_intersection"]):
                raise ValueError(f"intersection mismatch at {path}:{line_number}")
            if len(selected) > k:
                raise ValueError(f"more than K selected negatives at {path}:{line_number}")
            if denominator:
                events.append(
                    {
                        "intersection": intersection,
                        "actual": denominator,
                        "first_drop_depth": int(row["first_drop_depth"]),
                    }
                )
    if not events:
        raise ValueError(f"no nonempty actual-pruner events in {path}")
    return events


def summarize(events: list[dict[str, Any]], k: int) -> dict[str, Any]:
    metrics = capped_recall_metrics(
        ((row["intersection"], row["actual"]) for row in events), k
    )
    by_depth: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in events:
        by_depth[str(row["first_drop_depth"])].append(
            (row["intersection"], row["actual"])
        )
    metrics["by_first_drop_depth"] = {
        depth: capped_recall_metrics(rows, k)
        for depth, rows in sorted(by_depth.items(), key=lambda item: int(item[0]))
    }
    return metrics


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage18 S18-1R 容量校正 Gate 审计报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill：`academic-research-suite / experiment-agent`",
        "- Origin Mode：`validate + plan`",
        f"- Origin Date：{summary['completed_at']}",
        "- Verification Status：`ANALYZED`",
        "- Version Label：`s18_s1r_capacity_audit_v1`",
        "- Protected Data：未读取 I1/I2、D1/D2、official validation/test 或 Sports",
        "",
        "## 结论",
        "",
        f"回顾性容量校正裁决：`{summary['decision']}`。该裁决不覆盖 v0.2 原始 Gate，",
        "也不直接解锁 S18-2；必须先在预先冻结的未见 cohort 上确认。",
        "",
        "## Domain 结果",
        "",
        "| Domain | raw micro | mechanical ceiling | fraction of capacity | event any-hit | macro event recall | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for domain in ("Toys", "Beauty"):
        row = summary["domains"][domain]
        metric = row["metrics"]
        lines.append(
            f"| {domain} | {metric['raw_micro_recall']:.6f} | "
            f"{metric['raw_micro_mechanical_ceiling']:.6f} | "
            f"{metric['capacity_normalized_recall']:.6f} | "
            f"{metric['event_any_hit_rate']:.6f} | "
            f"{metric['macro_event_recall']:.6f} | `{row['gate']['decision']}` |"
        )
    beauty = summary["domains"]["Beauty"]["metrics"]
    depth_one = beauty["by_first_drop_depth"]["1"]
    lines.extend(
        [
            "",
            "## Beauty denominator 诊断",
            "",
            f"- Beauty 共 `{beauty['event_count']}` 个非空事件、`{beauty['actual_pruner_total']}` 个 actual-pruner items。",
            f"- K=8 最多覆盖 `{beauty['topk_capacity_total']}` 个，raw micro 的机械上限只有 "
            f"`{beauty['raw_micro_mechanical_ceiling']:.6f}`。",
            f"- 实际覆盖 `{beauty['intersection_total']}` 个，即达到可覆盖容量的 "
            f"`{beauty['capacity_normalized_recall']:.6f}`。",
            f"- depth-1 有 `{depth_one['event_count']}` 个事件、`{depth_one['actual_pruner_total']}` 个 items；"
            f"其 raw recall `{depth_one['raw_micro_recall']:.6f}`，机械上限 "
            f"`{depth_one['raw_micro_mechanical_ceiling']:.6f}`。",
            "",
            "## 下一强制 Gate",
            "",
            "按冻结 hash 顺序选择每域排名 `[1024,2048)` 的 1,024 名未见用户，在 I-1/I0 上复核同一容量校正 Gate。",
            "该 confirmation 通过前，不训练 treatment、不读取 I1/I2，也不启动 S18-2。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if OUTPUT.exists() or STATUS.exists() or LEDGER.exists() or REPORT.exists():
        raise FileExistsError("S18-1R capacity audit artifacts already exist; automatic rerun forbidden")
    config = load_json(CONFIG)
    k = int(config["hard_negative_k"])
    canonical = load_json(verified_path(config["sources"]["canonical_summary"]))
    append_ledger(
        {
            "at": utc_now(),
            "attempt_id": "audit-0001",
            "event": "retrospective_capacity_audit_started",
            "automatic_s18_2": False,
        }
    )

    fold_events: dict[str, list[dict[str, Any]]] = {}
    folds: dict[str, Any] = {}
    for unit, source in config["sources"]["folds"].items():
        path = verified_path(source)
        events = load_events(path, k)
        fold_events[unit] = events
        folds[unit] = {
            "source_path": str(path.relative_to(ROOT)),
            "source_sha256": source["sha256"],
            "metrics": summarize(events, k),
        }

    domains = {}
    for domain in ("Toys", "Beauty"):
        events = [
            row
            for fold in ("I-1", "I0")
            for row in fold_events[f"{domain}:{fold}"]
        ]
        metrics = summarize(events, k)
        original = canonical["domains"][domain]
        if abs(metrics["raw_micro_recall"] - original["metrics"]["k8_actual_pruner_recall"]) > 1e-12:
            raise ValueError(f"{domain}: recomputed raw recall does not match canonical S18-1")
        inherited_checks = {
            key: bool(value)
            for key, value in original["gate"]["checks"].items()
            if key != "gate4_k8_actual_pruner_recall"
        }
        repaired = evaluate_capacity_gate(metrics, config["repaired_gate"])
        domains[domain] = {
            "metrics": metrics,
            "inherited_non_gate4_checks": inherited_checks,
            "gate": {
                **repaired,
                "all_inherited_checks_pass": all(inherited_checks.values()),
            },
        }

    passed = all(
        row["gate"]["decision"] == "CAPACITY_ADJUSTED_ACTIONABILITY_PASS"
        and row["gate"]["all_inherited_checks_pass"]
        for row in domains.values()
    )
    decision = (
        "S18_1R_RETROSPECTIVE_CAPACITY_PASS_REQUIRES_DISJOINT_CONFIRMATION"
        if passed
        else "S18_1R_CAPACITY_GATE_FAILED"
    )
    completed = utc_now()
    summary = {
        "schema_version": "phase18.s18_1r_capacity_audit_summary.v1",
        "experiment_id": "s18_s1r_capacity_audit",
        "attempt_id": "audit-0001",
        "status": "COMPLETED",
        "decision": decision,
        "retrospective": True,
        "hard_negative_k": k,
        "config_sha256": sha256(CONFIG),
        "canonical_s18_1_summary_sha256": config["sources"]["canonical_summary"]["sha256"],
        "domains": domains,
        "folds": folds,
        "disjoint_confirmation": config["disjoint_confirmation"],
        "automatic_s18_2": False,
        "treatment_training": False,
        "i1_i2_read": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
        "completed_at": completed,
    }
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT / "summary.json", summary)
    atomic_text(REPORT, render_report(summary))
    status = {
        "schema_version": "phase18.status.v1",
        "experiment_id": "s18_s1r_capacity_audit",
        "attempt_id": "audit-0001",
        "scientific_state": "COMPLETED",
        "execution_state": "SCIENTIFIC_COMPLETED",
        "status_code": decision,
        "stage": "retrospective_capacity_audit_complete",
        "process_alive": False,
        "launcher_pid": os.getpid(),
        "workload_pid": 0,
        "tmux_session": None,
        "progress": {"current": 4, "total": 4, "unit": "domain_fold_derived_diagnostic"},
        "summary_path": str((OUTPUT / "summary.json").relative_to(ROOT)),
        "report_path": str(REPORT.relative_to(ROOT)),
        "automatic_s18_2": False,
        "next_action": "Freeze and run the disjoint cohort confirmation; do not start S18-2 automatically.",
        "heartbeat_at": completed,
        "updated_at": completed,
    }
    atomic_json(STATUS, status)
    append_ledger(
        {
            "at": completed,
            "attempt_id": "audit-0001",
            "event": "retrospective_capacity_audit_completed",
            "decision": decision,
            "summary_sha256": sha256(OUTPUT / "summary.json"),
        }
    )
    print(json.dumps({"decision": decision, "summary": status["summary_path"]}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
