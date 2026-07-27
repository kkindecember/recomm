#!/usr/bin/env python3
"""CPU-only HBTR-v2 F0 quantile-tail margin distinguishability probe.

The single formula and gates are locked in
artifacts/phase3/configs/hbtr_v2_f0_preregistered.json. This script reads only
existing training-only caches, sequence[:-2] popularity, the preregistration,
and the upstream autopsy decision. It does not load validation effect data,
test data, checkpoints, or models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = ROOT / "artifacts/phase3/hbtr_pilot"
OUTPUT_ROOT = ROOT / "artifacts/phase3/hbtr_v2_f0"
CONFIG_PATH = ROOT / "artifacts/phase3/configs/hbtr_v2_f0_preregistered.json"
AUTOPSY_SUMMARY_PATH = ROOT / "artifacts/phase3/hbtr_v2_autopsy/summary.json"
REPORT_PATH = ROOT / "report/第三阶段/GRAM_第三阶段_HBTR_v2_F0可辨识性报告.md"
DATASETS = ("Toys", "Beauty")
BASE_MARGIN = 0.1
PREFIX_CAP = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_training_popularity(path: Path) -> Counter:
    popularity: Counter = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            tokens = line.split()
            if len(tokens) < 4:
                raise ValueError(f"sequence too short at {path}:{line_number}")
            popularity.update(tokens[1:-2])
    if not popularity:
        raise ValueError(f"empty training-only popularity at {path}")
    return popularity


def build_tail_quantiles(popularity: Counter) -> tuple[dict[str, float], int]:
    ordered = sorted(popularity, key=lambda item: (-popularity[item], item))
    item_count = len(ordered)
    head_count = math.ceil(item_count * 0.20)
    tail_count = item_count - head_count
    if not 0 < head_count < item_count:
        raise ValueError("head/tail split requires at least two popularity groups")
    quantiles = {}
    for rank, item in enumerate(ordered, start=1):
        quantiles[item] = (
            0.0 if rank <= head_count else (rank - head_count) / tail_count
        )
    return quantiles, head_count


def prefix_weight(depth: int) -> float:
    if depth < 0:
        raise ValueError("prefix depth must be non-negative")
    return 1.0 + min(depth, PREFIX_CAP) / PREFIX_CAP


def tail_weight(quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("tail quantile must be in [0,1]")
    return 1.0 + quantile


def margins(prefix_depth: int, quantile: float) -> dict[str, float]:
    prefix = prefix_weight(prefix_depth)
    tail = tail_weight(quantile)
    return {
        "C1": BASE_MARGIN,
        "C2": BASE_MARGIN * prefix,
        "C3_v2": BASE_MARGIN * tail,
        "C4_v2": BASE_MARGIN * prefix * tail,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty list")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "max": max(values),
    }


def analyze_dataset(
    cache: dict,
    popularity: Counter,
    gates: dict,
) -> tuple[dict, list[dict]]:
    quantiles, head_count = build_tail_quantiles(popularity)
    item_count = len(quantiles)
    rows = cache["rows"]
    if not rows:
        raise ValueError("negative cache is empty")

    values = {control: [] for control in ("C1", "C2", "C3_v2", "C4_v2")}
    tail_rows = 0
    joint_rows = 0
    frequency_mismatches = []
    for row in rows:
        positive = row["positive_item"]
        if positive not in quantiles:
            raise ValueError(f"cache positive absent from training popularity: {positive}")
        if int(row["positive_frequency"]) != int(popularity[positive]):
            frequency_mismatches.append(positive)
        quantile = quantiles[positive]
        is_tail = quantile > 0.0
        has_prefix = any(int(depth) > 0 for depth in row["prefix_depths"])
        tail_rows += is_tail
        joint_rows += is_tail and has_prefix
        for depth in row["prefix_depths"]:
            for control, value in margins(int(depth), quantile).items():
                values[control].append(value)
    if frequency_mismatches:
        raise ValueError(
            f"cache/training popularity mismatch examples={frequency_mismatches[:3]}"
        )

    differences = [
        abs(c4 - c2) for c4, c2 in zip(values["C4_v2"], values["C2"])
    ]
    equality_rate = sum(value <= 1e-12 for value in differences) / len(differences)
    ordered_quantiles = [
        quantiles[item]
        for item in sorted(popularity, key=lambda item: (-popularity[item], item))
    ]
    head_alignment = all(value == 0.0 for value in ordered_quantiles[:head_count])
    tail_alignment = all(value > 0.0 for value in ordered_quantiles[head_count:])
    monotonic = all(
        left <= right
        for left, right in zip(ordered_quantiles, ordered_quantiles[1:])
    )
    rarest_weight = tail_weight(ordered_quantiles[-1])
    max_margin = max(max(control_values) for control_values in values.values())

    metrics = {
        "item_count": item_count,
        "head_item_count": head_count,
        "tail_item_count": item_count - head_count,
        "cache_rows": len(rows),
        "cache_pairs": len(values["C1"]),
        "tail_nontrivial_rows": tail_rows,
        "tail_nontrivial_row_rate": tail_rows / len(rows),
        "joint_nontrivial_rows": joint_rows,
        "joint_nontrivial_row_rate": joint_rows / len(rows),
        "C4_v2_vs_C2_pair_margin_exact_equality_rate": equality_rate,
        "head_weight_exactly_one": head_alignment,
        "tail_weight_strictly_above_one": tail_alignment,
        "tail_weight_monotonic": monotonic,
        "rarest_item_tail_weight": rarest_weight,
        "maximum_observed_margin": max_margin,
        "cache_frequency_mismatches": 0,
    }
    gate_results = {
        "tail_nontrivial_row_rate": (
            metrics["tail_nontrivial_row_rate"]
            >= gates["tail_nontrivial_row_rate_min_each_dataset"]
        ),
        "joint_nontrivial_row_rate": (
            metrics["joint_nontrivial_row_rate"]
            >= gates["joint_nontrivial_row_rate_min_each_dataset"]
        ),
        "C4_v2_vs_C2_pair_margin_exact_equality_rate": (
            equality_rate
            <= gates[
                "C4_v2_vs_C2_pair_margin_exact_equality_rate_max_each_dataset"
            ]
        ),
        "head_tail_alignment": head_alignment and tail_alignment,
        "tail_weight_monotonic": monotonic,
        "tail_weight_cap": rarest_weight <= gates["tail_weight_max"] + 1e-12,
        "rarest_item_reaches_cap": abs(
            rarest_weight - gates["tail_weight_max"]
        )
        <= 1e-12,
        "margin_cap": max_margin <= gates["margin_max"] + 1e-12,
        "cache_frequency_lineage": not frequency_mismatches,
    }
    distributions = [
        {"control": control, **describe(control_values)}
        for control, control_values in values.items()
    ]
    return (
        {
            "metrics": metrics,
            "margin_distributions": {
                row["control"]: {key: value for key, value in row.items() if key != "control"}
                for row in distributions
            },
            "gates": {**gate_results, "passed": all(gate_results.values())},
            "validation_effect_data_read": False,
            "test_data_read": False,
        },
        distributions,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: dict) -> str:
    result_rows = []
    for dataset in DATASETS:
        metrics = summary["datasets"][dataset]["metrics"]
        result_rows.append(
            "| {dataset} | {tail:.2%} | {joint:.2%} | {equal:.2%} | "
            "{max_weight:.3f} | {max_margin:.3f} | {gate} |".format(
                dataset=dataset,
                tail=metrics["tail_nontrivial_row_rate"],
                joint=metrics["joint_nontrivial_row_rate"],
                equal=metrics["C4_v2_vs_C2_pair_margin_exact_equality_rate"],
                max_weight=metrics["rarest_item_tail_weight"],
                max_margin=metrics["maximum_observed_margin"],
                gate="PASS" if summary["datasets"][dataset]["gates"]["passed"] else "FAIL",
            )
        )
    return "\n".join(
        [
            "# GRAM 第三阶段 HBTR-v2 F0 可辨识性报告",
            "",
            "## Material Passport",
            "",
            "- Origin Skill: academic-research-suite / experiment-agent",
            "- Origin Mode: run",
            "- Origin Date: 2026-07-24",
            "- Verification Status: ANALYZED",
            "- Version Label: hbtr_v2_f0_v1",
            "- Design Status: RESULT-INFORMED NEW CYCLE",
            "",
            "## 结论",
            "",
            f"- F0 决策：**{summary['decision']}**",
            "- 本结果不改变 HBTR-v1 STOP，也不解锁 GPU 或效果实验。",
            "",
            "## 核心结果",
            "",
            "| 数据集 | tail非平凡/有效行 | joint非平凡/有效行 | C4-v2=C2 pair率 | "
            "最大tail权重 | 最大margin | gate |",
            "|---|---:|---:|---:|---:|---:|---|",
            *result_rows,
            "",
            "## 边界",
            "",
            "F0 只读取 training-only cache 与 sequence[:-2] popularity；未读取 validation",
            "效果、test、checkpoint 或模型。PASS 只允许设计数值等价的 negative-decoder",
            "micro-batching，不证明推荐效果。",
            "",
        ]
    )


def main() -> int:
    config = load_json(CONFIG_PATH)
    upstream = load_json(AUTOPSY_SUMMARY_PATH)
    if upstream.get("decision") != "V2_DESIGN_ALLOWED":
        raise ValueError("HBTR-v2 F0 requires upstream V2_DESIGN_ALLOWED")
    gates = config["gates"]
    results = {}
    activation_rows = []
    distribution_rows = []
    input_hashes = {
        "preregistration": sha256(CONFIG_PATH),
        "upstream_autopsy_summary": sha256(AUTOPSY_SUMMARY_PATH),
    }
    for dataset in DATASETS:
        cache_path = PILOT_ROOT / dataset / "cache/negative_cache.json"
        sequence_path = ROOT / f"GRAM/rec_datasets/{dataset}/user_sequence.txt"
        result, distributions = analyze_dataset(
            load_json(cache_path),
            read_training_popularity(sequence_path),
            gates,
        )
        results[dataset] = result
        activation_rows.append({"dataset": dataset, **result["metrics"]})
        distribution_rows.extend(
            {"dataset": dataset, **row} for row in distributions
        )
        input_hashes[f"{dataset}_negative_cache"] = sha256(cache_path)
        input_hashes[f"{dataset}_user_sequence"] = sha256(sequence_path)
        dataset_output = OUTPUT_ROOT / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        with (dataset_output / "diagnostic_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    decision = (
        "PASS_FOR_RESOURCE_REPAIR_DESIGN"
        if all(results[dataset]["gates"]["passed"] for dataset in DATASETS)
        else "STOP_HBTR"
    )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-24",
            "verification_status": "ANALYZED",
            "version_label": "hbtr_v2_f0_v1",
            "design_status": "RESULT_INFORMED_NEW_CYCLE",
        },
        "decision": decision,
        "gpu_unlocked": False,
        "hbtr_v1_decision_unchanged": "STOP",
        "datasets": results,
        "input_sha256": dict(sorted(input_hashes.items())),
        "validation_effect_data_read": False,
        "test_data_read": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "activation_metrics.csv", activation_rows)
    write_csv(OUTPUT_ROOT / "margin_distributions.csv", distribution_rows)
    with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "decision": decision,
                "dataset_gates": {
                    dataset: results[dataset]["gates"]["passed"]
                    for dataset in DATASETS
                },
                "output": str((OUTPUT_ROOT / "summary.json").relative_to(ROOT)),
                "report": str(REPORT_PATH.relative_to(ROOT)),
                "gpu_unlocked": False,
                "validation_effect_data_read": False,
                "test_data_read": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
