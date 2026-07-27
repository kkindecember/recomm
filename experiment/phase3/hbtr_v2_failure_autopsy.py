#!/usr/bin/env python3
"""Deterministic, CPU-only HBTR-v2 failure autopsy.

This result-informed diagnostic reads only existing HBTR-v1 training caches,
validation per-user outputs, summaries, preregistration, and training-only
popularity derived from sequence[:-2]. It never loads checkpoints or test
predictions and does not train or regenerate beams.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = ROOT / "artifacts/phase3/hbtr_pilot"
OUTPUT_ROOT = ROOT / "artifacts/phase3/hbtr_v2_autopsy"
CONFIG_PATH = ROOT / "artifacts/phase3/configs/hbtr_v2_autopsy_preregistered.json"
PILOT_CONFIG_PATH = ROOT / "artifacts/phase3/configs/hbtr_pilot_preregistered.json"
PILOT_SUMMARY_PATH = PILOT_ROOT / "summary.json"
REPORT_PATH = ROOT / "report/第三阶段/GRAM_第三阶段_HBTR_v2失效解剖报告.md"
DATASETS = ("Toys", "Beauty")
CONTROLS = ("C0", "C1", "C2", "C3", "C4")
COMPARISONS = (("C0", "C1"), ("C0", "C4"), ("C1", "C4"))
BASE_MARGIN = 0.1
PREFIX_DEPTH_CAP = 3


def prefix_weight(depth: int) -> float:
    if depth < 0:
        raise ValueError("prefix depth must be non-negative")
    return 1.0 + min(depth, PREFIX_DEPTH_CAP) / PREFIX_DEPTH_CAP


def tail_weight(positive_frequency: int, median_frequency: float) -> float:
    if positive_frequency < 0 or median_frequency < 0:
        raise ValueError("training frequencies must be non-negative")
    log_ratio = math.log(
        (float(median_frequency) + 1.0) / (positive_frequency + 1.0)
    )
    return 1.0 + min(1.0, max(0.0, log_ratio))


def component_margin(
    control: str,
    prefix_depth: int,
    positive_frequency: int,
    median_frequency: float,
) -> float:
    """Pure-standard-library copy of the locked HBTR-v1 margin formula."""
    if control == "C1":
        return BASE_MARGIN
    if control == "C2":
        return BASE_MARGIN * prefix_weight(prefix_depth)
    if control == "C3":
        return BASE_MARGIN * tail_weight(positive_frequency, median_frequency)
    if control == "C4":
        return (
            BASE_MARGIN
            * prefix_weight(prefix_depth)
            * tail_weight(positive_frequency, median_frequency)
        )
    raise ValueError(f"ranking margin is undefined for control {control!r}")


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


def read_validation_rows(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        user = row["user_id"]
        if user in result:
            raise ValueError(f"duplicate validation user {user} in {path}")
        result[user] = row
    return result


def parsed_rank(row: dict) -> int:
    return int(row["rank"]) if row["rank"] else 51


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
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


def activation_analysis(cache: dict, median_frequency: float) -> tuple[dict, dict]:
    rows = cache["rows"]
    sample_count = int(cache["samples"])
    if not rows or sample_count < len(rows):
        raise ValueError("invalid cache sample/row counts")
    pair_count = sum(len(row["prefix_depths"]) for row in rows)
    prefix_pairs = sum(
        depth > 0 for row in rows for depth in row["prefix_depths"]
    )
    prefix_rows = sum(any(depth > 0 for depth in row["prefix_depths"]) for row in rows)
    tail_rows = sum(row["positive_frequency"] < median_frequency for row in rows)
    joint_rows = sum(
        row["positive_frequency"] < median_frequency
        and any(depth > 0 for depth in row["prefix_depths"])
        for row in rows
    )

    margins = {control: [] for control in CONTROLS[1:]}
    for row in rows:
        for depth in row["prefix_depths"]:
            for control in margins:
                margins[control].append(
                    component_margin(
                        control,
                        int(depth),
                        int(row["positive_frequency"]),
                        median_frequency,
                    )
                )

    margin_summary = {control: describe(values) for control, values in margins.items()}
    for left, right in (("C1", "C2"), ("C1", "C3"), ("C1", "C4"),
                        ("C2", "C4"), ("C3", "C4")):
        differences = [abs(a - b) for a, b in zip(margins[left], margins[right])]
        margin_summary[f"{left}_vs_{right}"] = {
            "equal_rate": sum(value <= 1e-12 for value in differences)
            / len(differences),
            "mean_absolute_difference": statistics.fmean(differences),
            "max_absolute_difference": max(differences),
        }

    metrics = {
        "training_samples": sample_count,
        "eligible_rows": len(rows),
        "eligible_all_rate": len(rows) / sample_count,
        "pair_count": pair_count,
        "prefix_nontrivial_pairs": prefix_pairs,
        "prefix_nontrivial_pair_rate": prefix_pairs / pair_count,
        "prefix_nontrivial_rows": prefix_rows,
        "prefix_nontrivial_row_rate": prefix_rows / len(rows),
        "prefix_nontrivial_all_rate": prefix_rows / sample_count,
        "tail_nontrivial_rows": tail_rows,
        "tail_nontrivial_row_rate": tail_rows / len(rows),
        "tail_nontrivial_all_rate": tail_rows / sample_count,
        "joint_nontrivial_rows": joint_rows,
        "joint_nontrivial_row_rate": joint_rows / len(rows),
        "joint_nontrivial_all_rate": joint_rows / sample_count,
        "training_popularity_median": median_frequency,
    }
    return metrics, margin_summary


def validate_lineage(rows_by_control: dict[str, dict[str, dict]]) -> dict:
    reference_users = set(rows_by_control["C0"])
    if not reference_users:
        raise ValueError("empty validation rows")
    for control, rows in rows_by_control.items():
        if set(rows) != reference_users:
            raise ValueError(f"validation user mismatch for {control}")
        for user in reference_users:
            reference = rows_by_control["C0"][user]
            candidate = rows[user]
            for field in ("target_item", "target_group", "history_bin"):
                if candidate[field] != reference[field]:
                    raise ValueError(
                        f"validation lineage mismatch control={control} "
                        f"user={user} field={field}"
                    )
    return {"users": len(reference_users), "mismatches": 0}


def transition_groups(row: dict) -> Iterable[str]:
    yield "overall"
    yield row["target_group"]
    yield f"history_{row['history_bin']}"


def transition_summary(
    dataset: str,
    baseline: str,
    candidate: str,
    rows_by_control: dict[str, dict[str, dict]],
) -> list[dict]:
    accumulators: dict[str, dict[str, float]] = {}
    for user, base_row in rows_by_control[baseline].items():
        candidate_row = rows_by_control[candidate][user]
        rank0 = parsed_rank(base_row)
        rank1 = parsed_rank(candidate_row)
        for group in transition_groups(base_row):
            counts = accumulators.setdefault(
                group,
                {
                    "n": 0,
                    "rank_changed": 0,
                    "rank_improved": 0,
                    "rank_worsened": 0,
                    "promoted_to_top10": 0,
                    "demoted_from_top10": 0,
                    "ndcg10_difference_sum": 0.0,
                    "recall10_difference_sum": 0.0,
                },
            )
            counts["n"] += 1
            counts["rank_changed"] += rank0 != rank1
            counts["rank_improved"] += rank1 < rank0
            counts["rank_worsened"] += rank1 > rank0
            counts["promoted_to_top10"] += rank0 > 10 and rank1 <= 10
            counts["demoted_from_top10"] += rank0 <= 10 and rank1 > 10
            counts["ndcg10_difference_sum"] += (
                float(candidate_row["NDCG@10"]) - float(base_row["NDCG@10"])
            )
            counts["recall10_difference_sum"] += (
                float(candidate_row["Recall@10"]) - float(base_row["Recall@10"])
            )

    output = []
    for group, counts in sorted(accumulators.items()):
        n = int(counts["n"])
        output.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "candidate": candidate,
                "group": group,
                "n": n,
                "rank_changed": int(counts["rank_changed"]),
                "rank_improved": int(counts["rank_improved"]),
                "rank_worsened": int(counts["rank_worsened"]),
                "promoted_to_top10": int(counts["promoted_to_top10"]),
                "demoted_from_top10": int(counts["demoted_from_top10"]),
                "net_top10_promotions": int(
                    counts["promoted_to_top10"] - counts["demoted_from_top10"]
                ),
                "mean_ndcg10_difference": counts["ndcg10_difference_sum"] / n,
                "mean_recall10_difference": counts["recall10_difference_sum"] / n,
            }
        )
    return output


def overall_transition(
    transitions: list[dict], dataset: str, baseline: str, candidate: str
) -> dict:
    matches = [
        row
        for row in transitions
        if row["dataset"] == dataset
        and row["baseline"] == baseline
        and row["candidate"] == candidate
        and row["group"] == "overall"
    ]
    if len(matches) != 1:
        raise ValueError("missing or duplicate overall transition")
    return matches[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: dict) -> str:
    dataset_rows = []
    for dataset in DATASETS:
        result = summary["datasets"][dataset]
        activation = result["activation"]
        c4 = result["transitions"]["C0_to_C4"]
        dataset_rows.append(
            "| {dataset} | {eligible:.2%} | {prefix:.2%} | {tail:.2%} | "
            "{joint:.2%} | {ndcg:+.3%} | {net:+d} |".format(
                dataset=dataset,
                eligible=activation["eligible_all_rate"],
                prefix=activation["prefix_nontrivial_row_rate"],
                tail=activation["tail_nontrivial_row_rate"],
                joint=activation["joint_nontrivial_row_rate"],
                ndcg=result["pilot_effects"]["C4_vs_C0_ndcg10_relative"],
                net=c4["net_top10_promotions"],
            )
        )
    gates = summary["gates"]
    return "\n".join(
        [
            "# GRAM 第三阶段 HBTR-v2 Failure Autopsy 报告",
            "",
            "## Material Passport",
            "",
            "- Origin Skill: academic-research-suite / experiment-agent",
            "- Origin Mode: run",
            "- Origin Date: 2026-07-24",
            "- Verification Status: ANALYZED",
            "- Version Label: hbtr_v2_autopsy_v1",
            "- Design Status: RESULT-INFORMED POST-HOC EXPLORATORY",
            "",
            "## 结论",
            "",
            f"- 诊断决策：**{summary['decision']}**",
            "- HBTR-v1 保持 STOP；本报告不解锁 GPU、25%、全量、更多 seed 或 test。",
            "",
            "## 核心结果",
            "",
            "| 数据集 | eligible/all | prefix非平凡/有效行 | tail非平凡/有效行 | "
            "joint非平凡/有效行 | C4 vs C0 NDCG@10 | C4 top-10净迁入 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *dataset_rows,
            "",
            "## 锁定诊断门槛",
            "",
            f"- [{'PASS' if gates['eligible_support_both'] else 'FAIL'}] "
            "两数据集 eligible/all ≥15%。",
            f"- [{'PASS' if gates['prefix_support_both'] else 'FAIL'}] "
            "两数据集 prefix 非平凡行 ≥25%。",
            f"- [{'PASS' if gates['tail_distinguishability_both'] else 'FAIL'}] "
            "两数据集 tail 非平凡行 ≥20%。",
            f"- [{'PASS' if gates['joint_distinguishability_both'] else 'FAIL'}] "
            "两数据集 joint 非平凡行 ≥10%。",
            f"- [{'PASS' if gates['generic_signal'] else 'FAIL'}] "
            "至少一个数据集在同一 C1/C4 对照中同时具有正 NDCG@10 差和正 top-10 净迁入。",
            "",
            "## 解释边界",
            "",
            "该诊断在看到 HBTR-v1 pilot 后建立，只判断联合机制是否缺乏可辨识激活以及",
            "是否允许设计独立 HBTR-v2。阈值不是效果门槛，不能把 HBTR-v1 STOP 改判为",
            "MODIFY/GO；所有指标均来自既有 training-only cache 与 validation 结果，未读取 test。",
            "",
        ]
    )


def main() -> int:
    config = load_json(CONFIG_PATH)
    pilot_config = load_json(PILOT_CONFIG_PATH)
    pilot_summary = load_json(PILOT_SUMMARY_PATH)
    if pilot_summary["decision"] != "STOP":
        raise ValueError("failure autopsy requires the frozen HBTR-v1 STOP decision")

    thresholds = config["activation_thresholds"]
    input_hashes = {
        "autopsy_preregistration": sha256(CONFIG_PATH),
        "pilot_preregistration": sha256(PILOT_CONFIG_PATH),
        "pilot_summary": sha256(PILOT_SUMMARY_PATH),
    }
    results = {}
    activation_rows = []
    transition_rows = []

    for dataset in DATASETS:
        dataset_root = PILOT_ROOT / dataset
        cache_path = dataset_root / "cache/negative_cache.json"
        sequence_path = ROOT / f"GRAM/rec_datasets/{dataset}/user_sequence.txt"
        cache = load_json(cache_path)
        popularity = read_training_popularity(sequence_path)
        median_frequency = float(statistics.median(popularity.values()))
        activation, margins = activation_analysis(cache, median_frequency)

        rows_by_control = {}
        for control in CONTROLS:
            validation_path = dataset_root / control / "validation_per_user.csv"
            training_path = dataset_root / control / "training_summary.json"
            rows_by_control[control] = read_validation_rows(validation_path)
            load_json(training_path)
            input_hashes[f"{dataset}_{control}_validation_per_user"] = sha256(
                validation_path
            )
            input_hashes[f"{dataset}_{control}_training_summary"] = sha256(
                training_path
            )
        lineage = validate_lineage(rows_by_control)
        input_hashes[f"{dataset}_negative_cache"] = sha256(cache_path)
        input_hashes[f"{dataset}_user_sequence"] = sha256(sequence_path)

        dataset_transitions = []
        for baseline, candidate in COMPARISONS:
            dataset_transitions.extend(
                transition_summary(
                    dataset, baseline, candidate, rows_by_control
                )
            )
        transition_rows.extend(dataset_transitions)

        c0_ndcg = pilot_summary["metrics"][dataset]["C0"]["groups"]["overall"][
            "NDCG@10"
        ]
        pilot_effects = {}
        for control in ("C1", "C4"):
            ndcg = pilot_summary["metrics"][dataset][control]["groups"]["overall"][
                "NDCG@10"
            ]
            pilot_effects[f"{control}_vs_C0_ndcg10_absolute"] = ndcg - c0_ndcg
            pilot_effects[f"{control}_vs_C0_ndcg10_relative"] = ndcg / c0_ndcg - 1.0

        results[dataset] = {
            "activation": activation,
            "margin_distributions": margins,
            "lineage": lineage,
            "pilot_effects": pilot_effects,
            "transitions": {
                "C0_to_C1": overall_transition(
                    dataset_transitions, dataset, "C0", "C1"
                ),
                "C0_to_C4": overall_transition(
                    dataset_transitions, dataset, "C0", "C4"
                ),
                "C1_to_C4": overall_transition(
                    dataset_transitions, dataset, "C1", "C4"
                ),
            },
            "test_data_read": False,
        }
        activation_rows.append({"dataset": dataset, **activation})
        dataset_output = OUTPUT_ROOT / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        with (dataset_output / "diagnostic_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(results[dataset], handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    eligible_support = all(
        results[dataset]["activation"]["eligible_all_rate"]
        >= thresholds["eligible_all_rate_min_each_dataset"]
        for dataset in DATASETS
    )
    prefix_support = all(
        results[dataset]["activation"]["prefix_nontrivial_row_rate"]
        >= thresholds["prefix_nontrivial_row_rate_min_each_dataset"]
        for dataset in DATASETS
    )
    tail_distinguishability = all(
        results[dataset]["activation"]["tail_nontrivial_row_rate"]
        >= thresholds["tail_nontrivial_row_rate_min_each_dataset"]
        for dataset in DATASETS
    )
    joint_distinguishability = all(
        results[dataset]["activation"]["joint_nontrivial_row_rate"]
        >= thresholds["joint_nontrivial_row_rate_min_each_dataset"]
        for dataset in DATASETS
    )
    generic_signal_by_dataset = {}
    for dataset in DATASETS:
        generic_signal_by_dataset[dataset] = any(
            results[dataset]["pilot_effects"][
                f"{control}_vs_C0_ndcg10_absolute"
            ]
            > 0
            and results[dataset]["transitions"][f"C0_to_{control}"][
                "net_top10_promotions"
            ]
            > 0
            for control in ("C1", "C4")
        )
    generic_signal = any(generic_signal_by_dataset.values())
    underactivated = not (tail_distinguishability and joint_distinguishability)
    decision = (
        "V2_DESIGN_ALLOWED"
        if eligible_support and prefix_support and underactivated and generic_signal
        else "STOP_HBTR"
    )
    gates = {
        "eligible_support_both": eligible_support,
        "prefix_support_both": prefix_support,
        "tail_distinguishability_both": tail_distinguishability,
        "joint_distinguishability_both": joint_distinguishability,
        "underactivated": underactivated,
        "generic_signal_by_dataset": generic_signal_by_dataset,
        "generic_signal": generic_signal,
    }
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-24",
            "verification_status": "ANALYZED",
            "version_label": "hbtr_v2_autopsy_v1",
            "design_status": "RESULT_INFORMED_POST_HOC_EXPLORATORY",
        },
        "decision": decision,
        "gpu_unlocked": False,
        "hbtr_v1_decision_unchanged": "STOP",
        "gates": gates,
        "datasets": results,
        "input_sha256": dict(sorted(input_hashes.items())),
        "pilot_config_version": pilot_config["material_passport"]["version_label"],
        "test_data_read": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "activation_metrics.csv", activation_rows)
    write_csv(OUTPUT_ROOT / "rank_transitions.csv", transition_rows)
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
                "gates": gates,
                "output": str((OUTPUT_ROOT / "summary.json").relative_to(ROOT)),
                "report": str(REPORT_PATH.relative_to(ROOT)),
                "test_data_read": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
