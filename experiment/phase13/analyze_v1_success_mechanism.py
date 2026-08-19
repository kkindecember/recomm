#!/usr/bin/env python3
"""Diagnose why Phase-13 v1 improves GRAM cold metrics.

The analysis is deliberately training-free.  It joins the frozen v0/v1/v2
hierarchical-ID files with their test predictions and reports:

* warm-cluster size at every semantic prefix depth;
* exact lexical-ID overlap between cold and warm items;
* cold-cold collision rates;
* hit rate conditioned on cluster size and exact overlap; and
* collision-aware metrics that count an ambiguous gold lexical ID as zero.

The last metric matters because GRAM evaluates decoded lexical-ID strings.  If
two item IDs share the same complete lexical ID, the saved prediction cannot
establish which underlying item was retrieved.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Iterable


METRIC_NAMES = [
    "hit@1",
    "hit@3",
    "hit@5",
    "hit@10",
    "hit@20",
    "hit@50",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "ndcg@50",
]


DATASETS = {
    "Toys": {
        "levels": 5,
        "id_stem": "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split",
        "runs": {"v0": "v0_toys", "v1": "v1_toys", "v2_iter2": "v2_toys_iter2"},
    },
    "Beauty": {
        "levels": 7,
        "id_stem": "item_generative_indexing_hierarchy_v1_c128_l7_len32768_split",
        "runs": {
            "v0": "v0_beauty",
            "v1": "v1_beauty",
            "v2_iter2": "v2_beauty_iter2",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase13/explore/v1_success_mechanism"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "report/第十三阶段/"
            "GRAM_第十三阶段_v1_success-mechanism_碰撞审计报告.md"
        ),
    )
    return parser.parse_args()


def read_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def read_id_map(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item_id, encoded = line.split(" |", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed ID row {path}:{line_no}") from exc
        result[item_id] = tuple(encoded.split("|"))
    return result


def read_targets(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3:
            result[fields[0]] = fields[-1]
    return result


def read_prediction_rows(path: Path) -> list[tuple[str, list[float]]]:
    result: list[tuple[str, list[float]]] = []
    with path.open() as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if not fields or fields[0] == "idx" or fields[0].startswith(("hit@", "ndcg@")):
                continue
            if len(fields) < 1 + len(METRIC_NAMES):
                continue
            try:
                values = [float(value) for value in fields[1 : 1 + len(METRIC_NAMES)]]
            except ValueError:
                continue
            result.append((fields[0], values))
    return result


def choose_test_predictions(run_dir: Path, expected_rows: int) -> tuple[Path, list]:
    candidates = sorted(run_dir.rglob("*_pred_test.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No test prediction TSV below {run_dir}")
    parsed = [(path, read_prediction_rows(path)) for path in candidates]
    exact = [(path, rows) for path, rows in parsed if len(rows) == expected_rows]
    if exact:
        return max(exact, key=lambda pair: pair[0].stat().st_mtime)
    path, rows = max(parsed, key=lambda pair: len(pair[1]))
    raise ValueError(
        f"No complete test prediction TSV below {run_dir}; best is {path} "
        f"with {len(rows)}/{expected_rows} rows"
    )


def quantile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": mean(values),
        "median": median(values),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "max": max(values),
        "nonzero_n": sum(value > 0 for value in values),
        "nonzero_rate": mean([value > 0 for value in values]),
    }


def collision_summary(paths: Iterable[tuple[str, ...]]) -> dict[str, float | int]:
    paths = list(paths)
    buckets = Counter(paths)
    duplicate_excess = len(paths) - len(buckets)
    items_in_collision_buckets = sum(size for size in buckets.values() if size > 1)
    return {
        "n_items": len(paths),
        "n_unique_ids": len(buckets),
        "duplicate_excess": duplicate_excess,
        "duplicate_excess_rate": duplicate_excess / len(paths),
        "items_in_collision_buckets": items_in_collision_buckets,
        "items_in_collision_buckets_rate": items_in_collision_buckets / len(paths),
        "max_bucket": max(buckets.values()),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float | None]:
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def group_metrics(rows: list[list[float]]) -> dict[str, float | int]:
    if not rows:
        return {"n": 0, **{name: 0.0 for name in METRIC_NAMES}}
    return {
        "n": len(rows),
        **{name: mean(row[index] for row in rows) for index, name in enumerate(METRIC_NAMES)},
    }


def analyze_version(
    version: str,
    id_map: dict[str, tuple[str, ...]],
    cold: set[str],
    warm: set[str],
    levels: int,
    targets: dict[str, str],
    prediction_rows: list[tuple[str, list[float]]],
) -> dict:
    warm_prefix_counts = [
        Counter(id_map[item][:depth] for item in warm) for depth in range(1, levels + 1)
    ]
    warm_exact_counts = Counter(id_map[item] for item in warm)
    all_exact_counts = Counter(id_map.values())

    prefix_sizes_by_item = {
        item: [
            warm_prefix_counts[depth - 1][id_map[item][:depth]]
            for depth in range(1, levels + 1)
        ]
        for item in cold
    }
    exact_overlap_by_item = {
        item: warm_exact_counts[id_map[item]] for item in cold
    }

    cold_events: list[dict] = []
    for user_id, metrics in prediction_rows:
        item = targets.get(user_id)
        if item not in cold:
            continue
        cold_events.append(
            {
                "user_id": user_id,
                "item_id": item,
                "metrics": metrics,
                "base_cluster_size": prefix_sizes_by_item[item][-1],
                "exact_warm_overlap": exact_overlap_by_item[item],
                "all_id_multiplicity": all_exact_counts[id_map[item]],
            }
        )

    original_rows = [event["metrics"] for event in cold_events]
    ambiguous_rows = [
        event["metrics"] for event in cold_events if event["all_id_multiplicity"] > 1
    ]
    unambiguous_rows = [
        event["metrics"] for event in cold_events if event["all_id_multiplicity"] == 1
    ]
    strict_rows = [
        event["metrics"] if event["all_id_multiplicity"] == 1 else [0.0] * len(METRIC_NAMES)
        for event in cold_events
    ]

    bins: dict[str, list[list[float]]] = defaultdict(list)
    for event in cold_events:
        size = event["base_cluster_size"]
        label = "0" if size == 0 else "1" if size == 1 else "2-4" if size <= 4 else "5+"
        bins[label].append(event["metrics"])

    hit10_index = METRIC_NAMES.index("hit@10")
    original_hit10 = sum(row[hit10_index] > 0 for row in original_rows)
    ambiguous_hit10 = sum(row[hit10_index] > 0 for row in ambiguous_rows)
    unambiguous_hit10 = sum(row[hit10_index] > 0 for row in unambiguous_rows)

    return {
        "version": version,
        "n_cold_items": len(cold),
        "n_warm_items": len(warm),
        "prefix_cluster_distribution": {
            f"L{depth}": distribution(
                [prefix_sizes_by_item[item][depth - 1] for item in sorted(cold)]
            )
            for depth in range(1, levels + 1)
        },
        "cold_id_collision": collision_summary(id_map[item] for item in cold),
        "exact_warm_overlap": {
            "n_items": sum(exact_overlap_by_item[item] > 0 for item in cold),
            "item_rate": mean([exact_overlap_by_item[item] > 0 for item in cold]),
            "max_warm_multiplicity": max(exact_overlap_by_item.values()),
        },
        "test_events": {
            "n": len(cold_events),
            "original": group_metrics(original_rows),
            "ambiguous_id": group_metrics(ambiguous_rows),
            "unambiguous_id": group_metrics(unambiguous_rows),
            "collision_aware_strict": group_metrics(strict_rows),
            "hit10_counts": {
                "original": original_hit10,
                "ambiguous": ambiguous_hit10,
                "unambiguous": unambiguous_hit10,
                "ambiguous_share_of_hits": (
                    ambiguous_hit10 / original_hit10 if original_hit10 else 0.0
                ),
                "original_wilson95": wilson_interval(original_hit10, len(cold_events)),
                "strict_wilson95": wilson_interval(unambiguous_hit10, len(cold_events)),
            },
            "by_base_cluster_size": {
                label: group_metrics(bins.get(label, [])) for label in ["0", "1", "2-4", "5+"]
            },
        },
    }


def version_paths(dataset_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "v0": dataset_dir / f"{stem}.txt",
        "v1": dataset_dir / f"{stem}_v1_mlpcold.txt",
        "v2_iter2": dataset_dir / f"{stem}_v2iter2_mlpcold_llmprior.txt",
    }


def fmt_rate(value: float) -> str:
    return f"{100 * value:.3f}%"


def render_report(payload: dict) -> str:
    lines = [
        "# GRAM 第十三阶段：v1 成功机制与 lexical-ID 碰撞审计",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {payload['generated_at']}",
        "- Verification Status: ANALYZED",
        "- Version Label: phase13_v1_success_mechanism_v1",
        "",
        "## 口径",
        "",
        "- `base cluster size`：与 cold item 共享前 L 个语义 token 的 warm item 数。",
        "- `exact warm overlap`：cold 与 warm 的完整 lexical-ID 字符串完全相同。",
        "- `ambiguous ID`：同一完整 lexical ID 映射到多个 item。GRAM 原评测只比较该字符串，无法确认具体 item。",
        "- `collision-aware strict`：ambiguous gold ID 的整行指标置零；这是保守的 item-level 可确认下界，不是新模型结果。",
        "",
        "## 核心结果",
        "",
        "| Dataset | Version | Original H@10 | Strict H@10 | Ambiguous hits / all hits | Exact warm-overlap items | Cold collision rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, dataset_result in payload["datasets"].items():
        for version, result in dataset_result["versions"].items():
            events = result["test_events"]
            hit_counts = events["hit10_counts"]
            lines.append(
                f"| {dataset} | {version} | {fmt_rate(events['original']['hit@10'])} "
                f"| {fmt_rate(events['collision_aware_strict']['hit@10'])} "
                f"| {hit_counts['ambiguous']} / {hit_counts['original']} "
                f"| {fmt_rate(result['exact_warm_overlap']['item_rate'])} "
                f"| {fmt_rate(result['cold_id_collision']['duplicate_excess_rate'])} |"
            )

    lines.extend(["", "## 逐层 warm 簇频率", ""])
    for dataset, dataset_result in payload["datasets"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Version | Level | Mean | Median | P90 | Non-zero |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for version, result in dataset_result["versions"].items():
            for level, stats in result["prefix_cluster_distribution"].items():
                lines.append(
                    f"| {version} | {level} | {stats['mean']:.3f} | {stats['median']:.1f} "
                    f"| {stats['p90']} | {fmt_rate(stats['nonzero_rate'])} |"
                )
        lines.append("")

    lines.extend(["## H@10 × 最深层 base cluster size", ""])
    for dataset, dataset_result in payload["datasets"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Version | Warm items in base cluster | Events | H@10 |",
                "|---|---:|---:|---:|",
            ]
        )
        for version, result in dataset_result["versions"].items():
            for label, group in result["test_events"]["by_base_cluster_size"].items():
                lines.append(f"| {version} | {label} | {group['n']} | {fmt_rate(group['hit@10'])} |")
        lines.append("")

    v1_invalidated = True
    for dataset_result in payload["datasets"].values():
        v0 = dataset_result["versions"]["v0"]["test_events"]["original"]["hit@10"]
        v1_strict = dataset_result["versions"]["v1"]["test_events"]["collision_aware_strict"]["hit@10"]
        if v1_strict > v0:
            v1_invalidated = False

    lines.extend(
        [
            "## 诊断结论",
            "",
            (
                "**双域一致：collision-aware strict H@10 均不超过 v0。** "
                "现有 v1 强提升主要来自 lexical-ID 别名，不能继续作为已验证的 item-level cold 推荐收益。"
                if v1_invalidated
                else "至少一个数据域在 collision-aware strict 口径下仍超过 v0；需要结合 JSON 明细继续判断。"
            ),
            "",
            "这项结果否定的是当前 v1 评测有效性，不是否定 semantic bridge 本身。下一道必要 Gate 应是 collision-safe ID 赋值与评测；在该 Gate 前，不应把 v1 当作可靠前提直接进入 v4-retriever 或 v5。",
            "",
            "## 产物",
            "",
            f"- Machine-readable: `{payload['analysis_json']}`",
            f"- 采用的 test prediction 文件记录在 JSON 的每个 dataset/version 下。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report_path = args.report if args.report.is_absolute() else root / args.report
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_json": str((output_dir / "analysis.json").relative_to(root)),
        "datasets": {},
    }

    for dataset, config in DATASETS.items():
        levels = config["levels"]
        dataset_dir = root / "GRAM" / "rec_datasets" / f"{dataset}_cold50"
        cold = read_set(dataset_dir / "cold_split_meta" / "cold_items.txt")
        warm = read_set(dataset_dir / "cold_split_meta" / "warm_items.txt")
        targets = read_targets(dataset_dir / "user_sequence.txt")
        paths = version_paths(dataset_dir, config["id_stem"])
        id_maps = {version: read_id_map(path) for version, path in paths.items()}

        expected_items = cold | warm
        for version, id_map in id_maps.items():
            if set(id_map) != expected_items:
                raise ValueError(
                    f"{dataset}/{version} ID coverage mismatch: "
                    f"got={len(id_map)} expected={len(expected_items)}"
                )
            too_short = [item for item, tokens in id_map.items() if len(tokens) < levels]
            if too_short:
                raise ValueError(f"{dataset}/{version} has {len(too_short)} IDs shorter than L={levels}")

        dataset_payload = {"levels": levels, "versions": {}}
        for version, run_name in config["runs"].items():
            prediction_path, prediction_rows = choose_test_predictions(
                root / "artifacts" / "phase13" / "explore" / run_name,
                len(targets),
            )
            version_result = analyze_version(
                version,
                id_maps[version],
                cold,
                warm,
                levels,
                targets,
                prediction_rows,
            )
            version_result["id_file"] = str(paths[version].relative_to(root))
            version_result["predictions_tsv"] = str(prediction_path.relative_to(root))
            dataset_payload["versions"][version] = version_result
        payload["datasets"][dataset] = dataset_payload

    json_path = output_dir / "analysis.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(payload))
    print(f"[analysis] wrote {json_path}")
    print(f"[analysis] wrote {report_path}")


if __name__ == "__main__":
    main()
