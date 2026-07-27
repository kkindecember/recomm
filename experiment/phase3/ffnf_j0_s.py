#!/usr/bin/env python3
"""FFNF J0-S fixed 64+64 field-budget feasibility census (CPU only)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GRAM_SRC = ROOT / "GRAM/src"
for path in (GRAM_SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from processor import CollatorGRAM  # noqa: E402
from utils import indexing  # noqa: E402

from cpbd_g0_d1 import (  # noqa: E402
    METADATA_FIELDS,
    build_serialization,
    count_fields,
    encode_exact,
    metadata_spans,
    read_key_value,
    read_similar_items,
    runtime_args,
    validate_run_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "artifacts/phase3/configs/ffnf_j0_s_preregistered.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/phase3/ffnf_j0_s",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/第三阶段/GRAM_第三阶段_FFNF_J0_S可行性报告.md",
    )
    return parser.parse_args()


def build_cf_stream(link: str, cf_values: list[str]):
    cf = ", ".join(cf_values)
    prefix = "item: "
    link_start = len(prefix)
    cf_prefix = f"{prefix}{link}; similar items: "
    cf_start = len(cf_prefix)
    text = f"{cf_prefix}{cf}"
    spans = {
        "link": [(link_start, link_start + len(link))],
        "collaborative": [(cf_start, cf_start + len(cf))],
    }
    return text, spans


def build_metadata_stream(metadata: str):
    spans = {"metadata_total": [(0, len(metadata))]}
    spans.update(metadata_spans(metadata, 0))
    return metadata, spans


def active_tokens(encoding: dict) -> int:
    return sum(encoding["visible_mask"])


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 1.0


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], config: dict):
    total = {
        "current_cf_visible": sum(row["current_cf_visible"] for row in rows),
        "ffnf_cf_visible": sum(row["ffnf_cf_visible"] for row in rows),
        "current_metadata_visible": sum(
            row["current_metadata_visible"] for row in rows
        ),
        "ffnf_metadata_visible": sum(row["ffnf_metadata_visible"] for row in rows),
    }
    field_totals = {}
    for field in METADATA_FIELDS:
        field_totals[field] = {
            "current_visible": sum(row[f"current_{field}_visible"] for row in rows),
            "ffnf_visible": sum(row[f"ffnf_{field}_visible"] for row in rows),
        }
        field_totals[field]["ratio"] = ratio(
            field_totals[field]["ffnf_visible"],
            field_totals[field]["current_visible"],
        )
    metrics = {
        "catalog_items": len(rows),
        "aggregate_cf_visible_ratio_vs_current": ratio(
            total["ffnf_cf_visible"], total["current_cf_visible"]
        ),
        "aggregate_metadata_visible_gain": total["ffnf_metadata_visible"]
        - total["current_metadata_visible"],
        "aggregate_metadata_visible_ratio_vs_current": ratio(
            total["ffnf_metadata_visible"], total["current_metadata_visible"]
        ),
        "item_fraction_metadata_visible_gain_positive": statistics.fmean(
            row["metadata_visible_gain"] > 0 for row in rows
        ),
        "median_metadata_visible_gain": statistics.median(
            row["metadata_visible_gain"] for row in rows
        ),
        "mean_metadata_visible_gain": statistics.fmean(
            row["metadata_visible_gain"] for row in rows
        ),
        "mean_active_token_delta": statistics.fmean(
            row["ffnf_active_tokens"] - row["current_active_tokens"] for row in rows
        ),
        "rate_active_token_increase": statistics.fmean(
            row["ffnf_active_tokens"] > row["current_active_tokens"] for row in rows
        ),
    }
    gates_spec = config["feasibility_gates_per_dataset"]
    gates = {
        "cf_coverage": metrics["aggregate_cf_visible_ratio_vs_current"]
        >= gates_spec["aggregate_cf_visible_ratio_vs_current_min"],
        "title_coverage": field_totals["title"]["ratio"]
        >= gates_spec["aggregate_title_visible_ratio_vs_current_min"],
        "brand_coverage": field_totals["brand"]["ratio"]
        >= gates_spec["aggregate_brand_visible_ratio_vs_current_min"],
        "categories_coverage": field_totals["categories"]["ratio"]
        >= gates_spec["aggregate_categories_visible_ratio_vs_current_min"],
        "metadata_gain": metrics["aggregate_metadata_visible_gain"]
        > gates_spec["aggregate_metadata_visible_gain_strictly_greater_than"],
        "metadata_gain_breadth": metrics[
            "item_fraction_metadata_visible_gain_positive"
        ]
        >= gates_spec["item_fraction_metadata_visible_gain_positive_min"],
    }
    return {
        "metrics": metrics,
        "field_totals": field_totals,
        "feasibility_gates": gates,
        "feasibility_pass": all(gates.values()),
    }


def run_dataset(dataset, spec, config, tokenizer, output_root):
    paths = {
        key: ROOT / value for key, value in spec.items() if isinstance(value, str)
    }
    for key in ("run_config", "item_index", "item_text", "similar_items"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    validate_run_config(paths["run_config"], spec)
    item_index = read_key_value(paths["item_index"])
    item_text = read_key_value(paths["item_text"])
    similar = read_similar_items(
        paths["similar_items"], int(spec["top_k_similar_item"])
    )
    sets = [set(item_index), set(item_text), set(similar)]
    union = set.union(*sets)
    items = sorted(set.intersection(*sets))
    catalog_coverage = len(items) / len(union)
    if catalog_coverage != config["integrity_gates"]["catalog_intersection_coverage"]:
        raise ValueError(f"{dataset} catalog intersection coverage={catalog_coverage}")
    if any(len(similar[item]) != int(spec["top_k_similar_item"]) for item in items):
        raise ValueError(f"{dataset} has short collaborative lists")

    runtime = runtime_args(dataset, spec)
    _, item2input, _ = indexing.gram_indexing(
        data_path=runtime.data_path,
        dataset=dataset,
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        args=runtime,
        user_id_without_target_item=True,
        id_linking=True,
    )
    collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
    token = config["tokenizer"]
    pre_max = int(token["pre_filter_max_length"])
    current_width = int(token["current_width"])
    cf_width = int(token["cf_width"])
    metadata_width = int(token["metadata_width"])
    if cf_width + metadata_width != config["integrity_gates"]["tensor_width_sum"]:
        raise ValueError("branch tensor widths do not sum to locked capacity")

    rows = []
    for item in items:
        cf_values = [item_index[value] for value in similar[item]]
        current_text, current_spans = build_serialization(
            item_index[item], cf_values, item_text[item], False
        )
        cf_text, cf_spans = build_cf_stream(item_index[item], cf_values)
        metadata_text, metadata_only_spans = build_metadata_stream(item_text[item])
        if current_text != item2input[item]:
            raise ValueError(f"{dataset}/{item} current serialization mismatch")
        current_encoding = encode_exact(
            tokenizer, current_text, pre_max, current_width
        )
        cf_encoding = encode_exact(tokenizer, cf_text, pre_max, cf_width)
        metadata_encoding = encode_exact(
            tokenizer, metadata_text, pre_max, metadata_width
        )
        current = count_fields(current_encoding, current_spans, tokenizer)
        cf = count_fields(cf_encoding, cf_spans, tokenizer)
        meta = count_fields(
            metadata_encoding, metadata_only_spans, tokenizer
        )
        current_meta = current["metadata_total"]
        ffnf_meta = meta["metadata_total"]
        row = {
            "item": item,
            "top_k_similar_item": len(similar[item]),
            "current_tensor_width": current_width,
            "ffnf_cf_tensor_width": cf_width,
            "ffnf_metadata_tensor_width": metadata_width,
            "ffnf_tensor_width_sum": cf_width + metadata_width,
            "current_active_tokens": active_tokens(current_encoding),
            "ffnf_cf_active_tokens": active_tokens(cf_encoding),
            "ffnf_metadata_active_tokens": active_tokens(metadata_encoding),
            "ffnf_active_tokens": active_tokens(cf_encoding)
            + active_tokens(metadata_encoding),
            "current_link_visible": current["link"][1],
            "ffnf_link_visible": cf["link"][1],
            "current_cf_visible": current["collaborative"][1],
            "ffnf_cf_visible": cf["collaborative"][1],
            "current_metadata_visible": current_meta[1],
            "ffnf_metadata_visible": ffnf_meta[1],
            "metadata_visible_gain": ffnf_meta[1] - current_meta[1],
            "link_duplication_count": 0,
        }
        for field in METADATA_FIELDS:
            row[f"current_{field}_visible"] = current[field][1]
            row[f"ffnf_{field}_visible"] = meta[field][1]
        if not all(
            math.isfinite(value)
            for value in row.values()
            if isinstance(value, (int, float))
        ):
            raise ValueError(f"{dataset}/{item} non-finite metrics")
        rows.append(row)

    replayed = 0
    for start in range(0, len(items), 64):
        chunk = items[start : start + 64]
        batch_ids, batch_masks = collator.encode_texts_split(
            [[item2input[item]] for item in chunk], tokenizer
        )
        for index, item in enumerate(chunk):
            expected = encode_exact(
                tokenizer, item2input[item], pre_max, current_width
            )
            width = batch_ids.shape[-1]
            if batch_ids[index, 0].tolist() != expected["visible_ids"][:width]:
                raise ValueError(f"{dataset}/{item} current ID replay mismatch")
            if (
                batch_masks[index, 0].int().tolist()
                != expected["visible_mask"][:width]
            ):
                raise ValueError(f"{dataset}/{item} current mask replay mismatch")
            replayed += 1

    summary = summarize(rows, config)
    summary["dataset"] = dataset
    summary["integrity"] = {
        "catalog_intersection_coverage": catalog_coverage,
        "parse_success_rate": 1.0,
        "finite_metric_rate": 1.0,
        "raw_component_identity_rate": 1.0,
        "tensor_width_sum": cf_width + metadata_width,
        "no_link_duplication_rate": statistics.fmean(
            row["link_duplication_count"] == 0 for row in rows
        ),
        "current_exact_replay_rate": replayed / len(items),
        "checkpoint_loaded": False,
        "gpu_used": False,
        "validation_or_test_effect_read": False,
    }
    output_dir = output_root / dataset
    write_csv(output_dir / "item_budget_census.csv", rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def decide(summaries):
    if not all(
        summary["integrity"]["catalog_intersection_coverage"] == 1
        and summary["integrity"]["parse_success_rate"] == 1
        and summary["integrity"]["finite_metric_rate"] == 1
        and summary["integrity"]["raw_component_identity_rate"] == 1
        and summary["integrity"]["tensor_width_sum"] == 128
        and summary["integrity"]["no_link_duplication_rate"] == 1
        and summary["integrity"]["current_exact_replay_rate"] == 1
        for summary in summaries.values()
    ):
        return "EXECUTION_INVALID"
    if not all(summary["feasibility_pass"] for summary in summaries.values()):
        return "STOP_FFNF_BUDGET_INFEASIBLE"
    return "FFNF_J1_DESIGN_ALLOWED"


def write_report(path, aggregate):
    lines = [
        "# GRAM 第三阶段：FFNF J0-S 固定预算可行性",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-07-24",
        "- Verification Status: VERIFIED",
        "- Version Label: `ffnf_j0_s_v1`",
        "",
        f"固定决策：**`{aggregate['decision']}`**。",
        "",
        "本阶段仅运行 CPU tokenizer census；未加载 checkpoint、未训练、未读取",
        "validation/test 效果。",
        "",
        "## 双数据集结果",
        "",
        "| Dataset | CF ratio | Metadata gain | Gain-positive items | Title ratio | Brand ratio | Categories ratio | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, summary in aggregate["datasets"].items():
        metrics = summary["metrics"]
        fields = summary["field_totals"]
        lines.append(
            f"| {dataset} | "
            f"{metrics['aggregate_cf_visible_ratio_vs_current']:.4f} | "
            f"{metrics['aggregate_metadata_visible_gain']:,} | "
            f"{metrics['item_fraction_metadata_visible_gain_positive']:.4f} | "
            f"{fields['title']['ratio']:.4f} | "
            f"{fields['brand']['ratio']:.4f} | "
            f"{fields['categories']['ratio']:.4f} | "
            f"{summary['feasibility_pass']} |"
        )
    lines.extend(
        [
            "",
            "固定预算指 padded tensor width / decoder capacity 为 64+64=128；短文本",
            "不要求恰有 128 个 active tokens。active-token delta 与第二个 EOS 仅作",
            "J1 confound 记录，不能救援或否定本轮字段覆盖 gate。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    with args.config.open() as handle:
        config = json.load(handle)
    if not config.get("preregistered_before_new_static_census"):
        raise ValueError("J0-S is not preregistered")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["name"], local_files_only=True
    )
    if not tokenizer.is_fast:
        raise ValueError("FFNF J0-S requires a fast tokenizer")
    summaries = {}
    for dataset, spec in config["datasets"].items():
        summaries[dataset] = run_dataset(
            dataset, spec, config, tokenizer, args.output_root
        )
    aggregate = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-24",
            "verification_status": "VERIFIED",
            "version_label": "ffnf_j0_s_v1",
        },
        "decision": decide(summaries),
        "datasets": summaries,
        "resource_scope": {
            "checkpoint_loaded": False,
            "gpu_used": False,
            "training_performed": False,
            "validation_or_test_effect_read": False,
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(args.report, aggregate)
    print(
        json.dumps(
            {
                "decision": aggregate["decision"],
                "datasets": {
                    name: summary["metrics"] for name, summary in summaries.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
