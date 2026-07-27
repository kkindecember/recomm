#!/usr/bin/env python3
"""CPBD G0-D1 exact static truncation census (CPU only)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = ROOT / "GRAM/src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))

from processor import CollatorGRAM  # noqa: E402
from utils import indexing  # noqa: E402

DELIMITER_TOKEN_IDS = {1820, 9175}
METADATA_FIELDS = (
    "title",
    "brand",
    "categories",
    "description",
    "price",
    "salesrank",
    "other_metadata",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "artifacts/phase3/configs/cpbd_g0_d1_preregistered.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/phase3/cpbd_g0",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/第三阶段/GRAM_第三阶段_CPBD_G0_D1诊断报告.md",
    )
    return parser.parse_args()


def read_key_value(path: Path) -> dict[str, str]:
    result = {}
    with path.open() as handle:
        for line in handle:
            key, value = line.rstrip("\n").split(" ", 1)
            result[key] = value
    return result


def read_similar_items(path: Path, top_k: int) -> dict[str, list[str]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("anchor"):
                continue
            item, values = line.split(" ", 1)
            result[item] = values.split()[:top_k]
    return result


def read_sequences(path: Path) -> dict[str, list[str]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            user, *items = line.split()
            result[user] = items
    return result


def metadata_spans(metadata: str, absolute_start: int) -> dict[str, list[tuple[int, int]]]:
    spans = {field: [] for field in METADATA_FIELDS}
    cursor = 0
    for segment in metadata.split("; "):
        start = metadata.find(segment, cursor)
        if start < cursor:
            raise ValueError("failed to locate metadata segment")
        end = start + len(segment)
        label = segment.split(":", 1)[0].strip().lower() if ":" in segment else ""
        field = label if label in METADATA_FIELDS[:-1] else "other_metadata"
        spans[field].append((absolute_start + start, absolute_start + end))
        cursor = end
    return spans


def build_serialization(
    link: str, cf_values: list[str], metadata: str, metadata_first: bool
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    cf = ", ".join(cf_values)
    link_prefix = "item: "
    link_start = len(link_prefix)
    if metadata_first:
        meta_prefix = f"{link_prefix}{link}; "
        meta_start = len(meta_prefix)
        cf_prefix = f"{meta_prefix}{metadata}; similar items: "
        cf_start = len(cf_prefix)
        text = f"{cf_prefix}{cf}"
    else:
        cf_prefix = f"{link_prefix}{link}; similar items: "
        cf_start = len(cf_prefix)
        meta_prefix = f"{cf_prefix}{cf}; "
        meta_start = len(meta_prefix)
        text = f"{meta_prefix}{metadata}"
    spans = {
        "link": [(link_start, link_start + len(link))],
        "collaborative": [(cf_start, cf_start + len(cf))],
        "metadata_total": [(meta_start, meta_start + len(metadata))],
    }
    spans.update(metadata_spans(metadata, meta_start))
    return text, spans


def overlaps(offset: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = offset
    return end > start and any(start < right and end > left for left, right in spans)


def encode_exact(tokenizer, text: str, pre_max: int, max_len: int) -> dict:
    encoded = tokenizer(
        text,
        max_length=pre_max,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    filtered = [
        (int(token_id), int(mask), tuple(offset))
        for token_id, mask, offset in zip(
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded["offset_mapping"],
        )
        if int(token_id) not in DELIMITER_TOKEN_IDS
    ]
    raw = list(filtered)
    visible = list(filtered[:max_len])
    visible_ids = [row[0] for row in visible]
    visible_mask = [row[1] for row in visible]
    visible_offsets = [row[2] for row in visible]
    if tokenizer.eos_token_id not in visible_ids:
        visible_ids[-1] = tokenizer.eos_token_id
        visible_offsets[-1] = (0, 0)
    if len(visible_ids) < max_len:
        padding = max_len - len(visible_ids)
        visible_ids.extend([tokenizer.pad_token_id] * padding)
        visible_mask.extend([0] * padding)
        visible_offsets.extend([(0, 0)] * padding)
    return {
        "raw": raw,
        "visible_ids": visible_ids,
        "visible_mask": visible_mask,
        "visible_offsets": visible_offsets,
    }


def count_fields(
    encoding: dict, spans: dict[str, list[tuple[int, int]]], tokenizer
) -> dict[str, tuple[int, int, int]]:
    raw_offsets = [
        offset
        for token_id, mask, offset in encoding["raw"]
        if mask
        and token_id not in {tokenizer.pad_token_id, tokenizer.eos_token_id}
        and offset != (0, 0)
    ]
    visible_offsets = [
        offset
        for token_id, mask, offset in zip(
            encoding["visible_ids"],
            encoding["visible_mask"],
            encoding["visible_offsets"],
        )
        if mask
        and token_id not in {tokenizer.pad_token_id, tokenizer.eos_token_id}
        and offset != (0, 0)
    ]
    result = {}
    for field, field_spans in spans.items():
        raw = sum(overlaps(offset, field_spans) for offset in raw_offsets)
        visible = sum(overlaps(offset, field_spans) for offset in visible_offsets)
        result[field] = (raw, visible, raw - visible)
    return result


def runtime_args(dataset: str, spec: dict) -> SimpleNamespace:
    index_name = Path(spec["item_index"]).stem.removeprefix(
        "item_generative_indexing_"
    )
    return SimpleNamespace(
        data_path=str(ROOT / "GRAM/rec_datasets"),
        datasets=dataset,
        rank=0,
        item_id_path="",
        hierarchical_id_type=index_name,
        item_prompt="all_text",
        top_k_similar_item=int(spec["top_k_similar_item"]),
        cf_model="sasrec",
        id_linking=1,
        item_prompt_max_len=128,
        target_max_len=32,
        max_his=20,
        item_id_type="split",
    )


def popularity_counts(sequences: dict[str, list[str]]) -> Counter:
    counts = Counter()
    for items in sequences.values():
        counts.update(items[:-2])
    return counts


def popularity_strata(counts: Counter, items: list[str]) -> dict[str, str]:
    nonzero = sorted(counts[item] for item in items if counts[item] > 0)
    median = statistics.median(nonzero) if nonzero else 0
    return {
        item: (
            "zero"
            if counts[item] == 0
            else "nonzero_bottom50"
            if counts[item] <= median
            else "top50"
        )
        for item in items
    }


def validate_run_config(path: Path, spec: dict) -> None:
    with path.open() as handle:
        run = json.load(handle)
    expected = {
        "item_prompt_max_len": 128,
        "id_linking": 1,
        "item_prompt": "all_text",
        "cf_model": "sasrec",
        "top_k_similar_item": int(spec["top_k_similar_item"]),
    }
    failures = {
        key: (run.get(key), value)
        for key, value in expected.items()
        if run.get(key) != value
    }
    if failures:
        raise ValueError(f"locked run config mismatch: {failures}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], gates: dict) -> dict:
    recoverable = [row["recoverable_metadata_tokens"] for row in rows]
    retention = [row["current_metadata_retention"] for row in rows]
    metrics = {
        "catalog_items": len(rows),
        "rate_recoverable_metadata_ge_8": sum(value >= 8 for value in recoverable)
        / len(rows),
        "median_recoverable_metadata_tokens": statistics.median(recoverable),
        "mean_recoverable_metadata_tokens": statistics.fmean(recoverable),
        "median_displaced_cf_tokens": statistics.median(
            row["displaced_cf_tokens"] for row in rows
        ),
        "median_current_metadata_retention": statistics.median(retention),
        "mean_current_metadata_retention": statistics.fmean(retention),
    }
    mechanism = {
        "rate_recoverable_metadata_ge_8": metrics[
            "rate_recoverable_metadata_ge_8"
        ]
        >= gates["rate_recoverable_metadata_ge_8"],
        "median_recoverable_metadata_tokens": metrics[
            "median_recoverable_metadata_tokens"
        ]
        >= gates["median_recoverable_metadata_tokens"],
        "median_current_metadata_retention": metrics[
            "median_current_metadata_retention"
        ]
        <= gates["median_current_metadata_retention_max"],
    }
    by_popularity = {}
    for stratum in ("zero", "nonzero_bottom50", "top50"):
        subset = [row for row in rows if row["popularity_stratum"] == stratum]
        if subset:
            by_popularity[stratum] = {
                "n": len(subset),
                "mean_recoverable_metadata_tokens": statistics.fmean(
                    row["recoverable_metadata_tokens"] for row in subset
                ),
                "rate_recoverable_metadata_ge_8": sum(
                    row["recoverable_metadata_tokens"] >= 8 for row in subset
                )
                / len(subset),
            }
    field_totals = {}
    for field in ("title", "brand", "categories", "description", "price", "salesrank"):
        field_totals[field] = {
            "current_raw": sum(row[f"current_{field}_raw"] for row in rows),
            "current_visible": sum(row[f"current_{field}_visible"] for row in rows),
            "metadata_first_visible": sum(
                row[f"metadata_first_{field}_visible"] for row in rows
            ),
        }
    return {
        "metrics": metrics,
        "mechanism_gates": mechanism,
        "mechanism_pass": all(mechanism.values()),
        "by_popularity": by_popularity,
        "field_totals": field_totals,
    }


def run_dataset(dataset: str, spec: dict, config: dict, tokenizer, output_root: Path):
    paths = {key: ROOT / value for key, value in spec.items() if isinstance(value, str)}
    for key in ("run_config", "item_index", "item_text", "similar_items", "user_sequence"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    validate_run_config(paths["run_config"], spec)
    item_index = read_key_value(paths["item_index"])
    item_text = read_key_value(paths["item_text"])
    similar = read_similar_items(paths["similar_items"], int(spec["top_k_similar_item"]))
    catalog_sets = [set(item_index), set(item_text), set(similar)]
    union = set.union(*catalog_sets)
    items = sorted(set.intersection(*catalog_sets))
    coverage = len(items) / len(union)
    if coverage != config["integrity_gates"]["catalog_intersection_coverage"]:
        raise ValueError(f"{dataset} catalog intersection coverage={coverage}")
    if any(len(similar[item]) != int(spec["top_k_similar_item"]) for item in items):
        raise ValueError(f"{dataset} contains short collaborative lists")

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
    counts = popularity_counts(read_sequences(paths["user_sequence"]))
    strata = popularity_strata(counts, items)
    collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
    pre_max = int(config["tokenizer"]["pre_filter_max_length"])
    max_len = int(config["tokenizer"]["item_prompt_max_len"])
    rows = []
    for item in items:
        cf_strings = [item_index[value] for value in similar[item]]
        current_text, current_spans = build_serialization(
            item_index[item], cf_strings, item_text[item], False
        )
        alternate_text, alternate_spans = build_serialization(
            item_index[item], cf_strings, item_text[item], True
        )
        if current_text != item2input[item]:
            raise ValueError(f"{dataset}/{item} current serialization mismatch")
        current_encoding = encode_exact(tokenizer, current_text, pre_max, max_len)
        alternate_encoding = encode_exact(tokenizer, alternate_text, pre_max, max_len)
        current = count_fields(current_encoding, current_spans, tokenizer)
        alternate = count_fields(alternate_encoding, alternate_spans, tokenizer)
        current_meta_raw, current_meta_visible, current_meta_lost = current[
            "metadata_total"
        ]
        alt_meta_raw, alt_meta_visible, alt_meta_lost = alternate["metadata_total"]
        if item_index[item] not in current_text or item_text[item] not in current_text:
            raise ValueError("component identity failure")
        row = {
            "item": item,
            "popularity_train": counts[item],
            "popularity_stratum": strata[item],
            "top_k_similar_item": len(similar[item]),
            "current_link_raw": current["link"][0],
            "current_link_visible": current["link"][1],
            "current_cf_raw": current["collaborative"][0],
            "current_cf_visible": current["collaborative"][1],
            "current_cf_lost": current["collaborative"][2],
            "current_metadata_raw": current_meta_raw,
            "current_metadata_visible": current_meta_visible,
            "current_metadata_lost": current_meta_lost,
            "metadata_first_cf_raw": alternate["collaborative"][0],
            "metadata_first_cf_visible": alternate["collaborative"][1],
            "metadata_first_cf_lost": alternate["collaborative"][2],
            "metadata_first_metadata_raw": alt_meta_raw,
            "metadata_first_metadata_visible": alt_meta_visible,
            "metadata_first_metadata_lost": alt_meta_lost,
            "recoverable_metadata_tokens": alt_meta_visible - current_meta_visible,
            "displaced_cf_tokens": current["collaborative"][1]
            - alternate["collaborative"][1],
            "current_metadata_retention": (
                current_meta_visible / current_meta_raw if current_meta_raw else 1.0
            ),
        }
        for field in METADATA_FIELDS:
            row[f"current_{field}_raw"] = current[field][0]
            row[f"current_{field}_visible"] = current[field][1]
            row[f"metadata_first_{field}_visible"] = alternate[field][1]
        if not all(
            math.isfinite(value)
            for value in row.values()
            if isinstance(value, (int, float))
        ):
            raise ValueError(f"{dataset}/{item} non-finite metric")
        rows.append(row)

    # Exact production replay is checked for every current passage in bounded chunks.
    replayed = 0
    for start in range(0, len(items), 64):
        chunk = items[start : start + 64]
        batch_ids, batch_masks = collator.encode_texts_split(
            [[item2input[item]] for item in chunk], tokenizer
        )
        for index, item in enumerate(chunk):
            expected = encode_exact(tokenizer, item2input[item], pre_max, max_len)
            width = batch_ids.shape[-1]
            if batch_ids[index, 0].tolist() != expected["visible_ids"][:width]:
                raise ValueError(f"{dataset}/{item} input-id replay mismatch")
            if (
                batch_masks[index, 0].int().tolist()
                != expected["visible_mask"][:width]
            ):
                raise ValueError(f"{dataset}/{item} attention replay mismatch")
            replayed += 1
    if replayed != len(items):
        raise ValueError("incomplete exact replay")

    summary = summarize(rows, config["mechanism_gates_per_dataset"])
    summary["dataset"] = dataset
    summary["integrity"] = {
        "catalog_intersection_coverage": coverage,
        "parse_success_rate": 1.0,
        "finite_metric_rate": 1.0,
        "fixed_component_identity_rate": 1.0,
        "exact_collator_replay_rate": replayed / len(items),
    }
    dataset_dir = output_root / dataset
    write_csv(dataset_dir / "item_truncation_census.csv", rows)
    with (dataset_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def write_report(path: Path, summaries: dict, decision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GRAM 第三阶段：CPBD G0-D1 static truncation census",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-07-24",
        "- Verification Status: VERIFIED",
        "- Version Label: `cpbd_g0_d1_v1`",
        "",
        "## 固定决策",
        "",
        f"**`{decision}`**",
        "",
        "本诊断只审计结构性截断；未加载 checkpoint、未评分、未训练、未使用 GPU，",
        "也未读取 validation/test target 或效果。",
        "",
        "## 双数据集主结果",
        "",
        "| 数据集 | items | recoverable>=8 | recoverable median | metadata retention median | displaced CF median | gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, summary in summaries.items():
        metrics = summary["metrics"]
        lines.append(
            f"| {dataset} | {metrics['catalog_items']:,} | "
            f"{metrics['rate_recoverable_metadata_ge_8']:.4f} | "
            f"{metrics['median_recoverable_metadata_tokens']:.2f} | "
            f"{metrics['median_current_metadata_retention']:.4f} | "
            f"{metrics['median_displaced_cf_tokens']:.2f} | "
            f"{'PASS' if summary['mechanism_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "通过只表示当前 GRAM serialization 在双数据集中存在广泛、可由固定内容重排",
            "机械恢复的 metadata displacement。它不表示被恢复 metadata 有推荐价值，",
            "也不表示 metadata-first 是最终方法。下一步若获准，只能先预注册固定预算、",
            "固定 CF identity、带位置 control 的 frozen outcome diagnosis。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        config = json.load(handle)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["name"], local_files_only=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("G0-D1 requires a fast tokenizer with offsets")
    summaries = {}
    for dataset, spec in config["datasets"].items():
        summaries[dataset] = run_dataset(
            dataset, spec, config, tokenizer, args.output_root
        )
    decision = (
        "G0_D2_DESIGN_ALLOWED"
        if all(summary["mechanism_pass"] for summary in summaries.values())
        else "STOP_CPBD_NO_STRUCTURAL_DISPLACEMENT"
    )
    combined = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-24",
            "verification_status": "VERIFIED",
            "version_label": "cpbd_g0_d1_v1",
        },
        "decision": decision,
        "datasets": summaries,
        "resource_scope": {
            "checkpoint_loaded": False,
            "gpu_used": False,
            "training_performed": False,
            "validation_or_test_effect_read": False
        }
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "summary.json").open("w") as handle:
        json.dump(combined, handle, indent=2, sort_keys=True)
    write_report(args.report, summaries, decision)
    print(json.dumps({"decision": decision, "datasets": summaries}, indent=2))


if __name__ == "__main__":
    main()
