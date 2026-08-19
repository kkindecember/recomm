#!/usr/bin/env python3
"""Make MLP-assigned cold IDs globally unique without changing warm IDs.

Phase-13 v1 assigned an L-token MLP prediction to every cold item but did not
deduplicate the merged ID file.  GRAM evaluates the decoded lexical-ID string,
so duplicate complete IDs make item-level hits ambiguous.  This postprocessor
preserves every warm ID byte-for-byte and appends the smallest available
numeric token to cold IDs whenever their complete ID is already reserved or
shared by multiple cold items.

The input is never modified.  Output order matches input order.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hierarchical_id_utils import format_id_line, parse_id_line, read_item_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-id-file", type=Path, required=True)
    parser.add_argument("--cold-items", type=Path, required=True)
    parser.add_argument("--output-id-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output after all checks pass; never overwrites the input.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    rows: list[tuple[str, tuple[str, ...]]] = []
    seen_items: set[str] = set()
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item, tokens = parse_id_line(line)
            if item in seen_items:
                raise ValueError(f"Duplicate item ID at {path}:{line_no}: {item}")
            seen_items.add(item)
            rows.append((item, tuple(tokens)))
    if not rows:
        raise ValueError(f"Empty ID file: {path}")
    return rows


def collision_stats(paths: list[tuple[str, ...]]) -> dict[str, int | float]:
    counts = Counter(paths)
    duplicate_excess = len(paths) - len(counts)
    collision_items = sum(size for size in counts.values() if size > 1)
    return {
        "n_items": len(paths),
        "n_unique_ids": len(counts),
        "duplicate_excess": duplicate_excess,
        "duplicate_excess_rate": duplicate_excess / len(paths),
        "items_in_collision_buckets": collision_items,
        "items_in_collision_buckets_rate": collision_items / len(paths),
        "max_bucket": max(counts.values()),
    }


def make_collision_safe(
    rows: list[tuple[str, tuple[str, ...]]], cold_items: set[str]
) -> tuple[list[tuple[str, tuple[str, ...]]], dict]:
    items = {item for item, _tokens in rows}
    missing_cold = cold_items - items
    if missing_cold:
        raise ValueError(
            f"Cold-item file contains {len(missing_cold)} items absent from ID file; "
            f"examples={sorted(missing_cold)[:5]}"
        )

    warm_items = items - cold_items
    input_map = dict(rows)
    cold_groups: dict[tuple[str, ...], list[str]] = {}
    for item, tokens in rows:
        if item in cold_items:
            cold_groups.setdefault(tokens, []).append(item)

    # Warm IDs are immutable and reserve their complete token paths.
    used_paths = {input_map[item] for item in warm_items}
    output_map = {item: input_map[item] for item in warm_items}
    modified_items: list[str] = []
    warm_overlap_groups = 0
    cold_collision_groups = 0
    suffix_candidates_skipped = 0
    max_suffix = -1

    for base_path, group_items in cold_groups.items():
        warm_overlap = base_path in used_paths
        cold_collision = len(group_items) > 1
        if warm_overlap:
            warm_overlap_groups += 1
        if cold_collision:
            cold_collision_groups += 1

        if not warm_overlap and not cold_collision:
            output_map[group_items[0]] = base_path
            used_paths.add(base_path)
            continue

        suffix = 0
        for item in group_items:
            while base_path + (str(suffix),) in used_paths:
                suffix_candidates_skipped += 1
                suffix += 1
            output_map[item] = base_path + (str(suffix),)
            used_paths.add(output_map[item])
            modified_items.append(item)
            max_suffix = max(max_suffix, suffix)
            suffix += 1

    output_rows = [(item, output_map[item]) for item, _tokens in rows]

    # Hard invariants: the transformation may only append one numeric token to
    # a modified cold ID.  Warm IDs, item coverage, and row order stay fixed.
    if [item for item, _ in output_rows] != [item for item, _ in rows]:
        raise AssertionError("Row order changed")
    if len(set(path for _item, path in output_rows)) != len(output_rows):
        raise AssertionError("Output IDs are not globally unique")
    for item in warm_items:
        if output_map[item] != input_map[item]:
            raise AssertionError(f"Warm ID changed: {item}")
    for item in cold_items:
        before = input_map[item]
        after = output_map[item]
        if after == before:
            continue
        if len(after) != len(before) + 1 or after[:-1] != before or not after[-1].isdigit():
            raise AssertionError(f"Invalid cold suffix transformation: {item}")

    report = {
        "n_items": len(rows),
        "n_warm": len(warm_items),
        "n_cold": len(cold_items),
        "n_cold_modified": len(modified_items),
        "cold_modified_rate": len(modified_items) / len(cold_items),
        "n_warm_overlap_groups": warm_overlap_groups,
        "n_cold_collision_groups": cold_collision_groups,
        "suffix_candidates_skipped": suffix_candidates_skipped,
        "max_assigned_suffix": max_suffix if max_suffix >= 0 else None,
        "warm_ids_unchanged": True,
        "row_order_unchanged": True,
        "cold_prefixes_unchanged": True,
        "input_collision": collision_stats([tokens for _item, tokens in rows]),
        "output_collision": collision_stats([tokens for _item, tokens in output_rows]),
        "sample_modified_items": [
            {
                "item_id": item,
                "before": list(input_map[item]),
                "after": list(output_map[item]),
            }
            for item in modified_items[:10]
        ],
    }
    return output_rows, report


def write_rows(path: Path, rows: list[tuple[str, tuple[str, ...]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for item, tokens in rows:
            handle.write(format_id_line(item, list(tokens)) + "\n")


def main() -> None:
    args = parse_args()
    input_path = args.input_id_file.resolve()
    output_path = args.output_id_file.resolve()
    report_path = args.report.resolve()
    if input_path == output_path:
        raise ValueError("Refusing to overwrite the input ID file")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {output_path}")

    rows = read_rows(input_path)
    cold_items = read_item_set(args.cold_items.resolve())
    output_rows, report = make_collision_safe(rows, cold_items)
    report.update(
        {
            "input_id_file": str(input_path),
            "cold_items_file": str(args.cold_items.resolve()),
            "output_id_file": str(output_path),
        }
    )

    # Compute and validate everything before making either output visible.
    write_rows(output_path, output_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[collision-safe] items={report['n_items']} cold={report['n_cold']} "
        f"modified={report['n_cold_modified']} "
        f"output_duplicate_excess={report['output_collision']['duplicate_excess']}"
    )
    print(f"[collision-safe] wrote {output_path}")
    print(f"[collision-safe] wrote {report_path}")


if __name__ == "__main__":
    main()
