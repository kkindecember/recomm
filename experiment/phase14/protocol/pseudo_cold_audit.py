"""Stage 14-1a deterministic item-disjoint pseudo-cold data audit.

This CPU-only preflight builds the item split and proves that pseudo-cold and
real-cold interactions are absent from student-readable warm CE histories and
targets.  Held pseudo-cold ground truth is written under a separately labelled
directory and is forbidden as an input to teacher fitting or student training.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Iterable

from cold_prefix_support import deepest_supported_prefix, load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-sequences", required=True)
    parser.add_argument("--item-path-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pseudo-fraction", type=float, default=0.20)
    parser.add_argument("--frequency-buckets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--max-history", type=int, default=20)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sequences(path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_no}: sequence is too short")
            if parts[0] in seen:
                raise ValueError(f"{path}:{line_no}: duplicate user {parts[0]}")
            seen.add(parts[0])
            rows.append((parts[0], parts[1:]))
    if not rows:
        raise ValueError(f"No sequences in {path}")
    return rows


def read_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def read_text_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line:
                continue
            item, separator, text = line.partition(" ")
            if item in lengths:
                raise ValueError(f"{path}:{line_no}: duplicate item {item}")
            lengths[item] = len(text.split()) if separator else 0
    return lengths


def frequency_map(sequences: Iterable[tuple[str, list[str]]]) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for _user, items in sequences:
        counts.update(items)
    return counts


def log_frequency_buckets(
    frequencies: dict[str, int], catalog: set[str], n_buckets: int
) -> dict[str, int]:
    if n_buckets < 1:
        raise ValueError("frequency_buckets must be positive")
    missing = catalog - frequencies.keys()
    if missing:
        raise ValueError(f"{len(missing)} catalog items have no source frequency")
    values = [math.log(frequencies[item] + 1.0) for item in catalog]
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return {item: 0 for item in catalog}
    return {
        item: min(
            int((math.log(frequencies[item] + 1.0) - low) / (high - low) * n_buckets),
            n_buckets - 1,
        )
        for item in catalog
    }


def _allocate_by_reference(
    total: int, reference_counts: dict[int, int], capacities: dict[int, int]
) -> dict[int, int]:
    if total <= 0 or total > sum(capacities.values()):
        raise ValueError("Requested pseudo-cold size is outside available capacity")
    reference_total = sum(reference_counts.values())
    if reference_total <= 0:
        raise ValueError("Reference cold distribution is empty")
    ideal = {bucket: total * reference_counts.get(bucket, 0) / reference_total for bucket in capacities}
    allocation = {
        bucket: min(int(math.floor(ideal[bucket])), capacities[bucket])
        for bucket in capacities
    }
    while sum(allocation.values()) < total:
        candidates = [
            bucket for bucket in capacities if allocation[bucket] < capacities[bucket]
        ]
        if not candidates:
            raise RuntimeError("Could not satisfy pseudo-cold allocation")
        bucket = max(
            candidates,
            key=lambda value: (ideal[value] - allocation[value], -value),
        )
        allocation[bucket] += 1
    return allocation


def select_pseudo_cold_items(
    eligible_warm: set[str],
    real_cold: set[str],
    buckets: dict[str, int],
    fraction: float,
    seed: int,
) -> tuple[set[str], dict]:
    if not 0 < fraction < 1:
        raise ValueError("pseudo_fraction must be in (0, 1)")
    total = int(round(fraction * len(eligible_warm)))
    warm_by_bucket: dict[int, list[str]] = collections.defaultdict(list)
    cold_counts: collections.Counter[int] = collections.Counter()
    for item in eligible_warm:
        warm_by_bucket[buckets[item]].append(item)
    for item in real_cold:
        cold_counts[buckets[item]] += 1
    capacities = {bucket: len(warm_by_bucket.get(bucket, [])) for bucket in set(buckets.values())}
    allocation = _allocate_by_reference(total, dict(cold_counts), capacities)

    selected: set[str] = set()
    for bucket, count in allocation.items():
        ranked = sorted(
            warm_by_bucket.get(bucket, []),
            key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(),
        )
        selected.update(ranked[:count])
    if len(selected) != total:
        raise RuntimeError(f"Pseudo-cold selection size mismatch: {len(selected)} != {total}")
    return selected, {
        "requested_fraction": fraction,
        "n_eligible_warm": len(eligible_warm),
        "n_selected": len(selected),
        "reference_real_cold_bucket_counts": {
            str(key): cold_counts[key] for key in sorted(cold_counts)
        },
        "eligible_warm_bucket_counts": {
            str(key): len(warm_by_bucket.get(key, [])) for key in sorted(capacities)
        },
        "selected_bucket_counts": {
            str(key): allocation[key] for key in sorted(allocation)
        },
    }


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def percentile(ratio: float) -> float:
        return ordered[min(int(round(ratio * (len(ordered) - 1))), len(ordered) - 1)]

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def item_profile(
    items: set[str],
    frequencies: dict[str, int],
    paths: dict[str, list[str]],
    text_lengths: dict[str, int],
    retained_warm: set[str],
) -> dict:
    warm_prefixes: dict[int, set[tuple[str, ...]]] = collections.defaultdict(set)
    for item in retained_warm:
        path = paths[item]
        for depth in range(1, len(path) + 1):
            warm_prefixes[depth].add(tuple(path[:depth]))
    depths = [deepest_supported_prefix(paths[item], warm_prefixes) for item in items]
    normalized = [depth / len(paths[item]) for item, depth in zip(items, depths)]
    return {
        "source_frequency": distribution([float(frequencies[item]) for item in items]),
        "path_length": distribution([float(len(paths[item])) for item in items]),
        "deepest_retained_warm_overlap": distribution([float(value) for value in depths]),
        "normalized_deepest_overlap": distribution(normalized),
        "text_length_words": distribution([float(text_lengths[item]) for item in items]),
    }


def audit_filtered_training(
    sequences: list[tuple[str, list[str]]],
    pseudo_cold: set[str],
    real_cold: set[str],
    max_history: int,
) -> tuple[list[dict], list[dict], dict]:
    held: list[dict] = []
    student_sequences: list[dict] = []
    counts: collections.Counter[str] = collections.Counter()
    for user, items in sequences:
        train_prefix = items[:-2]
        real_cold_in_prefix = [item for item in train_prefix if item in real_cold]
        if real_cold_in_prefix:
            raise RuntimeError(f"Real cold interaction leaked into filtered train prefix for {user}")
        visible: list[str] = []
        for position, item in enumerate(train_prefix):
            if item in pseudo_cold:
                counts["pseudo_occurrences_removed"] += 1
                if visible:
                    held.append(
                        {
                            "user_id": user,
                            "target_item": item,
                            "visible_history": visible[-max_history:],
                            "train_prefix_position": position,
                        }
                    )
                else:
                    counts["held_dropped_empty_history"] += 1
                continue
            visible.append(item)

        for position in range(1, len(visible)):
            history = visible[max(0, position - max_history):position]
            target = visible[position]
            counts["student_ce_examples"] += 1
            counts["teacher_visible_histories"] += 1
            counts["student_history_pseudo_leaks"] += sum(item in pseudo_cold for item in history)
            counts["student_history_real_cold_leaks"] += sum(item in real_cold for item in history)
            counts["student_target_pseudo_leaks"] += int(target in pseudo_cold)
            counts["student_target_real_cold_leaks"] += int(target in real_cold)

        if visible:
            student_sequences.append({"user_id": user, "train_items": visible})

    counts["held_events"] = len(held)
    counts["held_unique_items"] = len({row["target_item"] for row in held})
    counts["student_users"] = len(student_sequences)
    return held, student_sequences, dict(counts)


def main() -> None:
    args = parse_args()
    started = time.time()
    dataset_dir = Path(args.dataset_dir).resolve()
    source_sequences_path = Path(args.source_sequences).resolve()
    item_path_file = Path(args.item_path_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_existing = {"status.json", "run.log"}
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in allowed_existing]
    if unexpected:
        raise FileExistsError(f"Refusing to overwrite scientific artifacts: {unexpected}")

    inputs = {
        "filtered_sequences": dataset_dir / "user_sequence.txt",
        "source_sequences": source_sequences_path,
        "cold_items": dataset_dir / "cold_split_meta" / "cold_items.txt",
        "warm_items": dataset_dir / "cold_split_meta" / "warm_items.txt",
        "item_paths": item_path_file,
        "item_text": dataset_dir / "item_plain_text.txt",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        if "test" in path.name.lower():
            raise ValueError(f"Refusing test-labelled input: {path}")

    filtered_sequences = read_sequences(inputs["filtered_sequences"])
    source_sequences = read_sequences(inputs["source_sequences"])
    real_cold = read_set(inputs["cold_items"])
    warm = read_set(inputs["warm_items"])
    paths = load_paths(str(inputs["item_paths"]))
    text_lengths = read_text_lengths(inputs["item_text"])
    catalog = set(paths)
    if real_cold & warm or real_cold | warm != catalog:
        raise ValueError("Cold/warm sets do not form a disjoint catalog partition")
    if set(text_lengths) != catalog:
        raise ValueError("Text/path catalogs differ")
    if len({tuple(path) for path in paths.values()}) != len(paths):
        raise ValueError("Baseline item paths are not unique")

    source_frequency = frequency_map(source_sequences)
    filtered_train_frequency = frequency_map(
        (user, items[:-2]) for user, items in filtered_sequences
    )
    eligible_warm = {item for item in warm if filtered_train_frequency[item] > 0}
    buckets = log_frequency_buckets(source_frequency, catalog, args.frequency_buckets)
    pseudo_cold, selection = select_pseudo_cold_items(
        eligible_warm, real_cold, buckets, args.pseudo_fraction, args.seed
    )
    retained_warm = warm - pseudo_cold
    held, student_sequences, leakage = audit_filtered_training(
        filtered_sequences, pseudo_cold, real_cold, args.max_history
    )

    leakage_keys = [key for key in leakage if key.endswith("_leaks")]
    leakage_free = all(leakage[key] == 0 for key in leakage_keys)
    held_coverage_ok = leakage.get("held_events", 0) > 0 and leakage.get("held_unique_items", 0) > 0
    verdict = "PASS_STAGE14_1A_CPU_DATA_AUDIT" if leakage_free and held_coverage_ok else "FAIL_STAGE14_1A_CPU_DATA_AUDIT"

    ground_truth_dir = output_dir / "held_ground_truth_DO_NOT_USE_FOR_TRAINING"
    ground_truth_dir.mkdir(exist_ok=False)
    with (ground_truth_dir / "pseudo_cold_events.jsonl").open("w", encoding="utf-8") as handle:
        for row in held:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    student_dir = output_dir / "student_readable"
    student_dir.mkdir(exist_ok=False)
    with (student_dir / "filtered_train_sequences.jsonl").open("w", encoding="utf-8") as handle:
        for row in student_sequences:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "pseudo_cold_items.txt").write_text("\n".join(sorted(pseudo_cold)) + "\n")
    (output_dir / "retained_warm_items.txt").write_text("\n".join(sorted(retained_warm)) + "\n")

    hashes = {role: sha256_file(path) for role, path in inputs.items()}
    atomic_json(output_dir / "input_file_sha256.json", hashes)
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened_inputs": [
                {"role": role, "path": str(path), "sha256": hashes[role]}
                for role, path in inputs.items()
            ],
            "test_inputs_opened": False,
        },
    )
    config = {
        "experiment_id": "GRAM_PHASE14_STAGE14_1A_PSEUDO_COLD_CPU_AUDIT_TOYS",
        "dataset_dir": str(dataset_dir),
        "source_sequences": str(source_sequences_path),
        "item_path_file": str(item_path_file),
        "pseudo_fraction": args.pseudo_fraction,
        "frequency_buckets": args.frequency_buckets,
        "seed": args.seed,
        "max_history": args.max_history,
        "split": "train_only_pseudo_cold",
        "test_opened": False,
        "student_forbidden_inputs": [
            str(ground_truth_dir / "pseudo_cold_events.jsonl"),
            str(inputs["source_sequences"]),
        ],
    }
    atomic_json(output_dir / "config.json", config)
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "verdict": verdict,
        "selection": selection,
        "n_pseudo_cold": len(pseudo_cold),
        "n_retained_warm": len(retained_warm),
        "leakage_audit": leakage,
        "leakage_free": leakage_free,
        "profiles": {
            "real_cold": item_profile(real_cold, source_frequency, paths, text_lengths, retained_warm),
            "pseudo_cold": item_profile(pseudo_cold, source_frequency, paths, text_lengths, retained_warm),
            "retained_warm": item_profile(retained_warm, source_frequency, paths, text_lengths, retained_warm),
        },
        "held_ground_truth_path": str(
            (ground_truth_dir / "pseudo_cold_events.jsonl").relative_to(output_dir)
        ),
        "student_train_sequences_path": str(
            (student_dir / "filtered_train_sequences.jsonl").relative_to(output_dir)
        ),
        "student_training_must_not_read_held_ground_truth": True,
        "test_opened": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "catalog_known": True,
            "metadata_available": True,
            "actual_cold_interactions_used_for_training": False,
            "pseudo_cold_ground_truth_used_for_training": False,
            "source_sequences_use": "frequency-stratum audit and deterministic split construction only",
            "student_sequences_source": str(inputs["filtered_sequences"]),
        },
    )
    atomic_json(
        output_dir / "status.json",
        {
            "experiment_id": config["experiment_id"],
            "status": "completed" if verdict.startswith("PASS") else "failed",
            "stage": "stage14_1a_cpu_data_audit",
            "reason": verdict,
            "updated_at_epoch": time.time(),
            "automatic_retry": False,
            "test_opened": False,
            "summary_path": str((output_dir / "summary.json").resolve()),
        },
    )
    (output_dir / "run.log").write_text(
        f"{verdict}\nselected={len(pseudo_cold)} held_events={len(held)} "
        f"held_unique_items={leakage.get('held_unique_items', 0)}\n"
    )
    print(json.dumps({"verdict": verdict, "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
