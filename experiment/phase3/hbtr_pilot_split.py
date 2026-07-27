#!/usr/bin/env python3
"""Create locked, user-disjoint HBTR pilot splits from training-only signals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEED = 2023
TRAIN_FRACTION = 0.10
VALIDATION_USERS = 2048


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_sequences(path: Path) -> dict[str, list[str]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            user, *items = line.split()
            if len(items) >= 4:
                result[user] = items
    return result


def history_bin(length: int) -> str:
    if length <= 5:
        return "1-5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    return "21+"


def training_popularity(sequences: dict[str, list[str]]) -> Counter:
    counts: Counter = Counter()
    for items in sequences.values():
        counts.update(items[:-2])
    return counts


def head_items(popularity: Counter) -> set[str]:
    ordered = sorted(popularity, key=lambda item: (-popularity[item], item))
    return set(ordered[: max(1, math.ceil(len(ordered) * 0.20))])


def stable_order(dataset: str, user: str) -> str:
    return hashlib.sha256(f"{SEED}:{dataset}:{user}".encode()).hexdigest()


def largest_remainder_allocation(
    sizes: dict[str, int], total: int
) -> dict[str, int]:
    if total < 0 or total > sum(sizes.values()):
        raise ValueError("requested allocation is outside available population")
    population = sum(sizes.values())
    if population == 0:
        return {key: 0 for key in sizes}
    quotas = {key: total * size / population for key, size in sizes.items()}
    allocation = {key: min(size, int(math.floor(quotas[key]))) for key, size in sizes.items()}
    remaining = total - sum(allocation.values())
    order = sorted(
        sizes,
        key=lambda key: (-(quotas[key] - math.floor(quotas[key])), key),
    )
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < sizes[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("could not complete allocation")
    return allocation


def create_split(dataset: str, output_root: Path) -> dict:
    sequence_path = ROOT / "GRAM/rec_datasets" / dataset / "user_sequence.txt"
    sequences = read_sequences(sequence_path)
    popularity = training_popularity(sequences)
    heads = head_items(popularity)
    strata: dict[str, list[str]] = defaultdict(list)
    for user, items in sequences.items():
        training_target = items[-3]
        group = "head" if training_target in heads else "tail"
        stratum = f"{group}__{history_bin(len(items[:-3]))}"
        strata[stratum].append(user)
    for stratum in strata:
        strata[stratum].sort(key=lambda user: stable_order(dataset, user))

    train_total = int(round(len(sequences) * TRAIN_FRACTION))
    train_counts = largest_remainder_allocation(
        {key: len(users) for key, users in strata.items()}, train_total
    )
    train_users = []
    remaining_by_stratum = {}
    for key, users in strata.items():
        count = train_counts[key]
        train_users.extend(users[:count])
        remaining_by_stratum[key] = users[count:]

    validation_counts = largest_remainder_allocation(
        {key: len(users) for key, users in remaining_by_stratum.items()},
        VALIDATION_USERS,
    )
    validation_users = []
    for key, users in remaining_by_stratum.items():
        validation_users.extend(users[: validation_counts[key]])

    train_users.sort()
    validation_users.sort()
    if set(train_users) & set(validation_users):
        raise ValueError("pilot train/validation users overlap")
    if len(train_users) != train_total or len(validation_users) != VALIDATION_USERS:
        raise ValueError("pilot split size mismatch")

    out_dir = output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_users.txt"
    validation_path = out_dir / "validation_users.txt"
    train_path.write_text("".join(f"{user}\n" for user in train_users))
    validation_path.write_text("".join(f"{user}\n" for user in validation_users))

    strata_path = out_dir / "strata.csv"
    with strata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stratum", "population", "train", "validation", "remaining"],
        )
        writer.writeheader()
        for key in sorted(strata):
            writer.writerow(
                {
                    "stratum": key,
                    "population": len(strata[key]),
                    "train": train_counts[key],
                    "validation": validation_counts[key],
                    "remaining": len(strata[key]) - train_counts[key] - validation_counts[key],
                }
            )

    manifest = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "plan",
            "origin_date": "2026-07-22",
            "verification_status": "ANALYZED",
            "version_label": "hbtr_pilot_split_v1",
            "design_status": "LOCKED_ONCE",
        },
        "dataset": dataset,
        "seed": SEED,
        "population_users": len(sequences),
        "train_users": len(train_users),
        "validation_users": len(validation_users),
        "overlap_users": 0,
        "training_fraction_realized": len(train_users) / len(sequences),
        "stratification": "training-only target head/tail x training-anchor history length",
        "history_bins": ["1-5", "6-10", "11-20", "21+"],
        "training_anchor": "target=sequence[-3], history=sequence[:-3]",
        "popularity_source": "sequence[:-2]",
        "validation_target_read_for_split": False,
        "test_target_read": False,
        "input_sha256": sha256_file(sequence_path),
        "train_users_sha256": sha256_file(train_path),
        "validation_users_sha256": sha256_file(validation_path),
        "strata_sha256": sha256_file(strata_path),
        "strata": {
            key: {
                "population": len(strata[key]),
                "train": train_counts[key],
                "validation": validation_counts[key],
            }
            for key in sorted(strata)
        },
    }
    with (out_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/phase3/hbtr_pilot_splits",
    )
    args = parser.parse_args()
    results = {dataset: create_split(dataset, args.output_root) for dataset in ("Toys", "Beauty")}
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
