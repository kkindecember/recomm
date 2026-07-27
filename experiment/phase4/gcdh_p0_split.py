#!/usr/bin/env python3
"""Create the locked, user-disjoint GCDH P0 25%/4096 splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sequences(path: Path) -> dict[str, list[str]]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            user, *items = line.split()
            if len(items) >= 4:
                rows[user] = items
    return rows


def history_bin(length: int) -> str:
    if length <= 5:
        return "1-5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    return "21+"


def largest_remainder(sizes: dict[str, int], total: int) -> dict[str, int]:
    population = sum(sizes.values())
    if total < 0 or total > population:
        raise ValueError("allocation outside population")
    quotas = {key: total * value / population for key, value in sizes.items()}
    result = {key: min(sizes[key], int(math.floor(value))) for key, value in quotas.items()}
    remaining = total - sum(result.values())
    order = sorted(
        sizes,
        key=lambda key: (-(quotas[key] - math.floor(quotas[key])), key),
    )
    while remaining:
        progressed = False
        for key in order:
            if result[key] < sizes[key]:
                result[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("allocation stalled")
    return result


def create_split(dataset: str, config: dict, output_root: Path) -> dict:
    path = ROOT / "GRAM/rec_datasets" / dataset / "user_sequence.txt"
    sequences = read_sequences(path)
    counts = Counter(item for values in sequences.values() for item in values[:-2])
    ordered_items = sorted(counts, key=lambda item: (-counts[item], item))
    head = set(ordered_items[: max(1, math.ceil(len(ordered_items) * 0.2))])
    strata: dict[str, list[str]] = defaultdict(list)
    salt = config["split"]["salt"]
    for user, items in sequences.items():
        anchor = items[-3]
        group = "head" if anchor in head else "tail"
        key = f"{group}__{history_bin(len(items[:-3]))}"
        strata[key].append(user)
    for key in strata:
        strata[key].sort(
            key=lambda user: hashlib.sha256(
                f"{salt}|{dataset}|{user}".encode()
            ).hexdigest()
        )
    train_total = int(round(len(sequences) * config["split"]["training_user_fraction"]))
    train_alloc = largest_remainder(
        {key: len(users) for key, users in strata.items()}, train_total
    )
    remaining = {}
    train_users = []
    for key, users in strata.items():
        train_users.extend(users[: train_alloc[key]])
        remaining[key] = users[train_alloc[key] :]
    validation_alloc = largest_remainder(
        {key: len(users) for key, users in remaining.items()},
        int(config["split"]["validation_users"]),
    )
    validation_users = []
    for key, users in remaining.items():
        validation_users.extend(users[: validation_alloc[key]])
    train_users.sort()
    validation_users.sort()
    if set(train_users) & set(validation_users):
        raise AssertionError("train/validation user overlap")
    output = output_root / dataset
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train_users.txt"
    validation_path = output / "validation_users.txt"
    train_path.write_text("".join(f"{user}\n" for user in train_users))
    validation_path.write_text("".join(f"{user}\n" for user in validation_users))
    strata_path = output / "strata.csv"
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
                    "train": train_alloc[key],
                    "validation": validation_alloc[key],
                    "remaining": len(strata[key])
                    - train_alloc[key]
                    - validation_alloc[key],
                }
            )
    manifest = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "salt": salt,
        "population_users": len(sequences),
        "train_users": len(train_users),
        "validation_users": len(validation_users),
        "overlap_users": 0,
        "training_fraction_realized": len(train_users) / len(sequences),
        "stratification": config["split"]["stratification"],
        "validation_target_read_for_split": False,
        "test_target_read": False,
        "sequence_sha256": sha256(path),
        "train_users_sha256": sha256(train_path),
        "validation_users_sha256": sha256(validation_path),
        "strata_sha256": sha256(strata_path),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = {
        dataset: create_split(dataset, config, args.output_root)
        for dataset in config["datasets"]
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
