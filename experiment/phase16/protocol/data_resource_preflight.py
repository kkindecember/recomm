#!/usr/bin/env python3
"""Build sealed Stage16 train/internal-dev splits and freeze workload counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, domain: str, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{domain}|{namespace}|{value}".encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def reject_forbidden(path: Path, denylist: dict[str, Any]) -> None:
    relative = str(path.relative_to(ROOT))
    if path.is_symlink():
        raise ValueError(f"Symlink inputs are forbidden: {relative}")
    if path.name in set(denylist["exact_basenames"]):
        raise ValueError(f"Forbidden basename: {relative}")
    if any(token in relative for token in denylist["path_substrings"]):
        raise ValueError(f"Forbidden path substring: {relative}")


def verify_input(spec: dict[str, str], denylist: dict[str, Any]) -> Path:
    path = (ROOT / spec["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"Missing frozen input: {spec['path']}")
    reject_forbidden(path, denylist)
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise ValueError(f"SHA256 drift for {spec['path']}: expected {spec['sha256']}, got {actual}")
    return path


def read_sequences(path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    seen_users: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"Projected row {number} has fewer than user + two safe items")
        user, items = fields[0], fields[1:]
        if user in seen_users:
            raise ValueError(f"Duplicate projected user: {user}")
        seen_users.add(user)
        # The final projected item is the validation target. Its value is deliberately
        # neither retained nor emitted to any Stage16 artifact.
        rows.append((user, items[:-1]))
    return rows


def read_id_set(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path.relative_to(ROOT)}")
    return set(values)


def read_paths(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    reverse: dict[tuple[str, ...], str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed lexical-path row {number}")
        item, serialized = fields
        lexical = tuple(token for token in serialized.split("|") if token)
        if not lexical or item in result:
            raise ValueError(f"Empty path or duplicate item at lexical-path row {number}")
        if lexical in reverse:
            raise ValueError(f"Lexical collision: {item} and {reverse[lexical]}")
        result[item] = lexical
        reverse[lexical] = item
    return result


def read_metadata(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 2 or fields[0] in result:
            raise ValueError(f"Malformed or duplicate metadata row {number}")
        result[fields[0]] = len(fields) - 1
    return result


def quintile_thresholds(word_counts: dict[str, int]) -> tuple[int, int, int, int]:
    ordered = sorted(word_counts.values())
    if not ordered:
        raise ValueError("Empty metadata catalog")
    return tuple(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * q / 5) - 1)] for q in range(1, 5))


def bucket(value: int, thresholds: tuple[int, int, int, int]) -> int:
    return sum(value > threshold for threshold in thresholds)


def allocate_stratified(
    eligible: set[str],
    reference: set[str],
    strata: dict[str, tuple[int, int]],
    target: int,
    seed: int,
    domain: str,
) -> tuple[list[str], dict[str, Any]]:
    capacity = Counter(strata[item] for item in eligible)
    ref = Counter(strata[item] for item in reference)
    ref_total = sum(ref.values())
    if target <= 0 or target > len(eligible) or ref_total == 0:
        raise ValueError("Invalid pseudo-cold allocation target or empty reference")
    ideal = {key: target * ref.get(key, 0) / ref_total for key in capacity}
    allocated = Counter()
    while sum(allocated.values()) < target:
        available = [key for key, count in capacity.items() if allocated[key] < count]
        if not available:
            raise ValueError("Insufficient eligible warm-item capacity")
        chosen = max(
            available,
            key=lambda key: (
                ideal.get(key, 0.0) - allocated[key],
                ref.get(key, 0),
                -key[0],
                -key[1],
            ),
        )
        allocated[chosen] += 1
    selected: list[str] = []
    for key in sorted(allocated):
        candidates = sorted(
            (item for item in eligible if strata[item] == key),
            key=lambda item: stable_rank(seed, domain, "pseudo", item),
        )
        selected.extend(candidates[: allocated[key]])
    selected.sort(key=lambda item: stable_rank(seed, domain, "pseudo-output", item))
    audit = {
        "target": target,
        "selected": len(selected),
        "capacity_by_stratum": {f"path{key[0]}_textq{key[1]}": capacity[key] for key in sorted(capacity)},
        "reference_by_stratum": {f"path{key[0]}_textq{key[1]}": ref[key] for key in sorted(ref)},
        "selected_by_stratum": {f"path{key[0]}_textq{key[1]}": allocated[key] for key in sorted(allocated)},
    }
    return selected, audit


def build_domain(domain: dict[str, Any], config: dict[str, Any], output: Path) -> dict[str, Any]:
    name = domain["name"]
    denylist = config["denylist"]
    inputs = {key: verify_input(spec, denylist) for key, spec in domain.items() if isinstance(spec, dict) and "path" in spec}
    sequences = read_sequences(inputs["projected_sequence"])
    paths = read_paths(inputs["lexical_paths"])
    metadata = read_metadata(inputs["metadata"])
    cold = read_id_set(inputs["cold_items"])
    warm = read_id_set(inputs["warm_items"])
    catalog = set(paths)
    if cold & warm or cold | warm != catalog:
        raise ValueError(f"Cold/warm partition mismatch in {name}")
    if set(metadata) != catalog:
        raise ValueError(f"Metadata/catalog mismatch in {name}")

    train_items = [item for _, items in sequences for item in items]
    unknown = set(train_items) - catalog
    if unknown:
        raise ValueError(f"Unknown train items in {name}: {sorted(unknown)[:3]}")
    real_cold_train_leaks = set(train_items) & cold
    if real_cold_train_leaks:
        raise ValueError(f"Real-cold item leaked into safe train projection in {name}")

    eligible: set[str] = set()
    for _, items in sequences:
        eligible.update(item for item in items[1:] if item in warm)
    thresholds = quintile_thresholds(metadata)
    strata = {item: (len(paths[item]), bucket(metadata[item], thresholds)) for item in catalog}
    fraction = config["split_policy"]["pseudo_cold_fraction_of_eligible_warm"]
    target = max(1, round(len(eligible) * fraction))
    pseudo_list, allocation = allocate_stratified(eligible, cold, strata, target, config["seed"], name)
    pseudo = set(pseudo_list)

    minimum = config["split_policy"]["minimum_student_sequence_items"]
    retained_sequences: list[tuple[str, list[str]]] = []
    held_events: list[dict[str, Any]] = []
    for user, items in sequences:
        visible_prefix: list[str] = []
        for position, item in enumerate(items):
            if item in pseudo:
                if visible_prefix:
                    held_events.append(
                        {
                            "event_id": stable_rank(config["seed"], name, "held-event", f"{user}|{position}|{item}"),
                            "user_id": user,
                            "source_position": position,
                            "history": visible_prefix[-config["split_policy"]["maximum_history_items"] :],
                            "target_item": item,
                        }
                    )
            else:
                visible_prefix.append(item)
        if len(visible_prefix) >= minimum:
            retained_sequences.append((user, visible_prefix))

    ranked_users = sorted(
        (user for user, _ in retained_sequences),
        key=lambda user: stable_rank(config["seed"], name, "internal-dev-user", user),
    )
    dev_count = max(1, round(len(ranked_users) * config["split_policy"]["internal_dev_user_fraction"]))
    dev_users = set(ranked_users[:dev_count])
    train_rows = [(user, items) for user, items in retained_sequences if user not in dev_users]
    dev_rows = [(user, items) for user, items in retained_sequences if user in dev_users]

    student_items = {item for _, items in retained_sequences for item in items}
    if student_items & pseudo or student_items & cold:
        raise ValueError(f"Student-readable leakage in {name}")
    if {user for user, _ in train_rows} & {user for user, _ in dev_rows}:
        raise ValueError(f"Train/internal-dev user collision in {name}")

    domain_out = output / "splits" / name
    write_lines(domain_out / "pseudo_cold_items.txt", pseudo_list)
    write_lines(domain_out / "retained_warm_items.txt", sorted(warm - pseudo))
    write_lines(
        domain_out / "student_readable" / "interaction_train_sequences.jsonl",
        (json.dumps({"user_id": user, "items": items}, ensure_ascii=False, separators=(",", ":")) for user, items in train_rows),
    )
    write_lines(
        domain_out / "student_readable" / "interaction_internal_dev_sequences.jsonl",
        (json.dumps({"user_id": user, "items": items}, ensure_ascii=False, separators=(",", ":")) for user, items in dev_rows),
    )
    write_lines(
        domain_out / "held_ground_truth_DO_NOT_USE_FOR_TRAINING" / "pseudo_cold_events.jsonl",
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in held_events),
    )

    train_transitions = sum(len(items) - 1 for _, items in train_rows)
    dev_transitions = sum(len(items) - 1 for _, items in dev_rows)
    contexts = len(cold) * config["workload_policy"]["genrecedit_contexts_per_cold_item"]
    position_requests = sum(len(paths[item]) for item in cold) * config["workload_policy"]["genrecedit_contexts_per_cold_item"]
    covariance_rows = min(config["workload_policy"]["genrecedit_mom2_sample_cap"], train_transitions)
    manifest = {
        "domain": name,
        "seed": config["seed"],
        "source_projected_users": len(sequences),
        "source_train_items_after_validation_drop": len(train_items),
        "catalog_items": len(catalog),
        "real_cold_items": len(cold),
        "real_warm_items": len(warm),
        "eligible_pseudo_cold_items": len(eligible),
        "pseudo_cold_items": len(pseudo),
        "pseudo_cold_held_events": len(held_events),
        "train_users": len(train_rows),
        "internal_dev_users": len(dev_rows),
        "train_transitions": train_transitions,
        "internal_dev_transitions": dev_transitions,
        "s_aux_training_examples": train_transitions,
        "s_plus_pretrain_examples": train_transitions,
        "s_plus_finetune_examples": train_transitions,
        "g_full_edit_targets": len(cold),
        "g_full_contexts": contexts,
        "g_full_prefix_next_token_requests": position_requests,
        "g_full_covariance_rows": covariance_rows,
        "lexical_path_length_counts_real_cold": dict(sorted(Counter(len(paths[item]) for item in cold).items())),
        "catalog_text_word_count_quintile_thresholds": thresholds,
        "allocation": allocation,
        "leakage_audit": {
            "validation_target_values_logged": False,
            "test_files_opened": False,
            "real_cold_in_student_items": 0,
            "pseudo_cold_in_student_items": 0,
            "train_internal_dev_user_overlap": 0,
            "pseudo_cold_adaptation_scope_only": True,
        },
        "outputs": {},
    }
    for path in sorted(domain_out.rglob("*")):
        if path.is_file():
            manifest["outputs"][str(path.relative_to(ROOT))] = sha256(path)
    write_json(domain_out / "split_manifest.json", manifest)
    manifest["outputs"][str((domain_out / "split_manifest.json").relative_to(ROOT))] = sha256(domain_out / "split_manifest.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    if (output / "data_preflight_summary.json").exists():
        raise SystemExit("Refusing to overwrite an existing S16-1 data preflight summary")
    output.mkdir(parents=True, exist_ok=True)

    manifests = [build_domain(domain, config, output) for domain in config["domains"]]
    input_hashes = {
        spec["path"]: spec["sha256"]
        for domain in config["domains"]
        for spec in domain.values()
        if isinstance(spec, dict) and "path" in spec
    }
    generated = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "generated_at_utc": generated,
        "test_read": False,
        "network_used": False,
    }
    write_json(output / "config.json", config)
    write_json(output / "input_file_sha256.json", {**common, "files": input_hashes})
    write_json(
        output / "open_file_manifest.json",
        {
            **common,
            "opened_files": sorted(input_hashes),
            "denylist": config["denylist"],
            "test_read": False,
            "note": "The last validation-position value in each safe projected row was discarded in memory and never serialized.",
        },
    )
    write_json(
        output / "data_provenance.json",
        {
            **common,
            "projection_origin": "Stage15 redacted train+validation projections; original user_sequence.txt remained sealed",
            "stage16_transform": "drop validation position, select adaptation-level pseudo-cold, remove it from student data, user-disjoint train/dev split",
            "pseudo_cold_scope": config["split_policy"]["adaptation_scope_note"],
            "domains": [row["domain"] for row in manifests],
        },
    )
    write_json(
        output / "workload_counts.json",
        {
            **common,
            "counting_semantics": {
                "specgr_example": "one train-only next-item transition",
                "genrecedit_target": "one frozen real-cold catalog item",
                "genrecedit_context": "ten train-derived contexts per real-cold item",
                "genrecedit_request": "one context times one non-EOS lexical-path position",
                "covariance_row": "one train-only next-item transition, capped at 400000",
            },
            "domains": manifests,
        },
    )
    write_json(
        output / "data_preflight_summary.json",
        {
            **common,
            "verdict": "PASS_S16_1_DATA_LEAKAGE_PREFLIGHT_CPU",
            "domains_passed": len(manifests),
            "test_read": False,
            "resource_probe_pending": True,
            "final_gate_pending": "PASS_S16_1_DATA_LEAKAGE_RESOURCE_PREFLIGHT",
        },
    )
    print("PASS_S16_1_DATA_LEAKAGE_PREFLIGHT_CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
