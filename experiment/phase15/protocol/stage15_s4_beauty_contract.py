#!/usr/bin/env python3
"""Validate the frozen Stage15 S4 Beauty contract without opening test data."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE14_PROTOCOL = REPO_ROOT / "experiment" / "phase14" / "protocol"
if str(PHASE14_PROTOCOL) not in sys.path:
    sys.path.insert(0, str(PHASE14_PROTOCOL))

from item_level_eval import atomic_json  # noqa: E402
from oracle_prefix_probe import CollatorGRAM  # noqa: E402
from r2pd_pseudo_cold_screen import collator_args, load_paths  # noqa: E402

from common_adapter import (  # noqa: E402
    TrainTransition,
    read_projected_sequences,
    sha256_file,
    train_only_sequences,
)
from toys_b2_verifier_probe_smoke import _select_probe_transitions  # noqa: E402
from toys_s3a_admission import canonical_target_ids, encoded_catalog  # noqa: E402
from toys_s3b_full_validation import selected_train_rows  # noqa: E402


BACKBONE_PATH = REPO_ROOT / "artifacts" / "phase14" / "m2" / "pretrained" / "t5-small"


EXPECTED_COMMON = {
    "events": 10655,
    "catalog_items": 12101,
    "cold_items": 6052,
    "warm_items": 6049,
    "path_lengths": {"7": 11668, "8": 433},
    "lexical_positions": 8,
    "beam_size": 50,
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20260822,
    "seed": 1502,
    "similar_top_k": 10,
}

EXPECTED_B2 = {
    "train_transitions": 4096,
    "drafter_epochs": 2,
    "drafter_batch_size": 128,
    "drafter_learning_rate": 0.001,
    "draft_size": 10,
    "draft_rounds": 5,
    "verifier_threshold": -1.6,
    "candidate_chunk_size": 10,
}

EXPECTED_B3 = {
    "train_transitions": 4096,
    "contexts_per_pseudo_cold": 10,
    "covariance_transitions": 256,
    "covariance_long_path_minimum": 32,
    "covariance_batch_size": 32,
    "requests_per_position": 4,
    "z_steps": 30,
    "request_selection_rule": "catalog-only legal branching factor >=2; existing SHA rank and distinct-cold rule unchanged",
    "target_token_alignment": "canonical_catalog_token_ids_plus_eos",
}


def read_nonempty_set(path: Path) -> set[str]:
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not values:
        raise ValueError(f"Empty set file: {path}")
    return values


def validate_contract_payload(payload: dict) -> None:
    if payload.get("stage") != "S15-4" or payload.get("domain") != "Beauty_cold50":
        raise ValueError("S15-4 Beauty identity drift")
    if payload.get("test_read") is not False or payload.get("automatic_retry") is not False:
        raise ValueError("S15-4 safety flags drift")
    if payload.get("common") != EXPECTED_COMMON:
        raise ValueError(f"S15-4 common contract drift: {payload.get('common')}")
    if payload.get("b2") != EXPECTED_B2:
        raise ValueError(f"S15-4 B2 contract drift: {payload.get('b2')}")
    if payload.get("b3") != EXPECTED_B3:
        raise ValueError(f"S15-4 B3 contract drift: {payload.get('b3')}")
    if payload.get("execution", {}).get("b2_minimum_free_mib") != 16384:
        raise ValueError("S15-4 B2 GPU admission drift")
    if payload.get("execution", {}).get("b3_minimum_free_mib") != 15360:
        raise ValueError("S15-4 B3 GPU admission drift")
    if payload.get("execution", {}).get("hard_timeout_seconds") != 86400:
        raise ValueError("S15-4 hard-timeout drift")


def resolve_frozen_paths(payload: dict) -> dict[str, Path]:
    result = {}
    for name, relative in payload.get("paths", {}).items():
        path = (REPO_ROOT / relative).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"Path escapes repository: {name}={path}") from exc
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Frozen path must be a regular non-symlink file: {name}={path}")
        lowered = str(path.relative_to(REPO_ROOT)).lower()
        if lowered.endswith("/user_sequence.txt") or "predictions_test" in lowered or "metrics_test" in lowered:
            raise ValueError(f"Forbidden test/original input in S15-4 contract: {name}={path}")
        result[name] = path
    required = {
        "projected_sequences",
        "historical_config",
        "checkpoint",
        "frozen_b0_b1_predictions",
        "item_path_file",
        "item_text_file",
        "similar_items_file",
        "item_embeddings",
        "cold_items",
        "warm_items",
        "b1_source_summary",
        "b1_state",
        "s2_preflight_summary",
        "toys_b2_summary",
        "toys_b3_summary",
        "toys_b2_admission_summary",
        "toys_b3_admission_summary",
        "toys_b0_parity_summary",
    }
    if set(result) != required:
        raise ValueError(f"Frozen path keys drift: {sorted(set(result) ^ required)}")
    return result


def validate_frozen_code(payload: dict) -> dict[str, str]:
    code_paths = payload.get("code_paths", {})
    expected = payload.get("code_sha256", {})
    if set(code_paths) != set(expected) or not code_paths:
        raise ValueError("Frozen S15-4 code path/hash keys drift")
    observed = {}
    for name, relative in code_paths.items():
        path = (REPO_ROOT / relative).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"Code path escapes repository: {name}={path}") from exc
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Frozen code path must be a regular non-symlink file: {name}={path}")
        observed[name] = sha256_file(path)
    if observed != expected:
        changed = sorted(name for name in observed if observed[name] != expected.get(name))
        raise ValueError(f"Frozen S15-4 code SHA drift: {changed}")
    return observed


def validate_source_gates(paths: dict[str, Path]) -> dict:
    s2 = json.loads(paths["s2_preflight_summary"].read_text(encoding="utf-8"))
    b2 = json.loads(paths["toys_b2_summary"].read_text(encoding="utf-8"))
    b3 = json.loads(paths["toys_b3_summary"].read_text(encoding="utf-8"))
    b2_admission = json.loads(paths["toys_b2_admission_summary"].read_text(encoding="utf-8"))
    b3_admission = json.loads(paths["toys_b3_admission_summary"].read_text(encoding="utf-8"))
    parity = json.loads(paths["toys_b0_parity_summary"].read_text(encoding="utf-8"))
    if s2.get("overall_verdict") != "PASS_S15_2P_AND_TOYS_B0_PROJECTION_PARITY_SMOKE":
        raise ValueError("S15-2 dual-domain preflight Gate is not PASS")
    if b2.get("verdict") != "COMPLETED_S15_3B_TOYS_FULL_VALIDATION":
        raise ValueError("Toys B2 S15-3B is not complete")
    if b2.get("success_labels", {}).get("b2", {}).get("PASS_NATIVE_COLD_RECOVERY") is not True:
        raise ValueError("Toys B2 source label drift")
    if b2.get("success_labels", {}).get("b2", {}).get("PASS_OVER_R2_PARETO") is not False:
        raise ValueError("Toys B2 Pareto label drift")
    if b3.get("verdict") != "COMPLETED_S15_3B_TOYS_B3_FULL_VALIDATION":
        raise ValueError("Toys B3 S15-3B is not complete")
    if b3.get("success_labels", {}).get("b3", {}).get("PASS_NATIVE_COLD_RECOVERY") is not False:
        raise ValueError("Toys B3 source label drift")
    if b2_admission.get("verdict") != "PASS_S15_3A_B2_ITEM_DISJOINT_ADMISSION":
        raise ValueError("Toys B2 admission Gate drift")
    if b3_admission.get("verdict") != "PASS_S15_3A_B2_B3_ITEM_DISJOINT_ADMISSION":
        raise ValueError("Toys B3 exploratory admission Gate drift")
    if parity.get("verdict") != "PASS_B0_PROJECTION_PARITY":
        raise ValueError("Toys B0 projection parity Gate drift")
    return {
        "s2_dual_domain_preflight": s2["overall_verdict"],
        "toys_b2": b2["success_labels"]["b2"],
        "toys_b3_exploratory": b3["success_labels"]["b3"],
        "toys_b2_admission": b2_admission["verdict"],
        "toys_b3_admission": b3_admission["verdict"],
        "toys_b0_parity": parity["verdict"],
    }


def inspect_inputs(payload: dict, paths: dict[str, Path]) -> dict:
    projected = read_projected_sequences(paths["projected_sequences"])
    if len(projected) != EXPECTED_COMMON["events"]:
        raise ValueError("Beauty projected event count drift")
    cold = read_nonempty_set(paths["cold_items"])
    warm = read_nonempty_set(paths["warm_items"])
    if len(cold) != EXPECTED_COMMON["cold_items"] or len(warm) != EXPECTED_COMMON["warm_items"]:
        raise ValueError("Beauty cold/warm universe size drift")
    if cold & warm:
        raise ValueError("Beauty cold/warm universe overlap")
    item_paths = load_paths(paths["item_path_file"])
    if set(item_paths) != cold | warm or len(item_paths) != EXPECTED_COMMON["catalog_items"]:
        raise ValueError("Beauty catalog universe drift")
    path_lengths: dict[str, int] = {}
    for path in item_paths.values():
        path_lengths[str(len(path))] = path_lengths.get(str(len(path)), 0) + 1
    if path_lengths != EXPECTED_COMMON["path_lengths"]:
        raise ValueError(f"Beauty path-length drift: {path_lengths}")
    train = train_only_sequences(projected)
    if any(item in cold for items in train.values() for item in items):
        raise ValueError("Cold item entered Beauty train-only state input")
    if any(item not in item_paths for items in projected.values() for item in items):
        raise ValueError("Unknown item entered Beauty projection")
    positions = set(range(max(map(len, item_paths.values()))))
    if positions != set(range(EXPECTED_COMMON["lexical_positions"])):
        raise ValueError(f"Beauty lexical-position drift: {sorted(positions)}")
    cold_requests_by_position = {
        str(position): sum(len(item_paths[item]) > position for item in cold)
        for position in sorted(positions)
    }
    if any(count < EXPECTED_B3["requests_per_position"] for count in cold_requests_by_position.values()):
        raise ValueError("Beauty lacks four cold requests at a lexical position")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    expected_hashes = payload.get("sha256", {})
    if expected_hashes and hashes != expected_hashes:
        changed = sorted(name for name in hashes if hashes[name] != expected_hashes.get(name))
        raise ValueError(f"Frozen S15-4 input SHA drift: {changed}")
    return {
        "events": len(projected),
        "catalog_items": len(item_paths),
        "cold_items": len(cold),
        "warm_items": len(warm),
        "path_lengths": path_lengths,
        "lexical_positions": sorted(positions),
        "cold_requests_by_position": cold_requests_by_position,
        "train_only_cold_occurrences": 0,
        "input_sha256": hashes,
    }


def inspect_b3_token_alignment(
    paths: dict[str, Path],
    projected: dict[str, list[str]],
    item_paths: dict[str, tuple[str, ...]],
) -> dict:
    """Exercise the frozen 64-row probe labels without loading model weights."""

    if not BACKBONE_PATH.is_dir():
        raise FileNotFoundError(BACKBONE_PATH)
    historical = json.loads(paths["historical_config"].read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(BACKBONE_PATH), local_files_only=True)
    collator = CollatorGRAM(tokenizer, args=collator_args(historical), mode="train")
    encoded_paths = encoded_catalog(tokenizer, item_paths)
    train_rows = selected_train_rows(
        projected,
        argparse.Namespace(
            seed=EXPECTED_COMMON["seed"],
            train_transitions=EXPECTED_B3["train_transitions"],
        ),
    )
    transitions = [
        TrainTransition(
            user_id=row["user_id"],
            history=tuple(row["history"]),
            target=row["target"],
        )
        for row in train_rows
    ]
    selected = _select_probe_transitions(
        transitions,
        path_lengths={item: len(path) for item, path in item_paths.items()},
        sample_size=64,
        long_path_minimum=16,
        seed=EXPECTED_COMMON["seed"],
    )
    target_items = [row.target for row in selected]
    split_target = collator.encode_target_split(
        ["|".join(item_paths[item]) for item in target_items]
    )
    split_labels = split_target["input_ids"].masked_fill(
        ~split_target["attention_mask"].bool(), -100
    )
    canonical_labels = canonical_target_ids(
        target_items=target_items,
        encoded_paths=encoded_paths,
        eos_token_id=tokenizer.eos_token_id,
        device=torch.device("cpu"),
    )

    def active_tokens(labels: torch.Tensor, row: int) -> tuple[int, ...]:
        values = labels[row]
        active = values.ne(-100) & values.ne(tokenizer.eos_token_id)
        return tuple(int(value) for value in values[active].tolist())

    split_mismatch_rows = sum(
        active_tokens(split_labels, index) != encoded_paths[item]
        for index, item in enumerate(target_items)
    )
    canonical_mismatch_rows = sum(
        active_tokens(canonical_labels, index) != encoded_paths[item]
        for index, item in enumerate(target_items)
    )
    canonical_active = canonical_labels.ne(-100) & canonical_labels.ne(
        tokenizer.eos_token_id
    )
    canonical_counts = [
        int(canonical_active[:, position].sum())
        for position in range(canonical_labels.size(1))
        if bool(canonical_active[:, position].any())
    ]
    expected_counts = [
        sum(len(item_paths[item]) > position for item in target_items)
        for position in range(EXPECTED_COMMON["lexical_positions"])
    ]
    if canonical_mismatch_rows != 0 or canonical_counts != expected_counts:
        raise RuntimeError("Canonical B3 probe targets do not match constrained-generation paths")
    return {
        "probe_transitions": len(selected),
        "longest_path_transitions": sum(
            len(item_paths[item]) == EXPECTED_COMMON["lexical_positions"]
            for item in target_items
        ),
        "historical_split_collator_mismatch_rows": split_mismatch_rows,
        "canonical_mismatch_rows": canonical_mismatch_rows,
        "canonical_active_counts_by_position": canonical_counts,
        "canonical_positions": list(range(len(canonical_counts))),
        "target_token_alignment": EXPECTED_B3["target_token_alignment"],
    }


def run(contract_path: Path, output_dir: Path | None) -> dict:
    started = time.time()
    contract_path = contract_path.resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract_payload(payload)
    code_hashes = validate_frozen_code(payload)
    paths = resolve_frozen_paths(payload)
    gates = validate_source_gates(paths)
    inputs = inspect_inputs(payload, paths)
    projected = read_projected_sequences(paths["projected_sequences"])
    item_paths = load_paths(paths["item_path_file"])
    b3_token_alignment = inspect_b3_token_alignment(paths, projected, item_paths)
    summary = {
        "experiment_id": "GRAM_STAGE15_S4_BEAUTY_FROZEN_PREFLIGHT",
        "status": "completed",
        "verdict": "PASS_S15_4_BEAUTY_FROZEN_PREFLIGHT",
        "domain": "Beauty_cold50",
        "source_gates": gates,
        "inputs": inputs,
        "b3_token_alignment": b3_token_alignment,
        "code_sha256": code_hashes,
        "common": payload["common"],
        "b2": payload["b2"],
        "b3": payload["b3"],
        "test_read": False,
        "automatic_retry": False,
        "runtime_seconds": time.time() - started,
        "peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        unexpected = [path.name for path in output_dir.iterdir()]
        if unexpected:
            raise FileExistsError(f"Refusing existing S15-4 preflight artifacts: {unexpected}")
        atomic_json(output_dir / "config.json", payload)
        atomic_json(output_dir / "summary.json", summary)
        atomic_json(output_dir / "input_file_sha256.json", inputs["input_sha256"])
        atomic_json(
            output_dir / "open_file_manifest.json",
            {
                "opened": [str(path.relative_to(REPO_ROOT)) for path in paths.values()],
                "frozen_validation_predictions_content_parsed": False,
                "original_user_sequence_opened": False,
                "test_predictions_opened": False,
                "test_metrics_opened": False,
            },
        )
        atomic_json(
            output_dir / "resource_summary.json",
            {
                "runtime_seconds": summary["runtime_seconds"],
                "peak_cpu_rss_mib": summary["peak_cpu_rss_mib"],
                "gpu_used": False,
            },
        )
        atomic_json(
            output_dir / "status.json",
            {
                "experiment_id": summary["experiment_id"],
                "status": "completed",
                "status_code": "COMPLETED",
                "stage": "finished",
                "reason": summary["verdict"],
                "exit_code": 0,
                "test_read": False,
                "automatic_retry": False,
                "summary_path": str((output_dir / "summary.json").relative_to(REPO_ROOT)),
            },
        )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.contract, args.output_dir)


if __name__ == "__main__":
    main()
