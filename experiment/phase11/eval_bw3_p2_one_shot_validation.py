#!/usr/bin/env python3
"""Apply frozen P1C admission gates to the one-shot t=-2 validation split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE9 = REPO_ROOT / "experiment/phase9"
PHASE11 = REPO_ROOT / "experiment/phase11"
for directory in (PHASE9, PHASE11):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval_cf0_b3_beamfusion import load_users, metrics_from_ranks, score_item_head  # noqa: E402
from eval_p9x_fixed_pcrf import load_catalog  # noqa: E402
from train_bw3_listwise_admission import (  # noqa: E402
    FEATURES,
    anchor_apply,
    frozen_pcrf_joint,
    load_fresh_beams,
)


EXPERIMENT_ID = "GRAM_PHASE11_BW3_P2_LISTWISE_ADMISSION_ONE_SHOT_VALIDATION_V1"
DATASETS = {
    "Toys": {
        "item_index": "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt",
        "item_head": "artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt",
    },
    "Beauty": {
        "item_index": "item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
        "item_head": "artifacts/phase9/p9x_beauty_item_head/best_item_head.pt",
    },
}
EXPECTED_USERS = 512
TARGET_OFFSET = 2
MAX_ADMISSIONS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_status(path: Path, **changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    atomic_json(path, payload)
    return payload


def verify_files(root: Path, files: dict[str, str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing locked file: {relative}")
        digest = sha256(path)
        if digest != expected:
            raise ValueError(f"SHA mismatch: {relative}: {digest} != {expected}")
        actual[relative] = digest
    return actual


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected experiment_id")
    if config.get("execution_enabled") is not True:
        raise PermissionError("P2 formal execution is disabled")
    if config.get("decision_status") != "AUTHORIZED_P2_ONE_SHOT_FORMAL_RUN":
        raise PermissionError("P2 formal decision status is not authorized")
    scope = config["scientific_scope"]
    if scope.get("datasets") != ["Toys", "Beauty"]:
        raise ValueError("dataset order must be frozen as Toys then Beauty")
    if scope.get("validation_target_offset") != TARGET_OFFSET:
        raise ValueError("validation target offset must be 2")
    if scope.get("users_per_domain") != EXPECTED_USERS:
        raise ValueError("users_per_domain must be 512")
    if scope.get("features") != FEATURES:
        raise ValueError("feature schema differs from P1C")
    if scope.get("max_admissions") != MAX_ADMISSIONS:
        raise ValueError("max_admissions must be 3")
    if scope.get("automatic_retry") is not False:
        raise ValueError("automatic_retry must be false")
    return config


def validate_gate(gate: dict[str, Any], dataset: str) -> dict[str, Any]:
    if gate.get("dataset") != dataset:
        raise ValueError(f"{dataset}: gate dataset mismatch")
    if gate.get("experiment_id") != "GRAM_PHASE11_BW3_P1C_LISTWISE_ADMISSION_CORRECTION_V1":
        raise ValueError(f"{dataset}: gate parent experiment mismatch")
    if gate.get("feature_schema") != FEATURES:
        raise ValueError(f"{dataset}: gate feature schema mismatch")
    if gate.get("selected_margin") != 0.0:
        raise ValueError(f"{dataset}: selected margin is not frozen at 0.0")
    if gate.get("max_admissions") != MAX_ADMISSIONS:
        raise ValueError(f"{dataset}: max admissions mismatch")
    if gate.get("objective") != "per_user_listwise_cross_entropy_with_fixed_reject_logit_zero":
        raise ValueError(f"{dataset}: objective mismatch")
    arrays = {
        "weight": np.asarray(gate["weight"], dtype=np.float64),
        "mean": np.asarray(gate["feature_mean"], dtype=np.float64),
        "std": np.asarray(gate["feature_std"], dtype=np.float64),
    }
    if any(value.shape != (len(FEATURES),) for value in arrays.values()):
        raise ValueError(f"{dataset}: malformed gate vector")
    if not np.isfinite(np.concatenate([*arrays.values(), [float(gate["bias"])]] )).all():
        raise ValueError(f"{dataset}: non-finite gate")
    if np.any(arrays["std"] <= 0):
        raise ValueError(f"{dataset}: invalid feature std")
    return {**arrays, "bias": float(gate["bias"]), "margin": 0.0}


def rank_from_top(target: int, top: list[int]) -> int:
    return top.index(target) + 1 if target in top else 201


def score_and_admit(event: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    scored: list[tuple[float, int]] = []
    for candidate in event["expansion"]:
        feature = np.asarray(candidate["features"], dtype=np.float64)
        logit = float(((feature - model["mean"]) / model["std"]) @ model["weight"] + model["bias"])
        if not math.isfinite(logit):
            raise ValueError("non-finite admission logit")
        if logit >= model["margin"]:
            scored.append((logit, int(candidate["candidate_id"])))
    admitted_pairs = sorted(scored, key=lambda pair: (-pair[0], pair[1]))[:MAX_ADMISSIONS]
    admitted = [candidate_id for _, candidate_id in admitted_pairs]
    final_top10 = event["base_top10"][: 10 - len(admitted)] + admitted
    fallback = not admitted
    if fallback and final_top10 != event["base_top10"]:
        raise AssertionError("fallback identity failed")
    return {
        "admitted": admitted,
        "admitted_logits": [logit for logit, _ in admitted_pairs],
        "final_top10": final_top10,
        "final_rank": rank_from_top(event["target"], final_top10),
        "fallback": fallback,
    }


def build_event(
    user: str,
    target: int,
    target_frequency: int,
    q1: int,
    base: dict[str, Any],
    wide: dict[str, Any],
) -> dict[str, Any]:
    reliability = 1.0 - float(np.mean(base["candidate_frequencies"][:10] <= q1))
    base_joint = frozen_pcrf_joint(
        base["seq"], base["cf"], base["candidate_frequencies"], reliability
    )
    base_order = np.argsort(-base_joint, kind="stable")
    base_top10 = [base["candidate_ids"][index] for index in base_order[:10]]
    base_rank = rank_from_top(target, base_top10)

    base_pop_raw = np.log1p(base["candidate_frequencies"])
    wide_seq_anchor = anchor_apply(wide["seq"], base["seq"])
    wide_item_anchor = anchor_apply(wide["cf"], base["cf"])
    wide_pop_raw = np.log1p(wide["candidate_frequencies"])
    wide_pop_anchor = anchor_apply(wide_pop_raw, base_pop_raw)
    base_set = set(base["candidate_ids"])
    expansion: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(wide["candidate_ids"]):
        if candidate_id in base_set:
            continue
        feature = np.asarray(
            [
                wide["seq"][index],
                wide_seq_anchor[index],
                wide["cf"][index],
                wide_item_anchor[index],
                wide_pop_raw[index],
                wide_pop_anchor[index],
                (index + 1) / 200.0,
                reliability,
                wide_item_anchor[index] - 0.5 * wide_pop_anchor[index],
            ],
            dtype=np.float64,
        )
        if feature.shape != (len(FEATURES),) or not np.isfinite(feature).all():
            raise ValueError(f"{user}: invalid expansion feature")
        expansion.append({"candidate_id": candidate_id, "features": feature})
    return {
        "user": user,
        "target": target,
        "target_frequency": target_frequency,
        "q1": q1,
        "base_top10": base_top10,
        "base_rank": base_rank,
        "in_beam50": target in base_set,
        "in_beam200": target in set(wide["candidate_ids"]),
        "expansion": expansion,
        "reliability": reliability,
    }


def prepare_domain(dataset: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = config["domain_inputs"][dataset]
    beam50 = load_fresh_beams(REPO_ROOT / paths["beam50"], 50)
    beam200 = load_fresh_beams(REPO_ROOT / paths["beam200"], 200)
    if set(beam50) != set(beam200) or len(beam50) != EXPECTED_USERS:
        raise ValueError(f"{dataset}: expected identical fixed 512-user beam sets")

    dataset_config = DATASETS[dataset]
    data_dir = REPO_ROOT / "GRAM/rec_datasets" / dataset
    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(data_dir, dataset_config["item_index"])
    id_to_lexical = {raw_to_id[raw]: lexical for raw, lexical in raw_to_lexical.items()}
    users = load_users(data_dir, raw_to_id)
    frequencies: Counter[int] = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-TARGET_OFFSET])
    all_target_frequencies = sorted(frequencies[sequence[-TARGET_OFFSET]] for sequence in users.values())
    q1 = int(all_target_frequencies[len(all_target_frequencies) // 4])

    base_records: list[dict[str, Any]] = []
    wide_records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for user in sorted(beam50):
        if user not in users or len(users[user]) < TARGET_OFFSET:
            raise ValueError(f"{dataset} {user}: missing sequence")
        sequence = users[user]
        target = sequence[-TARGET_OFFSET]
        expected_gold = id_to_lexical[target]
        if beam50[user]["gold"] != expected_gold or beam200[user]["gold"] != expected_gold:
            raise ValueError(f"{dataset} {user}: frozen-beam gold mismatch")
        history = sequence[max(0, len(sequence) - TARGET_OFFSET - 20) : -TARGET_OFFSET]
        base_ids = [lexical_to_id[value] for value in beam50[user]["candidates"]]
        wide_ids = [lexical_to_id[value] for value in beam200[user]["candidates"]]
        base_records.append(
            {
                "history": history,
                "candidate_ids": base_ids,
                "seq": np.asarray(beam50[user]["seq"], dtype=np.float64),
                "candidate_frequencies": np.asarray([frequencies[item] for item in base_ids], dtype=np.float64),
            }
        )
        wide_records.append(
            {
                "history": history,
                "candidate_ids": wide_ids,
                "seq": np.asarray(beam200[user]["seq"], dtype=np.float64),
                "candidate_frequencies": np.asarray([frequencies[item] for item in wide_ids], dtype=np.float64),
            }
        )
        metadata.append({"user": user, "target": target, "target_frequency": frequencies[target]})

    item_head = REPO_ROOT / dataset_config["item_head"]
    score_item_head(base_records, item_head, 512)
    score_item_head(wide_records, item_head, 512)
    events = [
        build_event(meta["user"], meta["target"], meta["target_frequency"], q1, base, wide)
        for meta, base, wide in zip(metadata, base_records, wide_records)
    ]
    return events, {"q1": q1, "users": len(events), "item_head_sha256": sha256(item_head)}


def paired_hit_bootstrap(base_ranks: np.ndarray, final_ranks: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    if replicates <= 0:
        return {"replicates": 0, "lower": None, "upper": None}
    delta = (final_ranks <= 10).astype(np.float64) - (base_ranks <= 10).astype(np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        samples[index] = delta[rng.integers(0, len(delta), size=len(delta))].mean()
    return {
        "replicates": replicates,
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def evaluate_events(
    dataset: str,
    events: list[dict[str, Any]],
    model: dict[str, Any],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    applied = [score_and_admit(event, model) for event in events]
    base_ranks = np.asarray([event["base_rank"] for event in events], dtype=np.int64)
    final_ranks = np.asarray([row["final_rank"] for row in applied], dtype=np.int64)
    tail = np.asarray([event["target_frequency"] <= event["q1"] for event in events], dtype=bool)
    base = metrics_from_ranks(base_ranks)
    candidate = metrics_from_ranks(final_ranks)
    base_tail = metrics_from_ranks(base_ranks[tail])
    candidate_tail = metrics_from_ranks(final_ranks[tail])
    fallback_identity = all(
        not result["fallback"] or result["final_top10"] == event["base_top10"]
        for event, result in zip(events, applied)
    )
    finite = bool(
        np.isfinite(
            [base["Hit@10"], base["NDCG@10"], candidate["Hit@10"], candidate["NDCG@10"]]
        ).all()
    )
    per_user: list[dict[str, Any]] = []
    for event, result in zip(events, applied):
        base_hit = event["base_rank"] <= 10
        final_hit = result["final_rank"] <= 10
        per_user.append(
            {
                "user": event["user"],
                "target_item_id": event["target"],
                "base_rank": event["base_rank"],
                "final_rank": result["final_rank"],
                "target_in_beam50": int(event["in_beam50"]),
                "target_in_beam200": int(event["in_beam200"]),
                "expansion_pool_size": len(event["expansion"]),
                "admitted_item_ids": "||".join(map(str, result["admitted"])),
                "admitted_logits": "||".join(f"{value:.9g}" for value in result["admitted_logits"]),
                "admission_count": len(result["admitted"]),
                "promotion": int(not base_hit and final_hit),
                "regression": int(base_hit and not final_hit),
                "fallback": int(result["fallback"]),
                "target_frequency": event["target_frequency"],
                "target_group": "tail" if event["target_frequency"] <= event["q1"] else "non_tail",
                "reliability": event["reliability"],
            }
        )
    promotions = int(np.sum((base_ranks > 10) & (final_ranks <= 10)))
    regressions = int(np.sum((base_ranks <= 10) & (final_ranks > 10)))
    summary = {
        "dataset": dataset,
        "users": len(events),
        "base": base,
        "candidate": candidate,
        "hit10_delta": candidate["Hit@10"] - base["Hit@10"],
        "ndcg10_delta": candidate["NDCG@10"] - base["NDCG@10"],
        "tail": {"users": int(tail.sum()), "base": base_tail, "candidate": candidate_tail},
        "tail_hit10_delta": candidate_tail["Hit@10"] - base_tail["Hit@10"],
        "admissions": sum(len(row["admitted"]) for row in applied),
        "admission_users": sum(bool(row["admitted"]) for row in applied),
        "fallback_users": sum(row["fallback"] for row in applied),
        "promotions": promotions,
        "regressions": regressions,
        "unchanged": len(events) - promotions - regressions,
        "coverage": {
            "target_in_beam50": sum(event["in_beam50"] for event in events),
            "target_in_beam200": sum(event["in_beam200"] for event in events),
            "target_expansion_only": sum(event["in_beam200"] and not event["in_beam50"] for event in events),
            "target_outside_union": sum(not event["in_beam50"] and not event["in_beam200"] for event in events),
        },
        "paired_hit10_delta_95ci": paired_hit_bootstrap(
            base_ranks, final_ranks, bootstrap_replicates, bootstrap_seed
        ),
        "integrity": {
            "all_512_users": len(events) == EXPECTED_USERS,
            "fallback_identity": fallback_identity,
            "finite": finite,
            "max_three_admissions": all(len(row["admitted"]) <= MAX_ADMISSIONS for row in applied),
        },
    }
    return summary, per_user


def p2_scientific_gate(domain_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if [row["dataset"] for row in domain_summaries] != ["Toys", "Beauty"]:
        raise ValueError("P2 gate requires Toys and Beauty in frozen order")
    hit_deltas = [row["hit10_delta"] for row in domain_summaries]
    checks = {
        "both_hit10_nondegrade": all(value >= 0 for value in hit_deltas),
        "one_domain_hit10_at_least_0_002": any(value >= 0.002 for value in hit_deltas),
        "mean_hit10_at_least_0_001": float(np.mean(hit_deltas)) >= 0.001,
        "both_ndcg10_at_least_minus_0_001": all(row["ndcg10_delta"] >= -0.001 for row in domain_summaries),
        "both_tail_hit10_nondegrade": all(row["tail_hit10_delta"] >= 0 for row in domain_summaries),
        "both_admissions_nonzero": all(row["admissions"] > 0 for row in domain_summaries),
        "both_promotions_at_least_regressions": all(row["promotions"] >= row["regressions"] for row in domain_summaries),
        "both_domain_integrity": all(all(row["integrity"].values()) for row in domain_summaries),
    }
    passed = all(checks.values())
    return {
        "status": "passed_scientific_gate_awaiting_resource_audit" if passed else "failed_scientific_gate",
        "checks": checks,
        "mean_hit10_delta": float(np.mean(hit_deltas)),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty per-user output")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def atomic_reveal(output_dir: Path, payload: dict[str, Any], per_user: dict[str, list[dict[str, Any]]]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.transaction.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"transaction directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for summary in payload["datasets"]:
            domain_dir = temporary / summary["dataset"]
            domain_dir.mkdir()
            atomic_json(domain_dir / "summary.json", summary)
            write_tsv(domain_dir / "per_user.tsv", per_user[summary["dataset"]])
        atomic_json(temporary / "summary.json", payload)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise RuntimeError("P2 is CPU-only; CUDA_VISIBLE_DEVICES must be empty or -1")
    if torch.cuda.is_available():
        raise RuntimeError("P2 CPU-only assertion failed: CUDA remains visible")
    config = load_config(args.config)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    verified_locks = verify_files(REPO_ROOT, config["input_lock"]["files"])

    update_status(
        args.status_path,
        validation_access_started=True,
        validation_consumed=True,
        results_revealed=False,
        validation_users_expected=EXPECTED_USERS * 2,
        validation_users_processed=0,
        test_read=False,
        sports_read=False,
        stage="validation_consumed_evaluating_in_memory",
    )

    summaries: list[dict[str, Any]] = []
    per_user: dict[str, list[dict[str, Any]]] = {}
    for index, dataset in enumerate(config["scientific_scope"]["datasets"]):
        events, preparation = prepare_domain(dataset, config)
        gate_path = REPO_ROOT / config["domain_inputs"][dataset]["gate"]
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        model = validate_gate(gate, dataset)
        summary, rows = evaluate_events(
            dataset,
            events,
            model,
            config["scientific_scope"]["bootstrap_replicates"],
            config["scientific_scope"]["bootstrap_seed"] + index,
        )
        expected_item_head = gate["item_head_sha256"]
        summary["preparation"] = preparation
        summary["gate_sha256"] = sha256(gate_path)
        summary["base_identity"] = {
            "expected_hit10": config["domain_inputs"][dataset]["expected_base_hit10"],
            "expected_ndcg10": config["domain_inputs"][dataset]["expected_base_ndcg10"],
            "hit10_exact": summary["base"]["Hit@10"] == config["domain_inputs"][dataset]["expected_base_hit10"],
            "ndcg10_close": math.isclose(
                summary["base"]["NDCG@10"],
                config["domain_inputs"][dataset]["expected_base_ndcg10"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "q1_exact": preparation["q1"] == config["domain_inputs"][dataset]["expected_q1"],
            "item_head_exact": preparation["item_head_sha256"] == expected_item_head,
        }
        summary["integrity"]["base_identity"] = all(summary["base_identity"].values())
        summaries.append(summary)
        per_user[dataset] = rows
        update_status(args.status_path, validation_users_processed=(index + 1) * EXPECTED_USERS)

    aggregate = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "mode": "one_shot_validation",
        "compute_device": "cpu",
        "datasets": summaries,
        "p2_gate": p2_scientific_gate(summaries),
        "input_locks_verified": verified_locks,
        "validation_access_started": True,
        "validation_consumed": True,
        "results_revealed": False,
        "test_read": False,
        "sports_read": False,
        "automatic_retry": False,
    }
    atomic_reveal(args.output_dir, aggregate, per_user)
    update_status(
        args.status_path,
        stage="two_domain_results_staged_awaiting_resource_audit",
        scientific_status=aggregate["p2_gate"]["status"],
        results_revealed=False,
        validation_users_processed=EXPECTED_USERS * 2,
    )
    print(json.dumps({"status": aggregate["status"], "p2_gate": aggregate["p2_gate"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
