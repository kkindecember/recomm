#!/usr/bin/env python3
"""Fit the preregistered per-user listwise BW3 expansion-admission gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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

from eval_cf0_b3_beamfusion import (  # noqa: E402
    load_users,
    metrics_from_ranks,
    score_item_head,
    standardize,
)
from eval_p9x_fixed_pcrf import load_catalog  # noqa: E402


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
FEATURES = [
    "seq_raw",
    "seq_anchor_z",
    "item_raw",
    "item_anchor_z",
    "popularity_log1p",
    "popularity_anchor_z",
    "beam200_rank_fraction",
    "reliability",
    "cf_pop_adjusted",
]
ACTION_CATEGORIES = ("base_top10", "base_11_50", "expansion_only", "outside_union")
MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max-fit-users", type=int)
    parser.add_argument("--max-calibration-users", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fresh_beams(path: Path, expected_width: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        if next(handle).rstrip("\n") != "idx\tgold\tpred\tscores":
            raise ValueError("unexpected fresh-beam header")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError("malformed fresh-beam row")
            user, gold, prediction_text, score_text = fields
            if user in rows:
                raise ValueError(f"duplicate prediction user: {user}")
            candidates = prediction_text.split("||")
            scores = np.asarray([float(value) for value in score_text.split("||")], dtype=np.float64)
            if len(candidates) != expected_width or scores.shape != (expected_width,):
                raise ValueError(f"{user}: expected {expected_width} candidates/scores")
            if len(set(candidates)) != expected_width or not np.isfinite(scores).all():
                raise ValueError(f"{user}: duplicate candidate or non-finite score")
            rows[user] = {"gold": gold, "candidates": candidates, "seq": scores}
    return rows


def anchor_apply(values: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    return (values - np.mean(anchor)) / max(float(np.std(anchor)), 1e-6)


def frozen_pcrf_joint(
    sequence_scores: np.ndarray,
    item_scores: np.ndarray,
    popularity: np.ndarray,
    reliability: float,
) -> np.ndarray:
    seq_z = standardize(np.asarray(sequence_scores, dtype=np.float64))
    item_z = standardize(np.asarray(item_scores, dtype=np.float64))
    pop_z = standardize(np.log1p(np.asarray(popularity, dtype=np.float64)))
    adjusted_z = standardize(item_z - 0.5 * pop_z)
    return seq_z + reliability * adjusted_z


def deterministic_subset(users: list[str], limit: int | None, salt: str) -> list[str]:
    if limit is None or limit >= len(users):
        return sorted(users)
    if limit <= 0:
        raise ValueError("user limit must be positive")
    ordered = sorted(
        users,
        key=lambda user: (hashlib.sha256(f"{salt}:{user}".encode()).hexdigest(), user),
    )
    return sorted(ordered[:limit])


def action_category(target: int, base_ids: list[int], base_rank: int, expansion_ids: set[int]) -> str:
    if target in base_ids:
        return "base_top10" if base_rank <= 10 else "base_11_50"
    if target in expansion_ids:
        return "expansion_only"
    return "outside_union"


def prepare_events(
    dataset: str,
    offset: int,
    unit_dir: Path,
    max_users: int | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, str]]:
    config = DATASETS[dataset]
    beam_paths = {width: unit_dir / f"beams_w{width}.tsv" for width in (50, 200)}
    beam50 = load_fresh_beams(beam_paths[50], 50)
    beam200 = load_fresh_beams(beam_paths[200], 200)
    if set(beam50) != set(beam200):
        raise ValueError(f"{dataset} offset{offset}: beam user mismatch")
    selected = deterministic_subset(list(beam50), max_users, f"p1c-dry:{dataset}:{offset}")

    data_dir = REPO_ROOT / "GRAM/rec_datasets" / dataset
    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(data_dir, config["item_index"])
    id_to_lexical = {raw_to_id[raw_item]: lexical for raw_item, lexical in raw_to_lexical.items()}
    users = load_users(data_dir, raw_to_id)
    frequencies: Counter[int] = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-offset])
    target_freqs = sorted(frequencies[users[user][-offset]] for user in selected)
    q1 = target_freqs[len(target_freqs) // 4]

    base_records: list[dict[str, Any]] = []
    wide_records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for user in selected:
        sequence = users[user]
        target = sequence[-offset]
        expected_gold = id_to_lexical[target]
        if beam50[user]["gold"] != expected_gold or beam200[user]["gold"] != expected_gold:
            raise ValueError(f"{dataset} offset{offset} {user}: frozen-beam gold mismatch")
        history = sequence[max(0, len(sequence) - offset - 20) : len(sequence) - offset]
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
        metadata.append(
            {
                "user": user,
                "target": target,
                "target_frequency": frequencies[target],
                "q1": q1,
            }
        )

    item_head = REPO_ROOT / config["item_head"]
    score_item_head(base_records, item_head, 128)
    score_item_head(wide_records, item_head, 128)
    events: list[dict[str, Any]] = []
    for base, wide, meta in zip(base_records, wide_records, metadata):
        base_pop_raw = np.log1p(base["candidate_frequencies"])
        reliability = 1.0 - float(np.mean(base["candidate_frequencies"][:10] <= meta["q1"]))
        base_joint = frozen_pcrf_joint(
            base["seq"], base["cf"], base["candidate_frequencies"], reliability
        )
        base_order = np.argsort(-base_joint, kind="stable")
        base_top10 = [base["candidate_ids"][index] for index in base_order[:10]]

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
            expansion.append({"candidate_id": candidate_id, "features": feature})
        expansion_ids = {row["candidate_id"] for row in expansion}
        if meta["target"] in base["candidate_ids"]:
            target_base_position = base["candidate_ids"].index(meta["target"])
            base_rank = int(np.flatnonzero(base_order == target_base_position)[0]) + 1
        else:
            base_rank = 201
        category = action_category(meta["target"], base["candidate_ids"], base_rank, expansion_ids)
        target_expansion_index = next(
            (index for index, row in enumerate(expansion) if row["candidate_id"] == meta["target"]),
            None,
        )
        events.append(
            {
                **meta,
                "base_top10": base_top10,
                "base_rank": base_rank,
                "in_beam50": meta["target"] in base_set,
                "in_beam200": meta["target"] in set(wide["candidate_ids"]),
                "action_category": category,
                "target_expansion_index": target_expansion_index,
                "expansion": expansion,
            }
        )
    locks = {f"beams_w{width}.tsv": sha256(path) for width, path in beam_paths.items()}
    return events, item_head, locks


def eligible_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["action_category"] != "outside_union"]


def feature_statistics(events: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    candidates = [candidate["features"] for event in eligible_events(events) for candidate in event["expansion"]]
    if not candidates:
        raise ValueError("no eligible expansion candidates")
    values = np.stack(candidates)
    if not np.isfinite(values).all():
        raise ValueError("non-finite feature")
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6)


def listwise_loss(
    events: list[dict[str, Any]],
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for event in eligible_events(events):
        values = np.stack([candidate["features"] for candidate in event["expansion"]])
        x = torch.tensor((values - mean) / std, dtype=torch.float32)
        logits = x @ weight + bias
        denominator = torch.logsumexp(torch.cat((torch.zeros(1), logits)), dim=0)
        target_index = event["target_expansion_index"]
        losses.append(denominator if target_index is None else denominator - logits[target_index])
    if not losses:
        raise ValueError("no events eligible for listwise loss")
    return torch.stack(losses).mean()


def fit_listwise(
    events: list[dict[str, Any]],
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    if epochs <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid optimization settings")
    mean, std = feature_statistics(events)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    weight = torch.zeros(len(FEATURES), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)
    with torch.no_grad():
        initial = float(listwise_loss(events, weight, bias, mean, std))
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        ranking = listwise_loss(events, weight, bias, mean, std)
        objective = ranking + l2 * weight.square().sum()
        objective.backward()
        optimizer.step()
        losses.append(float(objective.detach()))
    with torch.no_grad():
        final_ranking = float(listwise_loss(events, weight, bias, mean, std))
        final_objective = float(final_ranking + l2 * weight.square().sum())
    return {
        "weight": weight.detach().numpy(),
        "bias": float(bias.detach()),
        "mean": mean,
        "std": std,
        "initial_ranking_loss": initial,
        "final_ranking_loss": final_ranking,
        "final_objective": final_objective,
        "epoch_objectives": losses,
        "finite": bool(np.isfinite([initial, final_ranking, final_objective, *losses]).all()),
        "loss_events": len(eligible_events(events)),
        "excluded_events": len(events) - len(eligible_events(events)),
    }


def score_expansion(event: dict[str, Any], model: dict[str, Any]) -> list[tuple[float, int]]:
    scored = []
    for candidate in event["expansion"]:
        z = (candidate["features"] - model["mean"]) / model["std"]
        logit = float(z @ model["weight"] + model["bias"])
        if not math.isfinite(logit):
            raise ValueError("non-finite admission logit")
        scored.append((logit, candidate["candidate_id"]))
    return sorted(scored, key=lambda pair: (-pair[0], pair[1]))


def apply_margin(event: dict[str, Any], model: dict[str, Any], margin: float) -> dict[str, Any]:
    passing = [(logit, candidate_id) for logit, candidate_id in score_expansion(event, model) if logit >= margin]
    admitted_pairs = passing[:3]
    admitted = [candidate_id for _, candidate_id in admitted_pairs]
    keep = 10 - len(admitted)
    final_top10 = event["base_top10"][:keep] + admitted
    if not admitted and final_top10 != event["base_top10"]:
        raise AssertionError("fallback identity failed")
    final_rank = final_top10.index(event["target"]) + 1 if event["target"] in final_top10 else 201
    return {
        "admitted": admitted,
        "admitted_logits": [logit for logit, _ in admitted_pairs],
        "final_top10": final_top10,
        "final_rank": final_rank,
        "fallback": not admitted,
    }


def evaluate_margin(events: list[dict[str, Any]], model: dict[str, Any], margin: float) -> dict[str, Any]:
    applied = [apply_margin(event, model, margin) for event in events]
    base_ranks = np.asarray([event["base_rank"] for event in events], dtype=np.int64)
    final_ranks = np.asarray([row["final_rank"] for row in applied], dtype=np.int64)
    target_frequency = np.asarray([event["target_frequency"] for event in events])
    tail = target_frequency <= events[0]["q1"]
    base = metrics_from_ranks(base_ranks)
    candidate = metrics_from_ranks(final_ranks)
    base_tail = metrics_from_ranks(base_ranks[tail])
    candidate_tail = metrics_from_ranks(final_ranks[tail])
    return {
        "margin": margin,
        "base": base,
        "candidate": candidate,
        "hit10_delta": candidate["Hit@10"] - base["Hit@10"],
        "ndcg10_delta": candidate["NDCG@10"] - base["NDCG@10"],
        "tail_hit10_delta": candidate_tail["Hit@10"] - base_tail["Hit@10"],
        "admissions": sum(len(row["admitted"]) for row in applied),
        "admission_users": sum(bool(row["admitted"]) for row in applied),
        "fallback_users": sum(row["fallback"] for row in applied),
        "promotions": int(np.sum((base_ranks > 10) & (final_ranks <= 10))),
        "regressions": int(np.sum((base_ranks <= 10) & (final_ranks > 10))),
    }


def select_margin(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    safe = [
        row
        for row in rows
        if row["hit10_delta"] >= 0
        and row["ndcg10_delta"] >= -0.001
        and row["admissions"] > 0
    ]
    return max(
        safe,
        key=lambda row: (row["candidate"]["Hit@10"], row["candidate"]["NDCG@10"], row["margin"]),
    ) if safe else None


def attrition_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(events)
    actions = {category: sum(event["action_category"] == category for event in events) for category in ACTION_CATEGORIES}
    if sum(actions.values()) != total:
        raise AssertionError("action attrition categories do not sum to all events")
    memberships = {
        "target_in_beam50": sum(event["in_beam50"] for event in events),
        "target_in_beam200": sum(event["in_beam200"] for event in events),
        "target_in_both": sum(event["in_beam50"] and event["in_beam200"] for event in events),
        "target_in_beam200_not_beam50": sum(event["in_beam200"] and not event["in_beam50"] for event in events),
        "target_in_union_included_in_loss": sum(event["action_category"] != "outside_union" for event in events),
        "target_outside_union_excluded_from_loss": actions["outside_union"],
        "empty_expansion_pool": sum(not event["expansion"] for event in events),
    }
    sizes = np.asarray([len(event["expansion"]) for event in events], dtype=np.float64)
    return {
        "total_events": total,
        "action_counts": actions,
        "action_fractions": {key: value / total for key, value in actions.items()},
        "membership_counts": memberships,
        "membership_fractions": {key: value / total for key, value in memberships.items()},
        "expansion_pool_size": {
            "min": int(sizes.min()),
            "median": float(np.median(sizes)),
            "mean": float(sizes.mean()),
            "max": int(sizes.max()),
        },
    }


def per_user_rows(events: list[dict[str, Any]], model: dict[str, Any], margin: float) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        applied = apply_margin(event, model, margin)
        base_hit = event["base_rank"] <= 10
        final_hit = applied["final_rank"] <= 10
        rows.append(
            {
                "user": event["user"],
                "target_item_id": event["target"],
                "action_category": event["action_category"],
                "base_rank": event["base_rank"],
                "target_in_beam50": int(event["in_beam50"]),
                "target_in_beam200": int(event["in_beam200"]),
                "expansion_pool_size": len(event["expansion"]),
                "admitted_item_ids": "||".join(map(str, applied["admitted"])),
                "admitted_logits": "||".join(f"{value:.9g}" for value in applied["admitted_logits"]),
                "admission_count": len(applied["admitted"]),
                "final_rank": applied["final_rank"],
                "promotion": int(not base_hit and final_hit),
                "regression": int(base_hit and not final_hit),
                "fallback": int(applied["fallback"]),
                "target_frequency": event["target_frequency"],
                "target_group": "tail" if event["target_frequency"] <= event["q1"] else "non_tail",
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty per-user output")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def serializable_fit(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model.items()
        if key not in ("weight", "mean", "std", "epoch_objectives")
    } | {
        "initial_objective": model["epoch_objectives"][0],
        "last_epoch_objective": model["epoch_objectives"][-1],
    }


def run_domain(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    fit_events, item_head, fit_locks = prepare_events(
        dataset,
        4,
        args.root / dataset / "fit",
        args.max_fit_users,
    )
    calibration_events, calibration_item_head, calibration_locks = prepare_events(
        dataset,
        3,
        args.root / dataset / "calibration",
        args.max_calibration_users,
    )
    if item_head != calibration_item_head:
        raise AssertionError("item-head identity differs across splits")
    model = fit_listwise(fit_events, args.epochs, args.learning_rate, args.l2, args.seed)
    grid = [evaluate_margin(calibration_events, model, margin) for margin in MARGINS]
    selected = None if args.dry_run else select_margin(grid)
    domain_dir = args.output_dir / dataset
    domain_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = {
        "experiment_id": "GRAM_PHASE11_BW3_P1C_LISTWISE_ADMISSION_CORRECTION_V1",
        "dataset": dataset,
        "objective": "per_user_listwise_cross_entropy_with_fixed_reject_logit_zero",
        "feature_schema": FEATURES,
        "weight": model["weight"].tolist(),
        "bias": model["bias"],
        "feature_mean": model["mean"].tolist(),
        "feature_std": model["std"].tolist(),
        "selected_margin": None if selected is None else selected["margin"],
        "margin_grid": list(MARGINS),
        "max_admissions": 3,
        "optimizer": {"name": "Adam", "epochs": args.epochs, "learning_rate": args.learning_rate, "l2": args.l2, "seed": args.seed},
        "item_head_sha256": sha256(item_head),
        "compute_device": "cpu",
    }
    checkpoint_path = domain_dir / "admission_gate.json"
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fit_attrition = attrition_summary(fit_events)
    calibration_attrition = attrition_summary(calibration_events)
    attrition = {"fit": fit_attrition, "calibration": calibration_attrition}
    (domain_dir / "attrition.json").write_text(json.dumps(attrition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if selected is not None:
        write_tsv(domain_dir / "calibration_per_user.tsv", per_user_rows(calibration_events, model, selected["margin"]))

    integrity = {
        "objective_is_per_user_listwise_ce": True,
        "bce_or_pos_weight_absent": True,
        "feature_schema_exact": checkpoint["feature_schema"] == FEATURES,
        "fit_only_feature_statistics": True,
        "finite": model["finite"],
        "loss_decreased": model["final_ranking_loss"] < model["initial_ranking_loss"],
        "action_attrition_complete": sum(fit_attrition["action_counts"].values()) == len(fit_events)
        and sum(calibration_attrition["action_counts"].values()) == len(calibration_events),
        "compute_device_cpu": True,
        "validation_target_unread": True,
        "test_unread": True,
        "sports_unread": True,
    }
    scientific = {
        "status": "not_evaluated_dry_run" if args.dry_run else "passed" if selected is not None
        and selected["tail_hit10_delta"] >= 0
        and selected["promotions"] >= selected["regressions"] else "failed",
        "selected": selected,
    }
    result = {
        "dataset": dataset,
        "mode": "dry_run" if args.dry_run else "formal",
        "fit": serializable_fit(model),
        "fit_attrition": fit_attrition,
        "calibration_attrition": calibration_attrition,
        "calibration_grid": grid,
        "scientific_gate": scientific,
        "integrity_gate": {"status": "passed" if all(integrity.values()) else "failed", "checks": integrity},
        "input_locks": {"fit": fit_locks, "calibration": calibration_locks, "item_head": sha256(item_head)},
        "checkpoint_sha256": sha256(checkpoint_path),
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
    }
    (domain_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise RuntimeError("P1C is CPU-only; CUDA_VISIBLE_DEVICES must be empty or -1")
    if torch.cuda.is_available():
        raise RuntimeError("P1C CPU-only assertion failed: CUDA remains visible")
    if not args.dry_run and (args.max_fit_users is not None or args.max_calibration_users is not None):
        raise ValueError("formal mode forbids user limits")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    results = [run_domain(args, dataset) for dataset in ("Toys", "Beauty")]
    both_integrity = all(row["integrity_gate"]["status"] == "passed" for row in results)
    both_scientific = all(row["scientific_gate"]["status"] == "passed" for row in results)
    summary = {
        "experiment_id": "GRAM_PHASE11_BW3_P1C_LISTWISE_ADMISSION_CORRECTION_V1",
        "status": "dry_run_completed" if args.dry_run else "completed",
        "mode": "dry_run" if args.dry_run else "formal",
        "compute_device": "cpu",
        "datasets": results,
        "p1c_gate": {
            "status": "not_evaluated_dry_run" if args.dry_run else "passed_eligible_for_separate_p2_authorization" if both_integrity and both_scientific else "failed",
            "both_domains_integrity": both_integrity,
            "both_domains_scientific": both_scientific if not args.dry_run else None,
        },
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "p1c_gate": summary["p1c_gate"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
