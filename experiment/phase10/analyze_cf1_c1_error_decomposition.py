#!/usr/bin/env python3
"""Read-only error decomposition for the completed CF1-C1 OOF experiment."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = REPO_ROOT / "artifacts/phase10/cf1_c0_toys_feature_audit/feature_table.npz"
DEFAULT_C1 = REPO_ROOT / "artifacts/phase10/cf1_c1_toys_crossfit_calibrator"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase10/cf1_c1_error_decomposition"
DEFAULT_PREDICTIONS = REPO_ROOT / "GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv"
FEATURE_NAMES = [
    "gram_z", "corrected_item_z", "source_both", "source_cf_only",
    "gram_rr", "cf_rr", "agreement", "reliability_x_item",
    "short_history_x_item", "long_history_x_item", "item_log_frequency",
]
SOURCE_NAMES = {0: "gram_only", 1: "both", 2: "cf_only"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--c1-dir", type=Path, default=DEFAULT_C1)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(values.std())
    if scale < 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


def load_cached_beams(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n")
        if not header.startswith("idx\t"):
            raise ValueError("unexpected prediction header")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 1:
                continue
            user, score_text = fields[0], fields[-1]
            scores = np.asarray([float(value) for value in score_text.split("||")])
            if scores.shape != (50,) or not np.isfinite(scores).all():
                raise ValueError(f"{user}: malformed cached scores")
            rows[user] = {"seq": scores}
    return rows


def build_base_features(data):
    lengths = np.diff(data["offsets"])
    reliability = np.repeat(data["user_reliability"], lengths)
    history = np.repeat(data["user_history_length"], lengths)
    source = data["source"]
    corrected = data["corrected_item_z"].astype(np.float64)
    return np.column_stack([
        data["gram_z"], corrected, source == 1, source == 2,
        data["gram_rr"], data["cf_rr"], data["agreement"],
        reliability * corrected, (history <= 5) * corrected,
        (history >= 11) * corrected, data["item_log_frequency"],
    ]).astype(np.float64)


def frozen_pcrf_ranks(data, cache):
    users = data["users"]
    offsets = data["offsets"]
    targets = data["user_target_position"]
    item_score = data["item_score"]
    item_log_frequency = data["item_log_frequency"]
    reliability = data["user_reliability"]
    ranks = np.empty(len(users), dtype=np.int64)
    for index, raw_user in enumerate(users):
        left = int(offsets[index])
        target = int(targets[index])
        if target < 0 or target >= 50:
            ranks[index] = 51
            continue
        seq_z = standardize(cache[str(raw_user)]["seq"])
        item_z = standardize(item_score[left : left + 50])
        pop_z = standardize(item_log_frequency[left : left + 50])
        adjusted = standardize(item_z - 0.5 * pop_z)
        scores = seq_z + float(reliability[index]) * adjusted
        order = np.argsort(-scores, kind="stable")
        ranks[index] = int(np.flatnonzero(order == target)[0]) + 1
    return ranks


def popularity_group(frequency):
    if frequency <= 5:
        return "tail"
    if frequency < 26:
        return "middle"
    return "head"


def history_group(length):
    if length <= 5:
        return "1-5"
    if length <= 10:
        return "6-10"
    return "11-20"


def transition_counts(baseline_ranks, candidate_ranks, mask, cutoff):
    base = baseline_ranks[mask] <= cutoff
    candidate = candidate_ranks[mask] <= cutoff
    gain = int(np.sum(~base & candidate))
    loss = int(np.sum(base & ~candidate))
    return {
        "users": int(np.sum(mask)),
        "baseline_hits": int(np.sum(base)),
        "candidate_hits": int(np.sum(candidate)),
        "gains": gain,
        "losses": loss,
        "net_hits": gain - loss,
        "delta": float((gain - loss) / np.sum(mask)),
        "discordant": gain + loss,
    }


def main():
    args = parse_args()
    summary_path = args.c1_dir / "summary.json"
    models_path = args.c1_dir / "fold_models.json"
    per_user_path = args.c1_dir / "per_user_oof.tsv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models = json.loads(models_path.read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("CF1-C1 is not completed")
    if summary["development_gate"]["status"] != "failed_development_gate":
        raise ValueError("diagnostic is frozen for the failed CF1-C1 result")
    if sha256(args.features) != summary["artifacts"]["c0_feature_table_sha256"]:
        raise ValueError("feature table hash mismatch")
    if sha256(models_path) != summary["artifacts"]["fold_models_sha256"]:
        raise ValueError("fold model hash mismatch")
    if sha256(per_user_path) != summary["artifacts"]["per_user_oof_sha256"]:
        raise ValueError("per-user OOF hash mismatch")

    data = np.load(args.features, allow_pickle=False)
    users = data["users"]
    offsets = data["offsets"].astype(np.int64)
    folds = data["fold"].astype(np.int64)
    targets = data["user_target_position"].astype(np.int64)
    target_frequency = data["user_target_frequency"].astype(np.int64)
    history = data["user_history_length"].astype(np.int64)
    reliability = data["user_reliability"].astype(np.float64)
    item_score = data["item_score"].astype(np.float64)
    item_log_frequency = data["item_log_frequency"].astype(np.float64)
    source = data["source"].astype(np.int64)
    base_x = build_base_features(data)

    cache = load_cached_beams(args.predictions)
    baseline_ranks = frozen_pcrf_ranks(data, cache).astype(np.int64)
    oof_ranks = np.full(len(users), 91, dtype=np.int64)
    entrant_sources = Counter()
    loss_user_entrant_sources = Counter()
    transition_rows = []

    for user_index, raw_user in enumerate(users):
        left, right = int(offsets[user_index]), int(offsets[user_index + 1])
        model = models[int(folds[user_index])]
        if model["feature_names"] != FEATURE_NAMES:
            raise ValueError("unexpected feature order")
        x_eval = base_x[left:right].copy()
        x_eval[:, 10] = (
            x_eval[:, 10] - float(model["item_log_frequency_mean"])
        ) / float(model["item_log_frequency_std"])
        oof_scores = x_eval @ np.asarray(model["weights"], dtype=np.float64)
        oof_order = np.argsort(-oof_scores, kind="stable")

        seq_z = standardize(np.asarray(cache[str(raw_user)]["seq"], dtype=np.float64))
        item_z = standardize(item_score[left : left + 50])
        pop_z = standardize(item_log_frequency[left : left + 50])
        adjusted = standardize(item_z - 0.5 * pop_z)
        pcrf_scores = seq_z + reliability[user_index] * adjusted
        baseline_order = np.argsort(-pcrf_scores, kind="stable")

        target = int(targets[user_index])
        if target >= 0:
            oof_ranks[user_index] = int(np.flatnonzero(oof_order == target)[0]) + 1
        baseline_top10 = set(int(value) for value in baseline_order[:10])
        oof_top10 = set(int(value) for value in oof_order[:10])
        entrants = sorted(oof_top10 - baseline_top10)
        for candidate in entrants:
            entrant_sources[SOURCE_NAMES[int(source[left + candidate])]] += 1

        baseline_hit = int(baseline_ranks[user_index] <= 10)
        oof_hit = int(oof_ranks[user_index] <= 10)
        transition = "stable_hit" if baseline_hit and oof_hit else (
            "gain" if not baseline_hit and oof_hit else (
                "loss" if baseline_hit and not oof_hit else "stable_miss"
            )
        )
        if transition == "loss":
            for candidate in entrants:
                loss_user_entrant_sources[SOURCE_NAMES[int(source[left + candidate])]] += 1
        target_source = "union_miss" if target < 0 else SOURCE_NAMES[int(source[left + target])]
        transition_rows.append({
            "user_id": str(raw_user),
            "fold": int(folds[user_index]),
            "history_group": history_group(int(history[user_index])),
            "target_popularity": popularity_group(int(target_frequency[user_index])),
            "target_source": target_source,
            "baseline_rank": int(baseline_ranks[user_index]),
            "oof_rank": int(oof_ranks[user_index]),
            "hit10_transition": transition,
            "oof_top10_new_gram_only": sum(
                SOURCE_NAMES[int(source[left + candidate])] == "gram_only" for candidate in entrants
            ),
            "oof_top10_new_both": sum(
                SOURCE_NAMES[int(source[left + candidate])] == "both" for candidate in entrants
            ),
            "oof_top10_new_cf_only": sum(
                SOURCE_NAMES[int(source[left + candidate])] == "cf_only" for candidate in entrants
            ),
        })

    expected = np.asarray([int(row["oof_rank"]) for row in csv.DictReader(
        per_user_path.open(encoding="utf-8"), delimiter="\t"
    )])
    if not np.array_equal(oof_ranks, expected):
        raise ValueError("recomputed OOF ranks do not match the frozen per-user artifact")

    all_users = np.ones(len(users), dtype=bool)
    groups = {
        "overall": all_users,
        "target_tail": target_frequency <= 5,
        "target_middle": (target_frequency > 5) & (target_frequency < 26),
        "target_head": target_frequency >= 26,
        "history_1-5": history <= 5,
        "history_6-10": (history >= 6) & (history <= 10),
        "history_11-20": history >= 11,
    }
    target_sources = np.full(len(users), "union_miss", dtype="<U16")
    present = targets >= 0
    target_sources[present] = np.asarray([
        SOURCE_NAMES[int(source[offsets[index] + targets[index]])]
        for index in np.flatnonzero(present)
    ])
    for name in SOURCE_NAMES.values():
        groups[f"target_source_{name}"] = target_sources == name
    groups["target_union_miss"] = ~present

    transitions = {
        name: {
            f"Hit@{cutoff}": transition_counts(baseline_ranks, oof_ranks, mask, cutoff)
            for cutoff in (1, 5, 10, 20, 50)
        }
        for name, mask in groups.items()
    }
    weights = np.asarray([model["weights"] for model in models], dtype=np.float64)
    coefficient_stability = {}
    for index, name in enumerate(FEATURE_NAMES):
        values = weights[:, index]
        coefficient_stability[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "all_positive": bool(np.all(values > 0)),
            "all_negative": bool(np.all(values < 0)),
        }

    oracle = 0.2647331547496394
    baseline_hit50 = summary["baseline"]["Hit@50"]
    delta_hit50 = summary["oof"]["delta"]["Hit@50"]
    diagnostic = {
        "analysis_id": "GRAM_PHASE10_CF1_C1_ERROR_DECOMPOSITION_V1",
        "status": "completed",
        "evidence_class": "post_C1_read_only_mechanism_diagnostic_not_model_selection",
        "dataset": "Toys",
        "split": "validation",
        "test_read": False,
        "beauty_read": False,
        "sports_read": False,
        "no_training_performed": True,
        "input_identity": {
            "feature_table_sha256": sha256(args.features),
            "c1_summary_sha256": sha256(summary_path),
            "fold_models_sha256": sha256(models_path),
            "per_user_oof_sha256": sha256(per_user_path),
            "predictions_sha256": sha256(args.predictions),
        },
        "transitions": transitions,
        "oof_top10_entrant_sources": dict(sorted(entrant_sources.items())),
        "loss_user_top10_entrant_sources": dict(sorted(loss_user_entrant_sources.items())),
        "coefficient_stability": coefficient_stability,
        "oracle_gap": {
            "baseline_Hit@50": baseline_hit50,
            "union_oracle_Hit@50": oracle,
            "available_gap": oracle - baseline_hit50,
            "C1_Hit@50_delta": delta_hit50,
            "fraction_of_gap_captured": delta_hit50 / (oracle - baseline_hit50),
        },
        "interpretation_boundary": (
            "Post-C1 diagnostic only. It may motivate one preregistered C2 specification, "
            "but it cannot retroactively pass C1 or authorize Beauty."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = args.output_dir / "hit10_transitions.tsv"
    with transitions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transition_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(transition_rows)
    diagnostic["artifacts"] = {"hit10_transitions_sha256": sha256(transitions_path)}
    summary_out = args.output_dir / "summary.json"
    summary_out.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": diagnostic["status"],
        "overall": diagnostic["transitions"]["overall"],
        "entrant_sources": diagnostic["oof_top10_entrant_sources"],
        "loss_user_entrant_sources": diagnostic["loss_user_top10_entrant_sources"],
        "fraction_of_gap_captured": diagnostic["oracle_gap"]["fraction_of_gap_captured"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
