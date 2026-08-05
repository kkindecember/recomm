#!/usr/bin/env python3
"""Evaluate parameter-free anchor50 normalization on frozen BW1 beam200 outputs."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE9 = REPO_ROOT / "experiment/phase9"
PHASE11 = REPO_ROOT / "experiment/phase11"
for directory in (PHASE9, PHASE11):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval_bw1_candidate_ceiling import DATASETS, build_records, ranks  # noqa: E402
from eval_cf0_b3_beamfusion import (  # noqa: E402
    bootstrap_hit10_delta,
    load_users,
    metrics_from_ranks,
    score_item_head,
)
from eval_p9x_fixed_pcrf import load_catalog  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw1-root", type=Path, default=REPO_ROOT / "artifacts/phase11/bw1_candidate_ceiling")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/phase11/bw2_anchored_expansion")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def load_fresh_beams(path, expected_width):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n")
        if header != "idx\tgold\tpred\tscores":
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


def anchor_standardize(values, anchor_size=50):
    anchor = np.asarray(values[:anchor_size], dtype=np.float64)
    return (np.asarray(values, dtype=np.float64) - np.mean(anchor)) / max(float(np.std(anchor)), 1e-6)


def anchored_values(record, beta=0.5, gamma=1.0, anchor_size=50):
    seq_z = anchor_standardize(record["seq"], anchor_size)
    cf_z = anchor_standardize(record["cf"], anchor_size)
    pop_z = anchor_standardize(np.log1p(record["candidate_frequencies"]), anchor_size)
    adjusted = cf_z - beta * pop_z
    adjusted_z = anchor_standardize(adjusted, anchor_size)
    reliability = (1.0 - record["tail_mass"]) ** gamma
    return seq_z + reliability * adjusted_z


def anchored_ranks(records, anchor_size=50):
    result, top10, anchor_identity = [], [], []
    for record in records:
        values = anchored_values(record, anchor_size=anchor_size)
        order = np.argsort(-values, kind="stable")
        top10.append(order[:10])
        position = record["target_position"]
        result.append(len(record["candidate_ids"]) + 1 if position < 0 else int(np.flatnonzero(order == position)[0]) + 1)

        truncated = dict(record)
        for key in ("seq", "cf", "candidate_frequencies", "candidate_ids"):
            truncated[key] = record[key][:anchor_size]
        regular_order = np.argsort(-anchored_values(truncated, anchor_size=anchor_size), kind="stable")
        anchor_order = np.asarray([index for index in order if index < anchor_size])
        anchor_identity.append(np.array_equal(anchor_order, regular_order))
    return np.asarray(result, dtype=np.int64), top10, anchor_identity


def scientific_gate(rows):
    checks = {
        "both_Hit10_non_degradation": all(row["hit10_delta"] >= 0 for row in rows),
        "at_least_one_Hit10_delta_at_least_0.002": any(row["hit10_delta"] >= 0.002 for row in rows),
        "both_NDCG10_delta_at_least_minus_0.001": all(row["ndcg10_delta"] >= -0.001 for row in rows),
        "expansion_candidates_used": any(row["users_with_expansion_in_top10"] > 0 for row in rows),
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, integrity_checks = [], {}
    per_user_path = args.output_dir / "per_user.tsv"
    with per_user_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["dataset", "user_id", "top50_overlap", "beam50_pcrf_rank", "anchored_beam200_rank", "expansion_in_top10"])
        for dataset in ("Toys", "Beauty"):
            config = DATASETS[dataset]
            bw1_summary = json.loads((args.bw1_root / dataset / "summary.json").read_text(encoding="utf-8"))
            beam50 = load_fresh_beams(args.bw1_root / dataset / "fresh_beams_w50.tsv", 50)
            beam200 = load_fresh_beams(args.bw1_root / dataset / "fresh_beams_w200.tsv", 200)
            selected = sorted(beam50)
            if selected != sorted(beam200):
                raise ValueError(f"{dataset}: beam user mismatch")
            data_dir = REPO_ROOT / "GRAM/rec_datasets" / dataset
            raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(data_dir, config["item_index"])
            users = load_users(data_dir, raw_to_id)
            frequencies = Counter()
            for sequence in users.values():
                frequencies.update(sequence[:-2])
            target_frequencies = sorted(frequencies[sequence[-2]] for sequence in users.values())
            q1 = target_frequencies[len(target_frequencies) // 4]
            records50 = build_records(selected, beam50, users, raw_to_lexical, raw_to_id, lexical_to_id, frequencies, q1)
            records200 = build_records(selected, beam200, users, raw_to_lexical, raw_to_id, lexical_to_id, frequencies, q1)
            item_head = REPO_ROOT / config["item_head"]
            score_item_head(records50, item_head, 512)
            score_item_head(records200, item_head, 512)
            base_rank = ranks(records50, True)
            anchored_rank, anchored_top10, anchor_identity = anchored_ranks(records200)
            base_metrics = metrics_from_ranks(base_rank)
            anchored_metrics = metrics_from_ranks(anchored_rank)
            overlaps, expansion_used = [], []
            for index, user in enumerate(selected):
                overlap = len(set(beam50[user]["candidates"]) & set(beam200[user]["candidates"][:50])) / 50.0
                used = any(candidate_index >= 50 for candidate_index in anchored_top10[index])
                overlaps.append(overlap)
                expansion_used.append(used)
                writer.writerow([dataset, user, overlap, base_rank[index], anchored_rank[index], int(used)])
            hit10_delta = anchored_metrics["Hit@10"] - base_metrics["Hit@10"]
            ndcg10_delta = anchored_metrics["NDCG@10"] - base_metrics["NDCG@10"]
            row = {
                "dataset": dataset,
                "users": len(selected),
                "mean_beam50_anchor50_overlap": float(np.mean(overlaps)),
                "anchor_top50_order_identity_fraction": float(np.mean(anchor_identity)),
                "beam50_pcrf": base_metrics,
                "anchored_beam200_pcrf": anchored_metrics,
                "hit10_delta": hit10_delta,
                "ndcg10_delta": ndcg10_delta,
                "users_with_expansion_in_top10": int(np.sum(expansion_used)),
                "target_promotions_into_top10": int(np.sum((base_rank > 10) & (anchored_rank <= 10))),
                "target_regressions_from_top10": int(np.sum((base_rank <= 10) & (anchored_rank > 10))),
                "Hit@10_paired_bootstrap_95ci": bootstrap_hit10_delta(base_rank, anchored_rank, args.bootstrap_replicates, args.seed),
            }
            rows.append(row)
            integrity_checks[dataset] = {
                "bw1_integrity_passed": bw1_summary["integrity_gate"]["status"] == "passed",
                "all_512_users": len(selected) == 512,
                "mean_anchor_overlap_at_least_0.98": row["mean_beam50_anchor50_overlap"] >= 0.98,
                "anchor_top50_order_identity": row["anchor_top50_order_identity_fraction"] == 1.0,
                "item_head_sha_matches_bw1": hashlib.sha256(item_head.read_bytes()).hexdigest() == bw1_summary["artifacts"]["item_head_sha256"],
            }
    integrity_flat = [value for checks in integrity_checks.values() for value in checks.values()]
    integrity_gate = {"status": "passed" if all(integrity_flat) else "failed", "checks": integrity_checks}
    gate = scientific_gate(rows) if integrity_gate["status"] == "passed" else {"status": "not_evaluated", "checks": {}}
    summary = {
        "experiment_id": "GRAM_PHASE11_BW2_ANCHORED_EXPANSION_VALIDATION_V1",
        "status": "completed",
        "split": "validation",
        "test_read": False,
        "sports_read": False,
        "formula": {"anchor_size": 50, "lambda": 1.0, "beta": 0.5, "gamma": 1.0},
        "rows": rows,
        "integrity_gate": integrity_gate,
        "scientific_gate": gate,
        "decision": "anchored_expansion_eligible" if gate["status"] == "passed" else "train_expansion_admission_gate",
        "artifacts": {"per_user_sha256": hashlib.sha256(per_user_path.read_bytes()).hexdigest()},
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"integrity_gate": integrity_gate, "scientific_gate": gate, "decision": summary["decision"], "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
