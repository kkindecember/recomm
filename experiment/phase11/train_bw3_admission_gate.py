#!/usr/bin/env python3
"""Fit and calibrate the BW3 expansion-admission gate on train-prefix pseudo-futures."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE9 = REPO_ROOT / "experiment/phase9"
PHASE11 = REPO_ROOT / "experiment/phase11"
for directory in (PHASE9, PHASE11):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval_bw1_candidate_ceiling import DATASETS  # noqa: E402
from eval_bw2_anchored_expansion import load_fresh_beams  # noqa: E402
from eval_cf0_b3_beamfusion import load_users, metrics_from_ranks, score_item_head, standardize  # noqa: E402
from eval_p9x_fixed_pcrf import load_catalog  # noqa: E402


FEATURES = [
    "seq_anchor_z",
    "cf_anchor_z",
    "pop_anchor_z",
    "adjusted_anchor_z",
    "beam200_rank_fraction",
    "reliability",
    "seq_gap_to_base10",
    "adjusted_gap_to_base10",
]
MARGINS = [0.0, 0.25, 0.5, 0.75, 1.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def anchor_apply(values, anchor):
    return (np.asarray(values, dtype=np.float64) - np.mean(anchor)) / max(float(np.std(anchor)), 1e-6)


def prepare_events(dataset, offset, unit_dir):
    config = DATASETS[dataset]
    beam50 = load_fresh_beams(unit_dir / "beams_w50.tsv", 50)
    beam200 = load_fresh_beams(unit_dir / "beams_w200.tsv", 200)
    selected = sorted(beam50)
    if selected != sorted(beam200):
        raise ValueError(f"{dataset} offset{offset}: user mismatch")
    data_dir = REPO_ROOT / "GRAM/rec_datasets" / dataset
    raw_to_lexical, raw_to_id, lexical_to_id = load_catalog(data_dir, config["item_index"])
    users = load_users(data_dir, raw_to_id)
    frequencies = Counter()
    for sequence in users.values():
        frequencies.update(sequence[:-offset])
    target_freqs = sorted(frequencies[users[user][-offset]] for user in selected)
    q1 = target_freqs[len(target_freqs) // 4]
    base_records, wide_records, metadata = [], [], []
    for user in selected:
        sequence = users[user]
        target = sequence[-offset]
        history = sequence[max(0, len(sequence) - offset - 20):len(sequence) - offset]
        base_ids = [lexical_to_id[value] for value in beam50[user]["candidates"]]
        wide_ids = [lexical_to_id[value] for value in beam200[user]["candidates"]]
        base_freq = np.asarray([frequencies[item] for item in base_ids], dtype=np.float64)
        wide_freq = np.asarray([frequencies[item] for item in wide_ids], dtype=np.float64)
        base_records.append({"history": history, "candidate_ids": base_ids, "seq": beam50[user]["seq"], "candidate_frequencies": base_freq})
        wide_records.append({"history": history, "candidate_ids": wide_ids, "seq": beam200[user]["seq"], "candidate_frequencies": wide_freq})
        metadata.append({"user": user, "target": target, "target_frequency": frequencies[target], "q1": q1})
    item_head = REPO_ROOT / config["item_head"]
    score_item_head(base_records, item_head, 512)
    score_item_head(wide_records, item_head, 512)
    events = []
    for base, wide, meta in zip(base_records, wide_records, metadata):
        base_seq_z = standardize(base["seq"])
        base_cf_z = standardize(base["cf"])
        base_pop_z = standardize(np.log1p(base["candidate_frequencies"]))
        base_adjusted_z = standardize(base_cf_z - 0.5 * base_pop_z)
        reliability = (1.0 - float(np.mean(base["candidate_frequencies"][:10] <= meta["q1"])))
        base_joint = base_seq_z + reliability * base_adjusted_z
        base_order = np.argsort(-base_joint, kind="stable")
        base_top10 = [base["candidate_ids"][index] for index in base_order[:10]]

        seq_z = anchor_apply(wide["seq"], base["seq"])
        cf_z = anchor_apply(wide["cf"], base["cf"])
        pop_z = anchor_apply(np.log1p(wide["candidate_frequencies"]), np.log1p(base["candidate_frequencies"]))
        base_adjusted_raw = anchor_apply(base["cf"], base["cf"]) - 0.5 * anchor_apply(np.log1p(base["candidate_frequencies"]), np.log1p(base["candidate_frequencies"]))
        adjusted_raw = cf_z - 0.5 * pop_z
        adjusted_z = anchor_apply(adjusted_raw, base_adjusted_raw)
        base_set = set(base["candidate_ids"])
        expansion = []
        for index, candidate_id in enumerate(wide["candidate_ids"]):
            if candidate_id in base_set:
                continue
            feature = np.asarray([
                seq_z[index], cf_z[index], pop_z[index], adjusted_z[index],
                (index + 1) / 200.0, reliability,
                seq_z[index] - base_seq_z[base_order[9]],
                adjusted_z[index] - base_adjusted_z[base_order[9]],
            ], dtype=np.float64)
            expansion.append({"candidate_id": candidate_id, "features": feature, "label": int(candidate_id == meta["target"])})
        events.append({
            **meta,
            "base_top10": base_top10,
            "base_rank": int(np.flatnonzero(base_order == base["candidate_ids"].index(meta["target"]))[0]) + 1 if meta["target"] in base["candidate_ids"] else 51,
            "expansion": expansion,
        })
    return events, item_head


def fit_logistic(events, epochs, learning_rate, l2, seed):
    features = np.concatenate([np.stack([candidate["features"] for candidate in event["expansion"]]) for event in events])
    labels = np.concatenate([np.asarray([candidate["label"] for candidate in event["expansion"]], dtype=np.float32) for event in events])
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("no positive expansion targets in fit data")
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-6)
    x = torch.tensor((features - mean) / std, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    torch.manual_seed(seed)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.tensor(math.log(positives / max(len(labels) - positives, 1)), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)
    pos_weight = torch.tensor(min((len(labels) - positives) / positives, 100.0))
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = x @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight) + l2 * weight.square().sum()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "weight": weight.detach().numpy(), "bias": float(bias.detach()),
        "mean": mean, "std": std, "positives": positives, "candidates": len(labels),
        "initial_loss": losses[0], "final_loss": losses[-1], "finite": bool(np.isfinite(losses).all()),
    }


def evaluate_margin(events, model, margin):
    base_ranks, final_ranks, admissions = [], [], 0
    target_freq, q1 = [], events[0]["q1"]
    for event in events:
        scored = []
        for candidate in event["expansion"]:
            z = (candidate["features"] - model["mean"]) / model["std"]
            logit = float(z @ model["weight"] + model["bias"])
            if logit >= margin:
                scored.append((logit, candidate["candidate_id"]))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        admitted = [candidate_id for _, candidate_id in scored[:3]]
        admissions += len(admitted)
        keep = 10 - len(admitted)
        final_top10 = event["base_top10"][:keep] + admitted
        base_ranks.append(event["base_rank"])
        final_ranks.append(final_top10.index(event["target"]) + 1 if event["target"] in final_top10 else 201)
        target_freq.append(event["target_frequency"])
    base_ranks = np.asarray(base_ranks)
    final_ranks = np.asarray(final_ranks)
    base = metrics_from_ranks(base_ranks)
    final = metrics_from_ranks(final_ranks)
    tail = np.asarray(target_freq) <= q1
    base_tail = metrics_from_ranks(base_ranks[tail])
    final_tail = metrics_from_ranks(final_ranks[tail])
    return {
        "margin": margin,
        "base": base,
        "candidate": final,
        "hit10_delta": final["Hit@10"] - base["Hit@10"],
        "ndcg10_delta": final["NDCG@10"] - base["NDCG@10"],
        "tail_hit10_delta": final_tail["Hit@10"] - base_tail["Hit@10"],
        "admissions": admissions,
        "promotions": int(np.sum((base_ranks > 10) & (final_ranks <= 10))),
        "regressions": int(np.sum((base_ranks <= 10) & (final_ranks > 10))),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_results = []
    for dataset in ("Toys", "Beauty"):
        fit_events, item_head = prepare_events(dataset, 4, args.root / dataset / "fit")
        calibration_events, _ = prepare_events(dataset, 3, args.root / dataset / "calibration")
        model = fit_logistic(fit_events, args.epochs, args.learning_rate, args.l2, args.seed)
        candidates = [evaluate_margin(calibration_events, model, margin) for margin in MARGINS]
        safe = [row for row in candidates if row["hit10_delta"] >= 0 and row["ndcg10_delta"] >= -0.001 and row["admissions"] > 0]
        selected = max(safe, key=lambda row: (row["candidate"]["Hit@10"], row["candidate"]["NDCG@10"], row["margin"])) if safe else None
        dataset_dir = args.output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "dataset": dataset,
            "feature_schema": FEATURES,
            "weight": model["weight"].tolist(),
            "bias": model["bias"],
            "feature_mean": model["mean"].tolist(),
            "feature_std": model["std"].tolist(),
            "selected_margin": None if selected is None else selected["margin"],
            "max_admissions": 3,
            "item_head_sha256": hashlib.sha256(item_head.read_bytes()).hexdigest(),
        }
        checkpoint_path = dataset_dir / "admission_gate.json"
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = {
            "dataset": dataset,
            "fit": {key: value for key, value in model.items() if key not in ("weight", "mean", "std")},
            "calibration_grid": candidates,
            "selected": selected,
            "calibration_gate": "passed" if selected is not None else "failed",
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        }
        dataset_results.append(result)
        (dataset_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"dataset_complete": dataset, "gate": result["calibration_gate"], "selected": selected}), flush=True)
    all_pass = all(row["calibration_gate"] == "passed" for row in dataset_results)
    summary = {
        "experiment_id": "GRAM_PHASE11_BW3_P1_TRAIN_PREFIX_ADMISSION_V1",
        "status": "completed",
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
        "datasets": dataset_results,
        "p1_gate": {"status": "passed" if all_pass else "failed", "both_domains_calibrated": all_pass},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"p1_gate": summary["p1_gate"]}), flush=True)


if __name__ == "__main__":
    main()
