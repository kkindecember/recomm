"""Recompute S17-4 metrics from saved rankings; never load or run a model.

The historical TSV header describes four metrics, but the rows contain twelve.
Read gold/predictions from the three trailing payload fields, independently
recompute ranking metrics, and validate against both logged aggregate values
and the twelve numeric row fields. Historical artifacts remain unchanged.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/phase17/s4_p1_targeted/run-0001/summary.json"
CUTOFFS = (1, 3, 5, 10, 20, 50)
METRICS = tuple(f"{name}@{k}" for name in ("hit", "ndcg") for k in CUTOFFS)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rankings(path):
    records = {}
    footer = {}
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n").split("\t")
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 1 and fields[0].partition(":")[0] in METRICS:
                name, _, value = fields[0].partition(":")
                if name in footer:
                    raise ValueError("Duplicate footer metric")
                footer[name] = float(value)
                continue
            if len(fields) != 16:
                raise ValueError(f"Unexpected row width in {path}: {len(fields)}")
            if footer:
                raise ValueError("Prediction row after metrics footer")
            user, gold, predictions = fields[0], fields[-3], fields[-2].split("||")
            if user in records:
                raise ValueError(f"Duplicate user in {path}")
            rank = predictions.index(gold) + 1 if gold in predictions else math.inf
            values = np.asarray(
                [float(rank <= k) for k in CUTOFFS]
                + [1.0 / math.log2(rank + 1) if rank <= k else 0.0 for k in CUTOFFS]
            )
            if not np.allclose(values, np.asarray(fields[1:-3], dtype=float), atol=1e-7, rtol=0):
                raise ValueError(f"Ranking/row-metric mismatch in {path}")
            records[user] = values
    if set(footer) != set(METRICS):
        raise ValueError("Incomplete metrics footer")
    return header, records, footer


def paired_effect(deltas, seed=2023, replicates=2000):
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, replicates, 32):
        count = min(32, replicates - start)
        indices = rng.integers(0, len(deltas), size=(count, len(deltas)))
        samples.extend(deltas[indices].mean(axis=1).tolist())
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"mean_delta": float(deltas.mean()), "ci95_low": float(low),
            "ci95_high": float(high), "replicates": replicates, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    report = {"source_summary": str(SOURCE.relative_to(ROOT)),
              "source_summary_sha256": sha256(SOURCE),
              "status": "RECOMPUTED_FROM_SAVED_RANKINGS_NO_NEW_INFERENCE",
              "model_runs": 0, "metric_order_in_rows": list(METRICS), "arms": {}}
    matrices = {}
    common_users = None
    for arm in source["formal_results"]:
        path = ROOT / arm["prediction_path"]
        digest = sha256(path)
        if digest != arm["prediction_sha256"]:
            raise ValueError(f"Historical prediction SHA mismatch: {path}")
        header, records, footer = read_rankings(path)
        users = sorted(records)
        if common_users is not None and users != common_users:
            raise ValueError("User cohorts differ")
        common_users = users
        matrix = np.stack([records[user] for user in users])
        means = dict(zip(METRICS, matrix.mean(axis=0).tolist()))
        logged = arm["parsed"]["validation_metrics"]
        discrepancy = max(abs(means[name] - reference[name])
                          for reference in (logged, footer) for name in METRICS)
        if discrepancy > 1e-7:
            raise ValueError(f"Logged aggregate mismatch: {discrepancy}")
        matrices[arm["arm_id"]] = matrix
        report["arms"][arm["arm_id"]] = {
            "prediction_path": arm["prediction_path"], "prediction_sha256": digest,
            "users": len(users), "header_columns": len(header), "data_columns": 16,
            "metrics": means, "max_difference_vs_logged_aggregates": discrepancy,
        }
    for name, matrix in matrices.items():
        if name == "gram_continue":
            continue
        report["arms"][name]["paired_vs_gram_continue"] = {
            metric: paired_effect(matrix[:, METRICS.index(metric)]
                                  - matrices["gram_continue"][:, METRICS.index(metric)])
            for metric in ("hit@10", "ndcg@10")
        }
    report["finding"] = (
        "The stale header makes historical DictReader parse H@3 as H@10 and "
        "H@10 as NDCG@10. Old paired deltas/intervals do not describe their "
        "labels. Corrected figures are exploratory and do not reopen old gates."
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
