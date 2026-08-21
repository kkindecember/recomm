"""Synthesize Stage14-0B dual-domain per-user oracle-prefix diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SOURCES = ("uniform", "popularity", "catalog_text", "r2")
BASELINES = SOURCES[:-1]
QUARTILES = (
    ("q1", 0.0, 0.25),
    ("q2", 0.25, 0.50),
    ("q3", 0.50, 0.75),
    ("q4", 0.75, 1.0000001),
)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_ci(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95": [mean, mean]}
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci95": [mean - half, mean + half]}


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def user_bin_value(row: dict, source: str, low: float, high: float, field: str) -> float:
    values = [
        float(point[field])
        for point in row["teacher_profiles"][source]["depth"]
        if low < float(point["normalized_depth"]) <= high
    ]
    if not values:
        raise RuntimeError(
            f"No normalized-depth observation for user={row['user_id']} source={source} bin=({low},{high}]"
        )
    return statistics.fmean(values)


def summarize_slice(rows: list[dict]) -> dict:
    result = {"n": len(rows), "item_rank": {}, "normalized_depth_quartiles": {}}
    for source in SOURCES:
        ranks = [float(row["teacher_profiles"][source]["item_rank"]) for row in rows]
        result["item_rank"][source] = {
            "mean": statistics.fmean(ranks),
            "median": median(ranks),
            "hit50": statistics.fmean(rank <= 50 for rank in ranks),
        }

    for label, low, high in QUARTILES:
        source_values = {}
        for source in SOURCES:
            masses = [user_bin_value(row, source, low, high, "target_prefix_mass") for row in rows]
            ranks = [user_bin_value(row, source, low, high, "target_prefix_rank") for row in rows]
            source_values[source] = {"mass": masses, "rank": ranks}

        strongest_mass = max(BASELINES, key=lambda name: statistics.fmean(source_values[name]["mass"]))
        strongest_rank = min(BASELINES, key=lambda name: statistics.fmean(source_values[name]["rank"]))
        r2_mass = source_values["r2"]["mass"]
        r2_rank = source_values["r2"]["rank"]
        mass_delta = [a - b for a, b in zip(r2_mass, source_values[strongest_mass]["mass"])]
        rank_advantage = [b - a for a, b in zip(r2_rank, source_values[strongest_rank]["rank"])]
        strict_mass_delta = [
            a - max(source_values[name]["mass"][idx] for name in BASELINES)
            for idx, a in enumerate(r2_mass)
        ]
        strict_rank_advantage = [
            min(source_values[name]["rank"][idx] for name in BASELINES) - a
            for idx, a in enumerate(r2_rank)
        ]
        result["normalized_depth_quartiles"][label] = {
            "normalized_depth_interval": [low, min(high, 1.0)],
            "sources": {
                source: {
                    "target_prefix_mass_mean": statistics.fmean(values["mass"]),
                    "target_prefix_rank_mean": statistics.fmean(values["rank"]),
                    "target_prefix_rank_median": median(values["rank"]),
                }
                for source, values in source_values.items()
            },
            "strongest_fixed_mass_baseline": strongest_mass,
            "strongest_fixed_rank_baseline": strongest_rank,
            "r2_minus_strongest_fixed_mass": mean_ci(mass_delta),
            "strongest_fixed_minus_r2_rank": mean_ci(rank_advantage),
            "r2_mass_win_rate_vs_strongest_fixed": statistics.fmean(value > 0 for value in mass_delta),
            "r2_rank_win_rate_vs_strongest_fixed": statistics.fmean(value > 0 for value in rank_advantage),
            "r2_minus_per_user_oracle_mass": mean_ci(strict_mass_delta),
            "per_user_oracle_minus_r2_rank": mean_ci(strict_rank_advantage),
        }
    return result


def summarize_domain(path: Path) -> dict:
    rows = load_rows(path)
    slices = {
        "all": rows,
        "warm": [row for row in rows if not row["is_cold"]],
        "cold": [row for row in rows if row["is_cold"]],
    }
    return {"n": len(rows), "slices": {name: summarize_slice(values) for name, values in slices.items()}}


def decide(domains: dict) -> dict:
    checks = []
    for domain, payload in domains.items():
        cold = payload["slices"]["cold"]["normalized_depth_quartiles"]
        for quartile, values in cold.items():
            mass_ci = values["r2_minus_strongest_fixed_mass"]["ci95"]
            rank_ci = values["strongest_fixed_minus_r2_rank"]["ci95"]
            checks.append(
                {
                    "domain": domain,
                    "quartile": quartile,
                    "mass_significantly_better": mass_ci[0] > 0,
                    "rank_significantly_better": rank_ci[0] > 0,
                }
            )
    passed = all(row["mass_significantly_better"] and row["rank_significantly_better"] for row in checks)
    return {
        "criterion": "On cold validation users in both domains and every normalized-depth quartile, R2 must beat the strongest fixed baseline in paired mean target-prefix mass and rank with 95% normal CIs above zero.",
        "checks": checks,
        "passed": passed,
        "route_decision": "PASS_PATH_TRANSFER_GATE" if passed else "FAIL_STOP_PATH_TRANSFER",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toys", required=True, type=Path)
    parser.add_argument("--beauty", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    domains = {"toys": summarize_domain(args.toys), "beauty": summarize_domain(args.beauty)}
    result = {
        "status": "completed",
        "split": "validation",
        "test_predictions_opened": False,
        "domains": domains,
        "gate": decide(domains),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({"status": "completed", "route_decision": result["gate"]["route_decision"]}))


if __name__ == "__main__":
    main()
