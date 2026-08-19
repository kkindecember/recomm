"""Phase-13 v1-R² B1 cross-domain portfolio confirmation analysis.

Reads frozen P0 predictions (v0_top50 / resolver_top50) and P6 predictions
(portfolio_candidates / p6_top50), reconstructs the unconditional portfolio
rankings with the frozen anchor rule, and reports the full Pareto front with
paired bootstrap confidence intervals.

This script is evaluation-only: it never trains, never selects parameters, and
never opens a test split.  Gate arithmetic is frozen in main() and applies only
to the pre-registered primary candidate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PORTFOLIO_SIZES = (2, 3)
ANCHOR_PREFIX = 7
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260818
PRIMARY_CANDIDATE = "unconditional_portfolio2"
MIN_COLD_H50_EVENTS = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p0-predictions", required=True)
    p.add_argument(
        "--p6-predictions",
        default=None,
        help="Optional. When absent the P6 comparison point is omitted; "
             "portfolio candidates are recomputed from P0 and cold state.",
    )
    p.add_argument("--cold-items", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--domain", required=True)
    return p.parse_args()


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def portfolio_ranking(
    gram_items: list[str], resolver_items: list[str], candidates: list[str], size: int
) -> list[str]:
    """Frozen anchor rule, identical to candidate_portfolio.portfolio_ranking."""
    if size not in PORTFOLIO_SIZES:
        raise ValueError(f"Unsupported portfolio size: {size}")
    gram = unique_in_order(gram_items)
    resolver = unique_in_order(resolver_items)
    portfolio = unique_in_order(candidates)[:size]
    if len(portfolio) != size:
        raise ValueError(f"Portfolio has only {len(portfolio)} unique candidates")
    anchor_count = 10 - size
    return unique_in_order(
        [*gram[:anchor_count], *portfolio, *gram[anchor_count:], *resolver]
    )


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_set(path: Path) -> set[str]:
    with path.open() as handle:
        return {line.strip() for line in handle if line.strip()}


def hit_and_ndcg(ranking: list[str], target: str, k: int) -> tuple[float, float]:
    for index, item in enumerate(ranking[:k]):
        if item == target:
            return 1.0, 1.0 / math.log2(index + 2)
    return 0.0, 0.0


def build_rows(
    p0_records: list[dict], p6_records: list[dict] | None, cold_items: set[str]
) -> tuple[list[dict], list[str]]:
    p6_by_uid = (
        {str(row["user_id"]): row for row in p6_records} if p6_records else None
    )
    rows: list[dict] = []
    method_names: list[str] = []
    skipped_insufficient_candidates = 0
    for source in p0_records:
        uid = str(source["user_id"])
        target = str(source["target"])
        gram = unique_in_order(source["v0_top50"])
        resolver_items = unique_in_order(source["resolver_top50"])

        # Candidates are recomputed from frozen P0 inputs and cold state.
        protected = set(gram[:ANCHOR_PREFIX])
        candidates = [
            item for item in resolver_items
            if item in cold_items and item not in protected
        ][:3]
        if len(candidates) < 3:
            # Cannot form the frozen 3-candidate portfolio; excluded and counted.
            skipped_insufficient_candidates += 1
            continue

        if p6_by_uid is not None:
            if uid not in p6_by_uid:
                raise ValueError(f"P6 predictions missing user {uid}")
            stored = unique_in_order(p6_by_uid[uid]["portfolio_candidates"])[:3]
            if candidates != stored:
                raise ValueError(
                    f"Portfolio candidate mismatch for {uid}: "
                    f"recomputed={candidates} stored={stored}"
                )

        variants = {
            "v0_gram": gram,
            "resolver_only": resolver_items,
            "unconditional_portfolio2": portfolio_ranking(
                gram, resolver_items, candidates, 2
            ),
            "unconditional_portfolio3": portfolio_ranking(
                gram, resolver_items, candidates, 3
            ),
        }
        if p6_by_uid is not None:
            variants["p6_candidate_portfolio"] = unique_in_order(
                p6_by_uid[uid]["p6_top50"]
            )

        record = {
            "user_id": uid,
            "is_cold": target in cold_items,
            "scores": {},
        }
        for name, ranking in variants.items():
            hit10, ndcg10 = hit_and_ndcg(ranking, target, 10)
            hit50, _ = hit_and_ndcg(ranking, target, 50)
            record["scores"][name] = {
                "hit@10": hit10, "ndcg@10": ndcg10, "hit@50": hit50,
            }
        rows.append(record)
        method_names = list(variants.keys())
    if not rows:
        raise ValueError("No evaluable users after candidate construction")
    return rows, method_names, skipped_insufficient_candidates


def aggregate(rows: list[dict], method: str, metric: str, subset: str) -> float:
    values = [
        row["scores"][method][metric] for row in rows
        if subset == "all"
        or (subset == "cold" and row["is_cold"])
        or (subset == "warm" and not row["is_cold"])
    ]
    return float(np.mean(values)) if values else 0.0


def event_count(rows: list[dict], method: str, metric: str, subset: str) -> int:
    return int(round(aggregate(rows, method, metric, subset) * subset_size(rows, subset)))


def subset_size(rows: list[dict], subset: str) -> int:
    if subset == "all":
        return len(rows)
    want = subset == "cold"
    return sum(1 for row in rows if row["is_cold"] == want)


def paired_bootstrap(
    rows: list[dict], method: str, metric: str, subset: str, rng: np.random.Generator
) -> dict:
    selected = [
        row for row in rows
        if subset == "all"
        or (subset == "cold" and row["is_cold"])
        or (subset == "warm" and not row["is_cold"])
    ]
    treatment = np.array([row["scores"][method][metric] for row in selected])
    control = np.array([row["scores"]["v0_gram"][metric] for row in selected])
    delta = treatment - control
    if delta.size == 0:
        return {"observed": 0.0, "ci_low": 0.0, "ci_high": 0.0, "verdict": "INCONCLUSIVE"}
    samples = rng.choice(delta, size=(BOOTSTRAP_RESAMPLES, delta.size), replace=True)
    means = samples.mean(axis=1)
    low = float(np.percentile(means, 2.5))
    high = float(np.percentile(means, 97.5))
    verdict = "PASS" if low > 0 else ("FAIL" if high < 0 else "INCONCLUSIVE")
    return {
        "observed": float(delta.mean()),
        "ci_low": low,
        "ci_high": high,
        "verdict": verdict,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    p0_path = Path(args.p0_predictions).resolve()
    cold_path = Path(args.cold_items).resolve()
    for path in (p0_path, cold_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if "test" in p0_path.name:
        raise ValueError("Refusing to read a test prediction file")

    cold_items = read_set(cold_path)
    p6_records = None
    if args.p6_predictions:
        p6_path = Path(args.p6_predictions).resolve()
        if not p6_path.is_file():
            raise FileNotFoundError(p6_path)
        if "test" in p6_path.name:
            raise ValueError("Refusing to read a test prediction file")
        p6_records = read_jsonl(p6_path)
    rows, methods, skipped = build_rows(read_jsonl(p0_path), p6_records, cold_items)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    pareto = {}
    for method in methods:
        pareto[method] = {
            subset: {
                metric: aggregate(rows, method, metric, subset)
                for metric in ("hit@10", "ndcg@10", "hit@50")
            }
            for subset in ("all", "warm", "cold")
        }
        pareto[method]["cold_hit50_events"] = event_count(rows, method, "hit@50", "cold")
        pareto[method]["cold_hit10_events"] = event_count(rows, method, "hit@10", "cold")
        baseline_warm = aggregate(rows, "v0_gram", "ndcg@10", "warm")
        pareto[method]["warm_retention_ndcg10"] = (
            aggregate(rows, method, "ndcg@10", "warm") / baseline_warm
            if baseline_warm > 0 else None
        )

    intervals = {
        method: {
            "overall_ndcg@10": paired_bootstrap(rows, method, "ndcg@10", "all", rng),
            "cold_hit@50": paired_bootstrap(rows, method, "hit@50", "cold", rng),
            "cold_ndcg@10": paired_bootstrap(rows, method, "ndcg@10", "cold", rng),
            "warm_ndcg@10": paired_bootstrap(rows, method, "ndcg@10", "warm", rng),
        }
        for method in methods if method != "v0_gram"
    }

    # Frozen Gate: primary candidate only, with the pre-registered event-density guard.
    primary = intervals[PRIMARY_CANDIDATE]
    baseline_cold_h50_events = event_count(rows, "v0_gram", "hit@50", "cold")
    density_guard_triggered = baseline_cold_h50_events < MIN_COLD_H50_EVENTS
    overall_gate = primary["overall_ndcg@10"]["verdict"]
    cold_gate = primary["cold_hit@50"]["verdict"]

    if density_guard_triggered:
        verdict = "INCONCLUSIVE"
        rationale = (
            f"Event-density guard: v0 cold H@50 events={baseline_cold_h50_events} "
            f"< {MIN_COLD_H50_EVENTS}; FAIL is not reportable at this density."
        )
    elif overall_gate == "PASS" and cold_gate == "PASS":
        verdict = "PASS"
        rationale = "Both primary intervals exclude zero from above."
    elif overall_gate == "FAIL" or cold_gate == "FAIL":
        verdict = "FAIL"
        rationale = f"overall={overall_gate}, cold_h@50={cold_gate}."
    else:
        verdict = "INCONCLUSIVE"
        rationale = f"overall={overall_gate}, cold_h@50={cold_gate}."

    summary = {
        "experiment_id": f"GRAM_PHASE13_V1_R2_{args.domain.upper()}_B1_PORTFOLIO_CONFIRMATION",
        "status": "completed",
        "domain": args.domain,
        "split": "validation",
        "test_predictions_opened": False,
        "primary_candidate": PRIMARY_CANDIDATE,
        "verdict": verdict,
        "verdict_rationale": rationale,
        "n_users": len(rows),
        "n_cold_users": subset_size(rows, "cold"),
        "n_warm_users": subset_size(rows, "warm"),
        "n_skipped_insufficient_candidates": skipped,
        "p6_comparison_included": p6_records is not None,
        "baseline_cold_hit50_events": baseline_cold_h50_events,
        "event_density_guard": {
            "threshold": MIN_COLD_H50_EVENTS,
            "triggered": density_guard_triggered,
        },
        "frozen_parameters": {
            "anchor_prefix": ANCHOR_PREFIX,
            "portfolio2_positions": "ranks 9-10",
            "portfolio3_positions": "ranks 8-10",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "migrated_from": "Toys; retuning on this domain is forbidden",
        },
        "pareto_front": pareto,
        "paired_bootstrap_vs_v0": intervals,
        "inputs": {
            "p0_predictions": str(p0_path),
            "p6_predictions": str(args.p6_predictions) if args.p6_predictions else None,
            "cold_items": str(cold_path),
        },
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"verdict": verdict, "rationale": rationale}, indent=2))


if __name__ == "__main__":
    main()
