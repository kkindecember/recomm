"""Phase-13 Tier-0 Experiment B: candidate-pool ceiling decomposition.

RESEARCH QUESTION (single):
    Every slate-allocation mechanism tried in P1-P7 and R2-v2 operates on the
    resolver's candidate list.  Before designing a ninth allocator, quantify how
    much headroom allocation can possibly have:

        For cold users, WHERE is the correct item in the resolver ranking?

    If the target is absent from the resolver's top-K for the large majority of
    cold users, then no allocator - however clever - can recover them, and the
    binding constraint is resolver RECALL, not slate allocation.

OUTPUT:
  - Distribution of the target's resolver rank for cold users (top-1/3/10/50/absent).
  - The same for the GRAM v0 ranking, for contrast.
  - Union coverage: fraction of cold users for whom the target is reachable by
    EITHER path within top-50.
  - The implied ceiling for a portfolio that inserts N candidates: it can only
    help users whose target is inside the eligible candidate slice.

This is a read-only descriptive decomposition.  No training, no gates, no test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ANCHOR_PREFIX = 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--p0-predictions", required=True)
    p.add_argument("--cold-items", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--domain", required=True)
    return p.parse_args()


def unique_in_order(items):
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def read_jsonl(path):
    with Path(path).open() as fh:
        return [json.loads(l) for l in fh if l.strip()]


def read_set(path):
    with Path(path).open() as fh:
        return {l.strip() for l in fh if l.strip()}


def rank_of(ranking, target):
    """1-based rank, or None if absent."""
    for i, it in enumerate(ranking):
        if it == target:
            return i + 1
    return None


def bucket(rank):
    if rank is None:
        return "absent_from_top50"
    if rank == 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_3"
    if rank <= 10:
        return "rank_4_10"
    if rank <= 50:
        return "rank_11_50"
    return "absent_from_top50"


BUCKETS = ["rank_1", "rank_2_3", "rank_4_10", "rank_11_50", "absent_from_top50"]


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.p0_predictions)
    cold_items = read_set(args.cold_items)

    cold_rows, warm_rows = [], []
    for src in records:
        target = str(src["target"])
        gram = unique_in_order(src["v0_top50"])
        resolver = unique_in_order(src["resolver_top50"])
        protected = set(gram[:ANCHOR_PREFIX])
        # The eligible candidate slice the portfolio actually draws from.
        eligible = [it for it in resolver
                    if it in cold_items and it not in protected]
        row = {
            "resolver_rank": rank_of(resolver, target),
            "gram_rank": rank_of(gram, target),
            "eligible_rank": rank_of(eligible, target),
            "n_eligible": len(eligible),
        }
        (cold_rows if src["is_cold"] else warm_rows).append(row)

    def dist(rows, key):
        counts = {b: 0 for b in BUCKETS}
        for r in rows:
            counts[bucket(r[key])] += 1
        n = len(rows)
        return {b: {"n": counts[b], "pct": 100.0 * counts[b] / n} for b in BUCKETS}

    n_cold = len(cold_rows)
    res_dist = dist(cold_rows, "resolver_rank")
    gram_dist = dist(cold_rows, "gram_rank")
    elig_dist = dist(cold_rows, "eligible_rank")

    reachable_either = sum(
        1 for r in cold_rows
        if r["resolver_rank"] is not None or r["gram_rank"] is not None)
    reachable_resolver = sum(1 for r in cold_rows if r["resolver_rank"] is not None)
    reachable_gram = sum(1 for r in cold_rows if r["gram_rank"] is not None)

    # Portfolio ceiling: inserting N candidates can only fix users whose target
    # sits within the first N of the eligible slice.
    ceilings = {}
    for n_ins in (1, 2, 3, 5, 10):
        hit = sum(1 for r in cold_rows
                  if r["eligible_rank"] is not None and r["eligible_rank"] <= n_ins)
        ceilings[f"portfolio@{n_ins}"] = {
            "cold_users_fixable": hit,
            "pct_of_cold": 100.0 * hit / n_cold,
        }

    summary = {
        "experiment": "TIER0_B_CANDIDATE_POOL_CEILING_DECOMPOSITION",
        "domain": args.domain,
        "evaluation_only": True,
        "test_read": False,
        "n_cold_users": n_cold,
        "n_warm_users": len(warm_rows),
        "cold_target_rank_distribution": {
            "resolver_top50": res_dist,
            "gram_v0_top50": gram_dist,
            "eligible_candidate_slice": elig_dist,
        },
        "cold_reachability": {
            "by_resolver_top50": {"n": reachable_resolver,
                                  "pct": 100.0 * reachable_resolver / n_cold},
            "by_gram_top50": {"n": reachable_gram,
                              "pct": 100.0 * reachable_gram / n_cold},
            "by_either": {"n": reachable_either,
                          "pct": 100.0 * reachable_either / n_cold},
        },
        "portfolio_insertion_ceiling": ceilings,
        "mean_eligible_candidates": float(np.mean([r["n_eligible"] for r in cold_rows])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Tier-0 B: candidate-pool ceiling [{args.domain}] ===")
    print(f"cold users = {n_cold}\n")
    print("Where is the cold target in each ranking?")
    hdr = f"{'bucket':22s} {'resolver':>18s} {'GRAM v0':>18s} {'eligible slice':>18s}"
    print(hdr); print("-" * len(hdr))
    for b in BUCKETS:
        print(f"{b:22s} {res_dist[b]['n']:8d} {res_dist[b]['pct']:8.2f}% "
              f"{gram_dist[b]['n']:8d} {gram_dist[b]['pct']:8.2f}% "
              f"{elig_dist[b]['n']:8d} {elig_dist[b]['pct']:8.2f}%")

    print(f"\nCold reachability within top-50:")
    for k, v in summary["cold_reachability"].items():
        print(f"  {k:20s} {v['n']:6d}  {v['pct']:6.2f}%")

    print(f"\nCeiling: cold users a portfolio of size N could POSSIBLY fix")
    for k, v in ceilings.items():
        print(f"  {k:14s} {v['cold_users_fixable']:6d}  {v['pct_of_cold']:6.2f}% of cold")
    print(f"\nmean eligible candidates per cold user = "
          f"{summary['mean_eligible_candidates']:.2f}")
    print(f"wrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
