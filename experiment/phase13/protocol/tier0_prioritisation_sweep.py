"""Phase-13 Tier-0 Experiment A2: learned vs random user prioritisation, swept
across the entire warm-cost front.

RESEARCH QUESTION (single):
    Experiment A showed that at P6's own operating point, learned gating beats a
    random subset of equal warm cost.  But P6 sits at only one point.  The real
    question for the paper is:

        At EVERY warm budget, does ranking users by a learned utility beat
        picking users at random?

    P6 emits a per-user `predicted_utilities` score for portfolio@2 / @3.  We use
    that score purely as a RANKING of users, sweep the intervention coverage from
    0 to 100%, and compare against random selection at the same coverage.  Both
    arms use the identical frozen portfolio insertion rule, so the ONLY thing
    that differs is which users get intervened on.

    This yields two Pareto fronts on the same axes (warm cost vs cold gain) and
    settles whether the learned signal has value, independent of any threshold.

DESIGN NOTES:
  - Evaluation-only.  No training, no deployable parameter selection, no test.
  - The learned utilities were fit by P6 on out-of-fold data; each user's score
    comes from a model that did not see that user.  Using them only to rank is
    strictly weaker than P6's own thresholding, so this is a fair - if anything
    conservative - reading of the learned signal.
  - Random arm is averaged over N seeds with the spread reported.
  - An ORACLE arm (rank users by realised gain) is reported as a ceiling only.
    It is not deployable and is never compared as a candidate method.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ANCHOR_PREFIX = 7
SEED = 20260819
N_RANDOM_SEEDS = 20
COVERAGE_GRID = [round(0.05 * i, 2) for i in range(0, 21)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--p0-predictions", required=True)
    p.add_argument("--p6-predictions", required=True)
    p.add_argument("--cold-items", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--size", type=int, default=2, choices=(2, 3))
    return p.parse_args()


def unique_in_order(items):
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def portfolio_ranking(gram, resolver, candidates, size):
    gram = unique_in_order(gram)
    resolver = unique_in_order(resolver)
    portfolio = unique_in_order(candidates)[:size]
    if len(portfolio) != size:
        raise ValueError("insufficient candidates")
    anchor = 10 - size
    return unique_in_order([*gram[:anchor], *portfolio, *gram[anchor:], *resolver])


def read_jsonl(path):
    with Path(path).open() as fh:
        return [json.loads(l) for l in fh if l.strip()]


def read_set(path):
    with Path(path).open() as fh:
        return {l.strip() for l in fh if l.strip()}


def hit_ndcg(ranking, target, k):
    for i, it in enumerate(ranking[:k]):
        if it == target:
            return 1.0, 1.0 / math.log2(i + 2)
    return 0.0, 0.0


def build(p0_records, p6_records, cold_items, size):
    p6_by_uid = {str(r["user_id"]): r for r in p6_records}
    rows = []
    for src in p0_records:
        uid = str(src["user_id"])
        target = str(src["target"])
        gram = unique_in_order(src["v0_top50"])
        resolver = unique_in_order(src["resolver_top50"])
        protected = set(gram[:ANCHOR_PREFIX])
        cands = [it for it in resolver
                 if it in cold_items and it not in protected][:3]
        if len(cands) < 3:
            continue
        p6row = p6_by_uid[uid]
        util = float(p6row["predicted_utilities"][f"portfolio@{size}"])

        rank_v0 = gram
        rank_pf = portfolio_ranking(gram, resolver, cands, size)
        h10_v0, n10_v0 = hit_ndcg(rank_v0, target, 10)
        h50_v0, _ = hit_ndcg(rank_v0, target, 50)
        h10_pf, n10_pf = hit_ndcg(rank_pf, target, 10)
        h50_pf, _ = hit_ndcg(rank_pf, target, 50)
        rows.append({
            "is_cold": bool(src["is_cold"]),
            "utility": util,
            "v0": (h10_v0, n10_v0, h50_v0),
            "pf": (h10_pf, n10_pf, h50_pf),
            # realised gain, used ONLY for the non-deployable oracle ceiling
            "_realised_n10_gain": n10_pf - n10_v0,
        })
    return rows


def evaluate(rows, selected_mask):
    """Aggregate metrics given a boolean per-user intervention mask."""
    cold_h50, cold_h10, cold_n10, warm_n10, all_n10 = [], [], [], [], []
    for r, sel in zip(rows, selected_mask):
        h10, n10, h50 = r["pf"] if sel else r["v0"]
        all_n10.append(n10)
        if r["is_cold"]:
            cold_h50.append(h50); cold_h10.append(h10); cold_n10.append(n10)
        else:
            warm_n10.append(n10)
    return {
        "cold_h50": float(np.mean(cold_h50)),
        "cold_h10": float(np.mean(cold_h10)),
        "cold_n10": float(np.mean(cold_n10)),
        "warm_n10": float(np.mean(warm_n10)),
        "all_n10": float(np.mean(all_n10)),
        "cold_h50_events": float(np.sum(cold_h50)),
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = build(read_jsonl(args.p0_predictions),
                 read_jsonl(args.p6_predictions),
                 read_set(args.cold_items), args.size)
    n = len(rows)
    n_cold = sum(1 for r in rows if r["is_cold"])
    warm_v0 = evaluate(rows, [False] * n)["warm_n10"]

    util_order = np.argsort([-r["utility"] for r in rows])       # learned ranking
    oracle_order = np.argsort([-r["_realised_n10_gain"] for r in rows])  # ceiling

    curve = []
    for cov in COVERAGE_GRID:
        k = int(round(cov * n))

        m_learned = np.zeros(n, dtype=bool); m_learned[util_order[:k]] = True
        learned = evaluate(rows, m_learned)

        m_oracle = np.zeros(n, dtype=bool); m_oracle[oracle_order[:k]] = True
        oracle = evaluate(rows, m_oracle)

        rand_runs = []
        for s in range(N_RANDOM_SEEDS):
            rng = np.random.default_rng(SEED + s)
            m = np.zeros(n, dtype=bool)
            m[rng.choice(n, size=k, replace=False)] = True
            rand_runs.append(evaluate(rows, m))
        rand = {k2: float(np.mean([x[k2] for x in rand_runs])) for k2 in rand_runs[0]}
        rand_sd = {k2: float(np.std([x[k2] for x in rand_runs])) for k2 in rand_runs[0]}

        curve.append({
            "coverage": cov, "k": k,
            "learned": learned, "random": rand, "random_sd": rand_sd,
            "oracle": oracle,
            "learned_warm_retention": learned["warm_n10"] / warm_v0,
            "random_warm_retention": rand["warm_n10"] / warm_v0,
            "learned_minus_random_cold_h50": learned["cold_h50"] - rand["cold_h50"],
            "learned_beats_random_cold_h50": bool(
                learned["cold_h50"] > rand["cold_h50"] + 2 * rand_sd["cold_h50"]),
        })

    n_beat = sum(1 for c in curve if c["k"] > 0 and c["learned_beats_random_cold_h50"])
    n_pts = sum(1 for c in curve if c["k"] > 0)

    summary = {
        "experiment": "TIER0_A2_LEARNED_VS_RANDOM_PRIORITISATION_SWEEP",
        "domain": args.domain,
        "portfolio_size": args.size,
        "evaluation_only": True,
        "test_read": False,
        "n_users": n, "n_cold": n_cold,
        "warm_v0_n10": warm_v0,
        "n_coverage_points_where_learned_beats_random_by_2sd": n_beat,
        "n_coverage_points": n_pts,
        "curve": curve,
        "seeds": {"seed": SEED, "n_random_seeds": N_RANDOM_SEEDS},
        "note": ("Oracle arm ranks users by realised gain; it is a non-deployable "
                 "ceiling and is not a candidate method."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Tier-0 A2: learned vs random user prioritisation "
          f"[{args.domain}, portfolio@{args.size}] ===")
    print(f"users={n}  cold={n_cold}  warm_v0_N@10={warm_v0:.6f}\n")
    hdr = (f"{'cov':>5s} {'warmRet_L':>9s} {'coldH50_L':>9s} {'coldH50_R':>9s} "
           f"{'+-sd':>7s} {'L-R':>9s} {'coldH50_O':>9s} {'allN10_L':>9s} {'allN10_R':>9s} {'>2sd':>5s}")
    print(hdr); print("-" * len(hdr))
    for c in curve:
        if c["k"] == 0:
            continue
        print(f"{c['coverage']:5.2f} {c['learned_warm_retention']:9.4f} "
              f"{c['learned']['cold_h50']:9.6f} {c['random']['cold_h50']:9.6f} "
              f"{c['random_sd']['cold_h50']:7.6f} "
              f"{c['learned_minus_random_cold_h50']:+9.6f} "
              f"{c['oracle']['cold_h50']:9.6f} "
              f"{c['learned']['all_n10']:9.6f} {c['random']['all_n10']:9.6f} "
              f"{'YES' if c['learned_beats_random_cold_h50'] else '-':>5s}")
    print(f"\nlearned beats random by >2sd at {n_beat}/{n_pts} coverage points")
    print(f"wrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
