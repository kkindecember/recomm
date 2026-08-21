"""Phase-13 Tier-0 Experiment A: warm-cost-matched random portfolio baseline.

RESEARCH QUESTION (single):
    The claim "simple unconditional portfolio beats P1-P7 learned gating" is
    currently confounded: portfolio@2 spends 3.65pp of warm NDCG to buy its cold
    gain, while P6 spends only 0.44pp.  They sit at different operating points,
    so the comparison is not a like-for-like test of "simple vs learned".

    This script removes the confound by constructing a *coverage-matched random*
    baseline: apply the frozen portfolio rule to a random subset of users, with
    the subset fraction chosen so the warm cost matches P6's warm cost.  If the
    random baseline reaches P6's cold performance at equal warm cost, then P6's
    learned features carry no information beyond "intervene on some users".

DESIGN NOTES:
  - Evaluation-only.  Never trains, never selects a deployable parameter, never
    opens a test split.  Reuses frozen P0/P6 validation predictions.
  - The random subset is target-free: selection depends only on a seeded RNG,
    never on the target, is_cold, or any outcome.
  - Coverage is calibrated on WARM cost only (the constraint), then cold is read
    out as the outcome.  This mirrors how P6 itself was constrained.
  - Multiple random seeds are averaged so the conclusion is not one lucky draw,
    and the across-seed spread is reported.

This is an exploratory diagnostic, not a pre-registered efficacy gate.  It is
designed to be able to REFUTE the project's headline claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ANCHOR_PREFIX = 7
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260819
N_RANDOM_SEEDS = 20
# Coverage grid searched to match a warm-cost target (target-free).
COVERAGE_GRID = [round(0.05 * i, 2) for i in range(1, 21)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p0-predictions", required=True)
    p.add_argument("--p6-predictions", required=True,
                   help="P6 predictions supply the learned-gating comparison point.")
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


def portfolio_ranking(gram, resolver, candidates, size):
    """Frozen anchor rule, identical to b1_portfolio_confirmation.portfolio_ranking."""
    gram = unique_in_order(gram)
    resolver = unique_in_order(resolver)
    portfolio = unique_in_order(candidates)[:size]
    if len(portfolio) != size:
        raise ValueError(f"Portfolio has only {len(portfolio)} unique candidates")
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


def build_rows(p0_records, p6_records, cold_items):
    """Per-user frozen rankings for every comparison point."""
    p6_by_uid = {str(r["user_id"]): r for r in p6_records}
    rows = []
    skipped = 0
    for src in p0_records:
        uid = str(src["user_id"])
        target = str(src["target"])
        gram = unique_in_order(src["v0_top50"])
        resolver = unique_in_order(src["resolver_top50"])

        protected = set(gram[:ANCHOR_PREFIX])
        candidates = [
            it for it in resolver if it in cold_items and it not in protected
        ][:3]
        if len(candidates) < 3:
            skipped += 1
            continue
        if uid not in p6_by_uid:
            raise ValueError(f"P6 predictions missing user {uid}")
        stored = unique_in_order(p6_by_uid[uid]["portfolio_candidates"])[:3]
        if candidates != stored:
            raise ValueError(f"Candidate mismatch for {uid}")

        variants = {
            "v0_gram": gram,
            "P6_learned": unique_in_order(p6_by_uid[uid]["p6_top50"]),
            "portfolio@2_always": portfolio_ranking(gram, resolver, candidates, 2),
            "portfolio@3_always": portfolio_ranking(gram, resolver, candidates, 3),
        }
        rec = {
            "user_id": uid,
            "is_cold": bool(src["is_cold"]),
            # Pre-materialise both the intervened and non-intervened ranking so the
            # random-subset variants are a pure per-user selection between them.
            "_rank_v0": gram,
            "_rank_p2": variants["portfolio@2_always"],
            "_rank_p3": variants["portfolio@3_always"],
        }
        for name, rl in variants.items():
            h10, n10 = hit_ndcg(rl, target, 10)
            h50, _ = hit_ndcg(rl, target, 50)
            rec[name] = (h10, n10, h50)
        rec["_target"] = target
        rows.append(rec)
    return rows, skipped


def metrics(rows, key, idx_subset):
    """(cold H@50, cold H@10, cold N@10, warm N@10, overall N@10)."""
    cold = [r for r in rows if r["is_cold"]]
    warm = [r for r in rows if not r["is_cold"]]
    return {
        "cold_h50": float(np.mean([r[key][2] for r in cold])),
        "cold_h10": float(np.mean([r[key][0] for r in cold])),
        "cold_n10": float(np.mean([r[key][1] for r in cold])),
        "warm_n10": float(np.mean([r[key][1] for r in warm])),
        "all_n10": float(np.mean([r[key][1] for r in rows])),
        "cold_h50_events": float(np.sum([r[key][2] for r in cold])),
        "cold_h10_events": float(np.sum([r[key][0] for r in cold])),
    }


def random_subset_metrics(rows, coverage, seed, size):
    """Apply portfolio to a random target-free subset of users at given coverage."""
    rng = np.random.default_rng(seed)
    draw = rng.random(len(rows))
    key = "_rank_p2" if size == 2 else "_rank_p3"
    cold_h50, cold_h10, cold_n10, warm_n10, all_n10 = [], [], [], [], []
    for r, d in zip(rows, draw):
        rl = r[key] if d < coverage else r["_rank_v0"]
        h10, n10 = hit_ndcg(rl, r["_target"], 10)
        h50, _ = hit_ndcg(rl, r["_target"], 50)
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
        "cold_h10_events": float(np.sum(cold_h10)),
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    p0 = read_jsonl(args.p0_predictions)
    p6 = read_jsonl(args.p6_predictions)
    cold_items = read_set(args.cold_items)
    rows, skipped = build_rows(p0, p6, cold_items)

    n_cold = sum(1 for r in rows if r["is_cold"])
    n_warm = len(rows) - n_cold

    base = {k: metrics(rows, k, None) for k in
            ["v0_gram", "P6_learned", "portfolio@2_always", "portfolio@3_always"]}

    # --- Calibrate random coverage to match P6's warm cost (target-free) ---
    warm_v0 = base["v0_gram"]["warm_n10"]
    warm_p6 = base["P6_learned"]["warm_n10"]
    p6_warm_cost = warm_v0 - warm_p6          # absolute warm NDCG@10 given up
    p6_warm_retention = warm_p6 / warm_v0

    matched = {}
    for size in (2, 3):
        # For each coverage, average warm cost across seeds; pick coverage whose
        # warm cost is closest to P6's.  Selection uses WARM only (the constraint).
        best = None
        curve = []
        for cov in COVERAGE_GRID:
            per_seed = [random_subset_metrics(rows, cov, BOOTSTRAP_SEED + s, size)
                        for s in range(N_RANDOM_SEEDS)]
            mean_warm = float(np.mean([m["warm_n10"] for m in per_seed]))
            cost = warm_v0 - mean_warm
            curve.append({"coverage": cov, "warm_n10": mean_warm,
                          "warm_cost": cost,
                          "warm_retention": mean_warm / warm_v0})
            gap = abs(cost - p6_warm_cost)
            if best is None or gap < best["gap"]:
                best = {"coverage": cov, "gap": gap, "per_seed": per_seed,
                        "mean_warm": mean_warm}
        ps = best["per_seed"]
        matched[f"random_portfolio@{size}_matched"] = {
            "coverage": best["coverage"],
            "warm_cost_gap_vs_p6": best["gap"],
            "n_seeds": N_RANDOM_SEEDS,
            "mean": {k: float(np.mean([m[k] for m in ps])) for k in ps[0]},
            "std": {k: float(np.std([m[k] for m in ps])) for k in ps[0]},
            "min_cold_h50": float(np.min([m["cold_h50"] for m in ps])),
            "max_cold_h50": float(np.max([m["cold_h50"] for m in ps])),
            "coverage_curve": curve,
        }

    # --- Verdict: does warm-cost-matched RANDOM reach learned P6 on cold? ---
    verdicts = {}
    for size in (2, 3):
        key = f"random_portfolio@{size}_matched"
        m = matched[key]["mean"]
        s = matched[key]["std"]
        beats_h50 = m["cold_h50"] > base["P6_learned"]["cold_h50"]
        # How many of the random seeds individually beat P6?
        cov = matched[key]["coverage"]
        per_seed = [random_subset_metrics(rows, cov, BOOTSTRAP_SEED + i, size)
                    for i in range(N_RANDOM_SEEDS)]
        n_beat = sum(1 for x in per_seed
                     if x["cold_h50"] > base["P6_learned"]["cold_h50"])
        verdicts[key] = {
            "random_cold_h50": m["cold_h50"],
            "random_cold_h50_std": s["cold_h50"],
            "p6_cold_h50": base["P6_learned"]["cold_h50"],
            "random_beats_p6_on_cold_h50": bool(beats_h50),
            "n_seeds_beating_p6": n_beat,
            "n_seeds": N_RANDOM_SEEDS,
            "random_warm_retention": m["warm_n10"] / warm_v0,
            "p6_warm_retention": p6_warm_retention,
        }

    summary = {
        "experiment": "TIER0_A_WARM_COST_MATCHED_RANDOM_BASELINE",
        "domain": args.domain,
        "purpose": ("Test whether the 'simple beats learned' claim survives once "
                    "warm cost is equalised. Refutable by design."),
        "evaluation_only": True,
        "test_read": False,
        "n_users_evaluated": len(rows),
        "n_cold": n_cold,
        "n_warm": n_warm,
        "n_skipped_insufficient_candidates": skipped,
        "p6_warm_cost_absolute": p6_warm_cost,
        "p6_warm_retention": p6_warm_retention,
        "fixed_points": base,
        "matched_random": matched,
        "verdicts": verdicts,
        "seeds": {"bootstrap_seed": BOOTSTRAP_SEED,
                  "n_random_seeds": N_RANDOM_SEEDS},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- Console report ---
    print(f"\n=== Tier-0 A: warm-cost-matched random baseline [{args.domain}] ===")
    print(f"users={len(rows)}  cold={n_cold}  warm={n_warm}  skipped={skipped}\n")
    hdr = f"{'method':32s} {'coldH@50':>10s} {'ev':>5s} {'coldH@10':>10s} {'ev':>5s} {'warmN@10':>10s} {'warmRet':>8s} {'allN@10':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for k, m in base.items():
        print(f"{k:32s} {m['cold_h50']:10.6f} {m['cold_h50_events']:5.0f} "
              f"{m['cold_h10']:10.6f} {m['cold_h10_events']:5.0f} "
              f"{m['warm_n10']:10.6f} {m['warm_n10']/warm_v0:8.4f} {m['all_n10']:10.6f}")
    for key, d in matched.items():
        m = d["mean"]
        label = f"{key}(cov={d['coverage']:.2f})"
        print(f"{label:32s} {m['cold_h50']:10.6f} {m['cold_h50_events']:5.0f} "
              f"{m['cold_h10']:10.6f} {m['cold_h10_events']:5.0f} "
              f"{m['warm_n10']:10.6f} {m['warm_n10']/warm_v0:8.4f} {m['all_n10']:10.6f}")

    print(f"\n=== VERDICT (does random, at P6's warm cost, match learned P6?) ===")
    for k, v in verdicts.items():
        flag = "RANDOM WINS" if v["random_beats_p6_on_cold_h50"] else "P6 WINS"
        print(f"{k}:")
        print(f"   random cold H@50 = {v['random_cold_h50']:.6f} "
              f"(sd {v['random_cold_h50_std']:.6f}, warm ret {v['random_warm_retention']:.4f})")
        print(f"   P6     cold H@50 = {v['p6_cold_h50']:.6f} "
              f"(warm ret {v['p6_warm_retention']:.4f})")
        print(f"   -> {flag};  {v['n_seeds_beating_p6']}/{v['n_seeds']} random seeds beat P6")
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
