"""Phase-13 Tier-1 Experiment A: item-level generative x retrieval score fusion.

RESEARCH QUESTION (single):
    The portfolio family can only touch ranks 8-10, and its ceiling is provably
    low (Tier-0 B: inserting 10 candidates reaches at most ~7% of cold users).
    P0's earlier fusion attempt failed, but its own root-cause analysis blamed
    the depth-3 ROUTE restriction, not fusion itself:

        "resolver 方向获得强支持，当前 depth3 route 接口被否定"

    A clean item-level fusion -- no routes, no prefix constraints, just combining
    the generative ranking and the retrieval ranking over the full candidate
    union -- has never actually been tested.  This script tests it.

METHOD:
    For each user, take the union of v0_top50 and resolver_top50 and score every
    item by a rank-based combination:

        RRF:      s(i) = w / (K + rank_gram(i))  +  (1-w) / (K + rank_res(i))
        Borda:    s(i) = w * (N - rank_gram(i))  +  (1-w) * (N - rank_res(i))

    Items absent from one list receive that list's worst rank (a deterministic,
    target-free convention).  w is swept from 0 (pure retrieval) to 1 (pure
    generative), producing the full Pareto front on identical axes to the
    portfolio family, so the two are directly comparable at matched warm cost.

    A cold-aware variant is also swept: apply fusion only when the resolver
    surfaces cold candidates, otherwise keep v0 untouched.  This isolates whether
    fusion's benefit comes from cold reachability or from general reranking.

DESIGN NOTES:
  - Evaluation-only.  No training, no test split, no GPU.
  - w is swept, not fitted: the whole front is reported and no single operating
    point is selected as a result.  Choosing w would require a preregistered
    protocol on unseen data.
  - Portfolio@2/@3 are recomputed with the frozen B1 rule as reference points.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ANCHOR_PREFIX = 7
RRF_K = 60
WEIGHT_GRID = [round(0.05 * i, 2) for i in range(21)]


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


def portfolio_ranking(gram, resolver, candidates, size):
    """Frozen B1 anchor rule."""
    pf = unique_in_order(candidates)[:size]
    if len(pf) != size:
        return None
    anchor = 10 - size
    return unique_in_order([*gram[:anchor], *pf, *gram[anchor:], *resolver])


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


def fuse(gram, resolver, w, scheme, out_len=50):
    """Rank-based fusion over the union. Target-free and deterministic."""
    n_g, n_r = len(gram), len(resolver)
    rg = {it: i + 1 for i, it in enumerate(gram)}
    rr = {it: i + 1 for i, it in enumerate(resolver)}
    union = unique_in_order([*gram, *resolver])
    miss_g, miss_r = n_g + 1, n_r + 1

    scored = []
    for it in union:
        a, b = rg.get(it, miss_g), rr.get(it, miss_r)
        if scheme == "rrf":
            s = w / (RRF_K + a) + (1.0 - w) / (RRF_K + b)
        else:  # borda
            s = w * (n_g + 1 - a) + (1.0 - w) * (n_r + 1 - b)
        scored.append((s, it))
    # Deterministic tie-break: higher score, then original union order.
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))
    return [scored[i][1] for i in order][:out_len]


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.p0_predictions)
    cold_items = read_set(args.cold_items)

    rows = []
    for src in records:
        gram = unique_in_order(src["v0_top50"])
        resolver = unique_in_order(src["resolver_top50"])
        protected = set(gram[:ANCHOR_PREFIX])
        cands = [it for it in resolver
                 if it in cold_items and it not in protected][:3]
        rows.append({
            "target": str(src["target"]),
            "is_cold": bool(src["is_cold"]),
            "gram": gram,
            "resolver": resolver,
            "cands": cands,
            "has_cold_cand": len(cands) > 0,
        })

    n_cold = sum(1 for r in rows if r["is_cold"])

    def evaluate(rankings):
        cold_h50, cold_h10, cold_n10, warm_n10, all_n10 = [], [], [], [], []
        for r, rl in zip(rows, rankings):
            h10, n10 = hit_ndcg(rl, r["target"], 10)
            h50, _ = hit_ndcg(rl, r["target"], 50)
            all_n10.append(n10)
            if r["is_cold"]:
                cold_h50.append(h50); cold_h10.append(h10); cold_n10.append(n10)
            else:
                warm_n10.append(n10)
        return {
            "cold_h50": float(np.mean(cold_h50)),
            "cold_h50_events": float(np.sum(cold_h50)),
            "cold_h10": float(np.mean(cold_h10)),
            "cold_n10": float(np.mean(cold_n10)),
            "warm_n10": float(np.mean(warm_n10)),
            "all_n10": float(np.mean(all_n10)),
        }

    # ---- reference points ----
    ref = {"v0_gram": evaluate([r["gram"] for r in rows]),
           "resolver_only": evaluate([r["resolver"] for r in rows])}
    for size in (2, 3):
        rk = []
        for r in rows:
            p = portfolio_ranking(r["gram"], r["resolver"], r["cands"], size)
            rk.append(p if p is not None else r["gram"])
        ref[f"portfolio@{size}"] = evaluate(rk)

    warm_v0 = ref["v0_gram"]["warm_n10"]

    # ---- fusion sweeps ----
    sweeps = {}
    for scheme in ("rrf", "borda"):
        for gated in (False, True):
            key = f"{scheme}{'_coldgated' if gated else ''}"
            pts = []
            for w in WEIGHT_GRID:
                rk = []
                for r in rows:
                    if gated and not r["has_cold_cand"]:
                        rk.append(r["gram"])
                    else:
                        rk.append(fuse(r["gram"], r["resolver"], w, scheme))
                m = evaluate(rk)
                m["w_gram"] = w
                m["warm_retention"] = m["warm_n10"] / warm_v0
                pts.append(m)
            sweeps[key] = pts

    # ---- headline comparison: best fusion at >= portfolio@2's warm retention ----
    p2_ret = ref["portfolio@2"]["warm_n10"] / warm_v0
    comparison = {}
    for key, pts in sweeps.items():
        feasible = [p for p in pts if p["warm_retention"] >= p2_ret]
        if feasible:
            best = max(feasible, key=lambda p: p["cold_h50"])
            comparison[key] = {
                "constraint": f"warm_retention >= {p2_ret:.4f} (portfolio@2)",
                "best_w_gram": best["w_gram"],
                "cold_h50": best["cold_h50"],
                "cold_h50_events": best["cold_h50_events"],
                "warm_retention": best["warm_retention"],
                "all_n10": best["all_n10"],
                "vs_portfolio2_cold_h50": best["cold_h50"] - ref["portfolio@2"]["cold_h50"],
                "beats_portfolio2": bool(best["cold_h50"] > ref["portfolio@2"]["cold_h50"]),
            }
        else:
            comparison[key] = {"constraint": f"warm_retention >= {p2_ret:.4f}",
                               "feasible_points": 0, "beats_portfolio2": False}

    summary = {
        "experiment": "TIER1_A_ITEM_LEVEL_SCORE_FUSION_SWEEP",
        "domain": args.domain,
        "evaluation_only": True,
        "test_read": False,
        "n_users": len(rows),
        "n_cold": n_cold,
        "rrf_k": RRF_K,
        "reference_points": ref,
        "sweeps": sweeps,
        "matched_warm_comparison": comparison,
        "note": ("w is swept and fully reported, not fitted. No operating point "
                 "is selected as a result; that needs preregistration on unseen data."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Tier-1 A: item-level fusion sweep [{args.domain}] ===")
    print(f"users={len(rows)}  cold={n_cold}\n")
    hdr = f"{'reference':20s} {'coldH@50':>10s} {'ev':>5s} {'coldH@10':>10s} {'warmRet':>9s} {'allN@10':>10s}"
    print(hdr); print("-" * len(hdr))
    for k, m in ref.items():
        print(f"{k:20s} {m['cold_h50']:10.6f} {m['cold_h50_events']:5.0f} "
              f"{m['cold_h10']:10.6f} {m['warm_n10']/warm_v0:9.4f} {m['all_n10']:10.6f}")

    for key, pts in sweeps.items():
        print(f"\n--- sweep: {key}  (w=1 -> pure GRAM, w=0 -> pure resolver) ---")
        h = f"{'w_gram':>7s} {'coldH@50':>10s} {'ev':>5s} {'coldH@10':>10s} {'warmRet':>9s} {'allN@10':>10s}"
        print(h); print("-" * len(h))
        for p in pts:
            if p["w_gram"] % 0.1 < 1e-9 or p["warm_retention"] >= p2_ret:
                print(f"{p['w_gram']:7.2f} {p['cold_h50']:10.6f} "
                      f"{p['cold_h50_events']:5.0f} {p['cold_h10']:10.6f} "
                      f"{p['warm_retention']:9.4f} {p['all_n10']:10.6f}")

    print(f"\n=== Best fusion at warm retention >= portfolio@2's ({p2_ret:.4f}) ===")
    print(f"portfolio@2 reference: cold H@50 = {ref['portfolio@2']['cold_h50']:.6f} "
          f"({ref['portfolio@2']['cold_h50_events']:.0f} events), "
          f"allN@10 = {ref['portfolio@2']['all_n10']:.6f}")
    for k, c in comparison.items():
        if c.get("feasible_points") == 0:
            print(f"  {k:18s} no feasible point at that warm retention")
            continue
        flag = "BEATS portfolio@2" if c["beats_portfolio2"] else "loses"
        print(f"  {k:18s} w={c['best_w_gram']:.2f}  cold H@50={c['cold_h50']:.6f} "
              f"({c['cold_h50_events']:.0f} ev)  warmRet={c['warm_retention']:.4f}  "
              f"allN@10={c['all_n10']:.6f}  -> {flag} "
              f"({c['vs_portfolio2_cold_h50']:+.6f})")
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
