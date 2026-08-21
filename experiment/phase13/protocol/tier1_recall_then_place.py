"""Phase-13 Tier-1 Experiment B: hybrid recall-then-place (RTP).

MOTIVATION (from Tier-1 A):
    Item-level fusion and the portfolio family fail in complementary ways.

      fusion      : cold H@50 2.2-2.3x portfolio, but cold H@10 UNCHANGED from v0
                    and overall NDCG@10 BELOW v0 -- it moves correct cold items
                    into ranks 11-50 and disturbs warm order, gaining nothing
                    at the cut-off that the headline metric measures.
      portfolio@N : surgical -- cold H@10 4.8x v0, overall NDCG@10 above v0 --
                    but its ceiling is tiny because it only ever considers the
                    resolver's own top-3 cold items.

    The two failures suggest the missing piece is not a better allocator and not
    a better fusion, but SEPARATING the two decisions:

        recall  -- which cold items deserve consideration  (fusion is good at this)
        place   -- where they go in the slate              (anchor rule is good at this)

METHOD (RTP):
    1. RECALL: fuse gram and resolver rankings (RRF, weight w) to produce a
       re-ranked cold-candidate ordering.  This draws on the FULL union, not just
       the resolver's top-3, so it can surface cold items either list ranks poorly.
    2. PLACE: keep v0's top `anchor` positions untouched, then insert the top `n`
       fused cold candidates into the remaining slots of the top-10, then append
       the rest of v0 followed by leftovers.

    Setting w=0 recovers portfolio@n exactly (pure resolver order), so the
    portfolio family is a strict special case of RTP and the comparison is nested.

DESIGN NOTES:
  - Evaluation-only.  No training, no GPU, no test split.
  - (w, anchor, n) is swept and the full grid reported.  Nothing is selected as a
    result; picking an operating point requires preregistration on unseen data.
  - Paired bootstrap CIs are reported against BOTH v0 and portfolio@2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RRF_K = 60
BOOTSTRAP = 10000
BOOT_CHUNK = 500          # bootstrap in chunks to bound peak memory
SEED = 20260819
W_GRID = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
ANCHOR_GRID = [7, 8]
N_GRID = [2, 3]


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


def hit_ndcg(ranking, target, k):
    for i, it in enumerate(ranking[:k]):
        if it == target:
            return 1.0, 1.0 / math.log2(i + 2)
    return 0.0, 0.0


def fused_cold_order(gram, resolver, cold_items, protected, w):
    """RECALL stage: RRF over the union, restricted to eligible cold items."""
    n_g, n_r = len(gram), len(resolver)
    rg = {it: i + 1 for i, it in enumerate(gram)}
    rr = {it: i + 1 for i, it in enumerate(resolver)}
    union = unique_in_order([*gram, *resolver])
    scored = []
    for idx, it in enumerate(union):
        if it not in cold_items or it in protected:
            continue
        a, b = rg.get(it, n_g + 1), rr.get(it, n_r + 1)
        s = w / (RRF_K + a) + (1.0 - w) / (RRF_K + b)
        scored.append((s, idx, it))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[2] for t in scored]


def rtp_ranking(gram, resolver, cold_order, anchor, n):
    """PLACE stage: frozen anchor rule, insert n fused cold candidates."""
    pf = cold_order[:n]
    if len(pf) < n:
        return None
    return unique_in_order([*gram[:anchor], *pf, *gram[anchor:], *resolver])


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
        rows.append({
            "target": str(src["target"]),
            "is_cold": bool(src["is_cold"]),
            "gram": gram,
            "resolver": resolver,
        })
    n_cold = sum(1 for r in rows if r["is_cold"])

    # Precompute fused cold orders per (w, anchor).
    cold_orders = {}
    for w in W_GRID:
        for anchor in ANCHOR_GRID:
            cold_orders[(w, anchor)] = [
                fused_cold_order(r["gram"], r["resolver"], cold_items,
                                 set(r["gram"][:anchor]), w)
                for r in rows
            ]

    def per_user(rankings):
        """Per-user (cold_h50, cold_h10, n10) so bootstrap can pair on users."""
        res = []
        for r, rl in zip(rows, rankings):
            h10, n10 = hit_ndcg(rl, r["target"], 10)
            h50, _ = hit_ndcg(rl, r["target"], 50)
            res.append((h50, h10, n10))
        return res

    def agg(pu):
        cold = [pu[i] for i, r in enumerate(rows) if r["is_cold"]]
        warm = [pu[i] for i, r in enumerate(rows) if not r["is_cold"]]
        return {
            "cold_h50": float(np.mean([x[0] for x in cold])),
            "cold_h50_events": float(np.sum([x[0] for x in cold])),
            "cold_h10": float(np.mean([x[1] for x in cold])),
            "cold_h10_events": float(np.sum([x[1] for x in cold])),
            "cold_n10": float(np.mean([x[2] for x in cold])),
            "warm_n10": float(np.mean([x[2] for x in warm])),
            "all_n10": float(np.mean([x[2] for x in pu])),
        }

    pu_v0 = per_user([r["gram"] for r in rows])
    ref = {"v0_gram": agg(pu_v0)}
    warm_v0 = ref["v0_gram"]["warm_n10"]

    # portfolio@N == RTP with w=0, anchor=10-N
    pu_ref = {"v0_gram": pu_v0}
    for size in (2, 3):
        anchor = 10 - size
        rk = []
        for i, r in enumerate(rows):
            p = rtp_ranking(r["gram"], r["resolver"],
                            cold_orders[(0.0, anchor)][i], anchor, size)
            rk.append(p if p is not None else r["gram"])
        pu = per_user(rk)
        pu_ref[f"portfolio@{size}"] = pu
        ref[f"portfolio@{size}"] = agg(pu)

    # ---- RTP grid ----
    grid = []
    pu_cache = {}
    for w in W_GRID:
        for anchor in ANCHOR_GRID:
            for n in N_GRID:
                if anchor + n > 10:
                    continue
                rk = []
                for i, r in enumerate(rows):
                    p = rtp_ranking(r["gram"], r["resolver"],
                                    cold_orders[(w, anchor)][i], anchor, n)
                    rk.append(p if p is not None else r["gram"])
                pu = per_user(rk)
                m = agg(pu)
                m.update({"w_gram": w, "anchor": anchor, "n_insert": n,
                          "warm_retention": m["warm_n10"] / warm_v0})
                key = (w, anchor, n)
                pu_cache[key] = pu
                grid.append(m)

    # ---- paired bootstrap for the best configs ----
    idx_cold = np.array([i for i, r in enumerate(rows) if r["is_cold"]])
    n_all, n_c = len(rows), len(idx_cold)

    def ci(a_pu, b_pu, which):
        """Chunked paired bootstrap: bounded memory, same seed => reproducible."""
        if which == "all_n10":
            d = np.array([a[2] - b[2] for a, b in zip(a_pu, b_pu)])
            m = n_all
        else:
            j = 0 if which == "cold_h50" else 1
            d = np.array([a_pu[i][j] - b_pu[i][j] for i in idx_cold])
            m = n_c
        rng_local = np.random.default_rng(SEED)
        means = np.empty(BOOTSTRAP)
        done = 0
        while done < BOOTSTRAP:
            b = min(BOOT_CHUNK, BOOTSTRAP - done)
            means[done:done + b] = d[rng_local.integers(0, m, size=(b, m))].mean(axis=1)
            done += b
        return {"diff": float(d.mean()),
                "lo": float(np.percentile(means, 2.5)),
                "hi": float(np.percentile(means, 97.5))}

    p2_ret = ref["portfolio@2"]["warm_n10"] / warm_v0
    feasible = [m for m in grid if m["warm_retention"] >= p2_ret]
    best_cold = max(feasible, key=lambda m: m["cold_h50"]) if feasible else None
    best_all = max(feasible, key=lambda m: m["all_n10"]) if feasible else None

    highlights = {}
    for label, m in (("best_cold_h50_at_matched_warm", best_cold),
                     ("best_overall_at_matched_warm", best_all)):
        if m is None:
            continue
        key = (m["w_gram"], m["anchor"], m["n_insert"])
        pu = pu_cache[key]
        highlights[label] = {
            "config": {"w_gram": m["w_gram"], "anchor": m["anchor"],
                       "n_insert": m["n_insert"]},
            "metrics": m,
            "vs_v0": {k: ci(pu, pu_ref["v0_gram"], k)
                      for k in ("cold_h50", "cold_h10", "all_n10")},
            "vs_portfolio2": {k: ci(pu, pu_ref["portfolio@2"], k)
                              for k in ("cold_h50", "cold_h10", "all_n10")},
        }

    summary = {
        "experiment": "TIER1_B_HYBRID_RECALL_THEN_PLACE",
        "domain": args.domain,
        "evaluation_only": True,
        "test_read": False,
        "n_users": len(rows), "n_cold": n_cold,
        "rrf_k": RRF_K, "bootstrap": BOOTSTRAP, "seed": SEED,
        "reference_points": ref,
        "grid": grid,
        "matched_warm_constraint": p2_ret,
        "highlights": highlights,
        "note": ("w=0 reproduces portfolio@n exactly, so portfolio is a nested "
                 "special case. Grid is swept and fully reported, not fitted."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Tier-1 B: hybrid recall-then-place [{args.domain}] ===")
    print(f"users={len(rows)}  cold={n_cold}\n")
    h = f"{'reference':16s} {'coldH@50':>10s} {'ev':>5s} {'coldH@10':>10s} {'ev':>5s} {'warmRet':>9s} {'allN@10':>10s}"
    print(h); print("-" * len(h))
    for k, m in ref.items():
        print(f"{k:16s} {m['cold_h50']:10.6f} {m['cold_h50_events']:5.0f} "
              f"{m['cold_h10']:10.6f} {m['cold_h10_events']:5.0f} "
              f"{m['warm_n10']/warm_v0:9.4f} {m['all_n10']:10.6f}")

    print(f"\n--- RTP grid, warm retention >= portfolio@2 ({p2_ret:.4f}), "
          f"top 12 by cold H@50 ---")
    h2 = f"{'w':>5s} {'anch':>5s} {'n':>3s} {'coldH@50':>10s} {'ev':>5s} {'coldH@10':>10s} {'ev':>5s} {'warmRet':>9s} {'allN@10':>10s}"
    print(h2); print("-" * len(h2))
    for m in sorted(feasible, key=lambda x: -x["cold_h50"])[:12]:
        print(f"{m['w_gram']:5.2f} {m['anchor']:5d} {m['n_insert']:3d} "
              f"{m['cold_h50']:10.6f} {m['cold_h50_events']:5.0f} "
              f"{m['cold_h10']:10.6f} {m['cold_h10_events']:5.0f} "
              f"{m['warm_retention']:9.4f} {m['all_n10']:10.6f}")

    for label, d in highlights.items():
        c = d["config"]
        print(f"\n=== {label}: w={c['w_gram']}, anchor={c['anchor']}, n={c['n_insert']} ===")
        for ref_name in ("vs_v0", "vs_portfolio2"):
            print(f"  {ref_name}:")
            for k, v in d[ref_name].items():
                verd = "PASS" if v["lo"] > 0 else ("FAIL" if v["hi"] < 0 else "INCONCL")
                print(f"    {k:10s} {v['diff']:+.6f}  [{v['lo']:+.6f},{v['hi']:+.6f}]  {verd}")
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
