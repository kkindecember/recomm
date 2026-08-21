"""Phase-13 Tier-0 Experiment A3: cross-domain replication of learned-vs-random
slate allocation at matched cost, using the trained CBSA allocator.

RESEARCH QUESTION (single):
    Experiment A2 showed that on Toys, ranking users by P6's learned utility beats
    random user selection at equal warm cost.  Does that replicate (a) on Beauty,
    and (b) with a DIFFERENT learned mechanism?

    The R2-v2 CBSA recovery run emitted a per-user `selected_action` in
    {a0, a2, a3} for both Toys and Beauty.  We hold that action MULTISET fixed and
    compare CBSA's actual assignment against random permutations of the same
    multiset.  Identical number of each action, identical insertion rule - the
    only difference is WHICH user receives WHICH action.

    This is the cleanest possible matched-cost test: cost is held fixed by
    construction, not by calibration.

DESIGN NOTES:
  - Evaluation-only.  Reuses the frozen CBSA recovery predictions and the frozen
    P0 predictions.  No training, no test split.
  - Action outcomes are recomputed from P0 + catalog cold state using the frozen
    B1 candidate filter and anchor rule, so all three actions are available per
    user counterfactually.
  - Recomputed a2 outcomes are cross-checked against the CBSA run's stored
    portfolio2 outcomes; any mismatch aborts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ANCHOR_PREFIX = 7
SEED = 20260819
N_PERMUTATIONS = 200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--p0-predictions", required=True)
    p.add_argument("--cbsa-predictions", required=True)
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
    gram = unique_in_order(gram)
    resolver = unique_in_order(resolver)
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


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    p0 = read_jsonl(args.p0_predictions)
    cbsa = [r for r in read_jsonl(args.cbsa_predictions)
            if str(r["domain"]) == args.domain]
    cbsa_by_uid = {str(r["user_id"]): r for r in cbsa}

    rows, mismatches, skipped = [], 0, 0
    for src in p0:
        uid = str(src["user_id"])
        if uid not in cbsa_by_uid:
            continue
        c = cbsa_by_uid[uid]
        target = str(src["target"])
        gram = unique_in_order(src["v0_top50"])
        resolver = unique_in_order(src["resolver_top50"])
        protected = set(gram[:ANCHOR_PREFIX])
        cands = [it for it in resolver
                 if it in cold_items_global and it not in protected][:3]
        r2 = portfolio_ranking(gram, resolver, cands, 2) if len(cands) >= 2 else None
        r3 = portfolio_ranking(gram, resolver, cands, 3) if len(cands) >= 3 else None
        if r2 is None or r3 is None:
            skipped += 1
            continue

        outcomes = {}
        for name, rl in (("a0", gram), ("a2", r2), ("a3", r3)):
            h10, n10 = hit_ndcg(rl, target, 10)
            h50, _ = hit_ndcg(rl, target, 50)
            outcomes[name] = (h10, n10, h50)

        # integrity: recomputed a2 must match the CBSA run's stored portfolio2
        if abs(outcomes["a2"][2] - float(c["portfolio2_hit50"])) > 1e-9:
            mismatches += 1
        rows.append({
            "is_cold": bool(src["is_cold"]),
            "action": str(c["effective_action"]),
            "out": outcomes,
        })

    if mismatches:
        raise SystemExit(f"ABORT: {mismatches} portfolio@2 outcome mismatches "
                         f"vs frozen CBSA predictions")

    n = len(rows)
    actions = [r["action"] for r in rows]
    mix = {a: actions.count(a) for a in ("a0", "a2", "a3")}

    def evaluate(assign):
        cold_h50, cold_h10, warm_n10, all_n10 = [], [], [], []
        for r, a in zip(rows, assign):
            h10, n10, h50 = r["out"][a]
            all_n10.append(n10)
            if r["is_cold"]:
                cold_h50.append(h50); cold_h10.append(h10)
            else:
                warm_n10.append(n10)
        return {
            "cold_h50": float(np.mean(cold_h50)),
            "cold_h10": float(np.mean(cold_h10)),
            "cold_h50_events": float(np.sum(cold_h50)),
            "warm_n10": float(np.mean(warm_n10)),
            "all_n10": float(np.mean(all_n10)),
        }

    learned = evaluate(actions)
    baseline_v0 = evaluate(["a0"] * n)

    rng = np.random.default_rng(SEED)
    perm_runs = []
    for _ in range(N_PERMUTATIONS):
        shuffled = list(actions)
        rng.shuffle(shuffled)
        perm_runs.append(evaluate(shuffled))

    rand_mean = {k: float(np.mean([x[k] for x in perm_runs])) for k in perm_runs[0]}
    rand_sd = {k: float(np.std([x[k] for x in perm_runs])) for k in perm_runs[0]}
    n_beat = sum(1 for x in perm_runs if x["cold_h50"] >= learned["cold_h50"])
    p_emp = (n_beat + 1) / (N_PERMUTATIONS + 1)

    summary = {
        "experiment": "TIER0_A3_CBSA_MATCHED_ACTION_MULTISET_PERMUTATION",
        "domain": args.domain,
        "evaluation_only": True,
        "test_read": False,
        "n_users": n,
        "n_cold": sum(1 for r in rows if r["is_cold"]),
        "n_skipped": skipped,
        "action_mix": mix,
        "v0_reference": baseline_v0,
        "learned_cbsa": learned,
        "random_permutation_mean": rand_mean,
        "random_permutation_sd": rand_sd,
        "n_permutations": N_PERMUTATIONS,
        "n_permutations_ge_learned_cold_h50": n_beat,
        "empirical_p_value_cold_h50": p_emp,
        "learned_beats_random": bool(p_emp < 0.05),
        "note": ("Action multiset is held fixed by construction, so warm cost is "
                 "matched exactly; only the user-to-action mapping differs."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Tier-0 A3: CBSA vs random permutation of same action multiset "
          f"[{args.domain}] ===")
    print(f"users={n}  cold={summary['n_cold']}  skipped={skipped}")
    print(f"action mix (held fixed): {mix}\n")
    hdr = f"{'arm':28s} {'coldH@50':>10s} {'ev':>6s} {'coldH@10':>10s} {'warmN@10':>10s} {'allN@10':>10s}"
    print(hdr); print("-" * len(hdr))
    print(f"{'v0 (no intervention)':28s} {baseline_v0['cold_h50']:10.6f} "
          f"{baseline_v0['cold_h50_events']:6.0f} {baseline_v0['cold_h10']:10.6f} "
          f"{baseline_v0['warm_n10']:10.6f} {baseline_v0['all_n10']:10.6f}")
    print(f"{'CBSA (learned mapping)':28s} {learned['cold_h50']:10.6f} "
          f"{learned['cold_h50_events']:6.0f} {learned['cold_h10']:10.6f} "
          f"{learned['warm_n10']:10.6f} {learned['all_n10']:10.6f}")
    print(f"{'random same-mix (mean)':28s} {rand_mean['cold_h50']:10.6f} "
          f"{rand_mean['cold_h50_events']:6.0f} {rand_mean['cold_h10']:10.6f} "
          f"{rand_mean['warm_n10']:10.6f} {rand_mean['all_n10']:10.6f}")
    print(f"{'  (sd across perms)':28s} {rand_sd['cold_h50']:10.6f} "
          f"{'':6s} {rand_sd['cold_h10']:10.6f} {rand_sd['warm_n10']:10.6f} "
          f"{rand_sd['all_n10']:10.6f}")
    print(f"\ncold H@50: learned={learned['cold_h50']:.6f} vs "
          f"random={rand_mean['cold_h50']:.6f} (sd {rand_sd['cold_h50']:.6f})")
    print(f"permutations >= learned: {n_beat}/{N_PERMUTATIONS}  "
          f"empirical p = {p_emp:.4f}  -> "
          f"{'LEARNED WINS' if p_emp < 0.05 else 'NO SIGNIFICANT DIFFERENCE'}")
    print(f"wrote {out/'summary.json'}")


if __name__ == "__main__":
    _a = parse_args()
    cold_items_global = read_set(_a.cold_items)
    main()
