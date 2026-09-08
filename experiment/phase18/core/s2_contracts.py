"""Deterministic, CPU-only contracts for the bounded S18-2 probe."""
from __future__ import annotations

import hashlib
import math
from typing import Any

ARMS = ("C0_CONT", "A0_LEGAL_GENERIC", "M0_PCPS", "S0_SHUFFLED_CF")


def training_view(materialized_i0_items):
    """I0 view is visible + I0 target + dummy guard; fit only visible."""
    visible = tuple(materialized_i0_items[:-2])
    if len(visible) < 2:
        raise ValueError("need at least two visible items")
    return visible[:-1], visible[-1]


def shuffled_teacher_map(users, domain, seed=2023):
    """One deterministic cycle: bijective, no fixed points, no outcome inputs."""
    ordered = sorted(users, key=lambda u: (hashlib.sha256(
        f"S18-2-shuffle|{seed}|{domain}|{u}".encode()).hexdigest(), u))
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("shuffle requires unique users and at least two rows")
    return {u: ordered[(i + 1) % len(ordered)] for i, u in enumerate(ordered)}


def choose_alpha(ratios_by_domain, candidates=(0.1, 0.3), lower=0.10, upper=0.30):
    if not ratios_by_domain or any(not math.isfinite(r) or r <= 0 for r in ratios_by_domain.values()):
        raise ValueError("nonfinite or zero calibration ratio")
    admissible = [a for a in candidates if all(lower <= a * r <= upper for r in ratios_by_domain.values())]
    if not admissible:
        return None
    return min(admissible, key=lambda a: (
        abs(sum(a * r for r in ratios_by_domain.values()) / len(ratios_by_domain) - 0.20), a))


def prefix_survival(active, target):
    """Per-user fraction of non-EOS target prefixes surviving active beam50."""
    depths = range(1, len(target))
    values = [tuple(target[:d]) in active.get(d, set()) for d in depths]
    if not values:
        raise ValueError("lexical path has no pre-EOS token")
    return sum(values) / len(values)


def mechanism_gate(arms: dict[str, dict[str, float]], gates: dict[str, Any]):
    c, a, m, s = (arms[k] for k in ARMS)
    deltas = {f"M0_minus_{name}": {k: m[k] - other[k] for k in
              ("prefix_survival", "path_margin", "Hit@50", "NDCG@10")}
              for name, other in (("C0", c), ("A0", a), ("S0", s))}
    d = deltas["M0_minus_C0"]
    checks = {
        "survival_vs_c0": d["prefix_survival"] >= gates["survival_vs_c0_min"],
        "path_margin_vs_c0": d["path_margin"] > 0,
        "hit50_vs_c0": d["Hit@50"] >= gates["hit50_vs_c0_min"],
        "survival_vs_s0": deltas["M0_minus_S0"]["prefix_survival"] >= gates["survival_vs_s0_min"],
        "ndcg10_noninferiority": d["NDCG@10"] >= -gates["ndcg10_max_drop"],
        "cf_additive": m["prefix_survival"] > a["prefix_survival"] and m["Hit@50"] > a["Hit@50"],
    }
    decision = ("MECHANISM_PASS" if all(checks.values()) else
                "CF_GUIDANCE_NOT_ADDITIVE" if not checks["cf_additive"] else "FAILED_SCIENTIFIC_GATE")
    return {"decision": decision, "checks": checks, "deltas": deltas}
