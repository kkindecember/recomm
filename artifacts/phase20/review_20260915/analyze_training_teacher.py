"""Describe frozen FIT teacher signals using NumPy; never load a model or test."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiment/phase20/gacr_list_distill_toys_recovery_v3.json"
OUTPUT = Path(__file__).resolve().parent / "training_teacher_diagnostic.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_ranks(scores, positions):
    order = np.argsort(-scores, axis=1, kind="stable")
    inverse = np.argsort(order, axis=1, kind="stable") + 1
    ranks = np.full(len(positions), 51, dtype=np.int64)
    covered = positions >= 0
    rows = np.flatnonzero(covered)
    ranks[rows] = inverse[rows, positions[covered]]
    # Independent count formulation, preserving original candidate order on ties.
    gold_scores = scores[rows, positions[covered]]
    expected = 1 + (scores[covered] > gold_scores[:, None]).sum(1)
    expected += ((scores[covered] == gold_scores[:, None]) &
                 (np.arange(50)[None, :] < positions[covered, None])).sum(1)
    assert np.array_equal(ranks[covered], expected)
    return ranks


def metrics(ranks):
    result = {"prefixes": len(ranks)}
    for k in (5, 10, 20, 50):
        result[f"hit@{k}"] = float((ranks <= k).mean())
        result[f"ndcg@{k}"] = float(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0).mean())
    return result


def main():
    config = json.loads(CONFIG.read_text())
    cache_path = ROOT / config["output"] / "cache/distillation_cache.npz"
    manifest_path = cache_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache_hash = digest(cache_path)
    assert cache_hash == manifest["distillation_cache_sha256"]
    with np.load(cache_path, allow_pickle=False) as z:
        fit = z["fit"]
        pos = z["positions"][fit]
        parent = z["parent_scores"][fit].astype(np.float64)
        teacher = z["true_probability"][fit]
        shuffled = z["shuffled_probability"][fit]
    assert len(pos) == manifest["fit_rows"]
    assert int((pos >= 0).sum()) == manifest["covered_fit_rows"]
    assert np.isfinite(parent).all() and np.isfinite(teacher).all()
    assert np.allclose(teacher.sum(1), 1., atol=1e-12, rtol=0)
    assert np.array_equal(np.sort(teacher, axis=1), np.sort(shuffled, axis=1))
    native = np.where(pos >= 0, pos + 1, 51)
    ranks = {"C1_native_candidate_order": native,
             "C1_teacher_forced_scores": target_ranks(parent, pos),
             "B1_PCRF_teacher": target_ranks(teacher, pos),
             "shuffled_teacher": target_ranks(shuffled, pos)}
    covered = pos >= 0
    groups = {"all_fit_prefixes": np.ones(len(pos), dtype=bool),
              "covered_fit_prefixes": covered,
              "uncovered_fit_prefixes": ~covered}
    comparisons = {}
    for baseline in ("C1_native_candidate_order", "C1_teacher_forced_scores", "shuffled_teacher"):
        a, b = ranks["B1_PCRF_teacher"], ranks[baseline]
        row = {"covered_target_rank_improved": int(((a < b) & covered).sum()),
               "covered_target_rank_worsened": int(((a > b) & covered).sum()),
               "covered_target_rank_unchanged": int(((a == b) & covered).sum())}
        for k in (5, 10, 20, 50):
            row[f"hit{k}_gained"] = int(((a <= k) & (b > k)).sum())
            row[f"hit{k}_lost"] = int(((b <= k) & (a > k)).sum())
        comparisons[baseline] = row
    for r in ranks.values():
        assert np.array_equal(r <= 50, covered)
    z_parent = (parent - parent.mean(1, keepdims=True)) / np.maximum(parent.std(1, keepdims=True), 1e-6)
    log_parent = z_parent - np.logaddexp.reduce(z_parent, axis=1)[:, None]
    kl = (teacher * (np.log(teacher) - log_parent)).sum(1)
    assert np.all(kl >= -1e-12)
    rows = np.flatnonzero(covered)
    target_q = teacher[rows, pos[covered]]
    target_p = np.exp(log_parent[rows, pos[covered]])
    result = {
        "state": "ANALYZED", "kind": "descriptive_in_sample_training_cache_analysis",
        "config": str(CONFIG.relative_to(ROOT)), "cache": str(cache_path.relative_to(ROOT)),
        "cache_sha256": cache_hash,
        "fit_prefixes": len(pos), "fit_users": manifest["cohort"]["fit_users"],
        "teacher_supervised_prefixes": int(covered.sum()),
        "teacher_supervised_fraction": float(covered.mean()),
        "teacher_unsupervised_prefixes": int((~covered).sum()),
        "metrics": {g: {name: metrics(r[mask]) for name, r in ranks.items()} for g, mask in groups.items()},
        "teacher_vs_baseline": comparisons,
        "native_vs_teacher_forced_target_rank_changed_fit_prefixes": int((native != ranks["C1_teacher_forced_scores"]).sum()),
        "covered_mean_teacher_KL_to_parent_anchor_distribution": float(kl[covered].mean()),
        "covered_mean_target_teacher_probability": float(target_q.mean()),
        "covered_mean_target_parent_probability": float(target_p.mean()),
        "covered_teacher_target_probability_higher_count": int((target_q > target_p).sum()),
        "teacher_distribution_temperature": config["loss"]["temperature"],
        "validation_read": False, "test_read": False, "model_forward_calls": 0, "optimizer_steps": 0,
        "checks": ["input_sha256", "fit_counts", "probability_simplex", "shuffled_probability_multiset", "independent_target_rank_count", "fixed_candidate_coverage"],
        "limitations": ["In-sample teacher diagnostics: teacher fitted on this fit cohort; not generalization evidence.",
                       "Multiple prefixes per user are correlated; no independence-based significance claim.",
                       "Native candidate ordering and cached teacher-forcing scores are reported separately.",
                       "No student checkpoint or post-training prediction is read.",
                       "Probability mass and KL use normalized scores, not calibrated full-catalog relevance probabilities."]
    }
    assert digest(cache_path) == cache_hash
    with OUTPUT.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(OUTPUT), "fit_prefixes": len(pos),
                      "covered": int(covered.sum()), "metrics": result["metrics"]["all_fit_prefixes"],
                      "teacher_vs_C1": comparisons["C1_native_candidate_order"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
