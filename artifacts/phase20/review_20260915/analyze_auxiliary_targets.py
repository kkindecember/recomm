"""Inspect frozen FIT targets, not trained students; NumPy only, no GPU."""
import json
from pathlib import Path

import numpy as np

from analyze_training_teacher import CONFIG, ROOT, digest, metrics, target_ranks


OUTPUT = Path(__file__).with_name("auxiliary_target_diagnostic.json")


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def kl(target, prediction):
    return (target * (np.log(target) - np.log(prediction))).sum(axis=1)


def main():
    config = json.loads(CONFIG.read_text())
    cache_path = ROOT / config["output"] / "cache/distillation_cache.npz"
    manifest = json.loads(cache_path.with_name("manifest.json").read_text())
    cache_hash = digest(cache_path)
    assert cache_hash == manifest["distillation_cache_sha256"]
    with np.load(cache_path, allow_pickle=False) as cache:
        fit = cache["fit"]
        positions = cache["positions"][fit]
        parent_scores = cache["parent_scores"][fit].astype(np.float64)
        teacher = cache["true_probability"][fit]
    covered = positions >= 0
    assert len(positions) == manifest["fit_rows"]
    assert int(covered.sum()) == manifest["covered_fit_rows"]
    temperature = config["loss"]["temperature"]
    z = (parent_scores - parent_scores.mean(1, keepdims=True)) / np.maximum(
        parent_scores.std(1, keepdims=True), 1e-6)
    parent = softmax(z / temperature)
    anchor_weight = config["loss"]["anchor_weight"]
    teacher_weight = config["loss"]["teacher_weight"] * len(positions) / covered.sum()
    row_weight = np.where(covered, teacher_weight, 0.)[:, None]
    mixture = (anchor_weight * parent + row_weight * teacher) / (anchor_weight + row_weight)
    assert np.allclose(mixture.sum(1), 1., rtol=0, atol=1e-12)

    # Verify the exact weighted-KL identity and gradient in logit space.
    # This says nothing about the CE term or the reachable model optimum.
    rows = np.flatnonzero(covered)[:256]
    p0, q, m = parent[rows], teacher[rows], mixture[rows]
    probe = softmax(z[rows] + np.random.default_rng(20260915).normal(size=p0.shape))
    a, b = anchor_weight, teacher_weight
    original = a * kl(p0, probe) + b * kl(q, probe)
    constant = a * kl(p0, m) + b * kl(q, m)
    identity_error = float(np.max(np.abs(original - ((a + b) * kl(m, probe) + constant))))
    gradient_error = float(np.max(np.abs(a * (probe - p0) + b * (probe - q) - (a + b) * (probe - m))))
    assert identity_error < 1e-12 and gradient_error < 1e-12

    parent_top = np.argsort(-parent, axis=1, kind="stable")[:, :10]
    teacher_top = np.argsort(-teacher, axis=1, kind="stable")[:, :10]
    up, down = [], []
    for p, q in zip(parent_top, teacher_top):
        up.append(len(set(q) - set(p)))
        down.append(len(set(p) - set(q)))
    up, down = np.array(up), np.array(down)
    assert np.array_equal(up, down) and np.all(up <= 10)
    assert np.all(50 - up - down >= 30)
    ranks = {"parent_teacher_forced_distribution": target_ranks(parent, positions),
             "external_teacher_distribution": target_ranks(teacher, positions),
             "hypothetical_auxiliary_target_NOT_student": target_ranks(mixture, positions)}
    pr, mr = ranks["parent_teacher_forced_distribution"], ranks["hypothetical_auxiliary_target_NOT_student"]
    result = {
        "kind": "in_sample_frozen_target_analysis_NOT_student_evaluation",
        "config": str(CONFIG.relative_to(ROOT)), "cache": str(cache_path.relative_to(ROOT)),
        "cache_sha256": cache_hash, "fit_prefixes": len(positions),
        "covered_fit_prefixes": int(covered.sum()), "temperature": temperature,
        "covered_anchor_weight": float(a), "covered_teacher_weight": float(b),
        "covered_teacher_fraction_in_two_KL_target": float(b / (a + b)),
        "identity_probe_rows": len(rows), "weighted_KL_identity_max_abs_error": identity_error,
        "logit_gradient_identity_max_abs_error": gradient_error,
        "hypothetical_target_metrics_all_fit": {name: metrics(r) for name, r in ranks.items()},
        "mixture_vs_parent_top10_gained": int(((mr <= 10) & (pr > 10)).sum()),
        "mixture_vs_parent_top10_lost": int(((mr > 10) & (pr <= 10)).sum()),
        "teacher_parent_top10_set_disagreement_fit_prefixes": int((up > 0).sum()),
        "mean_promoted_items_all_fit": float(up.mean()),
        "mean_promoted_items_covered_fit": float(up[covered].mean()),
        "mean_promoted_demoted_pairs_covered_fit": float((up[covered] * down[covered]).mean()),
        "minimum_unchanged_set_size_all_fit": int((50 - up - down).min()),
        "model_forward_calls": 0, "optimizer_steps": 0, "validation_read": False, "test_read": False,
        "limitations": [
            "Teacher fitted on FIT; target metrics are in-sample, not generalization evidence.",
            "The mixture identity concerns only the two KL terms on covered prefixes; excludes CE.",
            "Teacher mixture fraction is NOT a fraction of the total training gradient.",
            "Hypothetical target rankings are NOT trained student results or performance bounds.",
            "Full-list standardization and shared parameters constrain actual optimization.",
            "Uncovered rows have no teacher loss; set diagnostics on those rows are descriptive only.",
            "Top10 disagreement identifies requested changes, NOT proof the teacher is correct.",
            "Fewer ranking pairs does not imply fewer model forward passes with a 50-item anchor."
        ]
    }
    assert digest(cache_path) == cache_hash
    with OUTPUT.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
