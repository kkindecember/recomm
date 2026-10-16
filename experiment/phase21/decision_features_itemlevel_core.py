"""Seven observable features: expert item judgments plus initial GRAM decisions.

Only three coordinates change from decision7. No target, fused score, shortlist,
or newly selected temperature enters this feature interface.
"""
import numpy as np
from experiment.phase21 import decision_features_core as BASE

NAMES = ['log_history_length', 'entropy_gram_root', 'entropy_behavior_item',
         'entropy_text_item', 'js_behavior_text_item', 'js_behavior_gram_root',
         'js_text_gram_root']


def distribution(scores):
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.ndim == 1 and scores.size > 1 and np.isfinite(scores).all()
    mass = np.exp(20. * (scores - scores.max()))
    return mass / mass.sum()


def entropy(p):
    return float(-np.sum(p * np.log(np.maximum(p, 1e-300))))


def js(a, b):
    middle = (a + b) / 2
    # Direct KL definition, independently of the old entropy identity.
    def kl(p):
        return np.sum(p * (np.log(np.maximum(p, 1e-300)) -
                           np.log(np.maximum(middle, 1e-300))))
    return float(np.clip((kl(a) + kl(b)) / (2 * np.log(2.)), 0, 1))


def features(history_length, gram_root, behavior_scores, text_scores, groups):
    behavior, text = distribution(behavior_scores), distribution(text_scores)
    groups = np.asarray(groups)
    assert behavior.shape == text.shape == groups.shape
    branches = len(gram_root)
    assert groups.min() >= 0 and groups.max() < branches
    br = np.bincount(groups, weights=behavior, minlength=branches)
    tr = np.bincount(groups, weights=text, minlength=branches)
    root = BASE.features(history_length, gram_root, br, tr)
    result = root.copy()
    result[2] = entropy(behavior) / np.log(len(behavior))
    result[3] = entropy(text) / np.log(len(text))
    result[4] = js(behavior, text)
    assert np.isfinite(result).all() and ((result[1:] >= -1e-7) & (result[1:] <= 1+1e-7)).all()
    return result, root


def tests():
    checks = BASE.tests()
    # The same coarse branch masses can conceal disagreement between items.
    groups = np.array([0, 0, 1, 1])
    a, b = np.array([.45, .05, .45, .05]), np.array([.05, .45, .05, .45])
    scores_a, scores_b = np.log(a)/20, np.log(b)/20
    new, old = features(3, np.array([.5, .5]), scores_a, scores_b, groups)
    assert abs(old[4]) < 1e-7 and new[4] > .5
    np.testing.assert_array_equal(new[[0, 1, 5, 6]], old[[0, 1, 5, 6]])
    np.testing.assert_allclose(new[2:4], entropy(a)/np.log(4), atol=1e-7)
    swapped, _ = features(3, np.array([.5, .5]), scores_b, scores_a, groups)
    np.testing.assert_array_equal(swapped, new[[0, 1, 3, 2, 4, 6, 5]])
    perm = [2, 0, 3, 1]
    shuffled, _ = features(3, np.array([.5, .5]), scores_a[perm], scores_b[perm], groups[perm])
    np.testing.assert_allclose(shuffled, new, atol=1e-7)
    shifted, _ = features(3, np.array([.5, .5]), scores_a+7, scores_b-3, groups)
    np.testing.assert_allclose(shifted, new, atol=1e-7)
    uniform, _ = features(3, np.array([.5, .5]), np.zeros(4), np.zeros(4), groups)
    np.testing.assert_allclose(uniform[1:4], 1, atol=1e-7)
    np.testing.assert_allclose(uniform[4:], 0, atol=1e-7)
    assert js(np.array([1., 0]), np.array([0., 1])) == 1
    assert entropy(np.array([1., 0])) == 0
    # Entropy decomposition and JS contraction check the information lost by
    # mapping item IDs to first-token branches, using unnormalized entropies.
    rng = np.random.default_rng(79)
    for _ in range(10):
        a, b = [distribution(rng.random(4)) for _ in range(2)]
        ar, br = [np.bincount(groups, weights=p, minlength=2) for p in (a, b)]
        conditional = sum(ar[g]*entropy(a[groups == g]/ar[g]) for g in (0, 1))
        np.testing.assert_allclose(entropy(a), entropy(ar)+conditional, atol=1e-12)
        assert js(a, b) >= js(ar, br)-1e-12
    checks.update(same_root_different_item_disagreement_detected=True,
                  item_permutation_and_score_shift_invariant=True,
                  exactly_three_coordinates_replaced=True,
                  entropy_chain_rule=True, js_coarsening_contraction=True)
    return checks
