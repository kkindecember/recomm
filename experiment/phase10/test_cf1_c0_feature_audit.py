import sys
from pathlib import Path

import numpy as np


PHASE10 = Path(__file__).resolve().parent
if str(PHASE10) not in sys.path:
    sys.path.insert(0, str(PHASE10))

from eval_cf1_c0_feature_audit import make_folds, pcrf_scores, source_name, target_rank


def test_folds_are_deterministic_balanced_and_complete():
    users = [f"u{i}" for i in range(23)]
    first = make_folds(users)
    second = make_folds(list(reversed(users)))
    assert first == second
    assert set(first) == set(users)
    counts = np.bincount(list(first.values()), minlength=5)
    assert counts.max() - counts.min() <= 1


def test_source_and_target_rank_semantics():
    gram, cf = {"a", "b"}, {"b", "c"}
    assert source_name("a", gram, cf) == "gram"
    assert source_name("b", gram, cf) == "both"
    assert source_name("c", gram, cf) == "cf_only"
    assert target_rank([0.1, 0.9, 0.2], 1) == 1
    assert target_rank([0.1, 0.9, 0.2], -1, 91) == 91


def test_pcrf_reliability_can_switch_off_item_term():
    seq = np.asarray([0.1, 0.3, 0.2])
    item = np.asarray([0.9, 0.1, 0.2])
    freq = np.asarray([1, 2, 3])
    score = pcrf_scores(seq, item, freq, tail_mass=1.0)
    expected = (seq - seq.mean()) / seq.std()
    assert np.allclose(score, expected)
