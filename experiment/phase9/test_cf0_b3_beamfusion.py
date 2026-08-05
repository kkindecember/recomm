import sys
from pathlib import Path

import numpy as np


PHASE9 = Path(__file__).resolve().parent
if str(PHASE9) not in sys.path:
    sys.path.insert(0, str(PHASE9))

from eval_cf0_b3_beamfusion import (  # noqa: E402
    bootstrap_hit10_delta,
    make_partition,
    metrics_from_ranks,
    normalize_lexical_id,
    load_cached_beams,
    ranks_for_lambda,
    standardize,
)


def test_lexical_normalization_matches_generation_decode():
    assert normalize_lexical_id("|▁game|▁board|be|▁cu|▁contained") == "game boardbe cu contained"
    assert normalize_lexical_id("mel|▁puzzle|case") == "mel puzzlecase"


def test_hash_partition_is_order_invariant_and_exact():
    users = ["u3", "u1", "u2", "u4"]
    first, order = make_partition(users, 2, "2023")
    second, reverse_order = make_partition(list(reversed(users)), 2, "2023")
    assert first == second
    assert order == reverse_order
    assert len(first) == 2


def test_cache_parser_handles_stale_short_header(tmp_path):
    path = tmp_path / "pred.tsv"
    candidates = "||".join(f"item{i}" for i in range(50))
    scores = "||".join(str(-i) for i in range(50))
    path.write_text(
        "idx\tH@5\tH@10\tNDCG@5\tNDCG@10\tgold\tpred\tscores\n"
        f"u1\t0\t0\t0\t0\t0\t0\tgold item\t{candidates}\t{scores}\n"
        "hit@10: 0.1\n",
        encoding="utf-8",
    )
    rows, footer = load_cached_beams(path)
    assert rows["u1"]["gold"] == "gold item"
    assert len(rows["u1"]["candidates"]) == 50
    assert footer == {"hit@10": 0.1}


def test_zero_weight_is_identity_and_positive_weight_can_rerank():
    records = [
        {
            "seq_z": standardize(np.array([3.0, 2.0, 1.0])),
            "cf_z": standardize(np.array([0.0, 1.0, 4.0])),
            "target_position": 2,
        }
    ]
    assert ranks_for_lambda(records, [0], 0.0).tolist() == [3]
    assert ranks_for_lambda(records, [0], 2.0).tolist() == [1]


def test_metrics_and_paired_bootstrap_are_finite():
    baseline = np.array([11, 11, 1, 51])
    fused = np.array([1, 2, 1, 51])
    metrics = metrics_from_ranks(fused)
    assert metrics["Hit@10"] == 0.75
    interval = bootstrap_hit10_delta(baseline, fused, 100, 7)
    assert np.isfinite(interval["lower"])
    assert np.isfinite(interval["upper"])
