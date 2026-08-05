import numpy as np

from eval_bw2_anchored_expansion import anchor_standardize, anchored_ranks, load_fresh_beams, scientific_gate


def test_anchor_standardize_uses_only_prefix():
    values = np.arange(6, dtype=float)
    result = anchor_standardize(values, 3)
    expected = (values - np.mean(values[:3])) / np.std(values[:3])
    assert np.allclose(result, expected)


def test_anchored_rank_preserves_anchor_order():
    record = {
        "seq": np.asarray([3.0, 2.0, 1.0, 0.0]),
        "cf": np.asarray([0.0, 0.0, 0.0, 0.0]),
        "candidate_frequencies": np.ones(4),
        "candidate_ids": [1, 2, 3, 4],
        "tail_mass": 0.0,
        "target_position": 1,
    }
    ranks, _, identity = anchored_ranks([record], anchor_size=3)
    assert ranks.tolist() == [2]
    assert identity == [True]


def test_scientific_gate():
    rows = [
        {"hit10_delta": 0.002, "ndcg10_delta": 0.0, "users_with_expansion_in_top10": 1},
        {"hit10_delta": 0.0, "ndcg10_delta": -0.0005, "users_with_expansion_in_top10": 0},
    ]
    assert scientific_gate(rows)["status"] == "passed"


def test_load_fresh_beams_respects_expected_width(tmp_path):
    path = tmp_path / "beams.tsv"
    path.write_text("idx\tgold\tpred\tscores\nu1\tg\ta||b\t0.2||0.1\n", encoding="utf-8")
    loaded = load_fresh_beams(path, 2)
    assert loaded["u1"]["candidates"] == ["a", "b"]
