import numpy as np

from eval_p9x_fixed_pcrf import metric_delta, ranks_and_top10


def test_metric_delta():
    assert metric_delta({"count": 2, "Hit@10": 0.5}, {"count": 2, "Hit@10": 0.25}) == {"Hit@10": 0.25}


def test_zero_weight_preserves_sequence_rank():
    records = [{
        "seq": np.asarray([0.1, 0.3, 0.2]),
        "cf": np.asarray([3.0, 1.0, 2.0]),
        "candidate_frequencies": np.asarray([1.0, 2.0, 3.0]),
        "tail_mass": 0.0,
        "target_position": 2,
    }]
    ranks, top10 = ranks_and_top10(records, 0.0, 0.0, 0.0)
    assert ranks.tolist() == [2]
    assert top10[0].tolist() == [1, 2, 0]
