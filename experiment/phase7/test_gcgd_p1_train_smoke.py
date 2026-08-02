import torch

from experiment.phase7.gcgd_p1_train_smoke import root_reliability_feature


def test_root_reliability_feature_is_target_free_and_bounded():
    feature = root_reliability_feature(
        torch.tensor([2.0, 1.0, -1.0]),
        torch.log_softmax(torch.tensor([0.2, 0.3, 0.5]), dim=0),
        1.0,
    )
    assert len(feature) == 6
    assert all(0.0 <= value <= 1.0 for value in feature)
    assert feature[-1] == 0.0
