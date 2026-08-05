import torch

from experiment.phase6.gacr_v7 import assess_calibration_noninferiority
from experiment.phase6.gacr_v8 import ListwiseResidualRanker, _zscore, evaluate, integrity_is_valid, make_model, to_cpu_v8_record


def _record(group="head"):
    return {"sample_key":group,"target_group":group,"target_index":1,"gram_rank":2,
        "base":torch.tensor([1.,.5,0.]),"features6":torch.ones(3,6),"features10":torch.ones(3,10)}


def test_path_zscore_is_finite_and_centered():
    values=_zscore(torch.tensor([1.,2.,3.]))
    assert torch.isfinite(values).all()
    assert abs(float(values.mean())) < 1e-6


def test_v8_cpu_record_preserves_both_feature_interfaces_without_legacy_features_key():
    record = _record()
    copied = to_cpu_v8_record(record)
    assert set(("base", "features6", "features10")) <= set(copied)
    assert all(copied[key].device.type == "cpu" for key in ("base", "features6", "features10"))
    assert "features" not in copied


def test_integrity_treats_forbidden_data_reads_as_required_false_flags():
    good={"all_fit_records_used":True,"fit_calibration_user_disjoint":True,"parent_checkpoint_sha_unchanged_during_training":True,"backbone_optimizer_steps":0,"test_data_read":False,"sports_data_read":False}
    assert integrity_is_valid(good)
    assert not integrity_is_valid(good | {"test_data_read":True})


def test_listwise_zero_init_is_identity_and_isolated_per_list():
    model=ListwiseResidualRanker()
    first=torch.randn(3,10); second=torch.randn(3,10)
    with torch.no_grad():
        left=model(first); combined=model(torch.cat((first,second),0))[:3]
    assert torch.equal(left, torch.zeros_like(left))
    assert torch.equal(combined, torch.zeros_like(combined))


def test_pointwise_and_listwise_evaluation_preserve_identity_at_initialization():
    records=[_record("head"),_record("tail")]
    for arm in ("C","D","E"):
        model=make_model(arm,torch.device("cpu")); groups,rows=evaluate(records,model.state_dict(),arm,torch.device("cpu"))
        assert groups["overall"]["broad_harm_rate"] == 0.0
        assert all(row["candidate_rank"] == row["baseline_rank"] for row in rows)


def test_v8_gate_uses_the_frozen_v7_boundaries():
    groups={"overall":{"baseline_Recall@10":.2,"candidate_Recall@10":.198,"baseline_Recall@50":.3,"candidate_Recall@50":.298,"broad_harm_rate":.01},"tail":{"baseline_Recall@50":.2,"candidate_Recall@50":.196,"baseline_NDCG@10":.1,"candidate_NDCG@10":.0995}}
    config={"calibration_noninferiority":{"broad_harm_max":.01,"overall_recall10_absolute_delta_min":-.002,"overall_recall50_absolute_delta_min":-.002,"tail_recall50_absolute_delta_min":-.004,"tail_ndcg10_absolute_delta_min":-.0005}}
    assert assess_calibration_noninferiority(groups,config)["eligible"]


def test_v8_source_does_not_name_test_or_sports_paths():
    source=open("experiment/phase6/gacr_v8.py").read()
    assert "test_users.txt" not in source
    assert '"Sports"' not in source
