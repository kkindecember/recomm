import torch

from experiment.phase8.tipa_p0 import PathAlignmentAdapter, TIPAProcessor, normalized_child_features, rank_agreement, select_prefix_depth, stable_user_split


def test_zero_adapter_is_exact_identity_and_bounded_after_update():
    adapter=PathAlignmentAdapter(16,.3); features=torch.randn(4,6)
    assert torch.equal(adapter(features),torch.zeros(4))
    with torch.no_grad(): adapter.network[-1].weight.fill_(10)
    assert float(adapter(features).abs().max()) <= .300001


def test_processor_changes_only_declared_legal_children_and_null_path_is_identity():
    adapter=PathAlignmentAdapter(8,.2)
    with torch.no_grad(): adapter.network[-1].weight[0,0]=1
    processor=TIPAProcessor({(0,):{2:-.2,4:-1.7}},{(0,):.5},adapter,4)
    scores=torch.arange(6,dtype=torch.float32).unsqueeze(0); output=processor(torch.tensor([[0]]),scores)
    assert torch.equal(output[0,[0,1,3,5]],scores[0,[0,1,3,5]])
    null=processor(torch.tensor([[0,2]]),scores)
    assert torch.equal(null,scores)


def test_features_are_finite_and_teacher_mass_preserved():
    teacher=torch.log_softmax(torch.tensor([2.,1.,0.]),0); features=normalized_child_features(torch.tensor([1.,1.,1.]),teacher,depth=2,maximum_depth=5,leaf_fraction=.2)
    assert features.shape==(3,6) and torch.isfinite(features).all()
    assert abs(float(teacher.exp().sum())-1)<1e-6


def test_rank_agreement_has_expected_endpoints():
    scores={"a":3.,"b":2.,"c":1.}
    assert rank_agreement(["a","b","c"],scores)==1
    assert rank_agreement(["c","b","a"],scores)==-1


def test_user_split_is_deterministic_and_disjoint():
    users={f"u{i}" for i in range(20)}; a,b=stable_user_split(users,"Toys","salt",.8); c,d=stable_user_split(users,"Toys","salt",.8)
    assert (a,b)==(c,d) and not(a&b) and a|b==users


def test_recovery_sampling_selects_only_branching_prefixes_deterministically():
    path=(0,10,20,1); scores={(0,):{10:-.1,11:-2.},(0,10):{20:0.},(0,10,20):{1:0.}}
    first=select_prefix_depth(path,scores,2023,"Toys","sample","branching_teacher_path")
    second=select_prefix_depth(path,scores,2023,"Toys","sample","branching_teacher_path")
    assert first==second==1
    assert select_prefix_depth(path,{(0,):{10:0.}},2023,"Toys","sample","branching_teacher_path") is None
