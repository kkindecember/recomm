"""Resource revision: changed candidate batching must preserve the objective."""
import copy
import numpy as np
import pytest
import torch
from experiment.phase20.test_gacr_list_distill import fixture
from experiment.phase20.item_order_core import sample_backward, calibration_gradients


@pytest.mark.parametrize('arm,covered', [('CE_anchor',True),('order_teacher',True),('order_teacher',False)])
def test_chunk16_and_chunk4_parameter_gradients_match(arm,covered):
    small,paths,batch=fixture(); large=copy.deepcopy(small)
    settings=dict(margin=.01,order_weight=3.254811234597535,covered_normalizer=21839/12399,ce_weight=1.,anchor_weight=1.)
    ids=list(range(1,51)); parent=np.linspace(-1,0,50); probability=np.arange(1,51,dtype=float)/1275
    results=[]
    for model,chunk in ((small,4),(large,16)):
        results.append(sample_backward(model,batch,paths,ids,parent,probability,covered,arm,dict(settings,candidate_chunk=chunk),'cpu',79))
    assert abs(results[0]['loss']-results[1]['loss'])<1e-5
    for (n,p),(_,q) in zip(small.named_parameters(),large.named_parameters()):
        if p.requires_grad: torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)
        else: assert p.grad is None and q.grad is None,n


def test_gradient_calibration_is_consistent_across_chunks():
    small,paths,batch=fixture(); large=copy.deepcopy(small)
    cfg=dict(margin=.01,ce_weight=1.,covered_normalizer=21839/12399)
    ids=list(range(1,51)); p=np.arange(1,51,dtype=float)/1275
    a=calibration_gradients(small,batch,paths,ids,p,dict(cfg,candidate_chunk=4),'cpu')
    b=calibration_gradients(large,batch,paths,ids,p,dict(cfg,candidate_chunk=16),'cpu')
    for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)


def test_training_and_evaluation_code_unchanged_by_resource_revision():
    import inspect
    from experiment.phase20 import item_order_transfer as old,item_order_transfer_chunk16 as new
    for method in ('train_arm','evaluate','backward','load_cache','profile'):
        assert inspect.getsource(getattr(old.OrderWork,method))==inspect.getsource(getattr(new.OrderWork,method))
