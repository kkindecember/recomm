import copy
import numpy as np
import torch
from experiment.phase20.test_gacr_list_distill import fixture
from experiment.phase20.gacr_list_distill_diagnostic import component_gradients, flat_gradient, select_panel
from experiment.phase20.gacr_list_distill_core import sample_backward


def test_split_weighted_gradients_sum_to_original_and_do_not_update_parameters():
    model,paths,batch=fixture();reference=copy.deepcopy(model)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    parent=np.random.RandomState(12).normal(size=50).astype(np.float32)
    q=np.exp(np.arange(50)/50);q/=q.sum()
    settings=dict(candidate_chunk=4,temperature=1.,anchor_weight=1.,teacher_weight=.1,
                  covered_normalizer=21839/12399,ce_weight=1.)
    g,_=component_gradients(model,batch,paths,list(range(1,51)),parent,q,settings,'cpu')
    sample_backward(reference,batch,paths,list(range(1,51)),parent,q,True,'true_teacher',settings,'cpu')
    torch.testing.assert_close(sum(g.values()),flat_gradient(reference),rtol=1e-4,atol=2e-6)
    assert all(torch.equal(p,before[n]) and p.grad is None for n,p in model.named_parameters())


def test_selection_stays_disjoint_deterministic_and_does_not_drop_calibration():
    records=[dict(user_id=f'u{i//2}',sample_key=f'k{i}') for i in range(1200)]
    fit=np.arange(1200)<944;positions=np.where(np.arange(1200)%2==0,3,-1)
    first=select_panel(records,fit,positions);assert first==select_panel(records,fit,positions)
    panel,cal,gradient=first
    assert len(panel)==len({records[i]['user_id'] for i in panel})==256
    assert sum(positions[i]>=0 for i in panel)==192
    assert cal==list(range(944,1200)) and len(gradient)==32
    assert set(gradient)<=set(panel) and all(positions[i]>=0 for i in gradient)
