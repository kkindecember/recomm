import copy
import numpy as np
import pytest
import torch

from experiment.phase20.item_order_core import order_loss, auxiliary_loss, sample_backward, calibration_gradients
from experiment.phase20.test_gacr_list_distill import fixture
from experiment.phase20.gacr_list_distill_core import encode_history, decode_scores
from experiment.phase20.preference_core import path_labels


def settings():
    return dict(margin=.01, order_weight=2., covered_normalizer=21839/12399,
                ce_weight=1., anchor_weight=1., candidate_chunk=4)


def test_formula_and_gradient_against_independent_nested_loop():
    a = torch.tensor([-.4, -.8, -.1, -.2], dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    p = np.array([.2, .1, .4, .3])
    actual = order_loss(a, p, .01)
    order = sorted(range(4), key=lambda i: -p[i])
    expected = sum(torch.clamp(b[order[j]]-b[order[i]]+(j-i)*.01, min=0)
                   for i in range(4) for j in range(i+1,4))/6
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual,a)[0], torch.autograd.grad(expected,b)[0])


def test_teacher_swap_changes_direction_and_margin_satisfaction_zeroes_gradient():
    a = torch.tensor([0., 0.], requires_grad=True)
    g = torch.autograd.grad(order_loss(a, [1.,0.], .1), a)[0]
    h = torch.autograd.grad(order_loss(a, [0.,1.], .1), a)[0]
    torch.testing.assert_close(g, -h)
    assert g[0] < 0 and g[1] > 0
    a = torch.tensor([2.,0.], requires_grad=True)
    loss = order_loss(a,[1.,0.],.1)
    assert float(loss) == 0 and torch.autograd.grad(loss,a)[0].abs().sum() == 0


def test_additive_shift_and_aligned_permutation_invariance():
    a=torch.tensor([-.4,-.3,-.9,-.1],dtype=torch.float64)
    p=np.array([.3,.2,.4,.1]); perm=np.array([2,0,3,1])
    torch.testing.assert_close(order_loss(a,p,.01),order_loss(a+3,p,.01))
    torch.testing.assert_close(order_loss(a,p,.01),order_loss(a[perm],p[perm],.01))


@pytest.mark.parametrize('arm,covered,divisor', [('CE_anchor',True,128),('order_teacher',True,79),('order_teacher',False,1)])
def test_chunk_replay_matches_full_parameter_gradient(arm,covered,divisor):
    replay,paths,batch=fixture(); direct=copy.deepcopy(replay)
    parent=np.random.RandomState(5).normal(size=50).astype(np.float32)
    probability=np.arange(1,51,dtype=np.float64)/1275
    cfg=settings(); ids=list(range(1,51))
    result=sample_backward(replay,batch,paths,ids,parent,probability,covered,arm,cfg,'cpu',divisor)
    hidden,mask=encode_history(direct,batch,'cpu')
    scores=decode_scores(direct,hidden,mask,path_labels(paths,ids,'cpu'))
    aux,*_=auxiliary_loss(scores,parent,probability,covered,arm,cfg)
    ce=-decode_scores(direct,hidden,mask,batch['target_ids']).mean()
    ((ce+aux)/divisor).backward()
    assert abs(result['loss']-float(ce+aux)) < 1e-5
    for (n,a),(_,b) in zip(replay.named_parameters(),direct.named_parameters()):
        if a.requires_grad: torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)
        else: assert a.grad is None and b.grad is None,n


def test_uncovered_and_control_do_not_depend_on_teacher_order():
    a=torch.linspace(-1,0,50,requires_grad=True); p=np.arange(50,dtype=float)
    for arm,covered in [('CE_anchor',True),('order_teacher',False)]:
        x=auxiliary_loss(a,a.detach()+.1,p,covered,arm,settings())[0]
        y=auxiliary_loss(a,a.detach()+.1,p[::-1].copy(),covered,arm,settings())[0]
        torch.testing.assert_close(x,y)
        torch.testing.assert_close(torch.autograd.grad(x,a)[0],torch.autograd.grad(y,a)[0])


def test_calibration_gradients_match_combined_gradient_without_parameter_change():
    model,paths,batch=fixture(); initial={n:p.detach().clone() for n,p in model.named_parameters()}
    cfg=settings(); ids=list(range(1,51)); probability=np.arange(1,51,dtype=float)/1275
    ce,rank=calibration_gradients(model,batch,paths,ids,probability,cfg,'cpu')
    hidden,mask=encode_history(model,batch,'cpu')
    scores=decode_scores(model,hidden,mask,path_labels(paths,ids,'cpu'))
    objective=-decode_scores(model,hidden,mask,batch['target_ids']).mean()+cfg['covered_normalizer']*order_loss(scores,probability,.01)
    objective.backward()
    direct=torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.requires_grad])
    torch.testing.assert_close(ce+rank,direct,rtol=1e-4,atol=1e-5)
    assert all(torch.equal(p,initial[n]) for n,p in model.named_parameters())


def test_heartbeat_retains_progress_and_cannot_overwrite_terminal_state(tmp_path):
    import json
    from experiment.phase20.item_order_transfer import OrderWork
    job=OrderWork({'output':str(tmp_path)},'profile')
    job.status(stage='check',progress={'completed':3,'total':8})
    job.status()
    assert json.loads((job.path/'status.json').read_text())['progress']['completed']==3
    job.status('PASSED',stage='finished'); job.status()
    assert json.loads((job.path/'status.json').read_text())['state']=='PASSED'


def test_live_report_handles_not_yet_created_worker_outputs(tmp_path,monkeypatch):
    from experiment.phase20 import item_order_transfer as module
    monkeypatch.setattr(module,'OUT',tmp_path)
    monkeypatch.setattr(module,'REPORT',tmp_path/'report.md')
    module.render({'state':'RUNNING','updated_at':'test','details':{'profile':{}}})
    assert 'RUNNING' in (tmp_path/'report.md').read_text()
