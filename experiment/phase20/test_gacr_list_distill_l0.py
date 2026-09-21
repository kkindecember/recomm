"""Protect the short-run schedule and endpoint decision against silent changes."""
import copy
import json

import numpy as np
import pytest
import torch
from transformers import get_linear_schedule_with_warmup

from experiment.phase20.gacr_list_distill_l0 import optimizer_and_scheduler
from experiment.phase20 import analyze_gacr_list_distill_l0 as analysis


def test_short_run_is_original_schedule_prefix_with_tenfold_lr():
    a, b = torch.nn.Parameter(torch.tensor([1.])), torch.nn.Parameter(torch.tensor([1.]))
    original = torch.optim.AdamW([a], lr=1e-6, weight_decay=.01, eps=1e-8)
    original_schedule = get_linear_schedule_with_warmup(original, 9, 171)
    short, short_schedule = optimizer_and_scheduler([b], 1e-5)
    rates, old_rates = [], []
    for _ in range(32):
        a.grad = torch.ones_like(a); b.grad = torch.ones_like(b)
        rates.append(short.param_groups[0]['lr']); old_rates.append(original.param_groups[0]['lr'])
        original.step(); original_schedule.step(); short.step(); short_schedule.step()
    np.testing.assert_allclose(rates, np.asarray(old_rates)*10, rtol=1e-14, atol=0)
    assert rates[0] == 0 and np.count_nonzero(rates) == 31
    assert rates[9] == 1e-5 and rates[-1] == pytest.approx(1e-5*(171-31)/(171-9))
    assert short.state[b]['step'] == 32


def passing_comparison():
    return {'fit_covered': {'teacher_kl_delta': -.01, 'promoted_demoted_satisfaction_delta': .001},
            'calibration_covered': {'teacher_kl_delta': -.02, 'promoted_demoted_satisfaction_delta': .002},
            'calibration_all': {'ndcg10_delta_user_mean': .0001}}


@pytest.mark.parametrize('group,metric', [('fit_covered', 'teacher_kl_delta'),
    ('fit_covered', 'promoted_demoted_satisfaction_delta'), ('calibration_covered', 'teacher_kl_delta'),
    ('calibration_covered', 'promoted_demoted_satisfaction_delta'), ('calibration_all', 'ndcg10_delta_user_mean')])
def test_every_gate_is_required_and_ties_do_not_pass(group, metric):
    baseline = passing_comparison()
    assert analysis.decision(baseline)['passed']
    for value in (0., None, float('nan')):
        candidate = copy.deepcopy(baseline); candidate[group][metric] = value
        assert not analysis.decision(candidate)['passed']


def test_endpoint_analysis_handles_missing_targets_and_repeated_users(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    out = tmp_path/'run'; out.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value))
    write(out/'config.json', dict(distillation_cache='cache.npz', gate='fixture'))
    write(out/'training_subset.json', dict(examples=4096))
    indices = np.arange(8)
    # The three covered calibration prefixes include a repeated user.
    pos = np.asarray([40, 40, -1, -1, 40, 40, 40, -1])
    parent = np.tile(np.linspace(1, -1, 50), (8, 1))
    q = np.exp(analysis.logprob(-parent))
    np.savez(tmp_path/'cache.npz', positions=pos, true_probability=q, parent_scores=parent,
             fit=np.asarray([True]*4+[False]*4))
    write(out/'panel.json', dict(fit_indices=[0,1,2,3], calibration_indices=[4,5,6,7],
                                samples=[dict(user_id=u) for u in ['a','b','c','d','e','e','f','g']]))
    for arm in analysis.ARMS:
        folder = out/arm; folder.mkdir()
        write(folder/'summary.json', dict(state='COMPLETED', optimizer_steps=32))
        for step in (0,32):
            scores = -parent if arm == 'true_teacher' and step == 32 else parent
            np.savez(folder/f'step{step}.scores.npz', indices=indices, scores=scores, ce=np.ones(8))
    result = analysis.analyze(out)
    assert result['gate']['passed']
    groups = result['groups']['true_teacher_step32']
    assert groups['calibration_all']['prefixes'] == 4 and groups['calibration_all']['users'] == 3
    assert groups['calibration_all']['hit10'] == .75
    assert groups['calibration_all']['hit10_user_mean'] == pytest.approx(2/3)
    assert groups['fit_uncovered']['hit10'] == 0
    assert groups['fit_covered']['promoted_demoted_satisfaction'] == 1.
    assert groups['fit_covered']['teacher_kl'] == pytest.approx(0., abs=1e-12)
    assert (out/'metrics.csv').is_file()
