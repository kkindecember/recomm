"""Scientific-contract tests: train split, exact fusion, gradients and matched updates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiment.phase20.gacr_fusion_core import cohort, standardize_tensor, training_scores, pairwise_losses, numpy_ranks
from experiment.phase4.gacr_s0 import BoundedResidualRanker
from experiment.phase20.gacr_score_controls import pcrf_ranks


def test_cohort_matches_historical_prefix_keys_and_calibration():
    from experiment.phase4.gacr_p0 import split_training_users
    from experiment.phase4.gacr_s0 import select_stratified_samples
    from experiment.phase4.gcdh_p0 import build_train_samples, head_items, training_popularity
    users = {f'user{i}' for i in range(30)}
    names = [''] + [f'item{i}' for i in range(1, 12)]
    catalog = {'items': names}
    records, sequences = [], {}
    for i, user in enumerate(sorted(users)):
        # Repeated items and >20 prefix lengths exercise indexing and truncation.
        sequence = [1 + ((j + i) % 11) for j in range(27)] + [1] * 6
        sequences[user] = [names[k] for k in sequence] + ['HELD_OUT_VALIDATION', 'HELD_OUT_TEST']
        for p in range(1, len(sequence)):
            records.append({'user_id': user, 'history': sequence[max(0, p - 20):p], 'target': sequence[p]})
    fit, cal, frequency, meta = cohort(records, catalog, users, 'Toys', calibration_per_group=2)
    fit_users, cal_users = split_training_users(users, 2023, 'Toys')
    mapping = {n: n for n in names[1:]}
    original_fit = build_train_samples(sequences, fit_users, mapping, mapping)
    original_pool = build_train_samples(sequences, cal_users, mapping, mapping)
    heads = head_items(training_popularity(sequences))
    original_cal = select_stratified_samples(original_pool, heads, 2023, 'Toys|gacr-p0-calibration', 2, 2)
    assert [r['sample_key'] for r in fit] == [r['sample_key'] for r in original_fit]
    assert [r['sample_key'] for r in cal] == [r['sample_key'] for r in original_cal]
    assert meta['fit_calibration_user_overlap'] == 0
    expected_pop = training_popularity(sequences)
    assert frequency.tolist() == [expected_pop[n] for n in names]
    broken = [dict(r) for r in records]
    broken[1]['history'] = [11]
    with pytest.raises(AssertionError, match='nonsequential'):
        cohort(broken, catalog, users, 'Toys', calibration_per_group=2)


def test_fusion_forward_matches_original_and_missing_targets():
    rng = np.random.RandomState(4)
    n = 7
    g = rng.randn(n, 50)
    g[0] = 1.  # constant base vector exercises epsilon behavior and stable sorting.
    cf = rng.randn(n, 50)
    cf[1] = 0.
    pop = rng.randint(1, 20, size=(n, 50))
    pop[2] = 5; pop[3] = 6  # inclusive q1, and zero reliability in row2.
    positions = np.asarray([0, 49, -1, 7, 2, -1, 29])
    raw, final, reliability = numpy_ranks(g, cf, pop, positions, 5)
    all_pop = np.concatenate([[0], pop.reshape(-1)])
    records = []
    for i in range(n):
        order = np.argsort(-g[i], kind='stable')
        records.append({'candidate_ids': (i * 50 + order + 1).tolist(), 'seq': g[i, order],
                        'cf': cf[i, order], 'target_position': list(order).index(positions[i]) if positions[i] >= 0 else -1})
    expected = pcrf_ranks(records, all_pop, {'lambda': 1., 'beta': .5, 'gamma': 1., 'q1': 5})
    np.testing.assert_array_equal(final, expected)
    assert final[2] == final[5] == raw[2] == raw[5] == 51 and reliability[2] == 0.
    class Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x[..., 0])
    scores = training_scores(Zero(), torch.zeros(n, 50, 6, dtype=torch.float64), torch.tensor(g),
                             torch.tensor(cf), torch.tensor(pop), 5, 'B2').numpy()
    sorted_order = np.argsort(-g, axis=1, kind='stable')
    # Tie-breaking follows g-order before final PCRF, as in deployment.
    final_order = np.take_along_axis(sorted_order,
        np.argsort(-np.take_along_axis(scores, sorted_order, 1), axis=1, kind='stable'), 1)
    ranks = np.argsort(final_order, axis=1)[np.arange(n), np.maximum(positions, 0)] + 1
    np.testing.assert_array_equal(np.where(positions >= 0, ranks, 51), final)


def test_fusion_gradients_and_constant_vector_are_finite():
    torch.manual_seed(3)
    ranker = BoundedResidualRanker(6, 16, .2).double()
    with torch.no_grad():
        ranker.network[-1].weight.normal_(0, .03)
    x = torch.randn(3, 50, 6, dtype=torch.float64)
    base = torch.linspace(1, .02, 50, dtype=torch.float64).repeat(3, 1)
    cf = torch.randn(3, 50, dtype=torch.float64)
    pop = torch.randint(1, 20, (3, 50))
    target = torch.tensor([2, 14, 29])
    for arm in ('B1', 'B2'):
        ranker.zero_grad()
        loss = pairwise_losses(training_scores(ranker, x, base, cf, pop, 5, arm), target).mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ranker.parameters())
        assert ranker.network[-1].weight.grad.abs().sum() > 0
    values = torch.ones(2, 50, dtype=torch.float64, requires_grad=True)
    standardized = standardize_tensor(values)
    standardized.square().sum().backward()
    assert torch.isfinite(values.grad).all()
    # Check a differentiable score input away from rank boundaries.
    train_base = base.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda b: training_scores(ranker, x, b, cf, pop, 5, 'B2'),
                                    (train_base,), fast_mode=True)


def test_all_negative_loss_and_full_batch_chunk_gradient():
    torch.manual_seed(7)
    scores = torch.randn(9, 50, dtype=torch.float64, requires_grad=True)
    target = torch.arange(9)
    losses = pairwise_losses(scores, target)
    expected = torch.stack([torch.stack([torch.nn.functional.softplus(scores[i, j] - scores[i, i])
                            for j in range(50) if j != i]).mean() for i in range(9)])
    torch.testing.assert_close(losses, expected)
    head = torch.arange(9) < 2
    balanced = .5 * losses[head].mean() + .5 * losses[~head].mean()
    exact = torch.autograd.grad(balanced, scores, retain_graph=True)[0]
    for index in (torch.tensor([0, 1]), torch.tensor([2, 3, 4]), torch.tensor([5, 6, 7]), torch.tensor([8])):
        (losses[index].sum() / (2 * (2 if int(index[0]) < 2 else 7))).backward(retain_graph=True)
    torch.testing.assert_close(scores.grad, exact)
    with pytest.raises(AssertionError):
        pairwise_losses(scores, torch.full((9,), -1))


def test_worker_matched_training_saves_fixed_last_step(tmp_path):
    from experiment.phase20.gacr_fusion_train import Work
    torch.set_num_threads(1)
    torch.manual_seed(5)
    parent = tmp_path / 'parent.pt'
    torch.save(BoundedResidualRanker(6, 16, .2).state_dict(), parent)
    arrays = {'features': np.random.RandomState(1).normal(size=(16, 50, 6)).astype(np.float32),
              'base': np.tile((1 / np.arange(1, 51)).astype(np.float32), (16, 1)),
              'cf': np.random.RandomState(2).normal(size=(16, 50)),
              'frequencies': np.tile(np.arange(1, 51), (16, 1)),
              'positions': np.asarray([i % 50 for i in range(14)] + [-1, -1]),
              'head': np.arange(16) % 2 == 0, 'fit': np.arange(16) < 12}
    before = hashlib.sha256(parent.read_bytes()).hexdigest()
    job = object.__new__(Work)
    job.path, job.device = tmp_path, torch.device('cpu')
    job.domain = {'residual': str(parent), 'q1': 5}
    job.config = {'training_steps': 3, 'training_chunk_size': 3}
    job.steps = 0
    job.status = lambda *a, **k: None
    job.progress = lambda *a, **k: None
    results = {arm: job.train(arrays, arm) for arm in ('B1', 'B2')}
    assert job.steps == 6 and hashlib.sha256(parent.read_bytes()).hexdigest() == before
    assert results['B1']['initialization_sha256'] == results['B2']['initialization_sha256']
    for arm, result in results.items():
        assert result['optimizer_steps'] == 3 and result['curve'][-1]['parameter_change_l2'] > 0
        assert (tmp_path / f'{arm}.optimizer.pt').exists()
        assert result['train_only_calibration']['pcrf']['users'] == 4
