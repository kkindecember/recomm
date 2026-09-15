"""Contracts for immutable cohorts, matched baselines and seed aggregation."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiment.phase20.gacr_b1_multiseed import CachedWork, validate_cache
from experiment.phase20.gacr_b1_multiseed_queue import aggregate, KEYS


def test_cache_reuse_preserves_uncovered_and_calibration():
    n = 260
    arrays = {'features': np.zeros((n, 50, 6), np.float32), 'base': np.ones((n, 50), np.float32),
              'candidate_ids': np.tile(np.arange(1, 51), (n, 1)), 'cf': np.zeros((n, 50)),
              'fit': np.arange(n) < 4, 'head': np.arange(n) % 2 == 0,
              'positions': np.array([0, 2, -1, -1] + [0] * 256)}
    manifest = {'rows': n, 'cohort': {'fit_samples': 4, 'calibration_samples': 256, 'fit_calibration_user_overlap': 0},
                'fit_covered': 2, 'fit_uncovered': 2, 'calibration_covered': 256,
                'forced_positive_insertions': 0, 'test_read': False}
    validate_cache(arrays, manifest)
    arrays['positions'][2] = 0
    with pytest.raises(AssertionError):
        validate_cache(arrays, manifest)
    arrays['positions'][2] = -1
    arrays['fit'][5] = True
    with pytest.raises(AssertionError):
        validate_cache(arrays, manifest)


def test_mismatched_seed_baseline_blocks_optimizer(tmp_path):
    reference = tmp_path / 'reference.tsv'
    reference.write_text('user_id\tfixed\nuser1\t3\n')
    known = tmp_path / 'known.tsv'
    known.write_text('user_id\tc1\nuser1\t4\n')
    work = object.__new__(CachedWork)
    work.seed, work.path = 2024, tmp_path
    work.domain = {'reference_ranks': {'2024': str(reference)}, 'known_ranks': str(known), 'residual': 'unused.pt'}
    work.load_cache = lambda: {}
    work.validation_cache = lambda: {'users': ['user1']}
    work.score = lambda *a: (np.array([2]), np.array([2]), None, 0.)
    def forbid(*a):
        raise RuntimeError('Optimizer must not run before exact matched B0 parity')
    work.train = forbid
    with pytest.raises(AssertionError):
        work.run()
    assert not (tmp_path / 'B1.pt').exists()


def test_aggregation_uses_own_B0_and_marks_old_2023(tmp_path):
    config = {'output': str(tmp_path / 'out'), 'new_seeds': [2024, 2025], 'domains': {}}
    for d in ('Toys', 'Beauty'):
        known = tmp_path / f'{d}_known.json'
        config['domains'][d] = {'known_summary': str(known)}
        for seed, b0, gain in ((2023, .1, .01), (2024, .2, -.001), (2025, .3, .003)):
            g = {a: {k: b0 + (gain if a.startswith('B1') else 0) for k in KEYS} for a in ('B0_raw', 'B0', 'B1_raw', 'B1')}
            g['original_pcrf'] = {k: .09 for k in KEYS}
            path = known if seed == 2023 else Path(config['output']) / d / f'seed{seed}/summary.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'state': 'COMPLETED', 'new_optimizer_steps': 30, 'groups': {'all_validation': g}}))
    result = aggregate(config)
    assert result['new_optimizer_steps'] == 120
    for d in config['domains']:
        s = result['domains'][d]
        assert s['seed_sources']['2023'] == 'previously_observed'
        assert s['decision']['ndcg@10_vs_B0']['positive_count'] == 2
        assert s['decision']['ndcg@10_vs_B0']['mean_delta'] == pytest.approx(.004)
        assert s['aggregate']['all_validation']['B1']['ndcg@10']['mean'] == pytest.approx(.204)
