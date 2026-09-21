"""User isolation checks and a discarded, training-only CPU timing preflight."""
import copy
import io
import time
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiment.phase20.lime_learnability import (
    ARMS, CONFIG, ROOT, arrays, cohorts, fit, last_prefix_indices,
    predict, read, sha, training_bundle, user_folds, write, now)
from experiment.phase20.source_competition_core import target_positions
from experiment.phase20.test_source_competition import SourceCompetitionTests


def synthetic():
    rng = np.random.RandomState(79)
    users = np.asarray([f'user{i}' for i in range(60)])
    def make(u):
        n = len(u)
        ids = np.tile(np.arange(1, 7), (n, 1)); ids[:, -1] = 0
        targets = rng.randint(1, 6, size=n)
        return dict(users=u, ids=ids, targets=targets, positions=target_positions(ids, targets),
                    x=rng.randn(n, 6, 10).astype(np.float32),
                    prior=rng.randn(n, 6).astype(np.float32))
    train = make(np.repeat(users[:40], 2))
    train['train_indices'] = np.arange(80)
    # User 0's last prefix is uncovered, while the earlier one is covered.
    train['targets'][1] = 999; train['positions'][1] = -1
    return train, make(users)


class LearnabilityTests(unittest.TestCase):
    def setUp(self):
        self.c = read(CONFIG)
        self.train, self.dev = synthetic()

    def test_latest_prefix_precedes_coverage_filter_and_is_order_independent(self):
        latest = last_prefix_indices(self.train['users'], self.train['train_indices'])
        self.assertEqual(latest['user0'], 1)
        self.assertEqual(self.train['positions'][latest['user0']], -1)
        rng = np.random.RandomState(13); order = rng.permutation(80)
        other = last_prefix_indices(self.train['users'][order], self.train['train_indices'][order])
        for user, row in latest.items():
            self.assertEqual(order[other[user]], row)
        _, _, used = cohorts(self.train, self.dev, 'Toys', self.c)
        p = next(p for p in used if 1 in p['B_historical'])
        _, _, _, covered = training_bundle(self.train, p['B_historical'])
        self.assertNotIn(0, covered)
        self.assertNotIn(1, covered)

    def test_every_user_once_outside_fit_and_matched_before_coverage(self):
        folds, match, parts = cohorts(self.train, self.dev, 'Toys', self.c)
        self.assertEqual(int((match >= 0).sum()), 40)
        reverse = np.arange(59, -1, -1)
        np.testing.assert_array_equal(folds[reverse], user_folds(self.dev['users'][reverse], 'Toys'))
        visits = np.zeros(60, dtype=int)
        for p in parts:
            visits[p['evaluation']] += 1
            b = self.train['users'][p[ARMS[0]]]; c = self.dev['users'][p[ARMS[1]]]
            np.testing.assert_array_equal(b, c)
            self.assertFalse(set(b) & set(self.dev['users'][p['evaluation']]))
        np.testing.assert_array_equal(visits, np.ones(60, dtype=int))

    def test_heldout_features_and_labels_cannot_change_fit_or_normalization(self):
        _, _, parts = cohorts(self.train, self.dev, 'Beauty', self.c)
        p = parts[0]; ix = p['C_next_interaction']
        first, mu1, sd1, used1 = training_bundle(self.dev, ix)
        altered = copy.deepcopy(self.dev)
        alter = p['evaluation']
        altered['x'][alter] = 1e6
        altered['prior'][alter] = -1e6
        altered['targets'][alter] = 999
        altered['positions'][alter] = -1
        # Padding must have no influence on the fit-only feature statistics.
        altered['x'][altered['ids'] == 0] = -1e8
        second, mu2, sd2, used2 = training_bundle(altered, ix)
        for a, b in ((mu1, mu2), (sd1, sd2), (used1, used2)):
            np.testing.assert_array_equal(a, b)
        for k in first: torch.testing.assert_close(first[k], second[k], rtol=0, atol=0)
        m1, a = fit(first, self.c, steps=2); m2, b = fit(second, self.c, steps=2)
        self.assertEqual(a['final_sha256'], b['final_sha256'])

    def test_fixed_fit_reproducibility_and_different_inference_chunks(self):
        _, _, parts = cohorts(self.train, self.dev, 'Toys', self.c)
        p = parts[0]; bundle, mu, sd, _ = training_bundle(self.dev, p['C_next_interaction'])
        # Fit chunk is frozen in the protocol. Adam can amplify roundoff in
        # near-zero gradients, so optimizer trajectories across fit chunks are
        # not required to be bit-identical. Raw gradient equivalence is checked
        # separately in SourceCompetitionTests.
        m1, a = fit(bundle, self.c, steps=2); m2, b = fit(bundle, self.c, steps=2)
        self.assertEqual(a['initial_sha256'], b['initial_sha256'])
        self.assertEqual(a['final_sha256'], b['final_sha256'])
        ix = p['evaluation']
        s1 = predict(m1, self.dev['x'][ix], self.dev['prior'][ix], self.dev['ids'][ix], mu, sd, 128)
        s2 = predict(m2, self.dev['x'][ix], self.dev['prior'][ix], self.dev['ids'][ix], mu, sd, 3)
        np.testing.assert_allclose(s1, s2, atol=2e-6, rtol=1e-6)
        self.assertTrue(np.isneginf(s1[:, -1]).all())


def preflight():
    c = read(CONFIG); out = ROOT/c['preflight_output']
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(c['cpu_threads']); torch.set_num_interop_threads(1)
    started = time.monotonic()
    with threadpool_limits(c['cpu_threads']):
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(t)
                                   for t in (SourceCompetitionTests, LearnabilityTests))
        log = io.StringIO()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        (out/'unit_tests.log').write_text(log.getvalue())
        assert result.wasSuccessful(), log.getvalue()
        parent = read(ROOT/c['input']/'summary.json')
        assert parent['state'] == 'COMPLETED' and parent['verification']['state'] == 'VERIFIED'
        cache_hashes = {}; counts = {}; profiles = []
        for domain in c['domains']:
            data = []
            for name in ('train.npz', 'development.npz'):
                path = ROOT/c['input']/domain/name; rel = str(path.relative_to(ROOT))
                cache_hashes[rel] = sha(path)
                assert cache_hashes[rel] == parent['output_sha256'][rel]
                d = arrays(path)
                np.testing.assert_array_equal(target_positions(d['ids'], d['targets']), d['positions'])
                data.append(d)
            train, dev = data
            folds, match, parts = cohorts(train, dev, domain, c)
            assert len(dev['users']) == {'Toys': 17364, 'Beauty': 20315}[domain]
            assert (match >= 0).sum() == {'Toys': 3494, 'Beauty': 4085}[domain]
            counts[domain] = [dict(fold=p['fold'], fit_users=len(p['C_next_interaction']),
                historical_covered=int((train['positions'][p['B_historical']] >= 0).sum()),
                next_covered=int((dev['positions'][p['C_next_interaction']] >= 0).sum()),
                evaluation_users=len(p['evaluation'])) for p in parts]
            for arm in ARMS:
                d = train if arm == ARMS[0] else dev
                ix = parts[0][arm]; ix = ix[d['positions'][ix] >= 0]
                # Fixed first 512 covered fit users; no held-out scores or metrics.
                bundle, _, _, _ = training_bundle(d, ix[:512])
                _, meta = fit(bundle, c, steps=2)
                full_users = sum(q['historical_covered' if arm == ARMS[0] else 'next_covered']
                                 for q in counts[domain])
                estimate = meta['seconds']/2*c['steps']*full_users/len(bundle['x'])
                profiles.append(dict(domain=domain, arm=arm, discarded_updates=2,
                    sampled_fit_users=len(bundle['x']), seconds=meta['seconds'],
                    estimated_five_fold_fit_seconds=estimate))
        paths = list((ROOT/'experiment/phase20').glob('*lime_learnability*'))
        paths += [ROOT/'experiment/phase20'/name for name in (
            'source_competition_core.py', 'catalog_scope_core.py', 'guided_decoding_job.py',
            'test_source_competition.py')]
        paths += [ROOT/c['contract']]
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}
        receipt = dict(state='PASSED', completed_at=now(), unit_tests=result.testsRun,
            seconds=time.monotonic()-started, source_sha256=hashes, cache_sha256=cache_hashes,
            cohorts=counts, profiles=profiles, discarded_profile_updates=8,
            estimated_training_seconds=sum(p['estimated_five_fold_fit_seconds'] for p in profiles),
            estimate_excludes_cache_io_checkpoint_writes_prediction_and_verification=True,
            new_development_metrics_computed=False, gpu_used=False, test_read=False)
        write(out/'checks.json', receipt)
        print({k:v for k,v in receipt.items() if k not in ('source_sha256','cache_sha256','cohorts')}, flush=True)


if __name__ == '__main__': preflight()
