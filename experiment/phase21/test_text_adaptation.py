import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from experiment.phase21.text_adaptation_core import score_user, unit_rows, pool_inputs, query_weights, screen_rule


class TextAdaptationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.e = torch.tensor([[2., 0., 0.], [0., .5, 0.], [1., .2, .1], [.1, .8, .4], [0., 0., 1.]], dtype=torch.float64)
        self.v = unit_rows(self.e); self.history = [1, 2, 1, 4]

    def reference(self, arm, mapping=None):
        e = self.e.numpy(); h = e[np.array(self.history)-1]
        r = np.array([np.exp(-.1*(len(h)-1-t)) for t in range(len(h))], dtype=np.float32).astype(np.float64)
        norm = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        mean = (r[:, None] * h).sum(0) / r.sum()
        values = []
        for i, target in enumerate(e):
            if arm == 'A': pooled = mean
            else:
                query = mean if arm == 'B' else e[i if mapping is None else mapping[i]]
                z = norm(query) @ norm(h).T / .1
                a = r * np.exp(z - max(z)); a /= a.sum()
                pooled = a @ h
            values.append(float(norm(target) @ norm(pooled)))
        return np.array(values)

    def test_each_arm_matches_independent_loop_including_raw_norms_and_repeats(self):
        mapping = np.array([1, 2, 3, 4, 0])
        for arm in ('A', 'B', 'C', 'D'):
            got = score_user(self.e, self.v, self.history, arm, block=2,
                             query_map=torch.tensor(mapping) if arm == 'D' else None)
            np.testing.assert_allclose(got.numpy(), self.reference(arm, mapping if arm == 'D' else None), atol=1e-12, rtol=1e-12)

    def test_candidate_choice_changes_weights_and_blocking_does_not_change_scores(self):
        h, r, mean = pool_inputs(self.e, self.history)
        weights = query_weights(self.e[:2], h, r)
        self.assertFalse(torch.allclose(weights[0], weights[1]))
        a = score_user(self.e, self.v, self.history, 'C', block=1)
        b = score_user(self.e, self.v, self.history, 'C', block=5)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_single_history_and_same_query_map_reduce_to_expected_controls(self):
        for arm in ('A', 'B', 'C'):
            torch.testing.assert_close(score_user(self.e, self.v, [2], arm), self.v @ unit_rows(self.e[1]), rtol=1e-7, atol=1e-8)
        c = score_user(self.e, self.v, self.history, 'C')
        d = score_user(self.e, self.v, self.history, 'D', query_map=torch.arange(len(self.e)))
        torch.testing.assert_close(c, d, rtol=0, atol=0)

    def test_api_has_no_target_and_inputs_are_unchanged(self):
        before = self.e.clone(); history = list(self.history)
        self.assertNotIn('target', inspect.signature(score_user).parameters)
        score_user(self.e, self.v, self.history, 'C')
        self.assertEqual(history, self.history); torch.testing.assert_close(before, self.e, rtol=0, atol=0)

    def test_gate_requires_both_domains_and_net_hit_preservation(self):
        def domain(a=.1, b=.11, c=.12, ch=.2):
            return {'arms': {k: {'metrics': {'fusion_catalog': {'N10': n, 'H10': ch if k == 'C' else .2}}}
                             for k, n in [('A', a), ('B', b), ('C', c)]}}
        self.assertTrue(screen_rule({'Toys': domain(), 'Beauty': domain()})['pass_for_actual_generation'])
        self.assertFalse(screen_rule({'Toys': domain()})['pass_for_actual_generation'])
        self.assertFalse(screen_rule({'Toys': domain(), 'Beauty': domain(ch=.19)})['pass_for_actual_generation'])
        self.assertFalse(screen_rule({'Toys': domain(), 'Beauty': domain(b=.13)})['pass_for_actual_generation'])


class PipelineIntegrationTest(unittest.TestCase):
    def test_shared_queue_waits_while_old_scheduler_can_fill_slots(self):
        from experiment.phase21 import text_adaptation as app
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, 'RECOVERY', Path(tmp)):
            app.write(Path(tmp)/'status.json', {'state':'RUNNING','pending':[{'arm':'old'}]})
            self.assertEqual(app.external_capacity(), (0,set(),'existing_queue_has_priority'))

    def test_completed_old_queue_releases_capacity(self):
        from experiment.phase21 import text_adaptation as app
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, 'RECOVERY', Path(tmp)):
            app.write(Path(tmp)/'status.json', {'state':'COMPLETED','pending':[]})
            self.assertEqual(app.external_capacity(), (2,set(),None))

    def test_real_fusion_trainer_wiring_and_development_label_isolation(self):
        import sys
        from experiment.phase21 import text_adaptation as app
        sys.path.insert(0, str(app.ROOT/'third_party/LIME-Rec'))
        torch.set_num_threads(2)
        n, v = 1233, 7
        rng = np.random.default_rng(1)
        raw = rng.random((n+5,3,v),dtype=np.float32)
        replacement = rng.random((n,v),dtype=np.float32)
        samples = [dict(user_id=f'{j:04d}',source_sample_index=j+5,
                        role='gate_calibration' if j<721 else 'development',
                        full_history=[1,2],history=[1,2],target=3+j%5) for j in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'out'; source=root/'source'; src=source/'Toys'
            src.mkdir(parents=True); (out/'shared/Toys').mkdir(parents=True)
            np.save(src/'experts.npy',raw); np.save(out/'shared/Toys/donors.npy',np.arange(v))
            torch.save({'fit_users':[s['user_id'] for s in samples[:721]]},src/'gate.pt')
            states=[]
            with patch.object(app,'OUT',out), patch.object(app,'SOURCE',source), \
                 patch.object(app.runner,'transform_scores',app.core.transform_scores), \
                 patch.object(app.runner,'fuse',app.core.fuse):
                for arm in ('B','C'):
                    folder=out/arm/'Toys'; folder.mkdir(parents=True)
                    current=[dict(s) for s in samples]
                    if arm=='C':
                        for s in current[721:]: s['target']=7
                    app.write(folder/'samples.json',current); np.save(folder/'semantic.npy',replacement)
                    worker=app.SignalFit(arm,'Toys',lambda **kw: None); worker.device='cpu'
                    actual=worker.raw[np.array([5,725])]
                    np.testing.assert_array_equal(actual[:,:2],raw[[5,725],:2])
                    np.testing.assert_array_equal(actual[:,2],replacement[[0,720]])
                    worker.fit_signals()
                    self.assertEqual(worker.fused.shape,(n,v)); self.assertEqual(worker.fit_steps,12)
                    self.assertEqual(app.read(folder/'fit_checks.json')['state'],'PASSED')
                    saved=torch.load(folder/'gate.pt',map_location='cpu')
                    self.assertFalse(saved['development_labels_used']); states.append(saved['state_dict'])
                for key in states[0]: torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)
            np.testing.assert_array_equal(np.load(src/'experts.npy'),raw)


if __name__ == '__main__': unittest.main()
