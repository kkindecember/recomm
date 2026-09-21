"""Recovery and admission tests for expensive clean-upstream training."""
import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiment.phase20.lime_clean_gram_core import (
    epoch_order, update_windows, seed_all, pack_state, load_state, atomic_save, cpu_tree,
)
from experiment.phase20.lime_clean_gram_job import eligible_gpus, profile_passed


class CleanGramTests(unittest.TestCase):
    def test_partial_windows_cover_each_row_once_and_resume_without_duplicates(self):
        order = epoch_order(35, 2023, 1)
        windows = list(update_windows(order, 4, 3))
        self.assertEqual([list(map(len, w)) for w in windows], [[4, 4, 4], [4, 4, 4], [4, 4, 3]])
        completed = [x for w in windows[:2] for b in w for x in b]
        resumed = [x for w in list(update_windows(epoch_order(35, 2023, 1), 4, 3))[2:] for b in w for x in b]
        self.assertEqual(completed+resumed, order)
        self.assertEqual(sorted(completed+resumed), list(range(35)))
        random.random()
        self.assertEqual(order, epoch_order(35, 2023, 1))
        self.assertNotEqual(order, epoch_order(35, 2023, 2))

    def test_dropout_optimizer_scheduler_and_rng_resume_exactly(self):
        seed_all(123)
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.4), torch.nn.Linear(5, 1))
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1./(n+1))
        config, manifest = dict(seed=123), dict(input='fixed')

        def step():
            x = torch.rand(7, 3) + float(np.random.rand()) + random.random()
            loss = model(x).square().mean()
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            return loss.item()

        step()
        state = pack_state(model, opt, scheduler, dict(epoch=1, window=1, updates=1), manifest, 'gram', config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            atomic_save(path, state)
            expected_losses = [step(), step()]
            expected = cpu_tree(model.state_dict())
            expected_opt = cpu_tree(opt.state_dict())
            expected_scheduler = scheduler.state_dict()
            cp = torch.load(path, map_location='cpu')
            load_state(cp, model, opt, scheduler, manifest, 'gram', config)
            self.assertEqual([step(), step()], expected_losses)
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(expected[key], tensor))
            for key, values in expected_opt['state'].items():
                for name, value in values.items():
                    actual = opt.state_dict()['state'][key][name]
                    self.assertTrue(torch.equal(value, actual) if torch.is_tensor(value) else value == actual)
            self.assertEqual(scheduler.state_dict(), expected_scheduler)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])
            for altered, stage, changed_config in [(dict(input='changed'), 'gram', config),
                                                    (manifest, 'c1', config), (manifest, 'gram', dict(seed=124))]:
                with self.assertRaises(AssertionError):
                    load_state(cp, model, opt, scheduler, altered, stage, changed_config)

    def test_no_checkpoint_can_be_written_mid_accumulation(self):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
        model(torch.ones(1, 2)).sum().backward()
        with self.assertRaises(AssertionError):
            pack_state(model, opt, scheduler, {}, {}, 'gram', {})

    def test_gpu_admission_excludes_busy_small_and_wrong_profile_device(self):
        gpus = [dict(uuid='a', free_mib=40000), dict(uuid='b', free_mib=30720), dict(uuid='c', free_mib=30719)]
        self.assertEqual([g['uuid'] for g in eligible_gpus(gpus, {'a'}, 30720)], ['b'])
        self.assertEqual(eligible_gpus(gpus, set(), 30720, 'c'), [])
        self.assertEqual([g['uuid'] for g in eligible_gpus(gpus, set(), 30720, 'b')], ['b'])

    def test_failed_or_changed_profile_never_admits_training(self):
        manifest = dict(source='a')
        summary = dict(state='PASSED', input_sha256=manifest, profile_weights_discarded=True)
        self.assertTrue(profile_passed(summary, manifest))
        for key, value in [('state', 'COST_GATE_FAILED'), ('input_sha256', dict(source='b')), ('profile_weights_discarded', False)]:
            changed = copy.deepcopy(summary)
            changed[key] = value
            self.assertFalse(profile_passed(changed, manifest))


if __name__ == '__main__':
    unittest.main()
