"""Original LR continuation, absolute-epoch stopping, and validation isolation."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from experiment.phase18.core.screen_budget import cohort_indices, optimizer_steps
from experiment.phase18.protocol.s18_screen_resume import Run, load_checkpoint
from experiment.phase18.protocol.s18_diffgrm_full_resume import (
    matched_evaluate, restore_schedule, stopping_due, train_original,
)


class FullResumeTests(unittest.TestCase):
    def test_original_lr_and_adam_next_updates_equal_uninterrupted(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 20)
        x, y = torch.tensor([[1., 2.], [-2., 1.]]), torch.tensor([[.5], [-1.]])
        def step(m, opt, sched):
            opt.zero_grad(set_to_none=True)
            (m(x)-y).square().mean().backward()
            opt.step()
            sched.step()
        for _ in range(4):
            step(model, optimizer, scheduler)
        saved = copy.deepcopy(dict(epoch=4, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   scheduler=scheduler.state_dict()))
        restored = torch.nn.Linear(2, 1)
        restored.load_state_dict(saved['model'])
        resumed = torch.optim.AdamW(restored.parameters(), lr=9)
        resumed.load_state_dict(saved['optimizer'])
        schedule = restore_schedule(resumed, saved, dict(epochs=20, overrides={'warmup_steps':5}), 1)
        self.assertEqual(resumed.param_groups[0]['lr'], .008)
        for _ in range(4):  # Cross the warmup boundary into cosine decay.
            step(model, optimizer, scheduler)
            step(restored, resumed, schedule)
            self.assertEqual(schedule.get_last_lr(), scheduler.get_last_lr())
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(optimizer_steps(resumed.state_dict()), 8)
        with self.assertRaises(ValueError):
            restore_schedule(resumed, {k:v for k,v in saved.items() if k != 'scheduler'},
                             dict(epochs=20, overrides={'warmup_steps':5}), 1)

    def test_absolute_epoch_minimum_and_patience(self):
        self.assertFalse(stopping_due(90, 100, 8, 5))
        self.assertFalse(stopping_due(100, 100, 4, 5))
        self.assertTrue(stopping_due(100, 100, 5, 5))

    def test_monitoring_preserves_rng_and_compares_same_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = SimpleNamespace(inherited_epoch=33)
            references = {'gram': {'ndcg@10':.07}, 'gram_pcrf': {'ndcg@10':.08}}
            run = SimpleNamespace(directory=Path(directory), references={'matched_subset':references},
                                  context={}, event=lambda *a, **k:None)
            def evaluation(*args):
                torch.rand(15)
                return dict(scope='trend_subset', metrics={'ndcg@10':.05})
            before = torch.get_rng_state().clone()
            with patch('experiment.phase18.protocol.s18_diffgrm_full_resume.evaluate', side_effect=evaluation), \
                 patch('torch.cuda.get_rng_state_all', return_value=[]), patch('torch.cuda.set_rng_state_all'):
                result = matched_evaluate(backend, [], run, 7, False)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            self.assertEqual(result['epoch'], 40)
            self.assertAlmostEqual(result['delta']['ndcg@10'], -.02)
            self.assertAlmostEqual(result['delta_vs_gram_pcrf']['ndcg@10'], -.03)

    def test_train_tail_early_stop_and_best_start_model_selection(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 4, 16)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1, 1)).square().mean().backward()
            optimizer.step()
            scheduler.step()
        rows = [{'x':torch.tensor([float(i)]), 'y':torch.tensor([float(i)*.5]), 'user_id':str(i)}
                for i in range(5)]
        backend = SimpleNamespace(model=model, optimizer=optimizer, training=rows, validation=rows,
            records=rows, config={'seed':2023}, inherited_epoch=1, initial_steps=2,
            per_epoch=2, microbatch=2, effective=4, clip=1.)
        backend.loader = lambda ds, batch, generator=None: DataLoader(
            ds, batch_size=batch, shuffle=generator is not None, generator=generator)
        backend.loss = lambda batch: (model(batch['x'])-batch['y']).square().mean()
        backend.size = lambda batch:len(batch['x'])
        observed = []
        def evaluation(b, dataset, run, stage_epoch, full):
            observed.append((len(dataset), stage_epoch, full))
            return dict(metrics={'ndcg@10':.3 if stage_epoch==0 else .1})
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            run = Run(dict(run_directory=directory, revision='full_r2', dataset='Beauty', family='diffgrm',
                additional_epochs=7, total_epochs=8, trend_users=2, trend_interval=2,
                minimum_epochs=4, patience=2), 'test')
            run.directory.mkdir()
            previous = run.root/'screen_r1'
            previous.mkdir()
            indices = cohort_indices(rows, 'Beauty', 2023, 2)
            (previous/'trend_cohort.json').write_text(json.dumps({'user_ids':[rows[i]['user_id'] for i in indices]}))
            with patch('experiment.phase18.protocol.s18_diffgrm_full_resume.matched_evaluate', side_effect=evaluation), \
                 patch('torch.cuda.get_rng_state_all', return_value=[]):
                train_original(backend, run, scheduler)
            result = json.loads((run.directory/'result.json').read_text())
            self.assertEqual(result['epochs_completed'], 4)
            self.assertEqual(result['stopping_reason'], 'patience')
            self.assertEqual(result['selected_absolute_epoch'], 1)
            self.assertEqual(observed, [(2,0,False), (2,1,False), (2,3,False), (5,0,True)])
            last = load_checkpoint(run.directory/'last.pt')
            self.assertEqual(optimizer_steps(last['optimizer']), 8)
            self.assertEqual(last['scheduler']['last_epoch'], 8)
            self.assertEqual(json.loads((run.root/'status.json').read_text())['state'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
