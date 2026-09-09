"""Finish the LR tail without losing a historical best or stopping early again."""

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
from experiment.phase18.protocol.s18_diffgrm_full_resume import restore_schedule, train_original


class FinishTests(unittest.TestCase):
    def test_prior_best_survives_and_new_best_can_replace_it(self):
        for improves in (False, True):
            with self.subTest(improves=improves), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(2023)
                model = torch.nn.Linear(1, 1)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
                scheduler = get_cosine_schedule_with_warmup(optimizer, 2, 12)
                for _ in range(4):
                    optimizer.zero_grad(set_to_none=True)
                    model(torch.ones(1, 1)).square().mean().backward()
                    optimizer.step()
                    scheduler.step()
                rows = [{'x': torch.tensor([float(i)]), 'user_id': str(i)} for i in range(5)]
                backend = SimpleNamespace(model=model, optimizer=optimizer, training=rows, validation=rows,
                    records=rows, config={'seed': 2023}, inherited_epoch=2, initial_steps=4,
                    per_epoch=2, microbatch=2, effective=4, clip=1.)
                backend.loader = lambda ds, batch, generator=None: DataLoader(
                    ds, batch_size=batch, shuffle=generator is not None, generator=generator)
                backend.loss = lambda batch: model(batch['x']).square().mean()
                backend.size = lambda batch: len(batch['x'])
                run = Run(dict(run_directory=directory, revision='full_r3', dataset='Beauty', family='diffgrm',
                    additional_epochs=4, total_epochs=6, trend_users=2, trend_interval=2,
                    minimum_epochs=2, patience=1, disable_early_stopping=True), 'test')
                run.directory.mkdir()
                (run.root / 'screen_r1').mkdir()
                indices = cohort_indices(rows, 'Beauty', 2023, 2)
                (run.root / 'screen_r1/trend_cohort.json').write_text(json.dumps(
                    {'user_ids': [rows[i]['user_id'] for i in indices]}))
                prior_path = run.root / 'prior_best.pt'
                prior_weights = {k: torch.full_like(v, .75) for k, v in model.state_dict().items()}
                torch.save({'epoch': 1, 'model': prior_weights}, prior_path)
                prior = dict(epoch=1, score=.3, checkpoint=str(prior_path),
                    full_validation={'epoch': 1, 'metrics': {'ndcg@10': .25}},
                    full_summary_path='old/summary.json', full_predictions_path='old/predictions.jsonl', revision='full_r2')
                observed = []
                def evaluation(b, ds, r, stage_epoch, full):
                    observed.append((stage_epoch + b.inherited_epoch, full))
                    return dict(metrics={'ndcg@10': .4 if improves and stage_epoch == 2 else .1})
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch('experiment.phase18.protocol.s18_diffgrm_full_resume.matched_evaluate', side_effect=evaluation), \
                     patch('torch.cuda.get_rng_state_all', return_value=[]):
                    train_original(backend, run, scheduler, prior_best=prior)
                result = json.loads((run.directory / 'result.json').read_text())
                self.assertEqual(result['epochs_completed'], 6)
                self.assertEqual(result['stopping_reason'], 'epoch_limit')
                self.assertFalse(result['early_stopping_enabled'])
                self.assertEqual(result['full_validation_reused'], not improves)
                self.assertEqual(result['selected_absolute_epoch'], 4 if improves else 1)
                self.assertEqual(observed, [(2, False), (4, False), (6, False)] + ([(4, True)] if improves else []))
                last = load_checkpoint(run.directory / 'last.pt')
                self.assertEqual(optimizer_steps(last['optimizer']), 12)
                self.assertEqual(last['scheduler']['last_epoch'], 12)
                self.assertEqual(last['optimizer']['param_groups'][0]['lr'], 0.)
                selected = load_checkpoint(run.directory / 'best_trend.pt')
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, selected['model'][key], rtol=0, atol=0)
                if not improves:
                    self.assertEqual(result['full_validation']['source_revision'], 'full_r2')
                    for key in prior_weights:
                        torch.testing.assert_close(selected['model'][key], prior_weights[key], rtol=0, atol=0)

    def test_resumed_decay_tail_matches_uninterrupted_adam_to_zero_lr(self):
        torch.manual_seed(71)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 20)
        x = torch.tensor([[1., 2.], [-2., 1.]])
        def step(m, opt, sched):
            opt.zero_grad(set_to_none=True)
            m(x).square().mean().backward()
            opt.step()
            sched.step()
        for _ in range(16):
            step(model, optimizer, scheduler)
        saved = copy.deepcopy(dict(epoch=16, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   scheduler=scheduler.state_dict()))
        restored = torch.nn.Linear(2, 1)
        restored.load_state_dict(saved['model'])
        resumed = torch.optim.AdamW(restored.parameters(), lr=9.)
        schedule = restore_schedule(resumed, saved, dict(epochs=20, overrides={'warmup_steps': 5}), 1)
        self.assertEqual(schedule.get_last_lr(), scheduler.get_last_lr())
        for _ in range(4):
            step(model, optimizer, scheduler)
            step(restored, resumed, schedule)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(schedule.get_last_lr(), [0.])
        self.assertEqual(optimizer_steps(resumed.state_dict()), 20)


if __name__ == '__main__':
    unittest.main()
