"""Fixed rerun must cover all data/phases and evaluate the selected saved weights."""

import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiment.phase18.protocol import s18_diff_gram as original
from experiment.phase18.protocol.s18_diff_gram_fixed import FreshBackend, preserve_training_rng, train_fixed
from experiment.phase18.tests.test_diff_gram import ROOT, batch, tiny_model


class RecordingRun:
    def __init__(self, directory):
        self.root = self.directory = directory
        self.spec = dict(total_epochs=11, revision='fixed_r1')
        self.events = []
        self.context = {}

    def event(self, event, **values):
        self.events.append(dict(event=event, **values))

    def status(self, state, **values):
        self.state = state


class FixedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_phases_tail_group_and_best_weight_selection(self):
        torch.manual_seed(23)
        config = json.loads((ROOT / 'experiment/phase18/config/s18_diff_gram_beauty.json').read_text())
        config['training'].update(effective_batch_size=4, num_workers=0)
        backend = FreshBackend.__new__(FreshBackend)
        backend.old, backend.family, backend.config = original, 'diff_gram', config
        backend.model, backend.device = tiny_model(), torch.device('cpu')
        backend.optimizer = original.make_optimizer(backend.model, config)
        backend.initial_steps = backend.inherited_epoch = backend.workers = 0
        backend.microbatch, backend.effective, backend.per_epoch, backend.clip = 2, 4, 2, 1.0
        data = batch()
        backend.training = [{k: v[i % 2] for k, v in data.items()} for i in range(5)]
        backend.validation = backend.training

        def collate(rows):
            mapping = dict(input_ids='item_text_ids', attention_mask='item_text_masks',
                history_item_ids='history_item_ids', labels='target_ids', target_item_ids='target_item_ids')
            return {dest: torch.stack([row[key] for row in rows]) for key, dest in mapping.items()}

        backend.collator = collate
        initial_base = backend.model.gram.shared.weight.detach().clone()
        initial_new = backend.model.memory_projection.weight.detach().clone()
        checked, best_weights = [], []
        def validation(candidate, dataset, run, epoch, full):
            checked.append((epoch, full))
            if epoch == 1:
                self.assertTrue(candidate.model.backbone_frozen)
                torch.testing.assert_close(candidate.model.gram.shared.weight, initial_base, rtol=0, atol=0)
                self.assertFalse(torch.equal(candidate.model.memory_projection.weight, initial_new))
            if epoch == 3 and not full:
                best_weights.append(candidate.model.memory_projection.weight.detach().clone())
                self.assertFalse(torch.equal(candidate.model.gram.shared.weight, initial_base))
            if full:
                torch.testing.assert_close(candidate.model.memory_projection.weight, best_weights[0], rtol=0, atol=0)
            scores = {1: .01, 3: .06, 5: .05, 7: .04, 9: .03, 11: .02}
            return dict(scope='full_validation' if full else 'trend_subset', epoch=epoch,
                        metrics={'ndcg@10': scores[epoch]})

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            run = RecordingRun(Path(temporary))
            with patch('experiment.phase18.protocol.s18_diff_gram_fixed.matched_validation', side_effect=validation):
                train_fixed(backend, run, backend.validation)
            result = json.loads((run.directory / 'result.json').read_text())
            saved = torch.load(run.directory / 'last.pt', map_location='cpu')
            self.assertEqual(result['epochs_completed'], 11)  # four falling checks cannot truncate the budget
            self.assertEqual(result['optimizer_steps'], 22)  # one complete + one tail group each epoch
            self.assertEqual(result['selected_absolute_epoch'], 3)
            self.assertEqual(saved['epoch'], 11)
            self.assertEqual(saved['scheduler']['last_epoch'], 20)  # joint-local schedule, not global 22
            self.assertEqual([g['lr'] for g in saved['optimizer']['param_groups']], [0.0, 0.0])
            self.assertEqual(checked, [(1, False), (3, False), (5, False), (7, False),
                                      (9, False), (11, False), (3, True)])
            epochs = [row for row in run.events if row['event'] == 'epoch_completed']
            self.assertEqual([row['train_examples_seen'] for row in epochs], [5] * 11)
            self.assertAlmostEqual(epochs[2]['learning_rates'][0], .001 * 16 / 19)
            self.assertEqual(run.state, 'COMPLETED')

    def test_validation_rng_preserves_python_numpy_and_torch(self):
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)
        with preserve_training_rng():
            expected = random.random(), np.random.rand(), torch.rand(5)
        with preserve_training_rng():
            random.random()
            np.random.rand(300)
            torch.rand(100)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(5), expected[2], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
