"""Long fresh adaptation must retain a live LR after round 11 and run all 21 rounds."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiment.phase18.protocol import s18_diff_gram as original
from experiment.phase18.protocol.s18_diff_gram_fixed import FreshBackend, train_fixed
from experiment.phase18.protocol.s18_diff_gram_toys_long import validate_spec
from experiment.phase18.protocol.s18_screen_resume import load_checkpoint
from experiment.phase18.tests.test_diff_gram import ROOT, batch, tiny_model
from experiment.phase18.tests.test_diff_gram_fixed import RecordingRun


class ToysLongTests(unittest.TestCase):
    def test_contract_accepts_only_fresh_toys_and_the_declared_budget(self):
        spec = json.loads((ROOT/'experiment/phase18/config/s18_diff_gram_toys_long_fixed.json').read_text())
        config = validate_spec(spec)
        self.assertEqual(config['training']['joint_epochs'], 20)
        for field, value in [('dataset','Beauty'), ('total_epochs',11), ('additional_epochs',10), ('microbatch',8)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_spec(dict(spec, **{field:value}))

    def test_twenty_joint_epochs_live_midpoint_lr_and_selected_weights(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        spec = json.loads((ROOT/'experiment/phase18/config/s18_diff_gram_toys_long_fixed.json').read_text())
        config = copy.deepcopy(validate_spec(spec))
        config['training'].update(effective_batch_size=4,num_workers=0)
        backend = FreshBackend.__new__(FreshBackend)
        backend.old, backend.family, backend.config = original, 'diff_gram', config
        backend.model, backend.device = tiny_model(), torch.device('cpu')
        backend.optimizer = original.make_optimizer(backend.model, config)
        backend.initial_steps = backend.inherited_epoch = backend.workers = 0
        backend.microbatch, backend.effective, backend.per_epoch, backend.clip = 2,4,2,1.
        data = batch()
        backend.training = [{k:v[i%2] for k,v in data.items()} for i in range(5)]
        backend.validation = backend.training
        def collate(rows):
            mapping = dict(input_ids='item_text_ids',attention_mask='item_text_masks',
                history_item_ids='history_item_ids',labels='target_ids',target_item_ids='target_item_ids')
            return {dest:torch.stack([row[key] for row in rows]) for key,dest in mapping.items()}
        backend.collator = collate
        checked, selected = [], []
        initial = backend.model.gram.shared.weight.detach().clone()
        def validation(b, ds, run, epoch, full):
            checked.append((epoch,full))
            if epoch == 1:
                torch.testing.assert_close(b.model.gram.shared.weight,initial,rtol=0,atol=0)
            if epoch == 11:
                self.assertAlmostEqual(b.optimizer.param_groups[0]['lr'], .001*20/38)
            if epoch == 15 and not full:
                selected.append(b.model.memory_projection.weight.detach().clone())
            if full:
                torch.testing.assert_close(b.model.memory_projection.weight,selected[0],rtol=0,atol=0)
            return dict(metrics={'ndcg@10':1. if epoch==15 else .1},epoch=epoch)
        with tempfile.TemporaryDirectory() as directory:
            run = RecordingRun(Path(directory))
            run.spec.update(total_epochs=21,revision='long_fixed_r1')
            with patch('experiment.phase18.protocol.s18_diff_gram_fixed.matched_validation',side_effect=validation):
                train_fixed(backend,run,backend.validation)
            result = json.loads((run.directory/'result.json').read_text())
            self.assertEqual(result['epochs_completed'],21)
            self.assertEqual(result['optimizer_steps'],42)
            self.assertEqual(result['selected_absolute_epoch'],15)
            self.assertEqual(checked,[(e,False) for e in range(1,22,2)]+[(15,True)])
            saved = load_checkpoint(run.directory/'last.pt')
            self.assertEqual(saved['scheduler']['last_epoch'],40)
            self.assertEqual([g['lr'] for g in saved['optimizer']['param_groups']],[0.,0.])
            self.assertEqual([e['train_examples_seen'] for e in run.events if e['event']=='epoch_completed'],[5]*21)


if __name__ == '__main__':
    unittest.main()
