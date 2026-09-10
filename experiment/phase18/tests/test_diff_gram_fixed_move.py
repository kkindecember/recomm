"""An epoch-boundary device handoff must not restart the original training schedule."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiment.phase18.protocol import s18_diff_gram as original
from experiment.phase18.protocol.s18_diff_gram_fixed import FreshBackend,train_fixed
from experiment.phase18.protocol.s18_diff_gram_fixed_move import train_remaining,restore_joint,verify_boundary
from experiment.phase18.protocol.s18_diffgrm_full_resume import save_training
from experiment.phase18.protocol.s18_screen_resume import load_checkpoint
from experiment.phase18.tests.test_diff_gram import ROOT,batch,tiny_model
from experiment.phase18.tests.test_diff_gram_fixed import RecordingRun


def backend():
    config = json.loads((ROOT/'experiment/phase18/config/s18_diff_gram_beauty.json').read_text())
    config['training'].update(effective_batch_size=4,num_workers=0)
    b = FreshBackend.__new__(FreshBackend)
    b.old,b.family,b.config,b.device = original,'diff_gram',config,torch.device('cpu')
    b.model = tiny_model()
    b.optimizer = original.make_optimizer(b.model,config)
    b.initial_steps = b.inherited_epoch = b.workers = 0
    b.microbatch,b.effective,b.per_epoch,b.clip = 2,4,2,1.
    data = batch()
    b.training = [{k:v[i%2] for k,v in data.items()} for i in range(5)]
    b.validation = b.training
    def collate(rows):
        mapping = dict(input_ids='item_text_ids',attention_mask='item_text_masks',
            history_item_ids='history_item_ids',labels='target_ids',target_item_ids='target_item_ids')
        return {dest:torch.stack([row[key] for row in rows]) for key,dest in mapping.items()}
    b.collator = collate
    return b


class FixedMoveTests(unittest.TestCase):
    def test_completed_epoch_resume_matches_uninterrupted_weights_adam_rng_and_selection(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        full = backend()
        checks = []
        def validation(b,ds,run,epoch,is_full):
            checks.append((run.directory.name,epoch,is_full))
            return dict(epoch=epoch,metrics={'ndcg@10':1. if epoch==3 else .1})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root/'first';first.mkdir()
            resumed = root/'resumed';resumed.mkdir()
            run = RecordingRun(first)
            def capture(path,b,r,scheduler,epoch,steps,**extra):
                save_training(path,b,r,scheduler,epoch,steps,**extra)
                if path.name == 'last.pt' and epoch == 2:
                    shutil.copy2(path,root/'source.pt')
                    shutil.copy2(first/'best_trend.pt',resumed/'best_trend.pt')
            with patch('experiment.phase18.protocol.s18_diff_gram_fixed.matched_validation',side_effect=validation), \
                 patch('experiment.phase18.protocol.s18_diff_gram_fixed.save_training',side_effect=capture):
                train_fixed(full,run,full.validation)
            source = load_checkpoint(root/'source.pt')
            verify_boundary(source,run.spec,source['config'],per_epoch=2,cuda_rng_count=0)
            second = backend()
            with patch('experiment.phase18.protocol.s18_diff_gram_fixed_move.matched_validation',side_effect=validation):
                train_remaining(second,RecordingRun(resumed),second.validation,source)
            a,b = [load_checkpoint(p/'last.pt') for p in (first,resumed)]
            self.assertEqual(a['epoch'],b['epoch'])
            self.assertEqual(a['scheduler'],b['scheduler'])
            self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])
            for name,value in a['model'].items():
                torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            for index,state in a['optimizer']['state'].items():
                for key,value in state.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value,b['optimizer']['state'][index][key],rtol=0,atol=0)
                    else:self.assertEqual(value,b['optimizer']['state'][index][key])
            torch.testing.assert_close(a['cpu_rng'],b['cpu_rng'],rtol=0,atol=0)
            self.assertEqual(json.loads((resumed/'result.json').read_text())['selected_absolute_epoch'],3)
            self.assertEqual([(e,f) for r,e,f in checks if r=='resumed'],
                             [(3,False),(5,False),(7,False),(9,False),(11,False),(3,True)])
            invalid = copy.deepcopy(source)
            invalid['scheduler']['last_epoch'] += 1
            with self.assertRaises(ValueError):verify_boundary(invalid,run.spec,source['config'],per_epoch=2,cuda_rng_count=0)
            with self.assertRaises(ValueError):restore_joint(backend(),invalid)


if __name__ == '__main__':
    unittest.main()
