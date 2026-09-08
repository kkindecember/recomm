"""Check preservation, long horizons, phase-aware selection and safe source admission."""

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

from experiment.phase18.core.full_budget import FullBudgetSchedule
from experiment.phase18.core.screen_budget import cohort_indices, optimizer_steps
from experiment.phase18.protocol.s18_screen_resume import Run, load_checkpoint
from experiment.phase18.protocol.s18_diffgrm_full_resume import train_original
from experiment.phase18.protocol.s18_full_budget_resume import validate_source


class FullBudgetTests(unittest.TestCase):
    def test_lr_bridge_preserves_adam_and_long_schedule_stays_positive(self):
        for start, horizon, warmup, peak, offset, bridge, decay, short_end in [
            (2033,21400,10000,[.003],0,107,'cosine',2782),
            (1710,9405,427,[.001,1e-5],855,855,'linear',4275),
            (3081,11297,513,[.001,1e-5],1027,1027,'linear',5135),
        ]:
            params = [torch.nn.Parameter(torch.ones(1)) for _ in peak]
            opt = torch.optim.AdamW([{'params':[p], 'lr':lr*.8} for p,lr in zip(params,peak)])
            for p in params:
                p.grad = torch.ones_like(p)
            opt.step()
            saved = copy.deepcopy(opt.state_dict())
            schedule = FullBudgetSchedule(opt,start,horizon,warmup,peak,offset,bridge,decay)
            self.assertEqual(schedule.get_last_lr(), [g['lr'] for g in saved['param_groups']])
            self.assertEqual(optimizer_steps(opt.state_dict()), optimizer_steps(saved))
            for key, state in saved['state'].items():
                for name in state:
                    torch.testing.assert_close(state[name],opt.state_dict()['state'][key][name],rtol=0,atol=0)
            self.assertEqual(schedule.at(start), schedule.get_last_lr())
            self.assertEqual(schedule.at(start+bridge),schedule.original_at(start+bridge))
            self.assertTrue(all(lr>0 for lr in schedule.at(short_end)))
            self.assertEqual(schedule.at(horizon),[0.]*len(peak))
            schedule.step()
            self.assertEqual(schedule.get_last_lr(),schedule.at(start+1))
            with self.assertRaises(ValueError):
                schedule.at(horizon+1)

    def test_source_admission_preserves_original_schema_and_rejects_wrong_or_exhausted_run(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter],lr=.01)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        source = dict(original_config={'seed':2023}, revision_config={'revision':'screen_r1'},
                      epoch=2, model={'w':parameter.detach()}, optimizer=optimizer.state_dict(),
                      cpu_rng=torch.get_rng_state(), cuda_rng=[])
        normalized=validate_source(copy.deepcopy(source),{'seed':2023},{'total_epochs':11})
        self.assertEqual(normalized['config'],source['original_config'])
        self.assertNotIn('config',source)
        for bad in [dict(source,original_config={'seed':1}),dict(source,epoch=11)]:
            with self.assertRaises(ValueError):
                validate_source(bad,{'seed':2023},{'total_epochs':11})
        exhausted=copy.deepcopy(source)
        exhausted['optimizer']['param_groups'][0]['lr']=0
        with self.assertRaises(ValueError):
            validate_source(exhausted,{'seed':2023},{'total_epochs':11})

    def test_joint_validation_offset_tail_updates_and_early_stop(self):
        model=torch.nn.Linear(1,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1,1)).square().mean().backward()
            optimizer.step()
        rows=[{'x':torch.tensor([float(i)]),'user_id':str(i)} for i in range(5)]
        backend=SimpleNamespace(model=model,optimizer=optimizer,training=rows,validation=rows,records=rows,
            config={'seed':2023},inherited_epoch=2,initial_steps=4,per_epoch=2,microbatch=2,effective=4,clip=1.)
        backend.loader=lambda ds,batch,generator=None:DataLoader(ds,batch_size=batch,shuffle=generator is not None,generator=generator)
        backend.loss=lambda batch:model(batch['x']).square().mean()
        backend.size=lambda batch:len(batch['x'])
        scheduler=FullBudgetSchedule(optimizer,4,22,1,[.001],2,2,'linear')
        observed=[]
        def evaluation(b,ds,run,stage_epoch,full):
            observed.append((stage_epoch+b.inherited_epoch,full))
            return dict(metrics={'ndcg@10':.3 if stage_epoch==0 else .1})
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            run=Run(dict(run_directory=directory,revision='full_r2',family='diff_gram',dataset='Toys',
                additional_epochs=9,total_epochs=11,minimum_epochs=5,patience=3,trend_users=2,
                trend_interval=2,evaluation_epoch_offset=1,completion_scope='full_budget_preserved_short_prefix'),'test')
            run.directory.mkdir()
            (run.root/'screen_r1').mkdir()
            indices=cohort_indices(rows,'Toys',2023,2)
            (run.root/'screen_r1/trend_cohort.json').write_text(json.dumps({'user_ids':[rows[i]['user_id'] for i in indices]}))
            with patch('experiment.phase18.protocol.s18_diffgrm_full_resume.matched_evaluate',side_effect=evaluation),patch('torch.cuda.get_rng_state_all',return_value=[]):
                train_original(backend,run,scheduler)
            result=json.loads((run.directory/'result.json').read_text())
            self.assertEqual(result['epochs_completed'],7)  # Joint epochs 2, 4, 6: three bad checks.
            self.assertEqual(result['selected_absolute_epoch'],2)
            self.assertEqual(observed,[(2,False),(3,False),(5,False),(7,False),(2,True)])
            last=load_checkpoint(run.directory/'last.pt')
            self.assertEqual(optimizer_steps(last['optimizer']),14)
            self.assertEqual(last['scheduler']['last_epoch'],14)
            self.assertEqual(result['completion_scope'],'full_budget_preserved_short_prefix')


if __name__ == '__main__':
    unittest.main()
