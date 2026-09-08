"""Verify continuation, sampling boundaries, and stage-selection semantics."""

import copy
import io
import contextlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from experiment.phase18.core.screen_budget import BudgetSchedule, cohort_indices, optimizer_steps
from experiment.phase18.protocol.s18_screen_resume import Run, metric_result, train_stage, load_checkpoint


class ResumeContracts(unittest.TestCase):
    def test_cohort_invariant_to_order_and_labels(self):
        records = [{'user_id':str(i), 'target':i} for i in range(50)]
        other = [{'user_id':r['user_id'], 'target':999} for r in reversed(records)]
        chosen = lambda rows: [rows[i]['user_id'] for i in cohort_indices(rows, 'Toys', 2023, 20)]
        self.assertEqual(chosen(records), chosen(other))
        self.assertEqual(len(set(chosen(records))), 20)
        with self.assertRaises(ValueError):
            cohort_indices(records+records[:1], 'Toys', 2023, 20)

    def test_schedule_preserves_first_lr_and_has_finite_end(self):
        s = BudgetSchedule([.00012, 0], [.003, 1e-5], 500, 2140)
        self.assertEqual(s.at(0), [.00012, 0])
        self.assertEqual(s.at(500), [.003, 1e-5])
        self.assertEqual(s.at(2140), [0, 0])
        self.assertLess(s.at(1)[0]-.00012, 1e-5)
        self.assertLess(s.at(501)[0], s.at(500)[0])
        self.assertEqual(BudgetSchedule(**s.state_dict()).at(501), s.at(501))

    def test_adam_resume_matches_uninterrupted_next_update(self):
        torch.manual_seed(3)
        original = torch.nn.Linear(2,1)
        optimizer = torch.optim.AdamW(original.parameters(), lr=.001)
        x,y=torch.tensor([[1.,2.],[-1.,3.]]),torch.tensor([[.5],[2.]])
        def step(model, opt):
            opt.zero_grad(set_to_none=True)
            (model(x)-y).square().mean().backward()
            opt.step()
        for _ in range(3): step(original,optimizer)
        restored=torch.nn.Linear(2,1)
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        resumed=torch.optim.AdamW(restored.parameters(),lr=9)
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        schedule=BudgetSchedule([resumed.param_groups[0]['lr']],[.003],5,20)
        schedule.apply(resumed,0)
        step(original,optimizer)
        step(restored,resumed)
        for a,b in zip(original.parameters(),restored.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(optimizer_steps(resumed.state_dict()),4)
        for k,v in optimizer.state_dict()['state'].items():
            for name,tensor in v.items():
                torch.testing.assert_close(tensor,resumed.state_dict()['state'][k][name],rtol=0,atol=0)

    def test_subset_cannot_claim_delta_against_full_baseline(self):
        totals={'ndcg@10':1.,'hit@10':2.}
        self.assertIsNone(metric_result(totals,10,{'ndcg@10':.09},False)['delta'])
        self.assertAlmostEqual(metric_result(totals,10,{'ndcg@10':.09},True)['delta']['ndcg@10'],.01)

    def test_stage_preserves_tail_and_selects_before_full_validation(self):
        model=torch.nn.Linear(1,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        rows=[{'x':torch.tensor([float(i)]),'y':torch.tensor([float(i)*.5]),'user_id':str(i)} for i in range(5)]
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (model(torch.ones(1,1))-1).square().mean().backward();optimizer.step()
        backend=SimpleNamespace(model=model,optimizer=optimizer,training=rows,validation=rows,records=rows,
             config={'seed':2023},per_epoch=2,inherited_epoch=1,initial_steps=2,start_lrs=[.001],peak_lrs=[.002],
             family='diffgrm',microbatch=2,effective=4,clip=1.,baseline={'ndcg@10':.1})
        backend.loader=lambda dataset,batch_size,generator=None:DataLoader(dataset,batch_size=batch_size,shuffle=generator is not None,generator=generator)
        backend.loss=lambda b:(model(b['x'])-b['y']).square().mean()
        backend.size=lambda b:len(b['x'])
        seen=[]
        def evaluate(b,dataset,run,epoch,full):
            seen.append((len(dataset),epoch,full))
            return {'metrics':{'ndcg@10':.2 if epoch==1 else .1},'scope':'full_validation' if full else 'trend_subset'}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            spec=dict(run_directory=directory,revision='screen_r1',dataset='Toys',family='diffgrm',
                      additional_epochs=2,trend_users=2,trend_interval=1,warmup_steps=1)
            run=Run(spec,'test');run.directory.mkdir()
            with patch('experiment.phase18.protocol.s18_screen_resume.restore_probe'), patch('experiment.phase18.protocol.s18_screen_resume.evaluate',side_effect=evaluate),patch('torch.cuda.get_rng_state_all',return_value=[]):
                train_stage(backend,run,{})
            last=load_checkpoint(run.directory/'last.pt')
            self.assertEqual(last['optimizer_steps'],6)
            self.assertEqual(optimizer_steps(last['optimizer']),6)
            self.assertEqual(seen,[(2,1,False),(2,2,False),(5,1,True)])
            result=json.loads((run.root/'result.json').read_text())
            self.assertEqual(result['selected_stage_epoch'],1)
            self.assertFalse(result['direction_rejected'])
            self.assertEqual(json.loads((run.root/'status.json').read_text())['state'],'COMPLETED')


if __name__ == '__main__':
    unittest.main()
