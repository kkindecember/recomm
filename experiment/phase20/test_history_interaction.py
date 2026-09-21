import inspect
import unittest

import numpy as np
import torch

from experiment.phase20.history_interaction_core import (
    HistoryRanker, align_text, history_inputs, masked_attention, global_group_weights, dense_pairwise_losses)
from experiment.phase20.gacr_fusion_core import pairwise_losses, standardize_tensor


class HistoryInteractionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.args = [torch.randn(3,50,6), torch.arange(1,51).expand(3,-1),
                     torch.tensor([[1,2,3,0],[4,5,0,0],[6,7,8,9]]),
                     torch.randn(3,4,2), torch.randn(70,12), torch.randn(70,8)]

    def model(self, arm, nonzero=False):
        m=HistoryRanker(arm,12,8,4)
        if nonzero:
            torch.nn.init.normal_(m.head[-1].weight,std=.2)
        return m

    def test_asin_alignment_not_row_order(self):
        actual=align_text(['','a','b'],['b','a'],torch.tensor([[2.,3.],[4.,5.]]))
        torch.testing.assert_close(actual,torch.tensor([[0.,0.],[4.,5.],[2.,3.]]))

    def test_identity_errors_rejected(self):
        for ids in [['a','a'],['a','c']]:
            with self.assertRaises(ValueError):
                align_text(['','a','b'],ids,torch.zeros(2,2))

    def test_history_visibility_and_recency(self):
        ids,a=history_inputs([[1,2],[3]],np.array([0,5,8,1]))
        np.testing.assert_array_equal(ids[0,:3],[1,2,0])
        self.assertAlmostEqual(float(a[0,0,0]),.05,places=6)
        self.assertEqual(float(a[0,1,0]),0.)
        for fn in [history_inputs,HistoryRanker.forward]:
            self.assertFalse(any('target' in k for k in inspect.signature(fn).parameters))

    def test_zero_output_preserves_base_exactly(self):
        base=torch.randn(3,50,dtype=torch.float64)
        for arm in ['B_scores','C_attention','D_joint']:
            delta=self.model(arm)(*self.args)
            self.assertTrue(torch.equal(base,base+delta.double()))

    def test_padding_does_not_contribute(self):
        for arm in ['C_attention','D_joint']:
            m=self.model(arm,True)
            expected=m(*self.args)
            changed=[x.clone() for x in self.args]
            changed[4][0]=1000; changed[5][0]=-1000
            changed[3][changed[2]==0]=1000
            torch.testing.assert_close(m(*changed),expected,rtol=1e-5,atol=1e-6)

    def test_empty_history_rejected(self):
        with self.assertRaises(ValueError):
            masked_attention(torch.randn(2,50,4),torch.zeros(2,4,dtype=torch.bool))

    def test_candidate_permutation_equivariance(self):
        perm=torch.randperm(50)
        for arm in ['B_scores','C_attention','D_joint']:
            m=self.model(arm,True)
            changed=[x.clone() for x in self.args]
            changed[0]=changed[0][:,perm]; changed[1]=changed[1][:,perm]
            torch.testing.assert_close(m(*changed),m(*self.args)[:,perm],rtol=1e-5,atol=1e-6)

    def test_joint_history_permutation_with_attributes(self):
        perm=torch.tensor([2,0,3,1])
        m=self.model('D_joint',True)
        changed=[x.clone() for x in self.args]
        changed[2]=changed[2][:,perm]; changed[3]=changed[3][:,perm]
        torch.testing.assert_close(m(*changed),m(*self.args),rtol=1e-5,atol=1e-6)

    def test_joint_attention_manual_reconstruction(self):
        m=self.model('D_joint',True)
        _,d=m(*self.args,return_details=True)
        for k in ['semantic_weights','cf_weights','joint_weights']:
            torch.testing.assert_close(d[k].sum(-1),torch.ones(3,50))
            self.assertEqual(float(d[k].masked_select((self.args[2]==0)[:,None,:]).abs().sum()),0.)
        qs=m.sq(self.args[4][self.args[1]]); ks=m.sk(self.args[4][self.args[2]])
        qc=m.cq(self.args[5][self.args[1]]); kc=m.ck(self.args[5][self.args[2]])
        bias=m.position(self.args[3])
        ls=qs@ks.transpose(1,2)/2+bias[:,None,:,0]
        lc=qc@kc.transpose(1,2)/2+bias[:,None,:,1]
        rel=torch.cat([torch.stack([ls,lc,ls-lc,ls*lc],-1),self.args[3][:,None].expand(-1,50,-1,-1)],-1)
        expected=torch.softmax(m.joint(rel).squeeze(-1).masked_fill((self.args[2]==0)[:,None,:],-torch.inf),-1)
        torch.testing.assert_close(expected,d['joint_weights'])

    def test_item_tables_frozen_and_all_branches_learn(self):
        for arm in ['C_attention','D_joint']:
            m=self.model(arm,True)
            scores=torch.randn(3,50,dtype=torch.float64)+m(*self.args).double()
            pairwise_losses(standardize_tensor(scores),torch.tensor([1,2,3])).mean().backward()
            for name in ['sq','sk','sv','cq','ck','cv']:
                grad=getattr(m,name).weight.grad
                self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum()>0)
            self.assertIsNone(self.args[4].grad); self.assertIsNone(self.args[5].grad)

    def test_matched_common_initialization(self):
        torch.manual_seed(100); c=HistoryRanker('C_attention')
        torch.manual_seed(100); d=HistoryRanker('D_joint')
        for k in set(c.state_dict()) & set(d.state_dict()):
            self.assertTrue(torch.equal(c.state_dict()[k],d.state_dict()[k]),k)
        self.assertGreaterEqual(sum(x.numel() for x in c.parameters()),sum(x.numel() for x in d.parameters()))

    def test_group_weight_and_pairwise_mean(self):
        head=np.array([True,False,False,False])
        w=global_group_weights(head)
        self.assertAlmostEqual(w[head].sum()/len(w),.5)
        self.assertAlmostEqual(w[~head].sum()/len(w),.5)
        score=torch.randn(4,50,dtype=torch.float64,requires_grad=True)
        p=torch.tensor([0,1,2,3]); loss=pairwise_losses(score,p)
        reference=torch.stack([torch.stack([torch.nn.functional.softplus(score[i,j]-score[i,p[i]])
                                          for j in range(50) if j != p[i]]).mean() for i in range(4)])
        torch.testing.assert_close(loss,reference)
        self.assertTrue(torch.isfinite(torch.autograd.grad((loss*torch.tensor(w)).mean(),score)[0]).all())

    def test_dense_loss_matches_legacy_values_and_gradients(self):
        positions = torch.arange(50)
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(50, 50, dtype=dtype, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            old = pairwise_losses(standardize_tensor(x), positions)
            new = dense_pairwise_losses(standardize_tensor(y), positions)
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            a = torch.autograd.grad(old.sum(), x)[0]
            b = torch.autograd.grad(new.sum(), y)[0]
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__=='__main__':
    unittest.main()
