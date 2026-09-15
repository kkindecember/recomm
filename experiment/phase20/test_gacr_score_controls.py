import unittest
import numpy as np
import torch
from experiment.phase4.gacr_s0 import BoundedResidualRanker
from experiment.phase20.gacr_score_controls import residual_orders, reciprocal_scores, pcrf_ranks


class ScoreControlTest(unittest.TestCase):
    def test_zero_residual_recovers_generator_even_with_extra_candidates(self):
        row={'union_ids':list(range(1,66)), 'base':reciprocal_scores().tolist()+[0.]*15,
             'features':np.random.default_rng(8).normal(size=(65,6)).tolist()}
        ranker=BoundedResidualRanker(6,16,.2)
        for ids,scores in residual_orders(row,ranker):
            np.testing.assert_array_equal(ids,np.arange(1,51))
            np.testing.assert_array_equal(scores,reciprocal_scores())

    def test_fixed_candidate_control_never_admits_catalog_only_items(self):
        row={'union_ids':list(range(1,66)), 'base':reciprocal_scores().tolist()+[0.]*15,
             'features':[[float(i>=50)]+[0.]*5 for i in range(65)]}
        ranker=BoundedResidualRanker(6,16,.2)
        with torch.no_grad():
            for p in ranker.parameters(): p.zero_()
            ranker.network[0].weight[0,0]=10
            ranker.network[2].weight[0,0]=10
        full,fixed=residual_orders(row,ranker)
        self.assertTrue(any(i>50 for i in full[0]))
        self.assertEqual(set(fixed[0]),set(range(1,51)))

    def test_pcrf_affine_invariance_and_absent_target(self):
        cfg={'lambda':1.,'beta':.5,'gamma':1.,'q1':5}
        rng=np.random.default_rng(12)
        r={'candidate_ids':list(range(1,51)),'seq':rng.normal(size=50),'cf':rng.normal(size=50),'target_position':7}
        frequency=np.arange(51)
        base=pcrf_ranks([r],frequency,cfg)
        changed=pcrf_ranks([{**r,'seq':3*r['seq']+4}],frequency,cfg)
        np.testing.assert_array_equal(base,changed)
        self.assertEqual(pcrf_ranks([{**r,'target_position':-1}],frequency,cfg)[0],51)


if __name__=='__main__': unittest.main()
