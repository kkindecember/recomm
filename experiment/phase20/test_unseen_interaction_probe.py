import inspect
import unittest
import numpy as np
import torch
from experiment.phase20.unseen_interaction_probe import folds, outer_split, train_model, count_ranks
from experiment.phase20.conditional_reference_core import Correction, ranks


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_fold_assignment_independent_of_row_order_and_labels(self):
        users=np.asarray([f'u{i}' for i in range(40)])
        perm=np.arange(40)[::-1]
        np.testing.assert_array_equal(folds(users,9,5)[perm],folds(users[perm],9,5))
        self.assertEqual(list(inspect.signature(folds).parameters),['users','seed','count'])

    def test_same_cohort_and_no_outer_user_in_training(self):
        users=np.asarray([f'u{i}' for i in range(40)]); cohort=set(users[:12]); fs=folds(users,9,5)
        visits=np.zeros(len(users),int)
        for f in range(5):
            ti,ev=outer_split(users,cohort,fs,f)
            self.assertEqual(set(users[ti]),{u for u,k in zip(users,fs) if u in cohort and k!=f})
            self.assertTrue(set(users[ti]).isdisjoint(users[ev]));visits[ev]+=1
        np.testing.assert_array_equal(visits,np.ones(len(users),int))

    def test_outer_labels_cannot_change_fitted_weights(self):
        rng=np.random.default_rng(17); x=rng.normal(size=(12,50,6)).astype('float32'); b=rng.normal(size=(12,50))
        p=np.arange(12); config=dict(seed=91,bound=.5,learning_rate=.01,weight_decay=.01,steps=2,chunk_size=3,clip=5.)
        train=np.arange(8); m,_=train_model(x[train],b[train],p[train],config)
        changed=p.copy();changed[8:]=49
        other,_=train_model(x[train],b[train],changed[train],config)
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,other.state_dict()[k]))
        changed[0]=49; modified,_=train_model(x[train],b[train],changed[train],config)
        self.assertTrue(any(not torch.equal(v,modified.state_dict()[k]) for k,v in m.state_dict().items()))

    def test_independent_rank_ties_and_uncovered(self):
        s=np.zeros((3,50));s[1,2]=1;s[1,3]=1;p=np.array([4,3,-1])
        np.testing.assert_array_equal(count_ranks(s,p),[5,2,51])
        np.testing.assert_array_equal(count_ranks(s,p),ranks(s,p))

    def test_zero_start_and_uncovered_rows_not_inserted(self):
        m=Correction(.5);self.assertEqual(sum(p.numel() for p in m.parameters()),129)
        self.assertFalse(m(torch.randn(2,50,6)).count_nonzero())
        cfg=dict(seed=4,bound=.5,learning_rate=.01,weight_decay=.01,steps=1,chunk_size=2,clip=5.)
        x=np.zeros((2,50,6),np.float32); b=np.zeros((2,50),np.float64)
        with self.assertRaises(AssertionError):train_model(x,b,np.array([-1,-1]),cfg)


if __name__=='__main__':unittest.main()
