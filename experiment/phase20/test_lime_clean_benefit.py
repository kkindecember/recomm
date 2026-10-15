"""Regression checks for label isolation, masking, fit replay and benefit labels."""
import inspect
import unittest
import numpy as np
import torch
from experiment.phase20.lime_clean_benefit_core import (
    make_features,fold_ids,fit_ridge,predict_ridge,fit_head,predict_head,ranking,benefit_labels)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def data(self):
        rng=np.random.RandomState(3);n=20;k=50
        native=np.tile(np.linspace(0,-1,k),(n,1));mixed=native+rng.normal(0,.2,(n,k))
        return dict(ids=np.tile(np.arange(1,k+1),(n,1)),targets=rng.randint(1,15,n),
            native=native,mixed=mixed,valid=np.ones((n,k),bool),x=rng.normal(size=(n,k,10)).astype('float32'))

    def test_labels_include_harm_and_zero_and_absence(self):
        d=self.data();d['mixed']=d['native'].copy();d['targets'][:3]=[1,20,100]
        d['mixed'][0,0]=-100;d['mixed'][1,19]=100
        y=benefit_labels(d)
        self.assertLess(y[0],0);self.assertGreater(y[1],0);self.assertEqual(y[2],0)

    def test_history_mask_and_ties(self):
        d=self.data();d['targets'][:2]=[2,1];d['native'][0,:]=0;d['valid'][1,0]=False
        r=ranking(d,d['native']);self.assertEqual(r[0],2);self.assertGreater(r[1],50)

    def test_no_outcome_in_feature_api(self):
        self.assertNotIn('target',inspect.signature(make_features).parameters)
        x,q,v=make_features(np.arange(1,51),np.linspace(0,-1,50),np.linspace(0,-2,50),
            np.ones((3,60)),np.ones(60),np.arange(61),10,np.zeros(14),[1,2])
        self.assertEqual(x.shape,(50,10));self.assertEqual(q.shape,(24,));self.assertEqual(v.sum(),48)

    def test_unseen_targets_and_features_do_not_change_fit(self):
        d=self.data();ix=np.arange(12)
        a=fit_head(d,ix,'mlp',0)
        changed={k:v.copy() for k,v in d.items()};changed['targets'][12:]=1000;changed['x'][12:]=1000
        b=fit_head(changed,ix,'mlp',0)
        for k,v in a['state_dict'].items():torch.testing.assert_close(v,b['state_dict'][k],rtol=0,atol=0)
        np.testing.assert_array_equal(a['mean'],b['mean'])
        p=predict_head(a,d['x'][12:],d['mixed'][12:],chunk=3)
        q=predict_head(a,d['x'][12:],d['mixed'][12:],chunk=8)
        np.testing.assert_allclose(p,q,atol=2e-6,rtol=1e-6)

    def test_ridge_sign_predicts_signed_value(self):
        x=np.arange(-20,21,dtype=float)[:,None];y=x[:,0]/20
        a=fit_ridge(x,y);p=predict_ridge(a,np.array([[-5],[5.]]))
        self.assertLess(p[0],0);self.assertGreater(p[1],0)
        self.assertEqual(predict_ridge(fit_ridge(x,np.zeros(len(x))),x).sum(),0)

    def test_folds_follow_users_under_permutation(self):
        u=np.array([f'u{i}' for i in range(100)]);f=fold_ids(u,'Toys')
        np.testing.assert_array_equal(f[::-1],fold_ids(u[::-1],'Toys'));self.assertEqual(len(set(f)),5)


if __name__=='__main__':unittest.main()
