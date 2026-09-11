import unittest
import torch

from experiment.phase20.gacr_features import vector_features


class VectorFeatureTest(unittest.TestCase):
    def test_matches_original_scalar_formula_with_masked_and_missing_candidates(self):
        torch.manual_seed(3)
        logits = torch.randn(30)
        logits[[2,5]] = -torch.inf
        pooled = torch.randn(12)
        weights = torch.randn(30,12)
        weights[5].zero_()
        indices = torch.tensor([2,5,7,11,12,19])
        gram, catalog = [1,0,50,0,3,11], [0,2,50,4,0,9]
        finite = logits[torch.isfinite(logits)]
        mean,std = finite.mean(),finite.std(unbiased=False).clamp_min(1e-6)
        rows = []
        for index,gr,cr in zip(indices,gram,catalog):
            logit,weight = logits[index],weights[index]
            z = ((logit-mean)/std).clamp(-10,10) if torch.isfinite(logit) else torch.tensor(-10.)
            cosine = torch.dot(pooled,weight)/(torch.linalg.vector_norm(pooled).clamp_min(1e-6)*torch.linalg.vector_norm(weight).clamp_min(1e-6))
            rows.append(torch.stack([z,torch.tensor(1/gr if gr else 0.),torch.tensor(1/cr if cr else 0.),
                                    torch.tensor(float(gr>0)),torch.tensor(float(cr>0)),cosine]))
        torch.testing.assert_close(vector_features(logits,pooled,weights,indices,gram,catalog),torch.stack(rows),atol=1e-6,rtol=0)

    def test_constant_logits_and_zero_vectors_are_finite(self):
        result = vector_features(torch.ones(3),torch.zeros(4),torch.zeros(3,4),torch.tensor([0,1]),[1,0],[0,1])
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result,torch.tensor([[0.,1.,0.,1.,0.,0.],[0.,0.,1.,0.,1.,0.]]),atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
