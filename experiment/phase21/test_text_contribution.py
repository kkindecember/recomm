"""Scientific-control invariants, independent of training/evaluation outcomes."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party/LIME-Rec'))
from scripts.evaluation.run_repeat_aware_gate import RepeatAwareGate
from experiment.phase21.text_contribution_core import semantic_donors, transform_scores, fuse


class Controls(unittest.TestCase):
    def test_removed_semantic_information_has_no_path(self):
        rng = np.random.RandomState(7)
        raw = rng.rand(3, 3, 12).astype('float32')
        changed = raw.copy(); changed[:, 2] = rng.rand(3, 12) * 100
        history = torch.zeros(3, 12, dtype=torch.bool); history[:, :2] = True
        gate = RepeatAwareGate(14, [.6, .15, .25], .1)
        with torch.no_grad():
            gate.linear.weight.normal_()
        outputs = []
        for x in (raw, changed):
            scores = torch.from_numpy(transform_scores(x, 'without_text', np.arange(12)))
            result, features, weights, _ = fuse(gate, scores, torch.tensor([2., 4., 8.]), history, 'without_text')
            self.assertTrue(torch.equal(weights[:, 2], torch.zeros(3)))
            self.assertTrue(torch.equal(features[:, [3, 6, 9, 12]], torch.zeros(3, 4)))
            outputs.append(result)
        self.assertTrue(torch.equal(*outputs))

    def test_shuffle_preserves_scores_and_frequency_strata(self):
        ids = [f'item{i}' for i in range(12)]
        counts = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64])
        donors, audit = semantic_donors(ids, counts, 'Toys')
        donors2, _ = semantic_donors(ids, counts, 'Toys')
        np.testing.assert_array_equal(donors, donors2)
        self.assertEqual(audit['changed'], 10)
        raw = np.random.RandomState(8).rand(2, 3, 12).astype('float32')
        changed = transform_scores(raw, 'shuffled', donors)
        np.testing.assert_array_equal(changed[:, :2], raw[:, :2])
        np.testing.assert_array_equal(np.sort(changed[:, 2]), np.sort(raw[:, 2]))
        np.testing.assert_array_equal(changed[:, 2], raw[:, 2, donors])

    def test_real_fusion_matches_author_formula(self):
        from scripts.evaluation.run_repeat_aware_gate import features
        scores = torch.from_numpy(np.random.RandomState(9).rand(2, 3, 12).astype('float32'))
        history = torch.zeros(2, 12, dtype=torch.bool); history[:, 0] = True
        lengths = torch.tensor([1., 4.]); gate = RepeatAwareGate(14, [.6, .15, .25], .1)
        actual, _, _, _ = fuse(gate, scores, lengths, history, 'real')
        w, p = gate(features(scores, lengths, history))
        expected = (w[:, :, None] * scores).sum(1) - p[:, None] * history
        self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
