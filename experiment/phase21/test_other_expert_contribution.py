from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party/LIME-Rec'))
from scripts.evaluation.run_repeat_aware_gate import RepeatAwareGate
from experiment.phase21.other_expert_contribution_core import ARMS, REMOVED, transform_scores, fuse


class RemovalControls(unittest.TestCase):
    def check_removed_input(self, arm):
        removed = REMOVED[arm]; rng = np.random.RandomState(73)
        raw = rng.rand(4, 3, 15).astype('float32')
        changed = raw.copy(); changed[:, removed] = 100 * rng.rand(4, 15)
        history = torch.zeros(4, 15, dtype=torch.bool); history[:, :3] = True
        lengths = torch.tensor([3., 5., 7., 10.])
        gate = RepeatAwareGate(14, [.6, .15, .25], .1)
        with torch.no_grad(): gate.linear.weight.normal_()
        results = []
        for values in (raw, changed):
            transformed = transform_scores(values, arm)
            for channel in set(range(3)) - {removed}:
                np.testing.assert_array_equal(transformed[:, channel], raw[:, channel])
            score, x, weights, _ = fuse(gate, torch.from_numpy(transformed), lengths, history, arm)
            self.assertTrue(torch.equal(weights[:, removed], torch.zeros(4)))
            self.assertTrue(torch.equal(x[:, [1+removed, 4+removed, 7+removed, 10+removed]], torch.zeros(4, 4)))
            torch.testing.assert_close(weights.sum(1), torch.ones(4))
            results.append(score)
        self.assertTrue(torch.equal(*results))

    def test_remove_sasrec_completely(self): self.check_removed_input('without_sasrec')
    def test_remove_itemcf_completely(self): self.check_removed_input('without_itemcf')

    def test_remaining_experts_receive_training_gradient(self):
        scores = np.random.RandomState(74).rand(4, 3, 15).astype('float32')
        for arm in ARMS:
            gate = RepeatAwareGate(14, [.6, .15, .25], .1)
            result, _, _, _ = fuse(gate, torch.from_numpy(transform_scores(scores, arm)),
                                  torch.ones(4), torch.zeros(4, 15, dtype=torch.bool), arm)
            torch.nn.functional.cross_entropy(20 * result, torch.tensor([0, 3, 6, 9])).backward()
            self.assertTrue(torch.isfinite(gate.linear.weight.grad).all())
            for remaining in set(range(3)) - {REMOVED[arm]}:
                self.assertGreater(float(gate.linear.weight.grad[remaining].abs().sum()), 0.)


if __name__ == '__main__': unittest.main()
