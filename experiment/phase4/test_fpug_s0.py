import unittest

import torch

from experiment.phase4.fpug_s0 import FinePassageGate, masked_passage_mean


class FPUGS0Tests(unittest.TestCase):
    def test_zero_init_identity_and_coarse(self):
        hidden = torch.randn(2, 12, 4)
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        gate = FinePassageGate(4, 0.5)
        output, gates = gate(hidden, mask)
        self.assertTrue(torch.equal(output, hidden))
        self.assertTrue(torch.equal(gates, torch.ones_like(gates)))
        self.assertTrue(torch.equal(output.view(2, 3, 4, 4)[:, 0], hidden.view(2, 3, 4, 4)[:, 0]))

    def test_gate_bounds(self):
        hidden = torch.randn(2, 12, 4)
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        gate = FinePassageGate(4, 0.5)
        with torch.no_grad():
            gate.linear.weight.fill_(10.0)
        _, values = gate(hidden, mask)
        self.assertTrue(bool((values >= 0.5).all()))
        self.assertTrue(bool((values <= 1.5).all()))

    def test_masked_mean_shape(self):
        hidden = torch.ones(2, 12, 4)
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        self.assertEqual(masked_passage_mean(hidden, mask).shape, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
