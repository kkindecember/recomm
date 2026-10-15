"""Scientific interface checks for the frozen-backbone attribute residual."""
import unittest
import torch
from experiment.phase21.attribute_memory_core import AttributeMemoryResidual


class MemoryInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2023)
        self.branch = AttributeMemoryResidual(16, 4)
        self.head = torch.nn.Linear(16, 31, bias=False).requires_grad_(False)
        self.h = torch.randn(2, 3, 16)
        self.memory = torch.randn(2, 5, 16)
        self.valid = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        self.branch.set_memory(self.memory, self.valid)

    def test_zero_output_exact_and_branch_can_learn(self):
        expected = self.head(self.h)
        handle = self.branch.attach(self.head)
        actual = self.head(self.h)
        self.assertTrue(torch.equal(actual, expected))
        actual.square().mean().backward()
        self.assertGreater(float(self.branch.output.weight.grad.abs().sum()), 0)
        self.assertIsNone(self.head.weight.grad)
        with torch.no_grad():
            self.branch.output.weight.add_(-.1*self.branch.output.weight.grad)
        self.assertFalse(torch.equal(self.head(self.h), expected))
        handle.remove()
        self.assertTrue(torch.equal(self.head(self.h), expected))

    def test_masks_and_empty_history_remain_neutral(self):
        torch.nn.init.normal_(self.branch.output.weight)
        first = self.branch(self.h)
        other = self.memory.clone()
        other[~self.valid] = 9999
        self.branch.set_memory(other, self.valid)
        self.assertTrue(torch.equal(first, self.branch(self.h)))
        self.branch.set_memory(other, torch.zeros_like(self.valid))
        self.assertTrue(torch.equal(self.h, self.branch(self.h)))

    def test_beam_expansion_keeps_request_alignment(self):
        torch.nn.init.normal_(self.branch.output.weight)
        single = self.branch(self.h)
        beam = self.branch(self.h.repeat_interleave(3, dim=0))
        torch.testing.assert_close(beam, single.repeat_interleave(3, dim=0), atol=1e-6, rtol=1e-6)

    def test_only_history_memory_controls_update(self):
        torch.nn.init.normal_(self.branch.output.weight)
        first = self.branch(self.h)
        self.branch.set_memory(self.memory.flip(0), self.valid)
        self.assertFalse(torch.equal(first, self.branch(self.h)))
        self.branch.clear_memory()
        with self.assertRaises(RuntimeError):
            self.branch(self.h)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
