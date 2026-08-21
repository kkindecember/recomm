from __future__ import annotations

import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from r2pd_targets import (  # noqa: E402
    build_prefix_targets,
    compose_r2pd_loss,
    prefix_distillation_loss,
    retention_loss,
)


class TestPrefixTargets(unittest.TestCase):
    def test_absolute_mass_and_conditional_targets(self):
        paths = {"a": (1, 2, 3), "b": (1, 2, 4), "c": (5, 6)}
        targets, tail = build_prefix_targets(
            {"a": 0.6, "b": 0.3, "c": 0.05}, paths
        )
        by_prefix = {target.prefix: target for target in targets}
        self.assertAlmostEqual(tail, 0.05)
        self.assertAlmostEqual(by_prefix[()].mass, 0.95)
        self.assertEqual(by_prefix[()].children, (1, 5))
        self.assertAlmostEqual(by_prefix[()].probabilities[0], 0.9 / 0.95)
        self.assertAlmostEqual(by_prefix[(1, 2)].mass, 0.9)
        self.assertEqual(by_prefix[(1, 2)].children, (3, 4))
        self.assertAlmostEqual(by_prefix[(1, 2)].probabilities[0], 2 / 3)

    def test_min_mass_and_empty_effective_set(self):
        targets, _tail = build_prefix_targets(
            {"a": 0.01}, {"a": (1, 2)}, min_prefix_mass=0.02
        )
        self.assertEqual(targets, [])
        loss = prefix_distillation_loss([], [])
        self.assertEqual(float(loss), 0.0)

    def test_collision_hard_fails(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            build_prefix_targets({"a": 0.5, "b": 0.5}, {"a": (1, 2), "b": (1, 2)})


class TestMechanism(unittest.TestCase):
    def test_unseen_cold_path_log_probability_rises_at_every_prefix(self):
        targets, _tail = build_prefix_targets({"cold": 1.0}, {"cold": (1, 2, 3)})
        logits = [torch.nn.Parameter(torch.tensor([0.0, 0.0])) for _ in targets]
        optimizer = torch.optim.SGD(logits, lr=0.5)

        def target_log_probs() -> list[float]:
            return [float(torch.log_softmax(value, dim=-1)[0].detach()) for value in logits]

        before = target_log_probs()
        optimizer.zero_grad(set_to_none=True)
        # Each legal set contains the target child first and one competing child.
        expanded_targets = [
            type(target)(target.prefix, target.mass, (target.children[0], 99), (1.0, 0.0))
            for target in targets
        ]
        loss = prefix_distillation_loss(logits, expanded_targets)
        loss.backward()
        optimizer.step()
        after = target_log_probs()
        self.assertTrue(all(new > old for old, new in zip(before, after)))

    def test_zero_weights_recover_warm_ce_exactly(self):
        warm = torch.tensor(1.234567, requires_grad=True)
        kd = torch.tensor(9.0, requires_grad=True)
        keep = torch.tensor(7.0, requires_grad=True)
        total = compose_r2pd_loss(warm, kd, keep, lambda_cp=0.0, mu_keep=0.0)
        self.assertTrue(torch.equal(total, warm))

    def test_retention_teacher_is_stop_gradient(self):
        student = torch.tensor([[0.2, -0.1, 0.5]], requires_grad=True)
        teacher = torch.tensor([[0.4, 0.3, -0.2]], requires_grad=True)
        loss = retention_loss(student, teacher)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()
