"""Scientific invariants for S18-2, independent of GRAM data and GPU."""
import math
import unittest

import numpy as np
import torch

from experiment.phase18.core.pcps_loss import (
    combine_loss, full_path_loss, mine_prefix_nodes, path_weights, pcrf_scores, prefix_loss,
)
from experiment.phase18.core.s2_contracts import (
    ARMS, choose_alpha, mechanism_gate, prefix_survival, shuffled_teacher_map, training_view,
)


class S2Contracts(unittest.TestCase):
    def test_training_target_excludes_i0_and_sealed_suffix(self):
        self.assertEqual(training_view(["a", "b", "c", "I0_TARGET", "GUARD"]), (("a", "b"), "c"))
        with self.assertRaises(ValueError):
            training_view(["a", "I0_TARGET", "GUARD"])

    def test_shuffle_is_bijective_deterministic_no_fixed_points(self):
        users = [f"u{i}" for i in range(1000)]
        mapping = shuffled_teacher_map(users, "Toys")
        self.assertEqual(mapping, shuffled_teacher_map(users[::-1], "Toys"))
        self.assertEqual(set(mapping.values()), set(users))
        self.assertTrue(all(u != other for u, other in mapping.items()))

    def test_alpha_uses_weighted_ratio_and_common_domain_admissibility(self):
        self.assertEqual(choose_alpha({"Toys": 1.5, "Beauty": 1.4}), 0.1)
        self.assertEqual(choose_alpha({"Toys": 0.5, "Beauty": 0.6}), 0.3)
        self.assertIsNone(choose_alpha({"Toys": 4.0, "Beauty": 0.3}))
        with self.assertRaises(ValueError):
            choose_alpha({"Toys": math.nan})

    def test_survival_excludes_eos_and_does_not_equal_final_hit(self):
        target = (3, 4, 5, 1)
        active = {1: {(3,)}, 2: {(3, 4)}, 3: {(3, 8, 9)}}
        self.assertEqual(prefix_survival(active, target), 2 / 3)

    def test_generic_and_shuffle_are_necessary_mechanism_controls(self):
        gates = dict(survival_vs_c0_min=.02, hit50_vs_c0_min=.005,
                     survival_vs_s0_min=.01, ndcg10_max_drop=.001)
        c = dict(prefix_survival=.5, path_margin=-2., **{"Hit@50": .3, "NDCG@10": .1})
        m = dict(prefix_survival=.54, path_margin=-1.8, **{"Hit@50": .31, "NDCG@10": .1})
        arms = dict(zip(ARMS, [c, c, m, c]))
        self.assertEqual(mechanism_gate(arms, gates)["decision"], "MECHANISM_PASS")
        arms["A0_LEGAL_GENERIC"] = m
        self.assertEqual(mechanism_gate(arms, gates)["decision"], "CF_GUIDANCE_NOT_ADDITIVE")
        arms["A0_LEGAL_GENERIC"] = c
        arms["S0_SHUFFLED_CF"] = m
        self.assertFalse(mechanism_gate(arms, gates)["checks"]["survival_vs_s0"])


class PCPSLossContracts(unittest.TestCase):
    def test_alpha_zero_exact_value_gradient_and_no_cache_access(self):
        x = torch.tensor([1., -2.], requires_grad=True)
        ce = x.square().sum()
        actual, _ = combine_loss(ce, 0., cache=object())
        self.assertIs(actual, ce)
        actual.backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([2., -4.])))

    def test_prefix_gradients_raise_target_lower_only_legal_siblings(self):
        z = torch.zeros(2, 7, requires_grad=True)
        nodes = [{"depth": 0, "target_child": 2, "negative_children": [3, 4]}]
        loss = prefix_loss(z, nodes)
        self.assertAlmostEqual(float(loss), math.log(3), places=6)
        loss.backward()
        self.assertLess(float(z.grad[0, 2]), 0)
        self.assertTrue(bool((z.grad[0, [3, 4]] > 0).all()))
        self.assertEqual(float(z.grad[0, [0, 1, 5, 6]].abs().sum()), 0.)
        self.assertEqual(float(z.grad[1].abs().sum()), 0.)

    def test_complete_path_loss_has_correct_direction_and_stability(self):
        scores = torch.tensor([-1000., 1000., 0.], requires_grad=True)
        loss = full_path_loss(scores, [.75, .25])
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertLess(float(scores.grad[0]), 0.)
        self.assertGreater(float(scores.grad[1]), 0.)
        self.assertGreaterEqual(float(scores.grad[2]), 0.)

    def test_prefix_user_normalization_and_empty_fallback(self):
        z = torch.zeros(2, 8, requires_grad=True)
        node = {"depth": 0, "target_child": 2, "negative_children": [3]}
        self.assertEqual(float(prefix_loss(z, [node])), float(prefix_loss(z, [node, node])))
        self.assertEqual(float(prefix_loss(z, [])), 0.)
        self.assertEqual(float(full_path_loss(torch.tensor([-2.]), [])), 0.)

    def test_generic_mining_never_reads_cf_or_popularity(self):
        class Poison:
            def __getitem__(self, key):
                raise AssertionError("generic arm read CF/frequency")
        paths = {"t": (2, 5, 1), "a": (3, 5, 1), "b": (2, 6, 1), "c": (3, 6, 1)}
        nodes = mine_prefix_nodes(paths["t"], paths, ["b", "a"], [-1., -2.], None, Poison(), Poison())
        self.assertEqual(nodes[0]["negative_children"], [3])
        self.assertEqual(nodes[1]["negative_children"], [6])
        for node in nodes:
            self.assertNotIn(node["target_child"], node["negative_children"])

    def test_cf_guidance_changes_negative_selection_at_fixed_k(self):
        paths = {"t": (2, 1), "a": (3, 1), "b": (4, 1)}
        index = {"t": 1, "a": 2, "b": 3}
        args = (paths["t"], paths, ["a", "b"], [0., 0.])
        first = mine_prefix_nodes(*args, (np.array([0., 10., -10.]), 1.), np.ones(3), index, 1)
        second = mine_prefix_nodes(*args, (np.array([0., -10., 10.]), 1.), np.ones(3), index, 1)
        self.assertEqual(first[0]["negative_items"], ["a"])
        self.assertEqual(second[0]["negative_items"], ["b"])

    def test_cf_path_weights_preserve_parent_at_zero_reliability(self):
        generic = path_weights([-2., -3.])
        neutral = path_weights([-2., -3.], 1., [1000., -1000.], 0.)
        self.assertEqual(generic, neutral)
        guided = path_weights([-2., -3.], 1., [1000., -1000.], 1.)
        self.assertGreater(guided[0], generic[0])
        self.assertAlmostEqual(sum(guided), 1.)

    def test_pcrf_original_top10_reliability(self):
        joint, reliability = pcrf_scores(np.arange(50), np.arange(50), np.ones(50), 1)
        self.assertEqual(reliability, 0.)
        self.assertTrue(np.isfinite(joint).all())


if __name__ == "__main__":
    unittest.main()
