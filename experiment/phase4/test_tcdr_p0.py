import unittest

from experiment.phase4.tcdr_p0 import cycle_pairs, mechanism_gate


class TCDRP0Tests(unittest.TestCase):
    def test_cycle_pairs_wraps(self):
        rows = [{"value": index} for index in range(3)]
        self.assertEqual(
            [row["value"] for row in cycle_pairs(rows, 2, 4)],
            [2, 0, 1, 2],
        )

    def test_mechanism_gate(self):
        controls = {
            "C0": {
                "mean_paired_excess": 0.5,
                "lexical_ce": 1.0,
                "mapping_rate": 1.0,
                "trie_membership_rate": 1.0,
                "finite_rate": 1.0,
            },
            "C1": {
                "mean_paired_excess": 0.4,
                "lexical_ce": 1.005,
                "mapping_rate": 1.0,
                "trie_membership_rate": 1.0,
                "finite_rate": 1.0,
            },
        }
        config = {
            "mechanism_gates": {
                "mean_paired_excess_relative_decrease_min": 0.1,
                "lexical_ce_relative_increase_max": 0.01,
                "mapping_rate": 1.0,
                "trie_membership_rate": 1.0,
                "finite_rate": 1.0,
            }
        }
        result = mechanism_gate("Toys", controls, config)
        self.assertTrue(result["pass"])


if __name__ == "__main__":
    unittest.main()

