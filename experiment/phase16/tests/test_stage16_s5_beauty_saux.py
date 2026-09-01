from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiment.phase13.protocol.b1_portfolio_confirmation import portfolio_ranking
from experiment.phase16.protocol.stage16_s5_beauty_saux import (
    _exact_greater,
    _holm_single,
    _item_cluster_bootstrap,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiment/phase16/configs/stage16_s5_beauty_saux_gpu0_a1.json"
TRAIN_CONFIG = ROOT / "experiment/phase16/configs/stage16_s5_beauty_saux_train_gpu0_a1.json"
TOYS_CONFIG = ROOT / "experiment/phase16/configs/stage16_s4_toys_standalone_gpu4_a7.json"
INNER = ROOT / "experiment/phase16/run_stage16_s5_beauty_saux_gpu0_a1_inner.sh"


class Stage16S5BeautySauxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.train = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
        self.toys = json.loads(TOYS_CONFIG.read_text(encoding="utf-8"))

    def test_config_contract(self) -> None:
        validate_config(self.config)
        self.assertFalse(self.config["test_read"])
        self.assertFalse(self.config["automatic_retry"])
        self.assertEqual(self.config["physical_gpu"], 0)
        self.assertEqual(self.config["resources"]["minimum_free_mib"], 9216)

    def test_beauty_training_uses_toys_frozen_hyperparameters(self) -> None:
        toys_train = json.loads(
            (ROOT / "experiment/phase16/configs/stage16_s2_saux_formal_toys_a2.json").read_text(
                encoding="utf-8"
            )
        )
        for key in (
            "epochs",
            "early_stopping_patience_epochs",
            "evaluation_interval_epochs",
            "train_batch_size",
            "evaluation_batch_size",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "selection_metric",
            "maximum_history",
        ):
            self.assertEqual(self.train["training"][key], toys_train["training"][key])
        self.assertEqual(self.train["seed"], 1502)
        self.assertEqual(self.train["domain"], "Beauty_cold50")
        self.assertEqual(self.train["training"]["expected_train_transitions"], 33775)
        self.assertEqual(self.train["training"]["expected_pseudo_cold_events"], 9229)
        self.assertEqual(self.train["training"]["expected_pseudo_cold_items"], 1185)

    def test_beauty_inference_is_identical_to_toys_frozen_saux(self) -> None:
        frozen = self.toys["faithful_inference"]
        current = self.config["faithful_inference"]
        self.assertEqual(current["draft_size"], frozen["draft_size"]["S-AUX"])
        for key in (
            "threshold",
            "acceptance",
            "guided_redraft",
            "underfilled_live_round",
            "candidate_chunk_size",
            "fallback",
            "adaptive_exit",
        ):
            self.assertEqual(current[key], frozen[key])
        self.assertEqual(
            current["target_aware_prefix_length"],
            frozen["target_aware_prefix_length"]["S-AUX"],
        )

    def test_portfolio2_rule_inserts_at_ranks_nine_and_ten(self) -> None:
        gram = [f"g{index}" for index in range(50)]
        resolver = ["c1", "c2", "c3", *[f"r{index}" for index in range(47)]]
        ranking = portfolio_ranking(gram, resolver, ["c1", "c2", "c3"], 2)[:50]
        self.assertEqual(ranking[:8], gram[:8])
        self.assertEqual(ranking[8:10], ["c1", "c2"])
        self.assertEqual(len(ranking), 50)
        self.assertEqual(len(set(ranking)), 50)

    def test_exact_holm_and_item_bootstrap_are_finite(self) -> None:
        events = []
        for index in range(12):
            events.append(
                {
                    "is_cold": True,
                    "target_item": f"i{index // 2}",
                    "metrics": {
                        "S-AUX": {"hit@50": 1 if index < 9 else 0},
                        "F0": {"hit@50": 1 if index < 2 else 0},
                    },
                }
            )
        exact = _holm_single(_exact_greater(events, "S-AUX", "F0"), 0.05)
        self.assertEqual(exact["holm_family_size"], 1)
        self.assertGreaterEqual(exact["holm_adjusted_p_value"], 0.0)
        self.assertLessEqual(exact["holm_adjusted_p_value"], 1.0)
        diagnostic = _item_cluster_bootstrap(
            events, "S-AUX", "F0", resamples=100, seed=1502
        )
        self.assertEqual(diagnostic["unique_target_items"], 6)
        self.assertLessEqual(diagnostic["ci_low"], diagnostic["ci_high"])

    def test_runner_freezes_state_before_opening_validation(self) -> None:
        text = INNER.read_text(encoding="utf-8")
        train_at = text.index("saux_formal_train.py --config")
        freeze_at = text.index("freeze-comparators --config")
        validate_at = text.index("stage16_s5_beauty_saux validate --config")
        self.assertLess(train_at, freeze_at)
        self.assertLess(freeze_at, validate_at)
        self.assertIn('automatic_retry":false', text)
        self.assertNotIn("user_sequence.txt", text)

    def test_declared_inputs_exclude_test_and_original_sequence(self) -> None:
        for declaration in self.config["inputs"].values():
            name = Path(declaration["path"]).name.lower()
            self.assertNotIn("test", name)
            self.assertNotEqual(name, "user_sequence.txt")


if __name__ == "__main__":
    unittest.main()
