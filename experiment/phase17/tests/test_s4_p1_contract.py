from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "GRAM/src"))
from model.gram import GRAM  # noqa: E402
from model.gram_t5_config import T5Config  # noqa: E402
from experiment.phase17.protocol.s4_targeted_p1_runtime import (  # noqa: E402
    EXPERIMENT_ID,
    build_command,
    prediction_rows,
    runtime_recovery_inputs,
)


P1_MODULES = [
    "P1_pawa_lite",
    "P1_treecl_lite",
    "P1_pctx_root",
    "P1_sethead",
    "P1_ls_fid",
    "P1_mhm",
    "P1_graphmae_prompt",
    "P1_dcrec_cl",
    "P1_sprint",
    "P1_biflow_s2g",
    "P1_biflow_g2s",
]


def tiny_config(module_id: str) -> T5Config:
    config = T5Config(
        vocab_size=73,
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    config.max_seq_len = 4
    config.max_item_num = 4
    config.use_position_embedding = False
    config.sample_num = 1
    config.cf0_enabled = False
    config.hi_gram_enabled = False
    config.s17_modules = module_id
    config.s17_transition_map = ""
    return config


class S4P1ContractTests(unittest.TestCase):
    def test_all_p1_cards_exist(self) -> None:
        card_dir = ROOT / "experiment/phase17/registry/migration_cards"
        cards = list(card_dir.glob("P1*.yaml"))
        text = "\n".join(card.read_text(encoding="utf-8") for card in cards)
        for module_id in P1_MODULES:
            self.assertIn(module_id, text)

    def test_every_p1_module_runs_in_tiny_gram(self) -> None:
        input_ids = torch.tensor(
            [
                [[2, 3, 0, 0], [4, 5, 6, 0], [7, 8, 0, 0], [9, 0, 0, 0]],
                [[2, 10, 0, 0], [11, 12, 0, 0], [13, 14, 0, 0], [0, 0, 0, 0]],
            ]
        )
        attention = input_ids.ne(0).long()
        history_ids = torch.tensor([[4, 3, 2], [5, 4, 0]])
        labels = torch.tensor([[15, 16, 1], [17, 18, 1]])
        for module_id in P1_MODULES:
            with self.subTest(module_id=module_id):
                torch.manual_seed(1704)
                model = GRAM(tiny_config(module_id)).train()
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    history_item_ids=history_ids,
                    history_item_mask=history_ids.ne(0),
                    target_item_ids=torch.tensor([5, 6]),
                    labels=labels,
                )
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                metrics = model.encoder.migration_runtime.mechanism_metrics()
                self.assertTrue(metrics)

    def test_formal_sethead_composes_with_latte(self) -> None:
        model = GRAM(tiny_config("B1_latte,P1_sethead"))
        self.assertEqual(
            model.encoder.migration_runtime.enabled_modules,
            ("B1_latte", "P1_sethead"),
        )
        self.assertEqual(model.encoder.migration_runtime.item_aggregation, "logsumexp")

    def test_formal_budget_is_four_arm_paired_d0_and_gpu1_only_after_science(self) -> None:
        budget = json.loads(
            (ROOT / "experiment/phase17/config/s17_s4_targeted_budget.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(budget["fold"], "D0")
        self.assertEqual(
            [arm["arm_id"] for arm in budget["formal_arms"]],
            ["gram_continue", "pawa_lite", "latte_sethead", "biflow_s2g"],
        )
        self.assertTrue(budget["comparison_contract"]["paired_uncertainty_required"])
        self.assertEqual(budget["resources"]["post_science_occupancy_gpu_ids"], [1])
        self.assertTrue(budget["smoke"]["exclude_physical_gpu_ids"] == [1])
        self.assertFalse(budget["test_read"])
        self.assertFalse(budget["sports_read"])

    def test_formal_command_saves_validation_predictions_and_loads_parent(self) -> None:
        arm = {"arm_id": "pawa_lite", "track_id": "P1-A", "module_id": "P1_pawa_lite"}
        command = build_command(
            ROOT,
            arm=arm,
            dataset="Toys_s17_d0_full",
            data_root=ROOT / "frozen_d0",
            output_dir=ROOT / "out",
            epochs=1,
            save_predictions=True,
            transition_teacher=ROOT / "teacher.json",
        )
        self.assertEqual(command[command.index("--save_predictions") + 1], "1")
        self.assertEqual(command[command.index("--train") + 1], "1")
        self.assertEqual(command[command.index("--test_epoch_rec") + 1], "0")
        self.assertIn("--rec_model_path", command)
        self.assertNotIn("--test_by_valid", command)

    def test_prediction_parser_ignores_metric_footer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.tsv"
            path.write_text(
                "idx\tH@5\tH@10\tNDCG@5\tNDCG@10\tgold\tpred\tscores\n"
                "u1\t1\t1\t1\t1\tg\tp\t0.5\n"
                "validation hit@5: 1.0\n",
                encoding="utf-8",
            )
            self.assertEqual(prediction_rows(path)["u1"]["ndcg@10"], 1.0)

    def test_runtime_recovery_resumes_after_persistent_failure_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts/phase17/s4_p1_targeted/run-0001"
            status_dir = root / "artifacts/phase17/status"
            output.mkdir(parents=True)
            status_dir.mkdir(parents=True)
            arms = [
                {"arm_id": arm_id}
                for arm_id in (
                    "gram_continue",
                    "pawa_lite",
                    "latte_sethead",
                    "biflow_s2g",
                )
            ]
            (output / "portfolio_config.json").write_text(
                json.dumps({"formal_arms": arms}), encoding="utf-8"
            )
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "formal_results": [
                            {"arm_id": arm["arm_id"], "state": "COMPLETED"}
                            for arm in arms
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (status_dir / f"{EXPERIMENT_ID}.status.json").write_text(
                json.dumps(
                    {
                        "scientific_state": "COMPLETED",
                        "process_alive": False,
                        "occupancy_mode": "stopped_after_runtime_failure",
                        "repeat_iteration": 8,
                    }
                ),
                encoding="utf-8",
            )
            _, results, iteration = runtime_recovery_inputs(root)
            self.assertEqual(iteration, 9)
            self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
