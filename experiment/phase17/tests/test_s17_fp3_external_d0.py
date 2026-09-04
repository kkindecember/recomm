from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment.phase17.core.full_setrec_backend import SETREC_ARMS
from experiment.phase17.core.full_setrec_external import (
    attention_contract_diagnostics,
    fp3_gate,
    mechanism_active,
    read_sealed_bundle_views,
    summarize_mechanisms,
)
from experiment.phase17.protocol.s17_fp3_external_d0_runtime import (
    paths,
    source_paths,
    verify_readiness,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


def comparison(ndcg: float, ci_low: float, hit: float) -> dict:
    return {
        "effects": {
            "ndcg@10": {
                "mean_delta": ndcg,
                "ci95_low": ci_low,
                "ci95_high": ndcg + 0.001,
            },
            "hit@10": {
                "mean_delta": hit,
                "ci95_low": hit - 0.001,
                "ci95_high": hit + 0.001,
            },
        }
    }


def active_mechanism() -> dict:
    return {
        "continuous_identifier_active": True,
        "full_catalog_grounding": True,
        "full_set_recovery_rate": 0.001,
        "valid_item_rate": 1.0,
        "query_norms_finite_nonzero": True,
        "semantic_reconstruction_finite": True,
        "attention_contract": {
            "contract_pass": True,
            "forbidden_visibility_count": 0,
        },
    }


class FP3ExternalD0Tests(unittest.TestCase):
    def test_config_is_complete_and_unauthorized(self) -> None:
        config = json.loads(
            (ROOT / "experiment/phase17/config/s17_fp3_external_d0.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(config["checkpoints"]), set(SETREC_ARMS))
        self.assertFalse(
            config["authorization"]["external_bundle_content_read_authorized"]
        )
        self.assertFalse(config["authorization"]["gpu_execution_authorized"])
        self.assertTrue(config["data"]["raw_external_projection_reopen_forbidden"])
        self.assertEqual(config["statistics"]["paired_bootstrap_replicates"], 2000)
        self.assertEqual(config["resources"]["preferred_gpu_count"], 1)
        self.assertTrue(config["resources"]["gpu1_forbidden_for_this_runner"])

    def test_preflight_never_reads_bundle_content(self) -> None:
        with patch(
            "experiment.phase17.protocol.s17_fp3_external_d0_runtime."
            "read_sealed_bundle_views",
            side_effect=AssertionError("preflight must not read sealed D0 content"),
        ):
            readiness = verify_readiness(ROOT)
        self.assertEqual(
            readiness["verdict"],
            "READY_AWAITING_EXPLICIT_GPU_AND_BUNDLE_READ_AUTHORIZATION",
        )
        self.assertFalse(readiness["external_bundle_content_read"])
        self.assertFalse(readiness["raw_external_projection_reopened"])
        self.assertEqual(set(readiness["checkpoint_evidence"]), set(SETREC_ARMS))

    def test_worker_command_is_offline_background_and_gpu_isolated(self) -> None:
        command = worker_command(ROOT, 7)
        self.assertIn("CUDA_VISIBLE_DEVICES=7", command)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("TRANSFORMERS_OFFLINE=1", command)
        self.assertIn(str(paths(ROOT)["snapshot_worker"]), command)
        self.assertIn("worker", command)

    def test_snapshot_sources_exist(self) -> None:
        for source in source_paths(ROOT):
            self.assertTrue(source.is_file(), source)

    def test_all_attention_contracts_are_exact(self) -> None:
        for arm_id in SETREC_ARMS:
            diagnostics = attention_contract_diagnostics(arm_id)
            self.assertTrue(diagnostics["contract_pass"])
            self.assertEqual(diagnostics["forbidden_visibility_count"], 0)
        self.assertTrue(
            attention_contract_diagnostics("S1P_SETREC_PAPER_FAITHFUL")[
                "sparse_history_active"
            ]
        )
        self.assertFalse(
            attention_contract_diagnostics("S0_SETREC_ORDERED_CONTROL")[
                "independent_query_active"
            ]
        )

    def test_sealed_bundle_reader_requires_aligned_nonempty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle.jsonl"
            bundle.write_text(
                json.dumps(
                    {
                        "user_id": "u1",
                        "train_items": ["a", "b"],
                        "history": ["a", "b"],
                        "target": "c",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            users, examples = read_sealed_bundle_views(bundle)
            self.assertEqual(users[0].train_items, ("a", "b"))
            self.assertEqual(examples[0].target, "c")

    def test_mechanism_summary_and_gate_fail_closed(self) -> None:
        rows = {
            "u1": {
                "latency_seconds": 0.1,
                "mechanism": {
                    "continuous_identifier_active": True,
                    "full_catalog_grounding": True,
                    "full_set_recovered": True,
                    "per_query_target_top1_recovered": [True] * 5,
                    "per_query_target_ranks": [1, 1, 1, 1, 1],
                    "combined_grounding_target_rank": 1,
                    "query_norms": [1.0] * 5,
                    "semantic_reconstruction_mse": 0.01,
                    "semantic_reconstruction_finite": True,
                    "valid_item_ranking": True,
                },
            }
        }
        contract = {
            "contract_pass": True,
            "forbidden_visibility_count": 0,
        }
        mechanism = summarize_mechanisms(rows, attention_contract=contract)
        self.assertTrue(mechanism_active(mechanism))
        comparisons = {
            "S1P_MINUS_S0": comparison(0.002, 0.001, 0.001),
            "S2_MINUS_S0": comparison(0.002, 0.001, 0.001),
            "S2_MINUS_G0": comparison(0.002, 0.001, 0.001),
        }
        subgroups = {
            "history_length": {"short_le3": {"users": 2, "delta_ndcg@10": 0.0}},
            "target_frequency": {"tail": {"users": 2, "delta_ndcg@10": 0.0}},
            "memory": {"generalization": {"users": 2, "delta_ndcg@10": 0.0}},
        }
        mechanisms = {
            "S1R_SETREC_REPO_PARITY": active_mechanism(),
            "S1P_SETREC_PAPER_FAITHFUL": active_mechanism(),
            "S2_GRAM_SETREC_PAPER_FULL": active_mechanism(),
        }
        passed = fp3_gate(
            comparisons, subgroups, mechanisms, integrity_valid=True
        )
        self.assertEqual(passed["verdict"], "FP3_STRONG_PASS")
        mechanisms["S2_GRAM_SETREC_PAPER_FULL"]["full_set_recovery_rate"] = 0.0
        failed = fp3_gate(
            comparisons, subgroups, mechanisms, integrity_valid=True
        )
        self.assertEqual(failed["verdict"], "FP3_NOT_STRONG_PASS")
        self.assertFalse(failed["checks"]["s2_mechanism_active"])


if __name__ == "__main__":
    unittest.main()
