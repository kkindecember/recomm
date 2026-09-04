#!/usr/bin/env python3
"""Negative and identity tests for the frozen Stage18 S18-0 contracts."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from experiment.phase18.core.contracts import (
    ROOT,
    authorize_path,
    internal_fold_view,
    json_pointer,
    load_json,
    load_shadow_train_prefix_line,
    metrics_from_ranks,
)
from experiment.phase18.protocol.s18_s0_audit import audit_evidence


DATA_PATH = ROOT / "experiment/phase18/config/s18_data_contract.json"
EVIDENCE_PATH = ROOT / "experiment/phase18/config/s18_evidence_contract.json"


class DataDenyListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_json(DATA_PATH)

    def test_only_two_d0_shadow_files_enter_internal_runner(self) -> None:
        allowed = self.contract["access_profiles"]["s18_internal_runner"]["exact_allowlist"]
        self.assertEqual(len(allowed), 2)
        self.assertEqual({Path(path).parts[-3] for path in allowed}, {"Toys", "Beauty"})
        self.assertTrue(all("/D0/" in path for path in allowed))
        for path in allowed:
            self.assertEqual(authorize_path(path, "s18_internal_runner", self.contract), path)

    def test_d1_d2_official_sports_and_external_raw_fail_closed(self) -> None:
        for path in self.contract["representative_denied_paths"]:
            for profile in self.contract["access_profiles"]:
                with self.subTest(path=path, profile=profile):
                    with self.assertRaises(PermissionError):
                        authorize_path(path, profile, self.contract)

    def test_unknown_or_traversal_path_fails_closed(self) -> None:
        for path in ("artifacts/phase18/unknown.json", "../secret", "/absolute/path"):
            with self.subTest(path=path):
                with self.assertRaises(PermissionError):
                    authorize_path(path, "s18_internal_runner", self.contract)

    def test_shadow_loader_never_returns_external_target_or_guard(self) -> None:
        user, history = load_shadow_train_prefix_line(
            "u0 h0 h1 h2 SEALED_EXTERNAL_TARGET SEALED_GUARD\n"
        )
        self.assertEqual(user, "u0")
        self.assertEqual(history, ("h0", "h1", "h2"))
        self.assertNotIn("SEALED_EXTERNAL_TARGET", history)
        self.assertNotIn("SEALED_GUARD", history)

    def test_rolling_fold_offsets_are_exact(self) -> None:
        history = tuple(f"i{index}" for index in range(8))
        expected = {
            "I-1": (history[:-4], history[-4]),
            "I0": (history[:-3], history[-3]),
            "I1": (history[:-2], history[-2]),
            "I2": (history[:-1], history[-1]),
        }
        for fold, view in expected.items():
            self.assertEqual(internal_fold_view(history, fold), view)


class EvidenceAndNoRepeatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_json(DATA_PATH)
        cls.evidence = load_json(EVIDENCE_PATH)

    def test_all_historical_sources_are_explicitly_allowlisted(self) -> None:
        allowed = set(self.data["access_profiles"]["s18_s0_historical_audit"]["exact_allowlist"])
        sources = {row["path"] for row in self.evidence["sources"].values()}
        self.assertTrue(sources.issubset(allowed))
        reconstruction = self.evidence["pcrf_reconstruction"]
        self.assertIn(reconstruction["rank_cache_path"], allowed)
        self.assertIn(reconstruction["fresh_rank_cache_path"], allowed)

    def test_historical_claims_and_source_hashes_pass(self) -> None:
        result = audit_evidence(self.evidence, self.data)
        failures = [row for row in result["checks"] if not row["passed"]]
        self.assertEqual(failures, [])
        self.assertEqual(result["status"], "passed")

    def test_baseline_identity_is_native_lexical_pcrf(self) -> None:
        baseline = self.evidence["baseline_identity"]
        self.assertEqual(baseline["training_control"], "C0_CONT")
        self.assertEqual(baseline["inference_view"], "C1_CONT_PCRF")
        self.assertTrue(baseline["alpha_zero_must_equal_control"])
        self.assertEqual(baseline["beam_width"], 50)
        self.assertEqual(baseline["identifier"], "native_lexical")
        self.assertFalse(baseline["pcrf_parameter_search_allowed"])

    def test_all_twelve_hard_exclusions_are_frozen(self) -> None:
        exclusions = self.evidence["hard_exclusions"]
        self.assertEqual(len(exclusions), 12)
        required = {
            "replace_native_lexical_id",
            "posthoc_cf_only_or_beam200_admission_gate",
            "rerun_phase10_c1_or_c2",
            "rerun_stage17_a0_full_vocabulary_proxy",
            "automatic_scientific_retry",
        }
        self.assertTrue(required.issubset(exclusions))

    def test_json_pointer_supports_lists_and_metric_names(self) -> None:
        payload = {"rows": [{"delta": {"NDCG@10": 0.25}}]}
        self.assertEqual(json_pointer(payload, "/rows/0/delta/NDCG@10"), 0.25)


class RankMetricTests(unittest.TestCase):
    def test_rank_51_is_outside_beam_and_hit50(self) -> None:
        metrics = metrics_from_ranks([1, 10, 50, 51])
        self.assertEqual(metrics["count"], 4)
        self.assertEqual(metrics["Hit@1"], 0.25)
        self.assertEqual(metrics["Hit@10"], 0.5)
        self.assertEqual(metrics["Hit@50"], 0.75)
        self.assertTrue(math.isclose(metrics["mrr"], (1 + 0.1 + 0.02) / 4))

    def test_invalid_rank_cache_fails(self) -> None:
        with self.assertRaises(ValueError):
            metrics_from_ranks([0, 1])


if __name__ == "__main__":
    unittest.main()
