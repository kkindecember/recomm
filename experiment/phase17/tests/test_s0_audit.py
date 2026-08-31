#!/usr/bin/env python3
"""Unit tests for the leakage-safe S17-0 audit utilities."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "experiment/phase17/protocol/s0_audit.py"
SPEC = importlib.util.spec_from_file_location("s0_audit", MODULE_PATH)
assert SPEC and SPEC.loader
s0_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s0_audit)


class ShadowFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="s17-s0-test-")
        self.base = Path(self.temporary.name)
        self.source = self.base / "user_sequence.txt"
        self.output = self.base / "shadow.txt"
        self.source.write_text("u0 i0 i1 i2 i3 i4 i5 i6\n", encoding="utf-8")
        self.catalog = {f"i{index}" for index in range(7)}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_d2_excludes_official_validation_and_test(self) -> None:
        manifest = s0_audit.build_shadow_fold(self.source, self.output, 3, self.catalog)
        fields = self.output.read_text(encoding="utf-8").split()
        self.assertEqual(fields, ["u0", "i0", "i1", "i2", "i3", "i4", "i0"])
        self.assertNotIn("i5", fields)
        self.assertNotIn("i6", fields)
        self.assertFalse(manifest["official_validation_position_serialized"])
        self.assertFalse(manifest["official_test_position_serialized"])
        self.assertFalse(manifest["target_in_train_by_position"])

    def test_d0_requires_pre_target_history(self) -> None:
        self.source.write_text(
            "short i0 i1 i2 i3 i4\nlong i0 i1 i2 i3 i4 i5\n",
            encoding="utf-8",
        )
        manifest = s0_audit.build_shadow_fold(self.source, self.output, 5, self.catalog)
        self.assertEqual(manifest["excluded_users_without_train_history"], 1)
        self.assertEqual(self.output.read_text(encoding="utf-8").split(), ["long", "i0", "i1", "i0"])

    def test_unknown_catalog_item_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown item"):
            s0_audit.build_shadow_fold(self.source, self.output, 3, {"i0"})


class LexicalContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="s17-s0-lex-")
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unique_fixed_length_paths_pass(self) -> None:
        path = self.base / "paths.txt"
        path.write_text("i0 |a|b\ni1 |a|c\n", encoding="utf-8")
        mapping, audit = s0_audit.parse_lexical_paths(path)
        self.assertEqual(mapping["i0"], ("a", "b"))
        self.assertEqual(audit["duplicate_path_count"], 0)
        self.assertFalse(audit["variable_length"])
        self.assertFalse(audit["eos_serialized_in_path"])

    def test_duplicate_path_fails_closed(self) -> None:
        path = self.base / "paths.txt"
        path.write_text("i0 |a|b\ni1 |a|b\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Lexical collision"):
            s0_audit.parse_lexical_paths(path)


class RegistryAndHistoryTests(unittest.TestCase):
    def test_all_p0_tracks_have_migration_cards(self) -> None:
        card_dir = ROOT / "experiment/phase17/registry/migration_cards"
        cards = [
            card_dir / name
            for name in (
                "A0_bear_gram.yaml",
                "A1_prefixcurr_gram.yaml",
                "B0_mvi_gram.yaml",
                "B1_latte_gram.yaml",
                "C0_biflow_gram.yaml",
                "D0_ted_gram.yaml",
                "E0_shortcut_fid_gram.yaml",
            )
        ]
        self.assertEqual(len(cards), 7)
        self.assertTrue(all(path.is_file() for path in cards))
        text = "\n".join(path.read_text(encoding="utf-8") for path in cards)
        for track in ("A0", "A1", "B0", "B1", "C0", "D0", "E0"):
            self.assertIn(f"track_id: {track}", text)

    def test_status_schema_forbids_test_and_sports_read(self) -> None:
        schema = json.loads((ROOT / "experiment/phase17/schemas/status.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["test_read"], {"const": False})
        self.assertEqual(schema["properties"]["sports_read"], {"const": False})

    def test_phase12_forensics_detects_unusable_baseline(self) -> None:
        result = s0_audit.audit_phase12()
        self.assertTrue(result["retrain_required_for_phase17_baseline"])
        contradictions = [item for run in result["runs"] for item in run["contradictions"]]
        self.assertIn("STATUS_TEST_READ_FALSE_BUT_TEST_EVIDENCE_EXISTS", contradictions)
        self.assertIn("STATUS_SUCCEEDED_BUT_PLANNED_EPOCHS_INCOMPLETE", contradictions)


if __name__ == "__main__":
    unittest.main()
