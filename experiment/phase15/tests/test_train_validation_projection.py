from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from build_train_validation_projection import run_projection  # noqa: E402


class TestTrainValidationProjection(unittest.TestCase):
    def _write_config(self, root: Path, source_text: str) -> Path:
        source = root / "data" / "user_sequence.txt"
        source.parent.mkdir(parents=True)
        source.write_text(source_text, encoding="utf-8")
        config = {
            "schema_version": 1,
            "experiment_id": "TEST_PROJECTION",
            "output_root": "out",
            "audit_file": "projection_audit.json",
            "domains": [
                {
                    "name": "fixture",
                    "source": "data/user_sequence.txt",
                    "output": "fixture/user_sequence_train_validation.txt",
                }
            ],
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_drops_only_final_item_and_does_not_report_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(
                root,
                "u1 a b VALIDATION_1 SECRET_TEST_1\n"
                "u2 c VALIDATION_2 SECRET_TEST_2\n",
            )
            payload = run_projection(config, root)
            output = root / "out/fixture/user_sequence_train_validation.txt"
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "u1 a b VALIDATION_1\nu2 c VALIDATION_2\n",
            )
            serialized_audit = json.dumps(payload)
            self.assertNotIn("SECRET_TEST_1", serialized_audit)
            self.assertNotIn("SECRET_TEST_2", serialized_audit)
            self.assertFalse(payload["test_target_materialized"])
            self.assertFalse(payload["test_target_used"])
            self.assertTrue(
                all(not row["test_target_retained"] for row in payload["domains"])
            )

    def test_rejects_sequence_without_train_validation_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(root, "u1 train validation\n")
            with self.assertRaisesRegex(ValueError, "expected user id"):
                run_projection(config, root)

    def test_rejects_duplicate_users(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(
                root,
                "u1 a validation test\nu1 b validation test\n",
            )
            with self.assertRaisesRegex(ValueError, "duplicate user id"):
                run_projection(config, root)

    def test_refuses_to_overwrite_existing_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(root, "u1 a validation test\n")
            run_projection(config, root)
            with self.assertRaises(FileExistsError):
                run_projection(config, root)

    def test_rejects_output_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_config(root, "u1 a validation test\n")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["domains"][0]["output"] = "../escaped.txt"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes output root"):
                run_projection(config_path, root)


if __name__ == "__main__":
    unittest.main()
