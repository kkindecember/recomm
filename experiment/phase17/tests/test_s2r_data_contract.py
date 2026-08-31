from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment.phase17.protocol import s2r_data_contract


class S2RDataContractTests(unittest.TestCase):
    def test_parse_rows_rejects_duplicate_users(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate user"):
            s2r_data_contract.parse_rows(
                ["u1 i1 i2 i3", "u1 i4 i5 i6"]
            )

    def test_parse_rows_requires_train_target_and_guard(self) -> None:
        with self.assertRaisesRegex(ValueError, "train item"):
            s2r_data_contract.parse_rows(["u1 i1 i2"])

    def test_source_guard_rejects_original_or_arbitrary_sequence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-2r-data-") as temporary:
            path = Path(temporary) / "user_sequence.txt"
            path.write_text("u1 i1 i2 i3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only the sealed Toys D0"):
                s2r_data_contract.validate_source(path)

    def test_build_is_deterministic_disjoint_and_d0_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-2r-data-") as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            source.write_text(
                "\n".join(f"u{index} i1 i2 i3" for index in range(12)) + "\n",
                encoding="utf-8",
            )
            item_text = root / "item_plain_text.txt"
            item_text.write_text(
                "i1 first item\ni2 second item\ni3 third item\n",
                encoding="utf-8",
            )
            first = root / "first"
            second = root / "second"
            with patch.object(s2r_data_contract, "DEFAULT_SOURCE", source), patch.object(
                s2r_data_contract, "DEFAULT_ITEM_TEXT_SOURCE", item_text
            ), patch.object(
                s2r_data_contract, "ROOT", root
            ):
                manifest_a = s2r_data_contract.build_cohorts(
                    source,
                    first,
                    item_text_source=item_text,
                    seed=2023,
                    cohort_count=3,
                    users_per_cohort=3,
                )
                manifest_b = s2r_data_contract.build_cohorts(
                    source,
                    second,
                    item_text_source=item_text,
                    seed=2023,
                    cohort_count=3,
                    users_per_cohort=3,
                )

            self.assertEqual(
                (first / "user_sequence.txt").read_text(encoding="utf-8"),
                (second / "user_sequence.txt").read_text(encoding="utf-8"),
            )
            self.assertTrue(manifest_a["cohorts_disjoint"])
            self.assertFalse(manifest_a["official_test_read"])
            self.assertFalse(manifest_a["sports_read"])
            self.assertFalse(manifest_a["d1_read"])
            self.assertEqual(manifest_a["selected_users"], 9)
            self.assertEqual(manifest_a["r1_smoke_users"], 9)
            self.assertEqual(manifest_a["item_catalog_items"], 3)
            self.assertEqual(
                manifest_a["item_catalog_sha256"],
                manifest_a["item_catalog_source_sha256"],
            )
            self.assertEqual(
                (first / "r1_smoke_user_sequence.txt").read_text(encoding="utf-8"),
                (second / "r1_smoke_user_sequence.txt").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                [entry["users"] for entry in manifest_b["evaluation_cohorts"]],
                [3, 3, 3],
            )


if __name__ == "__main__":
    unittest.main()
