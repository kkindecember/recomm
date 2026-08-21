from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from item_level_eval import (  # noqa: E402
    EvaluationIntegrityError,
    decode_lexical_id,
    evaluate,
    load_item_paths,
    metrics_for_rank,
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestPathMultimap(unittest.TestCase):
    def test_decoding_matches_sentencepiece_join(self):
        self.assertEqual(decode_lexical_id("|▁game|board|▁cat"), "gameboard cat")

    def test_collision_is_preserved_as_multimap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            write(path, "I1 |▁a|b\nI2 |▁a|b\n")
            _forward, reverse = load_item_paths(path)
            self.assertEqual(reverse["ab"], ["I1", "I2"])


class TestStrictEvaluation(unittest.TestCase):
    def fixture(self, root: Path, collision: bool = False) -> tuple[Path, Path, Path]:
        dataset = root / "Dataset_cold50"
        ids = dataset / "ids.txt"
        second = "|▁a|one" if collision else "|▁b|two"
        write(ids, f"I1 |▁a|one\nI2 {second}\n")
        write(dataset / "user_sequence.txt", "U1 I2 I1 I2\n")
        write(dataset / "cold_split_meta" / "cold_items.txt", "I1\n")
        write(dataset / "cold_split_meta" / "warm_items.txt", "I2\n")
        predictions = root / "pred_validation.tsv"
        metrics = metrics_for_rank(1)
        saved = "\t".join(str(metrics[name]) for name in (
            "hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "hit@50",
            "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50",
        ))
        write(
            predictions,
            "idx\tH@5\tH@10\tNDCG@5\tNDCG@10\tgold\tpred\tscores\n"
            f"U1\t{saved}\taone\taone||btwo\t-0.1||-0.2\n",
        )
        return dataset, ids, predictions

    def test_unique_paths_reproduce_string_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, ids, predictions = self.fixture(Path(tmp))
            summary = evaluate(
                dataset_dir=dataset,
                item_path_file=ids,
                predictions_tsv=predictions,
                output_dir=Path(tmp) / "out",
            )
            self.assertTrue(summary["formal_item_level_valid"])
            self.assertEqual(summary["max_metric_abs_diff"], 0.0)
            self.assertEqual(summary["strict_item_metrics"]["hit@1"], 1.0)

    def test_formal_mode_hard_fails_on_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, ids, predictions = self.fixture(Path(tmp), collision=True)
            with self.assertRaises(EvaluationIntegrityError):
                evaluate(
                    dataset_dir=dataset,
                    item_path_file=ids,
                    predictions_tsv=predictions,
                    output_dir=Path(tmp) / "out",
                )

    def test_legacy_record_mode_never_credits_ambiguous_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, ids, predictions = self.fixture(Path(tmp), collision=True)
            summary = evaluate(
                dataset_dir=dataset,
                item_path_file=ids,
                predictions_tsv=predictions,
                output_dir=Path(tmp) / "out",
                invalid_policy="record",
            )
            self.assertFalse(summary["formal_item_level_valid"])
            self.assertEqual(summary["strict_item_metrics"]["hit@1"], 0.0)
            self.assertEqual(summary["alias_string_hits_removed_at_50"], 1)


if __name__ == "__main__":
    unittest.main()
