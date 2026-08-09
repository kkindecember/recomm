"""Unit tests for phase13 cold_split preprocessor and eval_cold_warm parser.

Run:
    /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \\
        -m unittest experiment/phase13/tests/test_cold_split.py -v
"""
from __future__ import annotations

import io
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE.parent / "protocol"
sys.path.insert(0, str(PROTOCOL_DIR))

import cold_split  # noqa: E402
import eval_cold_warm  # noqa: E402


class TestFrequencyStratifiedSampling(unittest.TestCase):
    def test_actual_fraction_close_to_eta(self):
        item_freq = {f"i{i}": (i % 20) + 1 for i in range(1000)}
        rng = random.Random(42)
        cold = cold_split.frequency_stratified_cold_sample(
            item_freq=item_freq, eta=0.5, n_buckets=10, rng=rng,
        )
        self.assertGreater(len(cold), 0)
        actual = len(cold) / len(item_freq)
        self.assertAlmostEqual(actual, 0.5, delta=0.05)

    def test_cold_spans_frequency_range(self):
        item_freq = {f"i{i}": i + 1 for i in range(500)}  # freq 1..500
        rng = random.Random(1)
        cold = cold_split.frequency_stratified_cold_sample(
            item_freq=item_freq, eta=0.5, n_buckets=10, rng=rng,
        )
        cold_freqs = [item_freq[it] for it in cold]
        # Should span from low to high freq (stratified)
        self.assertLess(min(cold_freqs), 50)
        self.assertGreater(max(cold_freqs), 400)

    def test_seed_deterministic(self):
        item_freq = {f"i{i}": (i % 7) + 1 for i in range(200)}
        cold1 = cold_split.frequency_stratified_cold_sample(
            item_freq=item_freq, eta=0.5, n_buckets=5, rng=random.Random(99),
        )
        cold2 = cold_split.frequency_stratified_cold_sample(
            item_freq=item_freq, eta=0.5, n_buckets=5, rng=random.Random(99),
        )
        self.assertEqual(cold1, cold2)


class TestUserFilter(unittest.TestCase):
    def test_prefix_filter_keeps_val_test_cold(self):
        users = [
            ("u1", ["a", "b", "c", "d", "e", "cold1", "cold2"]),  # val=cold1, test=cold2
            ("u2", ["a", "b", "c", "d", "e", "f", "g"]),
        ]
        cold = {"cold1", "cold2", "b"}
        new_users, dropped, stats = cold_split.filter_users(
            users=users, cold_items=cold, min_warm_history=3,
        )
        # u1: prefix=[a,b,c,d,e], warm=[a,c,d,e] len=4 >= 3, keep
        #     new_seq = [a,c,d,e] + [cold1, cold2]
        # u2: prefix=[a,b,c,d,e], warm=[a,c,d,e] len=4 >= 3, keep
        self.assertEqual(len(new_users), 2)
        uid, seq = new_users[0]
        self.assertEqual(uid, "u1")
        self.assertEqual(seq, ["a", "c", "d", "e", "cold1", "cold2"])
        self.assertEqual(stats["n_users_with_cold_val_target"], 1)
        self.assertEqual(stats["n_users_with_cold_test_target"], 1)

    def test_drops_users_with_short_warm_prefix(self):
        users = [
            ("u_short", ["a", "b", "cold1", "cold2", "cold3"]),  # prefix warm=[a,b] len=2
            ("u_ok", ["a", "b", "c", "d", "e", "f", "g"]),
        ]
        cold = {"cold1", "cold2", "cold3"}
        new_users, dropped, stats = cold_split.filter_users(
            users=users, cold_items=cold, min_warm_history=3,
        )
        self.assertEqual(len(new_users), 1)
        self.assertEqual(new_users[0][0], "u_ok")
        self.assertEqual(stats["n_users_dropped_warm_prefix_too_short"], 1)


class TestFullPipelineOnSyntheticDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "syn_src"
        self.dst = self.tmp / "syn_cold50"
        self.src.mkdir()
        # 20 users, 12 items, seq len ~ 6-10
        rng = random.Random(0)
        with open(self.src / "user_sequence.txt", "w") as f:
            for u in range(20):
                seq = [f"i{rng.randint(0, 11)}" for _ in range(rng.randint(6, 10))]
                f.write(f"u{u} " + " ".join(seq) + "\n")
        with open(self.src / "item_plain_text.txt", "w") as f:
            for i in range(12):
                f.write(f"i{i} title fake text\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_end_to_end(self):
        old_argv = sys.argv
        sys.argv = [
            "cold_split.py",
            "--source-dir", str(self.src),
            "--output-dir", str(self.dst),
            "--eta", "0.5",
            "--buckets", "4",
            "--seed", "7",
            "--min-warm-history", "2",
            "--force",
        ]
        try:
            cold_split.main()
        finally:
            sys.argv = old_argv

        self.assertTrue((self.dst / "user_sequence.txt").exists())
        self.assertTrue((self.dst / "item_plain_text.txt").exists())
        self.assertTrue((self.dst / "cold_split_meta" / "cold_items.txt").exists())
        cfg = json.loads((self.dst / "cold_split_meta" / "config.json").read_text())
        self.assertEqual(cfg["eta"], 0.5)
        self.assertGreater(cfg["n_items_cold"], 0)
        self.assertGreater(cfg["stats"]["n_users_kept"], 0)


class TestEvalColdWarmParser(unittest.TestCase):
    def test_parses_metric_header_and_skips_trailing_summary(self):
        content = (
            "idx\tH@5\tH@10\tNDCG@5\tNDCG@10\tgold\tpred\tscores\n"
            "u1\t0\t0.5\t0\t0.3\t0\t0.2\t0\t0\t0\t0\t0\t0\tgold text\tpred||stuff\t-0.1\n"
            "u2\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\tgold text\tpred||stuff\t-0.2\n"
            "hit@10: 0.25\n"
            "ndcg@10: 0.15\n"
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            f = tmp / "pred.tsv"
            f.write_text(content)
            rows = eval_cold_warm.parse_predictions_tsv(f)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0], "u1")
            self.assertEqual(rows[0][1][3], 0.3)  # hit@10 column (METRIC_NAMES index 3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestBeautyCold50Preflight(unittest.TestCase):
    """Real Beauty_cold50 dir must exist and be consistent."""

    ROOT = Path(__file__).resolve().parents[3]
    DATASET_DIR = ROOT / "GRAM" / "rec_datasets" / "Beauty_cold50"

    def test_dataset_dir_exists_and_has_files(self):
        self.assertTrue(self.DATASET_DIR.is_dir(),
                        f"missing {self.DATASET_DIR}; run cold_split.py first")
        for name in ["user_sequence.txt", "item_plain_text.txt",
                     "cold_split_meta/config.json",
                     "cold_split_meta/cold_items.txt",
                     "cold_split_meta/warm_items.txt"]:
            self.assertTrue((self.DATASET_DIR / name).exists(),
                            f"missing {self.DATASET_DIR / name}")

    def test_config_matches_v0_expectation(self):
        cfg = json.loads((self.DATASET_DIR / "cold_split_meta" / "config.json").read_text())
        self.assertEqual(cfg["eta"], 0.5)
        self.assertEqual(cfg["seed"], 12345)
        self.assertEqual(cfg["min_warm_history"], 3)
        self.assertEqual(cfg["buckets"], 10)
        # Sanity range checks
        self.assertGreater(cfg["stats"]["n_users_kept"], 5000)
        self.assertLess(cfg["stats"]["n_users_kept"], 22363)

    def test_cold_warm_disjoint_and_cover_all(self):
        cold = set((self.DATASET_DIR / "cold_split_meta" / "cold_items.txt").read_text().split())
        warm = set((self.DATASET_DIR / "cold_split_meta" / "warm_items.txt").read_text().split())
        self.assertEqual(cold & warm, set())
        # All items in dst user_sequence should be in cold ∪ warm
        useq_items = set()
        with open(self.DATASET_DIR / "user_sequence.txt") as f:
            for line in f:
                parts = line.strip().split()
                useq_items.update(parts[1:])
        missing = useq_items - (cold | warm)
        self.assertEqual(missing, set(),
                         f"items in user_sequence not in cold ∪ warm: {sorted(missing)[:5]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
