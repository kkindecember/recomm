from __future__ import annotations

import sys
import unittest
from pathlib import Path

from experiment.phase17.core.generation_hooks import assert_generated_paths_legal


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "GRAM/src"))
from utils.generation_trie import Trie  # noqa: E402


class TrieLegalityTests(unittest.TestCase):
    def test_locked_trie_prefixes_and_complete_paths(self) -> None:
        paths = [(0, 10, 20, 1), (0, 10, 30, 1)]
        trie = Trie([list(path) for path in paths])
        self.assertEqual(set(trie.get([0, 10])), {20, 30})
        assert_generated_paths_legal(paths, set(paths))

    def test_illegal_generated_path_fails(self) -> None:
        with self.assertRaises(ValueError):
            assert_generated_paths_legal([(0, 10, 99, 1)], {(0, 10, 20, 1)})
