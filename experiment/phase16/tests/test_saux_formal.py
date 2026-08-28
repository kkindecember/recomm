from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from saux_formal_train import build_pseudo_events, build_transitions


class SAuxFormalTests(unittest.TestCase):
    def test_transition_builder_uses_only_prior_history(self) -> None:
        sequences = [{"user_id": "u", "items": ["a", "b", "c"]}]
        histories, lengths, labels, label_items = build_transitions(sequences, {"a": 1, "b": 2, "c": 3}, 20)
        self.assertEqual(histories[:, :2].tolist(), [[1, 0], [1, 2]])
        self.assertEqual(lengths.tolist(), [1, 2])
        self.assertEqual(labels.tolist(), [2, 3])
        self.assertEqual(label_items, {"b", "c"})

    def test_pseudo_target_never_enters_history_index(self) -> None:
        events = [{"history": ["a", "b"], "target_item": "p"}]
        histories, lengths, targets = build_pseudo_events(events, {"a": 1, "b": 2}, {"a": 0, "b": 1, "p": 2}, 20)
        self.assertEqual(histories[0, :2].tolist(), [1, 2])
        self.assertEqual(lengths.tolist(), [2])
        self.assertEqual(targets.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
