from __future__ import annotations

import unittest
from dataclasses import dataclass

from experiment.phase17.protocol.s2r_r2_screen_runtime import (
    gryphon_beam_control,
    should_early_stop,
)


@dataclass(frozen=True)
class Row:
    item_id: str
    beam_score: float


class S2RR2ScreenRuntimeTests(unittest.TestCase):
    def test_early_stop_respects_minimum_epochs_and_patience(self) -> None:
        self.assertFalse(
            should_early_stop(
                completed_epochs=2, stale_epochs=2, minimum_epochs=3, patience=2
            )
        )
        self.assertTrue(
            should_early_stop(
                completed_epochs=3, stale_epochs=2, minimum_epochs=3, patience=2
            )
        )

    def test_gryphon_control_only_reorders_same_candidates(self) -> None:
        treatment = [[Row("b", 0.1), Row("a", 0.8), Row("c", 0.4)]]
        control = gryphon_beam_control(treatment)
        self.assertEqual([row.item_id for row in control[0]], ["a", "c", "b"])
        self.assertEqual(
            {row.item_id for row in treatment[0]},
            {row.item_id for row in control[0]},
        )


if __name__ == "__main__":
    unittest.main()
