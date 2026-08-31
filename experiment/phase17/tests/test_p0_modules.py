from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiment.phase17.core.feature_hooks import FeatureContext
from experiment.phase17.core.loss_hooks import LossContext, LossHookChain
from experiment.phase17.core.p0_modules import (
    BearSurvivalDecoderLoss,
    BiFlowFeatureHook,
    PrefixCurriculumDecoderLoss,
    ShortcutFiDFeatureHook,
    TransitionTeacherFeatureHook,
)


class P0LossModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.logits = torch.randn(3, 5, 61, requires_grad=True)
        self.labels = torch.tensor(
            [[2, 3, 4, 1, -100], [5, 6, 7, 8, 1], [9, 10, 1, -100, -100]]
        )

    def _run(self, decoder) -> None:
        chain = LossHookChain(decoder=decoder)
        base = self.logits.sum() * 0.0
        total, _ = chain.apply(
            base, LossContext(labels=self.labels, logits=self.logits)
        )
        self.assertEqual(total.ndim, 0)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(self.logits.grad)
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertTrue(decoder.last_metrics)

    def test_a0_bear_survival_has_finite_gradient_and_metric(self) -> None:
        self._run(BearSurvivalDecoderLoss(beam_width=10))

    def test_a1_prefix_curriculum_has_finite_gradient_and_metric(self) -> None:
        self._run(PrefixCurriculumDecoderLoss(steps_per_depth=1))


class P0FeatureModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.batch_size = 2
        self.n_passages = 5
        self.passage_length = 4
        self.d_model = 12
        self.hidden = torch.randn(
            self.batch_size * self.n_passages,
            self.passage_length,
            self.d_model,
            requires_grad=True,
        )
        self.mask = torch.ones(
            self.batch_size * self.n_passages, self.passage_length, dtype=torch.bool
        )
        self.history_ids = torch.tensor([[4, 3, 2, 1], [7, 6, 0, 0]])
        self.history_mask = self.history_ids.ne(0)

    def _context(self) -> FeatureContext:
        return FeatureContext(
            attention_mask=self.mask,
            history_item_ids=self.history_ids,
            history_item_mask=self.history_mask,
            extras={
                "n_passages": self.n_passages,
                "passage_length": self.passage_length,
            },
        )

    def _run(self, hook) -> None:
        output = hook(self.hidden, self._context())
        self.assertEqual(output.shape, self.hidden.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        gradients = [parameter.grad for parameter in hook.parameters()]
        self.assertTrue(gradients)
        self.assertTrue(all(value is not None for value in gradients))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertTrue(hook.last_metrics)

    def test_c0_bidirectional_exchange(self) -> None:
        hook = BiFlowFeatureHook()
        output = hook(self.hidden, self._context())
        output.mean().backward()
        self.assertGreater(hook.last_metrics["sequence_to_global_delta_norm"], 0.0)
        self.assertGreater(hook.last_metrics["global_to_sequence_delta_norm"], 0.0)

    def test_d0_fold_train_transition_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transition.json"
            path.write_text(
                json.dumps({"top_next_dense_id": [0, 2, 3, 4, 5, 6, 7, 1]}),
                encoding="utf-8",
            )
            self._run(TransitionTeacherFeatureHook(self.d_model, path))

    def test_e0_adaptive_shortcut_is_non_degenerate(self) -> None:
        hook = ShortcutFiDFeatureHook()
        self._run(hook)
        self.assertGreater(hook.last_metrics["selected_history_ratio"], 0.0)
        self.assertLess(hook.last_metrics["selected_history_ratio"], 1.0)

    def test_e0_controls_match_the_preregistered_cardinality(self) -> None:
        adaptive = ShortcutFiDFeatureHook()
        random_control = ShortcutFiDFeatureHook(selection_mode="random_same_size")
        full_control = ShortcutFiDFeatureHook(selection_mode="full")
        adaptive(self.hidden, self._context())
        random_control(self.hidden, self._context())
        full_control(self.hidden, self._context())
        self.assertAlmostEqual(
            adaptive.last_metrics["selected_history_ratio"],
            random_control.last_metrics["selected_history_ratio"],
        )
        self.assertEqual(full_control.last_metrics["selected_history_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
