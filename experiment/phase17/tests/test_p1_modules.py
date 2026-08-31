from __future__ import annotations

import unittest

import torch

from experiment.phase17.core.feature_hooks import FeatureContext
from experiment.phase17.core.loss_hooks import LossContext, LossHookChain
from experiment.phase17.core.p1_modules import (
    ContextRootPromptFeatureHook,
    LogitConcentrationAuxiliaryLoss,
    LongShortFiDFeatureHook,
    MaskedHistoryFeatureHook,
    OneWayBridgeFeatureHook,
    PawaLiteDecoderLoss,
    TokenSetAuxiliaryLoss,
    TreeContrastiveAuxiliaryLoss,
)


class P1LossModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1704)
        self.logits = torch.randn(4, 6, 71, requires_grad=True)
        self.labels = torch.tensor(
            [
                [2, 3, 4, 5, 1, -100],
                [2, 3, 9, 8, 1, -100],
                [6, 7, 4, 5, 1, -100],
                [10, 11, 12, 1, -100, -100],
            ]
        )

    def test_pawa_lite_is_finite_and_adaptive(self) -> None:
        decoder = PawaLiteDecoderLoss(beam_width=10)
        chain = LossHookChain(decoder=decoder)
        total, _ = chain.apply(
            self.logits.sum() * 0.0,
            LossContext(logits=self.logits, labels=self.labels),
        )
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(decoder.bucket_logits.grad)
        self.assertEqual(
            set(decoder.last_metrics),
            {
                "target_prefix_topB_survival",
                "early_target_rank",
                "late_target_rank",
                "early_depth_weight",
                "late_depth_weight",
                "mean_prune_risk",
            },
        )

    def test_p1_auxiliaries_are_finite_and_differentiable(self) -> None:
        for hook in (
            TreeContrastiveAuxiliaryLoss(),
            TokenSetAuxiliaryLoss(),
            LogitConcentrationAuxiliaryLoss(),
        ):
            logits = self.logits.detach().clone().requires_grad_(True)
            value = hook(LossContext(logits=logits, labels=self.labels))
            self.assertTrue(torch.isfinite(value))
            value.backward()
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertTrue(hook.last_metrics)


class P1FeatureModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1704)
        self.batch = 3
        self.passages = 7
        self.length = 4
        self.width = 16
        self.hidden = torch.randn(
            self.batch * self.passages,
            self.length,
            self.width,
            requires_grad=True,
        )
        self.mask = torch.ones(
            self.batch * self.passages, self.length, dtype=torch.bool
        )
        self.history_ids = torch.tensor(
            [[9, 8, 7, 6, 5, 4], [6, 5, 4, 3, 0, 0], [3, 2, 1, 0, 0, 0]]
        )
        self.context = FeatureContext(
            attention_mask=self.mask,
            history_item_ids=self.history_ids,
            history_item_mask=self.history_ids.ne(0),
            extras={"n_passages": self.passages, "passage_length": self.length},
        )

    def _run(self, hook) -> None:
        value = hook(self.hidden, self.context)
        self.assertEqual(value.shape, self.hidden.shape)
        self.assertTrue(torch.isfinite(value).all())
        value.square().mean().backward(retain_graph=True)
        gradients = [parameter.grad for parameter in hook.parameters()]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(hook.last_metrics)

    def test_context_root_and_long_short_hooks(self) -> None:
        self._run(ContextRootPromptFeatureHook())
        self._run(LongShortFiDFeatureHook())

    def test_one_way_bridge_blocks_exactly_one_direction(self) -> None:
        for direction in ("sequence_to_global", "global_to_sequence"):
            hook = OneWayBridgeFeatureHook(direction)
            self._run(hook)
            gates = (
                hook.last_metrics["sequence_to_global_gate"],
                hook.last_metrics["global_to_sequence_gate"],
            )
            self.assertEqual(sum(value == 0.0 for value in gates), 1)
            self.assertGreater(hook.last_metrics["active_delta_norm"], 0.0)

    def test_masked_history_is_train_only(self) -> None:
        hook = MaskedHistoryFeatureHook(mask_probability=0.5).train()
        train_value = hook(self.hidden, self.context)
        self.assertFalse(torch.equal(train_value, self.hidden))
        hook.eval()
        eval_value = hook(self.hidden, self.context)
        self.assertTrue(torch.equal(eval_value, self.hidden))


if __name__ == "__main__":
    unittest.main()
