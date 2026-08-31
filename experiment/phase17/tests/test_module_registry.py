from __future__ import annotations

import unittest

import torch
from torch import nn

from experiment.phase17.core.feature_hooks import FeatureContext, FeatureHook
from experiment.phase17.core.loss_hooks import AuxiliaryLossHook, LossContext
from experiment.phase17.registry.module_registry import ModuleRegistry, ModuleSpec, build_runtime


class ScaleHook(FeatureHook):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        return hidden_states * self.scale


class ExplodingAux(AuxiliaryLossHook):
    def forward(self, context: LossContext) -> torch.Tensor:
        raise AssertionError("zero-weight auxiliary hooks must not execute")


class ModuleRegistryTests(unittest.TestCase):
    def test_default_runtime_is_identity(self) -> None:
        self.assertTrue(build_runtime("").is_identity)

    def test_unknown_module_fails_closed(self) -> None:
        with self.assertRaises(KeyError):
            build_runtime("not_registered")

    def test_feature_module_has_finite_gradient(self) -> None:
        registry = ModuleRegistry()
        registry.register(ModuleSpec("scale", "T", feature_factory=ScaleHook))
        runtime = registry.build(["scale"])
        value = torch.ones(2, 3, requires_grad=True)
        result = runtime.apply_features(value).sum()
        result.backward()
        gradients = [parameter.grad for parameter in runtime.parameters()]
        self.assertTrue(gradients and all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_zero_weight_auxiliary_strictly_degenerates_to_parent(self) -> None:
        registry = ModuleRegistry()
        registry.register(ModuleSpec("zero", "T", auxiliary_factory=ExplodingAux, auxiliary_weight=0.0))
        runtime = registry.build(["zero"])
        parent = torch.tensor(2.0, requires_grad=True)
        total, _ = runtime.loss_hooks.apply(parent, LossContext())
        self.assertIs(total, parent)
