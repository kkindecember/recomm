"""Runtime bundle connecting the public hooks to GRAM."""

from __future__ import annotations

from torch import nn

from .feature_hooks import FeatureContext, FeatureHookChain
from .generation_hooks import GenerationHookChain
from .loss_hooks import LossHookChain
from .p0_modules import scalar_metrics


class MigrationRuntime(nn.Module):
    def __init__(
        self,
        enabled_modules: tuple[str, ...] = (),
        feature_hooks: FeatureHookChain | None = None,
        loss_hooks: LossHookChain | None = None,
        generation_hooks: GenerationHookChain | None = None,
        item_aggregation: str = "logsumexp",
    ) -> None:
        super().__init__()
        self.enabled_modules = tuple(enabled_modules)
        self.feature_hooks = feature_hooks or FeatureHookChain()
        self.loss_hooks = loss_hooks or LossHookChain()
        self.generation_hooks = generation_hooks or GenerationHookChain()
        self.item_aggregation = item_aggregation

    def apply_features(self, hidden_states, **context):
        return self.feature_hooks(hidden_states, FeatureContext(**context))

    def mechanism_metrics(self) -> dict[str, float]:
        """Return the most recent finite, scalar mechanism diagnostics."""

        result: dict[str, float] = {}
        modules = list(self.feature_hooks.hooks)
        if self.loss_hooks.decoder is not None:
            modules.append(self.loss_hooks.decoder)
        modules.extend(self.loss_hooks.auxiliary.values())
        for module in modules:
            prefix = getattr(module, "_s17_module_id", module.__class__.__name__)
            for name, value in scalar_metrics(module).items():
                result[f"{prefix}/{name}"] = value
        return result

    @property
    def is_identity(self) -> bool:
        return (
            not self.enabled_modules
            and self.feature_hooks.is_identity
            and self.loss_hooks.is_identity
            and self.generation_hooks.is_identity
        )
