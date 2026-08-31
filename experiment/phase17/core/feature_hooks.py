"""Composable encoder-feature hooks for GRAM mechanism migration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class FeatureContext:
    """Target-free metadata available to encoder feature hooks."""

    attention_mask: torch.Tensor | None = None
    history_item_ids: torch.Tensor | None = None
    history_item_mask: torch.Tensor | None = None
    cutoff_positions: torch.Tensor | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


class FeatureHook(nn.Module):
    """Base class used by every encoder/history migration module."""

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        raise NotImplementedError


class IdentityFeatureHook(FeatureHook):
    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        return hidden_states


class FeatureHookChain(FeatureHook):
    def __init__(self, hooks: list[FeatureHook] | None = None) -> None:
        super().__init__()
        self.hooks = nn.ModuleList(hooks or [])

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        output = hidden_states
        for hook in self.hooks:
            output = hook(output, context)
            if output.shape != hidden_states.shape:
                raise ValueError("feature hook changed the GRAM encoder tensor shape")
            if not torch.isfinite(output).all():
                raise FloatingPointError("feature hook produced NaN/Inf")
        return output

    @property
    def is_identity(self) -> bool:
        return len(self.hooks) == 0
