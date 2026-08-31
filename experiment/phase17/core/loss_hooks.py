"""Auxiliary and decoder-loss hooks with exact zero-weight degeneration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossContext:
    labels: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    target_item_ids: torch.Tensor | None = None
    extras: Mapping[str, Any] | None = None


class AuxiliaryLossHook(nn.Module):
    name = "auxiliary"

    def forward(self, context: LossContext) -> torch.Tensor:
        raise NotImplementedError


class DecoderLossHook(nn.Module):
    name = "decoder"

    def forward(self, token_losses: torch.Tensor, context: LossContext) -> torch.Tensor:
        raise NotImplementedError


class MeanDecoderLoss(DecoderLossHook):
    name = "mean_token_ce"

    def forward(self, token_losses: torch.Tensor, context: LossContext) -> torch.Tensor:
        if context.labels is None:
            return token_losses.mean()
        valid = context.labels.ne(-100)
        return token_losses[valid].mean() if valid.any() else token_losses.sum() * 0.0


class LossHookChain(nn.Module):
    """Combines one decoder reducer and named weighted auxiliary losses."""

    def __init__(
        self,
        decoder: DecoderLossHook | None = None,
        auxiliary: list[tuple[str, float, AuxiliaryLossHook]] | None = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.auxiliary = nn.ModuleDict({name: hook for name, _, hook in auxiliary or []})
        self.weights = {name: float(weight) for name, weight, _ in auxiliary or []}

    @staticmethod
    def token_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)

    def apply(
        self, base_loss: torch.Tensor, context: LossContext
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {"parent": base_loss.detach()}
        total = base_loss
        if self.decoder is not None:
            if context.logits is None or context.labels is None:
                raise ValueError("active decoder loss requires logits and labels")
            token_losses = self.token_cross_entropy(context.logits, context.labels)
            total = self.decoder(token_losses, context)
            components[f"decoder/{self.decoder.name}"] = total.detach()
        for name, hook in self.auxiliary.items():
            weight = self.weights[name]
            if weight == 0.0:
                components[f"aux/{name}"] = base_loss.detach().new_zeros(())
                continue
            value = hook(context)
            if value.ndim != 0 or not torch.isfinite(value):
                raise FloatingPointError(f"auxiliary loss {name} is not a finite scalar")
            total = total + weight * value
            components[f"aux/{name}"] = value.detach()
        components["total"] = total.detach()
        return total, components

    @property
    def is_identity(self) -> bool:
        return self.decoder is None and all(weight == 0.0 for weight in self.weights.values())
