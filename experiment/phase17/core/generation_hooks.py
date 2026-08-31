"""Shared legal-Trie score hooks used during constrained decoding."""

from __future__ import annotations

import torch
from torch import nn


class GenerationScoreHook(nn.Module):
    def forward(
        self, scores: torch.Tensor, legal_token_mask: torch.Tensor, depth: int
    ) -> torch.Tensor:
        raise NotImplementedError


class GenerationHookChain(GenerationScoreHook):
    def __init__(self, hooks: list[GenerationScoreHook] | None = None) -> None:
        super().__init__()
        self.hooks = nn.ModuleList(hooks or [])

    def forward(
        self, scores: torch.Tensor, legal_token_mask: torch.Tensor, depth: int
    ) -> torch.Tensor:
        if legal_token_mask.dtype is not torch.bool or legal_token_mask.shape != scores.shape:
            raise ValueError("legal token mask must be boolean and match score shape")
        output = scores
        for hook in self.hooks:
            output = hook(output, legal_token_mask, depth)
            if output.shape != scores.shape or not torch.isfinite(output[legal_token_mask]).all():
                raise FloatingPointError("generation hook violated shape/finite contract")
        return output.masked_fill(~legal_token_mask, float("-inf"))

    @property
    def is_identity(self) -> bool:
        return len(self.hooks) == 0


def assert_generated_paths_legal(
    paths: list[tuple[int, ...]], legal_paths: set[tuple[int, ...]]
) -> None:
    illegal = [path for path in paths if path not in legal_paths]
    if illegal:
        raise ValueError(f"generated paths outside locked Trie: {illegal[:3]}")
