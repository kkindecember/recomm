"""Independently implemented P1-lite mechanisms for the S17-4 gate.

The classes in this file transfer narrow mechanisms into GRAM's existing
hooks.  They are intentionally named ``lite`` and are not line-by-line
reproductions of the source systems named in the migration cards.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .feature_hooks import FeatureContext, FeatureHook
from .loss_hooks import AuxiliaryLossHook, DecoderLossHook, LossContext
from .p0_modules import _finite_scalar, _masked_pool, _passage_view


class PawaLiteDecoderLoss(DecoderLossHook):
    """Two-bucket prefix-adaptive survival loss.

    This is the S17-4 PAWA-lite transfer: early and late lexical depths have
    separately learned positive weights while a detached top-B difficulty
    proxy emphasizes targets at risk of pruning.  Legal generation remains
    enforced by GRAM's unchanged catalog Trie; the training rank proxy is over
    the decoder vocabulary and must not be called a legal-Trie PAWA replica.
    """

    name = "pawa_lite"

    def __init__(
        self,
        split_depth: int = 3,
        beam_width: int = 50,
        temperature: float = 0.25,
        difficulty_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.split_depth = max(1, int(split_depth))
        self.beam_width = max(1, int(beam_width))
        self.temperature = float(temperature)
        self.difficulty_weight = float(difficulty_weight)
        self.bucket_logits = nn.Parameter(torch.zeros(2))
        self.last_metrics: dict[str, float] = {}

    def forward(self, token_losses: torch.Tensor, context: LossContext) -> torch.Tensor:
        if context.logits is None or context.labels is None:
            raise ValueError("PAWA-lite requires decoder logits and labels")
        logits = context.logits.float()
        labels = context.labels
        valid = labels.ne(-100)
        if not valid.any():
            return token_losses.sum() * 0.0
        safe_labels = labels.masked_fill(~valid, 0)
        log_probs = F.log_softmax(logits, dim=-1)
        target_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        k = min(self.beam_width, logits.size(-1))
        kth_logp = log_probs.topk(k, dim=-1).values[..., -1]
        prune_risk = torch.sigmoid(
            (kth_logp - target_logp) / max(self.temperature, 1e-6)
        ).detach()

        depth = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
        bucket = depth.ge(self.split_depth).long().expand_as(labels)
        positive = F.softplus(self.bucket_logits) + 1e-4
        # A normalized parameterization keeps the initial loss exactly on the
        # parent scale while still allowing early/late adaptation.
        positive = positive / positive.mean()
        depth_weight = positive[bucket]
        weights = depth_weight * (1.0 + self.difficulty_weight * prune_risk)
        loss = (token_losses * weights.to(token_losses.dtype))[valid].sum()
        loss = loss / weights[valid].sum().clamp_min(1.0)

        with torch.no_grad():
            ranks = (log_probs > target_logp.unsqueeze(-1)).sum(dim=-1) + 1
            early = valid & bucket.eq(0)
            late = valid & bucket.eq(1)

            def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return value[mask].float().mean() if mask.any() else value.new_zeros(())

            self.last_metrics = {
                "target_prefix_topB_survival": _finite_scalar(
                    ((ranks <= k) & valid).sum() / valid.sum().clamp_min(1)
                ),
                "early_target_rank": _finite_scalar(masked_mean(ranks, early)),
                "late_target_rank": _finite_scalar(masked_mean(ranks, late)),
                "early_depth_weight": _finite_scalar(positive[0]),
                "late_depth_weight": _finite_scalar(positive[1]),
                "mean_prune_risk": _finite_scalar(prune_risk[valid].mean()),
            }
        return loss


class TreeContrastiveAuxiliaryLoss(AuxiliaryLossHook):
    """Hierarchy-aware prefix contrast over decoder distributions."""

    name = "treecl_lite"

    def __init__(self, prefix_depth: int = 2, margin: float = 0.1) -> None:
        super().__init__()
        self.prefix_depth = max(1, int(prefix_depth))
        self.margin = float(margin)
        self.last_metrics: dict[str, float] = {}

    def forward(self, context: LossContext) -> torch.Tensor:
        if context.logits is None or context.labels is None:
            raise ValueError("TreeCL-lite requires logits and labels")
        depth = min(self.prefix_depth, context.logits.size(1))
        representation = F.normalize(
            F.log_softmax(context.logits[:, :depth].float(), dim=-1).mean(dim=1),
            dim=-1,
        )
        labels = context.labels[:, :depth]
        valid = labels.ne(-100)
        safe = labels.masked_fill(~valid, -1)
        overlap = (safe[:, None, :] == safe[None, :, :]) & valid[:, None, :]
        overlap = overlap.float().sum(dim=-1) / valid[:, None, :].sum(dim=-1).clamp_min(1)
        similarity = representation @ representation.transpose(0, 1)
        eye = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
        positive = overlap.gt(0) & ~eye
        negative = overlap.eq(0) & ~eye
        if not positive.any() or not negative.any():
            value = similarity.sum() * 0.0
        else:
            positive_mean = similarity[positive].mean()
            negative_mean = similarity[negative].mean()
            value = F.relu(self.margin - positive_mean + negative_mean)
        with torch.no_grad():
            self.last_metrics = {
                "prefix_positive_pair_fraction": _finite_scalar(positive.float().mean()),
                "prefix_negative_pair_fraction": _finite_scalar(negative.float().mean()),
                "tree_contrastive_loss": _finite_scalar(value),
            }
        return value


class TokenSetAuxiliaryLoss(AuxiliaryLossHook):
    """Order-agnostic target-token coverage objective for SetHead-lite."""

    name = "sethead_lite"

    def __init__(self, exclude_eos: bool = True) -> None:
        super().__init__()
        self.exclude_eos = bool(exclude_eos)
        self.last_metrics: dict[str, float] = {}

    def forward(self, context: LossContext) -> torch.Tensor:
        if context.logits is None or context.labels is None:
            raise ValueError("SetHead-lite requires logits and labels")
        log_probs = F.log_softmax(context.logits.float(), dim=-1)
        labels = context.labels
        valid = labels.ne(-100)
        if self.exclude_eos:
            valid = valid & labels.ne(1)
        losses = []
        recalls = []
        for row in range(labels.size(0)):
            tokens = labels[row][valid[row]].unique()
            if tokens.numel() == 0:
                continue
            token_scores = log_probs[row, :, tokens].amax(dim=0)
            losses.append(-token_scores.mean())
            predicted = context.logits[row].argmax(dim=-1).unique()
            match = (tokens[:, None] == predicted[None, :]).any(dim=1)
            recalls.append(match.float().mean())
        if not losses:
            return context.logits.sum() * 0.0
        value = torch.stack(losses).mean()
        with torch.no_grad():
            self.last_metrics = {
                "token_set_loss": _finite_scalar(value),
                "target_token_set_recall": _finite_scalar(torch.stack(recalls).mean()),
                "set_examples": float(len(losses)),
            }
        return value


class ContextRootPromptFeatureHook(FeatureHook):
    """Target-free context route represented as a gated root prompt."""

    def __init__(self, initial_logit: float = -2.0) -> None:
        super().__init__()
        self.route_logit = nn.Parameter(torch.tensor(initial_logit))
        self.last_metrics: dict[str, float] = {}

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        states, masks, _, _ = _passage_view(hidden_states, context)
        if states.size(1) <= 1:
            return hidden_states
        item_states = _masked_pool(states[:, 1:], masks[:, 1:])
        item_valid = masks[:, 1:].any(dim=-1)
        positions = torch.arange(item_states.size(1), device=states.device).float()
        recency = torch.exp(-positions / max(1.0, item_states.size(1) / 3.0))
        weights = recency[None, :] * item_valid.to(states.dtype)
        route = (item_states * weights[:, :, None]).sum(dim=1)
        route = route / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        gate = torch.sigmoid(self.route_logit)
        output = states.clone()
        output[:, 0] = output[:, 0] + gate * route[:, None, :] * masks[:, 0, :, None]
        with torch.no_grad():
            self.last_metrics = {
                "context_root_gate": _finite_scalar(gate),
                "context_route_norm": _finite_scalar(route.norm(dim=-1).mean()),
            }
        return output.reshape_as(hidden_states)


class LongShortFiDFeatureHook(FeatureHook):
    """Separate recent and long-history summaries with normalized gates."""

    def __init__(self, recent_window: int = 3) -> None:
        super().__init__()
        self.recent_window = max(1, int(recent_window))
        self.gate_logits = nn.Parameter(torch.zeros(2))
        self.last_metrics: dict[str, float] = {}

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        states, masks, _, _ = _passage_view(hidden_states, context)
        if states.size(1) <= 1:
            return hidden_states
        item_states = _masked_pool(states[:, 1:], masks[:, 1:])
        valid = masks[:, 1:].any(dim=-1)
        recent_mask = valid.clone()
        recent_mask[:, self.recent_window :] = False
        long_mask = valid & ~recent_mask

        def pool(mask: torch.Tensor) -> torch.Tensor:
            weight = mask.unsqueeze(-1).to(states.dtype)
            return (item_states * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

        recent = pool(recent_mask)
        long = pool(long_mask)
        gates = torch.softmax(self.gate_logits, dim=0)
        fused = gates[0] * recent + gates[1] * long
        output = states.clone()
        output[:, 0] = output[:, 0] + fused[:, None, :] * masks[:, 0, :, None]
        with torch.no_grad():
            self.last_metrics = {
                "recent_gate": _finite_scalar(gates[0]),
                "long_gate": _finite_scalar(gates[1]),
                "long_history_available": _finite_scalar(long_mask.any(dim=1).float().mean()),
            }
        return output.reshape_as(hidden_states)


class OneWayBridgeFeatureHook(FeatureHook):
    """Ablated BiFlow bridge with exactly one permitted information direction."""

    DIRECTIONS = {"sequence_to_global", "global_to_sequence"}

    def __init__(self, direction: str = "sequence_to_global", initial_logit: float = -2.0):
        super().__init__()
        if direction not in self.DIRECTIONS:
            raise ValueError(f"unknown one-way bridge direction: {direction}")
        self.direction = direction
        self.gate_logit = nn.Parameter(torch.tensor(initial_logit))
        self.last_metrics: dict[str, float] = {}

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        states, masks, _, _ = _passage_view(hidden_states, context)
        if states.size(1) <= 1:
            return hidden_states
        global_state = _masked_pool(states[:, :1], masks[:, :1]).squeeze(1)
        item_states = _masked_pool(states[:, 1:], masks[:, 1:])
        valid = masks[:, 1:].any(dim=-1)
        weight = valid.unsqueeze(-1).to(states.dtype)
        sequence_state = (item_states * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        gate = torch.sigmoid(self.gate_logit)
        output = states.clone()
        if self.direction == "sequence_to_global":
            delta = gate * sequence_state
            output[:, 0] = output[:, 0] + delta[:, None, :] * masks[:, 0, :, None]
            s2g, g2s = gate, gate.new_zeros(())
        else:
            delta = gate * global_state
            output[:, 1:] = output[:, 1:] + delta[:, None, None, :] * masks[:, 1:, :, None]
            s2g, g2s = gate.new_zeros(()), gate
        with torch.no_grad():
            self.last_metrics = {
                "sequence_to_global_gate": _finite_scalar(s2g),
                "global_to_sequence_gate": _finite_scalar(g2s),
                "active_delta_norm": _finite_scalar(delta.norm(dim=-1).mean()),
                "blocked_direction_count": 1.0,
            }
        return output.reshape_as(hidden_states)


class MaskedHistoryFeatureHook(FeatureHook):
    """Deterministic train-only masked-history state perturbation smoke."""

    def __init__(self, mask_probability: float = 0.15, initial_logit: float = -2.0) -> None:
        super().__init__()
        if not 0.0 < mask_probability < 1.0:
            raise ValueError("mask_probability must be in (0, 1)")
        self.mask_probability = float(mask_probability)
        self.mask_logit = nn.Parameter(torch.tensor(initial_logit))
        self.last_metrics: dict[str, float] = {}

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        if not self.training:
            return hidden_states
        states, masks, _, _ = _passage_view(hidden_states, context)
        if states.size(1) <= 1:
            return hidden_states
        valid = masks[:, 1:].any(dim=-1)
        positions = torch.arange(valid.size(1), device=states.device)[None, :]
        ids = context.history_item_ids
        if ids is None:
            ids = positions.expand(valid.size(0), -1)
        else:
            ids = ids[:, : valid.size(1)].to(states.device)
        threshold = int(self.mask_probability * 997)
        selected = ((ids * 37 + positions * 101 + 17) % 997).lt(threshold) & valid
        gate = torch.sigmoid(self.mask_logit)
        history_scale = 1.0 - gate * selected[:, :, None, None].to(states.dtype)
        output = torch.cat((states[:, :1], states[:, 1:] * history_scale), dim=1)
        with torch.no_grad():
            self.last_metrics = {
                "masked_history_fraction": _finite_scalar(
                    selected.sum() / valid.sum().clamp_min(1)
                ),
                "mask_gate": _finite_scalar(gate),
            }
        return output.reshape_as(hidden_states)


class LogitConcentrationAuxiliaryLoss(AuxiliaryLossHook):
    """Batch-level output concentration penalty used as a SPRINT-lite smoke."""

    name = "sprint_lite"

    def __init__(self, topk: int = 128) -> None:
        super().__init__()
        self.topk = max(2, int(topk))
        self.last_metrics: dict[str, float] = {}

    def forward(self, context: LossContext) -> torch.Tensor:
        if context.logits is None:
            raise ValueError("SPRINT-lite requires logits")
        probabilities = F.softmax(context.logits.float(), dim=-1).mean(dim=(0, 1))
        k = min(self.topk, probabilities.numel())
        mass = probabilities.topk(k).values
        normalized = mass / mass.sum().clamp_min(1e-8)
        concentration = (normalized.square().sum() - 1.0 / k).clamp_min(0.0)
        entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum()
        with torch.no_grad():
            self.last_metrics = {
                "output_concentration_penalty": _finite_scalar(concentration),
                "topk_output_entropy": _finite_scalar(entropy),
            }
        return concentration
