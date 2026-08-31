"""Lite, independently implemented P0 mechanisms for the S17-2 probes.

These modules transfer the mechanism named in each migration card; they are not
intended to be line-by-line reproductions of the source repositories.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .feature_hooks import FeatureContext, FeatureHook
from .loss_hooks import DecoderLossHook, LossContext


def _finite_scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


class BearSurvivalDecoderLoss(DecoderLossHook):
    """BEAR-style target survival weighting using a full-vocabulary top-B proxy.

    S17-2 deliberately uses the full vocabulary rather than the catalog Trie in
    this loss-only probe.  The legal-Trie version is reserved for S17-3 after the
    probe has established that the weighting path is learnable and stable.
    """

    name = "bear_survival_proxy"

    def __init__(self, beam_width: int = 50, temperature: float = 0.2, topk_weight: float = 1.0):
        super().__init__()
        self.beam_width = int(beam_width)
        self.temperature = float(temperature)
        self.topk_weight = float(topk_weight)
        self.last_metrics: dict[str, float] = {}

    def forward(self, token_losses: torch.Tensor, context: LossContext) -> torch.Tensor:
        if context.logits is None or context.labels is None:
            raise ValueError("BEAR proxy requires decoder logits and labels")
        labels = context.labels
        valid = labels.ne(-100)
        if not valid.any():
            return token_losses.sum() * 0.0
        safe_labels = labels.masked_fill(~valid, 0)
        log_probs = F.log_softmax(context.logits.float(), dim=-1)
        target_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        k = min(self.beam_width, log_probs.size(-1))
        kth_logp = log_probs.topk(k, dim=-1).values[..., -1]
        # Official-code-inspired detached survival weighting.  Detaching the
        # weight prevents a second gradient path through the rank proxy.
        survival_score = torch.sigmoid(
            (target_logp - kth_logp) / max(self.temperature, 1e-6)
        )
        weights = (1.0 + self.topk_weight * survival_score).detach()
        weighted = token_losses * weights.to(token_losses.dtype)
        loss = weighted[valid].sum() / weights[valid].sum().clamp_min(1.0)
        with torch.no_grad():
            ranks = (log_probs > target_logp.unsqueeze(-1)).sum(dim=-1) + 1
            prefix_survival = (ranks <= k) & valid
            self.last_metrics = {
                "target_prefix_topB_survival": _finite_scalar(
                    prefix_survival.sum() / valid.sum().clamp_min(1)
                ),
                "mean_target_rank": _finite_scalar(ranks[valid].float().mean()),
                "mean_survival_weight": _finite_scalar(weights[valid].mean()),
            }
        return loss


class PrefixCurriculumDecoderLoss(DecoderLossHook):
    """Progressively exposes lexical-identifier positions to teacher forcing."""

    name = "prefix_curriculum"

    def __init__(self, steps_per_depth: int = 8):
        super().__init__()
        self.steps_per_depth = max(1, int(steps_per_depth))
        self.register_buffer("forward_calls", torch.zeros((), dtype=torch.long))
        self.last_metrics: dict[str, float] = {}

    def forward(self, token_losses: torch.Tensor, context: LossContext) -> torch.Tensor:
        if context.labels is None or context.logits is None:
            raise ValueError("prefix curriculum requires labels and logits")
        if self.training:
            self.forward_calls.add_(1)
        labels = context.labels
        valid = labels.ne(-100)
        max_depth = labels.size(1)
        active_depth = min(
            max_depth, 1 + int(self.forward_calls.item()) // self.steps_per_depth
        )
        positions = torch.arange(max_depth, device=labels.device).unsqueeze(0)
        active = valid & positions.lt(active_depth)
        if not active.any():
            return token_losses.sum() * 0.0
        loss = token_losses[active].mean()
        with torch.no_grad():
            predictions = context.logits.argmax(dim=-1)
            per_depth = []
            for depth in range(max_depth):
                mask = valid[:, depth]
                if mask.any():
                    per_depth.append(
                        (predictions[:, depth][mask] == labels[:, depth][mask])
                        .float()
                        .mean()
                    )
            self.last_metrics = {
                "active_identifier_depth": float(active_depth),
                "active_token_fraction": _finite_scalar(
                    active.sum() / valid.sum().clamp_min(1)
                ),
                "mean_per_depth_token_accuracy": _finite_scalar(
                    torch.stack(per_depth).mean()
                    if per_depth
                    else loss.detach().new_zeros(())
                ),
            }
        return loss


def _passage_view(
    hidden_states: torch.Tensor, context: FeatureContext
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    extras = dict(context.extras)
    n_passages = int(extras.get("n_passages", 0))
    passage_length = int(extras.get("passage_length", 0))
    if n_passages <= 0 or passage_length <= 0:
        raise ValueError("S17 feature module requires n_passages and passage_length")
    if hidden_states.size(0) % n_passages:
        raise ValueError("encoder batch is not divisible by n_passages")
    batch_size = hidden_states.size(0) // n_passages
    states = hidden_states.reshape(batch_size, n_passages, passage_length, -1)
    if context.attention_mask is None:
        masks = torch.ones(
            batch_size,
            n_passages,
            passage_length,
            dtype=torch.bool,
            device=hidden_states.device,
        )
    else:
        masks = context.attention_mask.reshape(
            batch_size, n_passages, passage_length
        ).bool()
    return states, masks, batch_size, passage_length


def _masked_pool(states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    weights = masks.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


class BiFlowFeatureHook(FeatureHook):
    """Two gated directions between the global prompt and history passages."""

    def __init__(self, initial_logit: float = -2.0):
        super().__init__()
        self.sequence_to_global_logit = nn.Parameter(torch.tensor(initial_logit))
        self.global_to_sequence_logit = nn.Parameter(torch.tensor(initial_logit))
        self.last_metrics: dict[str, float] = {}

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        states, masks, _, _ = _passage_view(hidden_states, context)
        if states.size(1) <= 1:
            return hidden_states
        global_state = _masked_pool(states[:, :1], masks[:, :1]).squeeze(1)
        item_states = _masked_pool(states[:, 1:], masks[:, 1:])
        item_valid = masks[:, 1:].any(dim=-1)
        item_weights = item_valid.unsqueeze(-1).to(states.dtype)
        sequence_state = (item_states * item_weights).sum(dim=1) / item_weights.sum(
            dim=1
        ).clamp_min(1.0)
        s2g_gate = torch.sigmoid(self.sequence_to_global_logit)
        g2s_gate = torch.sigmoid(self.global_to_sequence_logit)
        s2g_delta = s2g_gate * sequence_state
        g2s_delta = g2s_gate * global_state[:, None, :]
        output = states.clone()
        output[:, 0] = output[:, 0] + s2g_delta[:, None, :] * masks[:, 0, :, None]
        output[:, 1:] = output[:, 1:] + g2s_delta[:, :, None, :] * masks[:, 1:, :, None]
        with torch.no_grad():
            alignment = F.cosine_similarity(global_state, sequence_state, dim=-1).mean()
            self.last_metrics = {
                "sequence_to_global_gate": _finite_scalar(s2g_gate),
                "global_to_sequence_gate": _finite_scalar(g2s_gate),
                "sequence_to_global_delta_norm": _finite_scalar(s2g_delta.norm(dim=-1).mean()),
                "global_to_sequence_delta_norm": _finite_scalar(g2s_delta.norm(dim=-1).mean()),
                "representation_alignment": _finite_scalar(alignment),
            }
        return output.reshape_as(hidden_states)


def _load_transition_targets(path: str | Path | None) -> list[int]:
    if not path:
        return [0]
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    values = payload.get("top_next_dense_id")
    if not isinstance(values, list) or not values:
        raise ValueError(f"invalid transition teacher: {resolved}")
    return [int(value) for value in values]


class TransitionTeacherFeatureHook(FeatureHook):
    """Fold-train transition teacher plus recent/long multi-query pooling."""

    def __init__(self, d_model: int, transition_map: str | Path | None = None):
        super().__init__()
        transition_targets = _load_transition_targets(transition_map)
        self.register_buffer(
            "transition_targets", torch.tensor(transition_targets, dtype=torch.long)
        )
        self.teacher_embedding = nn.Embedding(len(transition_targets), d_model, padding_idx=0)
        self.recent_logit = nn.Parameter(torch.tensor(0.0))
        self.long_logit = nn.Parameter(torch.tensor(0.0))
        self.teacher_logit = nn.Parameter(torch.tensor(-1.0))
        self.last_metrics: dict[str, float] = {}

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        states, masks, batch_size, _ = _passage_view(hidden_states, context)
        if states.size(1) <= 1 or context.history_item_ids is None:
            return hidden_states
        item_states = _masked_pool(states[:, 1:], masks[:, 1:])
        item_valid = masks[:, 1:].any(dim=-1)
        history_ids = context.history_item_ids[:, : item_states.size(1)].to(states.device)
        if context.history_item_mask is not None:
            item_valid = item_valid & context.history_item_mask[:, : item_states.size(1)].to(
                states.device
            ).bool()
        # GRAM's frozen default is reverse_history=1, hence column 0 is most recent.
        recent_state = item_states[:, 0]
        valid_float = item_valid.unsqueeze(-1).to(states.dtype)
        long_state = (item_states * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        recent_ids = history_ids[:, 0].clamp(0, self.transition_targets.numel() - 1)
        teacher_ids = self.transition_targets[recent_ids]
        teacher_state = self.teacher_embedding(teacher_ids)
        recent_gate = torch.sigmoid(self.recent_logit)
        long_gate = torch.sigmoid(self.long_logit)
        teacher_gate = torch.sigmoid(self.teacher_logit)
        query = recent_gate * recent_state + long_gate * long_state + teacher_gate * teacher_state
        output = states.clone()
        output[:, 0] = output[:, 0] + query[:, None, :] * masks[:, 0, :, None]
        predicted_available = teacher_ids.ne(0)
        with torch.no_grad():
            self.last_metrics = {
                "transition_teacher_coverage": _finite_scalar(predicted_available.float().mean()),
                "recent_query_gate": _finite_scalar(recent_gate),
                "long_query_gate": _finite_scalar(long_gate),
                "teacher_gate": _finite_scalar(teacher_gate),
                "short_history_fraction": _finite_scalar(item_valid.sum(dim=1).le(3).float().mean()),
            }
        return output.reshape(batch_size * states.size(1), states.size(2), states.size(3))


class ShortcutFiDFeatureHook(FeatureHook):
    """Adds a FiD shortcut from a bounded, semantically coherent history subset.

    The S17-2 fixed-threshold probe connected every history item on Toys.  The
    preregistered S17-3 revision therefore uses a per-user similarity quantile
    and caps the shortcut branch at half of the observed history.  The parent
    FiD branch still sees the complete history.  ``full`` and
    ``random_same_size`` are mechanism controls; the random control is a stable
    function of observed history ids and never reads the target.
    """

    MODES = {"adaptive_semantic", "full", "random_same_size"}

    def __init__(
        self,
        similarity_quantile: float = 0.75,
        max_selected_ratio: float = 0.5,
        initial_logit: float = -1.5,
        selection_mode: str = "adaptive_semantic",
    ):
        super().__init__()
        if not 0.0 <= similarity_quantile <= 1.0:
            raise ValueError("similarity_quantile must be in [0, 1]")
        if not 0.0 < max_selected_ratio < 1.0:
            raise ValueError("max_selected_ratio must be in (0, 1)")
        if selection_mode not in self.MODES:
            raise ValueError(f"unknown shortcut selection mode: {selection_mode}")
        self.similarity_quantile = float(similarity_quantile)
        self.max_selected_ratio = float(max_selected_ratio)
        self.selection_mode = selection_mode
        self.shortcut_logit = nn.Parameter(torch.tensor(initial_logit))
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _largest_component(adjacency: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Histories are at most 20 items in the frozen GRAM protocol, so a
        # deterministic per-sample graph traversal is inexpensive and auditable.
        selected = torch.zeros_like(valid)
        valid_indices = valid.nonzero(as_tuple=False).flatten().tolist()
        components: list[list[int]] = []
        unseen = set(valid_indices)
        while unseen:
            root = min(unseen)
            stack = [root]
            unseen.remove(root)
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                neighbours = adjacency[node].nonzero(as_tuple=False).flatten().tolist()
                for neighbour in neighbours:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        stack.append(neighbour)
            components.append(sorted(component))
        if components:
            winner = sorted(components, key=lambda value: (-len(value), value))[0]
            selected[winner] = True
        return selected

    def _adaptive_selection(
        self, similarities: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        selected = torch.zeros_like(valid)
        valid_indices = valid.nonzero(as_tuple=False).flatten()
        count = int(valid_indices.numel())
        if count == 0:
            return selected
        if count == 1:
            selected[valid_indices] = True
            return selected

        valid_similarities = similarities[valid_indices][:, valid_indices]
        upper = torch.triu_indices(count, count, offset=1, device=similarities.device)
        pairwise = valid_similarities[upper[0], upper[1]]
        threshold = torch.quantile(pairwise, self.similarity_quantile)
        adjacency = similarities.ge(threshold)
        adjacency = adjacency & valid[:, None] & valid[None, :]
        adjacency.fill_diagonal_(True)
        component = self._largest_component(adjacency, valid)

        # The shortcut is an extra evidence branch, not a replacement for the
        # full-history FiD parent.  A strict cap guarantees that the revised
        # mechanism actually filters at least one item whenever count > 1.
        budget = max(1, int(math.ceil(count * self.max_selected_ratio)))
        component_indices = component.nonzero(as_tuple=False).flatten().tolist()
        if len(component_indices) <= budget:
            selected[component_indices] = True
            return selected

        ranked = []
        for index in component_indices:
            cohesion = similarities[index, component].mean()
            ranked.append((-float(cohesion.detach().cpu()), index))
        selected[[index for _, index in sorted(ranked)[:budget]]] = True
        return selected

    @staticmethod
    def _stable_random_same_size(
        valid: torch.Tensor, history_ids: torch.Tensor | None, size: int
    ) -> torch.Tensor:
        selected = torch.zeros_like(valid)
        candidates = valid.nonzero(as_tuple=False).flatten().tolist()
        ranked = []
        for position in candidates:
            item_id = int(history_ids[position]) if history_ids is not None else position
            key = (
                (item_id + 1) * 2654435761 + (position + 1) * 2246822519
            ) % (2**32)
            ranked.append((key, position))
        selected[[position for _, position in sorted(ranked)[:size]]] = True
        return selected

    def forward(self, hidden_states: torch.Tensor, context: FeatureContext) -> torch.Tensor:
        states, masks, _, _ = _passage_view(hidden_states, context)
        if states.size(1) <= 1:
            return hidden_states
        item_states = _masked_pool(states[:, 1:], masks[:, 1:])
        item_valid = masks[:, 1:].any(dim=-1)
        normalized = F.normalize(item_states.float(), dim=-1)
        similarities = torch.matmul(normalized, normalized.transpose(1, 2))
        adaptive_rows = []
        for row in range(states.size(0)):
            adaptive_rows.append(
                self._adaptive_selection(similarities[row], item_valid[row])
            )
        adaptive = torch.stack(adaptive_rows)
        if self.selection_mode == "adaptive_semantic":
            selected = adaptive
        elif self.selection_mode == "full":
            selected = item_valid
        else:
            selected_rows = []
            for row in range(states.size(0)):
                history_ids = None
                if context.history_item_ids is not None:
                    history_ids = context.history_item_ids[row, : item_states.size(1)]
                selected_rows.append(
                    self._stable_random_same_size(
                        item_valid[row], history_ids, int(adaptive[row].sum())
                    )
                )
            selected = torch.stack(selected_rows)
        weights = selected.unsqueeze(-1).to(states.dtype)
        shortcut = (item_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        gate = torch.sigmoid(self.shortcut_logit)
        output = states.clone()
        output[:, 0] = output[:, 0] + gate * shortcut[:, None, :] * masks[:, 0, :, None]
        with torch.no_grad():
            valid_counts = item_valid.sum(dim=1).clamp_min(1)
            selected_ratio = selected.sum(dim=1) / valid_counts
            self.last_metrics = {
                "selected_history_ratio": _finite_scalar(selected_ratio.float().mean()),
                "adaptive_target_ratio": _finite_scalar(
                    (adaptive.sum(dim=1) / valid_counts).float().mean()
                ),
                "shortcut_gate": _finite_scalar(gate),
                "noise_filtered_ratio": _finite_scalar((1.0 - selected_ratio.float()).mean()),
                "long_history_fraction": _finite_scalar(valid_counts.ge(10).float().mean()),
            }
        return output.reshape_as(hidden_states)


def scalar_metrics(module: nn.Module) -> dict[str, float]:
    metrics: dict[str, float] = {}
    raw: Any = getattr(module, "last_metrics", None)
    if isinstance(raw, dict):
        for name, value in raw.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1 and torch.isfinite(value):
                    metrics[name] = _finite_scalar(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[name] = float(value)
    return metrics
