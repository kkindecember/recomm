"""Single item-level aggregation path shared by all Stage 17 tracks."""

from __future__ import annotations

from collections import defaultdict

import torch


SUPPORTED = {"max", "logsumexp", "sum_probability"}


def aggregate_item_scores(
    item_ids: list[str], path_log_scores: torch.Tensor, method: str = "logsumexp"
) -> dict[str, torch.Tensor]:
    if method not in SUPPORTED:
        raise ValueError(f"unsupported item aggregation: {method}")
    if path_log_scores.ndim != 1 or len(item_ids) != path_log_scores.numel():
        raise ValueError("one scalar path score is required for every item id")
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for item_id, score in zip(item_ids, path_log_scores):
        grouped[item_id].append(score)
    result: dict[str, torch.Tensor] = {}
    for item_id, values in grouped.items():
        stacked = torch.stack(values)
        if method == "max":
            result[item_id] = stacked.max()
        elif method == "logsumexp":
            result[item_id] = torch.logsumexp(stacked, dim=0)
        else:
            result[item_id] = stacked.exp().sum()
    return result


def ranked_items(scores: dict[str, torch.Tensor]) -> list[str]:
    return sorted(scores, key=lambda item: (-float(scores[item]), item))


def assert_k1_equivalence(item_ids: list[str], path_log_scores: torch.Tensor) -> None:
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("K=1 equivalence requires one path per item")
    aggregated = aggregate_item_scores(item_ids, path_log_scores, "logsumexp")
    reconstructed = torch.stack([aggregated[item] for item in item_ids])
    if not torch.equal(reconstructed, path_log_scores):
        raise AssertionError("K=1 item aggregation did not exactly preserve path scores")
