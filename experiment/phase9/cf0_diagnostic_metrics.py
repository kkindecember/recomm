"""Pure metric helpers for CF0 diagnostics."""

import torch


def rank_from_logits(logits, targets):
    logits = logits.clone()
    logits[:, 0] = -torch.inf
    target_scores = logits.gather(1, targets[:, None])
    return 1 + (logits > target_scores).sum(dim=1)


def item_metrics_from_ranks(ranks):
    ranks = torch.as_tensor(ranks, dtype=torch.float64)
    result = {"count": int(ranks.numel()), "mean_rank": float(ranks.mean())}
    result["median_rank"] = float(ranks.median())
    result["mrr"] = float((1.0 / ranks).mean())
    for cutoff in (1, 5, 10, 20, 50):
        hits = ranks <= cutoff
        result[f"Recall@{cutoff}"] = float(hits.double().mean())
        result[f"NDCG@{cutoff}"] = float(
            torch.where(hits, 1.0 / torch.log2(ranks + 1.0), 0.0).mean()
        )
    return result
