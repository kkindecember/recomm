"""Batched equivalent of the original GACR six-feature construction."""
import torch


def vector_features(logits, pooled, weights, union_indices, gram_ranks, catalog_ranks):
    finite = logits[torch.isfinite(logits)]
    mean = finite.mean()
    std = finite.std(unbiased=False).clamp_min(1e-6)
    values = logits[union_indices]
    zscore = torch.where(torch.isfinite(values), ((values-mean)/std).clamp(-10,10), torch.full_like(values,-10))
    vectors = weights[union_indices]
    cosine = torch.mv(vectors, pooled) / (
        torch.linalg.vector_norm(pooled).clamp_min(1e-6) * torch.linalg.vector_norm(vectors,dim=1).clamp_min(1e-6))
    gram = torch.as_tensor(gram_ranks, dtype=logits.dtype, device=logits.device)
    catalog = torch.as_tensor(catalog_ranks, dtype=logits.dtype, device=logits.device)
    features = torch.stack([zscore, torch.where(gram>0,1/gram.clamp_min(1),torch.zeros_like(gram)),
        torch.where(catalog>0,1/catalog.clamp_min(1),torch.zeros_like(catalog)), (gram>0).to(logits.dtype),
        (catalog>0).to(logits.dtype), cosine],dim=1)
    if not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite GACR features')
    return features
