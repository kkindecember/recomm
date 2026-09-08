"""Legal-sibling and complete lexical-path objectives; no model parameters."""
from __future__ import annotations

import numpy as np
import torch

from experiment.phase18.core.s2_contracts import ARMS


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def pcrf_scores(sequence_scores, cf_scores, frequencies, q1):
    """Frozen Phase9 PCRF; input order is native beam order."""
    reliability = 1.0 - float(np.mean(np.asarray(frequencies)[:10] <= q1))
    corrected = zscore(zscore(cf_scores) - 0.5 * zscore(np.log1p(frequencies)))
    return zscore(sequence_scores) + reliability * corrected, reliability


def mine_prefix_nodes(target, item_paths, beam_items, beam_scores, cf, freq, item_index, k=8):
    """Target selects the legal sibling set only; teacher models are frozen.

    Generic mining never reads cf/freq: parent beam rank then lexical order.
    CF mining uses candidate PCRF rank, followed by full-catalog corrected CF.
    """
    beam_score = dict(zip(beam_items, beam_scores))
    if cf is not None:
        # Reliability is supplied by the original beam50, as in frozen PCRF.
        cf_values, reliability = cf
        positions = [item_index[i] - 1 for i in beam_items]
        corrected = zscore(zscore(cf_values[positions]) - 0.5 * zscore(np.log1p(freq[positions])))
        rank_score = dict(zip(beam_items, zscore(beam_scores) + reliability * corrected))
        global_cf = zscore(zscore(cf_values) - 0.5 * zscore(np.log1p(freq)))
    else:
        rank_score = beam_score
        global_cf = None
    nodes = []
    for depth, target_child in enumerate(target):
        prefix = target[:depth]
        siblings = [raw for raw, path in item_paths.items()
                    if len(path) > depth and path[:depth] == prefix and path[depth] != target_child]
        if not siblings:
            continue
        sibling_set = set(siblings)
        candidates = sorted((raw for raw in beam_items if raw in sibling_set),
                            key=lambda raw: (-float(rank_score[raw]), raw))[:k]
        remainder = [raw for raw in siblings if raw not in candidates]
        remainder.sort(key=(lambda raw: (-float(global_cf[item_index[raw] - 1]), raw))
                       if global_cf is not None else (lambda raw: raw))
        candidates += remainder[:k - len(candidates)]
        children = sorted({item_paths[raw][depth] for raw in candidates})
        nodes.append({"depth": depth, "target_child": target_child,
                      "negative_children": children, "negative_items": candidates})
    return nodes


def path_weights(parent_scores, target_cf_pc=None, negative_cf_pc=None, reliability=0.0):
    values = np.asarray(parent_scores, dtype=np.float64)
    if not len(values):
        return []
    weights = np.exp(values - values.max())
    if target_cf_pc is not None:
        delta = np.asarray(negative_cf_pc) - target_cf_pc
        sigmoid = np.exp(-np.logaddexp(0.0, -delta))
        weights *= 1.0 + reliability * sigmoid
    weights /= weights.sum()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise FloatingPointError("invalid frozen path weights")
    return weights.tolist()


def prefix_loss(logits, nodes, margin=0.0):
    terms = []
    for node in nodes:
        d, target = node["depth"], node["target_child"]
        children = node["negative_children"]
        if target in children or not children:
            raise ValueError("prefix loss requires wrong legal siblings")
        differences = logits[d, children].float() - logits[d, target].float() + margin
        terms.append(torch.logsumexp(torch.cat([differences.new_zeros(1), differences]), dim=0))
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def full_path_loss(scores, weights, margin=0.0):
    if scores.numel() == 1:
        return scores.sum() * 0.0
    w = scores.new_tensor(weights)
    if w.numel() != scores.numel() - 1 or not bool(torch.isfinite(w).all()) or not bool((w > 0).all()):
        raise ValueError("path weights must be finite positive and aligned")
    differences = margin + scores[1:] - scores[0] + w.log()
    return torch.logsumexp(torch.cat([scores.new_zeros(1), differences]), dim=0)


def combine_loss(ce, alpha, logits=None, scores=None, cache=None, arm="M0_PCPS"):
    if arm not in ARMS:
        raise ValueError("unknown S18-2 arm")
    if alpha == 0 or arm == "C0_CONT":
        # Exact identity: do not access negative cache or execute auxiliary forward.
        return ce, {"ce": ce}
    prefix = prefix_loss(logits, cache["nodes"])
    path = full_path_loss(scores, cache["weights"])
    auxiliary = 0.5 * prefix + 0.5 * path
    return ce + alpha * auxiliary, {"ce": ce, "prefix": prefix, "path": path, "auxiliary": auxiliary}
