"""Architecture contracts for the Stage17 LATTE/PSID full port.

The conflict-resolution and latent-path semantics follow the public LATTE
method at commit 05e4e6d983225bcb7172f148a076890e80c524d1 (MIT).  This module is
an independent project implementation; it does not import the official source.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PSIDResolutionSummary:
    catalog_items: int
    n_digit: int
    codebook_size: int
    collision_groups: int
    collisions_before: int
    collisions_after: int
    reassigned_items: int
    top_k_per_digit: int
    collision_suffix_size: int = 0


@dataclass(frozen=True)
class LatteBeamPath:
    tokens: tuple[int, ...]
    log_score: float


@dataclass(frozen=True)
class LatteItemPrediction:
    item_id: str
    score: float
    path_count: int


def _validate_rq_inputs(
    item_to_codes: Mapping[str, Sequence[int]], centroids: np.ndarray
) -> tuple[int, int]:
    if not item_to_codes:
        raise ValueError("PSID resolution requires a non-empty catalog")
    if centroids.ndim != 3:
        raise ValueError("RQ centroids must have shape [digit, code, embedding]")
    n_digit, codebook_size, _ = centroids.shape
    if n_digit <= 0 or codebook_size <= 1:
        raise ValueError("invalid RQ centroid shape")
    for item, codes in item_to_codes.items():
        if len(codes) != n_digit:
            raise ValueError(f"item {item} has {len(codes)} digits, expected {n_digit}")
        if any(code < 0 or code >= codebook_size for code in codes):
            raise ValueError(f"item {item} contains an out-of-range RQ code")
    return n_digit, codebook_size


def resolve_rqkmeans_psid_conflicts(
    item_to_codes: Mapping[str, Sequence[int]],
    centroids: np.ndarray,
    *,
    top_k_per_digit: int = 5,
) -> tuple[dict[str, tuple[int, ...]], PSIDResolutionSummary]:
    """Resolve RQ collisions by the nearest unused reconstructed code tuple.

    The first catalog item retains the original tuple.  Every later item in a
    collision group searches the Cartesian product of the closest centroid
    alternatives in each digit and takes the unused tuple with minimum
    reconstruction drift.  A missing candidate fails closed instead of adding
    a collision suffix.
    """

    n_digit, codebook_size = _validate_rq_inputs(item_to_codes, centroids)
    if top_k_per_digit <= 0:
        raise ValueError("top_k_per_digit must be positive")
    top_k = min(int(top_k_per_digit), codebook_size)

    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    normalized: dict[str, tuple[int, ...]] = {}
    for item, codes in item_to_codes.items():
        code_tuple = tuple(int(code) for code in codes)
        normalized[item] = code_tuple
        groups[code_tuple].append(item)

    collision_groups = {codes: items for codes, items in groups.items() if len(items) > 1}
    collisions_before = sum(len(items) - 1 for items in collision_groups.values())
    used = set(normalized.values())
    resolved = dict(normalized)
    reassigned = 0

    for original_codes, items in collision_groups.items():
        original_reconstruction = np.zeros(centroids.shape[-1], dtype=np.float64)
        closest_by_digit: list[list[int]] = []
        for digit, original_code in enumerate(original_codes):
            original_centroid = centroids[digit, original_code]
            original_reconstruction += original_centroid
            distances = np.square(centroids[digit] - original_centroid[None, :]).sum(axis=1)
            closest_by_digit.append(
                sorted(range(codebook_size), key=lambda code: (float(distances[code]), code))[:top_k]
            )

        for item in items[1:]:
            best_codes: tuple[int, ...] | None = None
            best_distance = float("inf")
            for candidate in product(*closest_by_digit):
                candidate_tuple = tuple(int(code) for code in candidate)
                if candidate_tuple in used:
                    continue
                reconstruction = np.zeros(centroids.shape[-1], dtype=np.float64)
                for digit, code in enumerate(candidate_tuple):
                    reconstruction += centroids[digit, code]
                distance = float(np.square(original_reconstruction - reconstruction).sum())
                if (distance, candidate_tuple) < (best_distance, best_codes or candidate_tuple):
                    best_distance = distance
                    best_codes = candidate_tuple
            if best_codes is None:
                raise RuntimeError(
                    f"no conflict-free PSID candidate for {item}; "
                    f"increase audited top_k from {top_k}"
                )
            resolved[item] = best_codes
            used.add(best_codes)
            reassigned += 1

    collisions_after = len(resolved) - len(set(resolved.values()))
    if collisions_after:
        raise AssertionError("PSID resolution left collision aliases")
    summary = PSIDResolutionSummary(
        catalog_items=len(resolved),
        n_digit=n_digit,
        codebook_size=codebook_size,
        collision_groups=len(collision_groups),
        collisions_before=collisions_before,
        collisions_after=collisions_after,
        reassigned_items=reassigned,
        top_k_per_digit=top_k,
    )
    return resolved, summary


class LattePathCodec:
    """Conflict-free PSID token layout plus a latent-root decoding forest."""

    padding_token = 0

    def __init__(
        self,
        item_to_codes: Mapping[str, Sequence[int]],
        *,
        codebook_sizes: Sequence[int],
        n_latent_tokens: int = 8,
        n_user_tokens: int = 1,
    ) -> None:
        if n_latent_tokens <= 0 or n_user_tokens < 0:
            raise ValueError("invalid LATTE latent/user token count")
        if not codebook_sizes or any(size <= 1 for size in codebook_sizes):
            raise ValueError("invalid semantic codebook sizes")
        self.item_ids = tuple(item_to_codes)
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        self.n_digit = len(self.codebook_sizes)
        self.n_latent_tokens = int(n_latent_tokens)
        self.n_user_tokens = int(n_user_tokens)
        self.base_latent_token = 1
        self.base_semantic_token = self.base_latent_token + self.n_latent_tokens
        offsets = []
        running = self.base_semantic_token
        for size in self.codebook_sizes:
            offsets.append(running)
            running += size
        self.semantic_offsets = tuple(offsets)
        self.base_user_token = running
        self.eos_token = self.base_user_token + self.n_user_tokens
        self.vocab_size = self.eos_token + 1

        semantic_by_item: dict[str, tuple[int, ...]] = {}
        item_by_semantic: dict[tuple[int, ...], str] = {}
        for item, raw_codes in item_to_codes.items():
            if len(raw_codes) != self.n_digit:
                raise ValueError(f"item {item} has the wrong semantic length")
            tokens = tuple(
                self.semantic_offsets[digit] + int(code)
                for digit, code in enumerate(raw_codes)
            )
            for digit, token in enumerate(tokens):
                lower = self.semantic_offsets[digit]
                upper = lower + self.codebook_sizes[digit]
                if token < lower or token >= upper:
                    raise ValueError(f"item {item} has an out-of-range semantic token")
            if tokens in item_by_semantic:
                raise ValueError(
                    f"semantic alias between {item_by_semantic[tokens]} and {item}"
                )
            semantic_by_item[item] = tokens
            item_by_semantic[tokens] = item
        if not semantic_by_item:
            raise ValueError("LATTE codec requires a non-empty catalog")
        self.semantic_by_item = semantic_by_item
        self.item_by_semantic = item_by_semantic
        self.legal_next = self._build_forest()

    @property
    def latent_tokens(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.base_latent_token,
                self.base_latent_token + self.n_latent_tokens,
            )
        )

    def _build_forest(self) -> dict[tuple[int, ...], tuple[int, ...]]:
        children: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for latent in self.latent_tokens:
            for semantic in self.semantic_by_item.values():
                path = (latent,) + semantic + (self.eos_token,)
                for position, token in enumerate(path):
                    children[path[:position]].add(token)
        return {
            prefix: tuple(sorted(tokens)) for prefix, tokens in children.items()
        }

    def sample_training_target(
        self, item_id: str, *, rng: random.Random
    ) -> tuple[int, ...]:
        if item_id not in self.semantic_by_item:
            raise KeyError(item_id)
        latent = self.latent_tokens[rng.randrange(self.n_latent_tokens)]
        return (latent,) + self.semantic_by_item[item_id] + (self.eos_token,)

    def decode_path(self, path: Sequence[int]) -> str | None:
        tokens = tuple(int(token) for token in path)
        if len(tokens) == self.n_digit + 2 and tokens[-1] == self.eos_token:
            tokens = tokens[:-1]
        if len(tokens) != self.n_digit + 1 or tokens[0] not in self.latent_tokens:
            return None
        return self.item_by_semantic.get(tokens[1:])

    def assert_legal_path(self, path: Sequence[int]) -> None:
        prefix: tuple[int, ...] = ()
        for token in path:
            if int(token) not in self.legal_next.get(prefix, ()):
                raise ValueError(f"illegal LATTE token {token} after prefix {prefix}")
            prefix = prefix + (int(token),)

    def aggregate_paths(
        self,
        paths: Iterable[LatteBeamPath],
        *,
        method: str = "agg_max",
        top_k: int = 50,
    ) -> list[LatteItemPrediction]:
        if method not in {"agg_max", "agg_sum"}:
            raise ValueError(f"unknown LATTE aggregation method: {method}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores: dict[str, float] = {}
        counts: dict[str, int] = defaultdict(int)
        for prediction in paths:
            item = self.decode_path(prediction.tokens)
            if item is None:
                continue
            score = float(prediction.log_score)
            counts[item] += 1
            if item not in scores:
                scores[item] = score
            elif method == "agg_max":
                scores[item] = max(scores[item], score)
            else:
                high = max(scores[item], score)
                low = min(scores[item], score)
                scores[item] = high + math.log1p(math.exp(low - high))
        ranked = sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:top_k]
        return [
            LatteItemPrediction(item_id=item, score=score, path_count=counts[item])
            for item, score in ranked
        ]
