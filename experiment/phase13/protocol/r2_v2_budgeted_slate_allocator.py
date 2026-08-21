"""R²-v2 cross-domain budget-conditioned slate allocator (CBSA).

This module implements the frozen Phase-13 preregistration.  Toys and Beauty
are source/development domains and are evaluated with user-level five-fold OOF
predictions.  Sports and every test split are rejected before any configured
file is opened.

Feature extraction is target-free.  Targets are attached only after features
have been built, to construct the three deterministic action rewards and the
aggregate warm constraint.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .route_resolve import (
        ResidualUserProjector,
        decode_lexical_id,
        parse_gram_predictions,
        read_key_value_lines,
        recency_weighted_history,
    )
except ImportError:  # Direct script execution.
    from route_resolve import (  # type: ignore
        ResidualUserProjector,
        decode_lexical_id,
        parse_gram_predictions,
        read_key_value_lines,
        recency_weighted_history,
    )


ACTIONS = ("a0", "a2", "a3")
BUDGETS = (0.93, 0.95, 0.97, 0.99)
PRIMARY_BUDGET = 0.97
FOLDS = 5
FOLD_SALT = "GRAM_PHASE13_R2_V2_CBSA_SOURCE_OOF_V1_20260819"
SEED = 20260819
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260819
RRF_K = 60

# Every name describes state available before the next-item target is known.
FEATURE_NAMES = (
    "v0_score_rank1",
    "v0_score_rank2",
    "v0_score_rank3",
    "v0_score_gap_1_2",
    "v0_score_gap_2_3",
    "v0_score_gap_7_8",
    "v0_score_gap_8_9",
    "v0_score_mean_top10",
    "v0_score_std_top10",
    "v0_score_entropy_top10",
    "resolver_cosine_rank1",
    "resolver_cosine_rank2",
    "resolver_cosine_rank3",
    "resolver_cosine_margin_1_2",
    "resolver_cosine_margin_2_3",
    "resolver_cosine_mean_top10",
    "resolver_cosine_std_top10",
    "resolver_cosine_entropy_top10",
    "overlap_count_top10",
    "overlap_count_top50",
    "jaccard_top10",
    "rank_agreement_spearman",
    "rrf_agreement",
    "resolver_cold_count_top10",
    "resolver_cold_ratio_top10",
    "resolver_cold_count_top50",
    "resolver_cold_ratio_top50",
    "usable_unique_candidates_a2",
    "usable_unique_candidates_a3",
    "history_length",
    "history_cold_count",
    "history_warm_count",
    "history_cold_ratio",
    "history_candidate_similarity_mean_top3",
    "history_candidate_similarity_std_top3",
    "history_candidate_similarity_max_top3",
)

FORBIDDEN_FEATURE_TOKENS = ("target", "label", "reward", "oracle", "ndcg", "hit")
TEST_TOKEN = re.compile(r"(^|[_\-.])test([_\-.]|$)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True,
        choices=("freeze-preflight", "verify-preflight", "run-source"),
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--canonical-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def read_sequences(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with path.open() as handle:
        for line_number, raw in enumerate(handle, 1):
            parts = raw.strip().split()
            if len(parts) < 4:
                continue
            uid, items = parts[0], parts[1:]
            if uid in rows:
                raise ValueError(f"Duplicate user in {path}:{line_number}: {uid}")
            rows[uid] = items
    if not rows:
        raise ValueError(f"No sequences parsed from {path}")
    return rows


def guard_source_path(path: Path) -> None:
    """Reject confirmation-domain and test paths before opening them."""
    parts = [part.casefold() for part in path.parts]
    if any("sports" in part for part in parts):
        raise ValueError(f"Sports guard: refusing source-stage path before open: {path}")
    if any(TEST_TOKEN.search(part) for part in parts):
        raise ValueError(f"Test guard: refusing source-stage path before open: {path}")


def resolve_source_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    guard_source_path(path)
    return path.resolve()


def stable_fold(user_id: str, domain: str, salt: str = FOLD_SALT, folds: int = FOLDS) -> int:
    payload = f"{salt}|{domain}|{user_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _candidate_items(
    v0_items: Sequence[str],
    resolver_items: Sequence[str],
    size: int,
    catalog: set[str],
    cold_items: set[str],
) -> list[str]:
    """Return the frozen B1 candidate prefix for an action.

    B1 constructs one shared three-item pool from catalog-cold resolver items
    outside the v0 top-7 anchor, then portfolio@2 uses its first two items.
    Keeping the anchor fixed here is essential: using top-(10-size) while
    filtering candidates would define a different portfolio@2 incumbent.
    """
    protected = set(unique_in_order(v0_items)[:7])
    return [
        item for item in unique_in_order(resolver_items)
        if item in catalog and item in cold_items and item not in protected
    ][:size]


def build_action_ranking(
    v0_items: Sequence[str],
    resolver_items: Sequence[str],
    requested_action: str,
    catalog: set[str],
    cold_items: set[str],
    top_k: int = 50,
) -> tuple[str, list[str]]:
    """Build a deterministic exact-item action, degrading a3 -> a2 -> a0."""
    if requested_action not in ACTIONS:
        raise ValueError(f"Unsupported action: {requested_action}")
    v0 = [item for item in unique_in_order(v0_items) if item in catalog]
    resolver = [item for item in unique_in_order(resolver_items) if item in catalog]
    if requested_action == "a0":
        ranking = unique_in_order([*v0, *resolver])[:top_k]
        return "a0", ranking
    size = 3 if requested_action == "a3" else 2
    candidates = _candidate_items(v0, resolver, size, catalog, cold_items)
    if len(candidates) < size:
        return build_action_ranking(
            v0,
            resolver,
            "a2" if requested_action == "a3" else "a0",
            catalog,
            cold_items,
            top_k,
        )
    anchor = 10 - size
    ranking = unique_in_order([*v0[:anchor], *candidates, *v0[anchor:], *resolver])[:top_k]
    if len(ranking) != len(set(ranking)) or not set(ranking) <= catalog:
        raise RuntimeError("Action output is not an exact-item unique catalog ranking")
    return requested_action, ranking


def safe_argmax(logits: torch.Tensor) -> torch.Tensor:
    """torch.argmax's first-index behavior implements a0 > a2 > a3 ties."""
    if logits.ndim != 2 or logits.shape[1] != len(ACTIONS):
        raise ValueError(f"Expected [N,3] logits, got {tuple(logits.shape)}")
    return torch.argmax(logits, dim=1)


def hit_and_ndcg(ranking: Sequence[str], target: str, k: int) -> tuple[float, float]:
    try:
        rank = list(ranking[:k]).index(target) + 1
    except ValueError:
        return 0.0, 0.0
    return 1.0, 1.0 / math.log2(rank + 1)


def _nan_at(values: Sequence[float], index: int) -> float:
    return float(values[index]) if index < len(values) else math.nan


def _finite(values: Sequence[float], limit: int | None = None) -> np.ndarray:
    chosen = list(values if limit is None else values[:limit])
    return np.asarray([value for value in chosen if math.isfinite(value)], dtype=np.float64)


def _mean(values: Sequence[float], limit: int) -> float:
    array = _finite(values, limit)
    return float(array.mean()) if array.size else math.nan


def _std(values: Sequence[float], limit: int) -> float:
    array = _finite(values, limit)
    return float(array.std()) if array.size else math.nan


def _entropy(values: Sequence[float], limit: int) -> float:
    array = _finite(values, limit)
    if not array.size:
        return math.nan
    shifted = array - array.max()
    probability = np.exp(shifted)
    probability /= probability.sum()
    return float(-(probability * np.log(probability + 1e-12)).sum())


def _spearman_on_overlap(left: Sequence[str], right: Sequence[str], k: int = 50) -> float:
    left_rank = {item: rank for rank, item in enumerate(unique_in_order(left)[:k], 1)}
    right_rank = {item: rank for rank, item in enumerate(unique_in_order(right)[:k], 1)}
    common = sorted(set(left_rank) & set(right_rank))
    if len(common) < 2:
        return math.nan
    x = np.asarray([left_rank[item] for item in common], dtype=np.float64)
    y = np.asarray([right_rank[item] for item in common], dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _rrf_agreement(left: Sequence[str], right: Sequence[str], k: int = 50) -> float:
    left_rank = {item: rank for rank, item in enumerate(unique_in_order(left)[:k], 1)}
    right_rank = {item: rank for rank, item in enumerate(unique_in_order(right)[:k], 1)}
    return float(sum(
        1.0 / (RRF_K + left_rank[item]) + 1.0 / (RRF_K + right_rank[item])
        for item in set(left_rank) & set(right_rank)
    ))


def extract_features(
    v0_items: Sequence[str],
    v0_scores: Sequence[float],
    resolver_items: Sequence[str],
    resolver_scores: Sequence[float],
    cold_items: set[str],
    catalog: set[str],
    history_items: Sequence[str],
    history_candidate_similarities: Sequence[float],
) -> list[float]:
    """Construct the frozen target-free feature vector."""
    v0 = unique_in_order(v0_items)
    resolver = unique_in_order(resolver_items)
    top10_left, top10_right = set(v0[:10]), set(resolver[:10])
    top50_left, top50_right = set(v0[:50]), set(resolver[:50])
    cold10 = sum(item in cold_items for item in resolver[:10])
    cold50 = sum(item in cold_items for item in resolver[:50])
    history_cold = sum(item in cold_items for item in history_items)
    history_warm = sum(item in catalog and item not in cold_items for item in history_items)
    candidate_sims = _finite(history_candidate_similarities, 3)
    usable2 = len(_candidate_items(v0, resolver, 2, catalog, cold_items))
    usable3 = len(_candidate_items(v0, resolver, 3, catalog, cold_items))
    values = [
        _nan_at(v0_scores, 0), _nan_at(v0_scores, 1), _nan_at(v0_scores, 2),
        _nan_at(v0_scores, 0) - _nan_at(v0_scores, 1),
        _nan_at(v0_scores, 1) - _nan_at(v0_scores, 2),
        _nan_at(v0_scores, 6) - _nan_at(v0_scores, 7),
        _nan_at(v0_scores, 7) - _nan_at(v0_scores, 8),
        _mean(v0_scores, 10), _std(v0_scores, 10), _entropy(v0_scores, 10),
        _nan_at(resolver_scores, 0), _nan_at(resolver_scores, 1),
        _nan_at(resolver_scores, 2),
        _nan_at(resolver_scores, 0) - _nan_at(resolver_scores, 1),
        _nan_at(resolver_scores, 1) - _nan_at(resolver_scores, 2),
        _mean(resolver_scores, 10), _std(resolver_scores, 10),
        _entropy(resolver_scores, 10),
        float(len(top10_left & top10_right)), float(len(top50_left & top50_right)),
        float(len(top10_left & top10_right) / max(len(top10_left | top10_right), 1)),
        _spearman_on_overlap(v0, resolver), _rrf_agreement(v0, resolver),
        float(cold10), cold10 / max(len(resolver[:10]), 1),
        float(cold50), cold50 / max(len(resolver[:50]), 1),
        float(usable2), float(usable3), float(len(history_items)),
        float(history_cold), float(history_warm),
        history_cold / max(history_cold + history_warm, 1),
        float(candidate_sims.mean()) if candidate_sims.size else math.nan,
        float(candidate_sims.std()) if candidate_sims.size else math.nan,
        float(candidate_sims.max()) if candidate_sims.size else math.nan,
    ]
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("Feature schema/value length mismatch")
    return [value if math.isfinite(value) else math.nan for value in values]


@dataclass(frozen=True)
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "Standardizer":
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("Unexpected raw feature shape")
        missing = torch.isnan(values)
        counts = (~missing).sum(0).clamp_min(1)
        clean = torch.where(missing, torch.zeros_like(values), values)
        mean = clean.sum(0) / counts
        centered = torch.where(missing, torch.zeros_like(values), values - mean)
        variance = (centered.square().sum(0) / counts).clamp_min(0)
        scale = variance.sqrt()
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        missing = torch.isnan(values)
        filled = torch.where(missing, self.mean, values)
        normalized = (filled - self.mean) / self.scale
        return torch.cat([normalized, missing.float()], dim=1)

    def as_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


class BudgetConditionedAllocator(nn.Module):
    def __init__(self, base_input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(base_input_dim + 1, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, len(ACTIONS)),
        )

    def forward(self, features: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        if budget.ndim == 1:
            budget = budget[:, None]
        return self.net(torch.cat([features, budget], dim=1))


@dataclass
class SourceRecord:
    domain: str
    user_id: str
    fold: int
    raw_features: list[float]
    target: str
    is_cold: bool
    action_rewards: list[float]
    action_rankings: list[list[str]]
    effective_actions: list[str]
    portfolio2_ranking: list[str]


def _ranked_scores(items: Sequence[str], score_by_item: dict[str, float]) -> list[float]:
    return [float(score_by_item.get(item, math.nan)) for item in items]


def _load_resolver(
    checkpoint_path: Path, embeddings: torch.Tensor, device: torch.device
) -> ResidualUserProjector:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ResidualUserProjector(
        int(checkpoint["dim"]), int(checkpoint["hidden_dim"]), float(checkpoint["dropout"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    if checkpoint["dim"] != embeddings.shape[1]:
        raise ValueError("Resolver/embedding dimension mismatch")
    return model


def _source_paths(project_root: Path, source_config: dict) -> dict[str, Path]:
    paths = {
        key: resolve_source_path(project_root, value)
        for key, value in source_config.items()
    }
    dataset_dir = paths["dataset_dir"]
    paths["user_sequence"] = resolve_source_path(project_root, str(dataset_dir / "user_sequence.txt"))
    return paths


def _decoded_score_maps(
    gram_path: Path, item_to_lexical: dict[str, str]
) -> dict[str, dict[str, float]]:
    decoded_to_item: dict[str, str] = {}
    for item, lexical in item_to_lexical.items():
        decoded = decode_lexical_id(lexical)
        if decoded in decoded_to_item:
            raise ValueError(f"Decoded ID collision in frozen source catalog: {decoded!r}")
        decoded_to_item[decoded] = item
    rows = parse_gram_predictions(gram_path)
    output: dict[str, dict[str, float]] = {}
    for uid, row in rows.items():
        score_map: dict[str, float] = {}
        for decoded, score in zip(row["predictions"], row["scores"]):
            item = decoded_to_item.get(decoded)
            if item is None:
                raise KeyError(f"Unmapped legal GRAM prediction for {uid}: {decoded!r}")
            score_map.setdefault(item, float(score))
        output[uid] = score_map
    return output


def audit_source(
    project_root: Path, domain: str, source_config: dict, include_hashes: bool = True
) -> tuple[dict, dict[str, Path]]:
    """Perform cache/schema alignment without producing efficacy metrics."""
    if domain not in {"Toys", "Beauty"}:
        raise ValueError(f"Source-stage domain is not allowed: {domain}")
    paths = _source_paths(project_root, source_config)
    for path in paths.values():
        guard_source_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    p0_config = read_json(paths["p0_config"])
    if p0_config.get("split") != "validation" or p0_config.get("test_predictions_opened") is not False:
        raise ValueError(f"{domain} P0 is not a sealed validation artifact")
    gram_path = Path(p0_config["gram_validation_predictions"])
    guard_source_path(gram_path)
    if not gram_path.is_file():
        raise FileNotFoundError(gram_path)
    paths["gram_validation_predictions"] = gram_path.resolve()

    p0_rows = read_jsonl(paths["p0_predictions"])
    p0_by_uid = {str(row["user_id"]): row for row in p0_rows}
    if len(p0_by_uid) != len(p0_rows):
        raise ValueError(f"Duplicate P0 users in {domain}")
    sequences = read_sequences(paths["user_sequence"])
    cold_items = read_set(paths["cold_items"])
    item_to_lexical = read_key_value_lines(paths["item_id_file"])
    catalog = set(item_to_lexical)
    if not cold_items <= catalog:
        raise ValueError(f"{domain} cold state is not a catalog subset")

    embedding_payload = torch.load(paths["item_embeddings"], map_location="cpu")
    item_ids = list(embedding_payload["item_ids"])
    embeddings = embedding_payload["embeddings"]
    if len(item_ids) != len(set(item_ids)) or set(item_ids) != catalog:
        raise ValueError(f"{domain} embedding/catalog mismatch")
    checkpoint = torch.load(paths["resolver_checkpoint"], map_location="cpu")
    if int(checkpoint["dim"]) != int(embeddings.shape[1]):
        raise ValueError(f"{domain} resolver/embedding dimension mismatch")

    gram_rows = parse_gram_predictions(paths["gram_validation_predictions"])
    if set(p0_by_uid) != set(gram_rows):
        raise ValueError(f"{domain} P0/raw-GRAM user alignment mismatch")
    if not set(p0_by_uid) <= set(sequences):
        raise ValueError(f"{domain} P0/sequence user alignment mismatch")
    for uid, row in p0_by_uid.items():
        v0 = row.get("v0_top50", [])
        resolver = row.get("resolver_top50", [])
        if len(v0) != 50 or len(resolver) != 50:
            raise ValueError(f"{domain}/{uid} does not have both frozen top-50 lists")
        if len(v0) != len(set(v0)) or len(resolver) != len(set(resolver)):
            raise ValueError(f"{domain}/{uid} has duplicate exact-item candidates")
        if not set(v0) <= catalog or not set(resolver) <= catalog:
            raise ValueError(f"{domain}/{uid} has non-catalog candidates")
        sequence = sequences[uid]
        if str(row["target"]) != sequence[-2]:
            raise ValueError(f"{domain}/{uid} validation target alignment mismatch")
        if bool(row["is_cold"]) != (sequence[-2] in cold_items):
            raise ValueError(f"{domain}/{uid} catalog-state alignment mismatch")

    score_maps = _decoded_score_maps(paths["gram_validation_predictions"], item_to_lexical)
    for uid, row in p0_by_uid.items():
        if any(item not in score_maps[uid] for item in row["v0_top50"]):
            raise ValueError(f"{domain}/{uid} P0 ranking lacks a frozen GRAM score")

    report = {
        "domain": domain,
        "split": "validation",
        "n_users": len(p0_rows),
        "n_catalog_items": len(catalog),
        "embedding_shape": list(embeddings.shape),
        "p0_raw_gram_user_alignment": True,
        "p0_sequence_user_alignment": True,
        "target_alignment_checked_for_integrity_only": True,
        "efficacy_metrics_computed": False,
        "sports_read": False,
        "test_read": False,
    }
    if include_hashes:
        report["input_sha256"] = {
            str(path.relative_to(project_root) if path.is_relative_to(project_root) else path): sha256_file(path)
            for path in sorted(set(paths.values())) if path.is_file()
        }
    return report, paths


def load_source_records(
    project_root: Path,
    domain: str,
    source_config: dict,
    device: torch.device,
) -> list[SourceRecord]:
    _audit, paths = audit_source(project_root, domain, source_config, include_hashes=False)
    p0_config = read_json(paths["p0_config"])
    p0_rows = read_jsonl(paths["p0_predictions"])
    sequences = read_sequences(paths["user_sequence"])
    cold_items = read_set(paths["cold_items"])
    item_to_lexical = read_key_value_lines(paths["item_id_file"])
    catalog = set(item_to_lexical)
    score_maps = _decoded_score_maps(
        Path(p0_config["gram_validation_predictions"]), item_to_lexical
    )

    embedding_payload = torch.load(paths["item_embeddings"], map_location="cpu")
    item_ids = list(embedding_payload["item_ids"])
    embeddings = F.normalize(embedding_payload["embeddings"].float(), dim=1)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    resolver_model = _load_resolver(paths["resolver_checkpoint"], embeddings, device)
    embeddings_device = embeddings.to(device)

    records: list[SourceRecord] = []
    with torch.no_grad():
        for offset in range(0, len(p0_rows), 256):
            batch = p0_rows[offset: offset + 256]
            history_vectors: list[torch.Tensor] = []
            history_by_uid: dict[str, list[str]] = {}
            for row in batch:
                uid = str(row["user_id"])
                history = sequences[uid][max(0, len(sequences[uid]) - 22):-2]
                history_by_uid[uid] = history
                history_vectors.append(recency_weighted_history(
                    (item_to_index[item] for item in history), embeddings, 0.85
                ))
            history_tensor = torch.stack(history_vectors).to(device)
            projected = resolver_model(history_tensor)
            for index, row in enumerate(batch):
                uid = str(row["user_id"])
                target = str(row["target"])
                v0_items = unique_in_order(row["v0_top50"])
                resolver_items = unique_in_order(row["resolver_top50"])
                v0_scores = _ranked_scores(v0_items, score_maps[uid])
                resolver_indices = torch.tensor(
                    [item_to_index[item] for item in resolver_items], device=device
                )
                resolver_scores = (
                    embeddings_device[resolver_indices] @ projected[index]
                ).detach().cpu().tolist()
                top3_indices = torch.tensor(
                    [item_to_index[item] for item in resolver_items[:3]], device=device
                )
                history_candidate_sims = (
                    embeddings_device[top3_indices] @ history_tensor[index]
                ).detach().cpu().tolist()
                raw_features = extract_features(
                    v0_items, v0_scores, resolver_items, resolver_scores,
                    cold_items, catalog, history_by_uid[uid], history_candidate_sims,
                )
                rankings: list[list[str]] = []
                effective: list[str] = []
                rewards: list[float] = []
                for action in ACTIONS:
                    effective_action, ranking = build_action_ranking(
                        v0_items, resolver_items, action, catalog, cold_items
                    )
                    effective.append(effective_action)
                    rankings.append(ranking)
                    rewards.append(hit_and_ndcg(ranking, target, 10)[1])
                _portfolio2_action, portfolio2 = build_action_ranking(
                    v0_items, resolver_items, "a2", catalog, cold_items
                )
                records.append(SourceRecord(
                    domain=domain,
                    user_id=uid,
                    fold=stable_fold(uid, domain),
                    raw_features=raw_features,
                    target=target,
                    is_cold=target in cold_items,
                    action_rewards=rewards,
                    action_rankings=rankings,
                    effective_actions=effective,
                    portfolio2_ranking=portfolio2,
                ))
            print(f"[features] {domain} {min(offset + 256, len(p0_rows))}/{len(p0_rows)}", flush=True)
    return records


def domain_balanced_batches(
    records: Sequence[SourceRecord],
    batch_size: int,
    generator: torch.Generator,
) -> Iterator[list[int]]:
    domains = sorted({record.domain for record in records})
    if len(domains) != 2 or batch_size % len(domains):
        raise ValueError("Frozen source batching requires two domains and divisible batch size")
    per_domain = batch_size // len(domains)
    indices = {
        domain: torch.tensor(
            [index for index, record in enumerate(records) if record.domain == domain],
            dtype=torch.long,
        )
        for domain in domains
    }
    shuffled = {
        domain: values[torch.randperm(len(values), generator=generator)]
        for domain, values in indices.items()
    }
    steps = max(math.ceil(len(values) / per_domain) for values in shuffled.values())
    for step in range(steps):
        batch: list[int] = []
        for domain in domains:
            values = shuffled[domain]
            start = (step * per_domain) % len(values)
            selection = [int(values[(start + offset) % len(values)]) for offset in range(per_domain)]
            batch.extend(selection)
        yield batch


def _domain_balanced_mean(values: torch.Tensor, records: Sequence[SourceRecord]) -> torch.Tensor:
    means = []
    for domain in sorted({record.domain for record in records}):
        mask = torch.tensor([record.domain == domain for record in records], device=values.device)
        means.append(values[mask].mean())
    return torch.stack(means).mean()


def train_allocator(
    records: Sequence[SourceRecord],
    raw_features: torch.Tensor,
    standardizer: Standardizer,
    device: torch.device,
    seed: int,
) -> tuple[BudgetConditionedAllocator, list[dict], float, int]:
    torch.manual_seed(seed)
    random.seed(seed)
    features = standardizer.transform(raw_features).to(device)
    rewards = torch.tensor([record.action_rewards for record in records], dtype=torch.float32, device=device)
    model = BudgetConditionedAllocator(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    dual = 0.0
    history: list[dict] = []
    steps = 0
    for epoch in range(1, 51):
        model.train()
        epoch_losses: list[float] = []
        epoch_violations: list[float] = []
        for batch_indices in domain_balanced_batches(records, 512, generator):
            budget = BUDGETS[int(torch.randint(len(BUDGETS), (1,), generator=generator))]
            index_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
            batch_records = [records[index] for index in batch_indices]
            batch_features = features[index_tensor]
            batch_rewards = rewards[index_tensor]
            budget_tensor = torch.full((len(batch_indices),), budget, device=device)
            logits = model(batch_features, budget_tensor)
            probability = F.softmax(logits / 1.0, dim=1)
            expected_reward = (probability * batch_rewards).sum(1)
            utility = _domain_balanced_mean(expected_reward, batch_records)

            warm_positions = [i for i, record in enumerate(batch_records) if not record.is_cold]
            if warm_positions:
                warm_index = torch.tensor(warm_positions, dtype=torch.long, device=device)
                warm_records = [batch_records[i] for i in warm_positions]
                warm_expected = _domain_balanced_mean(expected_reward[warm_index], warm_records)
                warm_baseline = _domain_balanced_mean(batch_rewards[warm_index, 0], warm_records)
                violation = budget * warm_baseline - warm_expected
            else:
                violation = torch.tensor(0.0, device=device)
            loss = -utility + dual * torch.relu(violation)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            dual = max(0.0, dual + 1e-2 * float(violation.detach()))
            steps += 1
            epoch_losses.append(float(loss.detach()))
            epoch_violations.append(float(violation.detach()))
        history.append({
            "epoch": epoch,
            "mean_loss": float(np.mean(epoch_losses)),
            "mean_constraint_violation": float(np.mean(epoch_violations)),
            "dual": dual,
        })
    model.eval()
    return model, history, dual, steps


def _checkpoint_payload(
    model: BudgetConditionedAllocator,
    standardizer: Standardizer,
    fold: int,
    steps: int,
    dual: float,
) -> dict:
    return {
        "state_dict": model.state_dict(),
        "standardizer": standardizer.as_dict(),
        "feature_names": FEATURE_NAMES,
        "fold": fold,
        "seed": SEED,
        "training_steps": steps,
        "dual_final": dual,
        "actions": ACTIONS,
        "primary_budget": PRIMARY_BUDGET,
    }


def evaluate_selected_record(record: SourceRecord, action_index: int) -> dict[str, float]:
    selected = record.action_rankings[action_index]
    selected_hit10, selected_ndcg10 = hit_and_ndcg(selected, record.target, 10)
    selected_hit50, _ = hit_and_ndcg(selected, record.target, 50)
    baseline_hit10, baseline_ndcg10 = hit_and_ndcg(record.portfolio2_ranking, record.target, 10)
    baseline_hit50, _ = hit_and_ndcg(record.portfolio2_ranking, record.target, 50)
    return {
        "cbsa_hit10": selected_hit10,
        "cbsa_ndcg10": selected_ndcg10,
        "cbsa_hit50": selected_hit50,
        "portfolio2_hit10": baseline_hit10,
        "portfolio2_ndcg10": baseline_ndcg10,
        "portfolio2_hit50": baseline_hit50,
    }


def aggregate_metric(rows: Sequence[dict], method: str, metric: str, subset: str) -> float:
    domain_means: list[float] = []
    for domain in sorted({row["domain"] for row in rows}):
        values = [
            row[f"{method}_{metric}"] for row in rows
            if row["domain"] == domain
            and (subset == "all" or (subset == "cold") == bool(row["is_cold"]))
        ]
        if values:
            domain_means.append(float(np.mean(values)))
    return float(np.mean(domain_means)) if domain_means else math.nan


def paired_domain_bootstrap(
    rows: Sequence[dict], metric: str, subset: str, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    arrays: list[np.ndarray] = []
    domains = sorted({row["domain"] for row in rows})
    for domain in domains:
        selected = [
            row for row in rows if row["domain"] == domain
            and (subset == "all" or (subset == "cold") == bool(row["is_cold"]))
        ]
        arrays.append(np.asarray([
            row[f"cbsa_{metric}"] - row[f"portfolio2_{metric}"] for row in selected
        ], dtype=np.float64))
    observed = float(np.mean([array.mean() for array in arrays]))
    bootstrap = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk_size = 128
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk_size):
        count = min(chunk_size, BOOTSTRAP_RESAMPLES - start)
        chunk_means = []
        for array in arrays:
            indices = rng.integers(0, len(array), size=(count, len(array)))
            chunk_means.append(array[indices].mean(axis=1))
        bootstrap[start:start + count] = np.stack(chunk_means).mean(axis=0)
    return {
        "observed": observed,
        "ci_low": float(np.percentile(bootstrap, 2.5)),
        "ci_high": float(np.percentile(bootstrap, 97.5)),
    }


def _ci_state(interval: dict, boundary: float) -> str:
    if interval["ci_low"] > boundary:
        return "PASS"
    if interval["ci_high"] < boundary:
        return "FAIL"
    return "INCONCLUSIVE"


def summarize_source_gate(rows: Sequence[dict]) -> dict:
    intervals = {
        "overall_ndcg10": paired_domain_bootstrap(rows, "ndcg10", "all", BOOTSTRAP_SEED),
        "warm_ndcg10": paired_domain_bootstrap(rows, "ndcg10", "warm", BOOTSTRAP_SEED),
        "cold_hit50": paired_domain_bootstrap(rows, "hit50", "cold", BOOTSTRAP_SEED),
    }
    baseline_cold = aggregate_metric(rows, "portfolio2", "hit50", "cold")
    cold_boundary = -0.05 * baseline_cold
    states = {
        "overall_ndcg10": _ci_state(intervals["overall_ndcg10"], 0.0),
        "warm_ndcg10": _ci_state(intervals["warm_ndcg10"], 0.0),
        "cold_hit50_noninferiority": _ci_state(intervals["cold_hit50"], cold_boundary),
    }
    per_domain = {}
    sparse = False
    directions_ok = True
    for domain in sorted({row["domain"] for row in rows}):
        domain_rows = [row for row in rows if row["domain"] == domain]
        cold_rows = [row for row in domain_rows if row["is_cold"]]
        overall_delta = float(np.mean([
            row["cbsa_ndcg10"] - row["portfolio2_ndcg10"] for row in domain_rows
        ]))
        cbsa_cold = float(np.mean([row["cbsa_hit50"] for row in cold_rows]))
        baseline_cold_domain = float(np.mean([row["portfolio2_hit50"] for row in cold_rows]))
        events = int(sum(row["portfolio2_hit50"] for row in cold_rows))
        per_domain[domain] = {
            "overall_ndcg10_delta": overall_delta,
            "cbsa_cold_hit50": cbsa_cold,
            "portfolio2_cold_hit50": baseline_cold_domain,
            "portfolio2_cold_hit50_events": events,
        }
        sparse |= events < 30
        directions_ok &= overall_delta > 0 and cbsa_cold >= 0.95 * baseline_cold_domain
    coverage = float(np.mean([
        row.get("effective_action", row["selected_action"]) != "a0" for row in rows
    ]))
    coverage_ok = 0.05 <= coverage <= 0.95
    integrity_ok = all(row["fold_isolation"] and row["catalog_unique"] for row in rows)
    if not integrity_ok or not directions_ok or not coverage_ok:
        verdict = "FAIL_STOP_R2_V2_SOURCE"
    elif sparse:
        verdict = "INCONCLUSIVE_STOP_R2_V2_SOURCE"
    elif "FAIL" in states.values():
        verdict = "FAIL_STOP_R2_V2_SOURCE"
    elif "INCONCLUSIVE" in states.values():
        verdict = "INCONCLUSIVE_STOP_R2_V2_SOURCE"
    else:
        verdict = "PASS_TO_R2_V2_SPORTS_CONFIRMATION_DISCUSSION"
    return {
        "verdict": verdict,
        "paired_bootstrap_vs_portfolio2": intervals,
        "gate_states": states,
        "cold_hit50_noninferiority_boundary": cold_boundary,
        "per_domain": per_domain,
        "intervention_coverage": coverage,
        "direction_consistency": directions_ok,
        "coverage_gate": coverage_ok,
        "event_density_guard_triggered": sparse,
        "integrity_gate": integrity_ok,
    }


def summarize_budget_curve(rows_by_budget: dict[float, Sequence[dict]]) -> dict:
    curve = {}
    for budget in BUDGETS:
        rows = rows_by_budget[budget]
        curve[f"rho_{budget:.2f}"] = {
            "rho": budget,
            "intervention_coverage": float(np.mean([
                row.get("effective_action", row["selected_action"]) != "a0"
                for row in rows
            ])),
            "metrics": {
                method: {
                    subset: {
                        metric: aggregate_metric(rows, method, metric, subset)
                        for metric in ("ndcg10", "hit50")
                    }
                    for subset in ("all", "warm", "cold")
                }
                for method in ("cbsa", "portfolio2")
            },
            "used_for_primary_gate": budget == PRIMARY_BUDGET,
        }
    return curve


def _canonical_and_artifact(
    project_root: Path, canonical_path: Path, output_dir: Path
) -> tuple[dict, dict | None]:
    canonical = read_json(canonical_path)
    if canonical.get("guards", {}).get("sports_read") is not False:
        raise ValueError("Frozen config does not keep Sports sealed")
    if canonical.get("guards", {}).get("test_read") is not False:
        raise ValueError("Frozen config does not keep test sealed")
    artifact_path = output_dir / "frozen_config.json"
    artifact = read_json(artifact_path) if artifact_path.exists() else None
    return canonical, artifact


def freeze_preflight(project_root: Path, canonical_path: Path, output_dir: Path) -> None:
    canonical, artifact = _canonical_and_artifact(project_root, canonical_path, output_dir)
    if artifact is not None:
        raise FileExistsError(f"Refusing to overwrite frozen config: {output_dir / 'frozen_config.json'}")
    if tuple(canonical["actions"]) != ACTIONS or tuple(canonical["budgets"]) != BUDGETS:
        raise ValueError("Canonical action/budget grid differs from preregistration")
    if canonical["primary_budget"] != PRIMARY_BUDGET:
        raise ValueError("Canonical primary budget differs from preregistration")
    audits = {}
    all_hashes = {}
    for domain, source_config in canonical["source_domains"].items():
        audit, _paths = audit_source(project_root, domain, source_config)
        audits[domain] = audit
        all_hashes.update(audit["input_sha256"])
    source_path = Path(inspect.getsourcefile(freeze_preflight) or __file__).resolve()
    expanded = {
        "canonical_config": canonical,
        "canonical_config_path": str(canonical_path.relative_to(project_root)),
        "canonical_config_sha256": sha256_file(canonical_path),
        "allocator_code_path": str(source_path.relative_to(project_root)),
        "allocator_code_sha256": sha256_file(source_path),
        "feature_schema": list(FEATURE_NAMES),
        "feature_schema_sha256": sha256_json(FEATURE_NAMES),
        "input_sha256": dict(sorted(all_hashes.items())),
        "sports_read": False,
        "test_read": False,
        "source_outcomes_read": False,
        "created_for_stage": "T0_IMPLEMENTATION_FREEZE",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "frozen_config.json", expanded)
    atomic_json(output_dir / "preflight_summary.json", {
        "experiment_id": canonical["experiment_id"],
        "status": "T0_PREFLIGHT_PASSED_STAGE_S_NOT_STARTED",
        "domains": audits,
        "feature_count": len(FEATURE_NAMES),
        "feature_schema_sha256": expanded["feature_schema_sha256"],
        "canonical_config_sha256": expanded["canonical_config_sha256"],
        "allocator_code_sha256": expanded["allocator_code_sha256"],
        "efficacy_metrics_computed": False,
        "sports_read": False,
        "test_read": False,
    })


def verify_preflight(project_root: Path, canonical_path: Path, output_dir: Path) -> dict:
    canonical, artifact = _canonical_and_artifact(project_root, canonical_path, output_dir)
    if artifact is None:
        raise FileNotFoundError(output_dir / "frozen_config.json")
    checks = {
        "canonical_config_sha256": sha256_file(canonical_path) == artifact["canonical_config_sha256"],
        "allocator_code_sha256": sha256_file(Path(__file__).resolve()) == artifact["allocator_code_sha256"],
        "feature_schema_sha256": sha256_json(FEATURE_NAMES) == artifact["feature_schema_sha256"],
        "sports_read_false": artifact.get("sports_read") is False,
        "test_read_false": artifact.get("test_read") is False,
    }
    current_hashes = {}
    for domain, source_config in canonical["source_domains"].items():
        audit, _paths = audit_source(project_root, domain, source_config)
        current_hashes.update(audit["input_sha256"])
    checks["input_sha256"] = dict(sorted(current_hashes.items())) == artifact["input_sha256"]
    if not all(checks.values()):
        raise RuntimeError(f"Frozen preflight verification failed: {checks}")
    return checks


def run_source(project_root: Path, canonical_path: Path, output_dir: Path, device: torch.device) -> None:
    checks = verify_preflight(project_root, canonical_path, output_dir)
    if (output_dir / "summary.json").exists() or (output_dir / "predictions_oof.jsonl").exists():
        raise FileExistsError("Refusing to overwrite Stage-S scientific artifacts")
    canonical = read_json(canonical_path)
    started = time.time()
    records: list[SourceRecord] = []
    for domain, source_config in canonical["source_domains"].items():
        records.extend(load_source_records(project_root, domain, source_config, device))
    raw_features = torch.tensor([record.raw_features for record in records], dtype=torch.float32)
    oof_rows: list[dict] = []
    budget_rows: dict[float, list[dict]] = {budget: [] for budget in BUDGETS}
    checkpoint_hashes = {}
    fold_audits = []
    for fold in range(FOLDS):
        train_indices = [i for i, record in enumerate(records) if record.fold != fold]
        held_indices = [i for i, record in enumerate(records) if record.fold == fold]
        if set(train_indices) & set(held_indices):
            raise RuntimeError("Fold overlap detected")
        train_records = [records[i] for i in train_indices]
        held_records = [records[i] for i in held_indices]
        standardizer = Standardizer.fit(raw_features[train_indices])
        model, history, dual, steps = train_allocator(
            train_records, raw_features[train_indices], standardizer, device, SEED
        )
        checkpoint_path = output_dir / f"allocator_fold{fold}.pt"
        torch.save(_checkpoint_payload(model, standardizer, fold, steps, dual), checkpoint_path)
        checkpoint_hashes[f"fold{fold}"] = sha256_file(checkpoint_path)
        held_features = standardizer.transform(raw_features[held_indices]).to(device)
        for budget in BUDGETS:
            budgets = torch.full((len(held_indices),), budget, device=device)
            with torch.no_grad():
                logits = model(held_features, budgets)
                selected_indices = safe_argmax(logits).cpu().tolist()
            for record, selected_index in zip(held_records, selected_indices):
                metrics = evaluate_selected_record(record, selected_index)
                selected_ranking = record.action_rankings[selected_index]
                row = {
                    "domain": record.domain,
                    "user_id": record.user_id,
                    "fold": fold,
                    "is_cold": record.is_cold,
                    "selected_action": ACTIONS[selected_index],
                    "effective_action": record.effective_actions[selected_index],
                    "fold_isolation": record.fold == fold,
                    "catalog_unique": len(selected_ranking) == len(set(selected_ranking)),
                    **metrics,
                }
                budget_rows[budget].append(row)
                if budget == PRIMARY_BUDGET:
                    oof_rows.append(row)
        fold_audits.append({
            "fold": fold,
            "n_train": len(train_indices),
            "n_held": len(held_indices),
            "train_held_overlap": 0,
            "training_steps": steps,
            "dual_final": dual,
            "last_epoch": history[-1],
        })
    if len(oof_rows) != len(records):
        raise RuntimeError("OOF predictions do not cover every source user exactly once")
    gate = summarize_source_gate(oof_rows)
    budget_curve = summarize_budget_curve(budget_rows)
    with (output_dir / "predictions_oof.jsonl").open("w") as handle:
        for row in sorted(oof_rows, key=lambda value: (value["domain"], value["user_id"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "experiment_id": canonical["experiment_id"],
        "status": "completed",
        **gate,
        "primary_budget": PRIMARY_BUDGET,
        "budget_curve": budget_curve,
        "actions": ACTIONS,
        "feature_schema": FEATURE_NAMES,
        "feature_target_free": True,
        "fold_audits": fold_audits,
        "checkpoint_sha256": checkpoint_hashes,
        "allocator_parameter_count": parameter_count,
        "allocator_seed": SEED,
        "training_steps": sum(item["training_steps"] for item in fold_audits),
        "preflight_checks": checks,
        "source_domains": sorted(canonical["source_domains"]),
        "sports_read": False,
        "test_read": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    next_action = (
        "discuss Sports confirmation; do not start automatically"
        if gate["verdict"] == "PASS_TO_R2_V2_SPORTS_CONFIRMATION_DISCUSSION"
        else "stop R²-v2; do not create a source-domain rescue"
    )
    decision = (
        "# R²-v2 Stage S Decision\n\n"
        f"- Verdict: `{gate['verdict']}`\n"
        f"- Next action: {next_action}.\n"
        "- Sports started: `false`\n"
        "- Toys/Beauty test read: `false`\n"
    )
    decision_path = output_dir / "decision.md"
    temporary = decision_path.with_suffix(".md.tmp")
    temporary.write_text(decision)
    temporary.replace(decision_path)


def validate_static_contract() -> None:
    if len(FEATURE_NAMES) != len(set(FEATURE_NAMES)):
        raise AssertionError("Duplicate feature name")
    lowered = " ".join(FEATURE_NAMES).casefold()
    if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
        raise AssertionError("Feature schema contains a target-derived token")
    signature = inspect.signature(extract_features)
    if any(token in signature.parameters for token in ("target", "label", "reward", "is_cold")):
        raise AssertionError("Feature extraction accepts a target-derived argument")


def main() -> None:
    validate_static_contract()
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    canonical_path = Path(args.canonical_config)
    if not canonical_path.is_absolute():
        canonical_path = project_root / canonical_path
    canonical_path = canonical_path.resolve()
    guard_source_path(canonical_path)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()
    guard_source_path(output_dir)
    if args.mode == "freeze-preflight":
        freeze_preflight(project_root, canonical_path, output_dir)
    elif args.mode == "verify-preflight":
        print(json.dumps(verify_preflight(project_root, canonical_path, output_dir), indent=2))
    else:
        run_source(project_root, canonical_path, output_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
