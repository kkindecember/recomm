"""S17-2R fold-safe data adapter and independent Semantic-ID primitives.

This module intentionally contains no copied third-party implementation.  It
provides the common Phase17 contract needed by architecture-native controls:
shadow-fold parsing, train-only quantizer fitting, deterministic collision
resolution, token layout, and user-level datasets.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ShadowUser:
    user_id: str
    train_items: tuple[str, ...]
    validation_target: str
    guard_item: str


@dataclass(frozen=True)
class SequenceExample:
    user_id: str
    history: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class SIDBuildSummary:
    seed: int
    embedding_method: str
    requested_codebook_size: int
    codebook_sizes: tuple[int, ...]
    n_codebooks: int
    catalog_items: int
    fit_items: int
    collisions_before_resolution: int
    collisions_after_resolution: int
    reassigned_items: int
    mean_reassignment_distance: float
    max_reassignment_distance: float
    train_only_quantizer_fit: bool
    collision_resolution: str = "nearest_unique_reassignment"
    collision_suffix_size: int = 0


@dataclass(frozen=True)
class CFBuildSummary:
    seed: int
    method: str
    catalog_items: int
    fit_items: int
    codebook_size: int
    observed_transitions: int
    train_only_fit: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_shadow_sequences(path: Path) -> list[ShadowUser]:
    normalized_path = path.resolve().as_posix()
    if "phase17/s2r_preflight/data/" not in normalized_path:
        raise ValueError(f"S17-2R may read only its frozen preflight projection: {path}")
    if "GRAM/rec_datasets" in normalized_path or "/D1/" in normalized_path:
        raise ValueError("original monolithic data and D1 are forbidden during S17-2R")

    users: list[ShadowUser] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if not fields:
            continue
        if len(fields) < 4:
            raise ValueError(f"line {line_number} lacks train/validation/guard positions")
        user_id, items = fields[0], fields[1:]
        if user_id in seen:
            raise ValueError(f"duplicate user id: {user_id}")
        seen.add(user_id)
        train_items = tuple(items[:-2])
        if not train_items:
            raise ValueError(f"line {line_number} has no train-prefix item")
        users.append(
            ShadowUser(
                user_id=user_id,
                train_items=train_items,
                validation_target=items[-2],
                guard_item=items[-1],
            )
        )
    if not users:
        raise ValueError("empty S17-2R shadow dataset")
    return users


def parse_item_text(path: Path) -> dict[str, str]:
    item_text: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        item_id, separator, text = stripped.partition(" ")
        if not separator or not text:
            raise ValueError(f"item text line {line_number} is malformed")
        if item_id in item_text:
            raise ValueError(f"duplicate item text id: {item_id}")
        item_text[item_id] = text
    if not item_text:
        raise ValueError("empty item text catalog")
    return item_text


def train_catalog_items(users: Sequence[ShadowUser]) -> set[str]:
    return {item for user in users for item in user.train_items}


def build_train_only_cf_codes(
    item_ids: Sequence[str],
    users: Sequence[ShadowUser],
    *,
    codebook_size: int = 32,
    hash_buckets: int = 64,
    seed: int = 2023,
) -> tuple[dict[str, int], CFBuildSummary]:
    """Build a small fold-safe collaborative token from training transitions.

    Validation and guard positions are structurally unavailable because the
    adapter exposes them separately from ``train_items``.  Hashed directed
    transition features keep this R1 tokenizer lightweight and deterministic.
    """

    item_to_row = {item: index for index, item in enumerate(item_ids)}
    features = np.zeros((len(item_ids), hash_buckets + 2), dtype=np.float32)
    transitions = 0
    for user in users:
        for current, following in zip(user.train_items, user.train_items[1:]):
            if current not in item_to_row or following not in item_to_row:
                continue
            current_row = item_to_row[current]
            following_row = item_to_row[following]
            forward = int(
                hashlib.sha256(f"out:{following}".encode("utf-8")).hexdigest()[:8],
                16,
            ) % hash_buckets
            backward = int(
                hashlib.sha256(f"in:{current}".encode("utf-8")).hexdigest()[:8], 16
            ) % hash_buckets
            features[current_row, forward] += 1.0
            features[following_row, backward] += 1.0
            features[current_row, -2] += 1.0
            features[following_row, -1] += 1.0
            transitions += 1
    fit_items = sorted(train_catalog_items(users) & set(item_ids))
    fit_rows = [item_to_row[item] for item in fit_items]
    if len(fit_rows) < 2 or transitions == 0:
        raise ValueError("CF tokenizer needs at least two train items and one transition")
    features = normalize(features, norm="l2")
    level_size = min(int(codebook_size), len(fit_rows))
    model = MiniBatchKMeans(
        n_clusters=level_size,
        random_state=seed,
        n_init=3,
        max_iter=100,
        batch_size=min(1024, max(level_size * 4, len(fit_rows))),
        reassignment_ratio=0.0,
    )
    model.fit(features[fit_rows])
    codes = model.predict(features)
    return (
        {item: int(codes[index]) for index, item in enumerate(item_ids)},
        CFBuildSummary(
            seed=seed,
            method="train_prefix_hashed_directed_transition_kmeans",
            catalog_items=len(item_ids),
            fit_items=len(fit_rows),
            codebook_size=level_size,
            observed_transitions=transitions,
            train_only_fit=True,
        ),
    )


def build_examples(
    users: Sequence[ShadowUser], *, max_history_items: int = 20
) -> tuple[list[SequenceExample], list[SequenceExample]]:
    train: list[SequenceExample] = []
    validation: list[SequenceExample] = []
    for user in users:
        for target_position in range(1, len(user.train_items)):
            train.append(
                SequenceExample(
                    user_id=user.user_id,
                    history=user.train_items[:target_position][-max_history_items:],
                    target=user.train_items[target_position],
                )
            )
        validation.append(
            SequenceExample(
                user_id=user.user_id,
                history=user.train_items[-max_history_items:],
                target=user.validation_target,
            )
        )
    if not train:
        raise ValueError("no sliding train examples can be constructed")
    return train, validation


def select_r2_early_stop_users(
    users: Sequence[ShadowUser], *, count: int = 300, seed: int = 2023
) -> tuple[str, ...]:
    eligible = [user.user_id for user in users if len(user.train_items) >= 2]
    if len(eligible) < count:
        raise ValueError(
            f"requested {count} R2 early-stop users from only {len(eligible)} eligible users"
        )
    ranked = sorted(
        eligible,
        key=lambda user_id: (
            hashlib.sha256(
                f"s17-2r-r2-early-stop:{seed}:{user_id}".encode("utf-8")
            ).hexdigest(),
            user_id,
        ),
    )
    return tuple(ranked[:count])


def build_r2_examples(
    users: Sequence[ShadowUser],
    early_stop_user_ids: Sequence[str],
    *,
    max_history_items: int = 20,
) -> tuple[list[SequenceExample], list[SequenceExample], list[SequenceExample]]:
    """Build supervised train, internal early-stop, and external R2 examples.

    The internal target is the final *training-prefix* item for selected users.
    That position is removed from their supervised train examples.  External
    validation targets remain untouched until the best checkpoint is frozen.
    """

    train, early_stop = build_r2_training_examples(
        users, early_stop_user_ids, max_history_items=max_history_items
    )
    external = build_r2_external_examples(users, max_history_items=max_history_items)
    return train, early_stop, external


def build_r2_training_examples(
    users: Sequence[ShadowUser],
    early_stop_user_ids: Sequence[str],
    *,
    max_history_items: int = 20,
) -> tuple[list[SequenceExample], list[SequenceExample]]:
    """Build train-prefix supervision without materializing external targets."""

    early_stop_set = set(early_stop_user_ids)
    known_users = {user.user_id for user in users}
    unknown = early_stop_set - known_users
    if unknown:
        raise ValueError(f"unknown R2 early-stop users: {sorted(unknown)[:3]}")
    train: list[SequenceExample] = []
    early_stop: list[SequenceExample] = []
    for user in users:
        training_items = user.train_items
        if user.user_id in early_stop_set:
            if len(user.train_items) < 2:
                raise ValueError(f"early-stop user lacks two train items: {user.user_id}")
            training_items = user.train_items[:-1]
            early_stop.append(
                SequenceExample(
                    user_id=user.user_id,
                    history=user.train_items[:-1][-max_history_items:],
                    target=user.train_items[-1],
                )
            )
        for target_position in range(1, len(training_items)):
            train.append(
                SequenceExample(
                    user_id=user.user_id,
                    history=training_items[:target_position][-max_history_items:],
                    target=training_items[target_position],
                )
            )
    if len(early_stop) != len(early_stop_set):
        raise AssertionError("R2 early-stop split cardinality drifted")
    if not train:
        raise ValueError("R2 training split is empty")
    return train, early_stop


def build_r2_external_examples(
    users: Sequence[ShadowUser], *, max_history_items: int = 20
) -> list[SequenceExample]:
    """Materialize shadow targets only after all family checkpoints are frozen."""

    external = [
        SequenceExample(
            user_id=user.user_id,
            history=user.train_items[-max_history_items:],
            target=user.validation_target,
        )
        for user in users
    ]
    if not external:
        raise ValueError("R2 external evaluation split is empty")
    return external


def read_cohort_user_ids(paths: Sequence[Path]) -> tuple[tuple[str, ...], ...]:
    cohorts = []
    seen: set[str] = set()
    for path in paths:
        user_ids = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if len(user_ids) != len(set(user_ids)):
            raise ValueError(f"duplicate user in R2 cohort: {path}")
        overlap = seen & set(user_ids)
        if overlap:
            raise ValueError(f"R2 cohorts overlap at user {sorted(overlap)[0]}")
        seen.update(user_ids)
        cohorts.append(user_ids)
    return tuple(cohorts)


def tfidf_embeddings(
    item_ids: Sequence[str],
    item_text: dict[str, str],
    *,
    output_dim: int = 128,
    max_features: int = 4096,
    seed: int = 2023,
) -> np.ndarray:
    texts = [item_text[item] for item in item_ids]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        dtype=np.float32,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    sparse = vectorizer.fit_transform(texts)
    max_components = min(output_dim, sparse.shape[0] - 1, sparse.shape[1] - 1)
    if max_components >= 2:
        dense = TruncatedSVD(n_components=max_components, random_state=seed).fit_transform(
            sparse
        )
    else:
        dense = sparse.toarray()
    return normalize(np.asarray(dense, dtype=np.float32), norm="l2")


def _count_collisions(codes: np.ndarray) -> int:
    counts: dict[tuple[int, ...], int] = {}
    for row in codes.tolist():
        key = tuple(int(value) for value in row)
        counts[key] = counts.get(key, 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def _reconstruct(code: Sequence[int], centers: Sequence[np.ndarray]) -> np.ndarray:
    value = np.zeros(centers[0].shape[1], dtype=np.float32)
    for level, token in enumerate(code):
        value += centers[level][int(token)]
    return value


def _candidate_codes(
    base: tuple[int, ...], centers: Sequence[np.ndarray], item_embedding: np.ndarray
) -> Iterable[tuple[int, ...]]:
    scored: list[tuple[float, tuple[int, ...]]] = []
    for level, level_centers in enumerate(centers):
        for token in range(level_centers.shape[0]):
            if token == base[level]:
                continue
            candidate = list(base)
            candidate[level] = token
            candidate_tuple = tuple(candidate)
            distance = float(
                np.square(item_embedding - _reconstruct(candidate_tuple, centers)).sum()
            )
            scored.append((distance, candidate_tuple))
    scored.sort(key=lambda row: (row[0], row[1]))
    yielded: set[tuple[int, ...]] = set()
    for _, candidate in scored:
        yielded.add(candidate)
        yield candidate

    nearest_per_level: list[list[int]] = []
    for level_centers in centers:
        distances = np.square(level_centers - item_embedding[None, :]).sum(axis=1)
        nearest_per_level.append(np.argsort(distances)[: min(8, len(distances))].tolist())
    for candidate in product(*nearest_per_level):
        candidate_tuple = tuple(int(value) for value in candidate)
        if candidate_tuple not in yielded and candidate_tuple != base:
            yield candidate_tuple


def resolve_collisions(
    item_ids: Sequence[str],
    codes: np.ndarray,
    embeddings: np.ndarray,
    centers: Sequence[np.ndarray],
) -> tuple[dict[str, tuple[int, ...]], dict[str, float]]:
    capacity = math.prod(level.shape[0] for level in centers)
    if capacity < len(item_ids):
        raise ValueError(
            f"Semantic-ID capacity {capacity} is below catalog size {len(item_ids)}"
        )

    used: set[tuple[int, ...]] = set()
    resolved: dict[str, tuple[int, ...]] = {}
    reassignment_distances: list[float] = []
    for index, item_id in enumerate(item_ids):
        original = tuple(int(value) for value in codes[index].tolist())
        chosen = original
        if chosen in used:
            chosen = next(
                (
                    candidate
                    for candidate in _candidate_codes(original, centers, embeddings[index])
                    if candidate not in used
                ),
                None,
            )
            if chosen is None:
                radices = [level.shape[0] for level in centers]
                for ordinal in range(capacity):
                    value = ordinal
                    candidate_reversed = []
                    for radix in reversed(radices):
                        candidate_reversed.append(value % radix)
                        value //= radix
                    candidate = tuple(reversed(candidate_reversed))
                    if candidate not in used:
                        chosen = candidate
                        break
            if chosen is None:
                raise RuntimeError("could not assign a unique Semantic ID")
            reassignment_distances.append(
                float(np.linalg.norm(_reconstruct(chosen, centers) - embeddings[index]))
            )
        used.add(chosen)
        resolved[item_id] = chosen

    return resolved, {
        "reassigned_items": float(len(reassignment_distances)),
        "mean_reassignment_distance": float(np.mean(reassignment_distances))
        if reassignment_distances
        else 0.0,
        "max_reassignment_distance": float(np.max(reassignment_distances))
        if reassignment_distances
        else 0.0,
    }


def append_collision_suffix(
    item_ids: Sequence[str], codes: np.ndarray
) -> tuple[dict[str, tuple[int, ...]], int]:
    """Append a deterministic within-code ordinal without changing semantic digits."""

    groups: dict[tuple[int, ...], list[str]] = {}
    for item_id, row in zip(item_ids, codes.tolist()):
        base = tuple(int(value) for value in row)
        groups.setdefault(base, []).append(item_id)
    suffix_size = max(len(group) for group in groups.values())
    resolved: dict[str, tuple[int, ...]] = {}
    for base, group in groups.items():
        for ordinal, item_id in enumerate(sorted(group)):
            resolved[item_id] = (*base, ordinal)
    return resolved, suffix_size


def build_residual_kmeans_ids(
    item_ids: Sequence[str],
    embeddings: np.ndarray,
    fit_item_ids: set[str],
    *,
    n_codebooks: int = 3,
    codebook_size: int = 32,
    seed: int = 2023,
    embedding_method: str = "tfidf_bigram_truncated_svd_l2",
    collision_resolution: str = "nearest_unique_reassignment",
) -> tuple[dict[str, tuple[int, ...]], SIDBuildSummary]:
    if len(item_ids) != embeddings.shape[0]:
        raise ValueError("item_ids and embeddings length mismatch")
    item_to_row = {item: index for index, item in enumerate(item_ids)}
    fit_rows = [item_to_row[item] for item in sorted(fit_item_ids) if item in item_to_row]
    if len(fit_rows) < 2:
        raise ValueError("at least two train-prefix items are required for SID fitting")

    residual = embeddings.astype(np.float32, copy=True)
    codes: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for level in range(n_codebooks):
        level_size = min(codebook_size, len(fit_rows))
        model = MiniBatchKMeans(
            n_clusters=level_size,
            random_state=seed + level,
            n_init=3,
            max_iter=100,
            batch_size=min(1024, max(level_size * 4, len(fit_rows))),
            reassignment_ratio=0.0,
        )
        model.fit(residual[fit_rows])
        level_codes = model.predict(residual)
        level_centers = np.asarray(model.cluster_centers_, dtype=np.float32)
        codes.append(level_codes.astype(np.int64))
        centers.append(level_centers)
        residual = residual - level_centers[level_codes]

    code_matrix = np.stack(codes, axis=1)
    collisions_before = _count_collisions(code_matrix)
    if collision_resolution == "nearest_unique_reassignment":
        item_to_code, resolution = resolve_collisions(
            item_ids, code_matrix, embeddings, centers
        )
        suffix_size = 0
    elif collision_resolution == "append_group_ordinal":
        item_to_code, suffix_size = append_collision_suffix(item_ids, code_matrix)
        resolution = {
            "reassigned_items": 0.0,
            "mean_reassignment_distance": 0.0,
            "max_reassignment_distance": 0.0,
        }
    else:
        raise ValueError(f"unknown collision resolution: {collision_resolution}")
    final_codes = np.asarray([item_to_code[item] for item in item_ids], dtype=np.int64)
    collisions_after = _count_collisions(final_codes)
    if collisions_after:
        raise AssertionError("collision resolution failed")

    summary = SIDBuildSummary(
        seed=seed,
        embedding_method=embedding_method,
        requested_codebook_size=codebook_size,
        codebook_sizes=(
            *tuple(int(level.shape[0]) for level in centers),
            *((suffix_size,) if suffix_size else ()),
        ),
        n_codebooks=n_codebooks,
        catalog_items=len(item_ids),
        fit_items=len(fit_rows),
        collisions_before_resolution=collisions_before,
        collisions_after_resolution=collisions_after,
        reassigned_items=int(resolution["reassigned_items"]),
        mean_reassignment_distance=resolution["mean_reassignment_distance"],
        max_reassignment_distance=resolution["max_reassignment_distance"],
        train_only_quantizer_fit=True,
        collision_resolution=collision_resolution,
        collision_suffix_size=suffix_size,
    )
    return item_to_code, summary


def write_sid_artifact(
    output: Path,
    *,
    item_to_code: dict[str, tuple[int, ...]],
    summary: SIDBuildSummary,
    sequence_path: Path,
    item_text_path: Path,
) -> dict:
    payload = {
        "schema_version": "phase17.s17_2r_sid.v1",
        "formal_result_eligible": False,
        "fidelity": "R1_CONTRACT_TFIDF_RQKMEANS_NOT_PAPER_SENTENCE_T5",
        "sequence_path": str(sequence_path),
        "sequence_sha256": sha256_file(sequence_path),
        "item_text_path": str(item_text_path),
        "item_text_sha256": sha256_file(item_text_path),
        "summary": asdict(summary),
        "item_to_code": {item: list(code) for item, code in item_to_code.items()},
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


class SemanticIDCodec:
    """Shared token layout for PSID, Latte, Gryphon, and later R2 candidates."""

    padding_token = 0

    def __init__(
        self,
        item_to_code: dict[str, Sequence[int]],
        codebook_sizes: Sequence[int],
        *,
        n_latent_tokens: int = 8,
        n_user_tokens: int = 1,
        max_history_items: int = 20,
    ) -> None:
        self.item_to_code = {
            item: tuple(int(value) for value in code)
            for item, code in item_to_code.items()
        }
        self.codebook_sizes = tuple(int(value) for value in codebook_sizes)
        self.n_digit = len(self.codebook_sizes)
        self.n_latent_tokens = int(n_latent_tokens)
        self.n_user_tokens = int(n_user_tokens)
        self.max_history_items = int(max_history_items)
        if any(len(code) != self.n_digit for code in self.item_to_code.values()):
            raise ValueError("Semantic-ID digit count mismatch")

        self.base_latent_token = 1
        offset = self.base_latent_token + self.n_latent_tokens
        self.base_sem_token = offset
        offsets = []
        for size in self.codebook_sizes:
            offsets.append(offset)
            offset += size
        self.semantic_offsets = tuple(offsets)
        self.base_user_token = offset
        self.mask_token = self.base_user_token + self.n_user_tokens
        self.eos_token = self.mask_token + 1
        self.vocab_size = self.eos_token + 1
        self.max_input_length = 1 + self.max_history_items * self.n_digit + 1

        self.item_ids = tuple(sorted(self.item_to_code))
        self.item_to_index = {item: index for index, item in enumerate(self.item_ids)}
        self.index_to_item = dict(enumerate(self.item_ids))
        self.tokens_to_item: dict[tuple[int, ...], str] = {}
        for item in self.item_ids:
            tokens = self.semantic_tokens(item)
            if tokens in self.tokens_to_item:
                raise ValueError(f"non-unique Semantic-ID token tuple: {tokens}")
            self.tokens_to_item[tokens] = item

        legal_next: dict[tuple[int, ...], set[int]] = {}
        for tokens in self.tokens_to_item:
            for depth in range(self.n_digit):
                prefix = tokens[:depth]
                legal_next.setdefault(prefix, set()).add(tokens[depth])
        self.legal_next = {
            prefix: tuple(sorted(values)) for prefix, values in legal_next.items()
        }

    def semantic_tokens(self, item: str) -> tuple[int, ...]:
        code = self.item_to_code[item]
        return tuple(
            self.semantic_offsets[level] + int(code[level])
            for level in range(self.n_digit)
        )

    def user_token(self, user_id: str) -> int:
        hashed = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16], 16)
        return self.base_user_token + hashed % self.n_user_tokens

    def encode_input(self, user_id: str, history: Sequence[str]) -> tuple[list[int], list[int]]:
        tokens = [self.user_token(user_id)]
        for item in history[-self.max_history_items :]:
            tokens.extend(self.semantic_tokens(item))
        tokens.append(self.eos_token)
        attention = [1] * len(tokens)
        pad_count = self.max_input_length - len(tokens)
        if pad_count < 0:
            raise AssertionError("max input length calculation drifted")
        return tokens + [self.padding_token] * pad_count, attention + [0] * pad_count

    def encode_label(
        self, item: str, *, latent_token: int | None = None
    ) -> list[int]:
        tokens = list(self.semantic_tokens(item))
        if latent_token is not None:
            if not (
                self.base_latent_token
                <= latent_token
                < self.base_latent_token + self.n_latent_tokens
            ):
                raise ValueError("latent token is outside the reserved range")
            tokens.insert(0, latent_token)
        tokens.append(self.eos_token)
        return tokens

    def decode_semantic_tokens(self, tokens: Sequence[int]) -> str | None:
        return self.tokens_to_item.get(tuple(int(value) for value in tokens))

    def allowed_generation_tokens(
        self, generated: Sequence[int], *, latte: bool
    ) -> tuple[int, ...]:
        values = [int(value) for value in generated]
        if values and values[0] == self.padding_token:
            values = values[1:]
        if latte:
            if not values:
                return tuple(
                    range(
                        self.base_latent_token,
                        self.base_latent_token + self.n_latent_tokens,
                    )
                )
            if not (
                self.base_latent_token
                <= values[0]
                < self.base_latent_token + self.n_latent_tokens
            ):
                return (self.eos_token,)
            semantic_prefix = tuple(values[1:])
        else:
            semantic_prefix = tuple(values)
        if len(semantic_prefix) < self.n_digit:
            return self.legal_next.get(semantic_prefix, (self.eos_token,))
        return (self.eos_token,)


class SIDSequenceDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[SequenceExample],
        codec: SemanticIDCodec,
        *,
        latte_training: bool,
        seed: int,
    ) -> None:
        self.examples = list(examples)
        self.codec = codec
        self.latte_training = latte_training
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.examples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        input_ids, attention_mask = self.codec.encode_input(
            example.user_id, example.history
        )
        latent = None
        if self.latte_training:
            digest = hashlib.sha256(
                f"{self.seed}:{self.epoch}:{index}".encode("utf-8")
            ).hexdigest()
            latent = self.codec.base_latent_token + int(digest[:8], 16) % self.codec.n_latent_tokens
        labels = self.codec.encode_label(example.target, latent_token=latent)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_item_index": torch.tensor(
                self.codec.item_to_index[example.target], dtype=torch.long
            ),
            "user_id": example.user_id,
            "target_item": example.target,
        }


def collate_sid_batch(batch: Sequence[dict]) -> dict:
    return {
        "input_ids": torch.stack([row["input_ids"] for row in batch]),
        "attention_mask": torch.stack([row["attention_mask"] for row in batch]),
        "labels": torch.stack([row["labels"] for row in batch]),
        "target_item_index": torch.stack(
            [row["target_item_index"] for row in batch]
        ),
        "user_id": [row["user_id"] for row in batch],
        "target_item": [row["target_item"] for row in batch],
    }
