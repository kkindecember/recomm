"""Leakage-safe full-catalog SentenceT5 -> PCA -> RQ-KMeans tokenizer.

The pinned LATTE implementation fits PCA on the complete catalog before it
applies its training-item mask to residual quantization.  Stage17 requires all
learned tokenizer transforms to fit on the train-prefix catalog only.  This
module therefore precomputes the exact official ``.sem_ids`` cache: PCA and RQ
are fit on the same frozen 11,138-item mask, while every catalog item receives
only target-independent transforms and code assignment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .full_latte_contracts import PSIDResolutionSummary, resolve_rqkmeans_psid_conflicts
from .full_latte_native_adapter import LatteNativeDataBundle, build_latte_native_bundle
from .status_writer import atomic_json, utc_now


OFFICIAL_SEM_IDS_FILENAME = "sentence-t5-base_rqkmeans_3x256_psid.sem_ids"


@dataclass(frozen=True)
class FullLatteTokenizerSpec:
    model_revision: str = "fc5d4628481afbbaaacd7af6bb07cf9d3865f781"
    sentence_embedding_dim: int = 768
    pca_components: int = 192
    codebook_count: int = 3
    codebook_size: int = 256
    conflict_top_k_per_digit: int = 5
    sentence_batch_size: int = 32
    faiss_threads: int = 32
    seed: int = 2023

    def validate(self) -> None:
        if self.sentence_embedding_dim <= 0 or self.pca_components <= 0:
            raise ValueError("embedding and PCA dimensions must be positive")
        if self.pca_components > self.sentence_embedding_dim:
            raise ValueError("PCA dimension exceeds sentence embedding dimension")
        if self.codebook_count <= 0 or self.codebook_size <= 1:
            raise ValueError("invalid RQ codebook dimensions")
        if self.codebook_size & (self.codebook_size - 1):
            raise ValueError("official Faiss bit packing requires a power-of-two codebook")
        if self.sentence_batch_size <= 0 or self.faiss_threads <= 0:
            raise ValueError("batch size and Faiss thread count must be positive")


@dataclass(frozen=True)
class TokenizerBuildResult:
    manifest: dict[str, Any]
    semantic_ids: dict[str, tuple[int, ...]]
    resolution: PSIDResolutionSummary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tokenizer_fit_mask(
    catalog_items: Sequence[str], bundle: LatteNativeDataBundle
) -> np.ndarray:
    """Return the frozen tokenizer-fit mask without opening external targets."""

    if not catalog_items or len(catalog_items) != len(set(catalog_items)):
        raise ValueError("catalog item order must be non-empty and unique")
    catalog_set = set(catalog_items)
    training_items = {
        item for sequence in bundle.train_sequences for item in sequence.item_seq
    }
    unknown = training_items - catalog_set
    if unknown:
        raise ValueError(f"tokenizer-fit items outside catalog: {sorted(unknown)[:3]}")
    mask = np.asarray([item in training_items for item in catalog_items], dtype=bool)
    if int(mask.sum()) != bundle.tokenizer_fit_catalog_items:
        raise AssertionError(
            f"fit-mask count {int(mask.sum())} != frozen bundle count "
            f"{bundle.tokenizer_fit_catalog_items}"
        )
    return mask


def fit_pca_train_only(
    embeddings: np.ndarray,
    fit_mask: np.ndarray,
    *,
    n_components: int,
    seed: int,
    pca_factory: Callable[..., Any] | None = None,
) -> tuple[np.ndarray, Any]:
    """Fit whitened PCA on the fit mask, then transform the complete catalog."""

    values = np.asarray(embeddings, dtype=np.float32)
    mask = np.asarray(fit_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != (values.shape[0],):
        raise ValueError("embedding matrix and fit mask shapes do not align")
    if not np.isfinite(values).all():
        raise ValueError("sentence embeddings contain non-finite values")
    if n_components <= 0 or n_components > min(int(mask.sum()), values.shape[1]):
        raise ValueError("PCA component count is incompatible with fit data")
    if pca_factory is None:
        from sklearn.decomposition import PCA

        pca_factory = PCA
    pca = pca_factory(n_components=n_components, whiten=True, random_state=seed)
    # Do not use fit_transform(all_values): that is the leakage being prevented.
    pca.fit(values[mask])
    transformed = np.asarray(pca.transform(values), dtype=np.float32)
    if transformed.shape != (values.shape[0], n_components):
        raise AssertionError("unexpected PCA output shape")
    if not np.isfinite(transformed).all():
        raise RuntimeError("PCA output contains non-finite values")
    return transformed, pca


def _decode_faiss_codes(
    packed: np.ndarray, *, codebook_count: int, bits_per_code: int, faiss_module: Any
) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError("Faiss RQ codes must be a two-dimensional byte array")
    result = np.empty((packed.shape[0], codebook_count), dtype=np.int64)
    for row_index, row in enumerate(packed):
        reader = faiss_module.BitstringReader(
            faiss_module.swig_ptr(row), packed.shape[1]
        )
        for digit in range(codebook_count):
            result[row_index, digit] = reader.read(bits_per_code)
    return result


def train_rqkmeans_train_only(
    transformed_embeddings: np.ndarray,
    fit_mask: np.ndarray,
    *,
    codebook_count: int,
    codebook_size: int,
    faiss_threads: int,
    seed: int,
    faiss_module: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train official Faiss residual quantization on the frozen fit mask."""

    values = np.ascontiguousarray(transformed_embeddings, dtype=np.float32)
    mask = np.asarray(fit_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != (values.shape[0],):
        raise ValueError("RQ matrix and fit mask shapes do not align")
    if not np.isfinite(values).all() or not mask.any():
        raise ValueError("RQ input must be finite and have fit items")
    if codebook_size <= 1 or codebook_size & (codebook_size - 1):
        raise ValueError("RQ codebook size must be a power of two")
    if faiss_module is None:
        import faiss as faiss_module

    bits = int(math.log2(codebook_size))
    faiss_module.omp_set_num_threads(faiss_threads)
    index = faiss_module.IndexResidualQuantizer(
        values.shape[1],
        codebook_count,
        bits,
        faiss_module.METRIC_INNER_PRODUCT,
    )
    # Faiss defaults are deterministic, but freeze the clustering seed when the
    # installed version exposes it.
    if hasattr(index.rq, "cp") and hasattr(index.rq.cp, "seed"):
        index.rq.cp.seed = int(seed)
    index.train(np.ascontiguousarray(values[mask]))
    if not index.is_trained:
        raise RuntimeError("Faiss residual quantizer did not train")
    index.add(values)
    flat = faiss_module.vector_to_array(index.rq.codebooks)
    centroids = np.asarray(flat, dtype=np.float32).reshape(
        codebook_count, codebook_size, values.shape[1]
    )
    codes = _decode_faiss_codes(
        index.rq.compute_codes(values),
        codebook_count=codebook_count,
        bits_per_code=bits,
        faiss_module=faiss_module,
    )
    if codes.shape != (values.shape[0], codebook_count):
        raise AssertionError("unexpected RQ code shape")
    if int(codes.min()) < 0 or int(codes.max()) >= codebook_size:
        raise RuntimeError("Faiss emitted an out-of-range code")
    return codes, centroids


def semantic_token_strings(
    item_to_codes: Mapping[str, Sequence[int]], *, n_latent_tokens: int = 8
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Build GRAM-visible position-specific SID and latent token strings."""

    if not item_to_codes:
        raise ValueError("semantic token export requires a catalog")
    digit_count = len(next(iter(item_to_codes.values())))
    mapping: dict[str, str] = {}
    observed: set[str] = set()
    for item, codes in item_to_codes.items():
        if len(codes) != digit_count:
            raise ValueError("semantic IDs have inconsistent lengths")
        tokens = tuple(f"<s17_sid{digit}_{int(code)}>" for digit, code in enumerate(codes))
        mapping[item] = " ".join(tokens)
        observed.update(tokens)
    latent = tuple(f"<s17_latent_{index}>" for index in range(n_latent_tokens))
    vocabulary = tuple(sorted(observed)) + latent
    return mapping, vocabulary


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_tokenizer_artifacts(
    *,
    output_dir: Path,
    official_cache_dir: Path,
    catalog_items: Sequence[str],
    fit_mask: np.ndarray,
    sentence_embeddings: np.ndarray,
    transformed_embeddings: np.ndarray,
    pca: Any,
    raw_codes: np.ndarray,
    centroids: np.ndarray,
    resolved_codes: Mapping[str, Sequence[int]],
    resolution: PSIDResolutionSummary,
    spec: FullLatteTokenizerSpec,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically export native-LATTE and GRAM artifacts with one manifest."""

    output_dir.mkdir(parents=True, exist_ok=False)
    if tuple(resolved_codes) != tuple(catalog_items):
        raise ValueError("semantic-ID order differs from frozen metadata catalog")
    if len(set(tuple(int(x) for x in value) for value in resolved_codes.values())) != len(
        catalog_items
    ):
        raise ValueError("semantic-ID export contains aliases")

    raw_json = {
        item: [int(value) for value in resolved_codes[item]] for item in catalog_items
    }
    semantic_path = output_dir / "item_semantic_codes.json"
    atomic_json(semantic_path, raw_json)
    _atomic_numpy(output_dir / "sentence_embeddings.npy", sentence_embeddings)
    _atomic_numpy(output_dir / "pca_embeddings.npy", transformed_embeddings)
    _atomic_numpy(output_dir / "rq_raw_codes.npy", raw_codes)
    _atomic_numpy(output_dir / "rq_centroids.npy", centroids)
    _atomic_numpy(output_dir / "tokenizer_fit_mask.npy", np.asarray(fit_mask, dtype=bool))
    _atomic_npz(
        output_dir / "pca_state.npz",
        components_=np.asarray(pca.components_, dtype=np.float64),
        mean_=np.asarray(pca.mean_, dtype=np.float64),
        explained_variance_=np.asarray(pca.explained_variance_, dtype=np.float64),
        explained_variance_ratio_=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ),
        singular_values_=np.asarray(pca.singular_values_, dtype=np.float64),
    )

    gram_ids, vocabulary = semantic_token_strings(raw_json)
    _atomic_lines(
        output_dir / "gram_item_semantic_ids.txt",
        [f"{item} {gram_ids[item]}" for item in catalog_items],
    )
    _atomic_lines(output_dir / "gram_added_tokens.txt", list(vocabulary))

    official_path = official_cache_dir / "processed" / OFFICIAL_SEM_IDS_FILENAME
    if official_path.exists():
        existing = json.loads(official_path.read_text(encoding="utf-8"))
        if existing != raw_json:
            raise FileExistsError(
                f"official semantic-ID cache exists with different contents: {official_path}"
            )
    else:
        atomic_json(official_path, raw_json)

    artifact_paths = [
        semantic_path,
        output_dir / "sentence_embeddings.npy",
        output_dir / "pca_embeddings.npy",
        output_dir / "rq_raw_codes.npy",
        output_dir / "rq_centroids.npy",
        output_dir / "tokenizer_fit_mask.npy",
        output_dir / "pca_state.npz",
        output_dir / "gram_item_semantic_ids.txt",
        output_dir / "gram_added_tokens.txt",
        official_path,
    ]
    manifest = {
        "schema_version": "phase17.s17_fp0_full_data_tokenizer.v1",
        "created_at": utc_now(),
        "spec": asdict(spec),
        "catalog_items": len(catalog_items),
        "fit_catalog_items": int(np.asarray(fit_mask, dtype=bool).sum()),
        "fit_scope": "train_prefix_after_internal_dev_position_holdout",
        "catalog_transform_scope": "metadata_only_no_external_targets",
        "pca_fit_scope": "fit_mask_only",
        "rq_fit_scope": "fit_mask_only",
        "code_assignment_scope": "complete_frozen_metadata_catalog",
        "collision_resolution": asdict(resolution),
        "official_cache_path": str(official_path),
        "official_cache_filename": OFFICIAL_SEM_IDS_FILENAME,
        "official_cache_prevents_unmasked_pca_refit": True,
        "gram_added_token_count": len(vocabulary),
        "provenance": dict(provenance),
        "artifacts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths
        ],
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "external_target_materialized": False,
        "effect_experiment_started": False,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def build_full_data_tokenizer(
    *,
    root: Path,
    model_path: Path,
    output_dir: Path,
    official_cache_dir: Path,
    spec: FullLatteTokenizerSpec,
    device: str = "cuda",
    heartbeat: Callable[[str, dict[str, Any]], None] | None = None,
) -> TokenizerBuildResult:
    """Execute the full tokenizer build.  The protocol runner owns authorization."""

    spec.validate()
    root = root.resolve()
    bundle = build_latte_native_bundle(root=root)
    catalog_items = tuple(bundle.id_mapping["id2item"][1:])
    fit_mask = build_tokenizer_fit_mask(catalog_items, bundle)
    texts = [bundle.item2meta[item] for item in catalog_items]
    if heartbeat:
        heartbeat("encoding_catalog", {"current": 0, "total": len(texts), "unit": "items"})
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), local_files_only=True, device=device)
    sentence_embeddings = np.asarray(
        model.encode(
            texts,
            batch_size=spec.sentence_batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=device,
        ),
        dtype=np.float32,
    )
    if sentence_embeddings.shape != (len(catalog_items), spec.sentence_embedding_dim):
        raise RuntimeError(f"unexpected SentenceT5 shape: {sentence_embeddings.shape}")
    if heartbeat:
        heartbeat(
            "fitting_train_only_pca",
            {"current": int(fit_mask.sum()), "total": len(catalog_items), "unit": "fit/catalog_items"},
        )
    transformed, pca = fit_pca_train_only(
        sentence_embeddings,
        fit_mask,
        n_components=spec.pca_components,
        seed=spec.seed,
    )
    if heartbeat:
        heartbeat(
            "training_train_only_rqkmeans",
            {"current": int(fit_mask.sum()), "total": len(catalog_items), "unit": "fit/catalog_items"},
        )
    raw_codes, centroids = train_rqkmeans_train_only(
        transformed,
        fit_mask,
        codebook_count=spec.codebook_count,
        codebook_size=spec.codebook_size,
        faiss_threads=spec.faiss_threads,
        seed=spec.seed,
    )
    item_to_codes = {
        item: tuple(int(value) for value in raw_codes[index])
        for index, item in enumerate(catalog_items)
    }
    resolved, resolution = resolve_rqkmeans_psid_conflicts(
        item_to_codes,
        centroids,
        top_k_per_digit=spec.conflict_top_k_per_digit,
    )
    manifest = write_tokenizer_artifacts(
        output_dir=output_dir,
        official_cache_dir=official_cache_dir,
        catalog_items=catalog_items,
        fit_mask=fit_mask,
        sentence_embeddings=sentence_embeddings,
        transformed_embeddings=transformed,
        pca=pca,
        raw_codes=raw_codes,
        centroids=centroids,
        resolved_codes=resolved,
        resolution=resolution,
        spec=spec,
        provenance={
            "sentence_model_path": str(model_path),
            "model_revision": spec.model_revision,
            "data_adapter": "experiment.phase17.core.full_latte_native_adapter",
        },
    )
    if heartbeat:
        heartbeat(
            "tokenizer_artifacts_written",
            {"current": len(catalog_items), "total": len(catalog_items), "unit": "items"},
        )
    return TokenizerBuildResult(manifest=manifest, semantic_ids=resolved, resolution=resolution)
