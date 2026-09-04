"""Checkpoint-frozen GPU inference for the Stage17 FP1/FP2 D0 evaluator.

The caller supplies a sealed, already materialized example bundle.  This
module has no path to the original D0 projection and therefore cannot perform
an additional target read.
"""

from __future__ import annotations

import json
import inspect
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .full_latte_arm_contracts import ARM_IDS
from .fullport_data import FullportExternalExample


SEED = 2023
LATTE_ARMS = {"N1_NATIVE_LATTE", "G2_GRAM_LATTE_FULL"}
NATIVE_ARMS = {"N0_NATIVE_PSID", "N1_NATIVE_LATTE"}
GRAM_ARMS = {"G0_GRAM_B0_FRESH", "G1_GRAM_PSID_FULL", "G2_GRAM_LATTE_FULL"}


def read_materialized_bundle(path: Path) -> list[FullportExternalExample]:
    examples: list[FullportExternalExample] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user_id = str(row.get("user_id", ""))
            history = tuple(str(item) for item in row.get("history", []))
            target = str(row.get("target", ""))
            if not user_id or user_id in seen or not history or not target:
                raise ValueError(f"invalid materialized example at {path}:{line_number}")
            seen.add(user_id)
            examples.append(
                FullportExternalExample(
                    user_id=user_id,
                    history=history,
                    target=target,
                )
            )
    if not examples:
        raise ValueError("empty materialized external example bundle")
    return examples


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _move(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _trim_eos(sequence: Sequence[int], eos_token: int) -> tuple[int, ...]:
    values = tuple(int(token) for token in sequence)
    try:
        position = values.index(int(eos_token), 1)
        return values[: position + 1]
    except ValueError:
        return values


def _logaddexp(left: float, right: float) -> float:
    high, low = max(left, right), min(left, right)
    return high + math.log1p(math.exp(low - high))


def _aggregate_item_paths(
    paths: Sequence[tuple[str | None, int | None, float, tuple[int, ...]]],
    *,
    method: str,
    top_k: int = 50,
) -> tuple[list[str], list[float], Counter[str]]:
    if method not in {"identity", "agg_max", "agg_sum"}:
        raise ValueError(f"unknown aggregation method: {method}")
    scores: dict[str, float] = {}
    counts: Counter[str] = Counter()
    for item, _latent, score, _tokens in paths:
        if item is None:
            continue
        counts[item] += 1
        if item not in scores:
            scores[item] = float(score)
        elif method in {"identity", "agg_max"}:
            scores[item] = max(scores[item], float(score))
        else:
            scores[item] = _logaddexp(scores[item], float(score))
    ranked = sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:top_k]
    return [row[0] for row in ranked], [float(row[1]) for row in ranked], counts


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    position = 0
    while position < array.size:
        end = position + 1
        while end < array.size and array[order[end]] == array[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0
        position = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _mechanism_row(
    paths: Sequence[tuple[str | None, int | None, float, tuple[int, ...]]],
    *,
    target: str,
    ranking: Sequence[str],
    semantic_by_item: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    valid = [path for path in paths if path[0] is not None]
    counts = Counter(str(path[0]) for path in valid)
    latent_counts = Counter(
        str(path[1]) for path in valid if path[1] is not None
    )
    target_positions = [
        index for index, path in enumerate(valid, 1) if path[0] == target
    ]
    try:
        post_rank = ranking.index(target) + 1
    except ValueError:
        post_rank = 0
    distance_similarity: list[float] = []
    scores: list[float] = []
    target_semantic = tuple(int(value) for value in semantic_by_item[target])
    for item, _latent, score, _tokens in valid:
        item_semantic = tuple(int(value) for value in semantic_by_item[str(item)])
        distance = sum(a != b for a, b in zip(item_semantic, target_semantic))
        distance_similarity.append(float(-distance))
        scores.append(float(score))
    unique_items = len(counts)
    duplicate_paths = len(valid) - unique_items
    pre_rank = target_positions[0] if target_positions else 0
    pre_ndcg = 1.0 / math.log2(pre_rank + 1) if 0 < pre_rank <= 10 else 0.0
    post_ndcg = 1.0 / math.log2(post_rank + 1) if 0 < post_rank <= 10 else 0.0
    return {
        "generated_path_count": len(paths),
        "valid_path_count": len(valid),
        "valid_path_rate": len(valid) / len(paths) if paths else 0.0,
        "unique_item_count": unique_items,
        "duplicate_item_path_count": duplicate_paths,
        "duplicate_path_rate": duplicate_paths / len(valid) if valid else 0.0,
        "multi_path_item_rate": (
            sum(value > 1 for value in counts.values()) / unique_items
            if unique_items
            else 0.0
        ),
        "latent_counts": dict(sorted(latent_counts.items())),
        "latent_root_count": len(latent_counts),
        "target_path_survived": float(bool(target_positions)),
        "target_root_count": len(
            {
                path[1]
                for path in valid
                if path[0] == target and path[1] is not None
            }
        ),
        "pre_aggregation_target_rank": pre_rank,
        "post_aggregation_target_rank": post_rank,
        "pre_aggregation_ndcg@10": pre_ndcg,
        "post_aggregation_ndcg@10": post_ndcg,
        "aggregation_gain_ndcg@10": post_ndcg - pre_ndcg,
        "tree_distance_score_correlation": _spearman(distance_similarity, scores),
    }


def _prediction_row(
    *,
    arm_id: str,
    variant: str,
    example: FullportExternalExample,
    ranking: Sequence[str],
    scores: Sequence[float],
    mechanism: Mapping[str, Any] | None,
    latency_seconds: float,
) -> dict[str, Any]:
    try:
        target_rank = ranking.index(example.target) + 1
    except ValueError:
        target_rank = 0
    row: dict[str, Any] = {
        "schema_version": "phase17.s17_fp12_external_prediction.v1",
        "arm_id": arm_id,
        "variant": variant,
        "user_id": example.user_id,
        "target": example.target,
        "ranking": list(ranking),
        "scores": [float(value) for value in scores],
        "target_rank": target_rank,
        "latency_seconds": float(latency_seconds),
    }
    if mechanism is not None:
        row["mechanism"] = dict(mechanism)
    return row


def _native_paths(
    model,
    components,
    batch: Mapping[str, Any],
    *,
    beam: int,
) -> list[tuple[str | None, int | None, float, tuple[int, ...]]]:
    import torch

    tokenizer = components.tokenizer
    n_digit = int(tokenizer.n_digit)
    latte = components.arm_id == "N1_NATIVE_LATTE"
    with torch.no_grad():
        generated = model.t5.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=n_digit + (2 if latte else 1),
            num_beams=beam,
            num_return_sequences=beam,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
    sequences = generated.sequences.detach().cpu().tolist()
    scores = generated.sequences_scores.detach().cpu().tolist()
    item_by_semantic = {
        tuple(int(token) for token in tokens): str(item)
        for item, tokens in tokenizer.item2tokens.items()
    }
    output = []
    for sequence, score in zip(sequences, scores):
        values = tuple(int(token) for token in sequence)
        latent = values[1] if latte and len(values) >= n_digit + 2 else None
        start = 2 if latte else 1
        semantic = values[start : start + n_digit]
        valid_latent = (
            not latte
            or tokenizer.base_latent_token
            <= int(latent)
            < tokenizer.base_latent_token + tokenizer.n_latent_tokens
        )
        item = item_by_semantic.get(tuple(semantic)) if valid_latent else None
        output.append((item, int(latent) if latent is not None else None, float(score), values))
    return output


def _evaluate_native(
    root: Path,
    arm_id: str,
    checkpoint: Path,
    examples: Sequence[FullportExternalExample],
    heartbeat: Callable[[str, int, int], None] | None,
) -> list[dict[str, Any]]:
    import torch

    from .full_latte_native_backend import (
        build_official_native_components,
        collate_native_eval_batch,
        create_fresh_official_native_model,
    )

    device = torch.device("cuda:0")
    components = build_official_native_components(
        root, arm_id, device="cuda:0", num_beams=500
    )
    components.config["topk"] = [50]
    model = create_fresh_official_native_model(components, seed=SEED)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    semantic_by_item = {
        str(item): tuple(int(token) for token in tokens)
        for item, tokens in components.tokenizer.item2tokens.items()
    }
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, 1):
        batch = _move(collate_native_eval_batch(components, (example,)), device)
        for beam in (50, 500):
            started = time.monotonic()
            paths = _native_paths(model, components, batch, beam=beam)
            methods = ("agg_max", "agg_sum") if arm_id in LATTE_ARMS else ("identity",)
            for method in methods:
                ranking, scores, _counts = _aggregate_item_paths(paths, method=method)
                mechanism = (
                    _mechanism_row(
                        paths,
                        target=example.target,
                        ranking=ranking,
                        semantic_by_item=semantic_by_item,
                    )
                    if arm_id in NATIVE_ARMS
                    else None
                )
                rows.append(
                    _prediction_row(
                        arm_id=arm_id,
                        variant=f"beam{beam}_{method}",
                        example=example,
                        ranking=ranking,
                        scores=scores,
                        mechanism=mechanism,
                        latency_seconds=time.monotonic() - started,
                    )
                )
        if heartbeat is not None and (index == 1 or index % 25 == 0):
            heartbeat("external_inference", index, len(examples))
    return rows


def _gram_paths(
    model,
    tokenizer,
    batch: Mapping[str, Any],
    trie,
    *,
    item_paths: Mapping[str, Sequence[Sequence[int]]],
    max_length: int,
    beam: int,
    latte: bool,
) -> list[tuple[str | None, int | None, float, tuple[int, ...]]]:
    import torch

    path_to_item = {
        tuple(int(token) for token in path): str(item)
        for item, paths in item_paths.items()
        for path in paths
    }
    with torch.no_grad():
        generated = model.generate(
            input_ids=batch["item_text_ids"],
            attention_mask=batch["item_text_masks"],
            history_item_ids=batch["history_item_ids"],
            history_item_mask=batch["history_item_mask"],
            max_length=max_length,
            prefix_allowed_tokens_fn=trie.prefix_allowed_tokens_fn(),
            num_beams=beam,
            num_return_sequences=beam,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,
            use_cache=False,
        )
    sequences = generated["sequences"].detach().cpu().tolist()
    scores = generated["sequences_scores"].detach().cpu().tolist()
    output = []
    for sequence, score in zip(sequences, scores):
        trimmed = _trim_eos(sequence, tokenizer.eos_token_id)
        item = path_to_item.get(trimmed)
        latent = trimmed[1] if latte and len(trimmed) > 1 else None
        output.append((item, latent, float(score), trimmed))
    return output


def _load_trusted_checkpoint(torch_module: Any, checkpoint: Path) -> Any:
    """Load a frozen local checkpoint across old and new PyTorch signatures."""

    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch_module.load).parameters:
        kwargs["weights_only"] = False
    return torch_module.load(checkpoint, **kwargs)


def _evaluate_gram(
    root: Path,
    arm_id: str,
    checkpoint: Path,
    examples: Sequence[FullportExternalExample],
    heartbeat: Callable[[str, int, int], None] | None,
) -> list[dict[str, Any]]:
    import torch

    from .full_latte_gram_backend import (
        PrefixTree,
        build_gram_collator,
        create_fresh_gram_model,
        encoded_candidate_paths,
        load_gram_catalog,
        render_gram_example,
    )

    device = torch.device("cuda:0")
    catalog = load_gram_catalog(root, arm_id)
    tokenizer, collator = build_gram_collator(root, arm_id)
    item_paths = encoded_candidate_paths(tokenizer, arm_id, catalog)
    flat_paths = [path for paths in item_paths.values() for path in paths]
    trie = PrefixTree(flat_paths)
    max_length = max(len(path) for path in flat_paths)
    model = create_fresh_gram_model(root, arm_id, tokenizer, seed=SEED)
    payload = _load_trusted_checkpoint(torch, checkpoint)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    rows: list[dict[str, Any]] = []
    rng = random.Random(SEED)
    for index, example in enumerate(examples, 1):
        rendered = render_gram_example(
            example, arm_id=arm_id, catalog=catalog, rng=rng
        )
        batch = _move(collator([rendered.as_collator_row()]), device)
        for beam in (50, 500):
            started = time.monotonic()
            paths = _gram_paths(
                model,
                tokenizer,
                batch,
                trie,
                item_paths=item_paths,
                max_length=max_length,
                beam=beam,
                latte=arm_id in LATTE_ARMS,
            )
            methods = ("agg_max", "agg_sum") if arm_id in LATTE_ARMS else ("identity",)
            for method in methods:
                ranking, scores, _counts = _aggregate_item_paths(paths, method=method)
                mechanism = (
                    _mechanism_row(
                        paths,
                        target=example.target,
                        ranking=ranking,
                        semantic_by_item=catalog.semantic_codes,
                    )
                    if arm_id != "G0_GRAM_B0_FRESH"
                    else None
                )
                rows.append(
                    _prediction_row(
                        arm_id=arm_id,
                        variant=f"beam{beam}_{method}",
                        example=example,
                        ranking=ranking,
                        scores=scores,
                        mechanism=mechanism,
                        latency_seconds=time.monotonic() - started,
                    )
                )
        if heartbeat is not None and (index == 1 or index % 25 == 0):
            heartbeat("external_inference", index, len(examples))
    return rows


def evaluate_external_arm(
    root: Path,
    arm_id: str,
    checkpoint: Path,
    bundle: Path,
    output: Path,
    *,
    heartbeat: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen arm; the caller must verify hashes and authorization."""

    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown Stage17 FP arm: {arm_id}")
    if not checkpoint.is_file() or not bundle.is_file():
        raise FileNotFoundError("checkpoint or materialized example bundle is missing")
    if output.exists():
        raise FileExistsError("prediction output already exists")
    examples = read_materialized_bundle(bundle)
    started = time.monotonic()
    if arm_id in NATIVE_ARMS:
        rows = _evaluate_native(root, arm_id, checkpoint, examples, heartbeat)
    elif arm_id in GRAM_ARMS:
        rows = _evaluate_gram(root, arm_id, checkpoint, examples, heartbeat)
    else:  # pragma: no cover - protected by ARM_IDS
        raise AssertionError(arm_id)
    _atomic_jsonl(output, rows)
    return {
        "arm_id": arm_id,
        "external_users": len(examples),
        "prediction_rows": len(rows),
        "variants": sorted({row["variant"] for row in rows}),
        "wall_seconds": time.monotonic() - started,
        "external_target_materialized_by_worker": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }
