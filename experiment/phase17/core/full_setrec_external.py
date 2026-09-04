"""Frozen external-D0 inference and statistics for Stage17 FP3 Full SETRec.

The inference entry point consumes an already sealed FP1/FP2 materialized
bundle.  It has no path to the raw D0 projection, D1/D2, official test, or
Sports data.  All SETRec arms use their train-prefix-internal-dev-selected
beta and full-catalog item scores; the frozen FP2 G0 predictions are consumed
only by the family analysis layer.
"""

from __future__ import annotations

import gc
import inspect
import json
import math
import os
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

import torch

from .full_latte_external_evaluator import catastrophic_subgroups
from .full_setrec_backend import (
    N_QUERY,
    SETREC_ARMS,
    build_full_setrec_model,
    history_visibility_mask,
    query_visibility_mask,
)
from .full_setrec_contracts import full_set_recovery, paper_sparse_history_mask
from .full_setrec_executor import SetRecBatchBuilder, to_device
from .fullport_data import FullportExternalExample, FullportTrainUser


TOP_K = 50


def read_sealed_bundle_views(
    path: Path,
) -> tuple[list[FullportTrainUser], list[FullportExternalExample]]:
    """Read the sealed bundle once and return aligned train/external views."""

    train_users: list[FullportTrainUser] = []
    examples: list[FullportExternalExample] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user_id = str(row.get("user_id", ""))
            train_items = tuple(str(item) for item in row.get("train_items", []))
            history = tuple(str(item) for item in row.get("history", []))
            target = str(row.get("target", ""))
            if (
                not user_id
                or user_id in seen
                or not train_items
                or not history
                or not target
                or history != train_items[-20:]
            ):
                raise ValueError(f"invalid sealed FP3 bundle row at {path}:{line_number}")
            seen.add(user_id)
            train_users.append(FullportTrainUser(user_id, train_items))
            examples.append(FullportExternalExample(user_id, history, target))
    if not examples:
        raise ValueError("sealed FP3 external bundle is empty")
    return train_users, examples


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


def _load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise TypeError("FP3 checkpoint payload is not a mapping")
    return payload


def attention_contract_diagnostics(arm_id: str) -> dict[str, Any]:
    """Fail-closed structural check for the frozen history/query contracts."""

    if arm_id not in SETREC_ARMS:
        raise ValueError(f"unknown SETRec arm: {arm_id}")
    item_mask = torch.ones((1, 20), dtype=torch.bool)
    observed_history, token_valid = history_visibility_mask(arm_id, item_mask)
    if arm_id in {"S1P_SETREC_PAPER_FAITHFUL", "S2_GRAM_SETREC_PAPER_FULL"}:
        expected_history = paper_sparse_history_mask(
            n_items=20, n_tokens_per_item=N_QUERY
        )[None]
        sparse_history_active = True
    else:
        expected_history = torch.ones_like(observed_history)
        sparse_history_active = False
    observed_query = query_visibility_mask(arm_id, device=torch.device("cpu"))
    expected_query = (
        torch.ones((N_QUERY, N_QUERY), dtype=torch.bool).tril()
        if arm_id == "S0_SETREC_ORDERED_CONTROL"
        else torch.eye(N_QUERY, dtype=torch.bool)
    )
    forbidden_visibility_count = int(
        (observed_history & ~expected_history).sum().item()
        + (observed_query & ~expected_query).sum().item()
    )
    missing_expected_visibility_count = int(
        (~observed_history & expected_history).sum().item()
        + (~observed_query & expected_query).sum().item()
    )
    return {
        "arm_id": arm_id,
        "history_tokens": int(observed_history.shape[-1]),
        "all_history_tokens_valid": bool(token_valid.all()),
        "sparse_history_active": sparse_history_active,
        "independent_query_active": arm_id != "S0_SETREC_ORDERED_CONTROL",
        "ordered_query_control_active": arm_id == "S0_SETREC_ORDERED_CONTROL",
        "forbidden_visibility_count": forbidden_visibility_count,
        "missing_expected_visibility_count": missing_expected_visibility_count,
        "contract_pass": (
            forbidden_visibility_count == 0
            and missing_expected_visibility_count == 0
            and bool(token_valid.all())
        ),
    }


def _prediction_row(
    *,
    arm_id: str,
    variant: str,
    example: FullportExternalExample,
    ranking: Sequence[str],
    scores: Sequence[float],
    combined_target_rank: int,
    per_query_target_ranks: Sequence[int],
    per_query_recovered: Sequence[bool],
    full_set_recovered: bool,
    query_norms: Sequence[float],
    semantic_reconstruction_mse: float,
    selected_beta: float,
    latency_seconds: float,
) -> dict[str, Any]:
    try:
        target_rank = list(ranking).index(example.target) + 1
    except ValueError:
        target_rank = 0
    return {
        "schema_version": "phase17.s17_fp3_external_prediction.v1",
        "arm_id": arm_id,
        "variant": variant,
        "user_id": example.user_id,
        "target": example.target,
        "ranking": list(ranking),
        "scores": [float(value) for value in scores],
        "target_rank": target_rank,
        "latency_seconds": float(latency_seconds),
        "mechanism": {
            "continuous_identifier_active": True,
            "full_catalog_grounding": True,
            "candidate_eligibility": "all_catalog_items_including_history",
            "selected_beta": float(selected_beta),
            "combined_grounding_target_rank": int(combined_target_rank),
            "per_query_target_ranks": [int(value) for value in per_query_target_ranks],
            "per_query_target_top1_recovered": [
                bool(value) for value in per_query_recovered
            ],
            "full_set_recovered": bool(full_set_recovered),
            "query_norms": [float(value) for value in query_norms],
            "semantic_reconstruction_mse": float(semantic_reconstruction_mse),
            "semantic_reconstruction_finite": math.isfinite(
                semantic_reconstruction_mse
            ),
            "valid_item_ranking": (
                len(ranking) == TOP_K and len(set(ranking)) == TOP_K
            ),
        },
    }


@torch.no_grad()
def evaluate_external_arm(
    root: Path,
    arm_id: str,
    checkpoint: Path,
    examples: Sequence[FullportExternalExample],
    output: Path,
    *,
    selected_beta: float,
    batch_size: int,
    device: torch.device,
    heartbeat: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Run one frozen FP3 arm over an in-memory sealed external cohort."""

    if arm_id not in SETREC_ARMS:
        raise ValueError(f"unknown SETRec arm: {arm_id}")
    if device.type != "cuda" or not checkpoint.is_file() or not examples:
        raise ValueError("external SETRec inference needs CUDA, checkpoint and examples")
    if output.exists() or batch_size <= 0 or not 0.0 <= selected_beta <= 1.0:
        raise ValueError("invalid external SETRec output/batch/beta contract")
    payload = _load_trusted_checkpoint(checkpoint)
    if (
        payload.get("schema_version") != "phase17.s17_fp3_checkpoint.v1"
        or payload.get("arm_id") != arm_id
        or payload.get("external_target_materialized") is not False
        or float(payload["internal_dev"]["selected_beta"]) != float(selected_beta)
    ):
        raise RuntimeError("FP3 checkpoint identity or selected beta drifted")
    model, tokenizer = build_full_setrec_model(root, arm_id, seed=2023)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    builder = SetRecBatchBuilder(root, arm_id, tokenizer)
    model.to(device).eval()
    catalog = builder.catalog.ordered_items
    variant = f"beta{selected_beta:.1f}_full_catalog"
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch_started = time.monotonic()
        batch = to_device(builder(batch_examples), device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = model(batch, beta=selected_beta)
        per_dimension = result.grounding.per_dimension_scores.float()
        combined = result.grounding.item_scores.float()
        if not bool(torch.isfinite(per_dimension).all()) or not bool(
            torch.isfinite(combined).all()
        ):
            raise FloatingPointError(f"non-finite external grounding for {arm_id}")
        top_scores, top_indices = combined.topk(k=TOP_K, dim=1)
        target_scores = combined.gather(1, batch.target_item_indices[:, None])
        combined_ranks = combined.gt(target_scores).sum(dim=1) + 1
        query_target_scores = per_dimension.gather(
            2,
            batch.target_item_indices[None, :, None].expand(
                per_dimension.shape[0], -1, 1
            ),
        )
        query_ranks = per_dimension.gt(query_target_scores).sum(dim=2) + 1
        query_recovered = per_dimension.argmax(dim=2).eq(
            batch.target_item_indices[None]
        )
        set_recovered = full_set_recovery(
            per_dimension, batch.target_item_indices
        )
        query_norms = result.query_outputs.float().norm(dim=-1)
        reconstruction_mse = float(
            torch.nn.functional.mse_loss(
                result.semantic_reconstruction.float(),
                model.semantic_features.float(),
            )
        )
        torch.cuda.synchronize(device)
        per_user_latency = (time.monotonic() - batch_started) / len(batch_examples)
        for row_index, example in enumerate(batch_examples):
            ranking = [catalog[index] for index in top_indices[row_index].tolist()]
            scores = top_scores[row_index].tolist()
            if len(ranking) != TOP_K or len(set(ranking)) != TOP_K:
                raise RuntimeError("SETRec full-catalog top-50 is invalid")
            rows.append(
                _prediction_row(
                    arm_id=arm_id,
                    variant=variant,
                    example=example,
                    ranking=ranking,
                    scores=scores,
                    combined_target_rank=int(combined_ranks[row_index]),
                    per_query_target_ranks=query_ranks[:, row_index].tolist(),
                    per_query_recovered=query_recovered[:, row_index].tolist(),
                    full_set_recovered=bool(set_recovered[row_index]),
                    query_norms=query_norms[row_index].tolist(),
                    semantic_reconstruction_mse=reconstruction_mse,
                    selected_beta=selected_beta,
                    latency_seconds=per_user_latency,
                )
            )
        completed = min(start + len(batch_examples), len(examples))
        if heartbeat is not None:
            heartbeat("external_full_catalog_inference", completed, len(examples))
    _atomic_jsonl(output, rows)
    contract = attention_contract_diagnostics(arm_id)
    wall_seconds = time.monotonic() - started
    del model, tokenizer, builder, payload
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "schema_version": "phase17.s17_fp3_external_arm_result.v1",
        "arm_id": arm_id,
        "external_users": len(examples),
        "prediction_rows": len(rows),
        "variant": variant,
        "selected_beta": float(selected_beta),
        "batch_size": int(batch_size),
        "wall_seconds": wall_seconds,
        "attention_contract": contract,
        "raw_external_projection_reopened": False,
        "bundle_read_by_family_worker": True,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }


def summarize_mechanisms(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    attention_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty SETRec predictions")
    mechanisms = [row.get("mechanism") for row in rows.values()]
    if not all(isinstance(value, Mapping) for value in mechanisms):
        raise ValueError("SETRec prediction is missing mechanism diagnostics")
    values = [dict(value) for value in mechanisms]
    per_query_ranks = [
        fmean(float(row["per_query_target_ranks"][query]) for row in values)
        for query in range(N_QUERY)
    ]
    per_query_recovery = [
        fmean(
            float(row["per_query_target_top1_recovered"][query]) for row in values
        )
        for query in range(N_QUERY)
    ]
    norms = [float(value) for row in values for value in row["query_norms"]]
    latencies = [float(row.get("latency_seconds", 0.0)) for row in rows.values()]
    return {
        "available": True,
        "users": len(rows),
        "continuous_identifier_active": all(
            row.get("continuous_identifier_active") is True for row in values
        ),
        "full_catalog_grounding": all(
            row.get("full_catalog_grounding") is True for row in values
        ),
        "candidate_eligibility": "all_catalog_items_including_history",
        "full_set_recovery_rate": fmean(
            float(row["full_set_recovered"]) for row in values
        ),
        "per_query_target_top1_recovery_rate": per_query_recovery,
        "mean_per_query_target_rank": per_query_ranks,
        "mean_combined_grounding_target_rank": fmean(
            float(row["combined_grounding_target_rank"]) for row in values
        ),
        "query_norm_min": min(norms),
        "query_norm_max": max(norms),
        "query_norms_finite_nonzero": bool(norms)
        and all(math.isfinite(value) and value > 0 for value in norms),
        "semantic_reconstruction_finite": all(
            row.get("semantic_reconstruction_finite") is True for row in values
        ),
        "mean_semantic_reconstruction_mse": fmean(
            float(row["semantic_reconstruction_mse"]) for row in values
        ),
        "valid_item_rate": fmean(
            float(row["valid_item_ranking"]) for row in values
        ),
        "mean_latency_seconds_per_user": fmean(latencies),
        "attention_contract": dict(attention_contract),
    }


def mechanism_active(mechanism: Mapping[str, Any]) -> bool:
    contract = mechanism.get("attention_contract", {})
    return bool(
        mechanism.get("continuous_identifier_active") is True
        and mechanism.get("full_catalog_grounding") is True
        and float(mechanism.get("full_set_recovery_rate", 0.0)) > 0.0
        and float(mechanism.get("valid_item_rate", 0.0)) == 1.0
        and mechanism.get("query_norms_finite_nonzero") is True
        and mechanism.get("semantic_reconstruction_finite") is True
        and contract.get("contract_pass") is True
        and int(contract.get("forbidden_visibility_count", -1)) == 0
    )


def fp3_gate(
    comparisons: Mapping[str, Mapping[str, Any]],
    subgroup_s2_vs_s0: Mapping[str, Any],
    mechanisms: Mapping[str, Mapping[str, Any]],
    *,
    integrity_valid: bool,
) -> dict[str, Any]:
    s1p_s0 = comparisons["S1P_MINUS_S0"]["effects"]
    s2_s0 = comparisons["S2_MINUS_S0"]["effects"]
    s2_g0 = comparisons["S2_MINUS_G0"]["effects"]
    catastrophes = catastrophic_subgroups(subgroup_s2_vs_s0, threshold=-0.003)
    checks = {
        "s1r_mechanism_active": mechanism_active(mechanisms["S1R_SETREC_REPO_PARITY"]),
        "s1p_vs_s0_ndcg_positive": float(
            s1p_s0["ndcg@10"]["mean_delta"]
        )
        > 0.0,
        "s1p_mechanism_active": mechanism_active(
            mechanisms["S1P_SETREC_PAPER_FAITHFUL"]
        ),
        "s2_vs_s0_ndcg_ge_0.0015": float(
            s2_s0["ndcg@10"]["mean_delta"]
        )
        >= 0.0015,
        "s2_vs_s0_ndcg_ci95_low_positive": float(
            s2_s0["ndcg@10"]["ci95_low"]
        )
        > 0.0,
        "s2_vs_g0_ndcg_ge_0.0015": float(
            s2_g0["ndcg@10"]["mean_delta"]
        )
        >= 0.0015,
        "s2_vs_s0_hit_nonnegative": float(
            s2_s0["hit@10"]["mean_delta"]
        )
        >= 0.0,
        "s2_mechanism_active": mechanism_active(
            mechanisms["S2_GRAM_SETREC_PAPER_FULL"]
        ),
        "no_catastrophic_large_subgroup": not catastrophes,
        "integrity_valid": bool(integrity_valid),
    }
    return {
        "verdict": "FP3_STRONG_PASS" if all(checks.values()) else "FP3_NOT_STRONG_PASS",
        "checks": checks,
        "catastrophic_subgroups": catastrophes,
    }
