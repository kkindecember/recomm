#!/usr/bin/env python3
"""Frozen S16-5 Beauty S-AUX transfer: comparator freeze, validation, finalization."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from experiment.phase13.protocol.b1_portfolio_confirmation import (
    portfolio_ranking,
    unique_in_order,
)
from experiment.phase16.protocol.stage16_s4_toys_validation import (
    atomic_json,
    encode_token_catalog,
    faithful_rank,
    load_gram,
    ranking_metrics,
    read_metadata,
    read_paths,
    read_projected_sequences,
    read_set,
    read_validation_predictions,
    reset_peak_memory_stats_compat,
    saux_logits,
    saux_state,
    sha256_file,
    tokenize_history,
    verifier_score_lengths,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "stage16_s5_beauty_saux_v1"
STATE_FREEZE_VERDICT = "PASS_S16_5_BEAUTY_SAUX_STATE_AND_COMPARATOR_FREEZE"
COMPLETED_VERDICT = "COMPLETED_S16_5_BEAUTY_SAUX_FROZEN_TRANSFER"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    expected_method = {
        "draft_size": 50,
        "threshold": -1.8,
        "acceptance": "strict_gt",
        "guided_redraft": "current_live_verifier_beam_prefixes",
        "underfilled_live_round": "draft_all_finite_unseen_then_advance_verifier_beam",
        "candidate_chunk_size": 10,
        "target_aware_prefix_length": "max_2_and_longest_warm_prefix",
        "fallback": "official_rejected_plus_nonduplicated_live_verifier_beam",
        "adaptive_exit": "accepted_at_least_50_or_maximum_lexical_depth",
        "checkpoint_state_universe": "retained_warm_plus_padding",
        "history_content_universe": "complete_catalog_plus_padding",
        "official_predict_semantics": True,
        "beam_size": 50,
    }
    expected_portfolio = {
        "candidate_exclusion_prefix": 7,
        "ranking_anchor_count": 8,
        "candidate_count": 3,
        "portfolio_size": 2,
        "ranking_size": 50,
        "method": "stage13_v1_r2_unconditional_portfolio2",
    }
    universe = config.get("expected_universe", {})
    resources = config.get("resources", {})
    if config.get("schema_version") != SCHEMA:
        raise ValueError("Unexpected S16-5 Beauty schema")
    if (
        config.get("domain") != "Beauty_cold50"
        or config.get("split") != "validation"
        or config.get("seed") != 1502
        or config.get("physical_gpu") != 0
        or config.get("visible_gpu") != 0
    ):
        raise ValueError("S16-5 Beauty domain/seed/GPU identity drift")
    if config.get("test_read") is not False or config.get("automatic_retry") is not False:
        raise ValueError("S16-5 test/retry boundary drift")
    if config.get("validation_used_for_state_selection_or_tuning") is not False:
        raise ValueError("S16-5 validation selection boundary drift")
    if config.get("faithful_inference") != expected_method:
        raise ValueError("S16-5 Toys-frozen S-AUX inference contract drift")
    if config.get("portfolio2_contract") != expected_portfolio:
        raise ValueError("S16-5 portfolio@2 contract drift")
    if universe != {
        "validation_events": 10655,
        "cold_validation_events": 5287,
        "warm_validation_events": 5368,
        "catalog_items": 12101,
        "cold_catalog_items": 6052,
        "warm_catalog_items": 6049,
        "ranking_size": 50,
    }:
        raise ValueError("S16-5 Beauty universe contract drift")
    if (
        resources.get("gpu_count") != 1
        or resources.get("minimum_free_mib") != 9216
        or resources.get("hard_timeout_seconds") != 604800
        or resources.get("existing_processes_modified") is not False
    ):
        raise ValueError("S16-5 resource contract drift")
    if config.get("statistics", {}).get("holm_family") != ["S-AUX_vs_F0"]:
        raise ValueError("S16-5 Holm family drift")


def _config(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    return config


def _output(config: Mapping[str, Any]) -> Path:
    return ROOT / str(config["output_dir"])


def _verify_regular(declaration: Mapping[str, str], label: str) -> Path:
    path = ROOT / str(declaration["path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing/non-regular S16-5 input: {label}")
    if sha256_file(path) != declaration["sha256"]:
        raise ValueError(f"S16-5 frozen input SHA drift: {label}")
    return path


def _verify_all_inputs(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: _verify_regular(declaration, name)
        for name, declaration in config["inputs"].items()
    }


def _aggregate_rankings(
    rows: Sequence[Mapping[str, Any]], ranking_key: str, subset: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if subset == "overall" or bool(row["is_cold"]) == (subset == "cold")
    ]
    if not selected:
        raise ValueError(f"Empty S16-5 aggregate subset: {subset}")
    metrics = [ranking_metrics(row[ranking_key], str(row["target_item"])) for row in selected]
    return {
        "events": len(selected),
        "hit@50": float(np.mean([row["hit@50"] for row in metrics])),
        "ndcg@10": float(np.mean([row["ndcg@10"] for row in metrics])),
    }


def _assert_close(left: float, right: float, label: str, tolerance: float = 1e-15) -> None:
    if abs(float(left) - float(right)) > tolerance:
        raise ValueError(f"S16-5 comparator aggregate mismatch: {label}")


def freeze_comparators(config_path: Path) -> dict[str, Any]:
    """Open Beauty validation only after the train-only checkpoint is frozen."""

    config = _config(config_path)
    output = _output(config)
    training_summary_path = ROOT / config["training"]["summary_path"]
    checkpoint_path = ROOT / config["training"]["checkpoint_path"]
    if not training_summary_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("S16-5 train-only Beauty S-AUX state is not complete")
    training_summary = load_json(training_summary_path)
    if (
        training_summary.get("verdict") != "PASS_S16_5_BEAUTY_SAUX_STATE_FREEZE"
        or training_summary.get("test_read") is not False
    ):
        raise ValueError("S16-5 Beauty S-AUX state freeze Gate is not PASS")
    method_source = _verify_regular(config["toys_frozen_method_source"], "toys_frozen_method_source")
    inputs = _verify_all_inputs(config)
    projected = read_projected_sequences(inputs["projected_train_validation_sequences"])
    p0 = read_validation_predictions(inputs["phase13_p0_predictions"])
    cold = read_set(inputs["cold_items"])
    warm = read_set(inputs["warm_items"])
    universe = config["expected_universe"]
    if set(projected) != set(p0) or len(projected) != universe["validation_events"]:
        raise ValueError("S16-5 projected/P0 user universe drift")
    if len(cold) != universe["cold_catalog_items"] or len(warm) != universe["warm_catalog_items"]:
        raise ValueError("S16-5 Beauty cold/warm universe drift")

    contract = config["portfolio2_contract"]
    comparator_rows: list[dict[str, Any]] = []
    cold_events = 0
    path = output / "comparators/portfolio2_predictions_validation.jsonl"
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to overwrite the S16-5 comparator freeze")
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8") as handle:
        for event_index, (user, sequence) in enumerate(projected.items(), 1):
            source = p0[user]
            target = str(sequence[-1])
            if str(source.get("target")) != target:
                raise ValueError(f"S16-5 P0/projected target mismatch: {user}")
            gram = unique_in_order([str(item) for item in source["v0_top50"]])
            resolver = unique_in_order([str(item) for item in source["resolver_top50"]])
            protected = set(gram[: contract["candidate_exclusion_prefix"]])
            candidates = [item for item in resolver if item in cold and item not in protected][
                : contract["candidate_count"]
            ]
            if len(candidates) != contract["candidate_count"]:
                raise ValueError(f"S16-5 portfolio candidates underfilled: {user}")
            portfolio = portfolio_ranking(
                gram, resolver, candidates, contract["portfolio_size"]
            )[: contract["ranking_size"]]
            if len(gram) != 50 or len(portfolio) != 50 or len(set(portfolio)) != 50:
                raise ValueError(f"S16-5 invalid F0/portfolio ranking: {user}")
            is_cold = target in cold
            cold_events += int(is_cold)
            row = {
                "event_index": event_index,
                "user_id": user,
                "target_item": target,
                "is_cold": is_cold,
                "f0_top50": gram,
                "portfolio_candidates": candidates,
                "portfolio2_top50": portfolio,
                "method": contract["method"],
            }
            comparator_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    if cold_events != universe["cold_validation_events"]:
        raise ValueError("S16-5 Beauty validation subset drift")

    phase13 = load_json(inputs["phase13_portfolio2_summary"])
    if (
        phase13.get("primary_candidate") != "unconditional_portfolio2"
        or phase13.get("n_users") != universe["validation_events"]
        or phase13.get("n_cold_users") != universe["cold_validation_events"]
        or phase13.get("n_skipped_insufficient_candidates") != 0
        or phase13.get("test_predictions_opened") is not False
    ):
        raise ValueError("S16-5 Phase13 Beauty portfolio lineage drift")
    observed: dict[str, Any] = {}
    max_abs_error = 0.0
    for subset, phase13_subset in (("overall", "all"), ("cold", "cold"), ("warm", "warm")):
        observed[subset] = {
            "F0": _aggregate_rankings(comparator_rows, "f0_top50", subset),
            "R2": _aggregate_rankings(comparator_rows, "portfolio2_top50", subset),
        }
        for arm, phase13_arm in (("F0", "v0_gram"), ("R2", "unconditional_portfolio2")):
            for metric in ("hit@50", "ndcg@10"):
                left = observed[subset][arm][metric]
                right = phase13["pareto_front"][phase13_arm][phase13_subset][metric]
                max_abs_error = max(max_abs_error, abs(left - right))
                _assert_close(left, right, f"{arm}/{subset}/{metric}")

    input_sha = {name: sha256_file(path_value) for name, path_value in inputs.items()}
    state = {
        "verdict": STATE_FREEZE_VERDICT,
        "training_summary_path": str(training_summary_path.relative_to(ROOT)),
        "training_summary_sha256": sha256_file(training_summary_path),
        "saux_checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "saux_checkpoint_sha256": sha256_file(checkpoint_path),
        "toys_frozen_method_source": str(method_source.relative_to(ROOT)),
        "toys_frozen_method_source_sha256": sha256_file(method_source),
        "input_sha256": input_sha,
        "comparator_predictions_path": str(path.relative_to(ROOT)),
        "comparator_predictions_sha256": sha256_file(path),
        "comparator_aggregate": observed,
        "phase13_comparator_max_abs_error": max_abs_error,
        "validation_opened_after_state_freeze": True,
        "validation_used_for_tuning_or_state_selection": False,
        "test_read": False,
        "automatic_retry": False,
    }
    atomic_json(output / "state_and_comparator_freeze.json", state)
    print(STATE_FREEZE_VERDICT, flush=True)
    return state


def _load_comparator_rows(path: Path, expected_events: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user = str(row.get("user_id"))
            if user in rows or row.get("event_index") != line_number:
                raise ValueError(f"Invalid S16-5 comparator row: {line_number}")
            rows[user] = row
    if len(rows) != expected_events:
        raise ValueError("S16-5 comparator event count drift")
    return rows


@torch.inference_mode()
def validate(config_path: Path) -> dict[str, Any]:
    config = _config(config_path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S16-5 Beauty S-AUX requires exactly one visible GPU")
    output = _output(config)
    state_path = output / "state_and_comparator_freeze.json"
    if not state_path.is_file():
        raise ValueError("S16-5 state/comparator freeze is missing")
    state = load_json(state_path)
    if state.get("verdict") != STATE_FREEZE_VERDICT:
        raise ValueError("S16-5 state/comparator Gate is not PASS")
    checkpoint = ROOT / config["training"]["checkpoint_path"]
    comparator_path = ROOT / state["comparator_predictions_path"]
    if (
        sha256_file(checkpoint) != state["saux_checkpoint_sha256"]
        or sha256_file(comparator_path) != state["comparator_predictions_sha256"]
    ):
        raise ValueError("S16-5 frozen state/comparator SHA drift")
    inputs = _verify_all_inputs(config)
    if {name: sha256_file(path) for name, path in inputs.items()} != state["input_sha256"]:
        raise ValueError("S16-5 validation input identity changed after freeze")

    projected = read_projected_sequences(inputs["projected_train_validation_sequences"])
    comparators = _load_comparator_rows(
        comparator_path, config["expected_universe"]["validation_events"]
    )
    metadata = read_metadata(inputs["item_metadata"])
    lexical_paths = read_paths(inputs["lexical_paths"])
    cold = read_set(inputs["cold_items"])
    warm = read_set(inputs["warm_items"])
    retained = read_set(inputs["retained_warm_items"])
    pseudo = read_set(inputs["pseudo_cold_items"])
    if (
        set(projected) != set(comparators)
        or cold & warm
        or cold | warm != set(metadata)
        or set(lexical_paths) != set(metadata)
        or retained & pseudo
        or retained | pseudo != warm
    ):
        raise ValueError("S16-5 Beauty catalog/validation universe drift")
    if any(not set(sequence[:-1]).issubset(warm) for sequence in projected.values()):
        raise ValueError("S16-5 Beauty validation history contains a non-warm item")

    device = torch.device("cuda:0")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_index = reset_peak_memory_stats_compat(device)
    started = time.time()
    ordered_items = sorted(metadata)
    tokenizer = AutoTokenizer.from_pretrained(
        str(inputs["t5_config"].parent), local_files_only=True
    )
    token_paths = encode_token_catalog(tokenizer, lexical_paths)
    score_lengths = verifier_score_lengths(token_paths, warm, cold, arm="S-AUX")
    model = load_gram(inputs["gram_config"], inputs["gram_f0_checkpoint"], device).eval()
    aux = saux_state(
        checkpoint=checkpoint,
        embedding_path=inputs["content_embeddings"],
        retained_items=retained,
        ordered_items=ordered_items,
        device=device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    validation_dir = output / "validation"
    if validation_dir.exists() or validation_dir.is_symlink():
        raise FileExistsError("Refusing to overwrite S16-5 Beauty validation")
    validation_dir.mkdir(parents=True)
    prediction_path = validation_dir / "predictions_validation.jsonl"
    progress_path = output / "validation_progress.json"
    mechanism_totals = {
        "rounds": 0,
        "drafted": 0,
        "accepted": 0,
        "redraft_rounds": 0,
        "draft_capacity_shortfall_rounds": 0,
        "zero_finite_draft_rounds": 0,
        "beam_decoder_rows": 0,
        "candidate_verifier_forwards": 0,
        "rankings_different_from_f0": 0,
    }
    rows: list[dict[str, Any]] = []
    faithful = config["faithful_inference"]
    with prediction_path.open("x", encoding="utf-8") as handle:
        for event_index, (user, sequence) in enumerate(projected.items(), 1):
            history, target = sequence[:-1][-20:], sequence[-1]
            comparator = comparators[user]
            if comparator["target_item"] != target:
                raise ValueError(f"S16-5 comparator/projected target mismatch: {user}")
            context = tokenize_history(history, metadata, lexical_paths, tokenizer, device)
            logits = saux_logits(aux, history, device)
            ranking, mechanism = faithful_rank(
                model=model,
                context=context,
                draft_logits=logits,
                ordered_items=ordered_items,
                token_paths=token_paths,
                score_lengths=score_lengths,
                tokenizer=tokenizer,
                draft_size=faithful["draft_size"],
                threshold=faithful["threshold"],
                beam_size=faithful["beam_size"],
                candidate_chunk_size=faithful["candidate_chunk_size"],
            )
            if len(ranking) != 50 or len(set(ranking)) != 50 or not set(ranking).issubset(metadata):
                raise RuntimeError("S16-5 S-AUX lost strict unique catalog top-50")
            row = {
                "event_index": event_index,
                "user_id": user,
                "target_item": target,
                "is_cold": target in cold,
                "arm": "S-AUX",
                "top50": ranking,
                "metrics": ranking_metrics(ranking, target),
                "mechanism": mechanism,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            for key in mechanism_totals:
                if key != "rankings_different_from_f0":
                    mechanism_totals[key] += int(mechanism.get(key, 0))
            mechanism_totals["rankings_different_from_f0"] += int(
                ranking != comparator["f0_top50"]
            )
            if event_index % config["progress_interval_events"] == 0 or event_index == len(projected):
                handle.flush()
                os.fsync(handle.fileno())
                atomic_json(
                    progress_path,
                    {
                        "stage": "validation",
                        "progress_current": event_index,
                        "progress_total": len(projected),
                        "progress_unit": "validation_events",
                        "updated_at_epoch": time.time(),
                    },
                )
                print(f"[s16-s5-S-AUX] events={event_index}/{len(projected)}", flush=True)

    elapsed = time.time() - started
    summary = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "status": "completed",
        "verdict": COMPLETED_VERDICT,
        "events": len(rows),
        "metrics": {
            subset: {
                "events": len(selected := [
                    row for row in rows
                    if subset == "overall" or bool(row["is_cold"]) == (subset == "cold")
                ]),
                "hit@50": float(np.mean([row["metrics"]["hit@50"] for row in selected])),
                "ndcg@10": float(np.mean([row["metrics"]["ndcg@10"] for row in selected])),
            }
            for subset in ("overall", "cold", "warm")
        },
        "mechanism_totals": mechanism_totals,
        "inference_seconds": elapsed,
        "extra_state_bytes": checkpoint.stat().st_size,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device_index) / 1024**2,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device_index) / 1024**2,
        "peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "saux_checkpoint_sha256": state["saux_checkpoint_sha256"],
        "comparator_predictions_sha256": state["comparator_predictions_sha256"],
        "validation_target_used_for_state_selection_or_tuning": False,
        "test_read": False,
        "automatic_retry": False,
    }
    atomic_json(validation_dir / "summary.json", summary)
    atomic_json(
        validation_dir / "open_file_manifest.json",
        {
            "validation_projection_opened": True,
            "validation_used_for_evaluation_only": True,
            "validation_used_for_tuning_or_state_selection": False,
            "original_user_sequence_opened": False,
            "test_opened": False,
            "test_read": False,
        },
    )
    print(COMPLETED_VERDICT, flush=True)
    return summary


def _summarize_events(
    events: Sequence[Mapping[str, Any]], arm: str, subset: str
) -> dict[str, Any]:
    selected = [
        row
        for row in events
        if subset == "overall" or bool(row["is_cold"]) == (subset == "cold")
    ]
    if not selected:
        raise ValueError(f"Empty S16-5 final subset: {subset}")
    return {
        "events": len(selected),
        "unique_target_items": len({str(row["target_item"]) for row in selected}),
        "hit@50": float(np.mean([row["metrics"][arm]["hit@50"] for row in selected])),
        "ndcg@10": float(np.mean([row["metrics"][arm]["ndcg@10"] for row in selected])),
    }


def _paired_bootstrap(
    events: Sequence[Mapping[str, Any]], treatment: str, control: str,
    metric: str, subset: str, *, resamples: int, seed: int
) -> dict[str, Any]:
    selected = [
        row for row in events
        if subset == "overall" or bool(row["is_cold"]) == (subset == "cold")
    ]
    delta = np.asarray(
        [row["metrics"][treatment][metric] - row["metrics"][control][metric] for row in selected],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 250):
        count = min(250, resamples - start)
        indices = generator.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "events": len(delta),
        "observed": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "verdict": "PASS" if low > 0 else ("NEGATIVE" if high < 0 else "INCONCLUSIVE"),
    }


def _item_cluster_bootstrap(
    events: Sequence[Mapping[str, Any]], treatment: str, control: str,
    *, resamples: int, seed: int
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in events:
        if not row["is_cold"]:
            continue
        groups.setdefault(str(row["target_item"]), []).append(
            float(row["metrics"][treatment]["hit@50"])
            - float(row["metrics"][control]["hit@50"])
        )
    sums = np.asarray([sum(values) for values in groups.values()], dtype=np.float64)
    counts = np.asarray([len(values) for values in groups.values()], dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 100):
        count = min(100, resamples - start)
        indices = generator.integers(0, len(groups), size=(count, len(groups)))
        means[start : start + count] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "events": int(counts.sum()),
        "unique_target_items": len(groups),
        "observed_event_weighted": float(sums.sum() / counts.sum()),
        "ci_low": float(low),
        "ci_high": float(high),
        "role": "repeated-target diagnostic; event-level bootstrap remains primary",
    }


def _exact_greater(events: Sequence[Mapping[str, Any]], treatment: str, control: str) -> dict[str, Any]:
    selected = [row for row in events if row["is_cold"]]
    treatment_only = sum(
        row["metrics"][treatment]["hit@50"] == 1 and row["metrics"][control]["hit@50"] == 0
        for row in selected
    )
    control_only = sum(
        row["metrics"][treatment]["hit@50"] == 0 and row["metrics"][control]["hit@50"] == 1
        for row in selected
    )
    discordant = treatment_only + control_only
    p_value = 1.0 if discordant == 0 else float(
        sum(math.comb(discordant, successes) for successes in range(treatment_only, discordant + 1))
        / (1 << discordant)
    )
    return {
        "events": len(selected),
        "treatment_only_hits": int(treatment_only),
        "control_only_hits": int(control_only),
        "discordant_pairs": int(discordant),
        "alternative": "treatment_greater_than_control",
        "raw_p_value": p_value,
    }


def _holm_single(result: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    p_value = float(result["raw_p_value"])
    return {
        **result,
        "holm_rank": 1,
        "holm_family_size": 1,
        "holm_adjusted_p_value": p_value,
        "reject_at_alpha": p_value <= alpha,
        "alpha": alpha,
    }


def finalize(config_path: Path) -> dict[str, Any]:
    config = _config(config_path)
    output = _output(config)
    summary_path = output / "summary.json"
    if summary_path.exists() or summary_path.is_symlink():
        raise FileExistsError("Refusing to overwrite S16-5 final summary")
    state = load_json(output / "state_and_comparator_freeze.json")
    validation_summary = load_json(output / "validation/summary.json")
    if state.get("verdict") != STATE_FREEZE_VERDICT or validation_summary.get("verdict") != COMPLETED_VERDICT:
        raise ValueError("S16-5 state or validation parent is not complete")
    comparator_path = ROOT / state["comparator_predictions_path"]
    comparators = _load_comparator_rows(comparator_path, config["expected_universe"]["validation_events"])
    predictions: dict[str, dict[str, Any]] = {}
    prediction_path = output / "validation/predictions_validation.jsonl"
    with prediction_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user = str(row.get("user_id"))
            if row.get("event_index") != line_number or user in predictions:
                raise ValueError(f"Invalid S16-5 S-AUX prediction row: {line_number}")
            predictions[user] = row
    if set(predictions) != set(comparators):
        raise ValueError("S16-5 S-AUX/comparator user universe drift")

    events: list[dict[str, Any]] = []
    for event_index, (user, comparator) in enumerate(comparators.items(), 1):
        treatment = predictions[user]
        if treatment["target_item"] != comparator["target_item"] or treatment["is_cold"] != comparator["is_cold"]:
            raise ValueError(f"S16-5 treatment/comparator identity mismatch: {user}")
        target = comparator["target_item"]
        events.append(
            {
                "event_index": event_index,
                "user_id": user,
                "target_item": target,
                "is_cold": comparator["is_cold"],
                "metrics": {
                    "F0": ranking_metrics(comparator["f0_top50"], target),
                    "R2": ranking_metrics(comparator["portfolio2_top50"], target),
                    "S-AUX": treatment["metrics"],
                },
            }
        )

    metrics = {
        arm: {subset: _summarize_events(events, arm, subset) for subset in ("overall", "cold", "warm")}
        for arm in ("F0", "R2", "S-AUX")
    }
    statistics = config["statistics"]
    comparisons: dict[str, Any] = {}
    seed_offset = 0
    for control in ("F0", "R2"):
        key = f"S-AUX_vs_{control}"
        comparisons[key] = {}
        for label, metric, subset in (
            ("cold_hit@50", "hit@50", "cold"),
            ("cold_ndcg@10", "ndcg@10", "cold"),
            ("warm_ndcg@10", "ndcg@10", "warm"),
            ("overall_ndcg@10", "ndcg@10", "overall"),
        ):
            comparisons[key][label] = _paired_bootstrap(
                events, "S-AUX", control, metric, subset,
                resamples=statistics["paired_bootstrap_resamples"],
                seed=statistics["paired_bootstrap_seed"] + seed_offset,
            )
            seed_offset += 1
    exact = _holm_single(
        _exact_greater(events, "S-AUX", "F0"), statistics["familywise_alpha"]
    )
    primary = comparisons["S-AUX_vs_F0"]["cold_hit@50"]
    cold_signal = primary["ci_low"] > 0 and exact["reject_at_alpha"]
    gate = (
        "PASS_S16_5_BEAUTY_SAUX_COLD_SIGNAL"
        if cold_signal
        else "FAIL_S16_5_BEAUTY_SAUX_COLD_SIGNAL_STOP"
    )
    relative_r2 = {
        "cold_hit@50_delta": metrics["S-AUX"]["cold"]["hit@50"] - metrics["R2"]["cold"]["hit@50"],
        "cold_ndcg@10_delta": metrics["S-AUX"]["cold"]["ndcg@10"] - metrics["R2"]["cold"]["ndcg@10"],
        "warm_ndcg@10_delta": metrics["S-AUX"]["warm"]["ndcg@10"] - metrics["R2"]["warm"]["ndcg@10"],
        "overall_ndcg@10_delta": metrics["S-AUX"]["overall"]["ndcg@10"] - metrics["R2"]["overall"]["ndcg@10"],
        "interpretation_scope": "frozen Pareto/trade-off description; no Beauty retuning",
    }
    event_metrics_path = output / "event_metrics.jsonl"
    with event_metrics_path.open("x", encoding="utf-8") as handle:
        for row in events:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    training_summary = load_json(ROOT / config["training"]["summary_path"])
    summary = {
        "schema_version": SCHEMA,
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "status": "COMPLETED",
        "verdict": gate,
        "events": len(events),
        "metrics": metrics,
        "paired_bootstrap": comparisons,
        "multiplicity": {
            "method": "Holm",
            "family": statistics["holm_family"],
            "family_size": 1,
            "alpha": statistics["familywise_alpha"],
            "primary_test": exact,
        },
        "item_level_bootstrap_diagnostic": _item_cluster_bootstrap(
            events, "S-AUX", "F0",
            resamples=statistics["item_bootstrap_resamples"],
            seed=statistics["item_bootstrap_seed"],
        ),
        "relative_to_unconditional_portfolio2": relative_r2,
        "costs": {
            "training_seconds": training_summary["runtime_seconds"],
            "inference_seconds": validation_summary["inference_seconds"],
            "training_peak_cuda_reserved_mib": training_summary["peak_cuda_reserved_mib"],
            "validation_peak_cuda_reserved_mib": validation_summary["peak_cuda_reserved_mib"],
            "extra_state_bytes": validation_summary["extra_state_bytes"],
            "F0_and_R2_cost_note": "Frozen Phase13 controls; historical timing is not hardware-normalized to this run."
        },
        "mechanisms": validation_summary["mechanism_totals"],
        "state_and_comparator_freeze_sha256": sha256_file(output / "state_and_comparator_freeze.json"),
        "comparator_predictions_sha256": sha256_file(comparator_path),
        "saux_predictions_sha256": sha256_file(prediction_path),
        "event_metrics_sha256": sha256_file(event_metrics_path),
        "validation_used_for_state_selection_or_tuning": False,
        "test_read": False,
        "automatic_retry": False,
        "next_gate": (
            "eligible_for_new_portfolio2_default_plus_conditional_S-AUX_plan_amendment"
            if cold_signal
            else "stop_S-AUX_and_all_Stage16_composition_development"
        ),
    }
    atomic_json(summary_path, summary)
    artifact_contract = {
        "verdict": "PASS_S16_5_BEAUTY_SAUX_ARTIFACT_CONTRACT",
        "required": [
            "state_and_comparator_freeze.json",
            "comparators/portfolio2_predictions_validation.jsonl",
            "training/summary.json",
            "training/checkpoints/best_model.pt",
            "validation/summary.json",
            "validation/predictions_validation.jsonl",
            "event_metrics.jsonl",
            "summary.json",
        ],
        "sha256": {
            relative: sha256_file(output / relative)
            for relative in (
                "state_and_comparator_freeze.json",
                "comparators/portfolio2_predictions_validation.jsonl",
                "training/summary.json",
                "training/checkpoints/best_model.pt",
                "validation/summary.json",
                "validation/predictions_validation.jsonl",
                "event_metrics.jsonl",
                "summary.json",
            )
        },
        "test_read": False,
        "automatic_retry": False,
    }
    atomic_json(output / "artifact_contract.json", artifact_contract)
    print(gate, flush=True)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-config", "freeze-comparators", "validate", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    if args.command == "check-config":
        _config(config_path)
        print("PASS_S16_5_BEAUTY_SAUX_CONFIG_CONTRACT")
        return 0
    if args.command == "freeze-comparators":
        freeze_comparators(config_path)
    elif args.command == "validate":
        validate(config_path)
    else:
        finalize(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
