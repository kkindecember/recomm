#!/usr/bin/env python3
"""Formal full-universe Stage16 G-RIDGE contract/admission on frozen Toys data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import transformers

from experiment.phase16.protocol.genrecedit_data import (
    read_lexical_paths,
    read_train_sequences,
    resolve_stage16_toys_inputs,
)
from experiment.phase16.protocol.genrecedit_faithful import (
    FullTargetRequest,
    OneOneGenerationDeltaContext,
    ZOptimizationConfig,
    aggregate_updates,
    assert_base_parameter_parity,
    build_full_target_requests,
    build_one_one_position_bundles,
    edited_parameter_name,
    extract_keys,
    filter_valid_z,
    form_weight_delta_request_products,
    official_position_to_layer,
    optimize_z_vectors,
    prepare_weight_delta_covariance,
    snapshot_base_parameters,
)
from experiment.phase16.protocol.genrecedit_inspired import (
    GRIDGE_METHOD_NAME,
    GRIDGE_RIDGE_RULE,
    GRIDGE_SOLVE_VARIANT,
    form_condition_targeted_ridge_system,
    solve_condition_targeted_ridge_system,
    validate_gridge_method_config,
)
from experiment.phase16.protocol.gfull_objective_resource_sweep import (
    BatchedRealGRAMZRuntime,
    ProgressReporter,
    RealGRAMZRuntime,
    covariance_convergence_diagnostics,
    covariance_resource_probe,
    decoder_ids,
    encode_catalog_paths,
    gpu_readmission,
    key_forward_factory,
    load_frozen_tokenizer,
    read_contexts,
    request_rows,
    select_covariance_transitions,
    sha256,
    utc_now,
    verify_s1_resolved_inputs,
    write_json,
)
from experiment.phase16.protocol.resource_probe import load_gram
from experiment.phase16.protocol.specgr_contract_smoke import (
    read_metadata,
    tokenize_passage_batch,
)


ROOT = Path(__file__).resolve().parents[3]

FORMAL_CODE_PATHS = (
    "experiment/phase16/protocol/genrecedit_faithful.py",
    "experiment/phase16/protocol/genrecedit_inspired.py",
    "experiment/phase16/protocol/genrecedit_data.py",
    "experiment/phase16/protocol/gfull_objective_resource_sweep.py",
    "experiment/phase16/protocol/gridge_formal_admission.py",
    "experiment/phase16/protocol/finalize_s3r_gridge_formal.py",
    "experiment/phase16/protocol/gridge_stability_queue.py",
    "experiment/phase16/protocol/gridge_repeat_queue.py",
    "experiment/phase16/protocol/prepare_s3r_gridge_f2_runtime.py",
    "experiment/phase16/protocol/prepare_s3r_gridge_f3_runtime.py",
    "experiment/phase16/protocol/resource_probe.py",
    "experiment/phase16/protocol/specgr_contract_smoke.py",
    "experiment/phase16/tests/test_genrecedit_faithful.py",
    "experiment/phase16/tests/test_genrecedit_inspired.py",
    "experiment/phase16/tests/test_gridge_formal_admission.py",
    "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f1.sh",
    "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f1_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_stability_gpu5.sh",
    "experiment/phase16/run_stage16_s3r_gridge_stability_gpu5_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f2.sh",
    "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f2_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f2.sh",
    "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f2_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f3.sh",
    "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f3_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f3.sh",
    "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f3_inner.sh",
    "experiment/phase15/protocol/genrecedit_gram_adapter.py",
    "GRAM/src/model/__init__.py",
    "GRAM/src/model/gram.py",
    "GRAM/src/model/gram_t5.py",
    "GRAM/src/model/gram_t5_config.py",
    "GRAM/src/model/gram_t5_modeling.py",
    "GRAM/src/model/gram_t5_outputs.py"
)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_set(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate frozen IDs: {path.relative_to(ROOT)}")
    return set(values)


def execution_identity_payload(config_path: Path, config_sha256: str) -> dict[str, Any]:
    resolved = config_path.resolve()
    if sha256(resolved) != config_sha256:
        raise ValueError("Formal config changed during execution-identity capture")
    missing = [path for path in FORMAL_CODE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise ValueError(f"Formal execution code paths are missing: {missing}")
    return {
        "captured_at_utc": utc_now(),
        "config_path": str(resolved.relative_to(ROOT)),
        "config_sha256": config_sha256,
        "code_sha256": {path: sha256(ROOT / path) for path in FORMAL_CODE_PATHS},
    }


def capture_identity(config_path: Path, config_sha256: str, output: Path) -> None:
    identity = execution_identity_payload(config_path, config_sha256)
    write_json(output / "execution_identity.json", identity)


def verify_identity(
    config_path: Path, config_sha256: str, output: Path
) -> tuple[dict[str, Any], str]:
    path = output / "execution_identity.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Formal preflight execution identity is missing")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    current = execution_identity_payload(config_path, config_sha256)
    for key in ("config_path", "config_sha256", "code_sha256"):
        if frozen.get(key) != current[key]:
            raise ValueError(f"Formal execution identity drift: {key}")
    return frozen, sha256(path)


def verify_regular_sha(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or path.is_symlink() or sha256(path) != expected:
        raise ValueError(f"Missing or SHA-drifted formal input: {label}")


def verify_inputs_before_state(config: Mapping[str, Any]) -> list[str]:
    opened: list[str] = []
    deferred = "pseudo_cold_admission_events"
    for label, spec in config["inputs"].items():
        if label in {deferred, "official_genrecedit"}:
            continue
        if "sha256" not in spec:
            continue
        verify_regular_sha(ROOT / spec["path"], spec["sha256"], label=label)
        opened.append(spec["path"])
    deferred_path = ROOT / config["inputs"][deferred]["path"]
    if not deferred_path.is_file() or deferred_path.is_symlink():
        raise ValueError("Deferred admission input is not a regular file")
    source = ROOT / config["inputs"]["official_genrecedit"]["path"]
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if head != config["inputs"]["official_genrecedit"]["commit"] or dirty:
        raise ValueError("Pinned GenRecEdit source identity drift")
    return sorted(opened)


def verify_resource_parent(config: Mapping[str, Any]) -> dict[str, Any]:
    summary = json.loads(
        (ROOT / config["inputs"]["resource_r2_summary"]["path"]).read_text(encoding="utf-8")
    )
    status = json.loads(
        (ROOT / config["inputs"]["resource_r2_status"]["path"]).read_text(encoding="utf-8")
    )
    raw = json.loads(
        (ROOT / config["inputs"]["resource_r2_raw"]["path"]).read_text(encoding="utf-8")
    )
    parent = config["resource_parent"]
    if (
        summary.get("attempt_id") != parent["attempt_id"]
        or summary.get("verdict") != parent["required_verdict"]
        or status.get("status") != parent["required_status"]
        or status.get("status_code") != parent["required_verdict"]
        or raw.get("verdict") != "PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP_RAW"
        or summary.get("formal_gate") != "PENDING_PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION"
        or int(summary.get("selected_request_microbatch", -1))
        != int(parent["selected_request_microbatch"])
        or summary.get("test_read") is not False
        or summary.get("validation_used") is not False
    ):
        raise ValueError("Formal G-RIDGE resource-parent Gate is not admissible")
    return {
        "attempt_id": parent["attempt_id"],
        "verdict": summary["verdict"],
        "raw_verdict": raw["verdict"],
        "status": status["status"],
        "selected_request_microbatch": summary["selected_request_microbatch"],
        "formal_projection": summary["formal_projection"],
    }


def verify_request_dataset(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any], list[str]]:
    manifest_path = ROOT / config["inputs"]["resource_request_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    opened = [str(manifest_path.relative_to(ROOT))]
    for shard in manifest["shards"]:
        for label in ("pseudo_contexts", "position_requests"):
            spec = shard[label]
            path = root / spec["path"]
            verify_regular_sha(path, spec["sha256"], label=f"request_{label}")
            opened.append(str(path.relative_to(ROOT)))
    subset = manifest["covariance"]["resource_subset"]
    subset_path = root / subset["path"]
    verify_regular_sha(subset_path, subset["sha256"], label="request_covariance_subset")
    opened.append(str(subset_path.relative_to(ROOT)))
    checkpoint_path = root / manifest["resume_contract"]["checkpoint_manifest"]
    verify_regular_sha(
        checkpoint_path,
        manifest["resume_contract"]["checkpoint_sha256"],
        label="request_checkpoint",
    )
    if sha256(checkpoint_path) != config["inputs"]["resource_request_checkpoint"]["sha256"]:
        raise ValueError("Resource request checkpoint disagrees with formal config")
    opened.append(str(checkpoint_path.relative_to(ROOT)))
    expected = {
        "targets": int(config["frozen_workload"]["edit_targets"]),
        "contexts": int(config["frozen_workload"]["contexts"]),
        "requests": int(config["frozen_workload"]["prefix_next_token_requests"]),
    }
    if manifest.get("counts") != expected or manifest["leakage_audit"].get(
        "held_ground_truth_files_opened"
    ) != 0:
        raise ValueError("Resource request dataset full-universe/leakage contract drift")
    return root, manifest, sorted(opened)


def throttled_progress(
    reporter: ProgressReporter, stage: str, *, stride: int = 256
):
    last_bucket = -1

    def callback(current: int, total: int) -> None:
        nonlocal last_bucket
        bucket = int(current) // int(stride)
        if current == total or bucket != last_bucket:
            last_bucket = bucket
            reporter.set(stage, current, total, "requests")

    return callback


def final_z_diagnostics(
    *,
    model,
    requests: Sequence[FullTargetRequest],
    result,
    metadata: Mapping[str, str],
    lexical_paths: Mapping[str, Sequence[str]],
    tokenizer,
    layer: int,
    device: torch.device,
    batch_size: int,
    reporter: ProgressReporter,
    position: int,
) -> dict[str, Any]:
    diagnostic_deltas = [
        result.delta_vectors[index]
        if result.delta_vectors[index] is not None
        else result.terminal_delta_vectors[index]
        for index in range(len(requests))
    ]
    if any(value is None for value in diagnostic_deltas):
        raise RuntimeError("Formal final-z diagnostic lost a request delta")
    probabilities: list[float] = []
    legal_ranks: list[int] = []
    vocabulary_ranks: list[int] = []
    stage = f"final_z_diagnostics_position_{position}"
    progress = throttled_progress(reporter, stage)
    started = time.perf_counter()
    for start in range(0, len(requests), batch_size):
        batch = tuple(requests[start : start + batch_size])
        deltas = torch.stack(
            [value for value in diagnostic_deltas[start : start + batch_size] if value is not None]
        ).to(device)
        runtime = RealGRAMZRuntime(
            model=model,
            requests=batch,
            metadata=metadata,
            lexical_paths=lexical_paths,
            tokenizer=tokenizer,
            layer=layer,
            device=device,
        )
        observation = runtime(batch, deltas, tuple(range(len(batch))))
        logits_batch = observation.logits.detach().cpu().float()
        del runtime, observation, deltas
        targets = torch.tensor([row.target_token_id for row in batch], dtype=torch.long)
        target_values = logits_batch.gather(1, targets[:, None]).squeeze(1)
        batch_probabilities = torch.softmax(logits_batch, dim=-1).gather(
            1, targets[:, None]
        ).squeeze(1)
        vocabulary_order = torch.arange(logits_batch.shape[1])
        batch_vocabulary_ranks = (
            (logits_batch > target_values[:, None]).sum(dim=1)
            + (
                (logits_batch == target_values[:, None])
                & (vocabulary_order[None, :] < targets[:, None])
            ).sum(dim=1)
            + 1
        )
        probabilities.extend(float(value) for value in batch_probabilities)
        vocabulary_ranks.extend(int(value) for value in batch_vocabulary_ranks)
        for row, logits, target_value in zip(batch, logits_batch, target_values):
            legal = logits[torch.tensor(row.legal_token_ids)]
            legal_ranks.append(1 + int((legal > target_value).sum().item()))
        progress(min(start + batch_size, len(requests)), len(requests))
    return {
        "full_vocabulary_target_probabilities": probabilities,
        "legal_target_ranks": legal_ranks,
        "full_vocabulary_target_ranks": vocabulary_ranks,
        "elapsed_seconds": time.perf_counter() - started,
    }


def solve_position(
    *,
    model,
    position: int,
    layer: int,
    chosen: Sequence[FullTargetRequest],
    result,
    covariance: torch.Tensor,
    config: Mapping[str, Any],
    metadata: Mapping[str, str],
    lexical_paths: Mapping[str, Sequence[str]],
    tokenizer,
    device: torch.device,
    reporter: ProgressReporter,
) -> tuple[torch.Tensor, dict[str, Any]]:
    workload = config["frozen_workload"]
    valid = filter_valid_z(result.z_vectors, result.delta_vectors)
    if valid.valid_count == 0:
        raise RuntimeError(f"FORMAL_BLOCKED_NO_VALID_Z_POSITION_{position}")
    module = model.decoder.block[layer].layer[2].DenseReluDense.wo
    key_started = time.perf_counter()
    keys_all = extract_keys(
        module=module,
        requests=chosen,
        forward_batch=key_forward_factory(
            model=model,
            metadata=metadata,
            lexical_paths=lexical_paths,
            tokenizer=tokenizer,
            device=device,
        ),
        batch_size=int(workload["key_extraction_batch_size"]),
        progress_callback=throttled_progress(
            reporter, f"key_extraction_position_{position}"
        ),
    )
    key_seconds = time.perf_counter() - key_started
    reporter.set(f"linear_system_position_{position}", 0, 1, "systems")
    system_started = time.perf_counter()
    valid_keys = keys_all[list(valid.valid_indices)].to(device)
    residuals = torch.stack(valid.delta_vectors).to(device)
    key_gram, rhs = form_weight_delta_request_products(
        residuals=residuals.double(), keys=valid_keys.double()
    )
    covariance64 = covariance.to(device).double()
    scaled_covariance = prepare_weight_delta_covariance(
        covariance=covariance64,
        key_width=int(workload["linear_system_width"]),
        covariance_lambda=float(workload["cov_lambda"]),
    )
    unregularized = key_gram + scaled_covariance
    unregularized_eigenvalues = torch.linalg.eigvalsh(unregularized)
    system, ridge = form_condition_targeted_ridge_system(
        system=unregularized,
        eigenvalues=unregularized_eigenvalues,
        target_condition=float(config["method"]["target_condition_number"]),
        safety_margin=float(config["method"]["ridge_safety_margin"]),
    )
    regularized_eigenvalues = unregularized_eigenvalues + ridge.ridge_value
    _, regularized_cholesky_info = torch.linalg.cholesky_ex(system)
    _, covariance_cholesky_info = torch.linalg.cholesky_ex(covariance64)
    key_eigenvalues = torch.linalg.eigvalsh(key_gram)
    key_tolerance = (
        key_gram.shape[0]
        * torch.finfo(key_gram.dtype).eps
        * key_eigenvalues.abs().max()
    )
    system_tolerance = (
        system.shape[0]
        * torch.finfo(system.dtype).eps
        * regularized_eigenvalues.abs().max()
    )
    delta = solve_condition_targeted_ridge_system(
        system=system, rhs=rhs, output_like=residuals
    )
    relative_residual = float(
        torch.linalg.vector_norm(delta @ system - rhs)
        / torch.linalg.vector_norm(rhs).clamp_min(1e-30)
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - system_started
    reporter.set(f"linear_system_position_{position}", 1, 1, "systems")
    diagnostics = {
        "request_count": len(chosen),
        "cache_hit_count": 0,
        "valid_z_count": valid.valid_count,
        "failed_z_count": valid.failed_count,
        "key_extraction_seconds": key_seconds,
        "system_and_solve_seconds": elapsed,
        "key_extraction_batch_size": int(workload["key_extraction_batch_size"]),
        "key_extraction_layer": layer,
        "valid_key_rank": int((key_eigenvalues.abs() > key_tolerance).sum().item()),
        "valid_key_rank_tolerance": float(key_tolerance),
        "covariance_cholesky_info": int(covariance_cholesky_info.item()),
        "system_rank": int((regularized_eigenvalues.abs() > system_tolerance).sum().item()),
        "system_rank_tolerance": float(system_tolerance),
        "system_min_abs_eigenvalue": float(regularized_eigenvalues.abs().min()),
        "system_max_abs_eigenvalue": float(regularized_eigenvalues.abs().max()),
        "system_condition": float(
            regularized_eigenvalues.abs().max()
            / regularized_eigenvalues.abs().clamp_min(torch.finfo(system.dtype).tiny).min()
        ),
        "rank_tolerance_rule": workload["rank_tolerance_rule"],
        "method_name": GRIDGE_METHOD_NAME,
        "method_family": "GenRecEdit-inspired",
        "faithful_reproduction": False,
        "solve_variant": GRIDGE_SOLVE_VARIANT,
        "ridge_added": True,
        "pseudoinverse_used": False,
        "jitter_fallback_used": False,
        "outcome_resampling_used": False,
        **ridge.as_dict(),
        "regularized_system_cholesky_info": int(regularized_cholesky_info.item()),
        "solve_completed": True,
        "solve_relative_residual": relative_residual,
        "delta_norm": float(torch.linalg.vector_norm(delta)),
        "delta_rank": int(torch.linalg.matrix_rank(delta).item()),
    }
    del keys_all, valid_keys, residuals, key_gram, rhs, covariance64
    del scaled_covariance, unregularized, unregularized_eigenvalues
    del system, regularized_eigenvalues, key_eigenvalues
    return delta.detach().cpu(), diagnostics


class StrictBeamEvaluator:
    def __init__(
        self,
        *,
        model,
        catalog_ids: Mapping[str, Sequence[int]],
        metadata: Mapping[str, str],
        lexical_paths: Mapping[str, Sequence[str]],
        tokenizer,
        device: torch.device,
        beam_size: int,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.lexical_paths = lexical_paths
        self.tokenizer = tokenizer
        self.device = device
        self.beam_size = int(beam_size)
        self.complete_paths = {
            tuple(map(int, path)): item for item, path in catalog_ids.items()
        }
        if len(self.complete_paths) != len(catalog_ids):
            raise ValueError("Formal admission catalog paths collide")
        self.eos = int(tokenizer.eos_token_id)
        self.pad = int(tokenizer.pad_token_id)
        children: dict[tuple[int, ...], set[int]] = {}
        for path in self.complete_paths:
            for depth, token in enumerate((*path, self.eos)):
                children.setdefault(path[:depth], set()).add(int(token))
        self.children = {key: sorted(value) for key, value in children.items()}

    def run(self, row: Mapping[str, Any]) -> tuple[list[str], bool]:
        def allowed(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
            prefix = tuple(int(value) for value in input_ids.detach().cpu().tolist()[1:])
            return self.children.get(prefix, [])

        context, _ = tokenize_passage_batch(
            [{"history": list(row["history"]), "target": row["target_item"]}],
            self.metadata,
            self.lexical_paths,
            self.tokenizer,
            self.device,
        )
        generated = self.model.generate(
            input_ids=context["input_ids"],
            attention_mask=context["attention_mask"],
            max_length=max(map(len, self.complete_paths)) + 2,
            num_beams=self.beam_size,
            num_return_sequences=self.beam_size,
            prefix_allowed_tokens_fn=allowed,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
        )
        ranking: list[str] = []
        for raw in generated.sequences.detach().cpu().tolist():
            suffix: list[int] = []
            for token in raw[1:]:
                if token == self.eos:
                    break
                if token != self.pad:
                    suffix.append(int(token))
            path = tuple(suffix)
            if path not in self.complete_paths:
                raise RuntimeError("Formal strict beam produced a non-catalog path")
            ranking.append(self.complete_paths[path])
        finite = bool(torch.isfinite(generated.sequences_scores).all().item())
        if len(ranking) != self.beam_size or len(set(ranking)) != self.beam_size:
            raise RuntimeError("Formal strict beam lost exact unique ranking contract")
        return ranking, finite


def rank_metrics(ranking: Sequence[str], target: str) -> tuple[int | None, int, float]:
    rank = ranking.index(target) + 1 if target in ranking else None
    return rank, int(rank is not None), 0.0 if rank is None else 1.0 / rank


def select_warm_events(
    train_rows: Sequence[tuple[str, Sequence[str]]], *, seed: int, count: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for user, items in train_rows:
        for source_position in range(1, len(items)):
            target = str(items[source_position])
            rows.append(
                {
                    "user_id": str(user),
                    "source_position": source_position,
                    "history": list(items[max(0, source_position - 20) : source_position]),
                    "target_item": target,
                    "rank_key": hashlib.sha256(
                        f"{seed}|g-ridge-warm|{user}|{source_position}|{target}".encode()
                    ).hexdigest(),
                }
            )
    rows.sort(key=lambda row: (row["rank_key"], row["user_id"], row["source_position"]))
    selected = rows[:count]
    if len(selected) != count:
        raise RuntimeError("Insufficient train-only warm-preservation events")
    return selected


def evaluate_formal_admission(
    *,
    model,
    evaluator: StrictBeamEvaluator,
    bundles: Mapping[int, Mapping[str, torch.Tensor]],
    position_to_layer: Mapping[int, int],
    catalog_ids: Mapping[str, Sequence[int]],
    pseudo_events: Sequence[Mapping[str, Any]],
    warm_events: Sequence[Mapping[str, Any]],
    reporter: ProgressReporter,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pseudo_digest = hashlib.sha256()
    pseudo_hits = 0
    pseudo_mrr = 0.0
    pseudo_finite = True
    reporter.set("item_disjoint_admission", 0, len(pseudo_events), "events")
    with torch.inference_mode(), OneOneGenerationDeltaContext(
        model=model,
        deltas_by_position=bundles,
        position_to_layer=position_to_layer,
        encoded_catalog_paths=catalog_ids.values(),
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        eos_token_id=evaluator.eos,
        pad_token_id=evaluator.pad,
    ) as pseudo_trace:
        for index, event in enumerate(pseudo_events, 1):
            ranking, finite = evaluator.run(event)
            _, hit, reciprocal = rank_metrics(ranking, str(event["target_item"]))
            pseudo_hits += hit
            pseudo_mrr += reciprocal
            pseudo_finite = pseudo_finite and finite
            pseudo_digest.update(
                json.dumps(
                    [str(event["event_id"]), ranking],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            )
            if index % 8 == 0 or index == len(pseudo_events):
                reporter.set("item_disjoint_admission", index, len(pseudo_events), "events")
        pseudo_applied = dict(pseudo_trace.applied_rows_by_position)
        pseudo_dead = int(pseudo_trace.dead_prefix_rows)

    reporter.set("warm_preservation_base", 0, len(warm_events), "events")
    base_rankings: list[list[str]] = []
    base_finite = True
    with torch.inference_mode():
        for index, event in enumerate(warm_events, 1):
            ranking, finite = evaluator.run(event)
            base_rankings.append(ranking)
            base_finite = base_finite and finite
            if index % 8 == 0 or index == len(warm_events):
                reporter.set("warm_preservation_base", index, len(warm_events), "events")

    warm_digest = hashlib.sha256()
    edited_finite = True
    exact = 0
    overlap = 0.0
    base_hits = edited_hits = 0
    base_mrr = edited_mrr = 0.0
    reporter.set("warm_preservation_edited", 0, len(warm_events), "events")
    with torch.inference_mode(), OneOneGenerationDeltaContext(
        model=model,
        deltas_by_position=bundles,
        position_to_layer=position_to_layer,
        encoded_catalog_paths=catalog_ids.values(),
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        eos_token_id=evaluator.eos,
        pad_token_id=evaluator.pad,
    ) as warm_trace:
        for index, (event, base) in enumerate(zip(warm_events, base_rankings), 1):
            edited, finite = evaluator.run(event)
            edited_finite = edited_finite and finite
            exact += int(base == edited)
            overlap += len(set(base) & set(edited)) / len(base)
            _, base_hit, base_reciprocal = rank_metrics(base, str(event["target_item"]))
            _, edit_hit, edit_reciprocal = rank_metrics(edited, str(event["target_item"]))
            base_hits += base_hit
            edited_hits += edit_hit
            base_mrr += base_reciprocal
            edited_mrr += edit_reciprocal
            warm_digest.update(
                json.dumps(
                    [event["user_id"], event["source_position"], base, edited],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            )
            if index % 8 == 0 or index == len(warm_events):
                reporter.set("warm_preservation_edited", index, len(warm_events), "events")
        warm_applied = dict(warm_trace.applied_rows_by_position)
        warm_dead = int(warm_trace.dead_prefix_rows)

    pseudo = {
        "events": len(pseudo_events),
        "beam_size": evaluator.beam_size,
        "all_finite": pseudo_finite,
        "all_rankings_unique_known_topk": True,
        "hit_at_50_non_promotional": pseudo_hits / len(pseudo_events),
        "mrr_non_promotional": pseudo_mrr / len(pseudo_events),
        "prediction_digest_sha256": pseudo_digest.hexdigest(),
        "applied_rows_by_position": {str(k): v for k, v in pseudo_applied.items()},
        "dead_prefix_rows": pseudo_dead,
    }
    warm = {
        "events": len(warm_events),
        "beam_size": evaluator.beam_size,
        "base_all_finite": base_finite,
        "edited_all_finite": edited_finite,
        "exact_top50_fraction": exact / len(warm_events),
        "mean_top50_set_overlap": overlap / len(warm_events),
        "base_hit_at_50_non_promotional": base_hits / len(warm_events),
        "edited_hit_at_50_non_promotional": edited_hits / len(warm_events),
        "base_mrr_non_promotional": base_mrr / len(warm_events),
        "edited_mrr_non_promotional": edited_mrr / len(warm_events),
        "prediction_pair_digest_sha256": warm_digest.hexdigest(),
        "edited_applied_rows_by_position": {str(k): v for k, v in warm_applied.items()},
        "edited_dead_prefix_rows": warm_dead,
    }
    trigger = {
        "pseudo_applied_rows_by_position": pseudo["applied_rows_by_position"],
        "warm_applied_rows_by_position": warm["edited_applied_rows_by_position"],
        "all_positions_exercised": all(pseudo_applied.get(position, 0) > 0 for position in range(6)),
    }
    return pseudo, warm, trigger


def solve_contract_pass(row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    workload = config["frozen_workload"]
    method = config["method"]
    return (
        row.get("valid_z_count", 0) > 0
        and row.get("solve_completed") is True
        and row.get("method_name") == GRIDGE_METHOD_NAME
        and row.get("solve_variant") == GRIDGE_SOLVE_VARIANT
        and row.get("faithful_reproduction") is False
        and row.get("ridge_rule") == GRIDGE_RIDGE_RULE
        and row.get("target_condition") == method["target_condition_number"]
        and row.get("safety_margin") == method["ridge_safety_margin"]
        and row.get("regularized_rank") == workload["linear_system_width"]
        and row.get("regularized_nullity") == 0
        and row.get("system_rank") == workload["linear_system_width"]
        and row.get("regularized_system_cholesky_info") == 0
        and math.isfinite(float(row.get("regularized_condition", math.inf)))
        and float(row["regularized_condition"])
        <= float(method["target_condition_number"]) * (1.0 + 1e-9)
        and row.get("solve_relative_residual", math.inf)
        <= workload["maximum_solve_relative_residual"]
        and row.get("pseudoinverse_used") is False
        and row.get("jitter_fallback_used") is False
        and row.get("outcome_resampling_used") is False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capture-identity-only", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--admission-free-mib", type=int)
    parser.add_argument("--expected-peak-mib", type=int)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    validate_gridge_method_config(config)
    run_role = config.get("run_role")
    if run_role not in {"authoritative_formal", "stability_repeat"}:
        raise ValueError("G-RIDGE execution requires an explicit formal/stability run role")
    output = ROOT / config["output_dir"]
    raw_path = output / "formal_admission_summary.json"
    if args.capture_identity_only:
        if output.exists():
            raise SystemExit("Refusing existing formal G-RIDGE attempt root")
        output.mkdir(parents=True)
        capture_identity(args.config, config_sha, output)
        print("PASS_S16_3R_FORMAL_EXECUTION_IDENTITY_CAPTURE")
        return 0
    if None in (args.physical_gpu, args.admission_free_mib, args.expected_peak_mib):
        raise SystemExit("Formal GPU workload arguments are required")
    if raw_path.exists():
        raise SystemExit("Refusing to overwrite formal G-RIDGE raw admission")
    if int(args.physical_gpu) != int(config["resources"]["fixed_physical_gpu"]):
        raise SystemExit("Formal G-RIDGE fixed physical GPU mismatch")
    identity, identity_sha = verify_identity(args.config, config_sha, output)
    reporter = ProgressReporter(output / "progress.json")
    reporter.start()
    started = time.perf_counter()
    checkpoint = ROOT / config["inputs"]["gram_checkpoint"]["path"]
    checkpoint_before = sha256(checkpoint)
    checkpoint_dir = output / "checkpoints"
    checkpoint_manifest: dict[str, Any] = {
        "attempt_id": config["attempt_id"],
        "config_sha256": config_sha,
        "execution_identity_sha256": identity_sha,
        "automatic_resume": False,
        "manual_resume_requires_explicit_user_confirmation": True,
        "artifacts": {},
    }
    held_ground_truth_opened_after_state = False
    try:
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        opened = verify_inputs_before_state(config)
        resource_parent = verify_resource_parent(config)
        request_root, data_manifest, request_opened = verify_request_dataset(config)
        opened.extend(request_opened)
        inputs, counts, max_history = resolve_stage16_toys_inputs(
            ROOT / config["inputs"]["s1_preflight_config"]["path"]
        )
        s1_contract = verify_s1_resolved_inputs(config, inputs, counts, max_history)
        free_at_worker = gpu_readmission(
            int(args.physical_gpu), int(config["resources"]["minimum_free_mib"])
        )
        device = torch.device("cuda:0")
        tokenizer, tokenizer_provenance = load_frozen_tokenizer(config)
        lexical_paths = read_lexical_paths(
            ROOT / config["inputs"]["lexical_paths"]["path"]
        )
        metadata = read_metadata(ROOT / config["inputs"]["metadata"]["path"])
        catalog_ids = encode_catalog_paths(tokenizer, lexical_paths)
        cold_items = read_set(ROOT / config["inputs"]["cold_items"]["path"])
        retained_warm = read_set(ROOT / config["inputs"]["retained_warm_items"]["path"])
        pseudo_cold = read_set(ROOT / config["inputs"]["pseudo_cold_items"]["path"])
        contexts = read_contexts(request_root, data_manifest)
        reporter.set("full_request_materialization", 0, 1, "request_universes")
        all_requests = build_full_target_requests(
            catalog_paths=catalog_ids,
            cold_paths={item: catalog_ids[item] for item in cold_items},
            pseudo_contexts=contexts,
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
        )
        requests_by_position = {
            position: [row for row in all_requests if row.position == position]
            for position in range(6)
        }
        expected_request_counts = {
            int(key): int(value)
            for key, value in config["frozen_workload"]["request_counts_by_position"].items()
        }
        if {key: len(value) for key, value in requests_by_position.items()} != expected_request_counts:
            raise RuntimeError("Formal full-target request counts drifted")
        reporter.set("full_request_materialization", 1, 1, "request_universes")

        model = load_gram(
            ROOT / config["inputs"]["gram_config"]["path"], checkpoint, device
        ).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if int(model.config.d_ff) != int(config["frozen_workload"]["linear_system_width"]):
            raise ValueError("Formal GRAM linear-system width drift")
        train_rows = read_train_sequences(ROOT / config["inputs"]["train_sequences"]["path"])
        covariance_counts = {
            int(key): int(value)
            for key, value in config["frozen_workload"]["covariance_rows_by_position"].items()
        }
        covariance_rows = select_covariance_transitions(
            train_rows, lexical_paths, rows_by_position=covariance_counts, seed=int(config["seed"])
        )

        def covariance_progress(position: int, _elapsed: Mapping[int, float]) -> None:
            reporter.set("full_covariance_positions", position + 1, 6, "lexical_positions")

        reporter.set("full_covariance_positions", 0, 6, "lexical_positions")
        covariance, covariance_seconds, activations, covariance_seconds_by_position = covariance_resource_probe(
            model=model,
            rows_by_position=covariance_rows,
            metadata=metadata,
            lexical_paths=lexical_paths,
            tokenizer=tokenizer,
            device=device,
            batch_size=int(config["frozen_workload"]["covariance_batch_size"]),
            progress_callback=covariance_progress,
        )
        checkpoints = {
            position: tuple(
                covariance_counts[position] if value == "full" else min(int(value), covariance_counts[position])
                for value in config["frozen_workload"]["formal_covariance_convergence_checkpoints"]
                if value == "full" or int(value) <= covariance_counts[position]
            )
            for position in range(6)
        }
        convergence_started = time.perf_counter()
        convergence = covariance_convergence_diagnostics(activations, checkpoints)
        convergence_seconds = time.perf_counter() - convergence_started
        covariance_checkpoint = checkpoint_dir / "full_covariance.pt"
        atomic_torch(
            covariance_checkpoint,
            {
                "config_sha256": config_sha,
                "rows_by_position": covariance.used_rows_by_position,
                "covariance_by_position": covariance.covariance_by_position,
            },
        )
        checkpoint_manifest["artifacts"]["full_covariance"] = {
            "path": str(covariance_checkpoint.relative_to(ROOT)),
            "sha256": sha256(covariance_checkpoint),
        }
        write_json(checkpoint_dir / "manifest.json", checkpoint_manifest)
        del activations
        gc.collect()

        position_metrics: dict[str, Any] = {}
        updates: dict[int, dict[str, torch.Tensor]] = {}
        position_elapsed: dict[str, float] = {}
        for position in range(6):
            position_started = time.perf_counter()
            chosen = tuple(requests_by_position[position])
            layer = official_position_to_layer([position])[position]
            runtime = BatchedRealGRAMZRuntime(
                model=model,
                metadata=metadata,
                lexical_paths=lexical_paths,
                tokenizer=tokenizer,
                layer=layer,
                device=device,
            )
            reporter.set(f"z_optimization_position_{position}", 0, len(chosen), "requests")
            result = optimize_z_vectors(
                requests=chosen,
                vector_dimension=int(model.config.d_model),
                forward_batch=runtime,
                config=ZOptimizationConfig(
                    v_lr=float(config["frozen_workload"]["z_learning_rate"]),
                    v_num_grad_steps=int(config["frozen_workload"]["z_steps"]),
                    v_weight_decay=float(config["frozen_workload"]["z_weight_decay"]),
                    z_vector_max=float(config["frozen_workload"]["z_vector_max"]),
                    batch_size=int(config["frozen_workload"]["selected_request_microbatch"]),
                ),
                cache_hits={},
                device=device,
                progress_callback=throttled_progress(
                    reporter, f"z_optimization_position_{position}"
                ),
            )
            del runtime
            z_diagnostics = final_z_diagnostics(
                model=model,
                requests=chosen,
                result=result,
                metadata=metadata,
                lexical_paths=lexical_paths,
                tokenizer=tokenizer,
                layer=layer,
                device=device,
                batch_size=int(config["frozen_workload"]["selected_request_microbatch"]),
                reporter=reporter,
                position=position,
            )
            delta, solve_diagnostics = solve_position(
                model=model,
                position=position,
                layer=layer,
                chosen=chosen,
                result=result,
                covariance=covariance.covariance_by_position[position],
                config=config,
                metadata=metadata,
                lexical_paths=lexical_paths,
                tokenizer=tokenizer,
                device=device,
                reporter=reporter,
            )
            position_metrics[str(position)] = {
                **solve_diagnostics,
                **z_diagnostics,
                "z_objective_step_seconds": sum(
                    sum(trace) for trace in result.step_elapsed_seconds_by_batch
                ),
                "lifecycle_check_steps_by_batch": [
                    list(trace) for trace in result.lifecycle_check_steps_by_batch
                ],
                "diagnostic_logit_semantics": (
                    "valid rows re-probed with satisfaction-time delta; failed rows "
                    "re-probed with terminal optimizer delta"
                ),
            }
            updates[position] = {edited_parameter_name(layer): delta}
            delta_checkpoint = checkpoint_dir / f"position_{position}_delta.pt"
            atomic_torch(
                delta_checkpoint,
                {
                    "config_sha256": config_sha,
                    "position": position,
                    "layer": layer,
                    "parameter_name": edited_parameter_name(layer),
                    "delta": delta,
                    "diagnostics": solve_diagnostics,
                },
            )
            checkpoint_manifest["artifacts"][f"position_{position}_delta"] = {
                "path": str(delta_checkpoint.relative_to(ROOT)),
                "sha256": sha256(delta_checkpoint),
            }
            position_elapsed[str(position)] = time.perf_counter() - position_started
            write_json(
                checkpoint_dir / "manifest.json",
                {
                    **checkpoint_manifest,
                    "completed_positions": position + 1,
                    "position_elapsed_seconds": position_elapsed,
                },
            )
            del result, delta
            gc.collect()
            torch.cuda.empty_cache()

        if not all(solve_contract_pass(row, config) for row in position_metrics.values()):
            raise RuntimeError("FORMAL_BLOCKED_GRIDGE_LINEAR_SYSTEM_CONTRACT")
        aggregated = aggregate_updates(updates)
        aggregate_checkpoint = checkpoint_dir / "aggregate_deltas.pt"
        atomic_torch(
            aggregate_checkpoint,
            {
                "config_sha256": config_sha,
                "position_to_layer": official_position_to_layer(range(6)),
                "aggregated_updates": aggregated,
            },
        )
        checkpoint_manifest["artifacts"]["aggregate_deltas"] = {
            "path": str(aggregate_checkpoint.relative_to(ROOT)),
            "sha256": sha256(aggregate_checkpoint),
        }
        state_frozen_at = utc_now()
        write_json(
            checkpoint_dir / "manifest.json",
            {
                **checkpoint_manifest,
                "completed_positions": 6,
                "state_frozen": True,
                "state_frozen_at_utc": state_frozen_at,
                "held_ground_truth_opened": False,
            },
        )

        base_snapshot = snapshot_base_parameters(model, sorted(aggregated))
        deferred_spec = config["inputs"]["pseudo_cold_admission_events"]
        deferred_path = ROOT / deferred_spec["path"]
        verify_regular_sha(deferred_path, deferred_spec["sha256"], label="pseudo_cold_admission_events")
        pseudo_events = read_jsonl(deferred_path)
        opened.append(deferred_spec["path"])
        held_ground_truth_opened_after_state = True
        if (
            len(pseudo_events) != int(config["admission"]["item_disjoint_events"])
            or any(str(row["target_item"]) not in pseudo_cold for row in pseudo_events)
            or any(set(map(str, row["history"])) - retained_warm for row in pseudo_events)
        ):
            raise ValueError("Deferred pseudo-cold admission universe drift/leakage")
        warm_events = select_warm_events(
            train_rows,
            seed=int(config["seed"]),
            count=int(config["admission"]["warm_preservation_events"]),
        )
        position_to_layer = official_position_to_layer(range(6))
        bundles = build_one_one_position_bundles(
            position_to_layer=position_to_layer,
            aggregated_updates={name: delta.to(device) for name, delta in aggregated.items()},
        )
        evaluator = StrictBeamEvaluator(
            model=model,
            catalog_ids=catalog_ids,
            metadata=metadata,
            lexical_paths=lexical_paths,
            tokenizer=tokenizer,
            device=device,
            beam_size=int(config["admission"]["beam_size"]),
        )
        pseudo_admission, warm_preservation, trigger_evidence = evaluate_formal_admission(
            model=model,
            evaluator=evaluator,
            bundles=bundles,
            position_to_layer=position_to_layer,
            catalog_ids=catalog_ids,
            pseudo_events=pseudo_events,
            warm_events=warm_events,
            reporter=reporter,
        )
        parity = assert_base_parameter_parity(model, base_snapshot)
        checkpoint_after = sha256(checkpoint)
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        contract_checks = {
            "resource_parent_pass": resource_parent["verdict"]
            == config["resource_parent"]["required_verdict"],
            "full_request_universe_exact": len(all_requests)
            == config["frozen_workload"]["prefix_next_token_requests"]
            and {position: len(rows) for position, rows in requests_by_position.items()}
            == expected_request_counts,
            "full_covariance_universe_exact": covariance.used_rows_by_position
            == covariance_counts,
            "covariance_convergence_complete": set(convergence)
            == {str(position) for position in range(6)}
            and all(rows[-1]["relative_frobenius_drift_to_largest_resource_checkpoint"] == 0.0 for rows in convergence.values()),
            "all_position_z_counts_complete": all(
                row["valid_z_count"] + row["failed_z_count"] == row["request_count"]
                for row in position_metrics.values()
            ),
            "all_position_ridge_solves_pass": all(
                solve_contract_pass(row, config) for row in position_metrics.values()
            ),
            "aggregate_covers_four_edited_layers": sorted(aggregated)
            == sorted({edited_parameter_name(layer) for layer in range(4)}),
            "held_ground_truth_opened_after_state_freeze": held_ground_truth_opened_after_state,
            "item_disjoint_admission_exact_and_finite": pseudo_admission["events"]
            == config["admission"]["item_disjoint_events"]
            and pseudo_admission["all_finite"]
            and pseudo_admission["all_rankings_unique_known_topk"],
            "warm_preservation_exact_and_finite": warm_preservation["events"]
            == config["admission"]["warm_preservation_events"]
            and warm_preservation["base_all_finite"]
            and warm_preservation["edited_all_finite"],
            "every_trigger_position_exercised": trigger_evidence["all_positions_exercised"],
            "base_parameter_parity": parity.get("exact") is True,
            "base_checkpoint_unchanged": checkpoint_after == checkpoint_before,
            "fixed_gpu_contract": int(args.physical_gpu)
            == int(config["resources"]["fixed_physical_gpu"])
            and int(args.admission_free_mib) >= int(config["resources"]["minimum_free_mib"])
            and int(free_at_worker) >= int(config["resources"]["minimum_free_mib"])
            and int(args.expected_peak_mib)
            == int(config["resources"]["expected_peak_reserved_mib"]),
            "peak_within_admitted_free_memory": max_reserved
            <= int(config["resources"]["minimum_free_mib"]),
            "validation_and_test_sealed": config["validation_used"] is False
            and config["test_read"] is False,
            "automatic_retry_false": config["automatic_retry"] is False,
            "config_and_code_identity_unchanged": sha256(args.config.resolve()) == config_sha
            and verify_identity(args.config, config_sha, output)[1] == identity_sha,
        }
        compute_pass = all(contract_checks.values())
        if run_role == "authoritative_formal":
            verdict = (
                "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION_RAW"
                if compute_pass
                else "FAIL_S16_3R_GRIDGE_CONTRACT_ADMISSION_RAW"
            )
            formal_gate = (
                "PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION"
                if compute_pass
                else "FAIL_S16_3R_GRIDGE_CONTRACT_ADMISSION"
            )
        else:
            verdict = (
                "STABILITY_CYCLE_COMPUTE_COMPLETE"
                if compute_pass
                else "STABILITY_CYCLE_COMPUTE_FAILED"
            )
            formal_gate = "NOT_APPLICABLE_STABILITY_REPEAT"
        raw = {
            "schema_version": config["schema_version"],
            "experiment_id": config["experiment_id"],
            "attempt_id": config["attempt_id"],
            "run_role": run_role,
            "generated_at_utc": utc_now(),
            "verdict": verdict,
            "formal_gate": formal_gate,
            "authoritative_stage_status": (
                "RUNNING_UNTIL_FORMAL_FINALIZER"
                if run_role == "authoritative_formal"
                else config.get("stability", {}).get(
                    "authoritative_stage_status", "COMPLETED"
                )
            ),
            "affects_scientific_results": run_role == "authoritative_formal",
            "promotion_eligible": run_role == "authoritative_formal",
            "method": config["method"],
            "resource_parent": resource_parent,
            "physical_gpu": int(args.physical_gpu),
            "visible_gpu": 0,
            "admission_free_mib": int(args.admission_free_mib),
            "worker_readmission_free_mib": int(free_at_worker),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_peak_allocated_mib": max_allocated,
            "maximum_peak_reserved_mib": max_reserved,
            "full_universe": {
                "edit_targets": len(cold_items),
                "contexts": data_manifest["counts"]["contexts"],
                "prefix_next_token_requests": len(all_requests),
                "request_counts_by_position": {str(k): len(v) for k, v in requests_by_position.items()},
                "covariance_rows_by_position": {str(k): v for k, v in covariance.used_rows_by_position.items()},
            },
            "covariance": {
                "elapsed_seconds": covariance_seconds,
                "elapsed_seconds_by_position": {str(k): v for k, v in covariance_seconds_by_position.items()},
                "convergence_elapsed_seconds": convergence_seconds,
                "convergence": convergence,
                "primary_estimator": "full train-only raw E[x x^T] moment; FP64 accumulation then frozen FP32 finalize",
            },
            "position_diagnostics": position_metrics,
            "position_elapsed_seconds": position_elapsed,
            "aggregated_parameters": sorted(aggregated),
            "item_disjoint_admission_non_promotional": pseudo_admission,
            "warm_preservation_non_promotional": warm_preservation,
            "trigger_evidence": trigger_evidence,
            "base_parameter_parity": parity,
            "contract_checks": contract_checks,
            "checkpoint_manifest": {
                "path": str((checkpoint_dir / "manifest.json").relative_to(ROOT)),
                "sha256": sha256(checkpoint_dir / "manifest.json"),
                "state_frozen_at_utc": state_frozen_at,
            },
            "request_dataset_artifact": {
                "path": config["inputs"]["resource_request_manifest"]["path"],
                "sha256": config["inputs"]["resource_request_manifest"]["sha256"],
                "dataset_sha256": data_manifest["dataset_sha256"],
            },
            "s1_resolved_input_contract": s1_contract,
            "execution_identity": identity,
            "execution_identity_sha256": identity_sha,
            "opened_files": sorted(set(opened)),
            "deferred_input_policy": {
                "path": deferred_spec["path"],
                "opened_after_state_frozen": held_ground_truth_opened_after_state,
                "used_for_state_selection": False,
                "used_for_ridge_selection": False,
            },
            "tokenizer_provenance": tokenizer_provenance,
            "runtime_provenance": {
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_runtime_version": torch.version.cuda,
            },
            "resource_summary": {
                "gpu_count": 1,
                "physical_gpu": int(args.physical_gpu),
                "maximum_peak_allocated_mib": max_allocated,
                "maximum_peak_reserved_mib": max_reserved,
                "cpu_ram_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "disk_reservation_mib": config["resources"]["disk_reservation_mib"],
                "hard_timeout_seconds": config["resources"]["hard_timeout_seconds"],
            },
            "base_checkpoint_sha256_before": checkpoint_before,
            "base_checkpoint_sha256_after": checkpoint_after,
            "scientific_scope": "train-only contract/admission; metrics are non-promotional",
            "scientific_efficacy_metric_produced": False,
            "validation_used": False,
            "test_read": False,
            "automatic_retry": False,
        }
        write_json(raw_path, raw)
        print(verdict)
        return 0 if compute_pass else 3
    except RuntimeError as error:
        message = str(error)
        if message.startswith("FORMAL_BLOCKED_NO_VALID_Z"):
            print("BLOCKED_S16_3R_FORMAL_VALID_Z")
            return 11
        if message == "FORMAL_BLOCKED_GRIDGE_LINEAR_SYSTEM_CONTRACT":
            print("BLOCKED_S16_3R_FORMAL_GRIDGE_LINEAR_SYSTEM")
            return 10
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    raise SystemExit(main())
