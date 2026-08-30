#!/usr/bin/env python3
"""S16-3B full-universe rank-sufficiency diagnostic for faithful G-FULL."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import transformers

from experiment.phase16.protocol.genrecedit_data import (
    read_lexical_paths,
    read_train_sequences,
    resolve_stage16_toys_inputs,
)
from experiment.phase16.protocol.genrecedit_faithful import (
    FullTargetRequest,
    build_full_target_requests,
    official_position_to_layer,
)
from experiment.phase16.protocol.genrecedit_rank_sufficiency import (
    RANK_TOLERANCE_RULE,
    StreamingKeyGram,
    classify_all_request_upper_bound,
    deterministic_request_order,
    effective_checkpoints,
    ordered_request_sha256,
    symmetric_rank_diagnostics,
)
from experiment.phase16.protocol.gfull_objective_resource_sweep import (
    ROOT,
    ProgressReporter,
    covariance_resource_probe,
    encode_catalog_paths,
    gpu_readmission,
    key_forward_factory,
    load_frozen_tokenizer,
    read_contexts,
    select_covariance_transitions,
    sha256,
    utc_now,
    verify_inputs,
    verify_s1_resolved_inputs,
    write_json,
)
from experiment.phase16.protocol.resource_probe import load_gram
from experiment.phase16.protocol.specgr_contract_smoke import read_metadata


EXECUTED_CODE_PATHS = (
    "experiment/phase16/protocol/genrecedit_faithful.py",
    "experiment/phase16/protocol/genrecedit_rank_sufficiency.py",
    "experiment/phase16/protocol/genrecedit_data.py",
    "experiment/phase16/protocol/gfull_objective_resource_sweep.py",
    "experiment/phase16/protocol/gfull_rank_sufficiency_diagnostic.py",
    "experiment/phase16/protocol/finalize_s3b_rank_sufficiency.py",
    "experiment/phase16/protocol/resource_probe.py",
    "experiment/phase16/protocol/specgr_contract_smoke.py",
    "experiment/phase16/protocol/official_specgr_runtime.py",
    "experiment/phase16/protocol/specgr_faithful.py",
    "experiment/phase16/tests/test_genrecedit_faithful.py",
    "experiment/phase16/tests/test_genrecedit_data.py",
    "experiment/phase16/tests/test_gfull_resource_contract.py",
    "experiment/phase16/tests/test_gfull_rank_sufficiency.py",
    "experiment/phase16/run_stage16_s3b_gfull_rank_sufficiency.sh",
    "experiment/phase16/run_stage16_s3b_gfull_rank_sufficiency_b1_gpu4.sh",
    "experiment/phase16/run_stage16_s3b_gfull_rank_sufficiency_b1_gpu4_inner.sh",
    "experiment/phase15/protocol/genrecedit_gram_adapter.py",
    "GRAM/src/model/__init__.py",
    "GRAM/src/model/gram.py",
    "GRAM/src/model/gram_t5.py",
    "GRAM/src/model/gram_t5_config.py",
    "GRAM/src/model/gram_t5_modeling.py",
    "GRAM/src/model/gram_t5_outputs.py",
)


def execution_identity_payload(
    config_path: Path, loaded_config_sha256: str
) -> dict[str, Any]:
    resolved = config_path.resolve()
    if sha256(resolved) != loaded_config_sha256:
        raise ValueError("S16-3B config changed between load and identity capture")
    return {
        "captured_at_utc": utc_now(),
        "config_path": str(resolved.relative_to(ROOT)),
        "config_sha256": loaded_config_sha256,
        "code_sha256": {path: sha256(ROOT / path) for path in EXECUTED_CODE_PATHS},
    }


def capture_execution_identity(
    config_path: Path, loaded_config_sha256: str, output: Path
) -> tuple[dict[str, Any], str]:
    identity = execution_identity_payload(config_path, loaded_config_sha256)
    path = output / "execution_identity.json"
    write_json(path, identity)
    return identity, sha256(path)


def verify_execution_identity(
    config_path: Path, loaded_config_sha256: str, path: Path
) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing regular S16-3B execution identity")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    current = execution_identity_payload(config_path, loaded_config_sha256)
    for key in ("config_path", "config_sha256", "code_sha256"):
        if frozen.get(key) != current[key]:
            raise ValueError(f"S16-3B execution identity drift: {key}")
    if not isinstance(frozen.get("captured_at_utc"), str):
        raise ValueError("S16-3B execution identity lacks capture time")
    return frozen, sha256(path)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def verify_parent_request_dataset(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify every reused A4 request shard without modifying the parent root."""

    parent = config["parent_a4"]
    dataset_root = ROOT / parent["request_dataset_root"]
    expected_counts = {
        "targets": int(config["frozen_workload"]["edit_targets"]),
        "contexts": int(config["frozen_workload"]["contexts"]),
        "requests": int(config["frozen_workload"]["prefix_next_token_requests"]),
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("Parent request dataset full-universe count drift")
    if manifest.get("dataset_sha256") != parent["request_dataset_sha256"]:
        raise ValueError("Parent request dataset semantic SHA drift")
    expected_covariance = {
        str(position): int(count)
        for position, count in config["diagnostic"][
            "full_covariance_rows_by_position"
        ].items()
    }
    if manifest.get("covariance", {}).get("position_counts") != expected_covariance:
        raise ValueError("Parent request dataset covariance coverage drift")
    leakage = manifest.get("leakage_audit", {})
    if (
        any(
            value != 0
            for key, value in leakage.items()
            if key.endswith("_opened") or key.endswith("_occurrences")
        )
        or leakage.get("target_selection_uses_validation_or_test_occurrence") is not False
    ):
        raise ValueError("Parent request dataset is not train-only")
    shards = manifest.get("shards", [])
    if len(shards) != int(parent["completed_shards"]):
        raise ValueError("Parent request dataset shard-count drift")
    observed_contexts = 0
    observed_requests = 0
    for shard in shards:
        observed_contexts += int(shard["context_count"])
        observed_requests += int(shard["request_count"])
        for label, expected_lines in (
            ("pseudo_contexts", int(shard["context_count"])),
            ("position_requests", int(shard["request_count"])),
        ):
            spec = shard[label]
            path = dataset_root / spec["path"]
            if not path.is_file() or path.is_symlink() or sha256(path) != spec["sha256"]:
                raise ValueError(f"Parent request dataset shard drift: {label}")
            if _line_count(path) != expected_lines:
                raise ValueError(f"Parent request dataset shard line-count drift: {label}")
    if observed_contexts != expected_counts["contexts"] or observed_requests != expected_counts["requests"]:
        raise ValueError("Parent request dataset materialized count drift")
    return {
        "dataset_root": parent["request_dataset_root"],
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": parent["request_manifest_sha256"],
        "checkpoint_sha256": parent["request_checkpoint_sha256"],
        "completed_shards": len(shards),
        "counts": expected_counts,
        "train_only": True,
        "all_shards_verified": True,
        "parent_modified": False,
    }


def parent_a4_baseline(
    config: Mapping[str, Any], raw: Mapping[str, Any], status: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        raw.get("verdict") != "RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM"
        or status.get("status_code") != "RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM"
        or status.get("process_alive") is not False
    ):
        raise ValueError("S16-3B parent is not the frozen terminal A4 rank block")
    rows: dict[str, Any] = {}
    for position in range(6):
        source = raw["position_diagnostics"][str(position)]
        if source.get("solve_completed") is not False:
            raise ValueError("S16-3B parent position is not a preserved failed solve")
        rows[str(position)] = {
            "request_count": int(source["request_count"]),
            "valid_z_count": int(source["valid_z_count"]),
            "covariance_rank": int(source["covariance_rank"]),
            "valid_key_rank": int(source["valid_key_rank"]),
            "system_rank": int(source["system_rank"]),
            "solve_completed": bool(source["solve_completed"]),
        }
    return {
        "attempt_id": raw["attempt_id"],
        "verdict": raw["verdict"],
        "status_code": status["status_code"],
        "linear_system_width": int(raw["covariance_resource"]["linear_system_width"]),
        "position_diagnostics": rows,
    }


def stream_position_key_rank_curve(
    *,
    model,
    requests: Sequence[FullTargetRequest],
    covariance: torch.Tensor,
    covariance_lambda: float,
    layer: int,
    batch_size: int,
    checkpoints: Sequence[int],
    metadata: Mapping[str, str],
    lexical_paths: Mapping[str, Sequence[str]],
    tokenizer,
    device: torch.device,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream every key into a Gram matrix and diagnose fixed prefixes."""

    width = int(model.config.d_ff)
    if covariance.shape != (width, width):
        raise ValueError("S16-3B covariance width disagrees with the GRAM key width")
    effective = tuple(int(value) for value in checkpoints)
    if not effective or effective[-1] != len(requests):
        raise ValueError("S16-3B key checkpoints must end at the full position universe")
    accumulator = StreamingKeyGram(width, device=device)
    scaled_covariance = float(covariance_lambda) * covariance.to(device).double()
    covariance_diagnostics = symmetric_rank_diagnostics(
        covariance.to(device).double()
    ).as_dict()
    forward = key_forward_factory(
        model=model,
        metadata=metadata,
        lexical_paths=lexical_paths,
        tokenizer=tokenizer,
        device=device,
    )
    module = model.decoder.block[layer].layer[2].DenseReluDense.wo
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise ValueError("S16-3B key hook did not receive a tensor")
        values = inputs[0]
        if values.ndim == 3:
            values = values[:, -1, :]
        elif values.ndim != 2:
            raise ValueError("S16-3B key hook received an invalid tensor shape")
        captured.append(values.detach())

    curve: list[dict[str, Any]] = []
    cursor = 0
    handle = module.register_forward_pre_hook(capture)
    try:
        for checkpoint in effective:
            while cursor < checkpoint:
                stop = min(cursor + int(batch_size), checkpoint)
                batch = tuple(requests[cursor:stop])
                captured.clear()
                with torch.no_grad():
                    forward(batch)
                if len(captured) != 1 or captured[0].shape[0] != len(batch):
                    raise ValueError("S16-3B key hook did not capture one aligned batch")
                accumulator.update(captured[0])
                cursor = stop
                if progress_callback is not None:
                    progress_callback(cursor)
            torch.cuda.synchronize(device)
            key_diagnostics = symmetric_rank_diagnostics(accumulator.gram).as_dict()
            system = accumulator.gram + scaled_covariance
            system_diagnostics = symmetric_rank_diagnostics(system).as_dict()
            curve.append(
                {
                    "request_count": int(checkpoint),
                    "key_gram": key_diagnostics,
                    "system": system_diagnostics,
                    "algebraic_rank_capacity_upper_bound": min(
                        width,
                        int(covariance_diagnostics["rank"])
                        + int(key_diagnostics["rank"]),
                    ),
                }
            )
            del system
    finally:
        handle.remove()
    if accumulator.count != len(requests):
        raise RuntimeError("S16-3B did not consume the full ordered key universe")
    return curve, covariance_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capture-identity-only", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--admission-free-mib", type=int)
    parser.add_argument("--admission-util-percent", type=int)
    parser.add_argument("--worker-hard-timeout-seconds", type=int)
    parser.add_argument("--expected-peak-mib", type=int)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    loaded_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    output = ROOT / config["output_dir"]
    raw_path = output / "rank_diagnostic_raw.json"
    identity_path = output / "execution_identity.json"
    if args.capture_identity_only:
        if raw_path.exists() or identity_path.exists():
            raise SystemExit("Refusing to overwrite an existing S16-3B attempt")
        output.mkdir(parents=True, exist_ok=True)
        capture_execution_identity(args.config, loaded_config_sha256, output)
        print("PASS_S16_3B_EXECUTION_IDENTITY_CAPTURE")
        return 0
    if None in (
        args.physical_gpu,
        args.admission_free_mib,
        args.admission_util_percent,
        args.worker_hard_timeout_seconds,
        args.expected_peak_mib,
    ):
        raise SystemExit("S16-3B GPU workload arguments are required")
    if int(args.physical_gpu) != int(config["resources"]["fixed_physical_gpu"]):
        raise SystemExit("S16-3B requires the frozen physical GPU")
    if raw_path.exists():
        raise SystemExit("Refusing to overwrite an existing S16-3B raw diagnostic")

    execution_identity, execution_identity_sha = verify_execution_identity(
        args.config, loaded_config_sha256, identity_path
    )
    reporter = ProgressReporter(output / "progress.json")
    reporter.start()
    started = time.perf_counter()
    checkpoint_path = output / "rank_stage_checkpoint.json"
    checkpoint = ROOT / config["inputs"]["gram_checkpoint"]["path"]
    checkpoint_before = sha256(checkpoint)
    parent_hashes_before = {
        label: sha256(ROOT / config["inputs"][label]["path"])
        for label in (
            "parent_a4_raw",
            "parent_a4_status",
            "parent_request_manifest",
            "parent_request_checkpoint",
        )
    }
    try:
        reporter.set("train_only_contract", 0, 1, "full_universe")
        opened = verify_inputs(config)
        inputs, counts, max_history = resolve_stage16_toys_inputs(
            ROOT / config["inputs"]["s1_preflight_config"]["path"]
        )
        s1_contract = verify_s1_resolved_inputs(config, inputs, counts, max_history)
        parent_manifest = json.loads(
            (ROOT / config["inputs"]["parent_request_manifest"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        parent_dataset = verify_parent_request_dataset(config, parent_manifest)
        parent_raw = json.loads(
            (ROOT / config["inputs"]["parent_a4_raw"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        parent_status = json.loads(
            (ROOT / config["inputs"]["parent_a4_status"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        parent_baseline = parent_a4_baseline(config, parent_raw, parent_status)
        if parent_baseline["linear_system_width"] != int(
            config["diagnostic"]["linear_system_width"]
        ):
            raise ValueError("S16-3B parent and diagnostic linear-system widths differ")
        reporter.set("train_only_contract", 1, 1, "full_universe")

        try:
            free_at_worker = gpu_readmission(
                int(args.physical_gpu), int(config["resources"]["minimum_free_mib"])
            )
        except RuntimeError as error:
            if str(error).startswith("GPU_READMISSION_FAILED"):
                print(str(error))
                return 9
            raise

        device = torch.device("cuda:0")
        tokenizer, tokenizer_provenance = load_frozen_tokenizer(config)
        lexical_paths = read_lexical_paths(
            ROOT / config["inputs"]["lexical_paths"]["path"]
        )
        metadata = read_metadata(ROOT / config["inputs"]["metadata"]["path"])
        catalog_ids = encode_catalog_paths(tokenizer, lexical_paths)
        cold_items = {
            line.strip()
            for line in (ROOT / config["inputs"]["cold_items"]["path"]).read_text().splitlines()
            if line.strip()
        }
        contexts = read_contexts(
            ROOT / config["parent_a4"]["request_dataset_root"], parent_manifest
        )
        full_requests = build_full_target_requests(
            catalog_paths=catalog_ids,
            cold_paths={item: catalog_ids[item] for item in cold_items},
            pseudo_contexts={item: contexts[item] for item in cold_items},
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
        )
        requests_by_position: dict[int, tuple[FullTargetRequest, ...]] = {}
        for position in range(6):
            rows = [request for request in full_requests if request.position == position]
            requests_by_position[position] = deterministic_request_order(
                rows, seed=int(config["seed"])
            )
        del full_requests
        expected_request_counts = {
            int(position): int(value)
            for position, value in config["diagnostic"][
                "full_request_counts_by_position"
            ].items()
        }
        if {
            position: len(rows) for position, rows in requests_by_position.items()
        } != expected_request_counts:
            raise ValueError("S16-3B full request position counts drift")

        model = load_gram(
            ROOT / config["inputs"]["gram_config"]["path"], checkpoint, device
        ).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        width = int(config["diagnostic"]["linear_system_width"])
        if int(model.config.d_ff) != width:
            raise ValueError("S16-3B linear-system width disagrees with GRAM")
        train_rows = read_train_sequences(
            ROOT / config["inputs"]["train_sequences"]["path"]
        )
        full_covariance_counts = {
            int(position): int(value)
            for position, value in config["diagnostic"][
                "full_covariance_rows_by_position"
            ].items()
        }
        covariance_rows = select_covariance_transitions(
            train_rows,
            lexical_paths,
            rows_by_position=full_covariance_counts,
            seed=int(config["seed"]),
        )
        del train_rows

        position_results: dict[str, Any] = {}
        cumulative_keys = 0
        total_keys = sum(expected_request_counts.values())
        position_layers = official_position_to_layer(range(6))
        for position in range(6):
            reporter.set(
                "full_covariance",
                position,
                6,
                f"positions_before_{position}",
            )
            covariance_started = time.perf_counter()
            covariance_result, _, activations, covariance_elapsed_by_position = (
                covariance_resource_probe(
                    model=model,
                    rows_by_position={position: covariance_rows[position]},
                    metadata=metadata,
                    lexical_paths=lexical_paths,
                    tokenizer=tokenizer,
                    device=device,
                    batch_size=int(config["diagnostic"]["covariance_batch_size"]),
                )
            )
            covariance = covariance_result.covariance_by_position[position]
            covariance_seconds = time.perf_counter() - covariance_started
            if covariance_result.used_rows_by_position[position] != full_covariance_counts[position]:
                raise RuntimeError("S16-3B covariance did not consume its full universe")
            del activations, covariance_result

            ordered = requests_by_position[position]
            checkpoints = effective_checkpoints(
                config["diagnostic"]["request_key_checkpoints"], total=len(ordered)
            )
            position_key_start = cumulative_keys

            def report_keys(local_count: int) -> None:
                reporter.set(
                    "all_request_key_upper_bound",
                    position_key_start + int(local_count),
                    total_keys,
                    "train_only_request_keys",
                )

            key_started = time.perf_counter()
            curve, covariance_diagnostics = stream_position_key_rank_curve(
                model=model,
                requests=ordered,
                covariance=covariance,
                covariance_lambda=float(config["frozen_workload"]["cov_lambda"]),
                layer=position_layers[position],
                batch_size=int(config["diagnostic"]["key_batch_size"]),
                checkpoints=checkpoints,
                metadata=metadata,
                lexical_paths=lexical_paths,
                tokenizer=tokenizer,
                device=device,
                progress_callback=report_keys,
            )
            key_seconds = time.perf_counter() - key_started
            cumulative_keys += len(ordered)
            final = curve[-1]
            position_results[str(position)] = {
                "position": position,
                "layer": position_layers[position],
                "covariance_rows": full_covariance_counts[position],
                "request_count": len(ordered),
                "request_order_sha256": ordered_request_sha256(ordered),
                "effective_checkpoints": list(checkpoints),
                "covariance": covariance_diagnostics,
                "rank_curve": curve,
                "final_key_rank": int(final["key_gram"]["rank"]),
                "final_system_rank": int(final["system"]["rank"]),
                "final_system_nullity": int(final["system"]["nullity"]),
                "full_covariance_universe_processed": True,
                "full_request_key_universe_processed": True,
                "all_request_key_superset": True,
                "valid_z_filter_applied": False,
                "z_optimization_run": False,
                "weight_delta_solve_run": False,
                "ridge_added": False,
                "pseudoinverse_used": False,
                "jitter_fallback_used": False,
                "outcome_resampling_used": False,
                "covariance_elapsed_seconds": covariance_seconds,
                "covariance_probe_elapsed_seconds": covariance_elapsed_by_position[
                    position
                ],
                "key_and_rank_elapsed_seconds": key_seconds,
            }
            write_json(
                checkpoint_path,
                {
                    "attempt_id": config["attempt_id"],
                    "stage": "position_complete",
                    "execution_identity_sha256": execution_identity_sha,
                    "completed_positions": position + 1,
                    "position_diagnostics": position_results,
                    "automatic_resume": False,
                },
            )
            del covariance
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

        if cumulative_keys != total_keys:
            raise RuntimeError("S16-3B full request key count is incomplete")
        classification = classify_all_request_upper_bound(
            position_results, width=width
        )
        parent_hashes_after = {
            label: sha256(ROOT / config["inputs"][label]["path"])
            for label in parent_hashes_before
        }
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        contract_checks = {
            "parent_a4_terminal_linear_system_block_preserved": parent_baseline[
                "verdict"
            ]
            == "RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM",
            "parent_artifacts_unchanged": parent_hashes_before == parent_hashes_after,
            "parent_request_dataset_verified_train_only": parent_dataset["train_only"]
            and parent_dataset["all_shards_verified"],
            "full_request_universe_exact": sum(expected_request_counts.values())
            == int(config["frozen_workload"]["prefix_next_token_requests"])
            and all(
                position_results[str(position)]["request_count"]
                == expected_request_counts[position]
                for position in range(6)
            ),
            "full_covariance_universe_exact": all(
                position_results[str(position)]["covariance_rows"]
                == full_covariance_counts[position]
                for position in range(6)
            ),
            "all_request_upper_bound_complete": all(
                position_results[str(position)][
                    "full_request_key_universe_processed"
                ]
                and position_results[str(position)]["all_request_key_superset"]
                for position in range(6)
            ),
            "no_z_solve_or_fallback": all(
                position_results[str(position)]["z_optimization_run"] is False
                and position_results[str(position)]["weight_delta_solve_run"] is False
                and position_results[str(position)]["ridge_added"] is False
                and position_results[str(position)]["pseudoinverse_used"] is False
                and position_results[str(position)]["jitter_fallback_used"] is False
                and position_results[str(position)]["outcome_resampling_used"] is False
                for position in range(6)
            ),
            "rank_rule_exact": config["diagnostic"]["rank_tolerance_rule"]
            == RANK_TOLERANCE_RULE
            and all(
                row["covariance"]["tolerance_rule"] == RANK_TOLERANCE_RULE
                and all(
                    checkpoint_row["key_gram"]["tolerance_rule"]
                    == RANK_TOLERANCE_RULE
                    and checkpoint_row["system"]["tolerance_rule"]
                    == RANK_TOLERANCE_RULE
                    for checkpoint_row in row["rank_curve"]
                )
                for row in position_results.values()
            ),
            "positive_semidefinite_evidence": all(
                row["covariance"]["significant_negative_eigenvalues"] == 0
                and all(
                    checkpoint_row["key_gram"][
                        "significant_negative_eigenvalues"
                    ]
                    == 0
                    and checkpoint_row["system"][
                        "significant_negative_eigenvalues"
                    ]
                    == 0
                    for checkpoint_row in row["rank_curve"]
                )
                for row in position_results.values()
            ),
            "base_checkpoint_unchanged": sha256(checkpoint) == checkpoint_before,
            "fixed_gpu_resource_contract_exact": int(args.physical_gpu)
            == int(config["resources"]["fixed_physical_gpu"])
            and int(args.admission_free_mib)
            >= int(config["resources"]["minimum_free_mib"])
            and int(free_at_worker) >= int(config["resources"]["minimum_free_mib"])
            and int(args.worker_hard_timeout_seconds)
            == int(config["resources"]["hard_timeout_seconds"])
            and int(args.expected_peak_mib)
            == int(config["resources"]["expected_peak_mib"]),
            "peak_within_attempt_cap": peak_reserved
            <= float(config["resources"]["expected_peak_mib"]),
            "classification_does_not_promote_faithful_gate": classification[
                "faithful_gate_promoted"
            ]
            is False,
        }
        raw = {
            "schema_version": config["schema_version"],
            "experiment_id": config["experiment_id"],
            "attempt_id": config["attempt_id"],
            "verdict": (
                "PASS_S16_3B_RANK_DIAGNOSTIC_RAW"
                if all(contract_checks.values())
                else "FAIL_S16_3B_RANK_DIAGNOSTIC_RAW"
            ),
            "generated_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "physical_gpu": int(args.physical_gpu),
            "visible_gpu": 0,
            "admission_free_mib": int(args.admission_free_mib),
            "worker_readmission_free_mib": int(free_at_worker),
            "admission_util_percent": int(args.admission_util_percent),
            "worker_hard_timeout_seconds": int(args.worker_hard_timeout_seconds),
            "expected_peak_mib": int(args.expected_peak_mib),
            "maximum_peak_allocated_mib": peak_allocated,
            "maximum_peak_reserved_mib": peak_reserved,
            "diagnostic_question": config["diagnostic"]["question"],
            "upper_bound_semantics": config["diagnostic"][
                "all_request_upper_bound_semantics"
            ],
            "classification": classification,
            "parent_a4_baseline": parent_baseline,
            "parent_hashes_before": parent_hashes_before,
            "parent_hashes_after": parent_hashes_after,
            "parent_request_dataset": parent_dataset,
            "s1_resolved_input_contract": s1_contract,
            "position_diagnostics": position_results,
            "linear_system_width": width,
            "covariance_lambda": float(config["frozen_workload"]["cov_lambda"]),
            "request_key_batch_size": int(config["diagnostic"]["key_batch_size"]),
            "covariance_batch_size": int(
                config["diagnostic"]["covariance_batch_size"]
            ),
            "contract_checks": contract_checks,
            "execution_identity": execution_identity,
            "execution_identity_artifact": {
                "path": str(identity_path.relative_to(ROOT)),
                "sha256": execution_identity_sha,
            },
            "base_checkpoint_unchanged": sha256(checkpoint) == checkpoint_before,
            "tokenizer_provenance": tokenizer_provenance,
            "runtime_provenance": {
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_runtime_version": torch.version.cuda,
            },
            "cpu_ram_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "opened_files": sorted(set(opened)),
            "declared_scope": (
                "S16-1 train-only inputs plus immutable S16-3 A4 terminal/request "
                "artifacts; no validation, internal-dev occurrence, or test files"
            ),
            "scientific_efficacy_metric_produced": False,
            "faithful_gate_promoted": False,
            "validation_used": False,
            "test_read": False,
            "automatic_retry": False,
            "automatic_resume": False,
        }
        write_json(raw_path, raw)
        print(raw["verdict"])
        return 0 if raw["verdict"].startswith("PASS") else 3
    finally:
        reporter.close()


if __name__ == "__main__":
    raise SystemExit(main())
