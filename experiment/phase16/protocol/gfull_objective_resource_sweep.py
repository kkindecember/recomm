#!/usr/bin/env python3
"""Bounded real-GRAM, official-parameter resource sweep for Stage16 G-FULL."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import transformers
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


ROOT = Path(__file__).resolve().parents[3]

EXECUTED_CODE_PATHS = (
    "experiment/phase16/protocol/genrecedit_faithful.py",
    "experiment/phase16/protocol/genrecedit_inspired.py",
    "experiment/phase16/protocol/genrecedit_data.py",
    "experiment/phase16/protocol/gfull_objective_resource_sweep.py",
    "experiment/phase16/protocol/finalize_s3_gfull_resource_sweep.py",
    "experiment/phase16/protocol/resource_probe.py",
    "experiment/phase16/protocol/specgr_contract_smoke.py",
    "experiment/phase16/protocol/official_specgr_runtime.py",
    "experiment/phase16/protocol/specgr_faithful.py",
    "experiment/phase16/tests/test_genrecedit_faithful.py",
    "experiment/phase16/tests/test_genrecedit_inspired.py",
    "experiment/phase16/tests/test_genrecedit_data.py",
    "experiment/phase16/tests/test_gfull_resource_contract.py",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep.sh",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a2.sh",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a3.sh",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a4_gpu4.sh",
    "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a4_gpu4_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu4.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu4_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu5.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu5_inner.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve.sh",
    "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve_inner.sh",
    "experiment/phase15/protocol/genrecedit_gram_adapter.py",
    "GRAM/src/model/__init__.py",
    "GRAM/src/model/gram.py",
    "GRAM/src/model/gram_t5.py",
    "GRAM/src/model/gram_t5_config.py",
    "GRAM/src/model/gram_t5_modeling.py",
    "GRAM/src/model/gram_t5_outputs.py",
)

from experiment.phase16.protocol.genrecedit_data import (  # noqa: E402
    build_sharded_dataset,
    read_lexical_paths,
    repo_relative_path,
    read_train_sequences,
    resolve_stage16_toys_inputs,
)
from experiment.phase16.protocol.genrecedit_faithful import (  # noqa: E402
    FullTargetRequest,
    ZForwardBatch,
    ZOptimizationConfig,
    aggregate_updates,
    assert_base_parameter_parity,
    build_one_one_position_bundles,
    build_full_target_requests,
    collect_covariance,
    edited_parameter_name,
    extract_keys,
    filter_valid_z,
    form_weight_delta_request_products,
    official_position_to_layer,
    OneOneGenerationDeltaContext,
    PositionCovarianceResult,
    optimize_z_vectors,
    probe_cached_z,
    prepare_weight_delta_covariance,
    snapshot_base_parameters,
    solve_weight_delta_system,
)
from experiment.phase16.protocol.genrecedit_inspired import (  # noqa: E402
    GRIDGE_METHOD_NAME,
    GRIDGE_RIDGE_RULE,
    GRIDGE_SOLVE_VARIANT,
    form_condition_targeted_ridge_system,
    solve_condition_targeted_ridge_system,
    validate_gridge_method_config,
)
from experiment.phase16.protocol.resource_probe import load_gram  # noqa: E402
from experiment.phase16.protocol.specgr_contract_smoke import (  # noqa: E402
    read_metadata,
    tokenize_passage_batch,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ProgressReporter has a heartbeat thread and a foreground writer.  A
    # fixed .tmp name lets the two atomic replacements steal one another's
    # temporary file, which is exactly what happened near the end of f1.
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def execution_identity_payload(
    config_path: Path, loaded_config_sha256: str
) -> dict[str, Any]:
    resolved_config = config_path.resolve()
    if sha256(resolved_config) != loaded_config_sha256:
        raise ValueError("S16-3 config changed between load and identity capture")
    return {
        "captured_at_utc": utc_now(),
        "config_path": str(resolved_config.relative_to(ROOT)),
        "config_sha256": loaded_config_sha256,
        "code_sha256": {
            path: sha256(ROOT / path) for path in EXECUTED_CODE_PATHS
        },
    }


def capture_execution_identity(
    config_path: Path, loaded_config_sha256: str, output: Path
) -> tuple[dict[str, Any], str]:
    """Freeze the shared-worktree bytes before CPU preflight and GPU work."""

    identity = execution_identity_payload(config_path, loaded_config_sha256)
    identity_path = output / "execution_identity.json"
    write_json(identity_path, identity)
    return identity, sha256(identity_path)


def verify_execution_identity(
    config_path: Path, loaded_config_sha256: str, identity_path: Path
) -> tuple[dict[str, Any], str]:
    if not identity_path.is_file() or identity_path.is_symlink():
        raise ValueError("Missing regular preflight execution identity")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    current = execution_identity_payload(config_path, loaded_config_sha256)
    for key in ("config_path", "config_sha256", "code_sha256"):
        if identity.get(key) != current[key]:
            raise ValueError(f"Execution identity drift after CPU preflight: {key}")
    if not isinstance(identity.get("captured_at_utc"), str):
        raise ValueError("Execution identity lacks its capture time")
    return identity, sha256(identity_path)


def verify_s1_resolved_inputs(
    config: Mapping[str, Any], inputs: Any, counts: Mapping[str, int], max_history: int
) -> dict[str, Any]:
    """Prove that S16-1 resolution selected exactly the S16-3 frozen files."""

    resolved = {
        "train_sequences": inputs.train_sequences,
        "split_manifest": inputs.split_manifest,
        "retained_warm_items": inputs.retained_warm_items,
        "pseudo_cold_items": inputs.pseudo_cold_items,
        "cold_items": inputs.real_cold_items,
        "lexical_paths": inputs.lexical_paths,
        "metadata": inputs.metadata,
        "content_embeddings": inputs.content_embeddings,
    }
    s1_sha_labels = {
        "train_sequences": "train_sequences",
        "retained_warm_items": "retained_warm_items",
        "pseudo_cold_items": "pseudo_cold_items",
        "cold_items": "real_cold_items",
        "lexical_paths": "lexical_paths",
        "metadata": "metadata",
        "content_embeddings": "content_embeddings",
    }
    files: dict[str, dict[str, str]] = {}
    for label, path in resolved.items():
        if path is None:
            raise ValueError(f"S16-1 did not resolve required S16-3 input: {label}")
        relative = repo_relative_path(path)
        declared = config["inputs"][label]
        actual_sha = sha256(path)
        if relative != declared["path"] or actual_sha != declared["sha256"]:
            raise ValueError(f"S16-1/S16-3 resolved input mismatch: {label}")
        s1_label = s1_sha_labels.get(label)
        if s1_label is not None and inputs.expected_sha256.get(s1_label) != actual_sha:
            raise ValueError(f"S16-1 manifest SHA disagrees with S16-3: {label}")
        files[label] = {"path": relative, "sha256": actual_sha}
    expected_counts = {
        "targets": int(config["frozen_workload"]["edit_targets"]),
        "contexts": int(config["frozen_workload"]["contexts"]),
        "requests": int(config["frozen_workload"]["prefix_next_token_requests"]),
    }
    if dict(counts) != expected_counts:
        raise ValueError("S16-1/S16-3 full-universe counts disagree")
    return {
        "preflight_config": {
            "path": config["inputs"]["s1_preflight_config"]["path"],
            "sha256": config["inputs"]["s1_preflight_config"]["sha256"],
        },
        "files": files,
        "counts": expected_counts,
        "maximum_history_items": int(max_history),
        "pass": True,
    }


class ProgressReporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._state = ("initializing", 0, 1, "steps")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)

    def _write(self) -> None:
        with self._lock:
            stage, current, total, unit = self._state
        write_json(
            self.path,
            {
                "stage": stage,
                "progress_current": current,
                "progress_total": total,
                "progress_unit": unit,
                "updated_at": utc_now(),
            },
        )

    def _heartbeat(self) -> None:
        while not self._stop.wait(45):
            self._write()

    def start(self) -> None:
        self._write()
        self._thread.start()

    def set(self, stage: str, current: int, total: int, unit: str) -> None:
        with self._lock:
            self._state = (stage, int(current), int(total), unit)
        self._write()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._write()


def verify_inputs(config: dict[str, Any]) -> list[str]:
    opened: list[str] = []
    for label, spec in config["inputs"].items():
        if "sha256" not in spec:
            continue
        path = ROOT / spec["path"]
        if not path.is_file() or path.is_symlink() or sha256(path) != spec["sha256"]:
            raise ValueError(f"Frozen S16-3 input drift: {label}")
        opened.append(spec["path"])
    source = ROOT / config["inputs"]["official_genrecedit"]["path"]
    expected = config["inputs"]["official_genrecedit"]["commit"]
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if head != expected or dirty:
        raise ValueError("Pinned GenRecEdit source commit/worktree drift")
    return sorted(opened)


def read_contexts(dataset_root: Path, manifest: Mapping[str, Any]) -> dict[str, list[tuple[str, tuple[str, ...]]]]:
    result: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for shard in manifest["shards"]:
        path = dataset_root / shard["pseudo_contexts"]["path"]
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                row = json.loads(raw)
                result.setdefault(row["cold_item"], []).append(
                    (row["source_warm_item"], tuple(row["train_context_items"]))
                )
    if any(len(rows) != 10 for rows in result.values()):
        raise ValueError("Full-target context materialization lost the official ten contexts")
    return result


def encode_catalog_paths(
    tokenizer, lexical_paths: Mapping[str, Sequence[str]]
) -> dict[str, tuple[int, ...]]:
    encoded: dict[str, tuple[int, ...]] = {}
    for item, path in lexical_paths.items():
        ids = tuple(int(value) for value in tokenizer.convert_tokens_to_ids(list(path)))
        if any(value == tokenizer.unk_token_id for value in ids):
            raise ValueError(f"Lexical token maps to UNK: {item}")
        encoded[item] = ids
    if len(set(encoded.values())) != len(encoded):
        raise ValueError("Encoded lexical catalog has a collision")
    return encoded


def load_frozen_tokenizer(config: Mapping[str, Any]):
    spec = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["name"],
        revision=spec["revision"],
        local_files_only=spec["local_files_only"],
    )
    vocabulary_sha = hashlib.sha256(
        json.dumps(
            tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    vocab_file = Path(tokenizer.vocab_file)
    if (
        tokenizer.__class__.__name__ != spec["class"]
        or vocabulary_sha != spec["vocabulary_sha256"]
        or not vocab_file.is_file()
        or sha256(vocab_file) != spec["sentencepiece_sha256"]
    ):
        raise ValueError("Frozen t5-small tokenizer provenance drift")
    return tokenizer, {
        "name": spec["name"],
        "revision": spec["revision"],
        "class": tokenizer.__class__.__name__,
        "vocabulary_sha256": vocabulary_sha,
        "sentencepiece_sha256": sha256(vocab_file),
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "unk_token_id": int(tokenizer.unk_token_id),
    }


def clear_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(0)


def gpu_readmission(physical_gpu: int, minimum_free_mib: int) -> int:
    observed = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={int(physical_gpu)}",
            "--query-gpu=memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    physical_free_mib = int(observed.split(",", 1)[0].strip())
    if physical_free_mib < minimum_free_mib:
        raise RuntimeError(
            "GPU_READMISSION_FAILED_BEFORE_CUDA: "
            f"physical GPU {physical_gpu} has {physical_free_mib} MiB free, "
            f"needs {minimum_free_mib}"
        )
    torch.cuda.init()
    free_bytes, _ = torch.cuda.mem_get_info(0)
    cuda_free_mib = int(free_bytes // 1024**2)
    if cuda_free_mib < minimum_free_mib:
        raise RuntimeError(
            f"GPU_READMISSION_FAILED_AFTER_CUDA: visible GPU has {cuda_free_mib} MiB free, "
            f"needs {minimum_free_mib}"
        )
    return min(physical_free_mib, cuda_free_mib)


def request_rows(requests: Sequence[FullTargetRequest]) -> list[dict[str, Any]]:
    return [
        {"history": list(row.context_items), "target": row.cold_item}
        for row in requests
    ]


def decoder_ids(
    requests: Sequence[FullTargetRequest], start_token: int, device: torch.device
) -> torch.Tensor:
    rows = [(start_token,) + tuple(row.prefix_token_ids) for row in requests]
    if len({len(row) for row in rows}) != 1:
        raise ValueError("One z batch must contain one lexical position")
    return torch.tensor(rows, dtype=torch.long, device=device)


class RealGRAMZRuntime:
    def __init__(
        self,
        *,
        model,
        requests: Sequence[FullTargetRequest],
        metadata: Mapping[str, str],
        lexical_paths: Mapping[str, Sequence[str]],
        tokenizer,
        layer: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.requests = tuple(requests)
        self.layer = int(layer)
        self.device = device
        context, _ = tokenize_passage_batch(
            request_rows(self.requests), metadata, lexical_paths, tokenizer, device
        )
        self.input_ids = context["input_ids"]
        self.attention = context["attention_mask"]
        model.encoder.n_passages = self.input_ids.shape[1]
        with torch.no_grad():
            hidden = model.encoder(
                input_ids=self.input_ids.reshape(self.input_ids.shape[0], -1),
                attention_mask=self.attention.reshape(self.attention.shape[0], -1),
                return_dict=True,
            )[0]
        self.encoder_outputs = BaseModelOutput(last_hidden_state=hidden.detach())
        self.flat_attention = self.attention.reshape(self.attention.shape[0], -1)
        self.decoder = decoder_ids(
            self.requests, int(model.config.decoder_start_token_id), device
        )
        self.module = model.decoder.block[layer].layer[2].DenseReluDense.wo
        self.forward_seconds: list[float] = []
        self.last_logits: torch.Tensor | None = None

    def __call__(
        self,
        batch: Sequence[FullTargetRequest],
        deltas: torch.Tensor,
        active: Sequence[int],
    ) -> ZForwardBatch:
        if tuple(batch) != self.requests or deltas.shape[0] != len(self.requests):
            raise ValueError("Real-GRAM z callback received a different request batch")
        started = time.perf_counter()
        past = None
        if self.decoder.shape[1] > 1:
            with torch.no_grad():
                prefix = self.model(
                    encoder_outputs=self.encoder_outputs,
                    attention_mask=self.flat_attention,
                    decoder_input_ids=self.decoder[:, :-1],
                    use_cache=True,
                    return_dict=True,
                )
                past = prefix.past_key_values
        captured: list[torch.Tensor] = []
        active_index = torch.tensor(tuple(active), dtype=torch.long, device=self.device)

        def inject(_module, _inputs, output):
            base = output[:, -1, :]
            captured.append(base.detach().clone())
            modified = output.clone()
            if active_index.numel():
                modified[active_index, -1, :] = base[active_index] + deltas[active_index]
            return modified

        handle = self.module.register_forward_hook(inject)
        try:
            output = self.model(
                encoder_outputs=self.encoder_outputs,
                attention_mask=self.flat_attention,
                decoder_input_ids=self.decoder[:, -1:],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Real-GRAM z hook did not capture exactly one target output")
        logits = output.logits[:, -1, :]
        self.last_logits = logits.detach()
        torch.cuda.synchronize(self.device)
        self.forward_seconds.append(time.perf_counter() - started)
        return ZForwardBatch(logits=logits, target_inits=captured[0])


class BatchedRealGRAMZRuntime:
    """Own only the currently active optimizer microbatch's encoder cache."""

    def __init__(
        self,
        *,
        model,
        metadata: Mapping[str, str],
        lexical_paths: Mapping[str, Sequence[str]],
        tokenizer,
        layer: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.lexical_paths = lexical_paths
        self.tokenizer = tokenizer
        self.layer = int(layer)
        self.device = device
        self._batch: tuple[FullTargetRequest, ...] | None = None
        self._runtime: RealGRAMZRuntime | None = None
        self.forward_seconds_by_batch: list[list[float]] = []
        self.request_count_by_batch: list[int] = []
        self.last_logits_by_batch: list[torch.Tensor | None] = []

    def __call__(
        self,
        batch: Sequence[FullTargetRequest],
        deltas: torch.Tensor,
        active: Sequence[int],
    ) -> ZForwardBatch:
        normalized = tuple(batch)
        if normalized != self._batch:
            self._runtime = None
            gc.collect()
            self._runtime = RealGRAMZRuntime(
                model=self.model,
                requests=normalized,
                metadata=self.metadata,
                lexical_paths=self.lexical_paths,
                tokenizer=self.tokenizer,
                layer=self.layer,
                device=self.device,
            )
            self._batch = normalized
            self.forward_seconds_by_batch.append([])
            self.request_count_by_batch.append(len(normalized))
            self.last_logits_by_batch.append(None)
        if self._runtime is None:  # pragma: no cover - guarded above
            raise RuntimeError("Missing live real-GRAM z runtime")
        observation = self._runtime(normalized, deltas, active)
        self.forward_seconds_by_batch[-1].append(self._runtime.forward_seconds[-1])
        self.last_logits_by_batch[-1] = observation.logits.detach().cpu()
        return observation

    def last_logits(self) -> torch.Tensor:
        if not self.last_logits_by_batch or any(
            value is None for value in self.last_logits_by_batch
        ):
            raise RuntimeError("Real-GRAM z runtime has incomplete final logits")
        return torch.cat(
            [value for value in self.last_logits_by_batch if value is not None], dim=0
        )


def request_subset_sha256(
    requests_by_position: Mapping[int, Sequence[FullTargetRequest]],
) -> str:
    rows = [
        {
            "position": position,
            "cold_item": request.cold_item,
            "source_warm_item": request.source_warm_item,
            "context_items": list(request.context_items),
            "full_target_path": list(request.full_target_path),
            "prefix_token_ids": list(request.prefix_token_ids),
            "target_token_id": request.target_token_id,
            "legal_token_ids": list(request.legal_token_ids),
        }
        for position in sorted(requests_by_position)
        for request in requests_by_position[position]
    ]
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def request_subset_manifest(
    requests_by_position: Mapping[int, Sequence[FullTargetRequest]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in sorted(requests_by_position):
        for request in requests_by_position[position]:
            identity = {
                "position": position,
                "cold_item": request.cold_item,
                "source_warm_item": request.source_warm_item,
                "context_items": list(request.context_items),
                "full_target_path": list(request.full_target_path),
                "prefix_token_ids": list(request.prefix_token_ids),
                "target_token_id": request.target_token_id,
                "legal_token_ids": list(request.legal_token_ids),
            }
            rows.append(
                {
                    **identity,
                    "row_id": hashlib.sha256(
                        json.dumps(
                            identity, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                }
            )
    return rows


def run_z_candidate(
    *,
    size: int,
    requests_by_position: Mapping[int, Sequence[FullTargetRequest]],
    config: dict[str, Any],
    metadata: Mapping[str, str],
    lexical_paths: Mapping[str, Sequence[str]],
    tokenizer,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    clear_cuda(device)
    historical = ROOT / config["inputs"]["gram_config"]["path"]
    checkpoint = ROOT / config["inputs"]["gram_checkpoint"]["path"]
    model = load_gram(historical, checkpoint, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    fixed_per_position = int(config["sweep"]["candidate_requests_per_position"])
    positions = tuple(int(value) for value in config["sweep"]["candidate_positions"])
    fixed_requests = {
        position: tuple(requests_by_position[position][:fixed_per_position])
        for position in positions
    }
    if any(len(rows) != fixed_per_position for rows in fixed_requests.values()):
        raise RuntimeError("Frozen candidate subset lacks 16 requests at a lexical position")
    expected_trace = tuple(config["frozen_workload"]["satisfied_check_step_indices"])
    batch_records: list[dict[str, Any]] = []
    total_valid = 0
    total_failed = 0
    total_cache = 0
    first_ten_seconds = 0.0
    first_ten_request_steps = 0
    for position in positions:
        layer = official_position_to_layer([position])[position]
        runtime = BatchedRealGRAMZRuntime(
            model=model,
            metadata=metadata,
            lexical_paths=lexical_paths,
            tokenizer=tokenizer,
            layer=layer,
            device=device,
        )
        result = optimize_z_vectors(
            requests=fixed_requests[position],
            vector_dimension=int(model.config.d_model),
            forward_batch=runtime,
            config=ZOptimizationConfig(
                v_lr=config["frozen_workload"]["z_learning_rate"],
                v_num_grad_steps=config["frozen_workload"]["z_steps"],
                v_weight_decay=config["frozen_workload"]["z_weight_decay"],
                z_vector_max=config["frozen_workload"]["z_vector_max"],
                batch_size=size,
            ),
            cache_hits={},
            device=device,
        )
        if len(runtime.forward_seconds_by_batch) != len(result.lifecycle_check_steps_by_batch):
            raise RuntimeError("Observed z runtime batches do not align with lifecycle traces")
        for batch_index, (times, step_times, batch_count, trace, lrs) in enumerate(
            zip(
                runtime.forward_seconds_by_batch,
                result.step_elapsed_seconds_by_batch,
                runtime.request_count_by_batch,
                result.lifecycle_check_steps_by_batch,
                result.scheduler_lrs_by_batch,
            )
        ):
            if len(times) < 10 or len(step_times) < 10:
                raise RuntimeError("Official lifecycle did not execute its first ten steps")
            first_ten_seconds += sum(step_times[:10])
            first_ten_request_steps += batch_count * 10
            batch_records.append(
                {
                    "position": position,
                    "batch_index": batch_index,
                    "request_count": batch_count,
                    "forward_calls": len(times),
                    "first_ten_forward_seconds": times[:10],
                    "first_ten_objective_step_seconds": list(step_times[:10]),
                    "lifecycle_check_steps": list(trace),
                    "scheduler_step_count": len(lrs),
                    "scheduler_lr_first": lrs[0] if lrs else None,
                    "scheduler_lr_last": lrs[-1] if lrs else None,
                    "observed_step_29": tuple(trace) == expected_trace,
                    "official_lifecycle_prefix": tuple(trace)
                    == expected_trace[: len(trace)],
                }
            )
        total_valid += result.valid_count
        total_failed += result.failed_count
        total_cache += len(result.cache_hit_indices)
        del result, runtime
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    expected_request_count = int(config["sweep"]["candidate_total_cache_miss_requests"])
    payload = {
        "microbatch": size,
        "candidate_request_count": sum(len(rows) for rows in fixed_requests.values()),
        "candidate_requests_by_position": {
            str(position): len(rows) for position, rows in fixed_requests.items()
        },
        "candidate_subset_sha256": request_subset_sha256(fixed_requests),
        "elapsed_seconds": elapsed,
        "forward_calls": sum(row["forward_calls"] for row in batch_records),
        "steady_request_steps_per_second": first_ten_request_steps / first_ten_seconds,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "valid_z_count": total_valid,
        "failed_z_count": total_failed,
        "cache_hit_count": total_cache,
        "batch_records": batch_records,
        "semantic_checks": {
            "formal_cache_empty": total_cache == 0,
            "identical_fixed_request_count": sum(len(rows) for rows in fixed_requests.values())
            == expected_request_count,
            "full_30_step_budget_configured": config["frozen_workload"]["z_steps"] == 30,
            "official_lifecycle_prefix": bool(batch_records)
            and all(row["official_lifecycle_prefix"] for row in batch_records),
            "first_ten_outcome_independent_objective_timing": bool(batch_records)
            and all(len(row["first_ten_objective_step_seconds"]) == 10 for row in batch_records),
            "scheduler_finite": bool(batch_records)
            and all(
                row["scheduler_step_count"] > 0
                and math.isfinite(row["scheduler_lr_first"])
                and math.isfinite(row["scheduler_lr_last"])
                for row in batch_records
            ),
            "valid_failed_cover_fixed_subset": total_valid + total_failed
            == expected_request_count,
            "checkpoint_file_unchanged": sha256(checkpoint)
            == config["inputs"]["gram_checkpoint"]["sha256"],
        },
    }
    payload["eligible"] = (
        all(payload["semantic_checks"].values())
        and peak_reserved
        <= config["sweep"].get(
            "maximum_candidate_peak_reserved_mib",
            config["sweep"]["maximum_eligible_peak_reserved_mib"],
        )
    )
    del model
    clear_cuda(device)
    return payload


def choose_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("No G-FULL request microbatch passed the preregistered resource contract")
    best = max(eligible, key=lambda row: row["steady_request_steps_per_second"])
    near = [
        row
        for row in eligible
        if row["steady_request_steps_per_second"]
        >= 0.98 * best["steady_request_steps_per_second"]
    ]
    return min(near, key=lambda row: (row["microbatch"], row["peak_reserved_mib"]))


def solve_status_label(
    position_metrics: Mapping[str, Mapping[str, Any]],
    aggregated: Mapping[str, torch.Tensor],
) -> str:
    """Label solve coverage without conflating valid-z rows with solved updates."""

    if aggregated:
        return "SOLVE_AND_AGGREGATE_EXERCISED"
    if any(int(row.get("valid_z_count", 0)) > 0 for row in position_metrics.values()):
        return "VALID_Z_PRESENT_BUT_NO_POSITION_SOLVE_COMPLETED"
    return "NO_VALID_Z_IN_PREREGISTERED_RESOURCE_SUBSET"


def independent_full_lifecycle_probe(
    request: FullTargetRequest, *, vector_dimension: int
) -> dict[str, Any]:
    """Exercise step 29 independently of real z success and batch selection."""

    vocabulary = max(
        max(request.legal_token_ids), request.target_token_id
    ) + 2
    competitor = next(
        token for token in range(vocabulary) if token != request.target_token_id
    )
    calls = 0

    def forward(batch, deltas, active):
        nonlocal calls
        calls += 1
        logits = deltas[:, :1] * 0.0 + torch.zeros(len(batch), vocabulary)
        logits[:, request.target_token_id] = 1.0
        logits[:, competitor] = 2.0
        return ZForwardBatch(logits=logits, target_inits=torch.ones_like(deltas))

    result = optimize_z_vectors(
        requests=[request],
        vector_dimension=vector_dimension,
        forward_batch=forward,
        config=ZOptimizationConfig(
            v_lr=0.5,
            v_num_grad_steps=30,
            v_weight_decay=0.2,
            z_vector_max=8000.0,
            batch_size=1,
        ),
        cache_hits={},
        device="cpu",
    )
    expected = (10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
    return {
        "scope": "synthetic_failure_row_not_used_for_candidate_selection_or_runtime",
        "forward_calls": calls,
        "lifecycle_check_steps": list(result.lifecycle_check_steps_by_batch[0]),
        "scheduler_step_count": len(result.scheduler_lrs_by_batch[0]),
        "failed_z_count": result.failed_count,
        "pass": calls == 30
        and result.lifecycle_check_steps_by_batch == (expected,)
        and len(result.scheduler_lrs_by_batch[0]) == 30
        and result.failed_count == 1,
    }


def select_covariance_transitions(
    train_rows: Sequence[tuple[str, Sequence[str]]],
    lexical_paths: Mapping[str, Sequence[str]],
    *,
    rows_by_position: Mapping[int, int],
    seed: int,
) -> dict[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for user, items in train_rows:
        for position in range(1, len(items)):
            target = items[position]
            rows.append(
                {
                    "user": user,
                    "position": position,
                    "history": list(items[max(0, position - 20) : position]),
                    "target": target,
                    "rank": hashlib.sha256(
                        f"{seed}|gfull-cov|{user}|{position}|{target}".encode()
                    ).hexdigest(),
                }
            )
    ranked = sorted(rows, key=lambda row: (row["rank"], row["user"], row["position"]))
    selected: dict[int, list[dict[str, Any]]] = {}
    for lexical_position, count in sorted(rows_by_position.items()):
        eligible = [
            {**row, "lexical_position": int(lexical_position)}
            for row in ranked
            if len(lexical_paths[row["target"]]) > int(lexical_position)
        ]
        selected[int(lexical_position)] = eligible[: int(count)]
        if len(selected[int(lexical_position)]) != int(count):
            raise RuntimeError(
                f"Insufficient train-only covariance rows at lexical position {lexical_position}"
            )
    return selected


def covariance_resource_probe(
    *,
    model,
    rows_by_position: Mapping[int, Sequence[dict[str, Any]]],
    metadata: Mapping[str, str],
    lexical_paths: Mapping[str, Sequence[str]],
    tokenizer,
    device: torch.device,
    batch_size: int,
    progress_callback: Callable[[int, Mapping[int, float]], None] | None = None,
) -> tuple[Any, float, dict[int, torch.Tensor], dict[int, float]]:
    position_layers = official_position_to_layer(range(6))
    activations: dict[int, torch.Tensor] = {}
    covariance_by_position: dict[int, torch.Tensor] = {}
    available_rows_by_position: dict[int, int] = {}
    used_rows_by_position: dict[int, int] = {}
    elapsed_by_position: dict[int, float] = {}
    for position, rows in sorted(rows_by_position.items()):
        torch.cuda.synchronize(device)
        position_started = time.perf_counter()
        layer = position_layers[position]
        captured_chunks: list[torch.Tensor] = []
        for start in range(0, len(rows), batch_size):
            batch_rows = list(rows[start : start + batch_size])
            context, _ = tokenize_passage_batch(
                batch_rows, metadata, lexical_paths, tokenizer, device
            )
            captured: list[torch.Tensor] = []
            module = model.decoder.block[layer].layer[2].DenseReluDense.wo
            handle = module.register_forward_pre_hook(
                lambda _module, inputs: captured.append(inputs[0].detach().cpu())
            )
            try:
                with torch.no_grad():
                    model(**context, use_cache=False, return_dict=True)
            finally:
                handle.remove()
            if len(captured) != 1 or captured[0].shape[0] != len(batch_rows):
                raise RuntimeError("Covariance hook did not capture one aligned batch")
            captured_chunks.append(captured[0][:, position, :])
        position_activations = torch.cat(captured_chunks, dim=0)
        position_result = collect_covariance(
            {position: position_activations}, mom2_n_samples=len(rows)
        )
        torch.cuda.synchronize(device)
        elapsed_by_position[position] = time.perf_counter() - position_started
        activations[position] = position_activations
        covariance_by_position.update(position_result.covariance_by_position)
        available_rows_by_position.update(position_result.available_rows_by_position)
        used_rows_by_position.update(position_result.used_rows_by_position)
        if progress_callback is not None:
            progress_callback(position, dict(elapsed_by_position))
    result = PositionCovarianceResult(
        covariance_by_position=covariance_by_position,
        available_rows_by_position=available_rows_by_position,
        used_rows_by_position=used_rows_by_position,
        mom2_n_samples=max(len(rows) for rows in rows_by_position.values()),
    )
    return result, sum(elapsed_by_position.values()), activations, elapsed_by_position


def covariance_convergence_diagnostics(
    activations: Mapping[int, torch.Tensor],
    checkpoints_by_position: Mapping[int, Sequence[int]],
) -> dict[str, list[dict[str, float | int]]]:
    diagnostics: dict[str, list[dict[str, float | int]]] = {}
    for position, rows in sorted(activations.items()):
        checkpoints = tuple(int(value) for value in checkpoints_by_position[position])
        if not checkpoints or checkpoints[-1] != rows.shape[0] or any(
            left >= right for left, right in zip(checkpoints, checkpoints[1:])
        ):
            raise ValueError("Covariance convergence checkpoints must end at the resource row count")
        reference = rows.float().T @ rows.float() / float(rows.shape[0])
        reference_norm = float(torch.linalg.matrix_norm(reference, ord="fro"))
        position_rows: list[dict[str, float | int]] = []
        for count in checkpoints:
            if count == rows.shape[0]:
                drift = 0.0
            else:
                moment = rows[:count].float().T @ rows[:count].float() / float(count)
                drift = float(torch.linalg.matrix_norm(moment - reference, ord="fro"))
            position_rows.append(
                {
                    "rows": count,
                    "relative_frobenius_drift_to_largest_resource_checkpoint": drift
                    / max(reference_norm, 1e-30),
                }
            )
        diagnostics[str(position)] = position_rows
    return diagnostics


def convergence_row_equivalents(
    checkpoints_by_position: Mapping[int, Sequence[int | str]],
    available_rows_by_position: Mapping[int, int],
) -> int:
    """Count row-moment work across all requested convergence checkpoints."""

    total = 0
    for position, checkpoints in checkpoints_by_position.items():
        available = int(available_rows_by_position[int(position)])
        effective = {
            available if checkpoint == "full" else min(int(checkpoint), available)
            for checkpoint in checkpoints
        }
        total += available + sum(count for count in effective if count < available)
    return total


def key_forward_factory(
    *, model, metadata, lexical_paths, tokenizer, device
):
    def run(requests: Sequence[FullTargetRequest]) -> None:
        context, _ = tokenize_passage_batch(
            request_rows(requests), metadata, lexical_paths, tokenizer, device
        )
        dec = decoder_ids(
            requests, int(model.config.decoder_start_token_id), device
        )
        with torch.no_grad():
            model(
                input_ids=context["input_ids"],
                attention_mask=context["attention_mask"],
                decoder_input_ids=dec,
                use_cache=False,
                return_dict=True,
            )
    return run


def probe_final_z_logits(
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
) -> torch.Tensor:
    """Re-probe satisfied z states and failed terminal deltas explicitly."""

    diagnostic_deltas = [
        result.delta_vectors[index]
        if result.delta_vectors[index] is not None
        else result.terminal_delta_vectors[index]
        for index in range(len(requests))
    ]
    if any(value is None for value in diagnostic_deltas):
        raise RuntimeError("Final z diagnostic lost a request delta")
    logits: list[torch.Tensor] = []
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
        logits.append(observation.logits.detach().cpu())
        del runtime
    return torch.cat(logits, dim=0)


def trigger_parity_contract_probe(
    *,
    model,
    requests_by_position: Mapping[int, Sequence[FullTargetRequest]],
    catalog_ids: Mapping[str, Sequence[int]],
    metadata: Mapping[str, str],
    lexical_paths: Mapping[str, Sequence[str]],
    tokenizer,
    device: torch.device,
) -> dict[str, Any]:
    """Exercise every trigger position plus EOS/pad/dead inactive rows."""

    position_to_layer = official_position_to_layer(range(6))
    parameters = dict(model.named_parameters())
    synthetic_aggregates: dict[str, torch.Tensor] = {}
    for layer in sorted(set(position_to_layer.values())):
        name = edited_parameter_name(layer)
        delta = torch.zeros_like(parameters[name], device=device)
        diagonal = min(delta.shape)
        indices = torch.arange(diagonal)
        delta[indices, indices] = 0.1
        synthetic_aggregates[name] = delta
    bundles = build_one_one_position_bundles(
        position_to_layer=position_to_layer,
        aggregated_updates=synthetic_aggregates,
    )
    if bundles[0][edited_parameter_name(0)] is not bundles[4][edited_parameter_name(0)]:
        raise RuntimeError("Shared-layer trigger positions lost their aggregate tensor identity")
    if bundles[1][edited_parameter_name(1)] is not bundles[5][edited_parameter_name(1)]:
        raise RuntimeError("Shared-layer trigger positions lost their aggregate tensor identity")
    snapshot = snapshot_base_parameters(model, sorted(synthetic_aggregates))

    def logits_for(request: FullTargetRequest, decoder: torch.Tensor) -> torch.Tensor:
        context, _ = tokenize_passage_batch(
            request_rows([request]), metadata, lexical_paths, tokenizer, device
        )
        with torch.no_grad():
            return model(
                input_ids=context["input_ids"],
                attention_mask=context["attention_mask"],
                decoder_input_ids=decoder,
                use_cache=False,
                return_dict=True,
            ).logits.detach().cpu()

    position_cases = {
        position: (
            requests_by_position[position][0],
            decoder_ids(
                [requests_by_position[position][0]],
                int(model.config.decoder_start_token_id),
                device,
            ),
        )
        for position in range(6)
    }
    reference_request = requests_by_position[5][0]
    start = int(model.config.decoder_start_token_id)
    root_children = {int(path[0]) for path in catalog_ids.values()}
    dead_token = next(
        token
        for token in range(int(model.config.vocab_size))
        if token not in root_children
        and token not in {int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)}
    )
    inactive_cases = {
        "complete_path": torch.tensor(
            [(start,) + tuple(reference_request.full_target_path)],
            dtype=torch.long,
            device=device,
        ),
        "eos": torch.tensor(
            [[start, int(tokenizer.eos_token_id)]], dtype=torch.long, device=device
        ),
        "padding": torch.tensor(
            [[start, int(tokenizer.pad_token_id)]], dtype=torch.long, device=device
        ),
        "dead_prefix": torch.tensor([[start, dead_token]], dtype=torch.long, device=device),
    }
    baseline_positions = {
        position: logits_for(request, decoder)
        for position, (request, decoder) in position_cases.items()
    }
    baseline_inactive = {
        name: logits_for(reference_request, decoder)
        for name, decoder in inactive_cases.items()
    }
    with OneOneGenerationDeltaContext(
        model=model,
        deltas_by_position=bundles,
        position_to_layer=position_to_layer,
        encoded_catalog_paths=catalog_ids.values(),
        decoder_start_token_id=start,
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
    ) as context:
        edited_positions: dict[int, torch.Tensor] = {}
        for position, (request, decoder) in position_cases.items():
            model.prepare_inputs_for_generation(decoder)
            edited_positions[position] = logits_for(request, decoder)
        edited_inactive: dict[str, torch.Tensor] = {}
        for name, decoder in inactive_cases.items():
            model.prepare_inputs_for_generation(decoder)
            edited_inactive[name] = logits_for(reference_request, decoder)
        applied_rows = dict(context.applied_rows_by_position)
        dead_rows = int(context.dead_prefix_rows)
    restored_positions = {
        position: logits_for(request, decoder)
        for position, (request, decoder) in position_cases.items()
    }
    restored_inactive = {
        name: logits_for(reference_request, decoder)
        for name, decoder in inactive_cases.items()
    }
    parameter_parity = assert_base_parameter_parity(model, snapshot)
    edited_changed = {
        str(position): not torch.equal(
            baseline_positions[position], edited_positions[position]
        )
        for position in range(6)
    }
    inactive_exact = {
        name: torch.equal(baseline_inactive[name], edited_inactive[name])
        for name in inactive_cases
    }
    restored_exact = {
        str(position): torch.equal(
            baseline_positions[position], restored_positions[position]
        )
        for position in range(6)
    }
    restored_inactive_exact = {
        name: torch.equal(baseline_inactive[name], restored_inactive[name])
        for name in inactive_cases
    }
    passed = (
        all(edited_changed.values())
        and all(inactive_exact.values())
        and all(restored_exact.values())
        and all(restored_inactive_exact.values())
        and all(applied_rows[position] >= 1 for position in range(6))
        and dead_rows >= 1
        and parameter_parity.get("exact") is True
    )
    return {
        "positions": list(range(6)),
        "shared_aggregate_identity": {"0_4": True, "1_5": True},
        "edited_output_changed": edited_changed,
        "inactive_output_exact": inactive_exact,
        "restored_output_exact": restored_exact,
        "restored_inactive_output_exact": restored_inactive_exact,
        "applied_rows_by_position": {str(key): value for key, value in applied_rows.items()},
        "dead_prefix_rows": dead_rows,
        "base_parameter_parity": parameter_parity,
        "pass": passed,
    }


def generation_resource_probe(
    *,
    model,
    rows: Sequence[dict[str, Any]],
    catalog_ids: Mapping[str, Sequence[int]],
    metadata: Mapping[str, str],
    lexical_paths: Mapping[str, Sequence[str]],
    tokenizer,
    device: torch.device,
    beam_size: int,
) -> dict[str, Any]:
    """Time the strict base+edited beam path used by formal admission."""

    if not rows or beam_size < 1:
        raise ValueError("Generation resource probe requires rows and a beam budget")
    position_to_layer = official_position_to_layer(range(6))
    parameters = dict(model.named_parameters())
    aggregates: dict[str, torch.Tensor] = {}
    for layer in sorted(set(position_to_layer.values())):
        name = edited_parameter_name(layer)
        delta = torch.zeros_like(parameters[name], device=device)
        diagonal = min(delta.shape)
        indices = torch.arange(diagonal)
        delta[indices, indices] = 0.1
        aggregates[name] = delta
    bundles = build_one_one_position_bundles(
        position_to_layer=position_to_layer, aggregated_updates=aggregates
    )
    complete_paths = {tuple(map(int, path)): item for item, path in catalog_ids.items()}
    if len(complete_paths) != len(catalog_ids):
        raise ValueError("Generation resource catalog paths collide")
    children: dict[tuple[int, ...], set[int]] = {}
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id)
    for path in complete_paths:
        for depth, token in enumerate((*path, eos)):
            children.setdefault(path[:depth], set()).add(int(token))

    def allowed(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        prefix = tuple(int(value) for value in input_ids.detach().cpu().tolist()[1:])
        return sorted(children.get(prefix, ()))

    def run_one(row: dict[str, Any]) -> tuple[float, list[tuple[int, ...]], bool]:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        context, _ = tokenize_passage_batch(
            [row], metadata, lexical_paths, tokenizer, device
        )
        generated = model.generate(
            input_ids=context["input_ids"],
            attention_mask=context["attention_mask"],
            max_length=max(map(len, complete_paths)) + 2,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            prefix_allowed_tokens_fn=allowed,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        normalized: list[tuple[int, ...]] = []
        for raw in generated.sequences.detach().cpu().tolist():
            suffix: list[int] = []
            for token in raw[1:]:
                if token == eos:
                    break
                if token != pad:
                    suffix.append(int(token))
            normalized.append(tuple(suffix))
        finite = bool(torch.isfinite(generated.sequences_scores).all().item())
        if (
            len(normalized) != beam_size
            or len(set(normalized)) != beam_size
            or any(path not in complete_paths for path in normalized)
        ):
            raise RuntimeError("Strict admission beam did not yield a unique catalog ranking")
        return elapsed, normalized, finite

    base_seconds: list[float] = []
    edited_seconds: list[float] = []
    digest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        elapsed, ranking, finite = run_one(row)
        base_seconds.append(elapsed)
        digest_rows.append(
            {"event": index, "arm": "base", "ranking": ranking, "finite": finite}
        )
    with OneOneGenerationDeltaContext(
        model=model,
        deltas_by_position=bundles,
        position_to_layer=position_to_layer,
        encoded_catalog_paths=complete_paths,
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        eos_token_id=eos,
        pad_token_id=pad,
    ) as context:
        for index, row in enumerate(rows):
            elapsed, ranking, finite = run_one(row)
            edited_seconds.append(elapsed)
            digest_rows.append(
                {"event": index, "arm": "synthetic_edit", "ranking": ranking, "finite": finite}
            )
        applied_rows = dict(context.applied_rows_by_position)
        dead_rows = int(context.dead_prefix_rows)
    digest = hashlib.sha256(
        json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    all_finite = all(row["finite"] for row in digest_rows)
    return {
        "events": len(rows),
        "beam_size": beam_size,
        "timer_scope": "tokenization_context_transfer_and_generation",
        "catalog_items": len(complete_paths),
        "base_elapsed_seconds": sum(base_seconds),
        "edited_elapsed_seconds": sum(edited_seconds),
        "base_seconds_per_event": sum(base_seconds) / len(rows),
        "edited_seconds_per_event": sum(edited_seconds) / len(rows),
        "base_plus_edited_seconds_per_event": (
            sum(base_seconds) + sum(edited_seconds)
        )
        / len(rows),
        "prediction_digest_sha256": digest,
        "applied_rows_by_position": {str(key): value for key, value in applied_rows.items()},
        "dead_prefix_rows": dead_rows,
        "all_finite": all_finite,
        "pass": all_finite and sum(applied_rows.values()) > 0,
    }


def ceil_to_1024(value: float) -> int:
    return int(math.ceil(value / 1024.0) * 1024)


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
    gridge_method = (
        validate_gridge_method_config(config) if "method" in config else None
    )
    ridge_enabled = gridge_method is not None
    output = ROOT / config["output_dir"]
    raw_path = output / "resource_sweep_summary.json"
    identity_path = output / "execution_identity.json"
    if args.capture_identity_only:
        if raw_path.exists() or identity_path.exists():
            raise SystemExit("Refusing to overwrite an existing S16-3 execution attempt")
        output.mkdir(parents=True, exist_ok=True)
        capture_execution_identity(args.config, loaded_config_sha256, output)
        print("PASS_S16_3_EXECUTION_IDENTITY_CAPTURE")
        return 0
    if None in (
        args.physical_gpu,
        args.admission_free_mib,
        args.admission_util_percent,
        args.worker_hard_timeout_seconds,
        args.expected_peak_mib,
    ):
        raise SystemExit("GPU workload arguments are required outside identity capture")
    if args.physical_gpu in config["resources"]["excluded_physical_gpus"]:
        raise SystemExit("Refusing an explicitly excluded/reserved physical GPU")
    if (
        config["resources"].get("fixed_physical_gpu") is not None
        and int(args.physical_gpu)
        != int(config["resources"]["fixed_physical_gpu"])
    ):
        raise SystemExit("Selected physical GPU disagrees with the fixed resource contract")
    if raw_path.exists():
        raise SystemExit("Refusing to overwrite an existing S16-3 raw resource summary")
    execution_identity, execution_identity_sha = verify_execution_identity(
        args.config, loaded_config_sha256, identity_path
    )
    reporter = ProgressReporter(output / "progress.json")
    reporter.start()
    started = time.perf_counter()
    checkpoint = ROOT / config["inputs"]["gram_checkpoint"]["path"]
    checkpoint_before = sha256(checkpoint)
    try:
        opened = verify_inputs(config)
        opened.extend(
            [
                str(args.config),
                "experiment/phase16/configs/stage16_s1_data_resource_preflight.json",
                config["inputs"]["official_genrecedit"]["path"],
                f"hf://{config['tokenizer']['name']}@{config['tokenizer']['revision']}",
            ]
        )
        opened = sorted(set(opened))
        reporter.set("context_build", 0, 1, "full_target_dataset")
        context_build_started = time.perf_counter()
        inputs, counts, max_history = resolve_stage16_toys_inputs(
            ROOT / config["inputs"]["s1_preflight_config"]["path"]
        )
        s1_resolved_input_contract = verify_s1_resolved_inputs(
            config, inputs, counts, max_history
        )
        dataset_root = output / "request_dataset"
        data_manifest = build_sharded_dataset(
            inputs,
            dataset_root,
            seed=config["seed"],
            contexts_per_target=10,
            max_history=max_history,
            target_shard_size=128,
            similarity_batch_size=64,
            required_counts=counts,
            required_covariance_position_counts={0: 27659, 1: 27659, 2: 27659, 3: 27659, 4: 27659, 5: 2036},
            minimum_long_path_resource_rows=config["sweep"]["covariance_long_path_minimum"],
        )
        context_build_seconds = time.perf_counter() - context_build_started
        reporter.set("request_manifest", 1, 1, "full_target_dataset")

        try:
            free_at_worker = gpu_readmission(
                args.physical_gpu, config["resources"]["minimum_free_mib"]
            )
        except RuntimeError as error:
            if str(error).startswith("GPU_READMISSION_FAILED"):
                print(str(error))
                return 9
            raise
        device = torch.device("cuda:0")
        tokenizer, tokenizer_provenance = load_frozen_tokenizer(config)
        lexical_paths = read_lexical_paths(ROOT / config["inputs"]["lexical_paths"]["path"])
        metadata = read_metadata(ROOT / config["inputs"]["metadata"]["path"])
        catalog_ids = encode_catalog_paths(tokenizer, lexical_paths)
        cold_items = {
            line.strip()
            for line in (ROOT / config["inputs"]["cold_items"]["path"]).read_text().splitlines()
            if line.strip()
        }
        contexts = read_contexts(dataset_root, data_manifest)
        long_cold = sorted(item for item in cold_items if len(catalog_ids[item]) > 5)
        configured_position_counts = {
            int(position): int(count)
            for position, count in config["sweep"].get(
                "position_contract_requests_by_position",
                {
                    str(position): config["sweep"]["candidate_requests_per_position"]
                    for position in range(6)
                },
            ).items()
        }
        resource_pool_items = max(
            int(config["sweep"]["candidate_requests_per_position"]),
            max(configured_position_counts.values()),
        )
        sample_items = long_cold[:resource_pool_items]
        sample_requests = build_full_target_requests(
            catalog_paths=catalog_ids,
            cold_paths={item: catalog_ids[item] for item in sample_items},
            pseudo_contexts={item: contexts[item] for item in sample_items},
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
        )
        requests_by_position: dict[int, list[FullTargetRequest]] = {}
        for position in range(6):
            selected_rows: list[FullTargetRequest] = []
            seen_items: set[str] = set()
            for row in sample_requests:
                if row.position == position and row.cold_item not in seen_items:
                    selected_rows.append(row)
                    seen_items.add(row.cold_item)
            requests_by_position[position] = selected_rows
        candidate_requests_by_position = {
            position: rows[: int(config["sweep"]["candidate_requests_per_position"])]
            for position, rows in requests_by_position.items()
        }
        candidate_request_manifest = request_subset_manifest(
            candidate_requests_by_position
        )
        position_contract_requests_by_position = {
            position: rows[: configured_position_counts[position]]
            for position, rows in requests_by_position.items()
        }
        position_contract_request_manifest = request_subset_manifest(
            position_contract_requests_by_position
        )
        position_contract_subset_sha256 = request_subset_sha256(
            position_contract_requests_by_position
        )
        if (
            len(candidate_request_manifest)
            != config["sweep"]["candidate_total_cache_miss_requests"]
            or any(
                len({row.cold_item for row in candidate_requests_by_position[position]})
                != config["sweep"]["candidate_requests_per_position"]
                for position in range(6)
            )
        ):
            raise RuntimeError("Candidate request subset lost its distinct-item contract")
        lifecycle_probe = independent_full_lifecycle_probe(
            candidate_requests_by_position[0][0], vector_dimension=2
        )
        candidates: list[dict[str, Any]] = []
        reporter.set("z_batch_sweep", 0, len(config["sweep"]["candidate_request_microbatches"]), "microbatch_candidates")
        for index, size in enumerate(config["sweep"]["candidate_request_microbatches"], 1):
            run = run_z_candidate(
                size=size,
                requests_by_position=candidate_requests_by_position,
                config=config,
                metadata=metadata,
                lexical_paths=lexical_paths,
                tokenizer=tokenizer,
                device=device,
            )
            candidates.append(run)
            reporter.set("z_batch_sweep", index, len(config["sweep"]["candidate_request_microbatches"]), "microbatch_candidates")
        selected = choose_candidate(candidates)
        stage_checkpoint_path = output / "resource_stage_checkpoint.json"
        write_json(
            stage_checkpoint_path,
            {
                "attempt_id": config["attempt_id"],
                "stage": "z_batch_sweep_complete",
                "execution_identity_sha256": execution_identity_sha,
                "candidates": candidates,
                "selected_request_microbatch": selected["microbatch"],
                "automatic_resume": False,
            },
        )

        reporter.set("covariance_resource", 0, 1, "resource_covariance")
        model = load_gram(
            ROOT / config["inputs"]["gram_config"]["path"], checkpoint, device
        ).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        train_rows = read_train_sequences(ROOT / config["inputs"]["train_sequences"]["path"])
        covariance_allocation = {
            int(position): int(count)
            for position, count in config["sweep"]["covariance_rows_by_position"].items()
        }
        position_contract_counts = {
            int(position): int(count)
            for position, count in config["sweep"].get(
                "position_contract_requests_by_position",
                {
                    str(position): config["sweep"]["candidate_requests_per_position"]
                    for position in range(6)
                },
            ).items()
        }
        if set(covariance_allocation) != set(range(6)) or set(
            position_contract_counts
        ) != set(range(6)):
            raise ValueError("Covariance and position contracts must cover positions 0--5")
        if sum(covariance_allocation.values()) != config["sweep"]["covariance_rows"]:
            raise ValueError("Covariance position allocation does not sum to the frozen resource total")
        if int(model.config.d_ff) != int(config["sweep"]["linear_system_width"]):
            raise ValueError("Frozen linear-system width disagrees with the GRAM backbone")
        if any(
            covariance_allocation[position] + position_contract_counts[position]
            < int(model.config.d_ff)
            for position in range(6)
        ):
            raise ValueError("Resource covariance/key rank capacity cannot span the faithful solve")
        covariance_rows = select_covariance_transitions(
            train_rows,
            lexical_paths,
            rows_by_position=covariance_allocation,
            seed=config["seed"],
        )

        def covariance_progress(
            completed_position: int, elapsed: Mapping[int, float]
        ) -> None:
            reporter.set(
                "covariance_resource_positions",
                completed_position + 1,
                6,
                "lexical_positions",
            )
            write_json(
                stage_checkpoint_path,
                {
                    "attempt_id": config["attempt_id"],
                    "stage": "covariance_resource",
                    "execution_identity_sha256": execution_identity_sha,
                    "selected_request_microbatch": selected["microbatch"],
                    "completed_positions": completed_position + 1,
                    "covariance_elapsed_seconds_by_position": elapsed,
                    "covariance_rows_by_position": {
                        str(position): covariance_allocation[position]
                        for position in range(completed_position + 1)
                    },
                    "automatic_resume": False,
                },
            )

        (
            covariance,
            covariance_seconds,
            covariance_activations,
            covariance_seconds_by_position,
        ) = covariance_resource_probe(
            model=model,
            rows_by_position=covariance_rows,
            metadata=metadata,
            lexical_paths=lexical_paths,
            tokenizer=tokenizer,
            device=device,
            batch_size=config["sweep"]["covariance_batch_size"],
            progress_callback=covariance_progress,
        )
        convergence_started = time.perf_counter()
        covariance_convergence = covariance_convergence_diagnostics(
            covariance_activations,
            {
                int(position): tuple(int(value) for value in checkpoints)
                for position, checkpoints in config["sweep"][
                    "resource_covariance_convergence_checkpoints_by_position"
                ].items()
            },
        )
        covariance_convergence_seconds = time.perf_counter() - convergence_started
        write_json(
            stage_checkpoint_path,
            {
                "attempt_id": config["attempt_id"],
                "stage": "covariance_resource_complete",
                "execution_identity_sha256": execution_identity_sha,
                "candidates": candidates,
                "selected_request_microbatch": selected["microbatch"],
                "covariance_rows_by_position": covariance.used_rows_by_position,
                "covariance_elapsed_seconds_by_position": covariance_seconds_by_position,
                "covariance_convergence": covariance_convergence,
                "automatic_resume": False,
            },
        )
        del covariance_activations
        reporter.set("covariance_resource", 1, 1, "resource_covariance")

        reporter.set("trigger_contract", 0, 1, "real_model_contract_probe")
        trigger_contract_started = time.perf_counter()
        trigger_contract = trigger_parity_contract_probe(
            model=model,
            requests_by_position=requests_by_position,
            catalog_ids=catalog_ids,
            metadata=metadata,
            lexical_paths=lexical_paths,
            tokenizer=tokenizer,
            device=device,
        )
        trigger_contract_seconds = time.perf_counter() - trigger_contract_started
        reporter.set("trigger_contract", 1, 1, "real_model_contract_probe")

        reporter.set("generation_resource", 0, 1, "strict_beam_event_pairs")
        generation_resource = generation_resource_probe(
            model=model,
            rows=covariance_rows[0][: config["sweep"]["generation_resource_events"]],
            catalog_ids=catalog_ids,
            metadata=metadata,
            lexical_paths=lexical_paths,
            tokenizer=tokenizer,
            device=device,
            beam_size=config["sweep"]["admission_beam_size"],
        )
        reporter.set("generation_resource", 1, 1, "strict_beam_event_pairs")

        reporter.set("position_contract", 0, 6, "lexical_positions")
        position_metrics: dict[str, Any] = {}
        updates: dict[int, dict[str, torch.Tensor]] = {}
        position_contract_started = time.perf_counter()
        repeated_z_step_seconds = 0.0
        final_z_probe_seconds = 0.0
        post_z_filter_rank_seconds = 0.0
        key_extraction_seconds = 0.0
        system_fixed_setup_seconds = 0.0
        valid_z_matrix_products_seconds = 0.0
        system_formation_seconds = 0.0
        solve_factorization_diagnostics_seconds = 0.0
        solve_diagnostic_seconds = 0.0
        for position in range(6):
            fixed_for_position = position_contract_counts[position]
            chosen = tuple(position_contract_requests_by_position[position])
            if len(chosen) != fixed_for_position:
                raise RuntimeError("Objective-complete position contract lost its fixed request count")
            layer = official_position_to_layer([position])[position]
            runtime = BatchedRealGRAMZRuntime(
                model=model,
                metadata=metadata,
                lexical_paths=lexical_paths,
                tokenizer=tokenizer,
                layer=layer,
                device=device,
            )
            result = optimize_z_vectors(
                requests=chosen,
                vector_dimension=int(model.config.d_model),
                forward_batch=runtime,
                config=ZOptimizationConfig(
                    v_lr=config["frozen_workload"]["z_learning_rate"],
                    v_num_grad_steps=config["frozen_workload"]["z_steps"],
                    v_weight_decay=config["frozen_workload"]["z_weight_decay"],
                    z_vector_max=config["frozen_workload"]["z_vector_max"],
                    batch_size=int(selected["microbatch"]),
                ),
                cache_hits={},
                device=device,
            )
            position_z_step_seconds = sum(
                sum(trace) for trace in result.step_elapsed_seconds_by_batch
            )
            repeated_z_step_seconds += position_z_step_seconds
            del runtime
            final_probe_started = time.perf_counter()
            last_logits = probe_final_z_logits(
                model=model,
                requests=chosen,
                result=result,
                metadata=metadata,
                lexical_paths=lexical_paths,
                tokenizer=tokenizer,
                layer=layer,
                device=device,
                batch_size=int(selected["microbatch"]),
            )
            position_final_probe_seconds = time.perf_counter() - final_probe_started
            final_z_probe_seconds += position_final_probe_seconds
            post_z_started = time.perf_counter()
            probabilities: list[float] = []
            ranks: list[int] = []
            full_vocabulary_ranks: list[int] = []
            for row_index, request in enumerate(chosen):
                logits = last_logits[row_index].float()
                probabilities.append(float(torch.softmax(logits, dim=-1)[request.target_token_id]))
                legal = logits[torch.tensor(request.legal_token_ids)]
                target_value = logits[request.target_token_id]
                ranks.append(1 + int((legal > target_value).sum().item()))
                vocabulary_order = torch.arange(logits.numel())
                full_vocabulary_ranks.append(
                    1
                    + int((logits > target_value).sum().item())
                    + int(
                        ((logits == target_value) & (vocabulary_order < request.target_token_id))
                        .sum()
                        .item()
                    )
                )
            valid = filter_valid_z(result.z_vectors, result.delta_vectors)
            torch.cuda.synchronize(device)
            position_post_z_seconds = time.perf_counter() - post_z_started
            post_z_filter_rank_seconds += position_post_z_seconds
            position_metrics[str(position)] = {
                "request_count": len(chosen),
                "cache_hit_count": 0,
                "valid_z_count": result.valid_count,
                "failed_z_count": result.failed_count,
                "full_vocabulary_target_probabilities": probabilities,
                "legal_target_ranks": ranks,
                "full_vocabulary_target_ranks": full_vocabulary_ranks,
                "diagnostic_logit_semantics": (
                    "valid rows re-probed with satisfaction-time delta; failed rows "
                    "re-probed with terminal optimizer delta"
                ),
                "z_objective_step_seconds": position_z_step_seconds,
                "final_z_reprobe_seconds": position_final_probe_seconds,
                "post_z_filter_rank_diagnostics_seconds": position_post_z_seconds,
                "key_extraction_batch_size": int(selected["microbatch"]),
                "key_extraction_layer": layer,
                "lifecycle_check_steps_by_batch": [
                    list(trace) for trace in result.lifecycle_check_steps_by_batch
                ],
            }
            if valid.valid_count:
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
                    batch_size=int(selected["microbatch"]),
                )
                position_key_seconds = time.perf_counter() - key_started
                key_extraction_seconds += position_key_seconds
                fixed_covariance_started = time.perf_counter()
                covariance_for_solve = covariance.covariance_by_position[position].to(
                    device
                )
                covariance64 = covariance_for_solve.double()
                scaled_covariance64 = prepare_weight_delta_covariance(
                    covariance=covariance64,
                    key_width=int(model.config.d_ff),
                    covariance_lambda=config["frozen_workload"]["cov_lambda"],
                )
                torch.cuda.synchronize(device)
                position_system_fixed_seconds = (
                    time.perf_counter() - fixed_covariance_started
                )
                matrix_products_started = time.perf_counter()
                valid_keys = keys_all[list(valid.valid_indices)].to(device)
                residuals = torch.stack(valid.delta_vectors).to(device)
                key64 = valid_keys.double()
                residual64 = residuals.double()
                key_gram, rhs = form_weight_delta_request_products(
                    residuals=residual64,
                    keys=key64,
                )
                torch.cuda.synchronize(device)
                position_matrix_products_seconds = (
                    time.perf_counter() - matrix_products_started
                )
                fixed_assembly_started = time.perf_counter()
                unregularized_system = key_gram + scaled_covariance64
                torch.cuda.synchronize(device)
                position_system_fixed_seconds += (
                    time.perf_counter() - fixed_assembly_started
                )
                position_system_formation_seconds = (
                    position_system_fixed_seconds
                    + position_matrix_products_seconds
                )
                system_fixed_setup_seconds += position_system_fixed_seconds
                valid_z_matrix_products_seconds += position_matrix_products_seconds
                system_formation_seconds += position_system_formation_seconds
                factorization_started = time.perf_counter()
                _, covariance_cholesky_info = torch.linalg.cholesky_ex(
                    covariance64
                )
                if int(covariance_cholesky_info.item()) == 0:
                    covariance_rank = int(covariance64.shape[0])
                    covariance_rank_method = "cholesky_full_rank"
                    covariance_tolerance_value = None
                else:
                    covariance_eigenvalues = torch.linalg.eigvalsh(covariance64)
                    covariance_tolerance = (
                        covariance64.shape[0]
                        * torch.finfo(covariance64.dtype).eps
                        * covariance_eigenvalues.abs().max()
                    )
                    covariance_rank = int(
                        (covariance_eigenvalues.abs() > covariance_tolerance)
                        .sum()
                        .item()
                    )
                    covariance_rank_method = "symmetric_eigenvalue_tolerance"
                    covariance_tolerance_value = float(covariance_tolerance)
                unregularized_system_eigenvalues = torch.linalg.eigvalsh(
                    unregularized_system
                )
                ridge_diagnostics: dict[str, Any] = {}
                if ridge_enabled:
                    system, ridge_result = form_condition_targeted_ridge_system(
                        system=unregularized_system,
                        eigenvalues=unregularized_system_eigenvalues,
                        target_condition=float(
                            gridge_method["target_condition_number"]
                        ),
                        safety_margin=float(gridge_method["ridge_safety_margin"]),
                    )
                    system_eigenvalues = (
                        unregularized_system_eigenvalues + ridge_result.ridge_value
                    )
                    _, regularized_cholesky_info = torch.linalg.cholesky_ex(system)
                    ridge_diagnostics = {
                        **ridge_result.as_dict(),
                        "regularized_system_cholesky_info": int(
                            regularized_cholesky_info.item()
                        ),
                    }
                else:
                    system = unregularized_system
                    system_eigenvalues = unregularized_system_eigenvalues
                system_tolerance = (
                    system.shape[0]
                    * torch.finfo(system.dtype).eps
                    * system_eigenvalues.abs().max()
                )
                system_rank = int(
                    (system_eigenvalues.abs() > system_tolerance).sum().item()
                )
                positive_system = system_eigenvalues.abs()
                system_min_abs_eigenvalue = float(positive_system.min())
                system_max_abs_eigenvalue = float(positive_system.max())
                system_condition = float(
                    positive_system.max()
                    / positive_system.clamp_min(torch.finfo(system.dtype).tiny).min()
                )
                key_gram_eigenvalues = torch.linalg.eigvalsh(key_gram)
                key_rank_tolerance = (
                    key_gram.shape[0]
                    * torch.finfo(key_gram.dtype).eps
                    * key_gram_eigenvalues.abs().max()
                )
                key_rank = int(
                    (key_gram_eigenvalues.abs() > key_rank_tolerance).sum().item()
                )
                spectral_diagnostics = {
                    "covariance_rank": covariance_rank,
                    "covariance_rank_method": covariance_rank_method,
                    "covariance_rank_tolerance": covariance_tolerance_value,
                    "covariance_cholesky_info": int(covariance_cholesky_info.item()),
                    "valid_key_rank": key_rank,
                    "valid_key_rank_method": "symmetric_key_gram_eigenvalue_tolerance",
                    "valid_key_rank_tolerance": float(key_rank_tolerance),
                    "system_rank": system_rank,
                    "system_rank_tolerance": float(system_tolerance),
                    "system_min_abs_eigenvalue": system_min_abs_eigenvalue,
                    "system_max_abs_eigenvalue": system_max_abs_eigenvalue,
                    "system_condition": system_condition,
                    "rank_tolerance_rule": config["sweep"]["rank_tolerance_rule"],
                    "method_name": (
                        GRIDGE_METHOD_NAME if ridge_enabled else "G-FULL"
                    ),
                    "method_family": (
                        "GenRecEdit-inspired" if ridge_enabled else "GenRecEdit-faithful"
                    ),
                    "faithful_reproduction": not ridge_enabled,
                    "solve_variant": (
                        GRIDGE_SOLVE_VARIANT if ridge_enabled else "faithful_no_ridge"
                    ),
                    "ridge_added": ridge_enabled,
                    "pseudoinverse_used": False,
                    "jitter_fallback_used": False,
                    "outcome_resampling_used": False,
                    **ridge_diagnostics,
                }
                try:
                    if ridge_enabled:
                        delta = solve_condition_targeted_ridge_system(
                            system=system,
                            rhs=rhs,
                            output_like=residuals,
                        )
                    else:
                        delta = solve_weight_delta_system(
                            system=system,
                            rhs=rhs,
                            output_like=residuals,
                        )
                except ValueError as error:
                    position_factorization_seconds = (
                        time.perf_counter() - factorization_started
                    )
                    solve_factorization_diagnostics_seconds += (
                        position_factorization_seconds
                    )
                    position_solve_seconds = (
                        position_system_formation_seconds
                        + position_factorization_seconds
                    )
                    solve_diagnostic_seconds += position_solve_seconds
                    position_metrics[str(position)].update(
                        {
                            "key_extraction_seconds": position_key_seconds,
                            "solve_diagnostic_seconds": position_solve_seconds,
                            "system_fixed_setup_seconds": position_system_fixed_seconds,
                            "valid_z_matrix_products_seconds": position_matrix_products_seconds,
                            "system_formation_seconds": position_system_formation_seconds,
                            "solve_factorization_diagnostics_seconds": position_factorization_seconds,
                            "solve_completed": False,
                            "solve_error": str(error),
                            **spectral_diagnostics,
                        }
                    )
                else:
                    updates[position] = {edited_parameter_name(layer): delta}
                    relative_residual = float(
                        torch.linalg.vector_norm(delta.double() @ system - rhs)
                        / torch.linalg.vector_norm(rhs).clamp_min(1e-30)
                    )
                    position_metrics[str(position)]["delta_norm"] = float(delta.double().norm())
                    position_metrics[str(position)]["delta_rank"] = int(torch.linalg.matrix_rank(delta.double()))
                    position_factorization_seconds = (
                        time.perf_counter() - factorization_started
                    )
                    solve_factorization_diagnostics_seconds += (
                        position_factorization_seconds
                    )
                    position_solve_seconds = (
                        position_system_formation_seconds
                        + position_factorization_seconds
                    )
                    solve_diagnostic_seconds += position_solve_seconds
                    position_metrics[str(position)].update(
                        {
                            "key_extraction_seconds": position_key_seconds,
                            "solve_diagnostic_seconds": position_solve_seconds,
                            "system_fixed_setup_seconds": position_system_fixed_seconds,
                            "valid_z_matrix_products_seconds": position_matrix_products_seconds,
                            "system_formation_seconds": position_system_formation_seconds,
                            "solve_factorization_diagnostics_seconds": position_factorization_seconds,
                            "solve_completed": True,
                            **spectral_diagnostics,
                            "solve_relative_residual": relative_residual,
                        }
                    )
            del result, last_logits
            write_json(
                stage_checkpoint_path,
                {
                    "attempt_id": config["attempt_id"],
                    "stage": "position_contract",
                    "execution_identity_sha256": execution_identity_sha,
                    "selected_request_microbatch": selected["microbatch"],
                    "completed_positions": position + 1,
                    "position_diagnostics": position_metrics,
                    "automatic_resume": False,
                },
            )
            reporter.set("position_contract", position + 1, 6, "lexical_positions")

        aggregated = aggregate_updates(updates) if updates else {}
        solve_status = solve_status_label(position_metrics, aggregated)
        actual_trigger_exercised = False
        parity_evidence: dict[str, Any] = {}
        trigger_rows_by_position: dict[str, int] = {}
        if aggregated:
            full_position_map = official_position_to_layer(range(6))
            live_position_map = {
                position: layer
                for position, layer in full_position_map.items()
                if edited_parameter_name(layer) in aggregated
            }
            live_bundles = build_one_one_position_bundles(
                position_to_layer=live_position_map,
                aggregated_updates={
                    name: delta.to(device) for name, delta in aggregated.items()
                },
            )
            base_snapshot = snapshot_base_parameters(model, sorted(aggregated))
            trigger_position = min(live_position_map)
            trigger_request = requests_by_position[trigger_position][0]
            trigger_context, _ = tokenize_passage_batch(
                request_rows([trigger_request]), metadata, lexical_paths, tokenizer, device
            )
            trigger_decoder = decoder_ids(
                [trigger_request], int(model.config.decoder_start_token_id), device
            )
            with OneOneGenerationDeltaContext(
                model=model,
                deltas_by_position=live_bundles,
                position_to_layer=live_position_map,
                encoded_catalog_paths=catalog_ids.values(),
                decoder_start_token_id=int(model.config.decoder_start_token_id),
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
            ) as delta_context:
                model.prepare_inputs_for_generation(trigger_decoder)
                with torch.no_grad():
                    model(
                        input_ids=trigger_context["input_ids"],
                        attention_mask=trigger_context["attention_mask"],
                        decoder_input_ids=trigger_decoder,
                        use_cache=False,
                        return_dict=True,
                    )
                trigger_rows_by_position = {
                    str(position): count
                    for position, count in delta_context.applied_rows_by_position.items()
                }
            parity_evidence = assert_base_parameter_parity(model, base_snapshot)
            actual_trigger_exercised = trigger_rows_by_position.get(str(trigger_position), 0) > 0
        position_contract_seconds = time.perf_counter() - position_contract_started
        # Isolated cache probe only: the pinned official primary has no cache
        # population path, so formal cache hits remain exactly zero.
        cache_logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
        isolated_cache = probe_cached_z(
            cache_logits,
            target_token_id=0,
            legal_token_ids=(0, 1),
            probability_threshold=0.3,
        )
        maximum_peak_allocated = max(
            [row["peak_allocated_mib"] for row in candidates]
            + [torch.cuda.max_memory_allocated(device) / 1024**2]
        )
        maximum_peak_reserved = max(
            [row["peak_reserved_mib"] for row in candidates]
            + [torch.cuda.max_memory_reserved(device) / 1024**2]
        )
        counts_by_position = {
            position: sum(
                10 for item in cold_items if len(catalog_ids[item]) > position
            )
            for position in range(6)
        }
        z_core_seconds = (
            sum(counts_by_position.values())
            * config["frozen_workload"]["z_steps"]
            / float(selected["steady_request_steps_per_second"])
        )
        covariance_formal_counts = {
            int(position): int(count)
            for position, count in data_manifest["covariance"]["position_counts"].items()
        }
        covariance_formal_seconds = sum(
            covariance_seconds_by_position[position]
            * covariance_formal_counts[position]
            / covariance_allocation[position]
            for position in range(6)
        )
        resource_convergence_checkpoints = {
            int(position): tuple(int(value) for value in checkpoints)
            for position, checkpoints in config["sweep"][
                "resource_covariance_convergence_checkpoints_by_position"
            ].items()
        }
        formal_convergence_checkpoints = {
            position: tuple(config["sweep"]["formal_covariance_convergence_checkpoints"])
            for position in range(6)
        }
        resource_convergence_row_equivalents = convergence_row_equivalents(
            resource_convergence_checkpoints, covariance_allocation
        )
        formal_convergence_row_equivalents = convergence_row_equivalents(
            formal_convergence_checkpoints, covariance_formal_counts
        )
        covariance_convergence_formal_seconds = (
            covariance_convergence_seconds
            * formal_convergence_row_equivalents
            / resource_convergence_row_equivalents
        )
        resource_valid = sum(
            row["valid_z_count"] for row in position_metrics.values()
        )
        resource_requests = sum(
            row["request_count"] for row in position_metrics.values()
        )
        solved_positions = sum(
            row.get("solve_completed") is True for row in position_metrics.values()
        )
        valid_positions = sum(
            int(row["valid_z_count"]) > 0 for row in position_metrics.values()
        )
        projection_objective_complete = (
            resource_valid > 0
            and solved_positions == valid_positions
            and valid_positions == 6
        )
        projected_valid_by_position = {
            position: counts_by_position[position]
            * position_metrics[str(position)]["valid_z_count"]
            / position_metrics[str(position)]["request_count"]
            for position in range(6)
        }
        projected_valid = sum(projected_valid_by_position.values())
        projected_key_seconds = sum(
            position_metrics[str(position)].get("key_extraction_seconds", 0.0)
            * counts_by_position[position]
            / position_metrics[str(position)]["request_count"]
            for position in range(6)
        )
        projected_post_z_seconds = sum(
            position_metrics[str(position)][
                "post_z_filter_rank_diagnostics_seconds"
            ]
            * counts_by_position[position]
            / position_metrics[str(position)]["request_count"]
            for position in range(6)
        )
        projected_matrix_products_seconds = sum(
            position_metrics[str(position)].get(
                "valid_z_matrix_products_seconds", 0.0
            )
            * projected_valid_by_position[position]
            / position_metrics[str(position)]["valid_z_count"]
            if position_metrics[str(position)]["valid_z_count"]
            else math.inf
            for position in range(6)
        )
        projected_system_fixed_seconds = (
            system_fixed_setup_seconds * 6 / valid_positions
            if valid_positions
            else math.inf
        )
        projected_solve_factorization_seconds = (
            solve_factorization_diagnostics_seconds * 6 / solved_positions
            if solved_positions
            else math.inf
        )
        projected_final_z_reprobe_seconds = sum(
            position_metrics[str(position)]["final_z_reprobe_seconds"]
            * counts_by_position[position]
            / position_metrics[str(position)]["request_count"]
            for position in range(6)
        )
        admission_seconds = (
            generation_resource["edited_seconds_per_event"]
            * config["sweep"]["formal_item_disjoint_admission_events"]
        )
        warm_preservation_seconds = (
            generation_resource["base_plus_edited_seconds_per_event"]
            * config["sweep"]["formal_warm_preservation_events"]
        )
        fixed_trigger_seconds = trigger_contract_seconds + max(
            0.0,
            position_contract_seconds
            - repeated_z_step_seconds
            - final_z_probe_seconds
            - post_z_filter_rank_seconds
            - key_extraction_seconds
            - solve_diagnostic_seconds,
        )
        core_seconds = (
            context_build_seconds
            + z_core_seconds
            + covariance_formal_seconds
            + covariance_convergence_formal_seconds
            + projected_final_z_reprobe_seconds
            + projected_post_z_seconds
            + projected_key_seconds
            + projected_matrix_products_seconds
            + projected_system_fixed_seconds
            + projected_solve_factorization_seconds
            + fixed_trigger_seconds
            + admission_seconds
            + warm_preservation_seconds
        )
        lower_seconds = core_seconds * config["sweep"]["runtime_projection_lower_multiplier"]
        upper_seconds = core_seconds * config["sweep"]["runtime_projection_upper_multiplier"]
        admission = max(
            8192,
            ceil_to_1024(
                maximum_peak_reserved
                + max(4096.0, 0.5 * maximum_peak_reserved)
            ),
        )
        full_universe = {
            "edit_targets": data_manifest["counts"]["targets"],
            "contexts": data_manifest["counts"]["contexts"],
            "prefix_next_token_requests": data_manifest["counts"]["requests"],
            "covariance_rows": data_manifest["train_transitions"],
            "request_counts_by_position": {str(k): v for k, v in counts_by_position.items()},
            "covariance_counts_by_position": data_manifest["covariance"]["position_counts"],
        }
        if ridge_enabled:
            solve_contract_key = "inspired_ridge_solve_completed_for_every_valid_position"
            solve_contract_pass = all(
                row["valid_z_count"] > 0
                and row.get("solve_completed") is True
                and row.get("method_name") == GRIDGE_METHOD_NAME
                and row.get("method_family") == "GenRecEdit-inspired"
                and row.get("faithful_reproduction") is False
                and row.get("solve_variant") == GRIDGE_SOLVE_VARIANT
                and row.get("ridge_added") is True
                and isinstance(row.get("ridge_value"), (int, float))
                and math.isfinite(float(row["ridge_value"]))
                and float(row["ridge_value"]) > 0.0
                and row.get("ridge_rule") == GRIDGE_RIDGE_RULE
                and row.get("target_condition")
                == gridge_method["target_condition_number"]
                and row.get("safety_margin") == gridge_method["ridge_safety_margin"]
                and row.get("regularized_rank")
                == config["sweep"]["linear_system_width"]
                and row.get("regularized_nullity") == 0
                and row.get("system_rank")
                == config["sweep"]["linear_system_width"]
                and row.get("regularized_system_cholesky_info") == 0
                and isinstance(row.get("regularized_condition"), (int, float))
                and math.isfinite(float(row["regularized_condition"]))
                and float(row["regularized_condition"])
                <= float(gridge_method["target_condition_number"]) * (1.0 + 1e-9)
                and row.get("system_min_abs_eigenvalue", 0.0) > 0.0
                and row.get("solve_relative_residual", math.inf)
                <= config["sweep"]["maximum_solve_relative_residual"]
                and row.get("rank_tolerance_rule")
                == config["sweep"]["rank_tolerance_rule"]
                and row.get("pseudoinverse_used") is False
                and row.get("jitter_fallback_used") is False
                and row.get("outcome_resampling_used") is False
                for row in position_metrics.values()
            )
        else:
            solve_contract_key = "faithful_solve_completed_for_every_valid_position"
            solve_contract_pass = all(
                row["valid_z_count"] > 0
                and row.get("solve_completed") is True
                and row.get("system_rank") == config["sweep"]["linear_system_width"]
                and row.get("covariance_rank", 0) + row.get("valid_key_rank", 0)
                >= config["sweep"]["linear_system_width"]
                and row.get("valid_key_rank_method")
                == "symmetric_key_gram_eigenvalue_tolerance"
                and isinstance(row.get("valid_key_rank_tolerance"), (int, float))
                and math.isfinite(float(row["valid_key_rank_tolerance"]))
                and float(row["valid_key_rank_tolerance"]) >= 0.0
                and isinstance(row.get("system_condition"), (int, float))
                and math.isfinite(float(row["system_condition"]))
                and row.get("system_min_abs_eigenvalue", 0.0) > 0.0
                and row.get("solve_relative_residual", math.inf)
                <= config["sweep"]["maximum_solve_relative_residual"]
                and row.get("rank_tolerance_rule")
                == config["sweep"]["rank_tolerance_rule"]
                and row.get("ridge_added") is False
                and row.get("pseudoinverse_used") is False
                and row.get("jitter_fallback_used") is False
                and row.get("outcome_resampling_used") is False
                for row in position_metrics.values()
            )
        contract_checks = {
            "full_universe_counts_match": full_universe["edit_targets"] == 5963
            and full_universe["contexts"] == 59630
            and full_universe["prefix_next_token_requests"] == 302400
            and full_universe["covariance_rows"] == 27659,
            "train_only_zero_leakage": all(
                value == 0
                for key, value in data_manifest["leakage_audit"].items()
                if key.endswith("_opened") or key.endswith("_occurrences")
            )
            and data_manifest["leakage_audit"][
                "target_selection_uses_validation_or_test_occurrence"
            ]
            is False,
            "all_candidate_semantics_pass": all(
                all(row["semantic_checks"].values()) for row in candidates
            ),
            "independent_full_lifecycle_probe_pass": lifecycle_probe["pass"],
            "candidate_workload_identical": len(
                {row["candidate_subset_sha256"] for row in candidates}
            )
            == 1
            and all(
                row["candidate_request_count"]
                == config["sweep"]["candidate_total_cache_miss_requests"]
                for row in candidates
            ),
            "all_positions_exercised": set(position_metrics) == {str(i) for i in range(6)},
            "valid_failed_counts_complete": all(
                row["valid_z_count"] + row["failed_z_count"] == row["request_count"]
                for row in position_metrics.values()
            ),
            "position_contract_workload_exact": all(
                row["request_count"] == position_contract_counts[int(position)]
                for position, row in position_metrics.items()
            )
            and all(
                len(
                    {
                        request.cold_item
                        for request in position_contract_requests_by_position[position]
                    }
                )
                == position_contract_counts[position]
                for position in range(6)
            ),
            "covariance_position_coverage_exact": data_manifest["covariance"]["position_counts"]
            == {"0": 27659, "1": 27659, "2": 27659, "3": 27659, "4": 27659, "5": 2036},
            "long_position_resource_rows_present": covariance.used_rows_by_position[5]
            >= config["sweep"]["covariance_long_path_minimum"],
            "covariance_resource_allocation_exact": covariance.used_rows_by_position
            == covariance_allocation,
            "covariance_resource_rank_rule_exact": all(
                covariance_allocation[position]
                == min(
                    int(data_manifest["covariance"]["position_counts"][str(position)]),
                    2 * int(config["sweep"]["linear_system_width"]),
                )
                for position in range(6)
            )
            and config["sweep"]["covariance_resource_rule"]
            == "min(formal_position_rows, 2 * linear_system_width), with deterministic seed-ranked train-only rows",
            "covariance_convergence_report_complete": set(covariance_convergence)
            == {str(i) for i in range(6)}
            and all(rows[-1]["relative_frobenius_drift_to_largest_resource_checkpoint"] == 0.0
                    for rows in covariance_convergence.values()),
            "formal_cache_empty": all(row["cache_hit_count"] == 0 for row in position_metrics.values()),
            "key_extraction_engineering_contract_exact": config["sweep"].get(
                "key_extraction_batch_policy"
            )
            == "selected_z_microbatch"
            and config["sweep"].get("key_extraction_layer_policy")
            == "position_selected_layer_only_output_equivalent_to_unused_official_key_bank_elision"
            and all(
                row.get("key_extraction_batch_size") == int(selected["microbatch"])
                and row.get("key_extraction_layer")
                == official_position_to_layer([int(position)])[int(position)]
                for position, row in position_metrics.items()
            ),
            "isolated_cache_probe_pass": isolated_cache.cache_hit,
            "all_position_trigger_parity_contract_pass": trigger_contract["pass"],
            "strict_generation_resource_path_pass": generation_resource["pass"],
            "valid_z_filter_complete": all(
                row["valid_z_count"] + row["failed_z_count"] == row["request_count"]
                for row in position_metrics.values()
            ),
            "solve_aggregate_trigger_exercised_if_valid": (
                not any(row["valid_z_count"] for row in position_metrics.values())
                or (bool(updates) and bool(aggregated) and actual_trigger_exercised)
            ),
            solve_contract_key: solve_contract_pass,
            "base_parameter_parity_after_trigger": not aggregated
            or parity_evidence.get("exact") is True,
            "base_checkpoint_unchanged": sha256(checkpoint) == checkpoint_before,
            "peak_within_resource_attempt_cap": maximum_peak_reserved
            <= config["sweep"].get(
                "maximum_resource_peak_reserved_mib",
                config["sweep"]["maximum_eligible_peak_reserved_mib"],
            ),
            "fixed_gpu_resource_contract_exact": (
                config["resources"].get("fixed_physical_gpu") is None
                or int(args.physical_gpu)
                == int(config["resources"]["fixed_physical_gpu"])
            )
            and int(args.admission_free_mib)
            >= int(config["resources"]["minimum_free_mib"])
            and int(free_at_worker)
            >= int(config["resources"]["minimum_free_mib"])
            and int(args.worker_hard_timeout_seconds)
            == int(config["resources"]["hard_timeout_seconds"])
            and int(args.expected_peak_mib)
            == int(config["resources"].get("expected_peak_mib", 8192))
            == int(
                config["sweep"].get(
                    "maximum_resource_peak_reserved_mib",
                    config["sweep"]["maximum_eligible_peak_reserved_mib"],
                )
            ),
            "formal_projection_objective_complete": projection_objective_complete
            and math.isfinite(core_seconds),
        }
        valid_z_blocked = any(
            row["valid_z_count"] == 0 for row in position_metrics.values()
        )
        linear_system_blocked = (
            not valid_z_blocked
            and not contract_checks[solve_contract_key]
        )
        pass_raw_verdict = (
            "PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP_RAW"
            if ridge_enabled
            else "PASS_S16_3_GFULL_OBJECTIVE_RESOURCE_SWEEP_RAW"
        )
        linear_blocked_verdict = (
            "RESOURCE_BLOCKED_INSPIRED_RIDGE_LINEAR_SYSTEM"
            if ridge_enabled
            else "RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM"
        )
        valid_z_blocked_verdict = (
            "RESOURCE_BLOCKED_INSPIRED_VALID_Z"
            if ridge_enabled
            else "RESOURCE_BLOCKED_FAITHFUL_VALID_Z"
        )
        failure_verdict = (
            "FAIL_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP_RAW"
            if ridge_enabled
            else "FAIL_S16_3_GFULL_OBJECTIVE_RESOURCE_SWEEP_RAW"
        )
        raw = {
            "schema_version": config["schema_version"],
            "experiment_id": config["experiment_id"],
            "attempt_id": config["attempt_id"],
            "verdict": (
                pass_raw_verdict
                if all(contract_checks.values())
                else (
                    linear_blocked_verdict
                    if linear_system_blocked
                    else (
                        valid_z_blocked_verdict
                        if valid_z_blocked
                        else failure_verdict
                    )
                )
            ),
            "generated_at_utc": utc_now(),
            "physical_gpu": args.physical_gpu,
            "visible_gpu": 0,
            "admission_free_mib": args.admission_free_mib,
            "worker_readmission_free_mib": free_at_worker,
            "admission_util_percent": args.admission_util_percent,
            "resource_attempt_hard_timeout_seconds": args.worker_hard_timeout_seconds,
            "resource_attempt_expected_peak_mib": args.expected_peak_mib,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_peak_allocated_mib": maximum_peak_allocated,
            "maximum_peak_reserved_mib": maximum_peak_reserved,
            "z_steps_per_candidate": 30,
            "candidates": candidates,
            "independent_full_lifecycle_probe": lifecycle_probe,
            "selected_request_microbatch": selected["microbatch"],
            "selected_candidate_subset_sha256": selected["candidate_subset_sha256"],
            "candidate_request_manifest": candidate_request_manifest,
            "position_contract_request_manifest": position_contract_request_manifest,
            "position_contract_subset_sha256": position_contract_subset_sha256,
            "request_dataset_artifact": {
                "manifest_path": str((dataset_root / "manifest.json").relative_to(ROOT)),
                "manifest_sha256": sha256(dataset_root / "manifest.json"),
                "dataset_sha256": data_manifest["dataset_sha256"],
                "checkpoint_path": str(
                    (dataset_root / data_manifest["resume_contract"]["checkpoint_manifest"])
                    .relative_to(ROOT)
                ),
                "checkpoint_sha256": data_manifest["resume_contract"][
                    "checkpoint_sha256"
                ],
                "completed_shards": data_manifest["resume_contract"]["completed_shards"],
            },
            "execution_identity": execution_identity,
            "execution_identity_artifact": {
                "path": str(identity_path.relative_to(ROOT)),
                "sha256": execution_identity_sha,
            },
            "s1_resolved_input_contract": s1_resolved_input_contract,
            "selection_rule": config["sweep"]["candidate_selection_rule"],
            "method": (
                dict(config["method"])
                if ridge_enabled
                else {
                    "name": "G-FULL",
                    "family": "GenRecEdit-faithful",
                    "faithful_reproduction": True,
                    "solve_variant": "faithful_no_ridge",
                }
            ),
            "position_diagnostics": position_metrics,
            "aggregated_parameters": sorted(aggregated),
            "solve_status": solve_status,
            "all_position_trigger_parity_contract": trigger_contract,
            "generation_resource_probe": generation_resource,
            "actual_aggregate_trigger_exercised": actual_trigger_exercised,
            "trigger_rows_by_position": trigger_rows_by_position,
            "base_parameter_parity": parity_evidence,
            "covariance_resource": {
                "rows_by_position": {str(key): value for key, value in covariance_allocation.items()},
                "linear_system_width": int(model.config.d_ff),
                "row_selection_rule": config["sweep"]["covariance_resource_rule"],
                "algebraic_rank_capacity_by_position": {
                    str(position): covariance_allocation[position]
                    + position_contract_counts[position]
                    for position in range(6)
                },
                "elapsed_seconds": covariance_seconds,
                "elapsed_seconds_by_position": {
                    str(key): value for key, value in covariance_seconds_by_position.items()
                },
                "convergence_elapsed_seconds": covariance_convergence_seconds,
                "resource_convergence_row_equivalents": resource_convergence_row_equivalents,
                "formal_convergence_row_equivalents": formal_convergence_row_equivalents,
                "convergence": covariance_convergence,
                "formal_convergence_checkpoints": config["sweep"][
                    "formal_covariance_convergence_checkpoints"
                ],
                "primary_formal_estimator": "full train-only raw E[x x^T] moment without ridge",
                "system_solve_variant": (
                    GRIDGE_SOLVE_VARIANT if ridge_enabled else "faithful_no_ridge"
                ),
            },
            "position_contract_seconds": position_contract_seconds,
            "projection_measurements": {
                "context_build_seconds": context_build_seconds,
                "trigger_contract_seconds": trigger_contract_seconds,
                "repeated_z_step_seconds": repeated_z_step_seconds,
                "final_z_probe_seconds": final_z_probe_seconds,
                "post_z_filter_rank_diagnostics_seconds": post_z_filter_rank_seconds,
                "key_extraction_seconds": key_extraction_seconds,
                "solve_diagnostic_seconds": solve_diagnostic_seconds,
                "system_fixed_setup_seconds": system_fixed_setup_seconds,
                "valid_z_matrix_products_seconds": valid_z_matrix_products_seconds,
                "system_formation_seconds": system_formation_seconds,
                "solve_factorization_diagnostics_seconds": solve_factorization_diagnostics_seconds,
            },
            "full_universe": full_universe,
            "formal_projection": {
                "measured_core_seconds": core_seconds,
                "component_seconds": {
                    "full_context_and_request_manifest": context_build_seconds,
                    "full_z_optimization": z_core_seconds,
                    "full_position_covariance": covariance_formal_seconds,
                    "formal_covariance_convergence_diagnostics": covariance_convergence_formal_seconds,
                    "full_final_z_reprobe_diagnostics": projected_final_z_reprobe_seconds,
                    "full_post_z_filter_and_rank_diagnostics": projected_post_z_seconds,
                    "full_request_key_extraction": projected_key_seconds,
                    "projected_valid_z_matrix_products": projected_matrix_products_seconds,
                    "six_position_system_fixed_setup": projected_system_fixed_seconds,
                    "six_position_solve_factorization_and_diagnostics": (
                        projected_solve_factorization_seconds
                    ),
                    "aggregation_and_trigger_contract": fixed_trigger_seconds,
                    "fixed_7435_event_item_disjoint_admission": admission_seconds,
                    "fixed_512_event_warm_preservation_base_plus_edit": warm_preservation_seconds,
                },
                "resource_valid_z_count": resource_valid,
                "resource_request_count": resource_requests,
                "projected_valid_z_count": projected_valid,
                "projected_valid_z_count_by_position": {
                    str(position): value
                    for position, value in projected_valid_by_position.items()
                },
                "key_extraction_batch_policy": config["sweep"][
                    "key_extraction_batch_policy"
                ],
                "key_extraction_layer_policy": config["sweep"][
                    "key_extraction_layer_policy"
                ],
                "key_extraction_batch_size": int(selected["microbatch"]),
                "projection_objective_complete": projection_objective_complete,
                "lower_wall_seconds": lower_seconds,
                "upper_wall_seconds": upper_seconds,
                "lower_gpu_hours": lower_seconds / 3600,
                "upper_gpu_hours": upper_seconds / 3600,
                "minimum_free_mib_per_gpu": admission,
                "expected_peak_reserved_mib_per_gpu": maximum_peak_reserved,
                "gpu_count": 1,
                "cpu_ram_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "disk_reservation_mib": 32768,
                "hard_timeout_seconds": 604800,
            },
            "contract_checks": contract_checks,
            "base_checkpoint_unchanged": sha256(checkpoint) == checkpoint_before,
            "opened_files": opened,
            "declared_external_input_scope": (
                "explicit frozen data/config/source/tokenizer identities only; generated "
                "request shards are output artifacts covered by their manifest SHA; this is "
                "not an OS-level syscall open audit"
            ),
            "tokenizer_provenance": tokenizer_provenance,
            "runtime_provenance": {
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_runtime_version": torch.version.cuda,
                "model_config_sha256": hashlib.sha256(
                    json.dumps(
                        model.config.to_dict(), sort_keys=True, separators=(",", ":"), default=str
                    ).encode()
                ).hexdigest(),
            },
            "scientific_efficacy_metric_produced": False,
            "validation_used": False,
            "test_read": False,
            "automatic_retry": False,
        }
        write_json(raw_path, raw)
        print(raw["verdict"])
        if raw["verdict"] in {
            "RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM",
            "RESOURCE_BLOCKED_INSPIRED_RIDGE_LINEAR_SYSTEM",
        }:
            return 10
        if raw["verdict"] in {
            "RESOURCE_BLOCKED_FAITHFUL_VALID_Z",
            "RESOURCE_BLOCKED_INSPIRED_VALID_Z",
        }:
            return 11
        return 0 if raw["verdict"].startswith("PASS") else 3
    finally:
        reporter.close()


if __name__ == "__main__":
    raise SystemExit(main())
