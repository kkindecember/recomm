#!/usr/bin/env python3
"""Bounded real-GRAM, official-parameter resource sweep for Stage16 G-FULL."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


ROOT = Path(__file__).resolve().parents[3]

from experiment.phase16.protocol.genrecedit_data import (  # noqa: E402
    build_sharded_dataset,
    read_lexical_paths,
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
    official_position_to_layer,
    OneOneGenerationDeltaContext,
    optimize_z_vectors,
    probe_cached_z,
    snapshot_base_parameters,
    solve_weight_delta,
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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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
        for batch_index, (times, batch_count, trace, lrs) in enumerate(
            zip(
                runtime.forward_seconds_by_batch,
                runtime.request_count_by_batch,
                result.lifecycle_check_steps_by_batch,
                result.scheduler_lrs_by_batch,
            )
        ):
            if len(times) < 10:
                raise RuntimeError("Official lifecycle did not execute its first ten steps")
            first_ten_seconds += sum(times[:10])
            first_ten_request_steps += batch_count * 10
            batch_records.append(
                {
                    "position": position,
                    "batch_index": batch_index,
                    "request_count": batch_count,
                    "forward_calls": len(times),
                    "first_ten_forward_seconds": times[:10],
                    "lifecycle_check_steps": list(trace),
                    "scheduler_step_count": len(lrs),
                    "scheduler_lr_first": lrs[0] if lrs else None,
                    "scheduler_lr_last": lrs[-1] if lrs else None,
                    "observed_step_29": tuple(trace) == expected_trace,
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
            "full_30_step_path_observed": bool(batch_records)
            and all(row["observed_step_29"] for row in batch_records),
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
        and peak_reserved <= config["sweep"]["maximum_eligible_peak_reserved_mib"]
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
) -> tuple[Any, float, dict[int, torch.Tensor]]:
    position_layers = official_position_to_layer(range(6))
    values: dict[int, list[torch.Tensor]] = {position: [] for position in range(6)}
    started = time.perf_counter()
    for position, rows in sorted(rows_by_position.items()):
        layer = position_layers[position]
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
            values[position].append(captured[0][:, position, :])
    activations = {position: torch.cat(chunks, dim=0) for position, chunks in values.items()}
    result = collect_covariance(
        activations,
        mom2_n_samples=max(len(rows) for rows in rows_by_position.values()),
    )
    return result, time.perf_counter() - started, activations


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


def ceil_to_1024(value: float) -> int:
    return int(math.ceil(value / 1024.0) * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--admission-free-mib", type=int, required=True)
    parser.add_argument("--admission-util-percent", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.physical_gpu in config["resources"]["excluded_physical_gpus"]:
        raise SystemExit("Refusing an explicitly excluded/reserved physical GPU")
    output = ROOT / config["output_dir"]
    raw_path = output / "resource_sweep_summary.json"
    if raw_path.exists():
        raise SystemExit("Refusing to overwrite an existing S16-3 raw resource summary")
    reporter = ProgressReporter(output / "progress.json")
    reporter.start()
    started = time.perf_counter()
    checkpoint = ROOT / config["inputs"]["gram_checkpoint"]["path"]
    checkpoint_before = sha256(checkpoint)
    try:
        opened = verify_inputs(config)
        opened.extend(
            [
                "experiment/phase16/configs/stage16_s1_data_resource_preflight.json",
                config["inputs"]["official_genrecedit"]["path"],
            ]
        )
        opened = sorted(set(opened))
        reporter.set("context_build", 0, 1, "full_target_dataset")
        inputs, counts, max_history = resolve_stage16_toys_inputs(
            ROOT / "experiment/phase16/configs/stage16_s1_data_resource_preflight.json"
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
        reporter.set("request_manifest", 1, 1, "full_target_dataset")

        free_at_worker = gpu_readmission(
            args.physical_gpu, config["resources"]["minimum_free_mib"]
        )
        device = torch.device("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
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
        sample_items = long_cold[: max(config["sweep"]["candidate_request_microbatches"])]
        sample_requests = build_full_target_requests(
            catalog_paths=catalog_ids,
            cold_paths={item: catalog_ids[item] for item in sample_items},
            pseudo_contexts={item: contexts[item] for item in sample_items},
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
        )
        requests_by_position = {
            position: [row for row in sample_requests if row.position == position]
            for position in range(6)
        }
        candidates: list[dict[str, Any]] = []
        reporter.set("z_batch_sweep", 0, len(config["sweep"]["candidate_request_microbatches"]), "microbatch_candidates")
        for index, size in enumerate(config["sweep"]["candidate_request_microbatches"], 1):
            run = run_z_candidate(
                size=size,
                requests_by_position=requests_by_position,
                config=config,
                metadata=metadata,
                lexical_paths=lexical_paths,
                tokenizer=tokenizer,
                device=device,
            )
            candidates.append(run)
            reporter.set("z_batch_sweep", index, len(config["sweep"]["candidate_request_microbatches"]), "microbatch_candidates")
        selected = choose_candidate(candidates)

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
        if sum(covariance_allocation.values()) != config["sweep"]["covariance_rows"]:
            raise ValueError("Covariance position allocation does not sum to the frozen resource total")
        covariance_rows = select_covariance_transitions(
            train_rows,
            lexical_paths,
            rows_by_position=covariance_allocation,
            seed=config["seed"],
        )
        covariance, covariance_seconds, covariance_activations = covariance_resource_probe(
            model=model,
            rows_by_position=covariance_rows,
            metadata=metadata,
            lexical_paths=lexical_paths,
            tokenizer=tokenizer,
            device=device,
            batch_size=config["sweep"]["covariance_batch_size"],
        )
        covariance_convergence = covariance_convergence_diagnostics(
            covariance_activations,
            {
                int(position): tuple(int(value) for value in checkpoints)
                for position, checkpoints in config["sweep"][
                    "resource_covariance_convergence_checkpoints_by_position"
                ].items()
            },
        )
        del covariance_activations
        reporter.set("covariance_resource", 1, 1, "resource_covariance")

        reporter.set("position_contract", 0, 6, "lexical_positions")
        position_metrics: dict[str, Any] = {}
        updates: dict[int, dict[str, torch.Tensor]] = {}
        position_contract_started = time.perf_counter()
        fixed_per_position = int(config["sweep"]["candidate_requests_per_position"])
        for position in range(6):
            chosen = tuple(requests_by_position[position][:fixed_per_position])
            if len(chosen) != fixed_per_position:
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
            last_logits = runtime.last_logits()
            probabilities: list[float] = []
            ranks: list[int] = []
            for row_index, request in enumerate(chosen):
                logits = last_logits[row_index].float()
                probabilities.append(float(torch.softmax(logits, dim=-1)[request.target_token_id]))
                legal = logits[torch.tensor(request.legal_token_ids)]
                target_value = logits[request.target_token_id]
                ranks.append(1 + int((legal > target_value).sum().item()))
            position_metrics[str(position)] = {
                "request_count": len(chosen),
                "cache_hit_count": 0,
                "valid_z_count": result.valid_count,
                "failed_z_count": result.failed_count,
                "full_vocabulary_target_probabilities": probabilities,
                "legal_target_ranks": ranks,
                "lifecycle_check_steps_by_batch": [
                    list(trace) for trace in result.lifecycle_check_steps_by_batch
                ],
            }
            valid = filter_valid_z(result.z_vectors, result.delta_vectors)
            if valid.valid_count:
                module = model.decoder.block[layer].layer[2].DenseReluDense.wo
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
                valid_keys = keys_all[list(valid.valid_indices)]
                residuals = torch.stack(valid.delta_vectors).cpu()
                delta = solve_weight_delta(
                    residuals=residuals,
                    keys=valid_keys,
                    covariance=covariance.covariance_by_position[position],
                    covariance_lambda=config["frozen_workload"]["cov_lambda"],
                )
                updates[position] = {edited_parameter_name(layer): delta}
                system = valid_keys.double().T @ valid_keys.double() + float(
                    config["frozen_workload"]["cov_lambda"]
                ) * covariance.covariance_by_position[position].double()
                position_metrics[str(position)]["delta_norm"] = float(delta.double().norm())
                position_metrics[str(position)]["delta_rank"] = int(torch.linalg.matrix_rank(delta.double()))
                position_metrics[str(position)]["system_condition"] = float(torch.linalg.cond(system))
            del result, runtime, last_logits
            reporter.set("position_contract", position + 1, 6, "lexical_positions")

        aggregated = aggregate_updates(updates) if updates else {}
        solve_status = (
            "SOLVE_AND_AGGREGATE_EXERCISED"
            if aggregated
            else "NO_VALID_Z_IN_PREREGISTERED_RESOURCE_SUBSET"
        )
        trigger_exercised = False
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
                aggregated_updates=aggregated,
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
            trigger_exercised = trigger_rows_by_position.get(str(trigger_position), 0) > 0
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
        covariance_formal_rows = sum(
            data_manifest["covariance"]["position_counts"].values()
        )
        covariance_scale = covariance_formal_rows / config["sweep"]["covariance_rows"]
        core_seconds = z_core_seconds + covariance_seconds * covariance_scale
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
            "covariance_position_coverage_exact": data_manifest["covariance"]["position_counts"]
            == {"0": 27659, "1": 27659, "2": 27659, "3": 27659, "4": 27659, "5": 2036},
            "long_position_resource_rows_present": covariance.used_rows_by_position[5]
            >= config["sweep"]["covariance_long_path_minimum"],
            "covariance_resource_allocation_exact": covariance.used_rows_by_position
            == covariance_allocation,
            "covariance_convergence_report_complete": set(covariance_convergence)
            == {str(i) for i in range(6)}
            and all(rows[-1]["relative_frobenius_drift_to_largest_resource_checkpoint"] == 0.0
                    for rows in covariance_convergence.values()),
            "formal_cache_empty": all(row["cache_hit_count"] == 0 for row in position_metrics.values()),
            "isolated_cache_probe_pass": isolated_cache.cache_hit,
            "valid_z_filter_complete": all(
                row["valid_z_count"] + row["failed_z_count"] == row["request_count"]
                for row in position_metrics.values()
            ),
            "solve_aggregate_trigger_exercised_if_valid": (
                not any(row["valid_z_count"] for row in position_metrics.values())
                or (bool(updates) and bool(aggregated) and trigger_exercised)
            ),
            "base_parameter_parity_after_trigger": not aggregated
            or parity_evidence.get("exact") is True,
            "base_checkpoint_unchanged": sha256(checkpoint) == checkpoint_before,
            "peak_within_small_experiment_cap": maximum_peak_reserved
            <= config["sweep"]["maximum_eligible_peak_reserved_mib"],
        }
        raw = {
            "schema_version": config["schema_version"],
            "experiment_id": config["experiment_id"],
            "attempt_id": config["attempt_id"],
            "verdict": "PASS_S16_3_GFULL_OBJECTIVE_RESOURCE_SWEEP_RAW"
            if all(contract_checks.values())
            else "FAIL_S16_3_GFULL_OBJECTIVE_RESOURCE_SWEEP_RAW",
            "generated_at_utc": utc_now(),
            "physical_gpu": args.physical_gpu,
            "visible_gpu": 0,
            "admission_free_mib": args.admission_free_mib,
            "worker_readmission_free_mib": free_at_worker,
            "admission_util_percent": args.admission_util_percent,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_peak_allocated_mib": maximum_peak_allocated,
            "maximum_peak_reserved_mib": maximum_peak_reserved,
            "z_steps_per_candidate": 30,
            "candidates": candidates,
            "selected_request_microbatch": selected["microbatch"],
            "selected_candidate_subset_sha256": selected["candidate_subset_sha256"],
            "selection_rule": config["sweep"]["candidate_selection_rule"],
            "position_diagnostics": position_metrics,
            "aggregated_parameters": sorted(aggregated),
            "solve_status": solve_status,
            "trigger_exercised": trigger_exercised,
            "trigger_rows_by_position": trigger_rows_by_position,
            "base_parameter_parity": parity_evidence,
            "covariance_resource": {
                "rows_by_position": {str(key): value for key, value in covariance_allocation.items()},
                "elapsed_seconds": covariance_seconds,
                "convergence": covariance_convergence,
                "formal_convergence_checkpoints": config["sweep"][
                    "formal_covariance_convergence_checkpoints"
                ],
                "primary_formal_estimator": "full train-only raw E[x x^T] moment without ridge",
            },
            "position_contract_seconds": position_contract_seconds,
            "full_universe": full_universe,
            "formal_projection": {
                "measured_core_seconds": core_seconds,
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
            "scientific_efficacy_metric_produced": False,
            "validation_used": False,
            "test_read": False,
            "automatic_retry": False,
        }
        write_json(raw_path, raw)
        print(raw["verdict"])
        return 0 if raw["verdict"].startswith("PASS") else 3
    finally:
        reporter.close()


if __name__ == "__main__":
    raise SystemExit(main())
