#!/usr/bin/env python3
"""Result-parity and peak-memory smoke for S18-1 cache-off generation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase18.core.contracts import load_json, sha256
from experiment.phase18.protocol import s18_s1_recovery as recovery
from experiment.phase18.protocol import s18_s1_runtime as base


REFERENCE = ROOT / "artifacts/phase18/s1_actionability/recovery-smoke-run-0002/units/toys_i0"
STATUS = ROOT / "artifacts/phase18/status/s18_s1_memory_smoke.status.json"


def output_root(
    max_users: int,
    generation_use_cache: bool = False,
    cross_attention_cache: bool = True,
    decoder_model_parallel: bool = False,
) -> Path:
    if not generation_use_cache:
        cache_mode = "cache-off"
    elif cross_attention_cache:
        cache_mode = "cache-on-release"
    else:
        cache_mode = "cache-on-cross-off-release"
    if decoder_model_parallel:
        cache_mode += "-decoder-mp2"
    return ROOT / f"artifacts/phase18/s1_actionability/resource-smoke-run-0003-{cache_mode}-u{max_users}"


def first_tsv_row(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"missing TSV data row: {path}")
    return lines[0].split("\t"), lines[1].split("\t")


def compare_float_list(left: str, right: str, tolerance: float = 1e-6) -> float:
    lhs = np.asarray([float(value) for value in left.split("||")], dtype=np.float64)
    rhs = np.asarray([float(value) for value in right.split("||")], dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"score shape mismatch: {lhs.shape} != {rhs.shape}")
    delta = float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0
    if not math.isfinite(delta) or delta > tolerance:
        raise RuntimeError(f"score mismatch: max_abs_delta={delta} > {tolerance}")
    return delta


def compare_first_user(candidate: Path) -> dict[str, Any]:
    maxima: dict[str, float] = {}
    for name in ("beams_w50.tsv", "beams_w200.tsv"):
        reference_header, reference_row = first_tsv_row(REFERENCE / name)
        candidate_header, candidate_row = first_tsv_row(candidate / name)
        if candidate_header != reference_header:
            raise RuntimeError(f"{name}: header mismatch")
        if candidate_row[:3] != reference_row[:3]:
            raise RuntimeError(f"{name}: user, target, or candidate ordering changed")
        maxima[f"{name}:normalized"] = compare_float_list(candidate_row[3], reference_row[3])
        maxima[f"{name}:raw"] = compare_float_list(candidate_row[4], reference_row[4])

    reference_diag = json.loads(
        (REFERENCE / "per_user_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    candidate_diag = json.loads(
        (candidate / "per_user_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    if candidate_diag != reference_diag:
        raise RuntimeError("first-user diagnostic record changed")
    return {
        "status": "PASSED",
        "candidate_order_exact": True,
        "diagnostic_record_exact": True,
        "score_max_abs_deltas": maxima,
        "score_tolerance": 1e-6,
    }


def update_status(**fields: Any) -> None:
    current = load_json(STATUS) if STATUS.is_file() else {}
    current.update(fields)
    current["updated_at"] = base.utc_now()
    base.atomic_json(STATUS, current)


def run(
    physical_gpu: int,
    max_users: int,
    generation_use_cache: bool,
    cross_attention_cache: bool,
    secondary_physical_gpu: int | None,
) -> int:
    if max_users < 1 or max_users > 1024:
        raise ValueError("max-users must be in [1, 1024]")
    config, authorization = recovery.verify_authorization()
    physical_gpus = [physical_gpu]
    if secondary_physical_gpu is not None:
        physical_gpus.append(secondary_physical_gpu)
    if len(set(physical_gpus)) != len(physical_gpus):
        raise RuntimeError("memory smoke GPUs must be distinct")
    if set(physical_gpus) - {0, 1, 4, 6, 7}:
        raise RuntimeError(f"GPU set {physical_gpus} is outside the researcher-authorized candidate set")
    decoder_model_parallel = secondary_physical_gpu is not None
    target_root = output_root(
        max_users,
        generation_use_cache,
        cross_attention_cache,
        decoder_model_parallel,
    )
    if target_root.exists():
        raise FileExistsError(f"memory smoke output already exists: {target_root}")
    target = target_root / "units" / "toys_i0"
    target.mkdir(parents=True)
    base.OUTPUT = target_root
    base.atomic_json(
        STATUS,
        {
            "schema_version": "phase18.s18_1_memory_smoke_status.v1",
            "experiment_id": "s18_s1_memory_smoke",
            "attempt_id": (
                f"run-0003-cache-on-release-u{max_users}"
                if generation_use_cache
                else f"run-0003-cache-off-u{max_users}"
            ),
            "execution_state": "RUNNING",
            "process_alive": True,
            "pid": os.getpid(),
            "physical_gpus": physical_gpus,
            "max_users": max_users,
            "generation_use_cache": generation_use_cache,
            "cross_attention_cache": cross_attention_cache,
            "release_cuda_cache_per_user": True,
            "scientific_result_eligible": False,
            "started_at": base.utc_now(),
            "updated_at": base.utc_now(),
        },
    )
    started = time.time()
    try:
        base.set_seed(config["seed"])
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        for visible_gpu in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(visible_gpu)
        tokenizer = AutoTokenizer.from_pretrained(
            config["backbone"]["snapshot"], local_files_only=True
        )
        parent, args, item_head, item_to_id, frequencies, sequences, provenance = recovery.load_frozen_models(
            config, authorization, "Toys", "I0", device
        )
        decoder_device_map = (
            base.enable_two_gpu_decoder_parallel(parent)
            if decoder_model_parallel
            else None
        )
        args.tokenizer = tokenizer
        diagnostic = base.diagnose(
            config,
            "Toys",
            "I0",
            device,
            tokenizer,
            parent,
            args,
            item_head,
            item_to_id,
            frequencies,
            sequences,
            max_users=max_users,
            generation_use_cache=generation_use_cache,
            cross_attention_cache=cross_attention_cache,
            release_cuda_cache_per_user=True,
        )
        parity = compare_first_user(target)
        peak_by_visible_gpu = {
            str(index): {
                "allocated_mib": torch.cuda.max_memory_allocated(index) / 1024**2,
                "reserved_mib": torch.cuda.max_memory_reserved(index) / 1024**2,
                "physical_gpu": physical_gpus[index],
            }
            for index in range(torch.cuda.device_count())
        }
        payload = {
            "schema_version": "phase18.s18_1_memory_smoke.v1",
            "status": "PASSED",
            "attempt_id": (
                f"run-0003-cache-on-release-u{max_users}"
                if generation_use_cache
                else f"run-0003-cache-off-u{max_users}"
            ),
            "physical_gpus": physical_gpus,
            "max_users": max_users,
            "generation_use_cache": generation_use_cache,
            "cross_attention_cache": cross_attention_cache,
            "release_cuda_cache_per_user": True,
            "decoder_model_parallel": decoder_model_parallel,
            "decoder_device_map": decoder_device_map,
            "peak_allocated_mib": diagnostic["peak_allocated_mib"],
            "peak_reserved_mib": diagnostic["peak_reserved_mib"],
            "peak_by_visible_gpu": peak_by_visible_gpu,
            "wall_time_seconds": time.time() - started,
            "parity": parity,
            "checkpoint_provenance": provenance,
            "source_runtime_sha256": sha256(Path(base.__file__).resolve()),
            "scientific_parameters_changed": False,
            "scientific_result_eligible": False,
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
        }
        base.atomic_json(target_root / "summary.json", payload)
        update_status(
            execution_state="COMPLETED",
            process_alive=False,
            status="PASSED",
            peak_allocated_mib=diagnostic["peak_allocated_mib"],
            peak_reserved_mib=diagnostic["peak_reserved_mib"],
            wall_time_seconds=time.time() - started,
            summary_path=str((target_root / "summary.json").relative_to(ROOT)),
            summary_sha256=sha256(target_root / "summary.json"),
        )
        print(json.dumps(payload, default=base.json_default))
        return 0
    except Exception as error:
        update_status(
            execution_state="FAILED_NO_RETRY",
            process_alive=False,
            status="FAILED",
            error_type=type(error).__name__,
            error=str(error),
            wall_time_seconds=time.time() - started,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--max-users", type=int, required=True)
    parser.add_argument("--generation-cache", choices=("on", "off"), required=True)
    parser.add_argument("--cross-attention-cache", choices=("on", "off"), required=True)
    parser.add_argument("--secondary-physical-gpu", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(
        run(
            arguments.physical_gpu,
            arguments.max_users,
            generation_use_cache=arguments.generation_cache == "on",
            cross_attention_cache=arguments.cross_attention_cache == "on",
            secondary_physical_gpu=arguments.secondary_physical_gpu,
        )
    )
