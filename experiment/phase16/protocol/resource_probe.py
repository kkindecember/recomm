#!/usr/bin/env python3
"""Bounded one-GPU resource probes for Stage16 formal workloads.

The SpecGR-Aux probe is explicitly a tensor/architecture resource proxy because
the pinned official class depends on RecBole, which is not installed. The GRAM
probes load the frozen Beauty checkpoint and execute real GRAM forwards.
Nothing produced here is a scientific metric.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5Config


ROOT = Path(__file__).resolve().parents[3]
GRAM_SRC = ROOT / "GRAM" / "src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))

from model import create_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class MoEAdaptorProxy(nn.Module):
    """Exact official adaptor tensor algebra, reproduced only for resource sizing."""

    def __init__(self, input_size: int = 1024, output_size: int = 300, experts: int = 8):
        super().__init__()
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_size)) for _ in range(experts)])
        self.experts = nn.ModuleList([nn.Linear(input_size, output_size, bias=False) for _ in range(experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, experts))
        self.w_noise = nn.Parameter(torch.zeros(input_size, experts))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        logits = values @ self.w_gate
        noise_scale = F.softplus(values @ self.w_noise) + 1e-2
        gates = F.softmax(logits + torch.randn_like(logits) * noise_scale, dim=-1)
        outputs = torch.stack(
            [layer(values - bias) for layer, bias in zip(self.experts, self.biases)], dim=-2
        )
        return (gates.unsqueeze(-1) * outputs).sum(dim=-2)


class UniSRecResourceProxy(nn.Module):
    """Official dimensions and dominant operations without unavailable RecBole kernels."""

    def __init__(self):
        super().__init__()
        self.adaptor = MoEAdaptorProxy()
        self.position = nn.Embedding(20, 300)
        layer = nn.TransformerEncoderLayer(
            d_model=300,
            nhead=2,
            dim_feedforward=256,
            dropout=0.5,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(300)

    def forward(self, catalog: torch.Tensor, indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        history = self.adaptor(catalog[indices])
        positions = self.position(torch.arange(indices.shape[1], device=indices.device))[None]
        sequence = self.encoder(self.norm(history + positions))[:, -1]
        all_items = self.adaptor(catalog)
        logits = F.normalize(sequence, dim=-1) @ F.normalize(all_items, dim=-1).T / 0.07
        return F.cross_entropy(logits, labels)


def clear_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def finish_measurement(device: torch.device, started: float) -> dict[str, float]:
    torch.cuda.synchronize(device)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def load_gram(historical_path: Path, checkpoint: Path, device: torch.device) -> nn.Module:
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    config = T5Config.from_pretrained(historical["backbone"], local_files_only=True)
    config.max_seq_len = historical["item_prompt_max_len"]
    config.max_item_num = historical["max_his"]
    config.use_position_embedding = historical["use_position_embedding"]
    config.sample_num = historical["sample_num"]
    config.cf0_arm = "A"
    config.cf0_enabled = False
    config.cf0_num_items = 0
    config.cf0_num_layers = 2
    config.cf0_num_heads = 4
    config.cf0_dropout = 0.1
    config.cf0_loss_weight = 0.1
    config.cf0_injection_scale = 0.1
    config.cf0_joint_score_weight = 0.25
    config.hi_gram_enabled = False
    config.hi_gram_local_window = 5
    config.hi_gram_local_layers = 2
    config.hi_gram_global_layers = 2
    config.hi_gram_num_heads = 4
    config.hi_gram_dropout = 0.1
    config.hi_gram_fusion_scale_init = 0.1
    config.hi_gram_include_user_prompt = False
    model = create_model("gram", config=config)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device)


def synthetic_gram_batch(model: nn.Module, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(1502 + batch_size)
    inputs = torch.randint(2, 1000, (batch_size, 21, 128), generator=generator, device=device)
    masks = torch.ones_like(inputs)
    labels = torch.randint(2, 1000, (batch_size, 9), generator=generator, device=device)
    labels[:, -1] = int(model.config.eos_token_id)
    return {"input_ids": inputs, "attention_mask": masks, "labels": labels}


def probe_saux(embedding_path: Path, device: torch.device) -> dict[str, Any]:
    clear_cuda(device)
    started = time.perf_counter()
    payload = torch.load(embedding_path, map_location="cpu")
    catalog = payload["embeddings"].to(device)
    model = UniSRecResourceProxy().to(device).train()
    batch_size = 64
    generator = torch.Generator(device=device).manual_seed(1502)
    indices = torch.randint(0, catalog.shape[0], (batch_size, 20), generator=generator, device=device)
    labels = torch.randint(0, catalog.shape[0], (batch_size,), generator=generator, device=device)
    loss = model(catalog, indices, labels)
    loss.backward()
    measurement = finish_measurement(device, started)
    result = {
        "name": "S-AUX",
        "execution_class": "RESOURCE_PROXY_NOT_SCIENTIFIC_EXECUTION",
        "fidelity": "Exact official MoE adaptor algebra and official 2-layer/2-head/300d/256-inner dimensions; PyTorch Transformer substitutes unavailable RecBole kernel.",
        "batch_size": batch_size,
        "catalog_shape": list(catalog.shape),
        "loss_finite": bool(torch.isfinite(loss).item()),
        **measurement,
    }
    del loss, labels, indices, model, catalog, payload
    clear_cuda(device)
    return result


def probe_splus(historical: Path, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    clear_cuda(device)
    started = time.perf_counter()
    model = load_gram(historical, checkpoint, device).train()
    batch = synthetic_gram_batch(model, 1, device)
    output = model(**batch, use_cache=False)
    generation_loss = output.loss
    # The official joint objective weights are exercised; this resource probe
    # uses a zero-valued differentiable contrastive placeholder because index
    # construction belongs to S16-2, not this preflight.
    contrastive_placeholder = sum(parameter.reshape(-1)[0] * 0 for parameter in model.parameters())
    loss = 6.0 * contrastive_placeholder + generation_loss
    loss.backward()
    measurement = finish_measurement(device, started)
    result = {
        "name": "S-PLUS",
        "execution_class": "REAL_FROZEN_GRAM_FORWARD_BACKWARD_RESOURCE_PROBE",
        "fidelity": "Real checkpoint, 21x128 encoder passages, 9 decoder tokens, physical microbatch 1; index/contrastive values are deferred.",
        "batch_size": 1,
        "loss_finite": bool(torch.isfinite(loss).item()),
        **measurement,
    }
    del loss, output, generation_loss, contrastive_placeholder, batch, model
    clear_cuda(device)
    return result


def probe_gfull(historical: Path, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    clear_cuda(device)
    started = time.perf_counter()
    model = load_gram(historical, checkpoint, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    batch = synthetic_gram_batch(model, 1, device)
    decoder_ids = torch.full((1, 4), int(model.config.decoder_start_token_id), dtype=torch.long, device=device)
    target = torch.tensor([42], dtype=torch.long, device=device)
    residual = torch.zeros(1, model.config.d_model, device=device, requires_grad=True)
    module = model.decoder.block[3].layer[2].DenseReluDense.wo

    def inject(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        modified = output.clone()
        modified[:, -1, :] = modified[:, -1, :] + residual
        return modified

    handle = module.register_forward_hook(inject)
    optimizer = torch.optim.Adam([residual], lr=0.5)
    try:
        for _ in range(3):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=decoder_ids,
                use_cache=False,
            ).logits[:, -1]
            loss = F.cross_entropy(logits, target) + 0.2 * residual.norm(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                norm = residual.norm().clamp_min(1e-12)
                residual.mul_(min(1.0, 8000.0 / float(norm)))
    finally:
        handle.remove()
    measurement = finish_measurement(device, started)
    result = {
        "name": "G-FULL",
        "execution_class": "REAL_FROZEN_GRAM_Z_OPTIMIZATION_RESOURCE_PROBE",
        "fidelity": "Real checkpoint and decoder FFN-output residual hook at layer 3; 3 of official 30 Adam steps, one request, no weight materialization.",
        "batch_size": 1,
        "probe_z_steps": 3,
        "formal_z_steps": 30,
        "loss_finite": bool(torch.isfinite(loss).item()),
        **measurement,
    }
    del loss, optimizer, residual, target, decoder_ids, batch, model
    clear_cuda(device)
    return result


def formal_resource_freeze(probes: list[dict[str, Any]], counts: dict[str, Any]) -> list[dict[str, Any]]:
    beauty = next(row for row in counts["domains"] if row["domain"] == "Beauty_cold50")
    by_name = {row["name"]: row for row in probes}
    return [
        {
            "workload": "S-AUX",
            "gpu_count": 1,
            "minimum_free_mib_per_gpu": 24576,
            "expected_peak_mib_per_gpu": 20480,
            "formal_timeout_seconds": 172800,
            "disk_reservation_mib": 8192,
            "training_examples": beauty["s_aux_training_examples"],
            "maximum_epochs": 300,
            "early_stopping_patience_epochs": 40,
            "exact_command_template": "CUDA_VISIBLE_DEVICES=<USER_GPU> bash experiment/phase16/run_stage16_s2_saux_formal.sh",
            "estimate_basis": by_name["S-AUX"]["execution_class"],
            "uncertainty": "HIGH_UNTIL_S16_2_NATIVE_MICROBATCH_SWEEP",
        },
        {
            "workload": "S-PLUS",
            "gpu_count": 1,
            "minimum_free_mib_per_gpu": 24576,
            "expected_peak_mib_per_gpu": 20480,
            "formal_timeout_seconds": 259200,
            "disk_reservation_mib": 16384,
            "pretrain_examples": beauty["s_plus_pretrain_examples"],
            "finetune_examples": beauty["s_plus_finetune_examples"],
            "physical_microbatch": 1,
            "gradient_accumulation_required": True,
            "exact_command_template": "CUDA_VISIBLE_DEVICES=<USER_GPU> bash experiment/phase16/run_stage16_s2_splus_formal.sh",
            "estimate_basis": by_name["S-PLUS"]["execution_class"],
            "uncertainty": "HIGH_UNTIL_S16_2_OBJECTIVE_COMPLETE_SWEEP",
        },
        {
            "workload": "G-FULL",
            "gpu_count": 1,
            "minimum_free_mib_per_gpu": 24576,
            "expected_peak_mib_per_gpu": 20480,
            "formal_timeout_seconds": 604800,
            "disk_reservation_mib": 32768,
            "edit_targets": beauty["g_full_edit_targets"],
            "contexts": beauty["g_full_contexts"],
            "prefix_next_token_requests": beauty["g_full_prefix_next_token_requests"],
            "covariance_rows": beauty["g_full_covariance_rows"],
            "z_steps": 30,
            "exact_command_template": "CUDA_VISIBLE_DEVICES=<USER_GPU> bash experiment/phase16/run_stage16_s3_gfull_formal.sh",
            "estimate_basis": by_name["G-FULL"]["execution_class"],
            "uncertainty": "HIGH; FULL REQUEST COUNT MAKES WALL-TIME THE PRIMARY RISK",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--admission-free-mib", type=int, required=True)
    parser.add_argument("--admission-util-percent", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    counts_path = output / "workload_counts.json"
    if not counts_path.is_file():
        raise SystemExit("CPU data preflight must pass before the GPU resource probe")
    if (output / "resource_probe_summary.json").exists():
        raise SystemExit("Refusing to overwrite an existing S16-1 resource probe")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Resource probe requires exactly one CUDA-visible GPU")
    device = torch.device("cuda:0")
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    beauty = next(domain for domain in config["domains"] if domain["name"] == "Beauty_cold50")
    paths = {key: ROOT / spec["path"] for key, spec in beauty.items() if isinstance(spec, dict) and "path" in spec}
    for key, path in paths.items():
        expected = beauty[key]["sha256"]
        if sha256(path) != expected:
            raise SystemExit(f"Frozen Beauty input drift before resource probe: {key}")

    started = time.perf_counter()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    probes = [
        probe_saux(paths["content_embeddings"], device),
        probe_splus(paths["gram_config"], paths["gram_checkpoint"], device),
        probe_gfull(paths["gram_config"], paths["gram_checkpoint"], device),
    ]
    total_seconds = time.perf_counter() - started
    peak = max(row["peak_allocated_mib"] for row in probes)
    cap = config["workload_policy"]["resource_probe_max_incremental_gpu_mib"]
    checks = {
        "single_visible_gpu": torch.cuda.device_count() == 1,
        "all_losses_finite": all(row["loss_finite"] for row in probes),
        "probe_peak_below_cap": peak <= cap,
        "probe_wall_below_10_minutes": total_seconds <= config["workload_policy"]["resource_probe_hard_timeout_seconds"],
    }
    verdict = "PASS_S16_1_RESOURCE_PROBE" if all(checks.values()) else "FAIL_S16_1_RESOURCE_PROBE"
    payload = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "scientific_results_produced": False,
        "test_read": False,
        "network_used": False,
        "physical_gpu": args.physical_gpu,
        "visible_gpu": 0,
        "gpu_name": torch.cuda.get_device_name(device),
        "admission_free_mib": args.admission_free_mib,
        "admission_util_percent": args.admission_util_percent,
        "total_probe_seconds": total_seconds,
        "maximum_peak_allocated_mib": peak,
        "checks": checks,
        "probes": probes,
        "formal_resource_freeze": formal_resource_freeze(probes, counts),
    }
    write_json(output / "resource_probe_summary.json", payload)
    print(verdict)
    return 0 if verdict.startswith("PASS") else 3


if __name__ == "__main__":
    raise SystemExit(main())
