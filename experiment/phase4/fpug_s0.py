#!/usr/bin/env python3
"""FPUG S0: frozen-backbone correctness smoke for bounded detail-passage gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.fpug_n1 import select_samples  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    ROOT,
    build_train_samples,
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)
from utils import generation_trie as gt  # noqa: E402


def masked_passage_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch, passages, width = mask.shape
    shaped = hidden.view(batch, passages, width, hidden.shape[-1])
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (shaped * weights).sum(2) / weights.sum(2).clamp_min(1.0)


class FinePassageGate(nn.Module):
    def __init__(self, hidden_size: int, bound: float) -> None:
        super().__init__()
        self.bound = float(bound)
        self.linear = nn.Linear(hidden_size * 3 + 1, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, passages, width = mask.shape
        pooled = masked_passage_mean(hidden, mask)
        coarse = pooled[:, :1].expand(-1, passages - 1, -1)
        detail = pooled[:, 1:]
        if passages <= 2:
            recency = torch.zeros(
                batch, passages - 1, 1, device=hidden.device, dtype=hidden.dtype
            )
        else:
            values = torch.linspace(
                0.0, 1.0, passages - 1, device=hidden.device, dtype=hidden.dtype
            )
            recency = values.view(1, -1, 1).expand(batch, -1, -1)
        features = torch.cat([coarse, detail, coarse * detail, recency], dim=-1)
        gates = 1.0 + self.bound * torch.tanh(self.linear(features)).squeeze(-1)
        shaped = hidden.view(batch, passages, width, hidden.shape[-1])
        gated = torch.cat(
            [
                shaped[:, :1],
                shaped[:, 1:] * gates[:, :, None, None],
            ],
            dim=1,
        )
        return gated.reshape_as(hidden), gates


@torch.no_grad()
def decoder_logits(backbone, hidden, flat_attention, labels):
    return backbone(
        input_ids=None,
        attention_mask=flat_attention,
        encoder_outputs=(hidden,),
        labels=labels,
        return_dict=True,
    ).logits


def run_dataset(
    dataset: str,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C0" / "model.pt"
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    prepared["model"].eval()
    backbone = prepared["model"].backbone
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    all_samples = build_train_samples(
        prepared["sequences"],
        users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    samples = select_samples(
        all_samples,
        prepared["heads"],
        int(config["seed"]),
        dataset,
        int(config["head_samples"]),
        int(config["tail_samples"]),
        int(config["minimum_history_items"]),
    )
    batch = collate(prepared["collator"], samples)
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    labels = batch["target_ids"].to(device)
    passages, width = input_ids.shape[1], input_ids.shape[2]
    backbone.encoder.n_passages = passages
    flat_ids = input_ids.view(input_ids.shape[0], -1)
    flat_attention = attention.view(attention.shape[0], -1)
    with torch.no_grad():
        hidden = backbone.encoder(
            input_ids=flat_ids,
            attention_mask=flat_attention,
            return_dict=True,
        )[0].detach()
        baseline_output = backbone(
            input_ids=None,
            attention_mask=flat_attention,
            encoder_outputs=(hidden,),
            labels=labels,
            return_dict=True,
        )
    gate = FinePassageGate(
        hidden.shape[-1], float(config["gate"]["bound"])
    ).to(device)
    zero_hidden, zero_gates = gate(hidden, attention)
    zero_logits = decoder_logits(backbone, zero_hidden, flat_attention, labels)
    zero_identity = float((zero_logits - baseline_output.logits).abs().max())
    shaped_original = hidden.view(
        hidden.shape[0], passages, width, hidden.shape[-1]
    )
    shaped_zero = zero_hidden.view_as(shaped_original)
    coarse_identity = float(
        (shaped_original[:, 0] - shaped_zero[:, 0]).abs().max()
    )
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(config["gate"]["learning_rate"]),
        weight_decay=float(config["gate"]["weight_decay"]),
    )
    losses, gradient_norms = [], []
    for _ in range(int(config["gate"]["training_steps"])):
        optimizer.zero_grad(set_to_none=True)
        gated_hidden, _ = gate(hidden, attention)
        output = backbone(
            input_ids=None,
            attention_mask=flat_attention,
            encoder_outputs=(gated_hidden,),
            labels=labels,
            return_dict=True,
        )
        loss = output.loss
        if not torch.isfinite(loss):
            raise ValueError("non-finite FPUG gate loss")
        loss.backward()
        gradient_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().float().pow(2).sum())
                for parameter in gate.parameters()
                if parameter.grad is not None
            )
        )
        gradient_norms.append(gradient_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
    trained_hidden, trained_gates = gate(hidden, attention)
    trained_logits = decoder_logits(
        backbone, trained_hidden, flat_attention, labels
    )
    active_detail = attention[:, 1:].any(-1)
    active_gates = trained_gates[active_detail]
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(gate.state_dict(), handle.name)
        reloaded = FinePassageGate(
            hidden.shape[-1], float(config["gate"]["bound"])
        ).to(device)
        reloaded.load_state_dict(torch.load(handle.name, map_location=device))
        reload_hidden, _ = reloaded(hidden, attention)
        reload_logits = decoder_logits(
            backbone, reload_hidden, flat_attention, labels
        )
    reload_difference = float((reload_logits - trained_logits).abs().max())
    relative_loss_decrease = (
        float(baseline_output.loss) - losses[-1]
    ) / float(baseline_output.loss)
    max_gate_deviation = float((active_gates - 1.0).abs().max())
    backbone_gradients_absent = all(
        parameter.grad is None for parameter in backbone.parameters()
    )
    trie = gt.Trie(prepared["encoded_candidates"])
    item_to_sequence = dict(zip(prepared["catalog"], prepared["encoded_candidates"]))
    trie_valid = all(
        all(
            sequence[position] in trie.get(sequence[:position])
            for position in range(1, len(sequence))
        )
        for sequence in [
            item_to_sequence[sample["positive_item"]] for sample in samples
        ]
    )
    gates = config["correctness_gates"]
    metrics = {
        "samples": len(samples),
        "head_samples": sum(
            sample["positive_item"] in prepared["heads"] for sample in samples
        ),
        "tail_samples": sum(
            sample["positive_item"] not in prepared["heads"] for sample in samples
        ),
        "zero_identity_max_abs_difference": zero_identity,
        "coarse_identity_max_abs_difference": coarse_identity,
        "zero_gate_min": float(zero_gates.min()),
        "zero_gate_max": float(zero_gates.max()),
        "initial_loss": float(baseline_output.loss),
        "final_loss": losses[-1],
        "relative_loss_decrease": relative_loss_decrease,
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "trained_gate_min": float(active_gates.min()),
        "trained_gate_max": float(active_gates.max()),
        "trained_gate_max_deviation": max_gate_deviation,
        "reload_max_abs_difference": reload_difference,
        "optimizer_steps": len(losses),
    }
    checks = {
        "zero_identity": zero_identity
        <= float(gates["zero_identity_tolerance"]),
        "coarse_identity": coarse_identity
        <= float(gates["coarse_identity_tolerance"]),
        "gate_bounds": metrics["trained_gate_min"] >= float(gates["gate_min"])
        and metrics["trained_gate_max"] <= float(gates["gate_max"]),
        "nonzero_finite_gradient": min(gradient_norms)
        >= float(gates["minimum_gradient_norm"])
        and all(math.isfinite(value) for value in gradient_norms),
        "loss_decrease": relative_loss_decrease
        >= float(gates["minimum_relative_loss_decrease"]),
        "trained_gate_nonidentity": max_gate_deviation
        >= float(gates["minimum_trained_gate_max_deviation"]),
        "reload_identity": reload_difference <= float(gates["reload_tolerance"]),
        "head_tail_presence": metrics["head_samples"] == int(config["head_samples"])
        and metrics["tail_samples"] == int(config["tail_samples"]),
        "trie_membership": trie_valid,
        "backbone_gradients_absent": backbone_gradients_absent,
    }
    integrity = {
        "finite_rate": float(
            all(math.isfinite(value) for value in losses + gradient_norms)
        ),
        "optimizer_steps": len(losses),
        "backbone_frozen": backbone_gradients_absent,
        "parameter_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "validation_test_predictions_read": False,
        "sports_read": False,
        "checkpoint_sha256": checkpoint_sha,
    }
    integrity_valid = (
        integrity["finite_rate"] == 1.0
        and integrity["optimizer_steps"] == int(config["gate"]["training_steps"])
        and integrity["backbone_frozen"]
        and integrity["parameter_sha_unchanged"]
        and not integrity["validation_test_predictions_read"]
        and not integrity["sports_read"]
    )
    result = {
        "metrics": metrics,
        "checks": checks,
        "correctness_pass": all(checks.values()),
        "integrity": integrity,
        "integrity_valid": integrity_valid,
    }
    write_json(output_root / dataset / "summary.json", result)
    del prepared, hidden
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FPUG S0 requires CUDA")
    config = json.loads(args.config.read_text())
    if sha256(Path(__file__)) != config["integrity"]["code_sha256"]:
        raise ValueError("FPUG S0 code SHA mismatch")
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    results = {
        dataset: run_dataset(dataset, config, p0_config, args.output_root, device)
        for dataset in config["datasets"]
    }
    integrity_valid = all(row["integrity_valid"] for row in results.values())
    correctness_pass = all(row["correctness_pass"] for row in results.values())
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "FPUG_S0_CORRECTNESS_PASS"
        if correctness_pass
        else "STOP_FPUG_S0_CORRECTNESS_FAILED"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "results": results,
        "integrity_valid": integrity_valid,
        "validation_test_predictions_read": False,
        "sports_read": False,
    }
    write_json(args.output_root / "summary.json", summary)
    lines = [
        "# FPUG-S0 Decision",
        "",
        f"- Fixed decision: **`{decision}`**",
        f"- Integrity valid: `{str(integrity_valid).lower()}`",
        "- Validation/test/Sports read: `false`",
        "",
    ]
    for dataset, result in results.items():
        lines.extend([f"## {dataset}", ""])
        for name, passed in result["checks"].items():
            lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
        lines.append("")
    (args.output_root / "decision.md").write_text("\n".join(lines))
    write_json(
        args.output_root / "status.json",
        {"experiment_id": config["experiment_id"], "status": "completed"},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
