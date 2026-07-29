#!/usr/bin/env python3
"""FPUG P0: frozen-backbone gate training and locked dual-domain effect pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase3.hbtr_b1_smoke import normalized_sequence  # noqa: E402
from experiment.phase4.fpug_s0 import FinePassageGate  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    ROOT,
    build_train_samples,
    build_validation_samples,
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)
from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions  # noqa: E402
from utils import generation_trie as gt  # noqa: E402


def hash_key(seed: int, dataset: str, salt: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|{salt}|{value}".encode()).hexdigest()


def stratified_unique_select(
    samples: list[dict],
    heads: set[str],
    seed: int,
    dataset: str,
    salt: str,
    head_count: int,
    tail_count: int,
    minimum_history: int,
    excluded_users: set[str] | None = None,
) -> list[dict]:
    excluded_users = excluded_users or set()
    ordered = sorted(
        (
            row
            for row in samples
            if len(row["history_items"]) >= minimum_history
            and row["user_id"] not in excluded_users
        ),
        key=lambda row: hash_key(seed, dataset, salt, row["sample_key"]),
    )
    selected, users = [], set()
    counts = {"head": 0, "tail": 0}
    limits = {"head": head_count, "tail": tail_count}
    for row in ordered:
        group = "head" if row["positive_item"] in heads else "tail"
        if row["user_id"] in users or counts[group] >= limits[group]:
            continue
        selected.append(row)
        users.add(row["user_id"])
        counts[group] += 1
        if counts == limits:
            break
    if counts != limits:
        raise ValueError(f"insufficient {dataset} {salt} samples: {counts}")
    return sorted(selected, key=lambda row: row["sample_key"])


def validation_users_by_hash(
    users: set[str], dataset: str, salt: str, count: int
) -> list[str]:
    return sorted(
        users, key=lambda user: hashlib.sha256(
            f"{salt}|{dataset}|{user}".encode()
        ).hexdigest()
    )[:count]


@torch.no_grad()
def encode_batch(backbone, input_ids: torch.Tensor, attention: torch.Tensor):
    passages = input_ids.shape[1]
    backbone.encoder.n_passages = passages
    flat_ids = input_ids.view(input_ids.shape[0], -1)
    flat_attention = attention.view(attention.shape[0], -1)
    hidden = backbone.encoder(
        input_ids=flat_ids,
        attention_mask=flat_attention,
        return_dict=True,
    )[0].detach()
    return hidden, flat_ids, flat_attention


def gate_loss(backbone, gate, hidden, attention, flat_attention, labels):
    gated_hidden, gates = gate(hidden, attention)
    output = backbone(
        input_ids=None,
        attention_mask=flat_attention,
        encoder_outputs=(gated_hidden,),
        labels=labels,
        return_dict=True,
    )
    return output.loss, output.logits, gates


@torch.no_grad()
def evaluate_ce(prepared: dict, samples: list[dict], gate, batch_size: int, device):
    total_loss, total_samples = 0.0, 0
    for start in range(0, len(samples), batch_size):
        rows = samples[start : start + batch_size]
        batch = collate(prepared["collator"], rows)
        input_ids = batch["item_text_ids"].to(device)
        attention = batch["item_text_masks"].to(device)
        labels = batch["target_ids"].to(device)
        hidden, _, flat_attention = encode_batch(
            prepared["model"].backbone, input_ids, attention
        )
        if gate is None:
            output = prepared["model"].backbone(
                input_ids=None,
                attention_mask=flat_attention,
                encoder_outputs=(hidden,),
                labels=labels,
                return_dict=True,
            )
            loss = output.loss
        else:
            loss, _, _ = gate_loss(
                prepared["model"].backbone,
                gate,
                hidden,
                attention,
                flat_attention,
                labels,
            )
        total_loss += float(loss) * len(rows)
        total_samples += len(rows)
    return total_loss / total_samples


def train_domain(dataset: str, config: dict, p0_config: dict, output_root: Path, device):
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
        prepared["sequences"], users, prepared["item2input"], prepared["item2lexid"]
    )
    fit = stratified_unique_select(
        all_samples,
        prepared["heads"],
        int(config["seed"]),
        dataset,
        "fpug-p0-fit-v1",
        int(config["fit"]["head_samples"]),
        int(config["fit"]["tail_samples"]),
        int(config["minimum_history_items"]),
    )
    fit_users = {row["user_id"] for row in fit}
    calibration = stratified_unique_select(
        all_samples,
        prepared["heads"],
        int(config["seed"]),
        dataset,
        "fpug-p0-calibration-v1",
        int(config["calibration"]["head_samples"]),
        int(config["calibration"]["tail_samples"]),
        int(config["minimum_history_items"]),
        fit_users,
    )
    calibration_users = {row["user_id"] for row in calibration}
    gate = FinePassageGate(
        backbone.config.d_model, float(config["training"]["gate_bound"])
    ).to(device)
    first_batch = collate(prepared["collator"], fit[: int(config["training"]["batch_size"])])
    first_ids = first_batch["item_text_ids"].to(device)
    first_attention = first_batch["item_text_masks"].to(device)
    first_labels = first_batch["target_ids"].to(device)
    first_hidden, _, first_flat_attention = encode_batch(
        backbone, first_ids, first_attention
    )
    with torch.no_grad():
        baseline_first = backbone(
            input_ids=None,
            attention_mask=first_flat_attention,
            encoder_outputs=(first_hidden,),
            labels=first_labels,
            return_dict=True,
        )
        _, zero_logits, _ = gate_loss(
            backbone,
            gate,
            first_hidden,
            first_attention,
            first_flat_attention,
            first_labels,
        )
    zero_identity = float((baseline_first.logits - zero_logits).abs().max())
    baseline_calibration_ce = evaluate_ce(
        prepared,
        calibration,
        None,
        int(config["training"]["batch_size"]),
        device,
    )
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epoch_rows = []
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        ordered = sorted(
            fit,
            key=lambda row: hash_key(
                int(config["seed"]), dataset, f"epoch-{epoch}", row["sample_key"]
            ),
        )
        losses, gradient_norms, active_gate_values = [], [], []
        for start in range(0, len(ordered), int(config["training"]["batch_size"])):
            rows = ordered[start : start + int(config["training"]["batch_size"])]
            batch = collate(prepared["collator"], rows)
            input_ids = batch["item_text_ids"].to(device)
            attention = batch["item_text_masks"].to(device)
            labels = batch["target_ids"].to(device)
            hidden, _, flat_attention = encode_batch(backbone, input_ids, attention)
            optimizer.zero_grad(set_to_none=True)
            loss, _, gates = gate_loss(
                backbone, gate, hidden, attention, flat_attention, labels
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite FPUG P0 loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                gate.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            gradient_norms.append(float(gradient_norm))
            active_gate_values.extend(
                gates[attention[:, 1:].any(-1)].detach().cpu().tolist()
            )
        calibration_ce = evaluate_ce(
            prepared,
            calibration,
            gate,
            int(config["training"]["batch_size"]),
            device,
        )
        relative_decrease = (
            baseline_calibration_ce - calibration_ce
        ) / baseline_calibration_ce
        gate_path = output_dir / f"gate_epoch_{epoch}.pt"
        torch.save(gate.state_dict(), gate_path)
        epoch_rows.append(
            {
                "epoch": epoch,
                "fit_mean_loss": float(np.mean(losses)),
                "calibration_ce": calibration_ce,
                "calibration_relative_decrease": relative_decrease,
                "gradient_norm_max": max(gradient_norms),
                "gate_min": min(active_gate_values),
                "gate_max": max(active_gate_values),
                "gate_sha256": sha256(gate_path),
            }
        )
        print(
            f"FPUG_P0_TRAIN dataset={dataset} epoch={epoch} "
            f"cal_rel={relative_decrease:.6f}",
            flush=True,
        )
    result = {
        "fit_samples": len(fit),
        "calibration_samples": len(calibration),
        "fit_users": len(fit_users),
        "calibration_users": len(calibration_users),
        "fit_calibration_user_overlap": len(fit_users & calibration_users),
        "zero_identity_max_abs_difference": zero_identity,
        "baseline_calibration_ce": baseline_calibration_ce,
        "epochs": epoch_rows,
        "checkpoint_sha256": checkpoint_sha,
        "parameter_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "backbone_gradients_absent": all(
            parameter.grad is None for parameter in backbone.parameters()
        ),
    }
    write_json(output_dir / "training_summary.json", result)
    del prepared
    torch.cuda.empty_cache()
    return result


def generate_with_hidden(
    backbone,
    flat_ids,
    flat_attention,
    hidden,
    trie,
    max_length: int,
    beam_size: int,
    length_penalty: float,
):
    encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden
    )
    parent = super(backbone.__class__, backbone)
    return parent.generate(
        input_ids=flat_ids,
        attention_mask=flat_attention,
        encoder_outputs=encoder_outputs,
        max_length=max_length,
        prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(trie),
        num_beams=beam_size,
        num_return_sequences=beam_size,
        length_penalty=length_penalty,
    )


def rank_metrics(items: list[str], target: str, cutoff: int = 10):
    rank = items.index(target) + 1 if target in items else None
    recall = float(rank is not None and rank <= cutoff)
    ndcg = 1.0 / math.log2(rank + 1) if recall else 0.0
    return rank, recall, ndcg


def bootstrap_relative(base, new, replicates: int, seed: int):
    base = np.asarray(base, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        indices = rng.integers(0, len(base), len(base))
        denominator = base[indices].mean()
        values.append(
            new[indices].mean() / denominator - 1.0
            if denominator > 0
            else 0.0
        )
    point = new.mean() / base.mean() - 1.0 if base.mean() > 0 else 0.0
    return {
        "point": float(point),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("empty validation rows")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate_domain(
    dataset: str,
    selected_epoch: int,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device,
):
    prepared = prepare(dataset, p0_config, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C0" / "model.pt"
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    prepared["model"].eval()
    backbone = prepared["model"].backbone
    gate = FinePassageGate(
        backbone.config.d_model, float(config["training"]["gate_bound"])
    ).to(device)
    gate_path = output_root / dataset / f"gate_epoch_{selected_epoch}.pt"
    gate.load_state_dict(torch.load(gate_path, map_location=device))
    gate.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    max_length = max(len(row) for row in prepared["encoded_candidates"])
    beam_size = int(config["validation"]["beam_size"])

    # Exercise the exact locked-generation path on training prefixes before the
    # validation user file is read. This is an interface/integrity smoke only.
    train_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    training_samples = build_train_samples(
        prepared["sequences"],
        train_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    smoke_samples = stratified_unique_select(
        training_samples,
        prepared["heads"],
        int(config["seed"]),
        dataset,
        "fpug-p0-generation-smoke-v1",
        1,
        1,
        int(config["minimum_history_items"]),
    )
    smoke_batch = collate(prepared["collator"], smoke_samples)
    smoke_ids = smoke_batch["item_text_ids"].to(device)
    smoke_attention = smoke_batch["item_text_masks"].to(device)
    smoke_hidden, smoke_flat_ids, smoke_flat_attention = encode_batch(
        backbone, smoke_ids, smoke_attention
    )
    smoke_baseline = generate_with_hidden(
        backbone,
        smoke_flat_ids,
        smoke_flat_attention,
        smoke_hidden,
        trie,
        max_length,
        beam_size,
        float(config["validation"]["length_penalty"]),
    )
    smoke_gated_hidden, _ = gate(smoke_hidden, smoke_attention)
    smoke_gated = generate_with_hidden(
        backbone,
        smoke_flat_ids,
        smoke_flat_attention,
        smoke_gated_hidden,
        trie,
        max_length,
        beam_size,
        float(config["validation"]["length_penalty"]),
    )
    smoke_sequences = torch.cat((smoke_baseline, smoke_gated), dim=0)
    smoke_mapping_rate = float(
        np.mean(
            [
                prepared["sequence_to_item"].get(
                    normalized_sequence(sequence.tolist())
                )
                is not None
                for sequence in smoke_sequences
            ]
        )
    )
    if smoke_mapping_rate != 1.0:
        raise ValueError("training-prefix generation smoke mapping failure")

    validation_pool = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "validation_users.txt"
    )
    selected_users = validation_users_by_hash(
        validation_pool,
        dataset,
        config["validation"]["selection_salt"],
        int(config["validation"]["users_per_dataset"]),
    )
    samples = build_validation_samples(
        prepared["sequences"],
        set(selected_users),
        prepared["item2input"],
        prepared["item2lexid"],
    )
    samples = sorted(samples, key=lambda row: selected_users.index(row["user_id"]))
    mapping_rate_numerator, mapping_rate_denominator = 0, 0
    rows = []
    batch_size = 4
    for start in range(0, len(samples), batch_size):
        selected = samples[start : start + batch_size]
        batch = collate(prepared["collator"], selected)
        input_ids = batch["item_text_ids"].to(device)
        attention = batch["item_text_masks"].to(device)
        hidden, flat_ids, flat_attention = encode_batch(
            backbone, input_ids, attention
        )
        baseline_sequences = generate_with_hidden(
            backbone,
            flat_ids,
            flat_attention,
            hidden,
            trie,
            max_length,
            beam_size,
            float(config["validation"]["length_penalty"]),
        )
        gated_hidden, active_gates = gate(hidden, attention)
        gated_sequences = generate_with_hidden(
            backbone,
            flat_ids,
            flat_attention,
            gated_hidden,
            trie,
            max_length,
            beam_size,
            float(config["validation"]["length_penalty"]),
        )
        for offset, sample in enumerate(selected):
            baseline_items = [
                prepared["sequence_to_item"].get(
                    normalized_sequence(sequence.tolist())
                )
                for sequence in baseline_sequences[
                    offset * beam_size : (offset + 1) * beam_size
                ]
            ]
            gated_items = [
                prepared["sequence_to_item"].get(
                    normalized_sequence(sequence.tolist())
                )
                for sequence in gated_sequences[
                    offset * beam_size : (offset + 1) * beam_size
                ]
            ]
            mapping_rate_denominator += beam_size * 2
            mapping_rate_numerator += sum(item is not None for item in baseline_items)
            mapping_rate_numerator += sum(item is not None for item in gated_items)
            if any(item is None for item in baseline_items + gated_items):
                raise ValueError("candidate mapping failure")
            baseline_rank, baseline_recall, baseline_ndcg = rank_metrics(
                baseline_items, sample["positive_item"]
            )
            gated_rank, gated_recall, gated_ndcg = rank_metrics(
                gated_items, sample["positive_item"]
            )
            rows.append(
                {
                    "sample_key": sample["sample_key"],
                    "user_id": sample["user_id"],
                    "target_item": sample["positive_item"],
                    "target_group": (
                        "head"
                        if sample["positive_item"] in prepared["heads"]
                        else "tail"
                    ),
                    "baseline_rank": baseline_rank or "",
                    "gated_rank": gated_rank or "",
                    "baseline_recall10": baseline_recall,
                    "gated_recall10": gated_recall,
                    "baseline_ndcg10": baseline_ndcg,
                    "gated_ndcg10": gated_ndcg,
                    "gate_min": float(
                        active_gates[offset][attention[offset, 1:].any(-1)].min()
                    ),
                    "gate_max": float(
                        active_gates[offset][attention[offset, 1:].any(-1)].max()
                    ),
                }
            )
        done = min(start + batch_size, len(samples))
        if done % 64 == 0:
            print(
                f"FPUG_P0_VALIDATION dataset={dataset} users={done}/{len(samples)}",
                flush=True,
            )
    validation_path = output_root / dataset / "validation_metrics.csv"
    write_csv(validation_path, rows)
    tail = [row for row in rows if row["target_group"] == "tail"]
    overall_relative = bootstrap_relative(
        [row["baseline_ndcg10"] for row in rows],
        [row["gated_ndcg10"] for row in rows],
        int(config["validation"]["bootstrap_replicates"]),
        int(config["validation"]["bootstrap_seed"]),
    )
    tail_relative = bootstrap_relative(
        [row["baseline_ndcg10"] for row in tail],
        [row["gated_ndcg10"] for row in tail],
        int(config["validation"]["bootstrap_replicates"]),
        int(config["validation"]["bootstrap_seed"]) + 1,
    )
    baseline_recall = float(np.mean([row["baseline_recall10"] for row in rows]))
    gated_recall = float(np.mean([row["gated_recall10"] for row in rows]))
    harm_rate = float(
        np.mean(
            [
                row["baseline_recall10"] == 1.0
                and row["gated_recall10"] == 0.0
                for row in rows
            ]
        )
    )
    metrics = {
        "users": len(rows),
        "tail_users": len(tail),
        "baseline_ndcg10": float(np.mean([row["baseline_ndcg10"] for row in rows])),
        "gated_ndcg10": float(np.mean([row["gated_ndcg10"] for row in rows])),
        "overall_ndcg10_relative": overall_relative,
        "baseline_recall10": baseline_recall,
        "gated_recall10": gated_recall,
        "overall_recall10_absolute_gain": gated_recall - baseline_recall,
        "baseline_tail_ndcg10": float(
            np.mean([row["baseline_ndcg10"] for row in tail])
        ),
        "gated_tail_ndcg10": float(
            np.mean([row["gated_ndcg10"] for row in tail])
        ),
        "tail_ndcg10_relative": tail_relative,
        "baseline_hit_gated_miss_rate": harm_rate,
    }
    gates = config["effect_gates"]
    checks = {
        "overall_ndcg10_relative_gain": overall_relative["point"]
        >= float(gates["overall_ndcg10_relative_gain_min"]),
        "overall_ndcg10_ci_lower": overall_relative["ci_low"]
        >= float(gates["overall_ndcg10_relative_gain_ci_lower_min"]),
        "overall_recall10_absolute_gain": metrics[
            "overall_recall10_absolute_gain"
        ]
        >= float(gates["overall_recall10_absolute_gain_min"]),
        "tail_ndcg10_relative_gain": tail_relative["point"]
        >= float(gates["tail_ndcg10_relative_gain_min"]),
        "tail_ndcg10_ci_lower": tail_relative["ci_low"]
        >= float(gates["tail_ndcg10_relative_gain_ci_lower_min"]),
        "baseline_hit_gated_miss_rate": harm_rate
        <= float(gates["baseline_hit_gated_miss_rate_max"]),
    }
    integrity = {
        "training_prefix_generation_smoke_mapping_rate": smoke_mapping_rate,
        "candidate_mapping_rate": mapping_rate_numerator
        / mapping_rate_denominator,
        "finite_rate": float(
            all(
                math.isfinite(float(row[key]))
                for row in rows
                for key in (
                    "baseline_recall10",
                    "gated_recall10",
                    "baseline_ndcg10",
                    "gated_ndcg10",
                    "gate_min",
                    "gate_max",
                )
            )
        ),
        "validation_selection_target_independent": True,
        "validation_user_sha256": hashlib.sha256(
            "\n".join(selected_users).encode()
        ).hexdigest(),
        "parameter_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "test_predictions_read": False,
        "sports_read": False,
        "validation_file_sha256": sha256(validation_path),
    }
    integrity_valid = (
        integrity["training_prefix_generation_smoke_mapping_rate"] == 1.0
        and integrity["candidate_mapping_rate"] == 1.0
        and integrity["finite_rate"] == 1.0
        and integrity["validation_selection_target_independent"]
        and integrity["parameter_sha_unchanged"]
        and not integrity["test_predictions_read"]
        and not integrity["sports_read"]
    )
    result = {
        "metrics": metrics,
        "checks": checks,
        "effect_pass": all(checks.values()),
        "integrity": integrity,
        "integrity_valid": integrity_valid,
    }
    write_json(output_root / dataset / "validation_summary.json", result)
    del prepared
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FPUG P0 requires CUDA")
    config = json.loads(args.config.read_text())
    if sha256(Path(__file__)) != config["integrity"]["code_sha256"]:
        raise ValueError("FPUG P0 code SHA mismatch")
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    training = {
        dataset: train_domain(dataset, config, p0_config, args.output_root, device)
        for dataset in config["datasets"]
    }
    epoch_scores = {}
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        epoch_scores[epoch] = float(
            np.mean(
                [
                    training[dataset]["epochs"][epoch - 1][
                        "calibration_relative_decrease"
                    ]
                    for dataset in config["datasets"]
                ]
            )
        )
    selected_epoch = max(epoch_scores, key=lambda epoch: (epoch_scores[epoch], -epoch))
    calibration_lock = {
        "selected_epoch": selected_epoch,
        "mean_relative_decrease_by_epoch": {
            str(epoch): score for epoch, score in epoch_scores.items()
        },
        "selection_uses_validation": False,
    }
    write_json(args.output_root / "calibration_lock.json", calibration_lock)
    validation = {
        dataset: validate_domain(
            dataset,
            selected_epoch,
            config,
            p0_config,
            args.output_root,
            device,
        )
        for dataset in config["datasets"]
    }
    training_integrity = all(
        row["fit_calibration_user_overlap"] == 0
        and row["zero_identity_max_abs_difference"]
        <= float(config["integrity"]["zero_gate_identity_tolerance"])
        and row["parameter_sha_unchanged"]
        and row["backbone_gradients_absent"]
        for row in training.values()
    )
    validation_integrity = all(row["integrity_valid"] for row in validation.values())
    integrity_valid = training_integrity and validation_integrity
    effect_pass = all(row["effect_pass"] for row in validation.values())
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "FPUG_FREEZE_FOR_CONFIRMATION"
        if effect_pass
        else "STOP_FPUG_EFFECT_GATE_FAILED"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "selected_epoch": selected_epoch,
        "decision": decision,
        "training": training,
        "validation": validation,
        "integrity_valid": integrity_valid,
        "test_predictions_read": False,
        "sports_read": False,
    }
    write_json(args.output_root / "summary.json", summary)
    lines = [
        "# FPUG-P0 Decision",
        "",
        f"- Selected shared epoch: `{selected_epoch}`",
        f"- Fixed decision: **`{decision}`**",
        f"- Integrity valid: `{str(integrity_valid).lower()}`",
        "- Test/Sports read: `false`",
        "",
    ]
    for dataset, result in validation.items():
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
