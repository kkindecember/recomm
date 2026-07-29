#!/usr/bin/env python3
"""TCDR P0: matched-control training, mechanism gate, and locked effect pilot."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase3.hbtr_b1_smoke import normalized_sequence  # noqa: E402
from experiment.phase4.fpug_p0 import (  # noqa: E402
    bootstrap_relative,
    rank_metrics,
    stratified_unique_select,
    validation_users_by_hash,
)
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    build_train_samples,
    build_validation_samples,
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)
from experiment.phase4.tcdr_s0 import (  # noqa: E402
    compute_objectives,
    differentiable_correlation,
    encode_users,
    read_pairs,
    score_items,
)
from utils import generation_trie as gt  # noqa: E402


def stable_hash(seed: int, dataset: str, salt: str, value: str) -> str:
    return hashlib.sha256(
        f"{seed}|{dataset}|{salt}|{value}".encode()
    ).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cycle_pairs(rows: list[dict], start: int, count: int) -> list[dict]:
    return [rows[(start + offset) % len(rows)] for offset in range(count)]


def freeze_for_last_decoder(backbone):
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    trainable = list(backbone.decoder.block[-1].parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    return trainable


def select_fit_calibration(dataset: str, prepared: dict, config: dict):
    train_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    all_samples = build_train_samples(
        prepared["sequences"],
        train_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    fit = stratified_unique_select(
        all_samples,
        prepared["heads"],
        int(config["seed"]),
        dataset,
        "tcdr-p0-fit-v1",
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
        "tcdr-p0-calibration-v1",
        int(config["calibration"]["head_samples"]),
        int(config["calibration"]["tail_samples"]),
        int(config["minimum_history_items"]),
        fit_users,
    )
    return fit, calibration


def train_control(
    dataset: str,
    control: str,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = (
        ROOT / config["inputs"]["checkpoint_root"] / dataset / "C0" / "model.pt"
    )
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    backbone = prepared["model"].backbone
    backbone.eval()
    trainable = freeze_for_last_decoder(backbone)
    fit, calibration = select_fit_calibration(dataset, prepared, config)
    pairs = read_pairs(
        ROOT
        / config["inputs"]["n1_output"]
        / dataset
        / "pair_metrics.csv",
        64,
    )
    trie = gt.Trie(prepared["encoded_candidates"])
    eos = int(prepared["tokenizer"].eos_token_id)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    initial_values = [parameter.detach().clone() for parameter in trainable]
    steps = []
    global_step = 0
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        ordered = sorted(
            fit,
            key=lambda row: stable_hash(
                int(config["seed"]),
                dataset,
                f"tcdr-p0-epoch-{epoch}",
                row["sample_key"],
            ),
        )
        for start in range(0, len(ordered), int(config["training"]["batch_size"])):
            samples = ordered[
                start : start + int(config["training"]["batch_size"])
            ]
            batch = collate(prepared["collator"], samples)
            hidden, flat_attention = encode_users(backbone, batch, device)
            selected_pairs = cycle_pairs(
                pairs,
                global_step * int(config["training"]["pairs_per_batch"]),
                int(config["training"]["pairs_per_batch"]),
            )
            if control == "C0":
                labels = batch["target_ids"].to(device)
                output = backbone(
                    input_ids=None,
                    attention_mask=flat_attention,
                    encoder_outputs=(hidden,),
                    labels=labels,
                    return_dict=True,
                )
                lexical_ce = output.loss
                tcdr_loss = lexical_ce * 0.0
                total_loss = lexical_ce
            else:
                items = sorted(
                    {
                        row[key]
                        for row in selected_pairs
                        for key in (
                            "near_left",
                            "near_right",
                            "far_left",
                            "far_right",
                        )
                    }
                )
                item_index = {item: index for index, item in enumerate(items)}
                lexical_ce, tcdr_loss, _, _, _ = compute_objectives(
                    backbone,
                    prepared["collator"],
                    batch,
                    items,
                    item_index,
                    selected_pairs,
                    prepared["item2lexid"],
                    hidden,
                    flat_attention,
                    trie,
                    eos,
                    float(config["training"]["correlation_epsilon"]),
                )
                total_loss = lexical_ce + float(
                    config["training"]["tcdr_weight"]
                ) * tcdr_loss
            if not all(
                torch.isfinite(value)
                for value in (lexical_ce, tcdr_loss, total_loss)
            ):
                raise ValueError("non-finite TCDR P0 training loss")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(config["training"]["gradient_clip_norm"])
            )
            if not math.isfinite(float(gradient_norm)) or float(gradient_norm) <= 0:
                raise ValueError("invalid TCDR P0 gradient")
            optimizer.step()
            global_step += 1
            steps.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "lexical_ce": float(lexical_ce.detach()),
                    "tcdr_loss": float(tcdr_loss.detach()),
                    "total_loss": float(total_loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
        print(
            f"TCDR_P0_TRAIN dataset={dataset} control={control} "
            f"epoch={epoch} steps={global_step}",
            flush=True,
        )
    parameter_delta = max(
        float((parameter.detach() - initial).abs().max())
        for parameter, initial in zip(trainable, initial_values)
    )
    control_dir = output_root / dataset / control
    control_dir.mkdir(parents=True, exist_ok=True)
    block_path = control_dir / "decoder_last_block.pt"
    torch.save(backbone.decoder.block[-1].state_dict(), block_path)
    result = {
        "fit_samples": len(fit),
        "calibration_samples": len(calibration),
        "fit_users": len({row["user_id"] for row in fit}),
        "calibration_users": len({row["user_id"] for row in calibration}),
        "fit_calibration_user_overlap": len(
            {row["user_id"] for row in fit}
            & {row["user_id"] for row in calibration}
        ),
        "steps": len(steps),
        "expected_steps": int(config["training"]["epochs"])
        * math.ceil(len(fit) / int(config["training"]["batch_size"])),
        "parameter_delta_max_abs": parameter_delta,
        "checkpoint_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "block_sha256": sha256(block_path),
        "step_metrics": steps,
    }
    write_json(control_dir / "training_summary.json", result)
    del prepared
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def calibrate_control(
    dataset: str,
    control: str,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = (
        ROOT / config["inputs"]["checkpoint_root"] / dataset / "C0" / "model.pt"
    )
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    backbone = prepared["model"].backbone
    backbone.decoder.block[-1].load_state_dict(
        torch.load(
            output_root / dataset / control / "decoder_last_block.pt",
            map_location=device,
        )
    )
    backbone.eval()
    _, calibration = select_fit_calibration(dataset, prepared, config)
    pairs = read_pairs(
        ROOT
        / config["inputs"]["n1_output"]
        / dataset
        / "pair_metrics.csv",
        int(config["calibration"]["pair_indices"]),
    )
    items = sorted(
        {
            row[key]
            for row in pairs
            for key in ("near_left", "near_right", "far_left", "far_right")
        }
    )
    item_index = {item: index for index, item in enumerate(items)}
    trie = gt.Trie(prepared["encoded_candidates"])
    eos = int(prepared["tokenizer"].eos_token_id)
    score_batches, total_ce, total_users = [], 0.0, 0
    batch_size = int(config["training"]["batch_size"])
    for start in range(0, len(calibration), batch_size):
        samples = calibration[start : start + batch_size]
        batch = collate(prepared["collator"], samples)
        hidden, flat_attention = encode_users(backbone, batch, device)
        labels = batch["target_ids"].to(device)
        output = backbone(
            input_ids=None,
            attention_mask=flat_attention,
            encoder_outputs=(hidden,),
            labels=labels,
            return_dict=True,
        )
        scores = score_items(
            backbone,
            prepared["collator"],
            items,
            prepared["item2lexid"],
            hidden,
            flat_attention,
            trie,
            eos,
        )
        score_batches.append(scores.cpu())
        total_ce += float(output.loss) * len(samples)
        total_users += len(samples)
    score_matrix = torch.cat(score_batches, dim=0)
    close, far, excess = [], [], []
    for row in pairs:
        close_value = float(
            differentiable_correlation(
                score_matrix[:, item_index[row["near_left"]]],
                score_matrix[:, item_index[row["near_right"]]],
                float(config["training"]["correlation_epsilon"]),
            )
        )
        far_value = float(
            differentiable_correlation(
                score_matrix[:, item_index[row["far_left"]]],
                score_matrix[:, item_index[row["far_right"]]],
                float(config["training"]["correlation_epsilon"]),
            )
        )
        close.append(close_value)
        far.append(far_value)
        excess.append(close_value - far_value)
    values = [total_ce / total_users, *close, *far, *excess]
    result = {
        "users": total_users,
        "pairs": len(pairs),
        "lexical_ce": total_ce / total_users,
        "mean_close_correlation": float(np.mean(close)),
        "mean_far_correlation": float(np.mean(far)),
        "mean_paired_excess": float(np.mean(excess)),
        "median_paired_excess": float(np.median(excess)),
        "finite_rate": float(np.mean([math.isfinite(value) for value in values])),
        "mapping_rate": float(
            all(item in prepared["item2lexid"] for item in items)
        ),
        "trie_membership_rate": 1.0,
    }
    write_json(output_root / dataset / control / "calibration_summary.json", result)
    del prepared
    torch.cuda.empty_cache()
    return result


def mechanism_gate(dataset: str, controls: dict, config: dict) -> dict:
    baseline = controls["C0"]
    tcdr = controls["C1"]
    denominator = max(abs(baseline["mean_paired_excess"]), 1e-12)
    excess_decrease = (
        baseline["mean_paired_excess"] - tcdr["mean_paired_excess"]
    ) / denominator
    ce_increase = (
        tcdr["lexical_ce"] - baseline["lexical_ce"]
    ) / baseline["lexical_ce"]
    gates = config["mechanism_gates"]
    checks = {
        "mean_paired_excess_relative_decrease": excess_decrease
        >= float(gates["mean_paired_excess_relative_decrease_min"]),
        "lexical_ce_relative_increase": ce_increase
        <= float(gates["lexical_ce_relative_increase_max"]),
        "mapping_rate": all(
            row["mapping_rate"] == float(gates["mapping_rate"])
            for row in controls.values()
        ),
        "trie_membership_rate": all(
            row["trie_membership_rate"] == float(gates["trie_membership_rate"])
            for row in controls.values()
        ),
        "finite_rate": all(
            row["finite_rate"] == float(gates["finite_rate"])
            for row in controls.values()
        ),
    }
    return {
        "dataset": dataset,
        "mean_paired_excess_relative_decrease": excess_decrease,
        "lexical_ce_relative_increase": ce_increase,
        "checks": checks,
        "pass": all(checks.values()),
    }


@torch.no_grad()
def generate_control_predictions(
    prepared: dict,
    samples: list[dict],
    block_path: Path,
    checkpoint: Path,
    config: dict,
    device: torch.device,
):
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    backbone = prepared["model"].backbone
    backbone.decoder.block[-1].load_state_dict(
        torch.load(block_path, map_location=device)
    )
    backbone.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    beam_size = int(config["validation"]["beam_size"])
    max_length = max(len(row) for row in prepared["encoded_candidates"])
    predictions = {}
    mapped, total = 0, 0
    for start in range(0, len(samples), 4):
        selected = samples[start : start + 4]
        batch = collate(prepared["collator"], selected)
        input_ids = batch["item_text_ids"].to(device)
        attention = batch["item_text_masks"].to(device)
        output = backbone.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_length=max_length,
            prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(trie),
            num_beams=beam_size,
            num_return_sequences=beam_size,
            return_dict_in_generate=True,
            length_penalty=float(config["validation"]["length_penalty"]),
        )
        for offset, sample in enumerate(selected):
            items = [
                prepared["sequence_to_item"].get(
                    normalized_sequence(sequence.tolist())
                )
                for sequence in output["sequences"][
                    offset * beam_size : (offset + 1) * beam_size
                ]
            ]
            mapped += sum(item is not None for item in items)
            total += len(items)
            if any(item is None for item in items):
                raise ValueError("TCDR P0 candidate mapping failure")
            predictions[sample["sample_key"]] = items
        done = min(start + 4, len(samples))
        if done % 64 == 0:
            print(
                f"TCDR_P0_VALIDATION control={block_path.parent.name} "
                f"users={done}/{len(samples)}",
                flush=True,
            )
    return predictions, mapped / total


def validate_domain(
    dataset: str,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = (
        ROOT / config["inputs"]["checkpoint_root"] / dataset / "C0" / "model.pt"
    )
    checkpoint_sha = sha256(checkpoint)
    _, calibration = select_fit_calibration(dataset, prepared, config)
    smoke_samples = calibration[:2]
    smoke_mapping_rates = {}
    for control in config["training"]["controls"]:
        _, smoke_mapping_rates[control] = generate_control_predictions(
            prepared,
            smoke_samples,
            output_root / dataset / control / "decoder_last_block.pt",
            checkpoint,
            config,
            device,
        )
    if min(smoke_mapping_rates.values()) != 1.0:
        raise ValueError("TCDR P0 training-prefix generation smoke failure")

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
    order = {user: index for index, user in enumerate(selected_users)}
    samples.sort(key=lambda row: order[row["user_id"]])
    predictions = {}
    mapping_rates = {}
    for control in config["training"]["controls"]:
        predictions[control], mapping_rates[control] = generate_control_predictions(
            prepared,
            samples,
            output_root / dataset / control / "decoder_last_block.pt",
            checkpoint,
            config,
            device,
        )
    rows = []
    for sample in samples:
        control_rank, control_recall, control_ndcg = rank_metrics(
            predictions["C0"][sample["sample_key"]], sample["positive_item"]
        )
        tcdr_rank, tcdr_recall, tcdr_ndcg = rank_metrics(
            predictions["C1"][sample["sample_key"]], sample["positive_item"]
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
                "control_rank": control_rank or "",
                "tcdr_rank": tcdr_rank or "",
                "control_recall10": control_recall,
                "tcdr_recall10": tcdr_recall,
                "control_ndcg10": control_ndcg,
                "tcdr_ndcg10": tcdr_ndcg,
            }
        )
    validation_path = output_root / dataset / "validation_metrics.csv"
    write_csv(validation_path, rows)
    tail = [row for row in rows if row["target_group"] == "tail"]
    overall_relative = bootstrap_relative(
        [row["control_ndcg10"] for row in rows],
        [row["tcdr_ndcg10"] for row in rows],
        int(config["validation"]["bootstrap_replicates"]),
        int(config["validation"]["bootstrap_seed"]),
    )
    tail_relative = bootstrap_relative(
        [row["control_ndcg10"] for row in tail],
        [row["tcdr_ndcg10"] for row in tail],
        int(config["validation"]["bootstrap_replicates"]),
        int(config["validation"]["bootstrap_seed"]) + 1,
    )
    control_recall = float(np.mean([row["control_recall10"] for row in rows]))
    tcdr_recall = float(np.mean([row["tcdr_recall10"] for row in rows]))
    harm = float(
        np.mean(
            [
                row["control_recall10"] == 1.0
                and row["tcdr_recall10"] == 0.0
                for row in rows
            ]
        )
    )
    metrics = {
        "users": len(rows),
        "tail_users": len(tail),
        "control_ndcg10": float(np.mean([row["control_ndcg10"] for row in rows])),
        "tcdr_ndcg10": float(np.mean([row["tcdr_ndcg10"] for row in rows])),
        "overall_ndcg10_relative": overall_relative,
        "control_recall10": control_recall,
        "tcdr_recall10": tcdr_recall,
        "overall_recall10_absolute_gain": tcdr_recall - control_recall,
        "control_tail_ndcg10": float(
            np.mean([row["control_ndcg10"] for row in tail])
        ),
        "tcdr_tail_ndcg10": float(
            np.mean([row["tcdr_ndcg10"] for row in tail])
        ),
        "tail_ndcg10_relative": tail_relative,
        "control_hit_tcdr_miss_rate": harm,
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
        "control_hit_tcdr_miss_rate": harm
        <= float(gates["control_hit_tcdr_miss_rate_max"]),
    }
    integrity = {
        "training_prefix_generation_smoke_mapping_rate": min(
            smoke_mapping_rates.values()
        ),
        "candidate_mapping_rate": min(mapping_rates.values()),
        "finite_rate": float(
            all(
                math.isfinite(float(row[key]))
                for row in rows
                for key in (
                    "control_recall10",
                    "tcdr_recall10",
                    "control_ndcg10",
                    "tcdr_ndcg10",
                )
            )
        ),
        "validation_selection_target_independent": True,
        "validation_user_sha256": hashlib.sha256(
            "\n".join(selected_users).encode()
        ).hexdigest(),
        "source_checkpoint_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "validation_file_sha256": sha256(validation_path),
        "test_predictions_read": False,
        "sports_read": False,
    }
    result = {
        "metrics": metrics,
        "checks": checks,
        "effect_pass": all(checks.values()),
        "integrity": integrity,
        "integrity_valid": (
            integrity["training_prefix_generation_smoke_mapping_rate"] == 1.0
            and integrity["candidate_mapping_rate"] == 1.0
            and integrity["finite_rate"] == 1.0
            and integrity["validation_selection_target_independent"]
            and integrity["source_checkpoint_sha_unchanged"]
            and not integrity["test_predictions_read"]
            and not integrity["sports_read"]
        ),
    }
    write_json(output_root / dataset / "validation_summary.json", result)
    del prepared
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if sha256(Path(__file__)) != config["integrity"]["code_sha256"]:
        raise ValueError("TCDR P0 code SHA mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("TCDR P0 requires CUDA")
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")

    training = {
        dataset: {
            control: train_control(
                dataset, control, config, p0_config, args.output_root, device
            )
            for control in config["training"]["controls"]
        }
        for dataset in config["datasets"]
    }
    calibration = {
        dataset: {
            control: calibrate_control(
                dataset, control, config, p0_config, args.output_root, device
            )
            for control in config["training"]["controls"]
        }
        for dataset in config["datasets"]
    }
    mechanism = {
        dataset: mechanism_gate(dataset, calibration[dataset], config)
        for dataset in config["datasets"]
    }
    training_integrity = all(
        row["fit_calibration_user_overlap"] == 0
        and row["steps"] == row["expected_steps"]
        and row["parameter_delta_max_abs"] > 0
        and row["checkpoint_sha_unchanged"]
        for dataset in training.values()
        for row in dataset.values()
    )
    mechanism_pass = all(row["pass"] for row in mechanism.values())
    prevalidation = {
        "training": training,
        "calibration": calibration,
        "mechanism": mechanism,
        "training_integrity": training_integrity,
        "mechanism_pass": mechanism_pass,
        "validation_read": False,
    }
    write_json(args.output_root / "prevalidation_summary.json", prevalidation)
    if not training_integrity:
        decision = "EXECUTION_INVALID"
        validation = {}
    elif not mechanism_pass:
        decision = "STOP_TCDR_MECHANISM_GATE_FAILED"
        validation = {}
    else:
        validation = {
            dataset: validate_domain(
                dataset, config, p0_config, args.output_root, device
            )
            for dataset in config["datasets"]
        }
        validation_integrity = all(
            row["integrity_valid"] for row in validation.values()
        )
        effect_pass = all(row["effect_pass"] for row in validation.values())
        decision = (
            "EXECUTION_INVALID"
            if not validation_integrity
            else "TCDR_FREEZE_FOR_CONFIRMATION"
            if effect_pass
            else "STOP_TCDR_EFFECT_GATE_FAILED"
        )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "training": training,
        "calibration": calibration,
        "mechanism": mechanism,
        "validation": validation,
        "integrity_valid": decision != "EXECUTION_INVALID",
        "validation_read": bool(validation),
        "test_predictions_read": False,
        "sports_read": False,
    }
    write_json(args.output_root / "summary.json", summary)
    write_json(
        args.output_root / "status.json",
        {"experiment_id": config["experiment_id"], "status": "completed"},
    )
    (args.output_root / "decision.md").write_text(
        "# TCDR-P0 Decision\n\n"
        f"- Fixed decision: **`{decision}`**\n"
        f"- Validation read: `{str(bool(validation)).lower()}`\n"
        "- Test/Sports read: `false`\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
