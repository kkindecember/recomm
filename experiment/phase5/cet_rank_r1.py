#!/usr/bin/env python3
"""CET Rank-R1: direct sequence-rank consistency correctness smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gcdh_p0 import (  # noqa: E402
    collate,
    prepare,
    read_users,
    sha256,
    stable_sha,
    write_json,
)
from experiment.phase5.cet_c1 import structured_passage_mask  # noqa: E402
from experiment.phase5.cet_c2 import backbone_forward  # noqa: E402
from experiment.phase5.cet_c2_optimization_audit import (  # noqa: E402
    ordered_calibration_samples,
)
from experiment.phase5.cet_rank_r0 import generate_ranked  # noqa: E402
from experiment.phase5.cet_rank_r0g import (  # noqa: E402
    flatten_gradients,
    jensen_shannon,
    sequence_rank_scores,
)


def load_configs(path: Path) -> tuple[dict, dict]:
    config = json.loads(path.read_text())
    p0 = json.loads(
        (ROOT / "artifacts/phase4/configs/gcdh_p0_preregistered.json").read_text()
    )
    return config, p0


def ordered_file_users(path: Path) -> list[str]:
    users = [value.strip() for value in path.read_text().splitlines() if value.strip()]
    if len(users) != len(set(users)):
        raise ValueError(f"duplicate users in {path}")
    return users


def exclusion_paths(dataset: str, config: dict) -> list[Path]:
    paths = [
        ROOT / pattern.format(dataset=dataset)
        for pattern in config["data"]["excluded_user_files"]
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing exclusion files: {missing}")
    return paths


def excluded_users(dataset: str, config: dict) -> set[str]:
    result: set[str] = set()
    for path in exclusion_paths(dataset, config):
        result.update(read_users(path))
    return result


def make_splits(config: dict, p0: dict, config_path: Path) -> dict:
    split_root = ROOT / config["data"]["split_root"]
    results = {}
    total = int(config["data"]["fit_users"]) + int(
        config["data"]["evaluation_users"]
    )
    for dataset in config["datasets"]:
        prepared = prepare(dataset, p0, torch.device("cpu"))
        excluded = excluded_users(dataset, config)
        samples = ordered_calibration_samples(
            dataset,
            prepared["sequences"],
            prepared["item2input"],
            prepared["item2lexid"],
            excluded,
            total,
            int(config["data"]["minimum_history_items"]),
            config["data"]["selection_salt"],
        )
        fit_count = int(config["data"]["fit_users"])
        subsets = {
            "fit": [row["user_id"] for row in samples[:fit_count]],
            "evaluation": [row["user_id"] for row in samples[fit_count:]],
        }
        if set(subsets["fit"]) & set(subsets["evaluation"]):
            raise ValueError(f"{dataset}: fit/evaluation overlap")
        if (set(subsets["fit"]) | set(subsets["evaluation"])) & excluded:
            raise ValueError(f"{dataset}: excluded user entered Rank-R1")
        subset_manifest = {}
        dataset_root = split_root / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        for name, users in subsets.items():
            path = dataset_root / f"{name}_users.txt"
            path.write_text("\n".join(users) + "\n")
            subset_manifest[name] = {
                "users": len(users),
                "user_sha256": stable_sha(set(users)),
                "file_sha256": sha256(path),
            }
        manifest = {
            "experiment_id": config["experiment_id"],
            "dataset": dataset,
            "selection_salt": config["data"]["selection_salt"],
            "selection": "SHA256(salt|dataset|user), ascending then fixed 64/64 split",
            "target": "sequence[-3]",
            "history": "sequence[:-3][-20:]",
            "excluded_users": len(excluded),
            "excluded_file_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in exclusion_paths(dataset, config)
            },
            "subsets": subset_manifest,
            "fit_evaluation_disjoint": True,
            "selection_uses_candidate_target": False,
            "validation_target_read": False,
            "test_read": False,
            "sports_read": False,
        }
        write_json(dataset_root / "manifest.json", manifest)
        results[dataset] = manifest
        del prepared
    frozen = {
        "experiment_id": config["experiment_id"],
        "code_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "datasets": results,
        "frozen_before_gpu_run": True,
    }
    write_json(split_root / "frozen_manifest.json", frozen)
    return frozen


def load_samples(
    dataset: str, subset: str, prepared: dict, config: dict
) -> list[dict]:
    dataset_root = ROOT / config["data"]["split_root"] / dataset
    path = dataset_root / f"{subset}_users.txt"
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    users = ordered_file_users(path)
    expected = manifest["subsets"][subset]
    if sha256(path) != expected["file_sha256"]:
        raise ValueError(f"{dataset}/{subset}: file SHA mismatch")
    if stable_sha(set(users)) != expected["user_sha256"]:
        raise ValueError(f"{dataset}/{subset}: user SHA mismatch")
    excluded = excluded_users(dataset, config)
    if set(users) & excluded:
        raise ValueError(f"{dataset}/{subset}: exclusion failure")
    total = int(config["data"]["fit_users"]) + int(
        config["data"]["evaluation_users"]
    )
    replay = ordered_calibration_samples(
        dataset,
        prepared["sequences"],
        prepared["item2input"],
        prepared["item2lexid"],
        excluded,
        total,
        int(config["data"]["minimum_history_items"]),
        config["data"]["selection_salt"],
    )
    by_user = {row["user_id"]: row for row in replay}
    if any(user not in by_user for user in users):
        raise ValueError(f"{dataset}/{subset}: deterministic replay failure")
    return [by_user[user] for user in users]


def prepare_view(
    dataset: str,
    sample: dict,
    prepared: dict,
    config: dict,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = collate(prepared["collator"], [sample])
    for key in ("item_text_ids", "item_text_masks", "target_ids"):
        batch[key] = batch[key].to(device)
    clean_attention = batch["item_text_masks"].bool()
    perturbed_attention, decisions = structured_passage_mask(
        clean_attention,
        [sample],
        dataset,
        int(config["views"]["mask_seed"]),
        float(config["views"]["mask_probability"]),
    )
    _, altered = structured_passage_mask(
        clean_attention,
        [dict(sample, positive_item="__altered__")],
        dataset,
        int(config["views"]["mask_seed"]),
        float(config["views"]["mask_probability"]),
    )
    if not torch.equal(decisions, altered):
        raise ValueError("Rank-R1 mask depends on target")
    if not torch.equal(clean_attention[:, 0], perturbed_attention[:, 0]):
        raise ValueError("Rank-R1 changed coarse passage")
    if clean_attention.shape[1] > 1 and not torch.equal(
        clean_attention[:, 1], perturbed_attention[:, 1]
    ):
        raise ValueError("Rank-R1 changed newest fine passage")
    return batch, clean_attention, perturbed_attention, decisions


@torch.inference_mode()
def freeze_candidate_bank(
    dataset: str,
    samples: list[dict],
    backbone,
    prepared: dict,
    config: dict,
    device: torch.device,
) -> dict[str, list[str]]:
    backbone.eval()
    bank = {}
    beam_size = int(config["surrogate"]["beam_size"])
    for index, sample in enumerate(samples, 1):
        batch, clean_attention, perturbed_attention, _ = prepare_view(
            dataset, sample, prepared, config, device
        )
        clean = generate_ranked(
            backbone, prepared, batch["item_text_ids"], clean_attention, beam_size
        )
        perturbed = generate_ranked(
            backbone,
            prepared,
            batch["item_text_ids"],
            perturbed_attention,
            beam_size,
        )
        union = list(dict.fromkeys(clean + perturbed))
        if not int(config["integrity"]["candidate_union_size_min"]) <= len(
            union
        ) <= int(config["integrity"]["candidate_union_size_max"]):
            raise ValueError("Rank-R1 candidate union outside frozen bounds")
        bank[sample["user_id"]] = union
        if index % 16 == 0:
            print(
                f"R1_CANDIDATES dataset={dataset} users={index}/{len(samples)}",
                flush=True,
            )
    return bank


def gradient_norms_for_gamma(
    dataset: str,
    fit_samples: list[dict],
    candidate_bank: dict[str, list[str]],
    backbone,
    prepared: dict,
    config: dict,
    trainable: list[torch.nn.Parameter],
    device: torch.device,
) -> tuple[float, dict]:
    backbone.eval()
    ratios = []
    rank_norms = []
    ce_norms = []
    masked_users = 0
    for index, sample in enumerate(fit_samples, 1):
        batch, clean_attention, perturbed_attention, decisions = prepare_view(
            dataset, sample, prepared, config, device
        )
        if int(decisions.sum()) == 0:
            continue
        masked_users += 1
        clean = backbone_forward(backbone, batch, clean_attention)
        perturbed = backbone_forward(backbone, batch, perturbed_attention)
        ce_loss = clean.loss + perturbed.loss
        ce_gradient = flatten_gradients(
            torch.autograd.grad(ce_loss, trainable, retain_graph=False)
        )
        union = candidate_bank[sample["user_id"]]
        clean_scores = sequence_rank_scores(
            backbone, prepared, batch["item_text_ids"], clean_attention, union
        )
        perturbed_scores = sequence_rank_scores(
            backbone,
            prepared,
            batch["item_text_ids"],
            perturbed_attention,
            union,
        )
        rank_loss = jensen_shannon(
            clean_scores / float(config["surrogate"]["rank_temperature"]),
            perturbed_scores / float(config["surrogate"]["rank_temperature"]),
        )
        rank_gradient = flatten_gradients(
            torch.autograd.grad(rank_loss, trainable, retain_graph=False)
        )
        ce_norm = float(ce_gradient.norm())
        rank_norm = float(rank_gradient.norm())
        threshold = float(config["gamma"]["nonzero_gradient_norm_threshold"])
        if not all(math.isfinite(value) for value in (ce_norm, rank_norm)):
            raise ValueError("non-finite Rank-R1 gamma gradient")
        if ce_norm <= threshold or rank_norm <= threshold:
            raise ValueError("zero Rank-R1 gamma gradient")
        ce_norms.append(ce_norm)
        rank_norms.append(rank_norm)
        ratios.append(ce_norm / rank_norm)
        if index % 16 == 0:
            print(
                f"R1_GAMMA dataset={dataset} users={index}/{len(fit_samples)}",
                flush=True,
            )
    if masked_users < int(config["gates"]["minimum_masked_users_per_subset"]):
        raise ValueError("insufficient masked fit users for gamma calibration")
    gamma = float(config["gamma"]["target_weighted_rank_to_ce_gradient_ratio"]) * float(
        np.median(ratios)
    )
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("invalid frozen gamma")
    return gamma, {
        "formula": "target_ratio * median(masked-user CE gradient norm / rank-JS gradient norm)",
        "masked_fit_users": masked_users,
        "median_ce_gradient_norm": float(np.median(ce_norms)),
        "median_rank_gradient_norm": float(np.median(rank_norms)),
        "median_ce_to_rank_gradient_norm_ratio": float(np.median(ratios)),
        "target_weighted_rank_to_ce_gradient_ratio": float(
            config["gamma"]["target_weighted_rank_to_ce_gradient_ratio"]
        ),
        "gamma": gamma,
    }


@torch.inference_mode()
def evaluate(
    dataset: str,
    samples: list[dict],
    candidate_bank: dict[str, list[str]],
    backbone,
    prepared: dict,
    config: dict,
    device: torch.device,
) -> dict:
    backbone.eval()
    rank_losses = []
    masked_rank_losses = []
    overlaps = []
    masked_overlaps = []
    clean_ce_weighted = 0.0
    label_tokens_total = 0
    masked_users = 0
    no_mask_identity = True
    beam_size = int(config["evaluation"]["overlap_beam_size"])
    tolerance = float(config["integrity"]["identity_absolute_tolerance"])
    for index, sample in enumerate(samples, 1):
        batch, clean_attention, perturbed_attention, decisions = prepare_view(
            dataset, sample, prepared, config, device
        )
        union = candidate_bank[sample["user_id"]]
        clean = backbone_forward(backbone, batch, clean_attention)
        perturbed = backbone_forward(backbone, batch, perturbed_attention)
        clean_scores = sequence_rank_scores(
            backbone, prepared, batch["item_text_ids"], clean_attention, union
        )
        perturbed_scores = sequence_rank_scores(
            backbone,
            prepared,
            batch["item_text_ids"],
            perturbed_attention,
            union,
        )
        rank_loss = float(
            jensen_shannon(
                clean_scores / float(config["surrogate"]["rank_temperature"]),
                perturbed_scores / float(config["surrogate"]["rank_temperature"]),
            )
        )
        clean_ranked = generate_ranked(
            backbone, prepared, batch["item_text_ids"], clean_attention, beam_size
        )
        perturbed_ranked = generate_ranked(
            backbone,
            prepared,
            batch["item_text_ids"],
            perturbed_attention,
            beam_size,
        )
        overlap = len(set(clean_ranked) & set(perturbed_ranked)) / beam_size
        label_tokens = int((batch["target_ids"] != -100).sum())
        clean_ce_weighted += float(clean.loss) * label_tokens
        label_tokens_total += label_tokens
        rank_losses.append(rank_loss)
        overlaps.append(overlap)
        if int(decisions.sum()) > 0:
            masked_users += 1
            masked_rank_losses.append(rank_loss)
            masked_overlaps.append(overlap)
        else:
            no_mask_identity = no_mask_identity and (
                clean_ranked == perturbed_ranked
                and float((clean_scores - perturbed_scores).abs().max()) <= tolerance
                and abs(rank_loss) <= tolerance
                and abs(float(clean.loss) - float(perturbed.loss)) <= tolerance
            )
        if not all(
            math.isfinite(value)
            for value in (rank_loss, overlap, float(clean.loss), float(perturbed.loss))
        ):
            raise ValueError("non-finite Rank-R1 evaluation")
        if index % 16 == 0:
            print(
                f"R1_EVAL dataset={dataset} users={index}/{len(samples)}",
                flush=True,
            )
    if masked_users < int(config["gates"]["minimum_masked_users_per_subset"]):
        raise ValueError("insufficient masked evaluation users")
    return {
        "users": len(samples),
        "masked_users": masked_users,
        "mean_rank_js_all_users": float(np.mean(rank_losses)),
        "mean_rank_js_masked_users": float(np.mean(masked_rank_losses)),
        "clean_lexical_ce": clean_ce_weighted / label_tokens_total,
        "mean_top10_overlap_all_users": float(np.mean(overlaps)),
        "mean_top10_overlap_masked_users": float(np.mean(masked_overlaps)),
        "no_mask_identity": no_mask_identity,
    }


def train(
    dataset: str,
    fit_samples: list[dict],
    candidate_bank: dict[str, list[str]],
    gamma: float,
    backbone,
    prepared: dict,
    config: dict,
    trainable: list[torch.nn.Parameter],
    device: torch.device,
) -> dict:
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    batch_size = int(config["optimization"]["batch_size"])
    steps = int(config["optimization"]["steps"])
    indices = list(range(len(fit_samples)))
    random.Random(int(config["seed"])).shuffle(indices)
    expected = steps * batch_size
    if expected != len(fit_samples):
        raise ValueError("Rank-R1 frozen steps*batch_size must cover fit users once")
    losses = []
    ce_losses = []
    rank_losses = []
    gradient_norms = []
    masked_users = 0
    backbone.train()
    started = time.time()
    for step in range(steps):
        selected = indices[step * batch_size : (step + 1) * batch_size]
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        ce_total = 0.0
        rank_total = 0.0
        for sample_index in selected:
            sample = fit_samples[sample_index]
            batch, clean_attention, perturbed_attention, decisions = prepare_view(
                dataset, sample, prepared, config, device
            )
            masked_users += int(bool(decisions.any()))
            clean = backbone_forward(backbone, batch, clean_attention)
            perturbed = backbone_forward(backbone, batch, perturbed_attention)
            ce = clean.loss + perturbed.loss
            union = candidate_bank[sample["user_id"]]
            clean_scores = sequence_rank_scores(
                backbone, prepared, batch["item_text_ids"], clean_attention, union
            )
            perturbed_scores = sequence_rank_scores(
                backbone,
                prepared,
                batch["item_text_ids"],
                perturbed_attention,
                union,
            )
            rank = jensen_shannon(
                clean_scores / float(config["surrogate"]["rank_temperature"]),
                perturbed_scores / float(config["surrogate"]["rank_temperature"]),
            )
            total = total + (ce + gamma * rank) / batch_size
            ce_total += float(ce.detach()) / batch_size
            rank_total += float(rank.detach()) / batch_size
        if not torch.isfinite(total):
            raise ValueError("non-finite Rank-R1 training loss")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config["optimization"]["gradient_clip_norm"])
        )
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise ValueError("non-finite or zero Rank-R1 gradient")
        optimizer.step()
        losses.append(float(total.detach()))
        ce_losses.append(ce_total)
        rank_losses.append(rank_total)
        gradient_norms.append(float(norm))
        if (step + 1) % 4 == 0:
            print(
                f"R1_TRAIN dataset={dataset} step={step+1}/{steps} "
                f"loss={losses[-1]:.6f} rank_js={rank_total:.6f} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    return {
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "fit_users_seen_once": expected,
        "masked_fit_users_seen": masked_users,
        "mean_training_loss": float(np.mean(losses)),
        "mean_training_ce": float(np.mean(ce_losses)),
        "mean_training_rank_js": float(np.mean(rank_losses)),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "wall_time_seconds": time.time() - started,
    }


def run_dataset(
    dataset: str,
    config: dict,
    p0: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0, device)
    backbone = prepared["model"].backbone
    checkpoint = ROOT / config["checkpoint"]["root"] / dataset / "C1/model.pt"
    checkpoint_sha_before = sha256(checkpoint)
    if checkpoint_sha_before != config["checkpoint"]["sha256"][dataset]:
        raise ValueError(f"{dataset}: Rank-R1 C1 checkpoint SHA mismatch")
    backbone.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    fit_samples = load_samples(dataset, "fit", prepared, config)
    evaluation_samples = load_samples(dataset, "evaluation", prepared, config)
    if {row["user_id"] for row in fit_samples} & {
        row["user_id"] for row in evaluation_samples
    }:
        raise ValueError("Rank-R1 fit/evaluation overlap")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    trainable = list(backbone.decoder.block[-1].parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    initial_parameters = [parameter.detach().clone() for parameter in trainable]

    all_samples = fit_samples + evaluation_samples
    candidate_bank = freeze_candidate_bank(
        dataset, all_samples, backbone, prepared, config, device
    )
    dataset_root = output_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    candidate_path = dataset_root / "candidate_bank.json"
    write_json(candidate_path, candidate_bank)
    candidate_sha = sha256(candidate_path)
    gamma, gamma_record = gradient_norms_for_gamma(
        dataset,
        fit_samples,
        candidate_bank,
        backbone,
        prepared,
        config,
        trainable,
        device,
    )
    gamma_path = dataset_root / "gamma.json"
    write_json(gamma_path, gamma_record)
    gamma_sha = sha256(gamma_path)
    initial_evaluation = evaluate(
        dataset,
        evaluation_samples,
        candidate_bank,
        backbone,
        prepared,
        config,
        device,
    )
    training = train(
        dataset,
        fit_samples,
        candidate_bank,
        gamma,
        backbone,
        prepared,
        config,
        trainable,
        device,
    )
    final_evaluation = evaluate(
        dataset,
        evaluation_samples,
        candidate_bank,
        backbone,
        prepared,
        config,
        device,
    )
    if sha256(candidate_path) != candidate_sha or sha256(gamma_path) != gamma_sha:
        raise ValueError("Rank-R1 candidate bank or gamma changed after freezing")
    parameter_change = max(
        float((after.detach() - before).abs().max())
        for before, after in zip(initial_parameters, trainable)
    )
    checkpoint_out = dataset_root / "decoder_last_layer.pt"
    torch.save(backbone.decoder.block[-1].state_dict(), checkpoint_out)
    saved_state = torch.load(checkpoint_out, map_location=device)
    with torch.no_grad():
        next(backbone.decoder.block[-1].parameters()).add_(1.0)
    backbone.decoder.block[-1].load_state_dict(saved_state, strict=True)
    reload_difference = max(
        float(
            (parameter.detach() - saved_state[name].to(parameter.device)).abs().max()
        )
        for name, parameter in backbone.decoder.block[-1].named_parameters()
    )
    rank_decrease = (
        initial_evaluation["mean_rank_js_masked_users"]
        - final_evaluation["mean_rank_js_masked_users"]
    ) / initial_evaluation["mean_rank_js_masked_users"]
    clean_ce_change = (
        final_evaluation["clean_lexical_ce"]
        - initial_evaluation["clean_lexical_ce"]
    ) / initial_evaluation["clean_lexical_ce"]
    overlap_change = (
        final_evaluation["mean_top10_overlap_masked_users"]
        - initial_evaluation["mean_top10_overlap_masked_users"]
    )
    integrity_checks = {
        "fit_evaluation_disjoint": True,
        "candidate_mapping_100_percent": len(candidate_bank) == len(all_samples),
        "candidate_union_size_valid": all(
            int(config["integrity"]["candidate_union_size_min"]) <= len(value)
            <= int(config["integrity"]["candidate_union_size_max"])
            for value in candidate_bank.values()
        ),
        "candidate_bank_frozen": sha256(candidate_path) == candidate_sha,
        "gamma_frozen_before_updates": sha256(gamma_path) == gamma_sha,
        "no_mask_identity": (
            initial_evaluation["no_mask_identity"]
            and final_evaluation["no_mask_identity"]
        ),
        "finite_nonzero_gradient": (
            math.isfinite(training["gradient_norm_min"])
            and training["gradient_norm_min"] > 0
        ),
        "parameter_change": parameter_change > 0,
        "checkpoint_reload": reload_difference <= float(
            config["integrity"]["identity_absolute_tolerance"]
        ),
        "source_checkpoint_sha_unchanged": sha256(checkpoint)
        == checkpoint_sha_before,
        "targets_sealed": True,
    }
    optimization_checks = {
        "minimum_masked_fit_users": training["masked_fit_users_seen"]
        >= int(config["gates"]["minimum_masked_users_per_subset"]),
        "minimum_masked_evaluation_users": final_evaluation["masked_users"]
        >= int(config["gates"]["minimum_masked_users_per_subset"]),
        "evaluation_rank_js_relative_decrease": rank_decrease
        >= float(config["gates"]["rank_js_relative_decrease_min"]),
        "clean_ce_safety": clean_ce_change
        <= float(config["gates"]["clean_ce_relative_increase_max"]),
        "top10_overlap_strict_improvement": overlap_change
        > float(config["gates"]["top10_overlap_absolute_change_min"]),
    }
    result = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "status": "COMPLETED",
        "gamma": gamma_record,
        "candidate_bank_sha256": candidate_sha,
        "candidate_bank_users": len(candidate_bank),
        "initial_evaluation": initial_evaluation,
        "training": training,
        "final_evaluation": final_evaluation,
        "rank_js_relative_decrease": rank_decrease,
        "clean_ce_relative_change": clean_ce_change,
        "top10_overlap_absolute_change": overlap_change,
        "parameter_max_abs_change": parameter_change,
        "decoder_checkpoint": str(checkpoint_out.relative_to(ROOT)),
        "decoder_checkpoint_sha256": sha256(checkpoint_out),
        "checkpoint_reload_max_abs_difference": reload_difference,
        "source_checkpoint_sha256": checkpoint_sha_before,
        "integrity_checks": integrity_checks,
        "optimization_checks": optimization_checks,
        "integrity_pass": all(integrity_checks.values()),
        "optimization_pass": all(optimization_checks.values()),
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(dataset_root / "summary.json", result)
    del prepared, backbone
    torch.cuda.empty_cache()
    return result


def route_decision(results: dict[str, dict]) -> str:
    if not all(result["integrity_pass"] for result in results.values()):
        return "INVALID_R1_FIX_AND_EXACT_RERUN"
    if all(result["optimization_pass"] for result in results.values()):
        return "CET_R1_RANK_CONSISTENCY_PASS"
    return "STOP_CET_RANK_NOT_OPTIMIZABLE"


def analyze(config: dict, output_root: Path) -> dict:
    results = {
        dataset: json.loads((output_root / dataset / "summary.json").read_text())
        for dataset in config["datasets"]
    }
    decision = route_decision(results)
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "results": results,
        "integrity_pass": all(
            result["integrity_pass"] for result in results.values()
        ),
        "optimization_pass": all(
            result["optimization_pass"] for result in results.values()
        ),
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("make-splits", "run", "analyze"), required=True
    )
    parser.add_argument("--dataset", choices=("Toys", "Beauty"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config, p0 = load_configs(args.config)
    actual_code_sha = sha256(Path(__file__))
    registered_code_sha = config["integrity"]["code_sha256"]
    if (
        registered_code_sha != "PENDING_FREEZE"
        and registered_code_sha != actual_code_sha
    ):
        raise ValueError(
            f"Rank-R1 code SHA mismatch: actual={actual_code_sha} "
            f"registered={registered_code_sha}"
        )
    if args.stage == "make-splits":
        print(
            json.dumps(make_splits(config, p0, args.config), ensure_ascii=False, indent=2)
        )
        return 0
    frozen = json.loads(
        (ROOT / config["data"]["split_root"] / "frozen_manifest.json").read_text()
    )
    if (
        frozen["code_sha256"] != actual_code_sha
        or frozen["config_sha256"] != sha256(args.config)
    ):
        raise ValueError("Rank-R1 frozen code/config SHA mismatch")
    if args.stage == "analyze":
        print(json.dumps(analyze(config, args.output_root), ensure_ascii=False, indent=2))
        return 0
    if args.dataset is None:
        parser.error("--dataset is required for run")
    if not torch.cuda.is_available():
        raise RuntimeError("CET Rank-R1 requires CUDA")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    result = run_dataset(
        args.dataset, config, p0, args.output_root, torch.device("cuda:0")
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
