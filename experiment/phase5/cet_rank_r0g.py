#!/usr/bin/env python3
"""CET Rank-R0G: local-KL versus direct-rank gradient alignment audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from experiment.phase5.cet_c2 import backbone_forward, candidate_sequences  # noqa: E402
from experiment.phase5.cet_c2_optimization_audit import (  # noqa: E402
    legal_child_symmetric_kl,
    ordered_calibration_samples,
)
from experiment.phase5.cet_rank_r0 import generate_ranked  # noqa: E402
from utils import generation_trie as gt  # noqa: E402


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
    users: set[str] = set()
    for path in exclusion_paths(dataset, config):
        users.update(read_users(path))
    return users


def make_splits(config: dict, p0: dict, config_path: Path) -> dict:
    split_root = ROOT / config["data"]["split_root"]
    results = {}
    for dataset in config["datasets"]:
        prepared = prepare(dataset, p0, torch.device("cpu"))
        excluded = excluded_users(dataset, config)
        samples = ordered_calibration_samples(
            dataset,
            prepared["sequences"],
            prepared["item2input"],
            prepared["item2lexid"],
            excluded,
            int(config["data"]["users_per_dataset"]),
            int(config["data"]["minimum_history_items"]),
            config["data"]["selection_salt"],
        )
        users = [row["user_id"] for row in samples]
        if set(users) & excluded:
            raise ValueError(f"{dataset}: excluded user entered Rank-R0G")
        path = split_root / dataset / "audit_users.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(users) + "\n")
        manifest = {
            "experiment_id": config["experiment_id"],
            "dataset": dataset,
            "users": len(users),
            "selection_salt": config["data"]["selection_salt"],
            "selection": "SHA256(salt|dataset|user), ascending",
            "target": "sequence[-3]",
            "history": "sequence[:-3][-20:]",
            "user_sha256": stable_sha(set(users)),
            "file_sha256": sha256(path),
            "excluded_users": len(excluded),
            "excluded_file_sha256": {
                str(value.relative_to(ROOT)): sha256(value)
                for value in exclusion_paths(dataset, config)
            },
            "all_prior_development_users_disjoint": True,
            "selection_uses_candidate_target": False,
            "validation_target_read": False,
            "test_read": False,
            "sports_read": False,
        }
        write_json(path.parent / "manifest.json", manifest)
        results[dataset] = manifest
        del prepared
    frozen = {
        "experiment_id": config["experiment_id"],
        "code_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "datasets": results,
        "frozen_before_gpu_audit": True,
    }
    write_json(split_root / "frozen_manifest.json", frozen)
    return frozen


def load_samples(dataset: str, prepared: dict, config: dict) -> list[dict]:
    root = ROOT / config["data"]["split_root"] / dataset
    path = root / "audit_users.txt"
    manifest = json.loads((root / "manifest.json").read_text())
    users = ordered_file_users(path)
    if sha256(path) != manifest["file_sha256"]:
        raise ValueError(f"{dataset}: Rank-R0G user file SHA mismatch")
    if stable_sha(set(users)) != manifest["user_sha256"]:
        raise ValueError(f"{dataset}: Rank-R0G user-set SHA mismatch")
    excluded = excluded_users(dataset, config)
    if set(users) & excluded:
        raise ValueError(f"{dataset}: Rank-R0G exclusion failure")
    replay = ordered_calibration_samples(
        dataset,
        prepared["sequences"],
        prepared["item2input"],
        prepared["item2lexid"],
        excluded,
        int(config["data"]["users_per_dataset"]),
        int(config["data"]["minimum_history_items"]),
        config["data"]["selection_salt"],
    )
    by_user = {row["user_id"]: row for row in replay}
    if any(user not in by_user for user in users):
        raise ValueError(f"{dataset}: deterministic sample replay failure")
    return [by_user[user] for user in users]


def jensen_shannon(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    if left_logits.shape != right_logits.shape or left_logits.ndim != 1:
        raise ValueError("rank logits must be equal-length vectors")
    left_log = F.log_softmax(left_logits, dim=0)
    right_log = F.log_softmax(right_logits, dim=0)
    left = left_log.exp()
    right = right_log.exp()
    mean_log = torch.logaddexp(left_log, right_log) - math.log(2.0)
    return 0.5 * (
        (left * (left_log - mean_log)).sum()
        + (right * (right_log - mean_log)).sum()
    )


def length_normalized_scores(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logit/label shape mismatch")
    valid = labels.ne(-100)
    safe = labels.masked_fill(~valid, 0)
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        -1, safe.unsqueeze(-1)
    ).squeeze(-1)
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("candidate has no prediction token")
    return (token_log_probs * valid).sum(dim=1) / counts


def candidate_labels(
    candidate_items: list[str], prepared: dict, device: torch.device
) -> torch.Tensor:
    item_to_sequence = dict(zip(prepared["catalog"], prepared["encoded_candidates"]))
    sequences = [item_to_sequence[item][1:] for item in candidate_items]
    if any(not sequence for sequence in sequences):
        raise ValueError("empty candidate identifier")
    width = max(len(sequence) for sequence in sequences)
    labels = torch.full(
        (len(sequences), width), -100, dtype=torch.long, device=device
    )
    for row, sequence in enumerate(sequences):
        labels[row, : len(sequence)] = torch.as_tensor(sequence, device=device)
    return labels


def sequence_rank_scores(
    backbone,
    prepared: dict,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    candidate_items: list[str],
) -> torch.Tensor:
    labels = candidate_labels(candidate_items, prepared, input_ids.device)
    count = len(candidate_items)
    output = backbone(
        input_ids=input_ids.expand(count, -1, -1).contiguous(),
        attention_mask=attention.expand(count, -1, -1).contiguous(),
        labels=labels,
        return_dict=True,
    )
    return length_normalized_scores(output.logits, labels)


def flatten_gradients(gradients: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    if any(value is None for value in gradients):
        raise ValueError("loss is disconnected from a decoder parameter")
    return torch.cat([value.detach().float().reshape(-1) for value in gradients])


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(left.norm())
    right_norm = float(right.norm())
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / (left.norm() * right.norm()))


def bootstrap_mean_interval(
    values: list[float], resamples: int, seed: int
) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def gradient_summary(rows: list[dict]) -> dict:
    cosines = [float(row["gradient_cosine"]) for row in rows]
    ratios = [
        float(row["rank_gradient_norm"]) / float(row["local_gradient_norm"])
        if float(row["local_gradient_norm"]) > 0 else float("inf")
        for row in rows
    ]
    return {
        "median_gradient_cosine": float(np.median(cosines)),
        "mean_gradient_cosine": float(np.mean(cosines)),
        "negative_cosine_prevalence": float(np.mean(np.asarray(cosines) < 0)),
        "median_gradient_norm_ratio_rank_over_local": float(np.median(ratios)),
    }


def route_decision(metrics: dict[str, dict], thresholds: dict, integrity: bool) -> str:
    if not integrity:
        return "INVALID_R0G_FIX_AND_EXACT_RERUN"
    usable = all(
        value["masked_users"] >= thresholds["minimum_masked_users_per_dataset"]
        and value["rank_loss_signal_coverage"]
        >= thresholds["minimum_rank_loss_signal_coverage"]
        and value["rank_nonzero_gradient_coverage"]
        >= thresholds["minimum_rank_gradient_nonzero_coverage"]
        for value in metrics.values()
    )
    if not usable:
        return "STOP_CET_RANK_NO_USABLE_GRADIENT"
    distinct = all(
        value["median_gradient_cosine"]
        < thresholds["distinct_median_cosine_max"]
        and value["mean_cosine_bootstrap_95ci"][1]
        < thresholds["distinct_bootstrap_ci_upper_max"]
        for value in metrics.values()
    )
    if distinct:
        return "CET_R0G_DIRECT_RANK_GRADIENT_DISTINCT"
    redundant = all(
        value["median_gradient_cosine"]
        >= thresholds["redundant_median_cosine_min"]
        and value["negative_cosine_prevalence"]
        < thresholds["redundant_negative_cosine_prevalence_max"]
        for value in metrics.values()
    )
    if redundant:
        return "STOP_CET_RANK_GRADIENT_REDUNDANT"
    return "CET_R0G_MIXED_GRADIENT_REVIEW_REQUIRED"


def audit_user(
    dataset: str,
    sample: dict,
    backbone,
    prepared: dict,
    config: dict,
    trainable: list[torch.nn.Parameter],
) -> dict:
    batch = collate(prepared["collator"], [sample])
    for key in ("item_text_ids", "item_text_masks", "target_ids"):
        batch[key] = batch[key].to(next(backbone.parameters()).device)
    clean_attention = batch["item_text_masks"].bool()
    perturbed_attention, decisions = structured_passage_mask(
        clean_attention,
        [sample],
        dataset,
        int(config["views"]["mask_seed"]),
        float(config["views"]["mask_probability"]),
    )
    _, altered_decisions = structured_passage_mask(
        clean_attention,
        [dict(sample, positive_item="__altered__")],
        dataset,
        int(config["views"]["mask_seed"]),
        float(config["views"]["mask_probability"]),
    )
    if not torch.equal(decisions, altered_decisions):
        raise ValueError("mask policy depends on target")
    if not torch.equal(clean_attention[:, 0], perturbed_attention[:, 0]):
        raise ValueError("coarse passage changed")
    if clean_attention.shape[1] > 1 and not torch.equal(
        clean_attention[:, 1], perturbed_attention[:, 1]
    ):
        raise ValueError("newest fine passage changed")
    masked = int(decisions.sum())
    beam_size = int(config["surrogates"]["beam_size"])
    with torch.no_grad():
        clean_ranked = generate_ranked(
            backbone, prepared, batch["item_text_ids"], clean_attention, beam_size
        )
        perturbed_ranked = generate_ranked(
            backbone, prepared, batch["item_text_ids"], perturbed_attention, beam_size
        )
    union = list(dict.fromkeys(clean_ranked + perturbed_ranked))
    union_min = int(config["integrity"]["candidate_union_size_min"])
    union_max = int(config["integrity"]["candidate_union_size_max"])
    if not union_min <= len(union) <= union_max:
        raise ValueError(f"candidate union size outside [{union_min},{union_max}]")

    trie = gt.Trie(prepared["encoded_candidates"])
    clean = backbone_forward(backbone, batch, clean_attention)
    perturbed = backbone_forward(backbone, batch, perturbed_attention)
    local_loss, competitive, eligible = legal_child_symmetric_kl(
        clean.logits,
        perturbed.logits,
        candidate_sequences(prepared, [sample]),
        trie,
        int(prepared["tokenizer"].eos_token_id),
        1.0,
    )
    local_grad = flatten_gradients(
        torch.autograd.grad(local_loss, trainable, retain_graph=False)
    )

    clean_scores = sequence_rank_scores(
        backbone, prepared, batch["item_text_ids"], clean_attention, union
    )
    perturbed_scores = sequence_rank_scores(
        backbone, prepared, batch["item_text_ids"], perturbed_attention, union
    )
    temperature = float(config["surrogates"]["rank_temperature"])
    rank_loss = jensen_shannon(clean_scores / temperature, perturbed_scores / temperature)
    rank_grad = flatten_gradients(
        torch.autograd.grad(rank_loss, trainable, retain_graph=False)
    )
    local_norm = float(local_grad.norm())
    rank_norm = float(rank_grad.norm())
    tolerance = float(config["measurement"]["identity_absolute_tolerance"])
    cosine = (
        0.0
        if masked == 0 and local_norm <= tolerance and rank_norm <= tolerance
        else cosine_similarity(local_grad, rank_grad)
    )
    finite = all(
        math.isfinite(value)
        for value in (float(local_loss), float(rank_loss), local_norm, rank_norm, cosine)
    )
    if not finite:
        raise ValueError("non-finite loss, gradient, or cosine")
    no_mask_identity = True
    if masked == 0:
        no_mask_identity = (
            clean_ranked == perturbed_ranked
            and float((clean_scores - perturbed_scores).abs().max()) <= tolerance
            and abs(float(local_loss)) <= tolerance
            and abs(float(rank_loss)) <= tolerance
            and local_norm <= tolerance
            and rank_norm <= tolerance
        )
    return {
        "user_id": sample["user_id"],
        "masked_passages": masked,
        "candidate_union_size": len(union),
        "candidate_mapping": True,
        "local_loss": float(local_loss.detach()),
        "rank_loss": float(rank_loss.detach()),
        "local_gradient_norm": local_norm,
        "rank_gradient_norm": rank_norm,
        "gradient_norm_ratio_rank_over_local": (
            rank_norm / local_norm if local_norm > 0 else float("inf")
        ),
        "gradient_cosine": cosine,
        "dot_product_negative": bool(float(torch.dot(local_grad, rank_grad)) < 0),
        "competitive_legal_child_steps": competitive,
        "eligible_lexical_steps": eligible,
        "same_views_for_losses": True,
        "target_independent_mask": True,
        "no_mask_identity": no_mask_identity,
    }


def audit_dataset(
    dataset: str,
    prepared: dict,
    samples: list[dict],
    config: dict,
    output_root: Path,
) -> dict:
    checkpoint = ROOT / config["checkpoint"]["root"] / dataset / "C1/model.pt"
    expected_sha = config["checkpoint"]["sha256"][dataset]
    before_sha = sha256(checkpoint)
    if before_sha != expected_sha:
        raise ValueError(f"{dataset}: C1 checkpoint SHA mismatch")
    backbone = prepared["model"].backbone
    backbone.load_state_dict(
        torch.load(checkpoint, map_location=next(backbone.parameters()).device), strict=True
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    trainable = list(backbone.decoder.block[-1].parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    rows = []
    started = time.time()
    for index, sample in enumerate(samples, 1):
        rows.append(audit_user(dataset, sample, backbone, prepared, config, trainable))
        if index % 8 == 0:
            print(
                f"R0G_PROGRESS dataset={dataset} users={index}/{len(samples)} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    masked_rows = [row for row in rows if row["masked_passages"] > 0]
    signal_threshold = float(config["measurement"]["rank_loss_signal_threshold"])
    gradient_threshold = float(
        config["measurement"]["gradient_nonzero_norm_threshold"]
    )
    metrics = gradient_summary(masked_rows) if masked_rows else {
        "median_gradient_cosine": None,
        "mean_gradient_cosine": None,
        "negative_cosine_prevalence": None,
        "median_gradient_norm_ratio_rank_over_local": None,
    }
    cosines = [row["gradient_cosine"] for row in masked_rows]
    metrics.update(
        {
            "users": len(rows),
            "masked_users": len(masked_rows),
            "rank_loss_signal_coverage": (
                float(np.mean([row["rank_loss"] > signal_threshold for row in masked_rows]))
                if masked_rows else 0.0
            ),
            "rank_nonzero_gradient_coverage": (
                float(np.mean([
                    row["rank_gradient_norm"] > gradient_threshold for row in masked_rows
                ])) if masked_rows else 0.0
            ),
            "local_nonzero_gradient_coverage": (
                float(np.mean([
                    row["local_gradient_norm"] > gradient_threshold for row in masked_rows
                ])) if masked_rows else 0.0
            ),
            "mean_local_loss": (
                float(np.mean([row["local_loss"] for row in masked_rows]))
                if masked_rows else None
            ),
            "mean_rank_loss": (
                float(np.mean([row["rank_loss"] for row in masked_rows]))
                if masked_rows else None
            ),
            "mean_cosine_bootstrap_95ci": bootstrap_mean_interval(
                cosines,
                int(config["measurement"]["bootstrap_resamples"]),
                int(config["measurement"]["bootstrap_seed"])
                + config["datasets"].index(dataset),
            ),
        }
    )
    integrity = {
        "candidate_mapping_100_percent": all(row["candidate_mapping"] for row in rows),
        "candidate_union_size_valid": all(
            int(config["integrity"]["candidate_union_size_min"])
            <= row["candidate_union_size"]
            <= int(config["integrity"]["candidate_union_size_max"])
            for row in rows
        ),
        "no_mask_identity": all(
            row["no_mask_identity"] for row in rows if row["masked_passages"] == 0
        ),
        "same_views_for_losses": all(row["same_views_for_losses"] for row in rows),
        "target_independent_mask": all(row["target_independent_mask"] for row in rows),
        "finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in ("local_loss", "rank_loss", "local_gradient_norm",
                        "rank_gradient_norm", "gradient_cosine")
        ),
        "checkpoint_sha_unchanged": sha256(checkpoint) == before_sha,
        "targets_sealed": True,
    }
    dataset_root = output_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    with (dataset_root / "per_user.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "control": "C1",
        "status": "AUDITED",
        "checkpoint_sha256": before_sha,
        "audit_user_sha256": stable_sha({row["user_id"] for row in rows}),
        "metrics": metrics,
        "integrity_checks": integrity,
        "per_user_sha256": sha256(dataset_root / "per_user.csv"),
        "wall_time_seconds": time.time() - started,
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(dataset_root / "summary.json", result)
    return result


def analyze(config: dict, output_root: Path) -> dict:
    results = {
        dataset: json.loads((output_root / dataset / "summary.json").read_text())
        for dataset in config["datasets"]
    }
    integrity_pass = all(
        all(result["integrity_checks"].values()) for result in results.values()
    )
    thresholds = {
        **config["measurement"],
        **config["routing"],
    }
    decision = route_decision(
        {dataset: result["metrics"] for dataset, result in results.items()},
        thresholds,
        integrity_pass,
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "results": results,
        "integrity_pass": integrity_pass,
        "routing_thresholds": thresholds,
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("make-splits", "audit", "analyze"), required=True)
    parser.add_argument("--dataset", choices=("Toys", "Beauty"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config, p0 = load_configs(args.config)
    actual_sha = sha256(Path(__file__))
    registered_sha = config["integrity"]["code_sha256"]
    if registered_sha != "PENDING_FREEZE" and actual_sha != registered_sha:
        raise ValueError(
            f"Rank-R0G code SHA mismatch: actual={actual_sha} registered={registered_sha}"
        )
    if args.stage == "make-splits":
        print(json.dumps(make_splits(config, p0, args.config), ensure_ascii=False, indent=2))
        return 0
    frozen = json.loads(
        (ROOT / config["data"]["split_root"] / "frozen_manifest.json").read_text()
    )
    if frozen["code_sha256"] != actual_sha or frozen["config_sha256"] != sha256(args.config):
        raise ValueError("Rank-R0G frozen code/config SHA mismatch")
    if args.stage == "analyze":
        print(json.dumps(analyze(config, args.output_root), ensure_ascii=False, indent=2))
        return 0
    if args.dataset is None:
        parser.error("--dataset is required for audit")
    if not torch.cuda.is_available():
        raise RuntimeError("Rank-R0G audit requires CUDA")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    prepared = prepare(args.dataset, p0, device)
    samples = load_samples(args.dataset, prepared, config)
    result = audit_dataset(args.dataset, prepared, samples, config, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
