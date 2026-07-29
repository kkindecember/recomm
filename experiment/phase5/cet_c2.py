#!/usr/bin/env python3
"""CET C2: preregistered three-arm recommendation-effect pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gcdh_p0 import (  # noqa: E402
    build_train_samples,
    build_validation_samples,
    collate,
    head_items,
    prepare,
    read_users,
    sha256,
    stable_sha,
    training_popularity,
    write_json,
)
from experiment.phase5.cet_c1 import (  # noqa: E402
    legal_child_kl,
    structured_passage_mask,
)
from utils import generation_trie as gt  # noqa: E402
from experiment.phase3.hbtr_b1_smoke import normalized_sequence  # noqa: E402


def load_configs(config_path: Path) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text())
    p0 = json.loads(
        (ROOT / "artifacts/phase4/configs/gcdh_p0_preregistered.json").read_text()
    )
    return config, p0


def select_validation_users(
    dataset: str,
    sequences: dict[str, list[str]],
    train_users: set[str],
    count: int,
    salt: str,
) -> list[str]:
    eligible = [
        user
        for user, items in sequences.items()
        if user not in train_users and len(items) >= 3
    ]
    eligible.sort(
        key=lambda user: hashlib.sha256(
            f"{salt}|{dataset}|{user}".encode()
        ).hexdigest()
    )
    if len(eligible) < count:
        raise ValueError(f"{dataset}: only {len(eligible)} eligible validation users")
    return eligible[:count]


def make_splits(config: dict, p0: dict) -> dict:
    from experiment.phase3.hbtr_b1_smoke import read_sequences

    result = {}
    split_root = ROOT / config["data"]["split_root"]
    for dataset in config["datasets"]:
        sequences = read_sequences(
            ROOT / "GRAM/rec_datasets" / dataset / "user_sequence.txt"
        )
        train_path = (
            ROOT
            / "artifacts/phase4/gcdh_p0_splits"
            / dataset
            / "train_users.txt"
        )
        train_users = read_users(train_path)
        users = select_validation_users(
            dataset,
            sequences,
            train_users,
            int(config["data"]["validation_users"]),
            config["data"]["validation_salt"],
        )
        if train_users.intersection(users):
            raise ValueError("train/validation user overlap")
        output = split_root / dataset / "validation_users.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(users) + "\n")
        manifest = {
            "dataset": dataset,
            "source_train_users": str(train_path.relative_to(ROOT)),
            "source_train_user_sha256": stable_sha(train_users),
            "validation_users": len(users),
            "validation_user_sha256": stable_sha(set(users)),
            "validation_file_sha256": sha256(output),
            "selection": "SHA256(salt|dataset|user), ascending",
            "selection_uses_candidate_target": False,
            "train_validation_disjoint": True,
            "test_read": False,
            "sports_read": False,
        }
        write_json(output.parent / "manifest.json", manifest)
        result[dataset] = manifest
    return result


def verify_frozen_splits(config: dict) -> None:
    for dataset in config["datasets"]:
        path = (
            ROOT
            / config["data"]["split_root"]
            / dataset
            / "validation_users.txt"
        )
        users = read_users(path)
        if sha256(path) != config["data"]["validation_file_sha256"][dataset]:
            raise ValueError(f"{dataset}: validation split file SHA mismatch")
        if stable_sha(users) != config["data"]["validation_user_sha256"][dataset]:
            raise ValueError(f"{dataset}: validation user-set SHA mismatch")


def candidate_sequences(prepared: dict, samples: list[dict]) -> list[list[int]]:
    item_to_sequence = dict(
        zip(prepared["catalog"], prepared["encoded_candidates"])
    )
    return [item_to_sequence[row["positive_item"]] for row in samples]


def backbone_forward(backbone, batch: dict, attention: torch.Tensor):
    return backbone(
        input_ids=batch["item_text_ids"],
        attention_mask=attention,
        labels=batch["target_ids"],
        return_dict=True,
    )


def train(
    dataset: str,
    control: str,
    prepared: dict,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> dict:
    train_users = read_users(
        ROOT / "artifacts/phase4/gcdh_p0_splits" / dataset / "train_users.txt"
    )
    samples = build_train_samples(
        prepared["sequences"],
        train_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    training = config["training"]
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation"])
    epochs = int(training["epochs"])
    batches = math.ceil(len(samples) / batch_size)
    updates_per_epoch = math.ceil(batches / accumulation)
    total_updates = updates_per_epoch * epochs
    backbone = prepared["model"].backbone
    backbone.train()
    optimizer = torch.optim.AdamW(
        backbone.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * float(training["warmup_fraction"])),
        num_training_steps=total_updates,
    )
    trie = gt.Trie(prepared["encoded_candidates"])
    alpha = 0.0 if control == "C0" else float(training["alpha"])
    beta = float(training["beta"]) if control == "C2" else 0.0
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    update_count = 0
    records = []
    mask_signature = hashlib.sha256()
    for epoch in range(1, epochs + 1):
        indices = list(range(len(samples)))
        random.Random(int(config["seed"]) + epoch).shuffle(indices)
        totals = defaultdict(float)
        competitive_steps = 0
        masked_passages = 0
        for batch_number, start in enumerate(range(0, len(indices), batch_size), 1):
            rows = [samples[index] for index in indices[start : start + batch_size]]
            batch = collate(prepared["collator"], rows)
            for key in ("item_text_ids", "item_text_masks", "target_ids"):
                batch[key] = batch[key].to(device)
            clean_attention = batch["item_text_masks"].bool()
            sequences = candidate_sequences(prepared, rows)
            clean = backbone_forward(backbone, batch, clean_attention)
            if not torch.isfinite(clean.loss) or not torch.isfinite(clean.logits).all():
                raise ValueError("non-finite clean output")
            clean_logits = clean.logits.detach()
            (clean.loss / accumulation).backward()
            total_value = float(clean.loss.detach())
            perturbed_ce_value = 0.0
            kl_value = 0.0
            if control != "C0":
                perturbed_attention, decisions = structured_passage_mask(
                    clean_attention,
                    rows,
                    dataset,
                    int(training["mask_seed"]) + epoch,
                    float(training["mask_probability"]),
                )
                if not torch.equal(perturbed_attention[:, 0], clean_attention[:, 0]):
                    raise ValueError("coarse passage changed")
                if clean_attention.shape[1] > 1 and not torch.equal(
                    perturbed_attention[:, 1], clean_attention[:, 1]
                ):
                    raise ValueError("newest fine passage changed")
                mask_signature.update(decisions.detach().cpu().numpy().tobytes())
                masked_passages += int(decisions.sum())
                perturbed = backbone_forward(backbone, batch, perturbed_attention)
                if (
                    not torch.isfinite(perturbed.loss)
                    or not torch.isfinite(perturbed.logits).all()
                ):
                    raise ValueError("non-finite perturbed output")
                extra = alpha * perturbed.loss
                perturbed_ce_value = float(perturbed.loss.detach())
                if control == "C2":
                    kl, steps = legal_child_kl(
                        clean_logits,
                        perturbed.logits,
                        sequences,
                        trie,
                        int(prepared["tokenizer"].eos_token_id),
                        float(training["temperature"]),
                    )
                    extra = extra + beta * kl
                    kl_value = float(kl.detach())
                    competitive_steps += steps
                (extra / accumulation).backward()
                total_value += float(extra.detach())
            should_step = batch_number % accumulation == 0 or batch_number == batches
            if should_step:
                norm = torch.nn.utils.clip_grad_norm_(
                    backbone.parameters(), float(training["gradient_clip_norm"])
                )
                if not torch.isfinite(norm):
                    raise ValueError("non-finite gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
            totals["total"] += total_value
            totals["clean_ce"] += float(clean.loss.detach())
            totals["perturbed_ce"] += perturbed_ce_value
            totals["legal_kl"] += kl_value
            if batch_number % 100 == 0:
                print(
                    f"TRAIN_PROGRESS dataset={dataset} control={control} "
                    f"epoch={epoch}/{epochs} batch={batch_number}/{batches} "
                    f"updates={update_count}/{total_updates} "
                    f"elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
        records.append(
            {
                "epoch": epoch,
                "mean_total_loss": totals["total"] / batches,
                "mean_clean_ce": totals["clean_ce"] / batches,
                "mean_perturbed_ce": totals["perturbed_ce"] / batches,
                "mean_legal_child_kl": totals["legal_kl"] / batches,
                "masked_passages": masked_passages,
                "competitive_legal_child_steps": competitive_steps,
                "optimizer_updates_cumulative": update_count,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    torch.save(backbone.state_dict(), checkpoint)
    result = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "control": control,
        "status": "TRAINED",
        "train_users": len(train_users),
        "train_user_sha256": stable_sha(train_users),
        "samples": len(samples),
        "epochs": epochs,
        "batches_per_epoch": batches,
        "optimizer_updates": update_count,
        "alpha": alpha,
        "beta": beta,
        "mask_signature_sha256": (
            None if control == "C0" else mask_signature.hexdigest()
        ),
        "epoch_records": records,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "wall_time_seconds": time.time() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "validation_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(output_dir / "training_summary.json", result)
    return result


def rank_metrics(rank: int | None) -> dict[str, float]:
    return {
        "Recall@10": float(rank is not None and rank <= 10),
        "NDCG@10": (
            1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0
        ),
        "Recall@50": float(rank is not None and rank <= 50),
    }


@torch.no_grad()
def validate(
    dataset: str,
    control: str,
    prepared: dict,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> dict:
    split_path = (
        ROOT / config["data"]["split_root"] / dataset / "validation_users.txt"
    )
    users = read_users(split_path)
    samples = build_validation_samples(
        prepared["sequences"], users, prepared["item2input"], prepared["item2lexid"]
    )
    backbone = prepared["model"].backbone
    backbone.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    prefix_fn = gt.prefix_allowed_tokens_fn(trie)
    max_length = max(len(row) for row in prepared["encoded_candidates"])
    rows = []
    started = time.time()
    for index, sample in enumerate(samples, 1):
        batch = collate(prepared["collator"], [sample])
        prediction = backbone.generate(
            input_ids=batch["item_text_ids"].to(device),
            attention_mask=batch["item_text_masks"].to(device),
            max_length=max_length,
            prefix_allowed_tokens_fn=prefix_fn,
            num_beams=50,
            num_return_sequences=50,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,
        )
        ranked = [
            prepared["sequence_to_item"].get(normalized_sequence(value.tolist()))
            for value in prediction["sequences"]
        ]
        if any(item is None for item in ranked) or len(set(ranked)) != 50:
            raise ValueError("constrained beam mapping failure")
        target = sample["positive_item"]
        rank = ranked.index(target) + 1 if target in ranked else None
        rows.append(
            {
                "user_id": sample["user_id"],
                "target_item": target,
                "target_group": (
                    "head" if target in prepared["heads"] else "tail"
                ),
                "raw_history_length": sample["raw_history_length"],
                "rank": "" if rank is None else rank,
                **rank_metrics(rank),
            }
        )
        if index % 250 == 0:
            print(
                f"VALID_PROGRESS dataset={dataset} control={control} "
                f"users={index}/{len(samples)} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user = output_dir / "validation_per_user.csv"
    with per_user.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def summarize(selected: list[dict]) -> dict:
        return {
            "n": len(selected),
            **{
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in ("Recall@10", "NDCG@10", "Recall@50")
            },
        }

    history_edges = config["evaluation"]["history_bin_edges"]
    groups = {"overall": summarize(rows)}
    for group in ("head", "tail"):
        groups[group] = summarize(
            [row for row in rows if row["target_group"] == group]
        )
    for lower, upper in zip(history_edges[:-1], history_edges[1:]):
        selected = [
            row
            for row in rows
            if int(lower) <= row["raw_history_length"] < int(upper)
        ]
        groups[f"history_{lower}_{upper}"] = summarize(selected)
    elapsed = time.time() - started
    result = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "control": control,
        "status": "VALIDATED",
        "users": len(rows),
        "validation_user_sha256": stable_sha(users),
        "groups": groups,
        "per_user_sha256": sha256(per_user),
        "wall_time_seconds": elapsed,
        "per_user_latency_seconds": elapsed / len(rows),
        "candidate_mapping_rate": 1.0,
        "test_read": False,
        "sports_read": False,
    }
    write_json(output_dir / "validation_summary.json", result)
    return result


def read_per_user(path: Path) -> dict[str, dict]:
    with path.open(newline="") as handle:
        return {row["user_id"]: row for row in csv.DictReader(handle)}


def paired_bootstrap(
    baseline: np.ndarray, candidate: np.ndarray, seed: int, resamples: int
) -> dict:
    rng = np.random.default_rng(seed)
    differences = candidate - baseline
    means = np.empty(resamples)
    for start in range(0, resamples, 500):
        width = min(500, resamples - start)
        indices = rng.integers(0, len(differences), size=(width, len(differences)))
        means[start : start + width] = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "changed_users": int(np.count_nonzero(differences)),
    }


def analyze(config: dict, output_root: Path) -> dict:
    controls = ("C0", "C1", "C2")
    data = {
        dataset: {
            control: read_per_user(
                output_root / dataset / control / "validation_per_user.csv"
            )
            for control in controls
        }
        for dataset in config["datasets"]
    }
    comparisons = {}
    for dataset in config["datasets"]:
        base_users = set(data[dataset]["C0"])
        if any(set(data[dataset][control]) != base_users for control in controls):
            raise ValueError("paired validation cohorts differ")
        comparisons[dataset] = {}
        for control in ("C1", "C2"):
            users = sorted(base_users)
            entry = {}
            for group in ("overall", "head", "tail"):
                selected = [
                    user
                    for user in users
                    if group == "overall"
                    or data[dataset]["C0"][user]["target_group"] == group
                ]
                entry[group] = {}
                for metric in ("Recall@10", "NDCG@10", "Recall@50"):
                    baseline = np.array(
                        [float(data[dataset]["C0"][user][metric]) for user in selected]
                    )
                    candidate = np.array(
                        [float(data[dataset][control][user][metric]) for user in selected]
                    )
                    stats = paired_bootstrap(
                        baseline,
                        candidate,
                        int(config["evaluation"]["bootstrap_seed"]),
                        int(config["evaluation"]["bootstrap_resamples"]),
                    )
                    base_mean = float(baseline.mean())
                    cand_mean = float(candidate.mean())
                    stats.update(
                        {
                            "baseline": base_mean,
                            "candidate": cand_mean,
                            "relative_change": (
                                (cand_mean - base_mean) / base_mean
                                if base_mean > 0
                                else None
                            ),
                        }
                    )
                    entry[group][metric] = stats
            entry["broad_harm_rate"] = float(
                np.mean(
                    [
                        float(data[dataset]["C0"][user]["Recall@10"]) == 1.0
                        and float(data[dataset][control][user]["Recall@10"]) == 0.0
                        for user in users
                    ]
                )
            )
            comparisons[dataset][control] = entry

    c2_ndcg = [
        comparisons[d]["C2"]["overall"]["NDCG@10"]["relative_change"]
        for d in config["datasets"]
    ]
    c1_ndcg = [
        comparisons[d]["C1"]["overall"]["NDCG@10"]["relative_change"]
        for d in config["datasets"]
    ]
    c2_recall = [
        comparisons[d]["C2"]["overall"]["Recall@10"]["mean_difference"]
        for d in config["datasets"]
    ]
    pooled_tail_base, pooled_tail_c2 = [], []
    for dataset in config["datasets"]:
        for user, row in data[dataset]["C0"].items():
            if row["target_group"] == "tail":
                pooled_tail_base.append(float(row["NDCG@10"]))
                pooled_tail_c2.append(float(data[dataset]["C2"][user]["NDCG@10"]))
    pooled_base = float(np.mean(pooled_tail_base))
    pooled_c2 = float(np.mean(pooled_tail_c2))
    pooled_tail_relative = (
        (pooled_c2 - pooled_base) / pooled_base if pooled_base > 0 else None
    )
    gates = config["development_gates"]
    effect_checks = {
        "macro_ndcg_relative_gain": float(np.mean(c2_ndcg))
        >= float(gates["macro_ndcg_relative_gain_min"]),
        "one_domain_ndcg_gain": max(c2_ndcg)
        >= float(gates["one_domain_ndcg_relative_gain_min"]),
        "other_domain_ndcg_floor": min(c2_ndcg)
        >= float(gates["other_domain_ndcg_relative_floor"]),
        "recall10_floor_each_domain": min(c2_recall)
        >= float(gates["recall10_absolute_floor"]),
        "pooled_tail_ndcg_floor": pooled_tail_relative
        >= float(gates["pooled_tail_ndcg_relative_floor"]),
        "broad_harm_rate_each_domain": max(
            comparisons[d]["C2"]["broad_harm_rate"] for d in config["datasets"]
        )
        <= float(gates["broad_harm_rate_max"]),
    }
    consistency_value = float(np.mean(c2_ndcg)) > float(np.mean(c1_ndcg))
    integrity = {
        "paired_users_all_controls": True,
        "candidate_mapping_rate": all(
            json.loads(
                (
                    output_root / d / c / "validation_summary.json"
                ).read_text()
            )["candidate_mapping_rate"]
            == 1.0
            for d in config["datasets"]
            for c in controls
        ),
        "c1_c2_mask_signature_equal": all(
            json.loads(
                (output_root / d / "C1" / "training_summary.json").read_text()
            )["mask_signature_sha256"]
            == json.loads(
                (output_root / d / "C2" / "training_summary.json").read_text()
            )["mask_signature_sha256"]
            for d in config["datasets"]
        ),
        "test_not_read": True,
        "sports_not_read": True,
    }
    decision = (
        "INVALID_RUN_FIX_AND_EXACT_RERUN"
        if not all(integrity.values())
        else "CET_FREEZE_FOR_CONFIRMATION"
        if all(effect_checks.values()) and consistency_value
        else "STOP_CET_NO_CONSISTENCY_VALUE"
        if all(effect_checks.values()) and not consistency_value
        else "STOP_CET_NO_REGULARIZATION_EFFECT"
    )
    result = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "comparisons_to_C0": comparisons,
        "macro_ndcg_relative_change": {
            "C1": float(np.mean(c1_ndcg)),
            "C2": float(np.mean(c2_ndcg)),
        },
        "pooled_tail_ndcg_relative_change_C2": pooled_tail_relative,
        "effect_checks": effect_checks,
        "consistency_value_point_estimate_C2_gt_C1": consistency_value,
        "integrity_checks": integrity,
        "confidence_intervals_are_descriptive_not_gates": True,
        "test_read": False,
        "sports_read": False,
    }
    write_json(output_root / "summary.json", result)
    write_json(
        output_root / "status.json",
        {"experiment_id": config["experiment_id"], "status": "completed"},
    )
    return result


def load_prepared(dataset: str, p0: dict, device: torch.device) -> dict:
    prepared = prepare(dataset, p0, device)
    popularity = training_popularity(prepared["sequences"])
    prepared["heads"] = head_items(popularity)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("make-splits", "smoke", "train", "validate", "analyze")
    )
    parser.add_argument("--dataset", choices=("Toys", "Beauty"))
    parser.add_argument("--control", choices=("C0", "C1", "C2"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config, p0 = load_configs(args.config)
    actual_code_sha = sha256(Path(__file__))
    registered_code_sha = config["integrity"]["code_sha256"]
    if registered_code_sha != "PENDING_FREEZE" and actual_code_sha != registered_code_sha:
        raise ValueError(
            f"CET C2 code SHA mismatch: actual={actual_code_sha} "
            f"registered={registered_code_sha}"
        )
    if args.stage == "make-splits":
        print(json.dumps(make_splits(config, p0), ensure_ascii=False, indent=2))
        return 0
    verify_frozen_splits(config)
    if args.stage == "analyze":
        print(json.dumps(analyze(config, args.output_root), ensure_ascii=False, indent=2))
        return 0
    if args.dataset is None or args.control is None:
        parser.error("--dataset and --control are required for model stages")
    if not torch.cuda.is_available():
        raise RuntimeError("CET C2 model stages require CUDA")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    prepared = load_prepared(args.dataset, p0, device)
    output_dir = args.output_root / args.dataset / args.control
    if args.stage == "smoke":
        users = read_users(
            ROOT
            / "artifacts/phase4/gcdh_p0_splits"
            / args.dataset
            / "train_users.txt"
        )
        all_samples = build_train_samples(
            prepared["sequences"],
            users,
            prepared["item2input"],
            prepared["item2lexid"],
        )
        # Exercise the actual perturbation path, independent of a chance q=0.25 draw.
        samples = [
            sample for sample in all_samples if len(sample["history_items"]) >= 3
        ][:2]
        if len(samples) != 2:
            raise ValueError("insufficient long-history smoke samples")
        batch = collate(prepared["collator"], samples[:2])
        for key in ("item_text_ids", "item_text_masks", "target_ids"):
            batch[key] = batch[key].to(device)
        clean = backbone_forward(
            prepared["model"].backbone, batch, batch["item_text_masks"].bool()
        )
        perturbed_attention, decisions = structured_passage_mask(
            batch["item_text_masks"].bool(),
            samples[:2],
            args.dataset,
            int(config["training"]["mask_seed"]) + 1,
            1.0,
        )
        if int(decisions.sum()) == 0:
            raise ValueError("smoke did not exercise passage masking")
        perturbed = backbone_forward(
            prepared["model"].backbone, batch, perturbed_attention
        )
        kl, steps = legal_child_kl(
            clean.logits.detach(),
            perturbed.logits,
            candidate_sequences(prepared, samples[:2]),
            gt.Trie(prepared["encoded_candidates"]),
            int(prepared["tokenizer"].eos_token_id),
            float(config["training"]["temperature"]),
        )
        total = clean.loss + perturbed.loss + float(config["training"]["beta"]) * kl
        total.backward()
        smoke = {
            "dataset": args.dataset,
            "status": "PASS",
            "finite": bool(torch.isfinite(total)),
            "masked_passages": int(decisions.sum()),
            "smoke_mask_probability": 1.0,
            "registered_training_mask_probability": float(
                config["training"]["mask_probability"]
            ),
            "competitive_steps": steps,
            "test_read": False,
            "sports_read": False,
        }
        write_json(output_dir / "smoke.json", smoke)
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "train":
        train(args.dataset, args.control, prepared, config, output_dir, device)
        return 0
    checkpoint = output_dir / "model.pt"
    summary = json.loads((output_dir / "training_summary.json").read_text())
    if sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    prepared["model"].backbone.load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    validate(args.dataset, args.control, prepared, config, output_dir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
