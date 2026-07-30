#!/usr/bin/env python3
"""CET Rank-R0: gold-prefix versus beam-ranking surrogate alignment audit."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase3.hbtr_b1_smoke import normalized_sequence  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    collate,
    prepare,
    read_users,
    sha256,
    stable_sha,
    write_json,
)
from experiment.phase5.cet_c1 import structured_passage_mask  # noqa: E402
from experiment.phase5.cet_c2 import (  # noqa: E402
    backbone_forward,
    candidate_sequences,
)
from experiment.phase5.cet_c2_optimization_audit import (  # noqa: E402
    legal_child_symmetric_kl,
    ordered_calibration_samples,
)
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


def make_splits(config: dict, p0: dict) -> dict:
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
            raise ValueError(f"{dataset}: excluded user entered Rank-R0")
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
                str(path.relative_to(ROOT)): sha256(path)
                for path in exclusion_paths(dataset, config)
            },
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
        "config_sha256": sha256(
            ROOT / "artifacts/phase5/configs/cet_rank_r0_preregistered.json"
        ),
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
        raise ValueError(f"{dataset}: Rank-R0 user file SHA mismatch")
    if stable_sha(set(users)) != manifest["user_sha256"]:
        raise ValueError(f"{dataset}: Rank-R0 user-set SHA mismatch")
    excluded = excluded_users(dataset, config)
    if set(users) & excluded:
        raise ValueError(f"{dataset}: Rank-R0 exclusion failure")
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


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 3:
        return None
    x = rankdata(np.asarray(values_x, dtype=float))
    y = rankdata(np.asarray(values_y, dtype=float))
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def union_rank_displacement(
    clean: list[str], perturbed: list[str], top_k: int, missing_rank: int
) -> float:
    union = set(clean[:top_k]) | set(perturbed[:top_k])
    clean_rank = {item: index + 1 for index, item in enumerate(clean)}
    perturbed_rank = {item: index + 1 for index, item in enumerate(perturbed)}
    return float(
        np.mean(
            [
                abs(
                    clean_rank.get(item, missing_rank)
                    - perturbed_rank.get(item, missing_rank)
                )
                for item in union
            ]
        )
    )


@torch.inference_mode()
def generate_ranked(
    backbone,
    prepared: dict,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    beam_size: int,
) -> list[str]:
    trie = gt.Trie(prepared["encoded_candidates"])
    prefix_fn = gt.prefix_allowed_tokens_fn(trie)
    max_length = max(len(row) for row in prepared["encoded_candidates"])
    output = backbone.generate(
        input_ids=input_ids,
        attention_mask=attention,
        max_length=max_length,
        prefix_allowed_tokens_fn=prefix_fn,
        num_beams=beam_size,
        num_return_sequences=beam_size,
        output_scores=True,
        return_dict_in_generate=True,
        length_penalty=1.0,
    )
    ranked = [
        prepared["sequence_to_item"].get(normalized_sequence(value.tolist()))
        for value in output["sequences"]
    ]
    if any(item is None for item in ranked):
        raise ValueError("beam candidate mapping failure")
    mapped = [str(item) for item in ranked]
    if len(mapped) != beam_size or len(set(mapped)) != beam_size:
        raise ValueError("beam candidates are not unique")
    return mapped


def summarize_rows(rows: list[dict]) -> dict:
    masked = [row for row in rows if row["masked_passages"] > 0]
    unmasked = [row for row in rows if row["masked_passages"] == 0]
    correlations_x = [row["gold_prefix_symmetric_kl"] for row in masked]
    correlations_y = [row["union_rank_displacement"] for row in masked]
    return {
        "users": len(rows),
        "masked_users": len(masked),
        "sample_mask_coverage": len(masked) / len(rows),
        "mean_masked_passages": float(
            np.mean([row["masked_passages"] for row in rows])
        ),
        "mean_gold_prefix_symmetric_kl_masked": (
            float(np.mean(correlations_x)) if correlations_x else None
        ),
        "rank_instability_prevalence_masked": (
            float(np.mean([row["rank_instability"] for row in masked]))
            if masked
            else None
        ),
        "mean_top10_overlap_masked": (
            float(np.mean([row["top10_overlap"] for row in masked]))
            if masked
            else None
        ),
        "mean_union_rank_displacement_masked": (
            float(np.mean(correlations_y)) if correlations_y else None
        ),
        "gold_kl_rank_displacement_spearman_masked": spearman(
            correlations_x, correlations_y
        ),
        "mean_absolute_target_rank_shift_masked": (
            float(np.mean([row["absolute_target_rank_shift"] for row in masked]))
            if masked
            else None
        ),
        "unmasked_users": len(unmasked),
        "unmasked_ranking_identity": all(
            not row["rank_instability"]
            and row["top10_overlap"] == 1.0
            and row["union_rank_displacement"] == 0.0
            for row in unmasked
        ),
    }


@torch.inference_mode()
def audit(
    dataset: str,
    control: str,
    prepared: dict,
    samples: list[dict],
    config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    arm_root = ROOT / config["checkpoints"]["root"] / dataset / control
    checkpoint = arm_root / "model.pt"
    expected_sha = config["checkpoints"]["sha256"][dataset][control]
    if sha256(checkpoint) != expected_sha:
        raise ValueError(f"{dataset}/{control}: checkpoint SHA mismatch")
    backbone = prepared["model"].backbone
    backbone.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    backbone.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    beam_size = int(config["evaluation"]["beam_size"])
    top_k = int(config["evaluation"]["top_k"])
    rows = []
    started = time.time()
    for index, sample in enumerate(samples, 1):
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
        altered = [dict(sample, positive_item="__altered__")]
        _, altered_decisions = structured_passage_mask(
            clean_attention,
            altered,
            dataset,
            int(config["views"]["mask_seed"]),
            float(config["views"]["mask_probability"]),
        )
        if not torch.equal(decisions, altered_decisions):
            raise ValueError("mask policy depends on target")
        clean = backbone_forward(backbone, batch, clean_attention)
        perturbed = backbone_forward(backbone, batch, perturbed_attention)
        symmetric_kl, competitive, eligible = legal_child_symmetric_kl(
            clean.logits,
            perturbed.logits,
            candidate_sequences(prepared, [sample]),
            trie,
            int(prepared["tokenizer"].eos_token_id),
            float(config["views"]["temperature"]),
        )
        clean_ranked = generate_ranked(
            backbone,
            prepared,
            batch["item_text_ids"],
            clean_attention,
            beam_size,
        )
        perturbed_ranked = generate_ranked(
            backbone,
            prepared,
            batch["item_text_ids"],
            perturbed_attention,
            beam_size,
        )
        target = sample["positive_item"]
        clean_target_rank = (
            clean_ranked.index(target) + 1 if target in clean_ranked else beam_size + 1
        )
        perturbed_target_rank = (
            perturbed_ranked.index(target) + 1
            if target in perturbed_ranked
            else beam_size + 1
        )
        rows.append(
            {
                "user_id": sample["user_id"],
                "masked_passages": int(decisions.sum()),
                "gold_prefix_symmetric_kl": float(symmetric_kl),
                "competitive_legal_child_steps": competitive,
                "eligible_lexical_steps": eligible,
                "top10_overlap": (
                    len(set(clean_ranked[:top_k]) & set(perturbed_ranked[:top_k]))
                    / top_k
                ),
                "rank_instability": clean_ranked[:top_k]
                != perturbed_ranked[:top_k],
                "union_rank_displacement": union_rank_displacement(
                    clean_ranked,
                    perturbed_ranked,
                    top_k,
                    beam_size + 1,
                ),
                "clean_target_rank": clean_target_rank,
                "perturbed_target_rank": perturbed_target_rank,
                "absolute_target_rank_shift": abs(
                    clean_target_rank - perturbed_target_rank
                ),
            }
        )
        if index % 16 == 0:
            print(
                f"R0_PROGRESS dataset={dataset} control={control} "
                f"users={index}/{len(samples)} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    summary = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "control": control,
        "status": "AUDITED",
        "checkpoint_sha256": expected_sha,
        "audit_user_sha256": stable_sha({row["user_id"] for row in rows}),
        "metrics": summarize_rows(rows),
        "candidate_mapping_rate": 1.0,
        "target_independent_mask": True,
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
        "wall_time_seconds": time.time() - started,
    }
    control_root = output_root / dataset / control
    control_root.mkdir(parents=True, exist_ok=True)
    with (control_root / "per_user.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["per_user_sha256"] = sha256(control_root / "per_user.csv")
    write_json(control_root / "summary.json", summary)
    return summary


def analyze(config: dict, output_root: Path) -> dict:
    results = {
        dataset: {
            control: json.loads(
                (output_root / dataset / control / "summary.json").read_text()
            )
            for control in config["controls"]
        }
        for dataset in config["datasets"]
    }
    routing_control = config["routing"]["control"]
    routing = {}
    integrity = {}
    for dataset in config["datasets"]:
        metrics = results[dataset][routing_control]["metrics"]
        prevalence = metrics["rank_instability_prevalence_masked"]
        rho = metrics["gold_kl_rank_displacement_spearman_masked"]
        routing[dataset] = {
            "control": routing_control,
            "rank_instability_prevalence": prevalence,
            "gold_kl_rank_displacement_spearman": rho,
            "mismatch_condition": (
                prevalence is not None
                and prevalence
                >= float(config["routing"]["instability_prevalence_min"])
                and rho is not None
                and abs(rho) < float(config["routing"]["absolute_spearman_max"])
            ),
            "low_instability_condition": (
                prevalence is not None
                and prevalence
                < float(config["routing"]["low_instability_max"])
            ),
        }
        integrity[dataset] = {
            "same_users_all_controls": len(
                {
                    results[dataset][control]["audit_user_sha256"]
                    for control in config["controls"]
                }
            )
            == 1,
            "candidate_mapping": all(
                results[dataset][control]["candidate_mapping_rate"] == 1.0
                for control in config["controls"]
            ),
            "unmasked_identity": all(
                results[dataset][control]["metrics"]["unmasked_ranking_identity"]
                for control in config["controls"]
            ),
            "target_independent_mask": all(
                results[dataset][control]["target_independent_mask"]
                for control in config["controls"]
            ),
            "targets_sealed": all(
                not results[dataset][control]["validation_target_read"]
                and not results[dataset][control]["test_read"]
                and not results[dataset][control]["sports_read"]
                for control in config["controls"]
            ),
        }
    integrity_pass = all(all(checks.values()) for checks in integrity.values())
    decision = (
        "INVALID_R0_FIX_AND_EXACT_RERUN"
        if not integrity_pass
        else "CET_R0_RANK_SURROGATE_MISMATCH"
        if all(value["mismatch_condition"] for value in routing.values())
        else "STOP_CET_NO_MEANINGFUL_RANK_INSTABILITY"
        if all(value["low_instability_condition"] for value in routing.values())
        else "CET_R0_MIXED_ALIGNMENT_REVIEW_REQUIRED"
    )
    c2_vs_c1 = {
        dataset: {
            "gold_kl_relative_change": (
                results[dataset]["C2"]["metrics"][
                    "mean_gold_prefix_symmetric_kl_masked"
                ]
                - results[dataset]["C1"]["metrics"][
                    "mean_gold_prefix_symmetric_kl_masked"
                ]
            )
            / results[dataset]["C1"]["metrics"][
                "mean_gold_prefix_symmetric_kl_masked"
            ],
            "rank_instability_absolute_change": (
                results[dataset]["C2"]["metrics"][
                    "rank_instability_prevalence_masked"
                ]
                - results[dataset]["C1"]["metrics"][
                    "rank_instability_prevalence_masked"
                ]
            ),
            "top10_overlap_change": (
                results[dataset]["C2"]["metrics"]["mean_top10_overlap_masked"]
                - results[dataset]["C1"]["metrics"]["mean_top10_overlap_masked"]
            ),
        }
        for dataset in config["datasets"]
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "results": results,
        "routing_evidence": routing,
        "c2_vs_c1": c2_vs_c1,
        "integrity_checks": integrity,
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
        "--stage", choices=("make-splits", "audit", "analyze"), required=True
    )
    parser.add_argument("--dataset", choices=("Toys", "Beauty"))
    parser.add_argument("--control", choices=("C0", "C1", "C2"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config, p0 = load_configs(args.config)
    actual_sha = sha256(Path(__file__))
    registered_sha = config["integrity"]["code_sha256"]
    if registered_sha != "PENDING_FREEZE" and actual_sha != registered_sha:
        raise ValueError(
            f"Rank-R0 code SHA mismatch: actual={actual_sha} "
            f"registered={registered_sha}"
        )
    if args.stage == "make-splits":
        print(json.dumps(make_splits(config, p0), ensure_ascii=False, indent=2))
        return 0
    frozen = json.loads(
        (ROOT / config["data"]["split_root"] / "frozen_manifest.json").read_text()
    )
    if frozen["code_sha256"] != actual_sha:
        raise ValueError("Rank-R0 frozen-manifest code SHA mismatch")
    if args.stage == "analyze":
        print(json.dumps(analyze(config, args.output_root), ensure_ascii=False, indent=2))
        return 0
    if args.dataset is None or args.control is None:
        parser.error("--dataset and --control are required for audit")
    if not torch.cuda.is_available():
        raise RuntimeError("Rank-R0 requires CUDA")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    prepared = prepare(args.dataset, p0, device)
    samples = load_samples(args.dataset, prepared, config)
    result = audit(
        args.dataset,
        args.control,
        prepared,
        samples,
        config,
        args.output_root,
        device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
