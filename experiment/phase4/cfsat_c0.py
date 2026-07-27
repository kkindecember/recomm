#!/usr/bin/env python3
"""CF-SAT C0: frozen clean-vs-collaborative-corruption premise audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "experiment/phase3"
GRAM_SRC = ROOT / "GRAM/src"
for candidate in (PHASE3, GRAM_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from processor import CollatorGRAM  # noqa: E402
from utils import generation_trie as gt  # noqa: E402
from utils import indexing  # noqa: E402

from cgi_e0 import bootstrap_mean, write_csv  # noqa: E402
from hbtr_b1_smoke import (  # noqa: E402
    create_model_and_tokenizer,
    make_runtime_args,
    read_sequences,
    sha256,
)
from marc_l0 import (  # noqa: E402
    build_passage,
    digest_int,
    encode_catalog_trie,
    read_keyed_text,
    read_neighbors,
    score_condition,
    selection_hash,
    user_split,
)

DELIMITER_TOKEN_IDS = {1820, 9175}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "artifacts/phase4/configs/cfsat_c0_preregistered.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/phase4/cfsat_c0",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/第四阶段/GRAM_第四阶段_CFSAT_C0报告.md",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def filtered_token_count(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return sum(int(value) not in DELIMITER_TOKEN_IDS for value in ids)


def cf_segment(
    item: str,
    item2lexid: dict[str, str],
    neighbors: dict[str, list[str]],
    k: int,
) -> str:
    selected = neighbors[item][:k]
    if len(selected) != k:
        raise ValueError(f"{item} has fewer than {k} neighbors")
    return "similar items: " + ", ".join(item2lexid[value] for value in selected)


def build_donor_buckets(
    catalog: list[str],
    item2lexid: dict[str, str],
    neighbors: dict[str, list[str]],
    tokenizer,
    k: int,
) -> tuple[dict[int, list[str]], dict[str, int]]:
    buckets: dict[int, list[str]] = defaultdict(list)
    lengths = {}
    for item in catalog:
        length = filtered_token_count(
            tokenizer, cf_segment(item, item2lexid, neighbors, k)
        )
        lengths[item] = length
        buckets[length].append(item)
    return dict(buckets), lengths


def choose_donor(
    *,
    seed: int,
    dataset: str,
    user: str,
    anchor: str,
    target: str,
    candidates: list[str],
    neighbors: dict[str, list[str]],
    k: int,
    maximum_jaccard: float,
) -> tuple[str, float] | None:
    clean = set(neighbors[anchor][:k])
    eligible = []
    for donor in candidates:
        donor_values = neighbors[donor][:k]
        if donor == anchor or donor == target or target in donor_values:
            continue
        overlap = jaccard(clean, set(donor_values))
        if overlap > maximum_jaccard:
            continue
        rank = hashlib.sha256(
            f"{seed}|{dataset}|{user}|{anchor}|{target}|{donor}".encode()
        ).hexdigest()
        eligible.append((rank, donor, overlap))
    if not eligible:
        return None
    _, donor, overlap = min(eligible)
    return donor, overlap


def metadata_start(
    tokenizer,
    item: str,
    neighbor_item: str,
    item2lexid: dict[str, str],
    neighbors: dict[str, list[str]],
    k: int,
) -> int:
    prefix = (
        f"item: {item2lexid[item]}; "
        + cf_segment(neighbor_item, item2lexid, neighbors, k)
        + "; "
    )
    return filtered_token_count(tokenizer, prefix)


def make_candidate_rows(
    dataset: str,
    sequences: dict[str, list[str]],
    valid_items: set[str],
    config: dict,
) -> dict[str, list[dict]]:
    result = {name: [] for name in config["splits"]}
    offset = config["target_offset_from_end"]
    for user, items in sequences.items():
        if len(items) < offset + config["min_history"]:
            continue
        target = items[-offset]
        history = items[:-offset][-config["max_history"] :]
        if (
            target not in valid_items
            or len(history) < config["min_history"]
            or any(item not in valid_items for item in history)
        ):
            continue
        split = user_split(config["seed"], dataset, user, config)
        result[split].append(
            {
                "user": user,
                "split": split,
                "target": target,
                "history": history,
                "history_length": len(history),
                "sample_hash": selection_hash(
                    config["seed"], dataset, split, user
                ),
            }
        )
    for rows in result.values():
        rows.sort(key=lambda row: row["sample_hash"])
    return result


def collator_attention_identity(
    tokenizer,
    runtime,
    cohort: list[dict],
    batch_size: int,
) -> float:
    collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
    equal = 0
    total = 0
    for start in range(0, len(cohort), batch_size):
        chunk = cohort[start : start + batch_size]
        clean = collator(
            [
                {
                    "input": row["clean_input"],
                    "output": row["output"],
                    "user_id": row["user"],
                }
                for row in chunk
            ]
        )
        corrupt = collator(
            [
                {
                    "input": row["corrupt_input"],
                    "output": row["output"],
                    "user_id": row["user"],
                }
                for row in chunk
            ]
        )
        clean_masks = clean["item_text_masks"]
        corrupt_masks = corrupt["item_text_masks"]
        if clean_masks.shape != corrupt_masks.shape:
            total += len(chunk)
            continue
        per_sample = (
            clean_masks.eq(corrupt_masks)
            .reshape(len(chunk), -1)
            .all(dim=1)
        )
        equal += int(per_sample.sum().item())
        total += len(chunk)
    return equal / total if total else 0.0


def prepare_dataset(dataset: str, spec: dict, config: dict) -> tuple[dict, dict]:
    from transformers import AutoTokenizer

    dataset_dir = ROOT / "GRAM/rec_datasets" / dataset
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    runtime = make_runtime_args(dataset)
    sequences = read_sequences(dataset_dir / "user_sequence.txt")
    _, item2input, item2lexid = indexing.gram_indexing(
        data_path=runtime.data_path,
        dataset=dataset,
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        args=runtime,
        user_id_without_target_item=True,
        id_linking=True,
    )
    item_text = read_keyed_text(dataset_dir / "item_plain_text.txt")
    neighbors = read_neighbors(dataset_dir / "similar_item_sasrec.txt")
    k = int(spec["baseline_neighbor_budget"])
    catalog = sorted(
        item
        for item in set(item2input) & set(item2lexid) & set(item_text) & set(neighbors)
        if len(neighbors[item]) >= k
        and all(value in item2lexid for value in neighbors[item][:k])
    )
    valid_items = set(catalog)
    buckets, cf_lengths = build_donor_buckets(
        catalog, item2lexid, neighbors, tokenizer, k
    )
    candidates = make_candidate_rows(dataset, sequences, valid_items, config)
    selected = []
    rejected = Counter()
    replay = []
    target_exclusion = []
    budget_identity = []
    overlap_gate = []
    attention_identity = []
    metadata_identity = []
    maximum_jaccard = float(config["corruption"]["maximum_neighbor_jaccard"])

    for split, rows in candidates.items():
        cap = int(config["splits"][split]["max_users"])
        for row in rows:
            if sum(value["split"] == split for value in selected) >= cap:
                break
            ordered = list(reversed(row["history"]))
            donors = []
            overlaps = []
            failed = False
            for anchor in ordered:
                choice = choose_donor(
                    seed=config["seed"],
                    dataset=dataset,
                    user=row["user"],
                    anchor=anchor,
                    target=row["target"],
                    candidates=buckets[cf_lengths[anchor]],
                    neighbors=neighbors,
                    k=k,
                    maximum_jaccard=maximum_jaccard,
                )
                if choice is None:
                    failed = True
                    break
                donor, overlap = choice
                donors.append(donor)
                overlaps.append(overlap)
            if failed:
                rejected["no_length_matched_donor"] += 1
                continue

            clean_passages = [
                build_passage(
                    anchor, item2lexid, item_text, neighbors, k, True
                )
                for anchor in ordered
            ]
            corrupt_passages = [
                build_passage(
                    anchor,
                    item2lexid,
                    item_text,
                    neighbors,
                    k,
                    True,
                    donor_item=donor,
                    corrupt_source="collaborative",
                )
                for anchor, donor in zip(ordered, donors)
            ]
            no_cf_passages = [
                build_passage(
                    anchor, item2lexid, item_text, neighbors, 0, True
                )
                for anchor in ordered
            ]
            clean_lengths = [
                min(runtime.item_prompt_max_len, filtered_token_count(tokenizer, text) + 1)
                for text in clean_passages
            ]
            corrupt_lengths = [
                min(runtime.item_prompt_max_len, filtered_token_count(tokenizer, text) + 1)
                for text in corrupt_passages
            ]
            clean_starts = [
                metadata_start(
                    tokenizer, anchor, anchor, item2lexid, neighbors, k
                )
                for anchor in ordered
            ]
            corrupt_starts = [
                metadata_start(
                    tokenizer, anchor, donor, item2lexid, neighbors, k
                )
                for anchor, donor in zip(ordered, donors)
            ]
            history_lex = " ; ".join(item2lexid[item] for item in ordered)
            coarse = f"What would user purchase after {history_lex} ?"
            sample = {
                **row,
                "donors": donors,
                "donor_overlap_max": max(overlaps, default=0.0),
                "clean_visible_lengths": clean_lengths,
                "corrupt_visible_lengths": corrupt_lengths,
                "clean_metadata_starts": clean_starts,
                "corrupt_metadata_starts": corrupt_starts,
                "clean_input": [coarse] + clean_passages,
                "corrupt_input": [coarse] + corrupt_passages,
                "no_cf_input": [coarse] + no_cf_passages,
                "output": item2lexid[row["target"]],
            }
            selected.append(sample)
            replay.extend(
                clean == item2input[anchor]
                for clean, anchor in zip(clean_passages, ordered)
            )
            target_exclusion.extend(
                row["target"] != donor
                and row["target"] not in neighbors[donor][:k]
                for donor in donors
            )
            budget_identity.extend(
                len(neighbors[anchor][:k]) == len(neighbors[donor][:k]) == k
                for anchor, donor in zip(ordered, donors)
            )
            overlap_gate.extend(value <= maximum_jaccard for value in overlaps)
            attention_identity.extend(
                left == right
                for left, right in zip(clean_lengths, corrupt_lengths)
            )
            metadata_identity.extend(
                left == right
                for left, right in zip(clean_starts, corrupt_starts)
            )

    counts = Counter(row["split"] for row in selected)
    expected = {
        split: int(value["max_users"]) for split, value in config["splits"].items()
    }
    user_sets = {
        split: {row["user"] for row in selected if row["split"] == split}
        for split in config["splits"]
    }
    overlaps = sum(
        len(user_sets[left] & user_sets[right])
        for left, right in (
            ("fit", "calibration"),
            ("fit", "audit"),
            ("calibration", "audit"),
        )
    )
    actual_collator_attention_identity = collator_attention_identity(
        tokenizer,
        runtime,
        selected,
        int(config["batch_size"]),
    )
    paths = {
        "checkpoint": ROOT / spec["checkpoint"],
        "run_config": ROOT / spec["run_config"],
        "user_sequence": dataset_dir / "user_sequence.txt",
        "item_index": dataset_dir
        / f"item_generative_indexing_{runtime.hierarchical_id_type}.txt",
        "item_text": dataset_dir / "item_plain_text.txt",
        "neighbors": dataset_dir / "similar_item_sasrec.txt",
    }
    integrity = {
        "catalog_size": len(catalog),
        "exact_split_caps": dict(counts) == expected,
        "counts": dict(counts),
        "user_overlap_across_splits": overlaps,
        "heldout_sequence_fields_read": False,
        "validation_or_test_read": False,
        "clean_serialization_replay_rate": float(np.mean(replay)) if replay else 0.0,
        "donor_target_exclusion_rate": (
            float(np.mean(target_exclusion)) if target_exclusion else 0.0
        ),
        "neighbor_budget_identity_rate": (
            float(np.mean(budget_identity)) if budget_identity else 0.0
        ),
        "donor_overlap_gate_rate": (
            float(np.mean(overlap_gate)) if overlap_gate else 0.0
        ),
        "attention_length_identity_rate": (
            float(np.mean(attention_identity)) if attention_identity else 0.0
        ),
        "collator_attention_mask_identity_rate": (
            actual_collator_attention_identity
        ),
        "metadata_start_identity_rate": (
            float(np.mean(metadata_identity)) if metadata_identity else 0.0
        ),
        "optimizer_steps": 0,
    }
    return {
        "tokenizer": tokenizer,
        "runtime": runtime,
        "item2input": item2input,
        "item2lexid": item2lexid,
        "item_text": item_text,
        "neighbors": neighbors,
        "cohort": selected,
    }, {
        "availability": {
            "candidate_counts": {
                split: len(rows) for split, rows in candidates.items()
            },
            "rejected": dict(rejected),
        },
        "integrity": integrity,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
    }


def condition_samples(cohort: list[dict], key: str) -> list[dict]:
    return [
        {
            "user": row["user"],
            "input": row[key],
            "output": row["output"],
        }
        for row in cohort
    ]


def model_parameter_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(value.tobytes())
    return digest.hexdigest()


def make_node_rows(
    dataset: str,
    cohort: list[dict],
    scores: dict[str, dict[str, list[dict]]],
    deficit_margin: float,
) -> list[dict]:
    lookup = {row["user"]: row for row in cohort}
    result = []
    for user, sample in lookup.items():
        clean = scores["clean"][user]
        corrupt = scores["corrupt"][user]
        no_cf = scores["no_cf"][user]
        if not (len(clean) == len(corrupt) == len(no_cf)):
            raise ValueError("condition target length mismatch")
        for clean_row, corrupt_row, no_cf_row in zip(clean, corrupt, no_cf):
            if not (
                clean_row["children"]
                == corrupt_row["children"]
                == no_cf_row["children"]
            ):
                raise ValueError("Trie children changed across conditions")
            utility = (
                clean_row["gold_log_probability"]
                - no_cf_row["gold_log_probability"]
            )
            margin = (
                clean_row["gold_log_probability"]
                - corrupt_row["gold_log_probability"]
            )
            helpful = utility > 0
            result.append(
                {
                    "dataset": dataset,
                    "user": user,
                    "split": sample["split"],
                    "sample_hash": sample["sample_hash"],
                    "depth": clean_row["depth"],
                    "child_count": len(clean_row["children"]),
                    "history_length": sample["history_length"],
                    "donor_overlap_max": sample["donor_overlap_max"],
                    "lp_clean": clean_row["gold_log_probability"],
                    "lp_corrupt": corrupt_row["gold_log_probability"],
                    "lp_no_cf": no_cf_row["gold_log_probability"],
                    "cf_utility": utility,
                    "cf_margin": margin,
                    "helpful": helpful,
                    "sensitivity_deficit": helpful and margin < deficit_margin,
                    "visible_length_identity": (
                        sample["clean_visible_lengths"]
                        == sample["corrupt_visible_lengths"]
                    ),
                    "metadata_start_identity": (
                        sample["clean_metadata_starts"]
                        == sample["corrupt_metadata_starts"]
                    ),
                }
            )
    return result


def analyze(rows: list[dict], config: dict, seed: int) -> dict:
    audit = [row for row in rows if row["split"] == "audit"]
    grouped = defaultdict(list)
    for row in audit:
        grouped[row["user"]].append(row)
    user_means = np.asarray(
        [np.mean([row["cf_margin"] for row in values]) for values in grouped.values()],
        dtype=np.float64,
    )
    margin_ci = bootstrap_mean(
        user_means,
        config["bootstrap_iterations"],
        seed,
    )
    helpful = [row for row in audit if row["helpful"]]
    deficits = [row for row in helpful if row["sensitivity_deficit"]]
    deficit_users = {row["user"] for row in deficits}
    helpful_depth_counts = Counter(
        int(row["depth"]) for row in helpful if int(row["depth"]) > 0
    )
    gates = config["scientific_gates_per_dataset"]
    supported_depths = sum(
        count >= gates["minimum_helpful_nodes_per_supported_depth"]
        for count in helpful_depth_counts.values()
    )
    metrics = {
        "audit_users": len(grouped),
        "audit_nodes": len(audit),
        "user_cluster_mean_cf_margin": margin_ci,
        "positive_user_mean_margin_rate": float((user_means > 0).mean()),
        "helpful_node_rate": len(helpful) / len(audit) if audit else 0.0,
        "sensitivity_deficit_margin": gates["sensitivity_deficit_margin"],
        "sensitivity_deficit_rate_among_helpful": (
            len(deficits) / len(helpful) if helpful else 0.0
        ),
        "deficit_user_coverage": (
            len(deficit_users) / len(grouped) if grouped else 0.0
        ),
        "helpful_nodes_by_nontrivial_depth": {
            str(depth): count for depth, count in sorted(helpful_depth_counts.items())
        },
        "supported_nontrivial_depths": supported_depths,
    }
    metrics["signal_gates"] = {
        "mean_margin_ci": margin_ci["ci95"][0]
        > gates["mean_margin_ci95_lower_strictly_greater_than"],
        "positive_user_rate": metrics["positive_user_mean_margin_rate"]
        >= gates["positive_user_mean_margin_rate_min"],
        "helpful_node_rate": metrics["helpful_node_rate"]
        >= gates["helpful_node_rate_min"],
    }
    metrics["deficit_gates"] = {
        "deficit_rate": metrics["sensitivity_deficit_rate_among_helpful"]
        >= gates["sensitivity_deficit_rate_among_helpful_min"],
        "deficit_user_coverage": metrics["deficit_user_coverage"]
        >= gates["deficit_user_coverage_min"],
        "depth_support": supported_depths
        >= gates["minimum_nontrivial_depths_with_support"],
    }
    return metrics


def integrity_pass(integrity: dict, config: dict) -> bool:
    gates = config["integrity_gates"]
    return (
        integrity["exact_split_caps"]
        and integrity["user_overlap_across_splits"]
        == gates["user_overlap_across_splits"]
        and integrity["heldout_sequence_fields_read"]
        == gates["heldout_sequence_fields_read"]
        and integrity["validation_or_test_read"] == gates["validation_or_test_read"]
        and integrity["clean_serialization_replay_rate"]
        == gates["clean_serialization_replay_rate"]
        and integrity["donor_target_exclusion_rate"]
        == gates["donor_target_exclusion_rate"]
        and integrity["neighbor_budget_identity_rate"]
        == gates["neighbor_budget_identity_rate"]
        and integrity["donor_overlap_gate_rate"] == gates["donor_overlap_gate_rate"]
        and integrity["attention_length_identity_rate"]
        == gates["attention_length_identity_rate"]
        and integrity["collator_attention_mask_identity_rate"]
        == gates["attention_length_identity_rate"]
        and integrity["metadata_start_identity_rate"]
        == gates["metadata_start_identity_rate"]
        and integrity["trie_child_membership_rate"]
        == gates["trie_child_membership_rate"]
        and integrity["finite_rate"] == gates["finite_rate"]
        and integrity["optimizer_steps"] == gates["optimizer_steps"]
        and integrity["parameter_sha_identity"] == gates["parameter_sha_identity"]
    )


def decide(results: dict) -> str:
    if not all(value["integrity_pass"] for value in results.values()):
        return "EXECUTION_INVALID"
    if not all(
        all(value["metrics"]["signal_gates"].values())
        for value in results.values()
    ):
        return "STOP_CFSAT_NO_CLEAN_CORRUPT_SIGNAL"
    if not all(
        all(value["metrics"]["deficit_gates"].values())
        for value in results.values()
    ):
        return "STOP_CFSAT_NO_TRAINABLE_DEFICIT"
    return "CFSAT_C1_DESIGN_ALLOWED"


def write_report(path: Path, aggregate: dict) -> None:
    lines = [
        "# GRAM 第四阶段：CF-SAT C0 报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-07-27",
        "- Verification Status: ANALYZED",
        "- Version Label: `cfsat_c0_v1`",
        "",
        f"固定决定：**`{aggregate['decision']}`**。",
        "",
        "| Dataset | Integrity | Margin mean [95% CI] | Positive users | Helpful nodes | Deficit/helpful | Deficit users | Supported depths |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in aggregate["datasets"].items():
        metrics = result["metrics"]
        margin = metrics["user_cluster_mean_cf_margin"]
        lines.append(
            f"| {dataset} | {result['integrity_pass']} | "
            f"{margin['mean']:.6f} [{margin['ci95'][0]:.6f}, {margin['ci95'][1]:.6f}] | "
            f"{metrics['positive_user_mean_margin_rate']:.4f} | "
            f"{metrics['helpful_node_rate']:.4f} | "
            f"{metrics['sensitivity_deficit_rate_among_helpful']:.4f} | "
            f"{metrics['deficit_user_coverage']:.4f} | "
            f"{metrics['supported_nontrivial_depths']} |"
        )
    lines.extend(
        [
            "",
            "C0 只使用 training-prefix frozen scoring，不训练、不生成 beam、",
            "不读取 validation/test。C0 通过也只解锁 C1 correctness smoke，",
            "不构成 Recall/NDCG 效果声明。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open() as handle:
        config = json.load(handle)
    if not config.get("preregistered_before_new_scores"):
        raise ValueError("C0 was not preregistered before scoring")
    started = time.time()
    prepared = {}
    preflight = {}
    for dataset, spec in config["datasets"].items():
        prepared[dataset], preflight[dataset] = prepare_dataset(
            dataset, spec, config
        )
        output_dir = args.output_root / dataset
        write_csv(
            output_dir / "cohort.csv",
            prepared[dataset]["cohort"],
            [
                "user",
                "split",
                "target",
                "history_length",
                "sample_hash",
                "donor_overlap_max",
            ],
        )
    preflight_summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-27",
            "verification_status": "ANALYZED_PREFLIGHT_ONLY",
            "version_label": "cfsat_c0_preflight_v1",
        },
        "decision": "PREFLIGHT_COMPLETE_SCORING_NOT_RUN",
        "datasets": preflight,
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
        "wall_time_seconds": time.time() - started,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "preflight_summary.json").write_text(
        json.dumps(preflight_summary, ensure_ascii=False, indent=2) + "\n"
    )
    if args.preflight_only:
        print(json.dumps({"decision": preflight_summary["decision"]}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CF-SAT C0 frozen scoring requires CUDA")

    results = {}
    device = torch.device("cuda:0")
    deficit_margin = config["scientific_gates_per_dataset"][
        "sensitivity_deficit_margin"
    ]
    for dataset, spec in config["datasets"].items():
        data = prepared[dataset]
        model, tokenizer, runtime = create_model_and_tokenizer(dataset, device)
        model.eval()
        parameter_sha_before = model_parameter_sha256(model)
        collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
        trie = encode_catalog_trie(collator, data["item2lexid"])
        scores = {}
        score_integrity = Counter()
        for condition, key in (
            ("clean", "clean_input"),
            ("corrupt", "corrupt_input"),
            ("no_cf", "no_cf_input"),
        ):
            condition_score, audit = score_condition(
                model,
                collator,
                condition_samples(data["cohort"], key),
                trie,
                config["batch_size"],
                device,
                tokenizer.eos_token_id,
            )
            scores[condition] = condition_score
            score_integrity.update(audit)
        rows = make_node_rows(
            dataset, data["cohort"], scores, deficit_margin
        )
        parameter_sha_after = model_parameter_sha256(model)
        integrity = dict(preflight[dataset]["integrity"])
        integrity["trie_child_membership_rate"] = (
            score_integrity["trie_valid"] / score_integrity["trie_checked"]
        )
        integrity["finite_rate"] = (
            score_integrity["finite_values"] / score_integrity["total_values"]
        )
        integrity["parameter_sha_identity"] = (
            parameter_sha_before == parameter_sha_after
        )
        metrics = analyze(
            rows, config, config["seed"] + (0 if dataset == "Toys" else 1)
        )
        result = {
            **preflight[dataset],
            "integrity": integrity,
            "metrics": metrics,
            "node_count": len(rows),
            "parameter_sha256": parameter_sha_after,
        }
        result["integrity_pass"] = integrity_pass(integrity, config)
        results[dataset] = result
        output_dir = args.output_root / dataset
        write_csv(
            output_dir / "node_margins.csv",
            rows,
            [
                "dataset",
                "user",
                "split",
                "sample_hash",
                "depth",
                "child_count",
                "history_length",
                "donor_overlap_max",
                "lp_clean",
                "lp_corrupt",
                "lp_no_cf",
                "cf_utility",
                "cf_margin",
                "helpful",
                "sensitivity_deficit",
                "visible_length_identity",
                "metadata_start_identity",
            ],
        )
        (output_dir / "diagnostic_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        del model
        torch.cuda.empty_cache()

    aggregate = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run + validate",
            "origin_date": "2026-07-27",
            "verification_status": "ANALYZED",
            "version_label": "cfsat_c0_v1",
        },
        "decision": decide(results),
        "datasets": results,
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
        "wall_time_seconds": time.time() - started,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(args.report, aggregate)
    print(
        json.dumps(
            {
                "decision": aggregate["decision"],
                "wall_time_seconds": aggregate["wall_time_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
