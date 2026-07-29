#!/usr/bin/env python3
"""CHPR A0: frozen earliest-divergence hard-negative prefix audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.gcdh_p0 import (  # noqa: E402
    ROOT,
    build_train_samples,
    collate,
    normalized_sequence,
    prepare,
    read_users,
    sha256,
    write_json,
)
from utils import generation_trie as gt  # noqa: E402


def hash_key(seed: int, dataset: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|chpr-a0|{value}".encode()).hexdigest()


def select_unique_user_samples(
    samples: list[dict],
    head_items: set[str],
    seed: int,
    dataset: str,
    head_count: int,
    tail_count: int,
) -> list[dict]:
    ordered = sorted(
        samples,
        key=lambda row: hash_key(seed, dataset, row["sample_key"]),
    )
    selected = []
    used_users = set()
    counts = {"head": 0, "tail": 0}
    limits = {"head": head_count, "tail": tail_count}
    for row in ordered:
        group = "head" if row["positive_item"] in head_items else "tail"
        if row["user_id"] in used_users or counts[group] >= limits[group]:
            continue
        selected.append(row)
        used_users.add(row["user_id"])
        counts[group] += 1
        if counts == limits:
            break
    if counts != limits:
        raise ValueError(f"insufficient unique-user samples for {dataset}: {counts}")
    return sorted(selected, key=lambda row: row["sample_key"])


def earliest_divergence(gold: list[int], negative: list[int]) -> int:
    for index, (left, right) in enumerate(zip(gold, negative)):
        if left != right:
            return index
    raise ValueError("identical or prefix-only lexical IDs are not valid negatives")


def pad_labels(sequences: list[list[int]], device: torch.device) -> torch.Tensor:
    maximum = max(len(sequence) - 1 for sequence in sequences)
    labels = torch.full(
        (len(sequences), maximum), -100, dtype=torch.long, device=device
    )
    for row, sequence in enumerate(sequences):
        values = torch.tensor(sequence[1:], dtype=torch.long, device=device)
        labels[row, : len(values)] = values
    return labels


@torch.no_grad()
def score_candidates(
    backbone,
    encoder_hidden: torch.Tensor,
    flat_attention: torch.Tensor,
    sequences: list[list[int]],
    trie: gt.Trie,
    batch_size: int,
) -> list[dict]:
    results = []
    for start in range(0, len(sequences), batch_size):
        selected = sequences[start : start + batch_size]
        labels = pad_labels(selected, encoder_hidden.device)
        repeated_hidden = encoder_hidden.expand(len(selected), -1, -1)
        repeated_attention = flat_attention.expand(len(selected), -1)
        output = backbone(
            input_ids=None,
            attention_mask=repeated_attention,
            encoder_outputs=(repeated_hidden,),
            labels=labels,
            return_dict=True,
        )
        log_probs = torch.log_softmax(output.logits.float(), dim=-1)
        valid = labels != -100
        gathered = log_probs.gather(
            -1, labels.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)
        scores = (gathered * valid).sum(dim=1) / valid.sum(dim=1)
        for offset, sequence in enumerate(selected):
            if not torch.isfinite(scores[offset]):
                raise ValueError("non-finite exact path score")
            results.append(
                {
                    "score": float(scores[offset]),
                    "log_probs": log_probs[offset].detach(),
                    "sequence": sequence,
                }
            )
    return results


@torch.no_grad()
def audit_sample(
    sample: dict,
    scorer: dict,
    proposer: dict,
    config: dict,
    device: torch.device,
) -> tuple[list[dict], dict]:
    scorer_model = scorer["model"]
    proposer_model = proposer["model"]
    inference_sample = dict(sample)
    inference_sample["output"] = scorer["item2lexid"][scorer["catalog"][0]]
    batch = collate(scorer["collator"], [inference_sample])
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)

    prediction = scorer_model.backbone.generate(
        input_ids=input_ids,
        attention_mask=attention,
        max_length=max(len(row) for row in scorer["encoded_candidates"]),
        prefix_allowed_tokens_fn=gt.prefix_allowed_tokens_fn(scorer["_chpr_trie"]),
        num_beams=int(config["proposal_construction"]["generator_top_k"]),
        num_return_sequences=int(config["proposal_construction"]["generator_top_k"]),
        return_dict_in_generate=True,
        length_penalty=1.0,
    )
    gram_items = [
        scorer["sequence_to_item"].get(normalized_sequence(row.tolist()))
        for row in prediction["sequences"]
    ]
    if any(item is None for item in gram_items) or len(set(gram_items)) != len(gram_items):
        raise ValueError("generator mapping or deduplication failure")

    proposer_model.backbone.encoder.n_passages = input_ids.size(1)
    proposer_logits = proposer_model.catalog_logits(input_ids, attention)[0]
    for item in sample["history_items"]:
        if item in proposer_model.item_to_index:
            proposer_logits[proposer_model.item_to_index[item]] = -torch.inf
    top_indices = torch.topk(
        proposer_logits,
        k=int(config["proposal_construction"]["catalog_top_k"]),
    ).indices.tolist()
    catalog_items = [proposer["catalog"][index] for index in top_indices]
    source = {}
    for item in gram_items:
        source[item] = "gram"
    for item in catalog_items:
        source[item] = "both" if item in source else "catalog"
    proposal_items = list(dict.fromkeys(gram_items + catalog_items))
    target = sample["positive_item"]
    forbidden = set(sample["history_items"]) | {target}
    negatives = [item for item in proposal_items if item not in forbidden]

    scorer_model.backbone.encoder.n_passages = input_ids.size(1)
    flat_ids = input_ids.view(1, -1)
    flat_attention = attention.view(1, -1)
    encoder_hidden = scorer_model.backbone.encoder(
        input_ids=flat_ids,
        attention_mask=flat_attention,
        return_dict=True,
    )[0]
    item_to_sequence = scorer["_chpr_item_to_sequence"]
    gold_sequence = item_to_sequence[target]
    negative_sequences = [item_to_sequence[item] for item in negatives]
    scored = score_candidates(
        scorer_model.backbone,
        encoder_hidden,
        flat_attention,
        negative_sequences,
        scorer["_chpr_trie"],
        int(config["execution"]["candidate_score_batch_size"]),
    )
    ranked = sorted(
        zip(negatives, scored),
        key=lambda pair: (-pair[1]["score"], pair[0]),
    )[: int(config["proposal_construction"]["hard_negatives_per_sample"])]
    pair_rows = []
    threshold = float(config["prefix_metric"]["deficit_threshold"])
    for negative_item, result in ranked:
        negative_sequence = result["sequence"]
        depth = earliest_divergence(gold_sequence, negative_sequence)
        if depth == 0:
            raise ValueError("BOS divergence is invalid")
        prefix = gold_sequence[:depth]
        allowed = scorer["_chpr_trie"].get(prefix)
        gold_child = gold_sequence[depth]
        negative_child = negative_sequence[depth]
        if gold_child not in allowed or negative_child not in allowed:
            raise ValueError("divergent child is not legal in Trie")
        legal_log_probs = torch.log_softmax(
            result["log_probs"][depth - 1, allowed], dim=0
        )
        child_to_position = {child: index for index, child in enumerate(allowed)}
        gold_lp = float(legal_log_probs[child_to_position[gold_child]])
        negative_lp = float(legal_log_probs[child_to_position[negative_child]])
        margin = gold_lp - negative_lp
        if not math.isfinite(margin):
            raise ValueError("non-finite prefix margin")
        pair_rows.append(
            {
                "sample_key": sample["sample_key"],
                "user_id": sample["user_id"],
                "target_group": (
                    "head" if target in scorer["heads"] else "tail"
                ),
                "negative_item": negative_item,
                "source": source[negative_item],
                "exact_path_score": result["score"],
                "divergence_depth": depth - 1,
                "legal_child_count": len(allowed),
                "gold_child_logp": gold_lp,
                "negative_child_logp": negative_lp,
                "margin": margin,
                "deficit": int(margin < threshold),
            }
        )
    gram_rank = gram_items.index(target) + 1 if target in gram_items else None
    sample_summary = {
        "sample_key": sample["sample_key"],
        "user_id": sample["user_id"],
        "target_group": "head" if target in scorer["heads"] else "tail",
        "valid_negative_count": len(negatives),
        "hard_negative_count": len(pair_rows),
        "deficit": int(any(row["deficit"] for row in pair_rows)),
        "minimum_margin": min((row["margin"] for row in pair_rows), default=float("nan")),
        "gram_rank": gram_rank,
        "gram_hit50": int(gram_rank is not None),
        "history_gold_exclusion": int(
            all(item not in forbidden for item in negatives)
        ),
    }
    return pair_rows, sample_summary


def run_dataset(
    dataset: str,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    scorer = prepare(dataset, p0_config, device)
    proposer = prepare(dataset, p0_config, device)
    if scorer["catalog"] != proposer["catalog"]:
        raise ValueError("scorer/proposer catalog mismatch")
    checkpoint_root = ROOT / config["inputs"]["checkpoint_root"] / dataset
    scorer_checkpoint = checkpoint_root / "C0" / "model.pt"
    proposer_checkpoint = checkpoint_root / "C1" / "model.pt"
    scorer_sha = sha256(scorer_checkpoint)
    proposer_sha = sha256(proposer_checkpoint)
    scorer["model"].load_state_dict(
        torch.load(scorer_checkpoint, map_location=device), strict=True
    )
    proposer["model"].load_state_dict(
        torch.load(proposer_checkpoint, map_location=device), strict=True
    )
    scorer["model"].eval()
    proposer["model"].eval()
    scorer["_chpr_trie"] = gt.Trie(scorer["encoded_candidates"])
    scorer["_chpr_item_to_sequence"] = {
        item: sequence
        for item, sequence in zip(scorer["catalog"], scorer["encoded_candidates"])
    }
    users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    samples = build_train_samples(
        scorer["sequences"],
        users,
        scorer["item2input"],
        scorer["item2lexid"],
    )
    selected = select_unique_user_samples(
        samples,
        scorer["heads"],
        int(config["seed"]),
        dataset,
        int(config["head_samples"]),
        int(config["tail_samples"]),
    )
    pair_rows, sample_rows = [], []
    for index, sample in enumerate(selected, 1):
        pairs, summary = audit_sample(
            sample, scorer, proposer, config, device
        )
        pair_rows.extend(pairs)
        sample_rows.append(summary)
        if index % 32 == 0:
            print(
                f"CHPR_A0_PROGRESS dataset={dataset} samples={index}/{len(selected)}",
                flush=True,
            )
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "prefix_pairs.csv"
    sample_path = output_dir / "sample_summary.csv"
    with pair_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    with sample_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    valid = [row for row in sample_rows if row["hard_negative_count"] >= 8]
    tail = [row for row in valid if row["target_group"] == "tail"]
    deficit = [row for row in valid if row["deficit"]]
    deficit_pairs = [row for row in pair_rows if row["deficit"]]
    depth_counts = Counter(row["divergence_depth"] for row in deficit_pairs)
    supported_depths = {
        str(depth): count
        for depth, count in sorted(depth_counts.items())
        if depth > 0
        and count
        >= int(
            config["scientific_gates"][
                "minimum_deficit_pairs_per_supported_depth"
            ]
        )
    }
    hit = [row for row in valid if row["gram_hit50"]]
    miss = [row for row in valid if not row["gram_hit50"]]

    def deficit_rate(rows: list[dict]) -> float:
        return float(np.mean([row["deficit"] for row in rows])) if rows else 0.0

    metrics = {
        "samples": len(sample_rows),
        "unique_users": len({row["user_id"] for row in sample_rows}),
        "valid_proposal_sample_rate": len(valid) / len(sample_rows),
        "deficit_sample_rate": deficit_rate(valid),
        "tail_samples": len(tail),
        "tail_deficit_sample_rate": deficit_rate(tail),
        "deficit_user_coverage": len({row["user_id"] for row in deficit})
        / len({row["user_id"] for row in valid}),
        "mean_minimum_margin": float(
            np.mean([row["minimum_margin"] for row in valid])
        ),
        "beam_hit50_deficit_rate": deficit_rate(hit),
        "beam_miss50_deficit_rate": deficit_rate(miss),
        "deficit_pairs": len(deficit_pairs),
        "deficit_pairs_by_depth": {
            str(depth): count for depth, count in sorted(depth_counts.items())
        },
        "supported_nontrivial_depths": supported_depths,
    }
    gates = config["scientific_gates"]
    checks = {
        "valid_proposal_sample_rate": metrics["valid_proposal_sample_rate"]
        >= gates["valid_proposal_sample_rate_min"],
        "deficit_sample_rate": metrics["deficit_sample_rate"]
        >= gates["deficit_sample_rate_min"],
        "tail_deficit_sample_rate": metrics["tail_deficit_sample_rate"]
        >= gates["tail_deficit_sample_rate_min"],
        "deficit_user_coverage": metrics["deficit_user_coverage"]
        >= gates["deficit_user_coverage_min"],
        "supported_nontrivial_depths": len(supported_depths)
        >= gates["minimum_supported_nontrivial_depths"],
    }
    integrity = {
        "mapping_rate": 1.0,
        "trie_membership_rate": 1.0,
        "finite_rate": 1.0,
        "history_gold_exclusion_rate": float(
            np.mean([row["history_gold_exclusion"] for row in sample_rows])
        ),
        "unique_user_rate": metrics["unique_users"] / metrics["samples"],
        "optimizer_steps": 0,
        "scorer_parameter_sha_unchanged": scorer_sha == sha256(scorer_checkpoint),
        "proposer_parameter_sha_unchanged": proposer_sha
        == sha256(proposer_checkpoint),
        "validation_test_read": False,
        "pair_file_sha256": sha256(pair_path),
        "sample_file_sha256": sha256(sample_path),
    }
    integrity_valid = (
        integrity["mapping_rate"] == 1.0
        and integrity["trie_membership_rate"] == 1.0
        and integrity["finite_rate"] == 1.0
        and integrity["history_gold_exclusion_rate"] == 1.0
        and integrity["unique_user_rate"] == 1.0
        and integrity["optimizer_steps"] == 0
        and integrity["scorer_parameter_sha_unchanged"]
        and integrity["proposer_parameter_sha_unchanged"]
        and not integrity["validation_test_read"]
    )
    del scorer, proposer
    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": all(checks.values()),
        "integrity": integrity,
        "integrity_valid": integrity_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CHPR A0 requires CUDA")
    config = json.loads(args.config.read_text())
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    results = {
        dataset: run_dataset(
            dataset, config, p0_config, args.output_root, device
        )
        for dataset in config["datasets"]
    }
    integrity_valid = all(
        result["integrity_valid"] for result in results.values()
    )
    scientific_pass = all(
        result["scientific_pass"] for result in results.values()
    )
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "CHPR_S0_DESIGN_ALLOWED"
        if scientific_pass
        else "STOP_CHPR_NO_PREFIX_RANKING_DEFICIT"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "results": results,
        "integrity_valid": integrity_valid,
        "test_data_read": False,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
