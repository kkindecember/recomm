#!/usr/bin/env python3
"""TCDR N1: training-only audit of lexical-tree score coupling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gcdh_p0 import (  # noqa: E402
    build_train_samples,
    prepare,
    read_users,
    sha256,
    write_json,
)
from experiment.phase3.marc_l0 import local_distribution  # noqa: E402
from model.gram_t5_outputs import (  # noqa: E402
    BaseModelOutputWithPastAndCrossAttentions,
)
from utils import generation_trie as gt  # noqa: E402


def stable_hash(seed: int, dataset: str, salt: str, value: str) -> str:
    return hashlib.sha256(
        f"{seed}|{dataset}|{salt}|{value}".encode()
    ).hexdigest()


def lcp_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def collaborative_cosine(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def frequency_bin(count: int) -> int:
    if count <= 0:
        raise ValueError("frequency must be positive")
    return int(math.floor(math.log2(count)))


def lexical_sequences(prepared: dict) -> dict[str, tuple[int, ...]]:
    excluded = {0, 1, 1820, 9175}
    result = {}
    for item, text in prepared["item2lexid"].items():
        sequence = tuple(
            token
            for token in prepared["tokenizer"].encode(text)
            if token not in excluded
        )
        if not sequence:
            raise ValueError(f"empty lexical sequence: {item}")
        result[item] = sequence
    return result


def item_incidence(prepared: dict, train_users: set[str]) -> dict[str, set[str]]:
    incidence: dict[str, set[str]] = defaultdict(set)
    for user in train_users:
        for item in set(prepared["sequences"][user][:-2]):
            if item in prepared["item2lexid"]:
                incidence[item].add(user)
    return incidence


def close_pairs(
    eligible: list[str],
    lexical: dict[str, tuple[int, ...]],
    incidence: dict[str, set[str]],
    config: dict,
    dataset: str,
) -> list[tuple[str, str]]:
    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    depth = int(config["close_lcp_tokens_min"])
    for item in eligible:
        if len(lexical[item]) >= depth:
            groups[lexical[item][:depth]].append(item)
    candidates = []
    maximum_cosine = float(config["collaborative_cosine_max"])
    for items in groups.values():
        ordered = sorted(items)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if collaborative_cosine(incidence[left], incidence[right]) <= maximum_cosine:
                    pair = (left, right)
                    candidates.append(pair)
    candidates.sort(
        key=lambda pair: stable_hash(
            int(config["seed"]), dataset, "tcdr-n1-close-v1", "|".join(pair)
        )
    )
    count = int(config["pairs_per_dataset"])
    if len(candidates) < count:
        raise ValueError(f"{dataset}: only {len(candidates)} eligible close pairs")
    return candidates[:count]


def matched_far_pairs(
    near_pairs: list[tuple[str, str]],
    eligible: list[str],
    lexical: dict[str, tuple[int, ...]],
    incidence: dict[str, set[str]],
    config: dict,
    dataset: str,
) -> list[tuple[str, str]]:
    by_bin: dict[int, list[str]] = defaultdict(list)
    for item in eligible:
        by_bin[frequency_bin(len(incidence[item]))].append(item)
    maximum_cosine = float(config["collaborative_cosine_max"])
    maximum_lcp = int(config["far_lcp_tokens_max"])
    used: set[tuple[str, str]] = set()
    controls = []
    for near_left, near_right in near_pairs:
        endpoint_bins = sorted(
            (
                (frequency_bin(len(incidence[near_left])), near_left),
                (frequency_bin(len(incidence[near_right])), near_right),
            )
        )
        left_bin, right_bin = endpoint_bins[0][0], endpoint_bins[1][0]
        salt = f"{near_left}|{near_right}"
        left_pool = sorted(
            by_bin[left_bin],
            key=lambda item: stable_hash(
                int(config["seed"]), dataset, f"far-left|{salt}", item
            ),
        )
        right_pool = sorted(
            by_bin[right_bin],
            key=lambda item: stable_hash(
                int(config["seed"]), dataset, f"far-right|{salt}", item
            ),
        )
        selected = None
        for left in left_pool[:256]:
            for right in right_pool[:256]:
                pair = tuple(sorted((left, right)))
                if (
                    left == right
                    or pair in used
                    or lcp_length(lexical[left], lexical[right]) > maximum_lcp
                    or collaborative_cosine(incidence[left], incidence[right])
                    > maximum_cosine
                ):
                    continue
                selected = pair
                break
            if selected is not None:
                break
        if selected is None:
            raise ValueError(f"{dataset}: no frequency-matched far pair for {salt}")
        used.add(selected)
        controls.append(selected)
    return controls


def select_user_samples(prepared: dict, train_users: set[str], config: dict, dataset: str):
    samples = build_train_samples(
        prepared["sequences"],
        train_users,
        prepared["item2input"],
        prepared["item2lexid"],
    )
    latest = {}
    for sample in samples:
        if len(sample["history_items"]) >= int(config["minimum_history_items"]):
            latest[sample["user_id"]] = sample
    ordered_users = sorted(
        latest,
        key=lambda user: stable_hash(
            int(config["seed"]), dataset, "tcdr-n1-users-v1", user
        ),
    )
    count = int(config["users_per_dataset"])
    if len(ordered_users) < count:
        raise ValueError(f"{dataset}: insufficient eligible users")
    return [latest[user] for user in ordered_users[:count]]


def bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        indices = rng.integers(0, len(values), len(values))
        estimates.append(float(values[indices].mean()))
    return {
        "point": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def score_user_items(
    model,
    collator,
    trie,
    sample: dict,
    items: list[str],
    item2lexid: dict[str, str],
    device: torch.device,
    eos_token_id: int,
    candidate_batch_size: int,
) -> tuple[list[float], dict]:
    input_batch = collator(
        [
            {
                "input": sample["input"],
                "output": item2lexid[items[0]],
                "user_id": sample["user_id"],
            }
        ]
    )
    input_ids = input_batch["item_text_ids"].to(device)
    attention = input_batch["item_text_masks"].to(device)
    model.encoder.n_passages = input_ids.size(1)
    flat_ids = input_ids.view(1, -1)
    flat_attention = attention.view(1, -1)
    hidden = model.encoder(
        input_ids=flat_ids,
        attention_mask=flat_attention,
        return_dict=True,
    )[0]
    scores = []
    counts = {"trie_checked": 0, "trie_valid": 0, "finite": 0, "values": 0}
    for start in range(0, len(items), candidate_batch_size):
        chunk = items[start : start + candidate_batch_size]
        target = collator.encode_target_split([item2lexid[item] for item in chunk])
        labels = target["input_ids"]
        masks = target["attention_mask"].bool()
        labels = labels.masked_fill(~masks, -100).to(device)
        count = len(chunk)
        output = model(
            input_ids=None,
            attention_mask=flat_attention.expand(count, -1),
            encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden.expand(count, -1, -1)
            ),
            labels=labels,
            return_dict=True,
        )
        for index in range(count):
            nodes, checked, valid = local_distribution(
                output.logits[index], labels[index], trie, eos_token_id
            )
            if not nodes:
                raise ValueError("candidate lexical path has no scored node")
            score = sum(node["gold_log_probability"] for node in nodes) / len(nodes)
            scores.append(float(score))
            counts["trie_checked"] += checked
            counts["trie_valid"] += valid
            counts["finite"] += int(math.isfinite(score))
            counts["values"] += 1
    return scores, counts


@torch.no_grad()
def audit_dataset(
    dataset: str,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    train_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    lexical = lexical_sequences(prepared)
    incidence = item_incidence(prepared, train_users)
    eligible = sorted(
        item
        for item in lexical
        if len(incidence.get(item, set()))
        >= int(config["minimum_train_user_incidence"])
    )
    near = close_pairs(eligible, lexical, incidence, config, dataset)
    far = matched_far_pairs(
        near, eligible, lexical, incidence, config, dataset
    )
    samples = select_user_samples(prepared, train_users, config, dataset)
    items = sorted({item for pair in near + far for item in pair})

    checkpoint = (
        ROOT / config["inputs"]["checkpoint_root"] / dataset / "C0" / "model.pt"
    )
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    model = prepared["model"].backbone
    model.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    eos = int(prepared["tokenizer"].eos_token_id)
    scores = np.empty((len(samples), len(items)), dtype=np.float64)
    integrity_counts = {
        "trie_checked": 0,
        "trie_valid": 0,
        "finite": 0,
        "values": 0,
    }
    batch_size = int(config["score"]["candidate_batch_size"])
    for user_index, sample in enumerate(samples):
        values, counts = score_user_items(
            model,
            prepared["collator"],
            trie,
            sample,
            items,
            prepared["item2lexid"],
            device,
            eos,
            batch_size,
        )
        for key in integrity_counts:
            integrity_counts[key] += counts[key]
        scores[user_index] = values
        if (user_index + 1) % 16 == 0:
            print(
                f"TCDR_N1_SCORE dataset={dataset} "
                f"users={user_index + 1}/{len(samples)} items={len(items)}",
                flush=True,
            )

    item_index = {item: index for index, item in enumerate(items)}
    rows = []
    for pair_index, (near_pair, far_pair) in enumerate(zip(near, far)):
        near_corr = float(
            np.corrcoef(
                scores[:, item_index[near_pair[0]]],
                scores[:, item_index[near_pair[1]]],
            )[0, 1]
        )
        far_corr = float(
            np.corrcoef(
                scores[:, item_index[far_pair[0]]],
                scores[:, item_index[far_pair[1]]],
            )[0, 1]
        )
        rows.append(
            {
                "pair_index": pair_index,
                "near_left": near_pair[0],
                "near_right": near_pair[1],
                "near_lcp": lcp_length(
                    lexical[near_pair[0]], lexical[near_pair[1]]
                ),
                "near_collaborative_cosine": collaborative_cosine(
                    incidence[near_pair[0]], incidence[near_pair[1]]
                ),
                "near_frequency_bins": "|".join(
                    map(
                        str,
                        sorted(
                            (
                                frequency_bin(len(incidence[near_pair[0]])),
                                frequency_bin(len(incidence[near_pair[1]])),
                            )
                        ),
                    )
                ),
                "far_left": far_pair[0],
                "far_right": far_pair[1],
                "far_lcp": lcp_length(
                    lexical[far_pair[0]], lexical[far_pair[1]]
                ),
                "far_collaborative_cosine": collaborative_cosine(
                    incidence[far_pair[0]], incidence[far_pair[1]]
                ),
                "far_frequency_bins": "|".join(
                    map(
                        str,
                        sorted(
                            (
                                frequency_bin(len(incidence[far_pair[0]])),
                                frequency_bin(len(incidence[far_pair[1]])),
                            )
                        ),
                    )
                ),
                "near_score_correlation": near_corr,
                "far_score_correlation": far_corr,
                "correlation_excess": near_corr - far_corr,
            }
        )
    output_dir = output_root / dataset
    pair_path = output_dir / "pair_metrics.csv"
    write_csv(pair_path, rows)
    excess = np.asarray([row["correlation_excess"] for row in rows])
    bootstrap = bootstrap_mean(
        excess,
        int(config["scientific_gates"]["bootstrap_replicates"]),
        int(config["seed"]),
    )
    metrics = {
        "users": len(samples),
        "matched_pairs": len(rows),
        "unique_scored_items": len(items),
        "median_close_score_correlation": float(
            np.median([row["near_score_correlation"] for row in rows])
        ),
        "median_far_score_correlation": float(
            np.median([row["far_score_correlation"] for row in rows])
        ),
        "median_paired_correlation_excess": float(np.median(excess)),
        "positive_excess_rate": float(np.mean(excess > 0)),
        "mean_excess_bootstrap": bootstrap,
    }
    gates = config["scientific_gates"]
    checks = {
        "matched_pairs": len(rows) >= int(gates["matched_pairs_min"]),
        "median_close_score_correlation": metrics[
            "median_close_score_correlation"
        ]
        >= float(gates["median_close_score_correlation_min"]),
        "median_paired_correlation_excess": metrics[
            "median_paired_correlation_excess"
        ]
        >= float(gates["median_paired_correlation_excess_min"]),
        "positive_excess_rate": metrics["positive_excess_rate"]
        >= float(gates["positive_excess_rate_min"]),
        "mean_excess_bootstrap_ci_lower": bootstrap["ci_low"] > 0,
    }
    frequency_matches = [
        row["near_frequency_bins"] == row["far_frequency_bins"] for row in rows
    ]
    all_values = [
        row[key]
        for row in rows
        for key in (
            "near_score_correlation",
            "far_score_correlation",
            "correlation_excess",
        )
    ]
    integrity = {
        "mapping_rate": len(lexical) / len(prepared["catalog"]),
        "trie_membership_rate": integrity_counts["trie_valid"]
        / integrity_counts["trie_checked"],
        "finite_rate": (
            integrity_counts["finite"] / integrity_counts["values"]
            if integrity_counts["values"]
            else 0.0
        ),
        "pair_metric_finite_rate": float(
            np.mean([math.isfinite(value) for value in all_values])
        ),
        "frequency_bin_match_rate": float(np.mean(frequency_matches)),
        "unique_user_rate": len({row["user_id"] for row in samples}) / len(samples),
        "optimizer_steps": 0,
        "checkpoint_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "validation_test_read": False,
        "sports_read": False,
        "pair_metrics_sha256": sha256(pair_path),
    }
    integrity_valid = (
        integrity["mapping_rate"] == 1.0
        and integrity["trie_membership_rate"] == 1.0
        and integrity["finite_rate"] == 1.0
        and integrity["pair_metric_finite_rate"] == 1.0
        and integrity["frequency_bin_match_rate"] == 1.0
        and integrity["unique_user_rate"] == 1.0
        and integrity["optimizer_steps"] == 0
        and integrity["checkpoint_sha_unchanged"]
        and not integrity["validation_test_read"]
        and not integrity["sports_read"]
    )
    result = {
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": all(checks.values()),
        "integrity": integrity,
        "integrity_valid": integrity_valid,
    }
    write_json(output_dir / "summary.json", result)
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
        raise ValueError("TCDR N1 code SHA mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("TCDR N1 requires CUDA")
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    results = {
        dataset: audit_dataset(
            dataset, config, p0_config, args.output_root, device
        )
        for dataset in config["datasets"]
    }
    integrity_valid = all(row["integrity_valid"] for row in results.values())
    scientific_pass = all(row["scientific_pass"] for row in results.values())
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "TCDR_S0_DESIGN_ALLOWED"
        if scientific_pass
        else "STOP_TCDR_NO_TREE_COUPLING_DEFICIT"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "datasets": results,
        "integrity_valid": integrity_valid,
        "optimizer_steps": 0,
        "validation_test_read": False,
        "sports_read": False,
    }
    write_json(args.output_root / "summary.json", summary)
    write_json(
        args.output_root / "status.json",
        {"experiment_id": config["experiment_id"], "status": "completed"},
    )
    (args.output_root / "decision.md").write_text(
        "# TCDR-N1 Decision\n\n"
        f"- Fixed decision: **`{decision}`**\n"
        f"- Integrity valid: `{str(integrity_valid).lower()}`\n"
        "- Validation/test/Sports read: `false`\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
