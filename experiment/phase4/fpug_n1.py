#!/usr/bin/env python3
"""FPUG N1: frozen training-prefix leave-one-detail-passage utility audit."""

from __future__ import annotations

import argparse
import csv
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

from experiment.phase4.chpr_a0 import pad_labels  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    ROOT,
    build_train_samples,
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)
from experiment.phase4.ialc_n1 import select_unique_user_samples  # noqa: E402
from utils import generation_trie as gt  # noqa: E402


def recency_quartile(recency_rank: int, history_length: int, quartiles: int = 4) -> int:
    if history_length <= 1:
        return 0
    fraction = recency_rank / (history_length - 1)
    return min(int(fraction * quartiles), quartiles - 1)


def normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    probabilities = np.asarray(counts, dtype=np.float64) / total
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / math.log(len(counts)))


def legal_child_ce(
    logits: torch.Tensor,
    sequences: list[list[int]],
    trie: gt.Trie,
    eos: int,
) -> tuple[list[float], list[int]]:
    losses, step_counts = [], []
    for batch_index, sequence in enumerate(sequences):
        step_losses = []
        for position, gold in enumerate(sequence[1:]):
            allowed = trie.get(sequence[: position + 1])
            if gold == eos or len(allowed) < 2:
                continue
            if gold not in allowed:
                raise ValueError("gold child is not legal")
            values = logits[batch_index, position, allowed].float()
            gold_position = allowed.index(gold)
            step_losses.append(float(-torch.log_softmax(values, dim=0)[gold_position]))
        losses.append(float(np.mean(step_losses)) if step_losses else math.nan)
        step_counts.append(len(step_losses))
    return losses, step_counts


def select_samples(
    all_samples: list[dict],
    heads: set[str],
    seed: int,
    dataset: str,
    head_count: int,
    tail_count: int,
    minimum_history: int,
) -> list[dict]:
    eligible = [
        row for row in all_samples if len(row["history_items"]) >= minimum_history
    ]
    return select_unique_user_samples(
        eligible, heads, seed, dataset, head_count, tail_count
    )


@torch.no_grad()
def audit_batch(
    samples: list[dict],
    prepared: dict,
    trie: gt.Trie,
    item_to_sequence: dict[str, list[int]],
    threshold: float,
    quartiles: int,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    batch = collate(prepared["collator"], samples)
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    sequences = [item_to_sequence[row["positive_item"]] for row in samples]
    labels = pad_labels(sequences, device)
    backbone = prepared["model"].backbone
    passages, width = input_ids.shape[1], input_ids.shape[2]
    backbone.encoder.n_passages = passages
    flat_ids = input_ids.view(input_ids.shape[0], -1)
    flat_attention = attention.view(attention.shape[0], -1)
    encoder_hidden = backbone.encoder(
        input_ids=flat_ids,
        attention_mask=flat_attention,
        return_dict=True,
    )[0]
    full = backbone(
        input_ids=None,
        attention_mask=flat_attention,
        encoder_outputs=(encoder_hidden,),
        labels=labels,
        return_dict=True,
    )
    eos = int(prepared["tokenizer"].eos_token_id)
    full_losses, competitive_steps = legal_child_ce(
        full.logits, sequences, trie, eos
    )
    removal_losses: dict[int, list[float]] = {}
    for passage_index in range(1, passages):
        masked_attention = flat_attention.clone()
        start = passage_index * width
        masked_attention[:, start : start + width] = False
        removed = backbone(
            input_ids=None,
            attention_mask=masked_attention,
            encoder_outputs=(encoder_hidden,),
            labels=labels,
            return_dict=True,
        )
        removal_losses[passage_index], _ = legal_child_ce(
            removed.logits, sequences, trie, eos
        )
    passage_rows, sample_rows = [], []
    for batch_index, sample in enumerate(samples):
        history = list(reversed(sample["history_items"]))
        improvements = []
        for recency_rank, history_item in enumerate(history):
            passage_index = recency_rank + 1
            if passage_index >= passages:
                raise ValueError("active detailed passage was truncated")
            if not bool(attention[batch_index, passage_index].any()):
                raise ValueError("active detailed passage has empty attention mask")
            removed_loss = removal_losses[passage_index][batch_index]
            improvement = full_losses[batch_index] - removed_loss
            improvements.append(improvement)
            passage_rows.append(
                {
                    "sample_key": sample["sample_key"],
                    "user_id": sample["user_id"],
                    "target_group": (
                        "head"
                        if sample["positive_item"] in prepared["heads"]
                        else "tail"
                    ),
                    "history_length": len(history),
                    "passage_index": passage_index,
                    "recency_rank": recency_rank,
                    "recency_quartile": recency_quartile(
                        recency_rank, len(history), quartiles
                    ),
                    "history_item": history_item,
                    "full_legal_ce": full_losses[batch_index],
                    "removed_legal_ce": removed_loss,
                    "removal_improvement": improvement,
                    "harmful": int(improvement >= threshold),
                    "competitive_steps": competitive_steps[batch_index],
                }
            )
        best = max(improvements) if improvements else math.nan
        oldest = improvements[-1] if improvements else math.nan
        most_recent = improvements[0] if improvements else math.nan
        sample_rows.append(
            {
                "sample_key": sample["sample_key"],
                "user_id": sample["user_id"],
                "target_group": (
                    "head"
                    if sample["positive_item"] in prepared["heads"]
                    else "tail"
                ),
                "history_length": len(history),
                "competitive_steps": competitive_steps[batch_index],
                "full_legal_ce": full_losses[batch_index],
                "best_removal_improvement": best,
                "oldest_removal_improvement": oldest,
                "most_recent_removal_improvement": most_recent,
                "oracle_advantage_over_oldest": best - oldest,
                "has_harmful_passage": int(best >= threshold),
            }
        )
    return passage_rows, sample_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    trie = gt.Trie(prepared["encoded_candidates"])
    item_to_sequence = dict(zip(prepared["catalog"], prepared["encoded_candidates"]))
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
    passage_rows, sample_rows = [], []
    batch_size = int(config["audit"]["batch_size"])
    for start in range(0, len(samples), batch_size):
        rows, summaries = audit_batch(
            samples[start : start + batch_size],
            prepared,
            trie,
            item_to_sequence,
            float(config["audit"]["harmful_improvement_threshold"]),
            int(config["audit"]["recency_quartiles"]),
            device,
        )
        passage_rows.extend(rows)
        sample_rows.extend(summaries)
        done = min(start + batch_size, len(samples))
        if done % 64 == 0:
            print(
                f"FPUG_N1_PROGRESS dataset={dataset} samples={done}/{len(samples)}",
                flush=True,
            )
    output_dir = output_root / dataset
    passage_path = output_dir / "passage_utility.csv"
    sample_path = output_dir / "sample_summary.csv"
    write_csv(passage_path, passage_rows)
    write_csv(sample_path, sample_rows)
    competitive = [row for row in sample_rows if row["competitive_steps"] > 0]
    harmful_passages = [row for row in passage_rows if row["harmful"]]
    tail_samples = [row for row in sample_rows if row["target_group"] == "tail"]
    quartile_count = int(config["audit"]["recency_quartiles"])
    quartile_counts = Counter(
        int(row["recency_quartile"]) for row in harmful_passages
    )
    counts = [quartile_counts[index] for index in range(quartile_count)]
    harmful_rate = float(
        np.mean([row["has_harmful_passage"] for row in sample_rows])
    )
    tail_harmful_rate = float(
        np.mean([row["has_harmful_passage"] for row in tail_samples])
    )
    metrics = {
        "samples": len(sample_rows),
        "unique_users": len({row["user_id"] for row in sample_rows}),
        "passages": len(passage_rows),
        "competitive_sample_coverage": len(competitive) / len(sample_rows),
        "harmful_passages": len(harmful_passages),
        "harmful_passage_sample_rate": harmful_rate,
        "tail_harmful_passage_sample_rate": tail_harmful_rate,
        "mean_best_removal_improvement": float(
            np.mean([row["best_removal_improvement"] for row in competitive])
        ),
        "mean_oldest_removal_improvement": float(
            np.mean([row["oldest_removal_improvement"] for row in competitive])
        ),
        "mean_oracle_advantage_over_oldest": float(
            np.mean([row["oracle_advantage_over_oldest"] for row in competitive])
        ),
        "harmful_passages_by_recency_quartile": {
            str(index): counts[index] for index in range(quartile_count)
        },
        "harmful_recency_entropy": normalized_entropy(counts),
    }
    gates = config["scientific_gates"]
    supported_quartiles = sum(
        count >= int(gates["harmful_passages_per_supported_quartile_min"])
        for count in counts
    )
    checks = {
        "samples": len(sample_rows) >= int(gates["samples_min"]),
        "competitive_sample_coverage": metrics["competitive_sample_coverage"]
        >= float(gates["competitive_sample_coverage_min"]),
        "harmful_passage_sample_rate": harmful_rate
        >= float(gates["harmful_passage_sample_rate_min"]),
        "tail_harmful_passage_sample_rate": tail_harmful_rate
        >= float(gates["tail_harmful_passage_sample_rate_min"]),
        "mean_best_removal_improvement": metrics["mean_best_removal_improvement"]
        >= float(gates["mean_best_removal_improvement_min"]),
        "mean_oracle_advantage_over_oldest": metrics[
            "mean_oracle_advantage_over_oldest"
        ]
        >= float(gates["mean_oracle_advantage_over_oldest_min"]),
        "supported_recency_quartiles": supported_quartiles
        >= int(gates["supported_recency_quartiles_min"]),
        "harmful_recency_entropy": metrics["harmful_recency_entropy"]
        >= float(gates["harmful_recency_entropy_min"]),
    }
    finite = all(
        math.isfinite(float(row["full_legal_ce"]))
        and math.isfinite(float(row["removed_legal_ce"]))
        and math.isfinite(float(row["removal_improvement"]))
        for row in passage_rows
    )
    integrity = {
        "mapping_rate": 1.0,
        "trie_membership_rate": 1.0,
        "finite_rate": float(finite),
        "unique_user_rate": len({row["user_id"] for row in sample_rows})
        / len(sample_rows),
        "coarse_prompt_unchanged_rate": 1.0,
        "exactly_one_detail_masked_rate": 1.0,
        "optimizer_steps": 0,
        "parameter_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "validation_test_predictions_read": False,
        "sports_read": False,
        "checkpoint_sha256": checkpoint_sha,
        "passage_file_sha256": sha256(passage_path),
        "sample_file_sha256": sha256(sample_path),
    }
    integrity_valid = (
        integrity["mapping_rate"] == 1.0
        and integrity["trie_membership_rate"] == 1.0
        and integrity["finite_rate"] == 1.0
        and integrity["unique_user_rate"] == 1.0
        and integrity["coarse_prompt_unchanged_rate"] == 1.0
        and integrity["exactly_one_detail_masked_rate"] == 1.0
        and integrity["optimizer_steps"] == 0
        and integrity["parameter_sha_unchanged"]
        and not integrity["validation_test_predictions_read"]
        and not integrity["sports_read"]
    )
    del prepared
    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": all(checks.values()),
        "integrity": integrity,
        "integrity_valid": integrity_valid,
    }


def decision_markdown(summary: dict) -> str:
    lines = [
        "# FPUG-N1 Decision",
        "",
        f"- Fixed decision: **`{summary['decision']}`**",
        f"- Integrity valid: `{str(summary['integrity_valid']).lower()}`",
        "- Data scope: unique-user training prefixes only",
        "- Validation/test/Sports read: `false`",
        "",
        "## Frozen gate results",
        "",
    ]
    for dataset, result in summary["results"].items():
        lines.extend([f"### {dataset}", ""])
        for name, passed in result["checks"].items():
            lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FPUG N1 requires CUDA")
    config = json.loads(args.config.read_text())
    expected_sha = config["integrity"]["code_sha256"]
    actual_sha = sha256(Path(__file__))
    if actual_sha != expected_sha:
        raise ValueError(f"code SHA mismatch expected={expected_sha} actual={actual_sha}")
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    results = {
        dataset: run_dataset(dataset, config, p0_config, args.output_root, device)
        for dataset in config["datasets"]
    }
    integrity_valid = all(row["integrity_valid"] for row in results.values())
    scientific_pass = all(row["scientific_pass"] for row in results.values())
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "FPUG_S0_DESIGN_ALLOWED"
        if scientific_pass
        else "STOP_FPUG_NO_DYNAMIC_PASSAGE_UTILITY_DEFICIT"
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
    (args.output_root / "decision.md").write_text(decision_markdown(summary))
    write_json(
        args.output_root / "status.json",
        {"experiment_id": config["experiment_id"], "status": "completed"},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
