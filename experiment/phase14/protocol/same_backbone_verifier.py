"""Stage14-0C frozen R2-candidate / GRAM-likelihood interface control.

The experiment is validation-only and does not train or mutate either model.
For each user it scores the already frozen R2 top-50 catalog candidates with
the frozen v0 GRAM decoder under teacher forcing.  The score is the mean raw
token log-likelihood over the complete lexical path (EOS excluded), matching
SpecGR's target-aware mean token likelihood while avoiding a domain-specific
acceptance threshold.

Four frozen controls are reported on exactly the same users and evaluator:

* ``r2_candidate_score_only``: the stored R2 ordering;
* ``gram_candidate_likelihood_only``: the stored native GRAM beam ordering;
* ``r2_plus_gram_verifier``: R2 top-50 reranked by GRAM path likelihood;
* ``r2_portfolio_at_2``: the Phase-13 frozen top-7 + two cold-candidate rule.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from item_level_eval import atomic_json, load_item_paths, metrics_for_rank, sha256_file
from oracle_prefix_probe import (
    CollatorGRAM,
    TestDatasetGRAM,
    batch_to_device,
    configure_model,
    encode_lexical_path,
    make_dataset_args,
    read_sequences,
    read_set,
)
from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions


METHODS = (
    "r2_candidate_score_only",
    "gram_candidate_likelihood_only",
    "r2_plus_gram_verifier",
    "r2_portfolio_at_2",
)


def read_r2_rows(path: Path) -> dict[str, dict]:
    if "test" in path.name.lower():
        raise ValueError(f"Stage14-0C refuses test predictions: {path}")
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            user = str(row["user_id"])
            if user in rows:
                raise ValueError(f"{path}:{line_no}: duplicate user {user}")
            rows[user] = row
    if not rows:
        raise ValueError(f"No R2 prediction rows in {path}")
    return rows


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def portfolio_at_2(gram: list[str], resolver: list[str], cold_items: set[str]) -> list[str]:
    """Frozen Phase-13 rule: protect top-7 and insert two cold resolver items at 9-10."""
    gram = unique_in_order(gram)
    resolver = unique_in_order(resolver)
    protected = set(gram[:7])
    candidates = [item for item in resolver if item in cold_items and item not in protected][:2]
    if len(candidates) != 2:
        raise ValueError("Cannot construct the frozen two-item cold portfolio")
    return unique_in_order([*gram[:8], *candidates, *gram[8:], *resolver])


def pad_candidate_paths(paths: list[tuple[int, ...]], device: torch.device) -> torch.Tensor:
    if not paths or any(not path for path in paths):
        raise ValueError("Candidate lexical paths must be non-empty")
    labels = torch.full((len(paths), max(map(len, paths))), -100, dtype=torch.long, device=device)
    for row, path in enumerate(paths):
        labels[row, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
    return labels


def mean_path_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean raw-vocabulary log P(path token | history, previous path tokens)."""
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logit/label shape mismatch: {logits.shape} vs {labels.shape}")
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logp = F.log_softmax(logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def rank_metrics(ranking: list[str], target: str) -> dict[str, float]:
    rank = ranking.index(target) + 1 if target in ranking else None
    return metrics_for_rank(rank)


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float | None]:
    names = tuple(metrics_for_rank(None))
    if not rows:
        return {name: None for name in names}
    return {name: float(np.mean([row[name] for row in rows])) for name in names}


def paired_bootstrap(
    rows: list[dict], method: str, baseline: str, metric: str, slice_name: str,
    resamples: int, seed: int,
) -> dict:
    selected = [
        row for row in rows
        if slice_name == "all" or (slice_name == "cold") == bool(row["is_cold"])
    ]
    delta = np.asarray(
        [row["metrics"][method][metric] - row["metrics"][baseline][metric] for row in selected],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 250):
        count = min(250, resamples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": len(delta),
        "observed": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def score_candidate_batch(
    model,
    batch: dict,
    candidate_items: list[list[str]],
    item_path_ids: dict[str, tuple[int, ...]],
    candidate_count: int,
    candidate_chunk_size: int,
) -> torch.Tensor:
    """Encode each history once, then decode bounded chunks of candidate paths."""
    device = batch["item_text_ids"].device
    if any(len(items) != candidate_count for items in candidate_items):
        raise ValueError("Every user must have exactly the frozen candidate budget")
    passages = batch["item_text_ids"].size(1)
    flat_input = batch["item_text_ids"].reshape(batch["item_text_ids"].size(0), -1)
    flat_mask = batch["item_text_masks"].reshape(batch["item_text_masks"].size(0), -1)
    model.encoder.n_passages = passages
    encoder_hidden = model.encoder(
        input_ids=flat_input,
        attention_mask=flat_mask,
        return_dict=True,
    )[0]
    score_chunks = []
    for start in range(0, candidate_count, candidate_chunk_size):
        stop = min(start + candidate_chunk_size, candidate_count)
        chunk_size = stop - start
        labels = pad_candidate_paths(
            [
                item_path_ids[item]
                for items in candidate_items
                for item in items[start:stop]
            ],
            device,
        )
        repeated_hidden = encoder_hidden.repeat_interleave(chunk_size, dim=0)
        repeated_mask = flat_mask.repeat_interleave(chunk_size, dim=0)
        outputs = model(
            encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=repeated_hidden
            ),
            attention_mask=repeated_mask,
            labels=labels,
        )
        score_chunks.append(
            mean_path_log_likelihood(outputs.logits, labels).view(len(candidate_items), chunk_size)
        )
    return torch.cat(score_chunks, dim=1)


def run(args: argparse.Namespace) -> dict:
    if args.split != "validation":
        raise ValueError("Stage14-0C permits validation only")
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    historical = json.loads(args.historical_config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(historical["backbone"])
    dataset_args = make_dataset_args(historical, args.dataset_dir.resolve())
    dataset = TestDatasetGRAM(
        args=dataset_args,
        dataset=args.dataset_dir.name,
        task="sequential",
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        debug_test_small_set=False,
        mode="validation",
    )
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CollatorGRAM(tokenizer, args=dataset_args, mode="valid"),
    )
    sequences = read_sequences(args.dataset_dir / "user_sequence.txt")
    cold_items = read_set(args.dataset_dir / "cold_split_meta" / "cold_items.txt")
    item_to_lexical, decoded_to_items = load_item_paths(args.item_path_file)
    if any(len(items) != 1 for items in decoded_to_items.values()):
        raise RuntimeError("Stage14-0C requires collision-free item identifiers")
    item_path_ids = {
        item: encode_lexical_path(tokenizer, lexical)
        for item, lexical in item_to_lexical.items()
    }
    frozen = read_r2_rows(args.r2_predictions)
    model = configure_model(historical, args.checkpoint, device)
    rows: list[dict] = []
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader, 1):
            batch = batch_to_device(raw_batch, device)
            users = [str(user) for user in batch["user_ids"]]
            source_rows = [frozen[user] for user in users]
            candidates = [unique_in_order(row["r2_top50"]) for row in source_rows]
            for user, items in zip(users, candidates):
                if len(items) != args.candidate_count:
                    raise ValueError(f"{user}: R2 candidate budget is {len(items)}, expected {args.candidate_count}")
                unknown = set(items) - set(item_to_lexical)
                if unknown:
                    raise ValueError(f"{user}: unknown candidates {sorted(unknown)[:3]}")
            scores = score_candidate_batch(
                model,
                batch,
                candidates,
                item_path_ids,
                args.candidate_count,
                args.candidate_chunk_size,
            ).cpu()
            for row_index, (user, source, r2_items) in enumerate(zip(users, source_rows, candidates)):
                target = sequences[user][-2]
                v0 = unique_in_order(source["v0_top50"])
                resolver = unique_in_order(source["resolver_top50"])
                order = sorted(
                    range(args.candidate_count),
                    key=lambda index: (-float(scores[row_index, index]), index),
                )
                verifier = [r2_items[index] for index in order]
                rankings = {
                    "r2_candidate_score_only": r2_items,
                    "gram_candidate_likelihood_only": v0,
                    "r2_plus_gram_verifier": verifier,
                    "r2_portfolio_at_2": portfolio_at_2(v0, resolver, cold_items),
                }
                rows.append(
                    {
                        "user_id": user,
                        "target_item": target,
                        "is_cold": target in cold_items,
                        "metrics": {name: rank_metrics(ranking, target) for name, ranking in rankings.items()},
                        "r2_top50": r2_items,
                        "gram_mean_path_log_likelihood": [float(value) for value in scores[row_index]],
                        "verifier_top50": verifier,
                    }
                )
            if batch_index % 25 == 0:
                print(f"[verifier] users={len(rows)}/{len(dataset)}", flush=True)

    with (output_dir / "predictions_validation.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_by_slice = {}
    for slice_name in ("all", "warm", "cold"):
        selected = [
            row for row in rows
            if slice_name == "all" or (slice_name == "cold") == bool(row["is_cold"])
        ]
        summary_by_slice[slice_name] = {
            "n": len(selected),
            **{
                method: average_metric_rows([row["metrics"][method] for row in selected])
                for method in METHODS
            },
        }
    intervals = {
        "verifier_vs_v0": {
            "overall_ndcg@10": paired_bootstrap(
                rows, "r2_plus_gram_verifier", "gram_candidate_likelihood_only",
                "ndcg@10", "all", args.bootstrap_resamples, args.bootstrap_seed,
            ),
            "cold_hit@50": paired_bootstrap(
                rows, "r2_plus_gram_verifier", "gram_candidate_likelihood_only",
                "hit@50", "cold", args.bootstrap_resamples, args.bootstrap_seed + 1,
            ),
        },
        "verifier_vs_r2_score": {
            "overall_ndcg@10": paired_bootstrap(
                rows, "r2_plus_gram_verifier", "r2_candidate_score_only",
                "ndcg@10", "all", args.bootstrap_resamples, args.bootstrap_seed + 2,
            ),
            "cold_ndcg@10": paired_bootstrap(
                rows, "r2_plus_gram_verifier", "r2_candidate_score_only",
                "ndcg@10", "cold", args.bootstrap_resamples, args.bootstrap_seed + 3,
            ),
        },
    }
    inputs = [
        args.historical_config,
        args.checkpoint,
        args.item_path_file,
        args.r2_predictions,
        args.dataset_dir / "user_sequence.txt",
        args.dataset_dir / "cold_split_meta" / "cold_items.txt",
        args.dataset_dir / "cold_split_meta" / "warm_items.txt",
        args.dataset_dir / "item_plain_text.txt",
        args.dataset_dir / f"similar_item_{historical['cf_model']}.txt",
    ]
    hashes = {str(path.resolve()): sha256_file(path) for path in inputs}
    summary = {
        "experiment_id": "GRAM_PHASE14_STAGE14_0C_SAME_BACKBONE_VERIFIER",
        "status": "completed",
        "dataset": args.dataset_dir.name,
        "split": "validation",
        "test_predictions_opened": False,
        "model_training": False,
        "candidate_budget": args.candidate_count,
        "candidate_chunk_size": args.candidate_chunk_size,
        "beam_k": 50,
        "score_definition": "mean raw token log-likelihood over complete lexical path, EOS excluded",
        "acceptance_threshold": None,
        "threshold_rationale": "exhaustive frozen top-50 rerank avoids validation-label threshold tuning",
        "n_users": len(rows),
        "summary_by_slice": summary_by_slice,
        "paired_bootstrap": intervals,
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "test_read": False,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "input_file_sha256.json", hashes)
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "scope": "application-level declared opens",
            "test_files_opened": [],
            "files": [{"path": path, "mode": "read", "sha256": digest} for path, digest in hashes.items()],
        },
    )
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "split": "validation",
            "target_rule": "user_sequence[-2], evaluation only",
            "candidate_source": "frozen Phase-13 R2 top-50",
            "model_or_teacher_training": False,
            "test_predictions_opened": False,
        },
    )
    atomic_json(
        output_dir / "config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    print(json.dumps({"status": "completed", "summary": str(output_dir / "summary.json")}), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--historical-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--item-path-file", required=True, type=Path)
    parser.add_argument("--r2-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=50)
    parser.add_argument("--candidate-chunk-size", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", default="validation")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
