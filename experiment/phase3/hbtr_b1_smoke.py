#!/usr/bin/env python3
"""HBTR-B1 correctness-only GPU smoke.

The script uses only training targets (sequence[-3]) and training histories
(sequence[:-3]). It mines a static cache from the locked baseline, performs a
small number of C4 optimization steps, verifies checkpoint reload, and checks
the constrained beam-50 path. Smoke weights are created in a temporary file
and are not retained for pilot use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = ROOT / "GRAM/src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import create_model  # noqa: E402
from processor import CollatorGRAM  # noqa: E402
from utils import generation_trie as gt  # noqa: E402
from utils import indexing  # noqa: E402

from hbtr_b1_objective import (  # noqa: E402
    NEGATIVE_COUNT,
    canonical_cache_sha256,
    common_prefix_depth,
    joint_margin,
    pairwise_ranking_loss,
    sequence_log_scores,
    total_loss,
    validate_cache_row,
)


DATASETS = {
    "Toys": {
        "checkpoint": ROOT / "GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt",
        "hierarchical_id_type": "hierarchy_v1_c32_l5_len32768_split",
        "top_k_similar_item": 5,
    },
    "Beauty": {
        "checkpoint": ROOT / "GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt",
        "hierarchical_id_type": "hierarchy_v1_c128_l7_len32768_split",
        "top_k_similar_item": 10,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sequences(path: Path) -> dict[str, list[str]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            user, *items = line.split()
            result[user] = items
    return result


def read_semantic_tokens(path: Path) -> dict[str, tuple[str, ...]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            item, raw = line.rstrip("\n").split(" ", 1)
            result[item] = tuple(token for token in raw.split("|") if token)
    return result


def normalized_sequence(ids: list[int]) -> tuple[int, ...]:
    values = list(ids)
    if 1 in values:
        values = values[: values.index(1) + 1]
    return tuple(values)


def encode_candidates(tokenizer, item2lexid: dict[str, str]):
    encoded = []
    sequence_to_item = {}
    for item, candidate in item2lexid.items():
        tokens = [tok for tok in tokenizer.encode(candidate) if tok not in (1820, 9175)]
        sequence = (0, *tokens)
        if sequence in sequence_to_item:
            raise ValueError(f"duplicate encoded semantic ID for {item}")
        sequence_to_item[sequence] = item
        encoded.append(list(sequence))
    return encoded, sequence_to_item


def make_runtime_args(dataset: str) -> SimpleNamespace:
    spec = DATASETS[dataset]
    return SimpleNamespace(
        data_path=str(ROOT / "GRAM/rec_datasets"),
        datasets=dataset,
        rank=0,
        item_id_path="",
        hierarchical_id_type=spec["hierarchical_id_type"],
        item_prompt="all_text",
        top_k_similar_item=spec["top_k_similar_item"],
        cf_model="sasrec",
        id_linking=1,
        item_prompt_max_len=128,
        target_max_len=32,
        max_his=20,
        item_id_type="split",
    )


def create_model_and_tokenizer(dataset: str, device: torch.device):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config

    runtime = make_runtime_args(dataset)
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    config = T5Config.from_pretrained("t5-small", local_files_only=True)
    config.max_seq_len = runtime.item_prompt_max_len
    config.max_item_num = runtime.max_his
    config.use_position_embedding = 1
    config.sample_num = "1"
    backbone = AutoModelForSeq2SeqLM.from_pretrained(
        "t5-small", config=config, local_files_only=True
    )
    model = create_model("gram", config=config)
    model.load_t5(backbone.state_dict())
    del backbone
    checkpoint = DATASETS[dataset]["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"checkpoint mismatch missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    return model, tokenizer, runtime


def build_training_samples(
    sequences: dict[str, list[str]],
    item2input: dict[str, str],
    item2lexid: dict[str, str],
    max_samples: int,
) -> list[dict]:
    samples = []
    for user in sorted(sequences, key=lambda value: hashlib.sha256(value.encode()).hexdigest()):
        items = sequences[user]
        if len(items) < 4:
            continue
        target = items[-3]
        history = items[:-3][-20:]
        if not history or target not in item2lexid or any(item not in item2input for item in history):
            continue
        history_for_model = list(reversed(history))
        history_lex = " ; ".join(item2lexid[item] for item in history_for_model)
        samples.append(
            {
                "sample_key": f"{user}:{target}:{len(history)}",
                "user_id": user,
                "positive_item": target,
                "history_items": history,
                "input": [f"What would user purchase after {history_lex} ?"]
                + [item2input[item] for item in history_for_model],
                "output": item2lexid[target],
            }
        )
        if len(samples) >= max_samples:
            break
    if len(samples) < max_samples:
        raise ValueError(f"only {len(samples)} eligible training samples were available")
    return samples


def collate_one(collator: CollatorGRAM, sample: dict, output: str | None = None):
    return collator(
        [
            {
                "input": sample["input"],
                "output": output if output is not None else sample["output"],
                "user_id": sample["user_id"],
            }
        ]
    )


@torch.no_grad()
def mine_cache(
    model,
    tokenizer,
    collator,
    samples,
    item2lexid,
    semantic_tokens,
    popularity,
    sequence_to_item,
    encoded_candidates,
    device,
):
    model.eval()
    trie = gt.Trie(encoded_candidates)
    prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(trie)
    max_length = max(len(candidate) for candidate in encoded_candidates)
    valid_items = set(item2lexid)
    rows = []
    generation_audit = []
    for sample in samples:
        batch = collate_one(collator, sample)
        prediction = model.generate(
            input_ids=batch["item_text_ids"].to(device),
            attention_mask=batch["item_text_masks"].to(device),
            max_length=max_length,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=50,
            num_return_sequences=50,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,
        )
        predicted_items = [
            sequence_to_item.get(normalized_sequence(ids.tolist()))
            for ids in prediction["sequences"]
        ]
        if any(item is None for item in predicted_items):
            raise ValueError("constrained beam produced an item outside the locked Trie")
        if len(set(predicted_items)) != len(predicted_items):
            raise ValueError("constrained beam produced duplicate items")
        positive = sample["positive_item"]
        rank = predicted_items.index(positive) + 1 if positive in predicted_items else None
        generation_audit.append({"sample_key": sample["sample_key"], "positive_rank": rank})
        if rank is None or not 11 <= rank <= 50:
            continue
        history = set(sample["history_items"])
        negatives = []
        for item in predicted_items[:10]:
            if item == positive or item in history or item in negatives:
                continue
            negatives.append(item)
            if len(negatives) == NEGATIVE_COUNT:
                break
        if len(negatives) != NEGATIVE_COUNT:
            continue
        row = {
            "sample_key": sample["sample_key"],
            "user_id": sample["user_id"],
            "positive_item": positive,
            "positive_rank": rank,
            "history_items": sample["history_items"],
            "negative_items": negatives,
            "prefix_depths": [
                common_prefix_depth(semantic_tokens[positive], semantic_tokens[item])
                for item in negatives
            ],
            "positive_frequency": int(popularity[positive]),
        }
        validate_cache_row(row, valid_items=valid_items)
        rows.append(row)
    return rows, generation_audit


def run_training_steps(
    model,
    collator,
    samples_by_key,
    cache_rows,
    item2lexid,
    median_frequency,
    device,
    max_steps,
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    records = []
    # Correctness smoke must exercise non-trivial weighting branches rather
    # than accidentally taking the first cache rows with wp=wt=1.
    joint_rows = [
        row
        for row in cache_rows
        if max(row["prefix_depths"]) > 0
        and row["positive_frequency"] < median_frequency
    ]
    prefix_rows = [row for row in cache_rows if max(row["prefix_depths"]) > 0]
    tail_rows = [
        row for row in cache_rows if row["positive_frequency"] < median_frequency
    ]
    selected = []
    for pool in (joint_rows, prefix_rows, tail_rows, cache_rows):
        for row in pool:
            if row not in selected:
                selected.append(row)
                break
        if len(selected) >= max_steps:
            break
    if len(selected) < max_steps:
        for row in cache_rows:
            if row not in selected:
                selected.append(row)
            if len(selected) >= max_steps:
                break
    for row in selected[:max_steps]:
        sample = samples_by_key[row["sample_key"]]
        positive_batch = collate_one(collator, sample)
        positive_labels = positive_batch["target_ids"].to(device)
        positive_output = model(
            input_ids=positive_batch["item_text_ids"].to(device),
            attention_mask=positive_batch["item_text_masks"].to(device),
            labels=positive_labels,
            return_dict=True,
        )
        negative_batch = collator(
            [
                {
                    "input": sample["input"],
                    "output": item2lexid[item],
                    "user_id": sample["user_id"],
                }
                for item in row["negative_items"]
            ]
        )
        negative_labels = negative_batch["target_ids"].to(device)
        negative_output = model(
            input_ids=negative_batch["item_text_ids"].to(device),
            attention_mask=negative_batch["item_text_masks"].to(device),
            labels=negative_labels,
            return_dict=True,
        )
        positive_score = sequence_log_scores(positive_output.logits, positive_labels)
        negative_score = sequence_log_scores(negative_output.logits, negative_labels)[None, :]
        margins = torch.tensor(
            [
                joint_margin(depth, row["positive_frequency"], median_frequency)
                for depth in row["prefix_depths"]
            ],
            device=device,
            dtype=positive_score.dtype,
        )[None, :]
        ranking = pairwise_ranking_loss(positive_score, negative_score, margins)
        loss = total_loss(positive_output.loss, ranking)
        fallback = total_loss(positive_output.loss, ranking, ranking_lambda=0.0)
        if not torch.equal(fallback, positive_output.loss):
            raise ValueError("lambda=0 failed exact loss fallback")
        if not torch.isfinite(loss):
            raise ValueError("non-finite HBTR smoke loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError("HBTR smoke produced invalid/zero gradient")
        optimizer.step()
        records.append(
            {
                "sample_key": row["sample_key"],
                "token_ce": float(positive_output.loss.detach().cpu()),
                "ranking_loss": float(ranking.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "max_prefix_depth": max(row["prefix_depths"]),
                "positive_frequency": row["positive_frequency"],
                "exercises_prefix_weight": max(row["prefix_depths"]) > 0,
                "exercises_tail_weight": row["positive_frequency"] < median_frequency,
                "margins": [float(value) for value in margins.detach().cpu().flatten()],
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_samples <= 100:
        raise ValueError("B1 preregistration permits 1 through 100 smoke samples")
    if not 1 <= args.max_train_steps <= 2:
        raise ValueError("B1 smoke permits at most two optimizer steps")

    started = time.time()
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("HBTR-B1 smoke requires CUDA")

    spec = DATASETS[args.dataset]
    checkpoint = spec["checkpoint"]
    dataset_dir = ROOT / "GRAM/rec_datasets" / args.dataset
    sequence_path = dataset_dir / "user_sequence.txt"
    index_path = dataset_dir / f"item_generative_indexing_{spec['hierarchical_id_type']}.txt"
    for path in (checkpoint, sequence_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    model, tokenizer, runtime = create_model_and_tokenizer(args.dataset, device)
    sequences = read_sequences(sequence_path)
    _, item2input, item2lexid = indexing.gram_indexing(
        data_path=runtime.data_path,
        dataset=args.dataset,
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        args=runtime,
        user_id_without_target_item=True,
        id_linking=True,
    )
    semantic_tokens = read_semantic_tokens(index_path)
    popularity = Counter()
    for items in sequences.values():
        popularity.update(items[:-2])
    median_frequency = float(statistics.median(popularity.values()))
    samples = build_training_samples(
        sequences, item2input, item2lexid, args.max_samples
    )
    samples_by_key = {sample["sample_key"]: sample for sample in samples}
    collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
    encoded_candidates, sequence_to_item = encode_candidates(tokenizer, item2lexid)

    cache_rows, generation_audit = mine_cache(
        model,
        tokenizer,
        collator,
        samples,
        item2lexid,
        semantic_tokens,
        popularity,
        sequence_to_item,
        encoded_candidates,
        device,
    )
    if not cache_rows:
        raise RuntimeError("no miss@10/hit@50 training pair found; no retry attempted")

    records = run_training_steps(
        model,
        collator,
        samples_by_key,
        cache_rows,
        item2lexid,
        median_frequency,
        device,
        args.max_train_steps,
    )
    if not records:
        raise RuntimeError("no HBTR optimizer step was executed")

    model.eval()
    verification_sample = samples[-1]
    verification_batch = collate_one(collator, verification_sample)
    with torch.no_grad():
        before_logits = model(
            input_ids=verification_batch["item_text_ids"].to(device),
            attention_mask=verification_batch["item_text_masks"].to(device),
            labels=verification_batch["target_ids"].to(device),
            return_dict=True,
        ).logits.detach().cpu()
    with tempfile.TemporaryDirectory(prefix=f"hbtr_b1_{args.dataset.lower()}_") as tmpdir:
        temporary_checkpoint = Path(tmpdir) / "discard_only_checkpoint.pt"
        torch.save(model.state_dict(), temporary_checkpoint)
        checkpoint_sha = sha256(temporary_checkpoint)
        reloaded, _, _ = create_model_and_tokenizer(args.dataset, device)
        reloaded.load_state_dict(torch.load(temporary_checkpoint, map_location="cpu"))
        reloaded.eval()
        with torch.no_grad():
            after_logits = reloaded(
                input_ids=verification_batch["item_text_ids"].to(device),
                attention_mask=verification_batch["item_text_masks"].to(device),
                labels=verification_batch["target_ids"].to(device),
                return_dict=True,
            ).logits.detach().cpu()
        reload_max_abs_diff = float((before_logits - after_logits).abs().max())
        if reload_max_abs_diff != 0.0:
            raise ValueError(f"checkpoint reload mismatch: {reload_max_abs_diff}")
        del reloaded

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "dataset": args.dataset,
        "source_checkpoint": str(checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(checkpoint),
        "refresh": "static_once_before_training",
        "test_data_read": False,
        "rows": cache_rows,
        "rows_sha256": canonical_cache_sha256(cache_rows),
    }
    with (args.output_dir / "negative_cache.json").open("w") as handle:
        json.dump(cache_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "hbtr_b1_smoke_v1",
            "design_status": "CORRECTNESS_ONLY_WEIGHTS_DISCARDED",
        },
        "dataset": args.dataset,
        "status": "PASS",
        "samples_mined": len(samples),
        "valid_cache_rows": len(cache_rows),
        "optimizer_steps": records,
        "median_training_frequency": median_frequency,
        "generation_audit": generation_audit,
        "cache_sha256": cache_payload["rows_sha256"],
        "temporary_checkpoint_sha256": checkpoint_sha,
        "temporary_checkpoint_retained": False,
        "checkpoint_reload_max_abs_diff": reload_max_abs_diff,
        "test_data_read": False,
        "pilot_split_created": False,
        "effect_claim_allowed": False,
        "wall_time_seconds": time.time() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
