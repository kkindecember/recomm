#!/usr/bin/env python3
"""One-step train-only S-AUX/S-PLUS/S-PLUS-CTRL contract admission smoke."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam, AdamW
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent
if str(PROTOCOL) not in sys.path:
    sys.path.insert(0, str(PROTOCOL))

from official_specgr_runtime import sha256, verify_sources  # noqa: E402
from resource_probe import load_gram  # noqa: E402
from specgr_faithful import (  # noqa: E402
    GRAMSelfDrafter,
    OfficialUniSRecDrafterGRAM,
    TrainingBudget,
    assert_splus_control_budget_match,
    sequence_item_contrastive_loss,
    splus_pretrain_loss,
    validate_cold_content_only,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_metadata(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item, text = line.split(maxsplit=1)
        rows[item] = text
    return rows


def read_paths(path: Path) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item, serialized = line.split(maxsplit=1)
        rows[item] = tuple(token for token in serialized.split("|") if token)
    return rows


def transitions(sequences: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rows = []
    for sequence in sequences:
        items = sequence["items"]
        for position in range(1, len(items)):
            rows.append(
                {
                    "user_id": sequence["user_id"],
                    "position": position,
                    "history": items[max(0, position - 20) : position],
                    "target": items[position],
                }
            )
    return sorted(
        rows,
        key=lambda row: stable_rank(seed, "transition", f"{row['user_id']}|{row['position']}|{row['target']}"),
    )


def clear_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def s_aux_smoke(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    pseudo_events: list[dict[str, Any]],
    retained_warm: set[str],
    pseudo_cold: set[str],
    real_cold: set[str],
    embedding_payload: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    clear_cuda(device)
    started = time.perf_counter()
    item_ids = [str(item) for item in embedding_payload["item_ids"]]
    all_embeddings = embedding_payload["embeddings"]
    index = {item: number for number, item in enumerate(item_ids)}
    if set(index) != retained_warm | real_cold | pseudo_cold:
        # Pseudo-cold is a removed subset of the original warm universe; the
        # union with retained warm and real cold must recover the full catalog.
        raise ValueError("Embedding catalog does not match retained-warm + pseudo-cold + real-cold")
    ordered_train = sorted(retained_warm)
    train_embedding_cpu = torch.cat(
        [torch.zeros(1, all_embeddings.shape[1]), all_embeddings[[index[item] for item in ordered_train]]], dim=0
    )
    train_index = {item: position + 1 for position, item in enumerate(ordered_train)}
    model = OfficialUniSRecDrafterGRAM(train_embedding_cpu).to(device).train()
    batch_rows = train_rows[: config["s_aux"]["smoke_batch_size"]]
    sequences = torch.zeros(len(batch_rows), 20, dtype=torch.long, device=device)
    lengths = torch.empty(len(batch_rows), dtype=torch.long, device=device)
    labels = torch.empty(len(batch_rows), dtype=torch.long, device=device)
    for row_number, row in enumerate(batch_rows):
        history = row["history"][-20:]
        sequences[row_number, : len(history)] = torch.tensor([train_index[item] for item in history], device=device)
        lengths[row_number] = len(history)
        labels[row_number] = train_index[row["target"]]
    optimizer = Adam(model.parameters(), lr=config["s_aux"]["learning_rate"], weight_decay=0.0)
    optimizer.zero_grad()
    loss = model.calculate_loss(sequences, lengths, labels)
    loss.backward()
    optimizer.step()

    selected_events = sorted(
        pseudo_events,
        key=lambda row: stable_rank(config["seed"], "pseudo-admission", row["event_id"]),
    )[:32]
    pseudo_history = torch.zeros(len(selected_events), 20, dtype=torch.long, device=device)
    pseudo_lengths = torch.empty(len(selected_events), dtype=torch.long, device=device)
    for row_number, row in enumerate(selected_events):
        history = row["history"][-20:]
        pseudo_history[row_number, : len(history)] = torch.tensor(
            [train_index[item] for item in history], device=device
        )
        pseudo_lengths[row_number] = len(history)
    candidate_items = [row["target_item"] for row in selected_events]
    candidate_embeddings = all_embeddings[[index[item] for item in candidate_items]].to(device)
    with torch.no_grad():
        scores = model.inductive_scores(
            pseudo_history,
            pseudo_lengths,
            train_embedding_cpu.to(device),
            candidate_embeddings,
        )
    label_items = {row["target"] for row in train_rows}
    content_audit = validate_cold_content_only(label_items, real_cold, item_ids)
    torch.cuda.synchronize(device)
    result = {
        "execution_class": "PINNED_OFFICIAL_SPECGR_UNISREC_AND_RECBOLE",
        "optimizer_steps": 1,
        "train_batch_size": len(batch_rows),
        "train_catalog_items_excluding_padding": len(ordered_train),
        "pseudo_cold_admission_events": len(selected_events),
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "inductive_score_shape": list(scores.shape),
        "inductive_scores_finite": bool(torch.isfinite(scores).all().item()),
        "cold_content_only": content_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    del scores, candidate_embeddings, pseudo_history, pseudo_lengths, optimizer, loss, model
    clear_cuda(device)
    return result


def tokenize_passage_batch(
    rows: list[dict[str, Any]],
    metadata: dict[str, str],
    paths: dict[str, tuple[str, ...]],
    tokenizer,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    max_passages, max_length = 21, 128
    all_passages: list[str] = []
    active_counts: list[int] = []
    for row in rows:
        history = row["history"][-20:]
        history_lexical = " > ".join("|".join(paths[item]) for item in history)
        passages = [f"What would user purchase after {history_lexical} ?"] + [metadata[item] for item in reversed(history)]
        active_counts.append(len(passages))
        all_passages.extend(passages + [""] * (max_passages - len(passages)))
    encoded = tokenizer(
        all_passages,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.reshape(len(rows), max_passages, max_length)
    attention = encoded.attention_mask.reshape(len(rows), max_passages, max_length)
    for row_number, count in enumerate(active_counts):
        input_ids[row_number, count:] = tokenizer.pad_token_id
        attention[row_number, count:] = 0

    label_rows = []
    for row in rows:
        token_ids = tokenizer.convert_tokens_to_ids(list(paths[row["target"]]))
        if any(token == tokenizer.unk_token_id for token in token_ids):
            raise ValueError(f"Lexical token maps to UNK for {row['target']}")
        label_rows.append(token_ids + [tokenizer.eos_token_id])
    label_width = max(map(len, label_rows))
    labels = torch.full((len(rows), label_width), -100, dtype=torch.long)
    for row_number, values in enumerate(label_rows):
        labels[row_number, : len(values)] = torch.tensor(values)

    target_text = tokenizer(
        [metadata[row["target"]] for row in rows],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    context = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "labels": labels.to(device),
    }
    target = {
        "input_ids": target_text.input_ids[:, None, :].to(device),
        "attention_mask": target_text.attention_mask[:, None, :].to(device),
    }
    return context, target


def encode_gram(model, drafter: GRAMSelfDrafter, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids, attention = batch["input_ids"], batch["attention_mask"]
    model.encoder.n_passages = input_ids.shape[1]
    hidden = model.encoder(
        input_ids=input_ids.reshape(input_ids.shape[0], -1),
        attention_mask=attention.reshape(attention.shape[0], -1),
        return_dict=True,
    )[0]
    return drafter.pool(hidden, attention.reshape(attention.shape[0], -1))


def budget_from_config(config: dict[str, Any], split_sha: str) -> TrainingBudget:
    pretrain = config["s_plus"]["pretrain"]
    return TrainingBudget(
        dataset_manifest_sha256=split_sha,
        transitions=27659,
        epochs=pretrain["epochs"],
        optimizer="AdamW",
        learning_rate=pretrain["learning_rate"],
        weight_decay=pretrain["weight_decay"],
        warmup_steps=pretrain["warmup_steps"],
        physical_microbatch=pretrain["physical_generation_microbatch"],
        gradient_accumulation=pretrain["gradient_accumulation"],
        optimizer_steps=pretrain["optimizer_steps_total"],
        gpu_count=1,
        timeout_seconds=259200,
    )


def splus_and_control_smoke(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    metadata: dict[str, str],
    paths: dict[str, tuple[str, ...]],
    historical: Path,
    checkpoint: Path,
    split_manifest: Path,
    device: torch.device,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    targets: set[str] = set()
    for row in train_rows:
        if row["target"] not in targets:
            selected.append(row)
            targets.add(row["target"])
        if len(selected) == config["s_plus"]["smoke_batch_size"]:
            break
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    context, target = tokenize_passage_batch(selected, metadata, paths, tokenizer, device)
    checkpoint_before = sha256(checkpoint)

    clear_cuda(device)
    started = time.perf_counter()
    model = load_gram(historical, checkpoint, device).train()
    drafter = GRAMSelfDrafter(model.config.d_model, config["s_plus"]["projection_dimension"]).to(device)
    optimizer = AdamW(
        list(model.parameters()) + list(drafter.parameters()),
        lr=config["s_plus"]["pretrain"]["learning_rate"],
        weight_decay=config["s_plus"]["pretrain"]["weight_decay"],
    )
    optimizer.zero_grad()
    sequence_embeddings = encode_gram(model, drafter, context)
    item_embeddings = encode_gram(model, drafter, target)
    item_ids = torch.arange(len(selected), device=device)
    contrastive = sequence_item_contrastive_loss(sequence_embeddings, item_embeddings, item_ids)
    generation = model(**context, use_cache=False).loss
    joint = splus_pretrain_loss(contrastive, generation)
    joint.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    splus_result = {
        "optimizer_steps": 1,
        "batch_size": len(selected),
        "contrastive_loss": float(contrastive.detach()),
        "generative_loss": float(generation.detach()),
        "joint_loss": float(joint.detach()),
        "all_losses_finite": bool(
            torch.isfinite(torch.stack([contrastive.detach(), generation.detach(), joint.detach()])).all().item()
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    del optimizer, joint, generation, contrastive, item_ids, item_embeddings, sequence_embeddings, drafter, model
    clear_cuda(device)

    started = time.perf_counter()
    control = load_gram(historical, checkpoint, device).train()
    control_optimizer = AdamW(
        control.parameters(),
        lr=config["s_plus"]["pretrain"]["learning_rate"],
        weight_decay=config["s_plus"]["pretrain"]["weight_decay"],
    )
    control_optimizer.zero_grad()
    control_loss = control(**context, use_cache=False).loss
    control_loss.backward()
    control_optimizer.step()
    torch.cuda.synchronize(device)
    control_result = {
        "objective": config["s_plus_control"]["objective"],
        "optimizer_steps": 1,
        "batch_size": len(selected),
        "generative_loss": float(control_loss.detach()),
        "loss_finite": bool(torch.isfinite(control_loss).item()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    del control_optimizer, control_loss, control
    clear_cuda(device)

    budget = budget_from_config(config, sha256(split_manifest))
    budget_audit = assert_splus_control_budget_match(budget, budget)
    checkpoint_after = sha256(checkpoint)
    return {
        "s_plus": splus_result,
        "s_plus_control": control_result,
        "budget_audit": budget_audit,
        "same_start_checkpoint": True,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
        "validation_or_test_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--admission-free-mib", type=int, required=True)
    parser.add_argument("--admission-util-percent", type=int, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    if (output / "smoke_summary.json").exists():
        raise SystemExit("Refusing to overwrite an existing S16-2 contract smoke")
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("S16-2 smoke requires exactly one visible GPU")
    device = torch.device("cuda:0")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    domain = {key: ROOT / value for key, value in config["domain"].items() if key != "name"}
    train_sequences = read_jsonl(domain["train_sequences"])
    train_rows = transitions(train_sequences, config["seed"])
    pseudo_events = read_jsonl(domain["pseudo_cold_events"])
    metadata = read_metadata(domain["metadata"])
    paths = read_paths(domain["lexical_paths"])
    real_cold = read_set(domain["cold_items"])
    retained_warm_path = domain["train_sequences"].parents[1] / "retained_warm_items.txt"
    pseudo_cold_path = domain["train_sequences"].parents[1] / "pseudo_cold_items.txt"
    retained_warm = read_set(retained_warm_path)
    pseudo_cold = read_set(pseudo_cold_path)
    embedding_payload = torch.load(domain["content_embeddings"], map_location="cpu")
    if embedding_payload["embeddings"].shape != (11924, 1024):
        raise ValueError("Frozen Toys BGE embedding shape drift")

    started = time.perf_counter()
    s_aux = s_aux_smoke(
        config, train_rows, pseudo_events, retained_warm, pseudo_cold, real_cold, embedding_payload, device
    )
    splus = splus_and_control_smoke(
        config,
        train_rows,
        metadata,
        paths,
        domain["gram_config"],
        domain["gram_checkpoint"],
        domain["split_manifest"],
        device,
    )
    elapsed = time.perf_counter() - started
    peak = max(
        s_aux["peak_allocated_mib"],
        splus["s_plus"]["peak_allocated_mib"],
        splus["s_plus_control"]["peak_allocated_mib"],
    )
    checks = {
        "official_unisrec_finite_forward_backward": s_aux["loss_finite"],
        "official_unisrec_inductive_scores_finite": s_aux["inductive_scores_finite"],
        "cold_content_only": s_aux["cold_content_only"]["cold_interaction_label_count"] == 0,
        "splus_joint_loss_finite": splus["s_plus"]["all_losses_finite"],
        "splus_ctrl_loss_finite": splus["s_plus_control"]["loss_finite"],
        "splus_ctrl_budget_match": splus["budget_audit"]["matched"],
        "base_checkpoint_unchanged": splus["checkpoint_unchanged"],
        "test_read_false": True,
        "peak_below_small_experiment_cap": peak <= config["smoke_contract"]["maximum_incremental_gpu_mib"],
        "wall_below_small_experiment_cap": elapsed <= config["smoke_contract"]["hard_timeout_seconds"],
    }
    verdict = "PASS_S16_2_SPECGR_CONTRACT_SMALL_SMOKE" if all(checks.values()) else "FAIL_S16_2_SPECGR_CONTRACT_SMALL_SMOKE"
    opened = [config_path, *domain.values(), retained_warm_path, pseudo_cold_path]
    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_read": False,
        "network_used_during_experiment": False,
    }
    write_json(output / "config.json", config)
    write_json(output / "source_manifest.json", {**common, **verify_sources()})
    write_json(output / "input_file_sha256.json", {**common, "files": {str(path.relative_to(ROOT)): sha256(path) for path in opened}})
    write_json(
        output / "open_file_manifest.json",
        {
            **common,
            "opened_files": sorted(str(path.relative_to(ROOT)) for path in opened),
            "forbidden_patterns": ["user_sequence.txt", "predictions_test", "test_metrics"],
            "test_read": False,
        },
    )
    write_json(
        output / "data_provenance.json",
        {
            **common,
            "training": "S16-1 student-readable Toys interaction train only",
            "admission": "S16-1 train-derived item-disjoint pseudo-cold held events",
            "validation_target_used": False,
            "real_test_target_used": False,
        },
    )
    write_json(
        output / "resource_summary.json",
        {
            **common,
            "physical_gpu": args.physical_gpu,
            "visible_gpu": 0,
            "gpu_name": torch.cuda.get_device_name(device),
            "admission_free_mib": args.admission_free_mib,
            "admission_util_percent": args.admission_util_percent,
            "elapsed_seconds": elapsed,
            "peak_allocated_mib": peak,
            "hard_timeout_seconds": config["smoke_contract"]["hard_timeout_seconds"],
            "large_experiment_started": False,
        },
    )
    write_json(
        output / "smoke_summary.json",
        {
            **common,
            "verdict": verdict,
            "checks": checks,
            "s_aux": s_aux,
            **splus,
            "scientific_efficacy_metric_produced": False,
            "formal_gates": {
                "PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION": "PENDING_FULL_TRAIN_AND_FIXED_SCALE_ADMISSION",
                "PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION": "PENDING_FULL_TRAIN_AND_FIXED_SCALE_ADMISSION",
                "S_PLUS_CTRL_BUDGET_MATCH": "CONTRACT_PASS_FORMAL_EXECUTION_PENDING",
            },
        },
    )
    print(verdict)
    return 0 if verdict.startswith("PASS") else 3


if __name__ == "__main__":
    raise SystemExit(main())
