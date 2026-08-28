#!/usr/bin/env python3
"""Formal official-UniSRec S-AUX training on sealed Stage16 train-only data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import Adam

from official_specgr_runtime import sha256, verify_sources
from specgr_faithful import OfficialUniSRecDrafterGRAM, validate_cold_content_only


ROOT = Path(__file__).resolve().parents[3]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_set(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate item IDs in {path.relative_to(ROOT)}")
    return set(values)


def build_transitions(
    sequences: list[dict[str, Any]], item_index: dict[str, int], maximum_history: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, set[str]]:
    histories: list[list[int]] = []
    lengths: list[int] = []
    labels: list[int] = []
    label_items: set[str] = set()
    for row in sequences:
        items = row["items"]
        for position in range(1, len(items)):
            history_items = items[max(0, position - maximum_history) : position]
            target = items[position]
            histories.append([item_index[item] for item in history_items])
            lengths.append(len(history_items))
            labels.append(item_index[target])
            label_items.add(target)
    matrix = torch.zeros(len(histories), maximum_history, dtype=torch.long)
    for row_number, history in enumerate(histories):
        matrix[row_number, : len(history)] = torch.tensor(history, dtype=torch.long)
    return matrix, torch.tensor(lengths), torch.tensor(labels), label_items


def build_pseudo_events(
    events: list[dict[str, Any]], item_index: dict[str, int], full_index: dict[str, int], maximum_history: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    histories = torch.zeros(len(events), maximum_history, dtype=torch.long)
    lengths = torch.empty(len(events), dtype=torch.long)
    targets = torch.empty(len(events), dtype=torch.long)
    for row_number, event in enumerate(events):
        history_items = event["history"][-maximum_history:]
        histories[row_number, : len(history_items)] = torch.tensor(
            [item_index[item] for item in history_items], dtype=torch.long
        )
        lengths[row_number] = len(history_items)
        targets[row_number] = full_index[event["target_item"]]
    return histories, lengths, targets


@torch.no_grad()
def evaluate_pseudo_cold(
    wrapper: OfficialUniSRecDrafterGRAM,
    histories: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    train_embeddings: torch.Tensor,
    full_embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    wrapper.eval()
    model = wrapper.model
    candidate_embeddings = F.normalize(model.moe_adaptor(full_embeddings.to(device)), dim=-1)
    hit50 = 0.0
    ndcg10 = 0.0
    reciprocal_rank = 0.0
    count = 0
    candidate_order = torch.arange(candidate_embeddings.shape[0], device=device)
    for start in range(0, len(histories), batch_size):
        end = min(len(histories), start + batch_size)
        sequence = histories[start:end].to(device)
        batch_lengths = lengths[start:end].to(device)
        batch_targets = targets[start:end].to(device)
        history_content = train_embeddings.to(device)[sequence]
        history_adapted = model.moe_adaptor(history_content)
        sequence_output = F.normalize(model.forward(sequence, history_adapted, batch_lengths), dim=-1)
        scores = sequence_output @ candidate_embeddings.T
        target_scores = scores.gather(1, batch_targets[:, None])
        greater = (scores > target_scores).sum(dim=1)
        tied_before = ((scores == target_scores) & (candidate_order[None, :] < batch_targets[:, None])).sum(dim=1)
        ranks = greater + tied_before + 1
        hit50 += float((ranks <= 50).sum())
        eligible = ranks <= 10
        ndcg10 += float((eligible / torch.log2(ranks.to(torch.float32) + 1)).sum())
        reciprocal_rank += float((1.0 / ranks.to(torch.float32)).sum())
        count += end - start
    wrapper.train()
    return {
        "pseudo_cold_hit_at_50": hit50 / count,
        "pseudo_cold_ndcg_at_10": ndcg10 / count,
        "pseudo_cold_mrr": reciprocal_rank / count,
        "events": count,
        "candidate_items": int(candidate_embeddings.shape[0]),
        "all_finite": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise SystemExit("Refusing to overwrite a completed formal S-AUX run")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Formal S-AUX requires exactly one visible GPU")
    device = torch.device("cuda:0")
    started = time.time()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_paths: dict[str, Path] = {}
    for name, spec in config["inputs"].items():
        path = (ROOT / spec["path"]).resolve()
        if path.name == "user_sequence.txt" or "test" in path.name.lower():
            raise ValueError(f"Forbidden formal input path: {path.relative_to(ROOT)}")
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise ValueError(f"Missing or SHA-drifted formal input: {name}")
        input_paths[name] = path
    sources = verify_sources()

    sequences = read_jsonl(input_paths["train_sequences"])
    events = read_jsonl(input_paths["pseudo_cold_events"])
    retained_warm = read_set(input_paths["retained_warm_items"])
    pseudo_cold = read_set(input_paths["pseudo_cold_items"])
    real_cold = read_set(input_paths["real_cold_items"])
    embedding_payload = torch.load(input_paths["content_embeddings"], map_location="cpu")
    item_ids = [str(item) for item in embedding_payload["item_ids"]]
    full_embeddings_cpu = embedding_payload["embeddings"].to(torch.float32)
    full_index = {item: index for index, item in enumerate(item_ids)}
    if set(item_ids) != retained_warm | pseudo_cold | real_cold:
        raise ValueError("Formal embedding/cold/warm/pseudo catalog partition mismatch")
    ordered_train = sorted(retained_warm)
    item_index = {item: index + 1 for index, item in enumerate(ordered_train)}
    train_embeddings_cpu = torch.cat(
        [torch.zeros(1, full_embeddings_cpu.shape[1]), full_embeddings_cpu[[full_index[item] for item in ordered_train]]],
        dim=0,
    )
    histories, lengths, labels, label_items = build_transitions(
        sequences, item_index, config["training"]["maximum_history"]
    )
    pseudo_histories, pseudo_lengths, pseudo_targets = build_pseudo_events(
        events, item_index, full_index, config["training"]["maximum_history"]
    )
    if len(histories) != config["training"]["expected_train_transitions"]:
        raise ValueError("Formal transition count drift")
    if len(events) != config["training"]["expected_pseudo_cold_events"] or len(pseudo_cold) != config["training"]["expected_pseudo_cold_items"]:
        raise ValueError("Formal pseudo-cold universe drift")
    pseudo_leaks = label_items & pseudo_cold
    real_leaks = label_items & real_cold
    if pseudo_leaks or real_leaks:
        raise ValueError("Cold or pseudo-cold labels leaked into formal training")
    cold_audit = validate_cold_content_only(label_items, real_cold | pseudo_cold, item_ids)

    wrapper = OfficialUniSRecDrafterGRAM(train_embeddings_cpu).to(device).train()
    optimizer = Adam(
        wrapper.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    config_sha = sha256(config_path)
    checkpoint_before = sha256(input_paths["content_embeddings"])
    metrics_path = output / "metrics.jsonl"
    progress_path = output / "progress.json"
    best_path = output / "checkpoints/best_model.pt"
    last_path = output / "checkpoints/last_state.pt"
    total_batches_per_epoch = math.ceil(len(histories) / config["training"]["train_batch_size"])
    maximum_steps = total_batches_per_epoch * config["training"]["epochs"]
    best_metric = float("-inf")
    best_epoch = 0
    global_step = 0
    stopped_early = False

    common = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "config_sha256": config_sha,
        "test_read": False,
        "network_used": False,
    }
    atomic_json(output / "config.json", config)
    atomic_json(output / "source_manifest.json", {**common, **sources})
    atomic_json(
        output / "input_file_sha256.json",
        {**common, "files": {spec["path"]: spec["sha256"] for spec in config["inputs"].values()}},
    )
    atomic_json(
        output / "data_provenance.json",
        {
            **common,
            "training": "S16-1 Toys student-readable interaction train transitions only",
            "selection_and_admission": "S16-1 train-derived item-disjoint pseudo-cold events only",
            "candidate_universe": "complete frozen Toys catalog content embeddings",
            "cold_content_only": cold_audit,
            "validation_used": False,
            "test_read": False,
        },
    )
    atomic_json(
        output / "open_file_manifest.json",
        {**common, "opened_files": sorted(spec["path"] for spec in config["inputs"].values()), "test_read": False},
    )

    for epoch in range(1, config["training"]["epochs"] + 1):
        wrapper.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        order = torch.randperm(len(histories), generator=generator)
        epoch_loss = 0.0
        epoch_started = time.time()
        for batch_number, start in enumerate(range(0, len(order), config["training"]["train_batch_size"]), 1):
            selected = order[start : start + config["training"]["train_batch_size"]]
            batch_histories = histories[selected].to(device)
            batch_lengths = lengths[selected].to(device)
            batch_labels = labels[selected].to(device)
            optimizer.zero_grad()
            loss = wrapper.calculate_loss(batch_histories, batch_lengths, batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite formal S-AUX loss at epoch {epoch}, batch {batch_number}")
            loss.backward()
            optimizer.step()
            global_step += 1
            epoch_loss += float(loss.detach())
            atomic_json(
                progress_path,
                {
                    **common,
                    "stage": "training",
                    "epoch": epoch,
                    "maximum_epochs": config["training"]["epochs"],
                    "batch": batch_number,
                    "batches_per_epoch": total_batches_per_epoch,
                    "global_step": global_step,
                    "maximum_steps": maximum_steps,
                    "last_loss": float(loss.detach()),
                    "best_epoch": best_epoch,
                    "best_metric": None if best_metric == float("-inf") else best_metric,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        train_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": epoch_loss / total_batches_per_epoch,
            "train_seconds": time.time() - epoch_started,
            "global_step": global_step,
        }
        if epoch % config["training"]["evaluation_interval_epochs"] == 0:
            eval_started = time.time()
            metrics = evaluate_pseudo_cold(
                wrapper,
                pseudo_histories,
                pseudo_lengths,
                pseudo_targets,
                train_embeddings_cpu,
                full_embeddings_cpu,
                config["training"]["evaluation_batch_size"],
                device,
            )
            train_record.update(metrics)
            train_record["evaluation_seconds"] = time.time() - eval_started
            selection_metric = float(metrics[config["training"]["selection_metric"]])
            if selection_metric > best_metric:
                best_metric = selection_metric
                best_epoch = epoch
                atomic_torch(
                    best_path,
                    {
                        "model": wrapper.state_dict(),
                        "epoch": epoch,
                        "selection_metric": config["training"]["selection_metric"],
                        "selection_value": selection_metric,
                        "config_sha256": config_sha,
                    },
                )
            if epoch - best_epoch >= config["training"]["early_stopping_patience_epochs"]:
                stopped_early = True
        append_jsonl(metrics_path, train_record)
        atomic_torch(
            last_path,
            {
                "model": wrapper.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "config_sha256": config_sha,
            },
        )
        if stopped_early:
            break

    if not best_path.is_file():
        raise RuntimeError("Formal S-AUX produced no internal-dev best checkpoint")
    best_state = torch.load(best_path, map_location=device)
    wrapper.load_state_dict(best_state["model"], strict=True)
    final_admission = evaluate_pseudo_cold(
        wrapper,
        pseudo_histories,
        pseudo_lengths,
        pseudo_targets,
        train_embeddings_cpu,
        full_embeddings_cpu,
        config["training"]["evaluation_batch_size"],
        device,
    )
    checkpoint_after = sha256(input_paths["content_embeddings"])
    verdict = (
        "PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION"
        if final_admission["all_finite"] and checkpoint_before == checkpoint_after
        else "FAIL_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION"
    )
    summary = {
        **common,
        "status": "completed",
        "verdict": verdict,
        "official_runtime": "PINNED_SPECGR_UNISREC_PLUS_RECBOLE_V1_2_0",
        "epochs_completed": epoch,
        "stopped_early": stopped_early,
        "global_steps": global_step,
        "best_epoch": best_epoch,
        "best_selection_metric": best_metric,
        "final_fixed_scale_admission": final_admission,
        "train_transitions": len(histories),
        "pseudo_cold_events": len(events),
        "pseudo_cold_items": len(pseudo_cold),
        "real_cold_items": len(real_cold),
        "cold_interaction_label_leaks": 0,
        "content_embedding_sha256_before": checkpoint_before,
        "content_embedding_sha256_after": checkpoint_after,
        "content_embedding_unchanged": checkpoint_before == checkpoint_after,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "runtime_seconds": time.time() - started,
        "scientific_scope": "train-only internal-development contract/admission; no source validation or test",
        "test_read": False,
    }
    atomic_json(output / "summary.json", summary)
    print(verdict, flush=True)
    return 0 if verdict.startswith("PASS") else 3


if __name__ == "__main__":
    raise SystemExit(main())
