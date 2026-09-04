#!/usr/bin/env python3
"""Run the frozen Stage18 S18-1 fold-local actionability diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config
from transformers.modeling_outputs import BaseModelOutput


ROOT = Path(__file__).resolve().parents[3]
GRAM_SRC = ROOT / "GRAM/src"
PHASE9 = ROOT / "experiment/phase9"
for directory in (ROOT, GRAM_SRC, PHASE9):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import utils  # noqa: E402,F401
from arguments import create_parser  # noqa: E402
from data import MultiTaskDatasetGRAM, TestDatasetGRAM  # noqa: E402
from model import create_model  # noqa: E402
from processor import CollatorGRAM  # noqa: E402
from train_cf0_b2_item_head import (  # noqa: E402
    CF0B2ItemHead,
    SequenceDataset,
    collate_sequences,
    scheduler_lambda,
)
from utils import generation_trie as gt  # noqa: E402

from experiment.phase18.core.contracts import load_json, sha256  # noqa: E402
from experiment.phase18.core.s1_contracts import (  # noqa: E402
    actual_pruner_items,
    catalog_standardized_target,
    evaluate_domain_gate,
    first_drop_depth,
    hard_negative_recall,
)
from experiment.phase17.core.run_manager import launch_background_tmux  # noqa: E402


PYTHON = Path("/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python")
CONFIG_PATH = ROOT / "experiment/phase18/config/s18_s1_actionability.json"
AUTH_PATH = ROOT / "experiment/phase18/config/s18_s1_resource_authorization.json"
PREFLIGHT = ROOT / "artifacts/phase18/s1_actionability/preflight"
OUTPUT = ROOT / "artifacts/phase18/s1_actionability/run-0001"
STATUS = ROOT / "artifacts/phase18/status/s18_s1_actionability.status.json"
LEDGER = ROOT / "artifacts/phase18/attempts/S18-1.attempts.jsonl"
SMOKE = ROOT / "artifacts/phase18/s1_actionability/smoke"
REPORT = ROOT / "report/第十八阶段/Stage18_S1_可作用性与FirstDrop诊断报告.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    """Convert NumPy values at artifact boundaries without changing computation."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=json_default,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def unit_key(domain: str, fold: str) -> str:
    return f"{domain.lower()}_{fold.lower().replace('-', 'm')}"


def unit_dir(domain: str, fold: str) -> Path:
    return OUTPUT / "units" / unit_key(domain, fold)


def unit_status_path(domain: str, fold: str) -> Path:
    return unit_dir(domain, fold) / "status.json"


def load_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    auth = load_json(AUTH_PATH)
    if sha256(CONFIG_PATH) != auth["scientific_config"]["sha256"]:
        raise RuntimeError("S18-1 scientific config hash mismatch")
    if sha256(PREFLIGHT / "manifest.json") != auth["preflight_manifest"]["sha256"]:
        raise RuntimeError("S18-1 preflight manifest hash mismatch")
    if config["authorization_boundary"]["automatic_retry"]:
        raise RuntimeError("automatic retry must remain disabled")
    return config, auth


def dataset_name_from_manifest(domain: str, fold: str) -> str:
    manifest = load_json(PREFLIGHT / "manifest.json")
    return manifest["domains"][domain]["folds"][fold]["dataset_name"]


def gram_args(config: dict[str, Any], domain: str, fold: str):
    dataset_name = dataset_name_from_manifest(domain, fold)
    domain_config = config["domains"][domain]
    args = create_parser().parse_args(
        [
            "--hierarchical_id_type",
            domain_config["hierarchy"],
            "--datasets",
            dataset_name,
            "--data_path",
            str(PREFLIGHT / "data"),
            "--prompt_file",
            str(ROOT / "GRAM/prompt.txt"),
            "--item_prompt",
            "all_text",
            "--top_k_similar_item",
            str(domain_config["top_k_similar_item"]),
            "--cf_model",
            "sasrec",
            "--id_linking",
            "1",
            "--max_his",
            str(config["parent_training"]["max_history"]),
            "--item_prompt_max_len",
            "128",
            "--target_max_len",
            "32",
            "--item_id_type",
            "split",
            "--reverse_history",
            "1",
            "--user_id_without_target_item",
            "1",
            "--skip_empty_his",
            "1",
            "--tasks",
            "sequential",
            "--sample_num",
            "1",
            "--train",
            "0",
        ]
    )
    args.rank = 0
    args.distributed = 0
    args.debug_train_100 = 0
    args.debug_test_100 = 0
    args.verbose_input_output = 0
    args.s17_modules = ""
    args.tokenizer = None
    return args


def initialize_parent(config: dict[str, Any], device: torch.device):
    snapshot = config["backbone"]["snapshot"]
    model_config = T5Config.from_pretrained(snapshot, local_files_only=True)
    model_config.max_seq_len = 128
    model_config.max_item_num = config["parent_training"]["max_history"]
    model_config.use_position_embedding = 1
    model_config.sample_num = "1"
    model_config.cf0_enabled = False
    model_config.cf0_num_items = 0
    model_config.s17_modules = ""
    backbone = AutoModelForSeq2SeqLM.from_pretrained(
        snapshot, config=model_config, local_files_only=True
    )
    model = create_model("gram", config=model_config)
    model.load_t5(backbone.state_dict())
    del backbone
    return model.to(device)


def load_parent(config: dict[str, Any], checkpoint: Path, device: torch.device):
    model = initialize_parent(config, device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device)


def update_unit_status(domain: str, fold: str, **fields: Any) -> None:
    path = unit_status_path(domain, fold)
    current = load_json(path) if path.is_file() else {
        "schema_version": "phase18.s18_1_unit_status.v1",
        "domain": domain,
        "fold": fold,
        "unit": unit_key(domain, fold),
        "scientific_attempt": "run-0001",
    }
    current.update(fields)
    current["heartbeat_at"] = utc_now()
    atomic_json(path, current)


def train_parent(
    config: dict[str, Any],
    domain: str,
    fold: str,
    device: torch.device,
    tokenizer,
):
    args = gram_args(config, domain, fold)
    args.tokenizer = tokenizer
    dataset_name = dataset_name_from_manifest(domain, fold)
    dataset = MultiTaskDatasetGRAM(
        args, dataset_name, "train", None, tokenizer, phase=0, regenerate=False
    )
    training = config["parent_training"]
    loader = DataLoader(
        dataset,
        batch_size=training["rec_batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        collate_fn=CollatorGRAM(tokenizer=tokenizer, args=args, mode="train"),
        num_workers=0,
    )
    model = initialize_parent(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        eps=1e-6,
    )
    update_steps_per_epoch = math.ceil(len(loader) / training["gradient_accumulation_steps"])
    total_steps = update_steps_per_epoch * training["epochs"]
    warmup_steps = int(total_steps * training["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_lambda(step, total_steps, warmup_steps),
    )
    history = []
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        loss_count = 0
        epoch_started = time.time()
        for batch_index, batch in enumerate(loader, 1):
            output = model(
                input_ids=batch["item_text_ids"].to(device),
                attention_mask=batch["item_text_masks"].to(device),
                history_item_ids=batch["history_item_ids"].to(device),
                history_item_mask=batch["history_item_mask"].to(device),
                target_item_ids=batch["target_item_ids"].to(device),
                labels=batch["target_ids"].to(device),
                return_dict=False,
            )
            raw_loss = output[0]
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"{domain}/{fold}: non-finite parent loss")
            (raw_loss / training["gradient_accumulation_steps"]).backward()
            loss_sum += float(raw_loss.detach())
            loss_count += 1
            if (
                batch_index % training["gradient_accumulation_steps"] == 0
                or batch_index == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_index % 50 == 0:
                update_unit_status(
                    domain,
                    fold,
                    execution_state="RUNNING_PARENT_TRAINING",
                    phase="parent_training",
                    epoch=epoch,
                    epochs=training["epochs"],
                    batch=batch_index,
                    batches=len(loader),
                    elapsed_seconds=time.time() - started,
                )
        history.append(
            {
                "epoch": epoch,
                "mean_loss": loss_sum / loss_count,
                "wall_time_seconds": time.time() - epoch_started,
            }
        )
        update_unit_status(
            domain,
            fold,
            execution_state="RUNNING_PARENT_TRAINING",
            phase="parent_training",
            epoch=epoch,
            epochs=training["epochs"],
            batch=len(loader),
            batches=len(loader),
            latest_mean_loss=history[-1]["mean_loss"],
            elapsed_seconds=time.time() - started,
        )
    checkpoint = unit_dir(domain, fold) / "parent_epoch10.pt"
    torch.save(
        {
            "schema_version": "phase18.s18_1_parent.v1",
            "domain": domain,
            "fold": fold,
            "seed": config["seed"],
            "epochs": training["epochs"],
            "target_based_selection": False,
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        checkpoint,
    )
    return model, args, dataset, {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "samples": len(dataset),
        "batches": len(loader),
        "history": history,
        "wall_time_seconds": time.time() - started,
    }


def read_numeric_fold_data(dataset_dir: Path, index_name: str):
    raw_items = []
    with (dataset_dir / index_name).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_items.append(line.split(" ", 1)[0])
    item_to_id = {item: index + 1 for index, item in enumerate(sorted(raw_items))}
    samples = []
    frequencies: Counter[int] = Counter()
    sequences: dict[str, list[int]] = {}
    with (dataset_dir / "user_sequence.txt").open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            user = fields[0]
            sequence = [item_to_id[item] for item in fields[1:]]
            visible = sequence[:-2]
            sequences[user] = sequence
            frequencies.update(visible)
            for index in range(1, len(visible)):
                samples.append((user, visible[max(0, index - 20) : index], visible[index]))
    return item_to_id, samples, frequencies, sequences


def train_item_head(
    config: dict[str, Any], domain: str, fold: str, device: torch.device
):
    domain_config = config["domains"][domain]
    index_name = f"item_generative_indexing_{domain_config['hierarchy']}.txt"
    dataset_dir = PREFLIGHT / "data" / dataset_name_from_manifest(domain, fold)
    item_to_id, samples, frequencies, sequences = read_numeric_fold_data(
        dataset_dir, index_name
    )
    training = config["item_head_training"]
    loader = DataLoader(
        SequenceDataset(samples),
        batch_size=training["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        collate_fn=collate_sequences,
        num_workers=0,
    )
    model = CF0B2ItemHead(
        num_items=len(item_to_id),
        max_history=training["max_history"],
        d_model=training["d_model"],
        num_layers=training["layers"],
        num_heads=training["heads"],
        dropout=training["dropout"],
        temperature=training["temperature"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    total_steps = len(loader) * training["epochs"]
    warmup_steps = int(total_steps * 0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: scheduler_lambda(step, total_steps, warmup_steps)
    )
    history = []
    started = time.time()
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        losses = []
        for batch in loader:
            histories = batch["history_item_ids"].to(device)
            masks = batch["history_item_mask"].to(device)
            targets = batch["target_item_ids"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(histories, masks, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{domain}/{fold}: non-finite item-head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "mean_loss": float(np.mean(losses))})
        update_unit_status(
            domain,
            fold,
            execution_state="RUNNING_ITEM_HEAD_TRAINING",
            phase="item_head_training",
            epoch=epoch,
            epochs=training["epochs"],
            latest_mean_loss=history[-1]["mean_loss"],
        )
    checkpoint = unit_dir(domain, fold) / "item_head_epoch10.pt"
    torch.save(
        {
            "schema_version": "phase18.s18_1_item_head.v1",
            "domain": domain,
            "fold": fold,
            "seed": config["seed"],
            "target_based_selection": False,
            "model_config": {
                "num_items": len(item_to_id),
                "max_history": training["max_history"],
                "d_model": training["d_model"],
                "num_layers": training["layers"],
                "num_heads": training["heads"],
                "dropout": training["dropout"],
                "temperature_initial": training["temperature"],
            },
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        checkpoint,
    )
    return model.eval(), item_to_id, frequencies, sequences, {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "samples": len(samples),
        "batches": len(loader),
        "history": history,
        "wall_time_seconds": time.time() - started,
    }


def normalize_lexical_id(raw: str) -> str:
    return raw.replace("|▁", " ").replace("|", "").strip()


def identifier_tokens(tokenizer, identifier: str) -> tuple[int, ...]:
    return tuple(
        token for token in tokenizer.encode(identifier) if token not in (1820, 9175)
    )


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-6)


class TracePrefixAllowed:
    def __init__(self, trie):
        self.trie = trie
        self.active: dict[int, set[tuple[int, ...]]] = defaultdict(set)

    def __call__(self, batch_id: int, sentence: torch.Tensor):
        values = sentence.tolist()
        generated = tuple(int(token) for token in values[1:])
        self.active[len(generated)].add(generated)
        return self.trie.get(values)


def trim_generated_path(row: torch.Tensor) -> tuple[int, ...]:
    values = [int(token) for token in row.tolist()[1:]]
    if 1 in values:
        values = values[: values.index(1) + 1]
    return tuple(values)


def generate_one(
    model,
    batch,
    trie,
    max_length: int,
    width: int,
    device: torch.device,
    *,
    use_cache: bool = True,
):
    tracer = TracePrefixAllowed(trie)
    with torch.no_grad():
        output = model.generate(
            input_ids=batch["item_text_ids"].to(device),
            attention_mask=batch["item_text_masks"].to(device),
            history_item_ids=batch["history_item_ids"].to(device),
            history_item_mask=batch["history_item_mask"].to(device),
            max_length=max_length,
            prefix_allowed_tokens_fn=tracer,
            num_beams=width,
            num_return_sequences=width,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,
            use_cache=use_cache,
        )
        transitions = model.compute_transition_scores(
            output.sequences,
            output.scores,
            output.beam_indices,
            normalize_logits=True,
        )
    paths = [trim_generated_path(row) for row in output.sequences]
    raw_logp = transitions.float().sum(dim=1).cpu().numpy().astype(np.float64)
    normalized = output.sequences_scores.float().cpu().numpy().astype(np.float64)
    return output, paths, raw_logp, normalized, tracer.active


def set_cross_attention_cache(model, enabled: bool) -> None:
    """Toggle only decoder cross-attention K/V retention for generation."""

    for block in model.decoder.block:
        if not hasattr(block, "cache_cross_attention"):
            raise RuntimeError("decoder block lacks the cross-attention cache control")
        block.cache_cross_attention = enabled


def enable_two_gpu_decoder_parallel(model) -> dict[int, list[int]]:
    """Split decoder layers over two visible GPUs while leaving the encoder on GPU0."""

    if torch.cuda.device_count() != 2:
        raise RuntimeError("two-GPU decoder parallelism requires exactly two visible GPUs")
    layer_count = len(model.decoder.block)
    split = math.ceil(layer_count / 2)
    device_map = {0: list(range(split)), 1: list(range(split, layer_count))}
    model.decoder.parallelize(device_map)
    model.lm_head = model.lm_head.to(model.decoder.first_device)
    model.model_parallel = True
    model.device_map = device_map
    # The customized T5 forward path uses this attribute only to place the LM
    # head output.  The wrapped encoder itself remains entirely on visible GPU0.
    model.encoder.first_device = model.decoder.first_device
    return device_map


def release_cuda_caches() -> None:
    """Release unused allocator blocks on every visible CUDA device."""

    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()


def decode_generated(tokenizer, output) -> list[str]:
    return [
        normalize_lexical_id(value)
        for value in tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    ]


def item_head_scores(
    model: CF0B2ItemHead,
    history: list[int],
    max_history: int,
    device: torch.device,
) -> np.ndarray:
    clipped = history[-max_history:]
    ids = torch.zeros(1, max_history, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    if clipped:
        ids[0, : len(clipped)] = torch.tensor(clipped, device=device)
        mask[0, : len(clipped)] = True
    with torch.no_grad():
        return model.score(ids, mask)[0].float().cpu().numpy().astype(np.float64)


def teacher_force_paths(
    model,
    batch,
    paths: list[tuple[int, ...]],
    device: torch.device,
) -> list[dict[str, Any]]:
    if not paths:
        return []
    max_length = max(len(path) for path in paths)
    labels = torch.full((len(paths), max_length), -100, dtype=torch.long, device=device)
    for row, path in enumerate(paths):
        labels[row, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    history_ids = batch["history_item_ids"].to(device)
    history_mask = batch["history_item_mask"].to(device)
    with torch.no_grad():
        model.encoder.n_passages = input_ids.size(1)
        model.encoder.set_migration_context(history_ids, history_mask)
        flat_ids = input_ids.reshape(1, -1)
        flat_attention = attention.reshape(1, -1)
        hidden = model.encoder(
            input_ids=flat_ids, attention_mask=flat_attention, return_dict=True
        )[0]
        output = model(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=hidden.repeat(len(paths), 1, 1)
            ),
            attention_mask=flat_attention.repeat(len(paths), 1),
            labels=labels,
            return_dict=True,
        )
        log_probs = torch.log_softmax(output.logits.float(), dim=-1)
    rows = []
    for index, path in enumerate(paths):
        positions = torch.arange(len(path), device=device)
        tokens = torch.tensor(path, dtype=torch.long, device=device)
        chosen = log_probs[index, positions, tokens]
        raw = float(chosen.sum().cpu())
        rows.append(
            {
                "finite": bool(torch.isfinite(chosen).all().cpu()),
                "raw_log_probability": raw,
                "normalized_log_probability": raw / len(path),
                "path_probability": math.exp(raw) if raw > -745 else 0.0,
                "path_length": len(path),
            }
        )
    return rows


def descriptive(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def diagnose(
    config: dict[str, Any],
    domain: str,
    fold: str,
    device: torch.device,
    tokenizer,
    parent,
    args,
    item_head,
    item_to_id: dict[str, int],
    frequencies: Counter[int],
    numeric_sequences: dict[str, list[int]],
    max_users: int | None = None,
    generation_use_cache: bool = True,
    cross_attention_cache: bool = True,
    release_cuda_cache_per_user: bool = False,
) -> dict[str, Any]:
    set_cross_attention_cache(parent, cross_attention_cache)
    dataset_name = dataset_name_from_manifest(domain, fold)
    dataset = TestDatasetGRAM(
        args,
        dataset_name,
        "sequential",
        None,
        tokenizer,
        regenerate=False,
        phase=0,
        mode="validation",
    )
    dataset_index = {user: index for index, user in enumerate(dataset.data["user_id"])}
    selected = (PREFLIGHT / f"cohort_{domain}.txt").read_text(encoding="utf-8").splitlines()
    if len(selected) != config["cohort"]["users_per_domain"] or set(selected) - set(dataset_index):
        raise RuntimeError(f"{domain}/{fold}: frozen cohort mismatch")
    if max_users is not None:
        if max_users < 1 or max_users > len(selected):
            raise ValueError("max_users must select a non-empty subset of the frozen cohort")
        selected = selected[:max_users]
    collator = CollatorGRAM(tokenizer=tokenizer, args=args, mode="valid")
    raw_to_identifier = dataset.item2lexid
    normalized_to_raw = {
        normalize_lexical_id(identifier): raw
        for raw, identifier in raw_to_identifier.items()
    }
    if len(normalized_to_raw) != len(raw_to_identifier):
        raise RuntimeError(f"{domain}/{fold}: normalized lexical collision")
    item_paths = {
        raw: identifier_tokens(tokenizer, identifier)
        for raw, identifier in raw_to_identifier.items()
    }
    encoded = [[0, *path] for path in item_paths.values()]
    trie = gt.Trie(encoded)
    max_length = max(len(row) for row in encoded)
    catalog_freq = np.asarray(
        [frequencies.get(index, 0) for index in range(1, len(item_to_id) + 1)],
        dtype=np.float64,
    )
    pop_z_global = standardize(np.log1p(catalog_freq))
    q1 = load_json(PREFLIGHT / "manifest.json")["domains"][domain]["folds"][fold]["q1"]
    hard_k = config["generation"]["hard_negative_k"]
    beta = config["pcrf"]["beta"]
    output50 = unit_dir(domain, fold) / "beams_w50.tsv"
    output200 = unit_dir(domain, fold) / "beams_w200.tsv"
    diagnostic_path = unit_dir(domain, fold) / "per_user_diagnostics.jsonl"
    rows: list[dict[str, Any]] = []
    started = time.time()
    with output50.open("w", encoding="utf-8", newline="") as h50, output200.open(
        "w", encoding="utf-8", newline=""
    ) as h200, diagnostic_path.open("w", encoding="utf-8") as diag:
        writers = {50: csv.writer(h50, delimiter="\t"), 200: csv.writer(h200, delimiter="\t")}
        for writer in writers.values():
            writer.writerow(["user", "target", "candidates", "normalized_scores", "raw_log_probabilities"])
        for ordinal, user in enumerate(selected, 1):
            index = dataset_index[user]
            sample = dataset.data_samples[index]
            batch = collator([dataset[index]])
            generations = {}
            all_legal = True
            all_finite = True
            for width in (50, 200):
                output, paths, raw_logp, normalized, active = generate_one(
                    parent,
                    batch,
                    trie,
                    max_length,
                    width,
                    device,
                    use_cache=generation_use_cache,
                )
                decoded = decode_generated(tokenizer, output)
                raw_items = [normalized_to_raw.get(value) for value in decoded]
                legal = (
                    len(raw_items) == width
                    and None not in raw_items
                    and len(set(raw_items)) == width
                    and all(tuple(path) == item_paths[raw] for raw, path in zip(raw_items, paths))
                )
                finite = bool(np.isfinite(raw_logp).all() and np.isfinite(normalized).all())
                all_legal = all_legal and legal
                all_finite = all_finite and finite
                if not legal or not finite:
                    raise RuntimeError(f"{domain}/{fold}/{user}: illegal or non-finite beam{width}")
                generations[width] = {
                    "items": raw_items,
                    "paths": paths,
                    "raw_logp": raw_logp,
                    "normalized": normalized,
                    "active": active,
                }
                writers[width].writerow(
                    [
                        user,
                        sample["target"],
                        "||".join(raw_items),
                        "||".join(map(str, normalized.tolist())),
                        "||".join(map(str, raw_logp.tolist())),
                    ]
                )
                # All result-bearing tensors have been copied to CPU containers above.
                # Releasing the generation object here prevents beam-search score/cache
                # tensors from surviving until the next user.  ``empty_cache`` is an
                # opt-in resource adaptation; it does not change model inputs or scores.
                del output
                if release_cuda_cache_per_user:
                    release_cuda_caches()
            target = sample["target"]
            target_path = item_paths[target]
            beam50_items = generations[50]["items"]
            beam200_items = generations[200]["items"]
            hit50 = target in beam50_items
            hit200 = target in beam200_items
            numeric = numeric_sequences[user]
            history = numeric[:-2][-config["item_head_training"]["max_history"] :]
            cf_scores = item_head_scores(
                item_head,
                history,
                config["item_head_training"]["max_history"],
                device,
            )
            target_id = item_to_id[target]
            target_cf_z = catalog_standardized_target(cf_scores[target_id - 1], cf_scores)
            record: dict[str, Any] = {
                "user": user,
                "target": target,
                "hit_beam50": hit50,
                "hit_beam200": hit200,
                "beam200_only": hit200 and not hit50,
                "trie_legal": all_legal,
                "finite": bool(all_finite and np.isfinite(cf_scores).all()),
                "target_cf_z": target_cf_z,
                "first_drop_depth": None,
                "actual_pruner_items": [],
                "actual_pruner_legal_fraction": None,
                "hard_negative_items": [],
                "hard_negative_recall": None,
                "hard_negative_intersection": 0,
                "hard_negative_actual_denominator": 0,
                "teacher_forced_finite": True,
                "target_minus_negative_path_gap": None,
                "target_minus_negative_cf_margin": None,
                "target_minus_negative_logpop_margin": None,
                "wrong_beam_normalized_log_probability": None,
                "wrong_beam_raw_log_probability": None,
                "wrong_beam_path_probability": None,
                "wrong_beam_pcrf_difficulty": None,
                "target_raw_log_probability": None,
                "target_path_probability": None,
            }
            if hit200 and not hit50:
                drop = first_drop_depth(generations[50]["active"], target_path)
                if drop is None:
                    drop = len(target_path)
                record["first_drop_depth"] = drop
                legal_children = set(trie.get([0, *target_path[: drop - 1]]))
                same_parent = {
                    raw
                    for raw in beam50_items
                    if len(item_paths[raw]) >= drop
                    and item_paths[raw][: drop - 1] == target_path[: drop - 1]
                    and item_paths[raw][drop - 1] != target_path[drop - 1]
                }
                actual = actual_pruner_items(
                    {raw: item_paths[raw] for raw in beam50_items},
                    target_path,
                    drop,
                    legal_children,
                )
                record["actual_pruner_items"] = sorted(actual)
                record["actual_pruner_legal_fraction"] = (
                    len(actual) / len(same_parent) if same_parent else None
                )
                candidate_ids = np.asarray([item_to_id[raw] for raw in beam200_items])
                candidate_cf = cf_scores[candidate_ids - 1]
                candidate_pop = catalog_freq[candidate_ids - 1]
                seq_z = standardize(generations[200]["normalized"])
                cf_z = standardize(candidate_cf)
                pop_z = standardize(np.log1p(candidate_pop))
                cf_pc = standardize(cf_z - beta * pop_z)
                tail_mass = float(
                    np.mean(
                        catalog_freq[
                            np.asarray([item_to_id[raw] for raw in beam50_items[:10]]) - 1
                        ]
                        <= q1
                    )
                )
                reliability = 1.0 - tail_mass
                joint = seq_z + reliability * cf_pc
                joint_by_item = {raw: float(joint[pos]) for pos, raw in enumerate(beam200_items)}
                normalized_by_item = {
                    raw: float(generations[200]["normalized"][pos])
                    for pos, raw in enumerate(beam200_items)
                }
                raw_logp_by_item = {
                    raw: float(generations[200]["raw_logp"][pos])
                    for pos, raw in enumerate(beam200_items)
                }
                record["target_raw_log_probability"] = raw_logp_by_item[target]
                record["target_path_probability"] = (
                    math.exp(raw_logp_by_item[target])
                    if raw_logp_by_item[target] > -745
                    else 0.0
                )
                competing = [
                    raw
                    for raw in beam200_items
                    if raw != target
                    and len(item_paths[raw]) >= drop
                    and item_paths[raw][: drop - 1] == target_path[: drop - 1]
                    and item_paths[raw][drop - 1] != target_path[drop - 1]
                    and item_paths[raw][drop - 1] in legal_children
                ]
                selected_negatives = sorted(
                    competing, key=lambda raw: (-joint_by_item[raw], raw)
                )[:hard_k]
                if len(selected_negatives) < hard_k:
                    global_cf_pc = standardize(
                        standardize(cf_scores) - beta * pop_z_global
                    )
                    supplements = [
                        raw
                        for raw, path in item_paths.items()
                        if raw != target
                        and raw not in selected_negatives
                        and len(path) >= drop
                        and path[: drop - 1] == target_path[: drop - 1]
                        and path[drop - 1] != target_path[drop - 1]
                        and path[drop - 1] in legal_children
                    ]
                    supplements.sort(
                        key=lambda raw: (-float(global_cf_pc[item_to_id[raw] - 1]), raw)
                    )
                    selected_negatives.extend(supplements[: hard_k - len(selected_negatives)])
                record["hard_negative_items"] = selected_negatives
                if actual:
                    recall = hard_negative_recall(selected_negatives, actual)
                    record["hard_negative_recall"] = recall
                    record["hard_negative_intersection"] = len(set(selected_negatives) & actual)
                    record["hard_negative_actual_denominator"] = len(actual)
                forced_items = [target, *selected_negatives]
                forced = teacher_force_paths(
                    parent,
                    batch,
                    [item_paths[raw] for raw in forced_items],
                    device,
                )
                forced_by_item = dict(zip(forced_items, forced))
                record["teacher_forced_finite"] = all(row["finite"] for row in forced)
                negative_path_scores = [
                    forced_by_item[raw]["normalized_log_probability"]
                    for raw in selected_negatives
                ]
                negative_cf = [cf_scores[item_to_id[raw] - 1] for raw in selected_negatives]
                negative_pop = [catalog_freq[item_to_id[raw] - 1] for raw in selected_negatives]
                if selected_negatives:
                    record["target_minus_negative_path_gap"] = float(
                        forced_by_item[target]["normalized_log_probability"]
                        - np.mean(negative_path_scores)
                    )
                    record["target_minus_negative_cf_margin"] = float(
                        cf_scores[target_id - 1] - np.mean(negative_cf)
                    )
                    record["target_minus_negative_logpop_margin"] = float(
                        math.log1p(catalog_freq[target_id - 1])
                        - np.mean(np.log1p(negative_pop))
                    )
                if competing:
                    record["wrong_beam_normalized_log_probability"] = float(
                        np.mean([normalized_by_item[raw] for raw in competing])
                    )
                    record["wrong_beam_raw_log_probability"] = float(
                        np.mean([raw_logp_by_item[raw] for raw in competing])
                    )
                    record["wrong_beam_path_probability"] = float(
                        np.mean(
                            [
                                math.exp(raw_logp_by_item[raw])
                                if raw_logp_by_item[raw] > -745
                                else 0.0
                                for raw in competing
                            ]
                        )
                    )
                    record["wrong_beam_pcrf_difficulty"] = float(
                        np.mean([joint_by_item[raw] for raw in competing])
                    )
            rows.append(record)
            diag.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=json_default,
                )
                + "\n"
            )
            if ordinal % 8 == 0:
                h50.flush()
                h200.flush()
                diag.flush()
            if ordinal % 16 == 0:
                update_unit_status(
                    domain,
                    fold,
                    execution_state="RUNNING_BOUNDED_GENERATION",
                    phase="beam50_beam200_diagnostic",
                    users_completed=ordinal,
                    users_total=len(selected),
                    elapsed_seconds=time.time() - started,
                )
            if release_cuda_cache_per_user:
                release_cuda_caches()

    beam200_only = [row for row in rows if row["beam200_only"]]
    actual_events = [row for row in beam200_only if row["actual_pruner_items"]]
    finite_legal = all(
        row["finite"] and row["trie_legal"] and row["teacher_forced_finite"]
        for row in rows
    )
    summary = {
        "schema_version": "phase18.s18_1_unit_summary.v1",
        "domain": domain,
        "fold": fold,
        "users": len(rows),
        "beam50_hits": sum(row["hit_beam50"] for row in rows),
        "beam200_hits": sum(row["hit_beam200"] for row in rows),
        "headroom": (
            sum(row["hit_beam200"] for row in rows)
            - sum(row["hit_beam50"] for row in rows)
        )
        / len(rows),
        "beam200_only_events": len(beam200_only),
        "first_drop_events": len(beam200_only),
        "first_drop_depth_distribution": dict(
            sorted(Counter(str(row["first_drop_depth"]) for row in beam200_only).items())
        ),
        "nonempty_actual_pruner_events": len(actual_events),
        "nonempty_actual_pruner_fraction": len(actual_events) / max(1, len(beam200_only)),
        "actual_pruner_items_total": sum(len(row["actual_pruner_items"]) for row in actual_events),
        "actual_pruner_legal_fraction": (
            float(
                np.mean(
                    [
                        row["actual_pruner_legal_fraction"]
                        for row in actual_events
                        if row["actual_pruner_legal_fraction"] is not None
                    ]
                )
            )
            if actual_events
            else None
        ),
        "hard_negative_intersection_total": sum(
            row["hard_negative_intersection"] for row in actual_events
        ),
        "hard_negative_actual_denominator_total": sum(
            row["hard_negative_actual_denominator"] for row in actual_events
        ),
        "k8_actual_pruner_recall": (
            sum(row["hard_negative_intersection"] for row in actual_events)
            / max(1, sum(row["hard_negative_actual_denominator"] for row in actual_events))
        ),
        "finite_and_trie_legal": finite_legal,
        "target_cf_z": descriptive([row["target_cf_z"] for row in rows]),
        "wrong_beam_normalized_log_probability": descriptive(
            [row["wrong_beam_normalized_log_probability"] for row in beam200_only if row["wrong_beam_normalized_log_probability"] is not None]
        ),
        "wrong_beam_raw_log_probability": descriptive(
            [row["wrong_beam_raw_log_probability"] for row in beam200_only if row["wrong_beam_raw_log_probability"] is not None]
        ),
        "wrong_beam_path_probability": descriptive(
            [row["wrong_beam_path_probability"] for row in beam200_only if row["wrong_beam_path_probability"] is not None]
        ),
        "target_raw_log_probability": descriptive(
            [row["target_raw_log_probability"] for row in beam200_only if row["target_raw_log_probability"] is not None]
        ),
        "target_path_probability": descriptive(
            [row["target_path_probability"] for row in beam200_only if row["target_path_probability"] is not None]
        ),
        "wrong_beam_pcrf_difficulty": descriptive(
            [row["wrong_beam_pcrf_difficulty"] for row in beam200_only if row["wrong_beam_pcrf_difficulty"] is not None]
        ),
        "target_minus_negative_path_gap": descriptive(
            [row["target_minus_negative_path_gap"] for row in beam200_only if row["target_minus_negative_path_gap"] is not None]
        ),
        "target_minus_negative_cf_margin": descriptive(
            [row["target_minus_negative_cf_margin"] for row in beam200_only if row["target_minus_negative_cf_margin"] is not None]
        ),
        "target_minus_negative_logpop_margin": descriptive(
            [row["target_minus_negative_logpop_margin"] for row in beam200_only if row["target_minus_negative_logpop_margin"] is not None]
        ),
        "attrition": {
            "cohort": len(rows),
            "beam200_only": len(beam200_only),
            "empty_actual_pruner": len(beam200_only) - len(actual_events),
            "nonempty_actual_pruner": len(actual_events),
            "hard_negative_fewer_than_k": sum(
                len(row["hard_negative_items"]) < hard_k for row in beam200_only
            ),
        },
        "files": {
            "beam50": {"path": str(output50.relative_to(ROOT)), "sha256": sha256(output50)},
            "beam200": {"path": str(output200.relative_to(ROOT)), "sha256": sha256(output200)},
            "diagnostics": {"path": str(diagnostic_path.relative_to(ROOT)), "sha256": sha256(diagnostic_path)},
        },
        "wall_time_seconds": time.time() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "generation_runtime": {
            "use_cache": generation_use_cache,
            "cross_attention_cache": cross_attention_cache,
            "release_cuda_cache_per_user": release_cuda_cache_per_user,
            "scientific_parameters_changed": False,
        },
    }
    return summary


def run_unit(domain: str, fold: str, physical_gpu: int) -> int:
    config, _ = load_contracts()
    expected_gpu = load_json(AUTH_PATH)["gpu_assignment"].get(f"{domain}:{fold}")
    if expected_gpu != physical_gpu:
        raise RuntimeError(f"unit GPU mismatch: expected {expected_gpu}, got {physical_gpu}")
    target = unit_dir(domain, fold)
    if target.exists():
        raise FileExistsError(f"unit output exists; automatic retry forbidden: {target}")
    target.mkdir(parents=True)
    started = time.time()
    update_unit_status(
        domain,
        fold,
        execution_state="STARTING",
        phase="initialization",
        physical_gpu=physical_gpu,
        visible_cuda_device=0,
        pid=os.getpid(),
        started_at=utc_now(),
        process_alive=True,
        automatic_retry=False,
    )
    try:
        set_seed(config["seed"])
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer = AutoTokenizer.from_pretrained(
            config["backbone"]["snapshot"], local_files_only=True
        )
        parent, args, _, parent_info = train_parent(
            config, domain, fold, device, tokenizer
        )
        item_head, item_to_id, frequencies, sequences, head_info = train_item_head(
            config, domain, fold, device
        )
        parent.eval()
        diagnostic = diagnose(
            config,
            domain,
            fold,
            device,
            tokenizer,
            parent,
            args,
            item_head,
            item_to_id,
            frequencies,
            sequences,
        )
        summary = {
            **diagnostic,
            "experiment_id": "s18_s1_actionability",
            "attempt_id": "run-0001",
            "status": "COMPLETED",
            "parent_training": parent_info,
            "item_head_training": head_info,
            "physical_gpu": physical_gpu,
            "wall_time_total_seconds": time.time() - started,
            "d1_read": False,
            "d2_read": False,
            "test_read": False,
            "sports_read": False,
            "treatment_training": False,
        }
        atomic_json(target / "summary.json", summary)
        update_unit_status(
            domain,
            fold,
            execution_state="COMPLETED",
            phase="complete",
            process_alive=False,
            summary_path=str((target / "summary.json").relative_to(ROOT)),
            summary_sha256=sha256(target / "summary.json"),
            elapsed_seconds=time.time() - started,
        )
        return 0
    except Exception as error:
        atomic_text(target / "failure.txt", f"{type(error).__name__}: {error}\n")
        update_unit_status(
            domain,
            fold,
            execution_state="FAILED_NO_RETRY",
            phase="failed",
            process_alive=False,
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.time() - started,
        )
        raise


def aggregate_results(config: dict[str, Any]) -> dict[str, Any]:
    units = {}
    for domain in config["domains"]:
        for fold in config["folds"]:
            path = unit_dir(domain, fold) / "summary.json"
            units[f"{domain}:{fold}"] = load_json(path)
    domains = {}
    for domain in config["domains"]:
        rows = [units[f"{domain}:{fold}"] for fold in ("I-1", "I0")]
        users = sum(row["users"] for row in rows)
        first_drop = sum(row["first_drop_events"] for row in rows)
        actual_events = sum(row["nonempty_actual_pruner_events"] for row in rows)
        actual_denominator = sum(row["hard_negative_actual_denominator_total"] for row in rows)
        metrics = {
            "pooled_headroom": (
                sum(row["beam200_hits"] for row in rows)
                - sum(row["beam50_hits"] for row in rows)
            )
            / users,
            "beam200_only_events": sum(row["beam200_only_events"] for row in rows),
            "nonempty_actual_pruner_fraction": actual_events / max(1, first_drop),
            "k8_actual_pruner_recall": sum(
                row["hard_negative_intersection_total"] for row in rows
            )
            / max(1, actual_denominator),
            "finite_and_trie_legal": all(row["finite_and_trie_legal"] for row in rows),
            "cf_target_z_mean_drift": units[f"{domain}:I0"]["target_cf_z"]["mean"]
            - units[f"{domain}:I-1"]["target_cf_z"]["mean"],
        }
        domains[domain] = {
            "metrics": metrics,
            "gate": evaluate_domain_gate(metrics, config["gates"]),
            "folds": {row["fold"]: row for row in rows},
        }
    decisions = [row["gate"]["decision"] for row in domains.values()]
    if "NO_ACTIONABLE_PREFIX_BOTTLENECK" in decisions:
        decision = "NO_ACTIONABLE_PREFIX_BOTTLENECK"
    elif "CF_TEACHER_UNSTABLE" in decisions:
        decision = "CF_TEACHER_UNSTABLE"
    else:
        decision = "S18_1_ACTIONABILITY_PASS_AWAIT_S18_2_AUTHORIZATION"
    return {
        "schema_version": "phase18.s18_1_summary.v1",
        "experiment_id": "s18_s1_actionability",
        "attempt_id": "run-0001",
        "status": "COMPLETED",
        "decision": decision,
        "domains": domains,
        "scientific_config_sha256": sha256(CONFIG_PATH),
        "resource_authorization_sha256": sha256(AUTH_PATH),
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
        "treatment_training": False,
        "automatic_retry": False,
        "automatic_s18_2": False,
        "completed_at": utc_now(),
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Stage18 S18-1 可作用性与 first-drop 诊断报告",
        "",
        "## Material Passport",
        "",
        f"- Experiment：`{summary['experiment_id']} / {summary['attempt_id']}`",
        "- Verification Status：`ANALYZED`",
        f"- Decision：`{summary['decision']}`",
        "- Protected data：D1/D2、official validation/test、Sports 均未读取",
        "- Treatment training：未发生",
        "",
        "## Domain Gate",
        "",
        "| Domain | headroom | beam200-only | nonempty pruner | K8 recall | CF z drift | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for domain, row in summary["domains"].items():
        metric = row["metrics"]
        lines.append(
            f"| {domain} | {metric['pooled_headroom']:.6f} | {metric['beam200_only_events']} "
            f"| {metric['nonempty_actual_pruner_fraction']:.6f} | {metric['k8_actual_pruner_recall']:.6f} "
            f"| {metric['cf_target_z_mean_drift']:.6f} | `{row['gate']['decision']}` |"
        )
    lines.extend(
        [
            "",
            "## Execution Contract",
            "",
            "- 每个 domain×fold parent 与 item-head 均只见 fold-visible transition。",
            "- parent 从冻结通用 t5-small 独立初始化；未复用未来泄漏 checkpoint。",
            "- beam200 只用于诊断；未训练 PCPS treatment。",
            "- 未自动重试，未自动启动 S18-2。",
            "",
        ]
    )
    if summary.get("recovery_of"):
        lines.extend(
            [
                "## Infrastructure Correction",
                "",
                f"- Recovery of：`{summary['recovery_of']}`",
                "- 仅加载原 attempt 的冻结 epoch-10 checkpoints 并重跑诊断；未重新训练。",
                "- 修复范围仅为 NumPy 标量的 JSON 序列化，不改变 cohort、模型、teacher、beam 或 Gate。",
                "",
            ]
        )
    atomic_text(REPORT, "\n".join(lines))


def update_aggregate_status(**fields: Any) -> None:
    current = load_json(STATUS) if STATUS.is_file() else {}
    current.update(fields)
    current["updated_at"] = utc_now()
    current["heartbeat_at"] = utc_now()
    atomic_json(STATUS, current)


def source_manifest() -> dict[str, str]:
    paths = [
        CONFIG_PATH,
        AUTH_PATH,
        ROOT / "plan/第十八阶段/GRAM_第十八阶段_PCPS-GRAM词法锚定协同前缀生存与低风险验证计划v0.2.md",
        ROOT / "plan/第十八阶段/GRAM_第十八阶段_S18-1可作用性诊断执行补遗v0.1.md",
        ROOT / "experiment/phase18/core/contracts.py",
        ROOT / "experiment/phase18/core/s1_contracts.py",
        ROOT / "experiment/phase18/protocol/s18_s1_prepare.py",
        ROOT / "experiment/phase18/protocol/s18_s1_runtime.py",
        ROOT / "GRAM/src/model/gram.py",
        ROOT / "GRAM/src/model/gram_t5.py",
        ROOT / "GRAM/src/data/multi_task_dataset_gram.py",
        ROOT / "GRAM/src/data/test_dataset_gram.py",
        ROOT / "GRAM/src/processor/Collator.py",
        ROOT / "GRAM/src/utils/generation_trie.py",
        ROOT / "experiment/phase9/train_cf0_b2_item_head.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = None
    errors = []
    for attempt in range(3):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            errors.append(repr(exc))
            if attempt < 2:
                time.sleep(1)
    if result is None:
        raise RuntimeError(f"nvidia-smi resource probe failed three times: {errors}")
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, used, free, utilization = [value.strip() for value in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization_gpu_percent": int(utilization),
            }
        )
    return rows


def takeover_guard() -> int:
    _, auth = load_contracts()
    takeover = auth["gpu4_takeover"]
    status_path = ROOT / takeover["source_status"]
    status = load_json(status_path)
    if (
        status.get("experiment_id") != "s17_fp12_external_d0_g1_guard_v3"
        or status.get("scientific_state") != takeover["required_scientific_state"]
        or status.get("result_selection_eligible") is not takeover["required_result_selection_eligible"]
        or status.get("target_gpu_id") != 4
    ):
        raise RuntimeError("refusing GPU4 takeover: Phase17 guard contract mismatch")
    session = status.get("tmux_session")
    before = ROOT / status["canonical_result_dir"] / "status_before_s18_1_takeover.json"
    atomic_json(before, status)
    exists = subprocess.run(
        ["tmux", "has-session", "-t", session], capture_output=True, check=False
    ).returncode == 0
    if exists:
        subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], check=True)
        deadline = time.time() + 30
        while time.time() < deadline:
            if subprocess.run(
                ["tmux", "has-session", "-t", session], capture_output=True, check=False
            ).returncode != 0:
                break
            time.sleep(1)
        else:
            # A sleeping occupancy worker can consume C-c without exiting. The
            # exact session and guard contract were validated above.
            subprocess.run(["tmux", "kill-session", "-t", session], check=True)
    final = {
        **status,
        "execution_state": "STOPPED_OCCUPANCY_RELEASED",
        "status_code": "SCIENTIFIC_COMPLETED_OCCUPANCY_RELEASED_FOR_S18_1",
        "stage": "occupancy_repeat_released_for_s18_1",
        "process_alive": False,
        "workload_pid": 0,
        "gpu_ids": [],
        "released_at": utc_now(),
        "released_for": "s18_s1_actionability/run-0001",
        "updated_at": utc_now(),
        "heartbeat_at": utc_now(),
        "scientific_state": "COMPLETED",
        "result_selection_eligible": False,
        "affects_scientific_result": False,
    }
    atomic_json(status_path, final)
    archive = status_path.with_name(
        "s17_fp12_external_d0_g1_guard_v3.attempt_001.released_for_s18_1.status.json"
    )
    atomic_json(archive, final)
    snapshot = gpu_snapshot()
    gpu4 = next(row for row in snapshot if row["index"] == 4)
    if gpu4["free_mib"] < 30000:
        raise RuntimeError(f"GPU4 takeover left only {gpu4['free_mib']} MiB free")
    print(json.dumps({"guard_released": True, "archive": str(archive.relative_to(ROOT)), "gpu4": gpu4}))
    return 0


def smoke(physical_gpu: int) -> int:
    config, _ = load_contracts()
    if SMOKE.exists():
        raise FileExistsError("S18-1 GPU smoke already exists; automatic rerun forbidden")
    SMOKE.mkdir(parents=True)
    started = time.time()
    set_seed(config["seed"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(
        config["backbone"]["snapshot"], local_files_only=True
    )
    args = gram_args(config, "Toys", "I0")
    args.tokenizer = tokenizer
    dataset_name = dataset_name_from_manifest("Toys", "I0")
    train_dataset = MultiTaskDatasetGRAM(
        args, dataset_name, "train", None, tokenizer, phase=0, regenerate=False
    )
    collator = CollatorGRAM(tokenizer=tokenizer, args=args, mode="train")
    batch = collator([train_dataset[0], train_dataset[1]])
    model = initialize_parent(config, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(
        input_ids=batch["item_text_ids"].to(device),
        attention_mask=batch["item_text_masks"].to(device),
        history_item_ids=batch["history_item_ids"].to(device),
        history_item_mask=batch["history_item_mask"].to(device),
        labels=batch["target_ids"].to(device),
        return_dict=False,
    )[0]
    loss.backward()
    optimizer.step()
    model.eval()
    valid_dataset = TestDatasetGRAM(
        args, dataset_name, "sequential", None, tokenizer, phase=0, regenerate=False, mode="validation"
    )
    valid_collator = CollatorGRAM(tokenizer=tokenizer, args=args, mode="valid")
    valid_batch = valid_collator([valid_dataset[0]])
    encoded = [[0, *identifier_tokens(tokenizer, item)] for item in valid_dataset.all_items]
    trie = gt.Trie(encoded)
    output, paths, raw_logp, normalized, _ = generate_one(
        model, valid_batch, trie, max(map(len, encoded)), 4, device
    )
    decoded = decode_generated(tokenizer, output)
    legal = set(decoded).issubset(
        {normalize_lexical_id(value) for value in valid_dataset.item2lexid.values()}
    )
    summary = {
        "schema_version": "phase18.s18_1_smoke.v1",
        "status": "PASSED" if torch.isfinite(loss) and legal and np.isfinite(raw_logp).all() else "FAILED",
        "physical_gpu": physical_gpu,
        "loss_finite": bool(torch.isfinite(loss)),
        "beam_paths": len(paths),
        "beam_legal": legal,
        "scores_finite": bool(np.isfinite(raw_logp).all() and np.isfinite(normalized).all()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "wall_time_seconds": time.time() - started,
        "scientific_result_eligible": False,
        "d1_read": False,
        "d2_read": False,
        "test_read": False,
        "sports_read": False,
    }
    atomic_json(SMOKE / "summary.json", summary)
    print(json.dumps(summary))
    return 0 if summary["status"] == "PASSED" else 1


def launch() -> int:
    config, auth = load_contracts()
    smoke_summary = load_json(SMOKE / "summary.json")
    if smoke_summary.get("status") != "PASSED":
        raise RuntimeError("formal launch requires a passed S18-1 smoke")
    if OUTPUT.exists():
        raise FileExistsError("S18-1 run-0001 already exists; automatic retry forbidden")
    snapshot = gpu_snapshot()
    by_gpu = {row["index"]: row for row in snapshot}
    for gpu in auth["gpu_assignment"].values():
        if by_gpu[gpu]["free_mib"] < 30000:
            raise RuntimeError(f"GPU{gpu} has only {by_gpu[gpu]['free_mib']} MiB free")
    OUTPUT.mkdir(parents=True)
    command = [str(PYTHON), str(Path(__file__).resolve()), "master"]
    manifest = {
        "schema_version": "phase18.s18_1_run_manifest.v1",
        "experiment_id": "s18_s1_actionability",
        "attempt_id": "run-0001",
        "created_at": utc_now(),
        "command": command,
        "working_directory": str(ROOT),
        "source_manifest": source_manifest(),
        "gpu_snapshot": snapshot,
        "gpu_assignment": auth["gpu_assignment"],
        "scientific_config_sha256": sha256(CONFIG_PATH),
        "resource_authorization_sha256": sha256(AUTH_PATH),
        "preflight_manifest_sha256": sha256(PREFLIGHT / "manifest.json"),
        "no_automatic_retry": True,
        "automatic_s18_2": False,
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    append_jsonl(
        LEDGER,
        {
            "event": "scientific_attempt_started",
            "at": utc_now(),
            "experiment_id": "s18_s1_actionability",
            "attempt_id": "run-0001",
            "gpu_assignment": auth["gpu_assignment"],
            "run_manifest_sha256": sha256(OUTPUT / "run_manifest.json"),
        },
    )
    session = auth["launch_contract"]["tmux_session"]
    env_command = [
        "/usr/bin/env",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={ROOT}",
        *command,
    ]
    update_aggregate_status(
        schema_version="phase18.status.v1",
        experiment_id="s18_s1_actionability",
        attempt_id="run-0001",
        step_id="S18-1",
        stage="formal_background_starting",
        execution_state="RUNNING_SCIENTIFIC",
        scientific_state="RUNNING",
        status_code="S18_1_FORMAL_STARTING",
        process_alive=True,
        tmux_session=session,
        workload_pid=0,
        gpu_ids=sorted(set(auth["gpu_assignment"].values())),
        gpu_assignment=auth["gpu_assignment"],
        progress={"current": 0, "total": 4, "unit": "domain_fold"},
        run_manifest_path=str((OUTPUT / "run_manifest.json").relative_to(ROOT)),
        run_manifest_sha256=sha256(OUTPUT / "run_manifest.json"),
        result_selection_eligible=True,
        automatic_retry=False,
        automatic_s18_2=False,
        d1_read=False,
        d2_read=False,
        test_read=False,
        sports_read=False,
        started_at=utc_now(),
    )
    try:
        launch_background_tmux(
            experiment_id="s18_s1_actionability",
            argv=env_command,
            cwd=ROOT,
            tmux_session=session,
            startup_log_path=OUTPUT / "master.log",
        )
    except Exception as error:
        update_aggregate_status(
            execution_state="FAILED_NO_RETRY",
            scientific_state="FAILED",
            status_code="S18_1_TMUX_START_FAILED_NO_RETRY",
            process_alive=False,
            error=str(error),
        )
        raise
    deadline = time.time() + 30
    while time.time() < deadline:
        status = load_json(STATUS)
        if status.get("workload_pid", 0) > 0 and status.get("status_code") == "S18_1_FORMAL_RUNNING":
            print(json.dumps({"tmux_session": session, "status": str(STATUS.relative_to(ROOT)), "gpu_assignment": auth["gpu_assignment"]}))
            return 0
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode != 0:
            break
        time.sleep(1)
    raise RuntimeError("S18-1 background master failed startup handshake")


def master() -> int:
    config, auth = load_contracts()
    update_aggregate_status(
        stage="four_unit_parallel_execution",
        execution_state="RUNNING_SCIENTIFIC",
        scientific_state="RUNNING",
        status_code="S18_1_FORMAL_RUNNING",
        process_alive=True,
        workload_pid=os.getpid(),
    )
    processes: dict[str, dict[str, Any]] = {}
    for label, physical_gpu in auth["gpu_assignment"].items():
        domain, fold = label.split(":", 1)
        log = unit_dir(domain, fold).parent / f"{unit_key(domain, fold)}.launcher.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w", encoding="utf-8")
        command = [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "unit",
            "--domain",
            domain,
            "--fold",
            fold,
            "--physical-gpu",
            str(physical_gpu),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes[label] = {
            "process": process,
            "handle": handle,
            "pid": process.pid,
            "started": time.time(),
            "gpu": physical_gpu,
            "log": str(log.relative_to(ROOT)),
        }
    timeout_seconds = auth["launch_contract"]["unit_hard_timeout_seconds"]
    while True:
        finished = 0
        states = {}
        for label, record in processes.items():
            process = record["process"]
            return_code = process.poll()
            if return_code is not None:
                finished += 1
                state = "COMPLETED" if return_code == 0 else "FAILED_NO_RETRY"
            else:
                state = "RUNNING"
                if time.time() - record["started"] > timeout_seconds:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    state = "HARD_TIMEOUT_NO_RETRY"
            unit_status = None
            domain, fold = label.split(":", 1)
            if unit_status_path(domain, fold).is_file():
                unit_status = load_json(unit_status_path(domain, fold))
            states[label] = {
                "state": state,
                "pid": record["pid"],
                "gpu": record["gpu"],
                "log": record["log"],
                "unit_status": unit_status,
            }
        update_aggregate_status(
            stage="four_unit_parallel_execution",
            execution_state="RUNNING_SCIENTIFIC" if finished < 4 else "FINALIZING",
            scientific_state="RUNNING",
            status_code="S18_1_FORMAL_RUNNING" if finished < 4 else "S18_1_FINALIZING",
            process_alive=True,
            workload_pid=os.getpid(),
            progress={"current": finished, "total": 4, "unit": "domain_fold"},
            units=states,
        )
        if finished == 4:
            break
        time.sleep(auth["launch_contract"]["heartbeat_seconds"])
    for record in processes.values():
        record["handle"].close()
    failures = [label for label, record in processes.items() if record["process"].returncode != 0]
    if failures:
        append_jsonl(
            LEDGER,
            {"event": "scientific_attempt_failed_no_retry", "at": utc_now(), "failed_units": failures},
        )
        update_aggregate_status(
            stage="terminal_failure",
            execution_state="FAILED_NO_RETRY",
            scientific_state="FAILED",
            status_code="S18_1_UNIT_FAILURE_NO_RETRY",
            process_alive=False,
            workload_pid=0,
            failed_units=failures,
            result_selection_eligible=False,
        )
        return 1
    summary = aggregate_results(config)
    atomic_json(OUTPUT / "summary.json", summary)
    write_report(summary)
    append_jsonl(
        LEDGER,
        {
            "event": "scientific_attempt_completed",
            "at": utc_now(),
            "decision": summary["decision"],
            "summary_sha256": sha256(OUTPUT / "summary.json"),
        },
    )
    update_aggregate_status(
        stage="scientific_complete",
        execution_state="SCIENTIFIC_COMPLETED",
        scientific_state="COMPLETED",
        status_code=summary["decision"],
        process_alive=False,
        workload_pid=0,
        progress={"current": 4, "total": 4, "unit": "domain_fold"},
        summary_path=str((OUTPUT / "summary.json").relative_to(ROOT)),
        summary_sha256=sha256(OUTPUT / "summary.json"),
        report_path=str(REPORT.relative_to(ROOT)),
        result_selection_eligible=True,
        automatic_s18_2=False,
        next_action="Await researcher review; S18-2 is not automatically authorized.",
    )
    launch_occupancy(auth)
    return 0


def launch_occupancy(auth: dict[str, Any]) -> None:
    occupancy = auth["postrun_occupancy"]
    source = unit_dir("Beauty", "I0") / "parent_epoch10.pt"
    if not source.is_file():
        return
    command = [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={occupancy['physical_gpu']}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={ROOT}",
        str(PYTHON),
        str(Path(__file__).resolve()),
        "occupancy",
        "--physical-gpu",
        str(occupancy["physical_gpu"]),
    ]
    try:
        launch_background_tmux(
            experiment_id="s18_s1_postrun_guard",
            argv=command,
            cwd=ROOT,
            tmux_session=occupancy["tmux_session"],
            startup_log_path=OUTPUT / "postrun_guard.log",
        )
        launch_return_code = 0
    except Exception:
        launch_return_code = 1
    update_aggregate_status(
        postrun_occupancy={
            "launch_return_code": launch_return_code,
            "tmux_session": occupancy["tmux_session"],
            "status_path": occupancy["status"],
            "physical_gpu": occupancy["physical_gpu"],
            "result_selection_eligible": False,
        }
    )


def occupancy(physical_gpu: int) -> int:
    config, auth = load_contracts()
    occupancy_config = auth["postrun_occupancy"]
    status_path = ROOT / occupancy_config["status"]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        config["backbone"]["snapshot"], local_files_only=True
    )
    args = gram_args(config, "Beauty", "I0")
    args.tokenizer = tokenizer
    dataset = MultiTaskDatasetGRAM(
        args,
        dataset_name_from_manifest("Beauty", "I0"),
        "train",
        None,
        tokenizer,
        phase=0,
        regenerate=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["parent_training"]["rec_batch_size"],
        shuffle=False,
        collate_fn=CollatorGRAM(tokenizer=tokenizer, args=args, mode="train"),
        num_workers=0,
    )
    model = load_parent(config, unit_dir("Beauty", "I0") / "parent_epoch10.pt", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    iteration = 0
    while True:
        iteration += 1
        model.train()
        for batch_index, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                input_ids=batch["item_text_ids"].to(device),
                attention_mask=batch["item_text_masks"].to(device),
                labels=batch["target_ids"].to(device),
                return_dict=False,
            )[0]
            loss.backward()
            optimizer.step()
            if batch_index % 50 == 0:
                atomic_json(
                    status_path,
                    {
                        "schema_version": "phase18.status.v1",
                        "experiment_id": "s18_s1_postrun_guard",
                        "execution_state": "RUNNING_OCCUPANCY_REPEAT",
                        "scientific_state": "COMPLETED",
                        "scientific_source": "s18_s1_actionability/run-0001",
                        "physical_gpu": physical_gpu,
                        "pid": os.getpid(),
                        "process_alive": True,
                        "repeat_iteration": iteration,
                        "batch": batch_index,
                        "batches": len(loader),
                        "result_selection_eligible": False,
                        "repeat_metrics_ignored": True,
                        "affects_scientific_result": False,
                        "heartbeat_at": utc_now(),
                        "updated_at": utc_now(),
                    },
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("takeover-guard", "smoke", "launch", "master", "unit", "occupancy")
    )
    parser.add_argument("--domain", choices=("Toys", "Beauty"))
    parser.add_argument("--fold", choices=("I-1", "I0"))
    parser.add_argument("--physical-gpu", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "takeover-guard":
        return takeover_guard()
    if args.action == "smoke":
        if args.physical_gpu is None:
            raise ValueError("smoke requires --physical-gpu")
        return smoke(args.physical_gpu)
    if args.action == "launch":
        return launch()
    if args.action == "master":
        return master()
    if args.action == "unit":
        if args.domain is None or args.fold is None or args.physical_gpu is None:
            raise ValueError("unit requires --domain --fold --physical-gpu")
        return run_unit(args.domain, args.fold, args.physical_gpu)
    if args.action == "occupancy":
        if args.physical_gpu is None:
            raise ValueError("occupancy requires --physical-gpu")
        return occupancy(args.physical_gpu)
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
