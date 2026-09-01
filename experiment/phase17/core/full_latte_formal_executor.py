"""Formal checkpoint-selection training for Stage17 FP1/FP2.

The executor only uses the rolling train-prefix examples and the frozen
train-prefix internal-dev cohort.  It never materializes the external D0
target.  External one-shot evaluation is a separate family-level action after
all matched checkpoints are frozen.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

from .status_writer import atomic_json, utc_now


SEED = 2023
GRAM_ARMS = {
    "G0_GRAM_B0_FRESH",
    "G1_GRAM_PSID_FULL",
    "G2_GRAM_LATTE_FULL",
}
NATIVE_ARMS = {"N0_NATIVE_PSID", "N1_NATIVE_LATTE"}


def _move(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: (value.to(device) if hasattr(value, "to") else value)
        for key, value in batch.items()
    }


def _save_checkpoint(torch, path: Path, model, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(
        {"model_state": model.state_dict(), "metadata": metadata},
        temporary,
    )
    os.replace(temporary, path)


def _ndcg_at_10(rank: int | None) -> float:
    if rank is None or rank > 10:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _heartbeat(
    callback: Callable[[str, dict[str, Any]], None] | None,
    stage: str,
    current: int,
    total: int,
    unit: str,
    **extra: Any,
) -> None:
    if callback is not None:
        callback(
            stage,
            {"current": current, "total": total, "unit": unit, **extra},
        )


def _native_eval(model, components, examples: Sequence, device) -> dict[str, float]:
    import torch

    from genrec.evaluator import Evaluator
    from .full_latte_native_backend import collate_native_eval_batch

    evaluator = Evaluator(components.config, components.tokenizer)
    metric_rows: dict[str, list[Any]] = {"ndcg@10": [], "recall@10": []}
    valid_rows = 0
    total_rows = 0
    model.eval()
    with torch.no_grad():
        # Evaluation batch one is the profiled, prediction-equivalent resource
        # setting.  It preserves beam/ranking semantics and only trades time.
        for example in examples:
            batch = _move(
                collate_native_eval_batch(components, (example,)), device
            )
            predictions = model.generate(batch, n_return_sequences=50)
            metrics = evaluator.calculate_metrics(predictions, batch["labels"])
            for key in metric_rows:
                metric_rows[key].append(metrics[key])
            width = int(predictions.shape[-1])
            valid_rows += int(width == int(components.tokenizer.n_digit))
            total_rows += 1
    return {
        "ndcg@10": float(torch.cat(metric_rows["ndcg@10"]).mean().item()),
        "recall@10": float(torch.cat(metric_rows["recall@10"]).mean().item()),
        "complete_width_rate": valid_rows / total_rows,
    }


def train_native_formal(
    root: Path,
    arm_id: str,
    result_dir: Path,
    *,
    heartbeat: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if arm_id not in NATIVE_ARMS:
        raise ValueError(f"not a native formal arm: {arm_id}")
    import torch
    from transformers.optimization import get_cosine_schedule_with_warmup

    from .full_latte_native_backend import (
        build_official_native_components,
        collate_native_train_batch,
        create_fresh_official_native_model,
        load_native_examples,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for native formal training")
    device = torch.device("cuda:0")
    components = build_official_native_components(
        root, arm_id, device="cuda:0", num_beams=50
    )
    # The frozen primary config reports top-50; internal checkpoint selection
    # uses official beam-50 and never opens the external target.
    components.config["topk"] = [10, 50]
    components.config["metrics"] = ["ndcg", "recall"]
    train_examples, internal_dev = load_native_examples(root)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = create_fresh_official_native_model(components, seed=SEED).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=0.05
    )
    batch_size = 256
    max_epochs = 150
    batches_per_epoch = math.ceil(len(train_examples) / batch_size)
    total_steps = max_epochs * batches_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10000,
        num_training_steps=total_steps,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / "best_checkpoint.pt"
    curve_path = result_dir / "learning_curve.json"
    curve: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch = 0
    started = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, max_epochs + 1):
        order = list(range(len(train_examples)))
        random.Random(SEED + epoch - 1).shuffle(order)
        model.train()
        losses: list[float] = []
        for batch_number, start in enumerate(
            range(0, len(order), batch_size), start=1
        ):
            rows = [train_examples[index] for index in order[start : start + batch_size]]
            # The same epoch/batch seed gives N0/N1 matched stochastic order;
            # N1's official collator still samples a fresh latent per exposure.
            torch.manual_seed(SEED + epoch * 100000 + batch_number)
            batch = _move(collate_native_train_batch(components, rows), device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite native loss at epoch={epoch} batch={batch_number}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))
            if batch_number == 1 or batch_number % 20 == 0:
                _heartbeat(
                    heartbeat,
                    "training",
                    epoch - 1,
                    max_epochs,
                    "epoch",
                    batch=batch_number,
                    batches=batches_per_epoch,
                )

        metrics = _native_eval(model, components, internal_dev, device)
        improved = best_epoch == 0 or metrics["ndcg@10"] > best_metric
        if improved:
            best_metric = metrics["ndcg@10"]
            best_epoch = epoch
            _save_checkpoint(
                torch,
                checkpoint,
                model,
                {
                    "arm_id": arm_id,
                    "epoch": epoch,
                    "internal_dev_ndcg@10": best_metric,
                    "external_target_materialized": False,
                    "saved_at": utc_now(),
                },
            )
        curve.append(
            {
                "epoch": epoch,
                "train_loss": fmean(losses),
                "internal_dev": metrics,
                "improved": improved,
                "best_epoch": best_epoch,
                "external_target_materialized": False,
            }
        )
        atomic_json(
            curve_path,
            {
                "arm_id": arm_id,
                "epochs": curve,
                "external_target_materialized": False,
            },
        )
        _heartbeat(heartbeat, "internal_dev_complete", epoch, max_epochs, "epoch")
        if epoch - best_epoch >= 50:
            break

    if not checkpoint.is_file():
        raise FileNotFoundError("native best checkpoint was not created")
    torch.cuda.synchronize()
    return {
        "arm_id": arm_id,
        "family": "native",
        "backend": f"pinned_official_{components.model_class.__name__}",
        "epochs_completed": len(curve),
        "best_epoch": best_epoch,
        "best_internal_dev_ndcg@10": best_metric,
        "checkpoint_path": str(checkpoint.relative_to(root)),
        "learning_curve_path": str(curve_path.relative_to(root)),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "wall_seconds": time.monotonic() - started,
        "train_examples": len(train_examples),
        "internal_dev_examples": len(internal_dev),
        "train_batch_size": batch_size,
        "internal_eval_batch_size": 1,
        "internal_eval_beam": 50,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }


def _gram_eval(
    *,
    model,
    tokenizer,
    collator,
    arm_id: str,
    catalog,
    examples: Sequence,
    item_paths,
    trie,
    max_length: int,
    device,
    heartbeat: Callable[[str, dict[str, Any]], None] | None,
    epoch: int,
) -> dict[str, float]:
    import torch

    from .full_latte_gram_backend import (
        aggregate_generated_paths,
        render_gram_example,
    )

    ndcg: list[float] = []
    hit: list[float] = []
    valid: list[float] = []
    model.eval()
    rng = random.Random(SEED)
    with torch.no_grad():
        for index, example in enumerate(examples, start=1):
            row = render_gram_example(
                example, arm_id=arm_id, catalog=catalog, rng=rng
            ).as_collator_row()
            batch = _move(collator([row]), device)
            generated = model.generate(
                input_ids=batch["item_text_ids"],
                attention_mask=batch["item_text_masks"],
                history_item_ids=batch["history_item_ids"],
                history_item_mask=batch["history_item_mask"],
                max_length=max_length,
                prefix_allowed_tokens_fn=trie.prefix_allowed_tokens_fn(),
                num_beams=50,
                num_return_sequences=50,
                output_scores=True,
                return_dict_in_generate=True,
                length_penalty=1.0,
                use_cache=False,
            )
            sequences = generated["sequences"].detach().cpu().tolist()
            scores = generated["sequences_scores"].detach().cpu().tolist()
            trimmed = []
            for sequence in sequences:
                try:
                    eos = sequence.index(tokenizer.eos_token_id, 1)
                    sequence = sequence[: eos + 1]
                except ValueError:
                    pass
                trimmed.append(sequence)
            ranked = aggregate_generated_paths(
                trimmed,
                scores,
                item_paths=item_paths,
                method="agg_max",
                top_k=50,
            )
            ranking = [item for item, _score, _count in ranked]
            try:
                rank = ranking.index(example.target) + 1
            except ValueError:
                rank = None
            ndcg.append(_ndcg_at_10(rank))
            hit.append(float(rank is not None and rank <= 10))
            valid.append(float(bool(ranking)))
            if index == 1 or index % 50 == 0:
                _heartbeat(
                    heartbeat,
                    "internal_dev",
                    epoch,
                    50,
                    "epoch",
                    dev_example=index,
                    dev_total=len(examples),
                )
    return {
        "ndcg@10": fmean(ndcg),
        "hit@10": fmean(hit),
        "nonempty_ranking_rate": fmean(valid),
    }


def train_gram_formal(
    root: Path,
    arm_id: str,
    result_dir: Path,
    *,
    heartbeat: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if arm_id not in GRAM_ARMS:
        raise ValueError(f"not a GRAM formal arm: {arm_id}")
    import torch
    from transformers.optimization import get_cosine_schedule_with_warmup

    from .full_latte_gram_backend import (
        PrefixTree,
        build_gram_collator,
        create_fresh_gram_model,
        encoded_candidate_paths,
        load_fullport_examples,
        load_gram_catalog,
        render_gram_example,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for GRAM formal training")
    device = torch.device("cuda:0")
    catalog = load_gram_catalog(root, arm_id)
    train_examples, internal_dev = load_fullport_examples(root)
    tokenizer, collator = build_gram_collator(root, arm_id)
    item_paths = encoded_candidate_paths(tokenizer, arm_id, catalog)
    flat_paths = [path for paths in item_paths.values() for path in paths]
    trie = PrefixTree(flat_paths)
    max_length = max(len(path) for path in flat_paths)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = create_fresh_gram_model(root, arm_id, tokenizer, seed=SEED).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01
    )
    # attempt_004 uses the largest safe per-device microbatch observed on each
    # shared card.  Formal attempt_003 showed that batch 16 left only 0.73 GiB
    # free on GPU1; batch 8's longest-input profile is safe there.  The
    # effective optimizer batch remains exactly 128 for every arm.
    microbatch = {
        "G0_GRAM_B0_FRESH": 16,
        "G1_GRAM_PSID_FULL": 8,
        "G2_GRAM_LATTE_FULL": 8,
    }[arm_id]
    accumulation = 128 // microbatch
    max_epochs = 50
    batches_per_epoch = math.ceil(len(train_examples) / microbatch)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_optimizer_steps = max_epochs * optimizer_steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_optimizer_steps * 0.05)),
        num_training_steps=total_optimizer_steps,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / "best_checkpoint.pt"
    curve_path = result_dir / "learning_curve.json"
    curve: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch = 0
    stale_evaluations = 0
    started = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, max_epochs + 1):
        order = list(range(len(train_examples)))
        random.Random(SEED + epoch - 1).shuffle(order)
        render_rng = random.Random(SEED + epoch - 1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for batch_number, start in enumerate(
            range(0, len(order), microbatch), start=1
        ):
            examples = [train_examples[index] for index in order[start : start + microbatch]]
            rows = [
                render_gram_example(
                    example, arm_id=arm_id, catalog=catalog, rng=render_rng
                ).as_collator_row()
                for example in examples
            ]
            batch = _move(collator(rows), device)
            outputs = model(
                input_ids=batch["item_text_ids"],
                attention_mask=batch["item_text_masks"],
                history_item_ids=batch["history_item_ids"],
                history_item_mask=batch["history_item_mask"],
                target_item_ids=batch["target_item_ids"],
                labels=batch["target_ids"],
            )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite GRAM loss at epoch={epoch} batch={batch_number}"
                )
            (loss / accumulation).backward()
            losses.append(float(loss.detach().cpu().item()))
            final_batch = batch_number == batches_per_epoch
            if batch_number % accumulation == 0 or final_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_number == 1 or batch_number % 100 == 0:
                _heartbeat(
                    heartbeat,
                    "training",
                    epoch - 1,
                    max_epochs,
                    "epoch",
                    batch=batch_number,
                    batches=batches_per_epoch,
                )

        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": fmean(losses),
            "external_target_materialized": False,
        }
        if epoch % 5 == 0:
            metrics = _gram_eval(
                model=model,
                tokenizer=tokenizer,
                collator=collator,
                arm_id=arm_id,
                catalog=catalog,
                examples=internal_dev,
                item_paths=item_paths,
                trie=trie,
                max_length=max_length,
                device=device,
                heartbeat=heartbeat,
                epoch=epoch,
            )
            improved = (
                best_epoch == 0 or metrics["ndcg@10"] > best_metric + 0.0001
            )
            if improved:
                best_metric = metrics["ndcg@10"]
                best_epoch = epoch
                stale_evaluations = 0
                _save_checkpoint(
                    torch,
                    checkpoint,
                    model,
                    {
                        "arm_id": arm_id,
                        "epoch": epoch,
                        "internal_dev_ndcg@10": best_metric,
                        "external_target_materialized": False,
                        "generation_kv_cache": False,
                        "saved_at": utc_now(),
                    },
                )
            else:
                stale_evaluations += 1
            record.update(
                internal_dev=metrics,
                improved=improved,
                best_epoch=best_epoch,
                stale_evaluations=stale_evaluations,
            )
        curve.append(record)
        atomic_json(
            curve_path,
            {
                "arm_id": arm_id,
                "epochs": curve,
                "external_target_materialized": False,
            },
        )
        _heartbeat(heartbeat, "epoch_complete", epoch, max_epochs, "epoch")
        if epoch >= 20 and stale_evaluations >= 3:
            break

    if not checkpoint.is_file():
        raise FileNotFoundError("GRAM best checkpoint was not created")
    torch.cuda.synchronize()
    return {
        "arm_id": arm_id,
        "family": "gram",
        "backend": "project_GRAM_FiD",
        "epochs_completed": len(curve),
        "best_epoch": best_epoch,
        "best_internal_dev_ndcg@10": best_metric,
        "checkpoint_path": str(checkpoint.relative_to(root)),
        "learning_curve_path": str(curve_path.relative_to(root)),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "wall_seconds": time.monotonic() - started,
        "train_examples": len(train_examples),
        "internal_dev_examples": len(internal_dev),
        "train_microbatch": microbatch,
        "gradient_accumulation": accumulation,
        "effective_batch": microbatch * accumulation,
        "internal_eval_batch_size": 1,
        "internal_eval_beam": 50,
        "generation_kv_cache": False,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }


def train_formal_arm(
    root: Path,
    arm_id: str,
    result_dir: Path,
    *,
    heartbeat: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if arm_id in GRAM_ARMS:
        return train_gram_formal(
            root, arm_id, result_dir, heartbeat=heartbeat
        )
    if arm_id in NATIVE_ARMS:
        return train_native_formal(
            root, arm_id, result_dir, heartbeat=heartbeat
        )
    raise ValueError(f"unknown formal arm: {arm_id}")
