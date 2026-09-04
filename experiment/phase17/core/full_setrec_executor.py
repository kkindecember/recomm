"""Execution helpers for FP3 Full SETRec resource and formal workloads."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from .full_latte_gram_backend import (
    ITEM_PROMPT_MAX_LENGTH,
    MAX_HISTORY_ITEMS,
    load_fullport_examples,
    load_gram_catalog,
)
from .full_setrec_backend import (
    SETREC_ARMS,
    FullSetRecModel,
    SetRecBatch,
    build_full_setrec_model,
    collate_setrec_examples,
    load_setrec_catalog,
)
from .full_setrec_contracts import full_set_recovery
from .status_writer import atomic_json, utc_now


PROFILE_MICROBATCH_BY_ARM = {
    "S0_SETREC_ORDERED_CONTROL": 128,
    "S1R_SETREC_REPO_PARITY": 128,
    "S1P_SETREC_PAPER_FAITHFUL": 128,
    "S2_GRAM_SETREC_PAPER_FULL": 16,
}
PROFILE_EVAL_BATCH_BY_ARM = {
    "S0_SETREC_ORDERED_CONTROL": 128,
    "S1R_SETREC_REPO_PARITY": 128,
    "S1P_SETREC_PAPER_FAITHFUL": 128,
    "S2_GRAM_SETREC_PAPER_FULL": 8,
}
GLOBAL_BATCH_SIZE = 512
BETA_GRID = tuple(round(index / 10, 1) for index in range(1, 11))


@dataclass(frozen=True)
class SetRecFormalSpec:
    seed: int = 2023
    maximum_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    global_batch_size: int = 512
    warmup_steps: int = 100
    eval_steps: int = 200
    save_steps: int = 200
    early_stopping_patience: int = 10
    alpha: float = 0.7
    precision: str = "fp16"
    top_k: int = 10


def to_device(batch: SetRecBatch, device: torch.device) -> SetRecBatch:
    return replace(
        batch,
        history_item_ids=batch.history_item_ids.to(device),
        history_item_mask=batch.history_item_mask.to(device),
        target_item_indices=batch.target_item_indices.to(device),
        gram_input_ids=(
            None if batch.gram_input_ids is None else batch.gram_input_ids.to(device)
        ),
        gram_attention_mask=(
            None
            if batch.gram_attention_mask is None
            else batch.gram_attention_mask.to(device)
        ),
    )


def encode_fid_passages(
    passage_rows: Sequence[Sequence[str]],
    *,
    tokenizer: Any,
    max_passages: int = MAX_HISTORY_ITEMS + 1,
    max_length: int = ITEM_PROMPT_MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the GRAM coarse prompt plus per-item passages without legacy APIs."""

    if not passage_rows or max_passages <= 0 or max_length <= 0:
        raise ValueError("invalid FiD passage batch")
    flat: list[str] = []
    for passages in passage_rows:
        values = list(passages[:max_passages])
        values.extend([""] * (max_passages - len(values)))
        flat.extend(values)
    encoded = tokenizer(
        flat,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_size = len(passage_rows)
    return (
        encoded["input_ids"].reshape(batch_size, max_passages, max_length),
        encoded["attention_mask"]
        .reshape(batch_size, max_passages, max_length)
        .bool(),
    )


class SetRecBatchBuilder:
    def __init__(self, root: Path, arm_id: str, tokenizer: Any) -> None:
        if arm_id not in SETREC_ARMS:
            raise ValueError(f"unknown SETRec arm: {arm_id}")
        self.root = root.resolve()
        self.arm_id = arm_id
        self.tokenizer = tokenizer
        self.catalog = load_setrec_catalog(self.root)
        self.gram_catalog = None
        self.gram_item_input_ids = None
        self.gram_item_attention_mask = None
        self.gram_empty_input_ids = None
        self.gram_empty_attention_mask = None
        self.gram_item_to_row: dict[str, int] = {}
        if arm_id == "S2_GRAM_SETREC_PAPER_FULL":
            self.gram_catalog = load_gram_catalog(
                self.root, "G0_GRAM_B0_FRESH"
            )
            if set(self.gram_catalog.ordered_items) != set(self.catalog.ordered_items):
                raise RuntimeError("SETRec and GRAM catalogs contain different items")
            self.gram_item_to_row = {
                item: index
                for index, item in enumerate(self.gram_catalog.ordered_items)
            }
            item_encoded = tokenizer(
                [
                    self.gram_catalog.item_passages[item]
                    for item in self.gram_catalog.ordered_items
                ],
                padding="max_length",
                truncation=True,
                max_length=ITEM_PROMPT_MAX_LENGTH,
                return_tensors="pt",
            )
            self.gram_item_input_ids = item_encoded["input_ids"]
            self.gram_item_attention_mask = item_encoded["attention_mask"].bool()
            empty_encoded = tokenizer(
                [""],
                padding="max_length",
                truncation=True,
                max_length=ITEM_PROMPT_MAX_LENGTH,
                return_tensors="pt",
            )
            self.gram_empty_input_ids = empty_encoded["input_ids"][0]
            self.gram_empty_attention_mask = empty_encoded["attention_mask"][0].bool()

    def __call__(self, examples: Sequence[Any]) -> SetRecBatch:
        gram_batch = None
        if self.gram_catalog is not None:
            reversed_histories = [
                tuple(reversed(example.history[-MAX_HISTORY_ITEMS:]))
                for example in examples
            ]
            coarse_texts = [
                "What would user purchase after "
                + " ; ".join(
                    self.gram_catalog.identity_text[item] for item in history
                )
                + " ?"
                for history in reversed_histories
            ]
            coarse = self.tokenizer(
                coarse_texts,
                padding="max_length",
                truncation=True,
                max_length=ITEM_PROMPT_MAX_LENGTH,
                return_tensors="pt",
            )
            if self.gram_empty_input_ids is None or self.gram_empty_attention_mask is None:
                raise AssertionError("S2 empty-passage encoding is missing")
            ids = self.gram_empty_input_ids.view(1, 1, -1).expand(
                len(examples), MAX_HISTORY_ITEMS + 1, -1
            ).clone()
            masks = self.gram_empty_attention_mask.view(1, 1, -1).expand_as(
                ids
            ).clone()
            ids[:, 0] = coarse["input_ids"]
            masks[:, 0] = coarse["attention_mask"].bool()
            if self.gram_item_input_ids is None or self.gram_item_attention_mask is None:
                raise AssertionError("S2 fixed item-passage cache is missing")
            for row, history in enumerate(reversed_histories):
                item_rows = torch.tensor(
                    [self.gram_item_to_row[item] for item in history],
                    dtype=torch.long,
                )
                ids[row, 1 : len(history) + 1] = self.gram_item_input_ids[item_rows]
                masks[row, 1 : len(history) + 1] = self.gram_item_attention_mask[
                    item_rows
                ]
            gram_batch = {"item_text_ids": ids, "item_text_masks": masks}
        return collate_setrec_examples(
            examples,
            item_to_index=self.catalog.item_to_index,
            max_history_items=MAX_HISTORY_ITEMS,
            gram_batch=gram_batch,
        )


def _combined_item_scores(
    per_dimension_scores: torch.Tensor, beta: float
) -> torch.Tensor:
    weights = torch.full(
        (per_dimension_scores.shape[0],),
        beta,
        dtype=per_dimension_scores.dtype,
        device=per_dimension_scores.device,
    )
    weights[0] = 1.0 - beta
    return (per_dimension_scores * weights[:, None, None]).sum(dim=0)


def _ranking_diagnostics(
    per_dimension_scores: torch.Tensor,
    targets: torch.Tensor,
    histories: torch.Tensor,
    *,
    beta: float,
) -> dict[str, Any]:
    scores = _combined_item_scores(per_dimension_scores, beta)
    for row in range(scores.shape[0]):
        seen = histories[row][histories[row].ne(0)].unique() - 1
        scores[row, seen] = -torch.inf
    top = scores.topk(k=min(10, scores.shape[1]), dim=1).indices
    matches = top.eq(targets[:, None])
    hit = matches.any(dim=1)
    gains = torch.zeros(scores.shape[0], device=scores.device)
    if bool(hit.any()):
        ranks = matches[hit].float().argmax(dim=1) + 1
        gains[hit] = 1.0 / torch.log2(ranks.float() + 1.0)
    target_scores = scores.gather(1, targets[:, None])
    target_ranks = scores.gt(target_scores).sum(dim=1) + 1
    recovery = full_set_recovery(per_dimension_scores, targets)
    return {
        "hit@10": float(hit.float().mean()),
        "ndcg@10": float(gains.mean()),
        "mean_grounding_target_rank": float(target_ranks.float().mean()),
        "full_set_recovery_rate": float(recovery.float().mean()),
        "valid_item_rate": float(torch.isfinite(top.float()).all()),
        "top10": top.detach().cpu(),
    }


@torch.no_grad()
def evaluate_setrec_internal_dev(
    model: FullSetRecModel,
    builder: SetRecBatchBuilder,
    examples: Sequence[Any],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if not examples or batch_size <= 0:
        raise ValueError("internal-dev evaluation requires examples and batch size")
    model.eval()
    totals = {
        beta: {"hit": 0.0, "ndcg": 0.0, "rank": 0.0} for beta in BETA_GRID
    }
    recovery = 0.0
    loss_sum = generation_sum = reconstruction_sum = 0.0
    query_norm_min = math.inf
    query_norm_max = 0.0
    reconstruction_finite = True
    processed = 0
    for start in range(0, len(examples), batch_size):
        rows = examples[start : start + batch_size]
        batch = to_device(builder(rows), device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch, beta=0.7)
        if output.loss is None:
            raise RuntimeError("internal-dev SETRec output lacks loss")
        if not bool(torch.isfinite(output.grounding.per_dimension_scores).all()):
            raise FloatingPointError("internal-dev grounding scores are non-finite")
        count = len(rows)
        processed += count
        loss_sum += float(output.loss) * count
        generation_sum += float(output.generation_loss) * count
        reconstruction_sum += float(output.reconstruction_loss) * count
        recovery += float(
            full_set_recovery(
                output.grounding.per_dimension_scores,
                batch.target_item_indices,
            ).sum()
        )
        norms = output.query_outputs.float().norm(dim=-1)
        query_norm_min = min(query_norm_min, float(norms.min()))
        query_norm_max = max(query_norm_max, float(norms.max()))
        reconstruction_finite &= bool(
            torch.isfinite(output.semantic_reconstruction).all()
        )
        for beta in BETA_GRID:
            diagnostic = _ranking_diagnostics(
                output.grounding.per_dimension_scores,
                batch.target_item_indices,
                batch.history_item_ids,
                beta=beta,
            )
            totals[beta]["hit"] += diagnostic["hit@10"] * count
            totals[beta]["ndcg"] += diagnostic["ndcg@10"] * count
            totals[beta]["rank"] += diagnostic["mean_grounding_target_rank"] * count
    if processed != len(examples):
        raise AssertionError("internal-dev row count drifted")
    beta_metrics = {
        f"{beta:.1f}": {
            "hit@10": values["hit"] / processed,
            "ndcg@10": values["ndcg"] / processed,
            "mean_grounding_target_rank": values["rank"] / processed,
        }
        for beta, values in totals.items()
    }
    selected_beta = max(
        BETA_GRID,
        key=lambda beta: (
            beta_metrics[f"{beta:.1f}"]["ndcg@10"],
            beta_metrics[f"{beta:.1f}"]["hit@10"],
            -beta,
        ),
    )
    selected = beta_metrics[f"{selected_beta:.1f}"]
    return {
        "n": processed,
        "loss": loss_sum / processed,
        "generation_loss": generation_sum / processed,
        "reconstruction_loss": reconstruction_sum / processed,
        "beta_metrics": beta_metrics,
        "selected_beta": selected_beta,
        "selected_ndcg@10": selected["ndcg@10"],
        "selected_hit@10": selected["hit@10"],
        "selected_mean_grounding_target_rank": selected[
            "mean_grounding_target_rank"
        ],
        "full_set_recovery_rate": recovery / processed,
        "valid_item_rate": 1.0,
        "query_norm_min": query_norm_min,
        "query_norm_max": query_norm_max,
        "query_norms_finite_nonzero": (
            math.isfinite(query_norm_min)
            and math.isfinite(query_norm_max)
            and query_norm_min > 0
        ),
        "semantic_reconstruction_finite": reconstruction_finite,
        "full_catalog_grounding": True,
    }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _cosine_schedule_lambda(
    step: int, *, warmup_steps: int, total_steps: int
) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_setrec_formal(
    root: Path,
    arm_id: str,
    *,
    output_dir: Path,
    device: torch.device,
    spec: SetRecFormalSpec = SetRecFormalSpec(),
    heartbeat: Any | None = None,
) -> dict[str, Any]:
    """Train one immutable FP3 arm and freeze its internal-dev-selected checkpoint."""

    if arm_id not in SETREC_ARMS or device.type != "cuda":
        raise ValueError("formal SETRec training requires a known arm and CUDA")
    if spec.global_batch_size != GLOBAL_BATCH_SIZE:
        raise ValueError("FP3 effective global batch must remain 512")
    train_microbatch = PROFILE_MICROBATCH_BY_ARM[arm_id]
    eval_batch_size = PROFILE_EVAL_BATCH_BY_ARM[arm_id]
    if spec.global_batch_size % train_microbatch:
        raise ValueError("global batch is not divisible by profiled microbatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_checkpoint.pt"
    latest_checkpoint = output_dir / "latest_checkpoint.pt"
    learning_curve_path = output_dir / "learning_curve.json"
    train_examples, dev_examples = load_fullport_examples(root)
    model, tokenizer = build_full_setrec_model(root, arm_id, seed=spec.seed)
    builder = SetRecBatchBuilder(root, arm_id, tokenizer)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_examples) / spec.global_batch_size)
    total_steps = steps_per_epoch * spec.maximum_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_schedule_lambda(
            step, warmup_steps=spec.warmup_steps, total_steps=total_steps
        ),
    )
    scaler = torch.amp.GradScaler("cuda")
    global_step = 0
    best_metric = -math.inf
    best_eval_index = 0
    no_improvement = 0
    learning_curve: list[dict[str, Any]] = []
    stopped_early = False
    last_eval_step = 0
    started = time.monotonic()
    for epoch in range(1, spec.maximum_epochs + 1):
        generator = torch.Generator().manual_seed(spec.seed + epoch)
        indices = torch.randperm(len(train_examples), generator=generator).tolist()
        model.train()
        epoch_loss_sum = 0.0
        epoch_rows = 0
        for group_start in range(0, len(indices), spec.global_batch_size):
            group = indices[group_start : group_start + spec.global_batch_size]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(group), train_microbatch):
                micro_indices = group[micro_start : micro_start + train_microbatch]
                rows = [train_examples[index] for index in micro_indices]
                batch = to_device(builder(rows), device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(batch, beta=0.7)
                    if output.loss is None:
                        raise RuntimeError("formal SETRec output lacks loss")
                    weighted_loss = output.loss * (len(rows) / len(group))
                if not bool(torch.isfinite(weighted_loss)):
                    raise FloatingPointError(
                        f"non-finite FP3 loss for {arm_id} at step {global_step + 1}"
                    )
                scaler.scale(weighted_loss).backward()
                epoch_loss_sum += float(output.loss.detach()) * len(rows)
                epoch_rows += len(rows)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if (
                heartbeat is not None
                and global_step % 5 == 0
                and global_step % spec.eval_steps != 0
            ):
                heartbeat(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "train_loss_running_epoch": epoch_loss_sum / epoch_rows,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "best_internal_dev_ndcg@10": (
                            None if best_metric == -math.inf else best_metric
                        ),
                        "no_improvement_evaluations": no_improvement,
                    }
                )
            if global_step % spec.eval_steps == 0:
                evaluation = evaluate_setrec_internal_dev(
                    model,
                    builder,
                    dev_examples,
                    batch_size=eval_batch_size,
                    device=device,
                )
                record = {
                    "eval_index": len(learning_curve) + 1,
                    "global_step": global_step,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_loss_running_epoch": epoch_loss_sum / epoch_rows,
                    "internal_dev": evaluation,
                }
                learning_curve.append(record)
                atomic_json(
                    learning_curve_path,
                    {
                        "schema_version": "phase17.s17_fp3_learning_curve.v1",
                        "arm_id": arm_id,
                        "updated_at": utc_now(),
                        "records": learning_curve,
                    },
                )
                checkpoint = {
                    "schema_version": "phase17.s17_fp3_checkpoint.v1",
                    "arm_id": arm_id,
                    "seed": spec.seed,
                    "global_step": global_step,
                    "epoch": epoch,
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "internal_dev": evaluation,
                    "spec": asdict(spec),
                    "external_target_materialized": False,
                    "test_read": False,
                    "sports_read": False,
                    "d1_read": False,
                    "d2_read": False,
                }
                _atomic_torch_save(latest_checkpoint, checkpoint)
                metric = float(evaluation["selected_ndcg@10"])
                if metric > best_metric:
                    best_metric = metric
                    best_eval_index = len(learning_curve)
                    no_improvement = 0
                    _atomic_torch_save(best_checkpoint, checkpoint)
                else:
                    no_improvement += 1
                last_eval_step = global_step
                if heartbeat is not None:
                    heartbeat(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "total_steps": total_steps,
                            "best_internal_dev_ndcg@10": best_metric,
                            "no_improvement_evaluations": no_improvement,
                            "latest_internal_dev": evaluation,
                        }
                    )
                model.train()
                if no_improvement >= spec.early_stopping_patience:
                    stopped_early = True
                    break
        if stopped_early:
            break
    if global_step != last_eval_step:
        evaluation = evaluate_setrec_internal_dev(
            model,
            builder,
            dev_examples,
            batch_size=eval_batch_size,
            device=device,
        )
        record = {
            "eval_index": len(learning_curve) + 1,
            "global_step": global_step,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss_running_epoch": epoch_loss_sum / epoch_rows,
            "internal_dev": evaluation,
            "final_noninterval_evaluation": True,
        }
        learning_curve.append(record)
        atomic_json(
            learning_curve_path,
            {
                "schema_version": "phase17.s17_fp3_learning_curve.v1",
                "arm_id": arm_id,
                "updated_at": utc_now(),
                "records": learning_curve,
            },
        )
        checkpoint = {
            "schema_version": "phase17.s17_fp3_checkpoint.v1",
            "arm_id": arm_id,
            "seed": spec.seed,
            "global_step": global_step,
            "epoch": epoch,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "internal_dev": evaluation,
            "spec": asdict(spec),
            "external_target_materialized": False,
            "test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
        }
        _atomic_torch_save(latest_checkpoint, checkpoint)
        metric = float(evaluation["selected_ndcg@10"])
        if metric > best_metric:
            best_metric = metric
            best_eval_index = len(learning_curve)
            _atomic_torch_save(best_checkpoint, checkpoint)
    if not best_checkpoint.is_file() or not learning_curve:
        raise RuntimeError("FP3 formal training did not freeze a best checkpoint")
    best_record = learning_curve[best_eval_index - 1]
    return {
        "arm_id": arm_id,
        "seed": spec.seed,
        "epochs_completed": epoch,
        "global_steps_completed": global_step,
        "total_planned_steps": total_steps,
        "stopped_early": stopped_early,
        "train_examples": len(train_examples),
        "internal_dev_examples": len(dev_examples),
        "train_microbatch": train_microbatch,
        "gradient_accumulation": spec.global_batch_size // train_microbatch,
        "effective_global_batch": spec.global_batch_size,
        "eval_batch_size": eval_batch_size,
        "best_eval_index": best_eval_index,
        "best_global_step": best_record["global_step"],
        "best_epoch": best_record["epoch"],
        "best_internal_dev": best_record["internal_dev"],
        "best_checkpoint_path": str(best_checkpoint),
        "latest_checkpoint_path": str(latest_checkpoint),
        "learning_curve_path": str(learning_curve_path),
        "wall_seconds": time.monotonic() - started,
        "external_target_materialized": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }


@torch.no_grad()
def fp16_forward_parity(
    model: FullSetRecModel,
    batch: SetRecBatch,
) -> dict[str, Any]:
    model.eval()
    fp32 = model(batch, beta=0.7)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fp16 = model(batch, beta=0.7)
    if fp32.loss is None or fp16.loss is None:
        raise RuntimeError("profile parity requires labeled losses")
    fp32_diag = _ranking_diagnostics(
        fp32.grounding.per_dimension_scores,
        batch.target_item_indices,
        batch.history_item_ids,
        beta=0.7,
    )
    fp16_diag = _ranking_diagnostics(
        fp16.grounding.per_dimension_scores,
        batch.target_item_indices,
        batch.history_item_ids,
        beta=0.7,
    )
    overlap = []
    for left, right in zip(fp32_diag["top10"], fp16_diag["top10"]):
        overlap.append(len(set(left.tolist()) & set(right.tolist())) / len(left))
    relative_loss = abs(float(fp16.loss) - float(fp32.loss)) / max(
        abs(float(fp32.loss)), 1e-12
    )
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            fp32.loss,
            fp16.loss,
            fp32.grounding.item_scores,
            fp16.grounding.item_scores,
        )
    )
    return {
        "fp32_loss": float(fp32.loss),
        "fp16_loss": float(fp16.loss),
        "relative_loss_difference": relative_loss,
        "mean_top10_set_overlap": sum(overlap) / len(overlap),
        "all_finite": finite,
        "pass": finite and relative_loss <= 0.05 and sum(overlap) / len(overlap) >= 0.8,
    }


def run_setrec_resource_profile(
    root: Path,
    arm_id: str,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if arm_id not in SETREC_ARMS:
        raise ValueError(f"unknown SETRec arm: {arm_id}")
    if device.type != "cuda":
        raise ValueError("FP3 resource profile requires CUDA")
    started = time.monotonic()
    train, internal_dev = load_fullport_examples(root)
    train_batch_size = PROFILE_MICROBATCH_BY_ARM[arm_id]
    eval_batch_size = PROFILE_EVAL_BATCH_BY_ARM[arm_id]
    longest = sorted(train, key=lambda row: len(row.history), reverse=True)
    model, tokenizer = build_full_setrec_model(root, arm_id)
    builder = SetRecBatchBuilder(root, arm_id, tokenizer)
    parity_batch = to_device(builder(longest[:1]), device)
    model.to(device)
    parity = fp16_forward_parity(model, parity_batch)
    if not parity["pass"]:
        raise RuntimeError(f"FP16 parity failed for {arm_id}: {parity}")

    train_batch = to_device(builder(longest[:train_batch_size]), device)
    eval_batch = to_device(builder(internal_dev[:eval_batch_size]), device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_started = time.monotonic()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(train_batch, beta=0.7)
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise FloatingPointError(f"non-finite FP3 profile loss for {arm_id}")
    scaler.scale(output.loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    train_seconds = time.monotonic() - train_started

    model.eval()
    eval_started = time.monotonic()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        evaluated = model(eval_batch, beta=0.7)
    torch.cuda.synchronize()
    eval_seconds = time.monotonic() - eval_started
    diagnostic = _ranking_diagnostics(
        evaluated.grounding.per_dimension_scores,
        eval_batch.target_item_indices,
        eval_batch.history_item_ids,
        beta=0.7,
    )
    query_norms = evaluated.query_outputs.float().norm(dim=-1)
    peak_allocated = torch.cuda.max_memory_allocated() / 1048576
    peak_reserved = torch.cuda.max_memory_reserved() / 1048576
    result = {
        "arm_id": arm_id,
        "precision": "fp16_autocast_with_grad_scaler",
        "train_microbatch": train_batch_size,
        "gradient_accumulation": GLOBAL_BATCH_SIZE // train_batch_size,
        "effective_global_batch": GLOBAL_BATCH_SIZE,
        "eval_batch_size": eval_batch_size,
        "train_step_seconds": train_seconds,
        "eval_step_seconds": eval_seconds,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_loss": float(output.loss.detach()),
        "generation_loss": float(output.generation_loss.detach()),
        "reconstruction_loss": float(output.reconstruction_loss.detach()),
        "fp16_parity": parity,
        "mechanism": {
            key: value
            for key, value in diagnostic.items()
            if key != "top10"
        }
        | {
            "query_norm_min": float(query_norms.min()),
            "query_norm_max": float(query_norms.max()),
            "query_norms_finite_nonzero": bool(
                torch.isfinite(query_norms).all() and (query_norms > 0).all()
            ),
            "semantic_reconstruction_finite": bool(
                torch.isfinite(evaluated.semantic_reconstruction).all()
            ),
            "per_dimension_shape": list(
                evaluated.grounding.per_dimension_scores.shape
            ),
            "full_catalog_grounding": (
                evaluated.grounding.item_scores.shape[1]
                == len(builder.catalog.ordered_items)
            ),
        },
        "gram_fid_shape": (
            None
            if eval_batch.gram_input_ids is None
            else list(eval_batch.gram_input_ids.shape)
        ),
        "wall_seconds": time.monotonic() - started,
        "external_target_materialized": False,
        "effect_metrics_computed": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }
    del optimizer, model, output, evaluated, train_batch, eval_batch, parity_batch
    torch.cuda.empty_cache()
    return result


def cpu_preflight_setrec(root: Path) -> dict[str, Any]:
    root = root.resolve()
    train, dev = load_fullport_examples(root)
    catalog = load_setrec_catalog(root)
    parameter_counts: dict[str, int] = {}
    fid_shape = None
    for arm_id in SETREC_ARMS:
        model, tokenizer = build_full_setrec_model(root, arm_id, catalog=catalog)
        parameter_counts[arm_id] = sum(p.numel() for p in model.parameters())
        builder = SetRecBatchBuilder(root, arm_id, tokenizer)
        batch = builder([max(train, key=lambda row: len(row.history))])
        if arm_id == "S2_GRAM_SETREC_PAPER_FULL":
            fid_shape = list(batch.gram_input_ids.shape)
        del model
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError("FP3 four-arm parameter capacity differs")
    return {
        "state": "PASS_S17_FP3_CPU_PREFLIGHT",
        "arms": list(SETREC_ARMS),
        "catalog_items": len(catalog.ordered_items),
        "rolling_train_examples": len(train),
        "internal_dev_examples": len(dev),
        "parameter_counts": parameter_counts,
        "s2_fid_shape": fid_shape,
        "profile_microbatch_by_arm": PROFILE_MICROBATCH_BY_ARM,
        "profile_eval_batch_by_arm": PROFILE_EVAL_BATCH_BY_ARM,
        "effective_global_batch": GLOBAL_BATCH_SIZE,
        "external_target_materialized": False,
        "effect_metrics_computed": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }
