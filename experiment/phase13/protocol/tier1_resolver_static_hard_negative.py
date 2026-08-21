"""Static warm-only hard-negative screen for Phase-13 Tier-1 resolver.

The frozen epoch-12 P0 recipe is first reproduced through a zero-hard-negative
control arm.  Experimental arms add 8/16/32 nearest warm items per target to
the contrastive denominator.  Cold items are never used as training targets or
negatives.  Only Toys validation development evidence is read.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_resolve as rr
import tier1_resolver_checkpoint_trajectory as trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_hard_negative_counts(values: Iterable[int]) -> list[int]:
    counts = [int(value) for value in values]
    if not counts or counts[0] != 0:
        raise ValueError("hard-negative counts must start with the zero control")
    if counts != sorted(set(counts)) or any(value < 0 for value in counts):
        raise ValueError("hard-negative counts must be unique and increasing")
    if len(counts) < 2:
        raise ValueError("at least one hard-negative arm is required")
    return counts


def set_arm_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_static_warm_negative_lookup(
    train_targets: torch.Tensor,
    embeddings: torch.Tensor,
    warm_indices: torch.Tensor,
    max_count: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, dict]:
    """Return catalog-indexed nearest-neighbour rows using warm items only."""
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    target_ids = torch.unique(train_targets.cpu(), sorted=True)
    warm_indices = torch.unique(warm_indices.cpu(), sorted=True)
    warm_set = set(warm_indices.tolist())
    nonwarm_targets = [index for index in target_ids.tolist() if index not in warm_set]
    if nonwarm_targets:
        raise ValueError(f"training targets outside warm catalog: {nonwarm_targets[:5]}")
    if max_count >= len(warm_indices):
        raise ValueError("hard-negative count must be smaller than warm catalog")

    lookup = torch.full((len(embeddings), max_count), -1, dtype=torch.long)
    warm_vectors = embeddings[warm_indices].to(device)
    warm_position = {catalog_index: position for position, catalog_index in enumerate(warm_indices.tolist())}
    for start in range(0, len(target_ids), chunk_size):
        target_chunk = target_ids[start:start + chunk_size]
        scores = embeddings[target_chunk].to(device) @ warm_vectors.T
        rows = torch.arange(len(target_chunk), device=device)
        own_positions = torch.tensor(
            [warm_position[index] for index in target_chunk.tolist()],
            device=device,
            dtype=torch.long,
        )
        scores[rows, own_positions] = -torch.inf
        nearest_positions = torch.topk(scores, k=max_count, dim=1).indices.cpu()
        lookup[target_chunk] = warm_indices[nearest_positions]

    selected = lookup[target_ids]
    self_negative_count = int((selected == target_ids[:, None]).sum())
    cold_negative_count = sum(
        int(index not in warm_set) for index in selected.flatten().tolist()
    )
    if self_negative_count or cold_negative_count or int((selected < 0).sum()):
        raise AssertionError("invalid static hard-negative lookup")
    audit = {
        "n_unique_train_targets": len(target_ids),
        "n_warm_catalog_items": len(warm_indices),
        "max_hard_negative_count": max_count,
        "self_negative_count": self_negative_count,
        "cold_negative_count": cold_negative_count,
        "all_training_targets_warm": not nonwarm_targets,
        "all_hard_negatives_warm": cold_negative_count == 0,
    }
    return lookup, audit


def hard_negative_augmented_loss(
    user_vec: torch.Tensor,
    target_vec: torch.Tensor,
    target_ids: torch.Tensor,
    hard_negative_vec: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    inbatch_logits = user_vec @ target_vec.T / temperature
    positives = target_ids[:, None].eq(target_ids[None, :])
    positive_logits = inbatch_logits.masked_fill(~positives, -torch.inf)
    denominator_logits = inbatch_logits
    if hard_negative_vec is not None and hard_negative_vec.shape[1] > 0:
        hard_logits = torch.einsum("bd,bhd->bh", user_vec, hard_negative_vec) / temperature
        denominator_logits = torch.cat([inbatch_logits, hard_logits], dim=1)
    return -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(denominator_logits, dim=1)
    ).mean()


def paired_bootstrap_interval(
    current: list[float],
    baseline: list[float],
    resamples: int,
    seed: int,
    confidence: float,
) -> dict:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    a = np.asarray(current, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    delta = a - b
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        batch = min(250, remaining)
        indices = rng.integers(0, len(delta), size=(batch, len(delta)))
        samples.append(delta[indices].mean(axis=1))
        remaining -= batch
    means = np.concatenate(samples)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {
        "difference": float(delta.mean()),
        "confidence": confidence,
        "ci": [float(low), float(high)],
        "resamples": resamples,
        "seed": seed,
        "interpretation": "positive" if low > 0 else "negative" if high < 0 else "inconclusive",
    }


def train_arm(
    context: dict,
    training: dict,
    hard_negative_lookup: torch.Tensor,
    hard_negative_count: int,
    epochs: int,
    seed: int,
    device: torch.device,
    status_path: Path | None,
) -> tuple[rr.ResidualUserProjector, list[dict]]:
    set_arm_seed(seed)
    model = rr.ResidualUserProjector(
        context["train_x"].shape[1], int(training["hidden_dim"]), float(training["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["lr"]), weight_decay=float(training["weight_decay"])
    )
    generator = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    batch_size = int(training["batch_size"])
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(context["train_x"]), generator=generator)
        total_loss = 0.0
        total_n = 0
        for start in range(0, len(order), batch_size):
            batch_ids = order[start:start + batch_size]
            x = context["train_x"][batch_ids].to(device)
            y_idx = context["train_y"][batch_ids].to(device)
            target_vec = context["embeddings"][y_idx].to(device)
            hard_vec = None
            if hard_negative_count:
                negative_ids = hard_negative_lookup[context["train_y"][batch_ids], :hard_negative_count]
                hard_vec = context["embeddings"][negative_ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            user_vec = model(x)
            if hard_negative_count:
                loss = hard_negative_augmented_loss(
                    user_vec, target_vec, y_idx, hard_vec, float(training["temperature"])
                )
            else:
                # Keep the control numerically identical to the frozen P0 implementation.
                loss = rr.multi_positive_inbatch_loss(
                    user_vec, target_vec, y_idx, float(training["temperature"])
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_ids)
            total_n += len(batch_ids)
        record = {
            "epoch": epoch,
            "loss": total_loss / max(total_n, 1),
            "residual_scale": float(model.residual_scale.detach().cpu()),
        }
        history.append(record)
        print(
            f"[train] hn={hard_negative_count} epoch={epoch}/{epochs} "
            f"loss={record['loss']:.6f} scale={record['residual_scale']:.4f}",
            flush=True,
        )
        trajectory.update_status(
            status_path,
            stage="training",
            current_arm_hard_negatives=hard_negative_count,
            current_epoch=epoch,
            total_epochs=epochs,
            last_loss=record["loss"],
        )
    model.eval()
    return model, history


def main() -> None:
    args = parse_args()
    started = time.time()
    root = Path(__file__).resolve().parents[3]
    config_path = Path(args.frozen_config).resolve()
    source_config = trajectory.load_json(config_path)
    if any(source_config.get(key) is not False for key in ("test_read", "beauty_read", "sports_read")):
        raise ValueError("frozen config must keep test/Beauty/Sports reads disabled")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in allowed]
    if unexpected:
        raise FileExistsError(f"refusing to overwrite scientific artifacts: {unexpected}")
    status_path = Path(args.status_path).resolve() if args.status_path else None

    training = source_config["training"]
    counts = validate_hard_negative_counts(training["hard_negative_counts"])
    epochs = int(training["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.smoke:
        counts = [0, counts[1]]
        epochs = 1
    seed = int(training["seed"])
    set_arm_seed(seed)
    context, input_hashes = trajectory.build_catalog_context(source_config, root)
    if context["train_report"]["cold_target_count"] != 0:
        raise ValueError("cold training targets are forbidden")
    if args.smoke:
        context["train_x"] = context["train_x"][:512]
        context["train_y"] = context["train_y"][:512]

    device = torch.device(args.device)
    warm_indices = torch.tensor(
        sorted(context["item_to_idx"][item] for item in context["warm_items"]),
        dtype=torch.long,
    )
    trajectory.update_status(status_path, stage="static_warm_negative_mining")
    hard_negative_lookup, negative_audit = build_static_warm_negative_lookup(
        context["train_y"],
        context["embeddings"],
        warm_indices,
        max(counts),
        device,
        int(training["mining_chunk_size"]),
    )
    config_record = {
        **source_config,
        "source_config_path": str(config_path),
        "source_config_sha256": rr.sha256_file(config_path),
        "input_sha256": input_hashes,
        "device": args.device,
        "smoke": args.smoke,
        "effective_hard_negative_counts": counts,
        "effective_epochs": epochs,
        "code_sha256": rr.sha256_file(Path(__file__).resolve()),
        "negative_pool_audit": negative_audit,
    }
    trajectory.atomic_json(output_dir / "config.json", config_record)

    arm_results: list[dict] = []
    observations_by_count: dict[int, dict[str, list[float]]] = {}
    control_ok = True
    for hard_negative_count in counts:
        trajectory.update_status(
            status_path,
            stage="training",
            current_arm_hard_negatives=hard_negative_count,
            completed_arms=[row["hard_negative_count"] for row in arm_results],
        )
        model, history = train_arm(
            context,
            training,
            hard_negative_lookup,
            hard_negative_count,
            epochs,
            seed,
            device,
            status_path,
        )
        trajectory.update_status(
            status_path,
            stage="arm_evaluation",
            current_arm_hard_negatives=hard_negative_count,
        )
        checkpoint, observations, mismatches, digest = trajectory.evaluate_checkpoint(
            model,
            context,
            device,
            int(source_config["evaluation"]["global_retrieve_k"]),
            eval_limit=128 if args.smoke else 0,
        )
        arm = {
            "hard_negative_count": hard_negative_count,
            "epochs": epochs,
            "training_history": history,
            **checkpoint,
        }
        trajectory.atomic_json(output_dir / f"arm_hn_{hard_negative_count:03d}.json", arm)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hard_negative_count": hard_negative_count,
                "epochs": epochs,
                "dim": context["embeddings"].shape[1],
                "hidden_dim": int(training["hidden_dim"]),
                "dropout": float(training["dropout"]),
                "seed": seed,
                "resolver_top50_sha256": digest,
            },
            output_dir / f"resolver_hn_{hard_negative_count:03d}.pt",
        )
        arm_results.append(arm)
        observations_by_count[hard_negative_count] = observations
        print(
            f"[arm] hn={hard_negative_count} cold_r50="
            f"{checkpoint['resolver_cold_rank']['recall']['@50']:.6f} "
            f"eligible_r3={checkpoint['eligible_cold_rank']['recall']['@3']:.6f} "
            f"warm_r50={checkpoint['metrics']['resolver']['warm']['hit@50']:.6f} "
            f"p0_top50_mismatch={mismatches}",
            flush=True,
        )
        if hard_negative_count == 0 and not args.smoke:
            reproduction = source_config["baseline_reproduction"]
            control_ok = (
                mismatches == int(reproduction["expected_exact_top50_mismatches"])
                and checkpoint["resolver_cold_rank"]["events_top50"]
                == int(reproduction["expected_cold_hit50_events"])
            )
            if not control_ok:
                print("[stop] zero-hard-negative control failed exact P0 reproduction", flush=True)
                break

    comparisons: dict[str, dict] = {}
    winner: dict | None = None
    verdict = "SMOKE_COMPLETED" if args.smoke else "CONTROL_REPRODUCTION_FAILED_STOP"
    if control_ok and not args.smoke and len(arm_results) == len(counts):
        control = arm_results[0]
        control_observations = observations_by_count[0]
        experimental_count = len(counts) - 1
        familywise_alpha = float(source_config["evaluation"]["familywise_alpha"])
        adjusted_confidence = 1.0 - familywise_alpha / experimental_count
        resamples = int(source_config["evaluation"]["bootstrap_resamples"])
        bootstrap_seed = int(source_config["evaluation"]["bootstrap_seed"])
        for arm_index, arm in enumerate(arm_results[1:], 1):
            count = arm["hard_negative_count"]
            observations = observations_by_count[count]
            comparisons[str(count)] = {
                "paired_bootstrap_95": {
                    key: trajectory.paired_bootstrap(
                        values,
                        control_observations[key],
                        resamples,
                        bootstrap_seed + count * 100 + metric_index,
                    )
                    for metric_index, (key, values) in enumerate(observations.items())
                },
                "primary_bonferroni_adjusted": paired_bootstrap_interval(
                    observations["resolver_cold_hit50"],
                    control_observations["resolver_cold_hit50"],
                    resamples,
                    bootstrap_seed + 900000 + arm_index,
                    adjusted_confidence,
                ),
            }

        winner = max(
            arm_results[1:],
            key=lambda arm: (
                arm["resolver_cold_rank"]["recall"]["@50"],
                arm["eligible_cold_rank"]["recall"]["@3"],
                -arm["hard_negative_count"],
            ),
        )
        winner_count = winner["hard_negative_count"]
        winner_primary = comparisons[str(winner_count)]["primary_bonferroni_adjusted"]
        control_warm = control["metrics"]["resolver"]["warm"]["hit@50"]
        winner_warm = winner["metrics"]["resolver"]["warm"]["hit@50"]
        warm_relative_change = winner_warm / control_warm - 1.0
        warm_guard = warm_relative_change >= -float(
            source_config["gate"]["maximum_relative_warm_recall50_degradation"]
        )
        positive_gate = winner_primary["ci"][0] > 0.0
        toys_gate = (
            winner["resolver_cold_rank"]["recall"]["@50"]
            >= float(source_config["gate"]["tier1_toys_absolute_cold_recall50_target"])
            and warm_guard
        )
        winner = {
            "hard_negative_count": winner_count,
            "cold_recall_at_50": winner["resolver_cold_rank"]["recall"]["@50"],
            "eligible_cold_recall_at_3": winner["eligible_cold_rank"]["recall"]["@3"],
            "warm_recall_at_50": winner_warm,
            "warm_recall50_relative_change_vs_control": warm_relative_change,
            "bonferroni_adjusted_primary_ci_positive": positive_gate,
            "warm_guard_passed": warm_guard,
            "tier1_toys_30pct_gate_passed": toys_gate,
        }
        verdict = (
            "PASS_T1_2_TOYS_GATE_REQUIRES_REPLICATION"
            if positive_gate and toys_gate
            else "PASS_T1_2_SIGNAL_REQUIRES_REPLICATION"
            if positive_gate and warm_guard
            else "FAIL_STOP_STATIC_HARD_NEGATIVE"
        )

    summary = {
        "experiment_id": source_config["experiment_id"],
        "status": "completed",
        "verdict": verdict,
        "evidence_role": source_config["evidence_role"],
        "test_read": False,
        "beauty_read": False,
        "sports_read": False,
        "control_reproduction_passed": control_ok if not args.smoke else None,
        "train_targets_all_warm": context["train_report"]["cold_target_count"] == 0,
        "hard_negatives_all_warm": negative_audit["all_hard_negatives_warm"],
        "negative_pool_audit": negative_audit,
        "train_report": context["train_report"],
        "arms": arm_results,
        "paired_comparisons_vs_hn0": comparisons,
        "winner": winner,
        "automatic_next_stage": False,
        "runtime_seconds": time.time() - started,
    }
    trajectory.atomic_json(output_dir / "summary.json", summary)
    trajectory.update_status(
        status_path,
        stage="finished",
        completed_arms=[row["hard_negative_count"] for row in arm_results],
        scientific_verdict=verdict,
    )
    print(f"[result] verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()
