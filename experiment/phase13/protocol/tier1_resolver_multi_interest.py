"""Frozen-projector multi-interest screen for Phase-13 Tier-1 resolver.

The HN=0 epoch-12 P0 projector and catalog are frozen.  The only experimental
factor is replacing one recency-weighted validation-history vector with two or
four deterministic semantic-interest centroids.  Toys validation is
development evidence only; test, Beauty, Sports, retraining, and route fusion
are forbidden.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_resolve as rr
import tier1_resolver_checkpoint_trajectory as trajectory
import tier1_resolver_static_hard_negative as hardneg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_interest_counts(values: Iterable[int]) -> list[int]:
    counts = [int(value) for value in values]
    if not counts or counts[0] != 1:
        raise ValueError("interest counts must start with the single-vector control")
    if counts != sorted(set(counts)) or any(value <= 0 for value in counts):
        raise ValueError("interest counts must be positive, unique, and increasing")
    if len(counts) < 2:
        raise ValueError("at least one multi-interest arm is required")
    return counts


def build_validation_histories(
    sequences: list[tuple[str, list[str]]],
    item_to_idx: dict[str, int],
    max_history: int,
) -> tuple[dict[str, list[int]], dict]:
    """Build histories from positions strictly before validation target (-2)."""
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    histories: dict[str, list[int]] = {}
    lengths: list[int] = []
    for uid, items in sequences:
        if len(items) < 3:
            raise ValueError(f"sequence too short for validation protocol: {uid}")
        history_items = items[max(0, len(items) - 2 - max_history):-2]
        if not history_items:
            raise ValueError(f"empty validation history: {uid}")
        histories[uid] = [item_to_idx[item] for item in history_items]
        lengths.append(len(history_items))
    return histories, {
        "n_users": len(histories),
        "minimum_history_length": min(lengths),
        "maximum_history_length": max(lengths),
        "validation_target_position": -2,
        "held_out_test_position": -1,
        "history_end_exclusive": -2,
    }


def deterministic_interest_vectors(
    item_indices: Iterable[int],
    embeddings: torch.Tensor,
    requested_count: int,
    decay: float,
    iterations: int,
) -> torch.Tensor:
    """Return deterministic recency-weighted spherical history centroids."""
    indices = list(item_indices)
    if not indices:
        raise ValueError("history cannot be empty")
    if requested_count <= 0 or iterations <= 0:
        raise ValueError("requested_count and iterations must be positive")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must lie in (0, 1]")
    if requested_count == 1:
        # Keep the control numerically identical to the frozen P0 construction.
        return rr.recency_weighted_history(indices, embeddings, decay).unsqueeze(0)

    history = embeddings[torch.tensor(indices, dtype=torch.long)]
    history = F.normalize(history.float(), dim=1)
    effective_count = min(requested_count, len(indices))

    selected = [len(indices) - 1]
    while len(selected) < effective_count:
        similarity = history @ history[selected].T
        nearest_selected = similarity.max(dim=1).values
        candidates = [position for position in range(len(indices)) if position not in selected]
        # Prefer the more recent position only when semantic distances tie exactly.
        selected.append(min(candidates, key=lambda pos: (float(nearest_selected[pos]), -pos)))
    centroids = history[selected].clone()

    ages = torch.arange(len(indices) - 1, -1, -1, dtype=history.dtype)
    weights = decay ** ages
    for _ in range(iterations):
        assignments = (history @ centroids.T).argmax(dim=1)
        updated: list[torch.Tensor] = []
        for cluster in range(effective_count):
            mask = assignments.eq(cluster)
            if not bool(mask.any()):
                updated.append(centroids[cluster])
                continue
            pooled = (history[mask] * weights[mask, None]).sum(dim=0)
            pooled = pooled / weights[mask].sum().clamp_min(1e-12)
            updated.append(F.normalize(pooled, dim=0))
        centroids = torch.stack(updated)
    return centroids


def max_interest_scores(projected: torch.Tensor, catalog: torch.Tensor) -> torch.Tensor:
    """Score each catalog item by maximum cosine over interest vectors."""
    if projected.ndim != 2 or catalog.ndim != 2 or projected.shape[1] != catalog.shape[1]:
        raise ValueError("projected interests and catalog must be aligned matrices")
    return (projected @ catalog.T).max(dim=0).values


def evaluate_interest_arm(
    model: rr.ResidualUserProjector,
    context: dict,
    histories: dict[str, list[int]],
    interest_count: int,
    decay: float,
    iterations: int,
    device: torch.device,
    retrieve_k: int,
    batch_size: int,
    eval_limit: int = 0,
) -> tuple[dict, dict[str, list[float]], int, str]:
    model.eval()
    item_ids = context["item_ids"]
    catalog = context["embeddings"].to(device)
    cold_items = context["cold_items"]
    eval_uids = context["eval_uids"][:eval_limit or None]
    result_rows = {
        model_name: {slice_name: [] for slice_name in ("all", "warm", "cold")}
        for model_name in ("resolver", "portfolio@2")
    }
    raw_cold_ranks: list[int | None] = []
    eligible_cold_ranks: list[int | None] = []
    top50_mismatches = 0
    digest = hashlib.sha256()
    observations = {
        "resolver_cold_hit50": [],
        "eligible_cold_hit3": [],
        "resolver_warm_hit50": [],
        "portfolio_all_ndcg10": [],
        "portfolio_cold_hit50": [],
        "portfolio_warm_ndcg10": [],
    }
    effective_counts: Counter[int] = Counter()

    with torch.no_grad():
        for offset in range(0, len(eval_uids), batch_size):
            batch_uids = eval_uids[offset:offset + batch_size]
            per_user: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            for uid in batch_uids:
                if interest_count == 1:
                    vectors = context["validation"][uid][0].unsqueeze(0)
                else:
                    vectors = deterministic_interest_vectors(
                        histories[uid], context["embeddings"], interest_count, decay, iterations
                    )
                effective_counts[len(vectors)] += 1
                padded = torch.zeros(interest_count, vectors.shape[1], dtype=vectors.dtype)
                padded[:len(vectors)] = vectors
                mask = torch.zeros(interest_count, dtype=torch.bool)
                mask[:len(vectors)] = True
                per_user.append(padded)
                masks.append(mask)

            base = torch.stack(per_user).to(device)
            valid = torch.stack(masks).to(device)
            if interest_count == 1:
                # Preserve the frozen P0 CUDA matmul shape and operation exactly.
                # Near-tied top-50 boundaries can change under a 3-D einsum.
                scores = model(base[:, 0]) @ catalog.T
            else:
                projected = model(base.flatten(0, 1)).reshape(
                    len(batch_uids), interest_count, -1
                )
                scores = torch.einsum("bmd,nd->bmn", projected, catalog)
                scores = scores.masked_fill(~valid[:, :, None], -torch.inf).max(dim=1).values

            for local_index, uid in enumerate(batch_uids):
                target = context["validation"][uid][2]
                split = "cold" if target in cold_items else "warm"
                indices = torch.topk(
                    scores[local_index], k=min(retrieve_k, len(item_ids))
                ).indices.tolist()
                resolver_items = [item_ids[index] for index in indices]
                resolver_top50 = resolver_items[:50]
                gram_items = context["gram_items_by_uid"][uid]
                candidates = trajectory.portfolio_candidates(gram_items, resolver_top50, cold_items)
                portfolio = trajectory.portfolio2_ranking(gram_items, resolver_top50, candidates)

                if resolver_top50 != context["baseline"][uid]["resolver_top50"]:
                    top50_mismatches += 1
                digest.update(uid.encode())
                digest.update(b"\0")
                digest.update("\x1f".join(resolver_top50).encode())
                digest.update(b"\n")

                resolver_metrics = rr.ranking_metrics(resolver_top50, target)
                portfolio_metrics = rr.ranking_metrics(portfolio, target)
                for name, metrics in (("resolver", resolver_metrics), ("portfolio@2", portfolio_metrics)):
                    result_rows[name]["all"].append(metrics)
                    result_rows[name][split].append(metrics)

                observations["portfolio_all_ndcg10"].append(portfolio_metrics["ndcg@10"])
                if split == "cold":
                    raw_rank = trajectory.target_rank(resolver_top50, target)
                    eligible_rank = trajectory.target_rank(candidates[:50], target)
                    raw_cold_ranks.append(raw_rank)
                    eligible_cold_ranks.append(eligible_rank)
                    observations["resolver_cold_hit50"].append(resolver_metrics["hit@50"])
                    observations["eligible_cold_hit3"].append(
                        float(eligible_rank is not None and eligible_rank <= 3)
                    )
                    observations["portfolio_cold_hit50"].append(portfolio_metrics["hit@50"])
                else:
                    observations["resolver_warm_hit50"].append(resolver_metrics["hit@50"])
                    observations["portfolio_warm_ndcg10"].append(portfolio_metrics["ndcg@10"])
            print(
                f"[eval] m={interest_count} {min(offset + batch_size, len(eval_uids))}/{len(eval_uids)}",
                flush=True,
            )

    metrics = {
        name: {split: rr.average_metrics(rows) for split, rows in slices.items()}
        for name, slices in result_rows.items()
    }
    summary = {
        "interest_count": interest_count,
        "n_users": len(eval_uids),
        "metrics": metrics,
        "resolver_cold_rank": trajectory.rank_summary(raw_cold_ranks),
        "eligible_cold_rank": trajectory.rank_summary(eligible_cold_ranks),
        "effective_interest_count_histogram": {
            str(key): value for key, value in sorted(effective_counts.items())
        },
        "exact_top50_mismatch_vs_p0": top50_mismatches,
        "resolver_top50_sha256": digest.hexdigest(),
    }
    return summary, observations, top50_mismatches, digest.hexdigest()


def load_frozen_projector(path: Path, embedding_dim: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    if int(checkpoint["dim"]) != embedding_dim:
        raise ValueError("frozen projector and catalog dimensions differ")
    if int(checkpoint.get("hard_negative_count", -1)) != 0:
        raise ValueError("T1-4 requires the HN=0 control projector")
    model = rr.ResidualUserProjector(
        int(checkpoint["dim"]), int(checkpoint["hidden_dim"]), float(checkpoint["dropout"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


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

    counts = validate_interest_counts(source_config["representation"]["interest_counts"])
    if args.smoke:
        counts = counts[:2]
    training = source_config["training"]
    representation = source_config["representation"]
    evaluation = source_config["evaluation"]
    context, input_hashes = trajectory.build_catalog_context(source_config, root)
    sequences = rr.read_sequences(context["paths"]["dataset_dir"] / "user_sequence.txt")
    histories, history_audit = build_validation_histories(
        sequences, context["item_to_idx"], int(training["max_history"])
    )
    if set(context["eval_uids"]) - set(histories):
        raise ValueError("missing validation histories for evaluation users")

    device = torch.device(args.device)
    model, checkpoint = load_frozen_projector(
        context["paths"]["control_projector_checkpoint"],
        context["embeddings"].shape[1],
        device,
    )
    config_record = {
        **source_config,
        "source_config_path": str(config_path),
        "source_config_sha256": rr.sha256_file(config_path),
        "input_sha256": input_hashes,
        "device": args.device,
        "smoke": args.smoke,
        "effective_interest_counts": counts,
        "history_audit": history_audit,
        "frozen_projector_epoch": checkpoint.get("epochs", checkpoint.get("epoch")),
        "frozen_projector_hard_negative_count": checkpoint.get("hard_negative_count"),
        "code_sha256": rr.sha256_file(Path(__file__).resolve()),
    }
    trajectory.atomic_json(output_dir / "config.json", config_record)

    arms: list[dict] = []
    observations_by_count: dict[int, dict[str, list[float]]] = {}
    control_ok = True
    for count in counts:
        trajectory.update_status(
            status_path,
            stage="arm_evaluation",
            current_interest_count=count,
            completed_arms=[row["interest_count"] for row in arms],
        )
        arm, observations, mismatches, _digest = evaluate_interest_arm(
            model,
            context,
            histories,
            count,
            float(training["recency_decay"]),
            int(representation["lloyd_iterations"]),
            device,
            int(evaluation["global_retrieve_k"]),
            int(evaluation["evaluation_batch_size"]),
            # One complete frozen P0 evaluation batch is required to audit exact
            # top-50 reproduction; smaller GEMM shapes are not bitwise equivalent.
            eval_limit=256 if args.smoke else 0,
        )
        trajectory.atomic_json(output_dir / f"arm_m_{count:03d}.json", arm)
        arms.append(arm)
        observations_by_count[count] = observations
        print(
            f"[arm] m={count} cold_r50={arm['resolver_cold_rank']['recall']['@50']:.6f} "
            f"eligible_r3={arm['eligible_cold_rank']['recall']['@3']:.6f} "
            f"warm_r50={arm['metrics']['resolver']['warm']['hit@50']:.6f} "
            f"p0_top50_mismatch={mismatches}",
            flush=True,
        )
        if count == 1 and not args.smoke:
            reproduction = source_config["baseline_reproduction"]
            control_ok = (
                mismatches == int(reproduction["expected_exact_top50_mismatches"])
                and arm["resolver_cold_rank"]["events_top50"]
                == int(reproduction["expected_cold_hit50_events"])
            )
            if not control_ok:
                print("[stop] single-vector control failed exact P0 reproduction", flush=True)
                break

    comparisons: dict[str, dict] = {}
    winner: dict | None = None
    verdict = "SMOKE_COMPLETED" if args.smoke else "CONTROL_REPRODUCTION_FAILED_STOP"
    if control_ok and not args.smoke and len(arms) == len(counts):
        control = arms[0]
        control_observations = observations_by_count[1]
        experimental_count = len(counts) - 1
        adjusted_confidence = 1.0 - float(evaluation["familywise_alpha"]) / experimental_count
        resamples = int(evaluation["bootstrap_resamples"])
        bootstrap_seed = int(evaluation["bootstrap_seed"])
        for arm_index, arm in enumerate(arms[1:], 1):
            count = arm["interest_count"]
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
                "primary_bonferroni_adjusted": hardneg.paired_bootstrap_interval(
                    observations["resolver_cold_hit50"],
                    control_observations["resolver_cold_hit50"],
                    resamples,
                    bootstrap_seed + 900000 + arm_index,
                    adjusted_confidence,
                ),
            }

        best_arm = max(
            arms[1:],
            key=lambda arm: (
                arm["resolver_cold_rank"]["recall"]["@50"],
                arm["eligible_cold_rank"]["recall"]["@3"],
                -arm["interest_count"],
            ),
        )
        winner_count = best_arm["interest_count"]
        winner_primary = comparisons[str(winner_count)]["primary_bonferroni_adjusted"]
        control_warm = control["metrics"]["resolver"]["warm"]["hit@50"]
        winner_warm = best_arm["metrics"]["resolver"]["warm"]["hit@50"]
        warm_relative_change = winner_warm / control_warm - 1.0
        warm_guard = warm_relative_change >= -float(
            source_config["gate"]["maximum_relative_warm_recall50_degradation"]
        )
        positive_gate = winner_primary["ci"][0] > 0.0
        toys_gate = (
            best_arm["resolver_cold_rank"]["recall"]["@50"]
            >= float(source_config["gate"]["tier1_toys_absolute_cold_recall50_target"])
            and warm_guard
        )
        winner = {
            "interest_count": winner_count,
            "cold_recall_at_50": best_arm["resolver_cold_rank"]["recall"]["@50"],
            "eligible_cold_recall_at_3": best_arm["eligible_cold_rank"]["recall"]["@3"],
            "warm_recall_at_50": winner_warm,
            "warm_recall50_relative_change_vs_control": warm_relative_change,
            "bonferroni_adjusted_primary_ci_positive": positive_gate,
            "warm_guard_passed": warm_guard,
            "tier1_toys_30pct_gate_passed": toys_gate,
        }
        verdict = (
            "PASS_T1_4_TOYS_GATE_REQUIRES_REPLICATION"
            if positive_gate and toys_gate
            else "PASS_T1_4_SIGNAL_REQUIRES_REPLICATION"
            if positive_gate and warm_guard
            else "FAIL_STOP_MULTI_INTEREST_TIER1"
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
        "frozen_projector": True,
        "history_audit": history_audit,
        "arms": arms,
        "paired_comparisons_vs_m1": comparisons,
        "winner": winner,
        "automatic_next_stage": False,
        "runtime_seconds": time.time() - started,
    }
    trajectory.atomic_json(output_dir / "summary.json", summary)
    trajectory.update_status(
        status_path,
        stage="finished",
        completed_arms=[row["interest_count"] for row in arms],
        scientific_verdict=verdict,
    )
    print(f"[result] verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()
