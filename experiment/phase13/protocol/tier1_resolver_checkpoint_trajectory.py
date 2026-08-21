"""Single-trajectory resolver convergence audit for Phase-13 Tier-1.

The script preserves the frozen P0 training recipe and evaluates checkpoints
from one deterministic training trajectory.  Toys validation is development
evidence only.  No test, Beauty, Sports, GRAM training, or route fusion is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_resolve as rr


AUDIT_KS = (1, 2, 3, 5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha256_text(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def update_status(status_path: Path | None, **fields: object) -> None:
    if status_path is None or not status_path.exists():
        return
    status = load_json(status_path)
    status.update(fields)
    status["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    atomic_json(status_path, status)


def validate_checkpoint_epochs(values: Iterable[int]) -> list[int]:
    epochs = [int(value) for value in values]
    if not epochs or any(value <= 0 for value in epochs):
        raise ValueError("checkpoint epochs must be positive")
    if epochs != sorted(set(epochs)):
        raise ValueError("checkpoint epochs must be unique and increasing")
    return epochs


def load_baseline_predictions(path: Path) -> dict[str, dict]:
    if "test" in path.name.lower():
        raise ValueError(f"Refusing test predictions: {path}")
    rows: dict[str, dict] = {}
    with path.open() as handle:
        for line_no, raw in enumerate(handle, 1):
            row = json.loads(raw)
            uid = str(row["user_id"])
            if uid in rows:
                raise ValueError(f"duplicate baseline user {uid} at line {line_no}")
            rows[uid] = row
    if not rows:
        raise ValueError(f"No baseline prediction rows in {path}")
    return rows


def portfolio_candidates(
    gram_items: list[str], resolver_items: list[str], cold_items: set[str]
) -> list[str]:
    protected = set(rr.unique_in_order(gram_items)[:7])
    return [
        item
        for item in rr.unique_in_order(resolver_items)
        if item in cold_items and item not in protected
    ]


def portfolio2_ranking(
    gram_items: list[str], resolver_items: list[str], candidates: list[str]
) -> list[str]:
    """Reproduce the frozen P6/B1 portfolio@2 ranking exactly."""
    gram = rr.unique_in_order(gram_items)
    resolver = rr.unique_in_order(resolver_items)
    chosen = rr.unique_in_order(candidates)[:2]
    if len(chosen) < 2:
        return gram
    return rr.unique_in_order([*gram[:8], *chosen, *gram[8:], *resolver])


def target_rank(ranking: list[str], target: str) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def rank_summary(ranks: list[int | None]) -> dict:
    n = len(ranks)
    finite = [rank for rank in ranks if rank is not None]
    return {
        "n": n,
        "events_top50": len(finite),
        "recall": {
            f"@{k}": sum(rank is not None and rank <= k for rank in ranks) / max(n, 1)
            for k in AUDIT_KS
        },
        "mrr_at_50": sum(1.0 / rank for rank in finite if rank <= 50) / max(n, 1),
        "rank_buckets": {
            "rank_1": sum(rank == 1 for rank in ranks),
            "rank_2_3": sum(rank is not None and 2 <= rank <= 3 for rank in ranks),
            "rank_4_10": sum(rank is not None and 4 <= rank <= 10 for rank in ranks),
            "rank_11_50": sum(rank is not None and 11 <= rank <= 50 for rank in ranks),
            "absent_top50": sum(rank is None or rank > 50 for rank in ranks),
        },
    }


def paired_bootstrap(
    current: list[float], baseline: list[float], resamples: int, seed: int
) -> dict:
    a = np.asarray(current, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    delta = a - b
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        batch = min(250, remaining)
        indices = rng.integers(0, len(delta), size=(batch, len(delta)))
        means.append(delta[indices].mean(axis=1))
        remaining -= batch
    samples = np.concatenate(means)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "difference": float(delta.mean()),
        "ci95": [float(low), float(high)],
        "resamples": resamples,
        "seed": seed,
        "interpretation": (
            "positive" if low > 0 else "negative" if high < 0 else "inconclusive"
        ),
    }


def build_catalog_context(
    config: dict, root: Path
) -> tuple[dict, dict]:
    paths = {name: (root / value).resolve() for name, value in config["paths"].items()}
    for name, path in paths.items():
        if name == "dataset_dir":
            if not path.is_dir():
                raise FileNotFoundError(f"{name}: {path}")
            continue
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
        if "test" in path.name.lower():
            raise ValueError(f"Refusing test-like input path: {path}")

    dataset_dir = paths["dataset_dir"]
    required_dataset_files = [
        dataset_dir / "user_sequence.txt",
        dataset_dir / "cold_split_meta" / "cold_items.txt",
        dataset_dir / "cold_split_meta" / "warm_items.txt",
    ]
    for path in required_dataset_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = torch.load(paths["item_embeddings"], map_location="cpu")
    item_ids = list(payload["item_ids"])
    embeddings = F.normalize(payload["embeddings"].float(), dim=1)
    item_to_idx = {item: index for index, item in enumerate(item_ids)}
    if len(item_to_idx) != len(item_ids):
        raise ValueError("embedding payload contains duplicate item IDs")

    item_to_lexical = rr.read_key_value_lines(paths["item_id_file"])
    if set(item_to_lexical) != set(item_ids):
        raise ValueError("embedding and hierarchical-ID catalogs differ")
    decoded_to_item: dict[str, str] = {}
    for item, lexical in item_to_lexical.items():
        decoded = rr.decode_lexical_id(lexical)
        if decoded in decoded_to_item:
            raise ValueError(f"decoded semantic-ID collision for {decoded!r}")
        decoded_to_item[decoded] = item

    gram_rows = rr.parse_gram_predictions(paths["gram_validation_predictions"])
    sequences = rr.read_sequences(dataset_dir / "user_sequence.txt")
    cold_items = rr.read_set(dataset_dir / "cold_split_meta" / "cold_items.txt")
    warm_items = rr.read_set(dataset_dir / "cold_split_meta" / "warm_items.txt")
    if cold_items & warm_items:
        raise ValueError("cold/warm item sets overlap")

    training = config["training"]
    train_x, train_y, train_report = rr.build_training_examples(
        sequences,
        item_to_idx,
        embeddings,
        cold_items,
        int(training["max_history"]),
        float(training["recency_decay"]),
    )
    validation = rr.build_validation_examples(
        sequences,
        item_to_idx,
        embeddings,
        int(training["max_history"]),
        float(training["recency_decay"]),
    )
    eval_uids = [uid for uid, _items in sequences if uid in gram_rows]
    baseline = load_baseline_predictions(paths["baseline_p0_predictions"])
    if set(eval_uids) != set(baseline):
        raise ValueError("baseline P0 users do not match the validation cohort")

    gram_items_by_uid: dict[str, list[str]] = {}
    for uid in eval_uids:
        items = []
        for decoded in gram_rows[uid]["predictions"]:
            item = decoded_to_item.get(decoded)
            if item is None:
                raise KeyError(f"unmapped legal GRAM beam for {uid}: {decoded!r}")
            items.append(item)
        gram_items_by_uid[uid] = rr.unique_in_order(items)

    context = {
        "paths": paths,
        "item_ids": item_ids,
        "embeddings": embeddings,
        "item_to_idx": item_to_idx,
        "cold_items": cold_items,
        "warm_items": warm_items,
        "train_x": train_x,
        "train_y": train_y,
        "train_report": train_report,
        "validation": validation,
        "eval_uids": eval_uids,
        "baseline": baseline,
        "gram_items_by_uid": gram_items_by_uid,
    }
    hashes = {
        str(path): rr.sha256_file(path)
        for path in [*required_dataset_files, *[p for name, p in paths.items() if name != "dataset_dir"]]
    }
    return context, hashes


def evaluate_checkpoint(
    model: rr.ResidualUserProjector,
    context: dict,
    device: torch.device,
    retrieve_k: int,
    eval_limit: int = 0,
) -> tuple[dict, dict[str, list[float]], int, str]:
    model.eval()
    item_ids = context["item_ids"]
    embeddings_device = context["embeddings"].to(device)
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

    with torch.no_grad():
        for offset in range(0, len(eval_uids), 256):
            batch_uids = eval_uids[offset:offset + 256]
            base = torch.stack([context["validation"][uid][0] for uid in batch_uids]).to(device)
            scores = model(base) @ embeddings_device.T
            for local_index, uid in enumerate(batch_uids):
                target = context["validation"][uid][2]
                split = "cold" if target in cold_items else "warm"
                indices = torch.topk(scores[local_index], k=min(retrieve_k, len(item_ids))).indices.tolist()
                resolver_items = [item_ids[index] for index in indices]
                resolver_top50 = resolver_items[:50]
                gram_items = context["gram_items_by_uid"][uid]
                candidates = portfolio_candidates(gram_items, resolver_top50, cold_items)
                portfolio = portfolio2_ranking(gram_items, resolver_top50, candidates)

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
                    raw_rank = target_rank(resolver_top50, target)
                    eligible_rank = target_rank(candidates[:50], target)
                    raw_cold_ranks.append(raw_rank)
                    eligible_cold_ranks.append(eligible_rank)
                    observations["resolver_cold_hit50"].append(resolver_metrics["hit@50"])
                    observations["eligible_cold_hit3"].append(float(eligible_rank is not None and eligible_rank <= 3))
                    observations["portfolio_cold_hit50"].append(portfolio_metrics["hit@50"])
                else:
                    observations["resolver_warm_hit50"].append(resolver_metrics["hit@50"])
                    observations["portfolio_warm_ndcg10"].append(portfolio_metrics["ndcg@10"])
            print(f"[eval] {min(offset + 256, len(eval_uids))}/{len(eval_uids)}", flush=True)

    metrics = {
        name: {
            split: rr.average_metrics(rows)
            for split, rows in slices.items()
        }
        for name, slices in result_rows.items()
    }
    summary = {
        "n_users": len(eval_uids),
        "metrics": metrics,
        "resolver_cold_rank": rank_summary(raw_cold_ranks),
        "eligible_cold_rank": rank_summary(eligible_cold_ranks),
        "exact_top50_mismatch_vs_p0": top50_mismatches,
        "resolver_top50_sha256": digest.hexdigest(),
    }
    return summary, observations, top50_mismatches, digest.hexdigest()


def main() -> None:
    args = parse_args()
    started = time.time()
    root = Path(__file__).resolve().parents[3]
    config_path = Path(args.frozen_config).resolve()
    source_config = load_json(config_path)
    if any(source_config.get(key) is not False for key in ("test_read", "beauty_read", "sports_read")):
        raise ValueError("frozen config must keep test/Beauty/Sports reads disabled")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in allowed]
    if unexpected:
        raise FileExistsError(f"refusing to overwrite scientific artifacts: {unexpected}")

    status_path = Path(args.status_path).resolve() if args.status_path else None
    checkpoints = validate_checkpoint_epochs(source_config["training"]["checkpoint_epochs"])
    if args.smoke:
        checkpoints = [1]
    training = source_config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    context, input_hashes = build_catalog_context(source_config, root)
    if args.smoke:
        context["train_x"] = context["train_x"][:512]
        context["train_y"] = context["train_y"][:512]
    config_record = {
        **source_config,
        "source_config_path": str(config_path),
        "source_config_sha256": rr.sha256_file(config_path),
        "input_sha256": input_hashes,
        "device": args.device,
        "smoke": args.smoke,
        "effective_checkpoint_epochs": checkpoints,
        "code_sha256": rr.sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output_dir / "config.json", config_record)

    device = torch.device(args.device)
    model = rr.ResidualUserProjector(
        context["train_x"].shape[1], int(training["hidden_dim"]), float(training["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["lr"]), weight_decay=float(training["weight_decay"])
    )
    generator = torch.Generator().manual_seed(seed)
    checkpoint_set = set(checkpoints)
    history: list[dict] = []
    trajectory: list[dict] = []
    observations_by_epoch: dict[int, dict[str, list[float]]] = {}
    baseline_ok = True

    print(
        f"[data] train_examples={len(context['train_x'])} eval_users={len(context['eval_uids'])} "
        f"checkpoints={checkpoints} smoke={args.smoke}",
        flush=True,
    )
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        order = torch.randperm(len(context["train_x"]), generator=generator)
        total_loss = 0.0
        total_n = 0
        for start in range(0, len(order), int(training["batch_size"])):
            batch_ids = order[start:start + int(training["batch_size"])]
            x = context["train_x"][batch_ids].to(device)
            y_idx = context["train_y"][batch_ids].to(device)
            target_vec = context["embeddings"][y_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            user_vec = model(x)
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
            f"[train] epoch={epoch}/{max(checkpoints)} loss={record['loss']:.6f} "
            f"scale={record['residual_scale']:.4f}",
            flush=True,
        )
        update_status(
            status_path,
            stage="training",
            current_epoch=epoch,
            total_epochs=max(checkpoints),
            last_loss=record["loss"],
            completed_checkpoints=[row["epoch"] for row in trajectory],
        )

        if epoch not in checkpoint_set:
            continue
        update_status(status_path, stage="checkpoint_evaluation", current_epoch=epoch)
        checkpoint_summary, observations, mismatches, digest = evaluate_checkpoint(
            model,
            context,
            device,
            int(source_config["evaluation"]["global_retrieve_k"]),
            eval_limit=128 if args.smoke else 0,
        )
        checkpoint_payload = {
            "epoch": epoch,
            "training": record,
            **checkpoint_summary,
        }
        atomic_json(output_dir / f"checkpoint_epoch_{epoch:03d}.json", checkpoint_payload)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "dim": context["embeddings"].shape[1],
                "hidden_dim": int(training["hidden_dim"]),
                "dropout": float(training["dropout"]),
                "seed": seed,
                "resolver_top50_sha256": digest,
            },
            output_dir / f"resolver_epoch_{epoch:03d}.pt",
        )
        trajectory.append(checkpoint_payload)
        observations_by_epoch[epoch] = observations
        print(
            f"[checkpoint] epoch={epoch} cold_r50="
            f"{checkpoint_summary['resolver_cold_rank']['recall']['@50']:.6f} "
            f"eligible_r3={checkpoint_summary['eligible_cold_rank']['recall']['@3']:.6f} "
            f"p2_all_n10={checkpoint_summary['metrics']['portfolio@2']['all']['ndcg@10']:.6f} "
            f"p0_top50_mismatch={mismatches}",
            flush=True,
        )

        required_epoch = int(source_config["baseline_reproduction"]["required_epoch"])
        if epoch == required_epoch and not args.smoke:
            expected_mismatch = int(
                source_config["baseline_reproduction"]["expected_exact_top50_mismatches"]
            )
            expected_events = int(
                source_config["baseline_reproduction"]["expected_cold_hit50_events"]
            )
            actual_events = int(checkpoint_summary["resolver_cold_rank"]["events_top50"])
            baseline_ok = mismatches == expected_mismatch and actual_events == expected_events
            if not baseline_ok:
                print(
                    f"[stop] baseline reproduction failed: mismatches={mismatches}, "
                    f"cold_hit50_events={actual_events}",
                    flush=True,
                )
                break

    bootstrap: dict[str, dict] = {}
    baseline_epoch = int(source_config["baseline_reproduction"]["required_epoch"])
    if baseline_ok and not args.smoke and baseline_epoch in observations_by_epoch:
        baseline_obs = observations_by_epoch[baseline_epoch]
        resamples = int(source_config["evaluation"]["bootstrap_resamples"])
        bootstrap_seed = int(source_config["evaluation"]["bootstrap_seed"])
        for epoch, observations in observations_by_epoch.items():
            if epoch == baseline_epoch:
                continue
            bootstrap[str(epoch)] = {
                key: paired_bootstrap(
                    values,
                    baseline_obs[key],
                    resamples,
                    bootstrap_seed + epoch * 100 + index,
                )
                for index, (key, values) in enumerate(observations.items())
            }

    summary = {
        "experiment_id": source_config["experiment_id"],
        "status": "completed",
        "verdict": (
            "SMOKE_COMPLETED"
            if args.smoke
            else "TRAJECTORY_COMPLETED_REVIEW_REQUIRED"
            if baseline_ok and len(trajectory) == len(checkpoints)
            else "BASELINE_REPRODUCTION_FAILED_STOP"
        ),
        "evidence_role": source_config["evidence_role"],
        "test_read": False,
        "beauty_read": False,
        "sports_read": False,
        "baseline_reproduction_passed": baseline_ok if not args.smoke else None,
        "train_targets_all_warm": context["train_report"]["cold_target_count"] == 0,
        "train_report": context["train_report"],
        "training_history": history,
        "trajectory": trajectory,
        "paired_bootstrap_vs_epoch12": bootstrap,
        "automatic_next_stage": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    update_status(
        status_path,
        stage="finished",
        current_epoch=history[-1]["epoch"],
        completed_checkpoints=[row["epoch"] for row in trajectory],
        scientific_verdict=summary["verdict"],
    )
    print(f"[result] verdict={summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
