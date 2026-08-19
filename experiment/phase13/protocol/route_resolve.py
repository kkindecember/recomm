"""Phase-13 v1-R² validation-only route-and-resolve screen.

The generator is kept frozen.  Its legal item beams are aggregated by the
first ``route_depth`` tokens of the original semantic ID and are used only as
a coarse route prior.  A warm-only residual user projector retrieves exact
catalog item IDs from frozen text embeddings.  The final R² ranking is a fixed
reciprocal-rank fusion of:

* the frozen GRAM item beam,
* global exact-item retrieval,
* retrieval inside the generated semantic routes, and
* the generated route rank.

The script deliberately evaluates validation only.  It never opens a test
prediction file and never rewrites a hierarchical-ID table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


METRIC_KS = (1, 3, 5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--item-id-file", required=True)
    p.add_argument("--item-embeddings", required=True)
    p.add_argument("--gram-validation-predictions", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--route-depth", type=int, default=3)
    p.add_argument("--max-history", type=int, default=20)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--recency-decay", type=float, default=0.85)
    p.add_argument("--global-retrieve-k", type=int, default=200)
    p.add_argument("--top-routes", type=int, default=8)
    p.add_argument("--per-route-k", type=int, default=50)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--route-prior-weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--train-example-limit", type=int, default=0,
                   help="Smoke-only prefix limit; 0 uses every warm transition")
    p.add_argument("--eval-user-limit", type=int, default=0)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_key_value_lines(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            key, sep, value = raw.partition(" ")
            if not sep:
                raise ValueError(f"Malformed key/value line in {path}: {raw!r}")
            if key in out:
                raise ValueError(f"Duplicate key in {path}: {key}")
            out[key] = value
    return out


def read_sequences(path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    with path.open() as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) >= 4:
                rows.append((parts[0], parts[1:]))
    return rows


def read_set(path: Path) -> set[str]:
    return {x.strip() for x in path.read_text().splitlines() if x.strip()}


def semantic_tokens(lexical_id: str) -> tuple[str, ...]:
    tokens = tuple(token for token in lexical_id.split("|") if token)
    if not tokens:
        raise ValueError(f"Semantic ID has no tokens: {lexical_id!r}")
    return tokens


def semantic_route(lexical_id: str, depth: int) -> tuple[str, ...]:
    tokens = semantic_tokens(lexical_id)
    if len(tokens) < depth:
        raise ValueError(
            f"Semantic ID has {len(tokens)} levels, shorter than route depth {depth}"
        )
    return tokens[:depth]


def decode_lexical_id(lexical_id: str) -> str:
    """Reproduce T5/SentencePiece decoding for the delimiter-separated IDs."""
    pieces = []
    for token in semantic_tokens(lexical_id):
        pieces.append(token.replace("▁", " "))
    return "".join(pieces).strip()


def parse_gram_predictions(path: Path) -> dict[str, dict]:
    if "test" in path.name.lower():
        raise ValueError(f"Refusing a test prediction file in P0: {path}")
    rows: dict[str, dict] = {}
    with path.open() as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw or raw.startswith(("idx\t", "hit@", "ndcg@")):
                continue
            fields = raw.split("\t")
            if len(fields) < 16:
                continue
            try:
                metrics = [float(x) for x in fields[1:13]]
            except ValueError:
                continue
            uid = fields[0]
            predictions = fields[14].split("||") if fields[14] else []
            scores = [float(x) for x in fields[15].split("||")] if fields[15] else []
            if len(predictions) != len(scores):
                raise ValueError(
                    f"Prediction/score mismatch at {path}:{line_no}: "
                    f"{len(predictions)} != {len(scores)}"
                )
            if uid in rows:
                raise ValueError(f"Duplicate prediction user {uid} in {path}")
            rows[uid] = {
                "saved_metrics": metrics,
                "gold_lexical": fields[13],
                "predictions": predictions,
                "scores": scores,
            }
    if not rows:
        raise ValueError(f"No prediction rows parsed from {path}")
    return rows


def recency_weighted_history(
    item_indices: Iterable[int], embeddings: torch.Tensor, decay: float
) -> torch.Tensor:
    indices = list(item_indices)
    if not indices:
        raise ValueError("History cannot be empty")
    history = embeddings[torch.tensor(indices, dtype=torch.long)]
    ages = torch.arange(len(indices) - 1, -1, -1, dtype=history.dtype)
    weights = decay ** ages
    pooled = (history * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-12)
    return F.normalize(pooled, dim=0)


def build_training_examples(
    sequences: list[tuple[str, list[str]]],
    item_to_idx: dict[str, int],
    embeddings: torch.Tensor,
    cold_items: set[str],
    max_history: int,
    recency_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    histories: list[torch.Tensor] = []
    targets: list[int] = []
    target_items: list[str] = []
    for _uid, items in sequences:
        train_prefix = items[:-2]
        for pos in range(1, len(train_prefix)):
            target = train_prefix[pos]
            if target in cold_items:
                raise RuntimeError(f"Cold target leaked into train prefix: {target}")
            history_items = train_prefix[max(0, pos - max_history):pos]
            histories.append(
                recency_weighted_history(
                    (item_to_idx[x] for x in history_items), embeddings, recency_decay
                )
            )
            targets.append(item_to_idx[target])
            target_items.append(target)
    if not histories:
        raise ValueError("No warm-only training transitions were constructed")
    report = {
        "n_examples": len(histories),
        "n_unique_targets": len(set(target_items)),
        "cold_target_count": sum(x in cold_items for x in target_items),
    }
    return torch.stack(histories), torch.tensor(targets, dtype=torch.long), report


def build_validation_examples(
    sequences: list[tuple[str, list[str]]],
    item_to_idx: dict[str, int],
    embeddings: torch.Tensor,
    max_history: int,
    recency_decay: float,
) -> dict[str, tuple[torch.Tensor, int, str]]:
    rows: dict[str, tuple[torch.Tensor, int, str]] = {}
    for uid, items in sequences:
        history_items = items[max(0, len(items) - 2 - max_history):-2]
        target = items[-2]
        rows[uid] = (
            recency_weighted_history(
                (item_to_idx[x] for x in history_items), embeddings, recency_decay
            ),
            item_to_idx[target],
            target,
        )
    return rows


class ResidualUserProjector(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.residual_scale * self.net(x), dim=-1)


def multi_positive_inbatch_loss(
    user_vec: torch.Tensor,
    target_vec: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = user_vec @ target_vec.T / temperature
    positives = target_ids[:, None].eq(target_ids[None, :])
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()


def train_projector(
    train_histories: torch.Tensor,
    train_targets: torch.Tensor,
    item_embeddings: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ResidualUserProjector, list[dict]]:
    model = ResidualUserProjector(
        train_histories.shape[1], args.hidden_dim, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_histories), generator=generator)
        total_loss = 0.0
        total_n = 0
        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start:start + args.batch_size]
            x = train_histories[batch_ids].to(device)
            y_idx = train_targets[batch_ids].to(device)
            target_vec = item_embeddings[y_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            user_vec = model(x)
            loss = multi_positive_inbatch_loss(
                user_vec, target_vec, y_idx, args.temperature
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
            f"[train] epoch={epoch}/{args.epochs} loss={record['loss']:.6f} "
            f"scale={record['residual_scale']:.4f}",
            flush=True,
        )
    model.eval()
    return model, history


def ranking_metrics(ranked_items: list[str], target: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        rank = ranked_items.index(target) + 1
    except ValueError:
        rank = math.inf
    for k in METRIC_KS:
        hit = float(rank <= k)
        out[f"hit@{k}"] = hit
        out[f"ndcg@{k}"] = (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
    return out


def average_metrics(rows: list[dict[str, float]]) -> dict:
    if not rows:
        return {"n": 0, **{f"hit@{k}": None for k in METRIC_KS},
                **{f"ndcg@{k}": None for k in METRIC_KS}}
    keys = rows[0].keys()
    return {"n": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in keys}}


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def fuse_r2(
    resolver_order: list[int],
    gram_items: list[str],
    route_order: list[tuple[str, ...]],
    route_to_ranked_indices: dict[tuple[str, ...], list[int]],
    item_ids: list[str],
    rrf_k: int,
    route_prior_weight: float,
    global_retrieve_k: int,
    per_route_k: int,
) -> list[str]:
    scores: defaultdict[str, float] = defaultdict(float)
    for rank, idx in enumerate(resolver_order[:global_retrieve_k], 1):
        scores[item_ids[idx]] += 1.0 / (rrf_k + rank)
    for rank, item in enumerate(gram_items, 1):
        scores[item] += 1.0 / (rrf_k + rank)
    for route_rank, route in enumerate(route_order, 1):
        route_bonus = route_prior_weight / (rrf_k + route_rank)
        for within_rank, idx in enumerate(
            route_to_ranked_indices.get(route, [])[:per_route_k], 1
        ):
            item = item_ids[idx]
            scores[item] += 1.0 / (rrf_k + within_rank) + route_bonus
    return [item for item, _score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    item_id_path = Path(args.item_id_file).resolve()
    embedding_path = Path(args.item_embeddings).resolve()
    prediction_path = Path(args.gram_validation_predictions).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
        unexpected = [p.name for p in output_dir.iterdir() if p.name not in allowed]
        if unexpected:
            raise FileExistsError(
                f"Refusing non-empty output with scientific artifacts: {unexpected}"
            )

    inputs = [
        dataset_dir / "user_sequence.txt",
        dataset_dir / "cold_split_meta" / "cold_items.txt",
        dataset_dir / "cold_split_meta" / "warm_items.txt",
        item_id_path,
        embedding_path,
        prediction_path,
    ]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    # The frozen gram-repro environment uses a PyTorch release predating the
    # ``weights_only`` keyword.  This file is a locally produced, trusted
    # Phase-13 payload containing metadata plus one tensor.
    payload = torch.load(embedding_path, map_location="cpu")
    item_ids = list(payload["item_ids"])
    item_embeddings = F.normalize(payload["embeddings"].float(), dim=1)
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    if len(item_to_idx) != len(item_ids):
        raise ValueError("Embedding payload has duplicate item IDs")

    item_to_lexical = read_key_value_lines(item_id_path)
    if set(item_to_lexical) != set(item_ids):
        raise ValueError(
            "Embedding/item-ID catalog mismatch: "
            f"embedding_only={len(set(item_ids)-set(item_to_lexical))}, "
            f"id_only={len(set(item_to_lexical)-set(item_ids))}"
        )
    decoded_to_item: dict[str, str] = {}
    for item, lexical in item_to_lexical.items():
        decoded = decode_lexical_id(lexical)
        if decoded in decoded_to_item:
            raise ValueError(
                f"Decoded semantic ID collision: {decoded_to_item[decoded]} and {item}"
            )
        decoded_to_item[decoded] = item

    item_routes = [
        semantic_route(item_to_lexical[item], args.route_depth) for item in item_ids
    ]
    route_to_indices: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, route in enumerate(item_routes):
        route_to_indices[route].append(idx)

    sequences = read_sequences(dataset_dir / "user_sequence.txt")
    cold_items = read_set(dataset_dir / "cold_split_meta" / "cold_items.txt")
    warm_items = read_set(dataset_dir / "cold_split_meta" / "warm_items.txt")
    if cold_items & warm_items:
        raise ValueError("Cold/warm item sets overlap")
    gram_rows = parse_gram_predictions(prediction_path)

    train_x, train_y, train_report = build_training_examples(
        sequences, item_to_idx, item_embeddings, cold_items,
        args.max_history, args.recency_decay,
    )
    train_report["n_examples_available"] = len(train_x)
    if args.train_example_limit:
        train_x = train_x[:args.train_example_limit]
        train_y = train_y[:args.train_example_limit]
    train_report["n_examples_used"] = len(train_x)
    validation = build_validation_examples(
        sequences, item_to_idx, item_embeddings,
        args.max_history, args.recency_decay,
    )
    eval_uids = [uid for uid, _items in sequences if uid in gram_rows]
    if args.eval_user_limit:
        eval_uids = eval_uids[:args.eval_user_limit]
    if not eval_uids:
        raise ValueError("No validation users overlap the GRAM prediction file")

    config = vars(args).copy()
    config.update({
        "dataset_dir": str(dataset_dir),
        "item_id_file": str(item_id_path),
        "item_embeddings": str(embedding_path),
        "gram_validation_predictions": str(prediction_path),
        "output_dir": str(output_dir),
        "split": "validation",
        "test_predictions_opened": False,
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "embedding_model": payload.get("model_name"),
        "embedding_pooling": payload.get("pooling"),
        "embedding_l2_normalized": payload.get("l2_normalized"),
        "n_catalog_items": len(item_ids),
        "n_routes": len(route_to_indices),
        "n_eval_users": len(eval_uids),
        "train_report": train_report,
    })
    atomic_json(output_dir / "config.json", config)

    print(
        f"[data] catalog={len(item_ids)} routes={len(route_to_indices)} "
        f"train_examples={len(train_x)} eval_users={len(eval_uids)}",
        flush=True,
    )
    model, training_history = train_projector(
        train_x, train_y, item_embeddings, args, device
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dim": item_embeddings.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "seed": args.seed,
        },
        output_dir / "resolver.pt",
    )

    item_embeddings_device = item_embeddings.to(device)
    results = {
        model_name: {slice_name: [] for slice_name in ("all", "warm", "cold")}
        for model_name in ("v0_gram", "resolver_only", "r2")
    }
    route_supported = {"all": 0, "warm": 0, "cold": 0}
    candidate_hit50 = {"all": 0, "warm": 0, "cold": 0}
    prediction_records: list[dict] = []

    with torch.no_grad():
        for offset in range(0, len(eval_uids), 256):
            batch_uids = eval_uids[offset:offset + 256]
            base = torch.stack([validation[uid][0] for uid in batch_uids]).to(device)
            projected = model(base)
            score_matrix = projected @ item_embeddings_device.T
            for row_idx, uid in enumerate(batch_uids):
                _history, _target_idx, target = validation[uid]
                slice_name = "cold" if target in cold_items else "warm"
                scores = score_matrix[row_idx]
                global_k = min(args.global_retrieve_k, len(item_ids))
                resolver_indices = torch.topk(scores, k=global_k).indices.tolist()
                resolver_items = [item_ids[idx] for idx in resolver_indices]

                gram_items: list[str] = []
                route_best_score: dict[tuple[str, ...], float] = {}
                for decoded, beam_score in zip(
                    gram_rows[uid]["predictions"], gram_rows[uid]["scores"]
                ):
                    item = decoded_to_item.get(decoded)
                    if item is None:
                        raise KeyError(f"Could not map legal GRAM beam to item: {decoded!r}")
                    gram_items.append(item)
                    route = semantic_route(item_to_lexical[item], args.route_depth)
                    route_best_score[route] = max(route_best_score.get(route, -math.inf), beam_score)
                gram_items = unique_in_order(gram_items)
                route_order = [
                    route for route, _score in sorted(
                        route_best_score.items(), key=lambda x: (-x[1], x[0])
                    )[:args.top_routes]
                ]
                target_route = semantic_route(item_to_lexical[target], args.route_depth)
                supported = target_route in route_order
                for name in ("all", slice_name):
                    route_supported[name] += int(supported)

                ranked_by_route: dict[tuple[str, ...], list[int]] = {}
                for route in route_order:
                    indices = route_to_indices[route]
                    local_scores = scores[torch.tensor(indices, device=device)]
                    local_k = min(args.per_route_k, len(indices))
                    local_order = torch.topk(local_scores, k=local_k).indices.tolist()
                    ranked_by_route[route] = [indices[i] for i in local_order]

                r2_items = fuse_r2(
                    resolver_indices,
                    gram_items,
                    route_order,
                    ranked_by_route,
                    item_ids,
                    args.rrf_k,
                    args.route_prior_weight,
                    args.global_retrieve_k,
                    args.per_route_k,
                )
                candidate_hit = target in r2_items[:50]
                for name in ("all", slice_name):
                    candidate_hit50[name] += int(candidate_hit)

                rankings = {
                    "v0_gram": gram_items,
                    "resolver_only": resolver_items,
                    "r2": r2_items,
                }
                for model_name, ranking in rankings.items():
                    if len(ranking) != len(set(ranking)):
                        raise RuntimeError(f"Duplicate output for {uid}/{model_name}")
                    if not set(ranking).issubset(item_to_idx):
                        raise RuntimeError(f"Non-catalog output for {uid}/{model_name}")
                    metrics = ranking_metrics(ranking, target)
                    results[model_name]["all"].append(metrics)
                    results[model_name][slice_name].append(metrics)

                prediction_records.append({
                    "user_id": uid,
                    "target": target,
                    "is_cold": target in cold_items,
                    "target_route_supported_top_routes": supported,
                    "v0_top50": gram_items[:50],
                    "resolver_top50": resolver_items[:50],
                    "r2_top50": r2_items[:50],
                })
            print(f"[eval] {min(offset + 256, len(eval_uids))}/{len(eval_uids)}", flush=True)

    metrics_summary = {
        model_name: {
            slice_name: average_metrics(rows)
            for slice_name, rows in slices.items()
        }
        for model_name, slices in results.items()
    }
    slice_counts = {
        "all": len(results["r2"]["all"]),
        "warm": len(results["r2"]["warm"]),
        "cold": len(results["r2"]["cold"]),
    }
    route_support_rate = {
        name: route_supported[name] / max(slice_counts[name], 1) for name in slice_counts
    }
    candidate_hit50_rate = {
        name: candidate_hit50[name] / max(slice_counts[name], 1) for name in slice_counts
    }

    v0_cold = metrics_summary["v0_gram"]["cold"]
    resolver_cold = metrics_summary["resolver_only"]["cold"]
    r2_cold = metrics_summary["r2"]["cold"]
    v0_warm = metrics_summary["v0_gram"]["warm"]
    r2_warm = metrics_summary["r2"]["warm"]
    gates = {
        "train_targets_all_warm": train_report["cold_target_count"] == 0,
        "r2_cold_ndcg10_ge_1_10x_v0": r2_cold["ndcg@10"] >= 1.10 * v0_cold["ndcg@10"],
        "r2_cold_hit50_ge_1_10x_v0": r2_cold["hit@50"] >= 1.10 * v0_cold["hit@50"],
        "r2_cold_ndcg10_gt_resolver": r2_cold["ndcg@10"] > resolver_cold["ndcg@10"],
        "r2_warm_ndcg10_ge_0_97x_v0": r2_warm["ndcg@10"] >= 0.97 * v0_warm["ndcg@10"],
        "catalog_outputs_unique": True,
        "validation_only": True,
    }
    verdict = (
        "PASS_TO_R2_MEDIUM_SMOKE_DISCUSSION"
        if all(gates.values())
        else "FAIL_STOP_R2_P0"
    )
    summary = {
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P0",
        "status": "completed",
        "verdict": verdict,
        "split": "validation",
        "test_predictions_opened": False,
        "metrics": metrics_summary,
        "route_support_rate": route_support_rate,
        "r2_candidate_hit50_rate": candidate_hit50_rate,
        "gates": gates,
        "train_report": train_report,
        "training_history": training_history,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions_validation.jsonl").open("w") as f:
        for record in prediction_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"[result] verdict={verdict} "
        f"v0_cold_ndcg10={v0_cold['ndcg@10']:.6f} "
        f"resolver={resolver_cold['ndcg@10']:.6f} "
        f"r2={r2_cold['ndcg@10']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
