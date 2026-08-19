"""Warm-only pseudo-cold setwise candidate selection for v1-R² P5-set."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from confidence_abstention import predict_gate
from counterfactual_slot_router import (
    affected_items,
    choose_best_action,
    extract_item_feature_vector,
    ranking_for_action,
    stable_fold,
)
from route_admission import read_prediction_records
from route_resolve import (
    ResidualUserProjector,
    atomic_json,
    average_metrics,
    build_validation_examples,
    ranking_metrics,
    read_key_value_lines,
    read_sequences,
    read_set,
    recency_weighted_history,
    semantic_route,
    sha256_file,
    unique_in_order,
)


SELECTOR_FEATURE_NAMES = (
    "resolver_cosine",
    "cosine_vs_set_top_gap",
    "set_reciprocal_rank",
    "history_item_max_cosine",
    "history_item_mean_cosine",
    "last_item_cosine",
    "set_cosine_zscore",
)
POOL_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--p0-predictions", required=True)
    parser.add_argument("--item-id-file", required=True)
    parser.add_argument("--item-embeddings", required=True)
    parser.add_argument("--resolver-checkpoint", required=True)
    parser.add_argument("--p4-checkpoint", required=True)
    parser.add_argument("--p4-summary", required=True)
    parser.add_argument("--cold-items", required=True)
    parser.add_argument("--warm-items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--recency-decay", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=32345)
    return parser.parse_args()


def is_pseudo_cold_item(item_id: str) -> bool:
    value = int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:8], "big")
    return value % 5 == 0


def build_warm_transitions(
    sequences: list[tuple[str, list[str]]],
    item_to_idx: dict[str, int],
    warm_items: set[str],
    max_history: int,
) -> list[tuple[list[int], int, str]]:
    transitions: list[tuple[list[int], int, str]] = []
    for _uid, items in sequences:
        prefix = items[:-2]
        for position in range(1, len(prefix)):
            target = prefix[position]
            if target not in warm_items:
                raise RuntimeError(f"Non-warm target in train prefix: {target}")
            history = prefix[max(0, position - max_history):position]
            transitions.append(
                ([item_to_idx[item] for item in history], item_to_idx[target], target)
            )
    return transitions


class SetwiseSelector(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.candidate_encoder(features)
        mean_context = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
        max_context = encoded.max(dim=1, keepdim=True).values.expand_as(encoded)
        return self.score_head(torch.cat([encoded, mean_context, max_context], dim=-1)).squeeze(-1)


def selector_feature_tensor(
    projected_users: torch.Tensor,
    candidate_indices: torch.Tensor,
    history_indices: torch.Tensor,
    history_mask: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Target-free setwise features; candidate order is frozen resolver order."""
    candidate_embeddings = embeddings[candidate_indices]
    cosines = torch.einsum("bd,bkd->bk", projected_users, candidate_embeddings)
    gaps = cosines - cosines[:, :1]
    ranks = torch.arange(1, candidate_indices.shape[1] + 1, device=cosines.device)
    reciprocal = (1.0 / torch.log2(ranks.float() + 1.0))[None, :].expand_as(cosines)
    history_embeddings = embeddings[history_indices]
    similarities = torch.einsum("bkd,bld->bkl", candidate_embeddings, history_embeddings)
    expanded_mask = history_mask[:, None, :]
    history_max = similarities.masked_fill(~expanded_mask, -torch.inf).max(dim=-1).values
    history_mean = (
        similarities.masked_fill(~expanded_mask, 0.0).sum(dim=-1)
        / expanded_mask.sum(dim=-1).clamp_min(1)
    )
    last_positions = history_mask.sum(dim=1).long() - 1
    last_embeddings = history_embeddings[
        torch.arange(len(history_embeddings), device=embeddings.device), last_positions
    ]
    last_similarity = torch.einsum("bkd,bd->bk", candidate_embeddings, last_embeddings)
    zscore = (cosines - cosines.mean(dim=1, keepdim=True)) / cosines.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    return torch.stack(
        [cosines, gaps, reciprocal, history_max, history_mean, last_similarity, zscore],
        dim=-1,
    )


def pad_histories(histories: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(history) for history in histories)
    indices = torch.zeros((len(histories), max_length), dtype=torch.long, device=device)
    mask = torch.zeros((len(histories), max_length), dtype=torch.bool, device=device)
    for row, history in enumerate(histories):
        indices[row, : len(history)] = torch.tensor(history, dtype=torch.long, device=device)
        mask[row, : len(history)] = True
    return indices, mask


def export_candidate_sets(
    transitions: list[tuple[list[int], int, str]],
    candidate_catalog_indices: torch.Tensor,
    resolver: ResidualUserProjector,
    embeddings_cpu: torch.Tensor,
    embeddings_device: torch.Tensor,
    pool_size: int,
    batch_size: int,
    recency_decay: float,
    device: torch.device,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    target_hit = 0
    for offset in range(0, len(transitions), batch_size):
        batch = transitions[offset:offset + batch_size]
        histories = [row[0] for row in batch]
        targets = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
        pooled = torch.stack([
            recency_weighted_history(history, embeddings_cpu, recency_decay)
            for history in histories
        ]).to(device)
        with torch.no_grad():
            projected = resolver(pooled)
            catalog_scores = projected @ embeddings_device[candidate_catalog_indices].T
            local_top = torch.topk(catalog_scores, k=pool_size, dim=1).indices
            candidates = candidate_catalog_indices[local_top]
            matches = candidates == targets[:, None]
            hit_mask = matches.any(dim=1)
            if hit_mask.any():
                history_indices, history_mask = pad_histories(
                    [histories[index] for index in torch.where(hit_mask)[0].tolist()], device
                )
                selected_features = selector_feature_tensor(
                    projected[hit_mask], candidates[hit_mask], history_indices,
                    history_mask, embeddings_device,
                )
                selected_labels = matches[hit_mask].float().argmax(dim=1).long()
                feature_batches.append(selected_features.cpu())
                label_batches.append(selected_labels.cpu())
                target_hit += int(hit_mask.sum())
        print(
            f"[{label}] {min(offset + batch_size, len(transitions))}/{len(transitions)} "
            f"eligible={target_hit}", flush=True,
        )
    if not feature_batches:
        raise ValueError(f"No target-in-pool examples for {label}")
    return (
        torch.cat(feature_batches), torch.cat(label_batches),
        {
            "n_transitions": len(transitions),
            "n_target_in_pool": target_hit,
            "target_in_pool_rate": target_hit / len(transitions),
        },
    )


def fit_selector(
    model: SetwiseSelector,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(features), generator=generator)
        losses: list[float] = []
        model.train()
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset:offset + batch_size]
            logits = model(features[indices].to(device))
            loss = F.cross_entropy(logits, labels[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses)})
        print(f"[train] epoch={epoch}/{epochs} loss={history[-1]['loss']:.6f}", flush=True)
    model.eval()
    return history


def selector_accuracy(model: SetwiseSelector, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict:
    correct = 0
    base_correct = int((labels == 0).sum())
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(features), 1024):
            logits = model(features[offset:offset + 1024].to(device))
            predicted = logits.argmax(dim=1).cpu()
            correct += int((predicted == labels[offset:offset + 1024]).sum())
    return {
        "n": len(labels),
        "resolver_top1_correct": base_correct,
        "resolver_top1_accuracy": base_correct / len(labels),
        "selector_top1_correct": correct,
        "selector_top1_accuracy": correct / len(labels),
        "selector_vs_resolver_ratio": correct / max(base_correct, 1),
    }


def summarize(metric_rows: dict[str, dict[str, list[dict[str, float]]]]) -> dict:
    return {
        model: {name: average_metrics(values) for name, values in slices.items()}
        for model, slices in metric_rows.items()
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    p0_path = Path(args.p0_predictions).resolve()
    item_id_path = Path(args.item_id_file).resolve()
    embedding_path = Path(args.item_embeddings).resolve()
    resolver_path = Path(args.resolver_checkpoint).resolve()
    p4_path = Path(args.p4_checkpoint).resolve()
    p4_summary_path = Path(args.p4_summary).resolve()
    cold_path = Path(args.cold_items).resolve()
    warm_path = Path(args.warm_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in allowed]
    if unexpected:
        raise FileExistsError(f"Refusing existing P5-set scientific artifacts: {unexpected}")
    inputs = [
        dataset_dir / "user_sequence.txt", p0_path, item_id_path, embedding_path,
        resolver_path, p4_path, p4_summary_path, cold_path, warm_path,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        if "test" in path.name.lower():
            raise ValueError(f"Refusing test input in P5-set: {path}")

    records = read_prediction_records(p0_path)
    item_to_lexical = read_key_value_lines(item_id_path)
    catalog = set(item_to_lexical)
    cold_items = read_set(cold_path)
    warm_items = read_set(warm_path)
    if cold_items & warm_items or cold_items | warm_items != catalog:
        raise ValueError("Invalid warm/cold catalog partition")
    selector_train_items = {item for item in warm_items if not is_pseudo_cold_item(item)}
    pseudo_cold_items = warm_items - selector_train_items
    item_routes = {item: semantic_route(lexical, 1) for item, lexical in item_to_lexical.items()}

    embedding_payload = torch.load(embedding_path, map_location="cpu")
    item_ids = list(embedding_payload["item_ids"])
    embeddings_cpu = F.normalize(embedding_payload["embeddings"].float(), dim=1)
    item_to_idx = {item: index for index, item in enumerate(item_ids)}
    if set(item_ids) != catalog:
        raise ValueError("Embedding/catalog mismatch")
    resolver_checkpoint = torch.load(resolver_path, map_location="cpu")
    resolver = ResidualUserProjector(
        resolver_checkpoint["dim"], resolver_checkpoint["hidden_dim"],
        resolver_checkpoint["dropout"],
    )
    resolver.load_state_dict(resolver_checkpoint["state_dict"])
    resolver.eval()
    device = torch.device(args.device)
    resolver.to(device)
    embeddings_device = embeddings_cpu.to(device)

    sequences = read_sequences(dataset_dir / "user_sequence.txt")
    transitions = build_warm_transitions(
        sequences, item_to_idx, warm_items, args.max_history
    )
    train_transitions = [row for row in transitions if row[2] in selector_train_items]
    pseudo_transitions = [row for row in transitions if row[2] in pseudo_cold_items]
    train_catalog_indices = torch.tensor(
        [item_to_idx[item] for item in item_ids if item in selector_train_items], device=device
    )
    pseudo_catalog_indices = torch.tensor(
        [item_to_idx[item] for item in item_ids if item in pseudo_cold_items], device=device
    )

    config = {
        **vars(args),
        "dataset_dir": str(dataset_dir),
        "p0_predictions": str(p0_path),
        "item_id_file": str(item_id_path),
        "item_embeddings": str(embedding_path),
        "resolver_checkpoint": str(resolver_path),
        "p4_checkpoint": str(p4_path),
        "p4_summary": str(p4_summary_path),
        "cold_items": str(cold_path),
        "warm_items": str(warm_path),
        "output_dir": str(output_dir),
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P5_SETWISE_SELECTOR",
        "selector_feature_names": SELECTOR_FEATURE_NAMES,
        "pseudo_cold_rule": "uint64_be(sha256(item_id)[:8]) mod 5 == 0",
        "n_selector_train_items": len(selector_train_items),
        "n_pseudo_cold_items": len(pseudo_cold_items),
        "selector_target_item_overlap": len(selector_train_items & pseudo_cold_items),
        "test_predictions_opened": False,
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
    }
    atomic_json(output_dir / "config.json", config)

    train_features, train_labels, train_export = export_candidate_sets(
        train_transitions, train_catalog_indices, resolver, embeddings_cpu,
        embeddings_device, args.pool_size, args.batch_size, args.recency_decay,
        device, "selector-train-export",
    )
    pseudo_features, pseudo_labels, pseudo_export = export_candidate_sets(
        pseudo_transitions, pseudo_catalog_indices, resolver, embeddings_cpu,
        embeddings_device, args.pool_size, args.batch_size, args.recency_decay,
        device, "pseudo-cold-export",
    )
    selector = SetwiseSelector(len(SELECTOR_FEATURE_NAMES), args.hidden_dim)
    training_history = fit_selector(
        selector, train_features, train_labels, args.epochs, args.batch_size,
        args.lr, args.weight_decay, args.seed, device,
    )
    pseudo_audit = selector_accuracy(selector, pseudo_features, pseudo_labels, device)
    torch.save({
        "state_dict": selector.state_dict(),
        "n_features": len(SELECTOR_FEATURE_NAMES),
        "hidden_dim": args.hidden_dim,
        "feature_names": SELECTOR_FEATURE_NAMES,
        "pool_size": args.pool_size,
        "seed": args.seed,
    }, output_dir / "setwise_selector.pt")

    p4_checkpoint = torch.load(p4_path, map_location="cpu")
    p4_summary = json.loads(p4_summary_path.read_text())
    fold_models = p4_checkpoint["fold_models"]
    fold_thresholds = {
        int(report["fold"]): float(report["selected"]["threshold"])
        for report in p4_summary["fold_reports"]
    }
    validation = build_validation_examples(
        sequences, item_to_idx, embeddings_cpu, args.max_history, args.recency_decay
    )
    sequence_by_uid = {uid: items for uid, items in sequences}
    record_by_uid = {str(row["user_id"]): row for row in records}
    ordered_uids = [uid for uid, _items in sequences if uid in record_by_uid]
    model_names = (
        "v0_gram", "resolver_only", "p5_setwise", "candidate_pool_oracle",
        "label_aware_oracle",
    )
    metric_rows = {
        model: {name: [] for name in ("all", "warm", "cold")}
        for model in model_names
    }
    base_candidate_correct = 0
    selector_candidate_correct = 0
    pool_target_hit = 0
    n_cold_users = 0
    actions: list[str] = []
    predictions: list[dict] = []
    selector.eval()
    with torch.no_grad():
        for offset in range(0, len(ordered_uids), args.batch_size):
            batch_uids = ordered_uids[offset:offset + args.batch_size]
            pooled = torch.stack([validation[uid][0] for uid in batch_uids]).to(device)
            projected = resolver(pooled)
            history_lists = [
                [item_to_idx[item] for item in sequence_by_uid[uid][max(0, len(sequence_by_uid[uid]) - 2 - args.max_history):-2]]
                for uid in batch_uids
            ]
            candidate_indices_rows: list[list[int]] = []
            cold_pools: list[list[str]] = []
            for uid in batch_uids:
                row = record_by_uid[uid]
                gram = unique_in_order(row["v0_top50"])
                protected = set(gram[:6])
                pool = [
                    item for item in unique_in_order(row["resolver_top50"])
                    if item in cold_items and item not in protected
                ][: args.pool_size]
                if len(pool) != args.pool_size:
                    raise ValueError(f"Cold candidate pool shorter than {args.pool_size} for {uid}")
                cold_pools.append(pool)
                candidate_indices_rows.append([item_to_idx[item] for item in pool])
            candidate_indices = torch.tensor(candidate_indices_rows, device=device)
            history_indices, history_mask = pad_histories(history_lists, device)
            selector_features = selector_feature_tensor(
                projected, candidate_indices, history_indices, history_mask,
                embeddings_device,
            )
            selected_positions = selector(selector_features).argmax(dim=1).tolist()
            for local_index, uid in enumerate(batch_uids):
                row = record_by_uid[uid]
                target = str(row["target"])
                slice_name = "cold" if target in cold_items else "warm"
                gram = unique_in_order(row["v0_top50"])
                resolver_top50 = unique_in_order(row["resolver_top50"])
                pool = cold_pools[local_index]
                base_candidate = pool[0]
                selected_candidate = pool[selected_positions[local_index]]
                if slice_name == "cold":
                    n_cold_users += 1
                    base_candidate_correct += int(base_candidate == target)
                    selector_candidate_correct += int(selected_candidate == target)
                    pool_target_hit += int(target in pool)
                fold = stable_fold(uid, len(fold_models))
                modeled = affected_items(gram, selected_candidate)
                p4_features = [
                    extract_item_feature_vector(
                        item, selected_candidate, projected[local_index],
                        embeddings_device, item_to_idx, item_routes, cold_items,
                        gram, resolver_top50,
                    )
                    for item in modeled
                ]
                probabilities = predict_gate(
                    fold_models[fold], torch.tensor(p4_features, dtype=torch.float32)
                ).tolist()
                policy_row = {
                    "v0_top50": gram,
                    "resolver_top50": resolver_top50,
                    "proposed_cold_item": selected_candidate,
                    "modeled_items": modeled,
                }
                best_action, best_utility, utilities = choose_best_action(policy_row, probabilities)
                action = (
                    best_action
                    if best_utility > 0 and best_utility >= fold_thresholds[fold]
                    else "abstain"
                )
                p5 = ranking_for_action(policy_row, action)
                pool_oracle_candidate = target if target in pool else selected_candidate
                oracle_row = {**policy_row, "proposed_cold_item": pool_oracle_candidate}
                pool_oracle = ranking_for_action(
                    oracle_row, "insert@7" if target in pool else "abstain"
                )
                label_oracle = resolver_top50 if slice_name == "cold" else gram
                rankings = {
                    "v0_gram": gram,
                    "resolver_only": resolver_top50,
                    "p5_setwise": p5,
                    "candidate_pool_oracle": pool_oracle,
                    "label_aware_oracle": label_oracle,
                }
                for model_name, ranking in rankings.items():
                    if len(ranking) != len(set(ranking)) or not set(ranking) <= catalog:
                        raise RuntimeError(f"Invalid ranking for {uid}/{model_name}")
                    metrics = ranking_metrics(ranking, target)
                    metric_rows[model_name]["all"].append(metrics)
                    metric_rows[model_name][slice_name].append(metrics)
                actions.append(action)
                predictions.append({
                    "user_id": uid,
                    "fold": fold,
                    "target": target,
                    "is_cold": target in cold_items,
                    "cold_candidate_pool": pool,
                    "base_candidate": base_candidate,
                    "selected_candidate": selected_candidate,
                    "pool_contains_target": target in pool,
                    "selected_candidate_is_target": selected_candidate == target,
                    "predicted_action_utilities": utilities,
                    "selected_action": action,
                    "p5_top50": p5[:50],
                })
            print(f"[validation] {min(offset + args.batch_size, len(ordered_uids))}/{len(ordered_uids)}", flush=True)

    metrics = summarize(metric_rows)
    v0 = metrics["v0_gram"]
    p5 = metrics["p5_setwise"]
    coverage = sum(action != "abstain" for action in actions) / len(actions)
    action_counts = {name: actions.count(name) for name in ("abstain", "insert@7", "insert@10")}
    candidate_report = {
        "n_cold_users": n_cold_users,
        "resolver_filtered_top1_correct": base_candidate_correct,
        "resolver_filtered_top1_accuracy": base_candidate_correct / n_cold_users,
        "selector_top1_correct": selector_candidate_correct,
        "selector_top1_accuracy": selector_candidate_correct / n_cold_users,
        "selector_vs_resolver_ratio": selector_candidate_correct / max(base_candidate_correct, 1),
        "pool_target_hit": pool_target_hit,
        "pool_recall_at_10": pool_target_hit / n_cold_users,
        "pool_recall_vs_top1_ratio": pool_target_hit / max(base_candidate_correct, 1),
    }
    gates = {
        "candidate_pool_recall10_ge_5x_resolver_top1": candidate_report["pool_recall_vs_top1_ratio"] >= 5.0,
        "pseudo_cold_selector_accuracy_ge_1_10x_resolver": pseudo_audit["selector_vs_resolver_ratio"] >= 1.10,
        "real_cold_selector_correct_ge_1_5x_resolver": candidate_report["selector_vs_resolver_ratio"] >= 1.5,
        "warm_ndcg10_ge_0_97x_v0": p5["warm"]["ndcg@10"] >= 0.97 * v0["warm"]["ndcg@10"],
        "cold_ndcg10_ge_2x_v0": p5["cold"]["ndcg@10"] >= 2.0 * v0["cold"]["ndcg@10"],
        "cold_hit10_ge_2x_v0": p5["cold"]["hit@10"] >= 2.0 * v0["cold"]["hit@10"],
        "all_ndcg10_gt_v0": p5["all"]["ndcg@10"] > v0["all"]["ndcg@10"],
        "intervention_coverage_lt_0_80": coverage < 0.80,
        "catalog_outputs_unique": all(
            len(row["p5_top50"]) == len(set(row["p5_top50"]))
            and set(row["p5_top50"]) <= catalog for row in predictions
        ),
        "selector_train_targets_all_warm": all(row[2] in warm_items for row in train_transitions),
        "pseudo_and_train_target_items_disjoint": not (selector_train_items & pseudo_cold_items),
        "validation_only": True,
    }
    verdict = (
        "PASS_TO_R2_P5_SET_NEW_DOMAIN_CONFIRMATION_DISCUSSION"
        if all(gates.values()) else "FAIL_STOP_R2_P5_SET"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "verdict": verdict,
        "evaluation_status": "warm_only_selector_plus_validation_held_fold_risk_gate",
        "metrics": metrics,
        "gates": gates,
        "train_export": train_export,
        "pseudo_cold_export": pseudo_export,
        "pseudo_cold_audit": pseudo_audit,
        "candidate_report": candidate_report,
        "diagnostics": {
            "intervention_coverage": coverage,
            "action_counts": action_counts,
            "training_history": training_history,
        },
        "test_predictions_opened": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions_validation.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[result] verdict={verdict} pseudo_ratio={pseudo_audit['selector_vs_resolver_ratio']:.3f} "
        f"real_ratio={candidate_report['selector_vs_resolver_ratio']:.3f} "
        f"warm={p5['warm']['ndcg@10']:.6f} cold={p5['cold']['ndcg@10']:.6f} "
        f"all={p5['all']['ndcg@10']:.6f}", flush=True,
    )


if __name__ == "__main__":
    main()
