"""One-shot disjoint Toys test confirmation for the frozen v1-R² P4 policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from confidence_abstention import auc_roc, predict_gate
from counterfactual_slot_router import (
    ACTION_POSITIONS,
    FEATURE_NAMES,
    affected_items,
    choose_best_action,
    extract_item_feature_vector,
    ranking_for_action,
)
from fresh_medium_smoke import (
    build_test_examples,
    read_selected_test_predictions,
    read_sequence_uids_only,
)
from route_resolve import (
    ResidualUserProjector,
    atomic_json,
    average_metrics,
    decode_lexical_id,
    ranking_metrics,
    read_key_value_lines,
    read_sequences,
    read_set,
    semantic_route,
    sha256_file,
    unique_in_order,
)


TRANCHE_START = 1000
TRANCHE_SIZE = 1000
FROZEN_UTILITY_THRESHOLD = 0.06854262202978134


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--gram-test-predictions", required=True)
    parser.add_argument("--item-id-file", required=True)
    parser.add_argument("--item-embeddings", required=True)
    parser.add_argument("--resolver-checkpoint", required=True)
    parser.add_argument("--p4-checkpoint", required=True)
    parser.add_argument("--previous-selection-manifest", required=True)
    parser.add_argument("--cold-items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tranche-start", type=int, default=TRANCHE_START)
    parser.add_argument("--sample-size", type=int, default=TRANCHE_SIZE)
    parser.add_argument("--frozen-utility-threshold", type=float, default=FROZEN_UTILITY_THRESHOLD)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--recency-decay", type=float, default=0.85)
    return parser.parse_args()


def ordered_uid_population(user_ids: list[str]) -> list[str]:
    return sorted(
        user_ids,
        key=lambda uid: (hashlib.sha256(uid.encode("utf-8")).hexdigest(), uid),
    )


def select_uid_tranche(user_ids: list[str], start: int, sample_size: int) -> list[str]:
    ordered = ordered_uid_population(user_ids)
    if start < 0 or sample_size <= 0 or start + sample_size > len(ordered):
        raise ValueError(
            f"Invalid tranche [{start}, {start + sample_size}) for {len(ordered)} users"
        )
    return ordered[start : start + sample_size]


def build_disjoint_manifest(
    selected_uids: list[str],
    population_size: int,
    tranche_start: int,
    previous_manifest: dict,
    previous_manifest_path: Path,
) -> dict:
    previous_uids = {
        str(row["user_id"]) for row in previous_manifest.get("selected_users", [])
    }
    overlap = sorted(previous_uids & set(selected_uids))
    if overlap:
        raise ValueError(f"Disjoint tranche overlaps previous sample: {overlap[:5]}")
    uid_hashes = [hashlib.sha256(uid.encode()).hexdigest() for uid in selected_uids]
    digest = hashlib.sha256(
        "".join(
            f"{uid}\t{uid_hash}\n"
            for uid, uid_hash in zip(selected_uids, uid_hashes)
        ).encode()
    ).hexdigest()
    return {
        "selection_rule": "ascending (sha256(user_id), user_id); zero-based slice [start,start+N)",
        "selection_inputs": ["user_id"],
        "target_used_for_selection": False,
        "population_size": population_size,
        "tranche_start_zero_based": tranche_start,
        "tranche_rank_one_based": [tranche_start + 1, tranche_start + len(selected_uids)],
        "sample_size": len(selected_uids),
        "previous_sample_size": len(previous_uids),
        "previous_sample_overlap": 0,
        "previous_selection_manifest": str(previous_manifest_path.resolve()),
        "previous_selection_manifest_sha256": sha256_file(previous_manifest_path),
        "selected_users": [
            {"user_id": uid, "user_id_sha256": uid_hash}
            for uid, uid_hash in zip(selected_uids, uid_hashes)
        ],
        "selection_manifest_sha256": digest,
        "written_before_test_prediction_open": True,
    }


def summarize(metric_rows: dict[str, dict[str, list[dict[str, float]]]]) -> dict:
    return {
        model: {
            slice_name: average_metrics(values)
            for slice_name, values in slices.items()
        }
        for model, slices in metric_rows.items()
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    dataset_dir = Path(args.dataset_dir).resolve()
    sequence_path = dataset_dir / "user_sequence.txt"
    test_path = Path(args.gram_test_predictions).resolve()
    item_id_path = Path(args.item_id_file).resolve()
    embedding_path = Path(args.item_embeddings).resolve()
    resolver_path = Path(args.resolver_checkpoint).resolve()
    p4_path = Path(args.p4_checkpoint).resolve()
    previous_manifest_path = Path(args.previous_selection_manifest).resolve()
    cold_path = Path(args.cold_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in allowed]
    if unexpected:
        raise FileExistsError(f"Refusing existing P4 confirmation artifacts: {unexpected}")
    required = [
        sequence_path, test_path, item_id_path, embedding_path, resolver_path,
        p4_path, previous_manifest_path, cold_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Only user IDs and the previous target-free selection manifest are read
    # before the new manifest becomes durable.
    population_uids = read_sequence_uids_only(sequence_path)
    ordered_population = ordered_uid_population(population_uids)
    selected_uids = select_uid_tranche(
        population_uids, args.tranche_start, args.sample_size
    )
    previous_manifest = json.loads(previous_manifest_path.read_text())
    previous_uids = [
        str(row["user_id"]) for row in previous_manifest.get("selected_users", [])
    ]
    if previous_uids != ordered_population[: args.tranche_start]:
        raise ValueError("Previous manifest is not the exact preceding UID tranche")
    manifest = build_disjoint_manifest(
        selected_uids, len(population_uids), args.tranche_start,
        previous_manifest, previous_manifest_path,
    )
    atomic_json(output_dir / "selection_manifest.json", manifest)
    atomic_json(output_dir / "test_access.json", {"test_predictions_opened": False})
    print(
        f"[selection] disjoint manifest written for ranks "
        f"{args.tranche_start + 1}-{args.tranche_start + args.sample_size} before test open",
        flush=True,
    )

    selected_set = set(selected_uids)
    gram_beams, parser_audit = read_selected_test_predictions(test_path, selected_set)
    atomic_json(output_dir / "test_access.json", {"test_predictions_opened": True})
    print(f"[test] parsed selected rows={len(gram_beams)} outside parsed=0", flush=True)

    item_to_lexical = read_key_value_lines(item_id_path)
    catalog = set(item_to_lexical)
    decoded_to_item: dict[str, str] = {}
    for item, lexical in item_to_lexical.items():
        decoded = decode_lexical_id(lexical)
        if decoded in decoded_to_item:
            raise ValueError(f"Decoded semantic ID collision for {item}")
        decoded_to_item[decoded] = item
    cold_items = read_set(cold_path)
    item_routes = {
        item: semantic_route(lexical, 1)
        for item, lexical in item_to_lexical.items()
    }

    embedding_payload = torch.load(embedding_path, map_location="cpu")
    item_ids = list(embedding_payload["item_ids"])
    embeddings_cpu = F.normalize(embedding_payload["embeddings"].float(), dim=1)
    item_to_idx = {item: index for index, item in enumerate(item_ids)}
    if set(item_ids) != catalog or not cold_items <= catalog:
        raise ValueError("Catalog, embedding, or cold-state mismatch")

    resolver_checkpoint = torch.load(resolver_path, map_location="cpu")
    resolver = ResidualUserProjector(
        resolver_checkpoint["dim"], resolver_checkpoint["hidden_dim"],
        resolver_checkpoint["dropout"],
    )
    resolver.load_state_dict(resolver_checkpoint["state_dict"])
    resolver.eval()

    p4_checkpoint = torch.load(p4_path, map_location="cpu")
    selected_policy = p4_checkpoint.get("full_selected_policy")
    saved_threshold = None if selected_policy is None else float(selected_policy["threshold"])
    if saved_threshold is None or abs(saved_threshold - args.frozen_utility_threshold) > 1e-12:
        raise ValueError(
            f"Frozen P4 threshold mismatch: checkpoint={saved_threshold} "
            f"cli={args.frozen_utility_threshold}"
        )
    if tuple(p4_checkpoint["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Frozen P4 feature schema mismatch")
    if tuple(p4_checkpoint["action_positions"]) != ACTION_POSITIONS:
        raise ValueError("Frozen P4 action schema mismatch")
    relevance_model = p4_checkpoint["full_model"]

    sequences = read_sequences(sequence_path)
    examples = build_test_examples(
        sequences, selected_set, item_to_idx, embeddings_cpu,
        args.max_history, args.recency_decay,
    )
    decoded_gram: dict[str, list[str]] = {}
    for uid in selected_uids:
        items: list[str] = []
        for decoded in gram_beams[uid]:
            item = decoded_to_item.get(decoded)
            if item is None:
                raise KeyError(f"Could not map legal test beam for {uid}: {decoded!r}")
            items.append(item)
        decoded_gram[uid] = unique_in_order(items)[:50]

    device = torch.device(args.device)
    resolver.to(device)
    embeddings_device = embeddings_cpu.to(device)
    model_names = (
        "v0_gram", "resolver_only", "p4_counterfactual_slot_router",
        "label_aware_oracle",
    )
    metric_rows = {
        model: {slice_name: [] for slice_name in ("all", "warm", "cold")}
        for model in model_names
    }
    predictions: list[dict] = []
    candidate_labels: list[int] = []
    candidate_probabilities: list[float] = []
    relevance_labels: list[int] = []
    relevance_probabilities: list[float] = []
    actions: list[str] = []
    with torch.no_grad():
        for offset in range(0, len(selected_uids), 256):
            batch_uids = selected_uids[offset : offset + 256]
            histories = torch.stack([examples[uid][0] for uid in batch_uids]).to(device)
            projected = resolver(histories)
            resolver_indices = torch.topk(
                projected @ embeddings_device.T, k=min(50, len(item_ids)), dim=1
            ).indices
            for local_index, uid in enumerate(batch_uids):
                target = examples[uid][1]
                slice_name = "cold" if target in cold_items else "warm"
                gram = decoded_gram[uid]
                resolver_top50 = [
                    item_ids[index] for index in resolver_indices[local_index].tolist()
                ]
                protected = set(gram[:6])
                eligible = [
                    item for item in resolver_top50
                    if item in cold_items and item not in protected
                ]
                if not eligible:
                    raise ValueError(f"No eligible cold proposal for {uid}")
                candidate = eligible[0]
                modeled = affected_items(gram, candidate)
                features = [
                    extract_item_feature_vector(
                        item, candidate, projected[local_index], embeddings_device,
                        item_to_idx, item_routes, cold_items, gram, resolver_top50,
                    )
                    for item in modeled
                ]
                probabilities = predict_gate(
                    relevance_model, torch.tensor(features, dtype=torch.float32)
                ).tolist()
                policy_row = {
                    "v0_top50": gram,
                    "resolver_top50": resolver_top50,
                    "proposed_cold_item": candidate,
                    "modeled_items": modeled,
                }
                best_action, best_utility, utilities = choose_best_action(
                    policy_row, probabilities
                )
                action = (
                    best_action
                    if best_utility > 0.0
                    and best_utility >= args.frozen_utility_threshold
                    else "abstain"
                )
                p4_ranking = ranking_for_action(policy_row, action)
                oracle = resolver_top50 if slice_name == "cold" else gram
                rankings = {
                    "v0_gram": gram,
                    "resolver_only": resolver_top50,
                    "p4_counterfactual_slot_router": p4_ranking,
                    "label_aware_oracle": oracle,
                }
                for model_name, ranking in rankings.items():
                    if len(ranking) != len(set(ranking)) or not set(ranking) <= catalog:
                        raise RuntimeError(f"Invalid ranking for {uid}/{model_name}")
                    metrics = ranking_metrics(ranking, target)
                    metric_rows[model_name]["all"].append(metrics)
                    metric_rows[model_name][slice_name].append(metrics)
                for item, probability in zip(modeled, probabilities):
                    relevance_labels.append(int(item == target))
                    relevance_probabilities.append(float(probability))
                candidate_index = modeled.index(candidate)
                candidate_labels.append(int(candidate == target))
                candidate_probabilities.append(float(probabilities[candidate_index]))
                actions.append(action)
                predictions.append({
                    "user_id": uid,
                    "target": target,
                    "is_cold": target in cold_items,
                    "proposed_cold_item": candidate,
                    "candidate_is_target": candidate == target,
                    "candidate_relevance_probability": float(probabilities[candidate_index]),
                    "predicted_action_utilities": utilities,
                    "selected_action": action,
                    "p4_top50": p4_ranking[:50],
                })
            print(f"[eval] {min(offset + 256, len(selected_uids))}/{len(selected_uids)}", flush=True)

    metrics = summarize(metric_rows)
    v0 = metrics["v0_gram"]
    p4 = metrics["p4_counterfactual_slot_router"]
    coverage = sum(action != "abstain" for action in actions) / len(actions)
    action_counts = {
        action: actions.count(action)
        for action in ("abstain", "insert@7", "insert@10")
    }
    candidate_label_tensor = torch.tensor(candidate_labels, dtype=torch.int64)
    candidate_probability_tensor = torch.tensor(candidate_probabilities)
    relevance_label_tensor = torch.tensor(relevance_labels, dtype=torch.int64)
    relevance_probability_tensor = torch.tensor(relevance_probabilities)
    candidate_auc = auc_roc(candidate_label_tensor, candidate_probability_tensor)
    relevance_auc = auc_roc(relevance_label_tensor, relevance_probability_tensor)
    gates = {
        "sample_size_eq_1000": len(selected_uids) == TRANCHE_SIZE,
        "previous_sample_overlap_eq_0": manifest["previous_sample_overlap"] == 0,
        "selection_target_free": manifest["target_used_for_selection"] is False,
        "manifest_written_before_test_open": manifest["written_before_test_prediction_open"],
        "outside_sample_rows_not_parsed": parser_audit["outside_sample_metric_or_prediction_rows_parsed"] == 0,
        "catalog_outputs_unique": True,
        "warm_ndcg10_ge_0_97x_v0": p4["warm"]["ndcg@10"] >= 0.97 * v0["warm"]["ndcg@10"],
        "cold_ndcg10_ge_1_5x_v0": p4["cold"]["ndcg@10"] >= 1.5 * v0["cold"]["ndcg@10"],
        "cold_hit10_ge_1_5x_v0": p4["cold"]["hit@10"] >= 1.5 * v0["cold"]["hit@10"],
        "all_ndcg10_gt_v0": p4["all"]["ndcg@10"] > v0["all"]["ndcg@10"],
        "intervention_coverage_in_0_35_0_65": 0.35 <= coverage <= 0.65,
    }
    verdict = (
        "PASS_TO_R2_FULL_TEST_OR_BEAUTY_DISCUSSION"
        if all(gates.values())
        else "FAIL_STOP_R2_P4_DISJOINT_CONFIRMATION"
    )
    config = {
        **vars(args),
        "dataset_dir": str(dataset_dir),
        "gram_test_predictions": str(test_path),
        "item_id_file": str(item_id_path),
        "item_embeddings": str(embedding_path),
        "resolver_checkpoint": str(resolver_path),
        "p4_checkpoint": str(p4_path),
        "previous_selection_manifest": str(previous_manifest_path),
        "cold_items": str(cold_path),
        "output_dir": str(output_dir),
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P4_DISJOINT_CONFIRMATION",
        "split": "test_hash_tranche_ranks_1001_2000_one_shot",
        "threshold_source": "frozen P4 full_selected_policy",
        "feature_names": FEATURE_NAMES,
        "action_positions": ACTION_POSITIONS,
        "test_predictions_opened": True,
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "input_sha256": {str(path): sha256_file(path) for path in required},
    }
    atomic_json(output_dir / "config.json", config)
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "verdict": verdict,
        "evaluation_status": "disjoint_test_hash_tranche_one_shot_not_full_test",
        "sample_size": len(selected_uids),
        "slice_counts": {
            name: len(metric_rows["p4_counterfactual_slot_router"][name])
            for name in ("all", "warm", "cold")
        },
        "metrics": metrics,
        "gates": gates,
        "diagnostics": {
            "intervention_coverage": coverage,
            "action_counts": action_counts,
            "n_candidate_positive": int(candidate_label_tensor.sum()),
            "candidate_correctness_auroc": candidate_auc,
            "n_relevance_positive": int(relevance_label_tensor.sum()),
            "shared_relevance_auroc": relevance_auc,
        },
        "parser_audit": parser_audit,
        "test_predictions_opened": True,
        "no_test_tuning_or_training": True,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions_test_disjoint.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[result] verdict={verdict} coverage={coverage:.4f} actions={action_counts} "
        f"warm={p4['warm']['ndcg@10']:.6f} cold={p4['cold']['ndcg@10']:.6f} "
        f"all={p4['all']['ndcg@10']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
