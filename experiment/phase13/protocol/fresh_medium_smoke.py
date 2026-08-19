"""One-shot fresh Toys test medium smoke for the frozen v1-R² P3 model.

The 1,000-user sample is selected only from user IDs and persisted before the
test prediction file is opened.  The parser rejects/ignores every unselected
row before decoding any metric, gold, prediction, or score field.  Nothing in
this program fits a model or chooses a threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from anchored_interleaving import anchored_interleave
from confidence_abstention import (
    COLD_QUOTA,
    FEATURE_NAMES,
    PROTECTED_PREFIX,
    auc_roc,
    extract_inference_features,
    predict_gate,
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
    recency_weighted_history,
    semantic_route,
    sha256_file,
    unique_in_order,
)


FROZEN_THRESHOLD = 0.3266778290271759
MEDIUM_SAMPLE_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--gram-test-predictions", required=True)
    parser.add_argument("--item-id-file", required=True)
    parser.add_argument("--item-embeddings", required=True)
    parser.add_argument("--resolver-checkpoint", required=True)
    parser.add_argument("--confidence-gates", required=True)
    parser.add_argument("--cold-items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-size", type=int, default=MEDIUM_SAMPLE_SIZE)
    parser.add_argument("--frozen-threshold", type=float, default=FROZEN_THRESHOLD)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--recency-decay", type=float, default=0.85)
    return parser.parse_args()


def read_sequence_uids_only(path: Path) -> list[str]:
    """Read only the first token, without materialising any sequence target."""
    uids: list[str] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line_no, raw in enumerate(handle, 1):
            uid = raw.split(maxsplit=1)[0] if raw.strip() else ""
            if not uid:
                continue
            if uid in seen:
                raise ValueError(f"Duplicate sequence user {uid} at {path}:{line_no}")
            seen.add(uid)
            uids.append(uid)
    return uids


def select_medium_uids(user_ids: list[str], sample_size: int) -> list[str]:
    if sample_size <= 0 or sample_size > len(user_ids):
        raise ValueError(f"Invalid sample size {sample_size} for {len(user_ids)} users")
    return sorted(
        user_ids,
        key=lambda uid: (hashlib.sha256(uid.encode("utf-8")).hexdigest(), uid),
    )[:sample_size]


def selection_manifest(selected_uids: list[str], population_size: int) -> dict:
    uid_hashes = [hashlib.sha256(uid.encode("utf-8")).hexdigest() for uid in selected_uids]
    digest = hashlib.sha256(
        "".join(f"{uid}\t{uid_hash}\n" for uid, uid_hash in zip(selected_uids, uid_hashes)).encode()
    ).hexdigest()
    return {
        "selection_rule": "ascending (sha256(user_id), user_id); first N",
        "selection_inputs": ["user_id"],
        "target_used_for_selection": False,
        "population_size": population_size,
        "sample_size": len(selected_uids),
        "selected_users": [
            {"user_id": uid, "user_id_sha256": uid_hash}
            for uid, uid_hash in zip(selected_uids, uid_hashes)
        ],
        "selection_manifest_sha256": digest,
        "written_before_test_prediction_open": True,
    }


def read_selected_test_predictions(path: Path, selected_uids: set[str]) -> tuple[dict[str, list[str]], dict]:
    """Parse prediction beams only after an early UID-only membership check."""
    if "test" not in path.name.lower():
        raise ValueError(f"Expected an explicitly named test prediction file: {path}")
    rows: dict[str, list[str]] = {}
    total_candidate_rows = 0
    outside_rows_skipped_before_field_parse = 0
    selected_rows_parsed = 0
    with path.open() as handle:
        for line_no, raw in enumerate(handle, 1):
            uid, separator, remainder = raw.rstrip("\n").partition("\t")
            if not separator or uid not in selected_uids:
                if separator and uid not in ("idx", "hit@1", "ndcg@1"):
                    outside_rows_skipped_before_field_parse += 1
                continue
            total_candidate_rows += 1
            fields = [uid, *remainder.split("\t")]
            if len(fields) < 16:
                raise ValueError(f"Malformed selected test row at {path}:{line_no}")
            predictions = fields[14].split("||") if fields[14] else []
            if not predictions:
                raise ValueError(f"Empty selected test beam at {path}:{line_no}")
            if uid in rows:
                raise ValueError(f"Duplicate selected test user {uid} at {path}:{line_no}")
            rows[uid] = predictions
            selected_rows_parsed += 1
    missing = selected_uids - rows.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} selected test users; examples={sorted(missing)[:5]}")
    audit = {
        "selected_rows_parsed": selected_rows_parsed,
        "selected_candidate_rows_seen": total_candidate_rows,
        "outside_rows_skipped_before_field_parse": outside_rows_skipped_before_field_parse,
        "outside_sample_metric_or_prediction_rows_parsed": 0,
        "saved_test_metric_fields_parsed": False,
        "saved_test_gold_fields_parsed": False,
    }
    return rows, audit


def build_test_examples(
    sequences: list[tuple[str, list[str]]],
    selected_uids: set[str],
    item_to_idx: dict[str, int],
    embeddings: torch.Tensor,
    max_history: int,
    recency_decay: float,
) -> dict[str, tuple[torch.Tensor, str]]:
    examples: dict[str, tuple[torch.Tensor, str]] = {}
    for uid, items in sequences:
        if uid not in selected_uids:
            continue
        if len(items) < 2:
            raise ValueError(f"Test user {uid} has no usable history")
        target = items[-1]
        history_items = items[max(0, len(items) - 1 - max_history):-1]
        try:
            history_indices = [item_to_idx[item] for item in history_items]
        except KeyError as error:
            raise ValueError(f"Non-catalog history item for {uid}: {error}") from error
        if target not in item_to_idx:
            raise ValueError(f"Non-catalog test target for {uid}: {target}")
        examples[uid] = (
            recency_weighted_history(history_indices, embeddings, recency_decay),
            target,
        )
    missing = selected_uids - examples.keys()
    if missing:
        raise ValueError(f"Missing sequence rows for {len(missing)} selected users")
    return examples


def summarize(metric_rows: dict[str, dict[str, list[dict[str, float]]]]) -> dict:
    return {
        model: {slice_name: average_metrics(values) for slice_name, values in slices.items()}
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
    gates_path = Path(args.confidence_gates).resolve()
    cold_path = Path(args.cold_items).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [
        path.name for path in output_dir.iterdir()
        if path.name not in allowed
        and not (path.name.startswith("status.") and path.name.endswith(".json"))
    ]
    if unexpected:
        raise FileExistsError(f"Refusing existing medium-smoke artifacts: {unexpected}")
    required = [
        sequence_path, test_path, item_id_path, embedding_path,
        resolver_path, gates_path, cold_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    # This is deliberately the only data read before the manifest is durable.
    population_uids = read_sequence_uids_only(sequence_path)
    selected_uids = select_medium_uids(population_uids, args.sample_size)
    manifest = selection_manifest(selected_uids, len(population_uids))
    manifest_path = output_dir / "selection_manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_json(output_dir / "test_access.json", {"test_predictions_opened": False})
    print(f"[selection] manifest written for {len(selected_uids)} users before test open", flush=True)

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
    item_routes = {item: semantic_route(lexical, 1) for item, lexical in item_to_lexical.items()}

    embedding_payload = torch.load(embedding_path, map_location="cpu")
    item_ids = list(embedding_payload["item_ids"])
    embeddings_cpu = F.normalize(embedding_payload["embeddings"].float(), dim=1)
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    if set(item_ids) != catalog or not cold_items <= catalog:
        raise ValueError("Catalog, embedding, or cold-state mismatch")

    checkpoint = torch.load(resolver_path, map_location="cpu")
    resolver = ResidualUserProjector(
        checkpoint["dim"], checkpoint["hidden_dim"], checkpoint["dropout"]
    )
    resolver.load_state_dict(checkpoint["state_dict"])
    resolver.eval()

    gate_payload = torch.load(gates_path, map_location="cpu")
    selected_gate = gate_payload.get("full_selected_threshold")
    saved_threshold = None if selected_gate is None else float(selected_gate["threshold"])
    if saved_threshold is None or abs(saved_threshold - args.frozen_threshold) > 1e-12:
        raise ValueError(
            f"Frozen threshold mismatch: checkpoint={saved_threshold} cli={args.frozen_threshold}"
        )
    if gate_payload["protected_prefix"] != PROTECTED_PREFIX or gate_payload["cold_quota"] != COLD_QUOTA:
        raise ValueError("Frozen P3 anchor mismatch")
    if tuple(gate_payload["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Frozen P3 feature schema mismatch")
    gate_model = gate_payload["full_model"]

    sequences = read_sequences(sequence_path)
    examples = build_test_examples(
        sequences, selected_set, item_to_idx, embeddings_cpu,
        args.max_history, args.recency_decay,
    )
    ordered_uids = selected_uids
    decoded_gram: dict[str, list[str]] = {}
    for uid in ordered_uids:
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
    model_names = ("v0_gram", "resolver_only", "p3_abstention", "label_aware_oracle")
    metric_rows = {
        model: {slice_name: [] for slice_name in ("all", "warm", "cold")}
        for model in model_names
    }
    prediction_rows: list[dict] = []
    labels: list[int] = []
    probabilities: list[float] = []
    admitted: list[bool] = []
    with torch.no_grad():
        for offset in range(0, len(ordered_uids), 256):
            batch_uids = ordered_uids[offset:offset + 256]
            histories = torch.stack([examples[uid][0] for uid in batch_uids]).to(device)
            projected = resolver(histories)
            resolver_indices = torch.topk(
                projected @ embeddings_device.T, k=min(50, len(item_ids)), dim=1
            ).indices
            for row_index, uid in enumerate(batch_uids):
                target = examples[uid][1]
                slice_name = "cold" if target in cold_items else "warm"
                gram = decoded_gram[uid]
                resolver_top50 = [item_ids[index] for index in resolver_indices[row_index].tolist()]
                feature_row = {
                    "user_id": uid,
                    "v0_top50": gram,
                    "resolver_top50": resolver_top50,
                }
                candidate, features = extract_inference_features(
                    feature_row, projected[row_index], embeddings_device,
                    item_to_idx, item_routes, cold_items,
                )
                probability = float(
                    predict_gate(gate_model, torch.tensor([features], dtype=torch.float32))[0]
                )
                use_slot = probability >= args.frozen_threshold
                p3 = (
                    anchored_interleave(
                        gram, resolver_top50, cold_items, PROTECTED_PREFIX, COLD_QUOTA
                    )[0]
                    if use_slot else gram
                )
                oracle = resolver_top50 if slice_name == "cold" else gram
                rankings = {
                    "v0_gram": gram,
                    "resolver_only": resolver_top50,
                    "p3_abstention": p3,
                    "label_aware_oracle": oracle,
                }
                for model_name, ranking in rankings.items():
                    if len(ranking) != len(set(ranking)) or not set(ranking) <= catalog:
                        raise RuntimeError(f"Invalid ranking for {uid}/{model_name}")
                    metrics = ranking_metrics(ranking, target)
                    metric_rows[model_name]["all"].append(metrics)
                    metric_rows[model_name][slice_name].append(metrics)
                is_correct = int(candidate == target)
                labels.append(is_correct)
                probabilities.append(probability)
                admitted.append(use_slot)
                prediction_rows.append({
                    "user_id": uid,
                    "target": target,
                    "is_cold": slice_name == "cold",
                    "proposed_cold_item": candidate,
                    "candidate_is_target": bool(is_correct),
                    "gate_probability": probability,
                    "admitted": use_slot,
                    "p3_top50": p3[:50],
                })
            print(f"[eval] {min(offset + 256, len(ordered_uids))}/{len(ordered_uids)}", flush=True)

    metrics = summarize(metric_rows)
    v0 = metrics["v0_gram"]
    p3 = metrics["p3_abstention"]
    coverage = sum(admitted) / len(admitted)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32)
    auc = auc_roc(labels_tensor, probabilities_tensor)
    admitted_count = sum(admitted)
    admitted_correct = sum(label and choice for label, choice in zip(labels, admitted))
    slice_counts = {
        name: len(metric_rows["p3_abstention"][name]) for name in ("all", "warm", "cold")
    }
    gates = {
        "sample_size_eq_1000": len(ordered_uids) == MEDIUM_SAMPLE_SIZE,
        "selection_target_free": manifest["target_used_for_selection"] is False,
        "manifest_written_before_test_open": manifest["written_before_test_prediction_open"],
        "outside_sample_rows_not_parsed": parser_audit["outside_sample_metric_or_prediction_rows_parsed"] == 0,
        "catalog_outputs_unique": True,
        "warm_ndcg10_ge_0_97x_v0": p3["warm"]["ndcg@10"] >= 0.97 * v0["warm"]["ndcg@10"],
        "cold_ndcg10_ge_1_5x_v0": p3["cold"]["ndcg@10"] >= 1.5 * v0["cold"]["ndcg@10"],
        "cold_hit10_ge_1_5x_v0": p3["cold"]["hit@10"] >= 1.5 * v0["cold"]["hit@10"],
        "all_ndcg10_gt_v0": p3["all"]["ndcg@10"] > v0["all"]["ndcg@10"],
        "admission_coverage_in_0_40_0_80": 0.40 <= coverage <= 0.80,
    }
    verdict = (
        "PASS_TO_R2_FULL_CONFIRMATION_DISCUSSION"
        if all(gates.values()) else "FAIL_STOP_R2_FRESH_MEDIUM_SMOKE"
    )
    config = {
        **vars(args),
        "dataset_dir": str(dataset_dir),
        "gram_test_predictions": str(test_path),
        "item_id_file": str(item_id_path),
        "item_embeddings": str(embedding_path),
        "resolver_checkpoint": str(resolver_path),
        "confidence_gates": str(gates_path),
        "cold_items": str(cold_path),
        "output_dir": str(output_dir),
        "experiment_id": "GRAM_PHASE13_V1_R2_TOYS_P3_FRESH_MEDIUM_SMOKE",
        "split": "test_hash_medium_1000_one_shot",
        "threshold_source": "frozen P3 full_selected_threshold",
        "protected_prefix": PROTECTED_PREFIX,
        "cold_quota": COLD_QUOTA,
        "feature_names": FEATURE_NAMES,
        "test_predictions_opened": True,
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "input_sha256": {str(path): sha256_file(path) for path in required},
    }
    atomic_json(output_dir / "config.json", config)
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "verdict": verdict,
        "evaluation_status": "fresh_test_hash_medium_one_shot_not_full_confirmation",
        "sample_size": len(ordered_uids),
        "slice_counts": slice_counts,
        "metrics": metrics,
        "gates": gates,
        "candidate_correctness_diagnostic": {
            "n_positive": int(labels_tensor.sum()),
            "base_rate": float(labels_tensor.float().mean()),
            "auroc": auc,
            "admission_coverage": coverage,
            "admitted_precision": admitted_correct / max(admitted_count, 1),
        },
        "parser_audit": parser_audit,
        "test_predictions_opened": True,
        "no_test_tuning_or_training": True,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "predictions_test_medium.jsonl").open("w") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[result] verdict={verdict} coverage={coverage:.4f} "
        f"warm={p3['warm']['ndcg@10']:.6f} cold={p3['cold']['ndcg@10']:.6f} "
        f"all={p3['all']['ndcg@10']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
