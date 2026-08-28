"""Stage15 S3B Toys full validation for exploratory GenRecEdit-GRAM B3.

The job rebuilds B3 state from the frozen full cold catalog and train-only
warm contexts before opening frozen validation rankings.  B0/B1 are replayed
from the same Phase13 validation artifact used by the concurrent B2 run; B3
uses the admitted One-One generation hook on the corresponding GRAM v0.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE14_PROTOCOL = REPO_ROOT / "experiment" / "phase14" / "protocol"
if str(PHASE14_PROTOCOL) not in sys.path:
    sys.path.insert(0, str(PHASE14_PROTOCOL))

from item_level_eval import atomic_json  # noqa: E402
from oracle_prefix_probe import CollatorGRAM  # noqa: E402
from r2pd_pseudo_cold_screen import (  # noqa: E402
    batch_to_device,
    build_filtered_item_inputs,
    collator_args,
    configure_fresh_model,
    load_paths,
    make_model_sample,
    read_key_value,
)

from common_adapter import (  # noqa: E402
    read_projected_sequences,
    read_validation_predictions,
    sha256_file,
    train_only_sequences,
)
from genrecedit_gram_adapter import OneOneGenerationDeltaContext  # noqa: E402
from specgr_gram_adapter import PathCatalog  # noqa: E402
from toys_b3_edit_state_smoke import _configure_determinism, _model_state_sha256  # noqa: E402
from toys_s3a_admission import (  # noqa: E402
    build_b3_state,
    build_pseudo_contexts,
    encoded_catalog,
    generate_beam,
    load_embeddings,
    probe_clean_base_layers,
)
from toys_s3b_full_validation import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    ensure_new_outputs,
    paired_bootstrap,
    portfolio_at_2,
    ranking_metrics,
    read_set,
    selected_train_rows,
    success_labels,
    summarize_arm,
    unique_in_order,
)


ARMS = ("b0", "b1", "b3")
EXPECTED_ADMISSION_VERDICT = "PASS_S15_3A_B2_B3_ITEM_DISJOINT_ADMISSION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="GRAM_STAGE15_S3B_TOYS_B3_FULL_VALIDATION_SEED0")
    parser.add_argument("--domain", default="Toys_cold50")
    parser.add_argument("--completed-verdict", default="COMPLETED_S15_3B_TOYS_B3_FULL_VALIDATION")
    parser.add_argument("--required-events", type=int, default=8789)
    parser.add_argument("--progress-marker", default="s3b-b3-eval")
    parser.add_argument("--projected-sequences", type=Path, required=True)
    parser.add_argument("--historical-config", type=Path, required=True)
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-b0-b1-predictions", type=Path, required=True)
    parser.add_argument("--item-path-file", type=Path, required=True)
    parser.add_argument("--item-text-file", type=Path, required=True)
    parser.add_argument("--similar-items-file", type=Path, required=True)
    parser.add_argument("--item-embeddings", type=Path, required=True)
    parser.add_argument("--cold-items", type=Path, required=True)
    parser.add_argument("--warm-items", type=Path, required=True)
    parser.add_argument("--s3a-b3-summary", type=Path, required=True)
    parser.add_argument("--b0-parity-summary", type=Path, required=True)
    parser.add_argument("--b1-source-summary", type=Path, required=True)
    parser.add_argument("--b1-state", type=Path, required=True)
    parser.add_argument("--frozen-contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-transitions", type=int, default=4096)
    parser.add_argument("--covariance-transitions", type=int, default=256)
    parser.add_argument("--covariance-long-path-minimum", type=int, default=32)
    parser.add_argument("--covariance-batch-size", type=int, default=32)
    parser.add_argument("--contexts-per-pseudo-cold", type=int, default=10)
    parser.add_argument("--requests-per-position", type=int, default=4)
    parser.add_argument("--z-steps", type=int, default=30)
    parser.add_argument("--similar-top-k", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--seed", type=int, default=1502)
    return parser.parse_args()


def validate_frozen_contract(args: argparse.Namespace, admission: dict) -> None:
    expected = {
        "train_transitions": 4096,
        "covariance_transitions": 256,
        "covariance_long_path_minimum": 32,
        "covariance_batch_size": 32,
        "contexts_per_pseudo_cold": 10,
        "requests_per_position": 4,
        "z_steps": 30,
        "beam_size": 50,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 20260822,
        "seed": 1502,
    }
    observed = {name: getattr(args, name) for name in expected}
    if observed != expected:
        raise ValueError(f"B3 full-validation contract drift: {observed}")
    if admission.get("verdict") != EXPECTED_ADMISSION_VERDICT:
        raise ValueError("Exploratory B3 S15-3A admission is not PASS")
    checks = admission.get("admission_checks", {})
    required_checks = {
        "all_rankings_unique_known_top50",
        "base_hash_unchanged",
        "held_ground_truth_not_used_for_training_or_state_selection",
        "test_not_opened",
        "b3_complete_one_one_edited_beam_path",
        "b3_delta_finite",
        "b3_delta_nonzero",
        "b3_every_position_exercised",
    }
    if any(checks.get(name) is not True for name in required_checks):
        raise ValueError("Exploratory B3 admission checks are incomplete")
    if admission.get("seed") != expected["seed"] or admission.get("beam_size") != 50:
        raise ValueError("B3 admission seed/beam contract differs")
    if admission.get("b3_covariance_transitions") != 256:
        raise ValueError("B3 admission covariance budget differs")
    if admission.get("b3_requests_per_position") != 4:
        raise ValueError("B3 admission request budget differs")


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    output_dir = args.output_dir.resolve()
    ensure_new_outputs(output_dir)

    paths = {
        name: Path(value).resolve()
        for name, value in {
            "projected_sequences": args.projected_sequences,
            "historical_config": args.historical_config,
            "checkpoint": args.checkpoint,
            "frozen_b0_b1_validation_predictions": args.frozen_b0_b1_predictions,
            "item_path_file": args.item_path_file,
            "item_text_file": args.item_text_file,
            "similar_items_b0_historical_only": args.similar_items_file,
            "item_embeddings": args.item_embeddings,
            "cold_items": args.cold_items,
            "warm_items": args.warm_items,
            "s3a_b3_summary": args.s3a_b3_summary,
            "b0_parity_summary": args.b0_parity_summary,
            "b1_source_summary": args.b1_source_summary,
            "b1_state": args.b1_state,
        }.items()
    }
    if args.frozen_contract is not None:
        paths["frozen_contract"] = args.frozen_contract.resolve()
    backbone = args.backbone_path.resolve()
    if paths["projected_sequences"].name != "user_sequence_train_validation.txt":
        raise ValueError("B3 full validation requires the audited projection")
    if "test" in paths["frozen_b0_b1_validation_predictions"].name.lower():
        raise ValueError("Refusing test predictions")
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Input must be a regular non-symlink file: {name}={path}")
    if not backbone.is_dir():
        raise FileNotFoundError(backbone)

    admission = json.loads(paths["s3a_b3_summary"].read_text(encoding="utf-8"))
    validate_frozen_contract(args, admission)
    parity = json.loads(paths["b0_parity_summary"].read_text(encoding="utf-8"))
    if parity.get("verdict") != "PASS_B0_PROJECTION_PARITY":
        raise ValueError("B0 projection-parity Gate is not PASS")

    numerical_mode = _configure_determinism()
    projected = read_projected_sequences(paths["projected_sequences"])
    if len(projected) != args.required_events:
        raise ValueError(
            f"Projected event count drift: {len(projected)} != {args.required_events}"
        )
    train_sequences = train_only_sequences(projected)
    cold = read_set(paths["cold_items"])
    warm = read_set(paths["warm_items"])
    if cold & warm:
        raise ValueError("Cold and warm sets overlap")
    if any(item not in warm for items in train_sequences.values() for item in items):
        raise ValueError("Frozen cold item entered train-only B3 context")

    item_paths = load_paths(paths["item_path_file"])
    item_text = read_key_value(paths["item_text_file"])
    item_ids, embeddings, embedding_meta = load_embeddings(paths["item_embeddings"])
    if cold | warm != set(item_paths) or set(item_ids) != set(item_paths) or set(item_text) != set(item_paths):
        raise ValueError("Catalog/path/text/embedding/cold-warm universes differ")

    device = torch.device(args.device)
    train_rows = selected_train_rows(projected, args)
    if any(row["target"] not in warm for row in train_rows):
        raise ValueError("Cold target entered B3 covariance/probe supervision")
    catalog = PathCatalog(paths=item_paths, warm_items=frozenset(warm), cold_items=frozenset(cold))

    update_started = time.time()
    requests, request_state = build_pseudo_contexts(
        train_sequences=[(user, list(items)) for user, items in train_sequences.items()],
        pseudo_items=cold,
        trainable_items=warm,
        catalog=catalog,
        item_ids=item_ids,
        embeddings=embeddings,
        args=args,
        output_dir=output_dir,
    )

    historical = json.loads(paths["historical_config"].read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True)
    collator = CollatorGRAM(tokenizer, args=collator_args(historical), mode="train")
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    model = configure_fresh_model(historical, backbone, device, 2023)
    model.load_state_dict(torch.load(paths["checkpoint"], map_location="cpu"), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base_hash_before = _model_state_sha256(model)

    # Frozen GRAM encoder inputs preserve the B0 backbone contract.  BGE alone
    # selects edit contexts; the historical SASRec file is never used for
    # request selection or target supervision.
    item_inputs, input_audit = build_filtered_item_inputs(
        item_paths,
        item_text,
        paths["similar_items_b0_historical_only"],
        set(),
        args.similar_top_k,
    )
    item_to_cfid = {item: index + 1 for index, item in enumerate(sorted(item_paths))}
    position_to_layer, probe_state = probe_clean_base_layers(
        model=model,
        train_rows=train_rows,
        item_paths=item_paths,
        item_text=item_text,
        item_to_cfid=item_to_cfid,
        tokenizer=tokenizer,
        collator=collator,
        device=device,
        seed=args.seed,
        output_dir=output_dir,
    )
    deltas, b3_state, position_to_layer = build_b3_state(
        model=model,
        train_rows=train_rows,
        requests=requests,
        item_paths=item_paths,
        item_text=item_text,
        item_to_cfid=item_to_cfid,
        tokenizer=tokenizer,
        collator=collator,
        device=device,
        args=args,
        output_dir=output_dir,
        selected_layers=position_to_layer,
    )
    base_hash_after_state = _model_state_sha256(model)
    if base_hash_before != base_hash_after_state:
        raise RuntimeError("B3 state construction mutated frozen GRAM v0")
    b3_update_seconds = time.time() - update_started

    # Validation-bearing frozen rankings are opened only after every B3 state
    # artifact has been frozen.  The projection's last item was materialized
    # earlier but was excluded mechanically from every state builder.
    frozen = read_validation_predictions(paths["frozen_b0_b1_validation_predictions"])
    if set(frozen) != set(projected):
        raise ValueError("Frozen baseline users do not exactly match the projection")
    for user, source in frozen.items():
        if str(source.get("target")) != projected[user][-1]:
            raise ValueError(f"Frozen/projected validation target mismatch: {user}")
    if any(item not in item_paths for items in projected.values() for item in items):
        raise ValueError("Projected sequence contains an unknown catalog item")

    token_paths = encoded_catalog(tokenizer, item_paths)
    prediction_path = output_dir / "predictions_validation.jsonl"
    rows: list[dict] = []
    totals = {
        "b0_replayed_users": 0,
        "b1_replayed_users": 0,
        "b3_model_forward_users": 0,
        "b3_rankings_different_from_b0": 0,
        "b3_generation_dead_prefix_rows": 0,
    }
    b3_trace = {position: 0 for position in position_to_layer}
    inference_started = time.time()
    with prediction_path.open("x", encoding="utf-8") as prediction_handle:
        for index, (user, projected_items) in enumerate(projected.items(), 1):
            history, target = projected_items[:-1][-20:], projected_items[-1]
            source = frozen[user]
            b0 = unique_in_order([str(item) for item in source["v0_top50"]])
            resolver = unique_in_order([str(item) for item in source["resolver_top50"]])
            b1 = portfolio_at_2(b0, resolver, cold)
            sample = make_model_sample(
                {"user_id": user, "history": history, "target": None},
                item_inputs,
                item_paths,
                item_to_cfid,
            )
            batch = batch_to_device(collator([sample]), device)
            with torch.inference_mode(), OneOneGenerationDeltaContext(
                model=model,
                deltas_by_position=deltas,
                position_to_layer=position_to_layer,
                encoded_catalog_paths=token_paths.values(),
                decoder_start_token_id=model.config.decoder_start_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ) as trace:
                b3_scored = generate_beam(
                    model=model,
                    batch=batch,
                    token_paths=token_paths,
                    tokenizer=tokenizer,
                    beam_size=args.beam_size,
                )
            b3 = [item for item, _score in b3_scored]
            for ranking in (b0, b1, b3):
                if len(ranking) != 50 or len(set(ranking)) != 50 or not set(ranking).issubset(item_paths):
                    raise RuntimeError("B3 full-validation ranking violates strict top-50 contract")
            row = {
                "event_index": index,
                "user_id": user,
                "target_item": target,
                "is_cold": target in cold,
                "metrics": {
                    arm: ranking_metrics(ranking, target)
                    for arm, ranking in zip(ARMS, (b0, b1, b3))
                },
                "b0_top50": b0,
                "b1_top50": b1,
                "b3_top50": b3,
                "b3_differs_from_b0": b3 != b0,
            }
            prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            totals["b0_replayed_users"] += 1
            totals["b1_replayed_users"] += 1
            totals["b3_model_forward_users"] += 1
            totals["b3_rankings_different_from_b0"] += int(b3 != b0)
            totals["b3_generation_dead_prefix_rows"] += trace.dead_prefix_rows
            for position, count in trace.applied_rows_by_position.items():
                b3_trace[position] += count
            if index % 16 == 0 or index == len(projected):
                print(f"[{args.progress_marker}] events={index}/{len(projected)}", flush=True)
    b3_inference_seconds = time.time() - inference_started

    base_hash_after = _model_state_sha256(model)
    if base_hash_before != base_hash_after:
        raise RuntimeError("B3 full validation mutated frozen GRAM v0")
    if any(count < 1 for count in b3_trace.values()):
        raise RuntimeError("B3 full validation did not exercise every lexical position")

    metrics = {
        arm: {subset: summarize_arm(rows, arm, subset) for subset in ("all", "warm", "cold")}
        for arm in ARMS
    }
    intervals = {}
    seed_offset = 0
    for treatment, control in (("b1", "b0"), ("b3", "b0"), ("b3", "b1")):
        comparison = f"{treatment}_vs_{control}"
        intervals[comparison] = {}
        for name, metric, subset in (
            ("cold_hit@50", "hit@50", "cold"),
            ("cold_ndcg@10", "ndcg@10", "cold"),
            ("warm_ndcg@10", "ndcg@10", "warm"),
            ("overall_ndcg@10", "ndcg@10", "all"),
        ):
            intervals[comparison][name] = paired_bootstrap(
                rows,
                treatment,
                control,
                metric,
                subset,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + seed_offset,
            )
            seed_offset += 1

    labels = {
        "b1": {
            "PASS_NATIVE_COLD_RECOVERY": intervals["b1_vs_b0"]["cold_hit@50"]["ci_low"] > 0,
            "PASS_OVER_R2_PARETO": True,
            "reference_arm": True,
        },
        "b3": success_labels(rows, intervals, "b3"),
    }
    quality_b1_dominates_b3 = (
        metrics["b1"]["cold"]["hit@50"] >= metrics["b3"]["cold"]["hit@50"]
        and metrics["b1"]["warm"]["ndcg@10"] >= metrics["b3"]["warm"]["ndcg@10"]
        and (
            metrics["b1"]["cold"]["hit@50"] > metrics["b3"]["cold"]["hit@50"]
            or metrics["b1"]["warm"]["ndcg@10"] > metrics["b3"]["warm"]["ndcg@10"]
        )
    )
    labels["b3"]["PASS_COST_QUALITY_CANDIDATE"] = bool(
        labels["b3"]["PASS_NATIVE_COLD_RECOVERY"] and not quality_b1_dominates_b3
    )
    labels["b3"]["cost_quality_caveat"] = (
        "B1 current full inference is replayed, so this label uses quality non-domination only."
    )

    b1_source = json.loads(paths["b1_source_summary"].read_text(encoding="utf-8"))
    b1_state_bytes = paths["b1_state"].stat().st_size
    b3_root = output_dir / "b3_genrecedit"
    b3_state_bytes = sum(path.stat().st_size for path in b3_root.rglob("*") if path.is_file())
    updated_parameter_elements = sum(
        tensor.numel() for bundle in deltas.values() for tensor in bundle.values()
    )
    cost = {
        "b0": {"run_mode": "frozen_validation_prediction_replay", "extra_state_bytes": 0},
        "b1": {
            "run_mode": "frozen_resolver_prediction_replay",
            "extra_state_bytes": b1_state_bytes,
            "source_train_plus_validation_runtime_seconds": b1_source.get("runtime_seconds"),
            "current_full_inference_recomputed": False,
        },
        "b3": {
            "offline_update_wall_seconds": b3_update_seconds,
            "inference_wall_seconds": b3_inference_seconds,
            "users_per_second": len(rows) / b3_inference_seconds,
            "extra_state_bytes": b3_state_bytes,
            "delta_positions": len(deltas),
            "updated_parameter_elements": updated_parameter_elements,
            "model_forward_users": totals["b3_model_forward_users"],
        },
        "comparability": "B3 measured in this run; B0/B1 replay the same frozen validation artifact.",
    }

    config = {
        "experiment_id": args.experiment_id,
        "domain": args.domain,
        "split": "validation",
        "arms": list(ARMS),
        "events": len(rows),
        "train_transitions": args.train_transitions,
        "cold_catalog_items": len(cold),
        "contexts_per_cold": args.contexts_per_pseudo_cold,
        "covariance_transitions": args.covariance_transitions,
        "covariance_long_path_minimum": args.covariance_long_path_minimum,
        "requests_per_position": args.requests_per_position,
        "z_steps": args.z_steps,
        "similar_top_k": args.similar_top_k,
        "beam_size": args.beam_size,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "seed": args.seed,
        "device": args.device,
        "request_selection_rule": b3_state["request_selection_rule"],
        "test_read": False,
        "automatic_retry": False,
        "numerical_mode": numerical_mode,
    }
    summary = {
        **config,
        "status": "completed",
        "verdict": args.completed_verdict,
        "metrics": metrics,
        "paired_bootstrap": intervals,
        "success_labels": labels,
        "b3_request_state": request_state,
        "b3_probe_selected_layer": {str(key): value for key, value in position_to_layer.items()},
        "b3_probe_state": probe_state,
        "b3_state": b3_state,
        "b3_generation_applied_rows_by_position": {
            str(key): value for key, value in b3_trace.items()
        },
        "forward_accounting": totals,
        "cost": cost,
        "base_hash_before": base_hash_before,
        "base_hash_after_state": base_hash_after_state,
        "base_hash_after": base_hash_after,
        "base_hash_unchanged": base_hash_before == base_hash_after,
        "validation_target_used_for_b3_state_or_selection": False,
        "original_user_sequence_opened": False,
        "test_predictions_opened": False,
        "test_metrics_opened": False,
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    hashes["backbone_config"] = sha256_file(backbone / "config.json")
    atomic_json(output_dir / "config.json", config)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "input_file_sha256.json", hashes)
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "development_view": "audited Stage15 train+validation projection",
            "training_slice": "projected_items[:-1] only",
            "validation_target": "projected_items[-1], excluded from every B3 state builder",
            "b0_b1": "frozen validation-only Phase13 rankings, rescored on projected target",
            "b3": "full frozen cold catalog plus train-only warm BGE contexts/covariance",
            "request_similarity_source": embedding_meta,
            "historical_similar_items_use": "frozen GRAM encoder prompt only; not request selection",
            "test_target_materialized": False,
            "test_target_used": False,
        },
    )
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened_before_state_freeze": [
                str(paths[name].relative_to(REPO_ROOT))
                for name in (
                    "projected_sequences",
                    "historical_config",
                    "checkpoint",
                    "item_path_file",
                    "item_text_file",
                    "similar_items_b0_historical_only",
                    "item_embeddings",
                    "cold_items",
                    "warm_items",
                    "s3a_b3_summary",
                    "b0_parity_summary",
                )
            ],
            "opened_after_state_freeze": [
                str(paths[name].relative_to(REPO_ROOT))
                for name in (
                    "frozen_b0_b1_validation_predictions",
                    "b1_source_summary",
                    "b1_state",
                )
            ],
            "backbone_dir": str(backbone.relative_to(REPO_ROOT)),
            "frozen_contract": (
                str(paths["frozen_contract"].relative_to(REPO_ROOT))
                if "frozen_contract" in paths
                else None
            ),
            "projected_validation_target_used_for_state": False,
            "original_user_sequence_opened": False,
            "test_predictions_opened": False,
            "test_metrics_opened": False,
        },
    )
    atomic_json(
        output_dir / "resource_summary.json",
        {
            "runtime_seconds": summary["runtime_seconds"],
            "peak_cuda_allocated_mib": summary["peak_cuda_allocated_mib"],
            "peak_cpu_rss_mib": summary["peak_cpu_rss_mib"],
            "cost": cost,
            "model_training": "none; B3 covariance/edit requests/deltaW only",
            "frozen_gram_training": False,
            "input_audit": input_audit,
        },
    )
    print(
        json.dumps(
            {"status": "completed", "verdict": summary["verdict"], "labels": labels},
            ensure_ascii=False,
        )
    )
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
