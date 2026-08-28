"""Stage15 S3B Toys full validation for frozen B0/B1 and admitted B2.

The job consumes only the audited train+validation projection.  B0 and B1 are
replayed from the frozen validation-only Phase13 artifact and rescored against
the projected target; B2 trains its frozen-budget train-only drafter before
the baseline artifact is opened, then runs the admitted draft/verify/redraft
path with the frozen GRAM v0 verifier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
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
    iter_train_transitions,
    read_projected_sequences,
    read_validation_predictions,
    sha256_file,
)
from specgr_gram_adapter import PathCatalog  # noqa: E402
from toys_b2_drafter_state_smoke import _stable_key  # noqa: E402
from toys_b3_edit_state_smoke import _configure_determinism, _model_state_sha256  # noqa: E402
from toys_s3a_admission import (  # noqa: E402
    b2_rank,
    encoded_catalog,
    load_embeddings,
    train_drafter,
    verifier_score_lengths,
)


ARMS = ("b0", "b1", "b2")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260822


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="GRAM_STAGE15_S3B_TOYS_FULL_VALIDATION_B0_B1_B2_SEED0")
    parser.add_argument("--domain", default="Toys_cold50")
    parser.add_argument("--completed-verdict", default="COMPLETED_S15_3B_TOYS_FULL_VALIDATION")
    parser.add_argument("--required-events", type=int, default=8789)
    parser.add_argument("--progress-marker", default="s3b-eval")
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
    parser.add_argument("--s3a-summary", type=Path, required=True)
    parser.add_argument("--b0-parity-summary", type=Path, required=True)
    parser.add_argument("--b1-source-summary", type=Path, required=True)
    parser.add_argument("--b1-state", type=Path, required=True)
    parser.add_argument("--frozen-contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-transitions", type=int, default=4096)
    parser.add_argument("--drafter-epochs", type=int, default=2)
    parser.add_argument("--drafter-batch-size", type=int, default=128)
    parser.add_argument("--drafter-learning-rate", type=float, default=1e-3)
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--draft-size", type=int, default=10)
    parser.add_argument("--draft-rounds", type=int, default=5)
    parser.add_argument("--verifier-threshold", type=float, default=-1.6)
    parser.add_argument("--candidate-chunk-size", type=int, default=10)
    parser.add_argument("--similar-top-k", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--seed", type=int, default=1502)
    return parser.parse_args()


def read_set(path: Path) -> set[str]:
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError(f"Empty item set: {path}")
    return values


def ensure_new_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in allowed]
    if unexpected:
        raise FileExistsError(f"Refusing existing S15-3B artifacts: {unexpected}")


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def portfolio_at_2(gram: list[str], resolver: list[str], cold_items: set[str]) -> list[str]:
    """Frozen Phase13 B1: protect top-7, insert two cold items at ranks 9-10."""
    gram = unique_in_order(gram)
    resolver = unique_in_order(resolver)
    if len(gram) != 50 or len(resolver) != 50:
        raise ValueError("B1 source rankings must each contain 50 unique items")
    protected = set(gram[:7])
    candidates = [item for item in resolver if item in cold_items and item not in protected][:2]
    if len(candidates) != 2:
        raise ValueError("Cannot construct frozen B1 portfolio@2")
    ranking = unique_in_order([*gram[:8], *candidates, *gram[8:], *resolver])[:50]
    if len(ranking) != 50:
        raise ValueError("Frozen B1 portfolio did not produce 50 unique items")
    return ranking


def ranking_metrics(ranking: list[str], target: str) -> dict[str, float | int | None]:
    rank = ranking.index(target) + 1 if target in ranking else None
    return {
        "rank": rank,
        "hit@50": int(rank is not None and rank <= 50),
        "ndcg@10": 1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0,
    }


def paired_bootstrap(
    rows: list[dict], treatment: str, control: str, metric: str, subset: str,
    *, resamples: int, seed: int,
) -> dict:
    selected = [
        row for row in rows
        if subset == "all" or (subset == "cold") == bool(row["is_cold"])
    ]
    delta = np.asarray(
        [row["metrics"][treatment][metric] - row["metrics"][control][metric] for row in selected],
        dtype=np.float64,
    )
    if not len(delta):
        raise ValueError(f"No rows for bootstrap subset {subset}")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 250):
        count = min(250, resamples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": len(delta),
        "observed": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "verdict": "PASS" if low > 0 else ("FAIL" if high < 0 else "INCONCLUSIVE"),
    }


def summarize_arm(rows: list[dict], arm: str, subset: str) -> dict:
    selected = [
        row for row in rows
        if subset == "all" or (subset == "cold") == bool(row["is_cold"])
    ]
    hits = sum(int(row["metrics"][arm]["hit@50"]) for row in selected)
    hit_targets = {
        row["target_item"] for row in selected if row["metrics"][arm]["hit@50"]
    }
    return {
        "events": len(selected),
        "hit@50": hits / len(selected),
        "ndcg@10": float(np.mean([row["metrics"][arm]["ndcg@10"] for row in selected])),
        "hit_events": hits,
        "unique_target_items": len({row["target_item"] for row in selected}),
        "unique_hit_target_items": len(hit_targets),
    }


def success_labels(rows: list[dict], intervals: dict, arm: str) -> dict:
    cold = summarize_arm(rows, arm, "cold")
    warm = summarize_arm(rows, arm, "warm")
    b1_cold = summarize_arm(rows, "b1", "cold")
    b1_warm = summarize_arm(rows, "b1", "warm")
    native = intervals[f"{arm}_vs_b0"]["cold_hit@50"]["ci_low"] > 0
    cold_over_b1 = intervals[f"{arm}_vs_b1"]["cold_hit@50"]["ci_low"] > 0
    warm_over_b1 = intervals[f"{arm}_vs_b1"]["warm_ndcg@10"]["ci_low"] > 0
    pareto = native and (
        (cold_over_b1 and warm["ndcg@10"] >= b1_warm["ndcg@10"])
        or (warm_over_b1 and cold["hit@50"] >= b1_cold["hit@50"])
    )
    return {
        "PASS_NATIVE_COLD_RECOVERY": native,
        "PASS_OVER_R2_PARETO": pareto,
        "native_cold_recovery_basis": intervals[f"{arm}_vs_b0"]["cold_hit@50"],
    }


def selected_train_rows(projected: dict[str, list[str]], args: argparse.Namespace) -> list[dict]:
    transitions = list(iter_train_transitions(projected))
    selected = sorted(
        transitions,
        key=lambda row: (
            _stable_key(args.seed, "train", row.user_id, len(row.history), row.target),
            row.user_id,
            len(row.history),
        ),
    )[: args.train_transitions]
    if len(selected) != args.train_transitions:
        raise ValueError("Insufficient train-only transitions for frozen B2 budget")
    return [
        {"user_id": row.user_id, "history": list(row.history[-20:]), "target": row.target}
        for row in selected
    ]


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    output_dir = args.output_dir.resolve()
    ensure_new_outputs(output_dir)
    if args.beam_size != 50 or args.draft_size * args.draft_rounds != 50:
        raise ValueError("S15-3B freezes beam and B2 candidate budgets at 50")
    if args.bootstrap_resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError("S15-3B freezes 10,000 paired bootstrap resamples")

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
            "s3a_summary": args.s3a_summary,
            "b0_parity_summary": args.b0_parity_summary,
            "b1_source_summary": args.b1_source_summary,
            "b1_state": args.b1_state,
        }.items()
    }
    if args.frozen_contract is not None:
        paths["frozen_contract"] = args.frozen_contract.resolve()
    backbone = args.backbone_path.resolve()
    if paths["projected_sequences"].name != "user_sequence_train_validation.txt":
        raise ValueError("S15-3B requires the audited projected sequence")
    if "test" in paths["frozen_b0_b1_validation_predictions"].name.lower():
        raise ValueError("Refusing test predictions")
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Input must be a regular non-symlink file: {name}={path}")
    if not backbone.is_dir():
        raise FileNotFoundError(backbone)
    s3a = json.loads(paths["s3a_summary"].read_text(encoding="utf-8"))
    parity = json.loads(paths["b0_parity_summary"].read_text(encoding="utf-8"))
    if s3a.get("verdict") != "PASS_S15_3A_B2_ITEM_DISJOINT_ADMISSION":
        raise ValueError("B2 S15-3A admission is not PASS")
    if parity.get("verdict") != "PASS_B0_PROJECTION_PARITY":
        raise ValueError("B0 projection-parity Gate is not PASS")

    numerical_mode = _configure_determinism()
    projected = read_projected_sequences(paths["projected_sequences"])
    if len(projected) != args.required_events:
        raise ValueError(
            f"Projected event count drift: {len(projected)} != {args.required_events}"
        )
    cold = read_set(paths["cold_items"])
    warm = read_set(paths["warm_items"])
    if cold & warm:
        raise ValueError("Cold and warm sets overlap")
    item_paths = load_paths(paths["item_path_file"])
    item_text = read_key_value(paths["item_text_file"])
    item_ids, embeddings, embedding_meta = load_embeddings(paths["item_embeddings"])
    if cold | warm != set(item_paths) or set(item_ids) != set(item_paths) or set(item_text) != set(item_paths):
        raise ValueError("Catalog/path/text/embedding/cold-warm universes differ")
    if any(item not in item_paths for items in projected.values() for item in items):
        raise ValueError("Projected sequence contains an unknown catalog item")

    device = torch.device(args.device)
    train_rows = selected_train_rows(projected, args)
    if any(row["target"] not in warm for row in train_rows):
        raise ValueError("Cold target entered B2 drafter supervision")
    update_started = time.time()
    drafter, b2_state = train_drafter(
        rows=train_rows,
        item_ids=item_ids,
        embeddings=embeddings,
        trainable_items=warm,
        device=device,
        args=args,
        output_dir=output_dir,
    )
    b2_update_seconds = time.time() - update_started

    # Validation-bearing baseline rows are intentionally opened only after B2
    # state is frozen.  Their targets are checked against, then ignored in
    # favour of, the audited projection.
    frozen = read_validation_predictions(paths["frozen_b0_b1_validation_predictions"])
    if set(frozen) != set(projected):
        raise ValueError("Frozen baseline users do not exactly match the projection")
    for user, source in frozen.items():
        if str(source.get("target")) != projected[user][-1]:
            raise ValueError(f"Frozen/projected validation target mismatch: {user}")

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

    item_inputs, input_audit = build_filtered_item_inputs(
        item_paths,
        item_text,
        paths["similar_items_b0_historical_only"],
        set(),
        args.similar_top_k,
    )
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    item_to_cfid = {item: index + 1 for index, item in enumerate(sorted(item_paths))}
    catalog = PathCatalog(paths=item_paths, warm_items=frozenset(warm), cold_items=frozenset(cold))
    token_paths = encoded_catalog(tokenizer, item_paths)
    score_lengths = verifier_score_lengths(catalog)

    prediction_path = output_dir / "predictions_validation.jsonl"
    rows: list[dict] = []
    totals = {"b0_replayed_users": 0, "b1_replayed_users": 0, "b2_verifier_forward_candidates": 0,
              "b2_encoder_forward_histories": 0, "b2_accepted_drafts": 0,
              "b2_rankings_different_from_b0": 0}
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
                item_inputs, item_paths, item_to_cfid,
            )
            batch = batch_to_device(collator([sample]), device)
            # Only rank order matters for the fallback.  Frozen B0 ordering is
            # represented by strictly decreasing synthetic scores.
            b0_scored = [(item, float(-rank)) for rank, item in enumerate(b0)]
            with torch.inference_mode():
                b2, budget = b2_rank(
                    model=model,
                    drafter=drafter,
                    batch=batch,
                    history=history,
                    item_ids=item_ids,
                    item_to_index=item_to_index,
                    token_paths=token_paths,
                    score_lengths=score_lengths,
                    catalog=catalog,
                    beam_fallback=b0_scored,
                    args=args,
                    device=device,
                )
            for ranking in (b0, b1, b2):
                if len(ranking) != 50 or len(set(ranking)) != 50 or not set(ranking).issubset(item_paths):
                    raise RuntimeError("S15-3B ranking violates strict item top-50 contract")
            row = {
                "event_index": index,
                "user_id": user,
                "target_item": target,
                "is_cold": target in cold,
                "metrics": {arm: ranking_metrics(ranking, target) for arm, ranking in zip(ARMS, (b0, b1, b2))},
                "b0_top50": b0,
                "b1_top50": b1,
                "b2_top50": b2,
                "b2_budget": budget,
            }
            prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            totals["b0_replayed_users"] += 1
            totals["b1_replayed_users"] += 1
            totals["b2_verifier_forward_candidates"] += int(budget["verifier_forward_candidates"])
            totals["b2_encoder_forward_histories"] += int(budget["rounds"])
            totals["b2_accepted_drafts"] += int(budget["accepted_drafts"])
            totals["b2_rankings_different_from_b0"] += int(b2 != b0)
            if index % 16 == 0 or index == len(projected):
                print(f"[{args.progress_marker}] events={index}/{len(projected)}", flush=True)
    b2_inference_seconds = time.time() - inference_started
    base_hash_after = _model_state_sha256(model)
    if base_hash_before != base_hash_after:
        raise RuntimeError("B2 full validation mutated frozen GRAM v0")

    metrics = {
        arm: {subset: summarize_arm(rows, arm, subset) for subset in ("all", "warm", "cold")}
        for arm in ARMS
    }
    intervals = {}
    seed_offset = 0
    for treatment, control in (("b1", "b0"), ("b2", "b0"), ("b2", "b1")):
        comparison = f"{treatment}_vs_{control}"
        intervals[comparison] = {}
        for name, metric, subset in (
            ("cold_hit@50", "hit@50", "cold"),
            ("cold_ndcg@10", "ndcg@10", "cold"),
            ("warm_ndcg@10", "ndcg@10", "warm"),
            ("overall_ndcg@10", "ndcg@10", "all"),
        ):
            intervals[comparison][name] = paired_bootstrap(
                rows, treatment, control, metric, subset,
                resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + seed_offset,
            )
            seed_offset += 1

    labels = {
        "b1": {
            "PASS_NATIVE_COLD_RECOVERY": intervals["b1_vs_b0"]["cold_hit@50"]["ci_low"] > 0,
            "PASS_OVER_R2_PARETO": True,
            "reference_arm": True,
        },
        "b2": success_labels(rows, intervals, "b2"),
        "b3": {
            "status": "FAIL_B3_S15_3A_EDIT_STATE_ADMISSION",
            "efficacy_metrics_generated": False,
        },
    }
    b1_source = json.loads(paths["b1_source_summary"].read_text(encoding="utf-8"))
    b1_state_bytes = paths["b1_state"].stat().st_size
    b2_state_path = output_dir / "b2_specgr" / "drafter" / "drafter_state.pt"
    b2_state_bytes = b2_state_path.stat().st_size
    quality_b1_dominates_b2 = (
        metrics["b1"]["cold"]["hit@50"] >= metrics["b2"]["cold"]["hit@50"]
        and metrics["b1"]["warm"]["ndcg@10"] >= metrics["b2"]["warm"]["ndcg@10"]
        and (
            metrics["b1"]["cold"]["hit@50"] > metrics["b2"]["cold"]["hit@50"]
            or metrics["b1"]["warm"]["ndcg@10"] > metrics["b2"]["warm"]["ndcg@10"]
        )
    )
    cost = {
        "b0": {"run_mode": "frozen_validation_prediction_replay", "extra_state_bytes": 0},
        "b1": {
            "run_mode": "frozen_resolver_prediction_replay",
            "extra_state_bytes": b1_state_bytes,
            "source_train_plus_validation_runtime_seconds": b1_source.get("runtime_seconds"),
            "current_full_inference_recomputed": False,
        },
        "b2": {
            "offline_update_wall_seconds": b2_update_seconds,
            "inference_wall_seconds": b2_inference_seconds,
            "users_per_second": len(rows) / b2_inference_seconds,
            "extra_state_bytes": b2_state_bytes,
            "trainable_parameters": b2_state["trainable_parameters"],
            "verifier_forward_candidates": totals["b2_verifier_forward_candidates"],
            "encoder_forward_histories": totals["b2_encoder_forward_histories"],
        },
        "comparability": "B2 measured in this run; B0/B1 replay frozen validation artifacts and retain source runtime separately",
    }
    labels["b2"]["PASS_COST_QUALITY_CANDIDATE"] = bool(
        labels["b2"]["PASS_NATIVE_COLD_RECOVERY"] and not quality_b1_dominates_b2
    )
    labels["b2"]["cost_quality_caveat"] = "B1 current full inference was replayed, so the label uses quality non-domination only."

    config = {
        "experiment_id": args.experiment_id,
        "domain": args.domain,
        "split": "validation",
        "arms": list(ARMS),
        "events": len(rows),
        "train_transitions": args.train_transitions,
        "drafter_epochs": args.drafter_epochs,
        "drafter_batch_size": args.drafter_batch_size,
        "drafter_learning_rate": args.drafter_learning_rate,
        "beam_size": args.beam_size,
        "b2_candidate_budget": args.draft_size * args.draft_rounds,
        "b2_draft_size": args.draft_size,
        "b2_draft_rounds": args.draft_rounds,
        "b2_verifier_threshold": args.verifier_threshold,
        "candidate_chunk_size": args.candidate_chunk_size,
        "similar_top_k": args.similar_top_k,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "seed": args.seed,
        "device": args.device,
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
        "b2_state": b2_state,
        "forward_accounting": totals,
        "cost": cost,
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_hash_unchanged": base_hash_before == base_hash_after,
        "validation_target_used_for_b2_training_or_state_selection": False,
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
            "validation_target": "projected_items[-1]",
            "b0_b1": "frozen validation-only Phase13 rankings, rescored on projected target",
            "b2": "4096 SHA-ranked train-only transitions; validation opened after state freeze",
            "test_target_materialized": False,
            "test_target_used": False,
        },
    )
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened": [str(path.relative_to(REPO_ROOT)) for path in paths.values()],
            "backbone_dir": str(backbone.relative_to(REPO_ROOT)),
            "original_user_sequence_opened": False,
            "similar_item_use": "frozen GRAM encoder inputs only; never drafter supervision or selection",
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
            "model_training": "B2 drafter only",
            "frozen_gram_training": False,
            "input_audit": input_audit,
        },
    )
    print(json.dumps({"status": "completed", "verdict": summary["verdict"], "labels": labels}, ensure_ascii=False))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
