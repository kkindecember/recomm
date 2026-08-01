#!/usr/bin/env python3
"""Phase 6 GACR-v4: target-free learned residual-application gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.gacr_p0 import relative_gain  # noqa: E402
from experiment.phase4.gacr_s0 import BoundedResidualRanker, stable_ranking  # noqa: E402
from experiment.phase4.gcdh_p0 import ROOT, sha256, write_json  # noqa: E402
from experiment.phase6.gacr_v2 import (  # noqa: E402
    build_training_records,
    build_validation_records,
    paired_bootstrap_candidate,
    rank_metrics,
    serializable_rows,
    summarize_seed_stability,
    validate_checkpoint_lineage,
)
from experiment.phase6.gacr_v3 import v3_method_result  # noqa: E402

GATE_FEATURE_DIM = 8


def target_free_gate_features(record: dict, residual: torch.Tensor) -> torch.Tensor:
    """Aggregate only inference-time candidate and score information."""
    base = record["base"].to(residual.device)
    features = record["features"].to(residual.device)
    if features.ndim != 2 or features.shape[1] != 6:
        raise ValueError("GACR-v4 expects the frozen six-feature schema")
    ordered = torch.sort(base, descending=True).values
    base_margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    gram_present = features[:, 3]
    catalog_present = features[:, 4]
    centered_base = base - base.mean()
    centered_residual = residual - residual.mean()
    alignment = torch.sum(centered_base * centered_residual) / (
        torch.linalg.vector_norm(centered_base)
        * torch.linalg.vector_norm(centered_residual)
    ).clamp_min(1e-6)
    result = torch.stack(
        [
            residual.max() - residual.min(),
            residual.abs().mean(),
            base_margin,
            (base > 0).float().mean(),
            (gram_present * catalog_present).mean(),
            ((1.0 - gram_present) * catalog_present).mean(),
            features[:, 0].std(unbiased=False),
            alignment,
        ]
    )
    if result.numel() != GATE_FEATURE_DIM or not torch.isfinite(result).all():
        raise ValueError("non-finite GACR-v4 target-free gate feature")
    return result


def rank_from_scores(record: dict, scores: torch.Tensor) -> int | None:
    if record["target_index"] is None:
        return None
    return stable_ranking(scores).index(int(record["target_index"])) + 1


def gate_training_examples(
    records: list[dict],
    residual_state: dict,
    bound: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    ranker = BoundedResidualRanker(6, 16, bound).to(device)
    ranker.load_state_dict(residual_state, strict=True)
    ranker.eval()
    xs, ys = [], []
    with torch.no_grad():
        for record in records:
            if record["target_index"] is None:
                continue
            residual = ranker(record["features"].to(device))
            candidate_rank = rank_from_scores(
                record, record["base"].to(device) + residual
            )
            baseline_rank = record["gram_rank"]
            sentinel = len(record["base"]) + 1
            baseline_value = sentinel if baseline_rank is None else baseline_rank
            candidate_value = sentinel if candidate_rank is None else candidate_rank
            if candidate_value == baseline_value:
                continue
            xs.append(target_free_gate_features(record, residual).cpu())
            ys.append(float(candidate_value < baseline_value))
    if not xs:
        raise ValueError("no changed covered examples for GACR-v4 gate")
    x = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.float32)
    positives = int(y.sum().item())
    negatives = int(len(y) - positives)
    if min(positives, negatives) < 5:
        raise ValueError(
            f"insufficient gate classes: positives={positives} negatives={negatives}"
        )
    return x, y, {
        "changed_covered_examples": len(y),
        "beneficial_examples": positives,
        "harmful_examples": negatives,
    }


def fit_gate(
    records: list[dict],
    residual_state: dict,
    config: dict,
    seed: int,
    device: torch.device,
) -> tuple[dict, dict]:
    x, y, stats = gate_training_examples(
        records,
        residual_state,
        float(config["residual"]["bound"]),
        device,
    )
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    z = ((x - mean) / std).to(device)
    y_device = y.to(device)
    torch.manual_seed(seed + int(config["gate"]["seed_offset"]))
    torch.cuda.manual_seed_all(seed + int(config["gate"]["seed_offset"]))
    model = torch.nn.Linear(GATE_FEATURE_DIM, 1).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    positives = float(y.sum())
    negatives = float(len(y) - y.sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["gate"]["learning_rate"]),
        weight_decay=float(config["gate"]["weight_decay"]),
    )
    first_loss = None
    last_loss = None
    for _ in range(int(config["gate"]["fixed_training_steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(z).squeeze(-1), y_device)
        if not torch.isfinite(loss):
            raise ValueError("non-finite GACR-v4 gate loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        first_loss = float(loss) if first_loss is None else first_loss
        last_loss = float(loss)
    state = {
        "mean": mean.cpu(),
        "std": std.cpu(),
        "weight": model.weight.detach().cpu().squeeze(0),
        "bias": model.bias.detach().cpu().squeeze(0),
    }
    with torch.no_grad():
        probabilities = torch.sigmoid(model(z).squeeze(-1)).cpu()
        predictions = probabilities >= 0.5
        accuracy = float((predictions == y.bool()).float().mean())
    return state, stats | {
        "first_loss": first_loss,
        "last_loss": last_loss,
        "training_accuracy_at_0.5": accuracy,
        "mean_probability": float(probabilities.mean()),
    }


def gate_probability(feature: torch.Tensor, gate_state: dict) -> float:
    z = (feature.cpu() - gate_state["mean"]) / gate_state["std"]
    logit = torch.dot(z, gate_state["weight"]) + gate_state["bias"]
    return float(torch.sigmoid(logit))


@torch.no_grad()
def evaluate_gate(
    records: list[dict],
    residual_state: dict,
    gate_state: dict,
    bound: float,
    threshold: float,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    ranker = BoundedResidualRanker(6, 16, bound).to(device)
    ranker.load_state_dict(residual_state, strict=True)
    ranker.eval()
    rows = []
    for record in records:
        residual = ranker(record["features"].to(device))
        feature = target_free_gate_features(record, residual)
        probability = gate_probability(feature, gate_state)
        applied = threshold <= 0.0 or probability >= threshold
        baseline_rank = record["gram_rank"]
        if applied:
            candidate_rank = rank_from_scores(
                record, record["base"].to(device) + residual
            )
        else:
            candidate_rank = baseline_rank
        b_r10, b_ndcg, b_r50 = rank_metrics(baseline_rank)
        c_r10, c_ndcg, c_r50 = rank_metrics(candidate_rank)
        rows.append(
            {
                "sample_key": record["sample_key"],
                "target_group": record["target_group"],
                "baseline_rank": baseline_rank,
                "candidate_rank": candidate_rank,
                "union_covered": int(record["target_index"] is not None),
                "baseline_Recall@10": b_r10,
                "baseline_NDCG@10": b_ndcg,
                "baseline_Recall@50": b_r50,
                "candidate_Recall@10": c_r10,
                "candidate_NDCG@10": c_ndcg,
                "candidate_Recall@50": c_r50,
                "changed": int(candidate_rank != baseline_rank),
                "broad_harm": int(b_r10 == 1.0 and c_r10 == 0.0),
                "gate_probability": probability,
                "gate_applied": int(applied),
                "residual_spread": float(residual.max() - residual.min()),
            }
        )

    def summarize(selected: list[dict]) -> dict:
        covered = [row for row in selected if row["union_covered"]]
        return {
            "n": len(selected),
            "baseline_Recall@10": float(np.mean([r["baseline_Recall@10"] for r in selected])),
            "baseline_NDCG@10": float(np.mean([r["baseline_NDCG@10"] for r in selected])),
            "baseline_Recall@50": float(np.mean([r["baseline_Recall@50"] for r in selected])),
            "candidate_Recall@10": float(np.mean([r["candidate_Recall@10"] for r in selected])),
            "candidate_NDCG@10": float(np.mean([r["candidate_NDCG@10"] for r in selected])),
            "candidate_Recall@50": float(np.mean([r["candidate_Recall@50"] for r in selected])),
            "union_coverage": float(np.mean([r["union_covered"] for r in selected])),
            "changed_user_coverage": float(np.mean([r["changed"] for r in selected])),
            "changed_covered_user_coverage": (
                float(np.mean([r["changed"] for r in covered])) if covered else 0.0
            ),
            "broad_harm_rate": float(np.mean([r["broad_harm"] for r in selected])),
            "gate_application_rate": float(np.mean([r["gate_applied"] for r in selected])),
            "mean_gate_probability": float(np.mean([r["gate_probability"] for r in selected])),
        }

    groups = {"overall": summarize(rows)}
    for group in ("head", "tail"):
        groups[group] = summarize([r for r in rows if r["target_group"] == group])
    return groups, rows


def select_domain_thresholds(training: dict, config: dict) -> tuple[dict, dict]:
    selected, audit = {}, {}
    for dataset in config["datasets"]:
        candidates = []
        for raw_threshold in config["gate"]["threshold_candidates"]:
            threshold = float(raw_threshold)
            cells = [
                training[dataset]["seeds"][str(seed)]["calibration"][str(raw_threshold)]
                for seed in config["training_seeds"]
            ]
            eligible = all(
                cell["overall"]["candidate_Recall@10"]
                >= cell["overall"]["baseline_Recall@10"]
                and cell["overall"]["candidate_Recall@50"]
                >= cell["overall"]["baseline_Recall@50"]
                and cell["tail"]["candidate_NDCG@10"]
                >= cell["tail"]["baseline_NDCG@10"]
                and cell["tail"]["candidate_Recall@50"]
                >= cell["tail"]["baseline_Recall@50"]
                and cell["overall"]["broad_harm_rate"]
                <= float(config["calibration_safety"]["broad_harm_max"])
                for cell in cells
            )
            candidates.append(
                {
                    "threshold": threshold,
                    "eligible": eligible,
                    "mean_overall_ndcg10": float(
                        np.mean([c["overall"]["candidate_NDCG@10"] for c in cells])
                    ),
                    "mean_tail_ndcg10": float(
                        np.mean([c["tail"]["candidate_NDCG@10"] for c in cells])
                    ),
                    "mean_gate_application_rate": float(
                        np.mean([c["overall"]["gate_application_rate"] for c in cells])
                    ),
                    "maximum_broad_harm_rate": float(
                        np.max([c["overall"]["broad_harm_rate"] for c in cells])
                    ),
                }
            )
        eligible = [row for row in candidates if row["eligible"]]
        if not eligible:
            raise RuntimeError(f"GACR-v3 identity control failed safety: {dataset}")
        chosen = sorted(
            eligible,
            key=lambda row: (
                -row["mean_overall_ndcg10"],
                -row["mean_tail_ndcg10"],
                row["maximum_broad_harm_rate"],
                row["threshold"],
            ),
        )[0]
        selected[dataset] = float(chosen["threshold"])
        audit[dataset] = candidates
    return selected, audit


def method_result(
    records: list[dict],
    residual_state: dict,
    gate_state: dict,
    config: dict,
    threshold: float,
    seed: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    groups, rows = evaluate_gate(
        records,
        residual_state,
        gate_state,
        float(config["residual"]["bound"]),
        threshold,
        device,
    )
    tail_rows = [row for row in rows if row["target_group"] == "tail"]
    overall = groups["overall"]
    return {
        "gate_threshold": threshold,
        "groups": groups,
        "gains": {
            "overall_ndcg10_relative_gain": relative_gain(
                overall["baseline_NDCG@10"], overall["candidate_NDCG@10"]
            ),
            "overall_recall10_absolute_gain": (
                overall["candidate_Recall@10"] - overall["baseline_Recall@10"]
            ),
            "overall_recall50_absolute_gain": (
                overall["candidate_Recall@50"] - overall["baseline_Recall@50"]
            ),
            "tail_ndcg10_relative_gain": relative_gain(
                groups["tail"]["baseline_NDCG@10"],
                groups["tail"]["candidate_NDCG@10"],
            ),
        },
        "bootstrap": {
            "overall_ndcg10_relative_gain_ci95": paired_bootstrap_candidate(
                rows, "NDCG@10", True, seed + 11
            ),
            "overall_recall10_absolute_gain_ci95": paired_bootstrap_candidate(
                rows, "Recall@10", False, seed + 21
            ),
            "tail_ndcg10_relative_gain_ci95": paired_bootstrap_candidate(
                tail_rows, "NDCG@10", True, seed + 31
            ),
        },
    }, rows


def load_residual_state(dataset: str, seed: int, config: dict) -> tuple[dict, str]:
    path = ROOT / config["inputs"]["v3_residual_root"] / dataset / f"residual_seed{seed}.pt"
    expected = config["inputs"]["expected_residual_sha256"][dataset][str(seed)]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"residual SHA mismatch {dataset}/{seed}: expected={expected} actual={actual}"
        )
    return torch.load(path, map_location="cpu"), actual


def validate_implementation(config: dict) -> None:
    expected = config["integrity"]["code_sha256"]
    actual = sha256(Path(__file__))
    if expected != actual:
        raise RuntimeError(
            f"implementation SHA mismatch: expected={expected} actual={actual}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GACR-v4 requires CUDA")
    config = json.loads(args.config.read_text())
    validate_implementation(config)
    validate_checkpoint_lineage(config)
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    device = torch.device("cuda:0")

    training = {}
    for dataset in config["datasets"]:
        metadata, fit_records, calibration_records = build_training_records(
            dataset, config, p0_config, device
        )
        seeds = {}
        for seed in config["training_seeds"]:
            residual_state, residual_sha = load_residual_state(dataset, int(seed), config)
            gate_state, gate_fit = fit_gate(
                fit_records, residual_state, config, int(seed), device
            )
            calibration = {}
            for threshold in config["gate"]["threshold_candidates"]:
                groups, _ = evaluate_gate(
                    calibration_records,
                    residual_state,
                    gate_state,
                    float(config["residual"]["bound"]),
                    float(threshold),
                    device,
                )
                calibration[str(threshold)] = groups
            seeds[str(seed)] = {
                "gate_state": gate_state,
                "gate_fit": gate_fit,
                "calibration": calibration,
                "residual_sha256": residual_sha,
            }
            print(
                f"GACR_V4_GATE dataset={dataset} seed={seed} "
                f"examples={gate_fit['changed_covered_examples']} "
                f"loss={gate_fit['last_loss']:.6f}",
                flush=True,
            )
        training[dataset] = metadata | {"seeds": seeds}
        del fit_records, calibration_records
        torch.cuda.empty_cache()

    selected_thresholds, threshold_selection = select_domain_thresholds(training, config)
    validation = {}
    for dataset in config["datasets"]:
        metadata, records = build_validation_records(dataset, config, p0_config, device)
        output_dir = args.output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = {}
        for seed in config["training_seeds"]:
            item = training[dataset]["seeds"][str(seed)]
            residual_state, _ = load_residual_state(dataset, int(seed), config)
            v3_result, v3_rows = v3_method_result(
                records,
                residual_state,
                config,
                float(config["v3_identity_budget"]),
                int(seed) + 1000,
                device,
            )
            v4_result, v4_rows = method_result(
                records,
                residual_state,
                item["gate_state"],
                config,
                selected_thresholds[dataset],
                int(seed) + 2000,
                device,
            )
            seeds[str(seed)] = {"gacr_v3": v3_result, "gacr_v4": v4_result}
            for method, rows in (("gacr_v3", v3_rows), ("gacr_v4", v4_rows)):
                path = output_dir / f"{method}_seed{seed}_per_user.csv"
                with path.open("w", newline="") as handle:
                    serial = serializable_rows(rows)
                    writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
                    writer.writeheader()
                    writer.writerows(serial)
                seeds[str(seed)][method]["per_user_sha256"] = sha256(path)
            gate_path = output_dir / f"gate_seed{seed}.pt"
            torch.save(item["gate_state"], gate_path)
            seeds[str(seed)]["gate_checkpoint_sha256"] = sha256(gate_path)
        validation[dataset] = metadata | {"seeds": seeds}
        del records
        torch.cuda.empty_cache()

    training_summary = {}
    for dataset, item in training.items():
        training_summary[dataset] = {k: v for k, v in item.items() if k != "seeds"}
        training_summary[dataset]["seeds"] = {
            seed: {k: v for k, v in value.items() if k != "gate_state"}
            for seed, value in item["seeds"].items()
        }
    integrity = {
        "fit_calibration_user_disjoint": all(
            training[d]["fit_calibration_user_overlap"] == 0 for d in config["datasets"]
        ),
        "parent_checkpoint_sha_unchanged": all(
            training[d]["parent_checkpoint_sha256_before"]
            == training[d]["parent_checkpoint_sha256_after"]
            == validation[d]["parent_checkpoint_sha256_before"]
            == validation[d]["parent_checkpoint_sha256_after"]
            for d in config["datasets"]
        ),
        "fresh_validation_zero_overlap": all(
            validation[d]["gcdh_or_training_overlap"] == 0
            and validation[d]["prior_gacr_p0_overlap"] == 0
            for d in config["datasets"]
        ),
        "frozen_v3_residuals_only": True,
        "gate_features_target_free": True,
        "backbone_optimizer_steps": 0,
        "test_data_read": False,
        "sports_data_read": False,
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "result_status": "RESULTS_READY_FOR_RESEARCHER_ANALYSIS",
        "single_changed_factor": "target_free_learned_residual_application_gate",
        "selected_domain_thresholds": selected_thresholds,
        "threshold_selection": threshold_selection,
        "training": training_summary,
        "validation": validation,
        "seed_stability": {
            method: summarize_seed_stability(validation, method)
            for method in ("gacr_v3", "gacr_v4")
        },
        "integrity": integrity,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
