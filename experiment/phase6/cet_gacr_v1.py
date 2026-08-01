#!/usr/bin/env python3
"""Phase 6 CET-v1 x frozen GACR-v3 paired four-arm validation."""

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

from experiment.phase4.gacr_p0 import (  # noqa: E402
    relative_gain,
    select_fresh_validation_users,
    to_cpu_record,
)
from experiment.phase4.gacr_s0 import build_candidate_record  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    ROOT,
    build_validation_samples,
    prepare,
    read_users,
    sha256,
    stable_sha,
    write_json,
)
from experiment.phase6.gacr_v2 import rank_metrics  # noqa: E402
from experiment.phase6.gacr_v3 import v3_method_result  # noqa: E402

METHODS = ("GRAM", "CET_v1", "GACR_v3", "CET_v1_GACR_v3")


def load_arm_checkpoint(
    prepared: dict,
    checkpoint: Path,
    scope: str,
    device: torch.device,
) -> None:
    state = torch.load(checkpoint, map_location=device)
    if scope == "full_model":
        prepared["model"].load_state_dict(state, strict=True)
    elif scope == "backbone":
        prepared["model"].backbone.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"unsupported checkpoint scope: {scope}")
    prepared["model"].eval()


def select_combo_users(
    dataset: str,
    sequences: dict,
    config: dict,
) -> tuple[list[str], dict]:
    split_root = ROOT / config["inputs"]["split_root"] / dataset
    train_users = read_users(split_root / "train_users.txt")
    gcdh_validation = read_users(split_root / "validation_users.txt")
    exclusions = train_users | gcdh_validation
    prior_gacr_users: set[str] = set()
    all_users = set(sequences)
    count = int(config["validation_users_per_dataset"])
    for salt in config["prior_gacr_validation_salts"]:
        users = set(
            select_fresh_validation_users(
                all_users, exclusions, dataset, salt, count
            )
        )
        prior_gacr_users |= users
        exclusions |= users

    prior_cet_users: set[str] = set()
    for pattern in config["inputs"]["prior_cet_validation_files"]:
        path = ROOT / pattern.format(dataset=dataset)
        if path.is_file():
            prior_cet_users |= read_users(path)
    exclusions |= prior_cet_users
    users = select_fresh_validation_users(
        all_users,
        exclusions,
        dataset,
        config["validation_salt"],
        count,
    )
    selected = set(users)
    selected_sha = stable_sha(selected)
    expected_sha = config["expected_validation_user_sha256"][dataset]
    if selected_sha != expected_sha:
        raise RuntimeError(
            f"{dataset} validation user SHA mismatch: "
            f"expected={expected_sha} actual={selected_sha}"
        )
    return users, {
        "users": len(users),
        "validation_user_sha256": selected_sha,
        "gcdh_or_training_overlap": len(
            selected & (train_users | gcdh_validation)
        ),
        "prior_gacr_overlap": len(selected & prior_gacr_users),
        "prior_cet_overlap": len(selected & prior_cet_users),
        "prior_gacr_cohorts_excluded": len(
            config["prior_gacr_validation_salts"]
        ),
        "prior_cet_users_excluded": len(prior_cet_users),
    }


def build_arm_records(
    dataset: str,
    config: dict,
    p0_config: dict,
    arm: str,
    checkpoint: Path,
    scope: str,
    device: torch.device,
    validation_users: list[str] | None = None,
) -> tuple[dict, list[dict], list[str]]:
    prepared = prepare(dataset, p0_config, device)
    before = sha256(checkpoint)
    load_arm_checkpoint(prepared, checkpoint, scope, device)
    selection = {}
    if validation_users is None:
        validation_users, selection = select_combo_users(
            dataset, prepared["sequences"], config
        )
    samples = build_validation_samples(
        prepared["sequences"],
        set(validation_users),
        prepared["item2input"],
        prepared["item2lexid"],
    )
    records = []
    for index, sample in enumerate(samples, 1):
        records.append(
            to_cpu_record(build_candidate_record(sample, prepared, config, device))
        )
        if index % 64 == 0:
            print(
                f"CET_GACR_VALIDATION arm={arm} dataset={dataset} "
                f"users={index}/{len(samples)}",
                flush=True,
            )
    metadata = selection | {
        "arm": arm,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_scope": scope,
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": sha256(checkpoint),
        "records": len(records),
    }
    del prepared
    torch.cuda.empty_cache()
    return metadata, records, validation_users


def load_residual_state(
    dataset: str, seed: int, config: dict, device: torch.device
) -> tuple[dict, str]:
    path = (
        ROOT
        / config["inputs"]["gacr_residual_root"]
        / dataset
        / f"residual_seed{seed}.pt"
    )
    expected = config["inputs"]["expected_residual_sha256"][dataset][str(seed)]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{dataset} seed {seed} residual SHA mismatch: "
            f"expected={expected} actual={actual}"
        )
    return torch.load(path, map_location=device), actual


def method_fields(rank: int | None) -> dict[str, float | int | str]:
    recall10, ndcg10, recall50 = rank_metrics(rank)
    return {
        "rank": "" if rank is None else rank,
        "Recall@10": recall10,
        "NDCG@10": ndcg10,
        "Recall@50": recall50,
        "covered": int(rank is not None),
    }


def unify_rows(
    gram_records: list[dict],
    cet_records: list[dict],
    gacr_rows: list[dict],
    combo_rows: list[dict],
) -> list[dict]:
    sources = []
    for rows in (gram_records, cet_records, gacr_rows, combo_rows):
        indexed = {row["sample_key"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("duplicate sample key in paired arm")
        sources.append(indexed)
    keys = set(sources[0])
    if any(set(source) != keys for source in sources[1:]):
        raise ValueError("four-arm validation cohorts differ")
    result = []
    for key in sorted(keys):
        gram, cet, gacr, combo = (source[key] for source in sources)
        row = {
            "sample_key": key,
            "target_group": gram["target_group"],
        }
        ranks = {
            "GRAM": gram["gram_rank"],
            "CET_v1": cet["gram_rank"],
            "GACR_v3": gacr["candidate_rank"],
            "CET_v1_GACR_v3": combo["candidate_rank"],
        }
        for method, rank in ranks.items():
            for field, value in method_fields(rank).items():
                row[f"{method}_{field}"] = value
        result.append(row)
    return result


def summarize_method(rows: list[dict], method: str) -> dict:
    def summarize(selected: list[dict]) -> dict:
        return {
            "n": len(selected),
            "Recall@10": float(
                np.mean([row[f"{method}_Recall@10"] for row in selected])
            ),
            "NDCG@10": float(
                np.mean([row[f"{method}_NDCG@10"] for row in selected])
            ),
            "Recall@50": float(
                np.mean([row[f"{method}_Recall@50"] for row in selected])
            ),
            "coverage": float(
                np.mean([row[f"{method}_covered"] for row in selected])
            ),
        }

    groups = {"overall": summarize(rows)}
    for group in ("head", "tail"):
        groups[group] = summarize(
            [row for row in rows if row["target_group"] == group]
        )
    return groups


def paired_bootstrap(
    rows: list[dict],
    baseline: str,
    candidate: str,
    metric: str,
    relative: bool,
    seed: int,
    samples: int,
) -> list[float]:
    base = np.asarray([row[f"{baseline}_{metric}"] for row in rows], dtype=float)
    cand = np.asarray([row[f"{candidate}_{metric}"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(rows), size=len(rows))
        b_value = float(base[indices].mean())
        c_value = float(cand[indices].mean())
        values.append(
            relative_gain(b_value, c_value)
            if relative
            else c_value - b_value
        )
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def compare_methods(
    rows: list[dict], baseline: str, candidate: str, seed: int, config: dict
) -> dict:
    base = summarize_method(rows, baseline)
    cand = summarize_method(rows, candidate)
    tail_rows = [row for row in rows if row["target_group"] == "tail"]
    broad_harm = np.mean(
        [
            row[f"{baseline}_Recall@10"] == 1.0
            and row[f"{candidate}_Recall@10"] == 0.0
            for row in rows
        ]
    )
    changed = np.mean(
        [row[f"{baseline}_rank"] != row[f"{candidate}_rank"] for row in rows]
    )
    samples = int(config["evaluation"]["bootstrap_resamples"])
    return {
        "baseline": baseline,
        "candidate": candidate,
        "overall_ndcg10_relative_gain": relative_gain(
            base["overall"]["NDCG@10"], cand["overall"]["NDCG@10"]
        ),
        "overall_recall10_absolute_gain": (
            cand["overall"]["Recall@10"] - base["overall"]["Recall@10"]
        ),
        "tail_ndcg10_relative_gain": relative_gain(
            base["tail"]["NDCG@10"], cand["tail"]["NDCG@10"]
        ),
        "broad_harm_rate": float(broad_harm),
        "changed_user_coverage": float(changed),
        "bootstrap": {
            "overall_ndcg10_relative_gain_ci95": paired_bootstrap(
                rows, baseline, candidate, "NDCG@10", True, seed + 11, samples
            ),
            "overall_recall10_absolute_gain_ci95": paired_bootstrap(
                rows, baseline, candidate, "Recall@10", False, seed + 21, samples
            ),
            "tail_ndcg10_relative_gain_ci95": paired_bootstrap(
                tail_rows, baseline, candidate, "NDCG@10", True, seed + 31, samples
            ),
        },
    }


def decide(validation: dict, config: dict) -> dict:
    domain_means = {}
    for dataset, data in validation.items():
        domain_means[dataset] = {}
        for method in METHODS:
            values = [
                result["methods"][method]["overall"]["NDCG@10"]
                for result in data["seeds"].values()
            ]
            domain_means[dataset][method] = float(np.mean(values))
    macro = {
        method: float(
            np.mean(
                [domain_means[dataset][method] for dataset in domain_means]
            )
        )
        for method in METHODS
    }
    combo = "CET_v1_GACR_v3"
    stronger_macro = max(macro["CET_v1"], macro["GACR_v3"])
    macro_superiority = macro[combo] > stronger_macro
    per_domain_superiority = all(
        domain_means[dataset][combo]
        > max(
            domain_means[dataset]["CET_v1"],
            domain_means[dataset]["GACR_v3"],
        )
        for dataset in domain_means
    )
    safe = all(
        result["comparisons"]["combo_vs_gram"]["broad_harm_rate"]
        <= float(config["decision_rule"]["broad_harm_rate_max"])
        and result["comparisons"]["combo_vs_gram"][
            "overall_recall10_absolute_gain"
        ]
        >= float(config["decision_rule"]["recall10_absolute_floor"])
        for data in validation.values()
        for result in data["seeds"].values()
    )
    passed = macro_superiority and per_domain_superiority and safe
    return {
        "decision": (
            "KEEP_CET_V1_GACR_V3_COMBINATION"
            if passed
            else "RETURN_TO_STRONGER_SINGLE_METHOD"
        ),
        "macro_ndcg10": macro,
        "domain_mean_ndcg10": domain_means,
        "macro_exceeds_both_single_components": macro_superiority,
        "each_domain_exceeds_both_single_components": per_domain_superiority,
        "safety_gate_passed": safe,
    }


def validate_inputs(config: dict, code_path: Path) -> None:
    expected_code = config["integrity"]["code_sha256"]
    actual_code = sha256(code_path)
    if expected_code != "PENDING_FREEZE" and actual_code != expected_code:
        raise RuntimeError(
            f"implementation SHA mismatch: expected={expected_code} actual={actual_code}"
        )
    for dataset in config["datasets"]:
        for key, expected in config["inputs"]["expected_checkpoint_sha256"].items():
            path = ROOT / config["inputs"][key].format(dataset=dataset)
            actual = sha256(path)
            if actual != expected[dataset]:
                raise RuntimeError(
                    f"{key} {dataset} SHA mismatch: expected={expected[dataset]} "
                    f"actual={actual}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CET-v1 x GACR-v3 validation requires CUDA")
    config = json.loads(args.config.read_text())
    validate_inputs(config, Path(__file__))
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    device = torch.device("cuda:0")
    validation = {}
    lineage = {}
    for dataset in config["datasets"]:
        gram_checkpoint = ROOT / config["inputs"]["gram_checkpoint"].format(
            dataset=dataset
        )
        cet_checkpoint = ROOT / config["inputs"]["cet_v1_checkpoint"].format(
            dataset=dataset
        )
        gram_meta, gram_records, users = build_arm_records(
            dataset,
            config,
            p0_config,
            "GRAM",
            gram_checkpoint,
            "full_model",
            device,
        )
        cet_meta, cet_records, cet_users = build_arm_records(
            dataset,
            config,
            p0_config,
            "CET_v1",
            cet_checkpoint,
            "backbone",
            device,
            users,
        )
        if users != cet_users:
            raise ValueError("paired checkpoint arms received different users")
        output_dir = args.output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = {}
        lineage[dataset] = {
            "GRAM": gram_meta,
            "CET_v1": cet_meta,
            "residuals": {},
        }
        for seed in config["residual_seeds"]:
            state, residual_sha = load_residual_state(
                dataset, int(seed), config, device
            )
            _, gacr_rows = v3_method_result(
                gram_records,
                state,
                config,
                float(config["frozen_gacr_v3_budget"][dataset]),
                int(seed),
                device,
            )
            _, combo_rows = v3_method_result(
                cet_records,
                state,
                config,
                float(config["frozen_gacr_v3_budget"][dataset]),
                int(seed) + 1000,
                device,
            )
            rows = unify_rows(gram_records, cet_records, gacr_rows, combo_rows)
            methods = {method: summarize_method(rows, method) for method in METHODS}
            comparisons = {
                "cet_vs_gram": compare_methods(
                    rows, "GRAM", "CET_v1", int(seed) + 100, config
                ),
                "gacr_vs_gram": compare_methods(
                    rows, "GRAM", "GACR_v3", int(seed) + 200, config
                ),
                "combo_vs_gram": compare_methods(
                    rows, "GRAM", "CET_v1_GACR_v3", int(seed) + 300, config
                ),
                "combo_vs_cet": compare_methods(
                    rows, "CET_v1", "CET_v1_GACR_v3", int(seed) + 400, config
                ),
                "combo_vs_gacr": compare_methods(
                    rows, "GACR_v3", "CET_v1_GACR_v3", int(seed) + 500, config
                ),
            }
            path = output_dir / f"four_arm_seed{seed}_per_user.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            seeds[str(seed)] = {
                "methods": methods,
                "comparisons": comparisons,
                "per_user_sha256": sha256(path),
            }
            lineage[dataset]["residuals"][str(seed)] = residual_sha
        validation[dataset] = {
            key: value
            for key, value in gram_meta.items()
            if key not in {"arm", "checkpoint", "checkpoint_scope"}
        } | {"seeds": seeds}
        del gram_records, cet_records
        torch.cuda.empty_cache()

    decision = decide(validation, config)
    integrity = {
        "paired_four_arm_cohorts": all(
            validation[d]["users"] == int(config["validation_users_per_dataset"])
            for d in config["datasets"]
        ),
        "fresh_validation_zero_overlap": all(
            validation[d]["gcdh_or_training_overlap"] == 0
            and validation[d]["prior_gacr_overlap"] == 0
            and validation[d]["prior_cet_overlap"] == 0
            for d in config["datasets"]
        ),
        "checkpoint_sha_unchanged": all(
            item["checkpoint_sha256_before"] == item["checkpoint_sha256_after"]
            for dataset in lineage.values()
            for key, item in dataset.items()
            if key in {"GRAM", "CET_v1"}
        ),
        "frozen_residuals_only": True,
        "backbone_optimizer_steps": 0,
        "test_data_read": False,
        "sports_data_read": False,
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "result_status": "RESULTS_READY_FOR_RESEARCHER_ANALYSIS",
        "design": "paired frozen four-arm validation",
        "methods": list(METHODS),
        "validation": validation,
        "decision_gate": decision,
        "lineage": lineage,
        "integrity": integrity,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
