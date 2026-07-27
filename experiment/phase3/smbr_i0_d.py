#!/usr/bin/env python3
"""SMBR I0-D: training-only fixed-budget benefit learnability audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GRAM_SRC = ROOT / "GRAM/src"
for path in (GRAM_SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from processor import CollatorGRAM  # noqa: E402
from utils import indexing  # noqa: E402

from cgi_e0 import bootstrap_mean, lexical_mean_logprob, write_csv  # noqa: E402
from cpbd_g0_d2 import metadata_first_passage  # noqa: E402
from hbtr_b1_smoke import (  # noqa: E402
    DATASETS,
    create_model_and_tokenizer,
    make_runtime_args,
    read_sequences,
    sha256,
)

FEATURES = [
    "history_length",
    "recoverable_sum",
    "recoverable_mean",
    "recoverable_max",
    "recoverable_positive_fraction",
    "recoverable_ge8_fraction",
    "displaced_cf_sum",
    "displaced_cf_mean",
    "metadata_lost_sum",
    "metadata_lost_mean",
    "metadata_retention_mean",
    "metadata_retention_min",
    "current_metadata_visible_sum",
    "current_cf_visible_sum",
    "log1p_popularity_mean",
    "log1p_popularity_min",
    "tail_item_fraction",
    "top_k_similar_item_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "artifacts/phase3/configs/smbr_i0_d_preregistered.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/phase3/smbr_i0_d",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/第三阶段/GRAM_第三阶段_SMBR_I0_D诊断报告.md",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def digest_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def split_for_user(seed: int, dataset: str, user: str, config: dict) -> str:
    bucket = digest_int(f"{seed}|{dataset}|{user}") % config["split_hash_modulus"]
    for name, spec in config["splits"].items():
        lower, upper = spec["hash_buckets"]
        if lower <= bucket < upper:
            return name
    raise ValueError(f"unassigned hash bucket {bucket}")


def selection_hash(seed: int, dataset: str, split: str, user: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|{split}|{user}".encode()).hexdigest()


def read_census(path: Path) -> dict[str, dict]:
    numeric = {
        "popularity_train",
        "top_k_similar_item",
        "current_cf_visible",
        "current_metadata_visible",
        "current_metadata_lost",
        "recoverable_metadata_tokens",
        "displaced_cf_tokens",
        "current_metadata_retention",
    }
    result = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in numeric:
                row[key] = float(row[key])
            result[row["item"]] = row
    return result


def feature_row(history: list[str], census: dict[str, dict]) -> dict[str, float]:
    rows = [census[item] for item in history]
    recoverable = np.asarray(
        [row["recoverable_metadata_tokens"] for row in rows], dtype=np.float64
    )
    displaced = np.asarray(
        [row["displaced_cf_tokens"] for row in rows], dtype=np.float64
    )
    metadata_lost = np.asarray(
        [row["current_metadata_lost"] for row in rows], dtype=np.float64
    )
    retention = np.asarray(
        [row["current_metadata_retention"] for row in rows], dtype=np.float64
    )
    popularity = np.asarray(
        [row["popularity_train"] for row in rows], dtype=np.float64
    )
    return {
        "history_length": float(len(history)),
        "recoverable_sum": float(recoverable.sum()),
        "recoverable_mean": float(recoverable.mean()),
        "recoverable_max": float(recoverable.max()),
        "recoverable_positive_fraction": float((recoverable > 0).mean()),
        "recoverable_ge8_fraction": float((recoverable >= 8).mean()),
        "displaced_cf_sum": float(displaced.sum()),
        "displaced_cf_mean": float(displaced.mean()),
        "metadata_lost_sum": float(metadata_lost.sum()),
        "metadata_lost_mean": float(metadata_lost.mean()),
        "metadata_retention_mean": float(retention.mean()),
        "metadata_retention_min": float(retention.min()),
        "current_metadata_visible_sum": float(
            sum(row["current_metadata_visible"] for row in rows)
        ),
        "current_cf_visible_sum": float(
            sum(row["current_cf_visible"] for row in rows)
        ),
        "log1p_popularity_mean": float(np.log1p(popularity).mean()),
        "log1p_popularity_min": float(np.log1p(popularity).min()),
        "tail_item_fraction": float(
            np.mean([row["popularity_stratum"] != "top50" for row in rows])
        ),
        "top_k_similar_item_mean": float(
            np.mean([row["top_k_similar_item"] for row in rows])
        ),
    }


def build_cohort(
    dataset: str,
    sequences: dict[str, list[str]],
    item2input: dict[str, str],
    item2lexid: dict[str, str],
    census: dict[str, dict],
    config: dict,
) -> tuple[list[dict], dict]:
    candidates = {name: [] for name in config["splits"]}
    rejection = Counter()
    offset = int(config["target_offset_from_end"])
    for user, items in sequences.items():
        if len(items) < offset + config["min_history"]:
            rejection["short_sequence"] += 1
            continue
        target = items[-offset]
        history = items[:-offset][-config["max_history"] :]
        if len(history) < config["min_history"]:
            rejection["short_history"] += 1
            continue
        if target not in item2lexid:
            rejection["target_not_indexed"] += 1
            continue
        if any(
            item not in item2input or item not in item2lexid or item not in census
            for item in history
        ):
            rejection["history_not_indexed"] += 1
            continue
        split = split_for_user(config["seed"], dataset, user, config)
        features = feature_row(history, census)
        candidates[split].append(
            {
                "user": user,
                "split": split,
                "target": target,
                "history": history,
                "newest_item": history[-1],
                "oldest_item": history[0],
                "sample_hash": selection_hash(
                    config["seed"], dataset, split, user
                ),
                **features,
            }
        )
    selected = []
    available = {}
    for split, values in candidates.items():
        values.sort(key=lambda row: row["sample_hash"])
        available[split] = len(values)
        selected.extend(values[: config["splits"][split]["max_users"]])
    selected.sort(key=lambda row: (row["split"], row["sample_hash"]))
    return selected, {"available": available, "rejection": dict(rejection)}


def make_samples(rows, item2input, item2lexid):
    samples = []
    for row in rows:
        ordered = list(reversed(row["history"]))
        history_lex = " ; ".join(item2lexid[item] for item in ordered)
        samples.append(
            {
                **row,
                "input": [f"What would user purchase after {history_lex} ?"]
                + [item2input[item] for item in ordered],
                "output": item2lexid[row["target"]],
            }
        )
    return samples


@torch.no_grad()
def score_samples(model, tokenizer, collator, samples, config, device):
    model.eval()
    output = []
    repeat_max = 0.0
    raw_identity = 0
    expected_passages = sum(len(row["history"]) for row in samples)
    for start in range(0, len(samples), config["batch_size"]):
        current = samples[start : start + config["batch_size"]]
        alternate = []
        for sample in current:
            alt = [sample["input"][0]]
            for passage in sample["input"][1:]:
                rebuilt, _, components = metadata_first_passage(passage)
                raw_identity += int(
                    components["link"] in rebuilt
                    and components["metadata"] in rebuilt
                    and all(value in rebuilt for value in components["cf_values"])
                )
                alt.append(rebuilt)
            alternate.append({**sample, "input": alt})
        packed = {}
        for name, chunk in (("current", current), ("recover", alternate)):
            packed[name] = collator(
                [
                    {
                        "input": row["input"],
                        "output": row["output"],
                        "user_id": row["user"],
                    }
                    for row in chunk
                ]
            )
        if not torch.equal(
            packed["current"]["target_ids"], packed["recover"]["target_ids"]
        ):
            raise ValueError("target changed across conditions")
        if (
            packed["current"]["item_text_ids"].shape
            != packed["recover"]["item_text_ids"].shape
        ):
            raise ValueError("fixed budget shape changed")
        scores = {}
        for name in ("current", "recover"):
            batch = packed[name]
            model_output = model(
                input_ids=batch["item_text_ids"].to(device),
                attention_mask=batch["item_text_masks"].to(device),
                labels=batch["target_ids"].to(device),
                return_dict=True,
            )
            scores[name] = lexical_mean_logprob(
                model_output.logits,
                batch["target_ids"].to(device),
                tokenizer.eos_token_id,
            ).cpu().numpy()
        repeated = model(
            input_ids=packed["current"]["item_text_ids"].to(device),
            attention_mask=packed["current"]["item_text_masks"].to(device),
            labels=packed["current"]["target_ids"].to(device),
            return_dict=True,
        )
        repeat = lexical_mean_logprob(
            repeated.logits,
            packed["current"]["target_ids"].to(device),
            tokenizer.eos_token_id,
        ).cpu().numpy()
        repeat_max = max(
            repeat_max, float(np.max(np.abs(repeat - scores["current"])))
        )
        for index, sample in enumerate(current):
            benefit = float(scores["recover"][index] - scores["current"][index])
            output.append(
                {
                    **{key: sample[key] for key in ("user", "split", "sample_hash")},
                    "lp_current": float(scores["current"][index]),
                    "lp_recover": float(scores["recover"][index]),
                    "benefit": benefit,
                    "positive_label": int(benefit > 0.02),
                }
            )
    return output, {
        "current_repeat_max_abs_error": repeat_max,
        "fixed_budget_rate": 1.0,
        "raw_component_identity_rate": raw_identity / expected_passages,
    }


def bootstrap_auc(labels, scores, iterations, seed):
    observed = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    values = []
    while len(values) < iterations:
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) == 2:
            values.append(roc_auc_score(labels[idx], scores[idx]))
    return {
        "value": observed,
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def wilson(successes, total):
    if total == 0:
        return [0.0, 0.0]
    z = float(norm.ppf(0.975))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2))
    return [center - half / denominator, center + half / denominator]


def ece(labels, probabilities, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            mask = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        if mask.any():
            value += mask.mean() * abs(
                probabilities[mask].mean() - labels[mask].mean()
            )
    return float(value)


def select_threshold(probabilities, labels, benefits, config):
    candidates = []
    lower_rate, upper_rate = config["eligible_active_rate"]
    for threshold in config["candidate_probability_thresholds"]:
        active = probabilities >= threshold
        rate = float(active.mean())
        if active.any():
            precision = float(labels[active].mean())
            mean_benefit = float(benefits[active].mean())
        else:
            precision = 0.0
            mean_benefit = float("-inf")
        if (
            lower_rate <= rate <= upper_rate
            and precision >= config["minimum_calibration_precision"]
        ):
            candidates.append(
                (mean_benefit, precision, threshold, rate, int(active.sum()))
            )
    if not candidates:
        return None
    mean_benefit, precision, threshold, rate, count = max(candidates)
    return {
        "threshold": threshold,
        "active_rate": rate,
        "active_count": count,
        "precision": precision,
        "mean_benefit": mean_benefit,
    }


def matched_baselines(rows, labels, benefits, active_count, seed):
    hashes = np.asarray([row["sample_hash"] for row in rows])
    recoverable = np.asarray([row["recoverable_sum"] for row in rows])
    ratio = np.asarray(
        [row["recoverable_sum"] / (1 + row["displaced_cf_sum"]) for row in rows]
    )

    def top_mask(values):
        order = sorted(
            range(len(values)), key=lambda i: (-values[i], hashes[i])
        )
        mask = np.zeros(len(values), dtype=bool)
        mask[order[:active_count]] = True
        return mask

    rng = np.random.default_rng(seed)
    random_gains = np.empty(1000)
    for index in range(1000):
        chosen = rng.choice(len(rows), active_count, replace=False)
        random_gains[index] = benefits[chosen].sum() / len(rows)
    result = {}
    for name, mask in (
        ("recoverable_threshold", top_mask(recoverable)),
        ("displacement_ratio", top_mask(ratio)),
    ):
        result[name] = {
            "policy_gain": float((benefits * mask).mean()),
            "active_mean_benefit": float(benefits[mask].mean()),
            "precision": float(labels[mask].mean()),
        }
    result["matched_random"] = {
        "policy_gain": float(random_gains.mean()),
        "policy_gain_ci95": [
            float(np.quantile(random_gains, 0.025)),
            float(np.quantile(random_gains, 0.975)),
        ],
    }
    result["always_recover"] = {
        "policy_gain": float(benefits.mean()),
        "active_mean_benefit": float(benefits.mean()),
        "precision": float(labels.mean()),
    }
    result["oracle"] = {
        "policy_gain": float(np.maximum(benefits, 0).mean()),
        "active_rate": float((benefits > 0).mean()),
    }
    return result


def analyze(dataset, cohort, score_rows, config):
    by_key = {(row["user"], row["split"]): row for row in score_rows}
    rows = []
    for row in cohort:
        score = by_key[(row["user"], row["split"])]
        rows.append({**row, **score})
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in config["splits"]
    }

    def matrix(values):
        return np.asarray(
            [[float(row[key]) for key in FEATURES] for row in values],
            dtype=np.float64,
        )

    fit = split_rows["fit"]
    y_fit = np.asarray([row["positive_label"] for row in fit])
    if len(np.unique(y_fit)) != 2:
        return rows, {"status": "NO_FIT_CLASS_VARIATION", "gates": {}}
    model_spec = config["primary_model"]
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=model_spec["C"],
                    penalty=model_spec["penalty"],
                    class_weight=model_spec["class_weight"],
                    solver=model_spec["solver"],
                    max_iter=model_spec["max_iter"],
                    random_state=model_spec["random_state"],
                ),
            ),
        ]
    )
    model.fit(matrix(fit), y_fit)
    for split, values in split_rows.items():
        probs = model.predict_proba(matrix(values))[:, 1]
        for row, probability in zip(values, probs):
            row["predicted_probability"] = float(probability)
    calibration = split_rows["calibration"]
    cal_probs = np.asarray([row["predicted_probability"] for row in calibration])
    cal_labels = np.asarray([row["positive_label"] for row in calibration])
    cal_benefits = np.asarray([row["benefit"] for row in calibration])
    threshold = select_threshold(
        cal_probs,
        cal_labels,
        cal_benefits,
        config["calibration_threshold_rule"],
    )
    if threshold is None:
        return rows, {
            "status": "NO_CALIBRATED_SUBSET",
            "threshold": None,
            "gates": {"calibrated_subset": False},
        }
    audit = split_rows["audit"]
    probabilities = np.asarray([row["predicted_probability"] for row in audit])
    labels = np.asarray([row["positive_label"] for row in audit])
    benefits = np.asarray([row["benefit"] for row in audit])
    if len(np.unique(labels)) != 2:
        return rows, {"status": "NO_AUDIT_CLASS_VARIATION", "gates": {}}
    active = probabilities >= threshold["threshold"]
    active_benefits = benefits[active]
    policy_gains = benefits * active
    auc = bootstrap_auc(
        labels, probabilities, config["bootstrap_iterations"], config["seed"]
    )
    recoverable_auc = float(
        roc_auc_score(
            labels, np.asarray([row["recoverable_sum"] for row in audit])
        )
    )
    active_stats = bootstrap_mean(
        active_benefits, config["bootstrap_iterations"], config["seed"] + 1
    )
    policy_stats = bootstrap_mean(
        policy_gains, config["bootstrap_iterations"], config["seed"] + 2
    )
    precision = float(labels[active].mean())
    precision_ci = wilson(int(labels[active].sum()), int(active.sum()))
    brier = float(brier_score_loss(labels, probabilities))
    prevalence_brier = float(
        brier_score_loss(labels, np.full(len(labels), labels.mean()))
    )
    baselines = matched_baselines(
        audit,
        labels,
        benefits,
        int(active.sum()),
        config["seed"] + 3,
    )
    gates_spec = config["learnability_gates_per_dataset"]
    non_oracle = [
        baselines[name]["policy_gain"]
        for name in (
            "recoverable_threshold",
            "displacement_ratio",
            "matched_random",
            "always_recover",
        )
    ]
    gates = {
        "calibrated_subset": True,
        "auroc": auc["value"] >= gates_spec["audit_auroc_point_min"]
        and auc["ci95"][0]
        > gates_spec["audit_auroc_ci95_lower_strictly_greater_than"],
        "calibration": brier < prevalence_brier
        and ece(labels, probabilities) <= gates_spec["audit_ece_max"],
        "active_rate": gates_spec["audit_active_rate_min"]
        <= active.mean()
        <= gates_spec["audit_active_rate_max"],
        "precision": precision_ci[0]
        > gates_spec[
            "audit_precision_wilson_ci95_lower_strictly_greater_than"
        ],
        "active_utility": active_stats["mean"]
        >= gates_spec["audit_active_mean_benefit_min"]
        and active_stats["ci95"][0]
        > gates_spec[
            "audit_active_mean_benefit_ci95_lower_strictly_greater_than"
        ],
        "policy_utility": policy_stats["ci95"][0]
        > gates_spec[
            "audit_policy_mean_gain_ci95_lower_strictly_greater_than"
        ],
        "recoverable_auc_margin": auc["value"] - recoverable_auc
        >= gates_spec["audit_auroc_margin_over_recoverable_threshold_min"],
        "non_oracle_baselines": policy_stats["mean"] > max(non_oracle),
    }
    return rows, {
        "status": "ANALYZED",
        "threshold": threshold,
        "audit": {
            "n": len(audit),
            "prevalence": float(labels.mean()),
            "auroc": auc,
            "recoverable_sum_auroc": recoverable_auc,
            "auroc_margin": auc["value"] - recoverable_auc,
            "brier": brier,
            "constant_prevalence_brier": prevalence_brier,
            "ece10": ece(labels, probabilities),
            "active_count": int(active.sum()),
            "active_rate": float(active.mean()),
            "precision": precision,
            "precision_wilson_ci95": precision_ci,
            "active_benefit": active_stats,
            "policy_gain": policy_stats,
        },
        "baselines": baselines,
        "gates": gates,
    }


def preflight_dataset(dataset, spec, config):
    dataset_dir = ROOT / "GRAM/rec_datasets" / dataset
    from transformers import AutoTokenizer

    runtime = make_runtime_args(dataset)
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    sequences = read_sequences(dataset_dir / "user_sequence.txt")
    _, item2input, item2lexid = indexing.gram_indexing(
        data_path=runtime.data_path,
        dataset=dataset,
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        args=runtime,
        user_id_without_target_item=True,
        id_linking=True,
    )
    census = read_census(ROOT / spec["census"])
    cohort, availability = build_cohort(
        dataset, sequences, item2input, item2lexid, census, config
    )
    counts = Counter(row["split"] for row in cohort)
    expected = {
        name: split["max_users"] for name, split in config["splits"].items()
    }
    users = {name: {row["user"] for row in cohort if row["split"] == name} for name in expected}
    overlap = sum(
        len(users[a] & users[b])
        for a, b in (("fit", "calibration"), ("fit", "audit"), ("calibration", "audit"))
    )
    integrity = {
        "exact_split_caps": dict(counts) == expected,
        "user_overlap_across_splits": overlap,
        "heldout_sequence_fields_read": False,
        "target_feature_inclusion_rate": 0.0,
        "feature_names_match_preregistration": FEATURES == config["features"],
        "finite_feature_rate": float(
            np.isfinite(
                [[row[key] for key in FEATURES] for row in cohort]
            ).mean()
        ),
    }
    return cohort, availability, integrity


def write_report(path, aggregate):
    lines = [
        "# GRAM 第三阶段：SMBR I0-D training-only benefit learnability",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-07-24",
        f"- Verification Status: {aggregate['material_passport']['verification_status']}",
        f"- Version Label: `{aggregate['material_passport']['version_label']}`",
        "",
        f"固定决策：**`{aggregate['decision']}`**。",
        "",
        "本阶段仅使用 `sequence[-3]` training target 与 `sequence[:-3]` history；",
        "未读取 validation/test，未更新 GRAM。",
        "",
        "## 数据集结果",
        "",
        "| Dataset | Status | Integrity | Learnability |",
        "|---|---|---:|---:|",
    ]
    for dataset, result in aggregate["datasets"].items():
        gates = result.get("analysis", {}).get("gates", {})
        integrity = result["integrity"]
        integrity_pass = (
            integrity.get("exact_split_caps", False)
            and integrity.get("user_overlap_across_splits") == 0
            and not integrity.get("heldout_sequence_fields_read", True)
            and integrity.get("target_feature_inclusion_rate") == 0
        )
        lines.append(
            f"| {dataset} | {result['status']} | "
            f"{integrity_pass} | "
            f"{bool(gates) and all(gates.values())} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    with args.config.open() as handle:
        config = json.load(handle)
    if not config.get("preregistered_before_new_training_prefix_scores"):
        raise ValueError("I0-D is not preregistered")
    if FEATURES != config["features"]:
        raise ValueError("feature list differs from preregistration")
    started = time.time()
    preflight = {}
    runtime_data = {}
    for dataset, spec in config["datasets"].items():
        cohort, availability, integrity = preflight_dataset(dataset, spec, config)
        preflight[dataset] = {
            "cohort": cohort,
            "availability": availability,
            "integrity": integrity,
        }
        output_dir = args.output_root / dataset
        write_csv(
            output_dir / "cohort.csv",
            cohort,
            ["user", "split", "target", "newest_item", "oldest_item", "sample_hash"]
            + FEATURES,
        )
        runtime_data[dataset] = {
            "status": "PREFLIGHT_COMPLETE",
            "counts": dict(Counter(row["split"] for row in cohort)),
            "availability": availability,
            "integrity": integrity,
        }
    if args.preflight_only:
        aggregate = {
            "material_passport": {
                "origin_skill": "academic-research-suite / experiment-agent",
                "origin_mode": "run",
                "origin_date": "2026-07-24",
                "verification_status": "ANALYZED_PREFLIGHT_ONLY",
                "version_label": "smbr_i0_d_preflight_v1",
            },
            "decision": "PREFLIGHT_COMPLETE_SCORING_NOT_RUN",
            "datasets": runtime_data,
            "config_sha256": sha256(args.config),
            "code_sha256": sha256(Path(__file__)),
            "wall_time_seconds": time.time() - started,
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "preflight_summary.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps({"decision": aggregate["decision"]}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-checkpoint label scoring")

    results = {}
    device = torch.device("cuda:0")
    for dataset, spec in config["datasets"].items():
        cohort = preflight[dataset]["cohort"]
        model, tokenizer, runtime = create_model_and_tokenizer(dataset, device)
        _, item2input, item2lexid = indexing.gram_indexing(
            data_path=runtime.data_path,
            dataset=dataset,
            model_gen=None,
            tokenizer=tokenizer,
            regenerate=False,
            phase=0,
            args=runtime,
            user_id_without_target_item=True,
            id_linking=True,
        )
        samples = make_samples(cohort, item2input, item2lexid)
        collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
        score_rows, score_integrity = score_samples(
            model, tokenizer, collator, samples, config, device
        )
        analyzed_rows, analysis = analyze(dataset, cohort, score_rows, config)
        integrity = {
            **preflight[dataset]["integrity"],
            **score_integrity,
            "finite_score_rate": float(
                np.isfinite(
                    [
                        [row["lp_current"], row["lp_recover"], row["benefit"]]
                        for row in score_rows
                    ]
                ).mean()
            ),
        }
        integrity_pass = (
            integrity["exact_split_caps"]
            and integrity["user_overlap_across_splits"] == 0
            and not integrity["heldout_sequence_fields_read"]
            and integrity["target_feature_inclusion_rate"] == 0
            and integrity["feature_names_match_preregistration"]
            and integrity["finite_feature_rate"] == 1
            and integrity["current_repeat_max_abs_error"]
            <= config["integrity_gates"]["current_repeat_max_abs_error"]
            and integrity["fixed_budget_rate"] == 1
            and integrity["raw_component_identity_rate"] == 1
            and integrity["finite_score_rate"] == 1
        )
        status = "ANALYZED" if integrity_pass else "EXECUTION_INVALID"
        output_dir = args.output_root / dataset
        write_csv(
            output_dir / "counterfactual_scores.csv",
            score_rows,
            list(score_rows[0]),
        )
        write_csv(
            output_dir / "policy_rows.csv",
            analyzed_rows,
            [
                "user",
                "split",
                "sample_hash",
                *FEATURES,
                "lp_current",
                "lp_recover",
                "benefit",
                "positive_label",
                "predicted_probability",
            ],
        )
        result = {
            "status": status,
            "counts": dict(Counter(row["split"] for row in cohort)),
            "integrity": integrity,
            "analysis": analysis,
        }
        (output_dir / "diagnostic_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        results[dataset] = result
        del model
        torch.cuda.empty_cache()

    if any(result["status"] == "EXECUTION_INVALID" for result in results.values()):
        decision = "EXECUTION_INVALID"
    elif any(
        result["analysis"]["status"] == "NO_CALIBRATED_SUBSET"
        for result in results.values()
    ):
        decision = "STOP_SMBR_NO_CALIBRATED_SUBSET"
    elif not all(
        result["analysis"].get("gates")
        and all(result["analysis"]["gates"].values())
        for result in results.values()
    ):
        decision = "STOP_SMBR_NO_BENEFIT_LEARNABILITY"
    else:
        decision = "I1_DESIGN_ALLOWED"
    aggregate = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-24",
            "verification_status": "ANALYZED",
            "version_label": "smbr_i0_d_v1",
        },
        "decision": decision,
        "datasets": results,
        "config_sha256": sha256(args.config),
        "code_sha256": sha256(Path(__file__)),
        "wall_time_seconds": time.time() - started,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(args.report, aggregate)
    print(json.dumps({"decision": decision}))


if __name__ == "__main__":
    main()
