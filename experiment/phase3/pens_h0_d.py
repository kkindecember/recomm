#!/usr/bin/env python3
"""PENS H0-D frozen direction-preserving norm diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GRAM_SRC = ROOT / "GRAM/src"
for path in (GRAM_SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from processor import CollatorGRAM  # noqa: E402
from utils import indexing  # noqa: E402

from cgi_e0 import (  # noqa: E402
    bootstrap_mean,
    build_cohort,
    lexical_mean_logprob,
    make_samples,
    write_csv,
)
from hbtr_b1_smoke import (  # noqa: E402
    DATASETS,
    create_model_and_tokenizer,
    read_sequences,
    sha256,
)
from lei_f0 import read_locked_cohort  # noqa: E402
from s0_offline_diagnostics import decode_item_ids, read_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "artifacts/phase3/configs/pens_h0_d_preregistered.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/phase3/pens_h0_d",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/第三阶段/GRAM_第三阶段_PENS_H0_D诊断报告.md",
    )
    return parser.parse_args()


def training_position_exposure(
    sequences: dict[str, list[str]], max_history: int = 20
) -> np.ndarray:
    """Replay load_train prefix lengths without using the final two held-out items."""
    exposure = np.zeros(max_history + 1, dtype=np.int64)
    for sequence in sequences.values():
        training_items = sequence[:-2]
        for prefix_end in range(1, len(training_items)):
            history_length = min(prefix_end, max_history)
            exposure[0] += 1
            exposure[1 : history_length + 1] += 1
    return exposure


def equalize_fine_norm(table: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if table.ndim != 2 or table.shape[0] != 21:
        raise ValueError(f"expected 21xd position table, got {tuple(table.shape)}")
    original = table.detach().clone()
    norms = original.norm(dim=1)
    if torch.any(norms[1:] <= 0):
        raise ValueError("fine position table contains a zero-norm vector")
    # The preregistered target is the conventional sample median.  There are
    # 20 fine positions, so use the midpoint of the two central order stats;
    # torch.median alone returns the lower central value for an even count.
    target = torch.quantile(norms[1:].float(), 0.5).to(norms.dtype)
    equalized = original.clone()
    equalized[1:] = original[1:] / norms[1:, None] * target
    cosine = torch.nn.functional.cosine_similarity(
        original[1:].float(), equalized[1:].float(), dim=1
    )
    audit = {
        "target_norm": float(target.item()),
        "equal_norm_max_abs_error": float(
            torch.max(torch.abs(equalized[1:].norm(dim=1) - target)).item()
        ),
        "direction_cosine_min": float(cosine.min().item()),
        "coarse_unchanged_max_abs_error": float(
            torch.max(torch.abs(equalized[0] - original[0])).item()
        ),
    }
    return equalized, audit


@torch.no_grad()
def score_samples(
    model,
    tokenizer,
    collator,
    samples: list[dict],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], dict]:
    model.eval()
    weight = model.position_embedding.weight
    if weight.data_ptr() != model.encoder.position_embedding.weight.data_ptr():
        raise ValueError("model and encoder position embeddings do not share storage")
    current = weight.detach().clone()
    equalized, intervention = equalize_fine_norm(current)
    zeroed = torch.zeros_like(current)
    tables = {
        "current": current,
        "equal_fine_norm": equalized,
        "zero_all_position": zeroed,
    }
    rows = []
    repeat_max = 0.0
    try:
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = collator(
                [
                    {
                        "input": row["input"],
                        "output": row["output"],
                        "user_id": row["user"],
                    }
                    for row in chunk
                ]
            )
            ids = batch["item_text_ids"].to(device)
            masks = batch["item_text_masks"].to(device)
            labels = batch["target_ids"].to(device)
            scores = {}
            for condition, table in tables.items():
                weight.copy_(table)
                output = model(
                    input_ids=ids,
                    attention_mask=masks,
                    labels=labels,
                    return_dict=True,
                )
                scores[condition] = lexical_mean_logprob(
                    output.logits, labels, tokenizer.eos_token_id
                ).cpu().numpy()
            weight.copy_(current)
            repeated = model(
                input_ids=ids,
                attention_mask=masks,
                labels=labels,
                return_dict=True,
            )
            repeated_scores = lexical_mean_logprob(
                repeated.logits, labels, tokenizer.eos_token_id
            ).cpu().numpy()
            repeat_max = max(
                repeat_max,
                float(np.max(np.abs(repeated_scores - scores["current"]))),
            )
            for index, sample in enumerate(chunk):
                lp_current = float(scores["current"][index])
                lp_equal = float(scores["equal_fine_norm"][index])
                lp_zero = float(scores["zero_all_position"][index])
                rows.append(
                    {
                        "user": sample["user"],
                        "stratum": sample["stratum"],
                        "history_length": sample["history_length"],
                        "lp_current": lp_current,
                        "lp_equal_fine_norm": lp_equal,
                        "lp_zero_all_position": lp_zero,
                        "norm_only_gain": lp_equal - lp_current,
                        "zero_position_delta": lp_zero - lp_current,
                    }
                )
    finally:
        weight.copy_(current)
    intervention["current_repeat_max_abs_error"] = repeat_max
    intervention["restored_current_max_abs_error"] = float(
        torch.max(torch.abs(weight - current)).item()
    )
    intervention["shared_position_storage"] = True
    return rows, intervention


def summarize(rows: list[dict], iterations: int, seed: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["stratum"]].append(row)
    summary = {}
    for stratum, values in sorted(grouped.items()):
        summary[stratum] = {}
        for offset, metric in enumerate(("norm_only_gain", "zero_position_delta")):
            array = np.asarray([row[metric] for row in values], dtype=np.float64)
            summary[stratum][metric] = bootstrap_mean(
                array, iterations, seed + offset
            )
    for label, lower, upper in (
        ("history_2_4", 2, 4),
        ("history_5_9", 5, 9),
        ("history_10_14", 10, 14),
        ("history_15_19", 15, 19),
        ("history_20", 20, 20),
    ):
        values = [
            row["norm_only_gain"]
            for row in rows
            if lower <= int(row["history_length"]) <= upper
        ]
        if values:
            summary[label] = bootstrap_mean(
                np.asarray(values, dtype=np.float64), iterations, seed + lower
            )
    return summary


def position_census(
    model, sequences: dict[str, list[str]], max_history: int
) -> tuple[list[dict], dict]:
    table = model.position_embedding.weight.detach().cpu()
    norms = table.norm(dim=1).numpy()
    exposure = training_position_exposure(sequences, max_history)
    pearson = float(np.corrcoef(exposure[1:], norms[1:])[0, 1])
    ratio = float(norms[20] / norms[1])
    rows = [
        {
            "position": position,
            "training_prefix_exposure": int(exposure[position]),
            "l2_norm": float(norms[position]),
        }
        for position in range(21)
    ]
    return rows, {
        "exposure_norm_pearson": pearson,
        "position20_to_position1_norm_ratio": ratio,
        "position1_norm": float(norms[1]),
        "position20_norm": float(norms[20]),
        "fine_norm_median": float(np.median(norms[1:])),
    }


def evaluate_gates(
    counts: dict,
    cohort_match: bool,
    stats: dict,
    structural: dict,
    intervention: dict,
    config: dict,
) -> dict:
    integrity_gates = config["integrity_gates"]
    mechanism = config["mechanism_gates_per_dataset"]
    integrity = (
        counts.get("tail_miss", 0) == integrity_gates["tail_miss_n"]
        and counts.get("tail_hit", 0) == integrity_gates["tail_hit_n"]
        and cohort_match
        and intervention["current_repeat_max_abs_error"]
        <= integrity_gates["current_repeat_max_abs_error"]
        and intervention["restored_current_max_abs_error"]
        <= integrity_gates["restored_current_max_abs_error"]
        and intervention["equal_norm_max_abs_error"]
        <= integrity_gates["equal_norm_max_abs_error"]
        and intervention["direction_cosine_min"]
        >= integrity_gates["direction_cosine_min"]
        and intervention["coarse_unchanged_max_abs_error"]
        <= integrity_gates["coarse_unchanged_max_abs_error"]
    )
    structural_replication = (
        structural["exposure_norm_pearson"]
        <= mechanism["exposure_norm_pearson_max"]
        and structural["position20_to_position1_norm_ratio"]
        >= mechanism["position20_to_position1_norm_ratio_min"]
    )
    miss = stats["tail_miss"]["norm_only_gain"]
    causal_benefit = (
        miss["mean"] >= mechanism["tail_miss_gain_mean"]
        and miss["ci95"][0] > mechanism["strict_bootstrap_lower_bound"]
        and miss["positive_rate"] >= mechanism["tail_miss_gain_positive_rate"]
    )
    no_broad_harm = (
        stats["tail_hit"]["norm_only_gain"]["mean"]
        >= mechanism["tail_hit_gain_mean_min"]
    )
    return {
        "integrity": bool(integrity),
        "structural_replication": bool(structural_replication),
        "causal_benefit": bool(causal_benefit),
        "no_broad_harm": bool(no_broad_harm),
    }


def run_dataset(
    dataset: str,
    spec: dict,
    config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    dataset_dir = ROOT / "GRAM/rec_datasets" / dataset
    paths = {
        "checkpoint": ROOT / spec["checkpoint"],
        "run_config": ROOT / spec["run_config"],
        "predictions": ROOT / spec["predictions"],
        "s0_summary": ROOT / spec["s0_summary"],
        "cgi_cohort": ROOT / spec["cgi_cohort"],
        "user_sequence": dataset_dir / "user_sequence.txt",
        "item_index": dataset_dir
        / f"item_generative_indexing_{DATASETS[dataset]['hierarchical_id_type']}.txt",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(DATASETS[dataset]["checkpoint"]).resolve() != paths[
        "checkpoint"
    ].resolve():
        raise ValueError("runtime checkpoint differs from preregistered checkpoint")
    with paths["run_config"].open() as handle:
        run_config = json.load(handle)
    if (
        int(run_config.get("reverse_history", -1)) != 1
        or int(run_config.get("max_his", -1)) != config["max_history"]
        or int(run_config.get("skip_empty_his", -1)) != 1
        or int(run_config.get("use_position_embedding", -1)) != 1
    ):
        raise ValueError("run configuration differs from locked PENS assumptions")

    model, tokenizer, runtime = create_model_and_tokenizer(dataset, device)
    sequences = read_sequences(paths["user_sequence"])
    position_rows, structural = position_census(
        model, sequences, config["max_history"]
    )
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
    _, text_to_item = decode_item_ids(paths["item_index"], "t5-small", True)
    prediction_rows, prediction_audit = read_predictions(
        paths["predictions"], text_to_item
    )
    cohort, available = build_cohort(
        dataset,
        sequences,
        prediction_rows,
        item2input,
        item2lexid,
        config["seed"],
        config["max_per_stratum"],
        config["min_history"],
        config["max_history"],
    )
    current_identity = sorted(
        (row["user"], row["stratum"], row["target"], row["selection_hash"])
        for row in cohort
    )
    cohort_match = current_identity == read_locked_cohort(paths["cgi_cohort"])
    if not cohort_match:
        raise ValueError(f"{dataset} cohort differs from locked CGI E0 cohort")
    scored = [
        row for row in cohort if row["stratum"] in set(config["scored_strata"])
    ]
    counts = Counter(row["stratum"] for row in scored)
    samples = make_samples(scored, item2input, item2lexid)
    collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
    score_rows, intervention = score_samples(
        model,
        tokenizer,
        collator,
        samples,
        config["batch_size"],
        device,
    )
    numeric_fields = (
        "lp_current",
        "lp_equal_fine_norm",
        "lp_zero_all_position",
        "norm_only_gain",
        "zero_position_delta",
    )
    finite_rate = float(
        np.mean(
            [
                math.isfinite(float(row[field]))
                for row in score_rows
                for field in numeric_fields
            ]
        )
    )
    stats = summarize(
        score_rows, config["bootstrap_iterations"], config["seed"]
    )
    gates = evaluate_gates(
        dict(counts), cohort_match, stats, structural, intervention, config
    )
    if finite_rate != config["integrity_gates"]["finite_metric_rate"]:
        gates["integrity"] = False

    output_dir = output_root / dataset
    write_csv(
        output_dir / "cohort.csv",
        scored,
        [
            "user",
            "stratum",
            "target",
            "history_length",
            "full_hit50",
            "target_tail",
            "selection_hash",
            "newest_item",
            "oldest_item",
        ],
    )
    write_csv(
        output_dir / "position_census.csv",
        position_rows,
        ["position", "training_prefix_exposure", "l2_norm"],
    )
    write_csv(
        output_dir / "counterfactual_scores.csv",
        score_rows,
        list(score_rows[0]),
    )
    result = {
        "dataset": dataset,
        "counts": dict(counts),
        "available_counts": available,
        "structural": structural,
        "intervention_audit": intervention,
        "statistics": stats,
        "gates": gates,
        "integrity": {
            "cohort_matches_cgi_e0": cohort_match,
            "finite_metric_rate": finite_rate,
            "prediction_audit": prediction_audit,
            "training_updates": False,
            "beam_generation": False,
            "sequence_final_two_excluded_from_exposure": True,
            "test_data_used": False,
        },
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    del model
    torch.cuda.empty_cache()
    return result


def decide(results: dict) -> str:
    for gate, failure in (
        ("integrity", "EXECUTION_INVALID"),
        ("structural_replication", "STOP_PENS_NO_STRUCTURAL_REPLICATION"),
        ("causal_benefit", "STOP_PENS_NO_CAUSAL_BENEFIT"),
        ("no_broad_harm", "STOP_PENS_BROAD_HARM"),
    ):
        if not all(result["gates"][gate] for result in results.values()):
            return failure
    return "H1_DESIGN_ALLOWED"


def write_report(path: Path, aggregate: dict) -> None:
    lines = [
        "# GRAM 第三阶段 PENS H0-D 诊断报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Verification Status: ANALYZED",
        f"- Version Label: `{aggregate['material_passport']['version_label']}`",
        "",
        f"- 决策：**`{aggregate['decision']}`**",
        "- 边界：冻结 validation checkpoint/cohort；未训练、未生成 beam、未读 test。",
        "",
        "## 双数据集 gate",
        "",
        "| Dataset | Integrity | Structural | Causal benefit | No broad harm |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, result in aggregate["datasets"].items():
        gate = result["gates"]
        lines.append(
            f"| {dataset} | {gate['integrity']} | "
            f"{gate['structural_replication']} | {gate['causal_benefit']} | "
            f"{gate['no_broad_harm']} |"
        )
    lines.extend(["", "## 锁定统计", ""])
    for dataset, result in aggregate["datasets"].items():
        structural = result["structural"]
        miss = result["statistics"]["tail_miss"]["norm_only_gain"]
        hit = result["statistics"]["tail_hit"]["norm_only_gain"]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- exposure–norm Pearson={structural['exposure_norm_pearson']:.6f}; "
                f"`||P20||/||P1||`={structural['position20_to_position1_norm_ratio']:.6f}",
                f"- tail-miss norm-only gain: mean={miss['mean']:.6f}, "
                f"95% CI=[{miss['ci95'][0]:.6f}, {miss['ci95'][1]:.6f}], "
                f"P(>0)={miss['positive_rate']:.6f}",
                f"- tail-hit norm-only gain mean={hit['mean']:.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释边界",
            "",
            "zero-position 与 history-length 分层仅为描述性结果。只有双数据集结构复制、",
            "tail-miss 因果收益和 tail-hit 无广泛伤害全部通过才解锁 H1。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open() as handle:
        config = json.load(handle)
    if not config.get("preregistered_before_new_counterfactual_scores"):
        raise ValueError("H0-D lacks a frozen preregistration boundary")
    if not torch.cuda.is_available():
        raise RuntimeError("H0-D requires CUDA for frozen checkpoint scoring")
    started = time.time()
    device = torch.device("cuda:0")
    results = {}
    for dataset, spec in config["datasets"].items():
        results[dataset] = run_dataset(
            dataset, spec, config, args.output_root, device
        )
    aggregate = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-24",
            "verification_status": "ANALYZED",
            "version_label": "pens_h0_d_v1",
        },
        "novelty_decision": "NOVELTY_SCOPE_PASS_WITH_STRONG_MECHANISTIC_NARROWING",
        "decision": decide(results),
        "datasets": results,
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
        "wall_time_seconds": time.time() - started,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n"
    )
    write_report(args.report, aggregate)
    print(
        json.dumps(
            {
                "decision": aggregate["decision"],
                "wall_time_seconds": aggregate["wall_time_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
