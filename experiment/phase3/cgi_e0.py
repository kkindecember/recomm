#!/usr/bin/env python3
"""CGI E0-D frozen-checkpoint counterfactual granularity diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from hbtr_b1_smoke import (  # noqa: E402
    DATASETS,
    create_model_and_tokenizer,
    read_sequences,
    sha256,
)
from s0_offline_diagnostics import (  # noqa: E402
    decode_item_ids,
    head_items,
    read_predictions,
    training_popularity,
)

CONDITIONS = ("full", "coarse_only", "minus_oldest", "minus_newest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/phase3/cgi_e0")
    parser.add_argument("--report", type=Path, default=ROOT / "report/第三阶段/GRAM_第三阶段_CGI_E0诊断报告.md")
    return parser.parse_args()


def selection_hash(seed: int, dataset: str, stratum: str, user: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|{stratum}|{user}".encode()).hexdigest()


def lexical_mean_logprob(logits, labels, eos_token_id: int) -> torch.Tensor:
    valid = (labels != -100) & (labels != eos_token_id)
    safe_labels = labels.masked_fill(~valid, 0)
    selected = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("target contains no lexical token after EOS/pad exclusion")
    return (selected * valid).sum(dim=1) / counts


def condition_mask(base: torch.Tensor, history_lengths: list[int], condition: str) -> torch.Tensor:
    result = base.clone()
    if condition == "full":
        return result
    if condition == "coarse_only":
        result[:, 1:, :] = False
    elif condition == "minus_newest":
        result[:, 1, :] = False
    elif condition == "minus_oldest":
        for row, length in enumerate(history_lengths):
            result[row, length, :] = False
    else:
        raise ValueError(condition)
    if torch.any(result[:, 0, :].sum(dim=1) == 0):
        raise ValueError("coarse passage was accidentally masked")
    return result


def build_cohort(
    dataset: str,
    sequences: dict[str, list[str]],
    prediction_rows: list[dict],
    item2input: dict[str, str],
    item2lexid: dict[str, str],
    seed: int,
    max_per_stratum: int,
    min_history: int,
    max_history: int,
) -> tuple[list[dict], dict]:
    popularity = training_popularity(sequences)
    heads = head_items(popularity)
    candidates = defaultdict(list)
    lineage_errors = []
    for row in prediction_rows:
        items = sequences.get(row["user"])
        if items is None or len(items) < 3:
            lineage_errors.append((row["user"], "missing_or_short"))
            continue
        history = items[:-2][-max_history:]
        target = items[-2]
        if row["gold"] != target:
            lineage_errors.append((row["user"], "target_mismatch"))
            continue
        if len(history) < min_history or target not in item2lexid:
            continue
        if any(item not in item2input or item not in item2lexid for item in history):
            continue
        hit = target in row["pred_items"][:50]
        pop = "head" if target in heads else "tail"
        stratum = f"{pop}_{'hit' if hit else 'miss'}"
        digest = selection_hash(seed, dataset, stratum, row["user"])
        candidates[stratum].append(
            {
                "user": row["user"],
                "stratum": stratum,
                "target": target,
                "history": history,
                "history_length": len(history),
                "full_hit50": int(hit),
                "target_tail": int(pop == "tail"),
                "selection_hash": digest,
                "newest_item": history[-1],
                "oldest_item": history[0],
            }
        )
    if lineage_errors:
        raise ValueError(f"prediction lineage errors: {lineage_errors[:3]}")
    selected = []
    available = {}
    for stratum in ("tail_miss", "tail_hit", "head_miss", "head_hit"):
        values = sorted(candidates[stratum], key=lambda row: row["selection_hash"])
        available[stratum] = len(values)
        selected.extend(values[:max_per_stratum])
    selected.sort(key=lambda row: (row["stratum"], row["selection_hash"]))
    return selected, available


def make_samples(cohort: list[dict], item2input: dict[str, str], item2lexid: dict[str, str]) -> list[dict]:
    result = []
    for row in cohort:
        newest_to_oldest = list(reversed(row["history"]))
        if newest_to_oldest[0] != row["newest_item"] or newest_to_oldest[-1] != row["oldest_item"]:
            raise ValueError(f"fine passage identity audit failed for {row['user']}")
        history_lex = " ; ".join(item2lexid[item] for item in newest_to_oldest)
        result.append(
            {
                **row,
                "input": [f"What would user purchase after {history_lex} ?"]
                + [item2input[item] for item in newest_to_oldest],
                "output": item2lexid[row["target"]],
            }
        )
    return result


@torch.no_grad()
def score_samples(model, tokenizer, collator, samples: list[dict], batch_size: int, device) -> tuple[list[dict], float]:
    model.eval()
    rows = []
    repeat_max = 0.0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        batch = collator(
            [{"input": row["input"], "output": row["output"], "user_id": row["user"]} for row in chunk]
        )
        ids = batch["item_text_ids"].to(device)
        base_mask = batch["item_text_masks"].to(device)
        labels = batch["target_ids"].to(device)
        lengths = [row["history_length"] for row in chunk]
        condition_scores = {}
        for condition in CONDITIONS:
            mask = condition_mask(base_mask, lengths, condition)
            output = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
            condition_scores[condition] = lexical_mean_logprob(
                output.logits, labels, tokenizer.eos_token_id
            ).cpu().numpy()
        repeat_output = model(
            input_ids=ids,
            attention_mask=condition_mask(base_mask, lengths, "full"),
            labels=labels,
            return_dict=True,
        )
        repeated = lexical_mean_logprob(
            repeat_output.logits, labels, tokenizer.eos_token_id
        ).cpu().numpy()
        repeat_max = max(repeat_max, float(np.max(np.abs(repeated - condition_scores["full"]))))
        for index, sample in enumerate(chunk):
            full = float(condition_scores["full"][index])
            coarse = float(condition_scores["coarse_only"][index])
            old = float(condition_scores["minus_oldest"][index])
            new = float(condition_scores["minus_newest"][index])
            rows.append(
                {
                    "user": sample["user"],
                    "stratum": sample["stratum"],
                    "history_length": sample["history_length"],
                    "lp_full": full,
                    "lp_coarse_only": coarse,
                    "lp_minus_oldest": old,
                    "lp_minus_newest": new,
                    "g_all": coarse - full,
                    "g_old": old - full,
                    "g_new": new - full,
                }
            )
    return rows, repeat_max


def bootstrap_mean(values: np.ndarray, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=np.float64)
    for begin in range(0, iterations, 1000):
        count = min(1000, iterations - begin)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        boot[begin : begin + count] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive_rate": float((values > 0).mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def bootstrap_difference(a: np.ndarray, b: np.ndarray, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        boot[index] = (
            a[rng.integers(0, len(a), len(a))].mean()
            - b[rng.integers(0, len(b), len(b))].mean()
        )
    return {
        "mean": float(a.mean() - b.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def summarize_scores(rows: list[dict], iterations: int, seed: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["stratum"]].append(row)
    summary = {}
    for stratum, values in sorted(grouped.items()):
        summary[stratum] = {}
        for offset, metric in enumerate(("g_all", "g_old", "g_new")):
            array = np.asarray([row[metric] for row in values], dtype=np.float64)
            summary[stratum][metric] = bootstrap_mean(array, iterations, seed + offset)
        temporal = np.asarray([row["g_old"] - row["g_new"] for row in values])
        summary[stratum]["g_old_minus_g_new"] = bootstrap_mean(temporal, iterations, seed + 3)
    miss = np.asarray([row["g_all"] for row in grouped["tail_miss"]])
    hit = np.asarray([row["g_all"] for row in grouped["tail_hit"]])
    summary["tail_miss_minus_tail_hit_g_all"] = bootstrap_difference(
        miss, hit, iterations, seed + 4
    )
    return summary


def evaluate_gates(summary: dict, counts: dict, repeat_max: float, gates: dict) -> dict:
    miss = summary["tail_miss"]
    integrity = (
        counts.get("tail_miss", 0) == gates["tail_miss_n"]
        and counts.get("tail_hit", 0) == gates["tail_hit_n"]
        and repeat_max <= gates["repeat_max_abs_error"]
    )
    cumulative = (
        miss["g_all"]["mean"] >= gates["cumulative_mean"]
        and miss["g_all"]["ci95"][0] > 0
        and miss["g_all"]["positive_rate"] >= gates["cumulative_positive_rate"]
    )
    old = (
        miss["g_old"]["mean"] >= gates["old_mean"]
        and miss["g_old"]["ci95"][0] > 0
    )
    temporal = (
        miss["g_old_minus_g_new"]["mean"] >= gates["temporal_difference_mean"]
        and miss["g_old_minus_g_new"]["ci95"][0] > 0
    )
    association = summary["tail_miss_minus_tail_hit_g_all"]
    failure = (
        association["mean"] >= gates["failure_association_mean"]
        and association["ci95"][0] > 0
    )
    return {
        "cohort_integrity": integrity,
        "cumulative_interference": cumulative,
        "old_passage_interference": old,
        "temporal_specificity": temporal,
        "failure_association": failure,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(dataset: str, spec: dict, config: dict, output_root: Path, device) -> dict:
    dataset_dir = ROOT / "GRAM/rec_datasets" / dataset
    paths = {
        "checkpoint": ROOT / spec["checkpoint"],
        "run_config": ROOT / spec["run_config"],
        "predictions": ROOT / spec["predictions"],
        "s0_summary": ROOT / spec["s0_summary"],
        "user_sequence": dataset_dir / "user_sequence.txt",
        "item_index": dataset_dir / f"item_generative_indexing_{DATASETS[dataset]['hierarchical_id_type']}.txt",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    with paths["run_config"].open() as handle:
        run_config = json.load(handle)
    if int(run_config.get("reverse_history", -1)) != 1:
        raise ValueError(f"{dataset} reverse_history is not locked to 1")

    model, tokenizer, runtime = create_model_and_tokenizer(dataset, device)
    sequences = read_sequences(paths["user_sequence"])
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
    prediction_rows, prediction_audit = read_predictions(paths["predictions"], text_to_item)
    cohort, available = build_cohort(
        dataset, sequences, prediction_rows, item2input, item2lexid,
        config["seed"], config["max_per_stratum"], config["min_history"], config["max_history"],
    )
    counts = Counter(row["stratum"] for row in cohort)
    if counts["tail_miss"] < config["gates"]["tail_miss_n"] or counts["tail_hit"] < config["gates"]["tail_hit_n"]:
        raise ValueError(f"{dataset} insufficient locked tail cohort: {dict(counts)}")
    samples = make_samples(cohort, item2input, item2lexid)
    collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
    score_rows, repeat_max = score_samples(
        model, tokenizer, collator, samples, config["batch_size"], device
    )
    if not all(math.isfinite(float(row[key])) for row in score_rows for key in (
        "lp_full", "lp_coarse_only", "lp_minus_oldest", "lp_minus_newest", "g_all", "g_old", "g_new"
    )):
        raise ValueError("NaN/Inf in counterfactual scores")
    stats = summarize_scores(score_rows, config["bootstrap_iterations"], config["seed"])
    gates = evaluate_gates(stats, dict(counts), repeat_max, config["gates"])
    output_dir = output_root / dataset
    write_csv(
        output_dir / "cohort.csv", cohort,
        ["user", "stratum", "target", "history_length", "full_hit50", "target_tail",
         "selection_hash", "newest_item", "oldest_item"],
    )
    write_csv(
        output_dir / "counterfactual_scores.csv", score_rows,
        ["user", "stratum", "history_length", "lp_full", "lp_coarse_only",
         "lp_minus_oldest", "lp_minus_newest", "g_all", "g_old", "g_new"],
    )
    result = {
        "dataset": dataset,
        "counts": dict(counts),
        "available_counts": available,
        "statistics": stats,
        "gates": gates,
        "integrity": {
            "full_repeat_max_abs_error": repeat_max,
            "finite": True,
            "reverse_history": 1,
            "order_identity_audit": True,
            "sequence_last_item_read": False,
            "test_data_read": False,
            "prediction_audit": prediction_audit,
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
    if not all(value["gates"]["cohort_integrity"] for value in results.values()):
        return "EXECUTION_INVALID"
    interference = ("cumulative_interference", "old_passage_interference", "temporal_specificity")
    if not all(value["gates"][gate] for value in results.values() for gate in interference):
        return "STOP_CGI_NO_INTERFERENCE"
    if not all(value["gates"]["failure_association"] for value in results.values()):
        return "STOP_CGI_NO_FAILURE_LINK"
    return "E0_MECHANISM_ALLOWED"


def write_report(path: Path, aggregate: dict) -> None:
    lines = [
        "# GRAM 第三阶段 CGI E0 诊断报告", "",
        f"- 决策：**`{aggregate['decision']}`**",
        "- 数据边界：validation history/target 与冻结 validation beam-50；未读 test，未训练。",
        "- 分数：gold lexical target token 的 mean log-prob；EOS/pad 排除。", "",
        "## 双数据集 gate", "",
        "| Dataset | Integrity | Cumulative | Old | Temporal | Failure association |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in aggregate["datasets"].items():
        g = result["gates"]
        lines.append(
            f"| {dataset} | {g['cohort_integrity']} | {g['cumulative_interference']} | "
            f"{g['old_passage_interference']} | {g['temporal_specificity']} | {g['failure_association']} |"
        )
    lines.extend(["", "## 锁定主统计", ""])
    for dataset, result in aggregate["datasets"].items():
        miss = result["statistics"]["tail_miss"]
        assoc = result["statistics"]["tail_miss_minus_tail_hit_g_all"]
        lines.extend([
            f"### {dataset}", "",
            f"- tail_miss `G_all`: mean={miss['g_all']['mean']:.6f}, "
            f"95% CI=[{miss['g_all']['ci95'][0]:.6f}, {miss['g_all']['ci95'][1]:.6f}], "
            f"P(>0)={miss['g_all']['positive_rate']:.6f}",
            f"- tail_miss `G_old`: mean={miss['g_old']['mean']:.6f}, "
            f"95% CI=[{miss['g_old']['ci95'][0]:.6f}, {miss['g_old']['ci95'][1]:.6f}]",
            f"- tail_miss `G_old-G_new`: mean={miss['g_old_minus_g_new']['mean']:.6f}, "
            f"95% CI=[{miss['g_old_minus_g_new']['ci95'][0]:.6f}, "
            f"{miss['g_old_minus_g_new']['ci95'][1]:.6f}]",
            f"- tail miss-hit `G_all`: mean={assoc['mean']:.6f}, "
            f"95% CI=[{assoc['ci95'][0]:.6f}, {assoc['ci95'][1]:.6f}]",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open() as handle:
        config = json.load(handle)
    if not torch.cuda.is_available():
        raise RuntimeError("CGI E0-D requires CUDA")
    started = time.time()
    results = {}
    device = torch.device("cuda:0")
    for dataset, spec in config["datasets"].items():
        results[dataset] = run_dataset(dataset, spec, config, args.output_root, device)
    aggregate = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-24",
            "verification_status": "ANALYZED",
            "version_label": "cgi_e0_d_v1",
        },
        "novelty_decision": "NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_NARROWING",
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
    print(json.dumps({"decision": aggregate["decision"], "wall_time_seconds": aggregate["wall_time_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
