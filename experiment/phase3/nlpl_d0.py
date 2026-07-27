#!/usr/bin/env python3
"""NLPL D0-D: CPU-only frozen-T5 native-prior exposure diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "artifacts/phase3/nlpl_d0"
REPORT_PATH = ROOT / "report/第三阶段/GRAM_第三阶段_NLPL_D0诊断报告.md"
DELIMITER_IDS = {1820, 9175}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_sequences(path: Path) -> dict[str, list[str]]:
    result = {}
    with path.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if fields:
                result[fields[0]] = fields[1:]
    return result


def training_popularity(sequences: dict[str, list[str]]) -> Counter:
    return Counter(item for sequence in sequences.values() for item in sequence[:-2])


def read_item_paths(path: Path, tokenizer) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    item_paths = {}
    text_to_item = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            item, raw = line.rstrip("\n").split(" ", 1)
            ids = tuple(
                token_id
                for token_id in tokenizer.encode(raw)
                if token_id not in DELIMITER_IDS and token_id != tokenizer.eos_token_id
            )
            expected_length = len([part for part in raw.split("|") if part])
            if len(ids) != expected_length:
                raise ValueError(
                    f"{path}:{line_number}: lexical segment/token mismatch "
                    f"{expected_length} != {len(ids)}"
                )
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            if decoded in text_to_item:
                raise ValueError(f"Non-unique decoded lexical ID: {decoded}")
            item_paths[item] = ids
            text_to_item[decoded] = item
    return item_paths, text_to_item


def read_predictions(path: Path, text_to_item: dict[str, str]) -> tuple[list[dict], dict]:
    rows = []
    footer = {}
    with path.open() as handle:
        header = next(handle, "")
        if not header.startswith("idx\t"):
            raise ValueError(f"Unexpected prediction header: {path}")
        for line_number, line in enumerate(handle, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 1 and ": " in fields[0]:
                key, value = fields[0].split(": ", 1)
                footer[key] = float(value)
                continue
            if len(fields) < 6:
                raise ValueError(f"Malformed prediction row {line_number}")
            user = fields[0]
            gold_text, pred_text, score_text = fields[-3:]
            if gold_text not in text_to_item:
                raise ValueError(f"Unknown gold lexical ID at row {line_number}")
            pred_strings = pred_text.split("||")
            scores = score_text.split("||")
            if len(pred_strings) != 50 or len(scores) != 50:
                raise ValueError(f"Expected exactly 50 candidates at row {line_number}")
            try:
                pred_items = [text_to_item[value] for value in pred_strings]
            except KeyError as error:
                raise ValueError(f"Unknown prediction at row {line_number}: {error}") from error
            if len(set(pred_items)) != 50:
                raise ValueError(f"Duplicate candidate item at row {line_number}")
            rows.append({"user": user, "gold": text_to_item[gold_text], "pred": pred_items})
    return rows, footer


def score_native_priors(
    item_paths: dict[str, tuple[int, ...]],
    model,
    tokenizer,
    batch_size: int,
) -> dict[str, dict]:
    items = sorted(item_paths)
    result = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            lengths = [len(item_paths[item]) for item in batch_items]
            max_length = max(lengths)
            labels = torch.full(
                (len(batch_items), max_length), -100, dtype=torch.long
            )
            for row, item in enumerate(batch_items):
                path = item_paths[item]
                labels[row, : len(path)] = torch.tensor(path)
            input_ids = torch.tensor(
                [[tokenizer.pad_token_id, tokenizer.eos_token_id]] * len(batch_items)
            )
            attention_mask = torch.ones_like(input_ids)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            safe_labels = labels.clamp_min(0)
            token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            for row, item in enumerate(batch_items):
                values = token_log_probs[row, : lengths[row]].cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite native prior for {item}")
                result[item] = {
                    "lp0": float(values.mean()),
                    "lp_last": float(values[-1]),
                }
    return result


def recall(rows: list[dict], k: int) -> float:
    return sum(row["gold"] in row["pred"][:k] for row in rows) / len(rows)


def build_pairs(
    item_paths: dict[str, tuple[int, ...]],
    popularity: Counter,
    priors: dict[str, dict],
    amplification: dict[str, float],
    max_ratio: float,
) -> list[dict]:
    groups = defaultdict(list)
    for item, path in item_paths.items():
        groups[path[:-1]].append(item)
    pairs = []
    for parent, items in sorted(groups.items()):
        for left, right in itertools.combinations(sorted(items), 2):
            left_freq, right_freq = popularity[left], popularity[right]
            if min(left_freq, right_freq) <= 0:
                continue
            if max(left_freq, right_freq) / min(left_freq, right_freq) > max_ratio:
                continue
            delta_lp = priors[right]["lp_last"] - priors[left]["lp_last"]
            delta_amp = amplification[right] - amplification[left]
            non_tie = delta_lp != 0.0 and delta_amp != 0.0
            pairs.append(
                {
                    "parent": " ".join(map(str, parent)),
                    "left_item": left,
                    "right_item": right,
                    "left_train_freq": left_freq,
                    "right_train_freq": right_freq,
                    "delta_lp_last": delta_lp,
                    "delta_amplification": delta_amp,
                    "non_tie": int(non_tie),
                    "concordant": int(non_tie and delta_lp * delta_amp > 0),
                }
            )
    return pairs


def clustered_inference(
    pairs: list[dict], seed: int, bootstrap_n: int, permutation_n: int
) -> dict:
    eligible = [row for row in pairs if row["non_tie"]]
    clusters = defaultdict(list)
    for row in eligible:
        clusters[row["parent"]].append(row["concordant"])
    if not eligible or not clusters:
        return {
            "eligible_non_tie_pairs": len(eligible),
            "parent_clusters": len(clusters),
            "concordance": None,
            "bootstrap_ci95": [None, None],
            "permutation_p": None,
        }
    names = sorted(clusters)
    observed = sum(row["concordant"] for row in eligible) / len(eligible)
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_n)
    for index in range(bootstrap_n):
        sampled = rng.integers(0, len(names), len(names))
        values = [value for cluster_index in sampled for value in clusters[names[cluster_index]]]
        boot[index] = sum(values) / len(values)
    cluster_success = np.array([sum(clusters[name]) for name in names], dtype=float)
    cluster_total = np.array([len(clusters[name]) for name in names], dtype=float)
    greater_equal = 0
    for _ in range(permutation_n):
        signs = rng.integers(0, 2, len(names))
        success = np.where(signs == 1, cluster_success, cluster_total - cluster_success).sum()
        greater_equal += success / cluster_total.sum() >= observed
    return {
        "eligible_non_tie_pairs": len(eligible),
        "parent_clusters": len(clusters),
        "concordance": observed,
        "bootstrap_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "permutation_p": (greater_equal + 1) / (permutation_n + 1),
    }


def tail_miss_analysis(
    rows: list[dict],
    item_paths: dict[str, tuple[int, ...]],
    popularity: Counter,
    priors: dict[str, dict],
) -> dict:
    ordered = sorted(item_paths, key=lambda item: (-popularity[item], item))
    head_n = max(1, math.ceil(len(ordered) * 0.20))
    tail = set(ordered[head_n:])
    parent_values = defaultdict(list)
    for item, path in item_paths.items():
        parent_values[path[:-1]].append(priors[item]["lp_last"])
    parent_mean = {parent: float(np.mean(values)) for parent, values in parent_values.items()}
    events = []
    for row in rows:
        item = row["gold"]
        if item in tail:
            residual = priors[item]["lp_last"] - parent_mean[item_paths[item][:-1]]
            events.append((residual, int(item not in row["pred"][:50])))
    median = float(np.median([event[0] for event in events]))
    cells = {"low_miss": 0, "low_hit": 0, "high_miss": 0, "high_hit": 0}
    for residual, miss in events:
        group = "low" if residual < median else "high"
        cells[f"{group}_{'miss' if miss else 'hit'}"] += 1
    odds_low = (cells["low_miss"] + 0.5) / (cells["low_hit"] + 0.5)
    odds_high = (cells["high_miss"] + 0.5) / (cells["high_hit"] + 0.5)
    return {
        "tail_target_events": len(events),
        "median_parent_centered_lp_last": median,
        **cells,
        "miss_odds_ratio_low_vs_high": odds_low / odds_high,
    }


def analyze_dataset(
    dataset: str, spec: dict, config: dict, tokenizer, model
) -> dict:
    started = time.time()
    paths = {key: ROOT / value for key, value in spec.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    item_paths, text_to_item = read_item_paths(paths["index"], tokenizer)
    rows, footer = read_predictions(paths["predictions"], text_to_item)
    sequences = read_sequences(paths["sequences"])
    popularity = training_popularity(sequences)
    for row in rows:
        if row["user"] not in sequences:
            raise ValueError(f"Prediction user absent from sequences: {row['user']}")
        if sequences[row["user"]][-2] != row["gold"]:
            raise ValueError(f"Validation target mismatch: {row['user']}")
    perturbed = {
        user: sequence[:-2] + ["__changed_validation__", "__changed_test__"]
        for user, sequence in sequences.items()
    }
    if training_popularity(perturbed) != popularity:
        raise AssertionError("Training popularity changed after perturbing held-out targets")
    s0 = load_json(paths["s0_summary"])
    recall10, recall50 = recall(rows, 10), recall(rows, 50)
    if abs(recall10 - s0["baseline"]["recall@10"]) > 1e-12:
        raise ValueError("Recall@10 reproduction failed")
    if abs(recall50 - s0["baseline"]["recall@50"]) > 1e-12:
        raise ValueError("Recall@50 reproduction failed")
    priors = score_native_priors(
        item_paths, model, tokenizer, int(config["batch_size"])
    )
    beam_count = Counter(item for row in rows for item in row["pred"])
    smooth = float(config["smoothing"])
    catalog_n = len(item_paths)
    total_train = sum(popularity.values())
    denominator_beam = 50 * len(rows) + smooth * catalog_n
    denominator_train = total_train + smooth * catalog_n
    amplification = {
        item: math.log((beam_count[item] + smooth) / denominator_beam)
        - math.log((popularity[item] + smooth) / denominator_train)
        for item in item_paths
    }
    parent_mean = defaultdict(list)
    for item, path in item_paths.items():
        parent_mean[path[:-1]].append(priors[item]["lp_last"])
    parent_mean = {key: float(np.mean(value)) for key, value in parent_mean.items()}
    item_rows = []
    for item in sorted(item_paths):
        item_rows.append(
            {
                "item": item,
                "path": " ".join(map(str, item_paths[item])),
                "parent": " ".join(map(str, item_paths[item][:-1])),
                "train_freq": popularity[item],
                "beam_count": beam_count[item],
                "lp0": priors[item]["lp0"],
                "lp_last": priors[item]["lp_last"],
                "lp_last_parent_centered": priors[item]["lp_last"]
                - parent_mean[item_paths[item][:-1]],
                "exposure_amplification": amplification[item],
            }
        )
    pairs = build_pairs(
        item_paths,
        popularity,
        priors,
        amplification,
        float(config["max_training_frequency_ratio"]),
    )
    inference = clustered_inference(
        pairs,
        int(config["seed"]),
        int(config["bootstrap_iterations"]),
        int(config["permutation_iterations"]),
    )
    tail = tail_miss_analysis(rows, item_paths, popularity, priors)
    gates = {
        "support": bool(
            inference["eligible_non_tie_pairs"] >= config["minimum_pairs"]
        ),
        "concordance": bool(
            inference["concordance"] is not None
            and inference["concordance"] >= config["minimum_concordance"]
        ),
        "uncertainty": bool(
            inference["bootstrap_ci95"][0] is not None
            and inference["bootstrap_ci95"][0] > config["minimum_ci_lower"]
        ),
        "randomization": bool(
            inference["permutation_p"] is not None
            and inference["permutation_p"] <= config["maximum_permutation_p"]
        ),
        "recommendation_link": bool(
            tail["miss_odds_ratio_low_vs_high"]
            >= config["minimum_tail_miss_odds_ratio"]
        ),
    }
    output = OUTPUT_ROOT / dataset
    write_csv(output / "item_native_prior.csv", item_rows, list(item_rows[0]))
    write_csv(output / "matched_pairs.csv", pairs, list(pairs[0]))
    summary = {
        "dataset": dataset,
        "integrity": {
            "rows": len(rows),
            "items": len(item_paths),
            "candidates_per_row": 50,
            "unknown_candidates": 0,
            "duplicate_candidate_rows": 0,
            "target_mismatches": 0,
            "training_only_perturbation_passed": true_value(),
            "recall@10": recall10,
            "recall@50": recall50,
            "s0_recall_absolute_error@10": abs(recall10 - s0["baseline"]["recall@10"]),
            "s0_recall_absolute_error@50": abs(recall50 - s0["baseline"]["recall@50"]),
            "prediction_footer": footer,
            "input_sha256": {key: sha256(path) for key, path in paths.items()},
        },
        "matched_sibling": {
            "all_frequency_matched_pairs": len(pairs),
            **inference,
        },
        "tail_miss": tail,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "wall_time_seconds": time.time() - started,
    }
    write_json(output / "diagnostic_summary.json", summary)
    return summary


def true_value() -> bool:
    """Named helper keeps the lineage assertion explicit and testable."""
    return True


def render_report(summary: dict) -> None:
    lines = [
        "# GRAM 第三阶段：NLPL D0-D 诊断报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-07-24",
        "- Verification Status: ANALYZED",
        "- Version Label: `nlpl_d0_diagnostic_v1`",
        "- Runtime: CPU only; no GRAM checkpoint; no test data",
        "",
        "## 决策",
        "",
        f"固定决策为 **`{summary['decision']}`**。",
        "",
        "## 预注册 gate",
        "",
        "| 数据集 | non-tie pairs | concordance | bootstrap 95% CI | permutation p | tail miss OR | 全部通过 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, value in summary["datasets"].items():
        matched, tail = value["matched_sibling"], value["tail_miss"]
        lines.append(
            f"| {dataset} | {matched['eligible_non_tie_pairs']} | "
            f"{matched['concordance']:.6f} | "
            f"[{matched['bootstrap_ci95'][0]:.6f}, {matched['bootstrap_ci95'][1]:.6f}] | "
            f"{matched['permutation_p']:.6g} | "
            f"{tail['miss_odds_ratio_low_vs_high']:.6f} | "
            f"{value['all_gates_passed']} |"
        )
    lines += [
        "",
        "全部门槛是双数据集必要条件，任一失败不能由其他门槛抵消。完整逐项结果见",
        "`artifacts/phase3/nlpl_d0/summary.json`。",
        "",
        "## 完整性",
        "",
        "- 两数据集 Recall@10/50 均按冻结 prediction 精确复算；",
        "- 每行 50 个候选均可映射且无重复；",
        "- 修改 `sequence[-2:]` 不改变 training-only frequency；",
        "- native prior 全部来自本地冻结原始 T5-small；",
        "- 未加载 GRAM checkpoint、未读取 test、未训练、未使用 GPU。",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_json(config_path)
    model_path = ROOT / config["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    model.to("cpu")
    model_hashes = {
        name: sha256(model_path / name)
        for name in (
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    datasets = {
        dataset: analyze_dataset(dataset, spec, config, tokenizer, model)
        for dataset, spec in config["datasets"].items()
    }
    all_passed = all(value["all_gates_passed"] for value in datasets.values())
    if all_passed:
        decision = "D0_MECHANISM_ALLOWED"
    else:
        exposure_keys = ("support", "concordance", "uncertainty", "randomization")
        exposure_pass = all(
            all(value["gates"][key] for key in exposure_keys)
            for value in datasets.values()
        )
        decision = (
            "STOP_NLPL_NO_TAIL_LINK" if exposure_pass else "STOP_NLPL_NO_EXPOSURE"
        )
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-24",
            "verification_status": "ANALYZED",
            "version_label": "nlpl_d0_diagnostic_v1",
        },
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(Path(__file__)),
        "model_sha256": model_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_visible_devices": "",
            "torch_cuda_available": torch.cuda.is_available(),
        },
        "datasets": datasets,
        "decision": decision,
        "d1_unlocked": decision == "D0_MECHANISM_ALLOWED",
    }
    write_json(OUTPUT_ROOT / "summary.json", summary)
    render_report(summary)
    print(json.dumps({"decision": decision, "datasets": datasets}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
