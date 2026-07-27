#!/usr/bin/env python3
"""LEI F0-D position-preserving span-factorized frozen diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
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
    bootstrap_difference,
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
from s0_offline_diagnostics import decode_item_ids, read_predictions  # noqa: E402

DELIMITER_TOKEN_IDS = {1820, 9175}
PASSAGE_RE = re.compile(
    r"^item: (?P<link>.*?); similar items: (?P<cf>.*?); (?P<meta>.*)$",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artifacts/phase3/lei_f0"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/第三阶段/GRAM_第三阶段_LEI_F0诊断报告.md",
    )
    return parser.parse_args()


def parse_role_char_spans(text: str) -> dict[str, list[tuple[int, int]]]:
    match = PASSAGE_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"fine passage does not match locked grammar: {text[:120]!r}")
    link = [match.span("link")]
    cf_start, _ = match.span("cf")
    cf_value = match.group("cf")
    cf_spans = []
    cursor = 0
    for value in cf_value.split(", "):
        if not value:
            raise ValueError("empty collaborative lexical ID")
        begin = cf_value.find(value, cursor)
        if begin < cursor:
            raise ValueError("failed to locate collaborative lexical ID")
        cf_spans.append((cf_start + begin, cf_start + begin + len(value)))
        cursor = begin + len(value)
    return {"link": link, "cf": cf_spans, "metadata": [match.span("meta")]}


def overlaps(offset: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = offset
    return end > start and any(start < span_end and end > span_start for span_start, span_end in spans)


def filtered_encoding_with_offsets(tokenizer, text: str, max_len: int) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("LEI role localization requires a fast tokenizer with offsets")
    encoded = tokenizer(
        text,
        max_length=999,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    kept = [
        (token_id, attention, tuple(offset))
        for token_id, attention, offset in zip(
            encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"]
        )
        if token_id not in DELIMITER_TOKEN_IDS
    ][:max_len]
    ids = [value[0] for value in kept]
    attention = [value[1] for value in kept]
    offsets = [value[2] for value in kept]
    if tokenizer.eos_token_id not in ids:
        ids[-1] = tokenizer.eos_token_id
        offsets[-1] = (0, 0)
    if len(ids) < max_len:
        padding = max_len - len(ids)
        ids.extend([tokenizer.pad_token_id] * padding)
        attention.extend([0] * padding)
        offsets.extend([(0, 0)] * padding)
    return ids, attention, offsets


def deterministic_metadata_positions(
    eligible: list[int],
    count: int,
    seed: int,
    dataset: str,
    user: str,
    passage_index: int,
    replicate: int,
) -> list[int]:
    ranked = sorted(
        eligible,
        key=lambda position: hashlib.sha256(
            f"{seed}|{dataset}|{user}|{passage_index}|{replicate}|{position}".encode()
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(
            f"insufficient metadata controls: eligible={len(ranked)} required={count}"
        )
    return ranked[:count]


def build_role_masks(
    tokenizer,
    collator,
    batch: dict,
    samples: list[dict],
    dataset: str,
    seed: int,
    replicates: int,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    ids = batch["item_text_ids"]
    base = batch["item_text_masks"]
    link = torch.zeros_like(base)
    cf = torch.zeros_like(base)
    controls = [torch.zeros_like(base) for _ in range(replicates)]
    audits = []
    max_len = collator.item_prompt_max_len
    for row_index, sample in enumerate(samples):
        for passage_index, text in enumerate(sample["input"][1:], start=1):
            char_spans = parse_role_char_spans(text)
            expected_ids, expected_attention, offsets = filtered_encoding_with_offsets(
                tokenizer, text, max_len
            )
            width = ids.shape[-1]
            if ids[row_index, passage_index].tolist() != expected_ids[:width]:
                raise ValueError(
                    f"filtered token identity mismatch for {sample['user']} passage {passage_index}"
                )
            if base[row_index, passage_index].int().tolist() != expected_attention[:width]:
                raise ValueError(
                    f"filtered attention identity mismatch for {sample['user']} passage {passage_index}"
                )
            link_positions = [
                position
                for position, offset in enumerate(offsets[:width])
                if expected_attention[position] and overlaps(offset, char_spans["link"])
            ]
            cf_positions = [
                position
                for position, offset in enumerate(offsets[:width])
                if expected_attention[position] and overlaps(offset, char_spans["cf"])
            ]
            metadata_span = char_spans["metadata"][0]
            eligible = []
            for position, offset in enumerate(offsets[:width]):
                if not expected_attention[position] or position in link_positions or position in cf_positions:
                    continue
                if not overlaps(offset, [metadata_span]):
                    continue
                start = max(offset[0], metadata_span[0])
                end = min(offset[1], metadata_span[1])
                if any(character.isalnum() for character in text[start:end]):
                    eligible.append(position)
            if not link_positions or not cf_positions:
                raise ValueError(
                    f"empty locked role for {sample['user']} passage {passage_index}"
                )
            link[row_index, passage_index, link_positions] = True
            cf[row_index, passage_index, cf_positions] = True
            control_positions = []
            for replicate in range(replicates):
                selected = deterministic_metadata_positions(
                    eligible,
                    len(link_positions),
                    seed,
                    dataset,
                    sample["user"],
                    passage_index,
                    replicate,
                )
                controls[replicate][row_index, passage_index, selected] = True
                control_positions.append("|".join(map(str, selected)))
            audits.append(
                {
                    "user": sample["user"],
                    "stratum": sample["stratum"],
                    "passage_index": passage_index,
                    "link_token_count": len(link_positions),
                    "cf_token_count": len(cf_positions),
                    "metadata_eligible_count": len(eligible),
                    "link_positions": "|".join(map(str, link_positions)),
                    "cf_positions": "|".join(map(str, cf_positions)),
                    **{
                        f"matched_positions_{index}": value
                        for index, value in enumerate(control_positions)
                    },
                }
            )
    if torch.any(link & cf):
        raise ValueError("link and CF role masks overlap")
    for control in controls:
        if torch.any(control & (link | cf)):
            raise ValueError("matched metadata control overlaps an identifier role")
        if torch.any(control & ~base):
            raise ValueError("matched metadata control selects inactive token")
    return {"link": link, "cf": cf, "controls": controls}, audits


def masked_attention(base: torch.Tensor, removed: torch.Tensor) -> torch.Tensor:
    result = base & ~removed
    if torch.any(result[:, 0, :].sum(dim=1) == 0):
        raise ValueError("coarse passage was accidentally masked")
    return result


@torch.no_grad()
def score_samples(
    model,
    tokenizer,
    collator,
    samples: list[dict],
    dataset: str,
    config: dict,
    device,
) -> tuple[list[dict], list[dict], float]:
    model.eval()
    rows, audits = [], []
    repeat_max = 0.0
    replicates = config["matched_control_replicates"]
    for start in range(0, len(samples), config["batch_size"]):
        chunk = samples[start : start + config["batch_size"]]
        batch = collator(
            [
                {"input": row["input"], "output": row["output"], "user_id": row["user"]}
                for row in chunk
            ]
        )
        ids = batch["item_text_ids"].to(device)
        base = batch["item_text_masks"].to(device)
        labels = batch["target_ids"].to(device)
        role_masks, chunk_audits = build_role_masks(
            tokenizer,
            collator,
            batch,
            chunk,
            dataset,
            config["seed"],
            replicates,
        )
        role_masks = {
            "link": role_masks["link"].to(device),
            "cf": role_masks["cf"].to(device),
            "controls": [value.to(device) for value in role_masks["controls"]],
        }
        conditions = {
            "full": base,
            "coarse_only": base.clone(),
            "minus_link_ids": masked_attention(base, role_masks["link"]),
            "minus_cf_ids": masked_attention(base, role_masks["cf"]),
            "minus_all_fine_ids": masked_attention(
                base, role_masks["link"] | role_masks["cf"]
            ),
        }
        conditions["coarse_only"][:, 1:, :] = False
        for replicate, control in enumerate(role_masks["controls"]):
            conditions[f"matched_{replicate}"] = masked_attention(base, control)
        scores = {}
        for name, attention in conditions.items():
            output = model(
                input_ids=ids, attention_mask=attention, labels=labels, return_dict=True
            )
            scores[name] = lexical_mean_logprob(
                output.logits, labels, tokenizer.eos_token_id
            ).cpu().numpy()
        repeated_output = model(
            input_ids=ids, attention_mask=base, labels=labels, return_dict=True
        )
        repeated = lexical_mean_logprob(
            repeated_output.logits, labels, tokenizer.eos_token_id
        ).cpu().numpy()
        repeat_max = max(
            repeat_max, float(np.max(np.abs(repeated - scores["full"])))
        )
        for index, sample in enumerate(chunk):
            full = float(scores["full"][index])
            controls = [
                float(scores[f"matched_{replicate}"][index])
                for replicate in range(replicates)
            ]
            control_effect = float(np.mean(controls) - full)
            raw_link = float(scores["minus_link_ids"][index] - full)
            row = {
                "user": sample["user"],
                "stratum": sample["stratum"],
                "history_length": sample["history_length"],
                "lp_full": full,
                "lp_coarse_only": float(scores["coarse_only"][index]),
                "lp_minus_link_ids": float(scores["minus_link_ids"][index]),
                "lp_minus_cf_ids": float(scores["minus_cf_ids"][index]),
                "lp_minus_all_fine_ids": float(scores["minus_all_fine_ids"][index]),
                "raw_link_harm": raw_link,
                "matched_control_effect": control_effect,
                "adjusted_link_echo": raw_link - control_effect,
                "metadata_benefit": float(
                    scores["minus_all_fine_ids"][index]
                    - scores["coarse_only"][index]
                ),
                "raw_cf_harm": float(scores["minus_cf_ids"][index] - full),
                "raw_all_id_harm": float(
                    scores["minus_all_fine_ids"][index] - full
                ),
            }
            for replicate, value in enumerate(controls):
                row[f"lp_matched_{replicate}"] = value
            rows.append(row)
        audits.extend(chunk_audits)
    return rows, audits, repeat_max


def summarize_scores(rows: list[dict], iterations: int, seed: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["stratum"]].append(row)
    metrics = (
        "raw_link_harm",
        "matched_control_effect",
        "adjusted_link_echo",
        "metadata_benefit",
        "raw_cf_harm",
        "raw_all_id_harm",
    )
    result = {}
    for stratum, values in sorted(grouped.items()):
        result[stratum] = {}
        for offset, metric in enumerate(metrics):
            array = np.asarray([row[metric] for row in values], dtype=np.float64)
            result[stratum][metric] = bootstrap_mean(
                array, iterations, seed + offset
            )
    miss = np.asarray(
        [row["adjusted_link_echo"] for row in grouped["tail_miss"]],
        dtype=np.float64,
    )
    hit = np.asarray(
        [row["adjusted_link_echo"] for row in grouped["tail_hit"]],
        dtype=np.float64,
    )
    result["tail_miss_minus_tail_hit_adjusted_link_echo"] = bootstrap_difference(
        miss, hit, iterations, seed + len(metrics)
    )
    return result


def evaluate_gates(
    summary: dict,
    counts: dict,
    repeat_max: float,
    role_rate: float,
    control_rate: float,
    cohort_match: bool,
    gates: dict,
) -> dict:
    miss = summary["tail_miss"]
    lower = gates["strict_bootstrap_lower_bound"]
    integrity = (
        counts.get("tail_miss", 0) == gates["tail_miss_n"]
        and counts.get("tail_hit", 0) == gates["tail_hit_n"]
        and repeat_max <= gates["repeat_max_abs_error"]
        and role_rate == gates["role_localization_rate"]
        and control_rate == gates["matched_control_eligibility_rate"]
        and cohort_match
    )
    raw = miss["raw_link_harm"]
    raw_gate = (
        raw["mean"] >= gates["raw_link_harm_mean"]
        and raw["ci95"][0] > lower
        and raw["positive_rate"] >= gates["raw_link_harm_positive_rate"]
    )
    adjusted = miss["adjusted_link_echo"]
    specificity = (
        adjusted["mean"] >= gates["adjusted_link_echo_mean"]
        and adjusted["ci95"][0] > lower
    )
    metadata = miss["metadata_benefit"]
    metadata_gate = (
        metadata["mean"] >= gates["metadata_benefit_mean"]
        and metadata["ci95"][0] > lower
    )
    association = summary["tail_miss_minus_tail_hit_adjusted_link_echo"]
    failure = (
        association["mean"] >= gates["failure_association_mean"]
        and association["ci95"][0] > lower
    )
    return {
        "integrity": bool(integrity),
        "raw_link_harm": bool(raw_gate),
        "role_specificity": bool(specificity),
        "separable_metadata_benefit": bool(metadata_gate),
        "failure_association": bool(failure),
    }


def read_locked_cohort(path: Path) -> list[tuple[str, str, str, str]]:
    with path.open(newline="") as handle:
        return sorted(
            (
                row["user"],
                row["stratum"],
                row["target"],
                row["selection_hash"],
            )
            for row in csv.DictReader(handle)
        )


def run_dataset(
    dataset: str, spec: dict, config: dict, output_root: Path, device
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
    with paths["run_config"].open() as handle:
        run_config = json.load(handle)
    if int(run_config.get("reverse_history", -1)) != 1:
        raise ValueError(f"{dataset} reverse_history is not locked to 1")
    if Path(DATASETS[dataset]["checkpoint"]).resolve() != paths["checkpoint"].resolve():
        raise ValueError("runtime checkpoint does not match preregistered checkpoint")

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
        raise ValueError(f"{dataset} cohort differs from locked CGI cohort")
    scored = [
        row for row in cohort if row["stratum"] in set(config["scored_strata"])
    ]
    counts = Counter(row["stratum"] for row in scored)
    samples = make_samples(scored, item2input, item2lexid)
    collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
    score_rows, audit_rows, repeat_max = score_samples(
        model, tokenizer, collator, samples, dataset, config, device
    )
    numeric_fields = [
        key
        for key in score_rows[0]
        if key.startswith("lp_")
        or key
        in {
            "raw_link_harm",
            "matched_control_effect",
            "adjusted_link_echo",
            "metadata_benefit",
            "raw_cf_harm",
            "raw_all_id_harm",
        }
    ]
    if not all(
        math.isfinite(float(row[key])) for row in score_rows for key in numeric_fields
    ):
        raise ValueError("NaN/Inf in LEI counterfactual scores")
    expected_passages = sum(row["history_length"] for row in scored)
    role_rate = len(audit_rows) / expected_passages if expected_passages else 0.0
    control_rate = (
        sum(
            row["metadata_eligible_count"] >= row["link_token_count"]
            for row in audit_rows
        )
        / expected_passages
        if expected_passages
        else 0.0
    )
    stats = summarize_scores(
        score_rows, config["bootstrap_iterations"], config["seed"]
    )
    gates = evaluate_gates(
        stats,
        dict(counts),
        repeat_max,
        role_rate,
        control_rate,
        cohort_match,
        config["gates"],
    )
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
    audit_fields = [
        "user",
        "stratum",
        "passage_index",
        "link_token_count",
        "cf_token_count",
        "metadata_eligible_count",
        "link_positions",
        "cf_positions",
    ] + [
        f"matched_positions_{index}"
        for index in range(config["matched_control_replicates"])
    ]
    write_csv(output_dir / "span_audit.csv", audit_rows, audit_fields)
    score_fields = [
        "user",
        "stratum",
        "history_length",
        "lp_full",
        "lp_coarse_only",
        "lp_minus_link_ids",
        "lp_minus_cf_ids",
        "lp_minus_all_fine_ids",
    ] + [
        f"lp_matched_{index}"
        for index in range(config["matched_control_replicates"])
    ] + [
        "raw_link_harm",
        "matched_control_effect",
        "adjusted_link_echo",
        "metadata_benefit",
        "raw_cf_harm",
        "raw_all_id_harm",
    ]
    write_csv(
        output_dir / "counterfactual_scores.csv", score_rows, score_fields
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
            "cohort_matches_cgi_e0": cohort_match,
            "role_localization_rate": role_rate,
            "matched_control_eligibility_rate": control_rate,
            "span_audit_rows": len(audit_rows),
            "expected_span_audit_rows": expected_passages,
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
    ordered = (
        ("integrity", "EXECUTION_INVALID"),
        ("raw_link_harm", "STOP_LEI_NO_RAW_ECHO"),
        ("role_specificity", "STOP_LEI_NO_ROLE_SPECIFICITY"),
        ("separable_metadata_benefit", "STOP_LEI_METADATA_NOT_SEPARABLE"),
        ("failure_association", "STOP_LEI_NO_FAILURE_LINK"),
    )
    for gate, failure in ordered:
        if not all(result["gates"][gate] for result in results.values()):
            return failure
    return "F0_MECHANISM_ALLOWED"


def write_report(path: Path, aggregate: dict) -> None:
    lines = [
        "# GRAM 第三阶段 LEI F0-D 诊断报告",
        "",
        f"- 决策：**`{aggregate['decision']}`**",
        "- 数据边界：冻结 validation checkpoint/cohort；未读 test，未训练，未生成 beam。",
        "- 干预边界：输入 token/position/passage 不变，只修改预注册 span 的 attention mask。",
        "",
        "## 双数据集 gate",
        "",
        "| Dataset | Integrity | Raw link harm | Role specificity | Metadata benefit | Failure association |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in aggregate["datasets"].items():
        gate = result["gates"]
        lines.append(
            f"| {dataset} | {gate['integrity']} | {gate['raw_link_harm']} | "
            f"{gate['role_specificity']} | {gate['separable_metadata_benefit']} | "
            f"{gate['failure_association']} |"
        )
    lines.extend(["", "## 锁定主统计", ""])
    for dataset, result in aggregate["datasets"].items():
        miss = result["statistics"]["tail_miss"]
        association = result["statistics"][
            "tail_miss_minus_tail_hit_adjusted_link_echo"
        ]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- tail-miss raw `R_link`: mean={miss['raw_link_harm']['mean']:.6f}, "
                f"95% CI=[{miss['raw_link_harm']['ci95'][0]:.6f}, "
                f"{miss['raw_link_harm']['ci95'][1]:.6f}], "
                f"P(>0)={miss['raw_link_harm']['positive_rate']:.6f}",
                f"- tail-miss adjusted `A_link`: mean={miss['adjusted_link_echo']['mean']:.6f}, "
                f"95% CI=[{miss['adjusted_link_echo']['ci95'][0]:.6f}, "
                f"{miss['adjusted_link_echo']['ci95'][1]:.6f}]",
                f"- tail-miss metadata `M_meta`: mean={miss['metadata_benefit']['mean']:.6f}, "
                f"95% CI=[{miss['metadata_benefit']['ci95'][0]:.6f}, "
                f"{miss['metadata_benefit']['ci95'][1]:.6f}]",
                f"- miss-hit adjusted association: mean={association['mean']:.6f}, "
                f"95% CI=[{association['ci95'][0]:.6f}, {association['ci95'][1]:.6f}]",
                f"- secondary tail-miss `R_cf`: mean={miss['raw_cf_harm']['mean']:.6f}; "
                f"`R_all`: mean={miss['raw_all_id_harm']['mean']:.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释边界",
            "",
            "只有五项 gate 在 Toys 与 Beauty 全部通过才允许进入 F1。CF-ID 结果为次要描述，",
            "不能挽救主 link-span gate；自然重复、语义相似或 matched control 差异本身不等于 echo。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open() as handle:
        config = json.load(handle)
    if not config.get("preregistered_before_new_checkpoint_scores"):
        raise ValueError("F0-D config lacks preregistration boundary")
    if not torch.cuda.is_available():
        raise RuntimeError("LEI F0-D requires CUDA for frozen checkpoint scoring")
    started = time.time()
    device = torch.device("cuda:0")
    results = {}
    for dataset, spec in config["datasets"].items():
        results[dataset] = run_dataset(
            dataset, spec, config, args.output_root, device
        )
    aggregate = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-07-24",
            "verification_status": "ANALYZED",
            "version_label": "lei_f0_d_v1",
        },
        "novelty_decision": "NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_GRAM_NARROWING",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
