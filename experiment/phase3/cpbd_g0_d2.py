#!/usr/bin/env python3
"""CPBD G0-D2 frozen fixed-budget outcome diagnosis."""

from __future__ import annotations

import argparse
import csv
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

from cpbd_g0_d1 import build_serialization, encode_exact, overlaps  # noqa: E402
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
from lei_f0 import read_locked_cohort  # noqa: E402
from s0_offline_diagnostics import decode_item_ids, read_predictions  # noqa: E402

PASSAGE_RE = re.compile(
    r"^item: (?P<link>.*?); similar items: (?P<cf>.*?); (?P<meta>.*)$",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "artifacts/phase3/configs/cpbd_g0_d2_preregistered.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/phase3/cpbd_g0_d2",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/第三阶段/GRAM_第三阶段_CPBD_G0_D2诊断报告.md",
    )
    return parser.parse_args()


def metadata_first_passage(text: str):
    match = PASSAGE_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"passage grammar mismatch: {text[:120]!r}")
    link = match.group("link")
    cf_values = match.group("cf").split(", ")
    metadata = match.group("meta")
    rebuilt, _ = build_serialization(link, cf_values, metadata, False)
    if rebuilt != text:
        raise ValueError("current passage did not round-trip")
    alternate, spans = build_serialization(link, cf_values, metadata, True)
    return alternate, spans, {
        "link": link,
        "cf_values": cf_values,
        "metadata": metadata,
    }


def metadata_positions(encoding: dict, spans, tokenizer) -> list[int]:
    return [
        position
        for position, (token_id, mask, offset) in enumerate(
            zip(
                encoding["visible_ids"],
                encoding["visible_mask"],
                encoding["visible_offsets"],
            )
        )
        if mask
        and token_id not in {tokenizer.pad_token_id, tokenizer.eos_token_id}
        and overlaps(offset, spans["metadata_total"])
    ]


def compare_encoded(batch_ids, batch_masks, row: int, passage: int, expected, label):
    width = batch_ids.shape[-1]
    if batch_ids[row, passage].tolist() != expected["visible_ids"][:width]:
        raise ValueError(f"{label} input identity mismatch row={row} passage={passage}")
    if (
        batch_masks[row, passage].int().tolist()
        != expected["visible_mask"][:width]
    ):
        raise ValueError(
            f"{label} attention identity mismatch row={row} passage={passage}"
        )


def build_masks(
    tokenizer,
    current_batch,
    alternate_batch,
    current_samples,
    alternate_samples,
    slice_size: int,
):
    current_ids = current_batch["item_text_ids"]
    current_masks = current_batch["item_text_masks"]
    alternate_ids = alternate_batch["item_text_ids"]
    alternate_masks = alternate_batch["item_text_masks"]
    all_recovered = torch.zeros_like(alternate_masks)
    recovered_slice = torch.zeros_like(alternate_masks)
    matched_slice = torch.zeros_like(alternate_masks)
    audits = []
    eligible = 0
    matched = 0
    component_ok = 0
    coarse_ok = 0
    max_len = alternate_ids.shape[-1]
    if torch.equal(current_ids[:, 0], alternate_ids[:, 0]) and torch.equal(
        current_masks[:, 0], alternate_masks[:, 0]
    ):
        coarse_ok = len(current_samples)
    for row, (current_sample, alternate_sample) in enumerate(
        zip(current_samples, alternate_samples)
    ):
        for passage, (current_text, alternate_text) in enumerate(
            zip(current_sample["input"][1:], alternate_sample["input"][1:]), start=1
        ):
            expected_alternate, alternate_spans, components = metadata_first_passage(
                current_text
            )
            if expected_alternate != alternate_text:
                raise ValueError("alternate passage identity mismatch")
            component_ok += int(
                all(
                    value in alternate_text
                    for value in (
                        components["link"],
                        components["metadata"],
                        *components["cf_values"],
                    )
                )
            )
            _, current_spans = build_serialization(
                components["link"],
                components["cf_values"],
                components["metadata"],
                False,
            )
            current_encoding = encode_exact(tokenizer, current_text, 999, 128)
            alternate_encoding = encode_exact(tokenizer, alternate_text, 999, 128)
            compare_encoded(
                current_ids,
                current_masks,
                row,
                passage,
                current_encoding,
                "current",
            )
            compare_encoded(
                alternate_ids,
                alternate_masks,
                row,
                passage,
                alternate_encoding,
                "metadata_first",
            )
            current_meta = metadata_positions(
                current_encoding, current_spans, tokenizer
            )
            alternate_meta = metadata_positions(
                alternate_encoding, alternate_spans, tokenizer
            )
            recovered = alternate_meta[len(current_meta) :]
            all_recovered[row, passage, recovered] = True
            selected_recovered = []
            selected_matched = []
            if len(recovered) >= slice_size:
                eligible += 1
                if len(current_meta) < slice_size:
                    raise ValueError("matched metadata slice is not eligible")
                selected_recovered = recovered[:slice_size]
                selected_matched = alternate_meta[
                    len(current_meta) - slice_size : len(current_meta)
                ]
                recovered_slice[row, passage, selected_recovered] = True
                matched_slice[row, passage, selected_matched] = True
                matched += int(
                    len(selected_recovered) == len(selected_matched) == slice_size
                )
            audits.append(
                {
                    "user": current_sample["user"],
                    "stratum": current_sample["stratum"],
                    "passage_index": passage,
                    "current_metadata_visible": len(current_meta),
                    "metadata_first_metadata_visible": len(alternate_meta),
                    "recovered_count": len(recovered),
                    "slice_eligible": int(len(recovered) >= slice_size),
                    "recovered_positions": "|".join(map(str, recovered)),
                    "recovered_slice_positions": "|".join(
                        map(str, selected_recovered)
                    ),
                    "matched_slice_positions": "|".join(map(str, selected_matched)),
                }
            )
    if torch.any(all_recovered & ~alternate_masks):
        raise ValueError("recovered mask selected inactive positions")
    if torch.any(recovered_slice & matched_slice):
        raise ValueError("slice masks overlap")
    expected_passages = sum(row["history_length"] for row in current_samples)
    return (
        {
            "all_recovered": all_recovered,
            "recovered_slice": recovered_slice,
            "matched_slice": matched_slice,
        },
        audits,
        {
            "coarse_prompt_identity_rate": coarse_ok / len(current_samples),
            "raw_component_identity_rate": component_ok / expected_passages,
            "recovered_mask_localization_rate": len(audits) / expected_passages,
            "matched_slice_eligibility_rate": matched / eligible if eligible else 0.0,
            "slice_eligible_passages": eligible,
            "expected_passages": expected_passages,
            "encoded_width": max_len,
        },
    )


def masked(base: torch.Tensor, removed: torch.Tensor) -> torch.Tensor:
    result = base & ~removed
    if torch.any(result[:, 0, :].sum(dim=1) == 0):
        raise ValueError("coarse prompt was masked")
    return result


@torch.no_grad()
def score_samples(model, tokenizer, collator, samples, config, device):
    model.eval()
    rows, audits = [], []
    repeat_max = 0.0
    integrity_sums = defaultdict(float)
    integrity_batches = 0
    for start in range(0, len(samples), config["batch_size"]):
        current = samples[start : start + config["batch_size"]]
        alternate = []
        for sample in current:
            alt_passages = [sample["input"][0]]
            for passage in sample["input"][1:]:
                alt_passages.append(metadata_first_passage(passage)[0])
            alternate.append({**sample, "input": alt_passages})
        current_batch = collator(
            [
                {"input": row["input"], "output": row["output"], "user_id": row["user"]}
                for row in current
            ]
        )
        alternate_batch = collator(
            [
                {"input": row["input"], "output": row["output"], "user_id": row["user"]}
                for row in alternate
            ]
        )
        if not torch.equal(
            current_batch["target_ids"], alternate_batch["target_ids"]
        ):
            raise ValueError("target identity changed")
        role_masks, chunk_audits, chunk_integrity = build_masks(
            tokenizer,
            current_batch,
            alternate_batch,
            current,
            alternate,
            config["matched_slice_tokens_per_passage"],
        )
        audits.extend(chunk_audits)
        for key, value in chunk_integrity.items():
            if isinstance(value, float):
                integrity_sums[key] += value
        integrity_batches += 1

        current_ids = current_batch["item_text_ids"].to(device)
        current_mask = current_batch["item_text_masks"].to(device)
        alternate_ids = alternate_batch["item_text_ids"].to(device)
        alternate_mask = alternate_batch["item_text_masks"].to(device)
        labels = current_batch["target_ids"].to(device)
        role_masks = {key: value.to(device) for key, value in role_masks.items()}
        conditions = {
            "current": (current_ids, current_mask),
            "metadata_first_full": (alternate_ids, alternate_mask),
            "metadata_first_minus_all_recovered": (
                alternate_ids,
                masked(alternate_mask, role_masks["all_recovered"]),
            ),
            "metadata_first_minus_recovered_slice8": (
                alternate_ids,
                masked(alternate_mask, role_masks["recovered_slice"]),
            ),
            "metadata_first_minus_matched_visible_slice8": (
                alternate_ids,
                masked(alternate_mask, role_masks["matched_slice"]),
            ),
        }
        scores = {}
        for name, (ids, attention) in conditions.items():
            output = model(
                input_ids=ids,
                attention_mask=attention,
                labels=labels,
                return_dict=True,
            )
            scores[name] = lexical_mean_logprob(
                output.logits, labels, tokenizer.eos_token_id
            ).cpu().numpy()
        repeated = model(
            input_ids=current_ids,
            attention_mask=current_mask,
            labels=labels,
            return_dict=True,
        )
        repeated_score = lexical_mean_logprob(
            repeated.logits, labels, tokenizer.eos_token_id
        ).cpu().numpy()
        repeat_max = max(
            repeat_max,
            float(np.max(np.abs(repeated_score - scores["current"]))),
        )
        for index, sample in enumerate(current):
            current_lp = float(scores["current"][index])
            full = float(scores["metadata_first_full"][index])
            minus_all = float(
                scores["metadata_first_minus_all_recovered"][index]
            )
            minus_recovered8 = float(
                scores["metadata_first_minus_recovered_slice8"][index]
            )
            minus_matched8 = float(
                scores["metadata_first_minus_matched_visible_slice8"][index]
            )
            rows.append(
                {
                    "user": sample["user"],
                    "stratum": sample["stratum"],
                    "history_length": sample["history_length"],
                    "lp_current": current_lp,
                    "lp_metadata_first_full": full,
                    "lp_metadata_first_minus_all_recovered": minus_all,
                    "lp_metadata_first_minus_recovered_slice8": minus_recovered8,
                    "lp_metadata_first_minus_matched_visible_slice8": minus_matched8,
                    "net_reallocation": full - current_lp,
                    "recovered_all_contribution": full - minus_all,
                    "residual_layout_effect": minus_all - current_lp,
                    "recovered_slice8_contribution": full - minus_recovered8,
                    "matched_visible_slice8_contribution": full - minus_matched8,
                }
            )
    mean_integrity = {
        key: value / integrity_batches for key, value in integrity_sums.items()
    }
    return rows, audits, repeat_max, mean_integrity


def summarize(rows, iterations: int, seed: int):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["stratum"]].append(row)
    metrics = (
        "net_reallocation",
        "recovered_all_contribution",
        "residual_layout_effect",
        "recovered_slice8_contribution",
        "matched_visible_slice8_contribution",
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
        [row["net_reallocation"] for row in grouped["tail_miss"]],
        dtype=np.float64,
    )
    hit = np.asarray(
        [row["net_reallocation"] for row in grouped["tail_hit"]],
        dtype=np.float64,
    )
    result["failure_association"] = bootstrap_difference(
        miss, hit, iterations, seed + len(metrics)
    )
    return result


def evaluate_gates(stats, counts, repeat_max, integrity, cohort_match, gates):
    miss = stats["tail_miss"]
    lower = gates["strict_bootstrap_lower_bound"]
    integrity_gate = (
        counts.get("tail_miss", 0) == gates["tail_miss_n"]
        and counts.get("tail_hit", 0) == gates["tail_hit_n"]
        and cohort_match
        and repeat_max <= gates["current_score_repeat_max_abs_error"]
        and integrity["coarse_prompt_identity_rate"]
        == gates["coarse_prompt_identity_rate"]
        and integrity["raw_component_identity_rate"]
        == gates["raw_component_identity_rate"]
        and integrity["recovered_mask_localization_rate"]
        == gates["recovered_mask_localization_rate"]
        and integrity["matched_slice_eligibility_rate"]
        == gates["matched_slice_eligibility_rate"]
    )
    net = miss["net_reallocation"]
    net_gate = (
        net["mean"] >= gates["tail_miss_net_reallocation_mean"]
        and net["ci95"][0] > lower
        and net["positive_rate"]
        >= gates["tail_miss_net_reallocation_positive_rate"]
    )
    recovered = miss["recovered_all_contribution"]
    recovered8 = miss["recovered_slice8_contribution"]
    recovered_gate = (
        recovered["mean"]
        >= gates["tail_miss_recovered_all_contribution_mean"]
        and recovered["ci95"][0] > lower
        and recovered["positive_rate"]
        >= gates["tail_miss_recovered_all_positive_rate"]
        and recovered8["mean"] >= gates["tail_miss_recovered_slice8_mean"]
        and recovered8["ci95"][0] > lower
        and recovered["mean"]
        >= gates["recovered_contribution_fraction_of_net"] * net["mean"]
    )
    broad_harm = (
        stats["tail_hit"]["net_reallocation"]["mean"]
        >= gates["tail_hit_net_reallocation_mean_min"]
    )
    return {
        "integrity": bool(integrity_gate),
        "net_value": bool(net_gate),
        "recovered_value": bool(recovered_gate),
        "no_broad_harm": bool(broad_harm),
    }


def run_dataset(dataset, spec, config, output_root, device):
    dataset_dir = ROOT / "GRAM/rec_datasets" / dataset
    paths = {
        "checkpoint": ROOT / spec["checkpoint"],
        "run_config": ROOT / spec["run_config"],
        "predictions": ROOT / spec["predictions"],
        "s0_summary": ROOT / spec["s0_summary"],
        "cgi_cohort": ROOT / spec["cgi_cohort"],
        "d1_census": ROOT / spec["d1_census"],
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
        2,
        20,
    )
    locked = read_locked_cohort(paths["cgi_cohort"])
    scored = [
        row for row in cohort if row["stratum"] in set(config["scored_strata"])
    ]
    current_identity = sorted(
        (row["user"], row["stratum"], row["target"], row["selection_hash"])
        for row in scored
    )
    locked_identity = sorted(
        row for row in locked if row[1] in set(config["scored_strata"])
    )
    cohort_match = current_identity == locked_identity
    if not cohort_match:
        raise ValueError(f"{dataset} cohort differs from locked CGI cohort")
    counts = Counter(row["stratum"] for row in scored)
    samples = make_samples(scored, item2input, item2lexid)
    collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
    score_rows, audit_rows, repeat_max, integrity = score_samples(
        model, tokenizer, collator, samples, config, device
    )
    numeric = [
        key
        for key in score_rows[0]
        if key.startswith("lp_")
        or key
        in {
            "net_reallocation",
            "recovered_all_contribution",
            "residual_layout_effect",
            "recovered_slice8_contribution",
            "matched_visible_slice8_contribution",
        }
    ]
    if not all(
        math.isfinite(float(row[key])) for row in score_rows for key in numeric
    ):
        raise ValueError("non-finite score")
    stats = summarize(
        score_rows, config["bootstrap_iterations"], config["seed"]
    )
    gates = evaluate_gates(
        stats,
        dict(counts),
        repeat_max,
        integrity,
        cohort_match,
        config["integrity_gates"] | config["mechanism_gates_per_dataset"],
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
    write_csv(
        output_dir / "span_audit.csv",
        audit_rows,
        list(audit_rows[0]),
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
        "statistics": stats,
        "gates": gates,
        "integrity": {
            **integrity,
            "current_score_repeat_max_abs_error": repeat_max,
            "cohort_matches_cgi_e0": cohort_match,
            "finite_metric_rate": 1.0,
            "fixed_budget_rate": 1.0,
            "prediction_audit": prediction_audit,
            "sequence_last_item_read": False,
            "test_data_read": False,
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


def decide(results):
    for gate, failure in (
        ("integrity", "EXECUTION_INVALID"),
        ("net_value", "STOP_CPBD_NO_NET_VALUE"),
        ("recovered_value", "STOP_CPBD_NO_RECOVERED_VALUE"),
        ("no_broad_harm", "STOP_CPBD_BROAD_HARM"),
    ):
        if not all(result["gates"][gate] for result in results.values()):
            return failure
    return "G1_DESIGN_ALLOWED"


def write_report(path: Path, aggregate):
    lines = [
        "# GRAM 第三阶段：CPBD G0-D2 frozen outcome diagnosis",
        "",
        f"- 决策：**`{aggregate['decision']}`**",
        "- 边界：锁定 validation checkpoint/cohort；固定 128-token budget 与 CF identity；未训练、未读 test。",
        "",
        "## 双数据集结果",
        "",
        "| Dataset | Integrity | Net value | Recovered value | No broad harm |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, result in aggregate["datasets"].items():
        gate = result["gates"]
        lines.append(
            f"| {dataset} | {gate['integrity']} | {gate['net_value']} | "
            f"{gate['recovered_value']} | {gate['no_broad_harm']} |"
        )
    lines.extend(["", "## 锁定主统计", ""])
    for dataset, result in aggregate["datasets"].items():
        miss = result["statistics"]["tail_miss"]
        hit = result["statistics"]["tail_hit"]
        net = miss["net_reallocation"]
        recovered = miss["recovered_all_contribution"]
        recovered8 = miss["recovered_slice8_contribution"]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- tail-miss net: mean={net['mean']:.6f}, 95% CI=[{net['ci95'][0]:.6f}, {net['ci95'][1]:.6f}], P(>0)={net['positive_rate']:.6f}",
                f"- tail-miss recovered-all: mean={recovered['mean']:.6f}, 95% CI=[{recovered['ci95'][0]:.6f}, {recovered['ci95'][1]:.6f}], P(>0)={recovered['positive_rate']:.6f}",
                f"- tail-miss recovered-slice8: mean={recovered8['mean']:.6f}, 95% CI=[{recovered8['ci95'][0]:.6f}, {recovered8['ci95'][1]:.6f}]",
                f"- tail-hit net mean={hit['net_reallocation']['mean']:.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释边界",
            "",
            "只有四类 gate 在双数据集同时通过才解锁 G1。matched visible slice、",
            "residual layout 与 failure association 均为 secondary descriptive，不能挽救主 gate。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    with config_path.open() as handle:
        config = json.load(handle)
    if not config.get("preregistered_before_new_checkpoint_scores"):
        raise ValueError("G0-D2 lacks preregistration boundary")
    if not torch.cuda.is_available():
        raise RuntimeError("G0-D2 requires CUDA for frozen checkpoint scoring")
    started = time.time()
    results = {}
    device = torch.device("cuda:0")
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
            "version_label": "cpbd_g0_d2_v1",
        },
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
