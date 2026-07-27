#!/usr/bin/env python3
"""CPGV V0: can frozen GRAM verify SASRec-recovered lexical paths?"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GRAM_SRC = ROOT / "GRAM/src"
PHASE3 = ROOT / "experiment/phase3"
for candidate in (ROOT, GRAM_SRC, PHASE3):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache/huggingface"))

from model.gram_t5_outputs import (  # noqa: E402
    BaseModelOutputWithPastAndCrossAttentions,
)
from processor import CollatorGRAM  # noqa: E402
from utils import generation_trie as gt  # noqa: E402
from utils import indexing  # noqa: E402

from hbtr_b1_smoke import (  # noqa: E402
    create_model_and_tokenizer,
    make_runtime_args,
    read_sequences,
)

from experiment.phase3.marc_l0 import encode_catalog_trie, local_distribution  # noqa: E402
from experiment.phase4.prpd_r0 import read_teacher  # noqa: E402
from experiment.phase4.rpcd_t0 import (  # noqa: E402
    deduplicate,
    load_dataset,
    popularity_tail,
    resolve_inputs,
    sha256,
    stable_fraction,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_from_scores(items: Sequence[str], scores: Sequence[float]) -> list:
    if len(items) != len(scores):
        raise ValueError("item/score length mismatch")
    if len(set(items)) != len(items):
        raise ValueError("candidate items are not unique")
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("candidate score is not finite")
    return [
        item
        for _, item in sorted(
            enumerate(items), key=lambda pair: (-float(scores[pair[0]]), pair[0])
        )
    ]


def summarize_rows(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("empty diagnostic subgroup")
    n = len(rows)
    sas_hit10 = sum(row["sasrec_rank"] <= 10 for row in rows) / n
    exact_hit10 = sum(row["exact_rank"] <= 10 for row in rows) / n
    return {
        "n": n,
        "sasrec_recall@10": sas_hit10,
        "exact_rescore_recall@10": exact_hit10,
        "exact_minus_sasrec_recall10_absolute": exact_hit10 - sas_hit10,
        "sasrec_mrr": sum(1.0 / row["sasrec_rank"] for row in rows) / n,
        "exact_rescore_mrr": sum(1.0 / row["exact_rank"] for row in rows) / n,
        "mean_rank_improvement": sum(
            row["sasrec_rank"] - row["exact_rank"] for row in rows
        )
        / n,
        "median_rank_improvement": float(
            np.median(
                [row["sasrec_rank"] - row["exact_rank"] for row in rows]
            )
        ),
        "pairwise_concordance": sum(row["pairwise_concordance"] for row in rows)
        / n,
    }


def build_validation_input(
    user: str,
    sequence: Sequence[str],
    item2input: Mapping[str, str],
    item2lexid: Mapping[str, str],
    max_history: int,
) -> dict:
    history = list(sequence[:-2])[-max_history:]
    target = sequence[-2]
    if target in history:
        raise ValueError(f"{user}: validation target appears in model history")
    missing = [item for item in history if item not in item2input or item not in item2lexid]
    if missing:
        raise ValueError(f"{user}: history item missing from GRAM mapping {missing[:3]}")
    ordered = list(reversed(history))
    history_lex = " ; ".join(item2lexid[item] for item in ordered)
    return {
        "user": user,
        "target": target,
        "history": history,
        "input": [f"What would user purchase after {history_lex} ?"]
        + [item2input[item] for item in ordered],
    }


@torch.no_grad()
def score_user_candidates(
    model,
    collator: CollatorGRAM,
    trie: gt.Trie,
    sample: dict,
    candidates: Sequence[str],
    item2lexid: Mapping[str, str],
    device: torch.device,
    eos_token_id: int,
) -> tuple[list[float], dict]:
    model.eval()
    input_batch = collator(
        [
            {
                "input": sample["input"],
                "output": item2lexid[candidates[0]],
                "user_id": sample["user"],
            }
        ]
    )
    input_ids = input_batch["item_text_ids"].to(device)
    input_masks = input_batch["item_text_masks"].to(device)
    model.encoder.n_passages = input_ids.size(1)
    flat_ids = input_ids.view(1, -1)
    flat_masks = input_masks.view(1, -1)
    encoder_hidden = model.encoder(
        input_ids=flat_ids,
        attention_mask=flat_masks,
        return_dict=True,
    )[0]
    target = collator.encode_target_split(
        [item2lexid[item] for item in candidates]
    )
    labels = target["input_ids"]
    target_masks = target["attention_mask"].bool()
    labels = labels.masked_fill(~target_masks, -100).to(device)
    count = len(candidates)
    encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=encoder_hidden.expand(count, -1, -1)
    )
    output = model(
        input_ids=None,
        attention_mask=flat_masks.expand(count, -1),
        encoder_outputs=encoder_outputs,
        labels=labels,
        return_dict=True,
    )
    scores = []
    trie_checked = trie_valid = finite = values = 0
    for index in range(count):
        node_rows, checked, valid = local_distribution(
            output.logits[index],
            labels[index],
            trie,
            eos_token_id,
        )
        if not node_rows:
            raise ValueError("candidate lexical path has no scored node")
        score = sum(row["gold_log_probability"] for row in node_rows) / len(node_rows)
        scores.append(float(score))
        trie_checked += checked
        trie_valid += valid
        finite += int(math.isfinite(score))
        values += 1
    return scores, {
        "trie_checked": trie_checked,
        "trie_valid": trie_valid,
        "finite": finite,
        "values": values,
    }


def prepare_dataset(
    dataset: str,
    spec: dict,
    source_config: dict,
    config: dict,
) -> tuple[dict, dict]:
    paths = resolve_inputs(source_config)[dataset]
    loaded = load_dataset(paths)
    teachers = read_teacher(ROOT / spec["teacher_top50"])
    if set(teachers) != set(loaded["sequences"]):
        raise ValueError(f"{dataset}: teacher/sequence user mismatch")
    runtime = make_runtime_args(dataset)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
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
    tail = popularity_tail(loaded["sequences"])
    cohort = []
    target_in_history = 0
    for row in loaded["rows"]:
        user = row["user"]
        if (
            stable_fraction(user, config["cohort"]["calibration_salt"])
            < config["cohort"]["calibration_fraction"]
        ):
            continue
        gram = deduplicate(row["pred_items"])
        sasrec = deduplicate(teachers[user]["items"])
        gold = row["gold"]
        if gold in gram[:50] or gold not in sasrec[:50]:
            continue
        sample = build_validation_input(
            user,
            loaded["sequences"][user],
            item2input,
            item2lexid,
            runtime.max_his,
        )
        target_in_history += int(gold in sample["history"])
        cohort.append(
            {
                **sample,
                "candidates": sasrec[:50],
                "sasrec_rank": sasrec.index(gold) + 1,
                "pop_group": "tail" if gold in tail else "head",
            }
        )
    cohort.sort(key=lambda row: hashlib.sha256(row["user"].encode()).hexdigest())
    checkpoint = ROOT / spec["checkpoint"]
    run_config = ROOT / spec["run_config"]
    teacher_path = ROOT / spec["teacher_top50"]
    for path in (checkpoint, run_config, teacher_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(cohort) < config["cohort"]["minimum_users_per_dataset"]:
        raise ValueError(f"{dataset}: only {len(cohort)} eligible users")
    mapping_ok = sum(
        all(item in item2lexid for item in row["candidates"]) for row in cohort
    )
    if mapping_ok != len(cohort):
        raise ValueError(f"{dataset}: proposal outside lexical mapping")
    preflight = {
        "eligible_users": len(cohort),
        "head_users": sum(row["pop_group"] == "head" for row in cohort),
        "tail_users": sum(row["pop_group"] == "tail" for row in cohort),
        "sasrec_proposal_recall@10": sum(row["sasrec_rank"] <= 10 for row in cohort)
        / len(cohort),
        "candidate_count_per_user": 50,
        "candidate_unique_rate": sum(
            len(set(row["candidates"])) == 50 for row in cohort
        )
        / len(cohort),
        "mapping_rate": mapping_ok / len(cohort),
        "target_input_inclusion_rate": target_in_history / len(cohort),
        "checkpoint_sha256": sha256(checkpoint),
        "run_config_sha256": sha256(run_config),
        "teacher_sha256": sha256(teacher_path),
    }
    return {
        **loaded,
        "teachers": teachers,
        "item2input": item2input,
        "item2lexid": item2lexid,
        "cohort": cohort,
    }, preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    config = json.loads(args.config.read_text())
    source_path = ROOT / config["source_rpcd_config"]
    source_config = json.loads(source_path.read_text())
    prepared = {}
    preflight = {
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(args.config),
        "source_rpcd_config_sha256": sha256(source_path),
        "test_predictions_read": False,
        "sequence_test_target_indexed": False,
        "datasets": {},
    }
    for dataset, spec in config["datasets"].items():
        prepared[dataset], preflight["datasets"][dataset] = prepare_dataset(
            dataset, spec, source_config, config
        )
        write_csv(
            args.output_dir / dataset / "cohort.csv",
            [
                {
                    "user": row["user"],
                    "target": row["target"],
                    "sasrec_rank": row["sasrec_rank"],
                    "pop_group": row["pop_group"],
                    "history_length": len(row["history"]),
                }
                for row in prepared[dataset]["cohort"]
            ],
            ("user", "target", "sasrec_rank", "pop_group", "history_length"),
        )
    write_json(args.output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CPGV V0 frozen scoring requires CUDA")
    device = torch.device(args.device)
    results = {}
    total_audit = Counter()
    for dataset, spec in config["datasets"].items():
        data = prepared[dataset]
        model, tokenizer, runtime = create_model_and_tokenizer(dataset, device)
        model.eval()
        collator = CollatorGRAM(tokenizer=tokenizer, args=runtime, mode="train")
        trie = encode_catalog_trie(collator, data["item2lexid"])
        user_rows = []
        candidate_rows = []
        score_audit = Counter()
        for index, sample in enumerate(data["cohort"], 1):
            scores, audit = score_user_candidates(
                model,
                collator,
                trie,
                sample,
                sample["candidates"],
                data["item2lexid"],
                device,
                tokenizer.eos_token_id,
            )
            score_audit.update(audit)
            ranking = rank_from_scores(sample["candidates"], scores)
            exact_rank = ranking.index(sample["target"]) + 1
            gold_score = scores[sample["candidates"].index(sample["target"])]
            lower = sum(
                score < gold_score
                for item, score in zip(sample["candidates"], scores)
                if item != sample["target"]
            )
            ties = sum(
                score == gold_score
                for item, score in zip(sample["candidates"], scores)
                if item != sample["target"]
            )
            concordance = (lower + 0.5 * ties) / 49.0
            user_rows.append(
                {
                    "user": sample["user"],
                    "target": sample["target"],
                    "pop_group": sample["pop_group"],
                    "history_length": len(sample["history"]),
                    "sasrec_rank": sample["sasrec_rank"],
                    "exact_rank": exact_rank,
                    "rank_improvement": sample["sasrec_rank"] - exact_rank,
                    "gold_exact_score": gold_score,
                    "pairwise_concordance": concordance,
                }
            )
            candidate_rows.extend(
                {
                    "user": sample["user"],
                    "candidate": item,
                    "sasrec_rank": rank + 1,
                    "exact_score": scores[rank],
                    "is_gold": int(item == sample["target"]),
                }
                for rank, item in enumerate(sample["candidates"])
            )
            if index % 50 == 0 or index == len(data["cohort"]):
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "scored_users": index,
                            "total_users": len(data["cohort"]),
                        }
                    ),
                    flush=True,
                )
        overall = summarize_rows(user_rows)
        groups = {
            group: summarize_rows([row for row in user_rows if row["pop_group"] == group])
            for group in ("head", "tail")
        }
        integrity = {
            "eligible_users": len(user_rows),
            "mapping_rate": preflight["datasets"][dataset]["mapping_rate"],
            "finite_rate": score_audit["finite"] / score_audit["values"],
            "target_input_inclusion_rate": preflight["datasets"][dataset][
                "target_input_inclusion_rate"
            ],
            "trie_membership_rate": score_audit["trie_valid"]
            / score_audit["trie_checked"],
            "model_optimizer_steps": 0,
        }
        gates = {
            "eligible_users": integrity["eligible_users"]
            >= config["gates"]["eligible_users_min"],
            "exact_rescore_recall10": overall["exact_rescore_recall@10"]
            >= config["gates"]["exact_rescore_recall10_min"],
            "exact_minus_sasrec_recall10": overall[
                "exact_minus_sasrec_recall10_absolute"
            ]
            >= config["gates"]["exact_minus_sasrec_recall10_absolute_min"],
            "mapping_rate": integrity["mapping_rate"]
            == config["gates"]["mapping_rate"],
            "finite_rate": integrity["finite_rate"] == config["gates"]["finite_rate"],
            "target_input_inclusion_rate": integrity["target_input_inclusion_rate"]
            == config["gates"]["target_input_inclusion_rate"],
            "trie_membership_rate": integrity["trie_membership_rate"]
            == config["gates"]["trie_membership_rate"],
            "model_optimizer_steps": integrity["model_optimizer_steps"]
            == config["gates"]["model_optimizer_steps"],
        }
        results[dataset] = {
            "overall": overall,
            "groups": groups,
            "integrity": integrity,
            "gates": gates,
            "pass": all(gates.values()),
        }
        total_audit.update(score_audit)
        write_csv(
            args.output_dir / dataset / "user_results.csv",
            user_rows,
            (
                "user",
                "target",
                "pop_group",
                "history_length",
                "sasrec_rank",
                "exact_rank",
                "rank_improvement",
                "gold_exact_score",
                "pairwise_concordance",
            ),
        )
        write_csv(
            args.output_dir / dataset / "candidate_scores.csv",
            candidate_rows,
            ("user", "candidate", "sasrec_rank", "exact_score", "is_gold"),
        )
        del model
        torch.cuda.empty_cache()
    decision = (
        "CPGV_V1_DESIGN_ALLOWED"
        if all(value["pass"] for value in results.values())
        else "STOP_CPGV_GRAM_CANNOT_VERIFY_PROPOSALS"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "datasets": results,
        "integrity": {
            "preflight_passed": True,
            "test_predictions_read": False,
            "sequence_test_target_indexed": False,
            "model_optimizer_steps": 0,
        },
        "config_sha256": preflight["config_sha256"],
        "code_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
