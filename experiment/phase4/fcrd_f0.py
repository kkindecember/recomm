#!/usr/bin/env python3
"""FCRD F0: full-catalog popularity-residual SASRec effect gate."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from experiment.phase4.prpd_r0 import paired_bootstrap, read_teacher
from experiment.phase4.rpcd_t0 import (
    ROOT,
    SASRec,
    deduplicate,
    load_dataset,
    make_eval_tensor,
    metric,
    resolve_inputs,
    sha256,
    stable_fraction,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fuse_indices(gram: Sequence[int], teacher: Sequence[int], weight: float) -> list:
    gram = deduplicate(gram)
    teacher = deduplicate(teacher)
    gram_score = {
        item: 1.0 / math.log2(rank + 1.0)
        for rank, item in enumerate(gram, 1)
    }
    teacher_score = {
        item: 1.0 / math.log2(rank + 1.0)
        for rank, item in enumerate(teacher, 1)
    }
    union = deduplicate(list(gram) + list(teacher))
    stable = {item: index for index, item in enumerate(union)}
    return sorted(
        union,
        key=lambda item: (
            -(
                (1.0 - weight) * gram_score.get(item, 0.0)
                + weight * teacher_score.get(item, 0.0)
            ),
            stable[item],
        ),
    )


def popularity_log_probability(
    sequences: Mapping[str, Sequence[str]], catalog: Sequence[str]
) -> np.ndarray:
    counts = Counter(item for sequence in sequences.values() for item in sequence[:-2])
    smoothed = np.asarray([counts.get(item, 0) + 1.0 for item in catalog], dtype=np.float64)
    smoothed /= smoothed.sum()
    return np.log(smoothed)


@torch.no_grad()
def full_catalog_topk(
    model: SASRec,
    users: Sequence[str],
    sequences: Mapping[str, Sequence[str]],
    item_to_index: Mapping[str, int],
    log_popularity: np.ndarray,
    gammas: Sequence[float],
    max_length: int,
    batch_size: int,
    top_k: int,
    device: torch.device,
    dataset: str,
) -> np.ndarray:
    model.eval()
    output = np.empty((len(gammas), len(users), top_k), dtype=np.int32)
    log_pop = torch.tensor(log_popularity, dtype=torch.float32, device=device)
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        histories = [sequences[user][:-2] for user in batch_users]
        tensors = torch.stack(
            [make_eval_tensor(history, item_to_index, max_length) for history in histories]
        ).to(device)
        encoded = model.encode(tensors)
        last_indices = tensors.ne(0).sum(dim=1) - 1
        representation = encoded[
            torch.arange(encoded.shape[0], device=device), last_indices
        ]
        logits = representation @ model.item_embedding.weight[1:].T
        for row, history in enumerate(histories):
            seen = {item_to_index[item] for item in history if item in item_to_index}
            if seen:
                columns = torch.tensor([value - 1 for value in seen], device=device)
                logits[row, columns] = -torch.inf
        for gamma_index, gamma in enumerate(gammas):
            residual = logits - float(gamma) * log_pop[None, :]
            indices = torch.topk(residual, k=top_k, dim=1).indices
            output[
                gamma_index, start : start + len(batch_users)
            ] = indices.cpu().numpy().astype(np.int32)
        if start == 0 or (start // batch_size + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "scored_users": min(start + batch_size, len(users)),
                        "total_users": len(users),
                    }
                ),
                flush=True,
            )
    return output


def encode_rows(loaded: dict, users: Sequence[str]) -> list:
    item_to_zero = {item: index for index, item in enumerate(loaded["catalog"])}
    by_user = {row["user"]: row for row in loaded["rows"]}
    rows = []
    for user in users:
        row = by_user[user]
        rows.append(
            {
                "user": user,
                "gold": item_to_zero[row["gold"]],
                "gram": [item_to_zero[item] for item in deduplicate(row["pred_items"])],
            }
        )
    return rows


def evaluate(
    rows: Sequence[dict],
    teacher: np.ndarray,
    user_indices: Sequence[int],
    tail_items: set,
    weight: float,
    keep_arrays: bool = False,
) -> dict:
    gram_r = hybrid_r = gram_n = hybrid_n = 0.0
    gram_r50 = union_r50 = 0.0
    tail_gram_r50 = tail_union_r50 = 0.0
    tail_gram_n = tail_hybrid_n = 0.0
    n = tail_n = 0
    ndcg_pairs = []
    recall_pairs = []
    tail_ndcg_pairs = []
    union_pairs = []
    tail_union_pairs = []
    for index in user_indices:
        row = rows[index]
        proposals = teacher[index].tolist()
        hybrid = fuse_indices(row["gram"], proposals, weight)
        gold = row["gold"]
        gr, gn = metric(row["gram"], gold, 10)
        hr, hn = metric(hybrid, gold, 10)
        g50 = int(gold in row["gram"][:50])
        u50 = int(gold in set(row["gram"][:50]) | set(proposals[:50]))
        gram_r += gr
        hybrid_r += hr
        gram_n += gn
        hybrid_n += hn
        gram_r50 += g50
        union_r50 += u50
        n += 1
        if keep_arrays:
            ndcg_pairs.append((gn, hn))
            recall_pairs.append((gr, hr))
            union_pairs.append((g50, u50))
        if gold in tail_items:
            tail_gram_r50 += g50
            tail_union_r50 += u50
            tail_gram_n += gn
            tail_hybrid_n += hn
            tail_n += 1
            if keep_arrays:
                tail_ndcg_pairs.append((gn, hn))
                tail_union_pairs.append((g50, u50))
    result = {
        "n": n,
        "gram_recall@10": gram_r / n,
        "hybrid_recall@10": hybrid_r / n,
        "recall10_absolute_gain": (hybrid_r - gram_r) / n,
        "gram_ndcg@10": gram_n / n,
        "hybrid_ndcg@10": hybrid_n / n,
        "ndcg10_relative_gain": hybrid_n / gram_n - 1.0,
        "gram_recall@50": gram_r50 / n,
        "union_recall@50": union_r50 / n,
        "overall_union_recall50_absolute_gain": (union_r50 - gram_r50) / n,
        "tail_n": tail_n,
        "tail_gram_recall@50": tail_gram_r50 / tail_n,
        "tail_union_recall@50": tail_union_r50 / tail_n,
        "tail_union_recall50_absolute_gain": (
            tail_union_r50 - tail_gram_r50
        )
        / tail_n,
        "tail_gram_ndcg@10": tail_gram_n / tail_n,
        "tail_hybrid_ndcg@10": tail_hybrid_n / tail_n,
        "tail_ndcg10_relative_gain": tail_hybrid_n / tail_gram_n - 1.0,
    }
    if keep_arrays:
        result["_arrays"] = {
            "ndcg": np.asarray(ndcg_pairs, dtype=np.float64),
            "recall": np.asarray(recall_pairs, dtype=np.float64),
            "tail_ndcg": np.asarray(tail_ndcg_pairs, dtype=np.float64),
            "union": np.asarray(union_pairs, dtype=np.float64),
            "tail_union": np.asarray(tail_union_pairs, dtype=np.float64),
        }
    return result


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
    resolved = resolve_inputs(source_config)
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
        loaded = load_dataset(resolved[dataset])
        users = list(loaded["sequences"])
        checkpoint_path = ROOT / spec["sasrec_checkpoint"]
        teacher_path = ROOT / spec["teacher_top50"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint["catalog"] != loaded["catalog"]:
            raise ValueError(f"{dataset}: SASRec checkpoint catalog mismatch")
        teachers = read_teacher(teacher_path)
        if set(teachers) != set(users):
            raise ValueError(f"{dataset}: teacher user mismatch")
        calibration_indices = [
            index
            for index, user in enumerate(users)
            if stable_fraction(user, config["hybrid"]["calibration_salt"])
            < config["hybrid"]["calibration_fraction"]
        ]
        audit_indices = sorted(set(range(len(users))) - set(calibration_indices))
        counts = Counter(item for sequence in loaded["sequences"].values() for item in sequence[:-2])
        ordered = sorted(counts, key=lambda item: (-counts[item], item))
        head_count = max(1, math.ceil(len(ordered) * 0.2))
        head = set(ordered[:head_count])
        item_to_zero = {item: index for index, item in enumerate(loaded["catalog"])}
        tail_indices = {
            item_to_zero[item] for item in loaded["catalog"] if item not in head
        }
        prepared[dataset] = {
            "loaded": loaded,
            "users": users,
            "checkpoint": checkpoint,
            "teachers": teachers,
            "calibration_indices": calibration_indices,
            "audit_indices": audit_indices,
            "tail_indices": tail_indices,
        }
        preflight["datasets"][dataset] = {
            "users": len(users),
            "catalog_items": len(loaded["catalog"]),
            "calibration_users": len(calibration_indices),
            "audit_users": len(audit_indices),
            "checkpoint_epoch": checkpoint["selected_shared_epoch"],
            "checkpoint_sha256": sha256(checkpoint_path),
            "teacher_sha256": sha256(teacher_path),
            "catalog_identity": True,
            "target_match_rate": 1.0,
        }
    write_json(args.output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("FCRD F0 requires CUDA for full-catalog inference")
    device = torch.device(args.device)
    candidate_arrays = {}
    encoded_rows = {}
    gamma0_matches = gamma0_total = 0
    for dataset, data in prepared.items():
        teacher_cfg = source_config["teacher"]
        catalog = data["loaded"]["catalog"]
        model = SASRec(
            len(catalog),
            int(teacher_cfg["hidden_size"]),
            int(teacher_cfg["max_length"]),
            int(teacher_cfg["num_blocks"]),
            int(teacher_cfg["num_heads"]),
            float(teacher_cfg["dropout"]),
        ).to(device)
        model.load_state_dict(data["checkpoint"]["state_dict"])
        item_to_index = {item: index + 1 for index, item in enumerate(catalog)}
        log_pop = popularity_log_probability(data["loaded"]["sequences"], catalog)
        arrays = full_catalog_topk(
            model,
            data["users"],
            data["loaded"]["sequences"],
            item_to_index,
            log_pop,
            config["residual"]["gammas"],
            int(teacher_cfg["max_length"]),
            int(teacher_cfg["batch_size"]),
            int(config["residual"]["top_k"]),
            device,
            dataset,
        )
        candidate_arrays[dataset] = arrays
        encoded_rows[dataset] = encode_rows(data["loaded"], data["users"])
        item_to_zero = {item: index for index, item in enumerate(catalog)}
        for index, user in enumerate(data["users"]):
            expected = [item_to_zero[item] for item in data["teachers"][user]["items"]]
            observed = arrays[0, index].tolist()
            gamma0_matches += int(observed == expected)
            gamma0_total += 1
        np.savez_compressed(
            args.output_dir / f"{dataset}_full_catalog_top50.npz",
            users=np.asarray(data["users"]),
            gammas=np.asarray(config["residual"]["gammas"], dtype=np.float32),
            item_indices=arrays,
        )
        del model
        torch.cuda.empty_cache()
    identity_rate = gamma0_matches / gamma0_total
    if identity_rate != config["integrity"]["gamma0_teacher_top50_identity_rate"]:
        raise AssertionError(f"gamma0 teacher identity rate={identity_rate}")
    grid = []
    selection_gates = config["audit_gates"]
    for gamma_index, gamma in enumerate(config["residual"]["gammas"]):
        for weight in config["hybrid"]["weights"]:
            per_dataset = {}
            eligible = True
            macro = []
            for dataset, data in prepared.items():
                result = evaluate(
                    encoded_rows[dataset],
                    candidate_arrays[dataset][gamma_index],
                    data["calibration_indices"],
                    data["tail_indices"],
                    float(weight),
                )
                per_dataset[dataset] = result
                macro.append(result["ndcg10_relative_gain"])
                eligible &= (
                    result["overall_union_recall50_absolute_gain"]
                    >= selection_gates["overall_union_recall50_absolute_gain_min"]
                    and result["tail_union_recall50_absolute_gain"]
                    >= selection_gates["tail_union_recall50_absolute_gain_min"]
                    and result["recall10_absolute_gain"]
                    >= selection_gates["hybrid_recall10_absolute_gain_min"]
                    and result["tail_ndcg10_relative_gain"]
                    >= selection_gates["tail_ndcg10_relative_gain_min"]
                )
            grid.append(
                {
                    "gamma_index": gamma_index,
                    "gamma": gamma,
                    "weight": weight,
                    "eligible": eligible,
                    "macro_ndcg10_relative_gain": sum(macro) / len(macro),
                    "datasets": per_dataset,
                }
            )
    eligible_rows = [row for row in grid if row["eligible"]]
    calibration_qualified = bool(eligible_rows)
    if calibration_qualified:
        selected = max(
            eligible_rows,
            key=lambda row: (
                row["macro_ndcg10_relative_gain"],
                -float(row["gamma"]),
                -float(row["weight"]),
            ),
        )
    else:
        selected = next(
            row for row in grid if float(row["gamma"]) == 0.0 and float(row["weight"]) == 0.0
        )
    audit = {}
    gate_rows = []
    for dataset, data in prepared.items():
        result = evaluate(
            encoded_rows[dataset],
            candidate_arrays[dataset][selected["gamma_index"]],
            data["audit_indices"],
            data["tail_indices"],
            float(selected["weight"]),
            keep_arrays=True,
        )
        arrays = result.pop("_arrays")
        bootstrap = {
            "ndcg10_relative_gain_ci95": paired_bootstrap(
                arrays["ndcg"], int(config["bootstrap"]["iterations"]), 2034, True
            ),
            "recall10_absolute_gain_ci95": paired_bootstrap(
                arrays["recall"], int(config["bootstrap"]["iterations"]), 2035, False
            ),
            "tail_ndcg10_relative_gain_ci95": paired_bootstrap(
                arrays["tail_ndcg"], int(config["bootstrap"]["iterations"]), 2036, True
            ),
            "overall_union_recall50_absolute_gain_ci95": paired_bootstrap(
                arrays["union"], int(config["bootstrap"]["iterations"]), 2037, False
            ),
            "tail_union_recall50_absolute_gain_ci95": paired_bootstrap(
                arrays["tail_union"], int(config["bootstrap"]["iterations"]), 2038, False
            ),
        }
        audit[dataset] = {**result, "bootstrap": bootstrap}
        checks = {
            "overall_union_recall50_absolute_gain": result[
                "overall_union_recall50_absolute_gain"
            ]
            >= selection_gates["overall_union_recall50_absolute_gain_min"],
            "tail_union_recall50_absolute_gain": result[
                "tail_union_recall50_absolute_gain"
            ]
            >= selection_gates["tail_union_recall50_absolute_gain_min"],
            "ndcg10_relative_gain": result["ndcg10_relative_gain"]
            >= selection_gates["hybrid_ndcg10_relative_gain_min"],
            "recall10_absolute_gain": result["recall10_absolute_gain"]
            >= selection_gates["hybrid_recall10_absolute_gain_min"],
            "tail_ndcg10_relative_gain": result["tail_ndcg10_relative_gain"]
            >= selection_gates["tail_ndcg10_relative_gain_min"],
        }
        gate_rows.append(
            {"dataset": dataset, "checks": checks, "pass": all(checks.values())}
        )
    passed = calibration_qualified and all(row["pass"] for row in gate_rows)
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": (
            "FCRD_F1_DESIGN_ALLOWED"
            if passed
            else "STOP_FCRD_NO_FULL_CATALOG_RESIDUAL_EFFECT"
        ),
        "calibration_qualified": calibration_qualified,
        "selected_shared_config": {
            "gamma": selected["gamma"],
            "weight": selected["weight"],
            "calibration_macro_ndcg10_relative_gain": selected[
                "macro_ndcg10_relative_gain"
            ],
        },
        "grid": grid,
        "audit": audit,
        "gate_rows": gate_rows,
        "integrity": {
            "preflight_passed": True,
            "gamma0_teacher_top50_identity_rate": identity_rate,
            "shared_config": True,
            "target_match_rate": 1.0,
            "test_predictions_read": False,
            "sequence_test_target_indexed": False,
            "model_optimizer_steps": 0,
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "calibration_qualified": calibration_qualified,
                "selected_shared_config": summary["selected_shared_config"],
                "audit": audit,
                "gate_rows": gate_rows,
                "integrity": summary["integrity"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
