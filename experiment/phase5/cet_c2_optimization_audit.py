#!/usr/bin/env python3
"""CET C2-O: frozen-checkpoint optimization evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase3.hbtr_b1_smoke import read_sequences  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    collate,
    prepare,
    read_users,
    sha256,
    stable_sha,
    write_json,
)
from experiment.phase5.cet_c1 import structured_passage_mask  # noqa: E402
from experiment.phase5.cet_c2 import (  # noqa: E402
    backbone_forward,
    candidate_sequences,
)
from utils import generation_trie as gt  # noqa: E402


def load_config(path: Path) -> tuple[dict, dict]:
    config = json.loads(path.read_text())
    p0 = json.loads(
        (ROOT / "artifacts/phase4/configs/gcdh_p0_preregistered.json").read_text()
    )
    return config, p0


def ordered_calibration_samples(
    dataset: str,
    sequences: dict[str, list[str]],
    item2input: dict[str, str],
    item2lexid: dict[str, str],
    excluded_users: set[str],
    count: int,
    minimum_history_items: int,
    salt: str,
) -> list[dict]:
    users = sorted(
        sequences,
        key=lambda user: hashlib.sha256(
            f"{salt}|{dataset}|{user}".encode()
        ).hexdigest(),
    )
    samples = []
    for user in users:
        if user in excluded_users:
            continue
        items = sequences[user]
        if len(items) < 4:
            continue
        target = items[-3]
        history = items[:-3][-20:]
        if (
            len(history) < minimum_history_items
            or target not in item2lexid
            or any(item not in item2input for item in history)
        ):
            continue
        reversed_history = list(reversed(history))
        history_lex = " ; ".join(item2lexid[item] for item in reversed_history)
        samples.append(
            {
                "sample_key": f"{user}:train-prefix:{len(history)}",
                "user_id": user,
                "positive_item": target,
                "history_items": history,
                "input": [f"What would user purchase after {history_lex} ?"]
                + [item2input[item] for item in reversed_history],
                "output": item2lexid[target],
            }
        )
        if len(samples) == count:
            break
    if len(samples) != count:
        raise ValueError(
            f"{dataset}: insufficient fit-disjoint calibration samples "
            f"({len(samples)} != {count})"
        )
    return samples


def select_calibration_users(
    dataset: str,
    config: dict,
    p0: dict,
) -> tuple[list[str], list[dict]]:
    prepared = prepare(dataset, p0, torch.device("cpu"))
    train_users = read_users(
        ROOT / "artifacts/phase4/gcdh_p0_splits" / dataset / "train_users.txt"
    )
    frozen_validation_users = read_users(
        ROOT
        / "artifacts/phase4/gcdh_p0_splits"
        / dataset
        / "validation_users.txt"
    )
    excluded = train_users | frozen_validation_users
    samples = ordered_calibration_samples(
        dataset,
        prepared["sequences"],
        prepared["item2input"],
        prepared["item2lexid"],
        excluded,
        int(config["data"]["users_per_dataset"]),
        int(config["data"]["minimum_history_items"]),
        config["data"]["selection_salt"],
    )
    return [row["user_id"] for row in samples], samples


def make_splits(config: dict, p0: dict) -> dict:
    split_root = ROOT / config["data"]["split_root"]
    result = {}
    for dataset in config["datasets"]:
        sequences = read_sequences(
            ROOT / "GRAM/rec_datasets" / dataset / "user_sequence.txt"
        )
        train_path = (
            ROOT
            / "artifacts/phase4/gcdh_p0_splits"
            / dataset
            / "train_users.txt"
        )
        validation_path = (
            ROOT
            / "artifacts/phase4/gcdh_p0_splits"
            / dataset
            / "validation_users.txt"
        )
        train_users = read_users(train_path)
        validation_users = read_users(validation_path)
        prepared = prepare(dataset, p0, torch.device("cpu"))
        samples = ordered_calibration_samples(
            dataset,
            sequences,
            prepared["item2input"],
            prepared["item2lexid"],
            train_users | validation_users,
            int(config["data"]["users_per_dataset"]),
            int(config["data"]["minimum_history_items"]),
            config["data"]["selection_salt"],
        )
        users = [row["user_id"] for row in samples]
        if set(users) & train_users or set(users) & validation_users:
            raise ValueError(f"{dataset}: calibration exclusion failure")
        output = split_root / dataset / "calibration_users.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(users) + "\n")
        manifest = {
            "experiment_id": config["experiment_id"],
            "dataset": dataset,
            "users": len(users),
            "selection": "SHA256(salt|dataset|user), ascending",
            "selection_salt": config["data"]["selection_salt"],
            "target": "sequence[-3]",
            "history": "sequence[:-3][-20:]",
            "calibration_user_sha256": stable_sha(set(users)),
            "calibration_file_sha256": sha256(output),
            "source_train_user_sha256": stable_sha(train_users),
            "source_validation_user_sha256": stable_sha(validation_users),
            "train_disjoint": True,
            "all_frozen_validation_disjoint": True,
            "validation_target_read": False,
            "test_read": False,
            "sports_read": False,
        }
        write_json(output.parent / "manifest.json", manifest)
        result[dataset] = manifest
    lock = {
        "experiment_id": config["experiment_id"],
        "code_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(
            ROOT / "artifacts/phase5/configs/cet_c2o_preregistered.json"
        ),
        "datasets": result,
        "frozen_before_gpu_audit": True,
    }
    write_json(split_root / "frozen_manifest.json", lock)
    return lock


def load_frozen_samples(
    dataset: str,
    prepared: dict,
    config: dict,
) -> list[dict]:
    split_root = ROOT / config["data"]["split_root"]
    user_path = split_root / dataset / "calibration_users.txt"
    manifest = json.loads((split_root / dataset / "manifest.json").read_text())
    users = read_users(user_path)
    if sha256(user_path) != manifest["calibration_file_sha256"]:
        raise ValueError(f"{dataset}: calibration file SHA mismatch")
    if stable_sha(users) != manifest["calibration_user_sha256"]:
        raise ValueError(f"{dataset}: calibration user SHA mismatch")
    train_users = read_users(
        ROOT / "artifacts/phase4/gcdh_p0_splits" / dataset / "train_users.txt"
    )
    validation_users = read_users(
        ROOT
        / "artifacts/phase4/gcdh_p0_splits"
        / dataset
        / "validation_users.txt"
    )
    if users & train_users or users & validation_users:
        raise ValueError(f"{dataset}: frozen calibration users are not disjoint")
    by_user = {
        row["user_id"]: row
        for row in ordered_calibration_samples(
            dataset,
            prepared["sequences"],
            prepared["item2input"],
            prepared["item2lexid"],
            train_users | validation_users,
            int(config["data"]["users_per_dataset"]),
            int(config["data"]["minimum_history_items"]),
            config["data"]["selection_salt"],
        )
    }
    if set(by_user) != users:
        raise ValueError(f"{dataset}: deterministic calibration replay mismatch")
    return [by_user[user] for user in sorted(users)]


def legal_child_symmetric_kl(
    clean_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    sequences: list[list[int]],
    trie: gt.Trie,
    eos_token_id: int,
    temperature: float,
) -> tuple[torch.Tensor, int, int]:
    if clean_logits.shape != perturbed_logits.shape:
        raise ValueError("clean/perturbed logits shape mismatch")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    losses = []
    eligible_steps = 0
    for batch_index, sequence in enumerate(sequences):
        for position, gold in enumerate(sequence[1:]):
            if gold == eos_token_id:
                continue
            eligible_steps += 1
            allowed = trie.get(sequence[: position + 1])
            if gold not in allowed:
                raise ValueError("gold child is not legal")
            if len(allowed) < 2:
                continue
            indices = torch.as_tensor(
                allowed, dtype=torch.long, device=clean_logits.device
            )
            clean_values = (
                clean_logits[batch_index, position].index_select(0, indices)
                / float(temperature)
            )
            perturbed_values = (
                perturbed_logits[batch_index, position].index_select(0, indices)
                / float(temperature)
            )
            clean_log = torch.log_softmax(clean_values, dim=0)
            perturbed_log = torch.log_softmax(perturbed_values, dim=0)
            clean_prob = clean_log.exp()
            perturbed_prob = perturbed_log.exp()
            forward = (clean_prob * (clean_log - perturbed_log)).sum()
            reverse = (perturbed_prob * (perturbed_log - clean_log)).sum()
            losses.append(0.5 * (forward + reverse))
    if not losses:
        raise ValueError("no competitive legal-child steps")
    return torch.stack(losses).mean(), len(losses), eligible_steps


@torch.inference_mode()
def audit_arm(
    dataset: str,
    control: str,
    prepared: dict,
    samples: list[dict],
    config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    arm_root = ROOT / config["checkpoints"]["root"] / dataset / control
    checkpoint = arm_root / "model.pt"
    training_summary = json.loads((arm_root / "training_summary.json").read_text())
    if sha256(checkpoint) != training_summary["checkpoint_sha256"]:
        raise ValueError(f"{dataset}/{control}: checkpoint SHA mismatch")
    expected_sha = config["checkpoints"]["sha256"][dataset][control]
    if sha256(checkpoint) != expected_sha:
        raise ValueError(f"{dataset}/{control}: preregistered checkpoint SHA mismatch")
    backbone = prepared["model"].backbone
    backbone.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    backbone.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
    batch_size = int(config["evaluation"]["batch_size"])
    totals = {
        "symmetric_kl_weighted": 0.0,
        "clean_ce_weighted": 0.0,
        "perturbed_ce_weighted": 0.0,
        "competitive_steps": 0,
        "eligible_steps": 0,
        "label_tokens": 0,
        "masked_passages": 0,
        "maskable_passages": 0,
    }
    mask_signature = hashlib.sha256()
    started = time.time()
    for batch_index, start in enumerate(range(0, len(samples), batch_size), 1):
        rows = samples[start : start + batch_size]
        batch = collate(prepared["collator"], rows)
        for key in ("item_text_ids", "item_text_masks", "target_ids"):
            batch[key] = batch[key].to(device)
        clean_attention = batch["item_text_masks"].bool()
        perturbed_attention, decisions = structured_passage_mask(
            clean_attention,
            rows,
            dataset,
            int(config["views"]["mask_seed"]),
            float(config["views"]["mask_probability"]),
        )
        altered = [dict(row, positive_item="__altered__") for row in rows]
        _, altered_decisions = structured_passage_mask(
            clean_attention,
            altered,
            dataset,
            int(config["views"]["mask_seed"]),
            float(config["views"]["mask_probability"]),
        )
        if not torch.equal(decisions, altered_decisions):
            raise ValueError("mask policy depends on target")
        if not torch.equal(perturbed_attention[:, 0], clean_attention[:, 0]):
            raise ValueError("coarse passage changed")
        if clean_attention.shape[1] > 1 and not torch.equal(
            perturbed_attention[:, 1], clean_attention[:, 1]
        ):
            raise ValueError("newest fine passage changed")
        mask_signature.update(decisions.detach().cpu().numpy().tobytes())
        sequences = candidate_sequences(prepared, rows)
        clean = backbone_forward(backbone, batch, clean_attention)
        perturbed = backbone_forward(backbone, batch, perturbed_attention)
        if (
            not torch.isfinite(clean.loss)
            or not torch.isfinite(perturbed.loss)
            or not torch.isfinite(clean.logits).all()
            or not torch.isfinite(perturbed.logits).all()
        ):
            raise ValueError("non-finite audit output")
        sym_kl, competitive, eligible = legal_child_symmetric_kl(
            clean.logits,
            perturbed.logits,
            sequences,
            trie,
            int(prepared["tokenizer"].eos_token_id),
            float(config["views"]["temperature"]),
        )
        label_tokens = int((batch["target_ids"] != -100).sum())
        totals["symmetric_kl_weighted"] += float(sym_kl) * competitive
        totals["clean_ce_weighted"] += float(clean.loss) * label_tokens
        totals["perturbed_ce_weighted"] += float(perturbed.loss) * label_tokens
        totals["competitive_steps"] += competitive
        totals["eligible_steps"] += eligible
        totals["label_tokens"] += label_tokens
        totals["masked_passages"] += int(decisions.sum())
        totals["maskable_passages"] += int(
            clean_attention[:, 2:].any(dim=-1).sum()
        )
        if batch_index % 8 == 0:
            print(
                f"AUDIT_PROGRESS dataset={dataset} control={control} "
                f"users={min(start + batch_size, len(samples))}/{len(samples)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    if totals["masked_passages"] == 0 or totals["maskable_passages"] == 0:
        raise ValueError("audit did not exercise maskable passages")
    result = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "control": control,
        "status": "AUDITED",
        "users": len(samples),
        "calibration_user_sha256": stable_sha(
            {row["user_id"] for row in samples}
        ),
        "symmetric_legal_child_kl": (
            totals["symmetric_kl_weighted"] / totals["competitive_steps"]
        ),
        "clean_lexical_ce": (
            totals["clean_ce_weighted"] / totals["label_tokens"]
        ),
        "perturbed_lexical_ce": (
            totals["perturbed_ce_weighted"] / totals["label_tokens"]
        ),
        "competitive_legal_child_steps": totals["competitive_steps"],
        "eligible_lexical_steps": totals["eligible_steps"],
        "competitive_step_coverage": (
            totals["competitive_steps"] / totals["eligible_steps"]
        ),
        "masked_passages": totals["masked_passages"],
        "maskable_passages": totals["maskable_passages"],
        "masked_passage_coverage": (
            totals["masked_passages"] / totals["maskable_passages"]
        ),
        "mask_signature_sha256": mask_signature.hexdigest(),
        "checkpoint_sha256": expected_sha,
        "wall_time_seconds": time.time() - started,
        "target_independent_mask": True,
        "train_disjoint": True,
        "all_frozen_validation_disjoint": True,
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
    }
    output = output_root / dataset / control / "optimization_audit.json"
    write_json(output, result)
    return result


def analyze(config: dict, output_root: Path) -> dict:
    results = {
        dataset: {
            control: json.loads(
                (
                    output_root
                    / dataset
                    / control
                    / "optimization_audit.json"
                ).read_text()
            )
            for control in config["controls"]
        }
        for dataset in config["datasets"]
    }
    domain_reductions = {}
    domain_clean_changes = {}
    integrity = {}
    for dataset in config["datasets"]:
        c1 = results[dataset]["C1"]
        c2 = results[dataset]["C2"]
        domain_reductions[dataset] = (
            c1["symmetric_legal_child_kl"] - c2["symmetric_legal_child_kl"]
        ) / c1["symmetric_legal_child_kl"]
        domain_clean_changes[dataset] = (
            c2["clean_lexical_ce"] - c1["clean_lexical_ce"]
        ) / c1["clean_lexical_ce"]
        integrity[dataset] = {
            "finite": all(
                math.isfinite(results[dataset][control][metric])
                for control in config["controls"]
                for metric in (
                    "symmetric_legal_child_kl",
                    "clean_lexical_ce",
                    "perturbed_lexical_ce",
                )
            ),
            "c2_kl_below_c1": (
                c2["symmetric_legal_child_kl"]
                < c1["symmetric_legal_child_kl"]
            ),
            "mask_signature_all_controls_equal": len(
                {
                    results[dataset][control]["mask_signature_sha256"]
                    for control in config["controls"]
                }
            )
            == 1,
            "calibration_users_all_controls_equal": len(
                {
                    results[dataset][control]["calibration_user_sha256"]
                    for control in config["controls"]
                }
            )
            == 1,
            "targets_sealed": all(
                not results[dataset][control]["validation_target_read"]
                and not results[dataset][control]["test_read"]
                and not results[dataset][control]["sports_read"]
                for control in config["controls"]
            ),
        }
    macro_reduction = sum(domain_reductions.values()) / len(domain_reductions)
    gates = {
        "c2_symmetric_kl_below_c1_each_domain": all(
            value["c2_kl_below_c1"] for value in integrity.values()
        ),
        "macro_symmetric_kl_relative_reduction": (
            macro_reduction
            >= float(config["gates"]["macro_symmetric_kl_relative_reduction_min"])
        ),
        "clean_ce_relative_increase_each_domain": (
            max(domain_clean_changes.values())
            <= float(config["gates"]["clean_ce_relative_increase_max"])
        ),
        "integrity": all(
            all(checks.values()) for checks in integrity.values()
        ),
    }
    decision = (
        "INVALID_C2O_FIX_AND_EXACT_RERUN"
        if not gates["integrity"]
        else "CET_C2O_OPTIMIZATION_PASS"
        if all(gates.values())
        else "STOP_CET_WEAK_OPTIMIZATION_SIGNAL"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "results": results,
        "c2_vs_c1_symmetric_kl_relative_reduction": domain_reductions,
        "c2_vs_c1_clean_ce_relative_change": domain_clean_changes,
        "macro_symmetric_kl_relative_reduction": macro_reduction,
        "gates": gates,
        "integrity_checks": integrity,
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("make-splits", "audit", "analyze"), required=True
    )
    parser.add_argument("--dataset", choices=("Toys", "Beauty"))
    parser.add_argument("--control", choices=("C0", "C1", "C2"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config, p0 = load_config(args.config)
    registered_sha = config["integrity"]["code_sha256"]
    actual_sha = sha256(Path(__file__))
    if registered_sha != "PENDING_FREEZE" and registered_sha != actual_sha:
        raise ValueError(
            f"C2-O code SHA mismatch: actual={actual_sha} "
            f"registered={registered_sha}"
        )
    if args.stage == "make-splits":
        print(json.dumps(make_splits(config, p0), ensure_ascii=False, indent=2))
        return 0
    frozen = json.loads(
        (
            ROOT
            / config["data"]["split_root"]
            / "frozen_manifest.json"
        ).read_text()
    )
    if frozen["code_sha256"] != actual_sha:
        raise ValueError("C2-O frozen manifest code SHA mismatch")
    if args.stage == "analyze":
        print(json.dumps(analyze(config, args.output_root), ensure_ascii=False, indent=2))
        return 0
    if args.dataset is None or args.control is None:
        parser.error("--dataset and --control are required for audit")
    if not torch.cuda.is_available():
        raise RuntimeError("C2-O checkpoint audit requires CUDA")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    prepared = prepare(args.dataset, p0, device)
    samples = load_frozen_samples(args.dataset, prepared, config)
    result = audit_arm(
        args.dataset,
        args.control,
        prepared,
        samples,
        config,
        args.output_root,
        device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
