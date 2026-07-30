#!/usr/bin/env python3
"""CET V2-A: beta-strength optimization smoke on fresh calibration-B users."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gcdh_p0 import (  # noqa: E402
    collate,
    prepare,
    read_users,
    sha256,
    stable_sha,
    write_json,
)
from experiment.phase5.cet_c1 import (  # noqa: E402
    legal_child_kl,
    structured_passage_mask,
)
from experiment.phase5.cet_c2 import (  # noqa: E402
    backbone_forward,
    candidate_sequences,
)
from experiment.phase5.cet_c2_optimization_audit import (  # noqa: E402
    legal_child_symmetric_kl,
    ordered_calibration_samples,
)
from utils import generation_trie as gt  # noqa: E402


def load_configs(path: Path) -> tuple[dict, dict]:
    config = json.loads(path.read_text())
    p0 = json.loads(
        (ROOT / "artifacts/phase4/configs/gcdh_p0_preregistered.json").read_text()
    )
    return config, p0


def ordered_file_users(path: Path) -> list[str]:
    users = [value.strip() for value in path.read_text().splitlines() if value.strip()]
    if len(users) != len(set(users)):
        raise ValueError(f"duplicate users in {path}")
    return users


def excluded_users(dataset: str, config: dict) -> set[str]:
    paths = [
        ROOT / pattern.format(dataset=dataset)
        for pattern in config["data"]["excluded_user_files"]
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing exclusion files: {missing}")
    result: set[str] = set()
    for path in paths:
        result.update(read_users(path))
    return result


def make_splits(config: dict, p0: dict) -> dict:
    split_root = ROOT / config["data"]["split_root"]
    result = {}
    for dataset in config["datasets"]:
        prepared = prepare(dataset, p0, torch.device("cpu"))
        excluded = excluded_users(dataset, config)
        count = int(config["data"]["fit_users"]) + int(
            config["data"]["evaluation_users"]
        )
        samples = ordered_calibration_samples(
            dataset,
            prepared["sequences"],
            prepared["item2input"],
            prepared["item2lexid"],
            excluded,
            count,
            int(config["data"]["minimum_history_items"]),
            config["data"]["selection_salt"],
        )
        fit_count = int(config["data"]["fit_users"])
        subsets = {
            "fit": [row["user_id"] for row in samples[:fit_count]],
            "evaluation": [row["user_id"] for row in samples[fit_count:]],
        }
        if set(subsets["fit"]) & set(subsets["evaluation"]):
            raise ValueError(f"{dataset}: fit/evaluation overlap")
        if (set(subsets["fit"]) | set(subsets["evaluation"])) & excluded:
            raise ValueError(f"{dataset}: excluded user entered calibration-B")
        dataset_root = split_root / dataset
        manifests = {}
        for subset, users in subsets.items():
            path = dataset_root / f"{subset}_users.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(users) + "\n")
            manifests[subset] = {
                "users": len(users),
                "user_sha256": stable_sha(set(users)),
                "file_sha256": sha256(path),
            }
        manifest = {
            "experiment_id": config["experiment_id"],
            "dataset": dataset,
            "selection_salt": config["data"]["selection_salt"],
            "selection": "SHA256(salt|dataset|user), ascending",
            "target": "sequence[-3]",
            "history": "sequence[:-3][-20:]",
            "excluded_users": len(excluded),
            "excluded_file_sha256": {
                pattern: sha256(ROOT / pattern.format(dataset=dataset))
                for pattern in config["data"]["excluded_user_files"]
            },
            "subsets": manifests,
            "fit_evaluation_disjoint": True,
            "validation_target_read": False,
            "test_read": False,
            "sports_read": False,
        }
        write_json(dataset_root / "manifest.json", manifest)
        result[dataset] = manifest
        del prepared
    lock = {
        "experiment_id": config["experiment_id"],
        "code_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(
            ROOT / "artifacts/phase5/configs/cet_v2a_preregistered.json"
        ),
        "datasets": result,
        "frozen_before_gpu_run": True,
    }
    write_json(split_root / "frozen_manifest.json", lock)
    return lock


def load_frozen_samples(
    dataset: str,
    subset: str,
    prepared: dict,
    config: dict,
) -> list[dict]:
    split_root = ROOT / config["data"]["split_root"] / dataset
    path = split_root / f"{subset}_users.txt"
    manifest = json.loads((split_root / "manifest.json").read_text())
    users = ordered_file_users(path)
    expected = manifest["subsets"][subset]
    if sha256(path) != expected["file_sha256"]:
        raise ValueError(f"{dataset}/{subset}: file SHA mismatch")
    if stable_sha(set(users)) != expected["user_sha256"]:
        raise ValueError(f"{dataset}/{subset}: user SHA mismatch")
    excluded = excluded_users(dataset, config)
    if set(users) & excluded:
        raise ValueError(f"{dataset}/{subset}: exclusion failure")
    count = int(config["data"]["fit_users"]) + int(
        config["data"]["evaluation_users"]
    )
    replay = ordered_calibration_samples(
        dataset,
        prepared["sequences"],
        prepared["item2input"],
        prepared["item2lexid"],
        excluded,
        count,
        int(config["data"]["minimum_history_items"]),
        config["data"]["selection_salt"],
    )
    by_user = {row["user_id"]: row for row in replay}
    if any(user not in by_user for user in users):
        raise ValueError(f"{dataset}/{subset}: deterministic replay failure")
    return [by_user[user] for user in users]


@torch.inference_mode()
def evaluate(
    dataset: str,
    backbone,
    prepared: dict,
    samples: list[dict],
    config: dict,
    device: torch.device,
) -> dict:
    backbone.eval()
    trie = gt.Trie(prepared["encoded_candidates"])
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
    signature = hashlib.sha256()
    batch_size = int(config["evaluation"]["batch_size"])
    for start in range(0, len(samples), batch_size):
        rows = samples[start : start + batch_size]
        batch = collate(prepared["collator"], rows)
        for key in ("item_text_ids", "item_text_masks", "target_ids"):
            batch[key] = batch[key].to(device)
        clean_attention = batch["item_text_masks"].bool()
        perturbed_attention, decisions = structured_passage_mask(
            clean_attention,
            rows,
            dataset,
            int(config["views"]["evaluation_mask_seed"]),
            float(config["views"]["mask_probability"]),
        )
        altered = [dict(row, positive_item="__altered__") for row in rows]
        _, altered_decisions = structured_passage_mask(
            clean_attention,
            altered,
            dataset,
            int(config["views"]["evaluation_mask_seed"]),
            float(config["views"]["mask_probability"]),
        )
        if not torch.equal(decisions, altered_decisions):
            raise ValueError("evaluation mask depends on target")
        if not torch.equal(perturbed_attention[:, 0], clean_attention[:, 0]):
            raise ValueError("coarse passage changed")
        if clean_attention.shape[1] > 1 and not torch.equal(
            perturbed_attention[:, 1], clean_attention[:, 1]
        ):
            raise ValueError("newest fine passage changed")
        signature.update(decisions.detach().cpu().numpy().tobytes())
        sequences = candidate_sequences(prepared, rows)
        clean = backbone_forward(backbone, batch, clean_attention)
        perturbed = backbone_forward(backbone, batch, perturbed_attention)
        if (
            not torch.isfinite(clean.loss)
            or not torch.isfinite(perturbed.loss)
            or not torch.isfinite(clean.logits).all()
            or not torch.isfinite(perturbed.logits).all()
        ):
            raise ValueError("non-finite evaluation output")
        symmetric_kl, competitive, eligible = legal_child_symmetric_kl(
            clean.logits,
            perturbed.logits,
            sequences,
            trie,
            int(prepared["tokenizer"].eos_token_id),
            float(config["views"]["temperature"]),
        )
        label_tokens = int((batch["target_ids"] != -100).sum())
        totals["symmetric_kl_weighted"] += float(symmetric_kl) * competitive
        totals["clean_ce_weighted"] += float(clean.loss) * label_tokens
        totals["perturbed_ce_weighted"] += float(perturbed.loss) * label_tokens
        totals["competitive_steps"] += competitive
        totals["eligible_steps"] += eligible
        totals["label_tokens"] += label_tokens
        totals["masked_passages"] += int(decisions.sum())
        totals["maskable_passages"] += int(
            clean_attention[:, 2:].any(dim=-1).sum()
        )
    if totals["competitive_steps"] == 0 or totals["masked_passages"] == 0:
        raise ValueError("evaluation did not exercise CET mechanism")
    return {
        "users": len(samples),
        "symmetric_legal_child_kl": (
            totals["symmetric_kl_weighted"] / totals["competitive_steps"]
        ),
        "clean_lexical_ce": totals["clean_ce_weighted"] / totals["label_tokens"],
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
        "mask_signature_sha256": signature.hexdigest(),
    }


def run_arm(
    dataset: str,
    arm: str,
    config: dict,
    p0: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0, device)
    fit_samples = load_frozen_samples(dataset, "fit", prepared, config)
    evaluation_samples = load_frozen_samples(
        dataset, "evaluation", prepared, config
    )
    if {row["user_id"] for row in fit_samples} & {
        row["user_id"] for row in evaluation_samples
    }:
        raise ValueError("fit/evaluation overlap")
    backbone = prepared["model"].backbone
    source_checkpoint = ROOT / p0["datasets"][dataset]["checkpoint"]
    source_sha = sha256(source_checkpoint)
    if source_sha != config["source_checkpoint_sha256"][dataset]:
        raise ValueError(f"{dataset}: source checkpoint SHA mismatch")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    trainable = list(backbone.decoder.block[-1].parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    initial_parameters = [value.detach().clone() for value in trainable]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    beta = float(config["arms"][arm]["beta"])
    alpha = float(config["views"]["alpha"])
    trie = gt.Trie(prepared["encoded_candidates"])
    batch_size = int(config["optimization"]["batch_size"])
    steps = int(config["optimization"]["steps"])
    gradient_norms = []
    losses = []
    kls = []
    train_signature = hashlib.sha256()
    masked_passages = 0
    started = time.time()
    backbone.train()
    indices: list[int] = []
    cursor = 0
    cycle = 0
    for step in range(steps):
        if cursor >= len(indices):
            indices = list(range(len(fit_samples)))
            random.Random(int(config["seed"]) + cycle).shuffle(indices)
            cursor = 0
            cycle += 1
        selected = indices[cursor : cursor + batch_size]
        cursor += batch_size
        if not selected:
            raise ValueError("empty fit batch")
        rows = [fit_samples[index] for index in selected]
        batch = collate(prepared["collator"], rows)
        for key in ("item_text_ids", "item_text_masks", "target_ids"):
            batch[key] = batch[key].to(device)
        clean_attention = batch["item_text_masks"].bool()
        perturbed_attention, decisions = structured_passage_mask(
            clean_attention,
            rows,
            dataset,
            int(config["views"]["fit_mask_seed"]) + step,
            float(config["views"]["mask_probability"]),
        )
        if not torch.equal(perturbed_attention[:, 0], clean_attention[:, 0]):
            raise ValueError("coarse passage changed")
        if clean_attention.shape[1] > 1 and not torch.equal(
            perturbed_attention[:, 1], clean_attention[:, 1]
        ):
            raise ValueError("newest fine passage changed")
        train_signature.update(decisions.detach().cpu().numpy().tobytes())
        masked_passages += int(decisions.sum())
        sequences = candidate_sequences(prepared, rows)
        optimizer.zero_grad(set_to_none=True)
        clean = backbone_forward(backbone, batch, clean_attention)
        if not torch.isfinite(clean.loss) or not torch.isfinite(clean.logits).all():
            raise ValueError("non-finite clean training output")
        clean_logits = clean.logits.detach()
        clean.loss.backward()
        perturbed = backbone_forward(backbone, batch, perturbed_attention)
        anchored_kl, competitive = legal_child_kl(
            clean_logits,
            perturbed.logits,
            sequences,
            trie,
            int(prepared["tokenizer"].eos_token_id),
            float(config["views"]["temperature"]),
        )
        extra = alpha * perturbed.loss + beta * anchored_kl
        extra.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config["optimization"]["gradient_clip_norm"])
        )
        if not torch.isfinite(norm):
            raise ValueError("non-finite gradient")
        optimizer.step()
        total = float(clean.loss.detach() + extra.detach())
        if not math.isfinite(total):
            raise ValueError("non-finite total loss")
        gradient_norms.append(float(norm))
        losses.append(total)
        kls.append(float(anchored_kl.detach()))
        if (step + 1) % 10 == 0:
            print(
                f"V2A_PROGRESS dataset={dataset} arm={arm} "
                f"step={step + 1}/{steps} loss={total:.6f} "
                f"kl={float(anchored_kl):.6f} "
                f"competitive={competitive} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    if masked_passages == 0:
        raise ValueError("fit did not mask any passage")
    evaluation = evaluate(
        dataset,
        backbone,
        prepared,
        evaluation_samples,
        config,
        device,
    )
    parameter_change = max(
        float((after.detach() - before).abs().max())
        for before, after in zip(initial_parameters, trainable)
    )
    arm_root = output_root / dataset / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint = arm_root / "decoder_last_layer.pt"
    torch.save(backbone.decoder.block[-1].state_dict(), checkpoint)
    saved_state = torch.load(checkpoint, map_location=device)
    with torch.no_grad():
        next(backbone.decoder.block[-1].parameters()).add_(1.0)
    backbone.decoder.block[-1].load_state_dict(saved_state, strict=True)
    reload_difference = max(
        float(
            (
                parameter.detach()
                - saved_state[name].to(parameter.device)
            )
            .abs()
            .max()
        )
        for (name, parameter) in backbone.decoder.block[-1].named_parameters()
    )
    result = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset,
        "arm": arm,
        "beta": beta,
        "status": "COMPLETED",
        "fit_users": len(fit_samples),
        "evaluation_users": len(evaluation_samples),
        "fit_user_sha256": stable_sha(
            {row["user_id"] for row in fit_samples}
        ),
        "evaluation_user_sha256": stable_sha(
            {row["user_id"] for row in evaluation_samples}
        ),
        "optimizer_steps": steps,
        "mean_training_loss": sum(losses) / len(losses),
        "mean_training_anchored_kl": sum(kls) / len(kls),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "parameter_max_abs_change": parameter_change,
        "fit_masked_passages": masked_passages,
        "fit_mask_signature_sha256": train_signature.hexdigest(),
        "evaluation": evaluation,
        "decoder_checkpoint": str(checkpoint.relative_to(ROOT)),
        "decoder_checkpoint_sha256": sha256(checkpoint),
        "reload_max_abs_parameter_difference": reload_difference,
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_sha_unchanged": sha256(source_checkpoint) == source_sha,
        "wall_time_seconds": time.time() - started,
        "fit_evaluation_disjoint": True,
        "validation_target_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(arm_root / "summary.json", result)
    del prepared, backbone
    torch.cuda.empty_cache()
    return result


def analyze(config: dict, output_root: Path) -> dict:
    results = {
        dataset: {
            arm: json.loads(
                (output_root / dataset / arm / "summary.json").read_text()
            )
            for arm in config["arms"]
        }
        for dataset in config["datasets"]
    }
    reductions = {}
    clean_changes = {}
    integrity = {}
    for dataset in config["datasets"]:
        v1 = results[dataset]["V1"]
        v2 = results[dataset]["V2"]
        v1_kl = float(v1["evaluation"]["symmetric_legal_child_kl"])
        v2_kl = float(v2["evaluation"]["symmetric_legal_child_kl"])
        v1_ce = float(v1["evaluation"]["clean_lexical_ce"])
        v2_ce = float(v2["evaluation"]["clean_lexical_ce"])
        reductions[dataset] = (v1_kl - v2_kl) / v1_kl
        clean_changes[dataset] = (v2_ce - v1_ce) / v1_ce
        integrity[dataset] = {
            "finite": all(
                math.isfinite(value)
                for arm in (v1, v2)
                for value in (
                    arm["mean_training_loss"],
                    arm["mean_training_anchored_kl"],
                    arm["gradient_norm_min"],
                    arm["gradient_norm_max"],
                    arm["evaluation"]["symmetric_legal_child_kl"],
                    arm["evaluation"]["clean_lexical_ce"],
                )
            ),
            "nonzero_gradient": min(
                v1["gradient_norm_min"], v2["gradient_norm_min"]
            )
            > 0,
            "parameter_change": min(
                v1["parameter_max_abs_change"],
                v2["parameter_max_abs_change"],
            )
            > 0,
            "checkpoint_reload": max(
                v1["reload_max_abs_parameter_difference"],
                v2["reload_max_abs_parameter_difference"],
            )
            <= 1e-8,
            "source_sha_unchanged": (
                v1["source_checkpoint_sha_unchanged"]
                and v2["source_checkpoint_sha_unchanged"]
            ),
            "fit_users_matched": v1["fit_user_sha256"] == v2["fit_user_sha256"],
            "evaluation_users_matched": (
                v1["evaluation_user_sha256"] == v2["evaluation_user_sha256"]
            ),
            "fit_masks_matched": (
                v1["fit_mask_signature_sha256"]
                == v2["fit_mask_signature_sha256"]
            ),
            "evaluation_masks_matched": (
                v1["evaluation"]["mask_signature_sha256"]
                == v2["evaluation"]["mask_signature_sha256"]
            ),
            "targets_sealed": all(
                not arm["validation_target_read"]
                and not arm["test_read"]
                and not arm["sports_read"]
                for arm in (v1, v2)
            ),
        }
    macro_reduction = sum(reductions.values()) / len(reductions)
    gates = {
        "v2_kl_below_v1_each_domain": all(value > 0 for value in reductions.values()),
        "macro_kl_relative_reduction": (
            macro_reduction
            >= float(config["gates"]["macro_kl_relative_reduction_min"])
        ),
        "clean_ce_safety_each_domain": (
            max(clean_changes.values())
            <= float(config["gates"]["clean_ce_relative_increase_max"])
        ),
        "integrity": all(all(checks.values()) for checks in integrity.values()),
    }
    decision = (
        "INVALID_V2A_FIX_AND_EXACT_RERUN"
        if not gates["integrity"]
        else "CET_V2A_OPTIMIZATION_PASS"
        if all(gates.values())
        else "STOP_CET_V2_STRENGTHENING_FAILED"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "results": results,
        "v2_vs_v1_symmetric_kl_relative_reduction": reductions,
        "v2_vs_v1_clean_ce_relative_change": clean_changes,
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
        "--stage", choices=("make-splits", "run", "analyze"), required=True
    )
    parser.add_argument("--dataset", choices=("Toys", "Beauty"))
    parser.add_argument("--arm", choices=("V1", "V2"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config, p0 = load_configs(args.config)
    code_sha = sha256(Path(__file__))
    registered_sha = config["integrity"]["code_sha256"]
    if registered_sha != "PENDING_FREEZE" and code_sha != registered_sha:
        raise ValueError(
            f"V2-A code SHA mismatch: actual={code_sha} "
            f"registered={registered_sha}"
        )
    if args.stage == "make-splits":
        print(json.dumps(make_splits(config, p0), ensure_ascii=False, indent=2))
        return 0
    frozen = json.loads(
        (ROOT / config["data"]["split_root"] / "frozen_manifest.json").read_text()
    )
    if frozen["code_sha256"] != code_sha:
        raise ValueError("V2-A frozen-manifest code SHA mismatch")
    if args.stage == "analyze":
        print(json.dumps(analyze(config, args.output_root), ensure_ascii=False, indent=2))
        return 0
    if args.dataset is None or args.arm is None:
        parser.error("--dataset and --arm are required for run")
    if not torch.cuda.is_available():
        raise RuntimeError("CET V2-A requires CUDA")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    result = run_arm(
        args.dataset,
        args.arm,
        config,
        p0,
        args.output_root,
        torch.device("cuda:0"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
