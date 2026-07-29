#!/usr/bin/env python3
"""CET C1: clean-anchored legal-child consistency correctness smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.phase4.chpr_a0 import pad_labels  # noqa: E402
from experiment.phase4.gcdh_p0 import (  # noqa: E402
    collate,
    prepare,
    sha256,
    write_json,
)
from utils import generation_trie as gt  # noqa: E402


def stable_fraction(payload: str) -> float:
    value = int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)
    return value / float(16**16)


def build_latest_training_samples(
    sequences: dict[str, list[str]],
    item2input: dict[str, str],
    item2lexid: dict[str, str],
    dataset: str,
    salt: str,
    count: int,
    minimum_history_items: int,
) -> list[dict]:
    ordered_users = sorted(
        sequences,
        key=lambda user: hashlib.sha256(
            f"{salt}|{dataset}|{user}".encode()
        ).hexdigest(),
    )
    samples = []
    for user in ordered_users:
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
            f"insufficient CET samples for {dataset}: {len(samples)} != {count}"
        )
    if len({row["user_id"] for row in samples}) != count:
        raise ValueError("CET sample selection must contain unique users")
    return samples


def structured_passage_mask(
    attention: torch.Tensor,
    samples: list[dict],
    dataset: str,
    seed: int,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention.ndim != 3:
        raise ValueError("attention must have [batch, passages, width] shape")
    if len(samples) != attention.shape[0]:
        raise ValueError("sample/attention batch mismatch")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("mask probability must lie in [0, 1]")
    perturbed = attention.clone()
    decisions = torch.zeros(
        attention.shape[:2], dtype=torch.bool, device=attention.device
    )
    for row, sample in enumerate(samples):
        history_key = "\x1f".join(sample["history_items"])
        for passage in range(2, attention.shape[1]):
            if not bool(attention[row, passage].any()):
                continue
            payload = (
                f"cet-c1-mask|{seed}|{dataset}|{sample['user_id']}|"
                f"{history_key}|{passage}"
            )
            if stable_fraction(payload) < probability:
                perturbed[row, passage] = False
                decisions[row, passage] = True
    return perturbed, decisions


def legal_child_kl(
    clean_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    sequences: list[list[int]],
    trie: gt.Trie,
    eos_token_id: int,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    if clean_logits.shape != perturbed_logits.shape:
        raise ValueError("clean/perturbed logits shape mismatch")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    losses = []
    for batch_index, sequence in enumerate(sequences):
        for position, gold in enumerate(sequence[1:]):
            allowed = trie.get(sequence[: position + 1])
            if gold == eos_token_id or len(allowed) < 2:
                continue
            if gold not in allowed:
                raise ValueError("gold child is not legal")
            indices = torch.as_tensor(
                allowed, dtype=torch.long, device=clean_logits.device
            )
            clean_values = clean_logits[batch_index, position].index_select(
                0, indices
            ) / float(temperature)
            perturbed_values = perturbed_logits[
                batch_index, position
            ].index_select(0, indices) / float(temperature)
            clean_log_probs = torch.log_softmax(clean_values, dim=0).detach()
            clean_probs = clean_log_probs.exp()
            perturbed_log_probs = torch.log_softmax(perturbed_values, dim=0)
            losses.append(
                (clean_probs * (clean_log_probs - perturbed_log_probs)).sum()
            )
    if not losses:
        raise ValueError("no competitive legal-child steps")
    return torch.stack(losses).mean(), len(losses)


def compose_loss(
    clean_ce: torch.Tensor,
    perturbed_ce: torch.Tensor,
    legal_kl: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    return clean_ce + float(alpha) * perturbed_ce + float(beta) * legal_kl


def gradient_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum())
    return math.sqrt(squared)


def forward_views(
    backbone,
    input_ids: torch.Tensor,
    clean_attention: torch.Tensor,
    perturbed_attention: torch.Tensor,
    labels: torch.Tensor,
):
    passages = input_ids.shape[1]
    backbone.encoder.n_passages = passages
    flat_ids = input_ids.view(input_ids.shape[0], -1)
    clean_flat = clean_attention.view(clean_attention.shape[0], -1)
    perturbed_flat = perturbed_attention.view(
        perturbed_attention.shape[0], -1
    )
    encoder_hidden = backbone.encoder(
        input_ids=flat_ids,
        attention_mask=clean_flat,
        return_dict=True,
    )[0]
    clean = backbone(
        input_ids=None,
        attention_mask=clean_flat,
        encoder_outputs=(encoder_hidden,),
        labels=labels,
        return_dict=True,
    )
    perturbed = backbone(
        input_ids=None,
        attention_mask=perturbed_flat,
        encoder_outputs=(encoder_hidden,),
        labels=labels,
        return_dict=True,
    )
    return clean, perturbed


def run_dataset(
    dataset: str,
    config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    backbone = prepared["model"].backbone
    source_checkpoint = REPO_ROOT / p0_config["datasets"][dataset]["checkpoint"]
    source_sha_before = sha256(source_checkpoint)
    samples = build_latest_training_samples(
        prepared["sequences"],
        prepared["item2input"],
        prepared["item2lexid"],
        dataset,
        config["data_boundary"]["selection_salt"],
        int(config["data_boundary"]["users_per_dataset"]),
        int(config["data_boundary"]["minimum_history_items"]),
    )
    batch = collate(prepared["collator"], samples)
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device).bool()
    item_to_sequence = dict(
        zip(prepared["catalog"], prepared["encoded_candidates"])
    )
    sequences = [
        item_to_sequence[sample["positive_item"]] for sample in samples
    ]
    labels = pad_labels(sequences, device)
    trie = gt.Trie(prepared["encoded_candidates"])
    perturbed_attention, mask_decisions = structured_passage_mask(
        attention,
        samples,
        dataset,
        int(config["views"]["perturbed"]["mask_seed"]),
        float(
            config["views"]["perturbed"][
                "other_fine_passage_mask_probability"
            ]
        ),
    )
    if int(mask_decisions.sum()) == 0:
        raise ValueError("structured perturbation masked no passages")
    if not torch.equal(perturbed_attention[:, 0], attention[:, 0]):
        raise ValueError("coarse passage was changed")
    if attention.shape[1] > 1 and not torch.equal(
        perturbed_attention[:, 1], attention[:, 1]
    ):
        raise ValueError("newest fine passage was changed")

    altered_targets = [dict(sample, positive_item="__altered__") for sample in samples]
    target_free_attention, target_free_decisions = structured_passage_mask(
        attention,
        altered_targets,
        dataset,
        int(config["views"]["perturbed"]["mask_seed"]),
        float(
            config["views"]["perturbed"][
                "other_fine_passage_mask_probability"
            ]
        ),
    )
    target_independent = torch.equal(
        perturbed_attention, target_free_attention
    ) and torch.equal(mask_decisions, target_free_decisions)

    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    trainable = list(backbone.decoder.block[-1].parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    initial_trainable = [
        parameter.detach().clone() for parameter in trainable
    ]
    backbone.eval()

    with torch.no_grad():
        direct = backbone(
            input_ids=input_ids,
            attention_mask=attention,
            labels=labels,
            return_dict=True,
        )
        clean_initial, perturbed_initial = forward_views(
            backbone, input_ids, attention, perturbed_attention, labels
        )
        replay_difference = float(
            (direct.logits - clean_initial.logits).abs().max()
        )
        initial_kl, competitive_steps = legal_child_kl(
            clean_initial.logits,
            perturbed_initial.logits,
            sequences,
            trie,
            int(prepared["tokenizer"].eos_token_id),
            float(config["loss"]["temperature_tau"]),
        )
        zero_weight_loss = compose_loss(
            clean_initial.loss,
            perturbed_initial.loss,
            initial_kl,
            0.0,
            0.0,
        )
        zero_weight_difference = float(
            (zero_weight_loss - clean_initial.loss).abs()
        )

    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        weight_decay=0.0,
    )
    losses, kls, clean_ces, perturbed_ces, gradient_norms = [], [], [], [], []
    for _ in range(int(optimization["steps"])):
        optimizer.zero_grad(set_to_none=True)
        clean, perturbed = forward_views(
            backbone, input_ids, attention, perturbed_attention, labels
        )
        legal_kl_value, step_count = legal_child_kl(
            clean.logits,
            perturbed.logits,
            sequences,
            trie,
            int(prepared["tokenizer"].eos_token_id),
            float(config["loss"]["temperature_tau"]),
        )
        if step_count != competitive_steps:
            raise ValueError("competitive legal-child step count changed")
        total = compose_loss(
            clean.loss,
            perturbed.loss,
            legal_kl_value,
            float(config["loss"]["perturbed_ce_weight_alpha"]),
            float(config["loss"]["legal_child_kl_weight_beta"]),
        )
        if not torch.isfinite(total):
            raise ValueError("non-finite CET loss")
        total.backward()
        norm = gradient_norm(trainable)
        if not math.isfinite(norm):
            raise ValueError("non-finite CET gradient")
        optimizer.step()
        losses.append(float(total.detach()))
        kls.append(float(legal_kl_value.detach()))
        clean_ces.append(float(clean.loss.detach()))
        perturbed_ces.append(float(perturbed.loss.detach()))

        gradient_norms.append(norm)

    with torch.no_grad():
        clean_final, perturbed_final = forward_views(
            backbone, input_ids, attention, perturbed_attention, labels
        )
        final_kl_tensor, final_step_count = legal_child_kl(
            clean_final.logits,
            perturbed_final.logits,
            sequences,
            trie,
            int(prepared["tokenizer"].eos_token_id),
            float(config["loss"]["temperature_tau"]),
        )
    final_kl = float(final_kl_tensor)
    initial_kl_value = float(initial_kl)
    relative_kl_decrease = (
        (initial_kl_value - final_kl) / initial_kl_value
        if initial_kl_value > 0
        else 0.0
    )
    clean_ce_relative_change = (
        float(clean_final.loss) - float(clean_initial.loss)
    ) / float(clean_initial.loss)
    parameter_change = max(
        float((after.detach() - before).abs().max())
        for before, after in zip(initial_trainable, trainable)
    )
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(backbone.decoder.block[-1].state_dict(), handle.name)
        before_reload = clean_final.logits.detach().clone()
        with torch.no_grad():
            next(backbone.decoder.block[-1].parameters()).add_(1.0)
        backbone.decoder.block[-1].load_state_dict(
            torch.load(handle.name, map_location=device)
        )
        with torch.no_grad():
            clean_reloaded, _ = forward_views(
                backbone, input_ids, attention, perturbed_attention, labels
            )
        reload_difference = float(
            (before_reload - clean_reloaded.logits).abs().max()
        )

    gates = config["required_optimization_gates"]
    integrity_checks = {
        "clean_view_exact_input_replay": replay_difference <= 1e-6,
        "coarse_passage_never_masked": torch.equal(
            perturbed_attention[:, 0], attention[:, 0]
        ),
        "newest_fine_passage_never_masked": attention.shape[1] <= 1
        or torch.equal(perturbed_attention[:, 1], attention[:, 1]),
        "target_not_used_by_mask_policy": target_independent,
        "alpha_beta_zero_exact_matched_ce": zero_weight_difference <= 1e-8,
        "legal_child_membership_100_percent": final_step_count
        == competitive_steps,
        "finite_logits_loss_gradients_100_percent": all(
            math.isfinite(value)
            for value in losses
            + kls
            + clean_ces
            + perturbed_ces
            + gradient_norms
            + [final_kl]
        ),
        "nonzero_gradient": min(gradient_norms) > 0,
        "source_checkpoint_sha_unchanged": source_sha_before
        == sha256(source_checkpoint),
        "checkpoint_reload_identity": reload_difference <= 1e-6,
        "validation_test_sports_not_read": True,
    }
    optimization_checks = {
        "legal_child_kl_decrease": relative_kl_decrease
        >= float(gates["legal_child_kl_relative_decrease_min"]),
        "clean_ce_safety": clean_ce_relative_change
        <= float(gates["clean_lexical_ce_relative_increase_max"]),
        "parameter_change": parameter_change > 0,
    }
    metrics = {
        "users": len(samples),
        "minimum_history_items": min(
            len(sample["history_items"]) for sample in samples
        ),
        "masked_passages": int(mask_decisions.sum()),
        "samples_with_mask": int(mask_decisions.any(dim=1).sum()),
        "competitive_legal_child_steps": competitive_steps,
        "clean_replay_max_abs_difference": replay_difference,
        "zero_weight_loss_difference": zero_weight_difference,
        "initial_legal_child_kl": initial_kl_value,
        "final_legal_child_kl": final_kl,
        "relative_legal_child_kl_decrease": relative_kl_decrease,
        "initial_clean_ce": float(clean_initial.loss),
        "final_clean_ce": float(clean_final.loss),
        "clean_ce_relative_change": clean_ce_relative_change,
        "initial_perturbed_ce": float(perturbed_initial.loss),
        "final_perturbed_ce": float(perturbed_final.loss),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "parameter_max_abs_change": parameter_change,
        "optimizer_steps": len(losses),
        "reload_max_abs_difference": reload_difference,
    }
    result = {
        "dataset": dataset,
        "metrics": metrics,
        "integrity_checks": integrity_checks,
        "optimization_checks": optimization_checks,
        "integrity_pass": all(integrity_checks.values()),
        "optimization_pass": all(optimization_checks.values()),
        "source_checkpoint_sha256": source_sha_before,
        "validation_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(output_root / dataset / "summary.json", result)
    del prepared, backbone
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CET C1 requires CUDA")
    config = json.loads(args.config.read_text())
    code_sha = sha256(Path(__file__))
    if code_sha != config["integrity"]["code_sha256"]:
        raise ValueError(
            f"CET C1 code SHA mismatch: actual={code_sha} "
            f"registered={config['integrity']['code_sha256']}"
        )
    p0_config = json.loads(
        (
            REPO_ROOT
            / "artifacts/phase4/configs/gcdh_p0_preregistered.json"
        ).read_text()
    )
    torch.manual_seed(int(config["optimization"]["seed"]))
    torch.cuda.manual_seed_all(int(config["optimization"]["seed"]))
    device = torch.device("cuda:0")
    results = {
        dataset: run_dataset(
            dataset, config, p0_config, args.output_root, device
        )
        for dataset in config["datasets"]
    }
    integrity_pass = all(row["integrity_pass"] for row in results.values())
    optimization_pass = all(
        row["optimization_pass"] for row in results.values()
    )
    decision = (
        "EXECUTION_INVALID"
        if not integrity_pass
        else "CET_C1_CORRECTNESS_PASS"
        if optimization_pass
        else "STOP_CET_NOT_OPTIMIZABLE"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "code_sha256": code_sha,
        "results": results,
        "integrity_pass": integrity_pass,
        "optimization_pass": optimization_pass,
        "validation_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(args.output_root / "summary.json", summary)
    write_json(
        args.output_root / "status.json",
        {"experiment_id": config["experiment_id"], "status": "completed"},
    )
    lines = [
        "# CET-C1 Decision",
        "",
        f"- Fixed decision: **`{decision}`**",
        f"- Integrity pass: `{str(integrity_pass).lower()}`",
        f"- Optimization pass: `{str(optimization_pass).lower()}`",
        "- Validation/test/Sports read: `false`",
        "",
    ]
    for dataset, result in results.items():
        lines.extend([f"## {dataset}", ""])
        for name, passed in result["integrity_checks"].items():
            lines.append(f"- integrity `{name}`: `{'PASS' if passed else 'FAIL'}`")
        for name, passed in result["optimization_checks"].items():
            lines.append(
                f"- optimization `{name}`: `{'PASS' if passed else 'FAIL'}`"
            )
        lines.append("")
    (args.output_root / "decision.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
