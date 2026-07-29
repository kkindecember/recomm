#!/usr/bin/env python3
"""TCDR S0: differentiability and bounded optimization correctness smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gcdh_p0 import (  # noqa: E402
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)
from experiment.phase4.tcdr_n1 import select_user_samples  # noqa: E402
from utils import generation_trie as gt  # noqa: E402


def differentiable_correlation(
    left: torch.Tensor, right: torch.Tensor, epsilon: float
) -> torch.Tensor:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    numerator = (left_centered * right_centered).sum()
    denominator = torch.sqrt(
        left_centered.square().sum()
        * right_centered.square().sum()
        + epsilon
    )
    return numerator / denominator


def differentiable_legal_path_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    trie,
    eos_token_id: int,
) -> torch.Tensor:
    prefix = [0]
    node_scores = []
    for depth, token_tensor in enumerate(labels):
        token = int(token_tensor.detach().item())
        if token == -100:
            break
        children = trie.get(prefix)
        if token not in children:
            raise ValueError(
                f"gold token {token} not in Trie children at prefix {prefix}"
            )
        child_logits = logits[depth, children].float()
        gold_index = children.index(token)
        if token != eos_token_id:
            node_scores.append(torch.log_softmax(child_logits, dim=-1)[gold_index])
        prefix.append(token)
    if not node_scores:
        raise ValueError("lexical path has no scored non-EOS node")
    return torch.stack(node_scores).mean()


def read_pairs(path: Path, count: int) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["pair_index"]) < count]
    selected.sort(key=lambda row: int(row["pair_index"]))
    if [int(row["pair_index"]) for row in selected] != list(range(count)):
        raise ValueError("TCDR S0 pair indices are incomplete")
    return selected


@torch.no_grad()
def encode_users(backbone, batch: dict, device: torch.device):
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    backbone.encoder.n_passages = input_ids.shape[1]
    flat_ids = input_ids.view(input_ids.shape[0], -1)
    flat_attention = attention.view(attention.shape[0], -1)
    hidden = backbone.encoder(
        input_ids=flat_ids,
        attention_mask=flat_attention,
        return_dict=True,
    )[0].detach()
    return hidden, flat_attention


def score_items(
    backbone,
    collator,
    items: list[str],
    item2lexid: dict[str, str],
    hidden: torch.Tensor,
    flat_attention: torch.Tensor,
    trie,
    eos_token_id: int,
) -> torch.Tensor:
    columns = []
    users = hidden.shape[0]
    for item in items:
        encoded = collator.encode_target_split([item2lexid[item]])
        labels = encoded["input_ids"]
        masks = encoded["attention_mask"].bool()
        labels = labels.masked_fill(~masks, -100).to(hidden.device)
        repeated_labels = labels.expand(users, -1)
        output = backbone(
            input_ids=None,
            attention_mask=flat_attention,
            encoder_outputs=(hidden,),
            labels=repeated_labels,
            return_dict=True,
        )
        user_scores = [
            differentiable_legal_path_score(
                output.logits[user_index],
                repeated_labels[user_index],
                trie,
                eos_token_id,
            )
            for user_index in range(users)
        ]
        columns.append(torch.stack(user_scores))
    return torch.stack(columns, dim=1)


def compute_objectives(
    backbone,
    collator,
    batch: dict,
    items: list[str],
    item_index: dict[str, int],
    pairs: list[dict],
    item2lexid: dict[str, str],
    hidden: torch.Tensor,
    flat_attention: torch.Tensor,
    trie,
    eos_token_id: int,
    epsilon: float,
):
    labels = batch["target_ids"].to(hidden.device)
    ce_output = backbone(
        input_ids=None,
        attention_mask=flat_attention,
        encoder_outputs=(hidden,),
        labels=labels,
        return_dict=True,
    )
    scores = score_items(
        backbone,
        collator,
        items,
        item2lexid,
        hidden,
        flat_attention,
        trie,
        eos_token_id,
    )
    close_correlations, far_correlations, pair_losses = [], [], []
    for row in pairs:
        close = differentiable_correlation(
            scores[:, item_index[row["near_left"]]],
            scores[:, item_index[row["near_right"]]],
            epsilon,
        )
        far = differentiable_correlation(
            scores[:, item_index[row["far_left"]]],
            scores[:, item_index[row["far_right"]]],
            epsilon,
        )
        close_correlations.append(close)
        far_correlations.append(far)
        pair_losses.append(torch.relu(close - far))
    close_tensor = torch.stack(close_correlations)
    far_tensor = torch.stack(far_correlations)
    tcdr_loss = torch.stack(pair_losses).mean()
    return ce_output.loss, tcdr_loss, scores, close_tensor, far_tensor


def audit_domain(
    dataset: str,
    config: dict,
    n1_config: dict,
    p0_config: dict,
    output_root: Path,
    device: torch.device,
) -> dict:
    prepared = prepare(dataset, p0_config, device)
    checkpoint = (
        ROOT / config["inputs"]["checkpoint_root"] / dataset / "C0" / "model.pt"
    )
    checkpoint_sha = sha256(checkpoint)
    prepared["model"].load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    backbone = prepared["model"].backbone
    backbone.eval()
    parameter_count_before = sum(parameter.numel() for parameter in backbone.parameters())
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    trainable_parameters = list(backbone.decoder.block[-1].parameters())
    for parameter in trainable_parameters:
        parameter.requires_grad_(True)

    train_users = read_users(
        ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt"
    )
    selection_config = dict(n1_config)
    selection_config["users_per_dataset"] = int(config["users_per_dataset"])
    samples = select_user_samples(
        prepared, train_users, selection_config, dataset
    )
    pairs = read_pairs(
        ROOT
        / config["inputs"]["n1_output"]
        / dataset
        / "pair_metrics.csv",
        int(config["pairs_per_dataset"]),
    )
    items = sorted(
        {
            row[key]
            for row in pairs
            for key in ("near_left", "near_right", "far_left", "far_right")
        }
    )
    if any(item not in prepared["item2lexid"] for item in items):
        raise ValueError("TCDR S0 item mapping failure")
    item_index = {item: index for index, item in enumerate(items)}
    batch = collate(prepared["collator"], samples)
    hidden, flat_attention = encode_users(backbone, batch, device)
    trie = gt.Trie(prepared["encoded_candidates"])
    eos = int(prepared["tokenizer"].eos_token_id)
    epsilon = float(config["training"]["correlation_epsilon"])

    initial_ce, initial_loss, initial_scores, initial_close, initial_far = (
        compute_objectives(
            backbone,
            prepared["collator"],
            batch,
            items,
            item_index,
            pairs,
            prepared["item2lexid"],
            hidden,
            flat_attention,
            trie,
            eos,
            epsilon,
        )
    )
    zero_identity = float(
        ((initial_ce + 0.0 * initial_loss) - initial_ce).detach().abs()
    )
    initial_parameter_values = [
        parameter.detach().clone() for parameter in trainable_parameters
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    step_rows = []
    gradient_norms = []
    for step in range(1, int(config["training"]["steps"]) + 1):
        if step == 1:
            ce, loss = initial_ce, initial_loss
        else:
            ce, loss, _, _, _ = compute_objectives(
                backbone,
                prepared["collator"],
                batch,
                items,
                item_index,
                pairs,
                prepared["item2lexid"],
                hidden,
                flat_attention,
                trie,
                eos,
                epsilon,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            float(config["training"]["gradient_clip_norm"]),
        )
        optimizer.step()
        gradient_norms.append(float(gradient_norm))
        step_rows.append(
            {
                "step": step,
                "lexical_ce_before_step": float(ce.detach()),
                "tcdr_loss_before_step": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
        )
        print(
            f"TCDR_S0 dataset={dataset} step={step} "
            f"loss={float(loss.detach()):.6f}",
            flush=True,
        )

    final_ce, final_loss, final_scores, final_close, final_far = compute_objectives(
        backbone,
        prepared["collator"],
        batch,
        items,
        item_index,
        pairs,
        prepared["item2lexid"],
        hidden,
        flat_attention,
        trie,
        eos,
        epsilon,
    )
    relative_decrease = float(
        (initial_loss.detach() - final_loss.detach())
        / initial_loss.detach().clamp_min(1e-12)
    )
    ce_relative_increase = float(
        (final_ce.detach() - initial_ce.detach())
        / initial_ce.detach().clamp_min(1e-12)
    )
    parameter_delta = max(
        float((parameter.detach() - before).abs().max())
        for parameter, before in zip(trainable_parameters, initial_parameter_values)
    )
    finite_values = torch.cat(
        (
            initial_scores.detach().flatten(),
            final_scores.detach().flatten(),
            initial_close.detach(),
            initial_far.detach(),
            final_close.detach(),
            final_far.detach(),
            torch.tensor(
                [
                    float(initial_ce.detach()),
                    float(initial_loss.detach()),
                    float(final_ce.detach()),
                    float(final_loss.detach()),
                    *gradient_norms,
                ],
                device=device,
            ),
        )
    )
    finite_rate = float(torch.isfinite(finite_values).float().mean())
    parameter_count_after = sum(parameter.numel() for parameter in backbone.parameters())
    tuned_path = output_root / dataset / "decoder_last_block_s0.pt"
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.decoder.block[-1].state_dict(), tuned_path)

    gates = config["correctness_gates"]
    checks = {
        "zero_lambda_identity": zero_identity
        <= float(gates["zero_lambda_identity_tolerance"]),
        "initial_tcdr_loss_positive": float(initial_loss.detach()) > 0,
        "initial_gradient_norm_positive": gradient_norms[0] > 0,
        "finite_rate": finite_rate == float(gates["finite_rate"]),
        "tcdr_relative_decrease": relative_decrease
        >= float(gates["tcdr_relative_decrease_min"]),
        "lexical_ce_relative_increase": ce_relative_increase
        <= float(gates["lexical_ce_relative_increase_max"]),
        "parameter_delta_positive": parameter_delta > 0,
        "checkpoint_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "exact_users": len(samples) == int(gates["exact_users"]),
        "exact_pairs": len(pairs) == int(gates["exact_pairs"]),
        "no_new_inference_parameters": parameter_count_before == parameter_count_after,
    }
    integrity = {
        "users": len(samples),
        "pairs": len(pairs),
        "unique_items": len(items),
        "finite_rate": finite_rate,
        "checkpoint_sha_unchanged": checkpoint_sha == sha256(checkpoint),
        "parameter_count_before": parameter_count_before,
        "parameter_count_after": parameter_count_after,
        "validation_test_read": False,
        "sports_read": False,
        "tuned_block_sha256": sha256(tuned_path),
    }
    result = {
        "initial_lexical_ce": float(initial_ce.detach()),
        "final_lexical_ce": float(final_ce.detach()),
        "lexical_ce_relative_increase": ce_relative_increase,
        "initial_tcdr_loss": float(initial_loss.detach()),
        "final_tcdr_loss": float(final_loss.detach()),
        "tcdr_relative_decrease": relative_decrease,
        "zero_lambda_identity_difference": zero_identity,
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "initial_gradient_norm": gradient_norms[0],
        "parameter_delta_max_abs": parameter_delta,
        "steps": step_rows,
        "checks": checks,
        "correctness_pass": all(checks.values()),
        "integrity": integrity,
    }
    write_json(output_root / dataset / "summary.json", result)
    del prepared
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if sha256(Path(__file__)) != config["integrity"]["code_sha256"]:
        raise ValueError("TCDR S0 code SHA mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("TCDR S0 requires CUDA")
    p0_config = json.loads((ROOT / config["inputs"]["p0_config"]).read_text())
    n1_config = json.loads((ROOT / config["inputs"]["n1_config"]).read_text())
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    device = torch.device("cuda:0")
    datasets = {
        dataset: audit_domain(
            dataset, config, n1_config, p0_config, args.output_root, device
        )
        for dataset in config["datasets"]
    }
    integrity_valid = all(
        row["integrity"]["checkpoint_sha_unchanged"]
        and row["integrity"]["finite_rate"] == 1.0
        and not row["integrity"]["validation_test_read"]
        and not row["integrity"]["sports_read"]
        for row in datasets.values()
    )
    correctness_pass = all(row["correctness_pass"] for row in datasets.values())
    decision = (
        "EXECUTION_INVALID"
        if not integrity_valid
        else "TCDR_S0_CORRECTNESS_PASS"
        if correctness_pass
        else "STOP_TCDR_S0_CORRECTNESS_FAILED"
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "datasets": datasets,
        "integrity_valid": integrity_valid,
        "validation_test_read": False,
        "sports_read": False,
    }
    write_json(args.output_root / "summary.json", summary)
    write_json(
        args.output_root / "status.json",
        {"experiment_id": config["experiment_id"], "status": "completed"},
    )
    (args.output_root / "decision.md").write_text(
        "# TCDR-S0 Decision\n\n"
        f"- Fixed decision: **`{decision}`**\n"
        f"- Integrity valid: `{str(integrity_valid).lower()}`\n"
        "- Validation/test/Sports read: `false`\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
