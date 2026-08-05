#!/usr/bin/env python3
"""Post-hoc diagnostics for the Phase-9 GRAM CF0 checkpoint.

The script is intentionally read-only with respect to the trained checkpoint. It
measures the independent item-head ranking quality on validation, performs
paired teacher-forced fusion ablations, and records per-loss gradient flow on a
small deterministic validation prefix.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Config


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAM_ROOT = REPO_ROOT / "GRAM"
GRAM_SRC = GRAM_ROOT / "src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))

# Preserve the import order used by GRAM's main entry point. The upstream
# package initializers have a data/utils cycle that is resolved by runner first.
import runner as _runner_import_guard  # noqa: E402,F401
from data import TestDatasetGRAM  # noqa: E402
from model import create_model  # noqa: E402
from processor import CollatorGRAM  # noqa: E402
from utils import set_seed  # noqa: E402
from cf0_diagnostic_metrics import item_metrics_from_ranks, rank_from_logits  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "artifacts/phase9/cf0_b_toys_p1/gram_logs/Toys/0_20260803_2333/config.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts/phase9/cf0_b_toys_p1/gram_logs/Toys/0_20260803_2333/"
    "id_0_rec_5/model_rec_phase_1_epoch_5.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase9/cf0_b_toys_p1_diagnostics"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--item-batch-size", type=int, default=256)
    parser.add_argument(
        "--item-samples",
        type=int,
        default=0,
        help="Validation prefix for item-head evaluation; 0 means the full split.",
    )
    parser.add_argument("--fusion-samples", type=int, default=512)
    parser.add_argument("--gradient-batches", type=int, default=4)
    return parser.parse_args()


def load_run_args(config_path):
    with config_path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    values["data_path"] = str(GRAM_ROOT / "rec_datasets")
    values["prompt_file"] = str(GRAM_ROOT / "prompt.txt")
    values["rank"] = 0
    values["distributed"] = 0
    values["debug_test_100"] = 0
    values["debug_test_small_set"] = 0
    values["eval_batch_size"] = 1
    return SimpleNamespace(**values)


def configure_model(run_args, device, checkpoint):
    config = T5Config.from_pretrained(run_args.backbone, local_files_only=True)
    config.max_seq_len = run_args.item_prompt_max_len
    config.max_item_num = run_args.max_his
    config.use_position_embedding = run_args.use_position_embedding
    config.sample_num = run_args.sample_num
    config.cf0_arm = run_args.cf0_arm
    config.cf0_enabled = run_args.cf0_arm in {"B", "C"}
    config.cf0_num_layers = run_args.cf0_num_layers
    config.cf0_num_heads = run_args.cf0_num_heads
    config.cf0_dropout = run_args.cf0_dropout
    config.cf0_loss_weight = run_args.cf0_loss_weight
    config.cf0_injection_scale = run_args.cf0_injection_scale
    config.cf0_joint_score_weight = run_args.cf0_joint_score_weight
    index_path = (
        Path(run_args.data_path)
        / run_args.datasets
        / f"item_generative_indexing_{run_args.hierarchical_id_type}.txt"
    )
    with index_path.open(encoding="utf-8") as handle:
        config.cf0_num_items = sum(1 for line in handle if line.strip())

    model = create_model("gram", config=config)
    state_dict = torch.load(checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    model.to(device).eval()
    return model


def build_validation_loader(run_args, tokenizer, batch_size, item_batch_size):
    dataset = TestDatasetGRAM(
        args=run_args,
        dataset=run_args.datasets,
        task=run_args.tasks.split(",")[0],
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        mode="validation",
    )
    collator = CollatorGRAM(tokenizer, args=run_args, mode="valid")
    text_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
    )
    history_loader = DataLoader(
        dataset,
        batch_size=item_batch_size,
        collate_fn=collate_history_only,
        shuffle=False,
        num_workers=0,
    )
    return dataset, text_loader, history_loader


def collate_history_only(batch):
    max_length = max(len(row["history_item_ids"]) for row in batch)
    history_item_ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    for row_index, row in enumerate(batch):
        values = row["history_item_ids"]
        history_item_ids[row_index, : len(values)] = torch.tensor(values)
    return {
        "user_ids": [row["user_id"] for row in batch],
        "history_item_ids": history_item_ids,
        "history_item_mask": history_item_ids.ne(0),
        "target_item_ids": torch.tensor(
            [row["target_item_id"] for row in batch], dtype=torch.long
        ),
    }


def encode_cf0_user_state(model, batch):
    input_ids = batch["item_text_ids"]
    attention_mask = batch["item_text_masks"]
    model.encoder.n_passages = input_ids.size(1)
    model.encoder.set_cf0_context(
        batch["history_item_ids"], batch["history_item_mask"]
    )
    model.encoder(
        input_ids=input_ids.reshape(input_ids.size(0), -1),
        attention_mask=attention_mask.reshape(attention_mask.size(0), -1),
        return_dict=True,
    )
    if model.encoder.last_cf0_user_state is None:
        raise RuntimeError("CF0 user state was not produced")
    return model.encoder.last_cf0_user_state


def encode_cf0_history_only(model, history_item_ids, history_item_mask):
    """Reproduce arm-B's collaborative user state without the text encoder."""
    chronological_ids = model.encoder._reverse_valid_prefix(
        history_item_ids, history_item_mask
    )
    chronological_mask = model.encoder._reverse_valid_prefix(
        history_item_mask, history_item_mask
    )
    length = chronological_ids.size(1)
    positions = torch.arange(length, device=history_item_ids.device)
    collaborative_input = model.encoder.cf0_item_embedding(chronological_ids)
    collaborative_input = collaborative_input + model.encoder.cf0_position_embedding(
        positions
    ).unsqueeze(0)
    causal_mask = torch.ones(
        length, length, dtype=torch.bool, device=history_item_ids.device
    ).triu(1)
    states = model.encoder.cf0_transformer(
        collaborative_input,
        mask=causal_mask,
        src_key_padding_mask=~chronological_mask,
    )
    states = model.encoder.cf0_sequence_norm(states)
    lengths = chronological_mask.long().sum(dim=1).clamp_min(1)
    return states[torch.arange(states.size(0), device=states.device), lengths - 1]


def popularity_counts(dataset):
    counts = Counter()
    for items in dataset.user_seq_dict.values():
        for item in items[:-2]:
            counts[dataset.item2cfid[item]] += 1
    return counts


def item_head_evaluation(model, dataset, loader, device, rank_path, sample_limit=0):
    counts = popularity_counts(dataset)
    records = []
    with rank_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "target_item_id", "history_length", "train_frequency", "item_rank"])
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                if sample_limit and len(records) >= sample_limit:
                    break
                if sample_limit and len(records) + len(batch["user_ids"]) > sample_limit:
                    remaining = sample_limit - len(records)
                    batch = {
                        key: value[:remaining] if torch.is_tensor(value) else value[:remaining]
                        for key, value in batch.items()
                    }
                user_state = encode_cf0_history_only(
                    model, batch["history_item_ids"], batch["history_item_mask"]
                )
                logits = F.linear(
                    user_state, model.encoder.cf0_item_embedding.weight
                ) / (model.config.d_model**0.5)
                ranks = rank_from_logits(logits, batch["target_item_ids"])
                lengths = batch["history_item_mask"].sum(dim=1)
                for user, target, length, rank in zip(
                    batch["user_ids"],
                    batch["target_item_ids"].tolist(),
                    lengths.tolist(),
                    ranks.tolist(),
                ):
                    frequency = counts[target]
                    records.append((rank, length, frequency))
                    writer.writerow([user, target, length, frequency, rank])

    ranks = [record[0] for record in records]
    summary = {"overall": item_metrics_from_ranks(ranks)}
    history_groups = {
        "1-5": [r for r, length, _ in records if 1 <= length <= 5],
        "6-10": [r for r, length, _ in records if 6 <= length <= 10],
        "11-20": [r for r, length, _ in records if 11 <= length <= 20],
    }
    summary["by_history_length"] = {
        name: item_metrics_from_ranks(group)
        for name, group in history_groups.items()
        if group
    }
    positive_frequencies = sorted(f for _, _, f in records)
    q1 = positive_frequencies[len(positive_frequencies) // 4]
    q3 = positive_frequencies[(3 * len(positive_frequencies)) // 4]
    popularity_groups = {
        "tail": [r for r, _, f in records if f <= q1],
        "middle": [r for r, _, f in records if q1 < f < q3],
        "head": [r for r, _, f in records if f >= q3],
    }
    summary["popularity_frequency_boundaries"] = {"q1": q1, "q3": q3}
    summary["by_target_popularity"] = {
        name: item_metrics_from_ranks(group)
        for name, group in popularity_groups.items()
        if group
    }
    return summary


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@contextmanager
def fusion_condition(model, condition):
    old_config_enabled = model.config.cf0_enabled
    old_encoder_enabled = model.encoder.cf0_enabled
    old_scale = model.encoder.cf0_injection_scale
    try:
        if condition == "full":
            pass
        elif condition == "zero_injection_keep_norm":
            model.encoder.cf0_injection_scale = 0.0
        elif condition == "bypass_cf0_path":
            model.config.cf0_enabled = False
            model.encoder.cf0_enabled = False
            model.encoder.set_cf0_context(None, None)
        else:
            raise ValueError(condition)
        yield
    finally:
        model.config.cf0_enabled = old_config_enabled
        model.encoder.cf0_enabled = old_encoder_enabled
        model.encoder.cf0_injection_scale = old_scale


def per_example_generation_nll(model, batch, condition):
    with fusion_condition(model, condition):
        outputs = model(
            input_ids=batch["item_text_ids"],
            attention_mask=batch["item_text_masks"],
            history_item_ids=batch["history_item_ids"],
            history_item_mask=batch["history_item_mask"],
            labels=batch["target_ids"],
            return_dict=True,
        )
    labels = batch["target_ids"]
    token_losses = F.cross_entropy(
        outputs.logits.transpose(1, 2), labels, reduction="none", ignore_index=-100
    )
    valid = labels.ne(-100)
    return (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def paired_summary(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "std": float(tensor.std(unbiased=True)) if tensor.numel() > 1 else 0.0,
        "positive_fraction": float((tensor > 0).double().mean()),
    }


def fusion_ablation(model, loader, device, sample_limit):
    conditions = ("full", "zero_injection_keep_norm", "bypass_cf0_path")
    losses = defaultdict(list)
    seen = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            remaining = sample_limit - seen
            if remaining <= 0:
                break
            if batch["target_ids"].size(0) > remaining:
                batch = {
                    key: value[:remaining] if torch.is_tensor(value) else value[:remaining]
                    for key, value in batch.items()
                }
            for condition in conditions:
                losses[condition].extend(
                    per_example_generation_nll(model, batch, condition).cpu().tolist()
                )
            seen += batch["target_ids"].size(0)

    result = {condition: paired_summary(losses[condition]) for condition in conditions}
    result["paired_deltas"] = {
        "full_minus_zero_injection": paired_summary(
            [a - b for a, b in zip(losses["full"], losses["zero_injection_keep_norm"])]
        ),
        "zero_injection_minus_bypass": paired_summary(
            [a - b for a, b in zip(losses["zero_injection_keep_norm"], losses["bypass_cf0_path"])]
        ),
        "full_minus_bypass": paired_summary(
            [a - b for a, b in zip(losses["full"], losses["bypass_cf0_path"])]
        ),
    }
    return result


def set_joint_stage_trainable(model, top_layers):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.encoder.named_parameters():
        if name.startswith("cf0_"):
            parameter.requires_grad = True
    for block in model.encoder.encoder.block[-top_layers:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for block in model.decoder.block[-top_layers:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.decoder.final_layer_norm.parameters():
        parameter.requires_grad = True
    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True


def parameter_groups(model, top_layers):
    named = dict(model.named_parameters())
    definitions = {
        "cf0_item_embedding": lambda n: n.startswith("encoder.cf0_item_embedding"),
        "cf0_position_embedding": lambda n: n.startswith("encoder.cf0_position_embedding"),
        "cf0_transformer": lambda n: n.startswith("encoder.cf0_transformer"),
        "cf0_sequence_norm": lambda n: n.startswith("encoder.cf0_sequence_norm"),
        "cf0_gate": lambda n: n.startswith("encoder.cf0_gate"),
        "cf0_token_norm": lambda n: n.startswith("encoder.cf0_token_norm"),
        "encoder_top": lambda n: any(
            n.startswith(f"encoder.encoder.block.{idx}.")
            for idx in range(len(model.encoder.encoder.block) - top_layers, len(model.encoder.encoder.block))
        ),
        "decoder_top": lambda n: any(
            n.startswith(f"decoder.block.{idx}.")
            for idx in range(len(model.decoder.block) - top_layers, len(model.decoder.block))
        ),
        "decoder_final_norm": lambda n: n.startswith("decoder.final_layer_norm"),
        # T5 ties lm_head.weight to shared.weight. named_parameters() exposes the
        # shared tensor under the first name only, while the training code
        # unfreezes it through the lm_head alias.
        "shared_embedding_lm_head": lambda n: n == "shared.weight" or n.startswith("lm_head"),
    }
    return {
        group: [(name, parameter) for name, parameter in named.items() if matcher(name) and parameter.requires_grad]
        for group, matcher in definitions.items()
    }


def group_gradient_stats(grads, grouped, flat_parameters):
    by_id = {id(parameter): grad for parameter, grad in zip(flat_parameters, grads)}
    output = {}
    for group, members in grouped.items():
        selected = [by_id[id(parameter)] for _, parameter in members if by_id[id(parameter)] is not None]
        squared = sum(float((grad.detach().float() ** 2).sum()) for grad in selected)
        output[group] = {
            "parameter_tensors": len(members),
            "gradient_tensors": len(selected),
            "l2_norm": math.sqrt(squared),
        }
    return output


def gradient_diagnostics(model, loader, device, batch_limit, loss_weight, top_layers):
    model.train()
    set_joint_stage_trainable(model, top_layers)
    grouped = parameter_groups(model, top_layers)
    flat_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    aggregate = defaultdict(lambda: defaultdict(float))
    conflict_dot = defaultdict(float)
    conflict_gen_sq = defaultdict(float)
    conflict_item_sq = defaultdict(float)
    batches = 0

    for raw_batch in loader:
        if batches >= batch_limit:
            break
        batch = move_batch(raw_batch, device)
        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["item_text_ids"],
            attention_mask=batch["item_text_masks"],
            history_item_ids=batch["history_item_ids"],
            history_item_mask=batch["history_item_mask"],
            labels=batch["target_ids"],
            return_dict=True,
        )
        generation_loss = F.cross_entropy(
            outputs.logits.transpose(1, 2), batch["target_ids"], ignore_index=-100
        )
        item_logits = model.encoder.score_all_items()
        item_loss = F.cross_entropy(item_logits, batch["target_item_ids"])
        gen_grads = torch.autograd.grad(
            generation_loss, flat_parameters, retain_graph=True, allow_unused=True
        )
        item_grads = torch.autograd.grad(
            loss_weight * item_loss, flat_parameters, retain_graph=False, allow_unused=True
        )
        gen_stats = group_gradient_stats(gen_grads, grouped, flat_parameters)
        item_stats = group_gradient_stats(item_grads, grouped, flat_parameters)
        for group in grouped:
            aggregate[group]["generation_l2_sum"] += gen_stats[group]["l2_norm"]
            aggregate[group]["weighted_item_l2_sum"] += item_stats[group]["l2_norm"]
            member_ids = {id(parameter) for _, parameter in grouped[group]}
            for parameter, gen_grad, item_grad in zip(flat_parameters, gen_grads, item_grads):
                if id(parameter) not in member_ids or gen_grad is None or item_grad is None:
                    continue
                gen_flat = gen_grad.detach().float()
                item_flat = item_grad.detach().float()
                conflict_dot[group] += float((gen_flat * item_flat).sum())
                conflict_gen_sq[group] += float((gen_flat**2).sum())
                conflict_item_sq[group] += float((item_flat**2).sum())
        aggregate["losses"]["generation_sum"] += float(generation_loss.detach())
        aggregate["losses"]["item_sum"] += float(item_loss.detach())
        batches += 1

    result = {
        "batches": batches,
        "loss_weight": loss_weight,
        "mean_generation_loss": aggregate["losses"]["generation_sum"] / batches,
        "mean_item_loss": aggregate["losses"]["item_sum"] / batches,
        "groups": {},
    }
    for group in grouped:
        denom = math.sqrt(conflict_gen_sq[group] * conflict_item_sq[group])
        result["groups"][group] = {
            "mean_generation_gradient_l2": aggregate[group]["generation_l2_sum"] / batches,
            "mean_weighted_item_gradient_l2": aggregate[group]["weighted_item_l2_sum"] / batches,
            "generation_item_gradient_cosine": conflict_dot[group] / denom if denom else None,
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in grouped[group]),
        }
    model.eval()
    return result


def main():
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    run_args = load_run_args(cli.config)
    set_seed(run_args.seed)
    device = torch.device(cli.device)
    tokenizer = AutoTokenizer.from_pretrained(run_args.backbone, local_files_only=True)
    model = configure_model(run_args, device, cli.checkpoint)
    dataset, text_loader, history_loader = build_validation_loader(
        run_args, tokenizer, cli.batch_size, cli.item_batch_size
    )

    result = {
        "experiment_id": "GRAM_PHASE9_CF0_B_TOYS_P1_DIAGNOSTICS_V1",
        "source_checkpoint": str(cli.checkpoint),
        "dataset": run_args.datasets,
        "split": "validation",
        "sample_count": len(dataset),
        "test_read": False,
        "sports_read": False,
        "item_head": item_head_evaluation(
            model,
            dataset,
            history_loader,
            device,
            cli.output_dir / "item_head_ranks.tsv",
            cli.item_samples,
        ),
        "fusion_ablation": fusion_ablation(
            model, text_loader, device, cli.fusion_samples
        ),
        "gradient_diagnostics": gradient_diagnostics(
            model,
            text_loader,
            device,
            cli.gradient_batches,
            run_args.cf0_loss_weight,
            run_args.cf0_unfreeze_top_layers,
        ),
    }
    with (cli.output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
