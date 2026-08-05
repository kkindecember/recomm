#!/usr/bin/env python3
"""Train a zero-initialized, GRAM-safe adapter over the frozen P9-2A item state."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, T5Config


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAM_ROOT = REPO_ROOT / "GRAM"
GRAM_SRC = GRAM_ROOT / "src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner as _runner_import_guard  # noqa: E402,F401
from data import MultiTaskDatasetGRAM  # noqa: E402
from model import create_model  # noqa: E402
from processor import CollatorGRAM  # noqa: E402
from utils import evaluate  # noqa: E402
from utils import generation_trie as gt  # noqa: E402
from train_cf0_b2_item_head import CF0B2ItemHead  # noqa: E402


DEFAULT_BASE_CONFIG = GRAM_ROOT / "log/Toys/1_20260720_1830/config.json"
DEFAULT_BASE_CHECKPOINT = (
    GRAM_ROOT
    / "log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt"
)
DEFAULT_ITEM_CHECKPOINT = (
    REPO_ROOT / "artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/phase9/cf0_b2_toys_safe_fusion_p2b"

BASELINE_BEAM = {
    "hit@1": 0.04167525242118277,
    "hit@3": 0.07361425922110035,
    "hit@5": 0.0909231403255718,
    "hit@10": 0.11941067381001443,
    "hit@20": 0.15444055223573047,
    "hit@50": 0.21193076447558212,
    "ndcg@1": 0.04167525242118277,
    "ndcg@3": 0.06000543033948122,
    "ndcg@5": 0.06713487106644757,
    "ndcg@10": 0.07627451426000033,
    "ndcg@20": 0.08510430222346706,
    "ndcg@50": 0.09653217535352887,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--item-checkpoint", type=Path, default=DEFAULT_ITEM_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-residual-scale", type=float, default=0.20)
    parser.add_argument("--identity-samples", type=int, default=128)
    parser.add_argument("--nll-samples", type=int, default=4096)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--skip-beam", action="store_true")
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=250)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run_args(path):
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    values.update(
        {
            "data_path": str(GRAM_ROOT / "rec_datasets"),
            "prompt_file": str(GRAM_ROOT / "prompt.txt"),
            "rank": 0,
            "distributed": 0,
            "debug_train_100": 0,
            "debug_test_100": 0,
            "debug_test_small_set": 0,
            "eval_batch_size": 1,
        }
    )
    return SimpleNamespace(**values)


def reverse_valid_prefix(values, valid_mask):
    positions = torch.arange(values.size(1), device=values.device).expand_as(valid_mask)
    lengths = valid_mask.long().sum(dim=1, keepdim=True)
    gather = torch.where(positions < lengths, lengths - 1 - positions, positions).clamp_min(0)
    if values.dim() == 3:
        gather = gather.unsqueeze(-1).expand_as(values)
    return values.gather(1, gather)


class ZeroInitSafeFusion(nn.Module):
    """Bounded residual adapter; alpha=0 is exactly the identity map."""

    def __init__(self, d_model, max_residual_scale=0.20):
        super().__init__()
        self.cf_projection = nn.Linear(d_model, d_model, bias=False)
        self.hidden_gate = nn.Linear(d_model, d_model, bias=False)
        self.cf_gate = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.max_residual_scale = float(max_residual_scale)
        self.enabled = True
        self.last_gate_mean = None
        self.last_gate_std = None
        nn.init.xavier_uniform_(self.cf_projection.weight)
        nn.init.xavier_uniform_(self.hidden_gate.weight)
        nn.init.xavier_uniform_(self.cf_gate.weight)
        nn.init.constant_(self.cf_gate.bias, -2.0)

    @property
    def actual_scale(self):
        return self.max_residual_scale * torch.tanh(self.alpha)

    def forward(self, hidden, cf_state, valid_mask=None):
        if not self.enabled:
            return hidden
        if cf_state is None:
            raise RuntimeError("safe-fusion collaborative context is missing")
        residual = F.normalize(self.cf_projection(cf_state), dim=-1).unsqueeze(1)
        gate = torch.sigmoid(
            self.hidden_gate(hidden) + self.cf_gate(cf_state).unsqueeze(1)
        )
        self.last_gate_mean = gate.detach().mean()
        self.last_gate_std = gate.detach().std(unbiased=False)
        update = self.actual_scale * gate * residual
        if valid_mask is not None:
            update = update * valid_mask.unsqueeze(-1).to(update.dtype)
        return hidden + update


class SafeFusionEncoder(nn.Module):
    """Wrap GRAM's encoder and alter only the coarse user-prompt passage."""

    def __init__(self, base_encoder, adapter):
        super().__init__()
        self.base_encoder = base_encoder
        self.adapter = adapter
        self.cf_state = None

    @property
    def main_input_name(self):
        return self.base_encoder.main_input_name

    @property
    def n_passages(self):
        return self.base_encoder.n_passages

    @n_passages.setter
    def n_passages(self, value):
        self.base_encoder.n_passages = value

    def set_cf_state(self, cf_state):
        self.cf_state = cf_state

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        outputs = self.base_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden = outputs[0]
        passage_length = hidden.size(1) // self.n_passages
        prompt_hidden = hidden[:, :passage_length]
        prompt_mask = None
        if attention_mask is not None:
            prompt_mask = attention_mask.reshape(hidden.size(0), -1)[:, :passage_length].bool()
        fused_prompt = self.adapter(prompt_hidden, self.cf_state, prompt_mask)
        fused = torch.cat([fused_prompt, hidden[:, passage_length:]], dim=1)
        return (fused,) + outputs[1:]


def configure_models(run_args, base_checkpoint, item_checkpoint, device, max_scale):
    config = T5Config.from_pretrained(run_args.backbone, local_files_only=True)
    config.max_seq_len = run_args.item_prompt_max_len
    config.max_item_num = run_args.max_his
    config.use_position_embedding = run_args.use_position_embedding
    config.sample_num = run_args.sample_num
    config.cf0_enabled = False
    config.cf0_arm = "A"
    model = create_model("gram", config=config)
    state = torch.load(base_checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"base checkpoint mismatch: {incompatible}")

    item_payload = torch.load(item_checkpoint, map_location="cpu")
    item_config = item_payload["model_config"]
    item_model = CF0B2ItemHead(
        num_items=item_config["num_items"],
        max_history=item_config["max_history"],
        d_model=item_config["d_model"],
        num_layers=item_config["num_layers"],
        num_heads=item_config["num_heads"],
        dropout=item_config["dropout"],
        temperature=item_config["temperature_initial"],
    )
    item_model.load_state_dict(item_payload["model_state_dict"], strict=True)
    if item_model.d_model != config.d_model:
        raise ValueError("item-state and GRAM dimensions differ")

    adapter = ZeroInitSafeFusion(config.d_model, max_scale)
    model.encoder = SafeFusionEncoder(model.encoder, adapter)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in adapter.parameters():
        parameter.requires_grad = True
    for parameter in item_model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()
    item_model.to(device).eval()
    return model, item_model, adapter


def build_loaders(run_args, tokenizer, batch_size, eval_batch_size, seed, max_train, max_val):
    train_dataset = MultiTaskDatasetGRAM(
        run_args, run_args.datasets, "train", None, tokenizer, phase=0, regenerate=False
    )
    validation_dataset = MultiTaskDatasetGRAM(
        run_args, run_args.datasets, "validation", None, tokenizer, phase=0, regenerate=False
    )
    if max_train:
        train_dataset = Subset(train_dataset, range(min(max_train, len(train_dataset))))
    if max_val:
        validation_dataset = Subset(
            validation_dataset, range(min(max_val, len(validation_dataset)))
        )
    collator = CollatorGRAM(tokenizer, args=run_args, mode="train")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    return train_dataset, validation_dataset, train_loader, validation_loader


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def collaborative_state(item_model, batch):
    ids = reverse_valid_prefix(batch["history_item_ids"], batch["history_item_mask"])
    mask = reverse_valid_prefix(batch["history_item_mask"], batch["history_item_mask"])
    with torch.no_grad():
        return item_model.encode(ids, mask)


def set_context(model, item_model, batch):
    model.encoder.set_cf_state(collaborative_state(item_model, batch))


def per_example_nll(logits, labels):
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    valid = labels.ne(-100)
    return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def model_logits(model, item_model, batch, fusion_enabled):
    model.encoder.adapter.enabled = fusion_enabled
    if fusion_enabled:
        set_context(model, item_model, batch)
    with torch.no_grad():
        output = model(
            input_ids=batch["item_text_ids"],
            attention_mask=batch["item_text_masks"],
            labels=batch["target_ids"],
            return_dict=True,
        )
    return output.logits


def identity_audit(model, item_model, loader, device, sample_limit):
    max_logit_delta = 0.0
    max_nll_delta = 0.0
    seen = 0
    for raw_batch in loader:
        if seen >= sample_limit:
            break
        batch = move_batch(raw_batch, device)
        remaining = sample_limit - seen
        if batch["target_ids"].size(0) > remaining:
            batch = {
                key: value[:remaining] if torch.is_tensor(value) else value[:remaining]
                for key, value in batch.items()
            }
        baseline = model_logits(model, item_model, batch, False)
        fused = model_logits(model, item_model, batch, True)
        max_logit_delta = max(max_logit_delta, float((fused - baseline).abs().max()))
        baseline_nll = per_example_nll(baseline, batch["target_ids"])
        fused_nll = per_example_nll(fused, batch["target_ids"])
        max_nll_delta = max(max_nll_delta, float((fused_nll - baseline_nll).abs().max()))
        seen += batch["target_ids"].size(0)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    passed = (
        seen == sample_limit
        and max_logit_delta <= 1e-7
        and max_nll_delta <= 1e-8
        and float(model.encoder.adapter.alpha.detach()) == 0.0
        and trainable
        and all(name.startswith("encoder.adapter.") for name in trainable)
    )
    return {
        "status": "passed" if passed else "failed",
        "samples": seen,
        "max_absolute_logit_delta": max_logit_delta,
        "max_absolute_per_example_nll_delta": max_nll_delta,
        "alpha": float(model.encoder.adapter.alpha.detach()),
        "trainable_parameter_names": trainable,
    }


def scheduler_factor(step, total_steps, warmup_steps):
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def train_epoch(model, item_model, adapter, loader, optimizer, scheduler, device, epoch, log_every):
    model.eval()
    adapter.train()
    loss_sum = 0.0
    sample_count = 0
    grad_sum = 0.0
    for step, raw_batch in enumerate(loader, 1):
        batch = move_batch(raw_batch, device)
        set_context(model, item_model, batch)
        adapter.enabled = True
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=batch["item_text_ids"],
            attention_mask=batch["item_text_masks"],
            labels=batch["target_ids"],
            return_dict=True,
        )
        loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}")
        loss.backward()
        leaked = [
            name
            for name, parameter in model.named_parameters()
            if not name.startswith("encoder.adapter.") and parameter.grad is not None
        ]
        if leaked:
            raise RuntimeError(f"gradient leaked into frozen parameters: {leaked[:5]}")
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        count = batch["target_ids"].size(0)
        loss_sum += float(loss.detach()) * count
        sample_count += count
        grad_sum += float(grad_norm)
        if step % log_every == 0 or step == len(loader):
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "epoch": epoch,
                        "step": step,
                        "steps": len(loader),
                        "mean_loss": loss_sum / sample_count,
                        "alpha": float(adapter.alpha.detach()),
                        "actual_scale": float(adapter.actual_scale.detach()),
                    }
                ),
                flush=True,
            )
    return {
        "train_loss": loss_sum / sample_count,
        "mean_preclip_gradient_norm": grad_sum / len(loader),
        "alpha": float(adapter.alpha.detach()),
        "actual_scale": float(adapter.actual_scale.detach()),
        "learning_rate_end": scheduler.get_last_lr()[0],
    }


def collect_nll(model, item_model, loader, device, sample_limit):
    baseline_values = []
    fused_values = []
    users = []
    model.eval()
    model.encoder.adapter.eval()
    for raw_batch in loader:
        if len(users) >= sample_limit:
            break
        batch = move_batch(raw_batch, device)
        remaining = sample_limit - len(users)
        if batch["target_ids"].size(0) > remaining:
            batch = {
                key: value[:remaining] if torch.is_tensor(value) else value[:remaining]
                for key, value in batch.items()
            }
        baseline_logits = model_logits(model, item_model, batch, False)
        fused_logits = model_logits(model, item_model, batch, True)
        baseline_values.extend(per_example_nll(baseline_logits, batch["target_ids"]).cpu().tolist())
        fused_values.extend(per_example_nll(fused_logits, batch["target_ids"]).cpu().tolist())
        users.extend(batch["user_ids"])
    return users, np.asarray(baseline_values), np.asarray(fused_values)


def summarize_nll(baseline, fused, bootstrap_repetitions, seed):
    delta = fused - baseline
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_repetitions, dtype=np.float64)
    for index in range(bootstrap_repetitions):
        selected = generator.integers(0, len(delta), size=len(delta))
        bootstrap[index] = delta[selected].mean()
    mean_delta = float(delta.mean())
    upper = float(np.quantile(bootstrap, 0.975))
    worse_fraction = float((delta > 0).mean())
    passed = mean_delta <= -0.002 and upper < 0.0 and worse_fraction <= 0.50
    return {
        "status": "passed" if passed else "failed",
        "count": len(delta),
        "baseline_mean_nll": float(baseline.mean()),
        "fused_mean_nll": float(fused.mean()),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(delta)),
        "delta_p95": float(np.quantile(delta, 0.95)),
        "fused_worse_fraction": worse_fraction,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            upper,
        ],
        "checks": {
            "mean_delta_le_minus_0.002": mean_delta <= -0.002,
            "bootstrap_upper_lt_zero": upper < 0.0,
            "fused_worse_fraction_le_0.50": worse_fraction <= 0.50,
        },
    }


def write_paired_nll(path, users, baseline, fused):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "baseline_nll", "fused_nll", "delta"])
        for user, base, value in zip(users, baseline, fused):
            writer.writerow([user, f"{base:.10f}", f"{value:.10f}", f"{value-base:.10f}"])


def encoded_candidates(dataset, tokenizer):
    output = []
    for candidate in dataset.all_items:
        values = [0]
        for token in tokenizer.encode(candidate):
            if token not in (1820, 9175):
                values.append(token)
        output.append(values)
    return output


def beam_validation(model, item_model, dataset, tokenizer, collator, device, beam_size, path):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)
    candidates = encoded_candidates(dataset, tokenizer)
    prefix_allowed = gt.prefix_allowed_tokens_fn(gt.Trie(candidates))
    max_length = max(len(value) for value in candidates)
    metric_names = list(BASELINE_BEAM)
    totals = np.zeros(len(metric_names), dtype=np.float64)
    records = 0
    model.eval()
    model.encoder.adapter.enabled = True
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["user_id", "gold", "top1", "gold_rank"] + metric_names)
        with torch.no_grad():
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                set_context(model, item_model, batch)
                prediction = model.generate(
                    input_ids=batch["item_text_ids"],
                    attention_mask=batch["item_text_masks"],
                    max_length=max_length,
                    prefix_allowed_tokens_fn=prefix_allowed,
                    num_beams=beam_size,
                    num_return_sequences=beam_size,
                    output_scores=True,
                    return_dict_in_generate=True,
                    length_penalty=1.0,
                )
                output_ids = torch.where(batch["target_ids"] == -100, 0, batch["target_ids"])
                gold = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                generated = tokenizer.batch_decode(prediction["sequences"], skip_special_tokens=True)
                scores = prediction["sequences_scores"]
                pairs = sorted(zip(generated, scores.tolist()), key=lambda pair: pair[1], reverse=True)
                ranked = [value for value, _ in pairs]
                relevance = [[1 if value == gold else 0 for value in ranked]]
                metrics = evaluate.get_metrics_results(relevance, metric_names)
                totals += metrics
                rank = ranked.index(gold) + 1 if gold in ranked else 0
                writer.writerow(
                    [batch["user_ids"][0], gold, ranked[0], rank]
                    + [f"{value:.10f}" for value in metrics]
                )
                records += 1
                if records % 250 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "beam_progress",
                                "records": records,
                                "total": len(dataset),
                                "hit@10": totals[metric_names.index("hit@10")] / records,
                            }
                        ),
                        flush=True,
                    )
    metrics = {name: float(value / records) for name, value in zip(metric_names, totals)}
    checks = {
        "hit@10": metrics["hit@10"] >= BASELINE_BEAM["hit@10"] - 0.002,
        "ndcg@10": metrics["ndcg@10"] >= BASELINE_BEAM["ndcg@10"] - 0.001,
        "hit@50": metrics["hit@50"] >= BASELINE_BEAM["hit@50"] - 0.003,
        "record_count": records == 19412,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "count": records,
        "beam_size": beam_size,
        "baseline": BASELINE_BEAM,
        "fused": metrics,
        "delta": {name: metrics[name] - BASELINE_BEAM[name] for name in metric_names},
        "checks": checks,
    }


def save_adapter(path, adapter, epoch, nll_summary):
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "adapter_config": {
                "d_model": adapter.cf_projection.in_features,
                "max_residual_scale": adapter.max_residual_scale,
            },
            "epoch": epoch,
            "nll": nll_summary,
        },
        path,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    set_seed(args.seed)
    started = time.perf_counter()
    device = torch.device(args.device)
    run_args = load_run_args(args.base_config)
    tokenizer = AutoTokenizer.from_pretrained(run_args.backbone, local_files_only=True)
    base_sha_before = sha256_file(args.base_checkpoint)
    item_sha_before = sha256_file(args.item_checkpoint)
    model, item_model, adapter = configure_models(
        run_args,
        args.base_checkpoint,
        args.item_checkpoint,
        device,
        args.max_residual_scale,
    )
    train_dataset, validation_dataset, train_loader, validation_loader = build_loaders(
        run_args,
        tokenizer,
        args.batch_size,
        args.eval_batch_size,
        args.seed,
        args.max_train_samples,
        args.max_validation_samples,
    )
    identity_limit = min(args.identity_samples, len(validation_dataset))
    nll_limit = min(args.nll_samples, len(validation_dataset))
    identity = identity_audit(model, item_model, validation_loader, device, identity_limit)
    print(json.dumps({"event": "identity_gate", **identity}), flush=True)
    if identity["status"] != "passed":
        raise RuntimeError(f"identity gate failed: {identity}")

    trainable_parameters = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_factor(step, total_steps, warmup_steps),
    )
    history = []
    best = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_stats = train_epoch(
            model,
            item_model,
            adapter,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            args.log_every,
        )
        users, baseline_nll, fused_nll = collect_nll(
            model, item_model, validation_loader, device, nll_limit
        )
        nll_summary = summarize_nll(
            baseline_nll, fused_nll, args.bootstrap_repetitions, args.seed
        )
        record = {
            "epoch": epoch,
            **train_stats,
            "validation_nll": nll_summary,
            "wall_time_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(json.dumps({"event": "epoch_complete", **record}), flush=True)
        selection = (nll_summary["fused_mean_nll"], nll_summary["delta_p95"])
        if best is None or selection < best[0]:
            best = (selection, epoch, nll_summary)
            save_adapter(args.output_dir / "best_adapter.pt", adapter, epoch, nll_summary)

    payload = torch.load(args.output_dir / "best_adapter.pt", map_location=device)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    users, baseline_nll, fused_nll = collect_nll(
        model, item_model, validation_loader, device, nll_limit
    )
    nll_gate = summarize_nll(
        baseline_nll, fused_nll, args.bootstrap_repetitions, args.seed
    )
    write_paired_nll(args.output_dir / "paired_nll.tsv", users, baseline_nll, fused_nll)
    base_sha_after = sha256_file(args.base_checkpoint)
    item_sha_after = sha256_file(args.item_checkpoint)
    checkpoint_integrity = {
        "base_sha256_before": base_sha_before,
        "base_sha256_after": base_sha_after,
        "item_sha256_before": item_sha_before,
        "item_sha256_after": item_sha_after,
        "passed": base_sha_before == base_sha_after and item_sha_before == item_sha_after,
    }
    if not checkpoint_integrity["passed"]:
        raise RuntimeError("frozen source checkpoint changed during training")

    beam = {"status": "not_run", "reason": "teacher-forced NLL gate failed"}
    formal_full_validation = (
        len(validation_dataset) == 19412
        and not args.max_validation_samples
        and not args.skip_beam
    )
    if nll_gate["status"] == "passed" and formal_full_validation:
        collator = CollatorGRAM(tokenizer, args=run_args, mode="valid")
        beam = beam_validation(
            model,
            item_model,
            validation_dataset,
            tokenizer,
            collator,
            device,
            args.beam_size,
            args.output_dir / "beam_validation.tsv",
        )
    elif args.skip_beam:
        beam = {"status": "not_run", "reason": "beam explicitly skipped for smoke"}

    if not formal_full_validation:
        scientific_status = "smoke_only"
    elif nll_gate["status"] != "passed":
        scientific_status = "failed_nll_gate"
    elif beam["status"] == "passed":
        scientific_status = "passed"
    else:
        scientific_status = "failed_beam_gate"
    summary = {
        "experiment_id": "GRAM_PHASE9_CF0_B2_TOYS_SAFE_FUSION_P2B_V1",
        "status": "completed",
        "scientific_status": scientific_status,
        "dataset": "Toys",
        "split": "validation",
        "seed": args.seed,
        "test_read": False,
        "sports_read": False,
        "beauty_read": False,
        "base_checkpoint": str(args.base_checkpoint),
        "item_checkpoint": str(args.item_checkpoint),
        "checkpoint_integrity": checkpoint_integrity,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "nll_selection_samples": nll_limit,
        "identity_gate": identity,
        "best_epoch": best[1],
        "nll_gate": nll_gate,
        "beam_gate": beam,
        "history": history,
        "adapter": {
            "trainable_parameters": trainable_parameters,
            "alpha": float(adapter.alpha.detach()),
            "actual_scale": float(adapter.actual_scale.detach()),
            "gate_mean_last_batch": (
                float(adapter.last_gate_mean) if adapter.last_gate_mean is not None else None
            ),
            "gate_std_last_batch": (
                float(adapter.last_gate_std) if adapter.last_gate_std is not None else None
            ),
        },
        "wall_time_seconds": time.perf_counter() - started,
        "resource": {
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda"
                else None
            ),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
