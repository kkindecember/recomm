"""Bounded DIFF→GRAM architecture screen; never evaluates official test."""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from experiment.phase18.core.diff_data import DiffDataset, native_collator, prepare, sha256, write_json
from experiment.phase18.core.diff_gram import DiffGram
from experiment.phase18.core.diff_sequence import DiffConfig
from experiment.phase17.core.full_latte_gram_backend import (
    PrefixTree, build_gram_collator, create_fresh_gram_model,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "experiment/phase18/config/s18_diff_gram_toys.json"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def status(directory, state, **values):
    path = directory / "status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(state=state, updated_at=timestamp(), pid=os.getpid(), **values)
    write_json(path, previous)


def record(directory, event):
    event = {"time": timestamp(), **event}
    with (directory / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_inputs(config):
    directory = ROOT / config["prepared"]
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["prepared_files"].items():
        if sha256(directory / name) != expected:
            raise AssertionError(f"Prepared data drift: {name}")
    if manifest["attribute_features"] != config["attribute_features"]:
        raise AssertionError("Attribute feature contract mismatch")
    if manifest.get("dataset", "Toys") != config.get("dataset", "Toys"):
        raise AssertionError("Prepared dataset mismatch")
    catalog = json.loads((directory / "catalog.json").read_text())
    train = DiffDataset.load(directory, "train", catalog)
    validation = DiffDataset.load(directory, "validation", catalog)
    tokenizer, collator = native_collator(ROOT, config)
    return catalog, train, validation, manifest, tokenizer, collator


def build_model(config, catalog, tokenizer, device):
    if sha256(ROOT / config["parent_checkpoint"]) != config["parent_sha256"]:
        raise AssertionError("Historical parent checkpoint drift")
    gram = create_fresh_gram_model(ROOT, "G0_GRAM_B0_FRESH", tokenizer, seed=config["seed"])
    parent = torch.load(ROOT / config["parent_checkpoint"], map_location="cpu")
    # The reusable fresh-model helper resizes to tokenizer length (32100),
    # whereas the original parent retains T5's padded vocabulary (32128).
    # Restore the parent's exact output space before strict loading.
    parent_vocabulary = parent["shared.weight"].size(0)
    if gram.config.vocab_size != parent_vocabulary:
        gram.resize_token_embeddings(parent_vocabulary)
    gram.load_state_dict(parent, strict=True)
    del parent
    model = DiffGram(gram, len(catalog["items"]) - 1, catalog["attribute_tables"],
                     catalog["attribute_sizes"], DiffConfig(**config["model"]))
    if config.get("reuse_cross_attention_cache", False):
        from experiment.phase18.core.diff_generation_cache import install_cross_cache_reuse
        install_cross_cache_reuse(model.gram, config["training"]["beam_size"])
    return model.to(device)


def snapshot(directory, config, args, manifest, model):
    sources = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and str(path).endswith(".py"):
            path = Path(path).resolve()
            if ROOT in path.parents:
                sources.add(path)
    sources.add((ROOT / args.config).resolve())
    sources.add(ROOT / "experiment/phase18/run_stage18_diff_gram.sh")
    hashes = {}
    for path in sorted(sources):
        relative = path.relative_to(ROOT)
        destination = directory / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = sha256(path)
    write_json(directory / "manifest.json", {
        "created_at": timestamp(), "config": config, "command_args": vars(args),
        "prepared_manifest": manifest, "source_sha256": hashes,
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "parameters": sum(p.numel() for p in model.parameters()),
        "new_parameters": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("gram.")),
    })


def model_batch(batch, device, training):
    result = {"input_ids": batch["item_text_ids"].to(device),
              "attention_mask": batch["item_text_masks"].to(device),
              "history_item_ids": batch["history_item_ids"].to(device)}
    if training:
        result.update(labels=batch["target_ids"].to(device), target_item_ids=batch["target_item_ids"].to(device))
    return result


def make_optimizer(model, config):
    train = config["training"]
    new = [p for name, p in model.named_parameters() if not name.startswith("gram.")]
    base = [p for name, p in model.named_parameters() if name.startswith("gram.")]
    return torch.optim.AdamW([{"params": new, "lr": train["new_learning_rate"]},
                             {"params": base, "lr": train["backbone_learning_rate"]}],
                            weight_decay=train["weight_decay"], eps=train["adam_epsilon"])


def save_checkpoint(path, model, **extra):
    temporary = path.with_suffix(".tmp.pt")
    # Move the detached state to CPU: saving must not allocate a second GPU model.
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"model": state, **extra}, temporary)
    temporary.replace(path)


def gradient_summary(model):
    groups = {"frequency": ".beta", "intermediate": ".intermediate.", "early": ".early.",
              "attribute": "attribute_embeddings", "memory": "memory_projection", "backbone": "gram."}
    result = {}
    for group, fragment in groups.items():
        grads = [p.grad for name, p in model.named_parameters() if fragment in name and p.grad is not None]
        if any(not torch.isfinite(grad).all() for grad in grads):
            raise FloatingPointError(f"Nonfinite {group} gradient")
        result[group] = sum(float(grad.abs().sum()) for grad in grads)
    return result


class CatalogDecoder:
    def __init__(self, catalog, config):
        self.paths = catalog["decoder_paths"]
        self.path_to_item = {tuple(path): index + 1 for index, path in enumerate(self.paths)}
        self.trie = PrefixTree(self.paths)
        self.beams = config["training"]["beam_size"]
        self.length_penalty = config["training"]["length_penalty"]

    def generate(self, model, inputs):
        output = model.generate(**inputs, max_length=max(map(len, self.paths)),
                                prefix_allowed_tokens_fn=self.trie.prefix_allowed_tokens_fn(),
                                num_beams=self.beams, num_return_sequences=self.beams,
                                output_scores=True, return_dict_in_generate=True,
                                length_penalty=self.length_penalty)
        sequences = output.sequences.cpu().tolist()
        scores = output.sequences_scores.cpu().tolist()
        result = []
        for start in range(0, len(sequences), self.beams):
            items, item_scores = [], []
            for sequence, score in zip(sequences[start:start + self.beams], scores[start:start + self.beams]):
                if 1 not in sequence or not math.isfinite(score):
                    raise AssertionError("Incomplete or nonfinite generated candidate")
                path = tuple(sequence[:sequence.index(1) + 1])
                if path not in self.path_to_item:
                    raise AssertionError("Generated path outside lexical catalog")
                items.append(self.path_to_item[path])
                item_scores.append(score)
            if len(items) != self.beams or len(set(items)) != self.beams:
                raise AssertionError("Candidate count or uniqueness mismatch")
            result.append((items, item_scores))
        return result


def smoke(model, train, collator, decoder, config, directory, device, microbatch):
    longest = sorted(range(len(train)), key=lambda i: len(train.records[i]["history"]), reverse=True)
    rows = [train[i] for i in longest[:microbatch]]
    inputs = model_batch(collator(rows), device, True)
    optimizer = make_optimizer(model, config)
    results = []
    for frozen in (True, False):
        model.set_backbone_frozen(frozen)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs)
            if not torch.isfinite(output.loss):
                raise FloatingPointError("Nonfinite smoke loss")
            output.loss.backward()
            gradients = gradient_summary(model)
            for group, magnitude in gradients.items():
                if not (frozen and group == "backbone") and magnitude <= 0:
                    raise AssertionError(f"No gradient reaches {group}")
            if frozen and gradients["backbone"] != 0:
                raise AssertionError("Frozen backbone received gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        results.append({"backbone_frozen": frozen, "steps": 2,
                        "seconds_per_step": (time.monotonic() - started) / 2,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                        "loss_components": {k: float(v) for k, v in model.last_loss_components.items()},
                        "gradients_abs_sum": gradients})
        record(directory, {"event": "smoke_training", **results[-1]})
    optimizer.zero_grad(set_to_none=True)
    del output, optimizer
    torch.cuda.empty_cache()
    model.eval()
    inputs = model_batch(collator(rows[:1]), device, False)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.no_grad():
        before = decoder.generate(model, inputs)
    torch.cuda.synchronize()
    generation_seconds = time.monotonic() - started
    peak = torch.cuda.max_memory_reserved() / 2**20
    save_checkpoint(directory / "smoke.pt", model)
    saved = torch.load(directory / "smoke.pt", map_location="cpu")
    model.load_state_dict(saved["model"], strict=True)
    del saved
    with torch.no_grad():
        after = decoder.generate(model, inputs)
    if before != after:
        raise AssertionError("Checkpoint reload changed deterministic generation")
    cache_parity = None
    if config.get("reuse_cross_attention_cache", False):
        short_inputs = model_batch(collator([train[0]]), device, False)
        optimized = decoder.generate(model, short_inputs)
        model.gram._diff_cross_cache_reuse = False
        original = decoder.generate(model, short_inputs)
        model.gram._diff_cross_cache_reuse = True
        cache_parity = original == optimized
        if not cache_parity:
            raise AssertionError("Cross cache reuse changed beam candidates or scores")
    summary = {"state": "SMOKE_PASSED", "microbatch": microbatch,
               "long_history_length": len(rows[0]["history_item_ids"]), "training": results,
               "generation_seconds": generation_seconds, "generation_peak_reserved_mib": peak,
               "beam_count": len(before[0][0]), "checkpoint_roundtrip_equal": True,
               "cross_cache_reuse_exact_parity": cache_parity,
               "efficacy_evidence": False}
    write_json(directory / "smoke_summary.json", summary)
    status(directory, "SMOKE_PASSED", summary=summary)
    record(directory, {"event": "smoke_completed", **summary})


@torch.no_grad()
def evaluate(model, dataset, collator, decoder, config, directory, epoch, device, baseline):
    model.eval()
    torch.cuda.empty_cache()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator,
                        num_workers=config["training"]["num_workers"])
    totals = {f"{metric}@{k}": 0.0 for metric in ("hit", "ndcg") for k in (5, 10, 20, 50)}
    path = directory / f"validation_epoch_{epoch:02d}.jsonl"
    temporary = path.with_suffix(".partial.jsonl")
    started = time.monotonic()
    last_update = started
    with temporary.open("w") as handle:
        for index, batch in enumerate(loader, start=1):
            ranked, scores = decoder.generate(model, model_batch(batch, device, False))[0]
            gold = int(batch["target_item_ids"][0])
            rank = ranked.index(gold) + 1 if gold in ranked else None
            if rank:
                for k in (5, 10, 20, 50):
                    if rank <= k:
                        totals[f"hit@{k}"] += 1
                        totals[f"ndcg@{k}"] += 1 / math.log2(rank + 1)
            handle.write(json.dumps({"split": "validation", "user_id": batch["user_ids"][0],
                                     "gold_item_id": gold, "ranked_item_ids": ranked,
                                     "sequence_scores": scores}) + "\n")
            now = time.monotonic()
            if now - last_update >= 30:
                handle.flush()
                status(directory, "VALIDATING", epoch=epoch, evaluated=index, validation_total=len(dataset),
                       validation_elapsed_seconds=now - started)
                record(directory, {"event": "validation_progress", "epoch": epoch,
                                   "examples": index, "total": len(dataset)})
                last_update = now
    temporary.replace(path)
    metrics = {key: value / len(dataset) for key, value in totals.items()}
    result = {"epoch": epoch, "split": "validation", "examples": len(dataset), "metrics": metrics,
              "historical_gram": baseline, "delta": {key: metrics[key] - value for key, value in baseline.items()},
              "seconds": time.monotonic() - started, "predictions": path.name,
              "predictions_sha256": sha256(path)}
    write_json(directory / f"validation_epoch_{epoch:02d}.json", result)
    record(directory, {"event": "validation_completed", **result})
    return result


def train_candidate(model, dataset, validation, collator, decoder, config, manifest, directory, device, microbatch):
    settings = config["training"]
    effective = settings["effective_batch_size"]
    if effective % microbatch:
        raise ValueError("Microbatch must divide effective batch size")
    accumulation = effective // microbatch
    optimizer = make_optimizer(model, config)
    steps_per_epoch = math.ceil(len(dataset) / effective)
    best, best_epoch, no_improvement = -math.inf, None, 0
    optimizer_steps = 0
    started = time.monotonic()
    phases = [("warmup", settings["warmup_epochs"]), ("joint", settings["joint_epochs"])]
    epoch = 0
    stop = False
    for phase, phase_epochs in phases:
        model.set_backbone_frozen(phase == "warmup")
        # Independent planned warmup/decay per phase; optimizer moments persist.
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = (settings["new_learning_rate"] if group is optimizer.param_groups[0]
                                                  else settings["backbone_learning_rate"])
        total_steps = steps_per_epoch * phase_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * settings["warmup_fraction"])), total_steps)
        for phase_epoch in range(1, phase_epochs + 1):
            epoch += 1
            generator = torch.Generator().manual_seed(config["seed"] + epoch)
            loader = DataLoader(dataset, batch_size=microbatch, shuffle=True, generator=generator,
                                collate_fn=collator, num_workers=settings["num_workers"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sums = dict(generation=0.0, item=0.0, alignment=0.0, total=0.0)
            seen, step_examples = 0, 0
            epoch_started = last_update = time.monotonic()
            status(directory, "TRAINING", phase=phase, epoch=epoch, phase_epoch=phase_epoch,
                   seen=0, train_total=len(dataset), effective_batch=effective, microbatch=microbatch)
            for batch_index, batch in enumerate(loader):
                count = batch["target_item_ids"].size(0)
                output = model(**model_batch(batch, device, True))
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("Nonfinite training loss")
                # Sum sample gradients, normalize by actual group size including final partial group.
                (output.loss * count).backward()
                for key, value in model.last_loss_components.items():
                    loss_sums[key] += float(value) * count
                seen += count
                step_examples += count
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(step_examples)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"],
                                                              error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    step_examples = 0
                now = time.monotonic()
                if now - last_update >= 30:
                    event = {"event": "train_progress", "phase": phase, "epoch": epoch,
                             "seen": seen, "total": len(dataset), "optimizer_steps": optimizer_steps,
                             "loss": {key: value / seen for key, value in loss_sums.items()},
                             "epoch_elapsed_seconds": now - epoch_started,
                             "epoch_eta_seconds": (len(dataset) - seen) * (now - epoch_started) / seen,
                             "learning_rates": [group["lr"] for group in optimizer.param_groups]}
                    status(directory, "TRAINING", **{key: value for key, value in event.items() if key != "event"})
                    record(directory, event)
                    last_update = now
            record(directory, {"event": "epoch_completed", "phase": phase, "epoch": epoch,
                               "loss": {key: value / seen for key, value in loss_sums.items()},
                               "seconds": time.monotonic() - epoch_started})
            save_checkpoint(directory / "last.pt", model, epoch=epoch, phase=phase, config=config,
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            optimizer_steps=optimizer_steps)
            if phase == "warmup" or phase_epoch % settings["evaluation_interval"] == 0:
                result = evaluate(model, validation, collator, decoder, config, directory, epoch,
                                  device, manifest["baseline_validation"])
                score = result["metrics"]["ndcg@10"]
                if score > best:
                    best, best_epoch, no_improvement = score, epoch, 0
                    save_checkpoint(directory / "best.pt", model, epoch=epoch, config=config, validation=result)
                else:
                    no_improvement += 1
                if (phase == "joint" and phase_epoch >= settings["min_joint_epochs"]
                        and no_improvement >= settings["patience"]):
                    stop = True
                    break
        if stop:
            break
    result = {"state": "COMPLETED", "best_epoch": best_epoch, "best_validation_ndcg@10": best,
              "historical_gram_validation_ndcg@10": manifest["baseline_validation"]["ndcg@10"],
              "delta_ndcg@10": best - manifest["baseline_validation"]["ndcg@10"],
              "completed_epochs": epoch, "early_stopped": stop, "seconds": time.monotonic() - started,
              "interpretation": "Single-seed architecture screen; historical parent has a different training budget."}
    write_json(directory / "result.json", result)
    status(directory, "COMPLETED", result=result)
    record(directory, {"event": "training_completed", **result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "train", "status"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output")
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--smoke-report", help="Required passing smoke summary for training")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if args.mode == "prepare":
        print(json.dumps(prepare(ROOT, config, ROOT / config["prepared"]), indent=2))
        return
    if not args.output:
        parser.error("--output is required")
    directory = (ROOT / args.output).resolve()
    if args.mode == "status":
        print((directory / "status.json").read_text())
        return
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite run {directory}")
    if args.microbatch <= 0 or config["training"]["effective_batch_size"] % args.microbatch:
        raise ValueError("Invalid microbatch")
    directory.mkdir(parents=True)
    try:
        status(directory, "INITIALIZING", mode=args.mode, config=args.config)
        if args.mode == "train":
            if not args.smoke_report:
                raise ValueError("A passing real-model smoke is required before training")
            report = json.loads((ROOT / args.smoke_report).read_text())
            report_manifest = json.loads((ROOT / args.smoke_report).with_name("manifest.json").read_text())
            if report["state"] != "SMOKE_PASSED" or report["microbatch"] != args.microbatch:
                raise AssertionError("Smoke report/microbatch mismatch")
            if report_manifest["config"] != config:
                raise AssertionError("Config changed since smoke")
            for name, expected in report_manifest["source_sha256"].items():
                if sha256(ROOT / name) != expected:
                    raise AssertionError(f"Source changed since smoke: {name}")
        seed_everything(config["seed"])
        if not torch.cuda.is_available():
            raise RuntimeError("Real-model smoke/training requires CUDA")
        catalog, dataset, validation, manifest, tokenizer, collator = load_inputs(config)
        device = torch.device("cuda:0")
        model = build_model(config, catalog, tokenizer, device)
        snapshot(directory, config, args, manifest, model)
        decoder = CatalogDecoder(catalog, config)
        if args.mode == "smoke":
            smoke(model, dataset, collator, decoder, config, directory, device, args.microbatch)
        else:
            train_candidate(model, dataset, validation, collator, decoder, config, manifest,
                            directory, device, args.microbatch)
    except BaseException as error:
        status(directory, "FAILED", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
