"""Pinned official DiffGRM, local original Beauty split, real item metrics.

Author model/tokenizer/CPD files are used unchanged. Local adapters control data
visibility, offline embeddings, item grounding, training budget, and artifacts.
"""

import argparse
from collections import Counter, defaultdict
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from transformers import get_cosine_schedule_with_warmup

from experiment.phase18.core.diff_data import baseline_reference, read_mapping, sha256, write_json

ROOT = Path(__file__).resolve().parents[3]


def now():
    return datetime.now(timezone.utc).isoformat()


def update(directory, state, **values):
    path = directory / "status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(state=state, updated_at=now(), pid=os.getpid(), **values)
    write_json(path, previous)


def event(directory, **values):
    value = {"time": now(), **values}
    with (directory / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def official_config(config, output, device):
    source = ROOT / config["source"]
    manifest = json.loads((ROOT / config["source_manifest"]).read_text())
    if manifest["commit"] != config["source_commit"]:
        raise AssertionError("Official commit mismatch")
    for name, expected in manifest["files"].items():
        if sha256(source / name) != expected:
            raise AssertionError(f"Official source modified: {name}")
    sys.path.insert(0, str(source))
    values = yaml.safe_load((source / "genrec/default.yaml").read_text())
    values.update(yaml.safe_load((source / "genrec/models/DIFF_GRM/config.yaml").read_text()))
    values.update(config["overrides"])
    values.update(device=device, category=config["dataset"], rand_seed=config["seed"],
                  max_history_len=config["max_history"], max_hist_len=config["max_history"],
                  sent_emb_model=str(ROOT / config["sentence_model"]), current_split="val",
                  cache_dir=str(output / "prepared" / "tokenizer"),
                  accelerator=SimpleNamespace(is_main_process=True))
    beam = config["beam"]
    values["vectorized_beam_search"].update(top_k_final=beam, dedup_strategy=values["dedup_strategy"],
                                             val={"beam_act": beam, "beam_max": beam})
    return values


class LocalOriginalDataset:
    def __init__(self, config, directory):
        from datasets import Dataset as HFDataset
        self.category = config["dataset"]
        source = ROOT / "GRAM/rec_datasets" / self.category
        self.item2meta = read_mapping(source / "item_plain_text.txt")
        self.names = ["[PAD]"] + sorted(self.item2meta)
        self.item2id = {name: i for i, name in enumerate(self.names) if i}
        self.id_mapping = {"id2item": self.names}
        self.n_items = len(self.names)
        self.cache_dir = str(directory / "tokenizer")
        sequences = read_mapping(source / "user_sequence.txt")
        prefixes = []
        self.train, self.validation = [], []
        self.counts = Counter()
        for user, text in sequences.items():
            sequence = text.split()
            prefix = sequence[:-2]
            if len(prefix) < 2:
                raise ValueError("Unexpected empty training history")
            prefixes.append(prefix)
            numeric = [self.item2id[name] for name in prefix]
            self.counts.update(numeric)
            for position in range(config["minimum_history"], len(numeric)):
                self.train.append({"user_id": user, "history": numeric[max(0, position - config["max_history"]):position],
                                   "target": numeric[position]})
            self.validation.append({"user_id": user, "history": numeric[-config["max_history"]:],
                                    "target": self.item2id[sequence[-2]]})
        # The official tokenizer uses only this training prefix view to fit PCA/OPQ.
        self.training_prefixes = HFDataset.from_dict({"item_seq": prefixes})

    def split(self):
        return {"train": self.training_prefixes}


class SidDataset(Dataset):
    def __init__(self, records, codes, max_history):
        self.records, self.codes, self.max_history = records, codes, max_history

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        ids = row["history"]
        history = torch.full((self.max_history, 4), -1, dtype=torch.long)
        history[:len(ids)] = self.codes[ids]
        mask = torch.arange(self.max_history) < len(ids)
        target = self.codes[row["target"]]
        return {"history_sid": history, "history_mask": mask,
                "decoder_input_ids": target, "decoder_labels": target,
                "target_item_id": row["target"], "user_id": row["user_id"]}


def inference_batch(batch):
    return {name: batch[name] for name in ("history_sid", "history_mask")}


def resolve_items(sequences, mapping, topk=50):
    items, seen_codes, seen_items = [], set(), set()
    for raw in sequences:
        code = tuple(int(x) for x in raw)
        if code in seen_codes:
            continue
        seen_codes.add(code)
        for item in mapping.get(code, ()):
            if item not in seen_items:
                items.append(item)
                seen_items.add(item)
                if len(items) == topk:
                    return items
    return items


def model_for(values, tokenizer):
    from genrec.models.DIFF_GRM import DIFF_GRM
    return DIFF_GRM(values, None, tokenizer)


def checkpoint(path, model, **extra):
    temporary = path.with_suffix(".tmp.pt")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **extra}, temporary)
    temporary.replace(path)


def profile(config, values, directory, device, real_batch=None, tokenizer=None):
    tokenizer = tokenizer or SimpleNamespace(vocab_size=1027, sid_offset=3, mask_token=-1,
                                             codebooks_to_item_id=lambda code: None)
    seed_all(config["seed"])
    model = model_for(values, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=values["lr"], weight_decay=values["weight_decay"])
    if real_batch is None:
        batch = {"history_sid": torch.randint(0, 256, (config["microbatch"], config["max_history"], 4)),
                 "history_mask": torch.ones(config["microbatch"], config["max_history"], dtype=torch.bool),
                 "decoder_input_ids": torch.randint(0, 256, (config["microbatch"], 4)),
                 "decoder_labels": torch.randint(0, 256, (config["microbatch"], 4))}
        batch["decoder_input_ids"] = batch["decoder_labels"].clone()
    else:
        batch = real_batch
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(batch)
        if not torch.isfinite(result.loss):
            raise FloatingPointError("Nonfinite official model loss")
        result.loss.backward()
        for fragment in ("item_mlp", "encoder_blocks", "decoder_blocks", "mask_emb_table"):
            grads = [p.grad for name, p in model.named_parameters() if fragment in name and p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads) or sum(float(g.abs().sum()) for g in grads) == 0:
                raise AssertionError(f"Missing/nonfinite gradient in {fragment}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    step_seconds = (time.monotonic() - start) / 2
    training_peak = torch.cuda.max_memory_reserved() / 2**20
    optimizer.zero_grad(set_to_none=True)
    del result, optimizer
    torch.cuda.empty_cache()
    model.eval()
    infer = {k: v[:config["evaluation_batch"]] for k, v in inference_batch(batch).items()}
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    generated = model.generate(infer, n_return_sequences=config["beam"], mode="confidence")
    torch.cuda.synchronize()
    generation_seconds = time.monotonic() - start
    if generated.shape != (config["evaluation_batch"], config["beam"], 4) or not ((generated >= 0) & (generated < 256)).all():
        raise AssertionError("Malformed official CPD generation")
    generation_peak = torch.cuda.max_memory_reserved() / 2**20
    tag = "real_smoke" if real_batch is not None else "profile"
    checkpoint(directory / f"{tag}.pt", model)
    state = torch.load(directory / f"{tag}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    repeated = model.generate(infer, n_return_sequences=config["beam"], mode="confidence")
    if not torch.equal(generated, repeated):
        raise AssertionError("Official checkpoint generation roundtrip mismatch")
    summary = {"state": "SMOKE_PASSED" if real_batch is not None else "PROFILE_PASSED", "microbatch": config["microbatch"],
               "model_parameters": sum(p.numel() for p in model.parameters()),
               "seconds_per_microbatch": step_seconds, "training_peak_reserved_mib": training_peak,
               "generation_seconds_per_batch": generation_seconds, "generation_peak_reserved_mib": generation_peak,
               "generation_shape": list(generated.shape), "checkpoint_roundtrip_equal": True,
               "efficacy_evidence": False}
    write_json(directory / f"{tag}.json", summary)
    event(directory, event=tag, **summary)
    del generated, repeated, state, model
    torch.cuda.empty_cache()
    return summary


def prepare(config, values, directory):
    from genrec.models.DIFF_GRM import DIFF_GRMTokenizer
    from sentence_transformers import SentenceTransformer

    seed_all(config["seed"])
    prepared = directory / "prepared"
    prepared.mkdir()
    data = LocalOriginalDataset(config, prepared)
    model_manifest = json.loads((ROOT / config["sentence_manifest"]).read_text())
    for row in model_manifest["files"]:
        if sha256(ROOT / config["sentence_model"] / row["path"]) != row["sha256"]:
            raise AssertionError("Offline Sentence-T5 cache drift")

    class OfflineTokenizer(DIFF_GRMTokenizer):
        # Same source normalize/encode algorithm, with local-files-only loading
        # and chunked progress updates. PCA and OPQ methods are inherited unchanged.
        def _encode_sent_emb(self, dataset, output_path):
            model = SentenceTransformer(self.config["sent_emb_model"], local_files_only=True,
                                        trust_remote_code=False, device=self.config["device"])
            text = [dataset.item2meta[dataset.names[i]] for i in range(1, dataset.n_items)]
            arrays = []
            for start in range(0, len(text), 128):
                part = model.encode(text[start:start + 128], convert_to_numpy=True, normalize_embeddings=True,
                                    batch_size=self.config["sent_emb_batch_size"], show_progress_bar=False)
                arrays.append(part.astype(np.float32))
                update(directory, "PREPARING", phase="sentence_t5", encoded_items=min(start + 128, len(text)), total_items=len(text))
            embeddings = np.concatenate(arrays)
            if embeddings.shape != (len(text), 768) or not np.isfinite(embeddings).all():
                raise AssertionError("Malformed Sentence-T5 embeddings")
            embeddings.tofile(output_path)
            del model
            torch.cuda.empty_cache()
            update(directory, "PREPARING", phase="pca_opq")
            return embeddings

    tokenizer = OfflineTokenizer(values, data)
    codes = torch.full((data.n_items, 4), -1, dtype=torch.long)
    mapping = defaultdict(list)
    for item, tokens in tokenizer.item2tokens.items():
        numeric = data.item2id[item]
        row = tuple(int(token) - (3 + 256 * digit) for digit, token in enumerate(tokens))
        if len(row) != 4 or any(x < 0 or x >= 256 for x in row):
            raise AssertionError("OPQ returned invalid digit")
        codes[numeric] = torch.tensor(row)
        mapping[row].append(numeric)
    if not codes[1:].ge(0).all():
        raise AssertionError("Missing item SID")
    for row in mapping:
        mapping[row].sort(key=lambda item: (-data.counts[item], item))
    np.save(prepared / "item_codes.npy", codes.numpy())
    write_json(prepared / "item_names.json", data.names)
    write_json(prepared / "sid_items.json", [{"sid": list(k), "items": v} for k, v in mapping.items()])
    for name, records in (("train", data.train), ("validation", data.validation)):
        with (prepared / f"{name}.jsonl").open("w") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
    baseline, baseline_hash, sources = baseline_reference(ROOT, config)
    sources += list((ROOT / "GRAM/rec_datasets" / config["dataset"]).glob("*.txt"))
    summary = {"dataset": config["dataset"], "train_examples": len(data.train), "validation_examples": len(data.validation),
               "catalog_items": data.n_items - 1, "distinct_sids": len(mapping),
               "colliding_sids": sum(len(v) > 1 for v in mapping.values()),
               "max_collision": max(map(len, mapping.values())), "baseline_validation": baseline,
               "historical_checkpoint_sha256": baseline_hash, "test_examples_constructed": 0,
               "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
               "prepared_sha256": {str(p.relative_to(prepared)): sha256(p) for p in sorted(prepared.rglob("*")) if p.is_file()}}
    write_json(prepared / "manifest.json", summary)
    event(directory, event="data_prepared", **{k: v for k, v in summary.items() if not isinstance(v, dict)})
    return data, tokenizer, codes, dict(mapping), summary


@torch.no_grad()
def evaluate(model, dataset, mapping, config, directory, epoch, baseline):
    model.eval()
    loader = DataLoader(dataset, batch_size=config["evaluation_batch"], shuffle=False, num_workers=0)
    totals = {f"{m}@{k}": 0.0 for m in ("hit", "ndcg") for k in (5, 10, 50)}
    counts, seen, short = [], 0, 0
    start = last_update = time.monotonic()
    path = directory / f"validation_epoch_{epoch:03d}.jsonl"
    temporary = path.with_suffix(".partial.jsonl")
    with temporary.open("w") as handle:
        for batch in loader:
            raw = model.generate(inference_batch(batch), n_return_sequences=config["beam"], mode="confidence").cpu().tolist()
            for row, gold, user in zip(raw, batch["target_item_id"].tolist(), batch["user_id"]):
                items = resolve_items(row, mapping)
                counts.append(len(items))
                short += len(items) < 50
                if gold in items:
                    rank = items.index(gold) + 1
                    for k in (5, 10, 50):
                        if rank <= k:
                            totals[f"hit@{k}"] += 1
                            totals[f"ndcg@{k}"] += 1 / math.log2(rank + 1)
                handle.write(json.dumps({"split": "validation", "user_id": user, "gold_item_id": gold,
                                         "ranked_item_ids": items}) + "\n")
                seen += 1
            current = time.monotonic()
            if current - last_update >= 30:
                handle.flush()
                update(directory, "VALIDATING", phase="validation", epoch=epoch, evaluated=seen, validation_total=len(dataset))
                event(directory, event="validation_progress", epoch=epoch, evaluated=seen, total=len(dataset))
                last_update = current
    temporary.replace(path)
    result = {"epoch": epoch, "split": "validation", "examples": seen,
              "metrics": {k: v / seen for k, v in totals.items()}, "historical_gram": baseline,
              "mean_returned_items": float(np.mean(counts)), "fewer_than_50_fraction": short / seen,
              "seconds": time.monotonic() - start, "predictions_sha256": sha256(path)}
    result["delta"] = {k: result["metrics"][k] - v for k, v in baseline.items()}
    write_json(directory / f"validation_epoch_{epoch:03d}.json", result)
    event(directory, event="validation_completed", **result)
    return result


def train(config, values, directory, data, tokenizer, codes, mapping, summary):
    seed_all(config["seed"])
    model = model_for(values, tokenizer).to(values["device"])
    training = SidDataset(data.train, codes, config["max_history"])
    validation = SidDataset(data.validation, codes, config["max_history"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=values["lr"], weight_decay=values["weight_decay"])
    per_epoch = math.ceil(len(training) / config["effective_batch"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, values["warmup_steps"], per_epoch * config["epochs"])
    best, best_epoch, bad, steps = -math.inf, None, 0, 0
    started = time.monotonic()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        loader = DataLoader(training, batch_size=config["microbatch"], shuffle=True, generator=generator, num_workers=0)
        optimizer.zero_grad(set_to_none=True)
        seen, group, total_loss = 0, 0, 0.0
        epoch_started = last_update = time.monotonic()
        update(directory, "TRAINING", phase="native_diffgrm", epoch=epoch, seen=0, train_total=len(training))
        for index, batch in enumerate(loader):
            result = model(batch)
            if not torch.isfinite(result.loss):
                raise FloatingPointError("Nonfinite native training loss")
            size = batch["history_sid"].size(0)
            (result.loss * size).backward()
            seen += size
            group += size
            total_loss += float(result.loss.detach()) * size
            if group == config["effective_batch"] or index + 1 == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group)
                torch.nn.utils.clip_grad_norm_(model.parameters(), values["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                group = 0
            current = time.monotonic()
            if current - last_update >= 30:
                info = {"epoch": epoch, "seen": seen, "train_total": len(training), "optimizer_steps": steps,
                        "loss": total_loss / seen, "learning_rate": scheduler.get_last_lr()[0],
                        "epoch_elapsed_seconds": current - epoch_started,
                        "epoch_eta_seconds": (len(training) - seen) * (current - epoch_started) / seen}
                update(directory, "TRAINING", phase="native_diffgrm", **info)
                event(directory, event="training_progress", **info)
                last_update = current
        event(directory, event="epoch_completed", epoch=epoch, loss=total_loss / seen,
              optimizer_steps=steps, seconds=time.monotonic() - epoch_started)
        checkpoint(directory / "last.pt", model, epoch=epoch, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                   config=config, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        if epoch >= config["evaluation_start"] and epoch % config["evaluation_interval"] == 0:
            torch.cuda.empty_cache()
            result = evaluate(model, validation, mapping, config, directory, epoch, summary["baseline_validation"])
            score = result["metrics"]["ndcg@10"]
            if score > best:
                best, best_epoch, bad = score, epoch, 0
                checkpoint(directory / "best.pt", model, epoch=epoch, config=config, validation=result)
            else:
                bad += 1
            if epoch >= config["minimum_epochs"] and bad >= config["patience"]:
                break
    result = {"state": "COMPLETED", "best_epoch": best_epoch, "best_validation_ndcg@10": best,
              "historical_gram": summary["baseline_validation"], "delta_ndcg@10": best - summary["baseline_validation"]["ndcg@10"],
              "epochs_completed": epoch, "optimizer_steps": steps, "seconds": time.monotonic() - started,
              "matched_training_and_decoding_budget": False}
    write_json(directory / "result.json", result)
    update(directory, "COMPLETED", result=result)
    event(directory, event="completed", **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pipeline", "status"))
    parser.add_argument("--config", default="experiment/phase18/config/s18_diffgrm_native_beauty.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = ROOT / args.output
    if args.mode == "status":
        print((directory / "status.json").read_text())
        return
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    try:
        config = json.loads((ROOT / args.config).read_text())
        if config["effective_batch"] % config["microbatch"] or config["evaluation_batch"] > config["microbatch"]:
            raise ValueError("Invalid batch configuration")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        seed_all(config["seed"])
        update(directory, "INITIALIZING", phase="profile")
        if not torch.cuda.is_available():
            raise RuntimeError("Native candidate pipeline requires CUDA")
        values = official_config(config, directory, "cuda:0")
        write_json(directory / "config.json", config)
        write_json(directory / "official_resolved.json", {k: v for k, v in values.items() if k != "accelerator"})
        paths = [Path(__file__).resolve(), ROOT / args.config, ROOT / "experiment/phase18/core/diff_data.py",
                 ROOT / "experiment/phase18/run_stage18_diffgrm_native.sh"]
        for path in paths:
            dest = directory / "sources" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        import transformers, faiss, sentence_transformers
        write_json(directory / "environment.json", {"torch": torch.__version__, "transformers": transformers.__version__,
                    "numpy": np.__version__, "faiss": faiss.__version__, "sentence_transformers": sentence_transformers.__version__,
                    "python": sys.version, "gpu": torch.cuda.get_device_name(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                    "local_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths}})
        profile(config, values, directory, "cuda:0")
        update(directory, "PREPARING", phase="tokenizer")
        data, tokenizer, codes, mapping, summary = prepare(config, values, directory)
        dataset = SidDataset(data.train, codes, config["max_history"])
        longest = sorted(range(len(dataset)), key=lambda i: len(data.train[i]["history"]), reverse=True)[:config["microbatch"]]
        real_batch = next(iter(DataLoader(torch.utils.data.Subset(dataset, longest), batch_size=config["microbatch"])))
        profile(config, values, directory, "cuda:0", real_batch, tokenizer)
        train(config, values, directory, data, tokenizer, codes, mapping, summary)
    except BaseException as error:
        update(directory, "FAILED", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
