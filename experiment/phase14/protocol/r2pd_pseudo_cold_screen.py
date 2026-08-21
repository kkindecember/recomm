"""Stage 14-1 clean-base A0/A1/A2/A3 pseudo-cold GRAM screen.

The clean base and item-disjoint teacher are both trained without pseudo-cold
or real-cold interactions.  Held pseudo-cold events are opened only after all
arm checkpoints have been written, for evaluation.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config


REPO_ROOT = Path(__file__).resolve().parents[3]
GRAM_SRC = REPO_ROOT / "GRAM" / "src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))
if "utils" not in sys.modules:
    package = types.ModuleType("utils")
    package.__path__ = [str(GRAM_SRC / "utils")]
    sys.modules["utils"] = package

from model import create_model  # noqa: E402
from processor.Collator import CollatorGRAM  # noqa: E402

from pseudo_cold_teacher import ResidualUserProjector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-config", required=True)
    parser.add_argument("--backbone-path", required=True)
    parser.add_argument("--train-sequences", required=True)
    parser.add_argument("--held-events", required=True)
    parser.add_argument("--pseudo-cold-items", required=True)
    parser.add_argument("--real-cold-items", required=True)
    parser.add_argument("--item-path-file", required=True)
    parser.add_argument("--item-text-file", required=True)
    parser.add_argument("--similar-items-file", required=True)
    parser.add_argument("--item-embeddings", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--teacher-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-examples", type=int, default=1024)
    parser.add_argument("--eval-events", type=int, default=512)
    parser.add_argument("--base-epochs", type=int, default=2)
    parser.add_argument("--adapt-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--synthetic-chunk-size", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-cp", type=float, default=1.0)
    parser.add_argument("--mu-keep", type=float, default=1.0)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--recency-decay", type=float, default=0.85)
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--pipeline-smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def read_key_value(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line:
                continue
            key, separator, value = line.partition(" ")
            if not separator or key in rows:
                raise ValueError(f"{path}:{line_no}: malformed or duplicate key")
            rows[key] = value
    return rows


def read_train_sequences(path: Path) -> list[tuple[str, list[str]]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            rows.append((str(row["user_id"]), [str(item) for item in row["train_items"]]))
    return rows


def read_held_events(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            rows.append(
                {
                    "user_id": str(row["user_id"]),
                    "target_item": str(row["target_item"]),
                    "visible_history": [str(item) for item in row["visible_history"]],
                    "train_prefix_position": int(row["train_prefix_position"]),
                }
            )
    return rows


def deterministic_rank(seed: int, *values: object) -> str:
    return hashlib.sha256(":".join([str(seed), *map(str, values)]).encode()).hexdigest()


def load_paths(path: Path) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            item, separator, encoded = line.partition(" |")
            if not separator or item in rows:
                raise ValueError(f"{path}:{line_no}: malformed or duplicate path")
            rows[item] = tuple(encoded.split("|"))
    if len(set(rows.values())) != len(rows):
        raise ValueError("Item path collision")
    return rows


def build_filtered_item_inputs(
    paths: dict[str, tuple[str, ...]],
    texts: dict[str, str],
    similar_path: Path,
    forbidden: set[str],
    top_k: int,
) -> tuple[dict[str, str], dict]:
    similar: dict[str, list[str]] = {}
    with similar_path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            parts = raw.strip().split()
            if not parts or parts[0] == "anchor":
                continue
            if parts[0] in similar:
                raise ValueError(f"{similar_path}:{line_no}: duplicate anchor")
            similar[parts[0]] = parts[1:]
    if set(paths) != set(texts) or not set(paths).issubset(similar):
        raise ValueError("Path/text/similar catalogs differ")
    outputs: dict[str, str] = {}
    removed = 0
    for item in paths:
        kept = []
        for candidate in similar[item]:
            if candidate in forbidden:
                removed += 1
                continue
            if candidate in paths:
                kept.append(candidate)
            if len(kept) == top_k:
                break
        lexical = "|".join(paths[item])
        cf_verbal = ["|".join(paths[candidate]) for candidate in kept]
        body = f"similar items: {', '.join(cf_verbal)}; {texts[item]}".strip()
        outputs[item] = f"item: {lexical}; {body}"
    return outputs, {"forbidden_similar_edges_removed": removed, "top_k": top_k}


def clean_transitions(
    sequences: list[tuple[str, list[str]]],
    forbidden: set[str],
    max_history: int,
    seed: int,
    limit: int,
) -> list[dict]:
    rows: list[dict] = []
    for user, items in sequences:
        if any(item in forbidden for item in items):
            raise RuntimeError(f"Forbidden item in clean sequence: {user}")
        for position in range(1, len(items)):
            rows.append(
                {
                    "user_id": user,
                    "position": position,
                    "history": items[max(0, position - max_history):position],
                    "target": items[position],
                }
            )
    rows.sort(key=lambda row: deterministic_rank(seed, row["user_id"], row["position"]))
    return rows[: min(limit, len(rows))]


def make_model_sample(
    row: dict,
    item_inputs: dict[str, str],
    item_paths: dict[str, tuple[str, ...]],
    item_to_cfid: dict[str, int],
) -> dict:
    history = row["history"]
    reversed_history = history[::-1]
    lexical_history = ["|".join(item_paths[item]) for item in reversed_history]
    prompt = f"What would user purchase after {' ; '.join(lexical_history)} ?"
    target = row.get("target")
    return {
        "input": [prompt] + [item_inputs[item] for item in reversed_history],
        "output": "|".join(item_paths[target]) if target is not None else "|".join(item_paths[history[-1]]),
        "user_id": row["user_id"],
        "history_item_ids": [item_to_cfid[item] for item in reversed_history],
        "target_item_id": item_to_cfid[target] if target is not None else 0,
    }


class TransitionDataset(Dataset):
    def __init__(self, rows: list[dict], item_inputs, item_paths, item_to_cfid):
        self.rows = rows
        self.item_inputs = item_inputs
        self.item_paths = item_paths
        self.item_to_cfid = item_to_cfid

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return make_model_sample(
            self.rows[index], self.item_inputs, self.item_paths, self.item_to_cfid
        )


def collator_args(historical: dict) -> SimpleNamespace:
    return SimpleNamespace(
        item_prompt_max_len=historical["item_prompt_max_len"],
        target_max_len=historical["target_max_len"],
        max_his=historical["max_his"],
        item_id_type=historical["item_id_type"],
        hierarchical_id_type=historical["hierarchical_id_type"],
    )


def configure_fresh_model(
    historical: dict, backbone_path: Path, device: torch.device, seed: int
):
    torch.manual_seed(seed)
    config = T5Config.from_pretrained(str(backbone_path), local_files_only=True)
    config.max_seq_len = historical["item_prompt_max_len"]
    config.max_item_num = historical["max_his"]
    config.use_position_embedding = historical["use_position_embedding"]
    config.sample_num = historical["sample_num"]
    config.cf0_enabled = False
    config.cf0_arm = "A"
    config.cf0_num_items = 0
    config.hi_gram_enabled = False
    config.hi_gram_include_user_prompt = False
    backbone = AutoModelForSeq2SeqLM.from_pretrained(
        str(backbone_path), config=config, local_files_only=True
    )
    model = create_model("gram", config=config)
    model.load_t5(backbone.state_dict())
    del backbone
    return model.to(device)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def warm_forward(model, batch: dict):
    return model(
        input_ids=batch["item_text_ids"],
        attention_mask=batch["item_text_masks"],
        labels=batch["target_ids"],
        return_dict=True,
    )


def train_clean_base(
    model,
    loader,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
    status,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch_to_device(batch, next(model.parameters()).device)
            optimizer.zero_grad(set_to_none=True)
            loss = warm_forward(model, batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        record = {"epoch": epoch, "warm_ce": total / count}
        history.append(record)
        status("clean_base_training", f"epoch {epoch}/{epochs} warm_ce={record['warm_ce']:.6f}")
    torch.save(model.state_dict(), output_dir / "clean_base.pt")
    return history


def load_embedding_payload(path: Path):
    payload = torch.load(path, map_location="cpu")
    item_ids = [str(item) for item in payload["item_ids"]]
    embeddings = F.normalize(payload["embeddings"].float(), dim=1)
    return item_ids, embeddings


def load_teacher(path: Path, embeddings: torch.Tensor, device: torch.device):
    payload = torch.load(path, map_location="cpu")
    model = ResidualUserProjector(
        embeddings.shape[1], int(payload["hidden_dim"]), float(payload["dropout"])
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def recency_history(history: list[int], embeddings: torch.Tensor, decay: float) -> torch.Tensor:
    values = embeddings[torch.tensor(history, dtype=torch.long)]
    ages = torch.arange(len(history) - 1, -1, -1, dtype=values.dtype)
    weights = decay**ages
    return F.normalize((values * weights[:, None]).sum(0) / weights.sum(), dim=0)


def teacher_candidates(
    rows: list[dict],
    teacher,
    item_ids: list[str],
    embeddings: torch.Tensor,
    item_to_idx: dict[str, int],
    top_m: int,
    temperature: float,
    decay: float,
    device: torch.device,
) -> tuple[list[dict], dict]:
    catalog = embeddings.to(device)
    outputs: list[dict] = []
    tail = []
    with torch.no_grad():
        for offset in range(0, len(rows), 128):
            batch = rows[offset:offset + 128]
            histories = torch.stack(
                [recency_history([item_to_idx[item] for item in row["history"]], embeddings, decay) for row in batch]
            ).to(device)
            scores = teacher(histories) @ catalog.T
            probabilities = F.softmax(scores / temperature, dim=1)
            mass, indices = torch.topk(probabilities, k=top_m, dim=1)
            score_top, _ = torch.topk(scores, k=top_m, dim=1)
            entropy = -(F.softmax(score_top / temperature, dim=1) * F.log_softmax(score_top / temperature, dim=1)).sum(dim=1) / math.log(top_m)
            margins = score_top[:, 0] - score_top[:, 1]
            for row_index in range(len(batch)):
                candidates = [item_ids[index] for index in indices[row_index].cpu().tolist()]
                masses = mass[row_index].cpu().tolist()
                tail_mass = 1.0 - sum(masses)
                tail.append(tail_mass)
                outputs.append(
                    {
                        "items": candidates,
                        "mass": masses,
                        "tail_mass": tail_mass,
                        "margin": float(margins[row_index]),
                        "normalized_entropy": float(entropy[row_index]),
                    }
                )
    return outputs, {
        "mean_tail_mass": float(np.mean(tail)),
        "max_tail_mass": float(np.max(tail)),
        "min_tail_mass": float(np.min(tail)),
    }


def history_confidence(record: dict, rule: dict) -> float:
    margin_denominator = max(rule["margin_q75"] - rule["margin_q25"], 1e-12)
    entropy_denominator = max(
        rule["normalized_entropy_q75"] - rule["normalized_entropy_q25"], 1e-12
    )
    margin = np.clip((record["margin"] - rule["margin_q25"]) / margin_denominator, 0, 1)
    entropy = np.clip(
        (rule["normalized_entropy_q75"] - record["normalized_entropy"]) / entropy_denominator,
        0,
        1,
    )
    return float(max(0.1, 0.5 * (margin + entropy)))


def encode_paths(tokenizer, item_paths: dict[str, tuple[str, ...]]) -> dict[str, tuple[int, ...]]:
    outputs = {}
    for item, tokens in item_paths.items():
        encoded = tokenizer.encode("|".join(tokens))
        filtered = [token for token in encoded if token not in (1820, 9175)]
        if not filtered or filtered[-1] != tokenizer.eos_token_id:
            raise ValueError(f"Encoded path lacks EOS: {item}")
        outputs[item] = tuple(filtered[:-1])
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("Tokenized path collision")
    return outputs


def candidate_labels(paths: list[tuple[int, ...]], eos: int, device: torch.device) -> torch.Tensor:
    length = max(len(path) + 1 for path in paths)
    labels = torch.full((len(paths), length), -100, dtype=torch.long, device=device)
    for row, path in enumerate(paths):
        values = list(path) + [eos]
        labels[row, : len(values)] = torch.tensor(values, device=device)
    return labels


def prefix_descendant_counts(paths: list[tuple[int, ...]]) -> collections.Counter:
    counts = collections.Counter()
    for path in paths:
        for depth in range(len(path) + 1):
            counts[path[:depth]] += 1
    return counts


def path_weight_normalizer(
    masses: list[float], paths: list[tuple[int, ...]], history_conf: float
) -> float:
    descendants = prefix_descendant_counts(paths)
    normalizer = 0.0
    for mass, path in zip(masses, paths):
        for depth in range(len(path) + 1):
            prefix = path[:depth]
            coverage = 0.5 + 0.5 * min(1.0, descendants[prefix] / 5.0)
            normalizer += float(mass) * history_conf * coverage
    return normalizer


def weighted_path_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masses: list[float],
    paths: list[tuple[int, ...]],
    history_conf: float,
) -> tuple[torch.Tensor, float]:
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none", ignore_index=-100
    ).reshape(labels.shape)
    descendants = prefix_descendant_counts(paths)
    weights = torch.zeros_like(token_loss)
    normalizer = 0.0
    for row, (mass, path) in enumerate(zip(masses, paths)):
        for depth in range(len(path) + 1):
            prefix = path[:depth]
            coverage = 0.5 + 0.5 * min(1.0, descendants[prefix] / 5.0)
            weight = float(mass) * history_conf * coverage
            if labels[row, depth] >= 0:
                weights[row, depth] = weight
                normalizer += weight
    return (token_loss * weights).sum(), normalizer


def legal_children(paths: dict[str, tuple[int, ...]], eos: int) -> dict[tuple[int, ...], tuple[int, ...]]:
    children: dict[tuple[int, ...], set[int]] = collections.defaultdict(set)
    for path in paths.values():
        for depth, token in enumerate((*path, eos)):
            children[path[:depth]].add(token)
    return {prefix: tuple(sorted(values)) for prefix, values in children.items()}


def retention_kl(current_logits, frozen_logits, labels, children) -> torch.Tensor:
    losses = []
    for row in range(labels.shape[0]):
        prefix: tuple[int, ...] = ()
        for depth in range(labels.shape[1]):
            token = int(labels[row, depth])
            if token < 0:
                break
            legal = children[prefix]
            indices = torch.tensor(legal, device=current_logits.device)
            student = current_logits[row, depth, indices]
            teacher = frozen_logits[row, depth, indices].detach()
            q = F.softmax(teacher, dim=-1)
            losses.append(torch.sum(q * (F.log_softmax(teacher, dim=-1) - F.log_softmax(student, dim=-1))))
            if token != 1:
                prefix = (*prefix, token)
    return torch.stack(losses).mean() if losses else current_logits.sum() * 0.0


def adapt_arm(
    arm: str,
    model,
    frozen,
    loader,
    rows,
    teacher_records,
    token_paths,
    children,
    tokenizer,
    epochs,
    learning_rate,
    lambda_cp,
    mu_keep,
    synthetic_chunk,
    confidence_rule,
    status,
) -> list[dict]:
    if arm == "A0":
        return []
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    device = next(model.parameters()).device
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = collections.Counter()
        for batch_index, batch in enumerate(loader):
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = warm_forward(model, batch)
            warm_loss = outputs.loss
            keep = warm_loss * 0.0
            if arm == "A3":
                frozen.eval()
                with torch.no_grad():
                    frozen_outputs = warm_forward(frozen, batch)
                keep = retention_kl(outputs.logits, frozen_outputs.logits, batch["target_ids"], children)
            (warm_loss + (mu_keep * keep if arm == "A3" else 0.0)).backward()
            kd_value = 0.0
            first = batch_index * loader.batch_size
            batch_rows = rows[first:first + len(batch["user_ids"])]
            batch_teachers = teacher_records[first:first + len(batch["user_ids"])]
            for local, (row, teacher_record) in enumerate(zip(batch_rows, batch_teachers)):
                model_sample = make_model_sample(row, loader.dataset.item_inputs, loader.dataset.item_paths, loader.dataset.item_to_cfid)
                single = loader.collate_fn([model_sample])
                single = batch_to_device(single, device)
                if arm == "A1":
                    items = teacher_record["items"][:1]
                    masses = [1.0]
                    hist_conf = 1.0
                else:
                    items = teacher_record["items"]
                    masses = teacher_record["mass"]
                    hist_conf = history_confidence(teacher_record, confidence_rule)
                paths = [token_paths[item] for item in items]
                # Compute Z without keeping a graph, then backpropagate bounded chunks.
                normalizer = path_weight_normalizer(masses, paths, hist_conf)
                for offset in range(0, len(paths), synthetic_chunk):
                    chunk_paths = paths[offset:offset + synthetic_chunk]
                    chunk_masses = masses[offset:offset + synthetic_chunk]
                    labels = candidate_labels(chunk_paths, tokenizer.eos_token_id, device)
                    repeat = len(chunk_paths)
                    result = model(
                        input_ids=single["item_text_ids"].expand(repeat, -1, -1),
                        attention_mask=single["item_text_masks"].expand(repeat, -1, -1),
                        labels=labels,
                        return_dict=True,
                    )
                    numerator, _ = weighted_path_nll(
                        result.logits, labels, chunk_masses, chunk_paths, hist_conf
                    )
                    kd_chunk = lambda_cp * numerator / max(normalizer, 1e-12)
                    kd_chunk.backward()
                    kd_value += float(kd_chunk.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["warm"] += float(warm_loss.detach())
            totals["keep"] += float(keep.detach())
            totals["kd"] += kd_value / max(len(batch_rows), 1)
            totals["batches"] += 1
        record = {
            "epoch": epoch,
            "warm_ce": totals["warm"] / totals["batches"],
            "prefix_kd": totals["kd"] / totals["batches"],
            "retention": totals["keep"] / totals["batches"],
        }
        history.append(record)
        status(f"{arm}_adaptation", f"epoch {epoch}/{epochs} {record}")
    return history


def normalize_generated(sequence: torch.Tensor, eos: int) -> tuple[int, ...]:
    values = [int(value) for value in sequence.tolist()]
    if values and values[0] == 0:
        values = values[1:]
    output = []
    for value in values:
        if value in (0, eos):
            break
        output.append(value)
    return tuple(output)


def evaluate_arm(model, events, item_inputs, item_paths, item_to_cfid, token_paths, children, tokenizer, collator, beam_size, device, status, arm):
    rows = []
    model.eval()
    max_length = max(len(path) for path in token_paths.values()) + 2
    for index, event in enumerate(events):
        sample_row = {
            "user_id": event["user_id"],
            "history": event["visible_history"],
            "target": event["target_item"],
        }
        batch = batch_to_device(
            collator([make_model_sample(sample_row, item_inputs, item_paths, item_to_cfid)]), device
        )

        def allowed(_batch_id: int, input_ids: torch.Tensor):
            prefix = tuple(int(value) for value in input_ids.tolist()[1:])
            return list(children.get(prefix, ()))

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["item_text_ids"],
                attention_mask=batch["item_text_masks"],
                max_length=max_length,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                prefix_allowed_tokens_fn=allowed,
                early_stopping=True,
            )
        generated_paths = [normalize_generated(sequence, tokenizer.eos_token_id) for sequence in generated]
        target_path = token_paths[event["target_item"]]
        try:
            rank = generated_paths.index(target_path) + 1
        except ValueError:
            rank = None
        longest = max(
            (next((depth for depth, (left, right) in enumerate(zip(target_path, candidate), 1) if left != right), min(len(target_path), len(candidate)) + 1) - 1 for candidate in generated_paths),
            default=0,
        )
        rows.append(
            {
                "user_id": event["user_id"],
                "target_item": event["target_item"],
                "rank": rank,
                "mrr": 0.0 if rank is None else 1.0 / rank,
                "recall50": float(rank is not None and rank <= 50),
                "longest_prefix_survival": longest,
                "normalized_prefix_survival": longest / len(target_path),
            }
        )
        if (index + 1) % 32 == 0:
            status(f"{arm}_evaluation", f"{index + 1}/{len(events)} held events")
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "exact_path_mrr": float(np.mean([row["mrr"] for row in rows])),
        "recall_at_50": float(np.mean([row["recall50"] for row in rows])),
        "mean_prefix_survival": float(np.mean([row["normalized_prefix_survival"] for row in rows])),
        "hit_events": int(sum(row["recall50"] for row in rows)),
        "unique_target_items": len({row["target_item"] for row in rows}),
    }


def item_paired_bootstrap(a2: list[dict], a1: list[dict], resamples: int, seed: int) -> dict:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for left, right in zip(a2, a1):
        if left["user_id"] != right["user_id"] or left["target_item"] != right["target_item"]:
            raise ValueError("A2/A1 evaluation rows are not paired")
        grouped[left["target_item"]].append(left["mrr"] - right["mrr"])
    values = np.array([np.mean(grouped[item]) for item in sorted(grouped)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return {
        "unit": "item",
        "n_items": len(values),
        "point": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "resamples": resamples,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in {"status.json", "run.log", "gpu_telemetry.csv"}]
    if unexpected:
        raise FileExistsError(f"Refusing existing screen artifacts: {unexpected}")
    experiment_id = "GRAM_PHASE14_STAGE14_1_PSEUDO_COLD_SCREEN_TOYS"

    def status(stage: str, reason: str, state: str = "running"):
        atomic_json(
            output_dir / "status.json",
            {
                "experiment_id": experiment_id,
                "status": state,
                "stage": stage,
                "reason": reason,
                "updated_at_epoch": time.time(),
                "automatic_retry": False,
                "test_opened": False,
                "held_ground_truth_opened_for_training": False,
            },
        )

    status("preflight", "Validating frozen M2 inputs.")
    input_paths = {
        name: Path(value).resolve()
        for name, value in {
            "historical_config": args.historical_config,
            "train_sequences": args.train_sequences,
            "held_events": args.held_events,
            "pseudo_cold_items": args.pseudo_cold_items,
            "real_cold_items": args.real_cold_items,
            "item_path_file": args.item_path_file,
            "item_text_file": args.item_text_file,
            "similar_items_file": args.similar_items_file,
            "item_embeddings": args.item_embeddings,
            "teacher_checkpoint": args.teacher_checkpoint,
            "teacher_summary": args.teacher_summary,
        }.items()
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if name != "held_events" and "held_ground_truth" in str(path):
            raise ValueError(f"Training input points into held ground truth: {name}")
    backbone_path = Path(args.backbone_path).resolve()
    if not backbone_path.is_dir():
        raise FileNotFoundError(backbone_path)
    backbone_weight_candidates = [
        backbone_path / "pytorch_model.bin",
        backbone_path / "model.safetensors",
    ]
    backbone_weight = next((path for path in backbone_weight_candidates if path.is_file()), None)
    if backbone_weight is None:
        raise FileNotFoundError(f"No PyTorch backbone weights under {backbone_path}")
    input_paths["backbone_config"] = backbone_path / "config.json"
    input_paths["backbone_weights"] = backbone_weight
    historical = json.loads(input_paths["historical_config"].read_text())
    teacher_summary = json.loads(input_paths["teacher_summary"].read_text())
    pseudo = read_set(input_paths["pseudo_cold_items"])
    real_cold = read_set(input_paths["real_cold_items"])
    forbidden = pseudo | real_cold
    item_paths = load_paths(input_paths["item_path_file"])
    texts = read_key_value(input_paths["item_text_file"])
    item_inputs, input_report = build_filtered_item_inputs(
        item_paths, texts, input_paths["similar_items_file"], forbidden, historical["top_k_similar_item"]
    )
    item_to_cfid = {item: index + 1 for index, item in enumerate(sorted(item_paths))}
    sequences = read_train_sequences(input_paths["train_sequences"])
    train_rows = clean_transitions(
        sequences, forbidden, args.max_history, args.seed, args.train_examples
    )
    tokenizer = AutoTokenizer.from_pretrained(str(backbone_path), local_files_only=True)
    collator = CollatorGRAM(tokenizer, collator_args(historical), mode="train")
    dataset = TransitionDataset(train_rows, item_inputs, item_paths, item_to_cfid)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    token_paths = encode_paths(tokenizer, item_paths)
    children = legal_children(token_paths, tokenizer.eos_token_id)
    item_ids, embeddings = load_embedding_payload(input_paths["item_embeddings"])
    if set(item_ids) != set(item_paths):
        raise ValueError("Embedding/path catalogs differ")
    item_to_idx = {item: index for index, item in enumerate(item_ids)}
    teacher = load_teacher(input_paths["teacher_checkpoint"], embeddings, device)
    teacher_records, teacher_report = teacher_candidates(
        train_rows,
        teacher,
        item_ids,
        embeddings,
        item_to_idx,
        args.top_m,
        teacher_summary["selected_score_temperature"],
        args.recency_decay,
        device,
    )
    del teacher
    torch.cuda.empty_cache()

    model = configure_fresh_model(historical, backbone_path, device, args.seed)
    base_history = train_clean_base(
        model, loader, args.base_epochs, args.learning_rate, output_dir, status
    )
    clean_state = copy.deepcopy(model.state_dict())
    arm_histories = {"A0": []}
    checkpoints = {"A0": output_dir / "arm_A0.pt"}
    torch.save(clean_state, checkpoints["A0"])
    confidence_rule = teacher_summary["calibration"]["confidence_rule"]
    for arm in ("A1", "A2", "A3"):
        model.load_state_dict(clean_state, strict=True)
        frozen = None
        if arm == "A3":
            frozen = configure_fresh_model(historical, backbone_path, device, args.seed)
            frozen.load_state_dict(clean_state, strict=True)
            frozen.eval()
            for parameter in frozen.parameters():
                parameter.requires_grad = False
        arm_histories[arm] = adapt_arm(
            arm,
            model,
            frozen,
            loader,
            train_rows,
            teacher_records,
            token_paths,
            children,
            tokenizer,
            args.adapt_epochs,
            args.learning_rate,
            args.lambda_cp,
            args.mu_keep,
            args.synthetic_chunk_size,
            confidence_rule,
            status,
        )
        checkpoints[arm] = output_dir / f"arm_{arm}.pt"
        torch.save(model.state_dict(), checkpoints[arm])
        del frozen
        torch.cuda.empty_cache()

    # Held ground truth is opened only after all training checkpoints exist.
    held = read_held_events(input_paths["held_events"])
    if any(
        item in forbidden - pseudo
        for row in held
        for item in row["visible_history"]
    ):
        raise RuntimeError("Real-cold item in held visible history")
    held.sort(
        key=lambda row: deterministic_rank(
            args.seed, row["user_id"], row["train_prefix_position"], row["target_item"]
        )
    )
    held = held[: min(args.eval_events, len(held))]
    arm_rows = {}
    arm_metrics = {}
    for arm in ("A0", "A1", "A2", "A3"):
        model.load_state_dict(torch.load(checkpoints[arm], map_location="cpu"), strict=True)
        model.to(device)
        rows = evaluate_arm(
            model,
            held,
            item_inputs,
            item_paths,
            item_to_cfid,
            token_paths,
            children,
            tokenizer,
            collator,
            args.beam_size,
            device,
            status,
            arm,
        )
        arm_rows[arm] = rows
        arm_metrics[arm] = summarize_rows(rows)
        with (output_dir / f"predictions_{arm}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    paired = item_paired_bootstrap(
        arm_rows["A2"], arm_rows["A1"], args.bootstrap_resamples, args.seed
    )
    gates = {
        "a2_minus_a1_mrr_ci_lower_gt_zero": paired["ci95"][0] > 0,
        "a2_vs_a0_recall50_non_degradation": arm_metrics["A2"]["recall_at_50"] >= arm_metrics["A0"]["recall_at_50"],
        "a2_vs_a0_beam_survival_non_degradation": arm_metrics["A2"]["mean_prefix_survival"] >= arm_metrics["A0"]["mean_prefix_survival"],
    }
    if args.pipeline_smoke:
        verdict = "PASS_STAGE14_1_PIPELINE_SMOKE_ONLY"
    else:
        verdict = "PASS_STAGE14_1_PSEUDO_COLD_SCREEN" if all(gates.values()) else "FAIL_STOP_PATH_TRANSFER_STAGE14_1"
    hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    config = vars(args).copy()
    config.update(
        {
            "experiment_id": experiment_id,
            "teacher_temperature": teacher_summary["selected_score_temperature"],
            "test_opened": False,
            "held_ground_truth_opened_after_training_only": True,
        }
    )
    atomic_json(output_dir / "config.json", config)
    atomic_json(output_dir / "input_file_sha256.json", hashes)
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened_inputs": [
                {"role": name, "path": str(path), "sha256": hashes[name]}
                for name, path in input_paths.items()
            ],
            "held_ground_truth_open_phase": "evaluation_after_all_arm_checkpoints",
            "test_opened": False,
        },
    )
    summary = {
        "experiment_id": experiment_id,
        "status": "completed",
        "verdict": verdict,
        "arm_metrics": arm_metrics,
        "paired_a2_minus_a1_mrr": paired,
        "gates": gates,
        "scientific_gate_eligible": not args.pipeline_smoke,
        "clean_base_history": base_history,
        "arm_training_history": arm_histories,
        "teacher_report": teacher_report,
        "input_filter_report": input_report,
        "test_opened": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "clean_base_from_fresh_backbone": True,
            "historical_v0_checkpoint_used": False,
            "historical_r2_teacher_used": False,
            "pseudo_or_real_cold_interactions_used_for_training": False,
            "held_ground_truth_used_for_training": False,
            "held_ground_truth_used_for_evaluation": True,
            "test_opened": False,
        },
    )
    status("finished", verdict, "completed")
    print(json.dumps({"verdict": verdict, "gates": gates}))


if __name__ == "__main__":
    main()
