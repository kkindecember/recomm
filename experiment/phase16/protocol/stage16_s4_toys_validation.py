#!/usr/bin/env python3
"""S16-4 frozen Toys standalone validation, one immutable arm per process.

The module deliberately separates GPU arm execution from the CPU finalizer.
Formal arm outputs are write-once.  ``--discard-output`` is reserved for the
post-terminal, non-promotional occupancy loop and never touches formal roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


ROOT = Path(__file__).resolve().parents[3]
PHASE15_PROTOCOL = ROOT / "experiment" / "phase15" / "protocol"
PHASE16_PROTOCOL = ROOT / "experiment" / "phase16" / "protocol"
for directory in (PHASE15_PROTOCOL, PHASE16_PROTOCOL):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common_adapter import read_projected_sequences, read_validation_predictions  # noqa: E402
from genrecedit_faithful import (  # noqa: E402
    OneOneGenerationDeltaContext,
    build_one_one_position_bundles,
)
from resource_probe import load_gram  # noqa: E402
from saux_formal_train import OfficialUniSRecDrafterGRAM  # noqa: E402
from specgr_contract_smoke import read_metadata, read_paths  # noqa: E402
from specgr_faithful import GRAMSelfDrafter, constrained_draft  # noqa: E402


FORMAL_ARMS = ("S-AUX", "S-PLUS-CTRL", "S-PLUS", "G-RIDGE")
SCIENTIFIC_ARMS = ("F0", "R2", *FORMAL_ARMS)
EXPECTED_CONTROLS = {
    "S-AUX": "F0",
    "S-PLUS-CTRL": "F0",
    "S-PLUS": "S-PLUS-CTRL",
    "G-RIDGE": "F0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def cuda_device_index(device: torch.device) -> int:
    """Return the integer device form required by the frozen PyTorch 1.11 allocator API."""
    if device.type != "cuda":
        raise ValueError("CUDA device required")
    return 0 if device.index is None else int(device.index)


def reset_peak_memory_stats_compat(device: torch.device) -> int:
    torch.cuda.init()
    index = cuda_device_index(device)
    torch.cuda.reset_peak_memory_stats(index)
    return index


def verify_regular(root: Path, declaration: Mapping[str, str], label: str) -> Path:
    relative = str(declaration["path"])
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing/non-regular frozen input: {label}={relative}")
    if sha256_file(path) != declaration["sha256"]:
        raise ValueError(f"Frozen input SHA drift: {label}")
    return path


def read_set(path: Path) -> set[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"Empty/duplicate frozen item set: {path}")
    return set(rows)


def ranking_metrics(ranking: Sequence[str], target: str) -> dict[str, float | int | None]:
    rank = ranking.index(target) + 1 if target in ranking else None
    return {
        "rank": rank,
        "hit@50": int(rank is not None and rank <= 50),
        "ndcg@10": 1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], *, require_all_subsets: bool = True
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for subset in ("overall", "cold", "warm"):
        selected = [
            row for row in rows
            if subset == "overall" or bool(row["is_cold"]) == (subset == "cold")
        ]
        if not selected:
            if require_all_subsets:
                raise ValueError(f"Empty validation subset: {subset}")
            continue
        result[subset] = {
            "events": len(selected),
            "hit@50": sum(float(row["metrics"]["hit@50"]) for row in selected) / len(selected),
            "ndcg@10": sum(float(row["metrics"]["ndcg@10"]) for row in selected) / len(selected),
        }
    return result


def tokenize_history(
    history: Sequence[str],
    metadata: Mapping[str, str],
    paths: Mapping[str, Sequence[str]],
    tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    history = list(history)[-20:]
    if not history or any(item not in metadata or item not in paths for item in history):
        raise ValueError("Validation history is empty or contains an unknown item")
    lexical = " > ".join("|".join(paths[item]) for item in history)
    passages = [f"What would user purchase after {lexical} ?"] + [
        metadata[item] for item in reversed(history)
    ]
    active = len(passages)
    passages.extend([""] * (21 - active))
    encoded = tokenizer(
        passages,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.reshape(1, 21, 128)
    attention = encoded.attention_mask.reshape(1, 21, 128)
    input_ids[:, active:] = int(tokenizer.pad_token_id)
    attention[:, active:] = 0
    return {"input_ids": input_ids.to(device), "attention_mask": attention.to(device)}


def encode_history(model, context: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = context["input_ids"]
    attention = context["attention_mask"]
    model.encoder.n_passages = int(input_ids.shape[1])
    flat_input = input_ids.reshape(input_ids.shape[0], -1)
    flat_mask = attention.reshape(attention.shape[0], -1)
    hidden = model.encoder(
        input_ids=flat_input,
        attention_mask=flat_mask,
        return_dict=True,
    )[0]
    return hidden, flat_mask


def catalog_children(
    token_paths: Mapping[str, Sequence[int]], eos_token_id: int
) -> tuple[dict[tuple[int, ...], list[int]], dict[tuple[int, ...], str]]:
    reverse: dict[tuple[int, ...], str] = {}
    children: dict[tuple[int, ...], set[int]] = {}
    for item, raw_path in token_paths.items():
        path = tuple(int(token) for token in raw_path)
        if not path or path in reverse:
            raise ValueError("Frozen tokenized catalog contains an empty/colliding path")
        reverse[path] = str(item)
        for depth, token in enumerate((*path, int(eos_token_id))):
            children.setdefault(path[:depth], set()).add(int(token))
    return {prefix: sorted(values) for prefix, values in children.items()}, reverse


@dataclass(frozen=True)
class BeamRow:
    prefix: tuple[int, ...]
    score: float


class LiveBeamStepper:
    """Output-equivalent variable-path beam steps used by guided re-drafting."""

    def __init__(
        self,
        *,
        model,
        encoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        children: Mapping[tuple[int, ...], Sequence[int]],
        reverse: Mapping[tuple[int, ...], str],
        decoder_start_token_id: int,
        eos_token_id: int,
        beam_size: int,
    ) -> None:
        self.model = model
        self.encoder_hidden = encoder_hidden
        self.encoder_mask = encoder_mask
        self.children = children
        self.reverse = reverse
        self.start = int(decoder_start_token_id)
        self.eos = int(eos_token_id)
        self.beam_size = int(beam_size)
        self.active = [BeamRow((), 0.0)]
        self.finished: dict[str, float] = {}
        self.depth = 0
        self.decoder_rows = 0

    def step(self) -> list[tuple[int, ...]]:
        if not self.active:
            self.depth += 1
            return []
        decoder = torch.tensor(
            [[self.start, *row.prefix] for row in self.active],
            dtype=torch.long,
            device=self.encoder_hidden.device,
        )
        repeated_hidden = self.encoder_hidden.repeat_interleave(len(self.active), dim=0)
        repeated_mask = self.encoder_mask.repeat_interleave(len(self.active), dim=0)
        output = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=repeated_hidden),
            attention_mask=repeated_mask,
            decoder_input_ids=decoder,
            use_cache=False,
            return_dict=True,
        )
        logp = F.log_softmax(output.logits[:, -1, :].float(), dim=-1)
        self.decoder_rows += len(self.active)
        candidates: list[tuple[float, int, int, tuple[int, ...]]] = []
        for parent_index, row in enumerate(self.active):
            allowed = self.children.get(row.prefix, ())
            if not allowed:
                continue
            for token in allowed:
                candidates.append(
                    (
                        row.score + float(logp[parent_index, int(token)]),
                        parent_index,
                        int(token),
                        row.prefix,
                    )
                )
        candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
        next_active: list[BeamRow] = []
        # HuggingFace beam search presents the top 2B candidates to the scorer.
        for score, _parent, token, prefix in candidates[: 2 * self.beam_size]:
            if token == self.eos:
                item = self.reverse.get(prefix)
                if item is not None:
                    # Official SpecGR beam scores are normalized over semantic
                    # digits; EOS only closes a variable-length path.
                    normalized = score if not prefix else score / len(prefix)
                    self.finished[item] = max(self.finished.get(item, float("-inf")), normalized)
                continue
            extended = (*prefix, token)
            if extended in self.children and len(next_active) < self.beam_size:
                next_active.append(BeamRow(extended, score))
        self.active = next_active
        self.depth += 1
        # A variable-length path that is already a complete item can still be
        # a live prefix of a longer item.  Keep it only when a non-EOS child
        # exists; an EOS-only row is complete and must not guide re-drafting.
        return [
            row.prefix
            for row in self.active
            if any(int(token) != self.eos for token in self.children.get(row.prefix, ()))
        ]

    def fallback(self, top_k: int) -> list[tuple[str, float]]:
        scores = dict(self.finished)
        for row in self.active:
            item = self.reverse.get(row.prefix)
            if item is not None:
                normalized = row.score / max(1, len(row.prefix))
                scores[item] = max(scores.get(item, float("-inf")), normalized)
        ranking = sorted(scores.items(), key=lambda value: (-value[1], value[0]))
        if len(ranking) < top_k:
            raise RuntimeError(f"Verifier beam produced only {len(ranking)} complete unique items")
        return ranking[:top_k]


def score_candidate_paths(
    *,
    model,
    encoder_hidden: torch.Tensor,
    encoder_mask: torch.Tensor,
    paths: Sequence[Sequence[int]],
    score_lengths: Sequence[int],
    chunk_size: int,
) -> torch.Tensor:
    if not paths or len(paths) != len(score_lengths):
        raise ValueError("Candidate paths/score lengths do not align")
    chunks: list[torch.Tensor] = []
    for start in range(0, len(paths), chunk_size):
        selected = [tuple(int(token) for token in path) for path in paths[start : start + chunk_size]]
        lengths = [int(value) for value in score_lengths[start : start + chunk_size]]
        width = max(map(len, selected))
        labels = torch.full(
            (len(selected), width), -100, dtype=torch.long, device=encoder_hidden.device
        )
        for row, path in enumerate(selected):
            labels[row, : len(path)] = torch.tensor(path, device=labels.device)
        hidden = encoder_hidden.repeat_interleave(len(selected), dim=0)
        mask = encoder_mask.repeat_interleave(len(selected), dim=0)
        output = model(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        active = labels.ne(-100)
        safe = labels.masked_fill(~active, 0)
        token_logp = F.log_softmax(output.logits.float(), dim=-1).gather(
            -1, safe.unsqueeze(-1)
        ).squeeze(-1)
        position = torch.arange(width, device=labels.device)[None, :]
        length_tensor = torch.tensor(lengths, device=labels.device)[:, None]
        selected_mask = active & (position < length_tensor)
        if bool((length_tensor[:, 0] < 1).any()) or bool(
            (length_tensor[:, 0] > active.sum(dim=1)).any()
        ):
            raise ValueError("Target-aware score length is outside candidate path")
        scores = (token_logp * selected_mask).sum(dim=1) / length_tensor[:, 0]
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Verifier candidate score is non-finite")
        chunks.append(scores)
    return torch.cat(chunks)


def official_finalize(
    verified: Sequence[tuple[str, float, bool]],
    beam_fallback: Sequence[tuple[str, float]],
    top_k: int,
) -> list[str]:
    candidate_ids = [item for item, _score, _accepted in verified]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("A faithful candidate was drafted more than once")
    accepted = [(item, float(score)) for item, score, flag in verified if flag]
    selected = accepted[:top_k]
    if len(selected) < top_k:
        drafted = set(candidate_ids)
        pool = [(item, float(score)) for item, score, flag in verified if not flag]
        pool.extend((item, float(score)) for item, score in beam_fallback if item not in drafted)
        pool.sort(key=lambda value: (-value[1], value[0]))
        used = {item for item, _score in selected}
        for item, score in pool:
            if item in used:
                continue
            selected.append((item, score))
            used.add(item)
            if len(selected) == top_k:
                break
    if len(selected) != top_k:
        raise RuntimeError("Faithful finalizer could not produce exact top-k")
    selected.sort(key=lambda value: (-value[1], value[0]))
    return [item for item, _score in selected]


def finite_unseen_constrained_draft(
    draft_logits: torch.Tensor, maximum_size: int
) -> torch.Tensor:
    """Draft up to ``maximum_size`` finite unseen candidates for one event.

    The pinned SpecGR helper always calls ``topk(k)``.  When live-prefix
    guidance leaves fewer than ``k`` candidates, that would return masked
    ``-inf`` entries.  Saturating at the finite count preserves the configured
    draft size as an upper bound without violating the live-beam constraint.
    """
    if draft_logits.ndim != 2 or draft_logits.shape[0] != 1:
        raise ValueError("Faithful finite drafting requires one event at a time")
    if isinstance(maximum_size, bool) or not isinstance(maximum_size, int) or maximum_size < 1:
        raise ValueError("Faithful maximum draft size must be a positive integer")
    finite_count = int(torch.isfinite(draft_logits[0]).sum().item())
    actual_size = min(maximum_size, finite_count)
    if actual_size == 0:
        return torch.empty((1, 0), dtype=torch.long, device=draft_logits.device)
    return constrained_draft(draft_logits, actual_size)


def faithful_rank(
    *,
    model,
    context: Mapping[str, torch.Tensor],
    draft_logits: torch.Tensor,
    ordered_items: Sequence[str],
    token_paths: Mapping[str, Sequence[int]],
    score_lengths: Mapping[str, int],
    tokenizer,
    draft_size: int,
    threshold: float,
    beam_size: int,
    candidate_chunk_size: int,
    encoder_state: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    if draft_logits.shape != (1, len(ordered_items)):
        raise ValueError("Drafter logits do not align with the frozen catalog")
    logits = draft_logits.clone()
    hidden, flat_mask = encoder_state if encoder_state is not None else encode_history(model, context)
    children, reverse = catalog_children(token_paths, int(tokenizer.eos_token_id))
    stepper = LiveBeamStepper(
        model=model,
        encoder_hidden=hidden,
        encoder_mask=flat_mask,
        children=children,
        reverse=reverse,
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        beam_size=beam_size,
    )
    verified: list[tuple[str, float, bool]] = []
    live_prefixes: list[tuple[int, ...]] = []
    max_depth = max(len(path) for path in token_paths.values())
    rounds = 0
    draft_capacity_shortfall_rounds = 0
    zero_finite_draft_rounds = 0
    for round_index in range(max_depth):
        if round_index > 1:
            live = set(live_prefixes)
            eligible = torch.tensor(
                [
                    len(token_paths[item]) >= round_index
                    and tuple(token_paths[item][:round_index]) in live
                    for item in ordered_items
                ],
                dtype=torch.bool,
                device=logits.device,
            )[None, :]
            logits = torch.where(eligible, logits, torch.full_like(logits, float("-inf")))
        finite_count = int(torch.isfinite(logits[0]).sum().item())
        if finite_count < draft_size:
            draft_capacity_shortfall_rounds += 1
        indices = finite_unseen_constrained_draft(logits, draft_size)[0]
        if indices.numel() == 0:
            zero_finite_draft_rounds += 1
        else:
            items = [ordered_items[int(index)] for index in indices]
            scores = score_candidate_paths(
                model=model,
                encoder_hidden=hidden,
                encoder_mask=flat_mask,
                paths=[token_paths[item] for item in items],
                score_lengths=[score_lengths[item] for item in items],
                chunk_size=candidate_chunk_size,
            )
            verified.extend(
                (item, float(score), bool(float(score) > threshold))
                for item, score in zip(items, scores.detach().cpu())
            )
        live_prefixes = stepper.step()
        rounds += 1
        if sum(int(flag) for _item, _score, flag in verified) >= beam_size:
            break
    accepted_count = sum(int(flag) for _item, _score, flag in verified)
    fallback = [] if accepted_count >= beam_size else stepper.fallback(beam_size)
    ranking = official_finalize(verified, fallback, beam_size)
    return ranking, {
        "rounds": rounds,
        "drafted": len(verified),
        "accepted": accepted_count,
        "redraft_rounds": max(0, rounds - 2),
        "draft_capacity_shortfall_rounds": draft_capacity_shortfall_rounds,
        "zero_finite_draft_rounds": zero_finite_draft_rounds,
        "beam_decoder_rows": stepper.decoder_rows,
        "candidate_verifier_forwards": len(verified),
        "candidate_reachability": len({item for item, _score, _flag in verified}) / len(ordered_items),
        "strict_acceptance": True,
        "live_beam_guidance": True,
    }


def encode_token_catalog(tokenizer, paths: Mapping[str, Sequence[str]]) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for item, path in paths.items():
        tokens = tuple(int(tokenizer.convert_tokens_to_ids(token)) for token in path)
        if not tokens or int(tokenizer.unk_token_id) in tokens:
            raise ValueError(f"Lexical tokenization failed: {item}")
        result[item] = tokens
    if len(set(result.values())) != len(result):
        raise ValueError("Frozen tokenized catalog paths collide")
    return result


def verifier_score_lengths(
    paths: Mapping[str, Sequence[int]],
    warm_items: set[str],
    cold_items: set[str],
    *,
    arm: str,
) -> dict[str, int]:
    if arm not in {"S-AUX", "S-PLUS"}:
        raise ValueError("Target-aware score lengths apply only to faithful SpecGR arms")
    warm_prefixes = {
        tuple(path[:depth])
        for item, path in paths.items()
        if item in warm_items
        for depth in range(1, len(path) + 1)
    }
    result: dict[str, int] = {}
    for item, path in paths.items():
        if item in warm_items:
            result[item] = len(path)
            continue
        if item not in cold_items or len(path) < 2:
            raise ValueError("Cold target-aware prefix contract failed")
        longest = max(
            (depth for depth in range(1, len(path) + 1) if tuple(path[:depth]) in warm_prefixes),
            default=0,
        )
        # Preserve the two distinct official policies.  Auxiliary SpecGR
        # clamps cold identifiers to at least two tokens.  Self-drafting
        # SpecGR++ ignores at most the final identifier token.
        minimum = 2 if arm == "S-AUX" else max(1, len(path) - 1)
        result[item] = min(len(path), max(minimum, longest))
    return result


def standard_beam_rank(
    *,
    model,
    context: Mapping[str, torch.Tensor],
    token_paths: Mapping[str, Sequence[int]],
    tokenizer,
    beam_size: int,
) -> list[str]:
    children, reverse = catalog_children(token_paths, int(tokenizer.eos_token_id))

    def allowed(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        prefix = tuple(int(value) for value in input_ids.detach().cpu().tolist()[1:])
        return children.get(prefix, [])

    generated = model.generate(
        input_ids=context["input_ids"],
        attention_mask=context["attention_mask"],
        max_length=max(len(path) for path in token_paths.values()) + 2,
        num_beams=beam_size,
        num_return_sequences=beam_size,
        prefix_allowed_tokens_fn=allowed,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=True,
    )
    ranking: list[str] = []
    for raw in generated.sequences.detach().cpu().tolist():
        suffix: list[int] = []
        for token in raw[1:]:
            if token == int(tokenizer.eos_token_id):
                break
            if token != int(tokenizer.pad_token_id):
                suffix.append(int(token))
        item = reverse.get(tuple(suffix))
        if item is None:
            raise RuntimeError("Strict beam produced a non-catalog path")
        ranking.append(item)
    if len(ranking) != beam_size or len(set(ranking)) != beam_size:
        raise RuntimeError("Strict beam violates exact unique top-50 contract")
    if not bool(torch.isfinite(generated.sequences_scores).all()):
        raise FloatingPointError("Strict beam score is non-finite")
    return ranking


def saux_state(
    *,
    checkpoint: Path,
    embedding_path: Path,
    retained_items: set[str],
    ordered_items: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(embedding_path, map_location="cpu")
    views = saux_embedding_views(
        item_ids=[str(item) for item in payload["item_ids"]],
        embeddings=payload["embeddings"].to(torch.float32),
        retained_items=retained_items,
        ordered_items=ordered_items,
    )
    train_embeddings = views["train_embeddings"]
    wrapper = OfficialUniSRecDrafterGRAM(train_embeddings).to(device)
    state = torch.load(checkpoint, map_location=device)
    wrapper.load_state_dict(state["model"], strict=True)
    wrapper.eval()
    with torch.inference_mode():
        candidates = F.normalize(
            wrapper.model.moe_adaptor(views["candidate_embeddings"].to(device)), dim=-1
        )
    return {
        "wrapper": wrapper,
        "history_index": views["history_index"],
        "history_embeddings": views["history_embeddings"],
        "candidate_embeddings": candidates,
    }


def saux_embedding_views(
    *,
    item_ids: Sequence[str],
    embeddings: torch.Tensor,
    retained_items: set[str],
    ordered_items: Sequence[str],
) -> dict[str, Any]:
    """Build official UniSRec train-state and inductive inference embedding views."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(item_ids):
        raise ValueError("S-AUX content embedding rows do not align with item IDs")
    if len(item_ids) != len(set(item_ids)) or set(item_ids) != set(ordered_items):
        raise ValueError("S-AUX content embedding/catalog universe drift")
    if not retained_items or not retained_items < set(ordered_items):
        raise ValueError("S-AUX retained training-item universe drift")
    source_index = {item: position for position, item in enumerate(item_ids)}
    train_order = sorted(retained_items)
    history_order = list(ordered_items)
    train_embeddings = torch.cat(
        [
            torch.zeros(1, embeddings.shape[1], dtype=embeddings.dtype),
            embeddings[[source_index[item] for item in train_order]],
        ],
        dim=0,
    )
    history_embeddings = torch.cat(
        [
            torch.zeros(1, embeddings.shape[1], dtype=embeddings.dtype),
            embeddings[[source_index[item] for item in history_order]],
        ],
        dim=0,
    )
    return {
        "train_embeddings": train_embeddings,
        "history_index": {item: position + 1 for position, item in enumerate(history_order)},
        "history_embeddings": history_embeddings,
        "candidate_embeddings": history_embeddings[1:],
    }


@torch.inference_mode()
def saux_logits(state: Mapping[str, Any], history: Sequence[str], device: torch.device) -> torch.Tensor:
    history = list(history)[-20:]
    history_index = state["history_index"]
    if any(item not in history_index for item in history):
        raise ValueError("S-AUX validation history contains an unknown catalog item")
    row = torch.zeros((1, 20), dtype=torch.long, device=device)
    row[0, : len(history)] = torch.tensor([history_index[item] for item in history], device=device)
    lengths = torch.tensor([len(history)], dtype=torch.long, device=device)
    content = state["history_embeddings"][row.cpu()].to(device)
    adapted = state["wrapper"].model.moe_adaptor(content)
    sequence = F.normalize(state["wrapper"].model.forward(row, adapted, lengths), dim=-1)
    return sequence @ state["candidate_embeddings"].T


def load_continued_model(
    *,
    historical: Path,
    base_checkpoint: Path,
    final_checkpoint: Path,
    arm: str,
    projection_dimension: int,
    device: torch.device,
) -> tuple[Any, GRAMSelfDrafter | None]:
    model = load_gram(historical, base_checkpoint, device)
    payload = torch.load(final_checkpoint, map_location=device)
    if payload.get("arm") != arm:
        raise ValueError(f"Continued checkpoint arm mismatch: {arm}")
    model.load_state_dict(payload["model"], strict=True)
    drafter = None
    if arm == "S-PLUS":
        drafter = GRAMSelfDrafter(model.config.d_model, projection_dimension).to(device)
        if payload.get("drafter") is None:
            raise ValueError("S-PLUS checkpoint lacks drafter state")
        drafter.load_state_dict(payload["drafter"], strict=True)
        drafter.eval()
    elif payload.get("drafter") is not None:
        raise ValueError("S-PLUS-CTRL checkpoint unexpectedly contains a drafter")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, drafter


def validate_config(config: Mapping[str, Any], arm: str) -> None:
    if config.get("schema_version") != "stage16_s4_toys_standalone_v1":
        raise ValueError("Unexpected S16-4 formal schema")
    if arm not in FORMAL_ARMS or tuple(config.get("formal_arms", ())) != FORMAL_ARMS:
        raise ValueError("S16-4 formal arm contract drift")
    physical_gpu = config.get("physical_gpu")
    if (
        config.get("seed") != 1502
        or isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
        or physical_gpu < 0
        or config.get("visible_gpu") != 0
    ):
        raise ValueError("S16-4 seed/GPU identity drift")
    if config.get("test_read") is not False or config.get("automatic_retry") is not False:
        raise ValueError("S16-4 test/retry boundary drift")
    faithful = config["faithful_inference"]
    if faithful.get("threshold") != -1.8 or faithful.get("acceptance") != "strict_gt":
        raise ValueError("S16-4 strict SpecGR acceptance drift")
    if faithful.get("guided_redraft") != "current_live_verifier_beam_prefixes":
        raise ValueError("S16-4 live-beam redraft drift")
    if (
        faithful.get("underfilled_live_round")
        != "draft_all_finite_unseen_then_advance_verifier_beam"
    ):
        raise ValueError("S16-4 underfilled live-round policy drift")
    if faithful.get("draft_size", {}).get("S-AUX") != 50 or faithful.get("draft_size", {}).get("S-PLUS") != 20:
        raise ValueError("S16-4 faithful draft-size drift")
    expected_prefix = {
        "S-AUX": "max_2_and_longest_warm_prefix",
        "S-PLUS": "max_path_length_minus_1_and_longest_warm_prefix",
    }
    if faithful.get("target_aware_prefix_length") != expected_prefix:
        raise ValueError("S16-4 target-aware prefix-length policy drift")
    saux_inference = config.get("saux_inference", {})
    if (
        saux_inference.get("checkpoint_state_universe") != "retained_warm_plus_padding"
        or saux_inference.get("history_content_universe") != "complete_catalog_plus_padding"
        or saux_inference.get("official_predict_semantics") is not True
    ):
        raise ValueError("S16-4 S-AUX official inductive semantics drift")
    saux_inputs = saux_inference.get("inputs", {})
    if set(saux_inputs) != {
        "source_training_config",
        "retained_warm_items",
        "pseudo_cold_items",
    }:
        raise ValueError("S16-4 S-AUX inductive input contract drift")
    if any(set(saux_inputs[name]) != {"path", "sha256"} for name in saux_inputs):
        raise ValueError("S16-4 S-AUX inductive input identity drift")


def run_arm(
    config_path: Path,
    arm: str,
    output_dir: Path | None,
    discard_output: bool,
    event_limit: int | None = None,
) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config, arm)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S16-4 arm execution requires exactly one visible GPU")
    device = torch.device("cuda:0")
    if discard_output:
        if output_dir is not None:
            raise ValueError("Discard mode cannot receive a formal output directory")
    else:
        if output_dir is None or output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Formal S16-4 arm output must be new: {output_dir}")
        output_dir.mkdir(parents=True)
    if event_limit is not None:
        if not discard_output:
            raise ValueError("An event limit is permitted only for discard-only resource smoke")
        if event_limit < 1:
            raise ValueError("Discard-only resource smoke event limit must be positive")

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device_index = reset_peak_memory_stats_compat(device)
    started = time.time()

    parent_config = verify_regular(ROOT, config["preflight"]["config"], "preflight_config")
    verify_regular(ROOT, config["preflight"]["status"], "preflight_status")
    parent_summary_path = verify_regular(ROOT, config["preflight"]["summary"], "preflight_summary")
    parent = load_json(parent_config)
    parent_summary = load_json(parent_summary_path)
    if parent_summary.get("verdict") != "PASS_S16_4_TOYS_INPUT_STATE_GATE_FREEZE":
        raise ValueError("S16-4 input/state preflight Gate is not PASS")
    inputs = {name: verify_regular(ROOT, spec, name) for name, spec in parent["inputs"].items()}

    projected = read_projected_sequences(inputs["projected_train_validation_sequences"])
    frozen = read_validation_predictions(inputs["frozen_f0_r2_validation_predictions"])
    if set(projected) != set(frozen) or len(projected) != config["validation_events"]:
        raise ValueError("S16-4 projected/frozen user universe drift")
    metadata = read_metadata(inputs["item_metadata"])
    lexical_paths = read_paths(inputs["lexical_paths"])
    cold = read_set(inputs["cold_items"])
    warm = read_set(inputs["warm_items"])
    retained_warm = read_set(
        verify_regular(
            ROOT,
            config["saux_inference"]["inputs"]["retained_warm_items"],
            "saux_retained_warm_items",
        )
    )
    pseudo_cold = read_set(
        verify_regular(
            ROOT,
            config["saux_inference"]["inputs"]["pseudo_cold_items"],
            "saux_pseudo_cold_items",
        )
    )
    if cold & warm or cold | warm != set(metadata) or set(lexical_paths) != set(metadata):
        raise ValueError("S16-4 cold/warm/metadata/path universe drift")
    if retained_warm & pseudo_cold or retained_warm | pseudo_cold != warm:
        raise ValueError("S16-4 S-AUX retained/pseudo partition drift")
    if any(not set(sequence[:-1]).issubset(warm) for sequence in projected.values()):
        raise ValueError("S16-4 validation history contains a non-warm item")
    ordered_items = sorted(metadata)
    backbone = inputs["t5_config"].parent
    tokenizer = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True)
    token_paths = encode_token_catalog(tokenizer, lexical_paths)
    score_lengths = (
        verifier_score_lengths(token_paths, warm, cold, arm=arm)
        if arm in {"S-AUX", "S-PLUS"}
        else {}
    )

    base_model = None
    model = None
    drafter = None
    aux = None
    delta_context = None
    if arm == "S-AUX":
        model = load_gram(inputs["gram_config"], inputs["gram_f0_checkpoint"], device).eval()
        aux = saux_state(
            checkpoint=inputs["saux_checkpoint"],
            embedding_path=inputs["content_embeddings"],
            retained_items=retained_warm,
            ordered_items=ordered_items,
            device=device,
        )
    elif arm in {"S-PLUS", "S-PLUS-CTRL"}:
        checkpoint_key = "splus_checkpoint" if arm == "S-PLUS" else "splus_ctrl_checkpoint"
        model, drafter = load_continued_model(
            historical=inputs["gram_config"],
            base_checkpoint=inputs["gram_f0_checkpoint"],
            final_checkpoint=inputs[checkpoint_key],
            arm=arm,
            projection_dimension=config["faithful_inference"]["projection_dimension"],
            device=device,
        )
    else:
        model = load_gram(inputs["gram_config"], inputs["gram_f0_checkpoint"], device).eval()
        aggregate = torch.load(inputs["gridge_aggregate_deltas"], map_location="cpu")
        bundles = build_one_one_position_bundles(
            position_to_layer={int(key): int(value) for key, value in aggregate["position_to_layer"].items()},
            aggregated_updates=aggregate["aggregated_updates"],
        )
        delta_context = OneOneGenerationDeltaContext(
            model=model,
            deltas_by_position=bundles,
            position_to_layer={int(key): int(value) for key, value in aggregate["position_to_layer"].items()},
            encoded_catalog_paths=token_paths.values(),
            decoder_start_token_id=int(model.config.decoder_start_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
        )
    assert model is not None
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    full_index = None
    if arm == "S-PLUS":
        assert drafter is not None
        from splus_formal_train import encode_item_index  # local import after device admission

        full_index = encode_item_index(
            model,
            drafter,
            ordered_items,
            metadata,
            tokenizer,
            config["faithful_inference"]["index_batch_size"],
            device,
        )

    rows: list[dict[str, Any]] = []
    run_total = min(len(projected), event_limit) if event_limit is not None else len(projected)
    mechanism_totals = {
        "rounds": 0,
        "drafted": 0,
        "accepted": 0,
        "redraft_rounds": 0,
        "draft_capacity_shortfall_rounds": 0,
        "zero_finite_draft_rounds": 0,
        "beam_decoder_rows": 0,
        "candidate_verifier_forwards": 0,
        "rankings_different_from_f0": 0,
    }
    prediction_handle = None
    if not discard_output:
        assert output_dir is not None
        prediction_handle = (output_dir / "predictions_validation.jsonl").open("x", encoding="utf-8")
    context_manager = delta_context if delta_context is not None else torch.inference_mode()
    with torch.inference_mode(), context_manager:
        for event_index, (user, sequence) in enumerate(projected.items(), 1):
            if event_index > run_total:
                break
            history, target = sequence[:-1][-20:], sequence[-1]
            source = frozen[user]
            if str(source.get("target")) != target:
                raise ValueError(f"Frozen/projected target mismatch: {user}")
            f0 = [str(item) for item in source["v0_top50"]]
            context = tokenize_history(history, metadata, lexical_paths, tokenizer, device)
            mechanism: dict[str, Any] = {}
            if arm == "S-AUX":
                assert aux is not None
                logits = saux_logits(aux, history, device)
                ranking, mechanism = faithful_rank(
                    model=model,
                    context=context,
                    draft_logits=logits,
                    ordered_items=ordered_items,
                    token_paths=token_paths,
                    score_lengths=score_lengths,
                    tokenizer=tokenizer,
                    draft_size=config["faithful_inference"]["draft_size"][arm],
                    threshold=config["faithful_inference"]["threshold"],
                    beam_size=config["beam_size"],
                    candidate_chunk_size=config["faithful_inference"]["candidate_chunk_size"],
                )
            elif arm == "S-PLUS":
                assert drafter is not None and full_index is not None
                hidden, mask = encode_history(model, context)
                sequence_embedding = drafter.pool(hidden, mask)
                logits = drafter.draft_logits(sequence_embedding, full_index)
                ranking, mechanism = faithful_rank(
                    model=model,
                    context=context,
                    draft_logits=logits,
                    ordered_items=ordered_items,
                    token_paths=token_paths,
                    score_lengths=score_lengths,
                    tokenizer=tokenizer,
                    draft_size=config["faithful_inference"]["draft_size"][arm],
                    threshold=config["faithful_inference"]["threshold"],
                    beam_size=config["beam_size"],
                    candidate_chunk_size=config["faithful_inference"]["candidate_chunk_size"],
                    encoder_state=(hidden, mask),
                )
            else:
                ranking = standard_beam_rank(
                    model=model,
                    context=context,
                    token_paths=token_paths,
                    tokenizer=tokenizer,
                    beam_size=config["beam_size"],
                )
            if len(ranking) != config["beam_size"] or len(set(ranking)) != config["beam_size"]:
                raise RuntimeError("S16-4 arm lost exact unique top-50 contract")
            if not set(ranking).issubset(metadata):
                raise RuntimeError("S16-4 arm ranking contains an unknown item")
            metrics = ranking_metrics(ranking, target)
            row = {
                "event_index": event_index,
                "user_id": user,
                "target_item": target,
                "is_cold": target in cold,
                "arm": arm,
                "top50": ranking,
                "metrics": metrics,
                "mechanism": mechanism,
            }
            if prediction_handle is not None:
                prediction_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if event_index % 32 == 0:
                    prediction_handle.flush()
                    os.fsync(prediction_handle.fileno())
            rows.append(row)
            for key in mechanism_totals:
                if key == "rankings_different_from_f0":
                    continue
                mechanism_totals[key] += int(mechanism.get(key, 0))
            mechanism_totals["rankings_different_from_f0"] += int(ranking != f0)
            if event_index % config["progress_interval_events"] == 0 or event_index == run_total:
                print(f"[s16-s4-{arm}] events={event_index}/{run_total}", flush=True)
    if prediction_handle is not None:
        prediction_handle.close()

    elapsed = time.time() - started
    summary = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "attempt_id": config["attempt_id"],
        "arm": arm,
        "status": "completed",
        "verdict": f"COMPLETED_S16_4_TOYS_{arm.replace('-', '_')}_FROZEN_VALIDATION",
        "events": len(rows),
        "metrics": summarize_rows(rows, require_all_subsets=event_limit is None),
        "mechanism_totals": mechanism_totals,
        "runtime_seconds": elapsed,
        "inference_seconds": elapsed,
        "update_seconds": 0.0,
        "extra_state_bytes": {
            "S-AUX": inputs["saux_checkpoint"].stat().st_size,
            "S-PLUS-CTRL": inputs["splus_ctrl_checkpoint"].stat().st_size,
            "S-PLUS": inputs["splus_checkpoint"].stat().st_size,
            "G-RIDGE": inputs["gridge_aggregate_deltas"].stat().st_size,
        }[arm],
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device_index) / 1024**2,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device_index) / 1024**2,
        "peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "validation_target_used_for_state_selection_or_tuning": False,
        "scientific_efficacy_metric_produced": not discard_output,
        "discard_output": discard_output,
        "promotion_eligible": not discard_output,
        "test_read": False,
        "automatic_retry": False,
        "event_limit": event_limit,
    }
    if not discard_output:
        assert output_dir is not None
        atomic_json(output_dir / "config.json", config)
        atomic_json(output_dir / "summary.json", summary)
        atomic_json(
            output_dir / "input_file_sha256.json",
            {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for name, path in inputs.items()},
        )
        atomic_json(
            output_dir / "open_file_manifest.json",
            {
                "validation_projection_opened": True,
                "validation_used_for_evaluation_only": True,
                "validation_used_for_tuning_or_state_selection": False,
                "original_user_sequence_opened": False,
                "test_opened": False,
                "test_read": False,
            },
        )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=FORMAL_ARMS, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--discard-output", action="store_true")
    parser.add_argument("--event-limit", type=int)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_arm(
        args.config.resolve(),
        args.arm,
        args.output_dir.resolve() if args.output_dir is not None else None,
        args.discard_output,
        args.event_limit,
    )
    print(json.dumps({"status": summary["status"], "verdict": summary["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
