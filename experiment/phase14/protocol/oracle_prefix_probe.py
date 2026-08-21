"""Stage 14-0B learned cold-path probe for a frozen GRAM checkpoint.

This validation-only diagnostic measures four distinct objects without training:

1. target-token raw-vocabulary and trie-legal NLL/rank under teacher forcing;
2. actual target-prefix survival inside the live width-K constrained beam;
3. frozen beam parity and item-level final-beam prefix survival;
4. target-subtree mass/rank for R2, catalog-text, popularity, and uniform priors.

The structural warm-prefix statistics are recorded only as covariates.  They never
decide the route by themselves.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList, T5Config


REPO_ROOT = Path(__file__).resolve().parents[3]
GRAM_SRC = REPO_ROOT / "GRAM" / "src"
if str(GRAM_SRC) not in sys.path:
    sys.path.insert(0, str(GRAM_SRC))

# GRAM's ``utils/__init__.py`` eagerly imports dataset_utils, which imports
# ``data`` again and creates a circular import when a standalone diagnostic
# imports TestDatasetGRAM.  Register only the package path here; Python then
# loads the required ``utils.indexing`` and ``utils.prompt`` submodules without
# executing that unrelated eager initializer.  No GRAM source file is changed.
if "utils" not in sys.modules:
    utils_package = types.ModuleType("utils")
    utils_package.__path__ = [str(GRAM_SRC / "utils")]
    sys.modules["utils"] = utils_package

from data import TestDatasetGRAM  # noqa: E402
from model import create_model  # noqa: E402
from processor.Collator import CollatorGRAM  # noqa: E402

from item_level_eval import (  # noqa: E402
    atomic_json,
    decode_lexical_id,
    load_item_paths,
    parse_prediction_rows,
    sha256_file,
)


class ResidualUserProjector(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
        )
        self.residual_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, histories: torch.Tensor) -> torch.Tensor:
        return F.normalize(histories + self.residual_scale * self.net(histories), dim=-1)


def read_sequences(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_no}: sequence too short")
            if parts[0] in rows:
                raise ValueError(f"{path}:{line_no}: duplicate user")
            rows[parts[0]] = parts[1:]
    return rows


def read_set(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def encode_lexical_path(tokenizer, lexical_id: str) -> tuple[int, ...]:
    ids = [token for token in tokenizer.encode(lexical_id) if token not in (1820, 9175)]
    if not ids or ids[-1] != tokenizer.eos_token_id:
        raise ValueError(f"Encoded lexical path lacks EOS: {lexical_id!r} -> {ids}")
    return tuple(ids[:-1])


def build_trie_children(paths: list[tuple[int, ...]], eos_id: int) -> dict[tuple[int, ...], tuple[int, ...]]:
    children: dict[tuple[int, ...], set[int]] = collections.defaultdict(set)
    for path in paths:
        for depth, token in enumerate((*path, eos_id)):
            children[path[:depth]].add(token)
    return {prefix: tuple(sorted(values)) for prefix, values in children.items()}


def tie_aware_midrank(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One-based midrank; equal scores share the average occupied rank."""
    greater = (values > target[..., None]).sum(dim=-1).float()
    equal = (values == target[..., None]).sum(dim=-1).float()
    return greater + (equal + 1.0) / 2.0


class BeamPrefixTracker:
    """Frozen-trie-compatible prefix callback plus score-aware observations."""

    def __init__(
        self,
        children: dict[tuple[int, ...], tuple[int, ...]],
        target: tuple[int, ...],
        inactive_score_floor: float = -1e8,
    ):
        self.children = children
        self.target = target
        self.inactive_score_floor = inactive_score_floor
        self.survived = [False] * (len(target) + 1)
        self.survived[0] = True
        self.empty_callback_count = 0

    def __call__(self, _batch_id: int, input_ids: torch.Tensor) -> list[int]:
        prefix = tuple(int(value) for value in input_ids.tolist()[1:])
        allowed = self.children.get(prefix)
        if allowed is None:
            # This is exactly GRAM Trie.get() semantics.  With num_beams larger
            # than the number of legal continuations, HF beam search retains
            # -inf/-1e9 bookkeeping rows and later calls the constraint on
            # arbitrary filler prefixes.  They must remain empty rather than
            # aborting or being counted as live.
            self.empty_callback_count += 1
            return []
        return list(allowed)

    def observe_processed_scores(self, input_ids: torch.Tensor, scores: torch.Tensor) -> None:
        """Record only prefixes on rows with a non-sentinel cumulative score."""
        row_best = scores.detach().amax(dim=-1)
        live_rows = torch.isfinite(row_best) & (row_best > self.inactive_score_floor)
        for row, is_live in zip(input_ids, live_rows.tolist()):
            if not is_live:
                continue
            prefix = tuple(int(value) for value in row.tolist()[1:])
            depth = len(prefix)
            if depth <= len(self.target) and prefix == self.target[:depth]:
                self.survived[depth] = True


class LiveBeamObserver(LogitsProcessor):
    """Observe scores after HF's prefix constraint without changing them."""

    def __init__(self, tracker: BeamPrefixTracker):
        self.tracker = tracker

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.tracker.observe_processed_scores(input_ids, scores)
        return scores


def configure_model(historical: dict, checkpoint: Path, device: torch.device):
    config = T5Config.from_pretrained(historical["backbone"])
    config.max_seq_len = historical["item_prompt_max_len"]
    config.max_item_num = historical["max_his"]
    config.use_position_embedding = historical["use_position_embedding"]
    config.sample_num = historical["sample_num"]
    config.cf0_arm = "A"
    config.cf0_enabled = False
    config.cf0_num_items = 0
    config.cf0_num_layers = historical.get("cf0_num_layers", 2)
    config.cf0_num_heads = historical.get("cf0_num_heads", 4)
    config.cf0_dropout = historical.get("cf0_dropout", 0.1)
    config.cf0_loss_weight = historical.get("cf0_loss_weight", 0.1)
    config.cf0_injection_scale = historical.get("cf0_injection_scale", 0.1)
    config.cf0_joint_score_weight = historical.get("cf0_joint_score_weight", 0.25)
    config.hi_gram_enabled = False
    config.hi_gram_local_window = historical.get("hi_gram_local_window", 5)
    config.hi_gram_local_layers = historical.get("hi_gram_local_layers", 2)
    config.hi_gram_global_layers = historical.get("hi_gram_global_layers", 2)
    config.hi_gram_num_heads = historical.get("hi_gram_num_heads", 4)
    config.hi_gram_dropout = historical.get("hi_gram_dropout", 0.1)
    config.hi_gram_fusion_scale_init = historical.get("hi_gram_fusion_scale_init", 0.1)
    config.hi_gram_include_user_prompt = False
    model = create_model("gram", config=config)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def make_dataset_args(historical: dict, dataset_dir: Path) -> SimpleNamespace:
    values = historical.copy()
    values.update(
        {
            "data_path": str(dataset_dir.parent.resolve()),
            "datasets": dataset_dir.name,
            "prompt_file": str((REPO_ROOT / "GRAM" / "prompt.txt").resolve()),
            "rank": 0,
            "distributed": 0,
            "debug_test_100": 0,
            "item_id_path": "",
            "verbose_input_output": 0,
        }
    )
    return SimpleNamespace(**values)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def target_token_observations(
    logits: torch.Tensor,
    labels: torch.Tensor,
    user_ids: list[str],
    user_targets: dict[str, str],
    item_paths: dict[str, tuple[int, ...]],
    trie_children: dict[tuple[int, ...], tuple[int, ...]],
) -> list[dict]:
    observations: list[dict] = []
    for row, user in enumerate(user_ids):
        target_item = user_targets[user]
        path = item_paths[target_item]
        active_labels = tuple(int(value) for value in labels[row].tolist() if value >= 0)
        if active_labels[: len(path)] != path:
            raise RuntimeError(
                f"Target token mismatch for {user}: collator={active_labels} catalog={path}"
            )
        per_depth = []
        for depth, token in enumerate(path, 1):
            step_logits = logits[row, depth - 1].float()
            target_logit = step_logits[token]
            raw_log_prob = F.log_softmax(step_logits, dim=-1)[token]
            raw_rank = float(tie_aware_midrank(step_logits, target_logit).item())
            prefix = path[: depth - 1]
            legal = trie_children[prefix]
            legal_tensor = torch.tensor(legal, device=step_logits.device)
            legal_logits = step_logits[legal_tensor]
            target_matches = (legal_tensor == token).nonzero(as_tuple=False)
            if target_matches.numel() != 1:
                raise RuntimeError(f"Target token {token} is not a unique legal child of {prefix}")
            legal_index = int(target_matches.item())
            legal_log_prob = F.log_softmax(legal_logits, dim=-1)[legal_index]
            legal_rank = float(
                tie_aware_midrank(legal_logits, legal_logits[legal_index]).item()
            )
            per_depth.append(
                {
                    "depth": depth,
                    "normalized_depth": depth / len(path),
                    "raw_nll": -float(raw_log_prob.item()),
                    "raw_rank": raw_rank,
                    "legal_nll": -float(legal_log_prob.item()),
                    "legal_rank": legal_rank,
                    "legal_branch_factor": len(legal),
                }
            )
        observations.append({"user_id": user, "target_item": target_item, "token_profile": per_depth})
    return observations


def recency_history(
    items: list[str], item_to_index: dict[str, int], embeddings: torch.Tensor, decay: float
) -> torch.Tensor:
    indices = torch.tensor([item_to_index[item] for item in items], dtype=torch.long)
    history = embeddings[indices]
    ages = torch.arange(len(items) - 1, -1, -1, dtype=history.dtype)
    weights = decay ** ages
    return F.normalize((history * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-12), dim=0)


def prefix_groups(item_ids: list[str], item_path_tokens: dict[str, tuple[str, ...]]):
    result = {}
    max_depth = max(len(item_path_tokens[item]) for item in item_ids)
    for depth in range(1, max_depth + 1):
        group_lookup: dict[tuple[str, ...], int] = {}
        group_ids = []
        for item in item_ids:
            path = item_path_tokens[item]
            prefix = path[: min(depth, len(path))]
            if prefix not in group_lookup:
                group_lookup[prefix] = len(group_lookup)
            group_ids.append(group_lookup[prefix])
        result[depth] = (group_lookup, torch.tensor(group_ids, dtype=torch.long))
    return result


def distribution_profile(
    scores: torch.Tensor,
    target_indices: torch.Tensor,
    target_items: list[str],
    item_path_tokens: dict[str, tuple[str, ...]],
    groups: dict,
    temperature: float,
) -> list[dict]:
    probs = F.softmax(scores.float() / temperature, dim=-1)
    target_scores = scores.gather(1, target_indices[:, None]).squeeze(1)
    item_ranks = tie_aware_midrank(scores, target_scores)
    outputs = [{"item_rank": float(item_ranks[row]), "depth": []} for row in range(len(target_items))]
    for depth, (lookup, group_cpu) in groups.items():
        group_ids = group_cpu.to(scores.device)
        masses = torch.zeros(scores.size(0), len(lookup), device=scores.device)
        masses.scatter_add_(1, group_ids[None].expand(scores.size(0), -1), probs)
        entropy = -(masses * masses.clamp_min(1e-30).log()).sum(1)
        target_group = torch.tensor(
            [lookup[item_path_tokens[item][: min(depth, len(item_path_tokens[item]))]] for item in target_items],
            device=scores.device,
        )
        target_mass = masses.gather(1, target_group[:, None]).squeeze(1)
        target_rank = tie_aware_midrank(masses, target_mass)
        for row, item in enumerate(target_items):
            if depth <= len(item_path_tokens[item]):
                outputs[row]["depth"].append(
                    {
                        "depth": depth,
                        "normalized_depth": depth / len(item_path_tokens[item]),
                        "target_prefix_mass": float(target_mass[row]),
                        "target_prefix_rank": float(target_rank[row]),
                        "prefix_entropy": float(entropy[row]),
                    }
                )
    return outputs


def fixed_prior_scores(
    sequences: dict[str, list[str]], item_ids: list[str], item_to_index: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    popularity = torch.zeros(len(item_ids), dtype=torch.float32)
    for items in sequences.values():
        for item in items[:-2]:
            popularity[item_to_index[item]] += 1
    popularity_scores = popularity.add(1e-12).log()
    uniform_scores = torch.zeros_like(popularity_scores)
    return uniform_scores, popularity_scores


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def summarise(rows: list[dict], cold_items: set[str]) -> dict:
    result = {}
    for slice_name, selected in (
        ("all", rows),
        ("warm", [row for row in rows if row["target_item"] not in cold_items]),
        ("cold", [row for row in rows if row["target_item"] in cold_items]),
    ):
        token_by_depth: dict[int, list[dict]] = collections.defaultdict(list)
        for row in selected:
            for obs in row["token_profile"]:
                token_by_depth[obs["depth"]].append(obs)
        token_summary = {
            str(depth): {
                "n": len(values),
                "normalized_depth_mean": mean([v["normalized_depth"] for v in values]),
                "legal_nll_mean": mean([v["legal_nll"] for v in values]),
                "legal_nll_median": median([v["legal_nll"] for v in values]),
                "legal_rank_mean": mean([v["legal_rank"] for v in values]),
                "legal_rank_median": median([v["legal_rank"] for v in values]),
                "raw_nll_mean": mean([v["raw_nll"] for v in values]),
                "raw_rank_median": median([v["raw_rank"] for v in values]),
                "branch_factor_mean": mean([v["legal_branch_factor"] for v in values]),
            }
            for depth, values in sorted(token_by_depth.items())
        }
        beam_rows = [row["beam"] for row in selected if "beam" in row]
        result[slice_name] = {
            "n": len(selected),
            "token_by_raw_depth": token_summary,
            "beam_trace_n": len(beam_rows),
            "beam_target_hit50": mean([float(row["target_final_rank"] is not None) for row in beam_rows]),
            "beam_first_dropout_depth_median": median(
                [row["first_dropout_depth"] for row in beam_rows if row["first_dropout_depth"] is not None]
            ),
            "beam_first_dropout_normalized_median": median(
                [row["first_dropout_normalized"] for row in beam_rows if row["first_dropout_normalized"] is not None]
            ),
        }
    return result


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    historical = json.loads(args.historical_config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(historical["backbone"])
    dataset_args = make_dataset_args(historical, dataset_dir)
    dataset = TestDatasetGRAM(
        args=dataset_args,
        dataset=dataset_dir.name,
        task="sequential",
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        debug_test_small_set=False,
        mode="validation",
    )
    if args.limit:
        indices = list(range(min(args.limit, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    collator = CollatorGRAM(tokenizer, args=dataset_args, mode="valid")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    item_to_lexical, decoded_to_items = load_item_paths(args.item_path_file)
    if any(len(items) != 1 for items in decoded_to_items.values()):
        raise RuntimeError("Stage14-0B requires a collision-free baseline identifier")
    item_path_token_strings = {
        item: tuple(token for token in lexical.split("|") if token)
        for item, lexical in item_to_lexical.items()
    }
    item_path_ids = {item: encode_lexical_path(tokenizer, lexical) for item, lexical in item_to_lexical.items()}
    trie_children = build_trie_children(list(item_path_ids.values()), tokenizer.eos_token_id)
    sequences = read_sequences(dataset_dir / "user_sequence.txt")
    cold_items = read_set(dataset_dir / "cold_split_meta" / "cold_items.txt")
    user_targets = {user: items[-2] for user, items in sequences.items()}
    model = configure_model(historical, args.checkpoint, device)

    rows: list[dict] = []
    row_by_user: dict[str, dict] = {}
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, 1):
            batch = batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["item_text_ids"],
                attention_mask=batch["item_text_masks"],
                history_item_ids=batch["history_item_ids"],
                history_item_mask=batch["history_item_mask"],
                target_item_ids=batch["target_item_ids"],
                labels=batch["target_ids"],
            )
            observations = target_token_observations(
                outputs.logits,
                batch["target_ids"],
                batch["user_ids"],
                user_targets,
                item_path_ids,
                trie_children,
            )
            for observation in observations:
                observation["is_cold"] = observation["target_item"] in cold_items
                rows.append(observation)
                row_by_user[observation["user_id"]] = observation
            if batch_idx % 50 == 0:
                print(f"[nll] users={len(rows)}", flush=True)

    embedding_payload = torch.load(args.item_embeddings, map_location="cpu")
    item_ids = list(embedding_payload["item_ids"])
    embeddings_cpu = embedding_payload["embeddings"].float()
    if set(item_ids) != set(item_to_lexical):
        raise RuntimeError("Embedding catalog and identifier catalog differ")
    item_to_index = {item: idx for idx, item in enumerate(item_ids)}
    embeddings = embeddings_cpu.to(device)
    resolver_payload = torch.load(args.resolver_checkpoint, map_location="cpu")
    resolver = ResidualUserProjector(
        resolver_payload["dim"], resolver_payload["hidden_dim"], resolver_payload["dropout"]
    )
    resolver.load_state_dict(resolver_payload["state_dict"], strict=True)
    resolver.to(device).eval()
    groups = prefix_groups(item_ids, item_path_token_strings)
    uniform_scores, popularity_scores = fixed_prior_scores(sequences, item_ids, item_to_index)

    selected_users = [row["user_id"] for row in rows]
    with torch.inference_mode():
        for start in range(0, len(selected_users), args.teacher_batch_size):
            users = selected_users[start : start + args.teacher_batch_size]
            histories = torch.stack(
                [
                    recency_history(
                        sequences[user][:-2][-historical["max_his"] :],
                        item_to_index,
                        embeddings_cpu,
                        args.recency_decay,
                    )
                    for user in users
                ]
            ).to(device)
            target_items = [user_targets[user] for user in users]
            target_indices = torch.tensor([item_to_index[item] for item in target_items], device=device)
            score_sets = {
                "uniform": uniform_scores.to(device)[None].expand(len(users), -1),
                "popularity": popularity_scores.to(device)[None].expand(len(users), -1),
                "catalog_text": histories @ embeddings.T,
                "r2": resolver(histories) @ embeddings.T,
            }
            profiles = {
                name: distribution_profile(
                    scores,
                    target_indices,
                    target_items,
                    item_path_token_strings,
                    groups,
                    args.teacher_temperature,
                )
                for name, scores in score_sets.items()
            }
            for row_idx, user in enumerate(users):
                row_by_user[user]["teacher_profiles"] = {
                    name: profiles[name][row_idx] for name in profiles
                }
            print(f"[teacher] users={min(start + len(users), len(selected_users))}", flush=True)

    frozen_rows = {row["user_id"]: row for row in parse_prediction_rows(args.frozen_predictions)}
    beam_parity_mismatches = 0
    if not args.skip_beam_trace:
        trace_collator = CollatorGRAM(tokenizer, args=dataset_args, mode="valid")
        for index in range(len(dataset)):
            sample = dataset[index]
            batch = batch_to_device(trace_collator([sample]), device)
            user = batch["user_ids"][0]
            target_item = user_targets[user]
            target_path = item_path_ids[target_item]
            tracker = BeamPrefixTracker(trie_children, target_path)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=batch["item_text_ids"],
                    attention_mask=batch["item_text_masks"],
                    history_item_ids=batch["history_item_ids"],
                    history_item_mask=batch["history_item_mask"],
                    max_length=max(len(path) for path in item_path_ids.values()) + 2,
                    prefix_allowed_tokens_fn=tracker,
                    num_beams=args.beam_size,
                    num_return_sequences=args.beam_size,
                    output_scores=True,
                    return_dict_in_generate=True,
                    length_penalty=1.0,
                    logits_processor=LogitsProcessorList([LiveBeamObserver(tracker)]),
                )
            decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
            frozen = frozen_rows[user]["predictions"][: args.beam_size]
            if decoded != frozen:
                beam_parity_mismatches += 1
            generated_items = [decoded_to_items[value][0] for value in decoded]
            target_rank = generated_items.index(target_item) + 1 if target_item in generated_items else None
            final_prefix_survival = [
                any(item_path_ids[item][:depth] == target_path[:depth] for item in generated_items)
                for depth in range(1, len(target_path) + 1)
            ]
            first_dropout = next(
                (depth for depth in range(1, len(target_path) + 1) if not tracker.survived[depth]),
                None,
            )
            row_by_user[user]["beam"] = {
                "beam_size": args.beam_size,
                "target_prefix_live_survival": tracker.survived[1:],
                "target_final_beam_descendant_survival": final_prefix_survival,
                "first_dropout_depth": first_dropout,
                "first_dropout_normalized": first_dropout / len(target_path) if first_dropout else None,
                "target_final_rank": target_rank,
                "frozen_prediction_parity": decoded == frozen,
                "trie_empty_callback_count": tracker.empty_callback_count,
            }
            if (index + 1) % 100 == 0:
                print(f"[beam] users={index + 1}/{len(dataset)} parity_mismatch={beam_parity_mismatches}", flush=True)

    if beam_parity_mismatches:
        raise RuntimeError(f"Frozen beam parity failed for {beam_parity_mismatches} users")

    ordered_rows = [row_by_user[user] for user in selected_users]
    with (output_dir / "per_user_validation.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    inputs = [
        args.historical_config,
        args.checkpoint,
        args.item_path_file,
        args.frozen_predictions,
        args.item_embeddings,
        args.resolver_checkpoint,
        dataset_dir / "user_sequence.txt",
        dataset_dir / "cold_split_meta" / "cold_items.txt",
        dataset_dir / "cold_split_meta" / "warm_items.txt",
        dataset_dir / "item_plain_text.txt",
        dataset_dir / f"similar_item_{historical['cf_model']}.txt",
    ]
    hashes = {str(path.resolve()): sha256_file(path) for path in inputs}
    summary = {
        "experiment_id": "GRAM_PHASE14_STAGE14_0B_ORACLE_PREFIX_PROBE",
        "status": "completed",
        "dataset": dataset_dir.name,
        "split": "validation",
        "test_predictions_opened": False,
        "n_users": len(ordered_rows),
        "beam_trace_enabled": not args.skip_beam_trace,
        "beam_size": args.beam_size,
        "beam_parity_mismatches": beam_parity_mismatches,
        "teacher_temperature": args.teacher_temperature,
        "recency_decay": args.recency_decay,
        "summary_by_slice": summarise(ordered_rows, cold_items),
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "test_read": False,
        "route_decision": "PENDING_DUAL_DOMAIN_SYNTHESIS",
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "input_file_sha256.json", hashes)
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "scope": "application-level declared opens",
            "test_files_opened": [],
            "files": [{"path": path, "mode": "read", "sha256": digest} for path, digest in hashes.items()],
        },
    )
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "split": "validation",
            "test_predictions_opened": False,
            "target_rule": "user_sequence[-2]",
            "teacher_history_rule": "visible history user_sequence[:-2], max_his truncated",
            "teacher_target_labels_used_for_scoring_only": True,
            "model_or_teacher_training": False,
        },
    )
    atomic_json(output_dir / "config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    print(json.dumps({"status": "completed", "summary": str(output_dir / "summary.json")}), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--historical-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--item-path-file", required=True, type=Path)
    parser.add_argument("--frozen-predictions", required=True, type=Path)
    parser.add_argument("--item-embeddings", required=True, type=Path)
    parser.add_argument("--resolver-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--teacher-temperature", type=float, default=0.07)
    parser.add_argument("--recency-decay", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-beam-trace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
