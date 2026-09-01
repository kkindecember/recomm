"""Leakage-safe GRAM backend for the Stage17 FP2 resource profiles.

The original GRAM dataset opens ``GRAM/rec_datasets/Toys/user_sequence.txt``
and derives validation/test positions internally.  FP2 must instead consume the
frozen D0 train-prefix adapter, so this module preserves GRAM's FiD passage and
collator/model interfaces while replacing only the data/identifier boundary.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from .full_latte_arm_contracts import (
    decoder_paths,
    full_semantic_vocabulary,
    gram_target_text,
)
from .full_latte_native_adapter import (
    APPROVED_DEV_IDS_SUFFIX,
    APPROVED_METADATA_SUFFIX,
    APPROVED_SEQUENCE_SUFFIX,
    read_frozen_internal_dev_ids,
    read_item_metadata_catalog,
)
from .fullport_data import (
    FullportExample,
    build_train_and_internal_dev_examples,
    read_train_prefix_users,
)


GRAM_ARMS = (
    "G0_GRAM_B0_FRESH",
    "G1_GRAM_PSID_FULL",
    "G2_GRAM_LATTE_FULL",
)
LEXICAL_ID_SUFFIX = Path(
    "GRAM/rec_datasets/Toys/"
    "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
)
SIMILAR_ITEM_SUFFIX = Path("GRAM/rec_datasets/Toys/similar_item_sasrec.txt")
SEMANTIC_ID_SUFFIX = Path(
    "artifacts/phase17/fullport/fp0/full_data_tokenizer/attempt_001/"
    "tokenizer/item_semantic_codes.json"
)
FULL_VOCABULARY_SUFFIX = Path(
    "artifacts/phase17/fullport/fp0/full_data_tokenizer/amendment_001/"
    "gram_full_added_tokens.txt"
)
T5_SMALL_SUFFIX = Path("artifacts/phase14/m2/pretrained/t5-small")
TOP_K_SIMILAR_ITEMS = 5
MAX_HISTORY_ITEMS = 20
ITEM_PROMPT_MAX_LENGTH = 128
TARGET_MAX_LENGTH = 32
SPLIT_DELIMITER_TOKEN_IDS = frozenset((1820, 9175))


@dataclass(frozen=True)
class GramCatalog:
    ordered_items: tuple[str, ...]
    lexical_ids: dict[str, str]
    semantic_codes: dict[str, tuple[int, ...]]
    identity_text: dict[str, str]
    item_passages: dict[str, str]
    item_numeric_ids: dict[str, int]
    top_k_similar_items: int


@dataclass(frozen=True)
class GramRenderedExample:
    input: tuple[str, ...]
    output: str
    user_id: str
    history_item_ids: tuple[int, ...]
    target_item_id: int

    def as_collator_row(self) -> dict[str, Any]:
        return {
            "input": list(self.input),
            "output": self.output,
            "user_id": self.user_id,
            "history_item_ids": list(self.history_item_ids),
            "target_item_id": self.target_item_id,
        }


def _read_key_value_lines(path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    ordered: list[str] = []
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split(maxsplit=1)
        if len(fields) != 2 or not fields[1].strip():
            raise ValueError(f"invalid key/value row at {path}:{line_number}")
        key, value = fields[0], fields[1].strip()
        if key in values:
            raise ValueError(f"duplicate key in {path}: {key}")
        ordered.append(key)
        values[key] = value
    if not ordered:
        raise ValueError(f"empty key/value artifact: {path}")
    return tuple(ordered), values


def _read_similar_items(path: Path, *, top_k: int) -> dict[str, tuple[str, ...]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive for the frozen GRAM passage contract")
    rows: dict[str, tuple[str, ...]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if not fields or fields[0] == "anchor":
            continue
        if len(fields) <= top_k:
            raise ValueError(f"too few similar items at {path}:{line_number}")
        if fields[0] in rows:
            raise ValueError(f"duplicate similar-item anchor: {fields[0]}")
        rows[fields[0]] = tuple(fields[1 : top_k + 1])
    if not rows:
        raise ValueError(f"empty similar-item artifact: {path}")
    return rows


def _semantic_text(codes: Sequence[int]) -> str:
    if len(codes) != 3:
        raise ValueError(f"expected three semantic digits, got {len(codes)}")
    values = tuple(int(code) for code in codes)
    if any(code < 0 or code >= 256 for code in values):
        raise ValueError(f"semantic code outside [0, 255]: {values}")
    return " ".join(
        f"<s17_sid{digit}_{code}>" for digit, code in enumerate(values)
    )


def load_gram_catalog(root: Path, arm_id: str) -> GramCatalog:
    """Load only frozen metadata, identifier and train-derived CF artifacts."""

    if arm_id not in GRAM_ARMS:
        raise ValueError(f"not a Stage17 GRAM arm: {arm_id}")
    root = root.resolve()
    metadata_order, metadata = read_item_metadata_catalog(
        root / APPROVED_METADATA_SUFFIX, root=root
    )
    lexical_order, lexical_ids = _read_key_value_lines(root / LEXICAL_ID_SUFFIX)
    raw_semantic = json.loads((root / SEMANTIC_ID_SUFFIX).read_text(encoding="utf-8"))
    semantic_codes = {
        item: tuple(int(code) for code in codes)
        for item, codes in raw_semantic.items()
    }
    similar = _read_similar_items(
        root / SIMILAR_ITEM_SUFFIX, top_k=TOP_K_SIMILAR_ITEMS
    )
    expected = set(metadata_order)
    artifacts = {
        "lexical": set(lexical_ids),
        "semantic": set(semantic_codes),
        "similar": set(similar),
    }
    for name, items in artifacts.items():
        if items != expected:
            raise ValueError(
                f"{name} catalog mismatch: missing={len(expected - items)}, "
                f"extra={len(items - expected)}"
            )
    if lexical_order != metadata_order:
        raise ValueError("lexical and metadata catalog order drifted")
    for item, candidates in similar.items():
        unknown = set(candidates) - expected
        if unknown:
            raise ValueError(f"similar-item row {item} has unknown candidates: {unknown}")
    if len(set(semantic_codes.values())) != len(semantic_codes):
        raise ValueError("frozen conflict-free semantic IDs contain aliases")

    if arm_id == "G0_GRAM_B0_FRESH":
        identity = dict(lexical_ids)
    else:
        identity = {
            item: _semantic_text(semantic_codes[item]) for item in metadata_order
        }
    passages = {
        item: (
            f"item: {identity[item]}; "
            f"similar items: {', '.join(identity[other] for other in similar[item])}; "
            f"{metadata[item]}"
        )
        for item in metadata_order
    }
    return GramCatalog(
        ordered_items=metadata_order,
        lexical_ids=lexical_ids,
        semantic_codes=semantic_codes,
        identity_text=identity,
        item_passages=passages,
        item_numeric_ids={item: index for index, item in enumerate(metadata_order, 1)},
        top_k_similar_items=TOP_K_SIMILAR_ITEMS,
    )


def load_fullport_examples(
    root: Path,
) -> tuple[list[FullportExample], list[FullportExample]]:
    """Materialize train-prefix rolling train and internal-dev examples only."""

    root = root.resolve()
    users = read_train_prefix_users(root / APPROVED_SEQUENCE_SUFFIX, root=root)
    dev_ids = read_frozen_internal_dev_ids(root / APPROVED_DEV_IDS_SUFFIX, root=root)
    return build_train_and_internal_dev_examples(
        users, dev_ids, max_history_items=MAX_HISTORY_ITEMS
    )


def render_gram_example(
    example: FullportExample,
    *,
    arm_id: str,
    catalog: GramCatalog,
    rng: random.Random,
) -> GramRenderedExample:
    unknown = (set(example.history) | {example.target}) - set(catalog.ordered_items)
    if unknown:
        raise ValueError(f"example contains items outside the catalog: {unknown}")
    # GRAM presents the most recent item first in both the coarse prompt and
    # the following fine-grained passages.
    reversed_history = tuple(reversed(example.history[-MAX_HISTORY_ITEMS:]))
    sequence_text = " ; ".join(catalog.identity_text[item] for item in reversed_history)
    user_sentence = f"What would user purchase after {sequence_text} ?"
    output = gram_target_text(
        arm_id,
        example.target,
        lexical_ids=catalog.lexical_ids,
        semantic_codes=catalog.semantic_codes,
        rng=rng,
    )
    return GramRenderedExample(
        input=(user_sentence,)
        + tuple(catalog.item_passages[item] for item in reversed_history),
        output=output,
        user_id=example.user_id,
        history_item_ids=tuple(
            catalog.item_numeric_ids[item] for item in reversed_history
        ),
        target_item_id=catalog.item_numeric_ids[example.target],
    )


def gram_collator_args(arm_id: str) -> SimpleNamespace:
    if arm_id not in GRAM_ARMS:
        raise ValueError(f"not a Stage17 GRAM arm: {arm_id}")
    return SimpleNamespace(
        item_prompt_max_len=ITEM_PROMPT_MAX_LENGTH,
        target_max_len=TARGET_MAX_LENGTH,
        max_his=MAX_HISTORY_ITEMS,
        item_id_type=("split" if arm_id == "G0_GRAM_B0_FRESH" else "t5_token"),
        hierarchical_id_type="hierarchy_v1_c32_l5_len32768_split",
    )


def load_gram_tokenizer(root: Path, arm_id: str):
    """Load the offline T5 tokenizer and freeze the G1/G2 776-token surface."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(root.resolve() / T5_SMALL_SUFFIX), local_files_only=True
    )
    if arm_id in {"G1_GRAM_PSID_FULL", "G2_GRAM_LATTE_FULL"}:
        frozen = tuple(
            token
            for token in (root.resolve() / FULL_VOCABULARY_SUFFIX)
            .read_text(encoding="utf-8")
            .splitlines()
            if token
        )
        if frozen != full_semantic_vocabulary():
            raise RuntimeError("complete GRAM vocabulary artifact drifted")
        added = tokenizer.add_tokens(list(frozen))
        if added != 776:
            raise RuntimeError(f"expected to add 776 fresh tokens, added {added}")
        ids = tokenizer.convert_tokens_to_ids(list(frozen))
        if len(set(ids)) != 776 or tokenizer.unk_token_id in ids:
            raise RuntimeError("G1/G2 added tokens are not one-to-one tokenizer IDs")
    return tokenizer


def build_gram_collator(root: Path, arm_id: str):
    gram_source = str(root.resolve() / "GRAM/src")
    if gram_source not in sys.path:
        sys.path.insert(0, gram_source)
    from processor import CollatorGRAM

    tokenizer = load_gram_tokenizer(root, arm_id)
    return tokenizer, CollatorGRAM(tokenizer, args=gram_collator_args(arm_id))


def encoded_candidate_paths(
    tokenizer, arm_id: str, catalog: GramCatalog
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Encode all constrained decoder paths, including decoder-start and EOS."""

    text_paths = decoder_paths(
        arm_id,
        lexical_ids=catalog.lexical_ids,
        semantic_codes=catalog.semantic_codes,
    )
    encoded: dict[str, tuple[tuple[int, ...], ...]] = {}
    for item in catalog.ordered_items:
        item_paths: list[tuple[int, ...]] = []
        for text in text_paths[item]:
            if arm_id == "G0_GRAM_B0_FRESH":
                path = (0,) + tuple(
                    token
                    for token in tokenizer.encode(text)
                    if token not in SPLIT_DELIMITER_TOKEN_IDS
                )
            else:
                path = (
                    (0,)
                    + tuple(tokenizer.convert_tokens_to_ids(text.split()))
                    + (tokenizer.eos_token_id,)
                )
            item_paths.append(path)
        encoded[item] = tuple(item_paths)
    flat = [path for paths in encoded.values() for path in paths]
    if len(flat) != len(set(flat)):
        raise ValueError("encoded constrained decoder paths contain item aliases")
    return encoded


class PrefixTree:
    """Small immutable-prefix view compatible with HF generation callbacks."""

    def __init__(self, paths: Iterable[Sequence[int]]) -> None:
        children: dict[tuple[int, ...], set[int]] = {}
        count = 0
        for raw_path in paths:
            path = tuple(int(token) for token in raw_path)
            if not path:
                raise ValueError("empty decoder path")
            count += 1
            for position, token in enumerate(path):
                children.setdefault(path[:position], set()).add(token)
        if count == 0:
            raise ValueError("cannot build an empty decoder trie")
        self.children = {
            prefix: tuple(sorted(tokens)) for prefix, tokens in children.items()
        }
        self.path_count = count

    def allowed(self, sentence: Sequence[int]) -> tuple[int, ...]:
        return self.children.get(tuple(int(token) for token in sentence), ())

    def prefix_allowed_tokens_fn(self):
        def allowed(_batch_id, sentence):
            return list(self.allowed(sentence.tolist()))

        return allowed


def aggregate_generated_paths(
    sequences: Sequence[Sequence[int]],
    log_scores: Sequence[float],
    *,
    item_paths: Mapping[str, Sequence[Sequence[int]]],
    method: str,
    top_k: int,
) -> list[tuple[str, float, int]]:
    """Resolve paths to items and aggregate latent variants at item level."""

    if method not in {"agg_max", "agg_sum"}:
        raise ValueError(f"unknown aggregation method: {method}")
    if len(sequences) != len(log_scores) or top_k <= 0:
        raise ValueError("invalid generated path batch")
    path_to_item = {
        tuple(int(token) for token in path): item
        for item, paths in item_paths.items()
        for path in paths
    }
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for sequence, raw_score in zip(sequences, log_scores):
        item = path_to_item.get(tuple(int(token) for token in sequence))
        if item is None:
            continue
        score = float(raw_score)
        counts[item] = counts.get(item, 0) + 1
        if item not in scores:
            scores[item] = score
        elif method == "agg_max":
            scores[item] = max(scores[item], score)
        else:
            high, low = max(scores[item], score), min(scores[item], score)
            scores[item] = high + math.log1p(math.exp(low - high))
    return [
        (item, score, counts[item])
        for item, score in sorted(scores.items(), key=lambda row: (-row[1], row[0]))[
            :top_k
        ]
    ]


def create_fresh_gram_model(root: Path, arm_id: str, tokenizer, *, seed: int = 2023):
    """Create the exact fresh FiD-GRAM backbone used by all three FP2 arms."""

    import torch
    from transformers import AutoModelForSeq2SeqLM, T5Config

    gram_source = str(root.resolve() / "GRAM/src")
    if gram_source not in sys.path:
        sys.path.insert(0, gram_source)
    from model import create_model

    if arm_id not in GRAM_ARMS:
        raise ValueError(f"not a Stage17 GRAM arm: {arm_id}")
    backbone_path = root.resolve() / T5_SMALL_SUFFIX
    torch.manual_seed(seed)
    config = T5Config.from_pretrained(str(backbone_path), local_files_only=True)
    config.max_seq_len = ITEM_PROMPT_MAX_LENGTH
    config.max_item_num = MAX_HISTORY_ITEMS
    config.use_position_embedding = 1
    config.sample_num = "1"
    config.cf0_arm = "A"
    config.cf0_enabled = False
    config.cf0_num_items = 0
    config.hi_gram_enabled = False
    config.s17_modules = ""
    config.s17_transition_map = ""
    backbone = AutoModelForSeq2SeqLM.from_pretrained(
        str(backbone_path), config=config, local_files_only=True
    )
    model = create_model("gram", config=config)
    model.load_t5(backbone.state_dict())
    if len(tokenizer) != model.config.vocab_size:
        torch.manual_seed(seed)
        model.resize_token_embeddings(len(tokenizer))
    return model


def cpu_preflight_gram_arm(root: Path, arm_id: str) -> dict[str, Any]:
    """Exercise the complete data/token/collator/trie contract without a GPU."""

    if arm_id not in GRAM_ARMS:
        raise ValueError(f"not a Stage17 GRAM arm: {arm_id}")
    root = root.resolve()
    catalog = load_gram_catalog(root, arm_id)
    train, internal_dev = load_fullport_examples(root)
    rng = random.Random(2023)
    longest_train = max(train, key=lambda example: len(example.history))
    rows = [
        render_gram_example(example, arm_id=arm_id, catalog=catalog, rng=rng)
        for example in (train[0], longest_train, internal_dev[0])
    ]
    tokenizer, collator = build_gram_collator(root, arm_id)
    batch = collator([row.as_collator_row() for row in rows])
    encoded = encoded_candidate_paths(tokenizer, arm_id, catalog)
    flat = [path for paths in encoded.values() for path in paths]
    trie = PrefixTree(flat)
    expected_paths = len(catalog.ordered_items) * (
        8 if arm_id == "G2_GRAM_LATTE_FULL" else 1
    )
    if trie.path_count != expected_paths:
        raise AssertionError("decoder path multiplicity drifted")
    for path in (flat[0], flat[len(flat) // 2], flat[-1]):
        for position, token in enumerate(path):
            if token not in trie.allowed(path[:position]):
                raise AssertionError("decoder trie rejected a frozen catalog path")
    expected_target_length = {
        "G0_GRAM_B0_FRESH": None,
        "G1_GRAM_PSID_FULL": 4,
        "G2_GRAM_LATTE_FULL": 5,
    }[arm_id]
    if expected_target_length is not None and batch["target_ids"].shape[1] != expected_target_length:
        raise AssertionError("semantic target tensor length drifted")
    return {
        "arm_id": arm_id,
        "state": "PASS_CPU_PREFLIGHT",
        "catalog_items": len(catalog.ordered_items),
        "rolling_train_examples": len(train),
        "internal_dev_examples": len(internal_dev),
        "rendered_examples_checked": len(rows),
        "passages_in_checked_batch": int(batch["item_text_ids"].shape[1]),
        "passage_token_length": int(batch["item_text_ids"].shape[2]),
        "target_token_length": int(batch["target_ids"].shape[1]),
        "decoder_paths": trie.path_count,
        "top_k_similar_items": catalog.top_k_similar_items,
        "added_vocabulary_size": (776 if arm_id != "G0_GRAM_B0_FRESH" else 0),
        "external_target_materialized": False,
        "effect_metrics_computed": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "gpu_used": False,
    }
