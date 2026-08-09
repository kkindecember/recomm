"""Utilities for parsing and manipulating GRAM hierarchical id files.

GRAM's `item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt` has
one line per item:

    <item_id> |<t0>|<t1>|<t2>|<t3>|<t4>|<t5>|<t6>

where <t*> are SentencePiece tokens from the T5 vocab (32768). The name c128
refers to top-level clustering (level 0 has ~108 distinct tokens across all
items); deeper levels are essentially per-item near-unique.

For v1 (semantic bridge), we need to:
  - Load id file → dict[item_id -> list[str] of 7 tokens]
  - Build per-level token vocabularies (int index ↔ token string)
  - Convert item ids to (7,) int tensors for MLP training targets
  - Reverse: (7,) int predictions → list of 7 tokens → new line in id file

Warm/cold partition is orthogonal: read cold_items.txt / warm_items.txt from
Beauty_cold50/cold_split_meta/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HierIdVocab:
    """Bidirectional token ↔ index mapping per hierarchical level."""
    per_level_token_to_idx: list[dict[str, int]] = field(default_factory=list)
    per_level_idx_to_token: list[list[str]] = field(default_factory=list)

    @property
    def level_sizes(self) -> list[int]:
        return [len(v) for v in self.per_level_token_to_idx]

    @property
    def n_levels(self) -> int:
        return len(self.per_level_token_to_idx)

    def encode(self, tokens: list[str]) -> list[int]:
        assert len(tokens) == len(self.per_level_token_to_idx), \
            f"expected {len(self.per_level_token_to_idx)} tokens, got {len(tokens)}"
        return [self.per_level_token_to_idx[i][tokens[i]]
                for i in range(len(tokens))]

    def decode(self, indices: list[int]) -> list[str]:
        assert len(indices) == len(self.per_level_idx_to_token)
        return [self.per_level_idx_to_token[i][indices[i]]
                for i in range(len(indices))]

    def save(self, path: Path):
        payload = {
            "per_level_idx_to_token": self.per_level_idx_to_token,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "HierIdVocab":
        with open(path) as f:
            payload = json.load(f)
        v = cls()
        v.per_level_idx_to_token = payload["per_level_idx_to_token"]
        v.per_level_token_to_idx = [{t: i for i, t in enumerate(l)}
                                    for l in v.per_level_idx_to_token]
        return v


def parse_id_line(line: str, n_levels: int | None = None) -> tuple[str, list[str]]:
    """Return (item_id, [t0, t1, ..., t{L-1}]).

    Format: '<item_id> |<t0>|<t1>|...'. If n_levels is given, take exactly that
    many tokens; if None, take all tokens from the line (the first line of an
    id file effectively defines the level count for the whole file).
    """
    line = line.rstrip("\n")
    if not line:
        raise ValueError("empty line")
    parts = line.split(" |", 1)
    if len(parts) != 2:
        raise ValueError(f"malformed line (no ' |' separator): {line[:80]!r}")
    item_id, rest = parts
    tokens = rest.split("|")
    if n_levels is not None:
        if len(tokens) < n_levels:
            raise ValueError(
                f"expected >= {n_levels} tokens, got {len(tokens)}: {line[:80]!r}"
            )
        tokens = tokens[:n_levels]
    return item_id, tokens


def format_id_line(item_id: str, tokens: list[str]) -> str:
    """Inverse of parse_id_line — produces canonical output line."""
    return f"{item_id} |" + "|".join(tokens)


def infer_n_levels(path: Path) -> int:
    """Peek at the first non-empty line of an id file to determine level count."""
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            _iid, tokens = parse_id_line(line)
            return len(tokens)
    raise ValueError(f"id file has no non-empty lines: {path}")


def read_id_file(path: Path) -> dict[str, list[str]]:
    n_levels = infer_n_levels(path)
    out: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item_id, tokens = parse_id_line(line, n_levels=n_levels)
            out[item_id] = tokens
    return out


def build_vocab_from_id_file(path: Path) -> HierIdVocab:
    """Build per-level token vocabularies from all items in the id file.

    Includes every item present in the file (warm + cold); this ensures the
    MLP's output space covers all tokens GRAM's original tokenizer knew about.
    """
    n_levels = infer_n_levels(path)
    per_level_seen: list[dict[str, int]] = [dict() for _ in range(n_levels)]
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            _item_id, tokens = parse_id_line(line, n_levels=n_levels)
            for i, t in enumerate(tokens):
                if t not in per_level_seen[i]:
                    per_level_seen[i][t] = len(per_level_seen[i])
    v = HierIdVocab()
    v.per_level_token_to_idx = per_level_seen
    v.per_level_idx_to_token = [
        [t for t, _ in sorted(d.items(), key=lambda kv: kv[1])]
        for d in per_level_seen
    ]
    return v


def read_item_texts(path: Path) -> dict[str, str]:
    """Read item_plain_text.txt: first space-separated token is item_id, rest is text."""
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            i = line.find(" ")
            if i < 0:
                out[line] = ""
            else:
                out[line[:i]] = line[i + 1:]
    return out


def read_item_set(path: Path) -> set[str]:
    """Read a one-item-per-line file into a set."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def write_id_file(
    path: Path,
    id_map: dict[str, list[str]],
    order_reference: Path | None = None,
):
    """Write id map back to a GRAM-compatible id file.

    If order_reference is given, preserve line order from that file (needed
    because GRAM indexing code may assume line order matches user_sequence
    processing order). Otherwise sort by item_id for determinism.
    """
    if order_reference is not None:
        order = []
        ref_n_levels = infer_n_levels(order_reference)
        with open(order_reference) as f:
            for line in f:
                if not line.strip():
                    continue
                item_id, _tokens = parse_id_line(line, n_levels=ref_n_levels)
                order.append(item_id)
    else:
        order = sorted(id_map.keys())

    with open(path, "w") as f:
        for item_id in order:
            if item_id not in id_map:
                continue
            f.write(format_id_line(item_id, id_map[item_id]) + "\n")
