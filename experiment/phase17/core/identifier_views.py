"""Deterministic multi-path lexical-identifier views used by B0/B1 probes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping


MULTI_PATH_MODULES = {"B0_mvi", "B1_latte"}


def enabled_module_names(value: str | list[str] | tuple[str, ...] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def _parts(identifier: str) -> list[str]:
    parts = [part for part in identifier.split("|") if part]
    if not parts:
        raise ValueError(f"identifier has no split-token parts: {identifier!r}")
    return parts


def _join(parts: list[str]) -> str:
    return "".join(f"|{part}" for part in parts)


def build_identifier_views(
    item2lexid: Mapping[str, str], module_id: str
) -> dict[str, list[str]]:
    """Create two legal, deterministic paths per item without changing its suffix.

    B0 provides an independently ordered native-token view.  B1 keeps the
    complete native path and adds a hash-bucket latent root drawn from a small
    fixed root vocabulary.  Full metadata/query views and learned roots are
    intentionally deferred until a probe passes.
    """

    if module_id not in MULTI_PATH_MODULES:
        return {item: [identifier] for item, identifier in item2lexid.items()}
    root_vocabulary = ["▁the", "▁a", "▁new", "▁one"]
    result: dict[str, list[str]] = {}
    occupied: dict[str, str] = {
        identifier: item_id for item_id, identifier in item2lexid.items()
    }
    for item_id in sorted(item2lexid):
        native = item2lexid[item_id]
        parts = _parts(native)
        if module_id == "B0_mvi":
            # A second native-token surface exposes the same item through a
            # different autoregressive prefix while preserving all information.
            candidates = []
            if len(parts) > 1:
                candidates.append(_join(parts[1:] + parts[:1]))
            candidates.extend(
                _join([root, *parts[1:], parts[0]]) for root in root_vocabulary
            )
        else:
            digest = hashlib.sha256(item_id.encode("utf-8")).digest()
            root = root_vocabulary[digest[0] % len(root_vocabulary)]
            candidates = [_join([root, *parts])]
            candidates.extend(
                _join([fallback_root, root, *parts])
                for fallback_root in root_vocabulary
            )
        alternative = next(
            (candidate for candidate in candidates if candidate not in occupied),
            native,
        )
        views = [native]
        if alternative != native:
            views.append(alternative)
        for view in views:
            previous = occupied.setdefault(view, item_id)
            if previous != item_id:
                raise ValueError(
                    f"multi-path identifier collision: {previous} and {item_id} -> {view}"
                )
        result[item_id] = views
    return result


def flatten_views(item2views: Mapping[str, list[str]]) -> list[str]:
    return [view for item_id in sorted(item2views) for view in item2views[item_id]]


def select_training_view(views: list[str], sample_index: int) -> str:
    if not views:
        raise ValueError("an item must expose at least one identifier view")
    return views[sample_index % len(views)]


def decoded_identifier(tokenizer, identifier: str) -> str:
    token_ids = [token for token in tokenizer.encode(identifier) if token not in {1820, 9175}]
    return tokenizer.decode(token_ids, skip_special_tokens=True)
