#!/usr/bin/env python3
"""Freeze the complete 3x256 semantic + 8 latent GRAM vocabulary.

The completed tokenizer attempt exported tokens observed in catalog paths.  One
valid codebook token was not observed, yielding 775 rather than the preregistered
776-token initialization surface.  This amendment is additive: it does not edit
the immutable tokenizer attempt or its official semantic-ID cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from experiment.phase17.core.full_latte_arm_contracts import full_semantic_vocabulary
from experiment.phase17.core.status_writer import atomic_json, utc_now


ROOT = Path(__file__).resolve().parents[3]
AMENDMENT_ID = "amendment_001"
SOURCE_STATUS = Path("artifacts/phase17/status/s17_fp0_full_data_tokenizer.status.json")
SOURCE_OUTPUT = Path(
    "artifacts/phase17/fullport/fp0/full_data_tokenizer/attempt_001/tokenizer"
)
OUTPUT = Path(
    "artifacts/phase17/fullport/fp0/full_data_tokenizer/amendment_001"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_amendment(root: Path, *, apply: bool) -> dict[str, Any]:
    root = root.resolve()
    status_path = root / SOURCE_STATUS
    source_dir = root / SOURCE_OUTPUT
    output_dir = root / OUTPUT
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status["scientific_state"] != "COMPLETED"
        or status["status_code"] != "PASS_S17_FP0_FULL_DATA_TOKENIZER"
    ):
        raise RuntimeError("full-data tokenizer is not a completed PASS dependency")
    source_manifest = source_dir / "manifest.json"
    if sha256(source_manifest) != status["tokenizer_manifest_sha256"]:
        raise RuntimeError("source tokenizer manifest hash drift")
    observed_path = source_dir / "gram_added_tokens.txt"
    observed = tuple(line for line in observed_path.read_text(encoding="utf-8").splitlines() if line)
    complete = full_semantic_vocabulary()
    if len(complete) != 776 or len(set(complete)) != 776:
        raise AssertionError("full vocabulary must contain exactly 776 unique tokens")
    if not set(observed) <= set(complete):
        raise RuntimeError("observed tokenizer vocabulary contains an invalid token")
    missing = tuple(token for token in complete if token not in set(observed))
    unexpected = tuple(token for token in observed if token not in set(complete))
    if len(observed) != 775 or len(missing) != 1 or unexpected:
        raise RuntimeError(
            "the frozen amendment applies only to the audited 775/776 vocabulary gap"
        )

    vocab_path = output_dir / "gram_full_added_tokens.txt"
    manifest_path = output_dir / "manifest.json"
    payload = {
        "schema_version": "phase17.s17_fp0_full_vocab_amendment.v1",
        "amendment_id": AMENDMENT_ID,
        "created_at": utc_now(),
        "source_attempt_id": status["attempt_id"],
        "source_status_path": SOURCE_STATUS.as_posix(),
        "source_status_sha256": sha256(status_path),
        "source_tokenizer_manifest_path": status["tokenizer_manifest_path"],
        "source_tokenizer_manifest_sha256": status["tokenizer_manifest_sha256"],
        "source_observed_vocabulary_path": str(observed_path.relative_to(root)),
        "source_observed_vocabulary_sha256": sha256(observed_path),
        "source_observed_token_count": len(observed),
        "complete_vocabulary_path": str(vocab_path.relative_to(root)),
        "complete_token_count": len(complete),
        "complete_unique_token_count": len(set(complete)),
        "semantic_token_count": 3 * 256,
        "latent_token_count": 8,
        "missing_observed_tokens_added": list(missing),
        "unexpected_observed_tokens": list(unexpected),
        "g1_g2_token_inventory_identical": True,
        "g1_g2_initialization_seed": 2023,
        "original_attempt_modified": False,
        "official_semantic_ids_modified": False,
        "external_target_materialized": False,
        "effect_experiment_started": False,
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
    }
    if apply:
        if output_dir.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("complete_token_count") != 776
                or not vocab_path.is_file()
                or tuple(vocab_path.read_text(encoding="utf-8").splitlines()) != complete
            ):
                raise FileExistsError("existing vocabulary amendment differs from contract")
            return existing
        output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_lines(vocab_path, complete)
        payload["complete_vocabulary_sha256"] = sha256(vocab_path)
        atomic_json(manifest_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_amendment(args.root, apply=args.apply),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
