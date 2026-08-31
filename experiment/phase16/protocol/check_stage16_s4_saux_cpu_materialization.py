#!/usr/bin/env python3
"""CPU-only materialization gate for the frozen S16-4 S-AUX checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment.phase16.protocol.stage16_s4_toys_validation import (
    FORMAL_ARMS,
    ROOT,
    OfficialUniSRecDrafterGRAM,
    load_json,
    read_metadata,
    read_projected_sequences,
    read_set,
    saux_embedding_views,
    sha256_file,
    validate_config,
    verify_regular,
)


def check(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    for arm in FORMAL_ARMS:
        validate_config(config, arm)

    source_config = verify_regular(
        ROOT,
        config["saux_inference"]["inputs"]["source_training_config"],
        "saux_source_training_config",
    )
    retained = read_set(
        verify_regular(
            ROOT,
            config["saux_inference"]["inputs"]["retained_warm_items"],
            "saux_retained_warm_items",
        )
    )
    pseudo = read_set(
        verify_regular(
            ROOT,
            config["saux_inference"]["inputs"]["pseudo_cold_items"],
            "saux_pseudo_cold_items",
        )
    )

    parent_config_path = verify_regular(ROOT, config["preflight"]["config"], "preflight_config")
    parent = load_json(parent_config_path)
    inputs = {
        name: verify_regular(ROOT, declaration, name)
        for name, declaration in parent["inputs"].items()
    }
    warm = read_set(inputs["warm_items"])
    cold = read_set(inputs["cold_items"])
    if retained & pseudo or retained | pseudo != warm:
        raise ValueError("CPU materialization retained/pseudo/warm partition drift")

    metadata = read_metadata(inputs["item_metadata"])
    ordered_items = sorted(metadata)
    payload = torch.load(inputs["content_embeddings"], map_location="cpu")
    views = saux_embedding_views(
        item_ids=[str(item) for item in payload["item_ids"]],
        embeddings=payload["embeddings"].to(torch.float32),
        retained_items=retained,
        ordered_items=ordered_items,
    )

    checkpoint = torch.load(inputs["saux_checkpoint"], map_location="cpu")
    if checkpoint.get("config_sha256") != sha256_file(source_config):
        raise ValueError("S-AUX checkpoint/source training-config identity drift")
    expected_shape = tuple(checkpoint["model"]["model.plm_embedding.weight"].shape)
    if expected_shape != tuple(views["train_embeddings"].shape):
        raise ValueError("S-AUX checkpoint/train embedding shape drift")

    wrapper = OfficialUniSRecDrafterGRAM(views["train_embeddings"]).cpu().eval()
    wrapper.load_state_dict(checkpoint["model"], strict=True)

    projected = read_projected_sequences(inputs["projected_train_validation_sequences"])
    selected_history: list[str] | None = None
    for sequence in projected.values():
        history = list(sequence[:-1][-20:])
        if set(history) & pseudo:
            selected_history = history
            break
    if selected_history is None or set(selected_history) & cold:
        raise ValueError("No valid pseudo-cold inductive validation history was found")

    history_index = views["history_index"]
    row = torch.zeros((1, 20), dtype=torch.long)
    row[0, : len(selected_history)] = torch.tensor(
        [history_index[item] for item in selected_history], dtype=torch.long
    )
    lengths = torch.tensor([len(selected_history)], dtype=torch.long)
    with torch.inference_mode():
        history_content = views["history_embeddings"][row]
        adapted_history = wrapper.model.moe_adaptor(history_content)
        sequence = F.normalize(wrapper.model.forward(row, adapted_history, lengths), dim=-1)
        candidates = F.normalize(
            wrapper.model.moe_adaptor(views["candidate_embeddings"]), dim=-1
        )
        scores = sequence @ candidates.T
    if scores.shape != (1, len(ordered_items)) or not bool(torch.isfinite(scores).all()):
        raise ValueError("S-AUX CPU inductive forward contract failed")

    return {
        "status": "PASS",
        "checkpoint_config_sha256": checkpoint["config_sha256"],
        "checkpoint_embedding_shape": list(expected_shape),
        "history_embedding_shape": list(views["history_embeddings"].shape),
        "candidate_count": len(ordered_items),
        "inductive_history_contains_pseudo_cold": True,
        "score_shape": list(scores.shape),
        "all_finite": True,
        "gpu_used": False,
        "test_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.config.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
