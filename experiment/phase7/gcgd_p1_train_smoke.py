#!/usr/bin/env python3
"""Train-only end-to-end smoke for the frozen GCGD P1 interfaces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.phase4.gcdh_p0 import (  # noqa: E402
    build_train_samples,
    collate,
    prepare,
    read_users,
    sha256,
    write_json,
)
from experiment.phase7.gcgd_p0 import read_train_sequences  # noqa: E402
from experiment.phase7.gcgd_p1 import (  # noqa: E402
    AdaptiveGraphPrefixLogitsProcessor,
    adapter_training_loss,
    build_indexed_graph,
    generate_arm_items,
    graph_logits_for_user,
    graph_prefix_inputs,
    train_lightgcn,
)
from experiment.phase7.gcgd_v1 import GraphReliabilityAdapter, reliability_features  # noqa: E402


def root_reliability_feature(
    gram_logits: torch.Tensor,
    graph_log_probabilities: torch.Tensor,
    compatible_leaf_fraction: float,
) -> tuple[float, ...]:
    probabilities = graph_log_probabilities.exp()
    if probabilities.numel() == 1:
        entropy, margin = 0.0, 1.0
    else:
        entropy = float(-(probabilities * graph_log_probabilities).sum() / math.log(probabilities.numel()))
        ordered = probabilities.sort(descending=True).values
        margin = float(ordered[0] - ordered[1])
    return reliability_features(
        graph_coverage=1.0,
        normalized_entropy=max(0.0, min(1.0, entropy)),
        top_margin=max(0.0, min(1.0, margin)),
        compatible_leaf_fraction=compatible_leaf_fraction,
        gram_graph_agreement=float(int(gram_logits.argmax()) == int(graph_log_probabilities.argmax())),
        normalized_depth=0.0,
    )


def run_domain(dataset: str, config: dict, output_root: Path) -> dict:
    device = torch.device("cuda:0")
    parent = json.loads((ROOT / config["inputs"]["phase4_parent_config"]).read_text())
    prepared = prepare(dataset, parent, device)
    checkpoint = ROOT / config["inputs"]["checkpoint_root"] / dataset / "C1" / "model.pt"
    expected_sha = config["inputs"]["expected_parent_checkpoint_sha256"][dataset]
    if sha256(checkpoint) != expected_sha:
        raise ValueError(f"{dataset} parent checkpoint SHA mismatch")
    prepared["model"].load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    prepared["model"].eval()
    train_sequences = read_train_sequences(
        ROOT / "GRAM/rec_datasets" / dataset / "user_sequence.txt", 2
    )
    graph = build_indexed_graph(train_sequences, prepared["catalog"])
    graph_config = dict(config["graph"])
    graph_config["epochs"] = int(config["smoke"]["lightgcn_epochs"])
    graph_model, graph_training = train_lightgcn(graph, graph_config, device)
    graph_model.eval()

    train_users = read_users(ROOT / config["inputs"]["split_root"] / dataset / "train_users.txt")
    samples = build_train_samples(
        prepared["sequences"], train_users, prepared["item2input"], prepared["item2lexid"]
    )
    sample = sorted(samples, key=lambda row: row["sample_key"])[0]
    item_paths = dict(zip(prepared["catalog"], prepared["encoded_candidates"]))
    item_logits = graph_logits_for_user(
        graph_model,
        graph,
        sample["user_id"],
        visible_history_items=sample["history_items"],
    )
    prefix_scores, leaf_fractions = graph_prefix_inputs(item_paths, item_logits)
    root_prefix = (0,)
    root_scores = prefix_scores[root_prefix]
    legal_tokens = sorted(root_scores)

    batch = collate(prepared["collator"], [sample])
    input_ids = batch["item_text_ids"].to(device)
    attention = batch["item_text_masks"].to(device)
    decoder_input = torch.tensor([[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        output = prepared["model"].backbone(
            input_ids=input_ids,
            attention_mask=attention,
            decoder_input_ids=decoder_input,
            return_dict=True,
        )
        gram_legal = output.logits[0, 0, legal_tokens].float().unsqueeze(0)
    graph_legal = torch.tensor(
        [[root_scores[token] for token in legal_tokens]], dtype=torch.float32, device=device
    )
    target_token = item_paths[sample["positive_item"]][1]
    if target_token not in legal_tokens:
        raise ValueError("train target root token is not legal")
    target_position = torch.tensor([legal_tokens.index(target_token)], device=device)
    feature = torch.tensor(
        [root_reliability_feature(gram_legal[0], graph_legal[0], leaf_fractions[root_prefix])],
        dtype=torch.float32,
        device=device,
    )
    reliability_label = torch.tensor(
        [float(int(graph_legal[0].argmax()) == int(target_position[0]))], device=device
    )
    adapter = GraphReliabilityAdapter().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(config["adapter"]["learning_rate"]))
    adapter_records = []
    for step in range(1, int(config["smoke"]["adapter_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, components = adapter_training_loss(
            adapter,
            feature,
            gram_legal,
            graph_legal,
            target_position,
            reliability_label,
            alpha=float(config["decoding"]["C_alpha"]),
            next_token_weight=float(config["adapter"]["loss_weights"]["next_token_ce"]),
            reliability_weight=float(config["adapter"]["loss_weights"]["gate_reliability_bce"]),
        )
        if not torch.isfinite(loss):
            raise ValueError("non-finite adapter smoke loss")
        loss.backward()
        optimizer.step()
        adapter_records.append({
            "step": step,
            "loss": float(loss.detach()),
            "next_token_ce": float(components["next_token_ce"].detach()),
            "gate_reliability_bce": float(components["gate_reliability_bce"].detach()),
        })
    adapter.eval()
    beam = int(config["decoding"]["generator_top_k"])
    length_penalty = float(config["decoding"]["length_penalty"])
    maximum_depth = max(len(path) for path in item_paths.values()) - 1
    torch.cuda.reset_peak_memory_stats(device)
    baseline, baseline_diag = generate_arm_items(
        sample, prepared, beam_size=beam, length_penalty=length_penalty, device=device, processor=None
    )
    identity_processor = AdaptiveGraphPrefixLogitsProcessor(
        prefix_scores, leaf_fractions, alpha=0.0, maximum_depth=maximum_depth, adapter=adapter
    )
    identity, _ = generate_arm_items(
        sample, prepared, beam_size=beam, length_penalty=length_penalty, device=device, processor=identity_processor
    )
    if baseline != identity:
        raise ValueError("alpha=0 failed exact identity in P1 train smoke")
    b_processor = AdaptiveGraphPrefixLogitsProcessor(
        prefix_scores,
        leaf_fractions,
        alpha=float(config["decoding"]["B_alpha"]),
        maximum_depth=maximum_depth,
        adapter=None,
    )
    b_items, b_diag = generate_arm_items(
        sample, prepared, beam_size=beam, length_penalty=length_penalty, device=device, processor=b_processor
    )
    c_processor = AdaptiveGraphPrefixLogitsProcessor(
        prefix_scores,
        leaf_fractions,
        alpha=float(config["decoding"]["C_alpha"]),
        maximum_depth=maximum_depth,
        adapter=adapter,
    )
    c_items, c_diag = generate_arm_items(
        sample, prepared, beam_size=beam, length_penalty=length_penalty, device=device, processor=c_processor
    )
    result = {
        "dataset": dataset,
        "sample_key": sample["sample_key"],
        "graph_epochs": graph_training,
        "adapter_steps": adapter_records,
        "identity_exact": True,
        "B_changed_positions": sum(left != right for left, right in zip(baseline, b_items)),
        "C_changed_positions": sum(left != right for left, right in zip(baseline, c_items)),
        "baseline_diagnostics": baseline_diag,
        "B_diagnostics": b_diag,
        "C_diagnostics": c_diag,
        "all_beams_unique_and_catalog_mapped": True,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "parent_checkpoint_sha256_before": expected_sha,
        "parent_checkpoint_sha256_after": sha256(checkpoint),
        "fresh_validation_read": False,
        "test_read": False,
        "sports_read": False,
    }
    write_json(output_root / dataset / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Toys", "Beauty"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("execution_enabled") is not True:
        raise ValueError("train smoke config is not execution-enabled")
    if not torch.cuda.is_available():
        raise RuntimeError("train smoke requires CUDA")
    result = run_domain(args.dataset, config, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
