"""Stage15 S2 Toys GPU smoke for the B2 verifier hook and B3 layer probe.

The smoke performs no optimization.  It scores a target-independent mixture of
warm/cold catalog candidates with the frozen GRAM checkpoint and probes decoder
layers using train-only next-item transitions.  Validation targets are never
used for candidate selection, acceptance tuning, or layer selection.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE14_PROTOCOL = REPO_ROOT / "experiment" / "phase14" / "protocol"
if str(PHASE14_PROTOCOL) not in sys.path:
    sys.path.insert(0, str(PHASE14_PROTOCOL))

from item_level_eval import atomic_json, load_item_paths  # noqa: E402
from oracle_prefix_probe import (  # noqa: E402
    CollatorGRAM,
    TestDatasetGRAM,
    batch_to_device,
    configure_model,
    encode_lexical_path,
    make_dataset_args,
)
from data import MultiTaskDatasetGRAM  # noqa: E402

from common_adapter import (  # noqa: E402
    TrainTransition,
    build_legacy_validation_view,
    iter_train_transitions,
    read_projected_sequences,
    sha256_file,
    stable_user_sample,
)
from genrecedit_gram_adapter import (  # noqa: E402
    accumulate_probe_predictions,
    merge_probe_counts,
    probe_accuracy_from_counts,
    select_probe_layers,
)
from specgr_gram_adapter import (  # noqa: E402
    PathCatalog,
    candidate_token_log_probabilities,
    padded_candidate_labels,
    score_candidate_paths_with_frozen_gram,
    target_aware_scores_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-sequences", type=Path, required=True)
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--historical-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--item-path-file", type=Path, required=True)
    parser.add_argument("--contract-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verifier-users", type=int, default=16)
    parser.add_argument("--candidates-per-split", type=int, default=16)
    parser.add_argument("--candidate-chunk-size", type=int, default=8)
    parser.add_argument("--probe-transitions", type=int, default=64)
    parser.add_argument("--probe-long-path-minimum", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1502)
    return parser.parse_args()


def _ensure_new_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_files = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in runtime_files]
    if unexpected:
        raise FileExistsError(f"Refusing existing scientific artifacts: {unexpected}")


def _read_set(path: Path) -> set[str]:
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError(f"Empty item set: {path}")
    return values


def _stable_rank(seed: int, *parts: object) -> bytes:
    return hashlib.sha256(":".join([str(seed), *map(str, parts)]).encode("utf-8")).digest()


def _candidate_rows(
    *, users: Iterable[str], warm: set[str], cold: set[str], count: int, seed: int
) -> dict[str, list[str]]:
    if count < 1 or count > min(len(warm), len(cold)):
        raise ValueError("Invalid per-split verifier candidate count")
    rows: dict[str, list[str]] = {}
    for user in users:
        warm_ranked = sorted(warm, key=lambda item: (_stable_rank(seed, "warm", user, item), item))
        cold_ranked = sorted(cold, key=lambda item: (_stable_rank(seed, "cold", user, item), item))
        interleaved = [item for pair in zip(warm_ranked[:count], cold_ranked[:count]) for item in pair]
        if len(interleaved) != 2 * count or len(set(interleaved)) != len(interleaved):
            raise RuntimeError("Verifier candidate construction violated its exact budget")
        rows[user] = interleaved
    return rows


def _select_probe_transitions(
    transitions: list[TrainTransition],
    *,
    path_lengths: dict[str, int],
    sample_size: int,
    long_path_minimum: int,
    seed: int,
) -> list[TrainTransition]:
    if sample_size < long_path_minimum or sample_size > len(transitions):
        raise ValueError("Invalid train-only probe sample contract")
    max_depth = max(path_lengths.values())
    ranked = sorted(
        transitions,
        key=lambda row: (_stable_rank(seed, "probe", row.user_id, len(row.history)), row.user_id, len(row.history)),
    )
    longest = [row for row in ranked if path_lengths[row.target] == max_depth]
    if len(longest) < long_path_minimum:
        raise ValueError("Insufficient train-only longest-path transitions for position coverage")
    selected = longest[:long_path_minimum]
    selected_keys = {(row.user_id, len(row.history)) for row in selected}
    for row in ranked:
        key = (row.user_id, len(row.history))
        if key in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(key)
        if len(selected) == sample_size:
            break
    if len(selected) != sample_size:
        raise RuntimeError("Could not fill the train-only probe sample")
    return selected


def _train_sample_lookup(samples: Iterable[dict]) -> dict[tuple[str, int], tuple[int, str]]:
    """Index augmented train samples by chronological target position.

    GRAM truncates ``history_item_ids`` to ``max_his``.  Consequently its
    length is not a unique transition key once a user history exceeds that
    limit.  Dataset order remains chronological, so the per-user one-based
    sample ordinal exactly matches ``len(TrainTransition.history)``.
    """

    ordinal: dict[str, int] = collections.defaultdict(int)
    lookup: dict[tuple[str, int], tuple[int, str]] = {}
    for index, sample in enumerate(samples):
        user = str(sample["user_id"])
        ordinal[user] += 1
        key = (user, ordinal[user])
        if key in lookup:
            raise ValueError("Duplicate chronological GRAM train-sample key")
        lookup[key] = (index, str(sample["target"]))
    return lookup


def _model_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _direct_candidate_score(model, batch: dict, path: tuple[int, ...], score_length: int) -> float:
    labels = padded_candidate_labels([path], device=batch["item_text_ids"].device)
    outputs = model(
        input_ids=batch["item_text_ids"],
        attention_mask=batch["item_text_masks"],
        labels=labels,
    )
    token_logp, mask = candidate_token_log_probabilities(outputs.logits, labels)
    score = target_aware_scores_tensor(token_logp, mask, [score_length])
    return float(score[0].detach().cpu())


def _configure_deterministic_smoke_math() -> dict[str, object]:
    """Remove TF32 batch-shape drift from the hook/direct parity Gate."""

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError(
            "Deterministic CUDA smoke requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": workspace,
    }


def _probe_layer_predictions(model, batch: dict) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer, block in enumerate(model.decoder.block):
        def capture(_module, _inputs, output, layer_index=layer):
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer_index] = value.detach()

        handles.append(block.register_forward_hook(capture))
    try:
        _ = model(
            input_ids=batch["item_text_ids"],
            attention_mask=batch["item_text_masks"],
            labels=batch["target_ids"],
        )
    finally:
        for handle in handles:
            handle.remove()
    expected = set(range(len(model.decoder.block)))
    if set(captured) != expected:
        raise RuntimeError("Decoder hook did not capture every GRAM layer")
    predictions: dict[int, torch.Tensor] = {}
    for layer, hidden in captured.items():
        readout = model.decoder.final_layer_norm(hidden)
        if model.config.tie_word_embeddings:
            readout = readout * (model.model_dim ** -0.5)
        predictions[layer] = model.lm_head(readout).argmax(dim=-1)
    return predictions


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    projected = args.projected_sequences.resolve()
    source_dataset = args.source_dataset_dir.resolve()
    historical_path = args.historical_config.resolve()
    checkpoint = args.checkpoint.resolve()
    item_path = args.item_path_file.resolve()
    contract_state = args.contract_state.resolve()
    output_dir = args.output_dir.resolve()
    _ensure_new_outputs(output_dir)

    if projected.name != "user_sequence_train_validation.txt":
        raise ValueError("Refusing a non-projected sequence input")
    required_inputs = [
        projected,
        historical_path,
        checkpoint,
        item_path,
        contract_state / "summary.json",
        contract_state / "specgr_gram" / "index" / "manifest.json",
        contract_state / "genrecedit_gram" / "edit_requests" / "position_map.json",
        source_dataset / "item_plain_text.txt",
        source_dataset / "similar_item_sasrec.txt",
        source_dataset / "cold_split_meta" / "cold_items.txt",
        source_dataset / "cold_split_meta" / "warm_items.txt",
    ]
    for path in required_inputs:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Input must be a regular non-symlink file: {path}")
    contract_summary = json.loads((contract_state / "summary.json").read_text(encoding="utf-8"))
    if contract_summary.get("verdict") != "PASS_B2_B3_INPUT_CONTRACT":
        raise ValueError("B2/B3 CPU input Gate is not PASS")

    projected_rows = read_projected_sequences(projected)
    item_to_lexical, decoded_to_items = load_item_paths(item_path)
    if any(len(items) != 1 for items in decoded_to_items.values()):
        raise ValueError("GPU hook smoke requires collision-free catalog paths")
    warm = _read_set(source_dataset / "cold_split_meta" / "warm_items.txt")
    cold = _read_set(source_dataset / "cold_split_meta" / "cold_items.txt")
    catalog = PathCatalog.build(item_to_lexical, warm, cold)

    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    if historical.get("backbone") != "t5-small" or int(historical.get("beam_size", 0)) != 50:
        raise ValueError("Historical GRAM contract does not match frozen Toys v0")
    tokenizer = AutoTokenizer.from_pretrained(historical["backbone"])
    encoded_paths = {item: encode_lexical_path(tokenizer, lexical) for item, lexical in item_to_lexical.items()}
    path_lengths = {item: len(path) for item, path in catalog.paths.items()}
    if any(len(encoded_paths[item]) != path_lengths[item] for item in catalog.paths):
        raise ValueError("Lexical segment count does not match GRAM token count")

    verifier_users = stable_user_sample(list(projected_rows), args.verifier_users, args.seed)
    candidate_items = _candidate_rows(
        users=verifier_users,
        warm=warm,
        cold=cold,
        count=args.candidates_per_split,
        seed=args.seed,
    )
    verifier_view = output_dir / "dataset_view_verifier" / "Toys_cold50"
    verifier_view_manifest = build_legacy_validation_view(
        projected_sequences=projected,
        selected_users=verifier_users,
        source_dataset_dir=source_dataset,
        item_path_file=item_path,
        view_dataset_dir=verifier_view,
    )
    atomic_json(output_dir / "dataset_view_verifier_manifest.json", verifier_view_manifest)

    dataset_args = make_dataset_args(historical, verifier_view)
    dataset_args.cf0_phase9 = 1
    dataset_args.valid_by_test = 0
    dataset_args.test_by_valid = 0
    dataset_args.debug_test_on_train = 0
    verifier_dataset = TestDatasetGRAM(
        args=dataset_args,
        dataset=verifier_view.name,
        task="sequential",
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        debug_test_small_set=False,
        mode="validation",
    )
    verifier_loader = DataLoader(
        verifier_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=CollatorGRAM(tokenizer, args=dataset_args, mode="valid"),
    )

    transitions = list(iter_train_transitions(projected_rows))
    selected_transitions = _select_probe_transitions(
        transitions,
        path_lengths=path_lengths,
        sample_size=args.probe_transitions,
        long_path_minimum=args.probe_long_path_minimum,
        seed=args.seed,
    )
    probe_users = sorted({row.user_id for row in selected_transitions})
    probe_view = output_dir / "dataset_view_probe" / "Toys_cold50"
    probe_view_manifest = build_legacy_validation_view(
        projected_sequences=projected,
        selected_users=probe_users,
        source_dataset_dir=source_dataset,
        item_path_file=item_path,
        view_dataset_dir=probe_view,
    )
    atomic_json(output_dir / "dataset_view_probe_manifest.json", probe_view_manifest)
    probe_args = make_dataset_args(historical, probe_view)
    probe_args.cf0_phase9 = 1
    probe_args.valid_by_test = 0
    probe_args.test_by_valid = 0
    probe_args.debug_test_on_train = 0
    probe_dataset = MultiTaskDatasetGRAM(
        args=probe_args,
        dataset=probe_view.name,
        mode="train",
        model_gen=None,
        tokenizer=tokenizer,
        phase=0,
        regenerate=False,
    )
    sample_lookup = _train_sample_lookup(probe_dataset.data_samples)
    probe_indices = []
    for transition in selected_transitions:
        index, observed_target = sample_lookup[(transition.user_id, len(transition.history))]
        if observed_target != transition.target:
            raise RuntimeError("GRAM train view does not match audited train-only transition")
        probe_indices.append(index)
    probe_loader = DataLoader(
        Subset(probe_dataset, probe_indices),
        batch_size=args.probe_batch_size,
        shuffle=False,
        collate_fn=CollatorGRAM(tokenizer, args=probe_args, mode="train"),
    )

    device = torch.device(args.device)
    numerical_mode = _configure_deterministic_smoke_math()
    torch.manual_seed(2023)
    model = configure_model(historical, checkpoint, device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    state_hash_before = _model_state_sha256(model)

    verifier_rows = []
    direct_pairs = []
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(verifier_loader):
            batch = batch_to_device(raw_batch, device)
            user = str(raw_batch["user_ids"][0])
            items = candidate_items[user]
            token_rows = [[encoded_paths[item] for item in items]]
            length_rows = [[catalog.score_length(item) for item in items]]
            hook = score_candidate_paths_with_frozen_gram(
                model=model,
                batch=batch,
                candidate_token_ids=token_rows,
                score_lengths=length_rows,
                candidate_chunk_size=args.candidate_chunk_size,
            )
            scores = hook["scores"][0].detach().cpu()
            if batch_index < 2:
                for candidate_index in range(4):
                    reference = _direct_candidate_score(
                        model,
                        batch,
                        encoded_paths[items[candidate_index]],
                        length_rows[0][candidate_index],
                    )
                    hook_score = float(scores[candidate_index])
                    direct_pairs.append(
                        {
                            "user_id": user,
                            "item_id": items[candidate_index],
                            "hook_score": hook_score,
                            "direct_score": reference,
                            "absolute_difference": abs(reference - hook_score),
                            "acceptance_equal": (hook_score >= -1.6) == (reference >= -1.6),
                        }
                    )
            verifier_rows.append(
                {
                    "user_id": user,
                    "candidate_selection_uses_validation_target": False,
                    "candidates": [
                        {
                            "item_id": item,
                            "split": "cold" if item in cold else "warm",
                            "path_length": len(catalog.paths[item]),
                            "score_length": length,
                            "target_aware_mean_log_likelihood": float(score),
                            "accepted_at_frozen_initial_threshold": float(score) >= -1.6,
                            "token_log_probabilities": token_logp,
                        }
                        for item, length, score, token_logp in zip(
                            items,
                            length_rows[0],
                            scores.tolist(),
                            hook["token_log_probabilities"][0],
                        )
                    ],
                }
            )
            print(f"[b2-verifier] users={batch_index + 1}/{len(verifier_dataset)}", flush=True)

        probe_counts: dict[int, dict[int, list[int]]] = {}
        for batch_index, raw_batch in enumerate(probe_loader):
            batch = batch_to_device(raw_batch, device)
            predictions = _probe_layer_predictions(model, batch)
            update = accumulate_probe_predictions(
                predictions_by_layer=predictions,
                labels=batch["target_ids"],
                eos_token_id=tokenizer.eos_token_id,
            )
            merge_probe_counts(probe_counts, update)
            print(f"[b3-probe] batches={batch_index + 1}/{len(probe_loader)}", flush=True)

    decoder_layers = len(model.decoder.block)
    probe_accuracy = probe_accuracy_from_counts(probe_counts, decoder_layers=decoder_layers)
    required_positions = set(range(catalog.max_depth))
    if set(probe_accuracy) != required_positions:
        raise RuntimeError("Train-only probe did not cover every lexical position")
    selected_layers = select_probe_layers(probe_accuracy, decoder_layers=decoder_layers)
    state_hash_after = _model_state_sha256(model)
    unchanged = state_hash_before == state_hash_after
    max_direct_difference = max(
        (row["absolute_difference"] for row in direct_pairs), default=float("inf")
    )
    direct_acceptance_equal = bool(direct_pairs) and all(
        row["acceptance_equal"] for row in direct_pairs
    )
    finite_scores = all(
        torch.isfinite(torch.tensor(candidate["target_aware_mean_log_likelihood"]))
        for row in verifier_rows
        for candidate in row["candidates"]
    )
    verdict = (
        "PASS_B2_VERIFIER_GPU_HOOK_AND_B3_TRAIN_ONLY_PROBE"
        if unchanged and finite_scores and direct_acceptance_equal and max_direct_difference <= 2e-5
        else "FAIL_B2_VERIFIER_GPU_HOOK_OR_B3_TRAIN_ONLY_PROBE"
    )

    with (output_dir / "b2_verifier_scores.jsonl").open("x", encoding="utf-8") as handle:
        for row in verifier_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(
        output_dir / "b2_verifier_direct_parity.json",
        {
            "absolute_tolerance": 2e-5,
            "numerical_mode": numerical_mode,
            "pairs": direct_pairs,
            "max_absolute_difference": max_direct_difference,
            "acceptance_equal": direct_acceptance_equal,
        },
    )
    probe_root = output_dir / "genrecedit_gram" / "probe"
    probe_root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        probe_root / "layer_probe.json",
        {
            "source": "train-only next-item transitions from audited projection",
            "selection_rule": "highest frozen logit-lens token accuracy; shallowest layer tie-break",
            "sample_selection": "16 longest-path train transitions then SHA-ranked train transitions; no validation/test",
            "transitions": len(selected_transitions),
            "users": len(probe_users),
            "counts": {str(p): {str(l): v for l, v in layers.items()} for p, layers in probe_counts.items()},
            "accuracy": {str(p): {str(l): v for l, v in layers.items()} for p, layers in probe_accuracy.items()},
            "selected_layer": {str(position): layer for position, layer in selected_layers.items()},
            "eos_included": False,
            "padding_included": False,
            "validation_used": False,
            "test_used": False,
        },
    )

    input_hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in required_inputs}
    config = {
        "experiment_id": "GRAM_STAGE15_S2_TOYS_B2_VERIFIER_PROBE_SMOKE",
        "split": "validation_for_verifier_train_only_for_probe",
        "verifier_users": args.verifier_users,
        "candidate_budget": 2 * args.candidates_per_split,
        "candidate_chunk_size": args.candidate_chunk_size,
        "probe_transitions": args.probe_transitions,
        "probe_long_path_minimum": args.probe_long_path_minimum,
        "probe_batch_size": args.probe_batch_size,
        "seed": args.seed,
        "device": args.device,
        "acceptance_threshold_initial": -1.6,
        "model_training": False,
        "automatic_retry": False,
        "test_read": False,
        "numerical_mode": numerical_mode,
    }
    summary = {
        **config,
        "status": "completed",
        "verdict": verdict,
        "verifier_scores_finite": finite_scores,
        "verifier_direct_reference_pairs": len(direct_pairs),
        "verifier_max_abs_difference_vs_direct": max_direct_difference,
        "verifier_direct_acceptance_equal": direct_acceptance_equal,
        "verifier_warm_candidates": args.verifier_users * args.candidates_per_split,
        "verifier_cold_candidates": args.verifier_users * args.candidates_per_split,
        "target_aware_variable_score_lengths": True,
        "probe_positions": len(probe_accuracy),
        "probe_decoder_layers": decoder_layers,
        "selected_layer_by_position": {str(position): layer for position, layer in selected_layers.items()},
        "frozen_gram_parameter_hash_before": state_hash_before,
        "frozen_gram_parameter_hash_after": state_hash_after,
        "frozen_gram_parameter_hash_unchanged": unchanged,
        "validation_target_used_for_candidate_selection": False,
        "validation_target_used_for_threshold_tuning": False,
        "validation_target_used_for_layer_probe": False,
        "original_user_sequence_opened": False,
        "test_predictions_opened": False,
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    atomic_json(output_dir / "config.json", config)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "input_file_sha256.json", input_hashes)
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "verifier_histories": "audited projection validation view; target-independent users/candidates",
            "probe_supervision": "audited projection with validation target removed before transition construction",
            "candidate_source": "frozen catalog, equal warm/cold SHA sample",
            "similar_item_sasrec_role": "frozen GRAM prompt reconstruction only; not drafter/request/candidate source",
            "validation_target_used_for_training_or_selection": False,
            "test_target_materialized": False,
            "test_read": False,
        },
    )
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened": sorted(input_hashes),
            "generated_views": [
                str(verifier_view.relative_to(REPO_ROOT)),
                str(probe_view.relative_to(REPO_ROOT)),
            ],
            "original_user_sequence_opened": False,
            "test_predictions_opened": False,
            "test_metrics_opened": False,
        },
    )
    atomic_json(
        output_dir / "resource_summary.json",
        {
            "runtime_seconds": summary["runtime_seconds"],
            "peak_cuda_allocated_mib": summary["peak_cuda_allocated_mib"],
            "model_training": False,
            "verifier_users": args.verifier_users,
            "verifier_candidates": args.verifier_users * 2 * args.candidates_per_split,
            "probe_transitions": args.probe_transitions,
        },
    )
    print(json.dumps({"status": "completed", "verdict": verdict, "summary": str(output_dir / "summary.json")}), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
