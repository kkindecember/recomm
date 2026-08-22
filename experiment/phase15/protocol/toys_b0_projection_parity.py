"""Stage15 S2 first Toys contract smoke: frozen B0 projection parity."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
    make_dataset_args,
)
from utils.generation_trie import Trie, prefix_allowed_tokens_fn  # noqa: E402

from common_adapter import (  # noqa: E402
    build_legacy_validation_view,
    compare_rankings,
    read_projected_sequences,
    read_validation_predictions,
    sha256_file,
    stable_user_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projected-sequences", type=Path, required=True)
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--historical-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--item-path-file", type=Path, required=True)
    parser.add_argument("--frozen-validation-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=1502)
    parser.add_argument("--beam-size", type=int, default=50)
    return parser.parse_args()


def _ensure_new_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_runtime_files = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [
        path.name for path in output_dir.iterdir() if path.name not in allowed_runtime_files
    ]
    if unexpected:
        raise FileExistsError(f"Refusing existing scientific artifacts: {unexpected}")


def _encoded_catalog_candidates(tokenizer, candidates: list[str]) -> list[list[int]]:
    encoded: list[list[int]] = []
    for candidate in candidates:
        tokens = [token for token in tokenizer.encode(candidate) if token not in (1820, 9175)]
        encoded.append([0, *tokens])
    if not encoded or any(sequence[-1] != tokenizer.eos_token_id for sequence in encoded):
        raise ValueError("Every encoded catalog path must terminate with EOS")
    return encoded


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    projected = args.projected_sequences.resolve()
    source_dataset = args.source_dataset_dir.resolve()
    historical_path = args.historical_config.resolve()
    checkpoint = args.checkpoint.resolve()
    item_path = args.item_path_file.resolve()
    frozen_predictions = args.frozen_validation_predictions.resolve()
    output_dir = args.output_dir.resolve()
    _ensure_new_outputs(output_dir)

    if projected.name != "user_sequence_train_validation.txt":
        raise ValueError("Refusing a non-projected sequence input")
    if args.beam_size != 50:
        raise ValueError("Stage15 B0 parity freezes beam_size=50")
    for path in (
        projected,
        historical_path,
        checkpoint,
        item_path,
        frozen_predictions,
        source_dataset / "item_plain_text.txt",
        source_dataset / "similar_item_sasrec.txt",
        source_dataset / "cold_split_meta" / "cold_items.txt",
        source_dataset / "cold_split_meta" / "warm_items.txt",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if "test" in frozen_predictions.name.lower():
        raise ValueError("Refusing test predictions")

    projected_rows = read_projected_sequences(projected)
    frozen_rows = read_validation_predictions(frozen_predictions)
    selected_users = stable_user_sample(
        list(projected_rows), args.sample_size, args.sample_seed
    )
    missing_frozen = set(selected_users) - frozen_rows.keys()
    if missing_frozen:
        raise ValueError(f"Selected users missing frozen B0 rows: {len(missing_frozen)}")

    view_dir = output_dir / "dataset_view" / "Toys_cold50"
    view_manifest = build_legacy_validation_view(
        projected_sequences=projected,
        selected_users=selected_users,
        source_dataset_dir=source_dataset,
        item_path_file=item_path,
        view_dataset_dir=view_dir,
    )
    atomic_json(output_dir / "dataset_view_manifest.json", view_manifest)

    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    if historical["backbone"] != "t5-small" or historical["beam_size"] != 50:
        raise ValueError("Historical backbone/beam does not match the frozen contract")
    dataset_args = make_dataset_args(historical, view_dir)
    dataset_args.cf0_phase9 = 1
    dataset_args.valid_by_test = 0
    dataset_args.test_by_valid = 0
    dataset_args.debug_test_on_train = 0
    tokenizer = AutoTokenizer.from_pretrained(historical["backbone"])
    dataset = TestDatasetGRAM(
        args=dataset_args,
        dataset=view_dir.name,
        task="sequential",
        model_gen=None,
        tokenizer=tokenizer,
        regenerate=False,
        phase=0,
        debug_test_small_set=False,
        mode="validation",
    )
    if len(dataset) != args.sample_size:
        raise RuntimeError(f"Dataset size mismatch: {len(dataset)} != {args.sample_size}")
    for sample in dataset.data_samples:
        expected_target = projected_rows[str(sample["user_id"])][-1]
        if sample["target"] != expected_target:
            raise RuntimeError(f"Projected validation target mismatch for {sample['user_id']}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=CollatorGRAM(tokenizer, args=dataset_args, mode="valid"),
    )
    item_to_lexical, decoded_to_items = load_item_paths(item_path)
    if any(len(items) != 1 for items in decoded_to_items.values()):
        raise RuntimeError("B0 smoke requires collision-free catalog paths")
    candidates = list(dataset.all_items)
    if len(candidates) != len(item_to_lexical):
        raise RuntimeError("Dataset candidate catalog size mismatch")
    encoded_candidates = _encoded_catalog_candidates(tokenizer, candidates)
    trie = Trie(encoded_candidates)
    allowed_tokens = prefix_allowed_tokens_fn(trie)
    max_length = max(map(len, encoded_candidates))

    device = torch.device(args.device)
    torch.manual_seed(2023)
    model = configure_model(historical, checkpoint, device)
    predictions: list[dict] = []
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader, 1):
            batch = batch_to_device(raw_batch, device)
            generated = model.generate(
                input_ids=batch["item_text_ids"],
                attention_mask=batch["item_text_masks"],
                history_item_ids=batch["history_item_ids"],
                history_item_mask=batch["history_item_mask"],
                max_length=max_length,
                prefix_allowed_tokens_fn=allowed_tokens,
                num_beams=args.beam_size,
                num_return_sequences=args.beam_size,
                output_scores=True,
                return_dict_in_generate=True,
                length_penalty=historical["length_penalty"],
            )
            decoded = tokenizer.batch_decode(
                generated["sequences"], skip_special_tokens=True
            )
            observed: list[str] = []
            for value in decoded:
                matches = decoded_to_items.get(value, [])
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Generated path has {len(matches)} catalog mappings: {value!r}"
                    )
                observed.append(matches[0])
            user = str(raw_batch["user_ids"][0])
            expected = list(frozen_rows[user]["v0_top50"])
            comparison = compare_rankings(observed, expected)
            predictions.append(
                {
                    "user_id": user,
                    "observed_v0_top50": observed,
                    "frozen_v0_top50": expected,
                    **comparison,
                }
            )
            print(
                f"[b0-parity] users={batch_index}/{len(dataset)} exact={comparison['exact']}",
                flush=True,
            )

    with (output_dir / "predictions_validation.jsonl").open(
        "x", encoding="utf-8"
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    exact_users = sum(int(row["exact"]) for row in predictions)
    verdict = (
        "PASS_B0_PROJECTION_PARITY"
        if exact_users == len(predictions)
        else "FAIL_B0_PROJECTION_PARITY"
    )
    input_paths = {
        "projected_sequences": projected,
        "historical_config": historical_path,
        "checkpoint": checkpoint,
        "item_path_file": item_path,
        "frozen_validation_predictions": frozen_predictions,
        "item_plain_text": source_dataset / "item_plain_text.txt",
        "similar_item_sasrec_b0_historical_only": source_dataset / "similar_item_sasrec.txt",
        "cold_items": source_dataset / "cold_split_meta" / "cold_items.txt",
        "warm_items": source_dataset / "cold_split_meta" / "warm_items.txt",
    }
    hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    config_payload = {
        "experiment_id": "GRAM_STAGE15_S2_TOYS_B0_PROJECTION_PARITY_SMOKE",
        "domain": "Toys_cold50",
        "split": "validation",
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "selection_rule": "lowest sha256(sample_seed:user_id), independent of target",
        "beam_size": args.beam_size,
        "batch_size": 1,
        "device": args.device,
        "model_training": False,
        "test_read": False,
    }
    summary = {
        **config_payload,
        "status": "completed",
        "verdict": verdict,
        "users": len(predictions),
        "exact_users": exact_users,
        "mismatched_users": len(predictions) - exact_users,
        "all_outputs_unique_catalog_items": True,
        "projected_target_parity": True,
        "model_or_adapter_opened_original_sequence": False,
        "test_predictions_opened": False,
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    atomic_json(output_dir / "config.json", config_payload)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "input_file_sha256.json", hashes)
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "source": "Stage15 audited Toys train+validation projection",
            "view_adapter": "fixed non-catalog sealed slot for historical GRAM [-2] validation semantics",
            "selected_users_sha256": view_manifest["selected_users_sha256"],
            "test_target_materialized": False,
            "test_target_used": False,
        },
    )
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened": [str(path.relative_to(REPO_ROOT)) for path in input_paths.values()],
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
            "model_forward_users": len(predictions),
            "beam_size": args.beam_size,
        },
    )
    print(json.dumps({"status": "completed", "verdict": verdict, "exact_users": exact_users}))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
