"""Train-only Toys SpecGR-GRAM auxiliary drafter state smoke.

This job trains only the inductive content drafter on a fixed SHA sample of
train transitions.  The GRAM verifier checkpoint is hashed but never loaded or
registered with the optimizer.  Validation targets are neither materialized in
outputs nor used for training, sampling, model selection, or hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE14_PROTOCOL = REPO_ROOT / "experiment" / "phase14" / "protocol"
if str(PHASE14_PROTOCOL) not in sys.path:
    sys.path.insert(0, str(PHASE14_PROTOCOL))

from item_level_eval import atomic_json  # noqa: E402

from common_adapter import (  # noqa: E402
    iter_train_transitions,
    read_projected_sequences,
    sha256_file,
    stable_user_sample,
)
from specgr_gram_adapter import (  # noqa: E402
    AuxiliaryContentDrafter,
    drafter_cross_entropy,
    rank_drafter_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-sequences", type=Path, required=True)
    parser.add_argument("--item-embeddings", type=Path, required=True)
    parser.add_argument("--item-metadata", type=Path, required=True)
    parser.add_argument("--warm-items", type=Path, required=True)
    parser.add_argument("--cold-items", type=Path, required=True)
    parser.add_argument("--gram-checkpoint", type=Path, required=True)
    parser.add_argument("--contract-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-transitions", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=300)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--feedforward-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--validation-users", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1502)
    return parser.parse_args()


def _ensure_new_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_files = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in runtime_files]
    if unexpected:
        raise FileExistsError(f"Refusing existing scientific artifacts: {unexpected}")


def _read_set(path: Path) -> set[str]:
    result = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not result:
        raise ValueError(f"Empty item set: {path}")
    return result


def _stable_key(seed: int, *parts: object) -> bytes:
    value = ":".join([str(seed), *map(str, parts)])
    return hashlib.sha256(value.encode("utf-8")).digest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_embeddings(path: Path, metadata_path: Path) -> tuple[list[str], torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu")
    required = {"item_ids", "embeddings", "model_name", "pooling", "l2_normalized", "text_source_sha256"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Unexpected item embedding payload")
    item_ids = [str(item) for item in payload["item_ids"]]
    embeddings = payload["embeddings"].float().contiguous()
    if embeddings.ndim != 2 or embeddings.size(0) != len(item_ids):
        raise ValueError("Embedding matrix and item IDs do not align")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Embedding payload has duplicate items")
    if not bool(payload["l2_normalized"]):
        raise ValueError("Drafter requires frozen normalized content embeddings")
    if payload["text_source_sha256"] != sha256_file(metadata_path):
        raise ValueError("Embedding metadata provenance mismatch")
    return item_ids, embeddings, {
        "model_name": payload["model_name"],
        "pooling": payload["pooling"],
        "l2_normalized": payload["l2_normalized"],
        "text_source_sha256": payload["text_source_sha256"],
        "shape": list(embeddings.shape),
    }


def _make_history_row(history: tuple[str, ...], item_to_index: dict[str, int], max_history: int) -> tuple[list[int], int]:
    retained = history[-max_history:]
    if not retained:
        raise ValueError("Drafter transition has an empty history")
    row = [item_to_index[item] for item in retained]
    return row + [-1] * (max_history - len(row)), len(row)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    output_dir = args.output_dir.resolve()
    _ensure_new_outputs(output_dir)
    paths = {
        "projected_sequences": args.projected_sequences.resolve(),
        "item_embeddings": args.item_embeddings.resolve(),
        "item_metadata": args.item_metadata.resolve(),
        "warm_items": args.warm_items.resolve(),
        "cold_items": args.cold_items.resolve(),
        "gram_checkpoint": args.gram_checkpoint.resolve(),
        "contract_summary": (args.contract_state.resolve() / "summary.json"),
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Input {name} must be a regular non-symlink file: {path}")
    if paths["projected_sequences"].name != "user_sequence_train_validation.txt":
        raise ValueError("Drafter must use the audited projected sequence")
    contract = json.loads(paths["contract_summary"].read_text(encoding="utf-8"))
    if contract.get("verdict") != "PASS_B2_B3_INPUT_CONTRACT":
        raise ValueError("B2/B3 input contract is not PASS")
    if args.train_transitions < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Invalid drafter training budget")

    warm = _read_set(paths["warm_items"])
    cold = _read_set(paths["cold_items"])
    if warm & cold:
        raise ValueError("Warm and cold sets overlap")
    item_ids, embeddings, embedding_meta = _load_embeddings(
        paths["item_embeddings"], paths["item_metadata"]
    )
    if warm | cold != set(item_ids):
        raise ValueError("Warm/cold sets do not partition the drafter catalog")
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    warm_indices = {item_to_index[item] for item in warm}
    rows = read_projected_sequences(paths["projected_sequences"])
    transitions = list(iter_train_transitions(rows))
    if args.train_transitions > len(transitions):
        raise ValueError("Requested drafter smoke sample exceeds train transitions")
    selected = sorted(
        transitions,
        key=lambda row: (
            _stable_key(args.seed, "train", row.user_id, len(row.history), row.target),
            row.user_id,
            len(row.history),
        ),
    )[: args.train_transitions]
    if any(row.target not in warm for row in selected):
        raise ValueError("Cold or unknown target entered drafter supervision")
    history_rows, history_lengths, labels = [], [], []
    for transition in selected:
        row, length = _make_history_row(transition.history, item_to_index, args.max_history)
        history_rows.append(row)
        history_lengths.append(length)
        labels.append(item_to_index[transition.target])
    train_histories = torch.tensor(history_rows, dtype=torch.long)
    train_lengths = torch.tensor(history_lengths, dtype=torch.long)
    train_labels = torch.tensor(labels, dtype=torch.long)

    validation_users = stable_user_sample(list(rows), args.validation_users, args.seed)
    validation_histories, validation_lengths = [], []
    for user in validation_users:
        full_projected_row = rows[user]
        history, length = _make_history_row(full_projected_row[:-1], item_to_index, args.max_history)
        validation_histories.append(history)
        validation_lengths.append(length)
    validation_histories_tensor = torch.tensor(validation_histories, dtype=torch.long)
    validation_lengths_tensor = torch.tensor(validation_lengths, dtype=torch.long)

    device = torch.device(args.device)
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Deterministic drafter smoke requires CUBLAS_WORKSPACE_CONFIG")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    numerical_mode = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": workspace,
    }
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AuxiliaryContentDrafter(
        item_content_embeddings=embeddings,
        hidden_size=args.hidden_size,
        max_history=args.max_history,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        feedforward_size=args.feedforward_size,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    trainable_names = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not trainable_names or any("item_content_embeddings" in name for name in trainable_names):
        raise RuntimeError("Drafter optimizer allowlist is invalid")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    initial_state_hash = _state_sha256(model)
    model.eval()
    with torch.inference_mode():
        initial_probe = model(
            validation_histories_tensor[:1].to(device),
            validation_lengths_tensor[:1].to(device),
        ).detach().cpu()

    epoch_losses = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(selected), generator=generator)
        total_loss = 0.0
        examples = 0
        for start in range(0, len(selected), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            histories = train_histories[indices].to(device)
            lengths = train_lengths[indices].to(device)
            batch_labels = train_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(histories, lengths)
            loss = drafter_cross_entropy(
                logits,
                batch_labels,
                warm_catalog_indices=warm_indices,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Drafter produced non-finite loss")
            loss.backward()
            optimizer.step()
            count = int(indices.numel())
            total_loss += float(loss.detach()) * count
            examples += count
        epoch_losses.append(total_loss / examples)
        print(f"[b2-drafter] epoch={epoch + 1}/{args.epochs} loss={epoch_losses[-1]:.8f}", flush=True)

    final_state_hash = _state_sha256(model)
    model.eval()
    with torch.inference_mode():
        final_probe = model(
            validation_histories_tensor[:1].to(device),
            validation_lengths_tensor[:1].to(device),
        ).detach().cpu()
        final_logits = []
        for start in range(0, args.validation_users, args.batch_size):
            final_logits.append(
                model(
                    validation_histories_tensor[start : start + args.batch_size].to(device),
                    validation_lengths_tensor[start : start + args.batch_size].to(device),
                ).detach().cpu()
            )
    score_change = float((final_probe - initial_probe).abs().max())
    state_changed = initial_state_hash != final_state_hash
    predictions = []
    logits_matrix = torch.cat(final_logits)
    cold_topk = 0
    for user, history_row, scores in zip(validation_users, validation_histories, logits_matrix):
        excluded = {item_ids[index] for index in history_row if index >= 0}
        ranking = rank_drafter_items(scores, item_ids, exclude_items=excluded)
        top_items = ranking[: args.top_k]
        if len(top_items) != args.top_k or len(set(top_items)) != args.top_k:
            raise RuntimeError("Drafter top-k output violates exact uniqueness")
        cold_count = sum(item in cold for item in top_items)
        cold_topk += cold_count
        predictions.append(
            {
                "user_id": user,
                "top_items": top_items,
                "cold_items": cold_count,
                "target_used": False,
            }
        )
    gram_sha_before = sha256_file(paths["gram_checkpoint"])
    state_root = output_dir / "specgr_gram" / "drafter"
    state_root.mkdir(parents=True, exist_ok=False)
    state_path = state_root / "drafter_state.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, state_path)
    with (output_dir / "drafter_predictions_validation.jsonl").open("x", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    gram_sha_after = sha256_file(paths["gram_checkpoint"])

    finite_losses = all(torch.isfinite(torch.tensor(value)) for value in epoch_losses)
    verdict = (
        "PASS_B2_TRAIN_ONLY_DRAFTER_STATE_SMOKE"
        if finite_losses and state_changed and score_change > 0 and cold_topk > 0 and gram_sha_before == gram_sha_after
        else "FAIL_B2_TRAIN_ONLY_DRAFTER_STATE_SMOKE"
    )
    config = {
        "experiment_id": "GRAM_STAGE15_S2_TOYS_B2_DRAFTER_STATE_SMOKE",
        "mechanism_name": "SpecGR-GRAM official-mechanism port auxiliary content drafter smoke",
        "split": "train_only_for_drafter_validation_history_only_for_output_contract",
        "train_transitions": args.train_transitions,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "max_history": args.max_history,
        "transformer_layers": args.transformer_layers,
        "attention_heads": args.attention_heads,
        "feedforward_size": args.feedforward_size,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "validation_users": args.validation_users,
        "top_k": args.top_k,
        "seed": args.seed,
        "device": args.device,
        "automatic_retry": False,
        "numerical_mode": numerical_mode,
    }
    summary = {
        **config,
        "status": "completed",
        "verdict": verdict,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "optimizer_parameter_names": trainable_names,
        "epoch_losses": epoch_losses,
        "finite_losses": finite_losses,
        "drafter_state_hash_before": initial_state_hash,
        "drafter_state_hash_after": final_state_hash,
        "drafter_state_changed": state_changed,
        "probe_max_absolute_score_change": score_change,
        "catalog_items": len(item_ids),
        "warm_catalog_items": len(warm),
        "cold_catalog_items": len(cold),
        "cold_items_in_validation_topk": cold_topk,
        "outputs_unique_known_items": True,
        "gram_checkpoint_loaded": False,
        "gram_checkpoint_registered_with_optimizer": False,
        "gram_checkpoint_sha256_before": gram_sha_before,
        "gram_checkpoint_sha256_after": gram_sha_after,
        "gram_checkpoint_unchanged": gram_sha_before == gram_sha_after,
        "validation_target_used_for_training": False,
        "validation_target_used_for_sampling": False,
        "validation_target_used_for_model_selection": False,
        "original_user_sequence_opened": False,
        "test_predictions_opened": False,
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "state_path": str(state_path.relative_to(REPO_ROOT)),
        "state_sha256": sha256_file(state_path),
        "embedding": embedding_meta,
    }
    input_hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths.values()}
    atomic_json(output_dir / "config.json", config)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "input_file_sha256.json", input_hashes)
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "training_supervision": "fixed SHA-ranked train-only next-item transitions",
            "candidate_catalog": "all frozen warm/cold items through BGE content embeddings",
            "validation_role": "history-only output contract; target never used or emitted",
            "test_read": False,
        },
    )
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened": sorted(input_hashes),
            "gram_checkpoint_opened_as_model": False,
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
            "trainable_parameters": summary["trainable_parameters"],
            "state_bytes": state_path.stat().st_size,
        },
    )
    print(json.dumps({"status": "completed", "verdict": verdict, "summary": str(output_dir / "summary.json")}), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
