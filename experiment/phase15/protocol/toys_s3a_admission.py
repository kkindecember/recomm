"""Stage15 S3A Toys item-disjoint 512-event admission for B0/B2/B3.

This is an engineering/scientific-contract admission, not an efficacy run.  It
uses the Stage14 item-disjoint clean base and filtered train data, builds B2/B3
method state before opening held events, then exercises complete beam,
candidate-verification, guided-redrafting, and One-One edited-beam paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE14_PROTOCOL = REPO_ROOT / "experiment" / "phase14" / "protocol"
if str(PHASE14_PROTOCOL) not in sys.path:
    sys.path.insert(0, str(PHASE14_PROTOCOL))

from item_level_eval import atomic_json  # noqa: E402
from oracle_prefix_probe import CollatorGRAM  # noqa: E402
from r2pd_pseudo_cold_screen import (  # noqa: E402
    batch_to_device,
    build_filtered_item_inputs,
    clean_transitions,
    collator_args,
    configure_fresh_model,
    deterministic_rank,
    load_paths,
    make_model_sample,
    normalize_generated,
    read_held_events,
    read_key_value,
    read_set,
    read_train_sequences,
)

from build_toys_b2_b3_contract_inputs import (  # noqa: E402
    choose_occurrence,
    collect_train_occurrences,
    deterministic_topk,
)
from common_adapter import TrainTransition, sha256_file  # noqa: E402
from genrecedit_gram_adapter import (  # noqa: E402
    OneOneGenerationDeltaContext,
    SecondMomentAccumulator,
    accumulate_probe_predictions,
    build_positionwise_requests,
    edited_parameter_name,
    merge_probe_counts,
    probe_accuracy_from_counts,
    select_positionwise_smoke_requests,
    select_probe_layers,
    solve_closed_form_delta,
    validate_delta_shapes,
    validate_request_universe,
)
from specgr_gram_adapter import (  # noqa: E402
    AuxiliaryContentDrafter,
    PathCatalog,
    VerifiedCandidate,
    drafter_cross_entropy,
    finalize_recommendations,
    guided_redraft,
    rank_drafter_items,
    score_candidate_paths_with_frozen_gram,
    validate_specgr_budget_trace,
)
from toys_b3_edit_state_smoke import (  # noqa: E402
    _capture_wo_inputs,
    _configure_determinism,
    _make_sample,
    _model_state_sha256,
    _optimize_z_residuals,
    select_covariance_transitions,
)
from toys_b2_verifier_probe_smoke import (  # noqa: E402
    _probe_layer_predictions,
    _select_probe_transitions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-config", type=Path, required=True)
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--clean-base", type=Path, required=True)
    parser.add_argument("--stage14-summary", type=Path, required=True)
    parser.add_argument("--train-sequences", type=Path, required=True)
    parser.add_argument("--held-events", type=Path, required=True)
    parser.add_argument("--pseudo-cold-items", type=Path, required=True)
    parser.add_argument("--real-cold-items", type=Path, required=True)
    parser.add_argument("--item-path-file", type=Path, required=True)
    parser.add_argument("--item-text-file", type=Path, required=True)
    parser.add_argument("--similar-items-file", type=Path, required=True)
    parser.add_argument("--item-embeddings", type=Path, required=True)
    parser.add_argument("--s2-report-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-transitions", type=int, default=4096)
    parser.add_argument("--eval-events", type=int, default=512)
    parser.add_argument("--drafter-epochs", type=int, default=2)
    parser.add_argument("--drafter-batch-size", type=int, default=128)
    parser.add_argument("--drafter-learning-rate", type=float, default=1e-3)
    parser.add_argument("--covariance-transitions", type=int, default=256)
    parser.add_argument("--covariance-long-path-minimum", type=int, default=32)
    parser.add_argument("--covariance-batch-size", type=int, default=32)
    parser.add_argument("--contexts-per-pseudo-cold", type=int, default=10)
    parser.add_argument("--requests-per-position", type=int, default=4)
    parser.add_argument("--z-steps", type=int, default=30)
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--draft-size", type=int, default=10)
    parser.add_argument("--draft-rounds", type=int, default=5)
    parser.add_argument("--verifier-threshold", type=float, default=-1.6)
    parser.add_argument("--candidate-chunk-size", type=int, default=10)
    parser.add_argument("--arms", choices=("b0,b2,b3", "b0,b2"), default="b0,b2,b3")
    parser.add_argument("--seed", type=int, default=1502)
    return parser.parse_args()


def ensure_new_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"status.json", "run.log", "gpu_telemetry.csv"}
    unexpected = [path.name for path in output_dir.iterdir() if path.name not in allowed]
    if unexpected:
        raise FileExistsError(f"Refusing existing admission artifacts: {unexpected}")


def stable_hash(seed: int, *parts: object) -> bytes:
    return hashlib.sha256(":\u241f".join([str(seed), *map(str, parts)]).encode()).digest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_embeddings(path: Path) -> tuple[list[str], torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu")
    required = {"item_ids", "embeddings", "model_name", "pooling", "l2_normalized"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Unexpected BGE embedding payload")
    item_ids = [str(item) for item in payload["item_ids"]]
    embeddings = payload["embeddings"].float().contiguous()
    if embeddings.ndim != 2 or embeddings.size(0) != len(item_ids):
        raise ValueError("Embedding IDs and matrix do not align")
    if len(item_ids) != len(set(item_ids)) or not bool(payload["l2_normalized"]):
        raise ValueError("Embeddings must be unique and L2 normalized")
    return item_ids, embeddings, {
        "model_name": payload["model_name"],
        "pooling": payload["pooling"],
        "l2_normalized": payload["l2_normalized"],
        "shape": list(embeddings.shape),
    }


def history_tensor(
    history: list[str], item_to_index: dict[str, int], max_history: int
) -> tuple[list[int], int]:
    retained = history[-max_history:]
    if not retained:
        raise ValueError("Drafter history is empty")
    values = [item_to_index[item] for item in retained]
    return values + [-1] * (max_history - len(values)), len(values)


def train_drafter(
    *,
    rows: list[dict],
    item_ids: list[str],
    embeddings: torch.Tensor,
    trainable_items: set[str],
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[AuxiliaryContentDrafter, dict]:
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    warm_indices = {item_to_index[item] for item in trainable_items}
    histories, lengths, labels = [], [], []
    for row in rows:
        if row["target"] not in trainable_items:
            raise ValueError("Forbidden item entered B2 supervision")
        encoded, length = history_tensor(row["history"], item_to_index, 20)
        histories.append(encoded)
        lengths.append(length)
        labels.append(item_to_index[row["target"]])
    history_values = torch.tensor(histories, dtype=torch.long)
    length_values = torch.tensor(lengths, dtype=torch.long)
    label_values = torch.tensor(labels, dtype=torch.long)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AuxiliaryContentDrafter(
        item_content_embeddings=embeddings,
        hidden_size=300,
        max_history=20,
        transformer_layers=2,
        attention_heads=2,
        feedforward_size=256,
        dropout=0.5,
        temperature=0.07,
    ).to(device)
    before = state_sha256(model)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.drafter_learning_rate,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    epoch_losses = []
    for epoch in range(args.drafter_epochs):
        model.train()
        permutation = torch.randperm(len(rows), generator=generator)
        total, count = 0.0, 0
        for start in range(0, len(rows), args.drafter_batch_size):
            indices = permutation[start : start + args.drafter_batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(history_values[indices].to(device), length_values[indices].to(device))
            loss = drafter_cross_entropy(
                logits, label_values[indices].to(device), warm_catalog_indices=warm_indices
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("B2 admission drafter loss is non-finite")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        epoch_losses.append(total / count)
        print(f"[s3a-b2-train] epoch={epoch + 1}/{args.drafter_epochs} loss={epoch_losses[-1]:.8f}", flush=True)
    after = state_sha256(model)
    state_root = output_dir / "b2_specgr" / "drafter"
    state_root.mkdir(parents=True, exist_ok=False)
    state_path = state_root / "drafter_state.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, state_path)
    if before == after:
        raise RuntimeError("B2 drafter state did not change")
    model.eval()
    return model, {
        "train_transitions": len(rows),
        "epochs": args.drafter_epochs,
        "epoch_losses": epoch_losses,
        "finite_loss": all(math.isfinite(value) for value in epoch_losses),
        "state_changed": before != after,
        "state_sha256": sha256_file(state_path),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def build_pseudo_contexts(
    *,
    train_sequences: list[tuple[str, list[str]]],
    pseudo_items: set[str],
    trainable_items: set[str],
    catalog: PathCatalog,
    item_ids: list[str],
    embeddings: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list, dict]:
    train_rows = {user: items for user, items in train_sequences}
    occurrences = collect_train_occurrences(
        train_rows, eligible_items=trainable_items, max_history=20
    )
    eligible = sorted(occurrences)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    eligible_matrix = embeddings[[item_to_index[item] for item in eligible]]
    pseudo_contexts: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    context_root = output_dir / "b3_genrecedit" / "edit_requests"
    context_root.mkdir(parents=True, exist_ok=False)
    context_path = context_root / "pseudo_contexts.jsonl"
    with context_path.open("x", encoding="utf-8") as handle:
        for offset, pseudo_item in enumerate(sorted(pseudo_items), 1):
            similarities = embeddings[item_to_index[pseudo_item]] @ eligible_matrix.T
            selected = deterministic_topk(
                similarities, eligible, args.contexts_per_pseudo_cold
            )
            rows = []
            for warm_index, similarity in selected:
                warm_item = eligible[warm_index]
                user, position, history = choose_occurrence(
                    occurrences[warm_item],
                    cold_item=pseudo_item,
                    warm_item=warm_item,
                    seed=args.seed,
                )
                rows.append((warm_item, tuple(history)))
                handle.write(
                    json.dumps(
                        {
                            "pseudo_cold_item": pseudo_item,
                            "source_warm_item": warm_item,
                            "train_context_items": history,
                            "similarity": similarity,
                            "source_occurrence_hash": hashlib.sha256(
                                f"{user}:{position}".encode()
                            ).hexdigest(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            pseudo_contexts[pseudo_item] = rows
            if offset % 256 == 0:
                print(f"[s3a-b3-context] items={offset}/{len(pseudo_items)}", flush=True)
    requests = build_positionwise_requests(
        cold_paths={item: catalog.paths[item] for item in pseudo_items},
        pseudo_contexts=pseudo_contexts,
    )
    by_position = validate_request_universe(
        requests=requests,
        cold_paths={item: catalog.paths[item] for item in pseudo_items},
        contexts_per_cold=args.contexts_per_pseudo_cold,
    )
    request_path = context_root / "positionwise_requests.jsonl"
    with request_path.open("x", encoding="utf-8") as handle:
        for row in requests:
            handle.write(
                json.dumps(
                    {
                        "cold_item": row.cold_item,
                        "context_items": row.context_items,
                        "prefix_tokens": row.prefix_tokens,
                        "target_token": row.target_token,
                        "position": row.position,
                        "source_warm_item": row.source_warm_item,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return requests, {
        "pseudo_cold_items": len(pseudo_items),
        "pseudo_contexts": sum(map(len, pseudo_contexts.values())),
        "positionwise_requests": len(requests),
        "requests_by_position": {str(key): value for key, value in by_position.items()},
        "context_sha256": sha256_file(context_path),
        "request_sha256": sha256_file(request_path),
    }


def build_b3_state(
    *,
    model,
    train_rows: list[dict],
    requests: list,
    item_paths: dict[str, tuple[str, ...]],
    item_text: dict[str, str],
    item_to_cfid: dict[str, int],
    tokenizer,
    collator,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    selected_layers: dict[int, int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict, dict[int, int]]:
    transitions = [
        TrainTransition(
            user_id=row["user_id"],
            history=tuple(row["history"]),
            target=row["target"],
        )
        for row in train_rows
    ]
    covariance_rows = select_covariance_transitions(
        transitions,
        path_lengths={item: len(path) for item, path in item_paths.items()},
        sample_size=args.covariance_transitions,
        long_path_minimum=args.covariance_long_path_minimum,
        seed=args.seed,
    )
    activation_bank: dict[int, list[torch.Tensor]] = {
        position: [] for position in selected_layers
    }
    for start in range(0, len(covariance_rows), args.covariance_batch_size):
        subset = covariance_rows[start : start + args.covariance_batch_size]
        samples = [
            _make_sample(
                context_items=row.history,
                target_item=row.target,
                user_id=row.user_id,
                item_to_lexical={item: "|".join(path) for item, path in item_paths.items()},
                item_text=item_text,
                item_to_cfid=item_to_cfid,
                max_history=20,
                reverse_history=True,
                history_separator=" ; ",
            )
            for row in subset
        ]
        batch = batch_to_device(collator(samples), device)
        captured = _capture_wo_inputs(model, batch, selected_layers.values())
        for position, layer in selected_layers.items():
            if position >= batch["target_ids"].size(1):
                continue
            active = batch["target_ids"][:, position].ne(-100) & batch["target_ids"][:, position].ne(
                tokenizer.eos_token_id
            )
            if bool(active.any()):
                activation_bank[position].append(captured[layer][:, position][active].cpu())
        print(f"[s3a-b3-cov] transitions={min(start + len(subset), len(covariance_rows))}/{len(covariance_rows)}", flush=True)
    covariance = {}
    counts = {}
    for position in selected_layers:
        values = torch.cat(activation_bank[position])
        accumulator = SecondMomentAccumulator(values.size(1))
        accumulator.update(values)
        covariance[position] = accumulator.moment(ridge=0.01).float()
        counts[position] = accumulator.count

    selected_requests = select_positionwise_smoke_requests(
        requests, requests_per_position=args.requests_per_position, seed=args.seed
    )
    encoded_paths = {
        item: tuple(tokenizer.convert_tokens_to_ids(token) for token in path)
        for item, path in item_paths.items()
    }
    base_parameters = dict(model.named_parameters())
    deltas: dict[int, dict[str, torch.Tensor]] = {}
    z_success = {}
    z_diagnostics = {}
    for position, rows in selected_requests.items():
        samples = [
            _make_sample(
                context_items=row.context_items,
                target_item=row.cold_item,
                user_id=f"s3a-edit-{position}-{index}",
                item_to_lexical={item: "|".join(path) for item, path in item_paths.items()},
                item_text=item_text,
                item_to_cfid=item_to_cfid,
                max_history=20,
                reverse_history=True,
                history_separator=" ; ",
            )
            for index, row in enumerate(rows)
        ]
        batch = batch_to_device(collator(samples), device)
        captured = _capture_wo_inputs(model, batch, [selected_layers[position]])
        keys = captured[selected_layers[position]][:, position].detach()
        residuals, success_mask, diagnostics = _optimize_z_residuals(
            model=model,
            batch=batch,
            requests=rows,
            encoded_paths=encoded_paths,
            layer=selected_layers[position],
            steps=args.z_steps,
            learning_rate=0.5,
            weight_decay=0.2,
            max_norm=8000,
            legal_probability_threshold=0.3,
        )
        if not bool(success_mask.any()):
            raise RuntimeError(f"No successful B3 request at position {position}")
        delta = solve_closed_form_delta(
            residual=residuals[success_mask].T,
            keys=keys[success_mask].T,
            covariance=covariance[position].to(device),
            preservation_lambda=10000,
        ).detach().cpu()
        name = edited_parameter_name(selected_layers[position])
        deltas[position] = {name: delta}
        z_success[position] = int(success_mask.sum())
        z_diagnostics[position] = diagnostics
        print(f"[s3a-b3-edit] position={position} success={z_success[position]}/{len(rows)}", flush=True)
    validate_delta_shapes(
        base_parameters=base_parameters,
        deltas_by_position=deltas,
        position_to_layer=selected_layers,
    )
    if not all(bool(torch.isfinite(next(iter(bundle.values()))).all()) for bundle in deltas.values()):
        raise RuntimeError("B3 delta is non-finite")
    delta_root = output_dir / "b3_genrecedit" / "deltaW"
    delta_root.mkdir(parents=True, exist_ok=False)
    for position, bundle in deltas.items():
        torch.save(bundle, delta_root / f"position_{position}.pt")
    covariance_root = output_dir / "b3_genrecedit" / "covariance"
    covariance_root.mkdir(parents=True, exist_ok=False)
    torch.save(covariance, covariance_root / "covariance_by_position.pt")
    atomic_json(output_dir / "b3_genrecedit" / "z_optimization.json", z_diagnostics)
    return deltas, {
        "covariance_transitions": len(covariance_rows),
        "covariance_counts": {str(key): value for key, value in counts.items()},
        "successful_z_requests": {str(key): value for key, value in z_success.items()},
        "delta_finite": True,
        "delta_nonzero": all(float(next(iter(bundle.values())).norm()) > 0 for bundle in deltas.values()),
    }, selected_layers


def probe_clean_base_layers(
    *,
    model,
    train_rows: list[dict],
    item_paths: dict[str, tuple[str, ...]],
    item_text: dict[str, str],
    item_to_cfid: dict[str, int],
    tokenizer,
    collator,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> tuple[dict[int, int], dict]:
    """Select edit layers on the actual admission base using train-only rows."""

    transitions = [
        TrainTransition(
            user_id=row["user_id"], history=tuple(row["history"]), target=row["target"]
        )
        for row in train_rows
    ]
    selected_rows = _select_probe_transitions(
        transitions,
        path_lengths={item: len(path) for item, path in item_paths.items()},
        sample_size=64,
        long_path_minimum=16,
        seed=seed,
    )
    lexical = {item: "|".join(path) for item, path in item_paths.items()}
    counts: dict[int, dict[int, list[int]]] = {}
    for start in range(0, len(selected_rows), 4):
        subset = selected_rows[start : start + 4]
        samples = [
            _make_sample(
                context_items=row.history,
                target_item=row.target,
                user_id=row.user_id,
                item_to_lexical=lexical,
                item_text=item_text,
                item_to_cfid=item_to_cfid,
                max_history=20,
                reverse_history=True,
                history_separator=" ; ",
            )
            for row in subset
        ]
        batch = batch_to_device(collator(samples), device)
        predictions = _probe_layer_predictions(model, batch)
        update = accumulate_probe_predictions(
            predictions_by_layer=predictions,
            labels=batch["target_ids"],
            eos_token_id=tokenizer.eos_token_id,
        )
        merge_probe_counts(counts, update)
        print(f"[s3a-b3-probe] batches={start // 4 + 1}/16", flush=True)
    decoder_layers = len(model.decoder.block)
    accuracy = probe_accuracy_from_counts(counts, decoder_layers=decoder_layers)
    if set(accuracy) != set(range(max(map(len, item_paths.values())))):
        raise RuntimeError("Admission train-only probe missed a lexical position")
    selected = select_probe_layers(accuracy, decoder_layers=decoder_layers)
    probe_root = output_dir / "b3_genrecedit" / "probe"
    probe_root.mkdir(parents=True, exist_ok=False)
    payload = {
        "source": "64 train-only transitions on Stage14 clean_base.pt",
        "selection_rule": "highest frozen logit-lens token accuracy; shallowest tie-break",
        "validation_or_held_used": False,
        "transitions": len(selected_rows),
        "counts": {str(p): {str(l): v for l, v in layers.items()} for p, layers in counts.items()},
        "accuracy": {str(p): {str(l): v for l, v in layers.items()} for p, layers in accuracy.items()},
        "selected_layer": {str(position): layer for position, layer in selected.items()},
    }
    atomic_json(probe_root / "layer_probe.json", payload)
    return selected, payload


def encoded_catalog(tokenizer, item_paths: dict[str, tuple[str, ...]]) -> dict[str, tuple[int, ...]]:
    result = {}
    for item, path in item_paths.items():
        values = tuple(tokenizer.convert_tokens_to_ids(token) for token in path)
        if len(values) != len(path) or tokenizer.unk_token_id in values:
            raise ValueError(f"Lexical token mapping failed: {item}")
        result[item] = values
    if len(set(result.values())) != len(result):
        raise ValueError("Tokenized catalog path collision")
    return result


def legal_children(paths: dict[str, tuple[int, ...]], eos: int) -> dict[tuple[int, ...], list[int]]:
    children: dict[tuple[int, ...], set[int]] = {}
    for path in paths.values():
        for depth, token in enumerate((*path, eos)):
            children.setdefault(path[:depth], set()).add(token)
    return {prefix: sorted(values) for prefix, values in children.items()}


def verifier_score_lengths(catalog: PathCatalog, minimum_cold_prefix: int = 2) -> dict[str, int]:
    """Compute target-aware lengths once; avoid rebuilding warm prefixes per event."""

    warm_prefixes = {
        path[:depth]
        for item, path in catalog.paths.items()
        if item in catalog.warm_items
        for depth in range(1, len(path) + 1)
    }
    result = {}
    for item, path in catalog.paths.items():
        if item in catalog.warm_items:
            result[item] = len(path)
            continue
        if len(path) < minimum_cold_prefix:
            raise ValueError("Cold path is shorter than the verifier prefix floor")
        longest = max(
            (depth for depth in range(1, len(path) + 1) if path[:depth] in warm_prefixes),
            default=0,
        )
        result[item] = min(len(path), max(minimum_cold_prefix, longest))
    return result


def generate_beam(
    *, model, batch: dict, token_paths: dict[str, tuple[int, ...]], tokenizer, beam_size: int
) -> list[tuple[str, float]]:
    reverse = {path: item for item, path in token_paths.items()}
    children = legal_children(token_paths, tokenizer.eos_token_id)

    def allowed(_batch_id: int, input_ids: torch.Tensor):
        prefix = tuple(int(value) for value in input_ids.tolist()[1:])
        return children.get(prefix, [])

    generated = model.generate(
        input_ids=batch["item_text_ids"],
        attention_mask=batch["item_text_masks"],
        max_length=max(map(len, token_paths.values())) + 2,
        num_beams=beam_size,
        num_return_sequences=beam_size,
        prefix_allowed_tokens_fn=allowed,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=True,
    )
    outputs = []
    for sequence, score in zip(generated.sequences, generated.sequences_scores):
        path = normalize_generated(sequence, tokenizer.eos_token_id)
        if path not in reverse:
            raise RuntimeError("Beam output does not map to exactly one catalog item")
        outputs.append((reverse[path], float(score)))
    if len(outputs) != beam_size or len({item for item, _score in outputs}) != beam_size:
        raise RuntimeError("Beam output violates exact budget or uniqueness")
    return outputs


def b2_rank(
    *,
    model,
    drafter,
    batch: dict,
    history: list[str],
    item_ids: list[str],
    item_to_index: dict[str, int],
    token_paths: dict[str, tuple[int, ...]],
    score_lengths: dict[str, int],
    catalog: PathCatalog,
    beam_fallback: list[tuple[str, float]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[str], dict]:
    row, length = history_tensor(history, item_to_index, 20)
    with torch.inference_mode():
        scores = drafter(
            torch.tensor([row], dtype=torch.long, device=device),
            torch.tensor([length], dtype=torch.long, device=device),
        )[0]
    ranking = rank_drafter_items(scores, item_ids, exclude_items=set(history))
    drafted_rounds = []
    verified: list[VerifiedCandidate] = []
    drafted: set[str] = set()
    guide_prefixes: list[tuple[str, ...]] = []
    verifier_forwards = 0
    for round_index in range(args.draft_rounds):
        prefix_depth = 0 if round_index == 0 else min(round_index, catalog.max_depth - 1)
        round_items = guided_redraft(
            ranking,
            catalog=catalog,
            verifier_prefixes=guide_prefixes,
            prefix_depth=prefix_depth,
            already_drafted=drafted,
            draft_size=args.draft_size,
        )
        if len(round_items) < args.draft_size:
            # Preserve the exact candidate budget while recording how many were
            # obtained through prefix guidance.
            round_items.extend(
                item for item in ranking if item not in drafted and item not in round_items
            )
            round_items = round_items[: args.draft_size]
        drafted_rounds.append(round_items)
        drafted.update(round_items)
        hook = score_candidate_paths_with_frozen_gram(
            model=model,
            batch=batch,
            candidate_token_ids=[[token_paths[item] for item in round_items]],
            score_lengths=[[score_lengths[item] for item in round_items]],
            candidate_chunk_size=args.candidate_chunk_size,
        )
        round_scores = [float(value) for value in hook["scores"][0].detach().cpu()]
        verifier_forwards += int(hook["verifier_forward_candidates"])
        verified.extend(
            VerifiedCandidate(item, score, score >= args.verifier_threshold)
            for item, score in zip(round_items, round_scores)
        )
        if round_index + 1 < args.draft_rounds:
            next_depth = min(round_index + 1, catalog.max_depth - 1)
            ranked_round = sorted(
                zip(round_items, round_scores), key=lambda row: (-row[1], row[0])
            )
            guide_prefixes = []
            for item, _score in ranked_round:
                prefix = catalog.paths[item][:next_depth]
                if len(prefix) == next_depth and prefix not in guide_prefixes:
                    guide_prefixes.append(prefix)
                if len(guide_prefixes) == 3:
                    break
    budget = validate_specgr_budget_trace(
        drafted_by_round=drafted_rounds,
        draft_size=args.draft_size,
        max_path_depth=catalog.max_depth,
        verifier_forward_candidates=verifier_forwards,
    )
    final = finalize_recommendations(
        verified=verified, beam_fallback=beam_fallback, catalog=catalog, k=args.beam_size
    )
    return [item for item, _score, _source in final], {
        **budget,
        "accepted_drafts": sum(row.accepted for row in verified),
        "finite_scores": all(math.isfinite(row.score) for row in verified),
    }


def rank_metrics(ranking: list[str], target: str) -> dict:
    rank = ranking.index(target) + 1 if target in ranking else None
    return {
        "rank": rank,
        "hit50": int(rank is not None and rank <= 50),
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def admission_verdict(run_b3: bool, checks: dict[str, bool]) -> str:
    """Return the method-specific verdict from positively phrased checks only."""
    method = "B2_B3" if run_b3 else "B2"
    prefix = "PASS" if checks and all(checks.values()) else "FAIL"
    return f"{prefix}_S15_3A_{method}_ITEM_DISJOINT_ADMISSION"


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.resolve()
    ensure_new_outputs(output_dir)
    if args.beam_size != 50 or args.draft_size * args.draft_rounds != 50:
        raise ValueError("Admission freezes beam/candidate budget at exactly 50")
    run_b3 = args.arms == "b0,b2,b3"
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "historical_config": args.historical_config,
            "backbone_path": args.backbone_path,
            "clean_base": args.clean_base,
            "stage14_summary": args.stage14_summary,
            "train_sequences": args.train_sequences,
            "held_events_deferred": args.held_events,
            "pseudo_cold_items": args.pseudo_cold_items,
            "real_cold_items": args.real_cold_items,
            "item_path_file": args.item_path_file,
            "item_text_file": args.item_text_file,
            "similar_items_file_gram_only": args.similar_items_file,
            "item_embeddings": args.item_embeddings,
            "s2_b3_summary": args.s2_report_summary,
        }.items()
    }
    for name, path in paths.items():
        if name == "backbone_path":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file() or path.is_symlink():
            raise ValueError(f"Admission input must be a regular file: {name}={path}")
    if "held_ground_truth" not in str(paths["held_events_deferred"]):
        raise ValueError("Held event path must remain visibly isolated")
    stage14 = json.loads(paths["stage14_summary"].read_text())
    if stage14.get("status") != "completed" or stage14.get("test_opened"):
        raise ValueError("Stage14 item-disjoint source is not admissible")
    s2_b3 = json.loads(paths["s2_b3_summary"].read_text())
    if s2_b3.get("verdict") != "PASS_B3_TRAIN_ONLY_EDIT_STATE_SMOKE":
        raise ValueError("S15-2 B3 state Gate is not PASS")

    numerical_mode = _configure_determinism()
    device = torch.device(args.device)
    historical = json.loads(paths["historical_config"].read_text())
    train_sequences = read_train_sequences(paths["train_sequences"])
    pseudo = read_set(paths["pseudo_cold_items"])
    real_cold = read_set(paths["real_cold_items"])
    forbidden = pseudo | real_cold
    item_paths = load_paths(paths["item_path_file"])
    item_text = read_key_value(paths["item_text_file"])
    if set(item_paths) != set(item_text):
        raise ValueError("Path and metadata catalogs differ")
    trainable_items = set(item_paths) - forbidden
    if any(item in forbidden for _user, items in train_sequences for item in items):
        raise ValueError("Filtered train sequences contain a forbidden item")
    item_inputs, filtered_input_audit = build_filtered_item_inputs(
        item_paths, item_text, paths["similar_items_file_gram_only"], forbidden, 5
    )
    item_ids, embeddings, embedding_meta = load_embeddings(paths["item_embeddings"])
    if set(item_ids) != set(item_paths):
        raise ValueError("Embedding and lexical catalogs differ")
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    item_to_cfid = {item: index + 1 for index, item in enumerate(sorted(item_paths))}
    catalog = PathCatalog(
        paths=item_paths,
        warm_items=frozenset(trainable_items),
        cold_items=frozenset(forbidden),
    )

    selected_train = clean_transitions(
        train_sequences, forbidden, 20, args.seed, args.train_transitions
    )
    if len(selected_train) != args.train_transitions:
        raise RuntimeError("Could not select the frozen B2 training budget")
    drafter, b2_state = train_drafter(
        rows=selected_train,
        item_ids=item_ids,
        embeddings=embeddings,
        trainable_items=trainable_items,
        device=device,
        args=args,
        output_dir=output_dir,
    )
    requests = []
    request_state = None
    if run_b3:
        requests, request_state = build_pseudo_contexts(
            train_sequences=train_sequences,
            pseudo_items=pseudo,
            trainable_items=trainable_items,
            catalog=catalog,
            item_ids=item_ids,
            embeddings=embeddings,
            args=args,
            output_dir=output_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(paths["backbone_path"]), local_files_only=True)
    collator = CollatorGRAM(tokenizer, args=collator_args(historical), mode="train")
    torch.manual_seed(1401)
    torch.cuda.manual_seed_all(1401)
    model = configure_fresh_model(historical, paths["backbone_path"], device, 1401)
    clean_state = torch.load(paths["clean_base"], map_location="cpu")
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base_hash_before = _model_state_sha256(model)
    position_to_layer = None
    probe_state = None
    deltas = None
    b3_state = None
    if run_b3:
        position_to_layer, probe_state = probe_clean_base_layers(
            model=model,
            train_rows=selected_train,
            item_paths=item_paths,
            item_text=item_text,
            item_to_cfid=item_to_cfid,
            tokenizer=tokenizer,
            collator=collator,
            device=device,
            seed=args.seed,
            output_dir=output_dir,
        )
        deltas, b3_state, position_to_layer = build_b3_state(
            model=model,
            train_rows=selected_train,
            requests=requests,
            item_paths=item_paths,
            item_text=item_text,
            item_to_cfid=item_to_cfid,
            tokenizer=tokenizer,
            collator=collator,
            device=device,
            args=args,
            output_dir=output_dir,
            selected_layers=position_to_layer,
        )
    base_hash_after_state = _model_state_sha256(model)
    if base_hash_before != base_hash_after_state:
        raise RuntimeError("B3 state construction mutated the clean base")

    # Held events are opened only after all B2/B3 state has been frozen.
    held = read_held_events(paths["held_events_deferred"])
    if any(row["target_item"] not in pseudo for row in held):
        raise ValueError("Held target is outside the frozen pseudo-cold universe")
    if any(set(row["visible_history"]) & forbidden for row in held):
        raise ValueError("Held visible history contains a forbidden item")
    held.sort(
        key=lambda row: deterministic_rank(
            1401, row["user_id"], row["train_prefix_position"], row["target_item"]
        )
    )
    held = held[: args.eval_events]
    if len(held) != args.eval_events:
        raise RuntimeError("Insufficient held events for admission")

    token_paths = encoded_catalog(tokenizer, item_paths)
    score_lengths = verifier_score_lengths(catalog)
    predictions = []
    totals = {
        "b0_model_forward_users": 0,
        "b2_verifier_forward_candidates": 0,
        "b2_accepted_drafts": 0,
        "b3_model_forward_users": 0,
    }
    b3_trace = None
    for index, event in enumerate(held, 1):
        sample = make_model_sample(
            {
                "user_id": event["user_id"],
                "history": event["visible_history"],
                "target": event["target_item"],
            },
            item_inputs,
            item_paths,
            item_to_cfid,
        )
        batch = batch_to_device(collator([sample]), device)
        with torch.inference_mode():
            b0_scored = generate_beam(
                model=model,
                batch=batch,
                token_paths=token_paths,
                tokenizer=tokenizer,
                beam_size=args.beam_size,
            )
        b0 = [item for item, _score in b0_scored]
        totals["b0_model_forward_users"] += 1
        with torch.inference_mode():
            b2, budget = b2_rank(
                model=model,
                drafter=drafter,
                batch=batch,
                history=event["visible_history"],
                item_ids=item_ids,
                item_to_index=item_to_index,
                token_paths=token_paths,
                score_lengths=score_lengths,
                catalog=catalog,
                beam_fallback=b0_scored,
                args=args,
                device=device,
            )
        totals["b2_verifier_forward_candidates"] += budget["verifier_forward_candidates"]
        totals["b2_accepted_drafts"] += budget["accepted_drafts"]
        b3 = None
        if run_b3:
            with torch.inference_mode(), OneOneGenerationDeltaContext(
                model=model,
                deltas_by_position=deltas,
                position_to_layer=position_to_layer,
                encoded_catalog_paths=token_paths.values(),
                decoder_start_token_id=model.config.decoder_start_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ) as trace:
                b3_scored = generate_beam(
                    model=model,
                    batch=batch,
                    token_paths=token_paths,
                    tokenizer=tokenizer,
                    beam_size=args.beam_size,
                )
            b3 = [item for item, _score in b3_scored]
            totals["b3_model_forward_users"] += 1
            if b3_trace is None:
                b3_trace = {position: 0 for position in position_to_layer}
            for position, count in trace.applied_rows_by_position.items():
                b3_trace[position] += count
        for ranking in ((b0, b2, b3) if run_b3 else (b0, b2)):
            if len(ranking) != 50 or len(set(ranking)) != 50 or not set(ranking).issubset(item_paths):
                raise RuntimeError("Admission ranking violates strict item contract")
        predictions.append(
            {
                "event_index": index,
                "user_id": event["user_id"],
                "target_item": event["target_item"],
                "b0": rank_metrics(b0, event["target_item"]),
                "b2": rank_metrics(b2, event["target_item"]),
                **({"b3": rank_metrics(b3, event["target_item"])} if run_b3 else {}),
                "b2_differs_from_b0": b2 != b0,
                **({"b3_differs_from_b0": b3 != b0} if run_b3 else {}),
                "b2_budget": budget,
            }
        )
        if index % 16 == 0:
            print(f"[s3a-eval] events={index}/{len(held)}", flush=True)

    base_hash_after_eval = _model_state_sha256(model)
    if base_hash_before != base_hash_after_eval:
        raise RuntimeError("Admission evaluation mutated the clean base")
    prediction_path = output_dir / "predictions_held_admission.jsonl"
    with prediction_path.open("x", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    b2_diff = sum(row["b2_differs_from_b0"] for row in predictions)
    b3_diff = sum(row["b3_differs_from_b0"] for row in predictions) if run_b3 else None
    if run_b3 and (
        not b3_trace or any(b3_trace[position] < 1 for position in position_to_layer)
    ):
        raise RuntimeError("One-One edited beam did not exercise every lexical position")
    admission_checks = {
        "b0_complete_beam_path": True,
        "b2_complete_draft_verify_redraft_path": True,
        "all_rankings_unique_known_top50": True,
        "b2_finite_loss_and_scores": b2_state["finite_loss"],
        "b2_state_changed": b2_state["state_changed"],
        "base_hash_unchanged": base_hash_before == base_hash_after_eval,
        "held_ground_truth_opened_after_state_only": True,
        "held_ground_truth_not_used_for_training_or_state_selection": True,
        "test_not_opened": True,
    }
    if run_b3:
        admission_checks.update(
            {
                "b3_complete_one_one_edited_beam_path": True,
                "b3_delta_finite": b3_state["delta_finite"],
                "b3_delta_nonzero": b3_state["delta_nonzero"],
                "b3_every_position_exercised": True,
            }
        )
    verdict = admission_verdict(run_b3, admission_checks)
    arm_metrics = {}
    for arm in (("b0", "b2", "b3") if run_b3 else ("b0", "b2")):
        arm_metrics[arm] = {
            "events": len(predictions),
            "hit50": sum(row[arm]["hit50"] for row in predictions),
            "mrr": sum(row[arm]["mrr"] for row in predictions) / len(predictions),
        }
    config = {
        "experiment_id": "GRAM_STAGE15_S3A_TOYS_ITEM_DISJOINT_ADMISSION",
        "scope": "engineering admission only; metrics are non-promotional",
        "arms": args.arms,
        "train_transitions": args.train_transitions,
        "eval_events": args.eval_events,
        "beam_size": args.beam_size,
        "b2_candidate_budget": args.draft_size * args.draft_rounds,
        "b2_draft_size": args.draft_size,
        "b2_draft_rounds": args.draft_rounds,
        "b2_verifier_threshold": args.verifier_threshold,
        "b3_covariance_transitions": args.covariance_transitions,
        "b3_requests_per_position": args.requests_per_position,
        "seed": args.seed,
        "device": args.device,
        "automatic_retry": False,
        "test_read": False,
        "numerical_mode": numerical_mode,
    }
    summary = {
        **config,
        "status": "completed",
        "verdict": verdict,
        "admission_checks": admission_checks,
        "b2_state": b2_state,
        **({"b3_request_state": request_state} if run_b3 else {}),
        **({"b3_probe_selected_layer": probe_state["selected_layer"]} if run_b3 else {}),
        **({"b3_state": b3_state} if run_b3 else {}),
        **(
            {
                "b3_generation_applied_rows_by_position": {
                    str(key): value for key, value in b3_trace.items()
                }
            }
            if run_b3
            else {}
        ),
        "arm_metrics_non_promotional": arm_metrics,
        "b2_rankings_different_from_b0": b2_diff,
        **({"b3_rankings_different_from_b0": b3_diff} if run_b3 else {}),
        "forward_accounting": totals,
        "clean_base_hash_before": base_hash_before,
        "clean_base_hash_after": base_hash_after_eval,
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    input_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "backbone_path"
    }
    input_hashes["backbone_config"] = sha256_file(paths["backbone_path"] / "config.json")
    atomic_json(output_dir / "config.json", config)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "input_file_sha256.json", input_hashes)
    atomic_json(
        output_dir / "data_provenance.json",
        {
            "base": "Stage14 item-disjoint clean_base.pt from fresh local t5-small",
            "historical_v0_checkpoint_used": False,
            "pseudo_or_real_cold_interactions_used_for_base_or_state_training": False,
            "held_ground_truth_opened_after_b2_b3_state": True,
            "held_ground_truth_used_for_training_or_selection": False,
            "test_opened": False,
            "gram_input_similar_items_filtered_forbidden": filtered_input_audit,
            "adapter_similarity_source": embedding_meta,
        },
    )
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened_before_state_freeze": [
                str(path.relative_to(REPO_ROOT))
                for name, path in paths.items()
                if name not in {"held_events_deferred", "backbone_path"}
            ],
            "opened_after_state_freeze": [str(paths["held_events_deferred"].relative_to(REPO_ROOT))],
            "held_ground_truth_used_for_training": False,
            "test_predictions_opened": False,
            "test_metrics_opened": False,
        },
    )
    atomic_json(
        output_dir / "resource_summary.json",
        {
            "runtime_seconds": summary["runtime_seconds"],
            "peak_cuda_allocated_mib": summary["peak_cuda_allocated_mib"],
            "b2_trainable_parameters": b2_state["trainable_parameters"],
            "b3_delta_positions": len(deltas) if run_b3 else 0,
            "eval_events": len(predictions),
            **totals,
        },
    )
    print(json.dumps({"status": "completed", "verdict": verdict, "summary": str(output_dir / 'summary.json')}))


if __name__ == "__main__":
    main()
