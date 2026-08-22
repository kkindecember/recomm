"""Toys GenRecEdit-GRAM covariance, request, deltaW, and trigger state smoke.

The job uses only audited train histories plus the frozen catalog and BGE-derived
pseudo contexts.  It expands every frozen cold item into position-wise requests,
extracts train-only second moments at probe-selected decoder FFN layers, optimizes
a small deterministic z-residual sample per lexical position, solves full-shape
deltaW bundles, and verifies One-One routing without mutating the base checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE14_PROTOCOL = REPO_ROOT / "experiment" / "phase14" / "protocol"
if str(PHASE14_PROTOCOL) not in sys.path:
    sys.path.insert(0, str(PHASE14_PROTOCOL))

from item_level_eval import atomic_json, load_item_paths  # noqa: E402
from oracle_prefix_probe import (  # noqa: E402
    CollatorGRAM,
    batch_to_device,
    configure_model,
    make_dataset_args,
)

from common_adapter import (  # noqa: E402
    TrainTransition,
    iter_train_transitions,
    read_projected_sequences,
    sha256_file,
)
from genrecedit_gram_adapter import (  # noqa: E402
    OneOneDeltaRouter,
    PositionWiseRequest,
    SecondMomentAccumulator,
    build_positionwise_requests,
    edited_parameter_name,
    legal_next_token_ids,
    legal_target_state,
    select_positionwise_smoke_requests,
    solve_closed_form_delta,
    validate_delta_shapes,
    validate_position_layer_selection,
    validate_request_universe,
)
from specgr_gram_adapter import PathCatalog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-sequences", type=Path, required=True)
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--historical-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--item-path-file", type=Path, required=True)
    parser.add_argument("--contract-state", type=Path, required=True)
    parser.add_argument("--probe-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--covariance-transitions", type=int, default=256)
    parser.add_argument("--covariance-long-path-minimum", type=int, default=32)
    parser.add_argument("--covariance-batch-size", type=int, default=32)
    parser.add_argument("--requests-per-position", type=int, default=4)
    parser.add_argument("--z-steps", type=int, default=30)
    parser.add_argument("--z-learning-rate", type=float, default=0.5)
    parser.add_argument("--z-weight-decay", type=float, default=0.2)
    parser.add_argument("--z-max-norm", type=float, default=8000.0)
    parser.add_argument("--legal-probability-threshold", type=float, default=0.3)
    parser.add_argument("--covariance-ridge", type=float, default=0.01)
    parser.add_argument("--preservation-lambda", type=float, default=10000.0)
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


def _read_item_text(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0] in result:
            raise ValueError(f"Invalid item metadata at line {line_number}")
        result[parts[0]] = parts[1]
    if not result:
        raise ValueError("Empty item metadata")
    return result


def _stable_rank(seed: int, *parts: object) -> bytes:
    return hashlib.sha256(":".join([str(seed), *map(str, parts)]).encode("utf-8")).digest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def select_covariance_transitions(
    transitions: Sequence[TrainTransition],
    *,
    path_lengths: Mapping[str, int],
    sample_size: int,
    long_path_minimum: int,
    seed: int,
) -> list[TrainTransition]:
    """Fixed train-only sample with explicit longest-path position coverage."""

    eligible = [row for row in transitions if row.history]
    if sample_size < long_path_minimum or sample_size > len(eligible):
        raise ValueError("Invalid covariance transition sample contract")
    max_depth = max(path_lengths.values())
    ranked = sorted(
        eligible,
        key=lambda row: (
            _stable_rank(seed, "b3-cov", row.user_id, len(row.history), row.target),
            row.user_id,
            len(row.history),
        ),
    )
    longest = [row for row in ranked if path_lengths[row.target] == max_depth]
    if len(longest) < long_path_minimum:
        raise ValueError("Insufficient longest-path train transitions for covariance")
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
        raise RuntimeError("Could not fill covariance transition sample")
    return selected


def _make_sample(
    *,
    context_items: Sequence[str],
    target_item: str,
    user_id: str,
    item_to_lexical: Mapping[str, str],
    item_text: Mapping[str, str],
    item_to_cfid: Mapping[str, int],
    max_history: int,
    reverse_history: bool,
    history_separator: str,
) -> dict:
    retained = tuple(context_items[-max_history:])
    if not retained:
        raise ValueError("GenRecEdit context must contain at least one train item")
    if target_item not in item_to_lexical or any(item not in item_to_lexical for item in retained):
        raise ValueError("GenRecEdit sample contains an unknown catalog item")
    ordered = tuple(reversed(retained)) if reverse_history else retained
    history_lexical = history_separator.join(item_to_lexical[item] for item in reversed(retained))
    return {
        "input": [f"What would user purchase after {history_lexical} ?"]
        + [item_text[item] for item in ordered],
        "output": item_to_lexical[target_item],
        "user_id": user_id,
        "history_item_ids": [item_to_cfid[item] for item in ordered],
        "target_item_id": item_to_cfid[target_item],
    }


def _decoder_wo(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    return model.decoder.block[layer].layer[2].DenseReluDense.wo


def _capture_wo_inputs(
    model: torch.nn.Module,
    batch: dict,
    layers: Iterable[int],
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in sorted(set(int(value) for value in layers)):
        def capture(_module, inputs, layer_index=layer):
            captured[layer_index] = inputs[0].detach()

        handles.append(_decoder_wo(model, layer).register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            _ = model(
                input_ids=batch["item_text_ids"],
                attention_mask=batch["item_text_masks"],
                labels=batch["target_ids"],
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(int(value) for value in layers):
        raise RuntimeError("Did not capture every requested decoder FFN input")
    return captured


def _decoder_inputs(
    requests: Sequence[PositionWiseRequest],
    encoded_paths: Mapping[str, Sequence[int]],
    *,
    decoder_start_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for request in requests:
        path = tuple(int(value) for value in encoded_paths[request.cold_item])
        rows.append((int(decoder_start_token_id),) + path[: request.position])
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError("Position batch decoder prefixes do not align")
    return torch.tensor(rows, dtype=torch.long, device=device)


def _optimize_z_residuals(
    *,
    model: torch.nn.Module,
    batch: dict,
    requests: Sequence[PositionWiseRequest],
    encoded_paths: Mapping[str, Sequence[int]],
    layer: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    max_norm: float,
    legal_probability_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    if not requests or len({row.position for row in requests}) != 1:
        raise ValueError("z optimization requires one non-empty lexical-position batch")
    device = batch["item_text_ids"].device
    position = requests[0].position
    target_ids = torch.tensor(
        [int(encoded_paths[row.cold_item][position]) for row in requests],
        dtype=torch.long,
        device=device,
    )
    decoder_ids = _decoder_inputs(
        requests,
        encoded_paths,
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        device=device,
    )
    with torch.inference_mode():
        baseline_logits = model(
            input_ids=batch["item_text_ids"],
            attention_mask=batch["item_text_masks"],
            decoder_input_ids=decoder_ids,
            use_cache=False,
        ).logits[:, -1, :].detach()

    residual = torch.zeros(
        len(requests), model.config.d_model, device=device, requires_grad=True
    )
    initial_output: list[torch.Tensor | None] = [None]

    def inject(_module, _inputs, output):
        if initial_output[0] is None:
            initial_output[0] = output[:, -1, :].detach()
        modified = output.clone()
        modified[:, -1, :] = modified[:, -1, :] + residual
        return modified

    handle = _decoder_wo(model, layer).register_forward_hook(inject)
    optimizer = torch.optim.Adam([residual], lr=learning_rate)
    try:
        for _ in range(steps):
            logits = model(
                input_ids=batch["item_text_ids"],
                attention_mask=batch["item_text_masks"],
                decoder_input_ids=decoder_ids,
                use_cache=False,
            ).logits[:, -1, :]
            loss = F.cross_entropy(logits, target_ids)
            if initial_output[0] is not None and weight_decay:
                reference_norm = initial_output[0].norm(dim=1).clamp_min(1e-8)
                loss = loss + weight_decay * (residual.norm(dim=1) / reference_norm).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                norms = residual.norm(dim=1)
                scale = (max_norm / norms.clamp_min(1e-12)).clamp_max(1.0)
                residual.mul_(scale[:, None])
        with torch.inference_mode():
            edited_logits = model(
                input_ids=batch["item_text_ids"],
                attention_mask=batch["item_text_masks"],
                decoder_input_ids=decoder_ids,
                use_cache=False,
            ).logits[:, -1, :].detach()
    finally:
        handle.remove()

    diagnostics = []
    successes = []
    for index, request in enumerate(requests):
        path = tuple(int(value) for value in encoded_paths[request.cold_item])
        legal = legal_next_token_ids(encoded_paths, path[:position])
        baseline_best, baseline_probability = legal_target_state(
            baseline_logits[index], target_token_id=int(target_ids[index]), legal_token_ids=legal
        )
        edited_best, edited_probability = legal_target_state(
            edited_logits[index], target_token_id=int(target_ids[index]), legal_token_ids=legal
        )
        success = bool(
            edited_best
            and edited_probability >= legal_probability_threshold
            and edited_probability > baseline_probability
        )
        successes.append(success)
        diagnostics.append(
            {
                "cold_item": request.cold_item,
                "source_warm_item": request.source_warm_item,
                "position": position,
                "layer": layer,
                "target_token_id": int(target_ids[index]),
                "legal_children": len(legal),
                "baseline_legal_argmax": baseline_best,
                "baseline_legal_probability": baseline_probability,
                "edited_legal_argmax": edited_best,
                "edited_legal_probability": edited_probability,
                "success": success,
                "residual_norm": float(residual[index].detach().norm().cpu()),
            }
        )
    return residual.detach(), torch.tensor(successes, dtype=torch.bool, device=device), diagnostics


def _parameter_logits(
    *,
    model: torch.nn.Module,
    batch: dict,
    requests: Sequence[PositionWiseRequest],
    encoded_paths: Mapping[str, Sequence[int]],
) -> torch.Tensor:
    decoder_ids = _decoder_inputs(
        requests,
        encoded_paths,
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        device=batch["item_text_ids"].device,
    )
    with torch.inference_mode():
        return model(
            input_ids=batch["item_text_ids"],
            attention_mask=batch["item_text_masks"],
            decoder_input_ids=decoder_ids,
            use_cache=False,
        ).logits[:, -1, :].detach().cpu()


def _configure_determinism() -> dict[str, object]:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Deterministic B3 smoke requires CUBLAS_WORKSPACE_CONFIG")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": workspace,
    }


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    output_dir = args.output_dir.resolve()
    _ensure_new_outputs(output_dir)
    source_dataset = args.source_dataset_dir.resolve()
    contract_state = args.contract_state.resolve()
    probe_state = args.probe_state.resolve()
    paths = {
        "projected_sequences": args.projected_sequences.resolve(),
        "historical_config": args.historical_config.resolve(),
        "checkpoint": args.checkpoint.resolve(),
        "item_paths": args.item_path_file.resolve(),
        "item_metadata": source_dataset / "item_plain_text.txt",
        "warm_items": source_dataset / "cold_split_meta" / "warm_items.txt",
        "cold_items": source_dataset / "cold_split_meta" / "cold_items.txt",
        "contract_summary": contract_state / "summary.json",
        "pseudo_contexts": contract_state
        / "genrecedit_gram"
        / "edit_requests"
        / "pseudo_contexts.jsonl",
        "probe_summary": probe_state / "summary.json",
        "layer_probe": probe_state / "genrecedit_gram" / "probe" / "layer_probe.json",
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Input {name} must be a regular non-symlink file: {path}")
    if paths["projected_sequences"].name != "user_sequence_train_validation.txt":
        raise ValueError("B3 must use the audited projected sequence")
    if min(
        args.covariance_transitions,
        args.covariance_long_path_minimum,
        args.covariance_batch_size,
        args.requests_per_position,
        args.z_steps,
    ) < 1:
        raise ValueError("B3 smoke budgets must be positive")
    if (
        args.z_learning_rate <= 0
        or args.z_weight_decay < 0
        or args.z_max_norm <= 0
        or not 0 < args.legal_probability_threshold <= 1
        or args.covariance_ridge <= 0
        or args.preservation_lambda <= 0
    ):
        raise ValueError("Invalid B3 edit hyperparameter")

    contract = json.loads(paths["contract_summary"].read_text(encoding="utf-8"))
    probe_summary = json.loads(paths["probe_summary"].read_text(encoding="utf-8"))
    if contract.get("verdict") != "PASS_B2_B3_INPUT_CONTRACT":
        raise ValueError("B2/B3 input contract is not PASS")
    if probe_summary.get("verdict") != "PASS_B2_VERIFIER_GPU_HOOK_AND_B3_TRAIN_ONLY_PROBE":
        raise ValueError("Frozen B3 layer probe is not PASS")

    historical = json.loads(paths["historical_config"].read_text(encoding="utf-8"))
    item_to_lexical, decoded_to_items = load_item_paths(paths["item_paths"])
    if any(len(items) != 1 for items in decoded_to_items.values()):
        raise ValueError("B3 requires collision-free catalog paths")
    warm = _read_set(paths["warm_items"])
    cold = _read_set(paths["cold_items"])
    catalog = PathCatalog.build(item_to_lexical, warm, cold)
    item_text = _read_item_text(paths["item_metadata"])
    if set(item_text) != set(catalog.paths):
        raise ValueError("Metadata does not exactly cover the path catalog")
    item_to_cfid = {item: index + 1 for index, item in enumerate(sorted(catalog.paths))}

    selected_layers = {
        int(position): int(layer)
        for position, layer in probe_summary["selected_layer_by_position"].items()
    }
    selected_layers = validate_position_layer_selection(
        cold_paths={item: catalog.paths[item] for item in cold},
        position_to_layer=selected_layers,
        decoder_layers=6,
    )
    if selected_layers != {0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 4}:
        raise ValueError("B3 selected layers drifted from the passed train-only probe")

    tokenizer = AutoTokenizer.from_pretrained(historical["backbone"])
    encoded_paths = {
        item: tuple(tokenizer.convert_tokens_to_ids(token) for token in path)
        for item, path in catalog.paths.items()
    }
    if any(
        len(encoded_paths[item]) != len(path)
        or any(token == tokenizer.unk_token_id for token in encoded_paths[item])
        for item, path in catalog.paths.items()
    ):
        raise ValueError("Frozen lexical paths do not map one-to-one to GRAM tokens")

    pseudo_contexts: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    with paths["pseudo_contexts"].open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            row = json.loads(raw)
            cold_item = str(row["cold_item"])
            if cold_item not in cold or tuple(row["cold_path"]) != catalog.paths[cold_item]:
                raise ValueError(f"Pseudo-context cold path mismatch at line {line_number}")
            context = tuple(str(item) for item in row["train_context_items"])
            source_warm = str(row["source_warm_item"])
            if source_warm not in warm or not context or any(item not in warm for item in context):
                raise ValueError(f"Pseudo-context is not warm train-only at line {line_number}")
            pseudo_contexts.setdefault(cold_item, []).append((source_warm, context))
    requests = build_positionwise_requests(
        cold_paths={item: catalog.paths[item] for item in cold},
        pseudo_contexts=pseudo_contexts,
    )
    requests_by_position = validate_request_universe(
        requests=requests,
        cold_paths={item: catalog.paths[item] for item in cold},
        contexts_per_cold=10,
    )
    if len(requests) != int(contract["genrecedit_positionwise_request_count"]):
        raise ValueError("Position-wise requests drifted from the input contract")
    selected_requests = select_positionwise_smoke_requests(
        requests, requests_per_position=args.requests_per_position, seed=args.seed
    )

    request_root = output_dir / "genrecedit_gram" / "edit_requests"
    request_root.mkdir(parents=True, exist_ok=False)
    request_path = request_root / "positionwise_requests.jsonl"
    with request_path.open("x", encoding="utf-8") as handle:
        for request in requests:
            handle.write(
                json.dumps(
                    {
                        "cold_item": request.cold_item,
                        "context_items": request.context_items,
                        "prefix_tokens": request.prefix_tokens,
                        "target_token": request.target_token,
                        "position": request.position,
                        "source_warm_item": request.source_warm_item,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    trigger_path = output_dir / "genrecedit_gram" / "one_one_trigger.jsonl"
    with trigger_path.open("x", encoding="utf-8") as handle:
        for cold_item in sorted(cold):
            handle.write(
                json.dumps(
                    {
                        "cold_item": cold_item,
                        "active_positions": list(range(len(catalog.paths[cold_item]))),
                        "eos_active": False,
                        "padding_active": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    rows = read_projected_sequences(paths["projected_sequences"])
    transitions = list(iter_train_transitions(rows))
    covariance_transitions = select_covariance_transitions(
        transitions,
        path_lengths={item: len(path) for item, path in catalog.paths.items()},
        sample_size=args.covariance_transitions,
        long_path_minimum=args.covariance_long_path_minimum,
        seed=args.seed,
    )
    if any(row.target not in warm for row in covariance_transitions):
        raise ValueError("Cold or unknown target entered covariance estimation")

    dataset_args = make_dataset_args(historical, source_dataset)
    collator = CollatorGRAM(tokenizer, args=dataset_args, mode="train")
    device = torch.device(args.device)
    numerical_mode = _configure_determinism()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = configure_model(historical, paths["checkpoint"], device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_hash_before = _model_state_sha256(model)
    checkpoint_hash_before = sha256_file(paths["checkpoint"])

    activation_bank: dict[int, list[torch.Tensor]] = {position: [] for position in selected_layers}
    for start in range(0, len(covariance_transitions), args.covariance_batch_size):
        subset = covariance_transitions[start : start + args.covariance_batch_size]
        samples = [
            _make_sample(
                context_items=row.history,
                target_item=row.target,
                user_id=row.user_id,
                item_to_lexical=item_to_lexical,
                item_text=item_text,
                item_to_cfid=item_to_cfid,
                max_history=int(historical["max_his"]),
                reverse_history=bool(historical["reverse_history"]),
                history_separator=str(historical["his_sep"]),
            )
            for row in subset
        ]
        batch = batch_to_device(collator(samples), device)
        captured = _capture_wo_inputs(model, batch, selected_layers.values())
        for position, layer in selected_layers.items():
            if position >= batch["target_ids"].size(1):
                continue
            active = batch["target_ids"][:, position].ne(-100) & batch["target_ids"][:, position].ne(
                int(tokenizer.eos_token_id)
            )
            if bool(active.any()):
                activation_bank[position].append(captured[layer][:, position, :][active].detach().cpu())
        print(
            f"[b3-cov] transitions={min(start + len(subset), len(covariance_transitions))}/{len(covariance_transitions)}",
            flush=True,
        )

    covariance_by_position: dict[int, torch.Tensor] = {}
    covariance_counts: dict[int, int] = {}
    for position in sorted(selected_layers):
        if not activation_bank[position]:
            raise RuntimeError(f"No covariance activations for lexical position {position}")
        activations = torch.cat(activation_bank[position], dim=0)
        accumulator = SecondMomentAccumulator(activations.size(1))
        accumulator.update(activations)
        covariance = accumulator.moment(ridge=args.covariance_ridge).float()
        if not bool(torch.isfinite(covariance).all()):
            raise RuntimeError("Covariance state is non-finite")
        covariance_by_position[position] = covariance
        covariance_counts[position] = accumulator.count

    covariance_root = output_dir / "genrecedit_gram" / "covariance"
    covariance_root.mkdir(parents=True, exist_ok=False)
    covariance_path = covariance_root / "covariance_by_position.pt"
    torch.save(covariance_by_position, covariance_path)

    delta_root = output_dir / "genrecedit_gram" / "deltaW"
    delta_root.mkdir(parents=True, exist_ok=False)
    base_parameters = dict(model.named_parameters())
    deltas_by_position: dict[int, dict[str, torch.Tensor]] = {}
    z_diagnostics: dict[int, list[dict]] = {}
    edit_batches: dict[int, dict] = {}
    successful_requests: dict[int, list[PositionWiseRequest]] = {}
    for position, position_requests in selected_requests.items():
        samples = [
            _make_sample(
                context_items=row.context_items,
                target_item=row.cold_item,
                user_id=f"b3-edit-{position}-{index}",
                item_to_lexical=item_to_lexical,
                item_text=item_text,
                item_to_cfid=item_to_cfid,
                max_history=int(historical["max_his"]),
                reverse_history=bool(historical["reverse_history"]),
                history_separator=str(historical["his_sep"]),
            )
            for index, row in enumerate(position_requests)
        ]
        batch = batch_to_device(collator(samples), device)
        target_ids = batch["target_ids"][:, position]
        expected_ids = torch.tensor(
            [encoded_paths[row.cold_item][position] for row in position_requests], device=device
        )
        if not torch.equal(target_ids, expected_ids):
            raise RuntimeError("Edit request token does not match the GRAM collator")
        captured = _capture_wo_inputs(model, batch, [selected_layers[position]])
        keys = captured[selected_layers[position]][:, position, :].detach()
        residuals, success_mask, diagnostics = _optimize_z_residuals(
            model=model,
            batch=batch,
            requests=position_requests,
            encoded_paths=encoded_paths,
            layer=selected_layers[position],
            steps=args.z_steps,
            learning_rate=args.z_learning_rate,
            weight_decay=args.z_weight_decay,
            max_norm=args.z_max_norm,
            legal_probability_threshold=args.legal_probability_threshold,
        )
        if not bool(success_mask.any()):
            raise RuntimeError(f"No successful z residual at lexical position {position}")
        successful = [row for row, keep in zip(position_requests, success_mask.tolist()) if keep]
        successful_requests[position] = successful
        z_diagnostics[position] = diagnostics
        delta = solve_closed_form_delta(
            residual=residuals[success_mask].T,
            keys=keys[success_mask].T,
            covariance=covariance_by_position[position].to(device),
            preservation_lambda=args.preservation_lambda,
        ).detach().cpu()
        parameter_name = edited_parameter_name(selected_layers[position])
        deltas_by_position[position] = {parameter_name: delta}
        torch.save(deltas_by_position[position], delta_root / f"position_{position}.pt")
        edit_batches[position] = batch
        print(
            f"[b3-edit] position={position} successful={int(success_mask.sum())}/{len(position_requests)}",
            flush=True,
        )

    validate_delta_shapes(
        base_parameters=base_parameters,
        deltas_by_position=deltas_by_position,
        position_to_layer=selected_layers,
    )
    router = OneOneDeltaRouter(
        deltas_by_position=deltas_by_position, position_to_layer=selected_layers
    )
    delta_finite = all(
        bool(torch.isfinite(delta).all())
        for bundle in deltas_by_position.values()
        for delta in bundle.values()
    )
    delta_nonzero = all(
        float(delta.norm()) > 0
        for bundle in deltas_by_position.values()
        for delta in bundle.values()
    )
    eos_padding_inactive = all(
        not router.active_bundle(position, is_eos=True)
        and not router.active_bundle(position, is_padding=True)
        for position in selected_layers
    )

    trigger_diagnostics = []
    unedited_exact = True
    for position in sorted(selected_layers):
        batch = edit_batches[position]
        position_requests = selected_requests[position]
        baseline = _parameter_logits(
            model=model, batch=batch, requests=position_requests, encoded_paths=encoded_paths
        )
        parameter_name = edited_parameter_name(selected_layers[position])
        parameter = base_parameters[parameter_name]
        original = parameter.detach().clone()
        try:
            with torch.no_grad():
                parameter.copy_(router.materialize_parameter(parameter_name, original, position))
            edited = _parameter_logits(
                model=model, batch=batch, requests=position_requests, encoded_paths=encoded_paths
            )
        finally:
            with torch.no_grad():
                parameter.copy_(original)
        restored = _parameter_logits(
            model=model, batch=batch, requests=position_requests, encoded_paths=encoded_paths
        )
        unedited_exact = unedited_exact and torch.equal(baseline, restored)
        target_ids = [encoded_paths[row.cold_item][position] for row in position_requests]
        target_changes = [
            float(edited[index, token] - baseline[index, token])
            for index, token in enumerate(target_ids)
        ]
        trigger_diagnostics.append(
            {
                "position": position,
                "layer": selected_layers[position],
                "parameter": parameter_name,
                "delta_norm": float(deltas_by_position[position][parameter_name].norm()),
                "target_logit_changes": target_changes,
                "any_logit_changed": not torch.equal(baseline, edited),
                "base_restored_exact": torch.equal(baseline, restored),
            }
        )

    model_hash_after = _model_state_sha256(model)
    checkpoint_hash_after = sha256_file(paths["checkpoint"])
    model_unchanged = model_hash_before == model_hash_after
    checkpoint_unchanged = checkpoint_hash_before == checkpoint_hash_after
    trigger_changes = all(row["any_logit_changed"] for row in trigger_diagnostics)
    all_positions_succeeded = set(successful_requests) == set(selected_layers)
    verdict = (
        "PASS_B3_TRAIN_ONLY_EDIT_STATE_SMOKE"
        if all(
            [
                len(requests) == 302400,
                set(requests_by_position) == set(selected_layers),
                all_positions_succeeded,
                delta_finite,
                delta_nonzero,
                eos_padding_inactive,
                trigger_changes,
                unedited_exact,
                model_unchanged,
                checkpoint_unchanged,
            ]
        )
        else "FAIL_B3_TRAIN_ONLY_EDIT_STATE_SMOKE"
    )

    atomic_json(
        request_root / "manifest.json",
        {
            "status": "completed",
            "cold_universe": len(cold),
            "cold_universe_covered": len(pseudo_contexts),
            "pseudo_contexts": sum(len(rows) for rows in pseudo_contexts.values()),
            "positionwise_requests": len(requests),
            "requests_by_position": {str(key): value for key, value in requests_by_position.items()},
            "requests_per_position_in_delta_smoke": args.requests_per_position,
            "validation_occurrence_used": False,
            "test_occurrence_used": False,
            "eos_edited": False,
            "padding_edited": False,
            "request_file": str(request_path.relative_to(REPO_ROOT)),
            "request_file_sha256": sha256_file(request_path),
        },
    )
    atomic_json(
        covariance_root / "manifest.json",
        {
            "status": "completed",
            "source": "fixed SHA-ranked train-only next-item transitions",
            "transitions": len(covariance_transitions),
            "longest_path_minimum": args.covariance_long_path_minimum,
            "activation_counts_by_position": {
                str(key): value for key, value in covariance_counts.items()
            },
            "selected_layer_by_position": {
                str(key): value for key, value in selected_layers.items()
            },
            "ridge": args.covariance_ridge,
            "matrix_shape": [2048, 2048],
            "state_file": str(covariance_path.relative_to(REPO_ROOT)),
            "state_sha256": sha256_file(covariance_path),
            "validation_used": False,
            "test_used": False,
        },
    )
    delta_manifest = {
        "status": "completed",
        "official_mechanism": "z residual optimization plus RK^T(lambda C + KK^T)^-1",
        "preservation_lambda": args.preservation_lambda,
        "position_to_layer": {str(key): value for key, value in selected_layers.items()},
        "successful_requests_by_position": {
            str(key): len(value) for key, value in successful_requests.items()
        },
        "bundles": {
            str(position): {
                "parameter": next(iter(bundle)),
                "shape": list(next(iter(bundle.values())).shape),
                "frobenius_norm": float(next(iter(bundle.values())).norm()),
                "file": str((delta_root / f"position_{position}.pt").relative_to(REPO_ROOT)),
                "sha256": sha256_file(delta_root / f"position_{position}.pt"),
            }
            for position, bundle in deltas_by_position.items()
        },
        "one_one_trigger": True,
        "eos_and_padding_inactive": eos_padding_inactive,
        "base_checkpoint_mutated": False,
    }
    atomic_json(delta_root / "manifest.json", delta_manifest)
    atomic_json(output_dir / "z_optimization.json", {str(k): v for k, v in z_diagnostics.items()})
    atomic_json(output_dir / "trigger_diagnostics.json", {"positions": trigger_diagnostics})

    input_hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths.values()}
    config = {
        "experiment_id": "GRAM_STAGE15_S2_TOYS_B3_EDIT_STATE_SMOKE",
        "mechanism_name": "GenRecEdit-GRAM official-mechanism port state smoke",
        "split": "train_only_for_covariance_and_requests",
        "covariance_transitions": args.covariance_transitions,
        "covariance_long_path_minimum": args.covariance_long_path_minimum,
        "covariance_batch_size": args.covariance_batch_size,
        "requests_per_position": args.requests_per_position,
        "z_steps": args.z_steps,
        "z_learning_rate": args.z_learning_rate,
        "z_weight_decay": args.z_weight_decay,
        "z_max_norm": args.z_max_norm,
        "legal_probability_threshold": args.legal_probability_threshold,
        "covariance_ridge": args.covariance_ridge,
        "preservation_lambda": args.preservation_lambda,
        "seed": args.seed,
        "device": args.device,
        "selected_layer_by_position": {str(key): value for key, value in selected_layers.items()},
        "base_model_training": False,
        "automatic_retry": False,
        "test_read": False,
        "numerical_mode": numerical_mode,
    }
    summary = {
        **config,
        "status": "completed",
        "verdict": verdict,
        "catalog_items": len(catalog.paths),
        "cold_items": len(cold),
        "pseudo_contexts": sum(len(rows) for rows in pseudo_contexts.values()),
        "positionwise_requests": len(requests),
        "requests_by_position": {str(key): value for key, value in requests_by_position.items()},
        "covariance_activation_counts": {
            str(key): value for key, value in covariance_counts.items()
        },
        "successful_z_requests_by_position": {
            str(key): len(value) for key, value in successful_requests.items()
        },
        "delta_finite": delta_finite,
        "delta_nonzero": delta_nonzero,
        "trigger_changes_every_position": trigger_changes,
        "one_one_eos_padding_inactive": eos_padding_inactive,
        "unedited_prompt_parity_exact": unedited_exact,
        "frozen_gram_model_hash_before": model_hash_before,
        "frozen_gram_model_hash_after": model_hash_after,
        "frozen_gram_model_hash_unchanged": model_unchanged,
        "gram_checkpoint_sha256_before": checkpoint_hash_before,
        "gram_checkpoint_sha256_after": checkpoint_hash_after,
        "gram_checkpoint_unchanged": checkpoint_unchanged,
        "validation_used_for_covariance_requests_or_layer_selection": False,
        "original_user_sequence_opened": False,
        "similar_item_sasrec_opened": False,
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
            "covariance": "fixed SHA-ranked audited train-only next-item transitions",
            "edit_contexts": "BGE-nearest warm items with audited train-only occurrence histories",
            "edit_targets": "complete frozen cold catalog, independent of validation/test occurrence",
            "layer_selection": "frozen attempt-4 train-only probe",
            "validation_target_used": False,
            "test_read": False,
        },
    )
    atomic_json(
        output_dir / "open_file_manifest.json",
        {
            "opened": sorted(input_hashes),
            "original_user_sequence_opened": False,
            "similar_item_sasrec_opened": False,
            "test_predictions_opened": False,
            "test_metrics_opened": False,
        },
    )
    artifact_files = [request_path, trigger_path, covariance_path] + [
        delta_root / f"position_{position}.pt" for position in sorted(deltas_by_position)
    ]
    atomic_json(
        output_dir / "resource_summary.json",
        {
            "runtime_seconds": summary["runtime_seconds"],
            "peak_cuda_allocated_mib": summary["peak_cuda_allocated_mib"],
            "base_model_training": False,
            "artifact_bytes": sum(path.stat().st_size for path in artifact_files),
            "positionwise_requests": len(requests),
            "delta_bundles": len(deltas_by_position),
        },
    )
    print(
        json.dumps(
            {"status": "completed", "verdict": verdict, "summary": str(output_dir / "summary.json")}
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run(parse_args())
