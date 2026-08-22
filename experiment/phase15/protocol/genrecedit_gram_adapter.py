"""Clean-room GenRecEdit-to-GRAM contract primitives for Stage15 B3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MethodType
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


class SecondMomentAccumulator:
    """Streaming E[x x^T] accumulator for train-only covariance state."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("Covariance dimension must be positive")
        self.dimension = int(dimension)
        self.count = 0
        self._sum_outer = torch.zeros(dimension, dimension, dtype=torch.float64)

    def update(self, activations: torch.Tensor) -> None:
        if activations.ndim != 2 or activations.size(1) != self.dimension:
            raise ValueError("Covariance activations have the wrong shape")
        values = activations.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Covariance activations must be finite")
        self._sum_outer.addmm_(values.T, values)
        self.count += int(values.size(0))

    def moment(self, *, ridge: float = 0.0) -> torch.Tensor:
        if self.count < 1:
            raise ValueError("Cannot finalize an empty covariance accumulator")
        if ridge < 0:
            raise ValueError("Covariance ridge must be non-negative")
        result = self._sum_outer / self.count
        if ridge:
            result = result + float(ridge) * torch.eye(self.dimension, dtype=result.dtype)
        return result


def edited_parameter_name(layer: int) -> str:
    if layer < 0:
        raise ValueError("Decoder layer must be non-negative")
    return f"decoder.block.{layer}.layer.2.DenseReluDense.wo.weight"


@dataclass(frozen=True)
class PositionWiseRequest:
    cold_item: str
    context_items: tuple[str, ...]
    prefix_tokens: tuple[str, ...]
    target_token: str
    position: int
    source_warm_item: str


def build_positionwise_requests(
    *,
    cold_paths: Mapping[str, Sequence[str]],
    pseudo_contexts: Mapping[str, Sequence[tuple[str, Sequence[str]]]],
) -> list[PositionWiseRequest]:
    """Expand every cold item into context-to-next-token edit requests.

    The caller must supply pseudo histories created from train-only warm-item
    occurrences.  EOS and padding are not represented in ``cold_paths`` and
    therefore can never become edit targets.
    """

    if set(pseudo_contexts) != set(cold_paths):
        raise ValueError("Pseudo contexts must cover the complete frozen cold catalog")
    requests: list[PositionWiseRequest] = []
    for cold_item in sorted(cold_paths):
        path = tuple(cold_paths[cold_item])
        if not path:
            raise ValueError(f"Cold item has an empty path: {cold_item}")
        contexts = pseudo_contexts[cold_item]
        if not contexts:
            raise ValueError(f"Cold item has no train-only pseudo context: {cold_item}")
        seen_contexts: set[tuple[str, tuple[str, ...]]] = set()
        for warm_item, context in contexts:
            context_tuple = tuple(context)
            if not context_tuple:
                raise ValueError(f"Pseudo context is empty for {cold_item}")
            key = (warm_item, context_tuple)
            if key in seen_contexts:
                raise ValueError(f"Duplicate pseudo context for {cold_item}")
            seen_contexts.add(key)
            for position, token in enumerate(path):
                requests.append(
                    PositionWiseRequest(
                        cold_item=cold_item,
                        context_items=context_tuple,
                        prefix_tokens=path[:position],
                        target_token=token,
                        position=position,
                        source_warm_item=warm_item,
                    )
                )
    return requests


def validate_request_universe(
    *,
    requests: Sequence[PositionWiseRequest],
    cold_paths: Mapping[str, Sequence[str]],
    contexts_per_cold: int,
) -> dict[int, int]:
    """Validate complete cold-catalog and lexical-position request coverage."""

    if contexts_per_cold < 1:
        raise ValueError("contexts_per_cold must be positive")
    expected_total = sum(len(path) * contexts_per_cold for path in cold_paths.values())
    if len(requests) != expected_total:
        raise ValueError("Position-wise request count does not match the frozen universe")
    per_cold: dict[str, int] = {item: 0 for item in cold_paths}
    per_position: dict[int, int] = {}
    for request in requests:
        if request.cold_item not in cold_paths:
            raise ValueError("Edit request contains an unknown cold item")
        path = tuple(cold_paths[request.cold_item])
        if (
            request.position < 0
            or request.position >= len(path)
            or request.prefix_tokens != path[: request.position]
            or request.target_token != path[request.position]
        ):
            raise ValueError("Edit request does not match its frozen cold path")
        per_cold[request.cold_item] += 1
        per_position[request.position] = per_position.get(request.position, 0) + 1
    for item, path in cold_paths.items():
        if per_cold[item] != len(path) * contexts_per_cold:
            raise ValueError(f"Cold item request coverage mismatch: {item}")
    return per_position


def select_positionwise_smoke_requests(
    requests: Sequence[PositionWiseRequest],
    *,
    requests_per_position: int,
    seed: int,
) -> dict[int, list[PositionWiseRequest]]:
    """Select deterministic, cold-item-disjoint request samples per position."""

    if requests_per_position < 1:
        raise ValueError("requests_per_position must be positive")
    grouped: dict[int, list[PositionWiseRequest]] = {}
    for request in requests:
        grouped.setdefault(request.position, []).append(request)
    if not grouped:
        raise ValueError("No position-wise requests supplied")
    result: dict[int, list[PositionWiseRequest]] = {}
    for position, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(
                    ":".join(
                        [
                            str(seed),
                            "b3-edit",
                            str(position),
                            row.cold_item,
                            row.source_warm_item,
                            *row.context_items,
                        ]
                    ).encode("utf-8")
                ).digest(),
                row.cold_item,
                row.source_warm_item,
                row.context_items,
            ),
        )
        selected: list[PositionWiseRequest] = []
        seen_cold: set[str] = set()
        for row in ranked:
            if row.cold_item in seen_cold:
                continue
            selected.append(row)
            seen_cold.add(row.cold_item)
            if len(selected) == requests_per_position:
                break
        if len(selected) != requests_per_position:
            raise ValueError(f"Insufficient distinct cold items at position {position}")
        result[position] = selected
    return result


def select_branching_positionwise_smoke_requests(
    requests: Sequence[PositionWiseRequest],
    *,
    catalog_paths: Mapping[str, Sequence[str]],
    requests_per_position: int,
    seed: int,
    minimum_legal_children: int = 2,
) -> dict[int, list[PositionWiseRequest]]:
    """Select deterministic edit requests only at non-trivial trie branches.

    A prefix with one legal continuation has legal-set target probability 1 by
    construction.  It therefore cannot satisfy the GenRecEdit admission rule
    requiring a strict probability increase, regardless of layer, optimizer,
    seed, or step budget.  Filtering those structurally solved requests is an
    item-catalog operation; it does not inspect interaction targets or model
    outcomes.  The existing SHA ranking and distinct-cold-item rule are then
    applied unchanged to the remaining requests.
    """

    if minimum_legal_children < 2:
        raise ValueError("Branching request selection requires at least two children")
    if not catalog_paths:
        raise ValueError("Branching request selection requires a catalog")
    filtered: list[PositionWiseRequest] = []
    children_by_prefix: dict[tuple[str, ...], set[str]] = {}
    for raw_path in catalog_paths.values():
        path = tuple(raw_path)
        for position, token in enumerate(path):
            children_by_prefix.setdefault(path[:position], set()).add(token)
    for request in requests:
        prefix = tuple(request.prefix_tokens)
        children = children_by_prefix.get(prefix, set())
        if request.target_token not in children:
            raise ValueError("Edit request target is not a catalog continuation")
        if len(children) >= minimum_legal_children:
            filtered.append(request)
    required_positions = {request.position for request in requests}
    try:
        selected = select_positionwise_smoke_requests(
            filtered,
            requests_per_position=requests_per_position,
            seed=seed,
        )
    except ValueError as error:
        raise ValueError(
            "Insufficient distinct branching cold items for every lexical position"
        ) from error
    if set(selected) != required_positions:
        raise ValueError(
            "Insufficient distinct branching cold items for every lexical position"
        )
    return selected


def legal_next_token_ids(
    encoded_paths: Mapping[str, Sequence[int]], prefix: Sequence[int]
) -> tuple[int, ...]:
    """Return the sorted lexical-trie children for an EOS-free prefix."""

    prefix_tuple = tuple(int(token) for token in prefix)
    children = {
        int(path[len(prefix_tuple)])
        for path in encoded_paths.values()
        if len(path) > len(prefix_tuple) and tuple(path[: len(prefix_tuple)]) == prefix_tuple
    }
    if not children:
        raise ValueError("Prefix has no legal catalog continuation")
    return tuple(sorted(children))


def legal_target_state(
    logits: torch.Tensor,
    *,
    target_token_id: int,
    legal_token_ids: Sequence[int],
) -> tuple[bool, float]:
    """Check GRAM lexical-trie argmax and return legal-set target probability."""

    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Token logits must be a finite vector")
    legal = tuple(int(token) for token in legal_token_ids)
    if not legal or len(legal) != len(set(legal)):
        raise ValueError("Legal token IDs must be non-empty and unique")
    if target_token_id not in legal:
        raise ValueError("Target token is not a legal continuation")
    if min(legal) < 0 or max(legal) >= logits.numel():
        raise ValueError("Legal token ID is outside the model vocabulary")
    legal_logits = logits[torch.tensor(legal, device=logits.device)]
    target_index = legal.index(int(target_token_id))
    target_probability = float(torch.softmax(legal_logits.float(), dim=0)[target_index])
    is_argmax = bool(legal_logits[target_index] >= legal_logits.max())
    return is_argmax, target_probability


def position_population(cold_paths: Mapping[str, Sequence[str]]) -> dict[int, int]:
    population: dict[int, int] = {}
    for path in cold_paths.values():
        for position in range(len(path)):
            population[position] = population.get(position, 0) + 1
    return population


def validate_position_layer_selection(
    *,
    cold_paths: Mapping[str, Sequence[str]],
    position_to_layer: Mapping[int, int],
    decoder_layers: int,
) -> dict[int, int]:
    required = set(position_population(cold_paths))
    if set(position_to_layer) != required:
        raise ValueError("Layer selection must cover every observed lexical position")
    if decoder_layers < 1:
        raise ValueError("decoder_layers must be positive")
    normalized = {int(position): int(layer) for position, layer in position_to_layer.items()}
    if any(layer < 0 or layer >= decoder_layers for layer in normalized.values()):
        raise ValueError("Selected edit layer is outside the GRAM decoder")
    return normalized


def select_probe_layers(
    probe_accuracy: Mapping[int, Mapping[int, float]],
    *,
    decoder_layers: int,
) -> dict[int, int]:
    """Select the best train-only probe layer with deterministic shallow tie-break."""

    if not probe_accuracy:
        raise ValueError("No probe accuracies supplied")
    selected: dict[int, int] = {}
    for position, scores in probe_accuracy.items():
        if set(scores) != set(range(decoder_layers)):
            raise ValueError("Every position must probe every decoder layer")
        if any(not torch.isfinite(torch.tensor(float(value))) for value in scores.values()):
            raise ValueError("Probe accuracies must be finite")
        selected[int(position)] = min(
            scores,
            key=lambda layer: (-float(scores[layer]), int(layer)),
        )
    return selected


def accumulate_probe_predictions(
    *,
    predictions_by_layer: Mapping[int, torch.Tensor],
    labels: torch.Tensor,
    eos_token_id: int,
) -> dict[int, dict[int, tuple[int, int]]]:
    """Count train-only token correctness by lexical position and layer."""

    if labels.ndim != 2:
        raise ValueError("Probe labels must be a matrix")
    if not predictions_by_layer:
        raise ValueError("No layer predictions supplied")
    expected_shape = tuple(labels.shape)
    if any(tuple(prediction.shape) != expected_shape for prediction in predictions_by_layer.values()):
        raise ValueError("Layer predictions do not align with labels")
    active = labels.ne(-100) & labels.ne(int(eos_token_id))
    counts: dict[int, dict[int, tuple[int, int]]] = {}
    for position in range(labels.size(1)):
        position_active = active[:, position]
        total = int(position_active.sum().item())
        if total == 0:
            continue
        counts[position] = {}
        for layer, predictions in predictions_by_layer.items():
            correct = int(
                (predictions[:, position][position_active] == labels[:, position][position_active])
                .sum()
                .item()
            )
            counts[position][int(layer)] = (correct, total)
    return counts


def merge_probe_counts(
    destination: dict[int, dict[int, list[int]]],
    update: Mapping[int, Mapping[int, tuple[int, int]]],
) -> None:
    """Merge correctness counts in place while preserving exact denominators."""

    for position, layers in update.items():
        destination.setdefault(int(position), {})
        for layer, (correct, total) in layers.items():
            current = destination[int(position)].setdefault(int(layer), [0, 0])
            current[0] += int(correct)
            current[1] += int(total)


def probe_accuracy_from_counts(
    counts: Mapping[int, Mapping[int, Sequence[int]]],
    *,
    decoder_layers: int,
) -> dict[int, dict[int, float]]:
    """Convert complete position/layer counts into deterministic accuracies."""

    if not counts:
        raise ValueError("No probe counts supplied")
    expected_layers = set(range(decoder_layers))
    result: dict[int, dict[int, float]] = {}
    for position, layers in counts.items():
        if set(layers) != expected_layers:
            raise ValueError("Every probe position must cover every decoder layer")
        result[int(position)] = {}
        totals = {int(values[1]) for values in layers.values()}
        if len(totals) != 1 or next(iter(totals)) < 1:
            raise ValueError("Probe denominators must be positive and layer-consistent")
        for layer, values in layers.items():
            correct, total = map(int, values)
            if correct < 0 or correct > total:
                raise ValueError("Invalid probe correctness count")
            result[int(position)][int(layer)] = correct / total
    return result


def solve_closed_form_delta(
    *,
    residual: torch.Tensor,
    keys: torch.Tensor,
    covariance: torch.Tensor,
    preservation_lambda: float,
) -> torch.Tensor:
    """Solve ΔW = R Kᵀ (λC + K Kᵀ)⁻¹ without forming an inverse."""

    if residual.ndim != 2 or keys.ndim != 2 or covariance.ndim != 2:
        raise ValueError("Residual, keys, and covariance must be matrices")
    if residual.shape[1] != keys.shape[1]:
        raise ValueError("Residual and keys must describe the same requests")
    if covariance.shape != (keys.shape[0], keys.shape[0]):
        raise ValueError("Covariance shape does not match key dimension")
    if preservation_lambda <= 0:
        raise ValueError("preservation_lambda must be positive")
    system = preservation_lambda * covariance + keys @ keys.T
    rhs = residual @ keys.T
    try:
        return torch.linalg.solve(system.T.double(), rhs.T.double()).T.to(residual.dtype)
    except RuntimeError as error:
        raise ValueError("GenRecEdit linear system is singular or invalid") from error


def validate_delta_shapes(
    *,
    base_parameters: Mapping[str, torch.Tensor],
    deltas_by_position: Mapping[int, Mapping[str, torch.Tensor]],
    position_to_layer: Mapping[int, int],
) -> None:
    if set(deltas_by_position) != set(position_to_layer):
        raise ValueError("There must be exactly one delta bundle per lexical position")
    for position, bundle in deltas_by_position.items():
        if not bundle:
            raise ValueError(f"Empty delta bundle for position {position}")
        expected_fragment = f"decoder.block.{position_to_layer[position]}.layer.2.DenseReluDense.wo.weight"
        for name, delta in bundle.items():
            if name != expected_fragment:
                raise ValueError(f"Unexpected edited parameter for position {position}: {name}")
            if name not in base_parameters:
                raise ValueError(f"Edited parameter is absent from frozen GRAM: {name}")
            if tuple(delta.shape) != tuple(base_parameters[name].shape):
                raise ValueError(f"deltaW shape mismatch for {name}")


class OneOneDeltaRouter:
    """Expose only the delta bundle belonging to the current lexical position."""

    def __init__(
        self,
        *,
        deltas_by_position: Mapping[int, Mapping[str, torch.Tensor]],
        position_to_layer: Mapping[int, int],
    ) -> None:
        self._deltas = {
            int(position): dict(bundle) for position, bundle in deltas_by_position.items()
        }
        self._position_to_layer = dict(position_to_layer)
        if set(self._deltas) != set(self._position_to_layer):
            raise ValueError("Router positions do not match the layer map")

    def active_bundle(
        self, position: int, *, is_eos: bool = False, is_padding: bool = False
    ) -> Mapping[str, torch.Tensor]:
        if is_eos or is_padding:
            return {}
        if position not in self._deltas:
            raise ValueError(f"No edit is defined for lexical position {position}")
        return self._deltas[position]

    def materialize_parameter(
        self,
        name: str,
        base_value: torch.Tensor,
        position: int,
        *,
        is_eos: bool = False,
        is_padding: bool = False,
    ) -> torch.Tensor:
        bundle = self.active_bundle(position, is_eos=is_eos, is_padding=is_padding)
        delta = bundle.get(name)
        return base_value if delta is None else base_value + delta.to(base_value)


class OneOneGenerationDeltaContext:
    """Apply position-wise ``deltaW`` during cached lexical beam generation.

    The context observes complete decoder prefixes before cached generation
    slices them to the newest token.  At the selected decoder FFN it adds the
    term equivalent to ``hidden @ deltaW.T`` without mutating base parameters.
    Complete catalog paths (whose next token is EOS) and EOS/padded rows remain
    inactive.
    """

    def __init__(
        self,
        *,
        model,
        deltas_by_position: Mapping[int, Mapping[str, torch.Tensor]],
        position_to_layer: Mapping[int, int],
        encoded_catalog_paths: Iterable[Sequence[int]],
        decoder_start_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> None:
        self.model = model
        self.router = OneOneDeltaRouter(
            deltas_by_position=deltas_by_position,
            position_to_layer=position_to_layer,
        )
        self.position_to_layer = {
            int(position): int(layer) for position, layer in position_to_layer.items()
        }
        path_rows = [tuple(map(int, path)) for path in encoded_catalog_paths]
        self.complete_paths = set(path_rows)
        if not self.complete_paths or () in self.complete_paths:
            raise ValueError("Encoded catalog paths must be non-empty")
        if len(self.complete_paths) != len(path_rows):
            raise ValueError("Encoded catalog paths contain a collision")
        self.valid_prefixes = {
            path[:depth]
            for path in self.complete_paths
            for depth in range(len(path))
        }
        self.decoder_start_token_id = int(decoder_start_token_id)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self._handles = []
        self._original_prepare = None
        self._active_position: int | None = None
        self._active_mask: torch.Tensor | None = None
        self.applied_rows_by_position = {position: 0 for position in self.position_to_layer}
        self.prepare_calls = 0

    def _parameter_name(self, position: int) -> str:
        bundle = self.router.active_bundle(position)
        if len(bundle) != 1:
            raise ValueError("Generation requires exactly one edited parameter per position")
        return next(iter(bundle))

    def _set_decoder_prefixes(self, decoder_input_ids: torch.Tensor) -> None:
        if decoder_input_ids.ndim != 2 or decoder_input_ids.size(1) < 1:
            raise ValueError("Generation decoder prefixes must be a non-empty matrix")
        active: list[bool] = []
        positions: set[int] = set()
        for raw in decoder_input_ids.detach().cpu().tolist():
            if int(raw[0]) != self.decoder_start_token_id:
                raise ValueError("Unexpected decoder start token during One-One generation")
            suffix = tuple(int(token) for token in raw[1:])
            if self.eos_token_id in suffix:
                active.append(False)
                continue
            while suffix and suffix[-1] == self.pad_token_id:
                suffix = suffix[:-1]
            position = len(suffix)
            positions.add(position)
            if suffix in self.complete_paths:
                active.append(False)
            elif suffix in self.valid_prefixes:
                active.append(position in self.position_to_layer)
            else:
                raise ValueError("Generation prefix is outside the frozen lexical trie")
        if len(positions) != 1:
            raise ValueError("Beam rows disagree on the current lexical position")
        position = next(iter(positions))
        self._active_position = position if position in self.position_to_layer else None
        self._active_mask = torch.tensor(active, dtype=torch.bool)
        self.prepare_calls += 1

    def _inject(self, layer: int, inputs, output):
        position = self._active_position
        mask = self._active_mask
        if position is None or mask is None or self.position_to_layer[position] != layer:
            return output
        hidden = inputs[0]
        if hidden.ndim != 3 or output.ndim != 3 or hidden.size(0) != mask.numel():
            raise ValueError("One-One generation hook batch shape mismatch")
        name = self._parameter_name(position)
        if name != edited_parameter_name(layer):
            raise ValueError("One-One generation delta targets the wrong decoder layer")
        delta = self.router.active_bundle(position)[name].to(hidden)
        addition = F.linear(hidden[:, -1, :], delta)
        active_mask = mask.to(device=addition.device)
        if not bool(active_mask.any()):
            return output
        modified = output.clone()
        modified[:, -1, :] = modified[:, -1, :] + addition * active_mask[:, None]
        self.applied_rows_by_position[position] += int(active_mask.sum().item())
        return modified

    def __enter__(self):
        if self._handles or self._original_prepare is not None:
            raise RuntimeError("One-One generation context cannot be entered twice")
        self._original_prepare = self.model.prepare_inputs_for_generation
        original = self._original_prepare

        def prepare(_model, decoder_input_ids, **kwargs):
            self._set_decoder_prefixes(decoder_input_ids)
            return original(decoder_input_ids, **kwargs)

        self.model.prepare_inputs_for_generation = MethodType(prepare, self.model)
        for layer in sorted(set(self.position_to_layer.values())):
            module = self.model.decoder.block[layer].layer[2].DenseReluDense.wo
            self._handles.append(
                module.register_forward_hook(
                    lambda _module, inputs, output, layer_index=layer: self._inject(
                        layer_index, inputs, output
                    )
                )
            )
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._original_prepare is not None:
            self.model.prepare_inputs_for_generation = self._original_prepare
            self._original_prepare = None
        self._active_position = None
        self._active_mask = None
        return False
