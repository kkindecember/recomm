#!/usr/bin/env python3
"""Clean-room faithful GenRecEdit-to-GRAM core primitives for Stage16.

The implementation follows the function-level S16-0 fidelity matrix while
keeping model- and dataset-specific plumbing outside this module.  It reuses
the project-internal Stage15 GRAM hook/second-moment foundations (and does not
import or copy third-party GenRecEdit code).  In particular, optimizer
satisfaction is legal-lexical argmax equality, whereas the frozen ``0.3``
probability threshold is used only when probing a cached z vector and is
computed in the full vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from experiment.phase15.protocol.genrecedit_gram_adapter import (
    OneOneGenerationDeltaContext as _Stage15OneOneGenerationDeltaContext,
    SecondMomentAccumulator,
    edited_parameter_name,
)


@dataclass(frozen=True)
class FullTargetRequest:
    """One train-context, lexical-prefix, next-token edit request."""

    cold_item: str
    source_warm_item: str
    context_items: tuple[str, ...]
    full_target_path: tuple[int, ...]
    prefix_token_ids: tuple[int, ...]
    target_token_id: int
    legal_token_ids: tuple[int, ...]
    position: int


def _normalize_path(path: Sequence[int], *, item: str) -> tuple[int, ...]:
    if not path or any(isinstance(token, bool) or not isinstance(token, int) for token in path):
        raise ValueError(f"Cold item requires a non-empty integer lexical path: {item}")
    return tuple(int(token) for token in path)


def build_full_target_requests(
    *,
    catalog_paths: Mapping[str, Sequence[int]],
    cold_paths: Mapping[str, Sequence[int]],
    pseudo_contexts: Mapping[str, Sequence[tuple[str, Sequence[str]]]],
    eos_token_id: int,
    pad_token_id: int,
) -> list[FullTargetRequest]:
    """Expand every frozen cold path and every train-only pseudo context.

    The caller supplies the complete, pre-frozen cold catalog and contexts
    derived from training interactions.  No outcome- or split-based selection
    is performed here.  EOS and padding are forbidden as edit targets.
    """

    if not catalog_paths or not cold_paths or set(pseudo_contexts) != set(cold_paths):
        raise ValueError("Pseudo contexts must cover the complete frozen cold catalog")
    catalog = {
        str(item): _normalize_path(path, item=str(item))
        for item, path in catalog_paths.items()
    }
    paths = {
        str(item): _normalize_path(path, item=str(item))
        for item, path in cold_paths.items()
    }
    if not set(paths).issubset(catalog) or any(catalog[item] != path for item, path in paths.items()):
        raise ValueError("Frozen cold paths must be an exact subset of the complete catalog")
    if len(set(catalog.values())) != len(catalog):
        raise ValueError("Frozen catalog lexical paths must be collision-free")
    excluded = {int(eos_token_id), int(pad_token_id)}
    if any(excluded.intersection(path) for path in catalog.values()):
        raise ValueError("EOS and padding cannot occur in a catalog lexical path")

    children_by_prefix: dict[tuple[int, ...], set[int]] = {}
    for path in catalog.values():
        for position, token in enumerate(path):
            children_by_prefix.setdefault(path[:position], set()).add(token)

    requests: list[FullTargetRequest] = []
    for item in sorted(paths):
        contexts = pseudo_contexts[item]
        if not contexts:
            raise ValueError(f"Cold item has no train-only pseudo context: {item}")
        seen_contexts: set[tuple[str, tuple[str, ...]]] = set()
        for raw_warm_item, raw_context in contexts:
            warm_item = str(raw_warm_item)
            context = tuple(str(value) for value in raw_context)
            if not warm_item or not context:
                raise ValueError(f"Cold item has an empty train-only context: {item}")
            context_key = (warm_item, context)
            if context_key in seen_contexts:
                raise ValueError(f"Cold item has a duplicate train-only context: {item}")
            seen_contexts.add(context_key)
            path = paths[item]
            for position, target in enumerate(path):
                prefix = path[:position]
                requests.append(
                    FullTargetRequest(
                        cold_item=item,
                        source_warm_item=warm_item,
                        context_items=context,
                        full_target_path=path,
                        prefix_token_ids=prefix,
                        target_token_id=target,
                        legal_token_ids=tuple(sorted(children_by_prefix[prefix])),
                        position=position,
                    )
                )
    return requests


def batch_full_target_requests(
    requests: Sequence[FullTargetRequest], *, batch_size: int
) -> dict[int, list[tuple[FullTargetRequest, ...]]]:
    """Return complete position-wise batches without sampling or truncation."""

    if batch_size < 1:
        raise ValueError("Full-target batch size must be positive")
    if not requests:
        raise ValueError("Full-target request universe is empty")
    grouped: dict[int, list[FullTargetRequest]] = {}
    for request in requests:
        grouped.setdefault(int(request.position), []).append(request)
    return {
        position: [
            tuple(rows[start : start + batch_size])
            for start in range(0, len(rows), batch_size)
        ]
        for position, rows in sorted(grouped.items())
    }


@dataclass(frozen=True)
class CacheProbeResult:
    legal_argmax: bool
    legal_rank: int
    full_vocabulary_probability: float
    cache_hit: bool


def probe_cached_z(
    logits: torch.Tensor,
    *,
    target_token_id: int,
    legal_token_ids: Sequence[int],
    probability_threshold: float = 0.3,
) -> CacheProbeResult:
    """Probe an already-injected z using the frozen split probability semantics."""

    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Cached-z probe logits must be a finite vector")
    legal = tuple(int(token) for token in legal_token_ids)
    if not legal or len(legal) != len(set(legal)) or int(target_token_id) not in legal:
        raise ValueError("Cached-z target must belong to a unique legal token set")
    if min(legal) < 0 or max(legal) >= logits.numel():
        raise ValueError("Cached-z legal token is outside the vocabulary")
    if not 0.0 <= float(probability_threshold) <= 1.0:
        raise ValueError("Cached-z probability threshold must be in [0, 1]")
    legal_logits = logits[torch.tensor(legal, device=logits.device)]
    target_index = legal.index(int(target_token_id))
    legal_argmax = bool(legal_logits[target_index] >= legal_logits.max())
    legal_rank = 1 + int((legal_logits > legal_logits[target_index]).sum().item())
    probability = float(torch.softmax(logits.float(), dim=-1)[int(target_token_id)])
    return CacheProbeResult(
        legal_argmax=legal_argmax,
        legal_rank=legal_rank,
        full_vocabulary_probability=probability,
        cache_hit=legal_argmax and probability > float(probability_threshold),
    )


@dataclass(frozen=True)
class CachedZObservation:
    """Model observation while probing a cached absolute z vector."""

    logits: torch.Tensor
    target_init: torch.Tensor


@dataclass(frozen=True)
class CachedZHit:
    z_vector: torch.Tensor
    delta_vector: torch.Tensor
    probe: CacheProbeResult


def try_cache_hits(
    *,
    requests: Sequence[FullTargetRequest],
    target_layer: int,
    z_cache: Mapping[tuple[int, str, int], Sequence[torch.Tensor]],
    probe_forward: Callable[[FullTargetRequest, torch.Tensor, int], CachedZObservation],
    probability_threshold: float = 0.3,
) -> dict[int, CachedZHit]:
    """Try cached absolute z vectors in official first-passing-candidate order."""

    if target_layer < 0:
        raise ValueError("Cached-z target layer must be non-negative")
    hits: dict[int, CachedZHit] = {}
    for index, request in enumerate(requests):
        key = (int(target_layer), str(request.target_token_id), int(request.position))
        for raw_z in z_cache.get(key, ()):
            z = raw_z.detach()
            observation = probe_forward(request, z, int(target_layer))
            if observation.target_init.shape != z.shape:
                raise ValueError("Cached z and current target initialization do not align")
            if not bool(torch.isfinite(observation.target_init).all()):
                raise ValueError("Cached-z target initialization must be finite")
            probe = probe_cached_z(
                observation.logits,
                target_token_id=request.target_token_id,
                legal_token_ids=request.legal_token_ids,
                probability_threshold=probability_threshold,
            )
            if probe.cache_hit:
                current_init = observation.target_init.detach().to(z)
                hits[index] = CachedZHit(
                    z_vector=z.clone(),
                    delta_vector=(z - current_init).clone(),
                    probe=probe,
                )
                break
    return hits


def optimizer_satisfied(
    logits: torch.Tensor, *, target_token_id: int, legal_token_ids: Sequence[int]
) -> bool:
    """Optimizer success is legal-set argmax equality and has no 0.3 gate."""

    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Optimizer logits must be a finite vector")
    legal = tuple(int(token) for token in legal_token_ids)
    if not legal or len(legal) != len(set(legal)) or int(target_token_id) not in legal:
        raise ValueError("Optimizer target must belong to a unique legal token set")
    if min(legal) < 0 or max(legal) >= logits.numel():
        raise ValueError("Optimizer legal token is outside the vocabulary")
    local = logits[torch.tensor(legal, device=logits.device)]
    return int(torch.argmax(local).item()) == legal.index(int(target_token_id))


@dataclass(frozen=True)
class ZLifecycleUpdate:
    active_indices: tuple[int, ...]
    satisfied_indices: tuple[int, ...]


def update_z_lifecycle(
    *,
    logits: torch.Tensor,
    requests: Sequence[FullTargetRequest],
    active_indices: Sequence[int],
) -> ZLifecycleUpdate:
    """Remove legal-argmax-satisfied rows from the active optimizer set."""

    if logits.ndim != 2 or logits.shape[0] != len(requests):
        raise ValueError("Lifecycle logits and request batch do not align")
    active = tuple(int(index) for index in active_indices)
    if len(active) != len(set(active)) or any(index < 0 or index >= len(requests) for index in active):
        raise ValueError("Lifecycle active indices are invalid")
    remaining: list[int] = []
    satisfied: list[int] = []
    for index in active:
        request = requests[index]
        if optimizer_satisfied(
            logits[index],
            target_token_id=request.target_token_id,
            legal_token_ids=request.legal_token_ids,
        ):
            satisfied.append(index)
        else:
            remaining.append(index)
    return ZLifecycleUpdate(tuple(remaining), tuple(satisfied))


def clip_delta_norm_(delta_tensor: torch.Tensor, maximum_norm: float = 8000.0) -> bool:
    """Clip one z delta to an absolute norm cap, in place."""

    if delta_tensor.ndim != 1 or not bool(torch.isfinite(delta_tensor).all()):
        raise ValueError("z delta must be one finite vector")
    if maximum_norm <= 0:
        raise ValueError("Absolute z-vector maximum must be positive")
    with torch.no_grad():
        norm = torch.linalg.vector_norm(delta_tensor)
        if float(norm) > float(maximum_norm):
            delta_tensor.mul_(float(maximum_norm) / norm)
            return True
    return False


@dataclass(frozen=True)
class ZOptimizationConfig:
    v_lr: float = 0.5
    v_num_grad_steps: int = 30
    v_weight_decay: float = 0.2
    z_vector_max: float = 8000.0
    eta_min: float = 0.01
    batch_size: int = 2048

    def __post_init__(self) -> None:
        if self.v_lr <= 0 or self.v_num_grad_steps < 1 or self.v_weight_decay < 0:
            raise ValueError("Invalid official z optimizer hyperparameters")
        if self.z_vector_max <= 0 or self.eta_min < 0 or self.batch_size < 1:
            raise ValueError("Invalid official z scheduler/batch hyperparameters")


@dataclass(frozen=True)
class ZForwardBatch:
    """Differentiable logits plus unedited FFN outputs for one z step."""

    logits: torch.Tensor
    target_inits: torch.Tensor


@dataclass(frozen=True)
class ZOptimizationResult:
    z_vectors: tuple[torch.Tensor | None, ...]
    delta_vectors: tuple[torch.Tensor | None, ...]
    cache_hit_indices: tuple[int, ...]
    optimizer_satisfied_indices: tuple[int, ...]
    failed_indices: tuple[int, ...]
    scheduler_lrs_by_batch: tuple[tuple[float, ...], ...]
    lifecycle_check_steps_by_batch: tuple[tuple[int, ...], ...]

    @property
    def valid_count(self) -> int:
        return len(self.z_vectors) - len(self.failed_indices)

    @property
    def failed_count(self) -> int:
        return len(self.failed_indices)


def _official_lifecycle_check(step: int, total_steps: int) -> bool:
    return step > 0 and (step % 10 == 0 or step > total_steps - 10)


def optimize_z_vectors(
    *,
    requests: Sequence[FullTargetRequest],
    vector_dimension: int,
    device: torch.device | str,
    forward_batch: Callable[
        [Sequence[FullTargetRequest], torch.Tensor, Sequence[int]], ZForwardBatch
    ],
    config: ZOptimizationConfig = ZOptimizationConfig(),
    cache_hits: Mapping[int, CachedZHit] | None = None,
) -> ZOptimizationResult:
    """Run official Adam+cosine z optimization over every supplied request.

    Calls are position-wise, matching the official editor.  Model hooks and
    encoder caching remain in ``forward_batch`` so this core is CPU-testable.
    """

    if not requests or vector_dimension < 1:
        raise ValueError("z optimization requires requests and a positive vector width")
    positions = {int(request.position) for request in requests}
    if len(positions) != 1:
        raise ValueError("z optimization batches must contain one lexical position")
    hits = dict(cache_hits or {})
    if any(index < 0 or index >= len(requests) for index in hits):
        raise ValueError("Cached-z hit index is outside the request universe")
    for hit in hits.values():
        if hit.z_vector.shape != (vector_dimension,) or hit.delta_vector.shape != (vector_dimension,):
            raise ValueError("Cached-z vector width does not match optimization width")

    z_vectors: list[torch.Tensor | None] = [None] * len(requests)
    delta_vectors: list[torch.Tensor | None] = [None] * len(requests)
    optimizer_satisfied_indices: list[int] = []
    scheduler_traces: list[tuple[float, ...]] = []
    lifecycle_traces: list[tuple[int, ...]] = []

    for start in range(0, len(requests), config.batch_size):
        end = min(start + config.batch_size, len(requests))
        batch = tuple(requests[start:end])
        batch_count = len(batch)
        batch_hits = {
            global_index - start: hit
            for global_index, hit in hits.items()
            if start <= global_index < end
        }
        batch_device = torch.device(device)
        deltas = torch.zeros(
            batch_count, vector_dimension, device=batch_device, requires_grad=True
        )
        target_inits: list[torch.Tensor | None] = [None] * batch_count
        active = [index for index in range(batch_count) if index not in batch_hits]
        for local_index, hit in batch_hits.items():
            z_vectors[start + local_index] = hit.z_vector.detach().to(batch_device).clone()
            delta_vectors[start + local_index] = hit.delta_vector.detach().to(batch_device).clone()

        optimizer = torch.optim.Adam([deltas], lr=config.v_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.v_num_grad_steps, eta_min=config.eta_min
        )
        batch_lrs: list[float] = []
        lifecycle_steps: list[int] = []

        for step in range(config.v_num_grad_steps):
            if not active:
                break
            observation = forward_batch(batch, deltas, tuple(active))
            if observation.logits.ndim != 2 or observation.logits.shape[0] != batch_count:
                raise ValueError("z forward logits do not align with the request batch")
            if observation.target_inits.shape != (batch_count, vector_dimension):
                raise ValueError("z target initializations have the wrong shape")
            if not bool(torch.isfinite(observation.logits).all()) or not bool(
                torch.isfinite(observation.target_inits).all()
            ):
                raise ValueError("z forward observation must be finite")
            if observation.logits.device != deltas.device or observation.target_inits.device != deltas.device:
                raise ValueError("z forward observation and deltas must share a device")
            for index in active:
                if target_inits[index] is None:
                    target_inits[index] = observation.target_inits[index].detach().clone()

            target_ids = torch.tensor(
                [request.target_token_id for request in batch], device=deltas.device
            )
            if int(target_ids.min()) < 0 or int(target_ids.max()) >= observation.logits.shape[1]:
                raise ValueError("z target token is outside the forward vocabulary")
            per_request_loss = F.cross_entropy(
                observation.logits, target_ids, reduction="none"
            )
            total_loss = torch.zeros((), device=deltas.device)
            for index in active:
                initial = target_inits[index]
                if initial is None:  # pragma: no cover - guarded by finite matrix validation
                    continue
                norm_penalty = config.v_weight_decay * (
                    torch.linalg.vector_norm(deltas[index])
                    / (torch.linalg.vector_norm(initial) + 1e-8)
                )
                total_loss = total_loss + per_request_loss[index] + norm_penalty

            optimizer.zero_grad()
            if float(total_loss.detach()) > 0.0:
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                batch_lrs.append(float(scheduler.get_last_lr()[0]))

            for index in active:
                clip_delta_norm_(deltas[index], config.z_vector_max)

            if _official_lifecycle_check(step, config.v_num_grad_steps):
                lifecycle_steps.append(step)
                lifecycle = update_z_lifecycle(
                    logits=observation.logits.detach(),
                    requests=batch,
                    active_indices=active,
                )
                for index in lifecycle.satisfied_indices:
                    initial = target_inits[index]
                    if initial is None:  # pragma: no cover - guarded above
                        raise RuntimeError("Satisfied z row has no target initialization")
                    z_vectors[start + index] = (initial + deltas[index].detach()).clone()
                    delta_vectors[start + index] = deltas[index].detach().clone()
                    optimizer_satisfied_indices.append(start + index)
                active = list(lifecycle.active_indices)

        scheduler_traces.append(tuple(batch_lrs))
        lifecycle_traces.append(tuple(lifecycle_steps))

    failed = tuple(index for index, vector in enumerate(z_vectors) if vector is None)
    return ZOptimizationResult(
        z_vectors=tuple(z_vectors),
        delta_vectors=tuple(delta_vectors),
        cache_hit_indices=tuple(sorted(hits)),
        optimizer_satisfied_indices=tuple(optimizer_satisfied_indices),
        failed_indices=failed,
        scheduler_lrs_by_batch=tuple(scheduler_traces),
        lifecycle_check_steps_by_batch=tuple(lifecycle_traces),
    )


@dataclass(frozen=True)
class PositionCovarianceResult:
    covariance_by_position: Mapping[int, torch.Tensor]
    available_rows_by_position: Mapping[int, int]
    used_rows_by_position: Mapping[int, int]
    mom2_n_samples: int


def collect_covariance(
    train_activations_by_position: Mapping[int, torch.Tensor],
    *,
    mom2_n_samples: int = 400_000,
) -> PositionCovarianceResult:
    """Collect separate train-only E[x x^T] moments for lexical positions."""

    if not train_activations_by_position or mom2_n_samples < 1:
        raise ValueError("Position covariance requires activations and a positive sample cap")
    normalized = {int(position): rows for position, rows in train_activations_by_position.items()}
    if any(position < 0 for position in normalized):
        raise ValueError("Covariance lexical positions must be non-negative")
    dimensions = {
        int(rows.shape[1])
        for rows in normalized.values()
        if rows.ndim == 2 and rows.shape[0] > 0
    }
    if len(dimensions) != 1 or len(dimensions) != len(
        {int(rows.shape[1]) for rows in normalized.values() if rows.ndim == 2}
    ):
        raise ValueError("Every position needs non-empty, width-aligned covariance rows")
    dimension = next(iter(dimensions))
    covariance: dict[int, torch.Tensor] = {}
    available: dict[int, int] = {}
    used: dict[int, int] = {}
    for position, rows in sorted(normalized.items()):
        if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != dimension:
            raise ValueError("Every position needs non-empty, width-aligned covariance rows")
        if not bool(torch.isfinite(rows).all()):
            raise ValueError("Covariance activations must be finite")
        count = min(int(rows.shape[0]), int(mom2_n_samples))
        accumulator = SecondMomentAccumulator(dimension)
        accumulator.update(rows[:count])
        covariance[position] = accumulator.moment()
        available[position] = int(rows.shape[0])
        used[position] = count
    return PositionCovarianceResult(covariance, available, used, int(mom2_n_samples))


def extract_keys(
    *,
    module: nn.Module,
    requests: Sequence[FullTargetRequest],
    forward_batch: Callable[[Sequence[FullTargetRequest]], object],
    batch_size: int = 2048,
) -> torch.Tensor:
    """Extract the selected decoder FFN input at each request's final token."""

    if not requests or batch_size < 1:
        raise ValueError("Key extraction requires requests and a positive batch size")
    batches = [
        tuple(requests[start : start + batch_size])
        for start in range(0, len(requests), batch_size)
    ]
    all_keys: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise ValueError("Edited FFN module did not receive a tensor input")
        values = inputs[0]
        if values.ndim == 3:
            values = values[:, -1, :]
        elif values.ndim != 2:
            raise ValueError("Edited FFN input must be [batch, sequence, width] or [batch, width]")
        captured.append(values.detach().to(device="cpu"))

    handle = module.register_forward_pre_hook(capture)
    try:
        for batch in batches:
            captured.clear()
            with torch.no_grad():
                forward_batch(batch)
            if len(captured) != 1 or captured[0].shape[0] != len(batch):
                raise ValueError("Edited FFN module must run exactly once for each key batch")
            all_keys.append(captured[0])
    finally:
        handle.remove()
    result = torch.cat(all_keys, dim=0)
    if result.shape[0] != len(requests) or not bool(torch.isfinite(result).all()):
        raise ValueError("Extracted keys do not align with the full request set")
    return result


@dataclass(frozen=True)
class ValidZSelection:
    valid_indices: tuple[int, ...]
    failed_indices: tuple[int, ...]
    z_vectors: tuple[torch.Tensor, ...]
    delta_vectors: tuple[torch.Tensor, ...]

    @property
    def valid_count(self) -> int:
        return len(self.valid_indices)

    @property
    def failed_count(self) -> int:
        return len(self.failed_indices)


def filter_valid_z(
    z_vectors: Sequence[torch.Tensor | None],
    delta_vectors: Sequence[torch.Tensor | None],
) -> ValidZSelection:
    """Select exactly non-null z rows and retain the complete failure count."""

    if len(z_vectors) != len(delta_vectors):
        raise ValueError("z and delta state lists do not align")
    valid_indices = tuple(index for index, vector in enumerate(z_vectors) if vector is not None)
    failed_indices = tuple(index for index, vector in enumerate(z_vectors) if vector is None)
    valid_z: list[torch.Tensor] = []
    valid_delta: list[torch.Tensor] = []
    for index in valid_indices:
        z, delta = z_vectors[index], delta_vectors[index]
        if z is None:  # pragma: no cover - selected above
            raise RuntimeError("Valid-z selection lost its z state")
        if delta is None:
            raise ValueError("A non-null z state must retain its aligned delta")
        if z.shape != delta.shape or not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(delta).all()):
            raise ValueError("Valid z/delta states must be finite and shape-aligned")
        valid_z.append(z)
        valid_delta.append(delta)
    return ValidZSelection(valid_indices, failed_indices, tuple(valid_z), tuple(valid_delta))


def solve_weight_delta(
    *,
    residuals: torch.Tensor,
    keys: torch.Tensor,
    covariance: torch.Tensor,
    covariance_lambda: float,
) -> torch.Tensor:
    """Solve dW = R^T K (K^T K + lambda C)^-1 without an inverse."""

    if residuals.ndim != 2 or keys.ndim != 2 or covariance.ndim != 2:
        raise ValueError("Residuals, keys, and covariance must be matrices")
    if residuals.shape[0] < 1 or residuals.shape[0] != keys.shape[0]:
        raise ValueError("Residuals and keys must describe the same non-empty requests")
    if covariance.shape != (keys.shape[1], keys.shape[1]):
        raise ValueError("Covariance shape does not match the key width")
    if covariance_lambda <= 0:
        raise ValueError("Covariance preservation lambda must be positive")
    if not all(bool(torch.isfinite(value).all()) for value in (residuals, keys, covariance)):
        raise ValueError("Closed-form solve inputs must be finite")
    key64 = keys.double()
    residual64 = residuals.double()
    system = key64.T @ key64 + float(covariance_lambda) * covariance.double()
    rhs = residual64.T @ key64
    try:
        return torch.linalg.solve(system.T, rhs.T).T.to(residuals)
    except RuntimeError as error:
        raise ValueError("GenRecEdit linear system is singular or invalid") from error


@dataclass(frozen=True)
class PositionAdmissionDiagnostics:
    position: int
    request_count: int
    cache_hit_count: int
    valid_z_count: int
    failed_z_count: int
    full_vocabulary_target_probabilities: tuple[float, ...]
    legal_target_ranks: tuple[int, ...]


@dataclass(frozen=True)
class LinearSystemDiagnostics:
    parameter_name: str
    contributing_positions: tuple[int, ...]
    delta_norm: float
    delta_rank: int
    system_condition: float


@dataclass(frozen=True)
class AdmissionDiagnostics:
    per_position: Mapping[int, PositionAdmissionDiagnostics]
    linear_systems: tuple[LinearSystemDiagnostics, ...]
    unedited_parity: Mapping[str, bool]
    warm_preservation: Mapping[str, float | int | bool]


def linear_system_diagnostics(
    *,
    parameter_name: str,
    contributing_positions: Sequence[int],
    weight_delta: torch.Tensor,
    keys: torch.Tensor,
    covariance: torch.Tensor,
    covariance_lambda: float,
) -> LinearSystemDiagnostics:
    """Compute frozen delta norm/rank/condition evidence without changing the solve."""

    if weight_delta.ndim != 2 or keys.ndim != 2 or covariance.ndim != 2:
        raise ValueError("Linear-system diagnostics require matrices")
    if covariance.shape != (keys.shape[1], keys.shape[1]) or covariance_lambda <= 0:
        raise ValueError("Linear-system diagnostics shapes/lambda are invalid")
    values = (weight_delta, keys, covariance)
    if not parameter_name or not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Linear-system diagnostics require finite named inputs")
    positions = tuple(sorted(int(position) for position in contributing_positions))
    if not positions or any(position < 0 for position in positions):
        raise ValueError("Linear-system diagnostics require contributing positions")
    system = keys.double().T @ keys.double() + float(covariance_lambda) * covariance.double()
    return LinearSystemDiagnostics(
        parameter_name=str(parameter_name),
        contributing_positions=positions,
        delta_norm=float(torch.linalg.vector_norm(weight_delta.double())),
        delta_rank=int(torch.linalg.matrix_rank(weight_delta.double()).item()),
        system_condition=float(torch.linalg.cond(system)),
    )


def validate_admission_diagnostics(
    diagnostics: AdmissionDiagnostics,
) -> dict[str, object]:
    """Validate the minimum G-FULL contract/admission evidence schema."""

    if not diagnostics.per_position:
        raise ValueError("Admission diagnostics require per-position evidence")
    total_requests = total_cache = total_valid = total_failed = 0
    for key, row in diagnostics.per_position.items():
        if int(key) != row.position or row.position < 0 or row.request_count < 1:
            raise ValueError("Admission position diagnostics are malformed")
        if row.valid_z_count + row.failed_z_count != row.request_count:
            raise ValueError("Admission valid/failed z counts do not cover requests")
        if row.cache_hit_count < 0 or row.cache_hit_count > row.valid_z_count:
            raise ValueError("Admission cache-hit count exceeds valid z states")
        if len(row.full_vocabulary_target_probabilities) != row.request_count or len(
            row.legal_target_ranks
        ) != row.request_count:
            raise ValueError("Admission probability/rank diagnostics do not cover requests")
        probabilities = torch.tensor(row.full_vocabulary_target_probabilities, dtype=torch.float64)
        if not bool(torch.isfinite(probabilities).all()) or bool(
            ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Admission full-vocabulary probabilities are invalid")
        if any(int(rank) < 1 for rank in row.legal_target_ranks):
            raise ValueError("Admission legal target ranks must be positive")
        total_requests += row.request_count
        total_cache += row.cache_hit_count
        total_valid += row.valid_z_count
        total_failed += row.failed_z_count
    if not diagnostics.linear_systems:
        raise ValueError("Admission diagnostics require delta/system evidence")
    for row in diagnostics.linear_systems:
        values = torch.tensor([row.delta_norm, row.system_condition], dtype=torch.float64)
        if (
            not row.parameter_name
            or not row.contributing_positions
            or row.delta_norm < 0
            or row.delta_rank < 0
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("Admission delta/system diagnostics are invalid")
    if not diagnostics.unedited_parity or not all(diagnostics.unedited_parity.values()):
        raise ValueError("Admission requires exact unedited/base parity evidence")
    if not diagnostics.warm_preservation:
        raise ValueError("Admission requires warm-preservation evidence")
    numeric_warm = [
        float(value)
        for value in diagnostics.warm_preservation.values()
        if isinstance(value, (int, float, bool))
    ]
    if len(numeric_warm) != len(diagnostics.warm_preservation) or not bool(
        torch.isfinite(torch.tensor(numeric_warm, dtype=torch.float64)).all()
    ):
        raise ValueError("Warm-preservation evidence must be finite scalar diagnostics")
    return {
        "request_count": total_requests,
        "cache_hit_count": total_cache,
        "valid_z_count": total_valid,
        "failed_z_count": total_failed,
        "positions": sorted(diagnostics.per_position),
        "linear_system_count": len(diagnostics.linear_systems),
        "unedited_parity": True,
        "warm_preservation_fields": sorted(diagnostics.warm_preservation),
    }


def aggregate_updates(
    updates_by_position: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Add every position contribution targeting the same base parameter."""

    if not updates_by_position:
        raise ValueError("At least one position update is required")
    result: dict[str, torch.Tensor] = {}
    for position, bundle in sorted(updates_by_position.items()):
        if int(position) < 0 or not bundle:
            raise ValueError("Every lexical position needs a non-empty update bundle")
        for name, update in bundle.items():
            if update.ndim != 2 or not bool(torch.isfinite(update).all()):
                raise ValueError("Parameter updates must be finite matrices")
            if name in result and result[name].shape != update.shape:
                raise ValueError(f"Position updates do not align for parameter: {name}")
            result[name] = update.detach().clone() if name not in result else result[name] + update
    return result


def official_position_to_layer(
    positions: Sequence[int], pos2layer: Sequence[int] = (0, 1, 2, 3)
) -> dict[int, int]:
    """Apply the frozen official modulo position-to-layer routing rule."""

    normalized_positions = tuple(int(position) for position in positions)
    layers = tuple(int(layer) for layer in pos2layer)
    if not normalized_positions or len(normalized_positions) != len(set(normalized_positions)):
        raise ValueError("One-One lexical positions must be non-empty and unique")
    if not layers or any(layer < 0 for layer in layers) or any(
        position < 0 for position in normalized_positions
    ):
        raise ValueError("One-One positions and decoder layers must be non-negative")
    return {
        position: layers[position % len(layers)]
        for position in sorted(normalized_positions)
    }


def build_one_one_position_bundles(
    *,
    position_to_layer: Mapping[int, int],
    aggregated_updates: Mapping[str, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    """Route live positions to their shared, already-aggregated layer update.

    Multiple positions mapped to one decoder layer intentionally reference the
    same aggregate tensor.  Position-local pre-aggregation contributions must
    never be activated independently at generation time.
    """

    if not position_to_layer or not aggregated_updates:
        raise ValueError("One-One routing requires positions and aggregate updates")
    bundles: dict[int, dict[str, torch.Tensor]] = {}
    for position, layer in sorted(position_to_layer.items()):
        if int(position) < 0 or int(layer) < 0:
            raise ValueError("One-One positions and decoder layers must be non-negative")
        name = edited_parameter_name(int(layer))
        if name not in aggregated_updates:
            raise ValueError(f"Aggregate update is missing for One-One layer: {layer}")
        bundles[int(position)] = {name: aggregated_updates[name]}
    return bundles


def snapshot_base_parameters(
    model: nn.Module, parameter_names: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Snapshot only the frozen parameters addressed by an edit bundle."""

    names = tuple(str(name) for name in parameter_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Base snapshot parameter names must be non-empty and unique")
    parameters = dict(model.named_parameters())
    missing = [name for name in names if name not in parameters]
    if missing:
        raise ValueError(f"Base snapshot parameters are missing: {missing}")
    return {name: parameters[name].detach().cpu().clone() for name in names}


def assert_base_parameter_parity(
    model: nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    """Hard-fail unless every edited base parameter is bitwise restored."""

    if not snapshot:
        raise ValueError("Base parity snapshot is empty")
    parameters = dict(model.named_parameters())
    missing = [name for name in snapshot if name not in parameters]
    mismatched = [
        name
        for name, expected in snapshot.items()
        if name in parameters and not torch.equal(parameters[name].detach().cpu(), expected.cpu())
    ]
    if missing or mismatched:
        raise ValueError(
            f"Frozen base parameter parity failed: missing={missing}, mismatched={mismatched}"
        )
    return {"exact": True, "parameter_count": len(snapshot), "parameters": sorted(snapshot)}


class OneOneGenerationDeltaContext(_Stage15OneOneGenerationDeltaContext):
    """Faithful variable-path One-One trigger over the Stage15 GRAM hook base.

    The inherited context injects ``hidden @ deltaW.T`` only for live lexical
    rows at the current position.  It never mutates a base weight, removes all
    hooks on exit, and leaves EOS, padding, completed paths, and dead beams
    unedited.  ``assert_base_parameter_parity`` provides the explicit restore
    contract used by Stage16 admission.
    """


__all__ = [
    "AdmissionDiagnostics",
    "CacheProbeResult",
    "CachedZHit",
    "CachedZObservation",
    "FullTargetRequest",
    "LinearSystemDiagnostics",
    "OneOneGenerationDeltaContext",
    "PositionAdmissionDiagnostics",
    "PositionCovarianceResult",
    "ValidZSelection",
    "ZForwardBatch",
    "ZLifecycleUpdate",
    "ZOptimizationConfig",
    "ZOptimizationResult",
    "aggregate_updates",
    "assert_base_parameter_parity",
    "batch_full_target_requests",
    "build_one_one_position_bundles",
    "build_full_target_requests",
    "clip_delta_norm_",
    "collect_covariance",
    "extract_keys",
    "filter_valid_z",
    "linear_system_diagnostics",
    "optimize_z_vectors",
    "official_position_to_layer",
    "optimizer_satisfied",
    "probe_cached_z",
    "snapshot_base_parameters",
    "solve_weight_delta",
    "try_cache_hits",
    "update_z_lifecycle",
    "validate_admission_diagnostics",
]
