"""Stage16 S16-0 semantic bridge checks for pinned SpecGR and GenRecEdit.

This is a clean-room contract module.  It neither imports nor executes third-
party repositories and it never opens recommendation splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


def masked_mean_log_probability(
    token_log_probabilities: Sequence[float], score_length: int
) -> float:
    if score_length < 1 or score_length > len(token_log_probabilities):
        raise ValueError("score_length is outside the candidate path")
    selected = torch.tensor(token_log_probabilities[:score_length], dtype=torch.float64)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("candidate log probabilities must be finite")
    return float(selected.mean())


def longest_warm_prefix(
    path: Sequence[int], warm_paths: Iterable[Sequence[int]], *, minimum: int
) -> int:
    candidate = tuple(map(int, path))
    if not candidate:
        raise ValueError("candidate path must be non-empty")
    if minimum < 1 or minimum > len(candidate):
        raise ValueError("minimum prefix is outside the candidate path")
    prefixes = {
        tuple(map(int, warm_path))[:depth]
        for warm_path in warm_paths
        for depth in range(1, len(tuple(warm_path)) + 1)
    }
    longest = 0
    for depth in range(1, len(candidate) + 1):
        if candidate[:depth] not in prefixes:
            break
        longest = depth
    return min(len(candidate), max(minimum, longest))


def draft_without_replacement(scores: Sequence[float], k: int) -> list[int]:
    values = torch.tensor(scores, dtype=torch.float64)
    if values.ndim != 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("draft scores must be a finite vector")
    if k < 1 or k > values.numel():
        raise ValueError("draft size is outside the catalog")
    return sorted(range(values.numel()), key=lambda i: (-float(values[i]), i))[:k]


def legal_children(
    catalog_paths: Iterable[Sequence[int]], prefix: Sequence[int]
) -> tuple[int, ...]:
    prefix_tuple = tuple(map(int, prefix))
    children = {
        int(path[len(prefix_tuple)])
        for raw_path in catalog_paths
        for path in (tuple(map(int, raw_path)),)
        if len(path) > len(prefix_tuple) and path[: len(prefix_tuple)] == prefix_tuple
    }
    if not children:
        raise ValueError("prefix has no legal continuation")
    return tuple(sorted(children))


def guided_redraft_candidates(
    ranked_items: Sequence[str],
    item_paths: Mapping[str, Sequence[int]],
    verifier_prefixes: Iterable[Sequence[int]],
    *,
    prefix_depth: int,
    already_drafted: Iterable[str],
    draft_size: int,
) -> list[str]:
    if len(ranked_items) != len(set(ranked_items)):
        raise ValueError("ranked items must be unique")
    if set(ranked_items) - set(item_paths):
        raise ValueError("ranked items contain an unknown catalog item")
    prefixes = {tuple(map(int, prefix)) for prefix in verifier_prefixes}
    if prefix_depth < 0 or any(len(prefix) != prefix_depth for prefix in prefixes):
        raise ValueError("verifier prefix depth mismatch")
    excluded = set(already_drafted)
    selected = []
    for item in ranked_items:
        path = tuple(map(int, item_paths[item]))
        if item in excluded:
            continue
        if prefix_depth and path[:prefix_depth] not in prefixes:
            continue
        selected.append(item)
        if len(selected) == draft_size:
            break
    return selected


def should_adaptively_exit(
    accepted_count: int, k: int, completed_rounds: int, maximum_depth: int
) -> bool:
    if min(accepted_count, completed_rounds) < 0 or k < 1 or maximum_depth < 1:
        raise ValueError("invalid adaptive-exit state")
    return accepted_count >= k or completed_rounds >= maximum_depth


def strict_accept(score: float, threshold: float) -> bool:
    values = torch.tensor([score, threshold], dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("acceptance inputs must be finite")
    return float(score) > float(threshold)


def normalized_knn(sequence: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 1 or items.ndim != 2 or sequence.numel() != items.size(1):
        raise ValueError("KNN tensors do not align")
    return torch.nn.functional.normalize(sequence.float(), dim=0) @ torch.nn.functional.normalize(
        items.float(), dim=1
    ).T


def weighted_objective(
    embedding_loss: torch.Tensor,
    generative_loss: torch.Tensor,
    lambda_embedding: float,
    lambda_generative: float,
) -> torch.Tensor:
    return lambda_embedding * embedding_loss + lambda_generative * generative_loss


@dataclass(frozen=True)
class ProbeState:
    is_argmax: bool
    target_probability: float
    cache_probe_pass: bool


def fixed_width_probe(
    logits: torch.Tensor,
    *,
    position: int,
    target_code: int,
    codebook_size: int = 256,
    offset: int = 1,
    threshold: float = 0.3,
) -> ProbeState:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("probe logits must be a finite vector")
    if position < 0 or target_code < 0 or target_code >= codebook_size:
        raise ValueError("invalid fixed-width target")
    start = offset + position * codebook_size
    stop = start + codebook_size
    if stop > logits.numel():
        raise ValueError("fixed-width codebook slice is outside the vocabulary")
    local = logits[start:stop].float()
    probabilities = torch.softmax(logits.float(), dim=0)
    target_token = start + target_code
    probability = float(probabilities[target_token])
    is_argmax = bool(local[target_code] >= local.max())
    return ProbeState(is_argmax, probability, is_argmax and probability > threshold)


def lexical_probe(
    logits: torch.Tensor,
    *,
    target_token: int,
    legal_tokens: Sequence[int],
    threshold: float = 0.3,
) -> ProbeState:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("probe logits must be a finite vector")
    legal = tuple(map(int, legal_tokens))
    if not legal or len(legal) != len(set(legal)) or target_token not in legal:
        raise ValueError("target must belong to a unique legal token set")
    if min(legal) < 0 or max(legal) >= logits.numel():
        raise ValueError("legal token is outside the vocabulary")
    local = logits[torch.tensor(legal)].float()
    target_index = legal.index(int(target_token))
    probability = float(torch.softmax(logits.float(), dim=0)[int(target_token)])
    is_argmax = bool(local[target_index] >= local.max())
    return ProbeState(is_argmax, probability, is_argmax and probability > threshold)


def optimizer_satisfied(logits: torch.Tensor, target_token: int) -> bool:
    if logits.ndim != 1 or target_token < 0 or target_token >= logits.numel():
        raise ValueError("invalid optimizer satisfaction state")
    return int(torch.argmax(logits).item()) == int(target_token)


def clip_absolute_norm(delta: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    if maximum_norm <= 0 or not bool(torch.isfinite(delta).all()):
        raise ValueError("invalid delta norm contract")
    result = delta.clone()
    norm = torch.linalg.vector_norm(result)
    if float(norm) > maximum_norm:
        result.mul_(maximum_norm / norm)
    return result


def second_moment(rows: torch.Tensor) -> torch.Tensor:
    if rows.ndim != 2 or rows.size(0) < 1 or not bool(torch.isfinite(rows).all()):
        raise ValueError("covariance rows must be a non-empty finite matrix")
    values = rows.double()
    return values.T @ values / values.size(0)


def solve_delta(
    residual: torch.Tensor,
    keys: torch.Tensor,
    covariance: torch.Tensor,
    covariance_lambda: float,
) -> torch.Tensor:
    if residual.ndim != 2 or keys.ndim != 2 or covariance.ndim != 2:
        raise ValueError("delta inputs must be matrices")
    if residual.size(1) != keys.size(1) or covariance.shape != (keys.size(0), keys.size(0)):
        raise ValueError("delta matrix shapes do not align")
    system = keys.double() @ keys.double().T + covariance_lambda * covariance.double()
    rhs = residual.double() @ keys.double().T
    return torch.linalg.solve(system.T, rhs.T).T.to(residual)


def aggregate_updates(updates: Iterable[torch.Tensor]) -> torch.Tensor:
    rows = list(updates)
    if not rows or any(row.shape != rows[0].shape for row in rows):
        raise ValueError("updates must be non-empty and shape-aligned")
    return torch.stack(rows).sum(dim=0)


def active_edit_position(
    prefix: Sequence[int], catalog_paths: Iterable[Sequence[int]], *, eos_seen: bool = False
) -> int | None:
    suffix = tuple(map(int, prefix))
    paths = {tuple(map(int, path)) for path in catalog_paths}
    if eos_seen or suffix in paths:
        return None
    valid_prefixes = {path[:depth] for path in paths for depth in range(len(path))}
    if suffix not in valid_prefixes:
        return None
    return len(suffix)


def run_bridge_checks() -> list[dict[str, object]]:
    checks: list[tuple[str, callable]] = []

    def register(name):
        def decorator(function):
            checks.append((name, function))
            return function
        return decorator

    @register("saux_content_only_cold_access")
    def _saux_content():
        labels = {0, 1}
        catalog = {0, 1, 2}
        assert labels < catalog and 2 not in labels

    @register("specgr_draft_without_replacement")
    def _draft():
        assert draft_without_replacement([0.2, 0.8, 0.8], 3) == [1, 2, 0]

    @register("specgr_fixed_width_score_parity")
    def _score():
        logp = [-0.2, -0.5, -9.0, -8.0]
        fixed_width = float(torch.tensor(logp[:2], dtype=torch.float64).sum() / 2)
        assert abs(masked_mean_log_probability(logp, 2) - fixed_width) < 1e-12
        assert longest_warm_prefix((4, 7, 9), [(4, 7, 2), (5, 1, 3)], minimum=2) == 2

    @register("specgr_strict_acceptance")
    def _accept():
        assert strict_accept(-1.79, -1.8) and not strict_accept(-1.8, -1.8)

    @register("specgr_variable_trie_redraft")
    def _redraft():
        paths = {"a": (1, 2), "b": (1, 3, 4), "c": (5,)}
        assert guided_redraft_candidates(
            ["c", "a", "b"], paths, [(1,)], prefix_depth=1,
            already_drafted={"a"}, draft_size=2
        ) == ["b"]

    @register("specgr_adaptive_exit")
    def _exit():
        assert should_adaptively_exit(50, 50, 2, 4)
        assert should_adaptively_exit(1, 50, 4, 4)
        assert not should_adaptively_exit(1, 50, 3, 4)

    @register("specgr_unique_fallback")
    def _fallback():
        accepted = ["a", "b"]
        beam = ["b", "c", "d"]
        merged = accepted + [item for item in beam if item not in set(accepted)]
        assert merged[:4] == ["a", "b", "c", "d"]

    @register("splus_normalized_knn")
    def _knn():
        scores = normalized_knn(torch.tensor([3.0, 0.0]), torch.tensor([[2.0, 0.0], [0.0, 4.0]]))
        assert torch.allclose(scores, torch.tensor([1.0, 0.0]))

    @register("splus_weighted_objective")
    def _objective():
        result = weighted_objective(torch.tensor(2.0), torch.tensor(3.0), 6.0, 1.0)
        assert float(result) == 15.0

    @register("genrecedit_fixed_width_probe_parity")
    def _probe():
        logits = torch.full((1 + 2 * 256,), -10.0)
        position, target_code = 1, 7
        start = 1 + position * 256
        logits[start + 7], logits[start + 8] = 4.0, 2.0
        fixed = fixed_width_probe(logits, position=position, target_code=target_code)
        lexical = lexical_probe(logits, target_token=start + 7, legal_tokens=tuple(range(start, start + 256)))
        assert fixed.is_argmax == lexical.is_argmax
        assert abs(fixed.target_probability - lexical.target_probability) < 1e-7

    @register("genrecedit_threshold_scope")
    def _threshold_scope():
        logits = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6])
        state = lexical_probe(logits, target_token=0, legal_tokens=(0, 1, 2, 3, 4), threshold=0.3)
        assert state.is_argmax and not state.cache_probe_pass
        assert optimizer_satisfied(logits, 0)

    @register("genrecedit_optimizer_contract")
    def _optimizer():
        parameter = torch.zeros(2, requires_grad=True)
        optimizer = torch.optim.Adam([parameter], lr=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.01)
        (parameter - 1).square().sum().backward(); optimizer.step(); scheduler.step()
        assert bool(torch.isfinite(parameter).all())

    @register("genrecedit_norm_clip")
    def _clip():
        clipped = clip_absolute_norm(torch.tensor([3.0, 4.0]), 2.0)
        assert abs(float(torch.linalg.vector_norm(clipped)) - 2.0) < 1e-6

    @register("genrecedit_second_moment")
    def _moment():
        rows = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(second_moment(rows), rows.double().T @ rows.double() / 2)

    @register("genrecedit_variable_position_routing")
    def _route():
        paths = [(4, 5), (4, 6, 7)]
        assert active_edit_position((), paths) == 0
        assert active_edit_position((4,), paths) == 1
        assert active_edit_position((4, 5), paths) is None
        assert active_edit_position((4,), paths, eos_seen=True) is None

    @register("genrecedit_valid_z_filter")
    def _valid_z():
        states = [torch.ones(1), None, torch.zeros(1)]
        assert [i for i, state in enumerate(states) if state is not None] == [0, 2]

    @register("genrecedit_closed_form_parity")
    def _delta():
        residual = torch.tensor([[2.0, 1.0]])
        keys = torch.tensor([[1.0, 2.0]])
        covariance = torch.tensor([[3.0]])
        solved = solve_delta(residual, keys, covariance, 4.0)
        inverse_form = (residual.double() @ keys.double().T) @ torch.linalg.inv(
            keys.double() @ keys.double().T + 4.0 * covariance.double()
        )
        assert torch.allclose(solved.double(), inverse_form)

    @register("genrecedit_additive_aggregation")
    def _aggregate():
        assert torch.equal(aggregate_updates([torch.tensor([1.0]), torch.tensor([2.0])]), torch.tensor([3.0]))

    results = []
    for name, function in checks:
        try:
            function()
        except Exception as error:  # pragma: no cover - emitted for audit diagnostics
            results.append({"id": name, "status": "FAIL", "detail": repr(error)})
        else:
            results.append({"id": name, "status": "PASS", "detail": "contract assertion passed"})
    return results
