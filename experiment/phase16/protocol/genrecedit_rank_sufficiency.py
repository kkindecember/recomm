#!/usr/bin/env python3
"""Train-only rank-sufficiency primitives for the Stage16 S16-3B diagnostic.

This module does not optimize z, solve for an edit, or add a numerical
fallback.  It evaluates a necessary condition for the faithful GenRecEdit
normal system: even the full train-only covariance plus the Gram matrix of
*all* train-only request keys must span the GRAM FFN input width.  The real
faithful system can only use a subset of those keys after valid-z filtering,
so a deficient all-request system is a strict structural impossibility proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch

from experiment.phase16.protocol.genrecedit_faithful import FullTargetRequest


RANK_TOLERANCE_RULE = "max(matrix_shape)*float64_eps*max_abs_eigenvalue"
STRUCTURAL_BLOCKED = "PROVEN_STRUCTURAL_RANK_BLOCKED"
VALID_Z_REQUIRED = "ALL_REQUEST_UPPER_BOUND_FULL_RANK_VALID_Z_DIAGNOSTIC_REQUIRED"


def _request_identity(request: FullTargetRequest) -> tuple[Any, ...]:
    return (
        request.cold_item,
        request.source_warm_item,
        request.context_items,
        request.full_target_path,
        request.prefix_token_ids,
        request.target_token_id,
        request.legal_token_ids,
        request.position,
    )


def deterministic_request_order(
    requests: Sequence[FullTargetRequest], *, seed: int
) -> tuple[FullTargetRequest, ...]:
    """Freeze one outcome-independent request prefix order for rank curves."""

    if not requests:
        raise ValueError("Rank diagnostic request universe is empty")
    identities = [_request_identity(request) for request in requests]
    if len(identities) != len(set(identities)):
        raise ValueError("Rank diagnostic request universe contains duplicate rows")

    def key(request: FullTargetRequest) -> tuple[bytes, tuple[Any, ...]]:
        identity = _request_identity(request)
        payload = repr((int(seed), "s16-3b-all-request-key", identity)).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).digest(), identity

    return tuple(sorted(requests, key=key))


def ordered_request_sha256(requests: Sequence[FullTargetRequest]) -> str:
    """Hash an already frozen ordered request universe without outcome fields."""

    if not requests:
        raise ValueError("Cannot hash an empty request universe")
    digest = hashlib.sha256()
    for request in requests:
        digest.update(repr(_request_identity(request)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def effective_checkpoints(
    configured: Sequence[int | str], *, total: int
) -> tuple[int, ...]:
    """Cap progressive checkpoints to a position's full request count."""

    if total < 1 or not configured:
        raise ValueError("Rank checkpoints require a positive full universe")
    raw: list[int] = []
    for value in configured:
        if value == "full":
            count = int(total)
        elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Rank checkpoints must be positive integers or 'full'")
        else:
            count = min(int(value), int(total))
        raw.append(count)
    values = tuple(sorted(set(raw + [int(total)])))
    if values[-1] != int(total):  # pragma: no cover - appended above
        raise RuntimeError("Effective rank checkpoints lost the full universe")
    return values


@dataclass(frozen=True)
class SymmetricRankDiagnostics:
    width: int
    rank: int
    nullity: int
    tolerance: float
    min_eigenvalue: float
    max_eigenvalue: float
    min_abs_eigenvalue: float
    max_abs_eigenvalue: float
    condition: float
    significant_negative_eigenvalues: int
    tolerance_rule: str = RANK_TOLERANCE_RULE

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "width": self.width,
            "rank": self.rank,
            "nullity": self.nullity,
            "tolerance": self.tolerance,
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "min_abs_eigenvalue": self.min_abs_eigenvalue,
            "max_abs_eigenvalue": self.max_abs_eigenvalue,
            "condition": self.condition,
            "significant_negative_eigenvalues": self.significant_negative_eigenvalues,
            "tolerance_rule": self.tolerance_rule,
        }


def symmetric_rank_diagnostics(matrix: torch.Tensor) -> SymmetricRankDiagnostics:
    """Apply the frozen A4 FP64 symmetric-eigenvalue numerical-rank rule."""

    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Rank diagnostics require a non-empty square matrix")
    values = matrix.double()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Rank diagnostics require a finite matrix")
    if not bool(torch.allclose(values, values.T, rtol=1e-10, atol=1e-10)):
        raise ValueError("Rank diagnostics require a symmetric matrix")
    eigenvalues = torch.linalg.eigvalsh(values)
    absolute = eigenvalues.abs()
    maximum = absolute.max()
    tolerance_tensor = (
        max(values.shape) * torch.finfo(values.dtype).eps * maximum
    )
    tolerance = float(tolerance_tensor)
    rank = int((absolute > tolerance_tensor).sum().item())
    min_abs = float(absolute.min())
    max_abs = float(maximum)
    condition = math.inf if min_abs == 0.0 else max_abs / min_abs
    return SymmetricRankDiagnostics(
        width=int(values.shape[0]),
        rank=rank,
        nullity=int(values.shape[0]) - rank,
        tolerance=tolerance,
        min_eigenvalue=float(eigenvalues.min()),
        max_eigenvalue=float(eigenvalues.max()),
        min_abs_eigenvalue=min_abs,
        max_abs_eigenvalue=max_abs,
        condition=condition,
        significant_negative_eigenvalues=int(
            (eigenvalues < -tolerance_tensor).sum().item()
        ),
    )


class StreamingKeyGram:
    """Accumulate K^T K in FP64 without retaining the full key bank."""

    def __init__(self, width: int, *, device: torch.device | str) -> None:
        if width < 1:
            raise ValueError("Key-Gram width must be positive")
        self.width = int(width)
        self.count = 0
        self.gram = torch.zeros(
            self.width, self.width, dtype=torch.float64, device=device
        )

    def update(self, keys: torch.Tensor) -> None:
        if keys.ndim != 2 or keys.shape[0] < 1 or keys.shape[1] != self.width:
            raise ValueError("Key-Gram update has the wrong shape")
        if not bool(torch.isfinite(keys).all()):
            raise ValueError("Key-Gram update contains a non-finite value")
        values = keys.detach().to(device=self.gram.device, dtype=torch.float64)
        self.gram.addmm_(values.T, values)
        self.count += int(values.shape[0])


def classify_all_request_upper_bound(
    positions: Mapping[int | str, Mapping[str, Any]], *, width: int
) -> dict[str, Any]:
    """Classify a complete all-request upper-bound diagnostic.

    A deficient all-request system proves every valid-z subset deficient.  A
    full-rank all-request system is only a necessary-condition pass; valid-z
    sufficiency remains untested and the S16-3 faithful Gate stays closed.
    """

    normalized = {int(position): row for position, row in positions.items()}
    if set(normalized) != set(range(6)) or width < 1:
        raise ValueError("Rank classification requires complete positions 0--5")
    deficient: list[int] = []
    final_ranks: dict[str, int] = {}
    for position, row in sorted(normalized.items()):
        if (
            row.get("full_covariance_universe_processed") is not True
            or row.get("full_request_key_universe_processed") is not True
            or row.get("all_request_key_superset") is not True
        ):
            raise ValueError("Rank classification requires complete full-universe evidence")
        rank = int(row["final_system_rank"])
        if rank < 0 or rank > width:
            raise ValueError("Rank classification contains an invalid system rank")
        final_ranks[str(position)] = rank
        if rank < width:
            deficient.append(position)
    classification = STRUCTURAL_BLOCKED if deficient else VALID_Z_REQUIRED
    return {
        "classification": classification,
        "linear_system_width": int(width),
        "final_system_rank_by_position": final_ranks,
        "structurally_blocked_positions": deficient,
        "faithful_gate_promoted": False,
        "logical_basis": (
            "For positive-semidefinite C and key Gram matrices, the faithful "
            "valid-z system uses a key subset of the measured all-request Gram. "
            "If lambda*C_full + K_all^T*K_all is rank-deficient, no valid-z subset "
            "can remove its remaining nullspace. Full rank of the superset is not "
            "sufficient to prove the valid-z system invertible."
        ),
    }


__all__ = [
    "RANK_TOLERANCE_RULE",
    "STRUCTURAL_BLOCKED",
    "StreamingKeyGram",
    "SymmetricRankDiagnostics",
    "VALID_Z_REQUIRED",
    "classify_all_request_upper_bound",
    "deterministic_request_order",
    "effective_checkpoints",
    "ordered_request_sha256",
    "symmetric_rank_diagnostics",
]
