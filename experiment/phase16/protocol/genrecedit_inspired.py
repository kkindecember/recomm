#!/usr/bin/env python3
"""GenRecEdit-inspired numerical adaptations for GRAM.

These primitives are intentionally separate from ``genrecedit_faithful``.
G-RIDGE preserves the official z, covariance, key, aggregation, and trigger
semantics but replaces the singular no-ridge solve with a preregistered,
scale-relative Tikhonov term.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch


GRIDGE_METHOD_NAME = "G-RIDGE"
GRIDGE_SOLVE_VARIANT = "condition_targeted_spectral_ridge_v1"
GRIDGE_RIDGE_RULE = (
    "mu=(1+safety_margin)*max(max_abs_eigenvalue/target_condition,"
    "(max_eigenvalue-target_condition*min_eigenvalue)/(target_condition-1));"
    "solve(A+mu*I) in FP64"
)


@dataclass(frozen=True)
class RidgeSystemDiagnostics:
    ridge_value: float
    ridge_relative_to_spectral_scale: float
    target_condition: float
    safety_margin: float
    unregularized_min_eigenvalue: float
    unregularized_max_eigenvalue: float
    unregularized_max_abs_eigenvalue: float
    unregularized_rank: int
    unregularized_nullity: int
    unregularized_rank_tolerance: float
    unregularized_significant_negative_eigenvalues: int
    regularized_min_eigenvalue: float
    regularized_max_eigenvalue: float
    regularized_rank: int
    regularized_nullity: int
    regularized_rank_tolerance: float
    regularized_condition: float
    ridge_rule: str = GRIDGE_RIDGE_RULE

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_gridge_method_config(config: Mapping[str, Any]) -> dict[str, float | str | bool]:
    """Fail closed unless the inspired-method boundary is explicit and frozen."""

    method = config.get("method")
    if not isinstance(method, Mapping):
        raise ValueError("G-RIDGE config requires an explicit method contract")
    target = method.get("target_condition_number")
    safety = method.get("ridge_safety_margin")
    if (
        method.get("name") != GRIDGE_METHOD_NAME
        or method.get("family") != "GenRecEdit-inspired"
        or method.get("faithful_reproduction") is not False
        or method.get("solve_variant") != GRIDGE_SOLVE_VARIANT
        or method.get("ridge_selection_uses_validation_or_test") is not False
        or method.get("ridge_added") is not True
        or method.get("pseudoinverse_used") is not False
        or method.get("jitter_fallback_used") is not False
        or method.get("outcome_resampling_used") is not False
        or not isinstance(target, (int, float))
        or not math.isfinite(float(target))
        or float(target) <= 1.0
        or not isinstance(safety, (int, float))
        or not math.isfinite(float(safety))
        or float(safety) <= 0.0
        or method.get("ridge_rule") != GRIDGE_RIDGE_RULE
    ):
        raise ValueError("Malformed G-RIDGE method contract")
    return {
        "name": GRIDGE_METHOD_NAME,
        "family": "GenRecEdit-inspired",
        "faithful_reproduction": False,
        "solve_variant": GRIDGE_SOLVE_VARIANT,
        "target_condition_number": float(target),
        "ridge_safety_margin": float(safety),
        "ridge_rule": GRIDGE_RIDGE_RULE,
    }


def condition_targeted_ridge_value(
    *,
    min_eigenvalue: float,
    max_eigenvalue: float,
    max_abs_eigenvalue: float,
    target_condition: float,
    safety_margin: float,
) -> float:
    """Return a scale-relative ridge that makes the shifted spectrum positive."""

    values = (
        min_eigenvalue,
        max_eigenvalue,
        max_abs_eigenvalue,
        target_condition,
        safety_margin,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("G-RIDGE spectrum and controls must be finite")
    if (
        max_eigenvalue <= 0.0
        or max_abs_eigenvalue < max(abs(min_eigenvalue), abs(max_eigenvalue))
        or target_condition <= 1.0
        or safety_margin <= 0.0
    ):
        raise ValueError("G-RIDGE spectrum or controls are invalid")
    scale_floor = max_abs_eigenvalue / target_condition
    exact_condition_floor = (
        max_eigenvalue - target_condition * min_eigenvalue
    ) / (target_condition - 1.0)
    ridge = (1.0 + safety_margin) * max(scale_floor, exact_condition_floor)
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("G-RIDGE failed to construct a positive ridge")
    return ridge


def form_condition_targeted_ridge_system(
    *,
    system: torch.Tensor,
    eigenvalues: torch.Tensor | None,
    target_condition: float,
    safety_margin: float,
) -> tuple[torch.Tensor, RidgeSystemDiagnostics]:
    """Shift one symmetric FP64 normal system using the frozen G-RIDGE rule."""

    if (
        system.ndim != 2
        or system.shape[0] < 1
        or system.shape[0] != system.shape[1]
        or system.dtype != torch.float64
        or not bool(torch.isfinite(system).all())
        or not bool(torch.allclose(system, system.T, rtol=1e-10, atol=1e-10))
    ):
        raise ValueError("G-RIDGE requires one finite symmetric FP64 system")
    spectrum = torch.linalg.eigvalsh(system) if eigenvalues is None else eigenvalues
    if (
        spectrum.ndim != 1
        or spectrum.shape[0] != system.shape[0]
        or spectrum.device != system.device
        or not bool(torch.isfinite(spectrum).all())
    ):
        raise ValueError("G-RIDGE eigenvalues do not match the system")
    minimum = float(spectrum.min())
    maximum = float(spectrum.max())
    max_abs = float(spectrum.abs().max())
    ridge = condition_targeted_ridge_value(
        min_eigenvalue=minimum,
        max_eigenvalue=maximum,
        max_abs_eigenvalue=max_abs,
        target_condition=float(target_condition),
        safety_margin=float(safety_margin),
    )
    shifted_spectrum = spectrum + ridge
    regularized_min = float(shifted_spectrum.min())
    regularized_max = float(shifted_spectrum.max())
    if regularized_min <= 0.0:
        raise ValueError("G-RIDGE did not make the normal system positive definite")
    unregularized_tolerance = (
        system.shape[0] * torch.finfo(system.dtype).eps * spectrum.abs().max()
    )
    regularized_tolerance = (
        system.shape[0]
        * torch.finfo(system.dtype).eps
        * shifted_spectrum.abs().max()
    )
    unregularized_rank = int(
        (spectrum.abs() > unregularized_tolerance).sum().item()
    )
    regularized_rank = int(
        (shifted_spectrum.abs() > regularized_tolerance).sum().item()
    )
    condition = regularized_max / regularized_min
    if condition > float(target_condition) * (1.0 + 1e-9):
        raise ValueError("G-RIDGE target condition was not achieved")
    identity = torch.eye(system.shape[0], dtype=system.dtype, device=system.device)
    regularized = system + ridge * identity
    diagnostics = RidgeSystemDiagnostics(
        ridge_value=ridge,
        ridge_relative_to_spectral_scale=ridge / max_abs,
        target_condition=float(target_condition),
        safety_margin=float(safety_margin),
        unregularized_min_eigenvalue=minimum,
        unregularized_max_eigenvalue=maximum,
        unregularized_max_abs_eigenvalue=max_abs,
        unregularized_rank=unregularized_rank,
        unregularized_nullity=int(system.shape[0]) - unregularized_rank,
        unregularized_rank_tolerance=float(unregularized_tolerance),
        unregularized_significant_negative_eigenvalues=int(
            (spectrum < -unregularized_tolerance).sum().item()
        ),
        regularized_min_eigenvalue=regularized_min,
        regularized_max_eigenvalue=regularized_max,
        regularized_rank=regularized_rank,
        regularized_nullity=int(system.shape[0]) - regularized_rank,
        regularized_rank_tolerance=float(regularized_tolerance),
        regularized_condition=condition,
    )
    return regularized, diagnostics


def solve_condition_targeted_ridge_system(
    *, system: torch.Tensor, rhs: torch.Tensor, output_like: torch.Tensor
) -> torch.Tensor:
    """Solve an already regularized system in FP64 without pinv or fallback.

    The edit is cast to the live model dtype only by the generation hook at
    application time.  Casting here would make the residual diagnostic measure
    an FP32-quantized update instead of the preregistered FP64 ridge solve.
    """

    if system.ndim != 2 or rhs.ndim != 2 or output_like.ndim != 2:
        raise ValueError("G-RIDGE solve inputs must be matrices")
    if system.shape[0] < 1 or system.shape[0] != system.shape[1]:
        raise ValueError("G-RIDGE system must be non-empty and square")
    if rhs.shape[1] != system.shape[0] or rhs.shape[0] != output_like.shape[1]:
        raise ValueError("G-RIDGE solve shapes are invalid")
    if not bool(torch.isfinite(system).all()) or not bool(torch.isfinite(rhs).all()):
        raise ValueError("G-RIDGE solve inputs must be finite")
    try:
        return torch.linalg.solve(system.T, rhs.T).T
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise
        raise ValueError("G-RIDGE regularized solve failed without fallback") from error


__all__ = [
    "GRIDGE_METHOD_NAME",
    "GRIDGE_RIDGE_RULE",
    "GRIDGE_SOLVE_VARIANT",
    "RidgeSystemDiagnostics",
    "condition_targeted_ridge_value",
    "form_condition_targeted_ridge_system",
    "solve_condition_targeted_ridge_system",
    "validate_gridge_method_config",
]
