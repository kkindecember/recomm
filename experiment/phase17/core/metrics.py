"""Shared evaluator-side score and eligibility contracts."""

from __future__ import annotations

import math
from typing import Iterable


def reconstruct_sequence_log_score(token_log_scores: Iterable[float]) -> float:
    values = list(token_log_scores)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("sequence score requires non-empty finite token log scores")
    return math.fsum(values)


def assert_score_reconstruction(
    token_log_scores: Iterable[float], recorded_sequence_score: float, tolerance: float = 1e-6
) -> None:
    reconstructed = reconstruct_sequence_log_score(token_log_scores)
    if not math.isclose(reconstructed, recorded_sequence_score, abs_tol=tolerance, rel_tol=0.0):
        raise AssertionError(
            f"sequence score mismatch: reconstructed={reconstructed}, recorded={recorded_sequence_score}"
        )


def require_result_selection_eligible(metadata: dict) -> None:
    if metadata.get("result_selection_eligible") is not True:
        raise PermissionError("evaluator rejected a runtime/non-canonical result")
    if metadata.get("affects_scientific_result") is not True:
        raise PermissionError("artifact is explicitly excluded from scientific results")
    if metadata.get("test_read") or metadata.get("sports_read"):
        raise PermissionError("artifact crossed a sealed evaluation boundary")
