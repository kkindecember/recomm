from __future__ import annotations

import unittest

import torch

from experiment.phase16.protocol.genrecedit_inspired import (
    GRIDGE_RIDGE_RULE,
    condition_targeted_ridge_value,
    form_condition_targeted_ridge_system,
    solve_condition_targeted_ridge_system,
    validate_gridge_method_config,
)


def method_config() -> dict:
    return {
        "method": {
            "name": "G-RIDGE",
            "family": "GenRecEdit-inspired",
            "faithful_reproduction": False,
            "solve_variant": "condition_targeted_spectral_ridge_v1",
            "target_condition_number": 1_000_000.0,
            "ridge_safety_margin": 1e-6,
            "ridge_rule": GRIDGE_RIDGE_RULE,
            "ridge_selection_uses_validation_or_test": False,
            "ridge_added": True,
            "pseudoinverse_used": False,
            "jitter_fallback_used": False,
            "outcome_resampling_used": False,
        }
    }


class GenRecEditInspiredTests(unittest.TestCase):
    def test_singular_system_becomes_full_rank_with_target_condition(self) -> None:
        system = torch.diag(
            torch.tensor([10.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        )
        regularized, diagnostics = form_condition_targeted_ridge_system(
            system=system,
            eigenvalues=None,
            target_condition=1_000_000.0,
            safety_margin=1e-6,
        )
        self.assertEqual(diagnostics.unregularized_rank, 2)
        self.assertEqual(diagnostics.regularized_rank, 4)
        self.assertGreater(diagnostics.ridge_value, 0.0)
        self.assertLessEqual(diagnostics.regularized_condition, 1_000_000.0)
        self.assertTrue(torch.linalg.cholesky_ex(regularized).info.item() == 0)

    def test_small_negative_eigenvalue_is_shifted_not_relabelled(self) -> None:
        spectrum = torch.tensor([-1e-5, 0.0, 2.0], dtype=torch.float64)
        system = torch.diag(spectrum)
        regularized, diagnostics = form_condition_targeted_ridge_system(
            system=system,
            eigenvalues=spectrum,
            target_condition=1_000_000.0,
            safety_margin=1e-6,
        )
        self.assertEqual(
            diagnostics.unregularized_significant_negative_eigenvalues, 1
        )
        self.assertGreater(diagnostics.regularized_min_eigenvalue, 0.0)
        self.assertGreater(float(torch.linalg.eigvalsh(regularized).min()), 0.0)

    def test_ridge_formula_is_scale_equivariant(self) -> None:
        first = condition_targeted_ridge_value(
            min_eigenvalue=0.0,
            max_eigenvalue=10.0,
            max_abs_eigenvalue=10.0,
            target_condition=1_000_000.0,
            safety_margin=1e-6,
        )
        second = condition_targeted_ridge_value(
            min_eigenvalue=0.0,
            max_eigenvalue=100.0,
            max_abs_eigenvalue=100.0,
            target_condition=1_000_000.0,
            safety_margin=1e-6,
        )
        self.assertAlmostEqual(second / first, 10.0)

    def test_regularized_solve_has_small_residual_without_pinv(self) -> None:
        base = torch.diag(torch.tensor([3.0, 1.0, 0.0], dtype=torch.float64))
        system, _ = form_condition_targeted_ridge_system(
            system=base,
            eigenvalues=None,
            target_condition=1_000_000.0,
            safety_margin=1e-6,
        )
        rhs = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        output_like = torch.empty((1, 2), dtype=torch.float64)
        delta = solve_condition_targeted_ridge_system(
            system=system, rhs=rhs, output_like=output_like
        )
        relative = torch.linalg.vector_norm(delta @ system - rhs) / torch.linalg.vector_norm(rhs)
        self.assertLess(float(relative), 1e-9)

    def test_regularized_solve_stays_fp64_until_model_application(self) -> None:
        torch.manual_seed(1502)
        width = 64
        rotation, _ = torch.linalg.qr(
            torch.randn(width, width, dtype=torch.float64)
        )
        spectrum = torch.logspace(0, 6, width, dtype=torch.float64)
        system = rotation @ torch.diag(spectrum) @ rotation.T
        rhs = torch.randn(32, width, dtype=torch.float64)
        fp32_parameter_template = torch.empty((1, 32), dtype=torch.float32)
        delta = solve_condition_targeted_ridge_system(
            system=system,
            rhs=rhs,
            output_like=fp32_parameter_template,
        )
        fp64_residual = torch.linalg.vector_norm(delta @ system - rhs) / torch.linalg.vector_norm(rhs)
        premature_fp32_residual = (
            torch.linalg.vector_norm(delta.float().double() @ system - rhs)
            / torch.linalg.vector_norm(rhs)
        )
        self.assertEqual(delta.dtype, torch.float64)
        self.assertLess(float(fp64_residual), 1e-8)
        self.assertGreater(float(premature_fp32_residual), 1e-6)

    def test_method_contract_is_explicitly_nonfaithful_and_train_only(self) -> None:
        validated = validate_gridge_method_config(method_config())
        self.assertEqual(validated["name"], "G-RIDGE")
        self.assertFalse(validated["faithful_reproduction"])
        broken = method_config()
        broken["method"]["faithful_reproduction"] = True
        with self.assertRaises(ValueError):
            validate_gridge_method_config(broken)

    def test_invalid_or_nonsymmetric_system_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            form_condition_targeted_ridge_system(
                system=torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64),
                eigenvalues=None,
                target_condition=1_000_000.0,
                safety_margin=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
