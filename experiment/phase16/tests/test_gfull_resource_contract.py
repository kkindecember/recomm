from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiment.phase16.protocol.finalize_s3_gfull_resource_sweep import (
    EXECUTED_CODE_PATHS as FINALIZER_EXECUTED_CODE_PATHS,
    GRIDGE_REQUIRED_CONTRACT_KEYS,
    REQUIRED_CANDIDATE_SEMANTIC_KEYS,
    expected_selected_microbatch,
    observed_full_30_step_path,
    recompute_projection,
)
from experiment.phase16.protocol.gfull_objective_resource_sweep import (
    EXECUTED_CODE_PATHS as WORKER_EXECUTED_CODE_PATHS,
    choose_candidate,
    covariance_convergence_diagnostics,
    independent_full_lifecycle_probe,
    solve_status_label,
)
from experiment.phase16.protocol.genrecedit_faithful import FullTargetRequest
from experiment.phase16.protocol.genrecedit_inspired import (
    GRIDGE_RIDGE_RULE,
    validate_gridge_method_config,
)


def candidate(microbatch: int, throughput: float, peak: float = 4000.0) -> dict:
    semantics = {key: True for key in REQUIRED_CANDIDATE_SEMANTIC_KEYS}
    return {
        "microbatch": microbatch,
        "steady_request_steps_per_second": throughput,
        "peak_reserved_mib": peak,
        "eligible": peak <= 8192,
        "semantic_checks": semantics,
        "candidate_request_count": 1,
        "candidate_requests_by_position": {"0": 1},
        "batch_records": [
            {
                "position": 0,
                "batch_index": 0,
                "request_count": 1,
                "first_ten_objective_step_seconds": [1.0 / throughput] * 10,
            }
        ],
    }


class ResourceContractTests(unittest.TestCase):
    def test_solve_status_distinguishes_valid_z_from_solved_updates(self) -> None:
        self.assertEqual(
            solve_status_label({"0": {"valid_z_count": 0}}, {}),
            "NO_VALID_Z_IN_PREREGISTERED_RESOURCE_SUBSET",
        )
        self.assertEqual(
            solve_status_label({"0": {"valid_z_count": 3}}, {}),
            "VALID_Z_PRESENT_BUT_NO_POSITION_SOLVE_COMPLETED",
        )
        self.assertEqual(
            solve_status_label(
                {"0": {"valid_z_count": 3}}, {"weight": torch.ones(1)}
            ),
            "SOLVE_AND_AGGREGATE_EXERCISED",
        )

    def test_execution_identity_covers_transitive_imports(self) -> None:
        self.assertEqual(WORKER_EXECUTED_CODE_PATHS, FINALIZER_EXECUTED_CODE_PATHS)
        self.assertIn(
            "experiment/phase16/protocol/official_specgr_runtime.py",
            WORKER_EXECUTED_CODE_PATHS,
        )
        self.assertIn(
            "experiment/phase16/protocol/specgr_faithful.py",
            WORKER_EXECUTED_CODE_PATHS,
        )

    def test_projection_is_recomputed_from_raw_measurements(self) -> None:
        config = {
            "frozen_workload": {"z_steps": 30},
            "sweep": {
                "covariance_rows": 64,
                "resource_covariance_convergence_checkpoints_by_position": {
                    str(position): [1, 2] for position in range(6)
                },
                "formal_covariance_convergence_checkpoints": [2, "full"],
                "formal_item_disjoint_admission_events": 7435,
                "formal_warm_preservation_events": 512,
                "generation_resource_events": 2,
                "key_extraction_batch_policy": "selected_z_microbatch",
                "key_extraction_layer_policy": "position_selected_layer_only_output_equivalent_to_unused_official_key_bank_elision",
            },
        }
        positions = {
            str(position): {
                "request_count": 16,
                "valid_z_count": 8,
                "z_objective_step_seconds": 1.0,
                "final_z_reprobe_seconds": 0.1,
                "post_z_filter_rank_diagnostics_seconds": 0.04,
                "key_extraction_seconds": 0.02,
                "system_fixed_setup_seconds": 0.004,
                "valid_z_matrix_products_seconds": 0.006,
                "system_formation_seconds": 0.01,
                "solve_factorization_diagnostics_seconds": 0.02,
                "solve_diagnostic_seconds": 0.03,
                "solve_completed": True,
                "key_extraction_batch_size": 8,
                "key_extraction_layer": position % 4,
            }
            for position in range(6)
        }
        raw = {
            "position_diagnostics": positions,
            "full_universe": {
                "request_counts_by_position": {str(position): 100 for position in range(6)},
                "covariance_rows": 27659,
                "covariance_counts_by_position": {
                    str(position): 10 for position in range(6)
                },
            },
            "candidates": [
                {
                    "microbatch": 8,
                    "steady_request_steps_per_second": 10.0,
                    "candidate_request_count": 1,
                    "candidate_requests_by_position": {"0": 1},
                    "batch_records": [
                        {
                            "position": 0,
                            "batch_index": 0,
                            "request_count": 1,
                            "first_ten_objective_step_seconds": [0.1] * 10,
                        }
                    ],
                }
            ],
            "selected_request_microbatch": 8,
            "projection_measurements": {
                "context_build_seconds": 3.0,
                "trigger_contract_seconds": 4.0,
                "repeated_z_step_seconds": 6.0,
                "final_z_probe_seconds": 0.6,
                "post_z_filter_rank_diagnostics_seconds": 0.24,
                "key_extraction_seconds": 0.12,
                "system_fixed_setup_seconds": 0.024,
                "valid_z_matrix_products_seconds": 0.036,
                "system_formation_seconds": 0.06,
                "solve_factorization_diagnostics_seconds": 0.12,
                "solve_diagnostic_seconds": 0.18,
            },
            "covariance_resource": {
                "elapsed_seconds": 2.0,
                "elapsed_seconds_by_position": {
                    str(position): 1.0 / 3.0 for position in range(6)
                },
                "rows_by_position": {str(position): 2 for position in range(6)},
                "convergence_elapsed_seconds": 0.5,
                "resource_convergence_row_equivalents": 18,
                "formal_convergence_row_equivalents": 72,
            },
            "generation_resource_probe": {
                "events": 2,
                "base_elapsed_seconds": 2.0,
                "edited_elapsed_seconds": 4.0,
                "base_seconds_per_event": 1.0,
                "edited_seconds_per_event": 2.0,
                "base_plus_edited_seconds_per_event": 3.0,
            },
            "position_contract_seconds": 10.0,
            "formal_projection": {
                "key_extraction_batch_policy": "selected_z_microbatch",
                "key_extraction_layer_policy": "position_selected_layer_only_output_equivalent_to_unused_official_key_bank_elision",
                "key_extraction_batch_size": 8,
            },
        }
        components, basis = recompute_projection(config, raw)
        self.assertAlmostEqual(
            components["full_final_z_reprobe_diagnostics"], 3.75
        )
        self.assertAlmostEqual(
            components["projected_valid_z_matrix_products"], 0.225
        )
        self.assertAlmostEqual(components["six_position_system_fixed_setup"], 0.024)
        self.assertAlmostEqual(
            components["full_post_z_filter_and_rank_diagnostics"], 1.5
        )
        self.assertAlmostEqual(components["full_request_key_extraction"], 0.75)
        self.assertEqual(basis["resource_request_count"], 96.0)
        self.assertEqual(
            basis["projected_valid_z_count_by_position"],
            {str(position): 50.0 for position in range(6)},
        )
        raw["projection_measurements"]["final_z_probe_seconds"] = 0.5
        with self.assertRaises(ValueError):
            recompute_projection(config, raw)

    def test_runner_declares_sub_ten_minute_component_bound(self) -> None:
        root = Path(__file__).resolve().parents[3]
        runner = (root / "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep.sh").read_text()
        self.assertIn("HARD_TIMEOUT=${S16_S3_HARD_TIMEOUT:-360}", runner)
        self.assertNotIn("HARD_TIMEOUT=420", runner)
        self.assertIn("--capture-identity-only", runner)
        self.assertIn("timeout --signal=TERM --kill-after=2 5 nvidia-smi", runner)
        self.assertIn(
            "-m experiment.phase16.protocol.gfull_objective_resource_sweep",
            runner,
        )
        self.assertIn('--worker-hard-timeout-seconds "$HARD_TIMEOUT"', runner)
        self.assertIn("terminal_status TIMEOUT RESOURCE_BLOCKED_BOUNDED_TIMEOUT", runner)
        self.assertIn("terminal_status BLOCKED GPU_ADMISSION_FAILED", runner)
        self.assertIn(
            'terminal_status COMPLETED "$SUCCESS_CODE" "$SUCCESS_REASON"', runner
        )

    def test_a4_changes_only_resource_attempt_contract_and_is_gpu4_background(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config_root = root / "experiment/phase16/configs"
        a3 = json.loads(
            (config_root / "stage16_s3_gfull_objective_resource_sweep_a3.json").read_text()
        )
        a4 = json.loads(
            (config_root / "stage16_s3_gfull_objective_resource_sweep_a4_gpu4.json").read_text()
        )
        for row in (a3, a4):
            for key in ("attempt_id", "output_dir", "exact_start_command"):
                row.pop(key)
        a4["sweep"].pop("maximum_candidate_peak_reserved_mib")
        a4["sweep"].pop("maximum_resource_peak_reserved_mib")
        a4_resources = a4.pop("resources")
        a3_resources = a3.pop("resources")
        self.assertEqual(a3, a4)
        self.assertEqual(a4_resources["fixed_physical_gpu"], 4)
        self.assertEqual(a4_resources["minimum_free_mib"], 18432)
        self.assertEqual(a4_resources["expected_peak_mib"], 12288)
        self.assertEqual(a4_resources["hard_timeout_seconds"], 900)
        self.assertEqual(a3_resources["minimum_free_mib"], 12288)
        expected_a4_resources = dict(a3_resources)
        expected_a4_resources.update(
            {
                "fixed_physical_gpu": 4,
                "minimum_free_mib": 18432,
                "expected_peak_mib": 12288,
                "hard_timeout_seconds": 900,
                "selection_rule": "use only physical GPU4 if it meets the frozen 18432 MiB admission; never auto-switch",
            }
        )
        self.assertEqual(a4_resources, expected_a4_resources)
        launcher = (
            root
            / "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a4_gpu4.sh"
        ).read_text()
        inner = (
            root
            / "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a4_gpu4_inner.sh"
        ).read_text()
        self.assertIn("tmux new-session -d", launcher)
        self.assertIn("S16_S3_FIXED_GPU=4", inner)
        self.assertIn("S16_S3_MINIMUM_FREE=18432", inner)
        self.assertIn("S16_S3_EXPECTED_PEAK=12288", inner)
        self.assertIn("S16_S3_HARD_TIMEOUT=900", inner)
        generic = (
            root / "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep.sh"
        ).read_text()
        self.assertIn('--expected-peak-mib "$EXPECTED_PEAK"', generic)

    def test_gridge_r1_is_distinct_nonfaithful_method_with_frozen_evidence(self) -> None:
        import hashlib

        root = Path(__file__).resolve().parents[3]
        config = json.loads(
            (
                root
                / "experiment/phase16/configs/stage16_s3r_gridge_resource_sweep_r1_gpu4.json"
            ).read_text()
        )
        method = validate_gridge_method_config(config)
        self.assertEqual(method["name"], "G-RIDGE")
        self.assertFalse(method["faithful_reproduction"])
        self.assertEqual(method["ridge_rule"], GRIDGE_RIDGE_RULE)
        self.assertEqual(method["target_condition_number"], 1_000_000.0)
        self.assertIn(
            "inspired_ridge_solve_completed_for_every_valid_position",
            GRIDGE_REQUIRED_CONTRACT_KEYS,
        )
        self.assertNotIn(
            "faithful_solve_completed_for_every_valid_position",
            GRIDGE_REQUIRED_CONTRACT_KEYS,
        )
        for label in (
            "faithful_a4_raw",
            "faithful_a4_status",
            "s3b_recovery_adjudication",
            "s3b_recovery_status",
        ):
            spec = config["inputs"][label]
            observed = hashlib.sha256((root / spec["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, spec["sha256"])

    def test_gridge_r1_changes_only_method_evidence_and_attempt_identity(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config_root = root / "experiment/phase16/configs"
        faithful = json.loads(
            (config_root / "stage16_s3_gfull_objective_resource_sweep_a4_gpu4.json").read_text()
        )
        inspired = json.loads(
            (config_root / "stage16_s3r_gridge_resource_sweep_r1_gpu4.json").read_text()
        )
        for row in (faithful, inspired):
            for key in (
                "schema_version",
                "experiment_id",
                "attempt_id",
                "output_dir",
                "exact_start_command",
            ):
                row.pop(key)
        method = inspired.pop("method")
        self.assertEqual(method["name"], "G-RIDGE")
        for label in (
            "faithful_a4_raw",
            "faithful_a4_status",
            "s3b_recovery_adjudication",
            "s3b_recovery_status",
        ):
            inspired["inputs"].pop(label)
        self.assertEqual(faithful, inspired)

    def test_gridge_gpu4_runner_is_isolated_and_no_auto_retry(self) -> None:
        root = Path(__file__).resolve().parents[3]
        launcher = (
            root
            / "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu4.sh"
        ).read_text()
        inner = (
            root
            / "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu4_inner.sh"
        ).read_text()
        self.assertIn("phase16_s3r_gridge_resource_r1_gpu4", launcher)
        self.assertIn("tmux new-session -d", launcher)
        self.assertIn("retries require a new attempt", launcher)
        self.assertIn("S16_S3_FIXED_GPU=4", inner)
        self.assertIn("S16_S3_HARD_TIMEOUT=900", inner)
        self.assertIn("PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP", inner)

    def test_gridge_gpu5_attempt_changes_only_execution_identity(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config_root = root / "experiment/phase16/configs"
        gpu4 = json.loads(
            (config_root / "stage16_s3r_gridge_resource_sweep_r1_gpu4.json").read_text()
        )
        gpu5 = json.loads(
            (config_root / "stage16_s3r_gridge_resource_sweep_r1_gpu5.json").read_text()
        )
        for row in (gpu4, gpu5):
            for key in ("attempt_id", "output_dir", "exact_start_command"):
                row.pop(key)
        gpu4_resources = gpu4.pop("resources")
        gpu5_resources = gpu5.pop("resources")
        self.assertEqual(gpu4, gpu5)
        self.assertEqual(gpu4_resources["fixed_physical_gpu"], 4)
        self.assertEqual(gpu5_resources["fixed_physical_gpu"], 5)
        self.assertEqual(gpu5_resources["minimum_free_mib"], 18432)
        self.assertEqual(gpu5_resources["expected_peak_mib"], 12288)
        self.assertNotIn(5, gpu5_resources["excluded_physical_gpus"])
        launcher = (
            root
            / "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu5.sh"
        ).read_text()
        inner = (
            root
            / "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r1_gpu5_inner.sh"
        ).read_text()
        self.assertIn("phase16_s3r_gridge_resource_r1_gpu5", launcher)
        self.assertIn("tmux new-session -d", launcher)
        self.assertIn("S16_S3_FIXED_GPU=5", inner)
        self.assertIn("S16_S3_EXCLUDED_GPUS=0,4,7", inner)
        self.assertIn("S16_S3_HARD_TIMEOUT=900", inner)

    def test_gridge_r2_is_an_isolated_engineering_retry(self) -> None:
        import hashlib

        root = Path(__file__).resolve().parents[3]
        config_root = root / "experiment/phase16/configs"
        r1 = json.loads(
            (config_root / "stage16_s3r_gridge_resource_sweep_r1_gpu5.json").read_text()
        )
        r2 = json.loads(
            (
                config_root
                / "stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve.json"
            ).read_text()
        )
        retry = r2.pop("engineering_retry")
        self.assertEqual(retry["parent_attempt_id"], r1["attempt_id"])
        self.assertFalse(retry["scientific_configuration_changed"])
        for label in ("parent_r1_raw", "parent_r1_status", "parent_r1_identity"):
            spec = r2["inputs"].pop(label)
            self.assertEqual(
                hashlib.sha256((root / spec["path"]).read_bytes()).hexdigest(),
                spec["sha256"],
            )
        for row in (r1, r2):
            for key in ("attempt_id", "output_dir", "exact_start_command"):
                row.pop(key)
        self.assertEqual(r1, r2)
        launcher = (
            root
            / "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve.sh"
        ).read_text()
        inner = (
            root
            / "experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve_inner.sh"
        ).read_text()
        self.assertIn("phase16_s3r_gridge_resource_r2_gpu5_fp64solve", launcher)
        self.assertIn("tmux new-session -d", launcher)
        self.assertIn("S16_S3_FIXED_GPU=5", inner)
        self.assertIn("S16_S3_HARD_TIMEOUT=900", inner)

    def test_a2_changes_only_attempt_identity_and_command(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config_root = root / "experiment/phase16/configs"
        a1 = json.loads(
            (config_root / "stage16_s3_gfull_objective_resource_sweep.json").read_text()
        )
        a2 = json.loads(
            (config_root / "stage16_s3_gfull_objective_resource_sweep_a2.json").read_text()
        )
        for key in ("attempt_id", "output_dir", "exact_start_command"):
            a1.pop(key)
            a2.pop(key)
        self.assertEqual(a1, a2)
        wrapper = (
            root / "experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a2.sh"
        ).read_text()
        self.assertIn("s16_s3_gfull_resource_a2", wrapper)

    def test_a3_rank_capacity_closes_a2_singular_design(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config_root = root / "experiment/phase16/configs"
        a2 = json.loads(
            (config_root / "stage16_s3_gfull_objective_resource_sweep_a2.json").read_text()
        )
        a3 = json.loads(
            (config_root / "stage16_s3_gfull_objective_resource_sweep_a3.json").read_text()
        )
        width = a3["sweep"]["linear_system_width"]
        a2_cov = a2["sweep"]["covariance_rows_by_position"]
        self.assertTrue(all(int(value) + 16 < width for value in a2_cov.values()))
        a3_cov = a3["sweep"]["covariance_rows_by_position"]
        a3_keys = a3["sweep"]["position_contract_requests_by_position"]
        self.assertEqual(sum(a3_cov.values()), a3["sweep"]["covariance_rows"])
        self.assertTrue(
            all(a3_cov[position] + a3_keys[position] >= width for position in a3_cov)
        )
        self.assertEqual(a3_cov["5"], 2036)
        self.assertEqual(a3_keys["5"], 64)

    def test_independent_lifecycle_probe_reaches_step_29_without_selection(self) -> None:
        request = FullTargetRequest(
            cold_item="cold",
            source_warm_item="warm",
            context_items=("warm",),
            full_target_path=(2,),
            prefix_token_ids=(),
            target_token_id=2,
            legal_token_ids=(2, 3),
            position=0,
        )
        probe = independent_full_lifecycle_probe(request, vector_dimension=2)
        self.assertTrue(probe["pass"])
        self.assertEqual(probe["forward_calls"], 30)
        self.assertIn("not_used_for_candidate_selection", probe["scope"])

    def test_selection_recomputed_with_two_percent_smaller_batch_tie_break(self) -> None:
        rows = [candidate(4, 98.5), candidate(8, 100.0), candidate(16, 99.0)]
        self.assertEqual(choose_candidate(rows)["microbatch"], 4)
        self.assertEqual(expected_selected_microbatch(rows, 8192), 4)

    def test_selection_rejects_self_reported_eligibility_or_missing_semantics(self) -> None:
        row = candidate(4, 10.0)
        row["eligible"] = False
        self.assertIsNone(expected_selected_microbatch([row], 8192))
        row = candidate(4, 10.0)
        row["semantic_checks"].pop("official_lifecycle_prefix")
        self.assertIsNone(expected_selected_microbatch([row], 8192))

    def test_step_29_is_observed_not_inferred_from_config(self) -> None:
        trace = [10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        probe = {
            "scope": "synthetic_failure_row_not_used_for_candidate_selection_or_runtime",
            "lifecycle_check_steps": trace,
            "forward_calls": 30,
            "scheduler_step_count": 30,
            "failed_z_count": 1,
            "pass": True,
        }
        self.assertTrue(observed_full_30_step_path(probe))
        probe["lifecycle_check_steps"] = [10, 20]
        probe["forward_calls"] = 21
        self.assertFalse(observed_full_30_step_path(probe))

    def test_covariance_convergence_ends_at_zero_reference_drift(self) -> None:
        activations = {
            0: torch.tensor([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [1.0, 1.0]]),
            1: torch.tensor([[1.0, 1.0], [2.0, 0.0]]),
        }
        result = covariance_convergence_diagnostics(
            activations, {0: [2, 4], 1: [1, 2]}
        )
        self.assertGreater(
            result["0"][0][
                "relative_frobenius_drift_to_largest_resource_checkpoint"
            ],
            0.0,
        )
        self.assertEqual(
            result["0"][-1][
                "relative_frobenius_drift_to_largest_resource_checkpoint"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
