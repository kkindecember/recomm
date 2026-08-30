#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from experiment.phase16.protocol.genrecedit_faithful import (
    FullTargetRequest,
    ZForwardBatch,
    ZOptimizationConfig,
    extract_keys,
    optimize_z_vectors,
)
from experiment.phase16.protocol.gridge_formal_admission import (
    FORMAL_CODE_PATHS,
    ROOT,
    rank_metrics,
    select_warm_events,
    solve_contract_pass,
    verify_request_dataset,
    verify_resource_parent,
)
from experiment.phase16.protocol.gridge_repeat_queue import (
    build_cycle_config as build_repeat_cycle_config,
)
from experiment.phase16.protocol.prepare_s3r_gridge_f3_runtime import (
    F2_IDENTITY_SHA256,
    F2_STATUS_SHA256,
    derive_f3_config,
)
from experiment.phase16.protocol.gridge_stability_queue import (
    build_cycle_config as build_stability_cycle_config,
)


CONFIG_PATH = ROOT / "experiment/phase16/configs/stage16_s3r_gridge_formal_admission_gpu5_f1.json"
F2_CONFIG_PATH = ROOT / "experiment/phase16/configs/stage16_s3r_gridge_formal_admission_gpu5_f2.json"


def request(index: int) -> FullTargetRequest:
    return FullTargetRequest(
        cold_item=f"cold-{index}",
        source_warm_item=f"warm-{index}",
        context_items=(f"warm-{index}",),
        full_target_path=(3,),
        prefix_token_ids=(),
        target_token_id=0,
        legal_token_ids=(0, 1),
        position=0,
    )


class FormalAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_formal_config_is_isolated_and_resource_gated(self) -> None:
        config = self.config
        self.assertEqual(config["resources"]["fixed_physical_gpu"], 5)
        self.assertEqual(config["resources"]["minimum_free_mib"], 13312)
        self.assertEqual(config["resources"]["hard_timeout_seconds"], 604800)
        self.assertFalse(config["automatic_retry"])
        self.assertFalse(config["validation_used"])
        self.assertFalse(config["test_read"])
        self.assertFalse(config["scientific_efficacy_metric"])
        self.assertEqual(
            config["resource_parent"]["required_verdict"],
            "PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP",
        )
        self.assertIn("/inspired_ridge/admission/", config["output_dir"])
        self.assertNotIn("stability", config["output_dir"])

    def test_frozen_resource_parent_and_request_artifact_pass(self) -> None:
        parent = verify_resource_parent(self.config)
        self.assertEqual(
            parent["verdict"], "PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP"
        )
        _root, manifest, opened = verify_request_dataset(self.config)
        self.assertEqual(manifest["counts"], {"targets": 5963, "contexts": 59630, "requests": 302400})
        self.assertEqual(len(manifest["shards"]), 47)
        self.assertGreater(len(opened), 90)

    def test_z_and_key_progress_callbacks_do_not_change_results(self) -> None:
        rows = [request(index) for index in range(3)]
        z_progress: list[tuple[int, int]] = []

        def forward(batch, deltas, _active):
            logits = torch.stack((deltas[:, 0], torch.zeros_like(deltas[:, 0])), dim=1)
            return ZForwardBatch(logits=logits, target_inits=torch.ones_like(deltas))

        result = optimize_z_vectors(
            requests=rows,
            vector_dimension=2,
            forward_batch=forward,
            config=ZOptimizationConfig(v_num_grad_steps=1, batch_size=2),
            progress_callback=lambda current, total: z_progress.append((current, total)),
            device="cpu",
        )
        self.assertEqual(z_progress, [(2, 3), (3, 3)])
        self.assertEqual(result.valid_count + result.failed_count, 3)

        module = torch.nn.Linear(2, 2, bias=False)
        key_progress: list[tuple[int, int]] = []

        def key_forward(batch):
            module(torch.ones(len(batch), 1, 2))

        keys = extract_keys(
            module=module,
            requests=rows,
            forward_batch=key_forward,
            batch_size=2,
            progress_callback=lambda current, total: key_progress.append((current, total)),
        )
        self.assertEqual(keys.shape, (3, 2))
        self.assertEqual(key_progress, [(2, 3), (3, 3)])

    def test_warm_event_selection_is_deterministic_and_complete(self) -> None:
        rows = [("u2", ("a", "b", "c")), ("u1", ("d", "e", "f"))]
        first = select_warm_events(rows, seed=1502, count=4)
        second = select_warm_events(rows, seed=1502, count=4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(row["history"] for row in first))

    def test_formal_solve_contract_is_fail_closed(self) -> None:
        row = {
            "valid_z_count": 1,
            "solve_completed": True,
            "method_name": "G-RIDGE",
            "solve_variant": "condition_targeted_spectral_ridge_v1",
            "faithful_reproduction": False,
            "ridge_rule": self.config["method"]["ridge_rule"],
            "target_condition": 1_000_000.0,
            "safety_margin": 1e-6,
            "regularized_rank": 2048,
            "regularized_nullity": 0,
            "system_rank": 2048,
            "regularized_system_cholesky_info": 0,
            "regularized_condition": 999_999.0,
            "solve_relative_residual": 1e-12,
            "pseudoinverse_used": False,
            "jitter_fallback_used": False,
            "outcome_resampling_used": False,
        }
        self.assertTrue(solve_contract_pass(row, self.config))
        row["solve_relative_residual"] = 1.1e-6
        self.assertFalse(solve_contract_pass(row, self.config))

    def test_rank_metrics_are_top50_contract_only(self) -> None:
        self.assertEqual(rank_metrics(["a", "b"], "b"), (2, 1, 0.5))
        self.assertEqual(rank_metrics(["a", "b"], "c"), (None, 0, 0.0))

    def test_runner_status_distinguishes_formal_from_stability(self) -> None:
        inner = (
            ROOT
            / "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f1_inner.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("authoritative experiment is finished", inner)
        self.assertIn('"stability_queue_started":%s', inner)
        self.assertIn("STABILITY_QUEUE_STARTED=false", inner)
        self.assertNotIn("scanholder", inner.lower())

    def test_stability_cycle_is_full_compute_but_cannot_promote(self) -> None:
        cycle = build_stability_cycle_config(
            formal_config=self.config,
            cycle=3,
            queue_root="artifacts/phase16/s3_genrecedit/inspired_ridge/stability/toys_seed1502_gpu5",
            summary_sha="a" * 64,
            completion_sha="b" * 64,
        )
        self.assertEqual(cycle["run_role"], "stability_repeat")
        self.assertEqual(cycle["frozen_workload"], self.config["frozen_workload"])
        self.assertEqual(cycle["stability"]["cycle"], 3)
        self.assertTrue(cycle["stability"]["full_reexecution"])
        self.assertFalse(cycle["stability"]["affects_scientific_results"])
        self.assertFalse(cycle["stability"]["promotion_eligible"])
        self.assertFalse(cycle["stability"]["automatic_retry"])
        self.assertIn("cycle_0003", cycle["output_dir"])
        queue_runner = (
            ROOT / "experiment/phase16/run_stage16_s3r_gridge_stability_gpu5_inner.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("full planned stability cycle", queue_runner)
        self.assertIn('"affects_scientific_results":false', queue_runner)
        self.assertNotIn("scanholder", queue_runner.lower())
        self.assertIn(
            "experiment/phase16/protocol/gridge_stability_queue.py", FORMAL_CODE_PATHS
        )

    def test_f2_is_an_isolated_same_science_attempt(self) -> None:
        f1 = self.config
        f2 = json.loads(F2_CONFIG_PATH.read_text(encoding="utf-8"))
        for key in ("seed", "domain", "method", "tokenizer", "frozen_workload", "admission", "resources", "resource_parent"):
            self.assertEqual(f2[key], f1[key])
        self.assertEqual(f2["attempt_id"], "s16_s3r_gridge_formal_gpu5_f2")
        self.assertTrue(f2["post_terminal_repeat_policy"]["enabled_by_user"])
        self.assertTrue(f2["post_terminal_repeat_policy"]["launch_after_any_formal_terminal"])
        self.assertFalse(f2["post_terminal_repeat_policy"]["affects_scientific_results"])
        self.assertFalse(f2["post_terminal_repeat_policy"]["promotion_eligible"])
        self.assertTrue(f2["post_terminal_repeat_policy"]["normal_experiment_priority"])
        self.assertEqual(
            f2["runtime_isolation"]["required_gram_py_sha256"],
            "275f10a94fdcfac9dd7323b43ba1932563bc21b4647906a2d7a0f70a75516466",
        )
        self.assertEqual(f2["inputs"]["failed_f1_status"]["sha256"], "a923a9108c3216a23a560fd5e8cec81927bbf530184f38b3139c81f1e841881d")

    def test_repeat_after_failed_formal_is_nonpromotional(self) -> None:
        f2 = json.loads(F2_CONFIG_PATH.read_text(encoding="utf-8"))
        terminal = {
            "attempt_id": "s16_s3r_gridge_formal_gpu5_f2",
            "status": "FAILED",
            "status_code": "FAILED",
        }
        cycle = build_repeat_cycle_config(
            formal_config=f2,
            formal_status=terminal,
            cycle=4,
            queue_root="artifacts/phase16/s3_genrecedit/inspired_ridge/stability/toys_seed1502_gpu5_f2",
        )
        self.assertEqual(cycle["run_role"], "stability_repeat")
        self.assertEqual(cycle["frozen_workload"], f2["frozen_workload"])
        self.assertEqual(cycle["stability"]["authoritative_stage_status"], "FAILED")
        self.assertFalse(cycle["stability"]["affects_scientific_results"])
        self.assertFalse(cycle["stability"]["promotion_eligible"])
        self.assertTrue(cycle["stability"]["normal_experiment_priority"])
        self.assertTrue(cycle["stability"]["continue_after_cycle_failure_by_user_authorization"])
        self.assertFalse(cycle["stability"]["automatic_retry"])

    def test_f2_repeat_runner_yields_only_its_own_cycle(self) -> None:
        runner = (
            ROOT / "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f2_inner.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("foreign_gpu5_pids", runner)
        self.assertIn("YIELDED_TO_PRIORITY_GPU5", runner)
        self.assertIn('kill -TERM "$CYCLE_PID"', runner)
        self.assertNotIn("pkill", runner)
        self.assertNotIn("killall", runner)
        self.assertNotIn("affects_scientific_results\":true", runner)
        for path in (
            "experiment/phase16/protocol/gridge_repeat_queue.py",
            "experiment/phase16/protocol/prepare_s3r_gridge_f2_runtime.py",
            "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f2.sh",
            "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f2_inner.sh",
            "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f2.sh",
            "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f2_inner.sh",
        ):
            self.assertIn(path, FORMAL_CODE_PATHS)

    def test_f3_is_mechanically_derived_same_science_with_f2_lineage(self) -> None:
        f2 = json.loads(F2_CONFIG_PATH.read_text(encoding="utf-8"))
        f3 = derive_f3_config(f2, f2_config_sha256="c" * 64)
        for key in (
            "seed",
            "domain",
            "method",
            "tokenizer",
            "frozen_workload",
            "admission",
            "resources",
            "resource_parent",
        ):
            self.assertEqual(f3[key], f2[key])
        self.assertEqual(f3["attempt_id"], "s16_s3r_gridge_formal_gpu5_f3")
        self.assertEqual(f3["inputs"]["failed_f2_config"]["sha256"], "c" * 64)
        self.assertEqual(f3["inputs"]["failed_f2_status"]["sha256"], F2_STATUS_SHA256)
        self.assertEqual(
            f3["inputs"]["failed_f2_identity"]["sha256"], F2_IDENTITY_SHA256
        )
        self.assertFalse(f3["post_terminal_repeat_policy"]["promotion_eligible"])
        self.assertEqual(
            f3["runtime_isolation"]["snapshot_root"],
            ".runtime/phase16_s3r_gridge_f3_runtime",
        )

    def test_f3_wrappers_preserve_formal_priority_and_repeat_isolation(self) -> None:
        formal = (
            ROOT
            / "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f3_inner.sh"
        ).read_text(encoding="utf-8")
        repeat = (
            ROOT / "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f3_inner.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("s16_s3r_gridge_formal_gpu5_f3", formal)
        self.assertIn("phase16_s3r_gridge_repeat_gpu5_f3", formal)
        self.assertIn("PHASE16_REPEAT_ATTEMPT_PREFIX=s16_s3r_gridge_repeat_gpu5_f3", repeat)
        self.assertIn("PHASE16_REPEAT_CONFIG_ROOT_REL=.runtime/phase16_s3r_gridge_f3_repeat_configs", repeat)
        for path in (
            "experiment/phase16/protocol/prepare_s3r_gridge_f3_runtime.py",
            "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f3.sh",
            "experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f3_inner.sh",
            "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f3.sh",
            "experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f3_inner.sh",
        ):
            self.assertIn(path, FORMAL_CODE_PATHS)


if __name__ == "__main__":
    unittest.main()
