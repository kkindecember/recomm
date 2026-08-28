import importlib.util
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "experiment/phase16/protocol/splus_ctrl_duplicate_guard.py"
SPEC = importlib.util.spec_from_file_location("splus_ctrl_duplicate_guard", MODULE_PATH)
guard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def scientific_config():
    return {
        "formal_budget": {
            "internal_dev_transitions": 3108,
            "pseudo_cold_events": 7435,
            "full_catalog_items": 11924,
            "pretrain": {"optimizer_steps": 10900},
            "finetune": {"optimizer_steps": 1635},
        },
        "admission": {"maximum_eligible_peak_reserved_mib": 28672},
    }


def plus_summary():
    return {
        "status": "completed",
        "verdict": "PASS_S16_2_S_PLUS_FORMAL_EXECUTION",
        "arm": "S-PLUS",
        "arm_optimizer_steps": 12535,
        "expected_arm_optimizer_steps": 12535,
        "internal_dev_generation_admission": {"all_finite": True, "events": 3108},
        "pseudo_cold_full_catalog_admission": {
            "all_finite": True,
            "events": 7435,
            "candidate_items": 11924,
        },
        "base_checkpoint_unchanged": True,
        "base_checkpoint_sha256_before": "abc",
        "base_checkpoint_sha256_after": "abc",
        "peak_cuda_reserved_mib": 17000,
        "test_read": False,
        "validation_used": False,
    }


def a4_arm_summary():
    return {
        "status": "completed",
        "verdict": "PASS_S16_2_S_PLUS_CTRL_FORMAL_EXECUTION",
        "arm": "S-PLUS-CTRL",
        "arm_optimizer_steps": 12535,
        "internal_dev_generation_admission": {"all_finite": True, "events": 3108},
        "pseudo_cold_full_catalog_admission": None,
        "base_checkpoint_unchanged": True,
        "test_read": False,
        "validation_used": False,
    }


def statuses(now):
    stamp = now.isoformat()
    a3 = {
        "attempt_id": "a3",
        "physical_gpu": 5,
        "status": "running",
        "status_code": "RUNNING",
        "current_arm": "S-PLUS",
        "updated_at": stamp,
        "test_read": False,
        "validation_used": False,
    }
    a4 = {
        "attempt_id": "a4",
        "physical_gpu": 7,
        "status": "running",
        "status_code": "RUNNING",
        "current_arm": "S-PLUS-CTRL",
        "process_alive": True,
        "progress_current": 100,
        "updated_at": stamp,
        "last_progress_at": stamp,
        "test_read": False,
        "validation_used": False,
    }
    return a3, a4


def guard_config():
    return {
        "a3": {"attempt_id": "a3"},
        "a4": {"attempt_id": "a4"},
        "requirements": {
            "minimum_a4_optimizer_steps": 1,
            "maximum_status_age_seconds": 300,
            "maximum_a4_progress_age_seconds": 3600,
        },
    }


class DuplicateGuardTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 28, 3, 0, tzinfo=timezone.utc)
        self.a3_status, self.a4_status = statuses(self.now)
        self.config = guard_config()
        self.scientific = scientific_config()

    def decide(self, plus=None, ctrl=None, a4_overall=None, a4_arm=None):
        return guard.document_decision(
            self.config,
            self.scientific,
            self.scientific,
            self.a3_status,
            plus,
            ctrl,
            self.a4_status,
            a4_overall,
            a4_arm,
            self.now,
        )

    def test_waits_while_splus_is_running(self):
        decision = self.decide()
        self.assertEqual(decision.state, "WAIT")
        self.assertFalse(decision.ready_to_signal)

    def test_ready_only_after_plus_pass_and_ctrl_transition(self):
        self.a3_status["current_arm"] = "S-PLUS-CTRL"
        decision = self.decide(plus=plus_summary())
        self.assertEqual(decision.state, "READY")
        self.assertTrue(decision.ready_to_signal)

    def test_invalid_plus_summary_blocks_signal(self):
        self.a3_status["current_arm"] = "S-PLUS-CTRL"
        broken = plus_summary()
        broken["arm_optimizer_steps"] = 12534
        decision = self.decide(plus=broken)
        self.assertEqual(decision.state, "BLOCKED")
        self.assertFalse(decision.ready_to_signal)

    def test_failed_a4_preserves_gpu5_fallback(self):
        self.a3_status["current_arm"] = "S-PLUS-CTRL"
        self.a4_status.update({"status": "failed", "status_code": "FAILED", "process_alive": False})
        decision = self.decide(plus=plus_summary())
        self.assertEqual(decision.state, "BLOCKED")
        self.assertIn("terminal non-PASS", decision.reason)

    def test_completed_a4_is_safe_replacement(self):
        self.a3_status["current_arm"] = "S-PLUS-CTRL"
        self.a4_status.update({"status": "completed", "status_code": "COMPLETED_CTRL_ONLY", "process_alive": False})
        overall = {
            "status": "completed",
            "verdict": "PASS_S16_2_S_PLUS_CTRL_SPLIT_FORMAL_EXECUTION",
            "same_frozen_scientific_config_as_parent": True,
            "formal_training_completed": True,
        }
        decision = self.decide(plus=plus_summary(), a4_overall=overall, a4_arm=a4_arm_summary())
        self.assertEqual(decision.state, "READY")

    def test_duplicate_already_complete_is_never_signaled(self):
        self.a3_status["current_arm"] = "S-PLUS-CTRL"
        decision = self.decide(plus=plus_summary(), ctrl={"status": "completed"})
        self.assertEqual(decision.state, "NO_ACTION")

    def test_process_identity_requires_start_ticks_and_exact_cmdline(self):
        spec = {"pid": 41, "start_ticks": 900, "cmdline": "bash frozen_runner.sh 5"}
        guard.require_frozen_process(guard.ProcessInfo(41, 1, 900, "bash frozen_runner.sh 5"), spec, "runner")
        with self.assertRaises(guard.GuardError):
            guard.require_frozen_process(guard.ProcessInfo(41, 1, 901, "bash frozen_runner.sh 5"), spec, "runner")

    def test_child_identity_requires_parent_and_ctrl_arm(self):
        info = guard.ProcessInfo(52, 41, 950, "python train.py --arm S-PLUS-CTRL")
        guard.require_child_process(info, 41, ["train.py", "--arm S-PLUS-CTRL"], "ctrl")
        with self.assertRaises(guard.GuardError):
            guard.require_child_process(info, 99, ["train.py", "--arm S-PLUS-CTRL"], "ctrl")


if __name__ == "__main__":
    unittest.main()
