from __future__ import annotations

import unittest
from unittest.mock import patch

from experiment.phase17.protocol import s17_fp12_g1_guard_v3_host_migration as migration
from experiment.phase17.protocol import s17_fp12_g1_runtime_guard_v3 as v3


class G1GuardV3HostMigrationTests(unittest.TestCase):
    def test_host_validation_requires_matching_tmux_and_gpu_identity(self) -> None:
        status = {
            "experiment_id": v3.V2_EXPERIMENT_ID,
            "scientific_state": "COMPLETED",
            "affects_scientific_result": False,
            "workload_pid": 2969343,
            "tmux_session": v3.V2_EXPERIMENT_ID,
        }
        with (
            patch.object(migration, "_read", return_value=status),
            patch.object(migration, "tmux_session_exists", return_value=True),
            patch.object(
                migration,
                "tmux_pane_identity",
                return_value={
                    "session": v3.V2_EXPERIMENT_ID,
                    "pid": 2969343,
                    "command": "python",
                    "dead": False,
                },
            ),
            patch.object(
                migration,
                "gpu_uuid_by_index",
                return_value={4: "GPU-4"},
            ),
            patch.object(
                migration,
                "gpu_processes",
                return_value=[
                    {
                        "gpu_uuid": "GPU-4",
                        "pid": 2969343,
                        "process_name": "/home/user/envs/gram-repro/bin/python",
                        "used_memory_mib": 17862,
                    }
                ],
            ),
        ):
            evidence = migration.validate_host_visible_v2(migration.ROOT)
        self.assertEqual(evidence["tmux_pane"]["pid"], 2969343)
        self.assertEqual(evidence["gpu_process"]["used_memory_mib"], 17862)

    def test_host_validation_rejects_gpu_mismatch(self) -> None:
        status = {
            "experiment_id": v3.V2_EXPERIMENT_ID,
            "scientific_state": "COMPLETED",
            "affects_scientific_result": False,
            "workload_pid": 2969343,
            "tmux_session": v3.V2_EXPERIMENT_ID,
        }
        with (
            patch.object(migration, "_read", return_value=status),
            patch.object(migration, "tmux_session_exists", return_value=True),
            patch.object(
                migration,
                "tmux_pane_identity",
                return_value={
                    "session": v3.V2_EXPERIMENT_ID,
                    "pid": 2969343,
                    "command": "python",
                    "dead": False,
                },
            ),
            patch.object(migration, "gpu_uuid_by_index", return_value={4: "GPU-4"}),
            patch.object(
                migration,
                "gpu_processes",
                return_value=[
                    {
                        "gpu_uuid": "GPU-5",
                        "pid": 2969343,
                        "process_name": "/home/user/envs/gram-repro/bin/python",
                        "used_memory_mib": 17862,
                    }
                ],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "physical GPU"):
                migration.validate_host_visible_v2(migration.ROOT)


if __name__ == "__main__":
    unittest.main()
