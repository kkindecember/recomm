from __future__ import annotations

import tarfile
import unittest
from pathlib import Path

from experiment.phase17.protocol.s17_fp0_native_env_runtime import (
    ATTEMPT_ID,
    BOOTSTRAP_PYTHON,
    BOOTSTRAP_PYTHON_SHA256,
    LATTE_COMMIT,
    archive_member_is_safe,
    build_uv_command,
    controlled_environment,
    paths,
    validation_script,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


class S17FP0NativeEnvironmentRuntimeTests(unittest.TestCase):
    def test_archive_rejects_traversal_links_and_devices(self) -> None:
        self.assertTrue(archive_member_is_safe(tarfile.TarInfo("Latte-main/LICENSE")))
        self.assertFalse(archive_member_is_safe(tarfile.TarInfo("../escape")))
        link = tarfile.TarInfo("Latte-main/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp/target"
        self.assertFalse(archive_member_is_safe(link))
        device = tarfile.TarInfo("Latte-main/device")
        device.type = tarfile.CHRTYPE
        self.assertFalse(archive_member_is_safe(device))

    def test_official_environment_is_isolated_and_gpu_hidden(self) -> None:
        resolved = paths(ROOT)
        environment = controlled_environment(resolved)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(Path(environment["UV_PROJECT_ENVIRONMENT"]), resolved["env"])
        self.assertTrue(str(resolved["env"]).startswith(str(ROOT / "artifacts/phase17")))
        command = build_uv_command(resolved)
        self.assertIn("--managed-python", command)
        self.assertIn("--no-dev", command)
        self.assertEqual(command[command.index("--python") + 1], str(BOOTSTRAP_PYTHON))
        self.assertEqual(len(BOOTSTRAP_PYTHON_SHA256), 64)

    def test_persistent_paths_are_commit_pinned(self) -> None:
        resolved = paths(ROOT)
        self.assertIn(LATTE_COMMIT, resolved["archive"].name)
        self.assertIn(LATTE_COMMIT, resolved["source"].name)
        self.assertNotIn("/tmp", str(resolved["env"]))

    def test_validation_script_is_valid_python(self) -> None:
        compile(validation_script(), "<s17-fp0-native-env-validation>", "exec")

    def test_recovery_worker_bootstraps_project_imports(self) -> None:
        self.assertEqual(ATTEMPT_ID, "attempt_003")
        command = worker_command(ROOT, paths(ROOT))
        self.assertEqual(command[0], "/usr/bin/env")
        self.assertEqual(command[1], f"PYTHONPATH={ROOT}")


if __name__ == "__main__":
    unittest.main()
