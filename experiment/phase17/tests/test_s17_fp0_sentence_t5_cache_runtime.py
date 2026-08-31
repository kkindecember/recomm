from __future__ import annotations

import unittest
from pathlib import Path

from experiment.phase17.protocol.s17_fp0_sentence_t5_cache_runtime import (
    ALLOW_PATTERNS,
    ATTEMPT_ID,
    CURL,
    MODEL_REVISION,
    TMUX_NETWORK_PROXY,
    TRANSFER_ATTEMPTS,
    controlled_environment,
    download_script,
    paths,
    validation_script,
    worker_command,
)


ROOT = Path(__file__).resolve().parents[3]


class S17FP0SentenceT5CacheRuntimeTests(unittest.TestCase):
    def test_model_revision_and_output_path_are_immutable(self) -> None:
        self.assertEqual(MODEL_REVISION, "fc5d4628481afbbaaacd7af6bb07cf9d3865f781")
        self.assertIn(MODEL_REVISION, paths(ROOT)["model"].name)

    def test_download_selects_safetensors_without_duplicate_large_backends(self) -> None:
        self.assertIn("model.safetensors", ALLOW_PATTERNS)
        self.assertIn("2_Dense/model.safetensors", ALLOW_PATTERNS)
        self.assertNotIn("pytorch_model.bin", ALLOW_PATTERNS)
        self.assertNotIn("rust_model.ot", ALLOW_PATTERNS)
        compile(download_script(), "<s17-fp0-sentence-t5-download>", "exec")
        self.assertIn("--continue-at", download_script())
        self.assertIn("transfer_retry", download_script())
        self.assertEqual(CURL, Path("/usr/bin/curl"))
        self.assertEqual(TRANSFER_ATTEMPTS, 5)

    def test_cache_task_hides_all_gpus(self) -> None:
        environment = controlled_environment(paths(ROOT))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue(environment["HF_HOME"].startswith(str(ROOT / "artifacts/phase17")))

    def test_offline_validation_script_is_valid_python(self) -> None:
        compile(validation_script(), "<s17-fp0-sentence-t5-validation>", "exec")
        self.assertIn("local_files_only=True", validation_script())

    def test_recovery_worker_bootstraps_project_imports(self) -> None:
        self.assertEqual(ATTEMPT_ID, "attempt_005")
        command = worker_command(ROOT, paths(ROOT))
        self.assertEqual(command[:2], ["/usr/bin/env", f"PYTHONPATH={ROOT}"])
        self.assertIn(f"http_proxy={TMUX_NETWORK_PROXY}", command)
        self.assertIn(f"https_proxy={TMUX_NETWORK_PROXY}", command)
        self.assertNotIn("@", TMUX_NETWORK_PROXY)


if __name__ == "__main__":
    unittest.main()
