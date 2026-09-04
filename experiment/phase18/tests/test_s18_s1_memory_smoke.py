from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from experiment.phase18.protocol import s18_s1_memory_smoke as smoke
from experiment.phase18.protocol import s18_s1_runtime as runtime


class S18S1MemorySmokeTests(unittest.TestCase):
    def test_batch_is_already_one_user(self) -> None:
        source = inspect.getsource(runtime.diagnose)
        self.assertIn("batch = collator([dataset[index]])", source)

    def test_resource_mode_preserves_beams_and_releases_cache(self) -> None:
        source = inspect.getsource(smoke.run)
        self.assertIn("generation_use_cache=generation_use_cache", source)
        self.assertIn("cross_attention_cache=cross_attention_cache", source)
        self.assertIn("release_cuda_cache_per_user=True", source)
        self.assertIn("compare_first_user(target)", source)

    def test_smoke_is_not_scientific_output(self) -> None:
        source = inspect.getsource(smoke.run)
        self.assertIn('"scientific_result_eligible": False', source)
        self.assertIn('"scientific_parameters_changed": False', source)

    def test_gpu1_is_now_in_researcher_authorized_smoke_set(self) -> None:
        source = inspect.getsource(smoke.run)
        self.assertIn("{0, 1, 4, 6, 7}", source)

    def test_output_is_disjoint_from_run0002(self) -> None:
        self.assertNotIn(
            Path("artifacts/phase18/s1_actionability/run-0002"),
            smoke.output_root(
                1,
                generation_use_cache=True,
                cross_attention_cache=True,
                decoder_model_parallel=True,
            ).parents,
        )


if __name__ == "__main__":
    unittest.main()
