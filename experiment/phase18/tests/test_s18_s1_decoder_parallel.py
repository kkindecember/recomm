from __future__ import annotations

import inspect
import unittest

from experiment.phase18.protocol import s18_s1_runtime as runtime


class S18S1DecoderParallelTests(unittest.TestCase):
    def test_only_decoder_is_parallelized(self) -> None:
        source = inspect.getsource(runtime.enable_two_gpu_decoder_parallel)
        self.assertIn("model.decoder.parallelize(device_map)", source)
        self.assertNotIn("model.encoder.parallelize", source)

    def test_exactly_two_visible_gpus_are_required(self) -> None:
        source = inspect.getsource(runtime.enable_two_gpu_decoder_parallel)
        self.assertIn("torch.cuda.device_count() != 2", source)

    def test_all_visible_gpu_caches_are_released(self) -> None:
        source = inspect.getsource(runtime.release_cuda_caches)
        self.assertIn("range(torch.cuda.device_count())", source)


if __name__ == "__main__":
    unittest.main()
