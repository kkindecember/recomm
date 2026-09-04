from __future__ import annotations

import inspect
import unittest

from experiment.phase18.protocol import s18_s1_runtime as runtime
from model.gram_t5_modeling import T5Block


class S18S1CrossAttentionCacheTests(unittest.TestCase):
    def test_default_behavior_remains_enabled(self) -> None:
        source = inspect.getsource(T5Block.__init__)
        self.assertIn("self.cache_cross_attention = True", source)

    def test_disabled_mode_keeps_two_self_attention_states(self) -> None:
        source = inspect.getsource(T5Block.forward)
        self.assertIn("not self.cache_cross_attention", source)
        self.assertIn("and self.cache_cross_attention", source)

    def test_runtime_toggle_requires_all_decoder_blocks(self) -> None:
        source = inspect.getsource(runtime.set_cross_attention_cache)
        self.assertIn("for block in model.decoder.block", source)
        self.assertIn("block.cache_cross_attention = enabled", source)


if __name__ == "__main__":
    unittest.main()
