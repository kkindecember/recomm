"""cold_prefix_support 的回归测试。

两类断言：

1. 合成数据上的正确性——前缀支持的语义必须精确，包括「不同长度的 warm item
   也能为一个短前缀提供支持」这一条（第一版口算脚本正是漏了它，把 Toys 的
   8.6% 少算成 8.3%）。
2. 真实 Toys/Beauty cold50 上的数字锁定——v0.2 计划的技术路线选择直接建立在
   这几个数上，任何改动导致它们变化都必须是显式的。
"""

from __future__ import annotations

import os
import sys
import textwrap
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protocol"))

from cold_prefix_support import (  # noqa: E402
    analyse,
    deepest_supported_prefix,
    deepest_token_supported,
    load_paths,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class TestDeepestSupportedPrefix(unittest.TestCase):
    def test_stops_at_first_unsupported_depth(self):
        warm = {1: {("a",)}, 2: {("a", "b")}, 3: set()}
        self.assertEqual(deepest_supported_prefix(["a", "b", "c"], warm), 2)

    def test_zero_when_root_token_unsupported(self):
        warm = {1: {("x",)}}
        self.assertEqual(deepest_supported_prefix(["a", "b"], warm), 0)

    def test_full_support(self):
        warm = {1: {("a",)}, 2: {("a", "b")}}
        self.assertEqual(deepest_supported_prefix(["a", "b"], warm), 2)

    def test_support_is_not_resumed_after_a_gap(self):
        # 前缀支持单调：z[:2] 无支持时，即便 z[:3] 恰好出现在索引里也不算支持，
        # 因为 decoder 根本走不到那一层。
        warm = {1: {("a",)}, 2: set(), 3: {("a", "b", "c")}}
        self.assertEqual(deepest_supported_prefix(["a", "b", "c"], warm), 1)


class TestLoadPaths(unittest.TestCase):
    def test_parses_item_and_tokens(self):
        import tempfile

        content = textwrap.dedent(
            """\
            ITEM1 |▁animals|stuffed|▁cat
            ITEM2 |▁train|mas

            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(content)
            tmp = fh.name
        try:
            paths = load_paths(tmp)
            self.assertEqual(paths["ITEM1"], ["▁animals", "stuffed", "▁cat"])
            self.assertEqual(paths["ITEM2"], ["▁train", "mas"])
            self.assertEqual(len(paths), 2)
        finally:
            os.unlink(tmp)


class TestCrossLengthPrefixSupport(unittest.TestCase):
    """回归：长度不同的 warm item 也必须能支持短前缀。"""

    def test_longer_warm_item_supports_shorter_cold_prefix(self):
        # cold 长度 3，其 all-but-last 前缀 ("a","b") 长度 2。
        # 唯一的 warm item 长度 4，自身 all-but-last 前缀是 ("a","b","z")，
        # 但它确实提供了长度 2 的前缀 ("a","b")。必须算作被支持。
        warm_prefixes = {}
        warm_tokens = ["a", "b", "z", "q"]
        for k in range(1, len(warm_tokens) + 1):
            warm_prefixes[k] = {tuple(warm_tokens[:k])}

        cold_tokens = ["a", "b", "c"]
        self.assertIn(tuple(cold_tokens[:-1]), warm_prefixes[len(cold_tokens) - 1])
        self.assertEqual(deepest_supported_prefix(cold_tokens, warm_prefixes), 2)


class TestRealDatasets(unittest.TestCase):
    """锁定 v0.2 计划所依赖的真实数字。"""

    EXPECTED = {
        "Toys_cold50": {
            "n_items_total": 11924,
            "n_cold": 5963,
            "n_warm": 5961,
            "duplicate_paths": 0,
            "cold_all_but_last_prefix_supported_pct": 8.6,
            "cumulative_at_depth_2": 67.8,
        },
        "Beauty_cold50": {
            "n_items_total": 12101,
            "n_cold": 6052,
            "n_warm": 6049,
            "duplicate_paths": 0,
            "cold_all_but_last_prefix_supported_pct": 2.66,
            "cumulative_at_depth_2": 82.5,
        },
    }

    def test_locked_numbers(self):
        for name, expected in self.EXPECTED.items():
            dataset_dir = os.path.join(REPO_ROOT, "GRAM", "rec_datasets", name)
            if not os.path.isdir(dataset_dir):
                self.skipTest(f"{dataset_dir} 不存在")
            with self.subTest(dataset=name):
                summary = analyse(dataset_dir)
                self.assertFalse(summary["test_read"])
                for key in ("n_items_total", "n_cold", "n_warm", "duplicate_paths"):
                    self.assertEqual(summary[key], expected[key], key)
                self.assertAlmostEqual(
                    summary["cold_all_but_last_prefix_supported_pct"],
                    expected["cold_all_but_last_prefix_supported_pct"],
                    places=2,
                )
                self.assertAlmostEqual(
                    summary["cold_depth_cumulative_pct"]["2"],
                    expected["cumulative_at_depth_2"],
                    places=1,
                )

    def test_break_is_structurally_early(self):
        """结构观察：多数 cold path 很早就失去完整 warm-prefix overlap。

        ⚠️ 这**不是**「断崖在中层」的证明（v0.2 曾如此声称，v0.3 已撤销）。
        neural decoder 可组合出从未完整出现过的 prefix，故本量不是 learned
        NLL 断崖的上界。它只锁定一个结构事实，路线选择由 14-0B 实测决定。
        """
        for name in self.EXPECTED:
            dataset_dir = os.path.join(REPO_ROOT, "GRAM", "rec_datasets", name)
            if not os.path.isdir(dataset_dir):
                self.skipTest(f"{dataset_dir} 不存在")
            with self.subTest(dataset=name):
                summary = analyse(dataset_dir)
                # 过半 cold item 在深度 ≤2 处失去完整 warm 前缀重叠
                self.assertGreater(summary["cold_depth_cumulative_pct"]["2"], 50.0)
                # 只有极少数是「只差最后一个 token」
                self.assertLess(summary["cold_all_but_last_prefix_supported_pct"], 10.0)


class TestTokenVsPrefixSupport(unittest.TestCase):
    """锁定「严格前缀」与「逐层 token」两个口径的区别。

    v0.2 曾声称严格口径是 learned NLL 断崖的上界，v0.3 撤销了该论断：
    decoder 可以走通一条每层 token 都见过、但该组合从未整体出现过的路径。
    """

    def test_composable_path_is_token_supported_but_not_prefix_supported(self):
        # 两个 warm item：a-b-c 与 x-b-z
        warm_paths = [["a", "b", "c"], ["x", "y", "z"]]
        warm_prefixes: dict = {}
        warm_tokens: dict = {}
        for tokens in warm_paths:
            for k in range(1, len(tokens) + 1):
                warm_prefixes.setdefault(k, set()).add(tuple(tokens[:k]))
                warm_tokens.setdefault(k, set()).add(tokens[k - 1])

        # cold path a-y-z：每层 token 都被见过（a@1, y@2, z@3），
        # 但完整前缀 ("a","y") 从未出现。
        cold = ["a", "y", "z"]
        self.assertEqual(deepest_supported_prefix(cold, warm_prefixes), 1)
        self.assertEqual(deepest_token_supported(cold, warm_tokens), 3)

    def test_real_datasets_show_large_gap(self):
        """真实数据上两个口径差距很大——这就是撤销「上界」论断的依据。"""
        expectations = {
            # dataset: (严格口径 depth<=2 累计, 宽松口径 depth<=2 累计)
            "Toys_cold50": (67.8, 32.3),
            "Beauty_cold50": (82.5, 47.5),
        }
        for name, (strict, loose) in expectations.items():
            dataset_dir = os.path.join(REPO_ROOT, "GRAM", "rec_datasets", name)
            if not os.path.isdir(dataset_dir):
                self.skipTest(f"{dataset_dir} 不存在")
            with self.subTest(dataset=name):
                summary = analyse(dataset_dir)
                self.assertAlmostEqual(
                    summary["cold_depth_cumulative_pct"]["2"], strict, places=1
                )
                self.assertAlmostEqual(
                    summary["cold_token_depth_cumulative_pct"]["2"], loose, places=1
                )
                # 宽松口径必然更浅（更晚断裂）——差距即可组合空间
                self.assertLess(
                    summary["cold_token_depth_cumulative_pct"]["2"],
                    summary["cold_depth_cumulative_pct"]["2"],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
