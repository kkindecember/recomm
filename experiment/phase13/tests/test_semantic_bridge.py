"""Unit tests for phase13 v1 (semantic bridge) code."""
from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE.parent / "protocol"
sys.path.insert(0, str(PROTOCOL_DIR))

from hierarchical_id_utils import (
    HierIdVocab,
    build_vocab_from_id_file,
    format_id_line,
    parse_id_line,
    read_id_file,
    read_item_set,
    write_id_file,
)
from precompute_item_embeddings import (
    add_text_prefix,
    l2_normalize,
    mean_pool,
    pool_hidden_state,
)


class TestEmbeddingProtocol(unittest.TestCase):
    def test_text_prefix_is_applied_without_mutating_input(self):
        raw = ["alpha", "", "beta"]
        prefixed = add_text_prefix(raw, "query: ")
        self.assertEqual(prefixed, ["query: alpha", "query: ", "query: beta"])
        self.assertEqual(raw, ["alpha", "", "beta"])

    def test_mean_pool_respects_attention_mask(self):
        import torch
        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = mean_pool(hidden, mask)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 4.0]])))

    def test_cls_pool_selects_first_token(self):
        import torch
        hidden = torch.tensor([[[1.0, 3.0], [8.0, 9.0], [5.0, 7.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = pool_hidden_state(hidden, mask, "cls")
        self.assertTrue(torch.equal(pooled, torch.tensor([[1.0, 3.0]])))

    def test_unknown_pooling_is_rejected(self):
        import torch
        with self.assertRaises(ValueError):
            pool_hidden_state(torch.zeros(1, 1, 2), torch.ones(1, 1), "bad")

    def test_l2_normalize_produces_unit_rows(self):
        import torch
        x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
        normalized = l2_normalize(x)
        self.assertTrue(torch.allclose(
            normalized.norm(p=2, dim=1), torch.ones(2), atol=1e-6
        ))


class TestHierarchicalIdUtils(unittest.TestCase):
    def test_parse_format_roundtrip(self):
        line = "B000P24EI2 |▁loss|▁brake|▁fur|▁consolid|ren|▁scalp|▁profound"
        item_id, tokens = parse_id_line(line)
        self.assertEqual(item_id, "B000P24EI2")
        self.assertEqual(len(tokens), 7)
        self.assertEqual(tokens[0], "▁loss")
        reformed = format_id_line(item_id, tokens)
        self.assertEqual(reformed, line)

    def test_vocab_build_and_encode_decode(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            f = tmp / "ids.txt"
            with open(f, "w") as g:
                g.write("i1 |a|b|c|d|e|f|g\n")
                g.write("i2 |a|x|c|d|e|f|g\n")
                g.write("i3 |z|b|c|d|e|f|g\n")
            v = build_vocab_from_id_file(f)
            self.assertEqual(v.level_sizes[0], 2)  # {a, z}
            self.assertEqual(v.level_sizes[1], 2)  # {b, x}
            self.assertEqual(v.level_sizes[2], 1)  # {c}
            # Roundtrip
            enc = v.encode(["a", "x", "c", "d", "e", "f", "g"])
            dec = v.decode(enc)
            self.assertEqual(dec, ["a", "x", "c", "d", "e", "f", "g"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_vocab_save_load(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            v = HierIdVocab()
            v.per_level_idx_to_token = [["a", "b"], ["c"], ["d", "e"]] + [["x"]] * 4
            v.per_level_token_to_idx = [
                {t: i for i, t in enumerate(l)} for l in v.per_level_idx_to_token
            ]
            p = tmp / "vocab.json"
            v.save(p)
            v2 = HierIdVocab.load(p)
            self.assertEqual(v2.level_sizes, v.level_sizes)
            self.assertEqual(v2.per_level_idx_to_token, v.per_level_idx_to_token)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_five_level_id_file_toys_style(self):
        """Toys uses 5-token ids (c32/l5); vocab must infer level count."""
        tmp = Path(tempfile.mkdtemp())
        try:
            f = tmp / "toys_ids.txt"
            with open(f, "w") as g:
                g.write("B0000A1Z5K |▁animals|stuffed|▁se|▁cat|hat\n")
                g.write("B009NFFYWM |▁train|mas|▁island|▁railway|th\n")
                g.write("B00XXXAAAA |▁animals|mas|▁se|▁railway|hat\n")
            v = build_vocab_from_id_file(f)
            self.assertEqual(v.n_levels, 5)
            self.assertEqual(len(v.level_sizes), 5)
            id_map = read_id_file(f)
            self.assertEqual(len(id_map["B0000A1Z5K"]), 5)
            # roundtrip
            item_id, tokens = parse_id_line(
                "B0000A1Z5K |▁animals|stuffed|▁se|▁cat|hat"
            )
            self.assertEqual(len(tokens), 5)
            self.assertEqual(
                format_id_line(item_id, tokens),
                "B0000A1Z5K |▁animals|stuffed|▁se|▁cat|hat",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSemanticBridgeModel(unittest.TestCase):
    def test_forward_shapes_and_loss_decrease(self):
        import torch
        import torch.nn as nn
        from semantic_bridge import build_model

        text_dim = 32
        level_sizes = [8, 16, 4]
        model = build_model(text_dim, level_sizes)

        # Fake batch
        rng = torch.Generator().manual_seed(0)
        x = torch.randn(64, text_dim, generator=rng)
        y = torch.stack([
            torch.randint(0, s, (64,), generator=rng) for s in level_sizes
        ], dim=1)

        # Forward shapes
        out = model(x)
        self.assertEqual(len(out), 3)
        for i, o in enumerate(out):
            self.assertEqual(o.shape, (64, level_sizes[i]))

        # Loss decreases on same batch (overfitting sanity)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        ce = nn.CrossEntropyLoss()
        losses = []
        for _ in range(50):
            out = model(x)
            loss = sum(ce(out[l], y[:, l]) for l in range(3))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.5,
                        f"loss did not decrease enough: {losses[0]:.3f} -> {losses[-1]:.3f}")


class TestResidualSemanticBridgeModel(unittest.TestCase):
    def test_forward_shapes_residual_width_and_loss_decrease(self):
        import torch
        import torch.nn as nn
        from semantic_bridge_residual import build_model_residual

        text_dim = 16
        hidden_dim = 32
        level_sizes = [5, 7, 3]
        model = build_model_residual(text_dim, level_sizes, hidden_dim)
        self.assertEqual(model.fc1.in_features, text_dim)
        self.assertEqual(model.fc1.out_features, hidden_dim)
        self.assertEqual(model.fc2.in_features, hidden_dim)
        self.assertEqual(model.fc2.out_features, text_dim)

        generator = torch.Generator().manual_seed(7)
        x = torch.randn(48, text_dim, generator=generator)
        y = torch.stack(
            [torch.randint(0, size, (48,), generator=generator) for size in level_sizes],
            dim=1,
        )
        outputs = model(x)
        self.assertEqual([tuple(output.shape) for output in outputs], [(48, 5), (48, 7), (48, 3)])

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        cross_entropy = nn.CrossEntropyLoss()
        losses = []
        for _ in range(80):
            outputs = model(x)
            loss = sum(cross_entropy(outputs[level], y[:, level]) for level in range(3))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        self.assertLess(losses[-1], losses[0] * 0.5)


class TestAssignColdIdsEndToEnd(unittest.TestCase):
    """Synthetic dataset — train MLP briefly, assign cold ids, verify output."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = random.Random(42)
        # 40 items, 5 warm-training-ready + 5 cold
        # 7-level ids, each level has a small vocab
        level_vocabs = [
            ["a0", "a1", "a2"],  # level 0
            ["b0", "b1", "b2", "b3"],
            ["c0", "c1"],
            ["d0", "d1"],
            ["e0", "e1"],
            ["f0", "f1"],
            ["g0", "g1"],
        ]
        # id file
        self.id_file = self.tmp / "ids.txt"
        self.cold_items_file = self.tmp / "cold.txt"
        self.item_text_file = self.tmp / "text.txt"
        item_ids = [f"i{i:03d}" for i in range(40)]
        # 30 warm, 10 cold
        cold = set(rng.sample(item_ids, 10))
        with open(self.id_file, "w") as f:
            for iid in item_ids:
                toks = [rng.choice(v) for v in level_vocabs]
                f.write(format_id_line(iid, toks) + "\n")
        with open(self.cold_items_file, "w") as f:
            for iid in sorted(cold):
                f.write(iid + "\n")
        with open(self.item_text_file, "w") as f:
            for iid in item_ids:
                f.write(f"{iid} text for {iid}\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip_write_read_id_file(self):
        id_map = read_id_file(self.id_file)
        out = self.tmp / "roundtrip.txt"
        write_id_file(out, id_map, order_reference=self.id_file)
        id_map2 = read_id_file(out)
        self.assertEqual(id_map, id_map2)

    def test_assign_cold_ids_uses_mlp_and_preserves_warm(self):
        import torch
        from semantic_bridge import build_model
        vocab = build_vocab_from_id_file(self.id_file)

        # Fake embeddings (random) + fake mlp checkpoint
        text_dim = 16
        n_items = 40
        item_ids = [f"i{i:03d}" for i in range(n_items)]
        rng = torch.Generator().manual_seed(0)
        embeddings = torch.randn(n_items, text_dim, generator=rng)
        emb_pt = self.tmp / "emb.pt"
        torch.save({"item_ids": item_ids, "embeddings": embeddings,
                    "model_name": "fake", "text_source_sha256": "n/a"}, emb_pt)

        model = build_model(text_dim, vocab.level_sizes)
        mlp_pt = self.tmp / "mlp.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "text_dim": text_dim,
            "level_sizes": vocab.level_sizes,
            "epoch": 1,
            "val_avg_acc": 0.5,
            "val_acc_per_level": [0.5] * 7,
            "text_source_sha256": "n/a",
            "encoder_model": "fake",
        }, mlp_pt)

        out_id = self.tmp / "merged.txt"
        # Directly call assign main via CLI-style args
        old_argv = sys.argv
        sys.argv = [
            "assign_cold_ids.py",
            "--embeddings", str(emb_pt),
            "--mlp", str(mlp_pt),
            "--source-id-file", str(self.id_file),
            "--cold-items", str(self.cold_items_file),
            "--output-id-file", str(out_id),
            "--device", "cpu",
        ]
        try:
            import assign_cold_ids
            # Reset cached module state if any; just call main.
            assign_cold_ids.main()
        finally:
            sys.argv = old_argv

        # Parse output, verify:
        # - line count matches source
        # - warm items unchanged, cold items = MLP predictions (from vocab tokens)
        src_map = read_id_file(self.id_file)
        out_map = read_id_file(out_id)
        self.assertEqual(set(src_map.keys()), set(out_map.keys()))
        cold = read_item_set(self.cold_items_file)
        for iid in src_map:
            if iid in cold:
                # All 7 tokens should be from the per-level vocab
                for l, t in enumerate(out_map[iid]):
                    self.assertIn(t, vocab.per_level_idx_to_token[l])
            else:
                self.assertEqual(out_map[iid], src_map[iid],
                                 f"warm item {iid} was modified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
