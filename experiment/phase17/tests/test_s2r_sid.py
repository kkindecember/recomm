from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment.phase17.core.s2r_sid import (
    SIDSequenceDataset,
    SemanticIDCodec,
    build_examples,
    build_r2_examples,
    build_residual_kmeans_ids,
    build_train_only_cf_codes,
    parse_shadow_sequences,
    read_cohort_user_ids,
    select_r2_early_stop_users,
    tfidf_embeddings,
    train_catalog_items,
)


class S2RSemanticIDTests(unittest.TestCase):
    def shadow_path(self, root: Path) -> Path:
        path = root / "phase17/s2r_preflight/data/Toys/r1_smoke_user_sequence.txt"
        path.parent.mkdir(parents=True)
        path.write_text(
            "u1 i1 i2 i3 i4 i5 i6\n"
            "u2 i7 i8 i9 i10 i11 i12\n"
            "u3 i13 i14 i15 i16 i17 i18\n"
            "u4 i19 i20 i1 i2 i3 i4\n",
            encoding="utf-8",
        )
        return path

    def test_shadow_adapter_never_uses_validation_or_guard_for_training(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-2r-sid-") as temporary:
            users = parse_shadow_sequences(self.shadow_path(Path(temporary)))
        train, validation = build_examples(users, max_history_items=3)
        by_user = {user.user_id: user for user in users}
        self.assertTrue(
            all(row.target != by_user[row.user_id].guard_item for row in train)
        )
        self.assertTrue(
            all(row.target != by_user[row.user_id].validation_target for row in train)
        )
        self.assertEqual(
            {row.user_id: row.target for row in validation},
            {user.user_id: user.validation_target for user in users},
        )
        self.assertTrue(
            all(
                row.history
                == by_user[row.user_id].train_items[-len(row.history) :]
                for row in validation
            )
        )

    def test_tfidf_residual_ids_are_unique_and_train_fit_only(self) -> None:
        item_ids = [f"i{index}" for index in range(1, 21)]
        item_text = {
            item: f"toy category {index % 5} color {index % 3} item {index}"
            for index, item in enumerate(item_ids)
        }
        embeddings = tfidf_embeddings(
            item_ids, item_text, output_dim=8, max_features=64, seed=17
        )
        mapping, summary = build_residual_kmeans_ids(
            item_ids,
            embeddings,
            set(item_ids[:16]),
            n_codebooks=3,
            codebook_size=4,
            seed=17,
        )
        self.assertEqual(len(set(mapping.values())), len(item_ids))
        self.assertEqual(summary.collisions_after_resolution, 0)
        self.assertEqual(summary.fit_items, 16)
        self.assertTrue(summary.train_only_quantizer_fit)

    def test_residual_id_summary_records_frozen_embedding_method(self) -> None:
        item_ids = [f"i{index}" for index in range(12)]
        rng = np.random.RandomState(17)
        embeddings = rng.normal(size=(12, 8)).astype(np.float32)
        _, summary = build_residual_kmeans_ids(
            item_ids,
            embeddings,
            set(item_ids[:10]),
            n_codebooks=2,
            codebook_size=4,
            seed=17,
            embedding_method="frozen_test_encoder_cls_l2",
        )
        self.assertEqual(summary.embedding_method, "frozen_test_encoder_cls_l2")

    def test_collision_suffix_preserves_semantic_digits_and_is_unique(self) -> None:
        item_ids = [f"i{index}" for index in range(6)]
        embeddings = np.ones((6, 4), dtype=np.float32)
        mapping, summary = build_residual_kmeans_ids(
            item_ids,
            embeddings,
            set(item_ids[:4]),
            n_codebooks=2,
            codebook_size=2,
            seed=17,
            collision_resolution="append_group_ordinal",
        )
        self.assertEqual(len(set(mapping.values())), len(item_ids))
        self.assertEqual(len({code[:-1] for code in mapping.values()}), 1)
        self.assertEqual({code[-1] for code in mapping.values()}, set(range(6)))
        self.assertEqual(summary.codebook_sizes, (2, 2, 6))
        self.assertEqual(summary.collision_resolution, "append_group_ordinal")
        self.assertEqual(summary.reassigned_items, 0)

    def test_codec_has_shared_semantic_layout_and_legal_generation(self) -> None:
        mapping = {
            "a": (0, 1, 0),
            "b": (1, 0, 1),
            "c": (1, 1, 0),
        }
        codec = SemanticIDCodec(mapping, [2, 2, 2], n_latent_tokens=3)
        a_tokens = codec.semantic_tokens("a")
        self.assertEqual(codec.decode_semantic_tokens(a_tokens), "a")
        self.assertEqual(
            codec.allowed_generation_tokens([0], latte=False),
            tuple(sorted({codec.semantic_tokens(item)[0] for item in mapping})),
        )
        latent_choices = codec.allowed_generation_tokens([0], latte=True)
        self.assertEqual(len(latent_choices), 3)
        allowed_after_latent = codec.allowed_generation_tokens(
            [0, latent_choices[0]], latte=True
        )
        self.assertEqual(allowed_after_latent, codec.allowed_generation_tokens([0], latte=False))
        self.assertEqual(
            codec.allowed_generation_tokens([0, *a_tokens], latte=False),
            (codec.eos_token,),
        )

    def test_latte_dataset_latent_is_deterministic_per_epoch(self) -> None:
        mapping = {
            "i1": (0, 0),
            "i2": (0, 1),
            "i3": (1, 0),
            "i4": (1, 1),
        }
        codec = SemanticIDCodec(mapping, [2, 2], n_latent_tokens=8)
        from experiment.phase17.core.s2r_sid import SequenceExample

        dataset = SIDSequenceDataset(
            [SequenceExample("u", ("i1", "i2"), "i3")],
            codec,
            latte_training=True,
            seed=2023,
        )
        first = dataset[0]["labels"].numpy().copy()
        self.assertTrue(
            codec.base_latent_token
            <= int(first[0])
            < codec.base_latent_token + codec.n_latent_tokens
        )
        np.testing.assert_array_equal(first, dataset[0]["labels"].numpy())
        dataset.set_epoch(1)
        second = dataset[0]["labels"].numpy()
        self.assertEqual(tuple(first[1:-1]), codec.semantic_tokens("i3"))
        self.assertEqual(tuple(second[1:-1]), codec.semantic_tokens("i3"))

    def test_train_catalog_excludes_validation_and_guard_only_items(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-2r-sid-") as temporary:
            users = parse_shadow_sequences(self.shadow_path(Path(temporary)))
        fit_items = train_catalog_items(users)
        self.assertNotIn("i6", fit_items)
        self.assertNotIn("i12", fit_items)

    def test_cf_codes_fit_training_prefix_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-2r-sid-") as temporary:
            users = parse_shadow_sequences(self.shadow_path(Path(temporary)))
        item_ids = [f"i{index}" for index in range(1, 21)]
        mapping, summary = build_train_only_cf_codes(
            item_ids, users, codebook_size=4, hash_buckets=8, seed=17
        )
        self.assertEqual(set(mapping), set(item_ids))
        self.assertEqual(summary.fit_items, len(train_catalog_items(users)))
        self.assertTrue(summary.train_only_fit)

    def test_r2_internal_targets_are_removed_from_supervised_training(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-2r-sid-") as temporary:
            users = parse_shadow_sequences(self.shadow_path(Path(temporary)))
        selected = select_r2_early_stop_users(users, count=2, seed=17)
        train, early_stop, external = build_r2_examples(users, selected)
        by_user = {user.user_id: user for user in users}
        self.assertEqual(len(early_stop), 2)
        self.assertEqual(len(external), len(users))
        for row in early_stop:
            self.assertEqual(row.target, by_user[row.user_id].train_items[-1])
            self.assertNotIn(
                (row.user_id, row.target),
                {(example.user_id, example.target) for example in train},
            )
        self.assertTrue(
            all(row.target == by_user[row.user_id].validation_target for row in external)
        )

    def test_r2_cohorts_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s17-2r-cohort-") as temporary:
            root = Path(temporary)
            first, second = root / "a.txt", root / "b.txt"
            first.write_text("u1\nu2\n", encoding="utf-8")
            second.write_text("u3\nu4\n", encoding="utf-8")
            cohorts = read_cohort_user_ids([first, second])
            self.assertEqual(cohorts, (("u1", "u2"), ("u3", "u4")))
            second.write_text("u2\nu4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                read_cohort_user_ids([first, second])


if __name__ == "__main__":
    unittest.main()
