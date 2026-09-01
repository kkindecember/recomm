from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiment.phase17.core.full_latte_contracts import PSIDResolutionSummary
from experiment.phase17.core.full_latte_native_adapter import (
    LatteNativeDataBundle,
    LatteNativeSequence,
)
from experiment.phase17.core.full_latte_tokenizer import (
    OFFICIAL_SEM_IDS_FILENAME,
    FullLatteTokenizerSpec,
    build_tokenizer_fit_mask,
    fit_pca_train_only,
    semantic_token_strings,
    write_tokenizer_artifacts,
)


def fake_bundle() -> LatteNativeDataBundle:
    return LatteNativeDataBundle(
        id_mapping={"id2item": ["[PAD]", "a", "b", "c"]},
        item2meta={"a": "A", "b": "B", "c": "C"},
        train_sequences=(LatteNativeSequence("u", ("a", "b")),),
        internal_dev_sequences=(),
        all_item_seqs={"u": ("a", "b")},
        internal_dev_user_ids=(),
        rolling_train_examples=1,
        train_catalog_items=2,
        tokenizer_fit_catalog_items=2,
        catalog_items=3,
    )


class RecordingPCA:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, values):
        self.fit_values = values.copy()
        self.components_ = np.eye(2, 3)
        self.mean_ = values.mean(axis=0)
        self.explained_variance_ = np.ones(2)
        self.explained_variance_ratio_ = np.ones(2) / 2
        self.singular_values_ = np.ones(2)
        return self

    def transform(self, values):
        return values[:, :2]


class FullLatteTokenizerTests(unittest.TestCase):
    def test_fit_mask_excludes_catalog_only_item(self) -> None:
        mask = build_tokenizer_fit_mask(("a", "b", "c"), fake_bundle())
        np.testing.assert_array_equal(mask, np.asarray([True, True, False]))

    def test_pca_fit_receives_only_masked_rows_but_transforms_all(self) -> None:
        values = np.arange(12, dtype=np.float32).reshape(4, 3)
        transformed, pca = fit_pca_train_only(
            values,
            np.asarray([True, False, True, False]),
            n_components=2,
            seed=2023,
            pca_factory=RecordingPCA,
        )
        np.testing.assert_array_equal(pca.fit_values, values[[0, 2]])
        np.testing.assert_array_equal(transformed, values[:, :2])
        self.assertTrue(pca.kwargs["whiten"])

    def test_semantic_tokens_are_position_specific_and_latents_are_shared(self) -> None:
        mapping, vocabulary = semantic_token_strings(
            {"a": (1, 2, 3), "b": (2, 2, 3)}, n_latent_tokens=2
        )
        self.assertEqual(mapping["a"], "<s17_sid0_1> <s17_sid1_2> <s17_sid2_3>")
        self.assertIn("<s17_latent_0>", vocabulary)
        self.assertIn("<s17_sid0_2>", vocabulary)
        self.assertIn("<s17_sid1_2>", vocabulary)

    def test_export_matches_official_cache_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "attempt"
            cache = base / "native_cache"
            pca = SimpleNamespace(
                components_=np.eye(2, 3),
                mean_=np.zeros(3),
                explained_variance_=np.ones(2),
                explained_variance_ratio_=np.ones(2) / 2,
                singular_values_=np.ones(2),
            )
            summary = PSIDResolutionSummary(
                catalog_items=3,
                n_digit=3,
                codebook_size=256,
                collision_groups=0,
                collisions_before=0,
                collisions_after=0,
                reassigned_items=0,
                top_k_per_digit=5,
            )
            codes = {"a": (1, 2, 3), "b": (2, 3, 4), "c": (3, 4, 5)}
            manifest = write_tokenizer_artifacts(
                output_dir=output,
                official_cache_dir=cache,
                catalog_items=("a", "b", "c"),
                fit_mask=np.asarray([True, True, False]),
                sentence_embeddings=np.ones((3, 3), dtype=np.float32),
                transformed_embeddings=np.ones((3, 2), dtype=np.float32),
                pca=pca,
                raw_codes=np.asarray(list(codes.values()), dtype=np.int64),
                centroids=np.ones((3, 256, 2), dtype=np.float32),
                resolved_codes=codes,
                resolution=summary,
                spec=FullLatteTokenizerSpec(sentence_embedding_dim=3, pca_components=2),
                provenance={"unit_test": True},
            )
            official = cache / "processed" / OFFICIAL_SEM_IDS_FILENAME
            self.assertEqual(json.loads(official.read_text()), {k: list(v) for k, v in codes.items()})
            self.assertEqual(manifest["fit_catalog_items"], 2)
            self.assertTrue(manifest["official_cache_prevents_unmasked_pca_refit"])
            self.assertFalse(manifest["external_target_materialized"])


if __name__ == "__main__":
    unittest.main()
