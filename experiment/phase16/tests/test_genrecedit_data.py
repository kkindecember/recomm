from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from genrecedit_data import (  # noqa: E402
    DatasetInputs,
    TrainOccurrence,
    build_sharded_dataset,
    choose_occurrence,
    collect_train_occurrences,
    covariance_position_coverage,
    deterministic_topk,
    deterministic_long_path_resource_subset,
    sha256_file,
    stable_sha256,
)


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


class GenRecEditDataTests(unittest.TestCase):
    def test_occurrences_are_retained_warm_and_use_only_preceding_history(self) -> None:
        rows = [("u", ("w0", "w1", "w2", "w1"))]
        occurrences = collect_train_occurrences(
            rows, retained_warm={"w0", "w1", "w2"}, max_history=2
        )
        self.assertEqual(
            occurrences["w1"],
            (
                TrainOccurrence("u", 1, ("w0",)),
                TrainOccurrence("u", 3, ("w1", "w2")),
            ),
        )
        with self.assertRaisesRegex(ValueError, "Non-retained-warm"):
            collect_train_occurrences(
                [("u", ("w0", "pseudo"))], retained_warm={"w0"}, max_history=2
            )

    def test_occurrence_choice_and_topk_are_deterministic(self) -> None:
        rows = (
            TrainOccurrence("u2", 2, ("b",)),
            TrainOccurrence("u1", 1, ("a",)),
        )
        first = choose_occurrence(rows, cold_item="c", warm_item="w", seed=1502)
        second = choose_occurrence(tuple(reversed(rows)), cold_item="c", warm_item="w", seed=1502)
        self.assertEqual(first, second)
        chosen = deterministic_topk(torch.tensor([0.5, 0.7, 0.5]), ["z", "m", "a"], 2)
        self.assertEqual([index for index, _ in chosen], [1, 2])

    def test_covariance_iterator_covers_every_legal_position_and_long_path(self) -> None:
        rows = [("u", ("w0", "w1", "w2"))]
        paths = {"w0": ("a",), "w1": tuple("abcdef"), "w2": ("g",)}
        coverage = covariance_position_coverage(
            rows, paths=paths, retained_warm=set(paths), max_history=3
        )
        self.assertEqual(coverage, {0: 2, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
        subset = deterministic_long_path_resource_subset(
            rows,
            paths=paths,
            retained_warm=set(paths),
            max_history=3,
            lexical_position=5,
            minimum_rows=1,
        )
        self.assertEqual(len(subset), 1)
        self.assertEqual(subset[0]["lexical_position"], 5)

    def _fixture(self, root: Path, *, leak_item: str | None = None) -> DatasetInputs:
        train = root / "s16" / "student_readable" / "interaction_train_sequences.jsonl"
        train.parent.mkdir(parents=True)
        train_rows = [
            {"user_id": "u1", "items": ["w0", "w1", "w2"]},
            {"user_id": "u2", "items": ["w2", "w1", "w0"]},
        ]
        if leak_item is not None:
            train_rows[0]["items"][1] = leak_item
        train.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in train_rows),
            encoding="utf-8",
        )
        retained = root / "s16" / "retained_warm_items.txt"
        pseudo = root / "s16" / "pseudo_cold_items.txt"
        cold = root / "frozen" / "cold_items.txt"
        paths = root / "frozen" / "paths.txt"
        metadata = root / "frozen" / "metadata.txt"
        embedding = root / "frozen" / "embeddings.pt"
        _write_lines(retained, ["w0", "w1", "w2"])
        _write_lines(pseudo, ["p"])
        _write_lines(cold, ["c1", "c2"])
        catalog_paths = {
            "w0": ("w0a",),
            "w1": ("w1a", "w1b", "w1c", "w1d", "w1e", "w1f"),
            "w2": ("w2a",),
            "p": ("pa",),
            "c1": ("c1a", "c1b"),
            "c2": ("c2a",),
        }
        _write_lines(paths, [f"{item} " + "".join(f"|{token}" for token in path) for item, path in catalog_paths.items()])
        _write_lines(metadata, [f"{item} title: {item}" for item in catalog_paths])
        item_ids = list(catalog_paths)
        vectors = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        torch.save(
            {
                "item_ids": item_ids,
                "embeddings": vectors,
                "model_name": "fixture",
                "pooling": "cls",
                "l2_normalized": True,
                "text_source_sha256": sha256_file(metadata),
            },
            embedding,
        )
        return DatasetInputs(
            train_sequences=train,
            retained_warm_items=retained,
            pseudo_cold_items=pseudo,
            real_cold_items=cold,
            lexical_paths=paths,
            metadata=metadata,
            content_embeddings=embedding,
        )

    def test_small_fixture_builds_full_positions_and_resumes_stably(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._fixture(root)
            output = root / "output"
            first = build_sharded_dataset(
                inputs,
                output,
                seed=1502,
                contexts_per_target=2,
                max_history=2,
                target_shard_size=1,
                similarity_batch_size=1,
                required_counts={"targets": 2, "contexts": 4, "requests": 6},
                required_covariance_position_counts={0: 4, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2},
                minimum_long_path_resource_rows=2,
            )
            before = {
                path.relative_to(output): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.rglob("*.jsonl")
            }
            second = build_sharded_dataset(
                inputs,
                output,
                seed=1502,
                contexts_per_target=2,
                max_history=2,
                target_shard_size=1,
                similarity_batch_size=1,
                required_counts={"targets": 2, "contexts": 4, "requests": 6},
                required_covariance_position_counts={0: 4, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2},
                minimum_long_path_resource_rows=2,
            )
            after = {
                path.relative_to(output): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.rglob("*.jsonl")
            }
            self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
            self.assertEqual(before, after)
            self.assertEqual(
                first["resume_contract"]["checkpoint_sha256"],
                sha256_file(output / "checkpoint_manifest.json"),
            )
            self.assertEqual(first["counts"], {"targets": 2, "contexts": 4, "requests": 6})
            requests = []
            for path in sorted((output / "position_requests").glob("*.jsonl")):
                requests.extend(json.loads(line) for line in path.read_text().splitlines())
            c1 = [row for row in requests if row["cold_item"] == "c1"]
            self.assertEqual(len(c1), 4)
            self.assertEqual({row["position"] for row in c1}, {0, 1})
            self.assertTrue(all(row["train_context_items"] for row in requests))
            self.assertTrue(all(row["source_warm_item"] in {"w0", "w1", "w2"} for row in requests))
            self.assertTrue(all(row["full_target_path"] for row in requests))
            self.assertEqual(first["leakage_audit"]["test_occurrence_files_opened"], 0)
            stable_payload = dict(first)
            recorded_stable_sha = stable_payload.pop("stable_manifest_payload_sha256")
            self.assertEqual(recorded_stable_sha, stable_sha256(stable_payload))
            self.assertEqual(first["covariance"]["resource_subset"]["rows"], 2)

    def test_resume_rejects_checkpoint_target_slice_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._fixture(root)
            output = root / "output"
            kwargs = {
                "contexts_per_target": 2,
                "max_history": 2,
                "target_shard_size": 1,
                "similarity_batch_size": 1,
                "required_counts": {"targets": 2, "contexts": 4, "requests": 6},
                "required_covariance_position_counts": {0: 4, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2},
                "minimum_long_path_resource_rows": 2,
            }
            build_sharded_dataset(inputs, output, **kwargs)
            (output / "manifest.json").unlink()
            checkpoint_path = output / "checkpoint_manifest.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["completed_shards"][0]["first_target"] = "tampered"
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target/count contract drift"):
                build_sharded_dataset(inputs, output, **kwargs)

    def test_pseudo_cold_or_real_cold_train_occurrence_hard_fails(self) -> None:
        for leaked in ("p", "c1"):
            with self.subTest(leaked=leaked), tempfile.TemporaryDirectory() as directory:
                inputs = self._fixture(Path(directory), leak_item=leaked)
                with self.assertRaisesRegex(ValueError, "Student-readable train leakage"):
                    build_sharded_dataset(
                        inputs,
                        Path(directory) / "output",
                        contexts_per_target=2,
                        required_counts={"targets": 2, "contexts": 4, "requests": 6},
                    )

    def test_validation_named_occurrence_source_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._fixture(Path(directory))
            bad = Path(directory) / "validation" / "interaction_train_sequences.jsonl"
            bad.parent.mkdir()
            bad.write_text(inputs.train_sequences.read_text(), encoding="utf-8")
            bad_inputs = DatasetInputs(**{**inputs.__dict__, "train_sequences": bad})
            with self.assertRaisesRegex(ValueError, "Forbidden occurrence-source path"):
                build_sharded_dataset(
                    bad_inputs,
                    Path(directory) / "out",
                    contexts_per_target=2,
                )


if __name__ == "__main__":
    unittest.main()
