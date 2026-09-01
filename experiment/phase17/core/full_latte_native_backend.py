"""Pinned-official PSID/LATTE backend for Stage17 FP1 resource profiles.

The expensive tokenizer fit has already produced the official ``.sem_ids``
cache.  This adapter imports the pinned official classes, injects the audited
D0 train-prefix dataset, and builds representative train/eval tensors without
calling the official external test pipeline.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .full_latte_native_adapter import (
    APPROVED_DEV_IDS_SUFFIX,
    APPROVED_SEQUENCE_SUFFIX,
    DEFAULT_NATIVE_CACHE_SUFFIX,
    make_official_latte_dataset_class,
    read_frozen_internal_dev_ids,
)
from .fullport_data import (
    FullportExample,
    build_train_and_internal_dev_examples,
    read_train_prefix_users,
)


NATIVE_ARMS = ("N0_NATIVE_PSID", "N1_NATIVE_LATTE")
OFFICIAL_SOURCE_SUFFIX = Path(
    "artifacts/phase17/fullport/sources/"
    "latte_05e4e6d983225bcb7172f148a076890e80c524d1_attempt_003"
)
OFFICIAL_COMMIT = "05e4e6d983225bcb7172f148a076890e80c524d1"
SEMANTIC_ID_SUFFIX = Path(
    "artifacts/phase17/fullport/fp0/full_data_tokenizer/attempt_001/"
    "tokenizer/item_semantic_codes.json"
)
MAX_HISTORY_ITEMS = 20


class _NoopAccelerator:
    is_main_process = True


@dataclass(frozen=True)
class OfficialNativeComponents:
    arm_id: str
    config: dict[str, Any]
    dataset: Any
    tokenizer: Any
    model_class: type


def _install_official_source(root: Path) -> Path:
    source = (root.resolve() / OFFICIAL_SOURCE_SUFFIX).resolve()
    if not (source / "genrec/models/PSID/model.py").is_file():
        raise FileNotFoundError(f"pinned official LATTE source is incomplete: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return source


def load_native_examples(
    root: Path,
) -> tuple[list[FullportExample], list[FullportExample]]:
    root = root.resolve()
    users = read_train_prefix_users(root / APPROVED_SEQUENCE_SUFFIX, root=root)
    dev_ids = read_frozen_internal_dev_ids(root / APPROVED_DEV_IDS_SUFFIX, root=root)
    return build_train_and_internal_dev_examples(
        users, dev_ids, max_history_items=MAX_HISTORY_ITEMS
    )


def build_official_native_components(
    root: Path, arm_id: str, *, device: str = "cpu", num_beams: int = 500
) -> OfficialNativeComponents:
    """Instantiate the pinned official config, dataset adapter and tokenizer."""

    if arm_id not in NATIVE_ARMS:
        raise ValueError(f"not a Stage17 native arm: {arm_id}")
    root = root.resolve()
    _install_official_source(root)
    from genrec.utils import get_config, get_model, get_tokenizer

    model_name = "PSID" if arm_id == "N0_NATIVE_PSID" else "Latte"
    overrides = {
        "rand_seed": 2023,
        "reproducibility": True,
        "metadata": "sentence",
        "sent_emb_model": "sentence-transformers/sentence-t5-base",
        "sent_emb_dim": 768,
        "sent_emb_pca": 192,
        "sent_emb_batch_size": 32,
        "vq_method": "rqkmeans",
        "vq_n_codebooks": 3,
        "vq_codebook_size": 256,
        "vq_permutation": 0,
        "faiss_omp_num_threads": 32,
        "n_user_tokens": 1,
        "n_latent_tokens": 8,
        "aggregation_method": "agg_max",
        "max_item_seq_len": MAX_HISTORY_ITEMS,
        "num_beams": int(num_beams),
        "train_batch_size": 256,
        "eval_batch_size": 128,
        "lr": 0.003,
        "weight_decay": 0.05,
        "warmup_steps": 10000,
        "epochs": 150,
        "eval_interval": 1,
        "patience": 50,
        "max_grad_norm": 1.0,
        "topk": [50],
        "metrics": ["ndcg", "recall"],
        "val_metric": "ndcg@10",
        "num_proc": 1,
        "device": device,
        "accelerator": _NoopAccelerator(),
        "stage17_native_cache_dir": str(root / DEFAULT_NATIVE_CACHE_SUFFIX),
        "external_target_authorized": False,
    }
    config = get_config(
        model_name=model_name,
        dataset_name="AmazonReviews2023",
        config_file=None,
        config_dict=overrides,
    )
    # ``get_config`` intentionally does not serialize runtime objects; restore
    # the local logger shim after its scalar conversion pass.
    config["accelerator"] = overrides["accelerator"]
    config["device"] = device
    adapter_class = make_official_latte_dataset_class(root=root)
    dataset = adapter_class(config)
    tokenizer = get_tokenizer(model_name)(config, dataset)
    model_class = get_model(model_name)
    return OfficialNativeComponents(
        arm_id=arm_id,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
        model_class=model_class,
    )


def _raw_semantic_from_tokenizer(components: OfficialNativeComponents) -> dict[str, tuple[int, ...]]:
    tokenizer = components.tokenizer
    if components.arm_id == "N0_NATIVE_PSID":
        offsets = (1, 257, 513)
    else:
        offsets = (9, 265, 521)
    raw: dict[str, tuple[int, ...]] = {}
    for item, tokens in tokenizer.item2tokens.items():
        if len(tokens) != 3:
            raise ValueError(f"official tokenizer emitted non-3-digit SID for {item}")
        raw[item] = tuple(int(token) - offsets[digit] for digit, token in enumerate(tokens))
    return raw


def _native_tokenize_once(tokenizer, example: FullportExample):
    return tokenizer._tokenize_once(
        {
            "user": example.user_id,
            "item_seq": list(example.history) + [example.target],
        }
    )


def collate_native_train_batch(
    components: OfficialNativeComponents, examples: Sequence[FullportExample]
) -> dict[str, Any]:
    import torch

    if not examples:
        raise ValueError("native profile train batch cannot be empty")
    tokenizer = components.tokenizer
    tokenized = [_native_tokenize_once(tokenizer, example) for example in examples]
    if components.arm_id == "N1_NATIVE_LATTE":
        rows = [
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "target_item": example.target,
            }
            for example, (input_ids, attention_mask, _labels) in zip(examples, tokenized)
        ]
        return tokenizer.collate_fn_train(rows)
    return {
        "input_ids": torch.tensor([row[0] for row in tokenized], dtype=torch.long),
        "attention_mask": torch.tensor([row[1] for row in tokenized], dtype=torch.long),
        "labels": torch.tensor([row[2] for row in tokenized], dtype=torch.long),
    }


def collate_native_eval_batch(
    components: OfficialNativeComponents, examples: Sequence[FullportExample]
) -> dict[str, Any]:
    import torch

    if not examples:
        raise ValueError("native profile eval batch cannot be empty")
    tokenized = [_native_tokenize_once(components.tokenizer, example) for example in examples]
    return {
        "input_ids": torch.tensor([row[0] for row in tokenized], dtype=torch.long),
        "attention_mask": torch.tensor([row[1] for row in tokenized], dtype=torch.long),
        "labels": torch.tensor([row[2] for row in tokenized], dtype=torch.long),
    }


def create_fresh_official_native_model(
    components: OfficialNativeComponents, *, seed: int = 2023
):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return components.model_class(
        components.config, components.dataset, components.tokenizer
    )


def cpu_preflight_native_arm(root: Path, arm_id: str) -> dict[str, Any]:
    """Validate the exact official adapter/tokenizer/model-class contract on CPU."""

    import torch

    root = root.resolve()
    components = build_official_native_components(root, arm_id, device="cpu")
    train, internal_dev = load_native_examples(root)
    longest_train = max(train, key=lambda example: len(example.history))
    batch_examples = (train[0], longest_train, train[-1])
    torch.manual_seed(2023)
    train_batch = collate_native_train_batch(components, batch_examples)
    eval_batch = collate_native_eval_batch(components, (internal_dev[0],))

    frozen = {
        item: tuple(int(code) for code in codes)
        for item, codes in json.loads(
            (root / SEMANTIC_ID_SUFFIX).read_text(encoding="utf-8")
        ).items()
    }
    official_raw = _raw_semantic_from_tokenizer(components)
    if official_raw != frozen:
        raise RuntimeError("official native tokenizer does not reproduce the frozen PSID cache")
    expected_vocab = 771 if arm_id == "N0_NATIVE_PSID" else 779
    expected_train_target = 4 if arm_id == "N0_NATIVE_PSID" else 5
    if components.tokenizer.vocab_size != expected_vocab:
        raise AssertionError("official native vocabulary size drifted")
    if train_batch["labels"].shape[1] != expected_train_target:
        raise AssertionError("official native training target length drifted")
    if eval_batch["labels"].shape[1] != 4:
        raise AssertionError("official native eval target must omit the latent root")
    latent_coverage: list[int] = []
    if arm_id == "N1_NATIVE_LATTE":
        rows = []
        for _ in range(128):
            rows.extend(batch_examples)
        sampled = collate_native_train_batch(components, rows)["labels"][:, 0]
        latent_coverage = sorted(set(int(value) for value in sampled.tolist()))
        if latent_coverage != list(range(1, 9)):
            raise AssertionError("official LATTE dynamic collator did not cover all latent roots")
    return {
        "arm_id": arm_id,
        "state": "PASS_CPU_PREFLIGHT",
        "official_commit": OFFICIAL_COMMIT,
        "official_model_module": components.model_class.__module__,
        "official_tokenizer_module": components.tokenizer.__class__.__module__,
        "catalog_items": len(frozen),
        "rolling_train_examples": len(train),
        "internal_dev_examples": len(internal_dev),
        "input_token_length": int(train_batch["input_ids"].shape[1]),
        "train_target_token_length": int(train_batch["labels"].shape[1]),
        "eval_target_token_length": int(eval_batch["labels"].shape[1]),
        "vocab_size": int(components.tokenizer.vocab_size),
        "latent_token_ids_observed": latent_coverage,
        "semantic_ids_identical_to_frozen_cache": True,
        "external_target_materialized": False,
        "effect_metrics_computed": False,
        "test_role": components.dataset.stage17_split_roles["test"],
        "test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "gpu_used": False,
    }
