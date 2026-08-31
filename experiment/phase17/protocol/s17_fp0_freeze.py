#!/usr/bin/env python3
"""Freeze official sources, configs, and fold-safe data for Stage17 FP0."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from pathlib import Path
from typing import Any

import yaml

from experiment.phase17.core.fullport_data import (
    build_train_and_internal_dev_examples,
    read_train_prefix_users,
    select_internal_dev_users,
)
from experiment.phase17.core.status_writer import (
    AttemptLedger,
    StatusWriter,
    atomic_json,
    utc_now,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "artifacts/phase17/fullport"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
CONFIG_DIR = OUTPUT_ROOT / "config"
FP0_DIR = OUTPUT_ROOT / "fp0/attempt_001"
STATUS_DIR = ROOT / "artifacts/phase17/status"
ATTEMPT_LEDGER = ROOT / "artifacts/phase17/attempts/S17-FP0.attempts.jsonl"
REPORT_PATH = ROOT / "report/第十七阶段/Stage17_FP0_来源数据与Fidelity冻结报告.md"

EXPERIMENT_ID = "s17_fp0_source_data_fidelity_freeze"
ATTEMPT_ID = "attempt_001"

LATTE_REPOSITORY = "https://github.com/hyp1231/Latte"
LATTE_COMMIT = "05e4e6d983225bcb7172f148a076890e80c524d1"
LATTE_ARCHIVE_SHA256 = "43ead8c1dd7dacf8a06c4bc4b6bce7b7f7645451f3733140f4aada05cf68f242"
SETREC_REPOSITORY = "https://github.com/Linxyhaha/SETRec"
SETREC_COMMIT = "2ed9a75ad1ad3784c61bba3c68cbedbe3cfce2d7"
SETREC_ARCHIVE_SHA256 = "f566902892a98021177bec454d83d97ba72aa1d627f9ed0da01a450801267fc4"

LATTE_FILES = (
    "LICENSE",
    "README.md",
    "genrec/default.yaml",
    "genrec/datasets/AmazonReviews2023/config.yaml",
    "genrec/models/Latte/config.yaml",
    "genrec/models/Latte/model.py",
    "genrec/models/Latte/tokenizer.py",
    "genrec/models/PSID/config.yaml",
    "genrec/models/PSID/model.py",
    "genrec/models/PSID/tokenizer.py",
    "genrec/trainer.py",
    "genrec/evaluator.py",
    "train_latte.py",
    "train_latte.sh",
)
SETREC_FILES = (
    "README.md",
    "requirements.txt",
    "code/scripts/train_t5.sh",
    "code/parse_utils.py",
    "code/finetune_t5.py",
    "code/inference_t5.py",
    "code/model_t5.py",
    "code/Q_t5.py",
    "code/AE/models/ae.py",
    "code/AE/models/layers.py",
    "code/utils/data_utils.py",
    "code/utils/eval_utils.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(source_root: Path, required: tuple[str, ...]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for relative in required:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"official source file is missing: {path}")
        records[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def merged_yaml(*paths: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"YAML config is not an object: {path}")
        merged.update(payload)
    return merged


def require_values(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"{label} official config mismatch: {mismatches}")


def require_patterns(text: str, patterns: dict[str, str], label: str) -> dict[str, bool]:
    checks = {name: re.search(pattern, text, flags=re.MULTILINE) is not None for name, pattern in patterns.items()}
    missing = [name for name, passed in checks.items() if not passed]
    if missing:
        raise AssertionError(f"{label} official source patterns missing: {missing}")
    return checks


def freeze_latte(source_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    default_path = source_root / "genrec/default.yaml"
    dataset_path = source_root / "genrec/datasets/AmazonReviews2023/config.yaml"
    latte_path = source_root / "genrec/models/Latte/config.yaml"
    psid_path = source_root / "genrec/models/PSID/config.yaml"
    latte = merged_yaml(default_path, dataset_path, latte_path)
    psid = merged_yaml(default_path, dataset_path, psid_path)
    expected = {
        "train_batch_size": 256,
        "eval_batch_size": 128,
        "lr": 0.003,
        "weight_decay": 0.05,
        "warmup_steps": 10000,
        "epochs": 150,
        "max_grad_norm": 1.0,
        "eval_interval": 1,
        "patience": 50,
        "sent_emb_model": "sentence-transformers/sentence-t5-base",
        "sent_emb_dim": 768,
        "sent_emb_pca": 192,
        "vq_method": "rqkmeans",
        "vq_n_codebooks": 3,
        "vq_codebook_size": 256,
        "n_latent_tokens": 8,
        "aggregation_method": "agg_max",
        "n_user_tokens": 1,
        "max_item_seq_len": 20,
        "num_beams": 50,
        "num_layers": 4,
        "num_decoder_layers": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "dropout_rate": 0.1,
    }
    require_values(latte, expected, "LATTE")
    quick_start = (source_root / "train_latte.py").read_text(encoding="utf-8")
    quick_checks = require_patterns(
        quick_start,
        {
            "wrapper_eval_interval_3": r"'eval_interval':\s*3",
            "final_beam_500": r"'num_beams':\s*500",
            "final_eval_batch_32": r"'eval_batch_size':\s*32",
            "quick_start_agg_max": r"'aggregation_method':\s*'agg_max'",
        },
        "LATTE quick-start",
    )
    license_path = source_root / "LICENSE"
    if "MIT License" not in license_path.read_text(encoding="utf-8"):
        raise AssertionError("LATTE root LICENSE is not the expected MIT license")

    manifest = {
        "schema_version": "phase17.s17_fp0_source_manifest.v1",
        "source_id": "latte_official",
        "repository_url": LATTE_REPOSITORY,
        "branch": "main",
        "commit": LATTE_COMMIT,
        "archive_url": f"https://codeload.github.com/hyp1231/Latte/tar.gz/{LATTE_COMMIT}",
        "downloaded_archive_sha256": LATTE_ARCHIVE_SHA256,
        "transport": "GitHub codeload snapshot after git GnuTLS failure",
        "license_status": "MIT",
        "license_file": "LICENSE",
        "license_sha256": sha256_file(license_path),
        "reuse_policy": "may adapt with attribution and MIT license preservation",
        "files": source_files(source_root, LATTE_FILES),
        "generated_at": utc_now(),
    }
    resolved = {
        "schema_version": "phase17.s17_fp0_latte_resolved_config.v1",
        "source_commit": LATTE_COMMIT,
        "official_config_primary": {key: latte[key] for key in expected},
        "official_quick_start_overrides": {
            "eval_interval": 3,
            "final_num_beams": 500,
            "final_eval_batch_size": 32,
        },
        "s17_native_matched_protocol": {
            **{key: latte[key] for key in expected},
            "rand_seed": 2023,
            "final_num_beams": 500,
            "final_top_k": 50,
            "eval_interval": 1,
            "external_d0_eval_count": 1,
            "official_test_read": False,
            "sports_read": False,
            "d1_read": False,
            "psid_vq_method_override": "rqkmeans",
        },
        "declared_interface_differences": [
            "Toys Stage17 D0 adapter replaces AmazonReviews2023 last-out adapter",
            "seed 2023 replaces official default seed 2024",
            "config-primary eval_interval=1 is used instead of quick-start wrapper eval_interval=3",
            "Native-PSID is explicitly forced to rqkmeans for a matched LATTE control",
            "official test is replaced by one sealed Stage17 D0 external evaluation",
        ],
        "parity_checks": {**{f"config_{key}": True for key in expected}, **quick_checks},
    }
    fidelity = {
        "schema_version": "phase17.s17_fp0_fidelity_matrix.v1",
        "family": "LATTE",
        "source_commit": LATTE_COMMIT,
        "copy_policy": "MIT_WITH_ATTRIBUTION",
        "existing_s17_2r_is_full": False,
        "components": [
            {"component": "official config and source identity", "required": True, "state": "FROZEN", "evidence": "latte_source_manifest.json"},
            {"component": "SentenceT5-base 768 -> PCA192", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "BGE/fixed feature path is not faithful"},
            {"component": "train-only rqkmeans 3x256", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "scaled MiniBatchKMeans is not faithful"},
            {"component": "PSID conflict-free reassignment", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "collision suffix is forbidden"},
            {"component": "uniform random latent path per exposure", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "fixed/hash root is forbidden"},
            {"component": "latent-conditioned forest decoding", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "scaled contract only"},
            {"component": "item-level agg_max and agg_sum diagnostic", "required": True, "state": "PARITY_TEST_PENDING", "existing_proxy": "reusable only after official parity test"},
            {"component": "150-epoch native budget and final beam500", "required": True, "state": "CONFIG_FROZEN", "existing_proxy": "3k/5-epoch result is not reusable efficacy evidence"},
        ],
        "full_name_allowed_before_components_pass": False,
        "implementation_ready": False,
    }
    return manifest, resolved, fidelity


def freeze_setrec(source_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_script = (source_root / "code/scripts/train_t5.sh").read_text(encoding="utf-8")
    trainer_source = (source_root / "code/finetune_t5.py").read_text(encoding="utf-8")
    model_source = (source_root / "code/model_t5.py").read_text(encoding="utf-8")
    q_source = (source_root / "code/Q_t5.py").read_text(encoding="utf-8")
    script_checks = require_patterns(
        train_script,
        {
            "n_query_is_n_sem_plus_one": r"n_query=\$\(\(n_sem \+ 1\)\)",
            "n_cf_1": r"^n_cf=1$",
            "seed_42": r"^seed=42$",
            "ae_layers": r'AE_layers="512 256 128"',
            "four_gpu_torchrun": r"--nproc_per_node=4",
            "batch_512": r"--batch_size 512",
            "micro_batch_128": r"--micro_batch_size 128",
            "epochs_30": r"--num_epochs 30",
            "cutoff_512": r"--cutoff_len 512",
            "val_2000": r"--val_set_size 2000",
            "cosine": r"--lr_scheduler 'cosine'",
            "warmup_100": r"--warmup_steps 100",
        },
        "SETRec Toys T5 script",
    )
    trainer_checks = require_patterns(
        trainer_source,
        {
            "fp16": r"fp16=True",
            "adamw_torch": r'optim="adamw_torch"',
            "eval_steps_200": r"eval_steps=200",
            "early_stop_patience_10": r"early_stopping_patience=10",
            "history_50": r"SequentialDataset\(args\.data_path, 50",
            "sasrec_item_embed": r"SASRec_item_embed\.pkl",
            "beta_validation_grid": r"\[0\.1,0\.2,0\.3,0\.4,0\.5,0\.6,0\.7,0\.8,0\.9,1\.0\]",
        },
        "SETRec trainer",
    )
    mechanism_checks = {}
    mechanism_checks.update(
        require_patterns(
            model_source,
            {
                "repo_item_group_position_ids": r"torch\.div\(\(vector\[middle_mask\] - i\), n, rounding_mode='floor'\)",
                "continuous_cf_projection": r"self\.input_proj = nn\.Linear",
                "semantic_ae": r"self\.tokenizer = AE",
                "history_identifier_concat": r"torch\.cat\(\[inputs, item_sem_token\]",
                "learnable_query_decode": r"decoder\.query_emb",
                "full_catalog_grounding": r"torch\.bmm\(output_emb, mat\)",
                "weighted_query_sum": r"torch\.sum\(output \* weight, dim=0\)",
                "ae_joint_loss": r"loss \+= self\.alpha \* loss_ae",
            },
            "SETRec model",
        )
    )
    mechanism_checks.update(
        require_patterns(
            q_source,
            {
                "independent_query_mask": r"torch\.eye\(n_query\)",
                "query_embedding": r"self\.query_emb = nn\.Embedding",
            },
            "SETRec query decoder",
        )
    )
    readme = (source_root / "README.md").read_text(encoding="utf-8")
    if (source_root / "LICENSE").exists():
        raise AssertionError("SETRec license policy changed; re-audit before copying source")
    if "NUS ©" not in readme or "NExT++" not in readme:
        raise AssertionError("SETRec README copyright notice is missing")

    manifest = {
        "schema_version": "phase17.s17_fp0_source_manifest.v1",
        "source_id": "setrec_official",
        "repository_url": SETREC_REPOSITORY,
        "branch": "main",
        "commit": SETREC_COMMIT,
        "archive_url": f"https://codeload.github.com/Linxyhaha/SETRec/tar.gz/{SETREC_COMMIT}",
        "downloaded_archive_sha256": SETREC_ARCHIVE_SHA256,
        "transport": "GitHub codeload snapshot after git GnuTLS failure",
        "license_status": "NO_STANDARD_LICENSE_FILE",
        "copyright_notice": "NUS © NExT++",
        "reuse_policy": "read-only semantic audit; project implementation must be clean-room",
        "files": source_files(source_root, SETREC_FILES),
        "generated_at": utc_now(),
    }
    resolved = {
        "schema_version": "phase17.s17_fp0_setrec_resolved_config.v1",
        "source_commit": SETREC_COMMIT,
        "official_toys_t5_command": "bash scripts/train_t5.sh toys 1e-3 4 0.7",
        "official_source_protocol": {
            "base_model": "t5-small local checkpoint",
            "learning_rate": 0.001,
            "n_sem": 4,
            "n_cf": 1,
            "n_query": 5,
            "alpha": 0.7,
            "seed": 42,
            "ae_layers": [512, 256, 128],
            "ae_dropout": 0.0,
            "ae_batch_norm": False,
            "ae_loss": "mse",
            "world_size": 4,
            "global_batch_size": 512,
            "per_device_micro_batch_size": 128,
            "epochs": 30,
            "max_history_items": 50,
            "cutoff_len": 512,
            "validation_users": 2000,
            "scheduler": "cosine",
            "warmup_steps": 100,
            "optimizer": "adamw_torch",
            "weight_decay": 0.0,
            "precision": "fp16",
            "eval_steps": 200,
            "save_steps": 200,
            "early_stopping_patience": 10,
            "beta_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "cf_embedding_dim": 64,
        },
        "s17_clean_room_matched_protocol": {
            "learning_rate": 0.001,
            "n_sem": 4,
            "n_cf": 1,
            "n_query": 5,
            "alpha": 0.7,
            "seed": 2023,
            "ae_layers": [512, 256, 128],
            "global_batch_size": 512,
            "epochs": 30,
            "max_history_items": 20,
            "scheduler": "cosine",
            "warmup_steps": 100,
            "optimizer": "adamw_torch",
            "precision": "fp16",
            "eval_steps": 200,
            "early_stopping_patience": 10,
            "beta_grid_internal_dev_only": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "runtime_world_size": "profile_decided; preserve effective batch 512",
            "full_catalog_grounding": True,
            "history_attention_primary": "paper-faithful sparse visibility",
            "repository_parity_attention": "shared position id within each item; no explicit encoder visibility mask",
            "official_test_read": False,
            "sports_read": False,
            "d1_read": False,
        },
        "declared_interface_differences": [
            "clean-room implementation is mandatory because no standard source license is present",
            "Toys Stage17 D0 adapter replaces bundled validation/testing dictionaries",
            "seed 2023 replaces official Toys script seed 42",
            "max history 20 replaces official source max history 50 for matched Stage17 arms",
            "world size is profile-decided while effective global batch 512 remains locked",
            "beta is selected only on train-prefix internal dev, never on external D0",
            "the paper-faithful sparse history mask and the public-repository shared-position implementation are separate arms",
        ],
        "parity_checks": {**script_checks, **trainer_checks, **mechanism_checks},
    }
    fidelity = {
        "schema_version": "phase17.s17_fp0_fidelity_matrix.v1",
        "family": "SETRec",
        "source_commit": SETREC_COMMIT,
        "copy_policy": "CLEAN_ROOM_ONLY",
        "existing_s17_2r_is_full": False,
        "components": [
            {"component": "official config and source identity", "required": True, "state": "FROZEN", "evidence": "setrec_source_manifest.json"},
            {"component": "fold-train SASRec CF embedding and projection", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "hashed transition code is not faithful"},
            {"component": "continuous semantic AE with four tokens", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "discrete codes are forbidden"},
            {"component": "five-token item history representation", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "permutation set loss is not faithful"},
            {"component": "paper sparse intra-item visibility", "required": True, "state": "PAPER_REQUIRED_REPO_GAP_CONTRACT_PASS", "existing_proxy": "public T5 commit groups position ids but does not explicitly mask encoder visibility"},
            {"component": "public-repository shared-within-item position ids", "required": True, "state": "REPO_PARITY_CONTRACT_PASS", "existing_proxy": "must remain a separate arm from paper-faithful sparse attention"},
            {"component": "five learnable independent queries", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "masked token positions are not faithful"},
            {"component": "per-dimension full-catalog grounding", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "beam candidate scoring is forbidden"},
            {"component": "joint recommendation plus alpha*AE loss", "required": True, "state": "IMPLEMENTATION_PENDING", "existing_proxy": "set permutation CE is not faithful"},
        ],
        "full_name_allowed_before_components_pass": False,
        "implementation_ready": False,
    }
    return manifest, resolved, fidelity


def freeze_data() -> tuple[dict[str, Any], tuple[str, ...]]:
    sequence_path = ROOT / "artifacts/phase17/s0_audit/shadow_data/Toys/D0/user_sequence.txt"
    item_path = ROOT / "GRAM/rec_datasets/Toys/item_plain_text.txt"
    users = read_train_prefix_users(sequence_path, root=ROOT)
    internal_dev_count = max(1, round(len(users) * 0.10))
    internal_dev_ids = select_internal_dev_users(
        users, count=internal_dev_count, seed=2023
    )
    train, internal_dev = build_train_and_internal_dev_examples(
        users, internal_dev_ids, max_history_items=20
    )
    item_ids: set[str] = set()
    for line_number, raw in enumerate(item_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        item_id = raw.split(maxsplit=1)[0]
        if item_id in item_ids:
            raise ValueError(f"duplicate item id at line {line_number}: {item_id}")
        item_ids.add(item_id)
    train_catalog = {item for user in users for item in user.train_items}
    if not train_catalog <= item_ids:
        raise AssertionError("D0 train prefix contains items missing from item text")
    manifest = {
        "schema_version": "phase17.s17_fp0_data_manifest.v1",
        "dataset": "Toys",
        "fold": "D0",
        "purpose": "full-data discovery",
        "sequence_path": str(sequence_path.relative_to(ROOT)),
        "sequence_sha256": sha256_file(sequence_path),
        "users": len(users),
        "train_prefix_item_occurrences": sum(len(user.train_items) for user in users),
        "train_catalog_items": len(train_catalog),
        "item_catalog_path": str(item_path.relative_to(ROOT)),
        "item_catalog_sha256": sha256_file(item_path),
        "item_catalog_items": len(item_ids),
        "rolling_train_examples": len(train),
        "internal_dev_users": len(internal_dev_ids),
        "internal_dev_examples": len(internal_dev),
        "internal_dev_selection": "lowest sha256(s17-fp0-internal-dev:<seed>:<user_id>)",
        "internal_dev_target": "last train-prefix item, position-held-out from that user's supervised examples",
        "external_d0_target_materialized": False,
        "external_d0_eval_count_allowed_after_checkpoint_freeze": 1,
        "official_test_read": False,
        "official_test_values_logged": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "original_monolithic_sequence_read": False,
        "max_history_items": 20,
        "seed": 2023,
        "generated_at": utc_now(),
    }
    return manifest, internal_dev_ids


def write_report(summary: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Stage17 FP0 来源、数据与 Fidelity 冻结报告

## Material Passport

- Step：`S17-FP0`
- Attempt：`{ATTEMPT_ID}`
- Status：`PASS_S17_FP0_SOURCE_DATA_FIDELITY_FREEZE`
- Generated：{summary['generated_at']}
- Scope：正常场景 GRAM；未读取 D1/D2、official test 或 Sports；未启动 GPU 任务

## 1. 冻结结论

LATTE 固定到 commit `{LATTE_COMMIT}`，许可证为 MIT。SETRec 固定到 commit
`{SETREC_COMMIT}`；仓库没有标准 LICENSE 文件，因此后续实现强制 clean-room，
不复制其源码。

现有 S17-2R LATTE/SETRec-style 结果继续作为方向选择证据，但两个实现均明确标记为
`not Full`，不能直接复用为 FP1/FP3 的正式效果结果。

## 2. 配置审计中的关键差异

- LATTE 基础 YAML 的 `eval_interval=1`，官方 quick-start wrapper 会覆盖为 3；
  S17 native matched protocol 采用 config-primary 的每 epoch 评估，并记录该差异。
- Native-PSID 强制使用与 LATTE 相同的 `rqkmeans`，避免把 PSID 默认 OPQ 差异混入因果比较。
- SETRec 官方 Toys T5 脚本为 30 epochs、全局 batch 512、4-GPU torchrun、FP16、
  seed 42、history 50，并在 validation 上搜索 beta。
- SETRec 论文要求 history sparse visibility；固定 public T5 commit 只实现同一 item
  共享 position id，未显式屏蔽同 item token 的 encoder visibility。两者必须分臂。
- S17 clean-room protocol 使用 seed 2023、history 20 和 train-prefix internal dev；
  保留有效 batch 512、30 epochs、五个连续 token、独立 query 与 full-catalog grounding。

## 3. 数据冻结

- Toys D0 full users：{summary['data']['users']}
- rolling train examples：{summary['data']['rolling_train_examples']}
- internal-dev users：{summary['data']['internal_dev_users']}
- item catalog：{summary['data']['item_catalog_items']}
- external D0 target：未 materialize；只允许 family checkpoint 冻结后读取一次

## 4. Gate 与下一步

FP0 来源、配置、许可和数据边界通过。Fidelity matrix 中的实现组件仍为
`IMPLEMENTATION_PENDING`，下一门是实现 full-data adapter tests、LATTE PSID/forest
合同和 SETRec clean-room 连续 token 合同；在这些合同通过前不得启动正式效果实验。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def run(latte_source: Path, setrec_source: Path) -> dict[str, Any]:
    latte_manifest, latte_config, latte_fidelity = freeze_latte(latte_source)
    setrec_manifest, setrec_config, setrec_fidelity = freeze_setrec(setrec_source)
    data_manifest, internal_dev_ids = freeze_data()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    FP0_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        MANIFEST_DIR / "latte_source_manifest.json": latte_manifest,
        MANIFEST_DIR / "latte_fidelity_matrix.json": latte_fidelity,
        MANIFEST_DIR / "setrec_source_manifest.json": setrec_manifest,
        MANIFEST_DIR / "setrec_fidelity_matrix.json": setrec_fidelity,
        MANIFEST_DIR / "data_manifest.json": data_manifest,
        CONFIG_DIR / "latte_native_toys_d0.json": latte_config,
        CONFIG_DIR / "setrec_native_toys_d0.json": setrec_config,
    }
    for path, payload in outputs.items():
        atomic_json(path, payload)
    internal_dev_path = MANIFEST_DIR / "toys_d0_internal_dev_user_ids.txt"
    internal_dev_path.write_text("\n".join(internal_dev_ids) + "\n", encoding="utf-8")

    summary = {
        "schema_version": "phase17.s17_fp0_summary.v1",
        "step_id": "S17-FP0",
        "attempt_id": ATTEMPT_ID,
        "verdict": "PASS_S17_FP0_SOURCE_DATA_FIDELITY_FREEZE",
        "generated_at": utc_now(),
        "latte_commit": LATTE_COMMIT,
        "setrec_commit": SETREC_COMMIT,
        "data": data_manifest,
        "internal_dev_ids_path": str(internal_dev_path.relative_to(ROOT)),
        "internal_dev_ids_sha256": sha256_file(internal_dev_path),
        "outputs": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in outputs
        },
        "effect_experiment_started": False,
        "gpu_used": False,
        "gpu1_touched": False,
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
        "d2_read": False,
        "automatic_retry": False,
        "next_gate": "S17-FP0-IMPLEMENTATION-CONTRACTS",
    }
    atomic_json(FP0_DIR / "summary.json", summary)
    write_report(summary)
    summary["report_path"] = str(REPORT_PATH.relative_to(ROOT))
    summary["report_sha256"] = sha256_file(REPORT_PATH)
    atomic_json(FP0_DIR / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latte-source", type=Path, required=True)
    parser.add_argument("--setrec-source", type=Path, required=True)
    args = parser.parse_args()

    writer = StatusWriter(STATUS_DIR, EXPERIMENT_ID)
    status_path = STATUS_DIR / f"{EXPERIMENT_ID}.status.json"
    if status_path.exists():
        raise FileExistsError(f"FP0 attempt already exists: {status_path}")
    started_at = utc_now()
    writer.initialize(
        step_id="S17-FP0",
        attempt_id=ATTEMPT_ID,
        canonical_result_dir=str(FP0_DIR.relative_to(ROOT)),
        log_path=None,
        extra={
            "d1_read": False,
            "d2_read": False,
            "automatic_retry": False,
            "gpu1_handoff_used": False,
            "gpu1_repeat_restored": None,
        },
    )
    writer.transition(
        "PREFLIGHT",
        "PREFLIGHT",
        "FP0_SOURCE_CONFIG_DATA_AUDIT",
        stage="source_freeze",
        progress={"current": 0, "total": 4, "unit": "contract"},
    )
    ledger = AttemptLedger(ATTEMPT_LEDGER)
    try:
        writer.transition(
            "RUNNING",
            "RUNNING_SCIENTIFIC",
            "FP0_FREEZING",
            process_alive=True,
            workload_pid=0,
            stage="freeze",
            progress={"current": 1, "total": 4, "unit": "contract"},
        )
        summary = run(args.latte_source.resolve(), args.setrec_source.resolve())
        ledger.append(
            {
                "attempt_id": ATTEMPT_ID,
                "step_id": "S17-FP0",
                "kind": "source_data_fidelity_freeze",
                "started_at": started_at,
                "ended_at": utc_now(),
                "state": "COMPLETED",
                "scientific_result_eligible": False,
                "verdict": summary["verdict"],
                "automatic_retry": False,
                "gpu_used": False,
            }
        )
        writer.transition(
            "COMPLETED",
            "SCIENTIFIC_COMPLETED",
            summary["verdict"],
            process_alive=False,
            stage="fp0_complete",
            progress={"current": 4, "total": 4, "unit": "contract"},
            result_selection_eligible=False,
            summary_path=str((FP0_DIR / "summary.json").relative_to(ROOT)),
            summary_sha256=sha256_file(FP0_DIR / "summary.json"),
            report_path=str(REPORT_PATH.relative_to(ROOT)),
            report_sha256=sha256_file(REPORT_PATH),
            next_gate=summary["next_gate"],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "phase17.s17_fp0_failure.v1",
            "step_id": "S17-FP0",
            "attempt_id": ATTEMPT_ID,
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "automatic_retry": False,
            "gpu_used": False,
            "gpu1_touched": False,
        }
        atomic_json(FP0_DIR / "failure.json", failure)
        current = writer.read()
        if current["scientific_state"] in {"PREFLIGHT", "RUNNING"}:
            writer.transition(
                "FAILED",
                "SCIENTIFIC_FAILED",
                "FP0_FREEZE_FAILED",
                process_alive=False,
                failure_path=str((FP0_DIR / "failure.json").relative_to(ROOT)),
                automatic_retry=False,
            )
        ledger.append(
            {
                "attempt_id": ATTEMPT_ID,
                "step_id": "S17-FP0",
                "kind": "source_data_fidelity_freeze",
                "started_at": started_at,
                "ended_at": utc_now(),
                "state": "FAILED",
                "scientific_result_eligible": False,
                "automatic_retry": False,
                "error_type": type(exc).__name__,
            }
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
