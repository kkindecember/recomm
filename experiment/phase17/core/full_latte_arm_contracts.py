"""Machine-checkable contracts for Stage17 FP1/FP2 LATTE arms."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ARM_IDS = (
    "N0_NATIVE_PSID",
    "N1_NATIVE_LATTE",
    "G0_GRAM_B0_FRESH",
    "G1_GRAM_PSID_FULL",
    "G2_GRAM_LATTE_FULL",
)


@dataclass(frozen=True)
class ArmDefinition:
    arm_id: str
    step_id: str
    family: str
    implementation: str
    identifier: str
    training_target: str
    decoder: str
    aggregation: str
    latent_tokens: int
    primary_control: str | None
    physical_gpu: int


ARM_DEFINITIONS = {
    "N0_NATIVE_PSID": ArmDefinition(
        arm_id="N0_NATIVE_PSID",
        step_id="S17-FP1",
        family="native",
        implementation="pinned_official_latte_PSID",
        identifier="conflict_free_rqkmeans_psid",
        training_target="sid_plus_eos",
        decoder="single_semantic_tree",
        aggregation="identity_item_resolution",
        latent_tokens=0,
        primary_control=None,
        physical_gpu=4,
    ),
    "N1_NATIVE_LATTE": ArmDefinition(
        arm_id="N1_NATIVE_LATTE",
        step_id="S17-FP1",
        family="native",
        implementation="pinned_official_latte_Latte",
        identifier="same_conflict_free_rqkmeans_psid_as_N0",
        training_target="uniform_random_latent_plus_sid_plus_eos",
        decoder="latent_conditioned_forest",
        aggregation="agg_max",
        latent_tokens=8,
        primary_control="N0_NATIVE_PSID",
        physical_gpu=4,
    ),
    "G0_GRAM_B0_FRESH": ArmDefinition(
        arm_id="G0_GRAM_B0_FRESH",
        step_id="S17-FP2",
        family="gram",
        implementation="project_GRAM_fresh_t5_small",
        identifier="native_lexical_id",
        training_target="lexical_id_plus_eos",
        decoder="lexical_trie",
        aggregation="identity_item_resolution",
        latent_tokens=0,
        primary_control=None,
        physical_gpu=1,
    ),
    "G1_GRAM_PSID_FULL": ArmDefinition(
        arm_id="G1_GRAM_PSID_FULL",
        step_id="S17-FP2",
        family="gram",
        implementation="project_GRAM_fresh_t5_small",
        identifier="conflict_free_rqkmeans_psid",
        training_target="sid_plus_eos",
        decoder="single_semantic_tree",
        aggregation="identity_item_resolution",
        latent_tokens=0,
        primary_control=None,
        physical_gpu=0,
    ),
    "G2_GRAM_LATTE_FULL": ArmDefinition(
        arm_id="G2_GRAM_LATTE_FULL",
        step_id="S17-FP2",
        family="gram",
        implementation="project_GRAM_fresh_t5_small",
        identifier="same_conflict_free_rqkmeans_psid_as_G1",
        training_target="uniform_random_latent_plus_sid_plus_eos",
        decoder="latent_conditioned_forest",
        aggregation="agg_max",
        latent_tokens=8,
        primary_control="G1_GRAM_PSID_FULL",
        physical_gpu=7,
    ),
}


def full_semantic_vocabulary(
    *, codebook_sizes: Sequence[int] = (256, 256, 256), n_latent_tokens: int = 8
) -> tuple[str, ...]:
    """Return the full shared G1/G2 token inventory, including unused codes."""

    if not codebook_sizes or any(size <= 1 for size in codebook_sizes):
        raise ValueError("invalid semantic codebook sizes")
    if n_latent_tokens <= 0:
        raise ValueError("n_latent_tokens must be positive")
    semantic = tuple(
        f"<s17_sid{digit}_{code}>"
        for digit, size in enumerate(codebook_sizes)
        for code in range(size)
    )
    latent = tuple(f"<s17_latent_{index}>" for index in range(n_latent_tokens))
    return semantic + latent


def gram_target_text(
    arm_id: str,
    item_id: str,
    *,
    lexical_ids: Mapping[str, str],
    semantic_codes: Mapping[str, Sequence[int]],
    rng: random.Random,
) -> str:
    """Create one exposure target; G2 samples a fresh latent from the supplied RNG."""

    if arm_id == "G0_GRAM_B0_FRESH":
        return lexical_ids[item_id]
    if arm_id not in {"G1_GRAM_PSID_FULL", "G2_GRAM_LATTE_FULL"}:
        raise ValueError(f"not a GRAM arm: {arm_id}")
    codes = semantic_codes[item_id]
    semantic = " ".join(
        f"<s17_sid{digit}_{int(code)}>" for digit, code in enumerate(codes)
    )
    if arm_id == "G1_GRAM_PSID_FULL":
        return semantic
    latent = rng.randrange(8)
    return f"<s17_latent_{latent}> {semantic}"


def decoder_paths(
    arm_id: str,
    *,
    lexical_ids: Mapping[str, str],
    semantic_codes: Mapping[str, Sequence[int]],
) -> dict[str, tuple[str, ...]]:
    """Enumerate item paths before model-tokenizer integer conversion."""

    if arm_id == "G0_GRAM_B0_FRESH":
        return {item: (lexical_ids[item],) for item in lexical_ids}
    if arm_id == "G1_GRAM_PSID_FULL":
        return {
            item: (
                " ".join(
                    f"<s17_sid{digit}_{int(code)}>" for digit, code in enumerate(codes)
                ),
            )
            for item, codes in semantic_codes.items()
        }
    if arm_id == "G2_GRAM_LATTE_FULL":
        return {
            item: tuple(
                f"<s17_latent_{latent}> "
                + " ".join(
                    f"<s17_sid{digit}_{int(code)}>" for digit, code in enumerate(codes)
                )
                for latent in range(8)
            )
            for item, codes in semantic_codes.items()
        }
    raise ValueError(f"not a GRAM arm: {arm_id}")


def build_preregistered_matrix() -> dict[str, Any]:
    native_training = {
        "seed": 2023,
        "max_history": 20,
        "max_epochs": 150,
        "early_stop_patience": 50,
        "eval_interval_epochs": 1,
        "train_batch_size": 256,
        "eval_batch_size": 128,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.05,
        "warmup_steps": 10000,
        "gradient_clip": 1.0,
    }
    gram_training = {
        "backbone": "t5-small",
        "backbone_path": "artifacts/phase14/m2/pretrained/t5-small",
        "seed": 2023,
        "max_history": 20,
        "max_epochs": 50,
        "minimum_epochs": 20,
        "eval_interval_epochs": 5,
        "early_stop_patience_evaluations": 3,
        "minimum_ndcg10_improvement": 0.0001,
        "train_microbatch": 16,
        "gradient_accumulation": 8,
        "effective_batch": 128,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "warmup_fraction": 0.05,
        "gradient_clip": 1.0,
        "precision": "fp32",
        "fresh_initialization": True,
    }
    return {
        "schema_version": "phase17.s17_fp12_latte_arm_matrix.v1",
        "state": "CONTRACTS_FROZEN_EXECUTORS_PENDING_FULL_TOKENIZER_REQUIRED",
        "dependencies": {
            "full_data_tokenizer": {
                "experiment_id": "s17_fp0_full_data_tokenizer",
                "required_status_code": "PASS_S17_FP0_FULL_DATA_TOKENIZER",
            }
        },
        "data_contract": {
            "train": "D0_train_prefix_rolling_examples",
            "internal_dev": "frozen_train_prefix_position_holdout_1283_users",
            "checkpoint_selection": "internal_dev_ndcg_at_10_only",
            "external_d0": "sealed_until_all_family_checkpoints_frozen_then_once",
            "official_test_read": False,
            "sports_read": False,
            "d1_read": False,
            "d2_read": False,
        },
        "shared_tokenizer_contract": {
            "semantic_id_artifact": "full_data_tokenizer/item_semantic_codes.json",
            "complete_added_vocabulary_artifact": "artifacts/phase17/fullport/fp0/full_data_tokenizer/amendment_001/gram_full_added_tokens.txt",
            "observed_token_export_is_not_the_complete_vocabulary": True,
            "pca_and_rq_fit_scope": "train_prefix_mask_only",
            "g1_g2_full_added_vocabulary_size": 3 * 256 + 8,
            "g1_g2_added_token_initialization_seed": 2023,
            "g1_g2_added_token_inventory_identical": True,
            "n0_n1_semantic_ids_identical": True,
        },
        "inference": {
            "standard_beam": 50,
            "compute_matched_beam": 500,
            "top_k": 50,
            "primary_aggregation": "agg_max",
            "frozen_checkpoint_ablation": "agg_sum",
        },
        "native_training": native_training,
        "gram_training": gram_training,
        "arms": {key: asdict(value) for key, value in ARM_DEFINITIONS.items()},
        "resource_profiles": {
            "state": "BLOCKED_FULL_TOKENIZER_NOT_COMPLETE",
            "profile_order": list(ARM_IDS[2:]) + list(ARM_IDS[:2]),
            "physical_gpu_by_arm": {
                key: value.physical_gpu for key, value in ARM_DEFINITIONS.items()
            },
            "resource_only": True,
            "external_target_materialized": False,
            "effect_metrics_forbidden": True,
            "formal_launch_authorized": False,
            "automatic_retry": False,
            "automatic_process_termination": False,
        },
        "formal_launch_authorized": False,
        "automatic_retry": False,
        "automatic_process_termination": False,
    }


def validate_preregistered_matrix(matrix: Mapping[str, Any]) -> None:
    if tuple(matrix["arms"]) != ARM_IDS:
        raise ValueError("arm order or membership drifted")
    if matrix["formal_launch_authorized"] is not False:
        raise PermissionError("formal FP1/FP2 launch cannot be pre-authorized")
    data = matrix["data_contract"]
    if any(data[key] for key in ("official_test_read", "sports_read", "d1_read", "d2_read")):
        raise PermissionError("sealed evaluation data was enabled")
    token_contract = matrix["shared_tokenizer_contract"]
    if token_contract["g1_g2_full_added_vocabulary_size"] != 776:
        raise ValueError("G1/G2 added vocabulary is not 3x256+8")
    if not token_contract["g1_g2_added_token_inventory_identical"]:
        raise ValueError("G1/G2 must share an identical added token inventory")
    arms = matrix["arms"]
    if arms["N1_NATIVE_LATTE"]["primary_control"] != "N0_NATIVE_PSID":
        raise ValueError("native LATTE causal control drifted")
    if arms["G2_GRAM_LATTE_FULL"]["primary_control"] != "G1_GRAM_PSID_FULL":
        raise ValueError("GRAM LATTE causal control drifted")
    training = matrix["gram_training"]
    if training["train_microbatch"] * training["gradient_accumulation"] != training[
        "effective_batch"
    ]:
        raise ValueError("GRAM effective batch contract drifted")


def load_and_validate_arm_matrix(path: Path) -> dict[str, Any]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    validate_preregistered_matrix(matrix)
    return matrix
