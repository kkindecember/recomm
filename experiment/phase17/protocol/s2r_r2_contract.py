from __future__ import annotations

import json
from pathlib import Path

from experiment.phase17.core.s2r_sid import (
    build_r2_examples,
    parse_shadow_sequences,
    read_cohort_user_ids,
    select_r2_early_stop_users,
    sha256_file,
)
from experiment.phase17.core.status_writer import atomic_json, utc_now


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "artifacts/phase17/s2r_preflight/data/Toys_s17_d0_3000"
SEQUENCE_PATH = DATA_DIR / "user_sequence.txt"
COHORT_PATHS = tuple(
    DATA_DIR / f"eval_cohort_c{index}_user_ids.txt" for index in range(3)
)
OUTPUT_DIR = ROOT / "artifacts/phase17/s2r_preflight/r2_contract"


def prepare_r2_contract(*, seed: int = 2023, early_stop_count: int = 300) -> dict:
    users = parse_shadow_sequences(SEQUENCE_PATH)
    early_stop_ids = select_r2_early_stop_users(
        users, count=early_stop_count, seed=seed
    )
    train, early_stop, external = build_r2_examples(users, early_stop_ids)
    cohorts = read_cohort_user_ids(COHORT_PATHS)
    all_users = {user.user_id for user in users}
    cohort_union = {user_id for cohort in cohorts for user_id in cohort}
    if cohort_union != all_users:
        raise AssertionError("R2 evaluation cohorts do not partition the selected 3k users")
    external_by_user = {row.user_id: row for row in external}
    if set(external_by_user) != all_users:
        raise AssertionError("R2 external examples do not cover every selected user")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    early_stop_path = OUTPUT_DIR / "early_stop_user_ids.txt"
    early_stop_path.write_text("\n".join(early_stop_ids) + "\n", encoding="utf-8")
    payload = {
        "schema_version": "phase17.s17_2r_r2_contract.v1",
        "step_id": "S17-2R",
        "gate": "R2",
        "seed": seed,
        "sequence_path": str(SEQUENCE_PATH.relative_to(ROOT)),
        "sequence_sha256": sha256_file(SEQUENCE_PATH),
        "selected_users": len(users),
        "supervised_train_examples": len(train),
        "internal_early_stop_examples": len(early_stop),
        "external_evaluation_examples": len(external),
        "early_stop_user_ids_path": str(early_stop_path.relative_to(ROOT)),
        "early_stop_user_ids_sha256": sha256_file(early_stop_path),
        "early_stop_target_position": "last train-prefix item",
        "early_stop_target_removed_from_selected_user_supervised_examples": True,
        "external_target_position": "shadow validation target",
        "external_target_read_during_early_stop": False,
        "cohorts": [
            {
                "cohort_id": f"c{index}",
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "users": len(cohort),
            }
            for index, (path, cohort) in enumerate(zip(COHORT_PATHS, cohorts))
        ],
        "cohorts_disjoint": sum(len(cohort) for cohort in cohorts)
        == len(cohort_union),
        "cohorts_partition_selected_users": cohort_union == all_users,
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
        "created_at": utc_now(),
    }
    atomic_json(OUTPUT_DIR / "manifest.json", payload)
    return payload


def main() -> None:
    print(json.dumps(prepare_r2_contract(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
