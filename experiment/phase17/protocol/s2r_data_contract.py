from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT / "artifacts/phase17/s0_audit/shadow_data/Toys/D0/user_sequence.txt"
)
DEFAULT_ITEM_TEXT_SOURCE = ROOT / "GRAM/rec_datasets/Toys/item_plain_text.txt"
DEFAULT_OUTPUT = ROOT / "artifacts/phase17/s2r_preflight/data/Toys_s17_d0_3000"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(user_id: str, seed: int) -> str:
    return hashlib.sha256(f"s17-2r:{seed}:{user_id}".encode("utf-8")).hexdigest()


def validate_source(path: Path) -> None:
    resolved = path.resolve()
    allowed = DEFAULT_SOURCE.resolve()
    if resolved != allowed:
        raise ValueError(
            "S17-2R cohort construction may read only the sealed Toys D0 shadow "
            f"projection, got {resolved}"
        )
    if "GRAM/rec_datasets" in resolved.as_posix():
        raise ValueError("original monolithic sequence files are forbidden")


def parse_rows(lines: Iterable[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(lines, start=1):
        normalized = " ".join(raw.strip().split())
        if not normalized:
            continue
        fields = normalized.split(" ")
        if len(fields) < 4:
            raise ValueError(
                f"line {line_number} has no train item + shadow target + guard item"
            )
        user_id = fields[0]
        if user_id in seen:
            raise ValueError(f"duplicate user id: {user_id}")
        seen.add(user_id)
        rows.append((user_id, normalized))
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_cohorts(
    source: Path,
    output: Path,
    *,
    item_text_source: Path = DEFAULT_ITEM_TEXT_SOURCE,
    seed: int = 2023,
    cohort_count: int = 3,
    users_per_cohort: int = 1000,
) -> dict:
    validate_source(source)
    if item_text_source.resolve() != DEFAULT_ITEM_TEXT_SOURCE.resolve():
        raise ValueError("S17-2R must use the frozen Toys item-text catalog")
    rows = parse_rows(source.read_text(encoding="utf-8").splitlines())
    requested = cohort_count * users_per_cohort
    if len(rows) < requested:
        raise ValueError(f"requested {requested} users from only {len(rows)} rows")

    selected = sorted(rows, key=lambda row: (stable_key(row[0], seed), row[0]))[
        :requested
    ]
    selected_user_ids = [row[0] for row in selected]
    if len(set(selected_user_ids)) != requested:
        raise AssertionError("selected users are not unique")

    combined_path = output / "user_sequence.txt"
    atomic_write(combined_path, "\n".join(row[1] for row in selected) + "\n")

    r1_rows = selected[: min(100, len(selected))]
    r1_path = output / "r1_smoke_user_sequence.txt"
    atomic_write(r1_path, "\n".join(row[1] for row in r1_rows) + "\n")

    # Item metadata has no held-out interaction labels, but projecting it here
    # prevents later runtimes from reopening the monolithic dataset directory.
    item_text_path = output / "item_plain_text.txt"
    item_text_payload = item_text_source.read_text(encoding="utf-8")
    atomic_write(item_text_path, item_text_payload)

    cohort_entries = []
    for index in range(cohort_count):
        start = index * users_per_cohort
        end = start + users_per_cohort
        cohort = selected[start:end]
        cohort_path = output / f"eval_cohort_c{index}_user_ids.txt"
        atomic_write(cohort_path, "\n".join(row[0] for row in cohort) + "\n")
        cohort_entries.append(
            {
                "cohort_id": f"Toys_s17_d0_eval_c{index}_1000",
                "path": cohort_path.relative_to(ROOT).as_posix(),
                "users": len(cohort),
                "sha256": sha256_file(cohort_path),
            }
        )

    manifest = {
        "schema_version": "phase17.s17_2r_data_contract.v1",
        "step_id": "S17-2R",
        "seed": seed,
        "selection": "lowest sha256(s17-2r:<seed>:<user_id>) values",
        "source": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(source),
        "source_users": len(rows),
        "selected_dataset": "Toys_s17_d0_3000",
        "selected_path": combined_path.relative_to(ROOT).as_posix(),
        "selected_sha256": sha256_file(combined_path),
        "selected_users": requested,
        "r1_smoke_dataset": "Toys_s17_d0_r1_100",
        "r1_smoke_path": r1_path.relative_to(ROOT).as_posix(),
        "r1_smoke_sha256": sha256_file(r1_path),
        "r1_smoke_users": len(r1_rows),
        "item_catalog_source": item_text_source.relative_to(ROOT).as_posix(),
        "item_catalog_source_sha256": sha256_file(item_text_source),
        "item_catalog_path": item_text_path.relative_to(ROOT).as_posix(),
        "item_catalog_sha256": sha256_file(item_text_path),
        "item_catalog_items": sum(
            1 for line in item_text_payload.splitlines() if line.strip()
        ),
        "evaluation_cohorts": cohort_entries,
        "cohorts_disjoint": len(set(selected_user_ids)) == requested,
        "official_test_read": False,
        "sports_read": False,
        "d1_read": False,
        "formal_result_eligible": False,
        "purpose": "S17-2R architecture screening data contract",
    }
    atomic_write(
        output / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--item-text-source", type=Path, default=DEFAULT_ITEM_TEXT_SOURCE)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    manifest = build_cohorts(
        args.source,
        args.output,
        item_text_source=args.item_text_source,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
