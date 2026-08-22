"""Build Stage15 train+validation-only sequence projections.

The frozen GRAM datasets store train interactions, the validation target, and
the test target in one row.  This tool is the only Stage15 component allowed to
stream that monolithic row.  It deterministically drops the final item without
logging, aggregating, or otherwise using its value.  Model and adapter jobs must
read the generated projection instead of the original sequence file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectionAudit:
    domain: str
    source_path: str
    output_path: str
    source_sha256: str
    output_sha256: str
    rows: int
    projected_items_min: int
    projected_items_max: int
    test_target_retained: bool = False
    discarded_target_logged: bool = False
    discarded_target_aggregated: bool = False
    operation: str = "preserve user_id and original items[:-1]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_within(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root: {relative}") from error
    return candidate


def _resolve_output(output_root: Path, relative: str, label: str) -> Path:
    candidate = (output_root / relative).resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes output root: {relative}") from error
    return candidate


def _stage_projection(
    domain: str,
    source: Path,
    output: Path,
    temporary: Path,
    repo_root: Path,
) -> ProjectionAudit:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{domain}: source must be a regular non-symlink file")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{domain}: refusing to overwrite {output}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"{domain}: temporary path already exists {temporary}")
    if source == output:
        raise ValueError(f"{domain}: source and output must differ")

    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    seen_users: set[str] = set()
    rows = 0
    min_items: int | None = None
    max_items = 0

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with source.open("rb") as input_handle, os.fdopen(
            descriptor, "wb", closefd=True
        ) as output_handle:
            descriptor = -1
            for line_number, raw in enumerate(input_handle, 1):
                source_digest.update(raw)
                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"{domain}:{line_number}: source is not UTF-8"
                    ) from error
                if not line:
                    raise ValueError(f"{domain}:{line_number}: blank row")

                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(
                        f"{domain}:{line_number}: expected user id, at least one "
                        "train item, validation target, and test target"
                    )
                user_id = fields[0]
                if user_id in seen_users:
                    raise ValueError(
                        f"{domain}:{line_number}: duplicate user id {user_id}"
                    )
                seen_users.add(user_id)

                fields.pop()  # Mechanical discard; the removed value is never retained.
                projected_items = len(fields) - 1
                encoded = (" ".join(fields) + "\n").encode("utf-8")
                output_handle.write(encoded)
                output_digest.update(encoded)
                rows += 1
                min_items = (
                    projected_items
                    if min_items is None
                    else min(min_items, projected_items)
                )
                max_items = max(max_items, projected_items)

            if rows == 0:
                raise ValueError(f"{domain}: source contains no rows")
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise

    return ProjectionAudit(
        domain=domain,
        source_path=str(source.relative_to(repo_root)),
        output_path=str(output.relative_to(repo_root)),
        source_sha256=source_digest.hexdigest(),
        output_sha256=output_digest.hexdigest(),
        rows=rows,
        projected_items_min=min_items if min_items is not None else 0,
        projected_items_max=max_items,
    )


def _write_new_json(path: Path, payload: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary path already exists {temporary}")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def run_projection(config_path: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    try:
        config_path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("Config must be inside the repository") from error
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported projection config schema_version")

    output_root = _resolve_within(
        repo_root, config["output_root"], "output_root"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink():
        raise ValueError("output_root must not be a symlink")

    audit_path = _resolve_output(output_root, config["audit_file"], "audit_file")
    if audit_path.exists() or audit_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {audit_path}")

    domains = config.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("Config must contain at least one domain")

    resolved: list[tuple[str, Path, Path, Path]] = []
    seen_domain_names: set[str] = set()
    seen_outputs: set[Path] = set()
    for entry in domains:
        domain = entry["name"]
        if domain in seen_domain_names:
            raise ValueError(f"Duplicate domain {domain}")
        seen_domain_names.add(domain)
        source = _resolve_within(repo_root, entry["source"], f"{domain}.source")
        output = _resolve_output(output_root, entry["output"], f"{domain}.output")
        if output in seen_outputs:
            raise ValueError(f"Duplicate output path {output}")
        seen_outputs.add(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
        resolved.append((domain, source, output, temporary))

    staged: list[tuple[Path, Path]] = []
    audits: list[ProjectionAudit] = []
    try:
        for domain, source, output, temporary in resolved:
            audits.append(
                _stage_projection(domain, source, output, temporary, repo_root)
            )
            staged.append((temporary, output))
        for temporary, output in staged:
            os.replace(temporary, output)
    except Exception:
        for temporary, _output in staged:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        raise

    payload: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "verdict": "PASS_TRAIN_VALIDATION_PROJECTION",
        "operation": "mechanical_per_row_final_item_redaction",
        "raw_monolithic_sequence_streamed_by_redactor": True,
        "test_target_materialized": False,
        "test_target_logged": False,
        "test_target_aggregated": False,
        "test_target_used": False,
        "model_or_adapter_opened_original_sequence": False,
        "domains": [asdict(audit) for audit in audits],
    }
    _write_new_json(audit_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_projection(Path(args.config), Path(args.repo_root))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "verdict": payload["verdict"],
                "domains": [
                    {
                        "domain": domain["domain"],
                        "rows": domain["rows"],
                        "output_sha256": domain["output_sha256"],
                    }
                    for domain in payload["domains"]
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
