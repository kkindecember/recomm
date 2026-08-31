"""Read-only GPU admission helpers; never preempt or migrate another process."""

from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import asdict, dataclass


MAX_USABLE_MIB_PER_JOB = 30 * 1024


@dataclass(frozen=True)
class GPURecord:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_percent: int


def parse_gpu_csv(text: str) -> list[GPURecord]:
    rows: list[GPURecord] = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 6:
            continue
        rows.append(
            GPURecord(
                index=int(row[0].strip()),
                name=row[1].strip(),
                total_mib=int(row[2].strip()),
                used_mib=int(row[3].strip()),
                free_mib=int(row[4].strip()),
                utilization_percent=int(row[5].strip()),
            )
        )
    return rows


def query_gpus(timeout_seconds: int = 15) -> list[GPURecord]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
    return parse_gpu_csv(completed.stdout)


def choose_idle_gpu(
    records: list[GPURecord], *, expected_peak_mib: int, safety_margin_mib: int = 4096
) -> GPURecord | None:
    if expected_peak_mib > MAX_USABLE_MIB_PER_JOB:
        raise ValueError("job exceeds the researcher-frozen 30 GiB usable-memory budget")
    eligible = [
        row for row in records if row.free_mib >= expected_peak_mib + safety_margin_mib
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (row.utilization_percent, -row.free_mib, row.index))


def validate_large_request(
    gpu_count: int,
    usable_mib_per_gpu: int,
    *,
    researcher_allocated_gpu_count: int,
) -> None:
    """Validate against the allocation granted for this experiment, not a global cap."""

    if gpu_count < 1:
        raise ValueError("a GPU request must contain at least one device")
    if researcher_allocated_gpu_count < 1:
        raise ValueError("large experiments require an explicit researcher allocation")
    if gpu_count > researcher_allocated_gpu_count:
        raise ValueError("request exceeds the GPU count allocated by the researcher")
    if usable_mib_per_gpu > MAX_USABLE_MIB_PER_JOB:
        raise ValueError("per-GPU usable memory request exceeds 30 GiB")


def snapshot(records: list[GPURecord]) -> list[dict]:
    return [asdict(row) for row in records]
