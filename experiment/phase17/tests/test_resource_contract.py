from __future__ import annotations

import unittest

from experiment.phase17.core.resource_profiler import (
    GPURecord,
    choose_idle_gpu,
    validate_large_request,
)


class ResourceContractTests(unittest.TestCase):
    def test_gpu_count_uses_explicit_allocation_not_global_ceiling(self) -> None:
        validate_large_request(
            4, 30 * 1024, researcher_allocated_gpu_count=4
        )
        with self.assertRaises(ValueError):
            validate_large_request(
                3, 30 * 1024, researcher_allocated_gpu_count=2
            )
        with self.assertRaises(ValueError):
            validate_large_request(
                2, 31 * 1024, researcher_allocated_gpu_count=2
            )

    def test_idle_selection_prefers_low_utilization_then_memory(self) -> None:
        records = [
            GPURecord(0, "A", 49140, 4000, 45000, 30),
            GPURecord(1, "A", 49140, 20000, 29000, 0),
        ]
        selected = choose_idle_gpu(records, expected_peak_mib=21916, safety_margin_mib=4096)
        self.assertEqual(selected.index, 1)

    def test_no_capacity_returns_none(self) -> None:
        records = [GPURecord(0, "A", 49140, 40000, 9000, 0)]
        self.assertIsNone(choose_idle_gpu(records, expected_peak_mib=21916))
