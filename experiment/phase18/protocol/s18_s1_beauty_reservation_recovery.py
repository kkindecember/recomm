#!/usr/bin/env python3
"""Run-0007: retain a preclaimed CUDA allocator pool for Beauty diagnostics."""

from pathlib import Path

from experiment.phase18.protocol import s18_s1_beauty_recovery as recovery


ROOT = Path(__file__).resolve().parents[3]


def configure() -> None:
    recovery.configure_attempt(
        entry_path=Path(__file__),
        experiment_id="s18_s1_actionability_beauty_reservation_recovery",
        attempt_id="run-0007",
        recovery_of="s18_s1_actionability_beauty_recovery/run-0006",
        auth_path=ROOT
        / "experiment/phase18/config/s18_s1_beauty_reservation_authorization.json",
        output=ROOT / "artifacts/phase18/s1_actionability/run-0007",
        smoke=ROOT
        / "artifacts/phase18/s1_actionability/beauty-reservation-smoke-run-0007",
        status=ROOT
        / "artifacts/phase18/status/s18_s1_actionability_beauty_reservation_recovery.status.json",
        smoke_status=ROOT
        / "artifacts/phase18/status/s18_s1_beauty_reservation_smoke_run0007.status.json",
        status_archive=ROOT
        / "artifacts/phase18/status/history/s18_s1_actionability.run-0006.status.json",
    )


def main() -> int:
    configure()
    return recovery.main()


if __name__ == "__main__":
    raise SystemExit(main())
