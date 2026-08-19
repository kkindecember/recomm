"""Unit tests for the Phase-13 v1-R² B1 cross-domain portfolio confirmation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol.b1_portfolio_confirmation import (  # noqa: E402
    ANCHOR_PREFIX,
    MIN_COLD_H50_EVENTS,
    hit_and_ndcg,
    portfolio_ranking,
    unique_in_order,
)

SCRIPT = (
    Path(__file__).resolve().parents[1] / "protocol" / "b1_portfolio_confirmation.py"
)


def test_portfolio2_places_candidates_at_ranks_9_and_10():
    gram = [f"g{i}" for i in range(1, 51)]
    ranking = portfolio_ranking(gram, ["r1"], ["c1", "c2", "c3"], 2)
    assert ranking[:8] == gram[:8]
    assert ranking[8:10] == ["c1", "c2"]


def test_portfolio3_places_candidates_at_ranks_8_9_10():
    gram = [f"g{i}" for i in range(1, 51)]
    ranking = portfolio_ranking(gram, ["r1"], ["c1", "c2", "c3"], 3)
    assert ranking[:7] == gram[:7]
    assert ranking[7:10] == ["c1", "c2", "c3"]


def test_anchor_prefix_is_frozen_at_seven():
    assert ANCHOR_PREFIX == 7


def test_portfolio_preserves_protected_gram_prefix_exactly():
    gram = [f"g{i}" for i in range(1, 51)]
    for size in (2, 3):
        ranking = portfolio_ranking(gram, ["r1"], ["c1", "c2", "c3"], size)
        assert ranking[:ANCHOR_PREFIX] == gram[:ANCHOR_PREFIX]


def test_portfolio_output_has_no_duplicates():
    gram = [f"g{i}" for i in range(1, 51)]
    ranking = portfolio_ranking(gram, gram[:10], ["c1", "c2", "c3"], 3)
    assert len(ranking) == len(set(ranking))


def test_portfolio_rejects_unsupported_size():
    with pytest.raises(ValueError):
        portfolio_ranking(["g1"], ["r1"], ["c1"], 1)


def test_portfolio_rejects_insufficient_candidates():
    gram = [f"g{i}" for i in range(1, 51)]
    with pytest.raises(ValueError):
        portfolio_ranking(gram, ["r1"], ["c1"], 2)


def test_hit_and_ndcg_discount_matches_rank():
    assert hit_and_ndcg(["a", "b"], "a", 10) == (1.0, 1.0)
    hit, ndcg = hit_and_ndcg(["x", "a"], "a", 10)
    assert hit == 1.0 and ndcg == pytest.approx(1.0 / 1.584962500721156)
    assert hit_and_ndcg(["x"], "a", 10) == (0.0, 0.0)


def test_hit_and_ndcg_respects_cutoff():
    ranking = [f"g{i}" for i in range(1, 21)]
    assert hit_and_ndcg(ranking, "g15", 10) == (0.0, 0.0)
    assert hit_and_ndcg(ranking, "g15", 50)[0] == 1.0


def test_unique_in_order_is_stable():
    assert unique_in_order(["b", "a", "b", "c"]) == ["b", "a", "c"]


def _write_case(
    tmp_path: Path, n_users: int, cold_hits: int, v0_reachable: bool = False
) -> tuple[Path, Path]:
    """Build a synthetic P0 file.

    The first `cold_hits` users have a cold target sitting in the resolver's
    top-3.  When `v0_reachable` is set the same target is also parked deep in
    the v0 ranking, so the v0 baseline itself registers cold H@50 events.
    """
    rows = []
    cold_items = []
    for i in range(n_users):
        gram = [f"g{i}_{j}" for j in range(50)]
        cold = [f"c{i}_{j}" for j in range(3)]
        cold_items.extend(cold)
        target = cold[0] if i < cold_hits else f"miss{i}"
        if i < cold_hits:
            cold_items.append(target)
            if v0_reachable:
                gram[40] = target
        resolver = cold + [f"r{i}_{j}" for j in range(47)]
        rows.append({
            "user_id": f"u{i}", "target": target,
            "v0_top50": gram, "resolver_top50": resolver,
        })
    p0 = tmp_path / "p0.jsonl"
    with p0.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    cold_file = tmp_path / "cold.txt"
    cold_file.write_text("\n".join(sorted(set(cold_items))) + "\n")
    return p0, cold_file


def _run(tmp_path: Path, p0: Path, cold: Path) -> dict:
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--p0-predictions", str(p0),
         "--cold-items", str(cold), "--domain", "unit", "--output-dir", str(out)],
        check=True, capture_output=True,
    )
    return json.loads((out / "summary.json").read_text())


def test_event_density_guard_forces_inconclusive_when_sparse(tmp_path):
    # Cold hits below the frozen threshold must never be reported as FAIL.
    p0, cold = _write_case(tmp_path, n_users=40, cold_hits=5)
    summary = _run(tmp_path, p0, cold)
    assert summary["baseline_cold_hit50_events"] < MIN_COLD_H50_EVENTS
    assert summary["event_density_guard"]["triggered"] is True
    assert summary["verdict"] == "INCONCLUSIVE"


def test_guard_not_triggered_when_events_sufficient(tmp_path):
    p0, cold = _write_case(tmp_path, n_users=200, cold_hits=60, v0_reachable=True)
    summary = _run(tmp_path, p0, cold)
    assert summary["baseline_cold_hit50_events"] >= MIN_COLD_H50_EVENTS
    assert summary["event_density_guard"]["triggered"] is False


def test_portfolio_promotes_cold_target_above_v0_baseline(tmp_path):
    # End-to-end: moving a deep v0 cold hit into rank 9 must raise cold NDCG@10.
    p0, cold = _write_case(tmp_path, n_users=200, cold_hits=60, v0_reachable=True)
    summary = _run(tmp_path, p0, cold)
    front = summary["pareto_front"]
    assert front["v0_gram"]["cold"]["ndcg@10"] == 0.0
    assert front["unconditional_portfolio2"]["cold"]["ndcg@10"] > 0.0
    assert (
        summary["paired_bootstrap_vs_v0"]["unconditional_portfolio2"]
        ["cold_hit@50"]["verdict"] in {"PASS", "INCONCLUSIVE"}
    )


def test_summary_records_validation_only_and_primary_candidate(tmp_path):
    p0, cold = _write_case(tmp_path, n_users=60, cold_hits=10)
    summary = _run(tmp_path, p0, cold)
    assert summary["test_predictions_opened"] is False
    assert summary["split"] == "validation"
    assert summary["primary_candidate"] == "unconditional_portfolio2"
    assert summary["frozen_parameters"]["anchor_prefix"] == 7
    assert summary["frozen_parameters"]["bootstrap_seed"] == 20260818


def test_refuses_test_prediction_file(tmp_path):
    p0, cold = _write_case(tmp_path, n_users=20, cold_hits=3)
    disguised = tmp_path / "predictions_test_medium.jsonl"
    disguised.write_text(p0.read_text())
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--p0-predictions", str(disguised),
         "--cold-items", str(cold), "--domain", "unit",
         "--output-dir", str(tmp_path / "out2")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "test prediction file" in result.stderr
