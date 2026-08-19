from pathlib import Path
import sys


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from anchored_interleaving import anchored_interleave, select_config  # noqa: E402


def test_protected_prefix_is_byte_for_byte_unchanged():
    gram = ["w1", "w2", "w3", "c1", "w4"]
    resolver = ["c2", "w5", "c1", "c3"]
    ranked, inserted = anchored_interleave(gram, resolver, {"c1", "c2", "c3"}, 3, 2)
    assert ranked[:3] == gram[:3]
    assert inserted == ["c2", "c1"]
    assert ranked == ["w1", "w2", "w3", "c2", "c1", "w4", "w5", "c3"]


def test_interleaving_is_unique_and_never_inserts_warm():
    ranked, inserted = anchored_interleave(
        ["w1", "w2", "c1", "w1"],
        ["w3", "c2", "c2", "w4"],
        {"c1", "c2"},
        2,
        3,
    )
    assert inserted == ["c2"]
    assert len(ranked) == len(set(ranked))
    assert set(inserted) <= {"c1", "c2"}


def test_select_config_uses_preregistered_tie_break():
    metrics = {
        "all": {"ndcg@10": 0.2},
        "warm": {"ndcg@10": 0.3},
        "cold": {"ndcg@10": 0.1},
    }
    rows = [
        {"protected_prefix": 8, "cold_quota": 2, "feasible": True, "metrics": metrics},
        {"protected_prefix": 9, "cold_quota": 1, "feasible": True, "metrics": metrics},
        {"protected_prefix": 7, "cold_quota": 1, "feasible": False, "metrics": metrics},
    ]
    assert select_config(rows)["protected_prefix"] == 9
    assert select_config(rows)["cold_quota"] == 1


def test_select_config_returns_none_without_feasible_row():
    assert select_config([{"feasible": False}]) is None
