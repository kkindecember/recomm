from pathlib import Path
import inspect
import sys


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from counterfactual_slot_router import extract_item_feature_vector  # noqa: E402
from p4_disjoint_confirmation import (  # noqa: E402
    build_disjoint_manifest,
    ordered_uid_population,
    select_uid_tranche,
)


def test_disjoint_tranches_follow_one_global_uid_hash_order(tmp_path):
    users = [f"u{i}" for i in range(40)]
    first = select_uid_tranche(users, 0, 10)
    second = select_uid_tranche(list(reversed(users)), 10, 10)
    assert first == ordered_uid_population(users)[:10]
    assert second == ordered_uid_population(users)[10:20]
    assert set(first).isdisjoint(second)


def test_manifest_proves_zero_overlap_and_target_free(tmp_path):
    previous_path = tmp_path / "previous.json"
    previous_path.write_text('{"selected_users":[{"user_id":"u0"}]}')
    manifest = build_disjoint_manifest(
        ["u1", "u2"], 3, 1,
        {"selected_users": [{"user_id": "u0"}]}, previous_path,
    )
    assert manifest["previous_sample_overlap"] == 0
    assert manifest["target_used_for_selection"] is False
    assert manifest["written_before_test_prediction_open"] is True
    assert manifest["tranche_rank_one_based"] == [2, 3]


def test_manifest_rejects_previous_sample_overlap(tmp_path):
    previous_path = tmp_path / "previous.json"
    previous_path.write_text('{"selected_users":[{"user_id":"u0"}]}')
    try:
        build_disjoint_manifest(
            ["u0"], 2, 1,
            {"selected_users": [{"user_id": "u0"}]}, previous_path,
        )
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected overlap rejection")


def test_frozen_item_feature_extractor_has_no_target_argument():
    parameters = inspect.signature(extract_item_feature_vector).parameters
    assert "target" not in parameters
    assert "label" not in parameters
