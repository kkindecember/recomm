from pathlib import Path
import sys

import torch


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"
sys.path.insert(0, str(PROTOCOL))

from fresh_medium_smoke import (  # noqa: E402
    build_test_examples,
    read_selected_test_predictions,
    select_medium_uids,
    selection_manifest,
)


def test_uid_selection_is_deterministic_and_target_free_by_interface():
    users = [f"u{i}" for i in range(30)]
    assert select_medium_uids(users, 10) == select_medium_uids(list(reversed(users)), 10)
    manifest = selection_manifest(select_medium_uids(users, 10), len(users))
    assert manifest["target_used_for_selection"] is False
    assert manifest["written_before_test_prediction_open"] is True
    assert len(manifest["selected_users"]) == 10


def test_selected_parser_skips_malformed_unselected_rows_before_parse(tmp_path):
    path = tmp_path / "pred_test.tsv"
    selected_fields = ["u1"] + ["not-a-metric"] * 13 + ["beam-a||beam-b", "bad-score"]
    path.write_text("u0\tmalformed\n" + "\t".join(selected_fields) + "\n")
    rows, audit = read_selected_test_predictions(path, {"u1"})
    assert rows == {"u1": ["beam-a", "beam-b"]}
    assert audit["outside_sample_metric_or_prediction_rows_parsed"] == 0
    assert audit["saved_test_metric_fields_parsed"] is False
    assert audit["saved_test_gold_fields_parsed"] is False


def test_test_example_uses_final_item_and_excludes_it_from_history():
    embeddings = torch.eye(4)
    item_to_idx = {f"i{i}": i for i in range(4)}
    examples = build_test_examples(
        [("u1", ["i0", "i1", "i2"])], {"u1"}, item_to_idx,
        embeddings, max_history=20, recency_decay=1.0,
    )
    history, target = examples["u1"]
    assert target == "i2"
    expected = torch.tensor([1.0, 1.0, 0.0, 0.0])
    expected = expected / expected.norm()
    assert torch.allclose(history, expected)
