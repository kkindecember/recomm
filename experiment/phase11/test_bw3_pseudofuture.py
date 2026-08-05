from smoke_bw3_pseudofuture import pseudo_sample_from_sequence


def test_pseudo_sample_uses_requested_offset_and_excludes_later_targets():
    items = ["a", "b", "c", "d", "e", "f"]
    mapping = {item: item.upper() for item in items}
    cfids = {item: index + 1 for index, item in enumerate(items)}
    sample = pseudo_sample_from_sequence(
        "u1", items, 3, "Toy", mapping, mapping, cfids, 20, " ; ", True
    )
    assert sample["target"] == "d"
    assert sample["target_offset"] == 3
    assert sample["history_item_ids"] == [3, 2, 1]
    assert "e" not in sample["history"] and "f" not in sample["history"]


def test_pseudo_sample_truncates_before_reversal():
    items = ["a", "b", "c", "d", "e", "f"]
    mapping = {item: item for item in items}
    cfids = {item: index + 1 for index, item in enumerate(items)}
    sample = pseudo_sample_from_sequence(
        "u1", items, 3, "Toy", mapping, mapping, cfids, 2, " ; ", True
    )
    assert sample["history_item_ids"] == [3, 2]
