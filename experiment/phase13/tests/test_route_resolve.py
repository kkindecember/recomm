import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "protocol" / "route_resolve.py"
SPEC = importlib.util.spec_from_file_location("route_resolve", MODULE_PATH)
rr = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rr)


def test_decode_and_route():
    lexical = "|▁game|▁board|be|▁cu|▁actions2"
    assert rr.decode_lexical_id(lexical) == "game boardbe cu actions2"
    assert rr.semantic_route(lexical, 3) == ("▁game", "▁board", "be")


def test_training_examples_never_use_validation_or_cold_targets():
    sequences = [("u1", ["a", "b", "c", "cold_val", "cold_test"])]
    item_to_idx = {item: i for i, item in enumerate(sequences[0][1])}
    embeddings = torch.eye(5)
    x, y, report = rr.build_training_examples(
        sequences,
        item_to_idx,
        embeddings,
        {"cold_val", "cold_test"},
        max_history=20,
        recency_decay=0.85,
    )
    assert x.shape == (2, 5)
    assert y.tolist() == [item_to_idx["b"], item_to_idx["c"]]
    assert report["cold_target_count"] == 0


def test_multi_positive_loss_is_finite_with_duplicate_targets():
    users = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    targets = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    target_ids = torch.tensor([1, 1, 2, 3])
    loss = rr.multi_positive_inbatch_loss(users, targets, target_ids, 0.07)
    assert torch.isfinite(loss)


def test_fusion_is_unique_and_catalog_only():
    item_ids = ["a", "b", "c", "d"]
    route = ("x", "y", "z")
    result = rr.fuse_r2(
        resolver_order=[0, 1, 2],
        gram_items=["b", "d"],
        route_order=[route],
        route_to_ranked_indices={route: [1, 2]},
        item_ids=item_ids,
        rrf_k=60,
        route_prior_weight=0.25,
        global_retrieve_k=3,
        per_route_k=2,
    )
    assert len(result) == len(set(result))
    assert set(result).issubset(item_ids)


def test_prediction_parser_rejects_test_file(tmp_path):
    path = tmp_path / "anything_test.tsv"
    path.write_text("idx\tplaceholder\n")
    try:
        rr.parse_gram_predictions(path)
    except ValueError as exc:
        assert "test prediction" in str(exc)
    else:
        raise AssertionError("test prediction file was not rejected")
