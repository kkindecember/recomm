import math

import numpy as np
import torch

from train_bw3_listwise_admission import (
    ACTION_CATEGORIES,
    FEATURES,
    action_category,
    apply_margin,
    attrition_summary,
    feature_statistics,
    frozen_pcrf_joint,
    listwise_loss,
    select_margin,
)


def candidate(candidate_id, features):
    return {"candidate_id": candidate_id, "features": np.asarray(features, dtype=np.float64)}


def event(category="base_top10", target_index=None, expansion=None, **overrides):
    row = {
        "user": "u",
        "target": 1,
        "target_frequency": 1,
        "q1": 1,
        "base_top10": list(range(1, 11)),
        "base_rank": 1,
        "in_beam50": category.startswith("base_"),
        "in_beam200": category != "outside_union",
        "action_category": category,
        "target_expansion_index": target_index,
        "expansion": expansion or [candidate(20, np.zeros(len(FEATURES)))],
    }
    row.update(overrides)
    return row


def test_feature_schema_is_preregistered_exactly():
    assert FEATURES == [
        "seq_raw", "seq_anchor_z", "item_raw", "item_anchor_z",
        "popularity_log1p", "popularity_anchor_z", "beam200_rank_fraction",
        "reliability", "cf_pop_adjusted",
    ]


def test_frozen_pcrf_keeps_adjusted_score_second_standardization():
    sequence = np.asarray([1.0, 4.0, 2.0])
    item = np.asarray([1.0, 2.0, 8.0])
    popularity = np.asarray([0.0, 3.0, 15.0])
    reliability = 0.7
    seq_z = (sequence - sequence.mean()) / sequence.std()
    item_z = (item - item.mean()) / item.std()
    pop = np.log1p(popularity)
    pop_z = (pop - pop.mean()) / pop.std()
    adjusted = item_z - 0.5 * pop_z
    adjusted_z = (adjusted - adjusted.mean()) / adjusted.std()
    assert np.allclose(frozen_pcrf_joint(sequence, item, popularity, reliability), seq_z + reliability * adjusted_z)


def test_action_category_covers_three_label_branches():
    base_ids = list(range(1, 51))
    assert action_category(1, base_ids, 1, {60}) == "base_top10"
    assert action_category(20, base_ids, 20, {60}) == "base_11_50"
    assert action_category(60, base_ids, 201, {60}) == "expansion_only"
    assert action_category(70, base_ids, 201, {60}) == "outside_union"


def test_listwise_reject_and_expansion_losses_match_hand_calculation():
    reject = event(expansion=[candidate(20, np.zeros(len(FEATURES)))])
    promote = event(
        category="expansion_only",
        target_index=0,
        expansion=[candidate(20, np.zeros(len(FEATURES)))],
        in_beam50=False,
    )
    weight = torch.zeros(len(FEATURES), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    mean = np.zeros(len(FEATURES))
    std = np.ones(len(FEATURES))
    assert math.isclose(float(listwise_loss([reject], weight, bias, mean, std)), math.log(2), rel_tol=1e-6)
    assert math.isclose(float(listwise_loss([promote], weight, bias, mean, std)), math.log(2), rel_tol=1e-6)


def test_listwise_averages_users_not_candidate_rows():
    one_candidate = event(expansion=[candidate(20, np.zeros(len(FEATURES)))])
    three_candidates = event(
        user="v",
        expansion=[candidate(20 + i, np.zeros(len(FEATURES))) for i in range(3)],
    )
    weight = torch.zeros(len(FEATURES), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    value = float(listwise_loss([one_candidate, three_candidates], weight, bias, np.zeros(len(FEATURES)), np.ones(len(FEATURES))))
    assert math.isclose(value, (math.log(2) + math.log(4)) / 2, rel_tol=1e-6)


def test_outside_union_does_not_affect_feature_statistics_or_loss():
    eligible = event(expansion=[candidate(20, np.arange(len(FEATURES)))])
    excluded = event(
        category="outside_union",
        in_beam50=False,
        in_beam200=False,
        expansion=[candidate(30, np.full(len(FEATURES), 1000.0))],
    )
    mean, _ = feature_statistics([eligible, excluded])
    assert np.allclose(mean, np.arange(len(FEATURES)))


def test_no_admission_is_exact_fallback_and_cap_is_three():
    expansion = [candidate(20 + i, np.full(len(FEATURES), i + 1.0)) for i in range(5)]
    row = event(expansion=expansion)
    model = {
        "mean": np.zeros(len(FEATURES)),
        "std": np.ones(len(FEATURES)),
        "weight": np.ones(len(FEATURES)),
        "bias": 0.0,
    }
    fallback = apply_margin(row, model, 1000.0)
    assert fallback["fallback"] and fallback["final_top10"] == row["base_top10"]
    admitted = apply_margin(row, model, 0.0)
    assert len(admitted["admitted"]) == 3
    assert admitted["final_top10"][:7] == row["base_top10"][:7]


def test_margin_selection_uses_frozen_lexicographic_rule():
    rows = [
        {"margin": 0.0, "hit10_delta": 0.1, "ndcg10_delta": 0.01, "admissions": 2, "candidate": {"Hit@10": 0.6, "NDCG@10": 0.4}},
        {"margin": 0.5, "hit10_delta": 0.1, "ndcg10_delta": 0.01, "admissions": 1, "candidate": {"Hit@10": 0.6, "NDCG@10": 0.4}},
        {"margin": 1.0, "hit10_delta": -0.1, "ndcg10_delta": 0.02, "admissions": 1, "candidate": {"Hit@10": 0.7, "NDCG@10": 0.5}},
    ]
    assert select_margin(rows)["margin"] == 0.5


def test_attrition_action_counts_are_mutually_exclusive_and_complete():
    events = []
    for index, category in enumerate(ACTION_CATEGORIES):
        events.append(
            event(
                category=category,
                user=str(index),
                in_beam50=category.startswith("base_"),
                in_beam200=category != "outside_union",
            )
        )
    result = attrition_summary(events)
    assert result["total_events"] == 4
    assert result["action_counts"] == {category: 1 for category in ACTION_CATEGORIES}
    assert sum(result["action_fractions"].values()) == 1.0
