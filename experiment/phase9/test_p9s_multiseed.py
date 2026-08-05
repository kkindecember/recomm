from summarize_p9s_multiseed import summarize


def test_summary_passes_expected_matrix():
    rows = []
    for dataset in ("Toys", "Beauty"):
        for seed, delta in zip((2023, 2024, 2025), (0.003, 0.004, 0.005)):
            rows.append({"dataset": dataset, "seed": seed, "item_gate": "frozen_existing" if seed == 2023 else "passed", "hit10_delta": delta, "ndcg10_delta": 0.001, "tail_hit10_delta": 0.0001, "hit50_delta": 0.0})
    grouped, checks = summarize(rows)
    assert grouped["Toys"]["Hit@10_delta_median"] == 0.004
    assert all(checks.values())


def test_negative_seed_fails_gate():
    rows = []
    for dataset in ("Toys", "Beauty"):
        for seed in (2023, 2024, 2025):
            rows.append({"dataset": dataset, "seed": seed, "item_gate": "passed", "hit10_delta": -0.001 if seed == 2025 else 0.003, "ndcg10_delta": 0.001, "tail_hit10_delta": 0.0, "hit50_delta": 0.0})
    _, checks = summarize(rows)
    assert not checks["all_six_Hit10_deltas_positive"]
