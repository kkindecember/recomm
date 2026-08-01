import torch

from experiment.phase6.cet_gacr_v1 import (
    METHODS,
    decide,
    load_arm_checkpoint,
    summarize_method,
    unify_rows,
)


class _Wrapper:
    def __init__(self):
        self.backbone = torch.nn.Linear(2, 2)

    def load_state_dict(self, state, strict=True):
        return self.backbone.load_state_dict(state, strict=strict)

    def eval(self):
        self.backbone.eval()


def test_checkpoint_scope_is_explicit(tmp_path):
    wrapper = _Wrapper()
    checkpoint = tmp_path / "model.pt"
    torch.save(wrapper.backbone.state_dict(), checkpoint)
    prepared = {"model": wrapper}
    load_arm_checkpoint(prepared, checkpoint, "backbone", torch.device("cpu"))
    try:
        load_arm_checkpoint(prepared, checkpoint, "unknown", torch.device("cpu"))
    except ValueError as exc:
        assert "unsupported checkpoint scope" in str(exc)
    else:
        raise AssertionError("unknown scope must fail closed")


def _record(key, group, base_rank, candidate_rank=None):
    row = {"sample_key": key, "target_group": group, "gram_rank": base_rank}
    if candidate_rank is not None:
        row["candidate_rank"] = candidate_rank
    return row


def test_four_arms_are_paired_and_summarized():
    gram = [_record("u1", "head", 2), _record("u2", "tail", 20)]
    cet = [_record("u1", "head", 1), _record("u2", "tail", 10)]
    gacr = [_record("u1", "head", 2, 1), _record("u2", "tail", 20, 9)]
    combo = [_record("u1", "head", 1, 1), _record("u2", "tail", 10, 8)]
    rows = unify_rows(gram, cet, gacr, combo)
    assert len(rows) == 2
    assert set(METHODS) == {"GRAM", "CET_v1", "GACR_v3", "CET_v1_GACR_v3"}
    summary = summarize_method(rows, "CET_v1_GACR_v3")
    assert summary["overall"]["Recall@10"] == 1.0
    assert summary["tail"]["Recall@10"] == 1.0


def test_unpaired_cohorts_fail_closed():
    gram = [_record("u1", "head", 2)]
    cet = [_record("u2", "head", 2)]
    gacr = [_record("u1", "head", 2, 1)]
    combo = [_record("u1", "head", 2, 1)]
    try:
        unify_rows(gram, cet, gacr, combo)
    except ValueError as exc:
        assert "cohorts differ" in str(exc)
    else:
        raise AssertionError("unpaired cohorts must fail closed")


def test_decision_requires_combo_to_exceed_both_singles_per_domain():
    validation = {}
    values = {
        "Toys": {"GRAM": 1.0, "CET_v1": 1.1, "GACR_v3": 1.2, "CET_v1_GACR_v3": 1.3},
        "Beauty": {"GRAM": 1.0, "CET_v1": 1.2, "GACR_v3": 1.1, "CET_v1_GACR_v3": 1.3},
    }
    for dataset in values:
        methods = {
            method: {"overall": {"NDCG@10": value}}
            for method, value in values[dataset].items()
        }
        validation[dataset] = {
            "seeds": {
                "1": {
                    "methods": methods,
                    "comparisons": {
                        "combo_vs_gram": {
                            "broad_harm_rate": 0.0,
                            "overall_recall10_absolute_gain": 0.0,
                        }
                    },
                }
            }
        }
    config = {
        "decision_rule": {
            "broad_harm_rate_max": 0.01,
            "recall10_absolute_floor": -0.002,
        }
    }
    result = decide(validation, config)
    assert result["decision"] == "KEEP_CET_V1_GACR_V3_COMBINATION"
    validation["Beauty"]["seeds"]["1"]["methods"]["CET_v1_GACR_v3"]["overall"]["NDCG@10"] = 1.15
    result = decide(validation, config)
    assert result["decision"] == "RETURN_TO_STRONGER_SINGLE_METHOD"
