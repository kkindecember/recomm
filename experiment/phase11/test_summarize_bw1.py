from summarize_bw1_candidate_ceiling import choose_decision


def row(coverage, pcrf, gate="passed"):
    return {"coverage_headroom": coverage, "pcrf_headroom": pcrf, "integrity_gate": gate}


def test_decision_candidate_expansion():
    assert choose_decision([row(0.01, 0.003), row(0.008, 0.001)]) == "candidate_expansion_eligible"


def test_decision_saturated_and_failed():
    assert choose_decision([row(0.004, 0.003), row(0.003, 0.002)]) == "beam_coverage_saturated"
    assert choose_decision([row(0.01, 0.003, "failed"), row(0.01, 0.003)]) == "integrity_failed"


def test_decision_unconverted_and_domain_dependent():
    assert choose_decision([row(0.01, 0.001), row(0.01, 0.001)]) == "coverage_not_converted_by_frozen_pcrf"
    assert choose_decision([row(0.01, 0.003), row(0.01, -0.001)]) == "domain_dependent"
