"""t1_gate.py: the registered decision of block T1 from the run summaries."""
import pytest

import t1_gate


def runs(arm, values):
    return [{"arm": arm, "seed": seed, "schedule": schedule, "A": a, "G": g}
            for (seed, schedule), (a, g) in zip(
                [(s, sch) for sch in "AB" for s in (123, 2024, 3407)], values)]


def test_criterion_needs_the_mean_and_five_of_six_runs():
    strong = [(0.03, 0.02)] * 6
    one_weak = [(0.03, 0.02)] * 5 + [(0.03, 0.004)]
    two_weak = [(0.03, 0.02)] * 4 + [(0.03, 0.004)] * 2
    assert t1_gate.criterion([g for _, g in strong], threshold=0.010)["passes"]
    assert t1_gate.criterion([g for _, g in one_weak], threshold=0.010)["passes"]
    assert not t1_gate.criterion([g for _, g in two_weak], threshold=0.010)["passes"]
    assert not t1_gate.criterion([0.009] * 6, threshold=0.010)["passes"]


def test_decision_reads_the_arms_of_the_registered_rule():
    d = runs("fedavg-replay", [(0.03, 0.02)] * 6)
    b = runs("local", [(0.02, 0.01)] * 6)
    e = runs("fedavg-replay-distill", [(0.029, 0.005)] * 6)
    decision = t1_gate.decide(d + b + e)
    assert decision["G2"]["passes"] and decision["G1"]["passes"]
    assert decision["G3"]["passes"] and decision["outcome"] == "method arms"
    assert decision["distillation_controls_regression"] is True


def test_missing_runs_are_an_error():
    with pytest.raises(ValueError, match="six"):
        t1_gate.decide(runs("fedavg-replay", [(0.03, 0.02)] * 5))
