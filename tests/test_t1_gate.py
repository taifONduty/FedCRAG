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


def write_run(out_dir, arm, schedule, seed, A, G, validated=True):
    import json
    d = out_dir / f"pilot-{arm}-{schedule}-s{seed}"
    d.mkdir()
    (d / f"continual_x_seed{seed}_{arm}.json").write_text(
        json.dumps({"summary": {"A": A, "G": G}}))
    if validated:
        (d / ".validated").write_text("ok\n")
    return d


def test_load_runs_reads_arm_schedule_and_seed_from_the_directory(tmp_path):
    write_run(tmp_path, "fedavg-replay-distill", "A", 123, 0.03, 0.01)
    write_run(tmp_path, "local", "B", 2024, 0.02, 0.00)
    rows = {(r["arm"], r["schedule"], r["seed"]): r for r in t1_gate.load_runs(str(tmp_path))}
    assert set(rows) == {("fedavg-replay-distill", "A", 123), ("local", "B", 2024)}
    assert rows[("fedavg-replay-distill", "A", 123)]["A"] == 0.03


def test_load_runs_skips_a_run_without_its_validation_marker(tmp_path):
    write_run(tmp_path, "fedavg", "A", 123, 0.03, 0.01, validated=False)
    assert t1_gate.load_runs(str(tmp_path)) == []
