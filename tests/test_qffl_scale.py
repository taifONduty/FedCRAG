"""q-FFL's Lipschitz constant L must be an explicit, recorded choice.

The transcription uses the paper's heuristic L = 1/lr. With full-epoch local
updates that heuristic makes every delta weight of order 1e-5 (a no-op arm;
see supervisor/2026-09-06_qffl_degeneracy_note.md). Registration §13 therefore
runs q-FFL with a rescaled L, so the driver needs a flag for it that (i) leaves
the historical behaviour and filenames untouched when absent, (ii) is recorded
and used when present, (iii) changes the filename so two L values never
overwrite each other, (iv) refuses illegal values, and (v) is honoured by the
validator's recomputation.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import driver_harness  # noqa: E402
import federated_forgetting as driver  # noqa: E402
from validate_e0 import E0ValidationError, validate_run_directory  # noqa: E402

COUNTS = {"c0": 1000, "c1": 250, "c2": 40}
STEPS = {"c0": 31, "c1": 8, "c2": 2}
LOSSES = {"c0": 1.35, "c1": 2.10, "c2": 0.72}


def reference_v(result, L):
    """Independent q-FFL delta weights from the record, q = qffl_q."""
    q = float(result["args"]["qffl_q"])
    slices = result["slices"]
    f = [max(result["client_losses"]["round_1"][s], 1e-8) for s in slices]
    d2 = [result["client_delta_norms"]["round_1"][s] ** 2 for s in slices]
    fq = [1.0 if q == 0 else x ** q for x in f]
    fqm1 = [0.0 if q == 0 else x ** (q - 1.0) for x in f]
    h = [q * a * L * L * b + L * c for a, b, c in zip(fqm1, d2, fq)]
    return [L * c / sum(h) for c in fq]


def test_absent_flag_keeps_L_equal_to_one_over_lr(monkeypatch, tmp_path):
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "qffl",
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES)
    assert result["args"]["qffl_L"] is None
    recorded = result["scheme_diagnostics"]["round_1"]["weights"]
    expected = reference_v(result, 1.0 / float(result["args"]["lr"]))
    assert recorded == pytest.approx(expected, rel=1e-6)


def test_supplied_L_is_recorded_used_and_validated(monkeypatch, tmp_path):
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "qffl",
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES,
        extra=("--qffl_L", "1.0"))
    assert result["args"]["qffl_L"] == 1.0
    recorded = result["scheme_diagnostics"]["round_1"]["weights"]
    assert recorded == pytest.approx(reference_v(result, 1.0), rel=1e-6)
    # a rescaled L is the whole point: the step is no longer negligible
    assert sum(recorded) > 1e-2
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1


def test_filenames_distinguish_L_and_preserve_the_legacy_name(
        monkeypatch, tmp_path):
    _, legacy = driver_harness.run_driver(
        monkeypatch, tmp_path / "legacy", "trainable-ab", "qffl",
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES)
    _, scaled = driver_harness.run_driver(
        monkeypatch, tmp_path / "scaled", "trainable-ab", "qffl",
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES,
        extra=("--qffl_L", "1.0"))
    assert legacy.name == "federated_bge-m3_seed42_weighted-qffl_r1.json"
    assert scaled.name != legacy.name
    assert "L1" in scaled.name


@pytest.mark.parametrize("extra", [
    ("--qffl_L", "0"),
    ("--qffl_L", "-2"),
    ("--qffl_L", "nan"),
])
def test_driver_rejects_nonpositive_or_nonfinite_L(monkeypatch, tmp_path,
                                                    extra):
    with pytest.raises(SystemExit):
        driver_harness.run_driver(
            monkeypatch, tmp_path, "trainable-ab", "qffl",
            example_counts=COUNTS, step_counts=STEPS, losses=LOSSES,
            extra=extra)


def test_driver_rejects_L_without_the_qffl_arm(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        driver_harness.run_driver(
            monkeypatch, tmp_path, "trainable-ab", "afl",
            example_counts=COUNTS, step_counts=STEPS, losses=LOSSES,
            extra=("--qffl_L", "1.0"))


def test_validator_refuses_a_tampered_L(monkeypatch, tmp_path):
    """The recomputation must read the recorded L: editing it in the record
    makes the recorded weights disagree with the reference."""
    _, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "qffl",
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES,
        extra=("--qffl_L", "1.0"))
    forged = json.loads(path.read_text())
    forged["args"]["qffl_L"] = 2.0
    path.write_text(json.dumps(forged))
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)
