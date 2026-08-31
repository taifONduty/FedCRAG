"""--weight_pow: the q-dose-response axis (registration SS9.3).

w_k proportional to n_k^q interpolates uniform (q=0) to canonical FedAvg n_k
(q=1). The flag is only-when-supplied so every historical filename and
configuration survives unchanged; distinct q values must produce distinct
filenames or two registered runs would silently overwrite each other.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver_harness  # noqa: E402
from validate_e0 import E0ValidationError, validate_run_directory  # noqa: E402

COUNTS = {"c0": 1000, "c1": 100, "c2": 10}


def test_q_run_validates_and_recomputation_uses_the_exponent(
        monkeypatch, tmp_path):
    """A q=0.5 run must validate — and it CAN only validate if the
    recomputation weights by n^0.5, because the aggregate it checks was built
    that way and the counts differ by 10x."""
    driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "examples",
        example_counts=COUNTS, extra=("--weight_pow", "0.5"))
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1


def test_tampered_exponent_is_refused(monkeypatch, tmp_path):
    result, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "examples",
        example_counts=COUNTS, extra=("--weight_pow", "0.5"))
    forged = json.loads(path.read_text())
    forged["args"]["weight_pow"] = 1.0
    path.write_text(json.dumps(forged))
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_q_zero_equals_uniform_aggregation(monkeypatch, tmp_path):
    """q=0 is the uniform endpoint: identical global state to an unweighted
    run over the same clients."""
    _, path_q0 = driver_harness.run_driver(
        monkeypatch, tmp_path / "q0", "trainable-ab", "examples",
        example_counts=COUNTS, extra=("--weight_pow", "0"))
    _, path_uni = driver_harness.run_driver(
        monkeypatch, tmp_path / "uni", "trainable-ab", "uniform",
        example_counts=COUNTS)
    s_q0, _ = driver_harness.load_round_states(tmp_path / "q0")
    s_uni, _ = driver_harness.load_round_states(tmp_path / "uni")
    for key in s_uni["global"]:
        torch.testing.assert_close(s_q0["global"][key], s_uni["global"][key])


def test_filenames_distinguish_q_and_preserve_legacy(monkeypatch, tmp_path):
    _, p_legacy = driver_harness.run_driver(
        monkeypatch, tmp_path / "legacy", "trainable-ab", "examples",
        example_counts=COUNTS)
    _, p_half = driver_harness.run_driver(
        monkeypatch, tmp_path / "half", "trainable-ab", "examples",
        example_counts=COUNTS, extra=("--weight_pow", "0.5"))
    _, p_quart = driver_harness.run_driver(
        monkeypatch, tmp_path / "quart", "trainable-ab", "examples",
        example_counts=COUNTS, extra=("--weight_pow", "0.25"))
    assert "weighted-examples_r1" in p_legacy.name          # unchanged
    assert "-q" not in p_legacy.name
    assert "weighted-examples-q0p5_r1" in p_half.name
    assert "weighted-examples-q0p25_r1" in p_quart.name
    assert p_half.name != p_quart.name


@pytest.mark.parametrize("argv_extra,message", [
    (("--weight_pow", "0.5", "--weight_by", "corpus"),
     "legal only with --weight_by examples"),
    (("--weight_pow", "-1"), "nonnegative"),
])
def test_driver_rejects_illegal_weight_pow(monkeypatch, tmp_path, capsys,
                                           argv_extra, message):
    with pytest.raises(SystemExit):
        driver_harness.run_driver(
            monkeypatch, tmp_path, "trainable-ab", "examples",
            extra=argv_extra)
    assert message in capsys.readouterr().err
