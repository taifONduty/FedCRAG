"""FedMGDA+ native-step arm (registration SS9.4) — source-faithful.

Transcribed from arXiv:2006.11489 (Hu, Shaloudegi, Zhang, Yu), Algorithm 1:
client updates normalized to unit length; lambda* = min-norm over the simplex
(epsilon = 1 makes the epsilon-ball constraint inactive); the server applies
w - eta_t * d_t with d_t = sum lambda*_k u_k UN-normalized, so the applied
norm is eta_t * ||z*|| and shrinks with the conflict level — the
self-annihilating step their Theorem 1b requires (eta_t -> 0, sum eta_t =
inf). Their settled schedule is exponential decay (SS6.1.5); at T < 100
rounds their 100-step staircase never fires, so the continuous form
eta_t = eta0 * decay^((t-1)/(T-1)) is used, preserving the design target
eta_T/eta_1 = decay. Registered defaults eta0=1.0, decay=0.1 (their grid
centres). The ONLY difference from FedSpan is the step law: same direction,
same solver, same gates.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver_harness  # noqa: E402
from aggregation_schemes import fedspan_delta_weights  # noqa: E402
from fedspan_fixtures import federation_with_cosine_gram  # noqa: E402
from validate_e0 import validate_run_directory  # noqa: E402

GRAM = np.array([
    [1.00, 0.55, 0.35, -0.10],
    [0.55, 1.00, 0.40, 0.05],
    [0.35, 0.40, 1.00, 0.20],
    [-0.10, 0.05, 0.20, 1.00],
])
RADII = [1.9, 2.0, 2.1, 0.2]


def solve(policy, **kwargs):
    broadcast, clients, scales = federation_with_cosine_gram(GRAM, RADII)
    return fedspan_delta_weights(
        clients, broadcast, module_scales=scales,
        direction_policy="exact", step_policy=policy, **kwargs)


def test_same_direction_as_fedspan_different_norm_law():
    """Identical lambda*, coefficients proportional (same direction), and the
    fedmgda applied norm is exactly eta_t * ||sum w u|| — the un-normalized
    min-norm step of Algorithm 1 line 9."""
    eta = 0.7
    fedspan = solve("median-active")
    fedmgda = solve("fedmgda", step_norm=eta)
    assert fedmgda["simplex_weights"] == fedspan["simplex_weights"]
    ratio = None
    for a, b in zip(fedmgda["delta_weights"], fedspan["delta_weights"]):
        if b:
            r = a / b
            ratio = r if ratio is None else ratio
            assert r == pytest.approx(ratio, rel=1e-12)
    assert fedmgda["resolved_step_norm"] == pytest.approx(
        eta * fedmgda["mixture_norm"], rel=1e-12)
    assert fedmgda["declared_step_norm"] == eta
    assert fedmgda["step_policy"] == "fedmgda"


def test_step_self_annihilates_with_conflict():
    """The scientific point of the arm: at fixed eta, more conflict (smaller
    mixture norm) means a smaller applied step — the property FedSpan's
    constant-norm policy deliberately removed."""
    mild = solve("fedmgda", step_norm=1.0)
    eps = 1e-3
    conflicted = np.array([
        [1.0, -0.9, 0.0, 0.0],
        [-0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, eps],
        [0.0, 0.0, eps, 1.0],
    ])
    broadcast, clients, scales = federation_with_cosine_gram(
        conflicted + np.eye(4) * 1e-6, RADII)
    hard = fedspan_delta_weights(
        clients, broadcast, module_scales=scales,
        direction_policy="exact", step_policy="fedmgda", step_norm=1.0)
    assert hard["resolved_step_norm"] < mild["resolved_step_norm"]
    assert hard["resolved_step_norm"] == pytest.approx(
        hard["mixture_norm"], rel=1e-12)


def test_fedmgda_requires_positive_eta():
    with pytest.raises(ValueError, match="fedmgda"):
        solve("fedmgda")
    with pytest.raises(ValueError, match="fedmgda"):
        solve("fedmgda", step_norm=-1.0)


# --- driver level ------------------------------------------------------------

FEDMGDA_EXTRA = ("--fedspan_step_policy", "fedmgda",
                 "--fedspan_direction_policy", "exact")


def run_fedmgda(monkeypatch, out, rounds=2, extra=()):
    return driver_harness.run_driver(
        monkeypatch, out, "frozen-a", "normmaxmin", num_rounds=rounds,
        row_scale="peft-init",
        extra=(*FEDMGDA_EXTRA, *extra))


def test_driver_schedule_matches_transcribed_form_and_validates(
        monkeypatch, tmp_path):
    result, _ = run_fedmgda(monkeypatch, tmp_path, rounds=2)
    diags = result["fedspan_diagnostics"]
    eta0, decay, T = 1.0, 0.1, 2
    for t in (1, 2):
        expected = eta0 * decay ** ((t - 1) / (T - 1))
        assert diags[f"round_{t}"]["declared_step_norm"] == pytest.approx(
            expected, rel=1e-12), f"round {t}"
    assert validate_run_directory(tmp_path)["rounds_validated"] == 2


def test_driver_single_round_uses_eta0(monkeypatch, tmp_path):
    result, _ = run_fedmgda(monkeypatch, tmp_path, rounds=1)
    assert result["fedspan_diagnostics"]["round_1"][
        "declared_step_norm"] == pytest.approx(1.0)


def test_driver_rejects_illegal_fedmgda_flags(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_fedmgda(monkeypatch, tmp_path,
                    extra=("--fedspan_step_norm", "0.5"))
    assert "rejects --fedspan_step_norm" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        driver_harness.run_driver(
            monkeypatch, tmp_path / "b", "frozen-a", "normmaxmin",
            row_scale="peft-init",
            extra=("--fedspan_step_policy", "fedmgda",
                   "--fedspan_direction_policy", "fixed",
                   "--fedspan_fixed_weights", "0.4", "0.3", "0.3"))
    assert "fedmgda" in capsys.readouterr().err


def test_filename_distinguishes_the_arm(monkeypatch, tmp_path):
    _, path = run_fedmgda(monkeypatch, tmp_path, rounds=1)
    assert "sfedmgda" in path.name
