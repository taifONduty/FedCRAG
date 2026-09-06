"""The canonical validator must recompute the published-baseline arms.

q-FFL, AFL and FedNova (registration §13, block A2) run through the same driver
as every other arm, but until now the validator had no recomputation reference
for them and refused their runs outright. A run that cannot be validated does
not count, so these references come first. Each reference derives the round's
coefficients from independently persisted quantities (pair counts and step
counts for FedNova; broadcast-point losses, update norms and the contract's
learning rate for q-FFL; the loss history and eta for AFL), checks them against
the recorded weights, and recomputes the server aggregate from the persisted
client states. Tampering with any input must be refused.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import driver_harness  # noqa: E402
from aggregation_schemes import state_dict_sha256  # noqa: E402
from validate_e0 import E0ValidationError, validate_run_directory  # noqa: E402

COUNTS = {"c0": 1000, "c1": 250, "c2": 40}
STEPS = {"c0": 31, "c1": 8, "c2": 2}
LOSSES = {"c0": 1.35, "c1": 2.10, "c2": 0.72}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, result):
    Path(path).write_text(json.dumps(result))


# ------------------------------------------------------------------- FedNova


def test_fednova_run_validates_in_the_trainable_coordinate(monkeypatch, tmp_path):
    driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "fednova",
        example_counts=COUNTS, step_counts=STEPS)
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1


def test_fednova_run_validates_in_the_frozen_a_coordinate(monkeypatch, tmp_path):
    driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "fednova", row_scale="peft-init",
        example_counts=COUNTS, step_counts=STEPS)
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1


def test_fednova_forged_step_count_is_refused(monkeypatch, tmp_path):
    """v_k = tau_eff p_k / tau_k: the steps are an input, so editing them must
    make the recorded weights disagree with the recomputation."""
    _, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "fednova",
        example_counts=COUNTS, step_counts=STEPS)
    forged = read(path)
    forged["clients"]["round_1"]["c0"]["num_steps"] = 3
    write(path, forged)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_fednova_forged_recorded_weights_are_refused(monkeypatch, tmp_path):
    _, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "fednova",
        example_counts=COUNTS, step_counts=STEPS)
    forged = read(path)
    forged["scheme_diagnostics"]["round_1"]["weights"] = [1 / 3, 1 / 3, 1 / 3]
    write(path, forged)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


# --------------------------------------------------------------------- q-FFL


def test_qffl_run_validates(monkeypatch, tmp_path):
    driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "qffl",
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES)
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1


def test_qffl_forged_loss_is_refused(monkeypatch, tmp_path):
    """The broadcast-point losses are the arm's whole reason to exist; a
    validator that ignores them would accept a q-FFL run with any weights."""
    _, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "qffl",
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES)
    forged = read(path)
    forged["client_losses"]["round_1"]["c2"] = 5.0
    write(path, forged)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_qffl_forged_global_is_refused_even_with_repaired_hashes(
        monkeypatch, tmp_path):
    """Delta-space weights do not sum to one, so a simplex-average reference
    would be wrong here; the reference must apply w^t + sum v_k (w_k - w^t)."""
    driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "qffl",
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES)
    payload, state_path = driver_harness.load_round_states(tmp_path, 1)
    b_key = driver_harness.B_KEY
    payload["global"][b_key] = payload["global"][b_key] + 1e-3
    payload["global_state_sha256"] = state_dict_sha256(payload["global"])
    torch.save(payload, state_path)
    with pytest.raises(E0ValidationError, match="persisted global disagrees"):
        validate_run_directory(tmp_path)


# ----------------------------------------------------------------------- AFL


def test_afl_two_round_run_validates(monkeypatch, tmp_path):
    """AFL's weights are a chain: lambda_t depends on lambda_{t-1}. The
    reference must replay the chain from the uniform start, so a two-round run
    is the smallest case that exercises it."""
    driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "afl", num_rounds=2,
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES)
    assert validate_run_directory(tmp_path)["rounds_validated"] == 2


def test_afl_forged_second_round_weights_are_refused(monkeypatch, tmp_path):
    _, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "afl", num_rounds=2,
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES)
    forged = read(path)
    forged["scheme_diagnostics"]["round_2"]["weights"] = [1 / 3, 1 / 3, 1 / 3]
    write(path, forged)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_afl_forged_first_round_loss_breaks_the_chain(monkeypatch, tmp_path):
    """Editing a round-1 loss must be caught even though the round-1 weights
    are still self-consistent with the *recorded* round-2 weights: the chain is
    recomputed from the losses, not copied from the record."""
    _, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "afl", num_rounds=2,
        example_counts=COUNTS, step_counts=STEPS, losses=LOSSES)
    forged = read(path)
    forged["client_losses"]["round_1"]["c1"] = 0.1
    write(path, forged)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)
