"""Driver-level reachability of the new arms, and the slice-identity guard.

An arm that exists in ``aggregation_schemes`` but has no command line is not
an arm: no campaign can request it and no result file records it. These tests
run the real driver and assert that the exact solver, the per-round optimality
certificate and the fixed-weight arms are reachable, recorded, and separated
in the configuration hash that names the output file.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver_harness  # noqa: E402
import federated_forgetting as driver  # noqa: E402

UNIFORM_OVER_CLIENTS = ["0.333333333333", "0.333333333333", "0.333333333333"]
UNIFORM_OVER_DISTRIBUTIONS = ["0.25", "0.25", "0.5"]


def load_result(out_directory):
    paths = list(Path(out_directory).glob("federated_*.json"))
    assert len(paths) == 1, paths
    with paths[0].open() as handle:
        return json.load(handle)


def fedspan_argv(tmp_path, direction_policy, extra=()):
    return ["federated_forgetting.py",
            "--slices", *driver_harness.SLICES,
            "--metrics", "ndcg@10",
            "--num_rounds", "1",
            "--lora_mode", "frozen-a", "--frozen_a_row_scale", "unit",
            "--save_states",
            "--weighted", "--weight_by", "normmaxmin",
            "--fedspan_step_policy", "median-active",
            "--fedspan_direction_policy", direction_policy,
            "--out", str(tmp_path), *extra]


# ---------------------------------------------------- the exact arm is live


def test_driver_runs_the_exact_direction_policy(monkeypatch, tmp_path):
    result, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy="exact")
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["direction_policy"] == "exact"
    assert diagnostic["min_norm_value_source"] == "exact-face-enumeration"
    assert diagnostic["exact_solver"]["algorithm"].startswith(
        "face-enumeration")
    assert diagnostic["status"] == "optimal"
    assert "-direxact" in path.name


def test_driver_records_the_exact_policy_in_the_configuration_hash(
        monkeypatch, tmp_path):
    exact, _ = driver_harness.run_driver(
        monkeypatch, tmp_path / "a", "frozen-a", "normmaxmin",
        direction_policy="exact")
    frank_wolfe, _ = driver_harness.run_driver(
        monkeypatch, tmp_path / "b", "frozen-a", "normmaxmin",
        direction_policy="minnorm")
    assert exact["method_contract"]["fedspan_direction_policy"] == "exact"
    assert (exact["method_contract"]["run_configuration_sha256"]
            != frank_wolfe["method_contract"]["run_configuration_sha256"])


# ------------------------------------------------- the certificate is logged


@pytest.mark.parametrize("policy", ["minnorm", "maxmin-lp", "exact"])
def test_driver_persists_a_wolfe_certificate_every_round(
        monkeypatch, tmp_path, policy):
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2,
        direction_policy=policy)
    for label in ("round_1", "round_2"):
        diagnostic = result["fedspan_diagnostics"][label]
        assert isinstance(diagnostic["wolfe_certificate"], float)
        assert diagnostic["wolfe_certificate"] >= 0.0
        assert isinstance(diagnostic["min_norm_value"], float)


# --------------------------------------------------------- the fixed arms


def test_driver_runs_the_norm_equalised_uniform_arm(monkeypatch, tmp_path):
    driver_harness.install_mocks(monkeypatch)
    monkeypatch.setattr(sys, "argv", fedspan_argv(
        tmp_path, "fixed",
        extra=["--fedspan_fixed_weights", *UNIFORM_OVER_CLIENTS]))
    driver.main()
    result = load_result(tmp_path)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["direction_policy"] == "fixed"
    assert diagnostic["status"] == "fixed"
    assert diagnostic["simplex_weights"] == pytest.approx(
        [1.0 / 3.0] * 3, abs=1e-9)
    assert result["method_contract"]["fedspan_fixed_weights"] == \
        pytest.approx([1.0 / 3.0] * 3, abs=1e-9)


def test_driver_separates_two_fixed_weight_arms(monkeypatch, tmp_path):
    hashes = []
    for weights in (UNIFORM_OVER_CLIENTS, UNIFORM_OVER_DISTRIBUTIONS):
        out = tmp_path / ("w" + weights[0])
        driver_harness.install_mocks(monkeypatch)
        monkeypatch.setattr(sys, "argv", fedspan_argv(
            out, "fixed", extra=["--fedspan_fixed_weights", *weights]))
        driver.main()
        result = load_result(out)
        hashes.append(result["method_contract"]["run_configuration_sha256"])
    assert hashes[0] != hashes[1]


def test_driver_rejects_fixed_policy_without_weights(
        monkeypatch, capsys, tmp_path):
    driver_harness.install_mocks(monkeypatch)
    monkeypatch.setattr(sys, "argv", fedspan_argv(tmp_path, "fixed"))
    with pytest.raises(SystemExit, match="2"):
        driver.main()
    assert "--fedspan_fixed_weights" in capsys.readouterr().err


def test_driver_rejects_fixed_weights_without_the_fixed_policy(
        monkeypatch, capsys, tmp_path):
    driver_harness.install_mocks(monkeypatch)
    monkeypatch.setattr(sys, "argv", fedspan_argv(
        tmp_path, "minnorm",
        extra=["--fedspan_fixed_weights", *UNIFORM_OVER_CLIENTS]))
    with pytest.raises(SystemExit, match="2"):
        driver.main()
    assert "--fedspan_fixed_weights" in capsys.readouterr().err


def test_driver_rejects_a_fixed_weight_vector_of_the_wrong_length(
        monkeypatch, capsys, tmp_path):
    driver_harness.install_mocks(monkeypatch)
    monkeypatch.setattr(sys, "argv", fedspan_argv(
        tmp_path, "fixed", extra=["--fedspan_fixed_weights", "0.5", "0.5"]))
    with pytest.raises(SystemExit, match="2"):
        driver.main()
    assert "one weight per slice" in capsys.readouterr().err


# ------------------------------------------------------- the slice guard


def test_assert_unique_slices_rejects_a_repeated_name():
    with pytest.raises(ValueError, match="repeated"):
        driver.assert_unique_slices(["nfcorpus", "nfcorpus", "arguana"])
    driver.assert_unique_slices(["nfcorpus", "arguana"])


def test_assert_data_matches_slices_catches_the_dict_collapse():
    slices = ["nfcorpus", "nfcorpus", "arguana"]
    data = {"nfcorpus": {}, "arguana": {}}
    with pytest.raises(ValueError, match="one payload per slice"):
        driver.assert_data_matches_slices(data, slices)


def test_driver_rejects_duplicate_slices(monkeypatch, capsys, tmp_path):
    """--slices nfcorpus nfcorpus nfcorpus arguana is a different experiment.

    Four clients from two datasets, three of them holding bit-identical data,
    with only two entries in the provenance fingerprints. It must not run.
    """
    driver_harness.install_mocks(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "federated_forgetting.py",
        "--slices", "c0", "c0", "c0", "c1",
        "--metrics", "ndcg@10", "--num_rounds", "1",
        "--out", str(tmp_path),
    ])
    with pytest.raises(SystemExit, match="2"):
        driver.main()
    assert "repeated" in capsys.readouterr().err
    assert list(Path(tmp_path).glob("federated_*.json")) == []


def test_driver_still_accepts_distinct_slices(monkeypatch, tmp_path):
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "uniform")
    assert result["slices"] == list(driver_harness.SLICES)
