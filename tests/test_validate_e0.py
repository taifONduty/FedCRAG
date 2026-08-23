"""validate_e0.py must refuse a corrupted run directory.

Each test builds a genuine run directory with the mocked driver, confirms it
validates, then introduces exactly one corruption and requires a refusal. A
corruption that repairs the persisted hashes it invalidates is the interesting
case: the older hash gates cannot see it, so only the independent aggregate
recomputation can.
"""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver_harness  # noqa: E402
from aggregation_schemes import state_dict_sha256  # noqa: E402
from validate_e0 import (  # noqa: E402
    E0ValidationError,
    validate_run_directory,
)

E0_CELLS = [
    ("trainable-ab", "uniform"),
    ("trainable-ab", "rawmaxmin"),
    ("frozen-a", "uniform"),
    ("frozen-a", "rawmaxmin"),
    ("frozen-a", "normmaxmin"),
]


def build_run(monkeypatch, tmp_path, lora_mode="frozen-a", arm="normmaxmin",
              num_rounds=1):
    result, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, lora_mode, arm, num_rounds=num_rounds)
    return result, result_path


def state_path(tmp_path, round_number=1):
    return driver_harness.load_round_states(tmp_path, round_number)[1]


def load_states(tmp_path, round_number=1):
    return driver_harness.load_round_states(tmp_path, round_number)[0]


def resave_states(tmp_path, payload, round_number=1, repair_hashes=False):
    """Persist a doctored payload, optionally making its own hashes agree."""
    if repair_hashes:
        payload["broadcast_state_sha256"] = state_dict_sha256(
            payload["broadcast"])
        payload["global_state_sha256"] = state_dict_sha256(payload["global"])
    torch.save(payload, state_path(tmp_path, round_number))


def rewrite(result_path, result):
    with Path(result_path).open("w") as handle:
        json.dump(result, handle)


# --------------------------------------------------------- the clean baseline


@pytest.mark.parametrize(("lora_mode", "arm"), E0_CELLS)
def test_clean_run_validates_for_every_e0_cell(
        monkeypatch, tmp_path, lora_mode, arm):
    build_run(monkeypatch, tmp_path, lora_mode, arm, num_rounds=2)
    report = validate_run_directory(tmp_path)

    assert report["rounds_validated"] == 2
    assert report["commit"] == driver_harness.CLEAN_COMMIT
    assert report["lora_mode"] == lora_mode
    assert report["manifest_verified"] is False
    # Headroom against false positives must stay visible, not merely pass.
    assert report["aggregate_recomputation_worst_tolerance_ratio"] < 0.5


# ----------------------------------------------------- forged server updates


@pytest.mark.parametrize(("lora_mode", "arm"), E0_CELLS)
def test_forged_global_is_refused_even_with_repaired_hashes(
        monkeypatch, tmp_path, lora_mode, arm):
    """The persisted global is rescaled and every hash it breaks is repaired."""
    build_run(monkeypatch, tmp_path, lora_mode, arm)
    validate_run_directory(tmp_path)

    payload = load_states(tmp_path)
    payload["global"][driver_harness.B_KEY] = (
        payload["global"][driver_harness.B_KEY] * 1.5)
    resave_states(tmp_path, payload, repair_hashes=True)
    if arm == "normmaxmin":
        result_path = next(Path(tmp_path).glob("federated_*.json"))
        with result_path.open() as handle:
            result = json.load(handle)
        application = result["fedspan_diagnostics"]["round_1"]["application"]
        application["applied_state_sha256"] = payload["global_state_sha256"]
        rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="persisted global disagrees"):
        validate_run_directory(tmp_path)


@pytest.mark.parametrize("arm", ["rawmaxmin", "normmaxmin"])
def test_arm_silently_downgraded_to_uniform_is_refused(
        monkeypatch, tmp_path, arm):
    """The recorded arm still claims adaptive weights; the global is FedAvg."""
    build_run(monkeypatch, tmp_path, "frozen-a", arm)
    payload = load_states(tmp_path)

    broadcast = payload["broadcast"]
    states = [payload["clients"][name] for name in driver_harness.SLICES]
    uniform_b = broadcast[driver_harness.B_KEY].double().clone()
    for state in states:
        uniform_b += (state[driver_harness.B_KEY].double()
                      - broadcast[driver_harness.B_KEY].double()) / len(states)
    payload["global"][driver_harness.B_KEY] = uniform_b.float()
    resave_states(tmp_path, payload, repair_hashes=True)
    if arm == "normmaxmin":
        result_path = next(Path(tmp_path).glob("federated_*.json"))
        with result_path.open() as handle:
            result = json.load(handle)
        result["fedspan_diagnostics"]["round_1"]["application"][
            "applied_state_sha256"] = payload["global_state_sha256"]
        rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="persisted global disagrees"):
        validate_run_directory(tmp_path)


def test_single_element_perturbation_of_the_global_is_refused(
        monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "uniform")
    payload = load_states(tmp_path)
    payload["global"][driver_harness.B_KEY][0, 0] += 1e-3
    resave_states(tmp_path, payload, repair_hashes=True)

    with pytest.raises(E0ValidationError, match="persisted global disagrees"):
        validate_run_directory(tmp_path)


# ------------------------------------------------------- persisted contracts


def test_changed_frozen_a_in_a_client_state_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    payload = load_states(tmp_path)
    payload["clients"]["c1"][driver_harness.A_KEY] = (
        payload["clients"]["c1"][driver_harness.A_KEY] + 1e-4)
    resave_states(tmp_path, payload)

    with pytest.raises(E0ValidationError, match="client c1 A changed"):
        validate_run_directory(tmp_path)


def test_changed_frozen_a_in_the_global_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "uniform")
    payload = load_states(tmp_path)
    payload["global"][driver_harness.A_KEY] = (
        payload["global"][driver_harness.A_KEY] + 1e-4)
    resave_states(tmp_path, payload, repair_hashes=True)

    with pytest.raises(E0ValidationError, match="global A changed"):
        validate_run_directory(tmp_path)


def test_broken_persisted_hash_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    payload = load_states(tmp_path)
    payload["global_state_sha256"] = "0" * 64
    resave_states(tmp_path, payload)

    with pytest.raises(E0ValidationError,
                       match="persisted global hash is invalid"):
        validate_run_directory(tmp_path)


def test_broadcast_hash_mismatch_between_json_and_states_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    result["fedspan_diagnostics"]["round_1"]["application"][
        "broadcast_state_sha256"] = "1" * 64
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="broadcast hash differs"):
        validate_run_directory(tmp_path)


def test_client_hash_mismatch_between_json_and_states_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    application = result["fedspan_diagnostics"]["round_1"]["application"]
    application["client_state_sha256"][1] = "2" * 64
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="client hashes differ"):
        validate_run_directory(tmp_path)


def test_applied_norm_disagreeing_with_the_resolved_norm_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    diagnostic["resolved_step_norm"] = float(
        diagnostic["resolved_step_norm"]) * 1.2
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="applied norm differs from resolved norm"):
        validate_run_directory(tmp_path)


def test_nonfinite_client_tensor_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "uniform")
    payload = load_states(tmp_path)
    payload["clients"]["c2"][driver_harness.B_KEY][0, 0] = float("nan")
    resave_states(tmp_path, payload)

    with pytest.raises(E0ValidationError, match="nonfinite entries"):
        validate_run_directory(tmp_path)


# --------------------------------------------------------- provenance gates


@pytest.mark.parametrize(
    "commit",
    ["abc123def456-dirty", "unknown", "abc123def456-unknown-worktree"])
def test_unclean_provenance_is_refused(monkeypatch, tmp_path, commit):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "uniform")
    with result_path.open() as handle:
        result = json.load(handle)
    result["commit"] = commit
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="clean Git provenance"):
        validate_run_directory(tmp_path)


# ------------------------------------------------------ D1 direction policy


def test_implicit_direction_policy_is_refused(monkeypatch, tmp_path):
    """A paper-grade run may not rely on the compatibility default."""
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    result["fedspan_diagnostics"]["round_1"][
        "direction_policy_specified"] = False
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="implicit default"):
        validate_run_directory(tmp_path)


def test_unconverged_min_norm_reference_is_refused(monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    result["fedspan_diagnostics"]["round_1"]["min_norm_solver"][
        "converged"] = False
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="did not converge"):
        validate_run_directory(tmp_path)


def test_direction_policy_disagreeing_with_the_contract_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    result["method_contract"]["fedspan_direction_policy"] = "maxmin-lp"
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="direction policy differs from method contract"):
        validate_run_directory(tmp_path)


def test_missing_direction_shortfall_is_refused(monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    result["fedspan_diagnostics"]["round_1"][
        "direction_solver_shortfall"] = None
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="direction_solver_shortfall"):
        validate_run_directory(tmp_path)


# --------------------------------------------------------- D2a audit trail


def test_missing_client_delta_norms_are_refused(monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "rawmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    del result["client_delta_norms"]["round_1"]
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="no per-client effective delta norms"):
        validate_run_directory(tmp_path)


def test_failed_client_delta_norms_are_refused(monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "rawmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    result["client_delta_norms"]["round_1"] = {"error": "RuntimeError: boom"}
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="per-client delta norms"):
        validate_run_directory(tmp_path)


# ------------------------------------------------------- scheme fallbacks


def test_scheme_fallback_round_is_refused(monkeypatch, tmp_path):
    """A round whose arm silently degraded to uniform is not E0 evidence."""
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "rawmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    record = result["scheme_diagnostics"]["round_1"]
    record["fallback"] = "uniform"
    record["status"] = "solver_failure"
    record["solver_message"] = "HiGHS returned no solution"
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="fell back to uniform"):
        validate_run_directory(tmp_path)


def test_missing_scheme_diagnostics_are_refused(monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "rawmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    del result["scheme_diagnostics"]["round_1"]
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="not recorded at full precision"):
        validate_run_directory(tmp_path)


def test_unsupported_arm_has_no_recomputation_reference(monkeypatch, tmp_path):
    """Fail closed rather than skipping the aggregate check for a new arm."""
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "rawmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    result["weight_by_canonical"] = "mgda"
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="no recomputation reference for arm"):
        validate_run_directory(tmp_path)


# --------------------------------------------------------- structural gates


def test_missing_round_state_file_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "uniform", num_rounds=2)
    state_path(tmp_path, 2).unlink()

    with pytest.raises(E0ValidationError, match="expected 2 state files"):
        validate_run_directory(tmp_path)


def test_recorded_weights_that_do_not_cover_every_client_are_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    diagnostic["delta_weights"] = diagnostic["delta_weights"][:-1]
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="do not cover every client"):
        validate_run_directory(tmp_path)


def manifest_row(tmp_path, lora_mode="frozen-a", arm="normmaxmin",
                 num_rounds=1):
    return {
        "run_id": tmp_path.name,
        "commit": driver_harness.CLEAN_COMMIT,
        "coordinate": lora_mode,
        "arm": arm,
        "argv": driver_harness.build_argv(
            tmp_path, lora_mode, arm, num_rounds=num_rounds)
                + ["--seed", "42", "--max_steps_per_round", "0"],
    }


def write_resource_record(tmp_path, num_rounds=1):
    record = {
        "schema": "fedcrag-e0-resources/1",
        "elapsed_seconds": 12.0,
        "round_elapsed_seconds": [1.0] * num_rounds,
        "deterministic_algorithms": False,
        "gpu_available": False,
        "peak_gpu_memory_mib": None,
    }
    with (tmp_path / "e0_resources.json").open("w") as handle:
        json.dump(record, handle)


def test_a_matching_manifest_row_validates(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)

    report = validate_run_directory(
        tmp_path, manifest_row=manifest_row(tmp_path))

    assert report["manifest_verified"] is True
    assert report["launched"]["seed"] == 42
    assert report["launched"]["slices"] == list(driver_harness.SLICES)


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [("--seed", "7", "launched seed=7"),
     ("--num_rounds", "9", "launched num_rounds=9"),
     ("--max_steps_per_round", "500", "launched max_steps_per_round=500"),
     ("--lora_mode", "trainable-ab", "launched lora_mode='trainable-ab'")],
)
def test_manifest_row_disagreeing_with_the_run_is_refused(
        monkeypatch, tmp_path, flag, value, message):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)

    drifted = manifest_row(tmp_path)
    drifted["argv"][drifted["argv"].index(flag) + 1] = value
    with pytest.raises(E0ValidationError, match=message):
        validate_run_directory(tmp_path, manifest_row=drifted)


def test_manifest_slice_order_is_significant(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)

    drifted = manifest_row(tmp_path)
    start = drifted["argv"].index("--slices") + 1
    drifted["argv"][start], drifted["argv"][start + 1] = (
        drifted["argv"][start + 1], drifted["argv"][start])
    with pytest.raises(E0ValidationError, match="launched slices"):
        validate_run_directory(tmp_path, manifest_row=drifted)


def test_manifest_commit_drift_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)

    drifted = manifest_row(tmp_path)
    drifted["commit"] = "0123456789ab"
    with pytest.raises(E0ValidationError, match="differs from the manifest"):
        validate_run_directory(tmp_path, manifest_row=drifted)


def test_missing_resource_record_is_refused_when_a_manifest_is_given(
        monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")

    with pytest.raises(E0ValidationError, match="unauditable"):
        validate_run_directory(
            tmp_path, manifest_row=manifest_row(tmp_path))


def test_truncated_round_timings_are_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2)
    write_resource_record(tmp_path, num_rounds=1)

    with pytest.raises(E0ValidationError, match="one finite elapsed time"):
        validate_run_directory(
            tmp_path, manifest_row=manifest_row(tmp_path, num_rounds=2))
