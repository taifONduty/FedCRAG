"""validate_e0.py must refuse a corrupted run directory.

Each test builds a genuine run directory with the mocked driver, confirms it
validates, then introduces exactly one corruption and requires a refusal. A
corruption that repairs the persisted hashes it invalidates is the interesting
case: the older hash gates cannot see it, so only the independent aggregate
recomputation can.
"""
import copy
import hashlib
import math
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver_harness  # noqa: E402
from aggregation_schemes import (  # noqa: E402
    ModuleScales,
    apply_fedspan_update,
    fedspan_delta_weights,
    state_dict_sha256,
)
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


def direct_effective_step_sha256(blocks):
    """Hash effective blocks directly; never reuse validator internals."""
    digest = hashlib.sha256()
    for name in sorted(blocks):
        tensor = blocks[name].detach().cpu().double().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def repair_fedspan_solution(tmp_path, result_path, active_weights,
                            round_number=1, coefficient_sign=1.0):
    """Replace a round decision and independently repair all affected data."""
    payload = load_states(tmp_path, round_number)
    with Path(result_path).open() as handle:
        result = json.load(handle)
    label = f"round_{round_number}"
    diagnostic = result["fedspan_diagnostics"][label]
    application = diagnostic["application"]
    active = list(diagnostic["active_indices"])
    weights = np.asarray(active_weights, dtype=np.float64)
    assert len(weights) == len(active)
    assert np.min(weights) >= 0.0
    assert np.sum(weights) == pytest.approx(1.0)

    scale = float(diagnostic["module_scales"][driver_harness.MODULE])
    broadcast_b = payload["broadcast"][driver_harness.B_KEY].double()
    blocks = []
    norms = []
    for name in result["slices"]:
        block = scale * (
            payload["clients"][name][driver_harness.B_KEY].double()
            - broadcast_b)
        blocks.append(block)
        norms.append(float(torch.linalg.vector_norm(block).item()))
    unit = torch.stack([
        blocks[index].reshape(-1) / norms[index] for index in active
    ]).numpy()
    cosine = unit @ unit.T
    mixture_sq = float(weights @ cosine @ weights)
    mixture_norm = math.sqrt(max(mixture_sq, 0.0))
    payoffs = cosine @ weights
    gamma = float(np.min(payoffs))
    step_norm = float(diagnostic["resolved_step_norm"])
    coefficients = [0.0] * len(result["slices"])
    for local_index, client_index in enumerate(active):
        coefficients[client_index] = float(
            coefficient_sign * step_norm * weights[local_index]
            / (norms[client_index] * mixture_norm))

    global_b = broadcast_b.clone()
    for coefficient, name in zip(coefficients, result["slices"]):
        global_b += coefficient * (
            payload["clients"][name][driver_harness.B_KEY].double()
            - broadcast_b)
    payload["global"][driver_harness.B_KEY] = global_b.float()
    payload["global_state_sha256"] = state_dict_sha256(payload["global"])
    torch.save(payload, state_path(tmp_path, round_number))

    solved_block = sum(
        coefficients[index] * blocks[index] for index in active)
    applied_block = scale * (
        payload["global"][driver_harness.B_KEY].double() - broadcast_b)
    solved_hash = direct_effective_step_sha256(
        {driver_harness.MODULE: solved_block})
    applied_hash = direct_effective_step_sha256(
        {driver_harness.MODULE: applied_block})
    solved_norm = float(torch.linalg.vector_norm(solved_block).item())
    applied_norm = float(torch.linalg.vector_norm(applied_block).item())
    direction_cosines = [None] * len(result["slices"])
    if applied_norm > 0.0:
        for index in active:
            direction_cosines[index] = float(
                torch.sum(blocks[index] * applied_block).item()
                / (norms[index] * applied_norm))

    simplex = [0.0] * len(result["slices"])
    for local_index, client_index in enumerate(active):
        simplex[client_index] = float(weights[local_index])
    diagnostic.update({
        "simplex_weights": simplex,
        "delta_weights": coefficients,
        "proposed_delta_weights": coefficients.copy(),
        "gamma": gamma,
        "mixture_norm": mixture_norm,
        "solver_objective_gamma": gamma,
        "solver_simplex_residual": abs(float(np.sum(weights)) - 1.0),
        "solver_constraint_violation": max(
            0.0, float(np.max(gamma - payoffs))),
        "achieved_min_direction_cosine": (
            None if mixture_norm <= 0.0
            else float(coefficient_sign * gamma / mixture_norm)),
        "certified_min_direction_cosine": (
            None if mixture_norm <= 0.0
            else float(coefficient_sign * gamma / mixture_norm)),
        "direction_solver_shortfall": float(
            diagnostic["min_norm_value"]
            - coefficient_sign * gamma / mixture_norm),
        "max_abs_delta_weight": max(abs(value) for value in coefficients),
        "proposed_max_abs_delta_weight": max(
            abs(value) for value in coefficients),
        "step_reconstruction_error": solved_norm - step_norm,
        "solved_effective_step_sha256": solved_hash,
    })
    application.update({
        "applied_step_norm": applied_norm,
        "max_effective_block_error": float(
            torch.max(torch.abs(applied_block - solved_block)).item()),
        "applied_delta_weights": coefficients.copy(),
        "applied_direction_cosines": direction_cosines,
        "applied_min_active_cosine": min(
            value for value in direction_cosines if value is not None),
        "applied_effective_step_sha256": applied_hash,
        "applied_state_sha256": payload["global_state_sha256"],
    })
    rewrite(result_path, result)
    return result, payload


def fabricate_zero_fallback(tmp_path, result_path, status, round_number=1):
    """Rewrite one genuine success as a self-consistent zero fallback."""
    payload = load_states(tmp_path, round_number)
    with Path(result_path).open() as handle:
        result = json.load(handle)
    label = f"round_{round_number}"
    diagnostic = result["fedspan_diagnostics"][label]
    application = diagnostic["application"]
    payload["global"] = {
        key: value.clone() for key, value in payload["broadcast"].items()
    }
    payload["global_state_sha256"] = state_dict_sha256(payload["global"])
    torch.save(payload, state_path(tmp_path, round_number))
    zero_block = torch.zeros_like(
        payload["broadcast"][driver_harness.B_KEY], dtype=torch.float64)
    zero_hash = direct_effective_step_sha256(
        {driver_harness.MODULE: zero_block})
    zeros = [0.0] * len(result["slices"])
    diagnostic.update({
        "status": status,
        "fallback": "zero_update",
        "delta_weights": zeros,
        "proposed_delta_weights": zeros,
        "max_abs_delta_weight": 0.0,
        "proposed_max_abs_delta_weight": 0.0,
        "solved_effective_step_sha256": None,
    })
    application.update({
        "applied_step_norm": 0.0,
        "max_effective_block_error": 0.0,
        "applied_delta_weights": zeros,
        "applied_direction_cosines": [None] * len(result["slices"]),
        "applied_min_active_cosine": None,
        "applied_effective_step_sha256": zero_hash,
        "applied_state_sha256": payload["global_state_sha256"],
    })
    rewrite(result_path, result)


def negate_only_materialized_step(tmp_path, result_path, round_number=1):
    """Negate the small persisted global step but keep the certified solve."""
    payload = load_states(tmp_path, round_number)
    with Path(result_path).open() as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"][f"round_{round_number}"]
    application = diagnostic["application"]
    broadcast_b = payload["broadcast"][driver_harness.B_KEY].double()
    original_global_b = payload["global"][driver_harness.B_KEY].double()
    payload["global"][driver_harness.B_KEY] = (
        2.0 * broadcast_b - original_global_b).float()
    payload["global_state_sha256"] = state_dict_sha256(payload["global"])
    torch.save(payload, state_path(tmp_path, round_number))

    scale = float(diagnostic["module_scales"][driver_harness.MODULE])
    applied_block = scale * (
        payload["global"][driver_harness.B_KEY].double() - broadcast_b)
    coefficients = [float(value) for value in diagnostic["delta_weights"]]
    solved_block = torch.zeros_like(applied_block)
    client_blocks = []
    client_norms = []
    for name in result["slices"]:
        block = scale * (
            payload["clients"][name][driver_harness.B_KEY].double()
            - broadcast_b)
        client_blocks.append(block)
        client_norms.append(float(torch.linalg.vector_norm(block).item()))
    for coefficient, block in zip(coefficients, client_blocks):
        solved_block += coefficient * block
    applied_norm = float(torch.linalg.vector_norm(applied_block).item())
    direction_cosines = [None] * len(result["slices"])
    for index in diagnostic["active_indices"]:
        direction_cosines[index] = float(
            torch.sum(client_blocks[index] * applied_block).item()
            / (client_norms[index] * applied_norm))
    application.update({
        "applied_step_norm": applied_norm,
        "max_effective_block_error": float(torch.max(torch.abs(
            applied_block - solved_block)).item()),
        # The certified production coefficients remain untouched. Only the
        # small persisted materialized state has been sign-reversed.
        "applied_delta_weights": coefficients,
        "applied_direction_cosines": direction_cosines,
        "applied_min_active_cosine": min(
            value for value in direction_cosines if value is not None),
        "applied_effective_step_sha256": direct_effective_step_sha256(
            {driver_harness.MODULE: applied_block}),
        "applied_state_sha256": payload["global_state_sha256"],
    })
    rewrite(result_path, result)
    return result


def replace_with_genuine_invalid_step_fallback(
        monkeypatch, tmp_path, result_path):
    """Install a real production invalid-step result in a persisted round."""
    payload = load_states(tmp_path)
    with Path(result_path).open() as handle:
        result = json.load(handle)
    states = [payload["clients"][name] for name in result["slices"]]
    scales = driver_harness.module_scales("frozen-a")
    # The driver's CLI excludes this defensive production branch. Force only
    # its deterministic median dependency to zero so the diagnostic itself is
    # still emitted by the real fedspan_delta_weights implementation.
    monkeypatch.setattr(np, "median", lambda values: 0.0)
    diagnostic = fedspan_delta_weights(
        states, payload["broadcast"], scales,
        step_norm=None, step_policy="median-active",
        direction_policy="minnorm", active_abs_tol=1e-12,
        active_rel_tol=1e-8, mixture_norm_tol=1e-6,
        max_abs_delta_weight=None)
    assert diagnostic["status"] == "invalid_step_norm"
    global_state, application = apply_fedspan_update(
        payload["broadcast"], states, diagnostic, scales)
    diagnostic = {**diagnostic, "application": application}
    payload["global"] = global_state
    payload["global_state_sha256"] = state_dict_sha256(global_state)
    torch.save(payload, state_path(tmp_path))

    result["fedspan_diagnostics"]["round_1"] = diagnostic
    rewrite(result_path, result)
    return result


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


def test_repaired_hash_opposite_direction_is_refused(monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        before = json.load(handle)
    weights = before["fedspan_diagnostics"]["round_1"]["simplex_weights"]
    repair_fedspan_solution(
        tmp_path, result_path, weights, coefficient_sign=-1.0)

    with pytest.raises(E0ValidationError, match="direction|coefficient"):
        validate_run_directory(tmp_path)


def test_small_negated_materialized_step_is_refused(
        monkeypatch, tmp_path):
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        extra=("--fedspan_step_policy", "fixed",
               "--fedspan_step_norm", "2e-6"))
    result = negate_only_materialized_step(tmp_path, result_path)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["achieved_min_direction_cosine"] > 0.0
    assert diagnostic["application"]["applied_min_active_cosine"] < 0.0

    with pytest.raises(
            E0ValidationError,
            match="materialized.*direction|too small.*direction"):
        validate_run_directory(tmp_path)


@pytest.mark.parametrize("direction_policy", ["minnorm", "maxmin-lp"])
def test_repaired_hash_suboptimal_direction_is_refused(
        monkeypatch, tmp_path, direction_policy):
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy=direction_policy)
    repair_fedspan_solution(tmp_path, result_path, [1.0, 0.0, 0.0])

    with pytest.raises(E0ValidationError, match="objective suboptimality"):
        validate_run_directory(tmp_path)


def test_repaired_near_balanced_antipodal_success_is_refused(
        monkeypatch, tmp_path):
    directions = ([1.0, 0.0], [-1.0, 0.0], [0.0, 0.0])
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        clients=clients_from_unit_directions(directions))
    result, _ = repair_fedspan_solution(
        tmp_path, result_path, [0.500006, 0.499994])
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    diagnostic["status"] = "optimal"
    diagnostic["fallback"] = None
    rewrite(result_path, result)
    assert diagnostic["achieved_min_direction_cosine"] == pytest.approx(-1.0)

    with pytest.raises(
            E0ValidationError,
            match="objective suboptimality|normalized direction|certificate|"
                  "boundary-indeterminate"):
        validate_run_directory(tmp_path)


def test_near_antipodal_objective_passes_but_certificate_refuses(
        monkeypatch, tmp_path):
    """A q-optimal repaired record still needs its directional certificate."""
    directions = [
        [0.6684167496930513, 0.4182758748494597, 0.6150319839233531],
        [-0.9083004866545865, 0.2549551570183043, 0.3316445293575836],
        [-0.4358725723652378, -0.061135324061740555,
         -0.8979296034832444],
    ]
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy="minnorm",
        clients=clients_from_unit_directions(directions))
    weights = [
        0.4446213294146499,
        0.13892965492126302,
        0.41644901566408704,
    ]
    result, _ = repair_fedspan_solution(tmp_path, result_path, weights)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    active = diagnostic["active_indices"]
    cosine = np.asarray(
        diagnostic["cosine_gram_active"], dtype=np.float64)
    active_weights = np.asarray(
        [diagnostic["simplex_weights"][index] for index in active])
    recorded_q = float(active_weights @ cosine @ active_weights)
    optimal_q = float(diagnostic["min_norm_value"]) ** 2

    assert abs(recorded_q - optimal_q) <= 1e-10
    assert min(cosine[np.triu_indices_from(cosine, k=1)]) < -0.85
    with pytest.raises(E0ValidationError, match="certificate"):
        validate_run_directory(tmp_path)


def test_exact_antipodal_fallback_is_boundary_indeterminate(
        monkeypatch, tmp_path):
    directions = ([1.0, 0.0], [-1.0, 0.0], [0.0, 0.0])
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        clients=clients_from_unit_directions(directions))
    assert result["fedspan_diagnostics"]["round_1"]["status"] == (
        "near_cancellation")

    with pytest.raises(E0ValidationError, match="boundary-indeterminate"):
        validate_run_directory(tmp_path)


def aligned_clients():
    base = driver_harness.broadcast_state()
    states = {}
    for multiplier, name in enumerate(driver_harness.SLICES, start=1):
        value = base[driver_harness.B_KEY].clone()
        value[0, 0] = float(multiplier)
        states[name] = {
            driver_harness.A_KEY: base[driver_harness.A_KEY].clone(),
            driver_harness.B_KEY: value,
        }
    return states


def clients_from_unit_directions(directions):
    base = driver_harness.broadcast_state()
    states = {}
    for name, direction in zip(driver_harness.SLICES, directions):
        value = base[driver_harness.B_KEY].clone()
        value[0, :len(direction)] = torch.tensor(direction)
        states[name] = {
            driver_harness.A_KEY: base[driver_harness.A_KEY].clone(),
            driver_harness.B_KEY: value,
        }
    return states


@pytest.mark.parametrize("direction_policy", ["minnorm", "maxmin-lp"])
def test_nonunique_optimal_face_accepts_alternate_weights(
        monkeypatch, tmp_path, direction_policy):
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy=direction_policy, clients=aligned_clients())
    repair_fedspan_solution(tmp_path, result_path, [0.2, 0.3, 0.5])

    report = validate_run_directory(tmp_path)

    assert report["fedspan_applied_rounds"] == 1


def test_direction_quantities_remain_numerically_distinct(
        monkeypatch, tmp_path):
    directions = [
        [0.7011839609197387, -0.39470784729742453,
         0.5883180802160277, 0.08017858019238598],
        [0.885407186197653, -0.381415825422982,
         -0.1801954606651313, 0.19520675885363425],
        [-0.727595824060434, 0.27288791414731695,
         0.5081307013877858, 0.3714023336347008],
    ]
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy="maxmin-lp",
        clients=clients_from_unit_directions(directions))
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    q_star = float(diagnostic["min_norm_value"]) ** 2
    quantities = [
        q_star,
        float(diagnostic["min_norm_value"]),
        float(diagnostic["solver_objective_gamma"]),
        float(diagnostic["achieved_min_direction_cosine"]),
    ]
    assert min(abs(left - right)
               for index, left in enumerate(quantities)
               for right in quantities[index + 1:]) > 0.03

    validate_run_directory(tmp_path)


def test_zero_gram_entry_uses_elementwise_tolerance(monkeypatch, tmp_path):
    directions = ([1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0])
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        clients=clients_from_unit_directions(directions))
    with result_path.open() as handle:
        result = json.load(handle)
    gram = result["fedspan_diagnostics"]["round_1"][
        "cosine_gram_active"]
    assert gram[0][1] == 0.0
    gram[0][1] = 5e-9
    gram[1][0] = 5e-9
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="cosine Gram"):
        validate_run_directory(tmp_path)


def test_negative_min_norm_value_is_not_hidden_by_squaring(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    diagnostic["min_norm_value"] = -float(diagnostic["min_norm_value"])
    diagnostic["direction_solver_shortfall"] = (
        diagnostic["min_norm_value"]
        - diagnostic["achieved_min_direction_cosine"])
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="min_norm_value"):
        validate_run_directory(tmp_path)


def test_genuine_singleton_direction_status_is_certified(
        monkeypatch, tmp_path):
    base = driver_harness.broadcast_state()
    clients = {
        name: {key: value.clone() for key, value in base.items()}
        for name in driver_harness.SLICES
    }
    clients["c0"][driver_harness.B_KEY][0, 0] = 1.0
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", clients=clients)
    assert result["fedspan_diagnostics"]["round_1"]["status"] == "singleton"

    validate_run_directory(tmp_path)


def test_genuine_coefficient_limit_fallback_is_certified_before_campaign_gate(
        monkeypatch, tmp_path):
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        extra=("--fedspan_max_abs_delta_weight", "0.01"))
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["status"] == "coefficient_limit"

    with pytest.raises(E0ValidationError, match="never applied a nonzero"):
        validate_run_directory(tmp_path)


def multi_module_peft_fixture():
    second_module = "encoder.layer0.value"
    second_a = f"{second_module}.lora_A.weight"
    second_b = f"{second_module}.lora_B.weight"
    stored_scales = {
        driver_harness.MODULE: float(torch.tensor(0.57735026).item()),
        second_module: float(torch.tensor(0.6123457).item()),
    }
    recorded_scales = {
        name: value * (1.0 + 2e-9)
        for name, value in stored_scales.items()
    }
    broadcast = {
        driver_harness.A_KEY: stored_scales[driver_harness.MODULE]
        * torch.eye(16),
        driver_harness.B_KEY: torch.zeros(3, 16),
        second_a: stored_scales[second_module] * torch.eye(16),
        second_b: torch.zeros(2, 16),
    }
    clients = {}
    for index, name in enumerate(driver_harness.SLICES):
        state = {key: value.clone() for key, value in broadcast.items()}
        state[driver_harness.B_KEY][:, :2] = torch.tensor(
            driver_harness.CLIENT_B_BLOCKS[name])
        state[second_b][0, :3] = torch.tensor([
            [0.3, -0.1, 0.2],
            [-0.2, 0.4, 0.1],
            [0.1, 0.2, -0.3],
        ][index])
        clients[name] = state
    scales = ModuleScales({
        name: 2.0 * value for name, value in recorded_scales.items()
    })
    for name, value in recorded_scales.items():
        scales.records[name] = {
            "peft_scale": 2.0,
            "row_scale_mode": "peft-init",
            "row_scale_c": value,
            "measured_init_row_rms": value,
            "geometry_scale": 2.0 * value,
        }
    return broadcast, clients, scales


def test_multimodule_peft_init_scale_perturbation_passes(
        monkeypatch, tmp_path):
    broadcast, clients, scales = multi_module_peft_fixture()
    driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        row_scale="peft-init", broadcast=broadcast, clients=clients,
        scale_override=scales)

    report = validate_run_directory(tmp_path)

    residuals = report["fedspan_direction_residuals"]["round_1"]
    assert residuals["delta_gram"] >= 0.0
    assert residuals["coefficient_error"] < 1e-10


def genuine_no_active_clients():
    base = driver_harness.broadcast_state()
    return {
        name: {key: value.clone() for key, value in base.items()}
        for name in driver_harness.SLICES
    }


@pytest.mark.parametrize("fallback_kind", ["no_active", "invalid_step_norm"])
@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("direction_policy_specified",), False,
         "direction policy.*explicit"),
        (("application", "applied_delta_weights"), [1.0, 0.0, 0.0],
         "application coefficient"),
        (("delta_weight_limit",), 10.0, "coefficient limit"),
        (("solver_status",), "forged", "solver status"),
        (("max_abs_delta_weight",), 1.0,
         "maximum applied coefficient"),
    ],
)
def test_genuine_pre_geometry_fallback_fields_are_bound(
        monkeypatch, tmp_path, fallback_kind, path, replacement, message):
    if fallback_kind == "no_active":
        result, result_path = driver_harness.run_driver(
            monkeypatch, tmp_path, "frozen-a", "normmaxmin",
            clients=genuine_no_active_clients())
    else:
        _, result_path = build_run(
            monkeypatch, tmp_path, "frozen-a", "normmaxmin")
        result = replace_with_genuine_invalid_step_fallback(
            monkeypatch, tmp_path, result_path)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["status"] == fallback_kind
    target = diagnostic
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match=message):
        validate_run_directory(tmp_path)


@pytest.mark.parametrize(
    "status",
    ["no_active", "invalid_step_norm", "solver_error", "solver_failure",
     "solver_invalid", "near_cancellation", "coefficient_limit",
     "reconstruction_failure"],
)
def test_repaired_fabricated_fallback_status_is_refused(
        monkeypatch, tmp_path, status):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2)
    fabricate_zero_fallback(tmp_path, result_path, status, round_number=1)

    with pytest.raises(E0ValidationError, match="fallback|status"):
        validate_run_directory(tmp_path)


def test_repaired_singleton_status_for_multiple_active_clients_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2)
    with result_path.open() as handle:
        result = json.load(handle)
    result["fedspan_diagnostics"]["round_1"]["status"] = "singleton"
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="fallback|status"):
        validate_run_directory(tmp_path)


def test_two_round_repaired_fabricated_fallback_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2)
    fabricate_zero_fallback(
        tmp_path, result_path, "solver_failure", round_number=1)

    with pytest.raises(E0ValidationError, match="fallback|status"):
        validate_run_directory(tmp_path)


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


def test_repaired_hash_lora_rank_shape_mismatch_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    payload = load_states(tmp_path)
    for state in [payload["broadcast"], payload["global"],
                  *payload["clients"].values()]:
        state[driver_harness.A_KEY] = state[driver_harness.A_KEY][:-1, :]
        state[driver_harness.B_KEY] = state[driver_harness.B_KEY][:, :-1]
    resave_states(tmp_path, payload, repair_hashes=True)

    with result_path.open() as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    application = diagnostic["application"]
    application["broadcast_state_sha256"] = payload["broadcast_state_sha256"]
    application["client_state_sha256"] = [
        state_dict_sha256(payload["clients"][name])
        for name in result["slices"]
    ]
    application["applied_state_sha256"] = payload["global_state_sha256"]
    scale = diagnostic["module_scales"][driver_harness.MODULE]
    broadcast_b = payload["broadcast"][driver_harness.B_KEY].double()
    solved = sum(
        float(coefficient) * scale * (
            payload["clients"][name][driver_harness.B_KEY].double()
            - broadcast_b)
        for coefficient, name in zip(
            diagnostic["delta_weights"], result["slices"])
    )
    applied = scale * (
        payload["global"][driver_harness.B_KEY].double() - broadcast_b)
    diagnostic["solved_effective_step_sha256"] = direct_effective_step_sha256(
        {driver_harness.MODULE: solved})
    application["applied_effective_step_sha256"] = direct_effective_step_sha256(
        {driver_harness.MODULE: applied})
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="lora_rank|rank"):
        validate_run_directory(tmp_path)


def test_unit_labeled_nonunit_frozen_a_is_refused(monkeypatch, tmp_path):
    _, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        row_scale="unit", row_scale_c=1.25)

    with pytest.raises(E0ValidationError, match="unit|row scale"):
        validate_run_directory(tmp_path)


def manifest_row(tmp_path, lora_mode="frozen-a", arm="normmaxmin",
                 num_rounds=1):
    return {
        "run_id": tmp_path.name,
        "commit": driver_harness.CLEAN_COMMIT,
        "coordinate": lora_mode,
        "arm": arm,
        "regime": "full",
        "max_steps": 0,
        "argv": ["/test/FedCRAG/.venv/bin/python"]
                + driver_harness.build_argv(
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
    assert report["dataset_content_verified"] is True


@pytest.mark.parametrize(
    ("flag", "replacement"),
    [
        ("--model", "different-model"),
        ("--metrics", "recall@10"),
        ("--seed", "7"),
        ("--num_rounds", "9"),
        ("--local_epochs", "2"),
        ("--lora_rank", "8"),
        ("--lora_mode", "trainable-ab"),
        ("--batch_size", "64"),
        ("--eval_batch_size", "64"),
        ("--lr", "3e-5"),
        ("--max_steps_per_round", "7"),
        ("--data_root", "missing-data-root"),
        ("--frozen_a_row_scale", "peft-init"),
        ("--fedspan_step_policy", "fixed"),
        ("--fedspan_direction_policy", "maxmin-lp"),
        ("--fedspan_active_abs_tol", "2e-12"),
        ("--fedspan_active_rel_tol", "2e-8"),
        ("--fedspan_mixture_norm_tol", "2e-6"),
        ("--weight_by", "rawmaxmin"),
        ("--out", "different-output"),
    ],
)
def test_every_manifest_value_drift_is_field_specifically_refused(
        monkeypatch, tmp_path, flag, replacement):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    drifted = manifest_row(tmp_path)
    if flag not in drifted["argv"]:
        insertion = 2
        drifted["argv"][insertion:insertion] = [flag, replacement]
    else:
        drifted["argv"][drifted["argv"].index(flag) + 1] = replacement

    field = flag.removeprefix("--")
    with pytest.raises(E0ValidationError, match=field):
        validate_run_directory(tmp_path, manifest_row=drifted)


@pytest.mark.parametrize(
    ("flag", "expected_field"),
    [
        ("--weighted", "weighted"),
        ("--save_states", "save_states"),
        ("--no_grad_ckpt", "no_grad_ckpt"),
        ("--allow_dirty_provenance", "allow_dirty_provenance"),
    ],
)
def test_manifest_boolean_drift_is_field_specifically_refused(
        monkeypatch, tmp_path, flag, expected_field):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    drifted = manifest_row(tmp_path)
    if flag in drifted["argv"]:
        drifted["argv"].remove(flag)
    else:
        drifted["argv"].append(flag)

    with pytest.raises(E0ValidationError, match=expected_field):
        validate_run_directory(tmp_path, manifest_row=drifted)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--qffl_q", "2"),
        ("--afl_eta", "0.2"),
        ("--loss_sample", "1024"),
        ("--fedspan_step_norm", "0.1"),
        ("--fedspan_max_abs_delta_weight", "10"),
    ],
)
def test_manifest_optional_value_drift_is_field_specifically_refused(
        monkeypatch, tmp_path, flag, value):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    drifted = manifest_row(tmp_path)
    drifted["argv"].extend([flag, value])

    with pytest.raises(E0ValidationError, match=flag.removeprefix("--")):
        validate_run_directory(tmp_path, manifest_row=drifted)


def test_manifest_duplicate_flag_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    drifted = manifest_row(tmp_path)
    drifted["argv"].extend(["--seed", "42"])

    with pytest.raises(E0ValidationError, match="duplicate.*--seed"):
        validate_run_directory(tmp_path, manifest_row=drifted)


def test_manifest_unknown_flag_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    drifted = manifest_row(tmp_path)
    drifted["argv"].append("--unknown_e0_flag")

    with pytest.raises(E0ValidationError, match="unknown.*--unknown_e0_flag"):
        validate_run_directory(tmp_path, manifest_row=drifted)


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, "/usr/bin/python", "interpreter"),
        (1, "different_driver.py", "script"),
    ],
)
def test_manifest_command_prefix_is_bound(
        monkeypatch, tmp_path, index, replacement, message):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    drifted = manifest_row(tmp_path)
    drifted["argv"][index] = replacement

    with pytest.raises(E0ValidationError, match=message):
        validate_run_directory(tmp_path, manifest_row=drifted)


@pytest.mark.parametrize(
    ("flag", "replacement"),
    [
        ("--weight_by", "invalid-arm"),
        ("--fedspan_step_policy", "invalid-step-policy"),
        ("--fedspan_direction_policy", "invalid-direction-policy"),
    ],
)
def test_manifest_uses_driver_cli_choices(
        monkeypatch, tmp_path, flag, replacement):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    drifted = manifest_row(tmp_path)
    drifted["argv"][drifted["argv"].index(flag) + 1] = replacement

    with pytest.raises(
            E0ValidationError,
            match=f"cannot be parsed.*{flag.removeprefix('--')}.*invalid"):
        validate_run_directory(tmp_path, manifest_row=drifted)


@pytest.mark.parametrize(
    ("field", "value"),
    [("coordinate", "trainable-ab"), ("arm", "rawmaxmin"),
     ("regime", "capped-500"), ("max_steps", 500)],
)
def test_manifest_row_metadata_drift_is_refused(
        monkeypatch, tmp_path, field, value):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    drifted = manifest_row(tmp_path)
    drifted[field] = value

    with pytest.raises(E0ValidationError, match=field):
        validate_run_directory(tmp_path, manifest_row=drifted)


def test_repaired_configuration_hash_drift_is_refused(monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    with result_path.open() as handle:
        result = json.load(handle)
    result["method_contract"]["run_configuration_sha256"] = "0" * 64
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="run_configuration_sha256"):
        validate_run_directory(tmp_path, manifest_row=manifest_row(tmp_path))


def test_archived_data_content_drift_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    corpus_path = tmp_path / "archived_data" / "c1" / "corpus.jsonl"
    corpus_path.write_text(json.dumps({
        "_id": "d0", "text": "tampered", "title": None,
    }) + "\n")

    with pytest.raises(E0ValidationError, match="dataset content|data_sha256"):
        validate_run_directory(tmp_path, manifest_row=manifest_row(tmp_path))


def test_missing_archived_data_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    root = tmp_path / "archived_data"
    root.rename(tmp_path / "unavailable_archived_data")

    with pytest.raises(E0ValidationError, match="dataset-content|data root"):
        validate_run_directory(tmp_path, manifest_row=manifest_row(tmp_path))


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


# ------------------------------------- the attribution axes themselves


def test_manifest_row_scale_drift_is_refused(monkeypatch, tmp_path):
    """e0-frozen-a-uniform-full and its unit-scale twin differ in this alone."""
    build_run(monkeypatch, tmp_path, "frozen-a", "uniform")
    write_resource_record(tmp_path)

    drifted = manifest_row(tmp_path, arm="uniform")
    index = drifted["argv"].index("--frozen_a_row_scale") + 1
    drifted["argv"][index] = "peft-init"
    with pytest.raises(E0ValidationError,
                       match="launched frozen_a_row_scale='peft-init'"):
        validate_run_directory(tmp_path, manifest_row=drifted)


def test_manifest_direction_policy_drift_is_refused(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)

    drifted = manifest_row(tmp_path)
    index = drifted["argv"].index("--fedspan_direction_policy") + 1
    drifted["argv"][index] = "maxmin-lp"
    with pytest.raises(E0ValidationError,
                       match="launched fedspan_direction_policy='maxmin-lp'"):
        validate_run_directory(tmp_path, manifest_row=drifted)


def test_frozen_a_run_that_defaulted_its_row_scale_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "uniform")
    with result_path.open() as handle:
        result = json.load(handle)
    result["method_contract"]["frozen_a_row_scale_specified"] = False
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="row scale was not specified explicitly"):
        validate_run_directory(tmp_path)


# ------------------------------------------- the applied step magnitude


def test_resolved_step_norm_must_equal_the_median_active_client_norm(
        monkeypatch, tmp_path):
    """The c double-count made these two persisted numbers disagree by 1.73x.

    They document one quantity in one file and nothing compared them.
    """
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    validate_run_directory(tmp_path)

    with result_path.open() as handle:
        result = json.load(handle)
    norms = result["client_delta_norms"]["round_1"]
    result["client_delta_norms"]["round_1"] = {
        name: value / math.sqrt(3.0) for name, value in norms.items()}
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="median active client delta norm"):
        validate_run_directory(tmp_path)


def test_a_self_consistent_but_wrong_step_magnitude_is_refused(
        monkeypatch, tmp_path):
    """Every number the run wrote about itself is rescaled together."""
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    diagnostic["resolved_step_norm"] *= 1.5
    diagnostic["requested_step_norm"] = diagnostic["resolved_step_norm"]
    diagnostic["application"]["applied_step_norm"] *= 1.5
    result["client_delta_norms"]["round_1"] = {
        name: value * 1.5
        for name, value in result["client_delta_norms"]["round_1"].items()}
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError,
                       match="applied step norm recomputed from the persisted"):
        validate_run_directory(tmp_path)


@pytest.mark.parametrize(
    "field", ["solved_effective_step_sha256", "applied_effective_step_sha256"])
def test_a_forged_effective_step_hash_is_refused(
        monkeypatch, tmp_path, field):
    """These were only length-checked, so any 64 hex characters passed."""
    _, result_path = build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    target = (diagnostic if field in diagnostic
              else diagnostic["application"])
    target[field] = "0" * 64
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match=f"{field} does not match"):
        validate_run_directory(tmp_path)


# --------------------------------------------------- the no-op FedSpan arm


def idle_clients():
    """Clients that trained to exactly the broadcast: every round no-ops."""
    return {name: driver_harness.broadcast_state()
            for name in driver_harness.SLICES}


def test_a_fedspan_arm_that_never_applied_anything_is_refused(
        monkeypatch, tmp_path):
    """Worst silent outcome available to E0: the frozen baseline's numbers
    reported under the FedSpan label, at the healthiest possible headroom."""
    driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2,
        clients=idle_clients())

    with pytest.raises(E0ValidationError,
                       match="never applied a nonzero update"):
        validate_run_directory(tmp_path)


def test_a_healthy_fedspan_run_reports_its_fallback_count(
        monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2)
    report = validate_run_directory(tmp_path)

    assert report["fedspan_fallback_rounds"] == 0
    assert report["fedspan_applied_rounds"] == 2
