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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import driver_harness  # noqa: E402
import validate_e0 as validator  # noqa: E402
from aggregation_schemes import (  # noqa: E402
    ModuleScales,
    apply_fedspan_update,
    fedspan_delta_weights,
    state_dict_sha256,
)
from validate_e0 import (  # noqa: E402
    E0ValidationError,
    validate_run_directory as _validate_run_directory,
)

E0_CELLS = [
    ("trainable-ab", "uniform"),
    ("trainable-ab", "rawmaxmin"),
    ("frozen-a", "uniform"),
    ("frozen-a", "rawmaxmin"),
    ("frozen-a", "normmaxmin"),
]


SOURCE_FILES = (
    "federated_forgetting.py",
    "aggregation_schemes.py",
    "fedcrag_common.py",
    "requirements.txt",
)


def validate_run_directory(run_directory, manifest_row=None,
                           execution_source_root=None):
    """Give manifest tests their independent frozen execution-source anchor."""
    if manifest_row is not None and execution_source_root is None:
        execution_source_root = prepare_execution_source(Path(run_directory))
    if execution_source_root is None:
        return _validate_run_directory(
            run_directory, manifest_row=manifest_row)
    return _validate_run_directory(
        run_directory, manifest_row=manifest_row,
        execution_source_root=execution_source_root)


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


def translate_round_b_state_and_repair_hashes(
        tmp_path, result_path, round_number,
        translation=2.0 ** -8):
    """Translate a whole round without changing any within-round delta."""
    payload = load_states(tmp_path, round_number)
    with Path(result_path).open() as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"][f"round_{round_number}"]
    application = diagnostic["application"]

    states = [payload["broadcast"], payload["global"],
              *payload["clients"].values()]
    original_b = [state[driver_harness.B_KEY].clone() for state in states]
    for state in states:
        state[driver_harness.B_KEY] = (
            state[driver_harness.B_KEY] + translation)
    for before, state in zip(original_b, states):
        assert torch.equal(
            state[driver_harness.B_KEY].double() - before.double(),
            torch.full_like(before.double(), translation))
    for before, state in zip(original_b[1:], states[1:]):
        assert torch.equal(
            state[driver_harness.B_KEY].double()
            - states[0][driver_harness.B_KEY].double(),
            before.double() - original_b[0].double())
    resave_states(
        tmp_path, payload, round_number=round_number, repair_hashes=True)

    application["broadcast_state_sha256"] = (
        payload["broadcast_state_sha256"])
    application["client_state_sha256"] = [
        state_dict_sha256(payload["clients"][name])
        for name in result["slices"]
    ]
    application["applied_state_sha256"] = payload["global_state_sha256"]

    scale = float(diagnostic["module_scales"][driver_harness.MODULE])
    broadcast_b = payload["broadcast"][driver_harness.B_KEY].double()
    client_blocks = [
        scale * (payload["clients"][name][driver_harness.B_KEY].double()
                 - broadcast_b)
        for name in result["slices"]
    ]
    solved_block = sum(
        float(coefficient) * block
        for coefficient, block in zip(
            diagnostic["delta_weights"], client_blocks)
    )
    applied_block = scale * (
        payload["global"][driver_harness.B_KEY].double() - broadcast_b)
    diagnostic["solved_effective_step_sha256"] = (
        direct_effective_step_sha256(
            {driver_harness.MODULE: solved_block}))
    application["applied_effective_step_sha256"] = (
        direct_effective_step_sha256(
            {driver_harness.MODULE: applied_block}))
    rewrite(result_path, result)


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


def rotate_only_materialized_step(
        tmp_path, result_path, angle_degrees=45.0, round_number=1):
    """Rotate only the persisted step while preserving its small-step norm."""
    payload = load_states(tmp_path, round_number)
    with Path(result_path).open() as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"][f"round_{round_number}"]
    application = diagnostic["application"]
    scale = float(diagnostic["module_scales"][driver_harness.MODULE])
    broadcast_b = payload["broadcast"][driver_harness.B_KEY].double()
    solved_block = scale * (
        payload["global"][driver_harness.B_KEY].double() - broadcast_b)
    solved_vector = solved_block.reshape(-1)
    solved_norm = float(torch.linalg.vector_norm(solved_vector).item())
    solved_unit = solved_vector / solved_norm

    active = diagnostic["active_indices"]
    client_blocks = []
    client_norms = []
    for name in result["slices"]:
        block = scale * (
            payload["clients"][name][driver_harness.B_KEY].double()
            - broadcast_b)
        client_blocks.append(block)
        client_norms.append(float(torch.linalg.vector_norm(block).item()))
    client_unit = (
        client_blocks[active[0]].reshape(-1) / client_norms[active[0]])
    perpendicular = -(client_unit - torch.dot(
        client_unit, solved_unit) * solved_unit)
    perpendicular /= torch.linalg.vector_norm(perpendicular)
    angle = math.radians(angle_degrees)
    rotated_vector = solved_norm * (
        math.cos(angle) * solved_unit + math.sin(angle) * perpendicular)
    rotated_block = rotated_vector.reshape_as(solved_block)
    payload["global"][driver_harness.B_KEY] = (
        broadcast_b + rotated_block / scale).float()
    payload["global_state_sha256"] = state_dict_sha256(payload["global"])
    torch.save(payload, state_path(tmp_path, round_number))

    applied_block = scale * (
        payload["global"][driver_harness.B_KEY].double() - broadcast_b)
    applied_norm = float(torch.linalg.vector_norm(applied_block).item())
    direction_cosines = [None] * len(result["slices"])
    for index in active:
        direction_cosines[index] = float(
            torch.sum(client_blocks[index] * applied_block).item()
            / (client_norms[index] * applied_norm))
    application.update({
        "applied_step_norm": applied_norm,
        "max_effective_block_error": float(torch.max(torch.abs(
            applied_block - solved_block)).item()),
        "applied_direction_cosines": direction_cosines,
        "applied_min_active_cosine": min(
            value for value in direction_cosines if value is not None),
        "applied_effective_step_sha256": direct_effective_step_sha256(
            {driver_harness.MODULE: applied_block}),
        "applied_state_sha256": payload["global_state_sha256"],
    })
    rewrite(result_path, result)
    return result, float(torch.linalg.vector_norm(
        applied_block - solved_block).item())


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
    with monkeypatch.context() as local:
        local.setattr(np, "median", lambda values: 0.0)
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
    assert report["initial_boundary_checked"] is True
    assert report["continuity_boundaries_checked"] == 1
    # Headroom against false positives must stay visible, not merely pass.
    assert report["aggregate_recomputation_worst_tolerance_ratio"] < 0.5


# ------------------------------------------------------- state-chain gates


def test_repaired_round_translation_breaking_continuity_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2)
    translate_round_b_state_and_repair_hashes(
        tmp_path, result_path, round_number=2)

    with pytest.raises(E0ValidationError, match="round_1 -> round_2"):
        validate_run_directory(tmp_path)


def test_repaired_round_1_initial_boundary_replacement_is_refused(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    translate_round_b_state_and_repair_hashes(
        tmp_path, result_path, round_number=1)

    with pytest.raises(E0ValidationError, match="initial -> round_1"):
        validate_run_directory(tmp_path)


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


def test_rotated_materialized_direction_with_noninformative_bound_is_refused(
        monkeypatch, tmp_path):
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        extra=("--fedspan_step_policy", "fixed",
               "--fedspan_step_norm", "6e-6"))
    result, vector_error = rotate_only_materialized_step(
        tmp_path, result_path)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert vector_error < 5e-6
    assert diagnostic["achieved_min_direction_cosine"] > 0.68
    assert diagnostic["application"]["applied_min_active_cosine"] < 0.0

    with pytest.raises(
            E0ValidationError,
            match="materialized.*certificate|non-informative"):
        validate_run_directory(tmp_path)


def test_small_rotation_crossing_positive_certificate_is_refused(
        monkeypatch, tmp_path):
    a = 0.1
    b = math.sqrt(1.0 - a * a)
    directions = ([a, b], [a, -b], [0.0, 0.0])
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        clients=clients_from_unit_directions(directions),
        extra=("--fedspan_step_policy", "fixed",
               "--fedspan_step_norm", "2e-5"))
    result, vector_error = rotate_only_materialized_step(
        tmp_path, result_path, angle_degrees=10.0)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert vector_error < 5e-6
    assert diagnostic["achieved_min_direction_cosine"] > 0.09
    assert diagnostic["application"]["applied_min_active_cosine"] < 0.0

    with pytest.raises(
            E0ValidationError,
            match="materialized.*certificate|sign crossing"):
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


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("solver_status", 1, "solver status"),
        ("solver_message", "forged convergence", "solver message"),
        ("proposed_max_abs_delta_weight", 0.0,
         "proposed maximum coefficient"),
        ("step_reconstruction_error", 0.0,
         "coefficient-limit.*reconstruction error"),
    ],
)
def test_coefficient_limit_fallback_solver_metadata_is_bound(
        monkeypatch, tmp_path, field, replacement, message):
    result, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        extra=("--fedspan_max_abs_delta_weight", "0.01"))
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["status"] == "coefficient_limit"
    diagnostic[field] = replacement
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match=message):
        validate_run_directory(tmp_path)


def test_coefficient_limit_fallback_cannot_claim_a_solved_step_hash(
        monkeypatch, tmp_path):
    result, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        extra=("--fedspan_max_abs_delta_weight", "0.01"))
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["status"] == "coefficient_limit"
    zero = torch.zeros_like(
        load_states(tmp_path)["broadcast"][driver_harness.B_KEY],
        dtype=torch.float64)
    diagnostic["solved_effective_step_sha256"] = (
        direct_effective_step_sha256({driver_harness.MODULE: zero}))
    rewrite(result_path, result)

    with pytest.raises(
            E0ValidationError,
            match="fallback.*solved_effective_step_sha256|solved step hash"):
        validate_run_directory(tmp_path)


def install_genuine_reconstruction_failure(
        monkeypatch, tmp_path, direction_policy="minnorm"):
    healthy_clients = genuine_no_active_clients()
    healthy_clients["c0"][driver_harness.B_KEY][0, 0] = 1.0
    healthy_clients["c1"][driver_harness.B_KEY][0, 0] = 0.75
    healthy_clients["c2"][driver_harness.B_KEY][0, 0] = 0.6
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2,
        direction_policy=direction_policy, clients=healthy_clients,
        extra=("--fedspan_step_policy", "fixed",
               "--fedspan_step_norm", "1.0"))
    payload = load_states(tmp_path, round_number=1)
    healthy_round_payload = copy.deepcopy(payload)
    a = 1.5e-5
    b = math.sqrt(1.0 - a * a)
    clients = clients_from_unit_directions(
        ([a, b], [a, -b], [0.0, 0.0]))
    states = [clients[name] for name in driver_harness.SLICES]
    scales = driver_harness.module_scales("frozen-a")
    diagnostic = fedspan_delta_weights(
        states, payload["broadcast"], scales, step_norm=1.0,
        step_policy="fixed", direction_policy=direction_policy,
        active_abs_tol=1e-12, active_rel_tol=1e-8,
        mixture_norm_tol=1e-6, max_abs_delta_weight=None)
    assert diagnostic["status"] == "reconstruction_failure"
    assert diagnostic["solver_message"].startswith(
        "coefficient reconstruction produced norm ")
    global_state, application = apply_fedspan_update(
        payload["broadcast"], states, diagnostic, scales)
    diagnostic = {**diagnostic, "application": application}
    payload["clients"] = {
        name: state for name, state in zip(driver_harness.SLICES, states)
    }
    payload["global"] = global_state
    payload["global_state_sha256"] = state_dict_sha256(global_state)
    torch.save(payload, state_path(tmp_path, round_number=1))
    torch.save(healthy_round_payload, state_path(tmp_path, round_number=2))
    with result_path.open() as handle:
        result = json.load(handle)
    result["fedspan_diagnostics"]["round_2"] = copy.deepcopy(
        result["fedspan_diagnostics"]["round_1"])
    result["client_delta_norms"]["round_2"] = copy.deepcopy(
        result["client_delta_norms"]["round_1"])
    result["fedspan_diagnostics"]["round_1"] = diagnostic
    result["client_delta_norms"]["round_1"] = {
        name: diagnostic["client_norms"][index]
        for index, name in enumerate(driver_harness.SLICES)
    }
    rewrite(result_path, result)
    return result, result_path


def test_genuine_reconstruction_failure_message_validates(
        monkeypatch, tmp_path):
    install_genuine_reconstruction_failure(monkeypatch, tmp_path)

    report = validate_run_directory(tmp_path)

    assert report["fedspan_fallback_rounds"] == 1
    assert report["fedspan_applied_rounds"] == 1


def test_reconstruction_failure_with_outer_success_message_is_refused(
        monkeypatch, tmp_path):
    result, result_path = install_genuine_reconstruction_failure(
        monkeypatch, tmp_path)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    solver = diagnostic["min_norm_solver"]
    diagnostic["solver_message"] = (
        "away-step Frank-Wolfe "
        f"{'converged' if solver['converged'] else 'STALLED'} at duality "
        f"gap {solver['gap']:.3e} after {solver['iterations']} iterations")
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="reconstruction.*message"):
        validate_run_directory(tmp_path)


def test_fallback_persisted_global_must_be_bit_exact_broadcast(
        monkeypatch, tmp_path):
    result, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", num_rounds=2,
        extra=("--fedspan_max_abs_delta_weight", "1.0"))
    assert result["fedspan_diagnostics"]["round_1"]["fallback"] is None
    diagnostic = result["fedspan_diagnostics"]["round_2"]
    assert diagnostic["status"] == "coefficient_limit"
    payload = load_states(tmp_path, round_number=2)
    payload["global"][driver_harness.B_KEY][0, -1] += 2e-10
    assert not torch.equal(
        payload["global"][driver_harness.B_KEY],
        payload["broadcast"][driver_harness.B_KEY])
    payload["global_state_sha256"] = state_dict_sha256(payload["global"])
    torch.save(payload, state_path(tmp_path, round_number=2))
    scale = float(diagnostic["module_scales"][driver_harness.MODULE])
    applied = scale * (
        payload["global"][driver_harness.B_KEY].double()
        - payload["broadcast"][driver_harness.B_KEY].double())
    application = diagnostic["application"]
    application["applied_state_sha256"] = payload["global_state_sha256"]
    application["applied_effective_step_sha256"] = (
        direct_effective_step_sha256({driver_harness.MODULE: applied}))
    rewrite(result_path, result)

    with pytest.raises(
            E0ValidationError,
            match="fallback.*global.*broadcast|exact zero"):
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
        monkeypatch, tmp_path, path, replacement, message):
    result, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        clients=genuine_no_active_clients())
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["status"] == "no_active"
    target = diagnostic
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match=message):
        validate_run_directory(tmp_path)


def test_genuine_invalid_step_fixture_is_independently_classified(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    result = replace_with_genuine_invalid_step_fallback(
        monkeypatch, tmp_path, result_path)
    assert result["fedspan_diagnostics"]["round_1"]["status"] == (
        "invalid_step_norm")
    assert float(np.median([1.0, 2.0, 3.0])) == 2.0

    with pytest.raises(
            E0ValidationError,
            match="production-impossible.*invalid_step_norm"):
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


@pytest.mark.parametrize("iterations", ["32", 0, 20001, True])
def test_min_norm_solver_iterations_are_bound(
        monkeypatch, tmp_path, iterations):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    result["fedspan_diagnostics"]["round_1"]["min_norm_solver"][
        "iterations"] = iterations
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="iterations"):
        validate_run_directory(tmp_path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("solver_status", 1, "solver status"),
        ("solver_message", "forged convergence", "solver message"),
    ],
)
def test_outer_direction_solver_metadata_is_bound(
        monkeypatch, tmp_path, field, replacement, message):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    with result_path.open() as handle:
        result = json.load(handle)
    result["fedspan_diagnostics"]["round_1"][field] = replacement
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match=message):
        validate_run_directory(tmp_path)


def test_outer_maxmin_solver_message_is_bound(monkeypatch, tmp_path):
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy="maxmin-lp")
    with result_path.open() as handle:
        result = json.load(handle)
    result["fedspan_diagnostics"]["round_1"]["solver_message"] = (
        "forged LP success")
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="solver message"):
        validate_run_directory(tmp_path)


def test_maxmin_shared_min_norm_gap_semantics_are_bound(
        monkeypatch, tmp_path):
    _, result_path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy="maxmin-lp")
    with result_path.open() as handle:
        result = json.load(handle)
    solver = result["fedspan_diagnostics"]["round_1"]["min_norm_solver"]
    solver["gap"] = 1.0
    solver["converged"] = True
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="min.?norm.*gap|convergence"):
        validate_run_directory(tmp_path)


@pytest.mark.parametrize("fallback_status", [
    "coefficient_limit", "reconstruction_failure",
])
def test_maxmin_post_geometry_fallback_binds_shared_min_norm_metadata(
        monkeypatch, tmp_path, fallback_status):
    if fallback_status == "coefficient_limit":
        result, result_path = driver_harness.run_driver(
            monkeypatch, tmp_path, "frozen-a", "normmaxmin",
            direction_policy="maxmin-lp",
            extra=("--fedspan_max_abs_delta_weight", "0.01"))
    else:
        result, result_path = install_genuine_reconstruction_failure(
            monkeypatch, tmp_path, direction_policy="maxmin-lp")
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["status"] == fallback_status
    diagnostic["min_norm_solver"]["gap"] = 1.0
    diagnostic["min_norm_solver"]["converged"] = True
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="min.?norm.*gap|convergence"):
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
    result["method_contract"]["initial_adapter_state_sha256"] = (
        payload["broadcast_state_sha256"])
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


def prepare_execution_source(tmp_path):
    """Create a real frozen Git source object and bind the synthetic result."""
    root = Path(tmp_path) / "execution-source"
    if not (root / ".git").is_dir():
        root.mkdir()
        for index, name in enumerate(SOURCE_FILES):
            (root / name).write_text(f"frozen source {index}: {name}\n")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "fixture@example.invalid"],
            cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.name", "E0 Fixture"],
            cwd=root, check=True)
        subprocess.run(["git", "add", *SOURCE_FILES], cwd=root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "frozen fixture"],
            cwd=root, check=True)
        (root / ".venv" / "bin").mkdir(parents=True)
        (root / "base-python").write_text("fixture interpreter\n")
        (root / ".venv" / "bin" / "python").symlink_to(
            Path("..") / ".." / "base-python")
    full_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, capture_output=True).stdout.strip()
    commit = full_commit[:12]
    source_hashes = {
        name: hashlib.sha256(subprocess.run(
            ["git", "show", f"{full_commit}:{name}"], cwd=root, check=True,
            capture_output=True).stdout).hexdigest()
        for name in SOURCE_FILES
    }
    result_path = _single_for_test(Path(tmp_path).glob("federated_*.json"))
    with result_path.open() as handle:
        result = json.load(handle)
    result["commit"] = commit
    result["provenance"]["git_commit"] = commit
    result["provenance"]["source_sha256"] = source_hashes
    rewrite(result_path, result)
    return root


def _single_for_test(paths):
    values = list(paths)
    assert len(values) == 1
    return values[0]


def manifest_row(tmp_path, lora_mode="frozen-a", arm="normmaxmin",
                 num_rounds=1):
    source_root = prepare_execution_source(tmp_path)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
        text=True, capture_output=True).stdout.strip()[:12]
    return {
        "run_id": tmp_path.name,
        "commit": commit,
        "coordinate": lora_mode,
        "arm": arm,
        "regime": "full",
        "max_steps": 0,
        "argv": [str(source_root / ".venv" / "bin" / "python")]
                + driver_harness.build_argv(
                    tmp_path, lora_mode, arm, num_rounds=num_rounds)
                + ["--seed", "42", "--max_steps_per_round", "0"],
    }


def write_resource_record(tmp_path, num_rounds=1):
    record = {
        "schema": "fedcrag-e0-resources/1",
        "run_id": tmp_path.name,
        "started_utc": "2026-08-26T00:00:00Z",
        "finished_utc": "2026-08-26T00:00:12Z",
        "elapsed_seconds": 12.0,
        "round_elapsed_seconds": [1.0] * num_rounds,
        "determinism_probe": "separate interpreter, same environment",
        "deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cublas_workspace_config": None,
        "python_hash_seed": None,
        "torch_version": torch.__version__,
        "gpu_available": False,
        "peak_gpu_memory_mib": None,
        "gpu_memory_samples": 0,
    }
    with (tmp_path / "e0_resources.json").open("w") as handle:
        json.dump(record, handle)


def write_boundary_sidecar(path, run_id, started_wall_ns, started_mono_ns,
                           finished_wall_ns, finished_mono_ns):
    path.write_text(
        f"E0_BOUNDARY\tstart\t{run_id}\t{started_wall_ns}\t"
        f"{started_mono_ns}\n"
        f"E0_BOUNDARY\tfinish\t{run_id}\t{finished_wall_ns}\t"
        f"{finished_mono_ns}\n")
    return path


def test_a_matching_manifest_row_validates(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)

    report = validate_run_directory(
        tmp_path, manifest_row=manifest_row(tmp_path))

    assert report["manifest_verified"] is True
    assert report["launched"]["seed"] == 42
    assert report["launched"]["slices"] == list(driver_harness.SLICES)
    assert report["dataset_content_verified"] is True
    assert report["resources"] == {
        "elapsed_seconds": 12.0,
        "peak_gpu_memory_mib": None,
        "deterministic_algorithms": False,
        "round_timing_valid": False,
        "round_timing_status": "legacy-buffered-unavailable",
        "round_elapsed_seconds": None,
    }


def test_manifest_relative_repo_interpreter_validates(monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    row = manifest_row(tmp_path)
    row["argv"][0] = ".venv/bin/python"

    report = validate_run_directory(tmp_path, manifest_row=row)

    assert report["manifest_verified"] is True


def test_manifest_requires_explicit_execution_source_anchor(
        monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    row = manifest_row(tmp_path)

    with pytest.raises(E0ValidationError, match="execution.source.root"):
        _validate_run_directory(tmp_path, manifest_row=row)


def test_manifest_cli_requires_execution_source_anchor(
        monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "fedcrag-e0-manifest/1",
        "commit": "0123456789ab",
        "rows": [],
    }))
    monkeypatch.setattr(sys, "argv", [
        "validate_e0.py", str(tmp_path), "--manifest", str(manifest),
    ])

    with pytest.raises(SystemExit) as error:
        validator.main()
    assert error.value.code == 2
    assert "--execution_source_root is required" in capsys.readouterr().err


def test_manifest_same_suffix_unrelated_interpreter_is_refused(
        monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    row = manifest_row(tmp_path)
    row["argv"][0] = "/tmp/unrelated/.venv/bin/python"

    with pytest.raises(E0ValidationError, match="interpreter"):
        validate_run_directory(tmp_path, manifest_row=row)


def test_manifest_source_hash_is_verified_from_recorded_git_object(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    row = manifest_row(tmp_path)
    source_root = tmp_path / "execution-source"
    with result_path.open() as handle:
        result = json.load(handle)
    result["provenance"]["source_sha256"]["aggregation_schemes.py"] = (
        "0" * 64)
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="source_sha256"):
        _validate_run_directory(
            tmp_path, manifest_row=row,
            execution_source_root=source_root)


def test_manifest_source_anchor_must_contain_recorded_commit(
        monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    row = manifest_row(tmp_path)
    unrelated = tmp_path / "unrelated-source"
    unrelated.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=unrelated, check=True)

    with pytest.raises(E0ValidationError, match="recorded commit|Git"):
        _validate_run_directory(
            tmp_path, manifest_row=row, execution_source_root=unrelated)


def test_manifest_commit_identity_must_be_frozen_12_hex(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    row = manifest_row(tmp_path)
    source_root = tmp_path / "execution-source"
    with result_path.open() as handle:
        result = json.load(handle)
    row["commit"] = "HEAD"
    result["commit"] = "HEAD"
    result["provenance"]["git_commit"] = "HEAD"
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="12 lowercase hex"):
        _validate_run_directory(
            tmp_path, manifest_row=row,
            execution_source_root=source_root)


def test_manifest_hex_named_branch_cannot_alias_recorded_commit(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    row = manifest_row(tmp_path)
    prefix = row["commit"]
    unrelated = tmp_path / "branch-alias-source"
    unrelated.mkdir()
    for index, name in enumerate(SOURCE_FILES):
        (unrelated / name).write_text(f"unrelated {index}: {name}\n")
    subprocess.run(["git", "init", "-q"], cwd=unrelated, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=unrelated, check=True)
    subprocess.run(
        ["git", "config", "user.name", "E0 Fixture"],
        cwd=unrelated, check=True)
    subprocess.run(["git", "add", *SOURCE_FILES], cwd=unrelated, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "unrelated fixture"],
        cwd=unrelated, check=True)
    subprocess.run(["git", "branch", "-m", prefix], cwd=unrelated, check=True)
    alias_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=unrelated, check=True,
        text=True, capture_output=True).stdout.strip()
    assert not alias_commit.startswith(prefix)
    with result_path.open() as handle:
        result = json.load(handle)
    result["provenance"]["source_sha256"] = {
        name: hashlib.sha256(subprocess.run(
            ["git", "show", f"{alias_commit}:{name}"], cwd=unrelated,
            check=True, capture_output=True).stdout).hexdigest()
        for name in SOURCE_FILES
    }
    row["argv"][0] = str(unrelated / ".venv" / "bin" / "python")
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="OID.*prefix|recorded commit"):
        _validate_run_directory(
            tmp_path, manifest_row=row,
            execution_source_root=unrelated)


def test_manifest_source_hash_ignores_git_replacement_objects(
        monkeypatch, tmp_path):
    _, result_path = build_run(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    row = manifest_row(tmp_path)
    source_root = tmp_path / "execution-source"
    original = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
        text=True, capture_output=True).stdout.strip()
    for index, name in enumerate(SOURCE_FILES):
        (source_root / name).write_text(f"replacement {index}: {name}\n")
    subprocess.run(["git", "add", *SOURCE_FILES], cwd=source_root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "replacement fixture"],
        cwd=source_root, check=True)
    replacement = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
        text=True, capture_output=True).stdout.strip()
    subprocess.run(
        ["git", "replace", original, replacement], cwd=source_root,
        check=True)
    with result_path.open() as handle:
        result = json.load(handle)
    result["provenance"]["source_sha256"] = {
        name: hashlib.sha256(subprocess.run(
            ["git", "show", f"{replacement}:{name}"], cwd=source_root,
            check=True, capture_output=True).stdout).hexdigest()
        for name in SOURCE_FILES
    }
    rewrite(result_path, result)

    with pytest.raises(E0ValidationError, match="source_sha256"):
        _validate_run_directory(
            tmp_path, manifest_row=row,
            execution_source_root=source_root)


def test_manifest_interpreter_symlink_target_name_is_refused(
        monkeypatch, tmp_path):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    row = manifest_row(tmp_path)
    source_root = tmp_path / "execution-source"
    interpreter = source_root / ".venv" / "bin" / "python"
    assert interpreter.is_symlink()
    assert interpreter.resolve() == (source_root / "base-python").resolve()
    row["argv"][0] = str(source_root / "base-python")

    with pytest.raises(E0ValidationError, match="interpreter"):
        _validate_run_directory(
            tmp_path, manifest_row=row,
            execution_source_root=source_root)


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
        (0, "/tmp/unrelated/.venv/bin/python", "interpreter"),
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
    ("mutate", "message"),
    [
        (lambda argv: argv.extend(["--fedspan_step_norm", "0"]),
         "positive finite.*fedspan_step_norm"),
        (lambda argv: argv.extend(["--fedspan_step_norm", "1e-4"]),
         "median-active.*rejects.*fedspan_step_norm"),
    ],
)
def test_manifest_enforces_driver_cross_field_legality(
        monkeypatch, tmp_path, mutate, message):
    build_run(monkeypatch, tmp_path, "frozen-a", "normmaxmin")
    write_resource_record(tmp_path)
    drifted = manifest_row(tmp_path)
    if "positive finite" in message:
        index = drifted["argv"].index("--fedspan_step_policy") + 1
        drifted["argv"][index] = "fixed"
    mutate(drifted["argv"])

    with pytest.raises(E0ValidationError, match=message):
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


def test_resource_schema_v2_is_replayed_from_raw_evidence(
        monkeypatch, tmp_path):
    run_dir = tmp_path / "e0-test-row"
    build_run(monkeypatch, run_dir, "frozen-a", "normmaxmin",
              num_rounds=2)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log = log_dir / "e0-test-row.log"
    log.write_text(
        "1000\t200\tE0_ROUND_START e0-test-row 1/2\n"
        "900\t350\tE0_ROUND_END e0-test-row 1/2\n"
        "1100\t500\tE0_ROUND_START e0-test-row 2/2\n"
        "800\t900\tE0_ROUND_END e0-test-row 2/2\n")
    samples = log_dir / "e0-test-row.gpu"
    samples.write_text("100\n250\n")
    boundaries = write_boundary_sidecar(
        log_dir / "e0-test-row.boundaries", "e0-test-row",
        1_700_000_000_000_000_000, 100,
        1_600_000_000_000_000_000, 1000)
    record = {
        "schema": "fedcrag-e0-resources/2",
        "run_id": "e0-test-row",
        "started_wall_ns": 1_700_000_000_000_000_000,
        "finished_wall_ns": 1_600_000_000_000_000_000,
        "started_utc": "2023-11-14T22:13:20.000000000Z",
        "finished_utc": "2020-09-13T12:26:40.000000000Z",
        "started_mono_ns": 100,
        "finished_mono_ns": 1000,
        "elapsed_seconds": 9e-7,
        "pre_ns": 100,
        "round_ns": [150, 400],
        "between_round_ns": [150],
        "post_ns": 100,
        "round_elapsed_seconds": [1.5e-7, 4e-7],
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
        "boundaries_sha256": hashlib.sha256(
            boundaries.read_bytes()).hexdigest(),
        "gpu_available": True,
        "peak_gpu_memory_mib": 250,
        "gpu_memory_samples": 2,
        "determinism_probe": "separate interpreter, same environment",
        "deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cublas_workspace_config": None,
        "python_hash_seed": None,
        "torch_version": torch.__version__,
    }
    (run_dir / "e0_resources.json").write_text(json.dumps(record))

    report = validate_run_directory(
        run_dir, manifest_row=manifest_row(run_dir, num_rounds=2))

    assert report["resources"]["round_timing_valid"] is True
    assert report["resources"]["round_timing_status"] == \
        "measured-monotonic"
    assert report["resources"]["round_elapsed_seconds"] == \
        [1.5e-7, 4e-7]


@pytest.mark.parametrize(("target", "mutation"), [
    ("json", "timing"),
    ("json", "gpu"),
    ("json", "boundaries"),
    ("log", "raw"),
    ("samples", "raw"),
    ("boundaries", "raw"),
])
def test_resource_schema_v2_refuses_mutated_claims_or_evidence(
        monkeypatch, tmp_path, target, mutation):
    run_dir = tmp_path / "e0-test-row"
    build_run(monkeypatch, run_dir, "frozen-a", "normmaxmin")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log = log_dir / "e0-test-row.log"
    log.write_text(
        "1000\t200\tE0_ROUND_START e0-test-row 1/1\n"
        "900\t900\tE0_ROUND_END e0-test-row 1/1\n")
    samples = log_dir / "e0-test-row.gpu"
    samples.write_text("250\n")
    boundaries = write_boundary_sidecar(
        log_dir / "e0-test-row.boundaries", "e0-test-row",
        1000, 100, 900, 1000)
    record = {
        "schema": "fedcrag-e0-resources/2",
        "run_id": "e0-test-row",
        "started_wall_ns": 1000,
        "finished_wall_ns": 900,
        "started_utc": "1970-01-01T00:00:00.000001000Z",
        "finished_utc": "1970-01-01T00:00:00.000000900Z",
        "started_mono_ns": 100,
        "finished_mono_ns": 1000,
        "elapsed_seconds": 9e-7,
        "pre_ns": 100,
        "round_ns": [700],
        "between_round_ns": [],
        "post_ns": 100,
        "round_elapsed_seconds": [7e-7],
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
        "boundaries_sha256": hashlib.sha256(
            boundaries.read_bytes()).hexdigest(),
        "gpu_available": True,
        "peak_gpu_memory_mib": 250,
        "gpu_memory_samples": 1,
        "deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cublas_workspace_config": None,
        "python_hash_seed": None,
        "torch_version": torch.__version__,
    }
    resource_path = run_dir / "e0_resources.json"
    resource_path.write_text(json.dumps(record))
    if target == "json":
        if mutation == "timing":
            record["round_ns"] = [699]
        elif mutation == "boundaries":
            record["started_mono_ns"] = 50
            record["pre_ns"] = 150
            record["elapsed_seconds"] = 9.5e-7
        else:
            record["peak_gpu_memory_mib"] = 999
        resource_path.write_text(json.dumps(record))
    elif target == "log":
        log.write_text(log.read_text() + "mutation\n")
    elif target == "samples":
        samples.write_text("999\n")
    else:
        boundaries.write_text(boundaries.read_text() + "mutation\n")

    with pytest.raises(E0ValidationError):
        validate_run_directory(
            run_dir, manifest_row=manifest_row(run_dir))


def test_resource_schema_v2_requires_raw_boundary_sidecar(
        monkeypatch, tmp_path):
    run_dir = tmp_path / "e0-test-row"
    build_run(monkeypatch, run_dir, "frozen-a", "normmaxmin")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log = log_dir / "e0-test-row.log"
    log.write_text(
        "1000\t200\tE0_ROUND_START e0-test-row 1/1\n"
        "900\t900\tE0_ROUND_END e0-test-row 1/1\n")
    samples = log_dir / "e0-test-row.gpu"
    samples.write_text("250\n")
    record = {
        "schema": "fedcrag-e0-resources/2",
        "run_id": "e0-test-row",
        "started_wall_ns": 1000,
        "finished_wall_ns": 900,
        "started_utc": "1970-01-01T00:00:00.000001000Z",
        "finished_utc": "1970-01-01T00:00:00.000000900Z",
        "started_mono_ns": 100,
        "finished_mono_ns": 1000,
        "elapsed_seconds": 9e-7,
        "pre_ns": 100,
        "round_ns": [700],
        "between_round_ns": [],
        "post_ns": 100,
        "round_elapsed_seconds": [7e-7],
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
        "gpu_available": True,
        "peak_gpu_memory_mib": 250,
        "gpu_memory_samples": 1,
        "determinism_probe": "separate interpreter, same environment",
        "deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cublas_workspace_config": None,
        "python_hash_seed": None,
        "torch_version": torch.__version__,
    }
    (run_dir / "e0_resources.json").write_text(json.dumps(record))

    with pytest.raises(E0ValidationError, match="boundar"):
        validate_run_directory(
            run_dir, manifest_row=manifest_row(run_dir))


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
