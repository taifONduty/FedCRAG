"""Fail-closed guards, the activity gate, and state-hash content sensitivity.

The guards in ``apply_fedspan_update`` exist for states that shipped numerics
will not produce on their own, so they are exercised here by doctoring a
solved result before applying it. A guard that stops firing must fail a test.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregation_schemes  # noqa: E402
from aggregation_schemes import (  # noqa: E402
    FedSpanContractError,
    apply_fedspan_update,
    fedspan_delta_weights,
    state_dict_sha256,
)
from fedspan_fixtures import (  # noqa: E402
    MODULES,
    effective_delta_vector,
    federation_from_unit_directions,
    federation_with_norms,
)


def solved_federation(step_norm=0.4, direction_policy="minnorm", seed=91):
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.6, 0.8, 0.0],
        [0.2, 0.3, math.sqrt(1.0 - 0.04 - 0.09)],
    ])
    broadcast, clients, scales = federation_from_unit_directions(
        directions, [0.4, 1.3, 2.9], seed=seed)
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=step_norm,
        direction_policy=direction_policy)
    assert result["status"] == "optimal"
    return broadcast, clients, scales, result


# -------------------------------------------------- relative activity gate


def test_relative_activity_gate_excludes_a_client_above_the_absolute_floor():
    """A client above active_abs_tol but below rel_tol * largest is inactive.

    Fixture norms are 100, 3, 1 and 1e-9 with the E0 tolerances
    (abs 1e-12, rel 1e-8), so the threshold is 1e-8 * 100 = 1e-6. The fourth
    client sits six orders above the absolute floor and three below the
    relative one, so only the relative term can exclude it. The median over
    the three surviving clients is then 3.0 by hand; admitting the fourth
    would move it to 2.0.
    """
    norms = [100.0, 3.0, 1.0, 1e-9]
    broadcast, clients, scales = federation_with_norms(norms, seed=37)

    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales,
        step_policy="median-active", step_norm=None,
        direction_policy="minnorm",
        active_abs_tol=1e-12, active_rel_tol=1e-8)

    assert result["activity_threshold"] == pytest.approx(1e-6, rel=1e-9)
    assert result["client_norms"][3] > 1e-12
    assert result["active_mask"] == [True, True, True, False]
    assert result["inactive_reasons"] == [None, None, None, "zero_or_tiny_delta"]
    assert result["active_indices"] == [0, 1, 2]
    assert result["simplex_weights"][3] == 0.0
    assert result["delta_weights"][3] == 0.0
    assert result["resolved_step_norm"] == pytest.approx(3.0, rel=1e-9)

    applied, diagnostic = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)
    assert diagnostic["applied_direction_cosines"][3] is None
    assert diagnostic["applied_step_norm"] == pytest.approx(3.0, rel=1e-6)


def test_relative_activity_gate_scales_with_the_largest_client():
    """The same absolute norm is active or not depending on the largest one."""
    small = federation_with_norms([1.0, 1.0, 1e-4], seed=39)
    large = federation_with_norms([1.0e6, 1.0e6, 1e-4], seed=39)
    common = dict(step_policy="median-active", step_norm=None,
                  direction_policy="minnorm",
                  active_abs_tol=1e-12, active_rel_tol=1e-8)

    kept = fedspan_delta_weights(
        small[1], small[0], module_scales=small[2], **common)
    dropped = fedspan_delta_weights(
        large[1], large[0], module_scales=large[2], **common)

    assert kept["active_mask"] == [True, True, True]
    assert dropped["active_mask"] == [True, True, False]
    assert dropped["inactive_reasons"][2] == "zero_or_tiny_delta"
    assert dropped["delta_weights"][2] == 0.0


# ------------------------------------------------ apply-time fail-closed set


def test_scaled_coefficients_are_refused_at_apply_time():
    """A result whose coefficients no longer produce the declared step."""
    broadcast, clients, scales, result = solved_federation()
    doctored = copy.deepcopy(result)
    doctored["delta_weights"] = [1.5 * value
                                 for value in result["delta_weights"]]

    with pytest.raises(RuntimeError, match="applied FedSpan norm"):
        apply_fedspan_update(
            broadcast, clients, doctored, module_scales=scales)


def test_doctored_resolved_step_norm_is_refused_at_apply_time():
    broadcast, clients, scales, result = solved_federation()
    doctored = copy.deepcopy(result)
    doctored["resolved_step_norm"] = 2.0 * float(result["resolved_step_norm"])
    doctored["requested_step_norm"] = doctored["resolved_step_norm"]

    with pytest.raises(RuntimeError, match="applied FedSpan norm"):
        apply_fedspan_update(
            broadcast, clients, doctored, module_scales=scales)


def test_non_fallback_result_without_a_resolved_norm_is_refused():
    broadcast, clients, scales, result = solved_federation()
    doctored = copy.deepcopy(result)
    doctored["resolved_step_norm"] = None
    doctored["requested_step_norm"] = None

    with pytest.raises(FedSpanContractError, match="resolved step norm"):
        apply_fedspan_update(
            broadcast, clients, doctored, module_scales=scales)


def test_fallback_result_carrying_a_nonzero_coefficient_is_refused():
    """A zero-update fallback must not be able to move the global model."""
    directions = np.array([[1.0, 0.0], [-1.0, 0.0]])
    broadcast, clients, scales = federation_from_unit_directions(
        directions, [1.0, 1.0], seed=93)
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.2,
        direction_policy="minnorm")
    assert result["fallback"] == "zero_update"

    doctored = copy.deepcopy(result)
    doctored["delta_weights"] = [0.25, 0.0]

    with pytest.raises(FedSpanContractError, match="zero update"):
        apply_fedspan_update(
            broadcast, clients, doctored, module_scales=scales)


def test_module_scale_change_between_solve_and_apply_is_refused():
    broadcast, clients, scales, result = solved_federation()
    shifted = {name: value * 1.5 for name, value in scales.items()}

    with pytest.raises(FedSpanContractError, match="module scale changed"):
        apply_fedspan_update(
            broadcast, clients, result, module_scales=shifted)


def test_applied_state_that_does_not_match_the_solved_step_is_refused(
        monkeypatch):
    """The independent reconstruction inside apply must be load-bearing."""
    broadcast, clients, scales, result = solved_federation()
    genuine = aggregation_schemes.apply_frozen_b_delta_weights

    def perturbed(broadcast_state, client_states, coefficients, module_scales):
        state = genuine(broadcast_state, client_states, coefficients,
                        module_scales)
        key = f"{MODULES[0]}.lora_B.weight"
        state[key] = state[key] + 0.05
        return state

    monkeypatch.setattr(
        aggregation_schemes, "apply_frozen_b_delta_weights", perturbed)
    with pytest.raises(RuntimeError, match="differs from solved update"):
        apply_fedspan_update(
            broadcast, clients, result, module_scales=scales)


def test_coefficient_count_mismatch_is_refused():
    broadcast, clients, scales, result = solved_federation()
    doctored = copy.deepcopy(result)
    doctored["delta_weights"] = result["delta_weights"][:-1]

    with pytest.raises(FedSpanContractError, match="one finite value"):
        apply_fedspan_update(
            broadcast, clients, doctored, module_scales=scales)


def test_nonfinite_coefficient_is_refused():
    broadcast, clients, scales, result = solved_federation()
    doctored = copy.deepcopy(result)
    doctored["delta_weights"] = list(result["delta_weights"])
    doctored["delta_weights"][0] = float("nan")

    with pytest.raises(FedSpanContractError, match="one finite value"):
        apply_fedspan_update(
            broadcast, clients, doctored, module_scales=scales)


# -------------------------------------------- solve-time reconstruction gate


class _RescaledMath:
    """``math`` with a corrupted ``sqrt``, to break internal norm bookkeeping."""

    def __init__(self, module, factor):
        self._module = module
        self._factor = factor

    def __getattr__(self, name):
        return getattr(self._module, name)

    def sqrt(self, value):
        return self._factor * self._module.sqrt(value)


def test_inconsistent_internal_norms_fail_closed_as_reconstruction_failure(
        monkeypatch):
    """If the norms and the Gram disagree, no update may be applied.

    The coefficient formula divides by both the client norms and the mixture
    norm, so a corrupted square root leaves the solved mixture at the wrong
    length. The reconstruction check is the only thing standing between that
    and a silently mis-scaled server step.
    """
    directions = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.5, 0.86]])
    broadcast, clients, scales = federation_from_unit_directions(
        directions, [0.4, 1.3, 2.9], seed=95)

    monkeypatch.setattr(
        aggregation_schemes, "math",
        _RescaledMath(math, 1.05))
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.4,
        direction_policy="minnorm")

    assert result["status"] == "reconstruction_failure"
    assert result["fallback"] == "zero_update"
    assert result["delta_weights"] == [0.0, 0.0, 0.0]
    assert "reconstruction produced norm" in result["solver_message"]
    assert any(abs(value) > 0
               for value in result["proposed_delta_weights"])


def test_reconstruction_failure_applies_nothing(monkeypatch):
    directions = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.5, 0.86]])
    broadcast, clients, scales = federation_from_unit_directions(
        directions, [0.4, 1.3, 2.9], seed=95)
    monkeypatch.setattr(
        aggregation_schemes, "math", _RescaledMath(math, 1.05))
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.4,
        direction_policy="minnorm")
    monkeypatch.undo()

    applied, diagnostic = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)
    assert diagnostic["applied_step_norm"] == 0.0
    for key in broadcast:
        assert torch.equal(applied[key], broadcast[key])


def test_declared_coefficient_limit_reports_what_it_refused():
    broadcast, clients, scales, unlimited = solved_federation()
    largest = max(abs(value) for value in unlimited["delta_weights"])

    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.4,
        direction_policy="minnorm",
        max_abs_delta_weight=largest * 0.5)

    assert result["status"] == "coefficient_limit"
    assert result["fallback"] == "zero_update"
    assert result["delta_weights"] == [0.0, 0.0, 0.0]
    assert result["proposed_max_abs_delta_weight"] == pytest.approx(
        largest, rel=1e-12)
    assert result["delta_weight_limit"] == pytest.approx(largest * 0.5)


# --------------------------------------------------- state hash sensitivity


def hash_fixture():
    return {
        "block.lora_A.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "block.lora_B.weight": torch.tensor([[0.5], [0.25]]),
        "other.weight": torch.tensor([7.0, 8.0, 9.0]),
    }


def test_state_hash_changes_when_a_single_element_changes():
    baseline = hash_fixture()
    changed = hash_fixture()
    changed["block.lora_A.weight"] = torch.tensor([[1.0, 2.0], [3.0, 4.5]])

    assert state_dict_sha256(baseline) != state_dict_sha256(changed)


def test_state_hash_detects_the_smallest_representable_change():
    baseline = hash_fixture()
    nudged = hash_fixture()
    key = "block.lora_B.weight"
    nudged[key] = torch.nextafter(baseline[key], torch.tensor(float("inf")))

    assert not torch.equal(baseline[key], nudged[key])
    assert state_dict_sha256(baseline) != state_dict_sha256(nudged)


def test_state_hash_is_insertion_order_independent():
    baseline = hash_fixture()
    reordered = {key: baseline[key].clone()
                 for key in reversed(list(baseline))}

    assert list(reordered) != list(baseline)
    assert state_dict_sha256(reordered) == state_dict_sha256(baseline)


def test_state_hash_changes_with_dtype():
    baseline = hash_fixture()
    recast = {key: value.double() for key, value in baseline.items()}

    assert state_dict_sha256(recast) != state_dict_sha256(baseline)


def test_state_hash_changes_with_shape_and_with_key_names():
    baseline = hash_fixture()
    reshaped = {key: value.clone() for key, value in baseline.items()}
    reshaped["block.lora_A.weight"] = baseline[
        "block.lora_A.weight"].reshape(4, 1)
    renamed = {(key + "x" if key == "other.weight" else key): value.clone()
               for key, value in baseline.items()}

    assert state_dict_sha256(reshaped) != state_dict_sha256(baseline)
    assert state_dict_sha256(renamed) != state_dict_sha256(baseline)


def test_state_hash_is_stable_across_clones_and_devices():
    baseline = hash_fixture()
    cloned = {key: value.clone() for key, value in baseline.items()}
    strided = {key: value.t().contiguous().t()
               for key, value in baseline.items()}

    assert state_dict_sha256(cloned) == state_dict_sha256(baseline)
    assert state_dict_sha256(strided) == state_dict_sha256(baseline)


def test_applied_state_hash_tracks_the_applied_update():
    """The persisted hash must distinguish two genuinely different globals."""
    broadcast, clients, scales, result = solved_federation()
    other = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.8,
        direction_policy="minnorm")

    applied, diagnostic = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)
    applied_other, diagnostic_other = apply_fedspan_update(
        broadcast, clients, other, module_scales=scales)

    assert diagnostic["applied_state_sha256"] == state_dict_sha256(applied)
    assert diagnostic["applied_state_sha256"] != diagnostic_other[
        "applied_state_sha256"]
    assert diagnostic["broadcast_state_sha256"] == diagnostic_other[
        "broadcast_state_sha256"]
    assert diagnostic["applied_effective_step_sha256"] != diagnostic_other[
        "applied_effective_step_sha256"]


def test_client_hashes_follow_the_client_states():
    broadcast, clients, scales, result = solved_federation()
    _, diagnostic = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)

    assert diagnostic["client_state_sha256"] == [
        state_dict_sha256(state) for state in clients]
    nudged = [{key: value.clone() for key, value in state.items()}
              for state in clients]
    key = f"{MODULES[1]}.lora_B.weight"
    nudged[1][key] = nudged[1][key] + 1e-3
    assert state_dict_sha256(nudged[1]) != diagnostic["client_state_sha256"][1]


def test_effective_step_hash_matches_an_independent_materialization():
    broadcast, clients, scales, result = solved_federation()
    applied, diagnostic = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)

    independent = effective_delta_vector(applied, broadcast, scales)
    reference = sum(
        coefficient * effective_delta_vector(state, broadcast, scales)
        for coefficient, state in zip(result["delta_weights"], clients))
    assert torch.linalg.vector_norm(independent - reference).item() < 1e-6
    assert len(diagnostic["applied_effective_step_sha256"]) == 64
