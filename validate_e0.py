"""Validate persisted E0 result/state contracts without evaluating metrics.

The aggregate recomputation below is written inline on purpose. Calling the
production aggregation functions would only re-run the code under test, which
cannot detect a persisted global that the production code never produced.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from aggregation_schemes import state_dict_sha256
from e0_direction_oracle import (
    maximin_simplex_oracle,
    min_norm_simplex_oracle,
)

_LORA_KEY = re.compile(r"(.*)\.lora_(A|B)\.weight$")

# Server aggregates are stored as float32, so an independent recomputation in
# float64 agrees only to float32 resolution. Tolerance is relative to the
# largest magnitude in the expected tensor, floored at 1 so near-zero blocks
# keep an absolute floor. Measured worst deviation/tolerance ratio across the
# five E0 aggregation paths on four-client rounds: 1.0e-02 (trainable-ab), and
# exactly 0 on every frozen-A path. A global rescaled by 1.5x scores 5.0e+04.
_AGGREGATE_RTOL = 1e-5

_SUPPORTED_ARMS = (None, "rawmaxmin", "normmaxmin")
_DIRECTION_POLICIES = ("minnorm", "maxmin-lp")
_FEDSPAN_STATUSES = {
    "no_active", "invalid_step_norm", "singleton", "optimal",
    "solver_error", "solver_failure", "solver_invalid",
    "near_cancellation", "coefficient_limit", "reconstruction_failure",
}

_SCALAR_ATOL = 1e-10
_SCALAR_RTOL = 1e-8
_SIMPLEX_TOL = 1e-10
_STORED_A_RTOL = 1e-6
_MATERIALIZED_VECTOR_RTOL = 5e-6

# The applied step norm and the median active client norm reach the same
# quantity by different routes: one through the float64 geometry scale
# sigma*c, the other through the float32 A the adapter state carries. The
# measured gap on a float32 frozen-A state is 1.8e-08 relative; the defect
# class this gate exists for (a lost divisor, a median over the wrong set, a
# double-counted row constant) is an O(1) fraction of the norm.
_STEP_NORM_RTOL = 1e-5
_APPLIED_NORM_RTOL = 1e-9

_MANIFEST_SCHEMA = "fedcrag-e0-manifest/1"
_RESOURCE_SCHEMA = "fedcrag-e0-resources/1"
_RESOURCE_FILENAME = "e0_resources.json"


class E0ValidationError(RuntimeError):
    """A persisted E0 run violates its declared implementation contract."""


def _require(condition, message):
    if not condition:
        raise E0ValidationError(message)


def _single(paths, label):
    paths = list(paths)
    _require(len(paths) == 1, f"expected one {label}, found {len(paths)}")
    return paths[0]


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _a_keys(state):
    return sorted(key for key in state if ".lora_A.weight" in key)


def _lora_modules(state, label):
    """{module: {"A": key, "B": key}} for every complete LoRA factor pair."""
    modules = {}
    for key in state:
        match = _LORA_KEY.match(key)
        if match:
            modules.setdefault(match.group(1), {})[match.group(2)] = key
    _require(modules, f"{label}: state contains no LoRA A/B factor keys")
    incomplete = sorted(name for name, factors in modules.items()
                        if set(factors) != {"A", "B"})
    _require(not incomplete,
             f"{label}: incomplete LoRA factor keys for modules {incomplete}")
    return modules


def _validate_fixed_a(payload, round_label):
    broadcast = payload["broadcast"]
    clients = payload["clients"]
    global_state = payload["global"]
    keys = _a_keys(broadcast)
    _require(keys, f"{round_label}: frozen-A state contains no LoRA A keys")
    for key in keys:
        _require(
            torch.equal(global_state[key], broadcast[key]),
            f"{round_label}: global A changed at {key}")
        for client_name, state in clients.items():
            _require(
                torch.equal(state[key], broadcast[key]),
                f"{round_label}: client {client_name} A changed at {key}")


def _validate_lora_shapes(result, payload, round_label):
    arguments = result.get("args") or {}
    _require("lora_rank" in arguments,
             f"{round_label}: result does not declare lora_rank")
    rank = int(arguments["lora_rank"])
    if result["lora_mode"] == "frozen-a":
        _require(rank == 16,
                 f"{round_label}: frozen E0 lora_rank is {rank}, not 16")
    reference_modules = _lora_modules(
        payload["broadcast"], f"{round_label} broadcast")
    states = [("broadcast", payload["broadcast"]),
              ("global", payload["global"])] + [
        (f"client {name}", state)
        for name, state in sorted(payload["clients"].items())
    ]
    reference_shapes = None
    for label, state in states:
        modules = _lora_modules(state, f"{round_label} {label}")
        _require(sorted(modules) == sorted(reference_modules),
                 f"{round_label} {label}: LoRA module set differs")
        shapes = {}
        for name, factors in modules.items():
            a_value = state[factors["A"]]
            b_value = state[factors["B"]]
            _require(a_value.ndim == 2 and b_value.ndim == 2,
                     f"{round_label} {label}: non-matrix LoRA factor at "
                     f"'{name}'")
            _require(
                a_value.shape[0] == b_value.shape[1] == rank,
                f"{round_label} {label}: '{name}' factor rank "
                f"A={a_value.shape[0]}, B={b_value.shape[1]} does not match "
                f"lora_rank={rank}")
            shapes[name] = (tuple(a_value.shape), tuple(b_value.shape))
        if reference_shapes is None:
            reference_shapes = shapes
        else:
            _require(shapes == reference_shapes,
                     f"{round_label} {label}: LoRA outer dimensions differ "
                     "from the broadcast")
    return reference_modules


def _derive_frozen_a_geometry_scales(result, payload, diagnostic,
                                     round_label):
    """Certify A A^T and derive sigma*c from the persisted float32 A."""
    modules = _lora_modules(payload["broadcast"], f"{round_label} broadcast")
    contract = result.get("method_contract") or {}
    records = contract.get("frozen_a_row_scale_records")
    _require(isinstance(records, dict)
             and sorted(records) == sorted(modules),
             f"{round_label}: frozen-A row scale records do not cover every "
             "module")
    declared_mode = contract.get("frozen_a_row_scale")
    _require(declared_mode is not None,
             f"{round_label}: frozen-A row scale mode is missing")
    diagnostic_scales = (diagnostic or {}).get("module_scales")
    if diagnostic is not None:
        _require(isinstance(diagnostic_scales, dict)
                 and sorted(diagnostic_scales) == sorted(modules),
                 f"{round_label}: diagnostic module_scales do not cover "
                 "every module")

    recorded = {}
    derived = {}
    details = {}
    for name, factors in sorted(modules.items()):
        a_value = payload["broadcast"][factors["A"]].detach().cpu().double()
        gram_a = a_value @ a_value.T
        diagonal = torch.diagonal(gram_a)
        c_squared = float(torch.mean(diagonal).item())
        _require(_finite(c_squared) and c_squared > 0.0,
                 f"{round_label}: invalid tensor-derived row scale at "
                 f"'{name}'")
        storage_tolerance = _STORED_A_RTOL * max(1.0, c_squared)
        diagonal_error = float(torch.max(
            torch.abs(diagonal - c_squared)).item())
        off_diagonal = gram_a - torch.diag(diagonal)
        off_diagonal_error = float(torch.max(torch.abs(off_diagonal)).item())
        _require(
            diagonal_error <= storage_tolerance
            and off_diagonal_error <= storage_tolerance,
            f"{round_label}: stored A rows at '{name}' are not orthogonal "
            "with a common norm")
        row_scale_c = math.sqrt(c_squared)
        geometry_scale = 2.0 * row_scale_c
        record = records[name]
        _require(isinstance(record, dict),
                 f"{round_label}: invalid scale record for '{name}'")
        _require(record.get("row_scale_mode") == declared_mode,
                 f"{round_label}: row scale mode for '{name}' differs from "
                 "the manifest/result mode")
        for field in ("peft_scale", "row_scale_c", "geometry_scale"):
            _require(_finite(record.get(field)),
                     f"{round_label}: invalid {field} for '{name}'")
        _require(_scalar_close(record["peft_scale"], 2.0),
                 f"{round_label}: '{name}' peft_scale does not bind "
                 "lora_alpha = 2*lora_rank")
        _require(_scalar_close(record["row_scale_c"], row_scale_c),
                 f"{round_label}: '{name}' recorded row scale differs from "
                 "the scale derived from saved A")
        _require(_scalar_close(record["geometry_scale"], geometry_scale),
                 f"{round_label}: '{name}' recorded geometry scale differs "
                 "from saved A")
        if diagnostic is not None:
            _require(_scalar_close(
                diagnostic_scales[name], record["geometry_scale"]),
                f"{round_label}: diagnostic module scale for '{name}' "
                "differs from its certified scale record")
        if declared_mode == "unit":
            _require(_scalar_close(row_scale_c, 1.0),
                     f"{round_label}: unit row scale label has non-unit A at "
                     f"'{name}'")
        elif declared_mode == "peft-init":
            # The pre-orthogonalization tensor was not archived. These three
            # values can certify mode/scale consistency, but cannot prove the
            # historical origin of measured_init_row_rms more strongly.
            measured = record.get("measured_init_row_rms")
            _require(_finite(measured)
                     and _scalar_close(measured, record["row_scale_c"])
                     and _scalar_close(measured, row_scale_c),
                     f"{round_label}: peft-init row scale record for '{name}' "
                     "is inconsistent with saved A")
        else:
            try:
                numeric_mode = float(declared_mode)
            except (TypeError, ValueError):
                raise E0ValidationError(
                    f"{round_label}: unknown frozen-A row scale mode "
                    f"{declared_mode!r}")
            _require(_scalar_close(row_scale_c, numeric_mode),
                     f"{round_label}: numeric row scale declaration differs "
                     f"from saved A at '{name}'")
        recorded[name] = float(record["geometry_scale"])
        derived[name] = geometry_scale
        details[name] = {
            "c_squared": c_squared,
            "row_scale_c": row_scale_c,
            "recorded_geometry_scale": recorded[name],
            "derived_geometry_scale": geometry_scale,
            "stored_a_diagonal_error": diagonal_error,
            "stored_a_off_diagonal_error": off_diagonal_error,
        }
    return modules, recorded, derived, details


# ------------------------------------------------------------- finiteness


def _assert_finite_state(state, label):
    for key in sorted(state):
        tensor = state[key]
        if not tensor.is_floating_point():
            continue
        nonfinite = int((~torch.isfinite(tensor)).sum().item())
        _require(
            nonfinite == 0,
            f"{label}: '{key}' has {nonfinite} of {tensor.numel()} "
            "nonfinite entries")


def _states_are_identical(left, right):
    if left is None or right is None or set(left) != set(right):
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def _validate_finite_states(payload, round_label):
    _assert_finite_state(payload["broadcast"], f"{round_label} broadcast")
    _assert_finite_state(payload["global"], f"{round_label} global")
    for name, state in sorted(payload["clients"].items()):
        _assert_finite_state(state, f"{round_label} client {name}")


# --------------------------------------------- independent recomputation


def _reference_weighted_average(client_states, weights):
    """Simplex average of complete client states (the trainable-A+B path)."""
    out = {}
    for key in client_states[0]:
        accumulator = None
        for weight, state in zip(weights, client_states):
            term = weight * state[key].detach().cpu().double()
            accumulator = term if accumulator is None else accumulator + term
        out[key] = accumulator.float()
    return out


def _reference_frozen_b_delta(broadcast_state, client_states, coefficients):
    """Raw-B delta application with A copied bit-for-bit (the frozen-A path)."""
    out = {key: value.detach().cpu().clone()
           for key, value in broadcast_state.items()}
    modules = _lora_modules(broadcast_state, "broadcast")
    for factors in modules.values():
        b_key = factors["B"]
        base = broadcast_state[b_key].detach().cpu().double()
        accumulator = base.clone()
        for coefficient, state in zip(coefficients, client_states):
            if coefficient == 0.0:
                continue
            accumulator += coefficient * (
                state[b_key].detach().cpu().double() - base)
        out[b_key] = accumulator.float()
    return out


def _compare_aggregates(expected, actual, label):
    """Worst (deviation, deviation/tolerance, key); raises past tolerance."""
    _require(set(expected) == set(actual),
             f"{label}: recomputed aggregate has a different key set")
    worst_deviation = 0.0
    worst_ratio = 0.0
    worst_key = None
    for key in sorted(expected):
        want = expected[key].detach().cpu().double()
        got = actual[key].detach().cpu().double()
        _require(tuple(want.shape) == tuple(got.shape),
                 f"{label}: recomputed aggregate shape differs at '{key}'")
        if not want.numel():
            continue
        deviation = float(torch.max(torch.abs(want - got)).item())
        scale = float(torch.max(torch.abs(want)).item())
        tolerance = _AGGREGATE_RTOL * max(1.0, scale)
        _require(
            deviation <= tolerance,
            f"{label}: persisted global disagrees with the aggregate implied "
            f"by the recorded weights at '{key}': max |delta| = "
            f"{deviation:.6g} exceeds tolerance {tolerance:.6g}")
        ratio = deviation / tolerance
        if ratio > worst_ratio:
            worst_deviation, worst_ratio, worst_key = deviation, ratio, key
    return worst_deviation, worst_ratio, worst_key


def _simplex_from_recorded(weights, label):
    weights = [float(value) for value in weights]
    _require(all(_finite(value) for value in weights),
             f"{label}: recorded weights contain a nonfinite value")
    _require(min(weights) >= 0.0,
             f"{label}: recorded simplex weights contain a negative value")
    total = sum(weights)
    _require(total > 0.0, f"{label}: recorded simplex weights sum to {total}")
    return [value / total for value in weights]


def _round_coefficients(result, round_label, num_clients):
    """(kind, coefficients) the record says were applied for this round."""
    arm = result.get("weight_by_canonical")
    frozen = result["lora_mode"] == "frozen-a"
    kind = "frozen-b-delta" if frozen else "fedavg"
    _require(arm in _SUPPORTED_ARMS,
             f"{round_label}: no recomputation reference for arm '{arm}'")

    if arm is None:
        _require(not result.get("weighted"),
                 f"{round_label}: weighted run records no weighting arm")
        return kind, [1.0 / num_clients] * num_clients

    if arm == "normmaxmin":
        _require(frozen,
                 f"{round_label}: normmaxmin requires the frozen-A coordinate")
        diagnostics = result.get("fedspan_diagnostics") or {}
        diagnostic = diagnostics.get(round_label)
        _require(diagnostic is not None,
                 f"{round_label}: normmaxmin round has no fedspan diagnostics")
        _require("delta_weights" in diagnostic,
                 f"{round_label}: fedspan diagnostics record no delta weights")
        coefficients = [float(value)
                        for value in diagnostic["delta_weights"]]
        _require(len(coefficients) == num_clients,
                 f"{round_label}: recorded delta weights do not cover every "
                 "client")
        _require(all(_finite(value) for value in coefficients),
                 f"{round_label}: recorded delta weights contain a nonfinite "
                 "value")
        return kind, coefficients

    diagnostics = result.get("scheme_diagnostics") or {}
    record = diagnostics.get(round_label)
    _require(record is not None,
             f"{round_label}: {arm} round has no scheme diagnostics, so the "
             "applied weights are not recorded at full precision")
    _require(record.get("scheme") == arm,
             f"{round_label}: scheme diagnostics name "
             f"'{record.get('scheme')}', not '{arm}'")
    weights = record.get("weights")
    _require(isinstance(weights, list) and len(weights) == num_clients,
             f"{round_label}: scheme diagnostics do not record one weight "
             "per client")
    return kind, _simplex_from_recorded(weights, round_label)


def _validate_recomputed_global(result, payload, round_label):
    slices = list(result["slices"])
    clients = payload["clients"]
    _require(sorted(clients) == sorted(slices),
             f"{round_label}: persisted client states do not match the slices")
    client_states = [clients[name] for name in slices]
    kind, coefficients = _round_coefficients(result, round_label, len(slices))
    if kind == "fedavg":
        expected = _reference_weighted_average(client_states, coefficients)
    else:
        expected = _reference_frozen_b_delta(
            payload["broadcast"], client_states, coefficients)
    deviation, ratio, key = _compare_aggregates(
        expected, payload["global"], round_label)
    return {"kind": kind, "max_abs_deviation": deviation,
            "max_tolerance_ratio": ratio, "worst_key": key}


# ------------------------------------------------------- scheme contracts


def _validate_scheme_round(result, round_label):
    diagnostics = result.get("scheme_diagnostics") or {}
    record = diagnostics.get(round_label)
    if record is None:
        return
    _require(
        record.get("fallback") is None,
        f"{round_label}: {record.get('scheme')} fell back to "
        f"{record.get('fallback')} ({record.get('status')}): "
        f"{record.get('solver_message')}")
    _require(record.get("status") != "unreported",
             f"{round_label}: {record.get('scheme')} reported no solver "
             "status")


def _validate_client_delta_norms(result, round_label, slices):
    norms = (result.get("client_delta_norms") or {}).get(round_label)
    _require(norms is not None,
             f"{round_label}: run records no per-client effective delta "
             "norms")
    _require("error" not in norms,
             f"{round_label}: per-client delta norms failed: "
             f"{norms.get('error')}")
    _require(sorted(norms) == sorted(slices),
             f"{round_label}: per-client delta norms do not cover every "
             "client")
    for name, value in sorted(norms.items()):
        _require(_finite(value) and float(value) >= 0.0,
                 f"{round_label}: client {name} has an invalid delta norm "
                 f"{value}")


def _validate_direction_policy(diagnostic, round_label):
    _require(
        diagnostic.get("direction_policy_specified") is True,
        f"{round_label}: the FedSpan direction policy was not specified "
        "explicitly, so the recorded direction is an implicit default")
    _require(
        diagnostic.get("direction_policy") in _DIRECTION_POLICIES,
        f"{round_label}: unknown FedSpan direction policy "
        f"'{diagnostic.get('direction_policy')}'")
    for field in ("achieved_min_direction_cosine", "min_norm_value",
                  "direction_solver_shortfall"):
        _require(_finite(diagnostic.get(field)),
                 f"{round_label}: invalid {field}")
    solver = diagnostic.get("min_norm_solver") or {}
    _require(solver.get("converged") is True,
             f"{round_label}: the min-norm reference solver did not converge "
             f"(gap {solver.get('gap')} after {solver.get('iterations')} "
             "iterations)")


def _fedspan_module_scales(diagnostic, module_names, round_label):
    """The per-module raw-B geometry scales the run recorded, validated."""
    scales = diagnostic.get("module_scales")
    _require(
        isinstance(scales, dict) and sorted(scales) == sorted(module_names),
        f"{round_label}: fedspan diagnostics record no module scale for every "
        "LoRA module")
    resolved = {}
    for name in module_names:
        value = scales[name]
        _require(_finite(value) and float(value) > 0,
                 f"{round_label}: invalid module scale for '{name}': {value!r}")
        resolved[name] = float(value)
    return resolved


def _raw_b_delta_blocks(broadcast_state, state, scales, modules):
    """``{module: scale * (B_state - B_broadcast)}`` in float64.

    Mirrors the arithmetic the aggregation performs so the recomputation can
    be compared bit-for-bit against the persisted effective-step hashes, but
    reads every operand out of the state files rather than out of the run.
    """
    return {
        name: scales[name] * (
            state[modules[name]["B"]].detach().cpu().double()
            - broadcast_state[modules[name]["B"]].detach().cpu().double())
        for name in sorted(modules)
    }


def _effective_step_sha256(blocks):
    digest = hashlib.sha256()
    for name in sorted(blocks):
        tensor = blocks[name].detach().cpu().double().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _block_norm(blocks):
    return math.sqrt(sum(float(torch.sum(value ** 2).item())
                         for value in blocks.values()))


def _max_scalar_error(actual, reference):
    return abs(float(actual) - float(reference))


def _require_scalar(actual, reference, message):
    _require(_finite(actual) and _finite(reference)
             and _scalar_close(actual, reference),
             f"{message}: recorded {actual!r}, recomputed {reference!r}")


def _geometry_from_scales(payload, slices, modules, scales,
                          abs_tol, rel_tol):
    blocks = [
        _raw_b_delta_blocks(
            payload["broadcast"], payload["clients"][name], scales, modules)
        for name in slices
    ]
    finite = [all(torch.isfinite(value).all().item()
                  for value in client.values()) for client in blocks]
    norms = [(_block_norm(client) if is_finite else None)
             for client, is_finite in zip(blocks, finite)]
    largest = max((value for value in norms if value is not None), default=0.0)
    threshold = max(float(abs_tol), float(rel_tol) * largest)
    active_mask = [is_finite and norm > threshold
                   for is_finite, norm in zip(finite, norms)]
    active = [index for index, flag in enumerate(active_mask) if flag]
    reasons = [
        None if is_active else (
            "nonfinite_delta" if not is_finite else "zero_or_tiny_delta")
        for is_active, is_finite in zip(active_mask, finite)
    ]
    gram = None
    if active:
        gram = np.empty((len(active), len(active)), dtype=np.float64)
        for left_position, left in enumerate(active):
            for right_position, right in enumerate(active):
                inner = sum(float(torch.sum(
                    blocks[left][name] * blocks[right][name]).item())
                    for name in sorted(modules))
                gram[left_position, right_position] = (
                    inner / (norms[left] * norms[right]))
        gram = 0.5 * (gram + gram.T)
        np.fill_diagonal(gram, 1.0)
    return {
        "blocks": blocks,
        "norms": norms,
        "threshold": threshold,
        "active_mask": active_mask,
        "active": active,
        "inactive_reasons": reasons,
        "gram": gram,
    }


def _weighted_blocks(blocks, coefficients, modules):
    return {
        name: sum((float(coefficient) * blocks[index][name]
                   for index, coefficient in enumerate(coefficients)
                   if float(coefficient) != 0.0),
                  torch.zeros_like(blocks[0][name]))
        for name in sorted(modules)
    }


def _flatten_blocks(blocks):
    return torch.cat([
        blocks[name].detach().cpu().double().reshape(-1)
        for name in sorted(blocks)
    ])


def _validate_recorded_activity(diagnostic, recorded, derived, round_label):
    recorded_norms = diagnostic.get("client_norms")
    _require(isinstance(recorded_norms, list)
             and len(recorded_norms) == len(recorded["norms"]),
             f"{round_label}: fallback/status diagnostics have invalid "
             "client_norms")
    for index, (actual, expected) in enumerate(zip(
            recorded_norms, recorded["norms"])):
        if expected is None:
            _require(actual is None,
                     f"{round_label}: fallback/status client norm {index} "
                     "should be nonfinite")
        else:
            _require_scalar(
                actual, expected,
                f"{round_label}: fallback/status client norm {index} differs")
    _require_scalar(
        diagnostic.get("activity_threshold"), recorded["threshold"],
        f"{round_label}: fallback/status activity threshold differs")
    _require(diagnostic.get("active_mask") == recorded["active_mask"],
             f"{round_label}: fallback/status active mask differs from saved "
             "client tensors")
    _require(diagnostic.get("active_indices") == recorded["active"],
             f"{round_label}: fallback/status active indices differ from "
             "saved client tensors")
    _require(diagnostic.get("inactive_reasons")
             == recorded["inactive_reasons"],
             f"{round_label}: fallback/status inactive reasons differ from "
             "saved client tensors")
    _require(derived["active"] == recorded["active"],
             f"{round_label}: activity classification changes between "
             "recorded and tensor-derived geometry")


def _recorded_active_weights(diagnostic, client_count, active, round_label):
    weights = diagnostic.get("simplex_weights")
    _require(isinstance(weights, list) and len(weights) == client_count,
             f"{round_label}: recorded simplex weights do not cover every "
             "client")
    values = np.asarray(weights, dtype=np.float64)
    _require(np.all(np.isfinite(values)),
             f"{round_label}: simplex weights contain nonfinite values")
    inactive = sorted(set(range(client_count)) - set(active))
    _require(all(values[index] == 0.0 for index in inactive),
             f"{round_label}: inactive simplex coefficients are not exactly "
             "zero")
    active_weights = values[active]
    _require(float(np.min(active_weights)) >= -_SIMPLEX_TOL,
             f"{round_label}: simplex feasibility has a negative weight")
    _require(abs(float(np.sum(active_weights)) - 1.0) <= _SIMPLEX_TOL,
             f"{round_label}: simplex feasibility sum differs from one")
    return active_weights


def _validate_pre_geometry_zero_fallback(
        diagnostic, slices, status, expected_resolved, round_label):
    """Bind every field emitted before Gram construction can begin."""
    for field in ("resolved_step_norm", "requested_step_norm"):
        actual = diagnostic.get(field)
        if expected_resolved is None:
            _require(actual is None,
                     f"{round_label}: {status} fallback has unexpected "
                     f"{field}")
        elif _finite(expected_resolved):
            _require_scalar(
                actual, expected_resolved,
                f"{round_label}: {status} fallback {field} differs")
        else:
            _require(not _finite(actual),
                     f"{round_label}: {status} fallback {field} should be "
                     "nonfinite")
    _require(diagnostic.get("solver_status") is None,
             f"{round_label}: {status} fallback has an unexpected solver "
             "status")
    expected_message = ("" if status == "no_active"
                        else "resolved step norm is nonpositive or nonfinite")
    _require(diagnostic.get("solver_message") == expected_message,
             f"{round_label}: {status} fallback solver message differs")
    _require(diagnostic.get("cosine_gram_active") is None,
             f"{round_label}: {status} fallback records a cosine Gram")
    zeros = [0.0] * len(slices)
    _require(diagnostic.get("simplex_weights") == zeros,
             f"{round_label}: {status} fallback has simplex weights")
    _require(diagnostic.get("delta_weights") == zeros,
             f"{round_label}: {status} fallback has applied coefficients")
    _require(diagnostic.get("proposed_delta_weights") is None,
             f"{round_label}: {status} fallback proposes coefficients")
    for field in ("gamma", "mixture_norm", "solver_objective_gamma",
                  "solver_simplex_residual", "solver_constraint_violation",
                  "min_norm_value", "min_norm_solver",
                  "achieved_min_direction_cosine",
                  "certified_min_direction_cosine",
                  "direction_solver_shortfall",
                  "proposed_max_abs_delta_weight",
                  "step_reconstruction_error",
                  "solved_effective_step_sha256"):
        _require(diagnostic.get(field) is None,
                 f"{round_label}: {status} fallback has unexpected {field}")
    _require(diagnostic.get("max_abs_delta_weight") == 0.0,
             f"{round_label}: {status} fallback maximum applied coefficient "
             "is not zero")
    application = diagnostic.get("application") or {}
    _require(application.get("applied_delta_weights") == zeros,
             f"{round_label}: {status} fallback application coefficient "
             "list is not exactly zero")
    _require(application.get("applied_step_norm") == 0.0,
             f"{round_label}: {status} fallback materialized a nonzero step")
    _require(application.get("max_effective_block_error") == 0.0,
             f"{round_label}: {status} fallback application has block error")
    _require(application.get("applied_direction_cosines")
             == [None] * len(slices),
             f"{round_label}: {status} fallback application records "
             "direction cosines")
    _require(application.get("applied_min_active_cosine") is None,
             f"{round_label}: {status} fallback application records a "
             "minimum direction cosine")


def _validate_fedspan_direction_decision(result, payload, diagnostic,
                                         round_label):
    """Independently reconstruct geometry, optimum, and coefficients."""
    arguments = result.get("args") or {}
    required_arguments = (
        "fedspan_active_abs_tol", "fedspan_active_rel_tol",
        "fedspan_mixture_norm_tol", "fedspan_max_abs_delta_weight",
        "fedspan_step_policy", "fedspan_step_norm",
        "fedspan_direction_policy",
    )
    for field in required_arguments:
        _require(field in arguments,
                 f"{round_label}: result args omit {field}")
    _require(diagnostic.get("step_policy")
             == arguments["fedspan_step_policy"],
             f"{round_label}: step policy diagnostic differs from args")
    _require(diagnostic.get("direction_policy")
             == arguments["fedspan_direction_policy"],
             f"{round_label}: direction policy diagnostic differs from args")
    _require(diagnostic.get("declared_step_norm")
             == arguments["fedspan_step_norm"],
             f"{round_label}: declared step norm diagnostic differs from args")
    _require(
        diagnostic.get("direction_policy_specified") is True,
        f"{round_label}: the FedSpan direction policy was not specified "
        "explicitly, so the recorded direction is an implicit default")
    _require(
        diagnostic.get("delta_weight_limit")
        == arguments["fedspan_max_abs_delta_weight"],
        f"{round_label}: coefficient limit diagnostic differs from the "
        "execution contract")
    modules, recorded_scales, derived_scales, _ = (
        _derive_frozen_a_geometry_scales(
            result, payload, diagnostic, round_label))
    slices = list(result["slices"])
    recorded = _geometry_from_scales(
        payload, slices, modules, recorded_scales,
        arguments["fedspan_active_abs_tol"],
        arguments["fedspan_active_rel_tol"])
    derived = _geometry_from_scales(
        payload, slices, modules, derived_scales,
        arguments["fedspan_active_abs_tol"],
        arguments["fedspan_active_rel_tol"])
    _validate_recorded_activity(
        diagnostic, recorded, derived, round_label)

    status = diagnostic.get("status")
    _require(status in _FEDSPAN_STATUSES,
             f"{round_label}: unknown fallback/status {status!r}")
    fallback = diagnostic.get("fallback")
    active = recorded["active"]
    if not active:
        _require(status == "no_active" and fallback == "zero_update",
                 f"{round_label}: fallback/status does not match the "
                 "independently recomputed no_active branch")
        expected_resolved = (arguments["fedspan_step_norm"]
                             if arguments["fedspan_step_policy"] == "fixed"
                             else None)
        _validate_pre_geometry_zero_fallback(
            diagnostic, slices, status, expected_resolved, round_label)
        return {
            "status": status,
            "delta_gram": 0.0,
            "policy_objective_error": 0.0,
            "coefficient_error": 0.0,
            "direction_uncertainty": 0.0,
        }

    if arguments["fedspan_step_policy"] == "fixed":
        resolved = arguments["fedspan_step_norm"]
        resolved_derived = resolved
    else:
        resolved = float(np.median([
            recorded["norms"][index] for index in active
        ]))
        resolved_derived = float(np.median([
            derived["norms"][index] for index in active
        ]))
    if not _finite(resolved) or float(resolved) <= 0.0:
        _require(status == "invalid_step_norm" and fallback == "zero_update",
                 f"{round_label}: fallback/status does not match the "
                 "invalid_step_norm branch")
        _validate_pre_geometry_zero_fallback(
            diagnostic, slices, status, resolved, round_label)
        return {
            "status": status,
            "delta_gram": 0.0,
            "policy_objective_error": 0.0,
            "coefficient_error": 0.0,
            "direction_uncertainty": 0.0,
        }
    _require_scalar(
        diagnostic.get("resolved_step_norm"), resolved,
        f"{round_label}: applied norm differs from resolved norm; applied "
        "step norm recomputed from the persisted tensors and deterministic "
        "policy differs")
    _require_scalar(
        diagnostic.get("requested_step_norm"), resolved,
        f"{round_label}: requested step norm differs from resolved policy")
    _require(
        abs(float(resolved) - float(resolved_derived))
        <= _SCALAR_ATOL + _SCALAR_RTOL * max(
            abs(float(resolved)), abs(float(resolved_derived))),
        f"{round_label}: median-active step differs between recorded and "
        "tensor-derived geometry")
    persisted_norms = (result.get("client_delta_norms") or {}).get(
        round_label) or {}
    for index, name in enumerate(slices):
        _require_scalar(
            persisted_norms.get(name), derived["norms"][index],
            f"{round_label}: tensor-derived client norm for {name} differs; "
            "the median active client delta norm is not independently bound")

    recorded_gram = recorded["gram"]
    derived_gram = derived["gram"]
    recorded_diagnostic_gram = np.asarray(
        diagnostic.get("cosine_gram_active"), dtype=np.float64)
    _require(recorded_diagnostic_gram.shape == recorded_gram.shape
             and np.all(np.isfinite(recorded_diagnostic_gram)),
             f"{round_label}: invalid recorded cosine Gram")
    gram_errors = np.abs(recorded_diagnostic_gram - recorded_gram)
    gram_limits = _SCALAR_ATOL + _SCALAR_RTOL * np.abs(recorded_gram)
    _require(np.all(gram_errors <= gram_limits),
             f"{round_label}: recorded cosine Gram differs from saved "
             "client tensors")
    delta_gram = float(np.max(np.abs(recorded_gram - derived_gram)))

    min_result = min_norm_simplex_oracle(derived_gram)
    max_result = maximin_simplex_oracle(derived_gram)
    _require(status not in {"solver_error", "solver_failure",
                            "solver_invalid"},
             f"{round_label}: fallback/status {status} is invalid because "
             "the independent audit oracle found a feasible optimum")
    active_weights = _recorded_active_weights(
        diagnostic, len(slices), active, round_label)
    recorded_payoffs = recorded_gram @ active_weights
    derived_payoffs = derived_gram @ active_weights
    q_recorded = float(active_weights @ recorded_gram @ active_weights)
    q_derived = float(active_weights @ derived_gram @ active_weights)
    mixture_recorded = math.sqrt(max(q_recorded, 0.0))
    mixture_derived = math.sqrt(max(q_derived, 0.0))
    gamma_recorded = float(np.min(recorded_payoffs))
    gamma_derived = float(np.min(derived_payoffs))
    _require_scalar(
        diagnostic.get("mixture_norm"), mixture_recorded,
        f"{round_label}: recorded mixture norm differs")
    _require_scalar(
        diagnostic.get("gamma"), gamma_recorded,
        f"{round_label}: recorded direction gamma differs")
    _require_scalar(
        diagnostic.get("solver_objective_gamma"), gamma_recorded,
        f"{round_label}: solver objective gamma differs")
    _require(abs(float(diagnostic.get("solver_simplex_residual"))
                 - abs(float(np.sum(active_weights)) - 1.0))
             <= _SCALAR_ATOL,
             f"{round_label}: solver simplex residual differs")
    expected_violation = max(
        0.0, float(np.max(gamma_recorded - recorded_payoffs)))
    _require_scalar(
        diagnostic.get("solver_constraint_violation"), expected_violation,
        f"{round_label}: solver constraint violation differs")

    objective_allowance = 2.0 * delta_gram + _SCALAR_ATOL
    q_star = float(min_result["objective"])
    t_star = float(max_result["objective"])
    min_norm_value = float(diagnostic.get("min_norm_value"))
    _require_scalar(
        min_norm_value, math.sqrt(max(q_star, 0.0)),
        f"{round_label}: recorded min_norm_value differs from the square "
        "root of the independent min-norm objective")
    policy = diagnostic.get("direction_policy")
    if policy == "minnorm":
        policy_error = max(0.0, q_derived - q_star)
        _require(
            policy_error <= objective_allowance,
            f"{round_label}: minnorm direction objective suboptimality "
            f"{policy_error:.6g} exceeds {objective_allowance:.6g}")
        solver = diagnostic.get("min_norm_solver") or {}
        _require_scalar(
            solver.get("tol"), 1e-14,
            f"{round_label}: minnorm solver tolerance differs from frozen E0")
        gap = q_recorded - float(np.min(recorded_payoffs))
        _require_scalar(
            solver.get("gap"), gap,
            f"{round_label}: minnorm Frank-Wolfe certificate gap differs")
        _require(solver.get("converged") is (gap <= 1e-14),
                 f"{round_label}: min-norm reference did not converge as "
                 "required; convergence status disagrees with the replayed "
                 "certificate gap")
    elif policy == "maxmin-lp":
        policy_error = max(0.0, t_star - gamma_derived)
        _require(
            policy_error <= objective_allowance,
            f"{round_label}: maxmin-lp direction objective suboptimality "
            f"{policy_error:.6g} exceeds {objective_allowance:.6g}")
    else:
        raise E0ValidationError(
            f"{round_label}: unknown direction policy {policy!r}")

    boundary = (float(arguments["fedspan_mixture_norm_tol"])
                + math.sqrt(delta_gram + _SCALAR_ATOL))
    _require(
        mixture_derived > boundary,
        f"{round_label}: boundary-indeterminate mixture norm "
        f"{mixture_derived:.6g} is within {boundary:.6g}")

    achieved_recorded = gamma_recorded / mixture_recorded
    shortfall_recorded = min_norm_value - achieved_recorded
    _require_scalar(
        diagnostic.get("achieved_min_direction_cosine"), achieved_recorded,
        f"{round_label}: achieved direction cosine differs")
    _require_scalar(
        diagnostic.get("certified_min_direction_cosine"), achieved_recorded,
        f"{round_label}: deprecated achieved direction alias differs")
    _require_scalar(
        diagnostic.get("direction_solver_shortfall"), shortfall_recorded,
        f"{round_label}: direction_solver_shortfall differs")

    expected_coefficients = [0.0] * len(slices)
    for local_index, client_index in enumerate(active):
        expected_coefficients[client_index] = float(
            float(resolved) * active_weights[local_index]
            / (recorded["norms"][client_index] * mixture_recorded))
    coefficient_max = max(abs(value) for value in expected_coefficients)
    solved_recorded = _weighted_blocks(
        recorded["blocks"], expected_coefficients, modules)
    reconstruction_error = _block_norm(solved_recorded) - float(resolved)
    cap = arguments["fedspan_max_abs_delta_weight"]
    if cap is not None and coefficient_max > float(cap):
        expected_status, expected_fallback = "coefficient_limit", "zero_update"
    elif abs(reconstruction_error) > 1e-9 * max(1.0, float(resolved)):
        expected_status = "reconstruction_failure"
        expected_fallback = "zero_update"
    else:
        expected_status = "singleton" if len(active) == 1 else "optimal"
        expected_fallback = None
    _require(status == expected_status and fallback == expected_fallback,
             f"{round_label}: fallback/status {fallback!r}/{status!r} differs "
             f"from independent decision {expected_fallback!r}/"
             f"{expected_status!r}")

    actual_coefficients = diagnostic.get("delta_weights")
    proposed_coefficients = diagnostic.get("proposed_delta_weights")
    application = diagnostic.get("application") or {}
    application_coefficients = application.get("applied_delta_weights")
    for field, values in (("delta coefficient", actual_coefficients),
                          ("application coefficient",
                           application_coefficients)):
        _require(isinstance(values, list) and len(values) == len(slices),
                 f"{round_label}: {field} list has the wrong length")
    _require(isinstance(proposed_coefficients, list)
             and len(proposed_coefficients) == len(slices),
             f"{round_label}: proposed coefficient list has the wrong length")
    coefficient_error = 0.0
    for index, expected in enumerate(expected_coefficients):
        if index not in active:
            _require(float(actual_coefficients[index]) == 0.0
                     and float(proposed_coefficients[index]) == 0.0
                     and float(application_coefficients[index]) == 0.0,
                     f"{round_label}: inactive coefficient is not exactly "
                     "zero")
        expected_applied = 0.0 if expected_fallback is not None else expected
        for field, values, target in (
                ("delta coefficient", actual_coefficients, expected_applied),
                ("proposed coefficient", proposed_coefficients, expected),
                ("application coefficient", application_coefficients,
                 expected_applied)):
            error = abs(float(values[index]) - target)
            coefficient_error = max(coefficient_error, error)
            _require(
                error <= _SCALAR_ATOL + _SCALAR_RTOL * abs(target),
                f"{round_label}: fallback/status {field} construction "
                f"differs at client {index}; direction/coefficient formula "
                "is invalid")
    _require_scalar(
        diagnostic.get("proposed_max_abs_delta_weight"), coefficient_max,
        f"{round_label}: proposed maximum coefficient differs")
    _require(diagnostic.get("delta_weight_limit") == cap,
             f"{round_label}: coefficient limit diagnostic differs from "
             "the execution contract")
    if expected_status == "coefficient_limit":
        _require(diagnostic.get("step_reconstruction_error") is None,
                 f"{round_label}: coefficient-limit fallback records a "
                 "reconstruction error it never evaluated")
    else:
        _require_scalar(
            diagnostic.get("step_reconstruction_error"), reconstruction_error,
            f"{round_label}: step reconstruction diagnostic differs")
    recorded_max = 0.0 if expected_fallback is not None else coefficient_max
    _require_scalar(
        diagnostic.get("max_abs_delta_weight"), recorded_max,
        f"{round_label}: maximum applied coefficient differs")

    if expected_fallback is not None:
        return {
            "status": status,
            "delta_gram": delta_gram,
            "policy_objective_error": policy_error,
            "coefficient_error": coefficient_error,
            "direction_uncertainty": None,
        }

    solved_recorded = _weighted_blocks(
        recorded["blocks"], expected_coefficients, modules)
    solved_derived = _weighted_blocks(
        derived["blocks"], expected_coefficients, modules)
    vector_recorded = _flatten_blocks(solved_recorded)
    vector_derived = _flatten_blocks(solved_derived)
    vector_delta = float(torch.linalg.vector_norm(
        vector_recorded - vector_derived).item())
    vector_limit = (_SCALAR_ATOL + _SCALAR_RTOL * max(
        float(torch.linalg.vector_norm(vector_recorded).item()),
        float(torch.linalg.vector_norm(vector_derived).item())))
    _require(vector_delta <= vector_limit,
             f"{round_label}: recorded-vs-derived effective vector differs "
             "beyond certified scale perturbation")

    actual_derived = _raw_b_delta_blocks(
        payload["broadcast"], payload["global"], derived_scales, modules)
    actual_recorded = _raw_b_delta_blocks(
        payload["broadcast"], payload["global"], recorded_scales, modules)
    max_block_error = max(
        float(torch.max(torch.abs(
            actual_recorded[name] - solved_recorded[name])).item())
        for name in sorted(modules))
    _require_scalar(
        application.get("max_effective_block_error"), max_block_error,
        f"{round_label}: application block reconstruction error differs")
    materialized_limit = _MATERIALIZED_VECTOR_RTOL * max(
        1.0, float(resolved))
    recorded_actual_norm = _block_norm(actual_recorded)
    _require(
        recorded_actual_norm > materialized_limit,
        f"{round_label}: materialized step is too small for direction "
        f"certification ({recorded_actual_norm:.6g} <= "
        f"{materialized_limit:.6g})")
    recorded_application_cosines = [None] * len(slices)
    for index in active:
        dot = sum(float(torch.sum(
            recorded["blocks"][index][name] * actual_recorded[name]).item())
            for name in sorted(modules))
        recorded_application_cosines[index] = float(
            dot / (recorded["norms"][index] * recorded_actual_norm))
    application_cosines = application.get("applied_direction_cosines")
    _require(isinstance(application_cosines, list)
             and len(application_cosines) == len(slices),
             f"{round_label}: application direction cosines are malformed")
    for index, expected in enumerate(recorded_application_cosines):
        if expected is None:
            _require(application_cosines[index] is None,
                     f"{round_label}: inactive application cosine is set")
        else:
            _require_scalar(
                application_cosines[index], expected,
                f"{round_label}: application direction cosine {index} differs")
    _require_scalar(
        application.get("applied_min_active_cosine"),
        min(value for value in recorded_application_cosines
            if value is not None),
        f"{round_label}: application minimum direction cosine differs")
    actual_vector = _flatten_blocks(actual_derived)
    solved_vector = _flatten_blocks(solved_derived)
    materialization_error = float(torch.linalg.vector_norm(
        actual_vector - solved_vector).item())
    _require(materialization_error <= materialized_limit,
        f"{round_label}: materialized effective vector differs from the "
        "certified coefficient reconstruction")
    actual_norm = float(torch.linalg.vector_norm(actual_vector).item())
    solved_norm_derived = float(torch.linalg.vector_norm(solved_vector).item())
    direction_conditioning_norm = min(actual_norm, solved_norm_derived)
    _require(
        direction_conditioning_norm > materialized_limit,
        f"{round_label}: materialized step is too small for direction "
        f"certification ({direction_conditioning_norm:.6g} <= "
        f"{materialized_limit:.6g})")
    _require(abs(actual_norm - float(resolved)) <= materialized_limit,
             f"{round_label}: materialized applied norm differs from the "
             "resolved step")
    scientific_cosines = []
    for index in active:
        client_vector = _flatten_blocks(derived["blocks"][index])
        scientific_cosines.append(float(torch.dot(
            client_vector, actual_vector).item()
            / (derived["norms"][index] * actual_norm)))
    materialized_scientific_achieved = min(scientific_cosines)
    scientific_achieved = gamma_derived / mixture_derived
    direction_uncertainty = ((4.0 * delta_gram + _SCALAR_ATOL)
                             / max(mixture_derived,
                                   float(arguments[
                                       "fedspan_mixture_norm_tol"])))
    materialization_direction_uncertainty = (
        2.0 * materialization_error / direction_conditioning_norm)
    _require(
        abs(materialized_scientific_achieved - scientific_achieved)
        <= (direction_uncertainty
            + materialization_direction_uncertainty + _SCALAR_RTOL),
        f"{round_label}: materialized scientific direction differs from "
        "the certified direction beyond conditioning and materialization "
        "uncertainty")
    if policy == "minnorm":
        derived_optimum = math.sqrt(max(q_star, 0.0))
        _require(
            abs(scientific_achieved - derived_optimum)
            <= direction_uncertainty + _SCALAR_RTOL,
            f"{round_label}: normalized direction/certificate is not "
            "independently optimal")
        scientific_shortfall = derived_optimum - scientific_achieved
        _require(abs(scientific_shortfall)
                 <= direction_uncertainty + _SCALAR_RTOL,
                 f"{round_label}: normalized direction shortfall exceeds "
                 "the certified scale perturbation")
    return {
        "status": status,
        "delta_gram": delta_gram,
        "policy_objective_error": policy_error,
        "coefficient_error": coefficient_error,
        "recorded_vs_derived_vector_delta": vector_delta,
        "direction_uncertainty": direction_uncertainty,
        "materialization_direction_uncertainty": (
            materialization_direction_uncertainty),
        "scientific_achieved_cosine": scientific_achieved,
        "materialized_scientific_achieved_cosine": (
            materialized_scientific_achieved),
    }


def _validate_median_active_step(result, diagnostic, round_label):
    """The step the server took must be the median of what the clients took.

    ``resolved_step_norm`` and ``client_delta_norms`` are two records of one
    quantity written into one file by two different computations. Comparing
    them is the only check that sees a scale leaking into one and not the
    other.
    """
    if diagnostic.get("step_policy") != "median-active":
        return
    active = diagnostic.get("active_indices") or []
    resolved = diagnostic.get("resolved_step_norm")
    if not active or not _finite(resolved) or float(resolved) <= 0:
        return
    slices = list(result["slices"])
    norms = (result.get("client_delta_norms") or {}).get(round_label) or {}
    _require(
        all(0 <= index < len(slices) for index in active),
        f"{round_label}: active client index is outside the slice list")
    values = []
    for index in active:
        value = norms.get(slices[index])
        _require(_finite(value),
                 f"{round_label}: active client {slices[index]} has no finite "
                 "recorded delta norm to compare the server step against")
        values.append(float(value))
    median = statistics.median(values)
    resolved = float(resolved)
    tolerance = _STEP_NORM_RTOL * max(1.0, abs(resolved))
    _require(
        abs(median - resolved) <= tolerance,
        f"{round_label}: resolved step norm {resolved:.10g} is not the median "
        f"active client delta norm {median:.10g} (ratio "
        f"{resolved / median if median else float('inf'):.10g})")


def _validate_recomputed_fedspan_step(result, payload, diagnostic,
                                      round_label):
    """Rebuild the applied and solved effective steps from the state files."""
    application = diagnostic["application"]
    modules = _lora_modules(payload["broadcast"], f"{round_label} broadcast")
    scales = _fedspan_module_scales(diagnostic, sorted(modules), round_label)
    slices = list(result["slices"])
    coefficients = [float(value) for value in diagnostic["delta_weights"]]
    _require(len(coefficients) == len(slices),
             f"{round_label}: recorded delta weights do not cover every client")

    applied_blocks = _raw_b_delta_blocks(
        payload["broadcast"], payload["global"], scales, modules)
    applied_norm = _block_norm(applied_blocks)
    recorded_norm = float(application["applied_step_norm"])
    tolerance = _APPLIED_NORM_RTOL * max(1.0, applied_norm)
    _require(
        abs(applied_norm - recorded_norm) <= tolerance,
        f"{round_label}: applied step norm recomputed from the persisted "
        f"states is {applied_norm:.10g}, but the run recorded "
        f"{recorded_norm:.10g}")
    _require(
        application.get("applied_effective_step_sha256")
        == _effective_step_sha256(applied_blocks),
        f"{round_label}: applied_effective_step_sha256 does not match the "
        "effective step implied by the persisted broadcast and global")

    solved_hash = diagnostic.get("solved_effective_step_sha256")
    if solved_hash is None:
        return applied_norm
    client_blocks = [
        _raw_b_delta_blocks(
            payload["broadcast"], payload["clients"][name], scales, modules)
        for name in slices
    ]
    active = diagnostic.get("active_indices") or []
    _require(
        active and all(0 <= index < len(slices) for index in active),
        f"{round_label}: a solved effective step was recorded but the active "
        f"client set {active!r} cannot produce one")
    solved_blocks = {
        name: sum(coefficients[index] * client_blocks[index][name]
                  for index in active)
        for name in sorted(modules)
    }
    _require(
        solved_hash == _effective_step_sha256(solved_blocks),
        f"{round_label}: solved_effective_step_sha256 does not match the "
        "coefficient mixture implied by the persisted client states")
    return applied_norm


def _validate_fedspan_round(result, payload, round_label):
    diagnostic = result["fedspan_diagnostics"][round_label]
    application = diagnostic["application"]
    _require(
        diagnostic["step_policy"]
        == result["method_contract"]["fedspan_step_policy"],
        f"{round_label}: step policy differs from method contract")
    _require(
        diagnostic.get("direction_policy")
        == result["method_contract"].get("fedspan_direction_policy"),
        f"{round_label}: direction policy differs from method contract")
    _require(
        application["broadcast_state_sha256"]
        == payload["broadcast_state_sha256"],
        f"{round_label}: broadcast hash differs between JSON and state file")
    _require(
        application["applied_state_sha256"]
        == payload["global_state_sha256"],
        f"{round_label}: applied hash differs between JSON and state file")

    slices = result["slices"]
    expected_client_hashes = [
        state_dict_sha256(payload["clients"][name]) for name in slices
    ]
    _require(
        application["client_state_sha256"] == expected_client_hashes,
        f"{round_label}: client hashes differ between JSON and state file")

    direction_residuals = _validate_fedspan_direction_decision(
        result, payload, diagnostic, round_label)

    fallback = diagnostic.get("fallback")
    applied_norm = float(application["applied_step_norm"])
    if fallback is None:
        _validate_direction_policy(diagnostic, round_label)
        resolved = diagnostic.get("resolved_step_norm")
        _require(
            resolved is not None and math.isfinite(float(resolved))
            and float(resolved) > 0,
            f"{round_label}: successful solve lacks a positive resolved norm")
        tolerance = 5e-6 * max(1.0, float(resolved))
        _require(
            abs(applied_norm - float(resolved)) <= tolerance,
            f"{round_label}: applied norm differs from resolved norm")
        _require(
            float(application["max_effective_block_error"]) <= tolerance,
            f"{round_label}: applied effective block error exceeds tolerance")
        for field in (
                "solved_effective_step_sha256",
                "applied_effective_step_sha256"):
            value = (diagnostic[field] if field in diagnostic
                     else application[field])
            _require(
                isinstance(value, str) and len(value) == 64,
                f"{round_label}: invalid {field}")
        for field in (
                "solver_simplex_residual",
                "solver_constraint_violation"):
            value = diagnostic.get(field)
            _require(
                value is not None and math.isfinite(float(value)),
                f"{round_label}: invalid {field}")
    else:
        _require(
            applied_norm == 0.0,
            f"{round_label}: fallback must apply an exact zero update")
        _require(
            not any(float(value) != 0.0
                    for value in diagnostic["delta_weights"]),
            f"{round_label}: fallback has a nonzero applied coefficient")
        _require(application.get("max_effective_block_error") == 0.0,
                 f"{round_label}: fallback application has block error")
        _require(application.get("applied_direction_cosines")
                 == [None] * len(slices),
                 f"{round_label}: fallback application records direction "
                 "cosines")
        _require(application.get("applied_min_active_cosine") is None,
                 f"{round_label}: fallback application records a minimum "
                 "direction cosine")

    # Independent of everything above: rebuild the step from the state files
    # rather than comparing two numbers the run wrote about itself.
    _validate_median_active_step(result, diagnostic, round_label)
    recomputed_norm = _validate_recomputed_fedspan_step(
        result, payload, diagnostic, round_label)
    return {"fallback": fallback, "applied_step_norm": recomputed_norm,
            "direction_residuals": direction_residuals}


# ------------------------------------------------------- launch provenance


def _manifest_argument_parser():
    """Independent mirror of the driver's public CLI parsing semantics."""
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--slices", nargs="+",
                        default=["nfcorpus", "fiqa", "scifact", "arguana"])
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--metrics", nargs="+",
                        default=["ndcg@10", "recall@10", "recall@100"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_mode", choices=["trainable-ab", "frozen-a"],
                        default="trainable-ab")
    parser.add_argument("--frozen_a_row_scale",
                        choices=["unit", "peft-init"])
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument(
        "--weight_by",
        choices=["examples", "corpus", "maxmin", "rawmaxmin",
                 "normmaxmin", "qffl", "afl", "mgda", "fednova"],
        default="examples")
    parser.add_argument("--qffl_q", type=float, default=1.0)
    parser.add_argument("--afl_eta", type=float, default=0.1)
    parser.add_argument("--loss_sample", type=int, default=2048)
    parser.add_argument("--fedspan_step_policy",
                        choices=["fixed", "median-active"])
    parser.add_argument("--fedspan_direction_policy",
                        choices=["minnorm", "maxmin-lp"])
    parser.add_argument("--fedspan_step_norm", type=float, default=None)
    parser.add_argument("--fedspan_active_abs_tol", type=float, default=1e-12)
    parser.add_argument("--fedspan_active_rel_tol", type=float, default=1e-8)
    parser.add_argument("--fedspan_mixture_norm_tol", type=float, default=1e-6)
    parser.add_argument("--fedspan_max_abs_delta_weight", type=float,
                        default=None)
    parser.add_argument("--allow_dirty_provenance", action="store_true")
    parser.add_argument("--save_states", action="store_true")
    parser.add_argument("--max_steps_per_round", type=int, default=0)
    parser.add_argument("--no_grad_ckpt", action="store_true")
    parser.add_argument("--data_root", default="./beir_data")
    parser.add_argument("--out", default="./results")
    return parser


def _parse_manifest_argv(argv):
    parser = _manifest_argument_parser()
    known = {option for action in parser._actions
             for option in action.option_strings}
    tokens = list(argv)
    _require(len(tokens) >= 3,
             "manifest command must contain an interpreter, script, and "
             "driver arguments")
    interpreter = tokens[0]
    _require(isinstance(interpreter, str)
             and Path(interpreter).parts[-3:] == (".venv", "bin", "python"),
             "manifest command interpreter is not the canonical "
             ".venv/bin/python E0 interpreter")
    _require(tokens[1] == "federated_forgetting.py",
             "manifest command script is not federated_forgetting.py")
    command = tokens[2:]
    _require(isinstance(command[0], str) and command[0].startswith("--"),
             "manifest command has noncanonical tokens between its script "
             "and driver flags")
    seen = set()
    for token in command:
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        option = token.partition("=")[0]
        _require(option in known,
                 f"manifest command has unknown flag {option}")
        _require(option not in seen,
                 f"manifest command has duplicate flag {option}")
        seen.add(option)
    try:
        namespace, unknown = parser.parse_known_args(command)
    except (argparse.ArgumentError, ValueError, TypeError) as error:
        raise E0ValidationError(
            f"manifest command cannot be parsed: {error}") from error
    _require(not unknown,
             f"manifest command has unknown arguments {unknown!r}")
    return vars(namespace)


_EXECUTION_FIELDS = (
    "model", "slices", "metrics", "seed", "num_rounds", "local_epochs",
    "batch_size", "eval_batch_size", "lr", "lora_rank", "lora_mode",
    "frozen_a_row_scale", "weighted", "weight_by", "qffl_q", "afl_eta",
    "loss_sample", "fedspan_step_policy", "fedspan_direction_policy",
    "fedspan_step_norm", "fedspan_active_abs_tol",
    "fedspan_active_rel_tol", "fedspan_mixture_norm_tol",
    "fedspan_max_abs_delta_weight", "allow_dirty_provenance", "save_states",
    "max_steps_per_round", "no_grad_ckpt", "data_root", "out",
)


def _scalar_close(actual, reference):
    return abs(float(actual) - float(reference)) <= (
        _SCALAR_ATOL + _SCALAR_RTOL * abs(float(reference)))


def _canonical_data_fingerprints(data):
    fingerprints = {}
    for slice_name, payload in sorted(data.items()):
        digest = hashlib.sha256()
        for section in ("corpus", "train_q", "train_qrels",
                        "eval_q", "eval_qrels"):
            digest.update(section.encode("utf-8") + b"\0")
            values = payload.get(section, {})
            for key in sorted(values, key=str):
                digest.update(str(key).encode("utf-8") + b"\0")
                encoded = json.dumps(
                    values[key], sort_keys=True, ensure_ascii=False,
                    separators=(",", ":")).encode("utf-8")
                digest.update(encoded + b"\0")
        fingerprints[slice_name] = digest.hexdigest()
    return fingerprints


def _read_jsonl(path, transform):
    values = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key, value = transform(record)
            values[key] = value
    return values


def _read_qrels(path):
    values = {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, delimiter="\t")
        next(rows)
        for query_id, corpus_id, score, *_ in rows:
            values.setdefault(query_id, {})[corpus_id] = int(score)
    return values


def _load_archived_slice(data_root, slice_name):
    """Load an already-present BEIR slice without a download code path."""
    directory = Path(data_root) / slice_name
    corpus_path = directory / "corpus.jsonl"
    queries_path = directory / "queries.jsonl"
    test_path = directory / "qrels" / "test.tsv"
    for path in (corpus_path, queries_path, test_path):
        _require(path.is_file(),
                 f"independent dataset-content gate cannot load {path}")
    corpus = _read_jsonl(
        corpus_path,
        lambda row: (row.get("_id"), {
            "text": row.get("text"), "title": row.get("title"),
        }))
    queries = _read_jsonl(
        queries_path,
        lambda row: (row.get("_id"), row.get("text")))
    eval_qrels = _read_qrels(test_path)
    eval_q = {query_id: queries[query_id] for query_id in eval_qrels}
    train_path = directory / "qrels" / "train.tsv"
    if train_path.is_file():
        train_qrels = _read_qrels(train_path)
        train_q = {query_id: queries[query_id] for query_id in train_qrels}
    else:
        query_ids = sorted(eval_qrels)
        midpoint = len(query_ids) // 2
        train_ids = set(query_ids[:midpoint])
        eval_ids = set(query_ids[midpoint:])
        train_q = {key: eval_q[key] for key in train_ids if key in eval_q}
        train_qrels = {key: eval_qrels[key] for key in train_ids}
        eval_q = {key: eval_q[key] for key in eval_ids if key in eval_q}
        eval_qrels = {key: eval_qrels[key] for key in eval_ids}
    return {
        "corpus": corpus,
        "train_q": train_q,
        "train_qrels": train_qrels,
        "eval_q": eval_q,
        "eval_qrels": eval_qrels,
    }


def _frozen_configuration_sha256(arguments, data_sha256, row_scale):
    fields = {
        "frozen_a_row_scale": row_scale,
        "fedspan_direction_policy": arguments.get(
            "fedspan_direction_policy"),
        "slices": arguments["slices"],
        "metrics": arguments["metrics"],
        "num_rounds": arguments["num_rounds"],
        "local_epochs": arguments["local_epochs"],
        "batch_size": arguments["batch_size"],
        "eval_batch_size": arguments.get("eval_batch_size"),
        "lr": arguments["lr"],
        "lora_rank": arguments["lora_rank"],
        "lora_mode": arguments["lora_mode"],
        "weighted": arguments["weighted"],
        "weight_by": arguments["weight_by"],
        "qffl_q": arguments["qffl_q"],
        "afl_eta": arguments["afl_eta"],
        "loss_sample": arguments["loss_sample"],
        "max_steps_per_round": arguments["max_steps_per_round"],
        "no_grad_ckpt": arguments.get("no_grad_ckpt", False),
        "fedspan_step_policy": arguments["fedspan_step_policy"],
        "fedspan_step_norm": arguments["fedspan_step_norm"],
        "fedspan_active_abs_tol": arguments["fedspan_active_abs_tol"],
        "fedspan_active_rel_tol": arguments["fedspan_active_rel_tol"],
        "fedspan_mixture_norm_tol": arguments["fedspan_mixture_norm_tol"],
        "fedspan_max_abs_delta_weight": arguments[
            "fedspan_max_abs_delta_weight"],
        "dirty_provenance_override": arguments.get(
            "allow_dirty_provenance", False),
        "data_sha256": data_sha256,
    }
    encoded = json.dumps(
        fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_configuration_hash(result, arguments, data_sha256):
    if result["lora_mode"] != "frozen-a":
        return
    contract = result.get("method_contract") or {}
    expected = _frozen_configuration_sha256(
        arguments, data_sha256, contract.get("frozen_a_row_scale"))
    _require(
        contract.get("run_configuration_sha256") == expected,
        "run_configuration_sha256 does not match the canonical execution "
        "contract and dataset fingerprints")


def _validate_manifest_row(result, row, run_id):
    launched = _parse_manifest_argv(row["argv"])
    _require(row["run_id"] == run_id,
             f"manifest row is for '{row['run_id']}', not '{run_id}'")
    _require(
        result["commit"] == row["commit"],
        f"run commit {result['commit']} differs from the manifest commit "
        f"{row['commit']}")

    arguments = result.get("args")
    _require(isinstance(arguments, dict),
             "result records no argument namespace to compare")
    contract = result.get("method_contract") or {}
    for field in _EXECUTION_FIELDS:
        _require(field in arguments,
                 f"result argument namespace omits {field}")
        expected = launched[field]
        _require(
            arguments[field] == expected,
            f"manifest row '{run_id}' launched {field}={expected!r} but the "
            f"result records {arguments[field]!r}")
    for field in ("model", "slices", "metrics", "seed", "num_rounds",
                  "lora_mode", "weighted"):
        _require(result[field] == arguments[field],
                 f"result {field} metadata differs from its argument record")
    _require(result["weight_by"] == (
        arguments["weight_by"] if arguments["weighted"] else None),
        "result weight_by metadata differs from its argument record")
    for field in ("frozen_a_row_scale", "fedspan_step_policy",
                  "fedspan_step_norm", "fedspan_direction_policy"):
        _require(contract.get(field) == arguments[field],
                 f"method contract {field} differs from its argument record")
    _require(row["coordinate"] == launched["lora_mode"],
             f"manifest row '{run_id}' coordinate differs from --lora_mode")
    expected_arm = None if row["arm"] == "uniform" else row["arm"]
    recorded_arm = launched["weight_by"] if launched["weighted"] else None
    _require(recorded_arm == expected_arm,
             f"manifest row '{run_id}' arm '{row['arm']}' differs from the "
             f"recorded weighting {recorded_arm!r}")
    _require(row.get("max_steps") == launched["max_steps_per_round"],
             f"manifest row '{run_id}' max_steps differs from argv")
    expected_regime = ("full" if launched["max_steps_per_round"] == 0
                       else f"capped-{launched['max_steps_per_round']}")
    _require(row.get("regime") == expected_regime,
             f"manifest row '{run_id}' regime differs from argv/result")

    data_root = Path(launched["data_root"]).expanduser()
    provenance = result.get("provenance") or {}
    recorded_fingerprints = provenance.get("data_sha256")
    _require(isinstance(recorded_fingerprints, dict),
             "result provenance has no recorded data_sha256 fingerprints")
    _validate_configuration_hash(result, launched, recorded_fingerprints)
    _require(
        Path(provenance.get("data_root", "")).resolve()
        == data_root.resolve(),
        "provenance data_root differs from the exact manifest data_root")
    _require(
        data_root.is_dir(),
        f"independent dataset-content gate: archived data root {data_root} "
        "is unavailable; recorded-fingerprint cross-binding passed but "
        "content verification is impossible")
    data = {
        name: _load_archived_slice(data_root, name)
        for name in launched["slices"]
    }
    fingerprints = _canonical_data_fingerprints(data)
    _require(
        provenance.get("data_sha256") == fingerprints,
        "independently recomputed dataset content/data_sha256 differs from "
        "result provenance")
    _validate_configuration_hash(result, launched, fingerprints)
    return launched, fingerprints


def _load_manifest_row(manifest_path, run_id):
    with Path(manifest_path).open() as handle:
        manifest = json.load(handle)
    _require(manifest.get("schema") == _MANIFEST_SCHEMA,
             f"manifest schema is {manifest.get('schema')!r}, expected "
             f"{_MANIFEST_SCHEMA!r}")
    commit = manifest.get("commit")
    _require(isinstance(commit, str) and commit != "unknown"
             and not commit.endswith("-dirty")
             and not commit.endswith("-unknown-worktree"),
             f"manifest does not have clean Git provenance: {commit!r}")
    rows = [row for row in manifest["rows"] if row["run_id"] == run_id]
    _require(len(rows) == 1,
             f"manifest has {len(rows)} rows for run id '{run_id}'")
    return {**rows[0], "commit": commit}


def _validate_resource_record(run_directory, result):
    path = Path(run_directory) / _RESOURCE_FILENAME
    _require(path.is_file(),
             f"run directory has no {_RESOURCE_FILENAME}, so its GPU-hour "
             "and determinism claims are unauditable")
    with path.open() as handle:
        record = json.load(handle)
    _require(record.get("schema") == _RESOURCE_SCHEMA,
             f"{_RESOURCE_FILENAME} schema is {record.get('schema')!r}, "
             f"expected {_RESOURCE_SCHEMA!r}")
    elapsed = record.get("elapsed_seconds")
    _require(_finite(elapsed) and float(elapsed) > 0,
             f"{_RESOURCE_FILENAME} has an invalid elapsed_seconds "
             f"{elapsed!r}")
    rounds = record.get("round_elapsed_seconds")
    _require(isinstance(rounds, list)
             and len(rounds) == int(result["num_rounds"])
             and all(_finite(value) and float(value) >= 0
                     for value in rounds),
             f"{_RESOURCE_FILENAME} does not carry one finite elapsed time "
             f"for each of the {result['num_rounds']} rounds")
    _require(isinstance(record.get("deterministic_algorithms"), bool),
             f"{_RESOURCE_FILENAME} does not record whether deterministic "
             "algorithms were enabled")
    _require(isinstance(record.get("gpu_available"), bool),
             f"{_RESOURCE_FILENAME} does not record GPU availability")
    peak = record.get("peak_gpu_memory_mib")
    if record["gpu_available"]:
        _require(_finite(peak) and float(peak) > 0,
                 f"{_RESOURCE_FILENAME} reports a GPU but no positive peak "
                 f"memory: {peak!r}")
    else:
        _require(peak is None,
                 f"{_RESOURCE_FILENAME} reports no GPU but a peak memory "
                 f"{peak!r}")
    return {
        "elapsed_seconds": float(elapsed),
        "peak_gpu_memory_mib": (float(peak) if record["gpu_available"]
                                else None),
        "deterministic_algorithms": record["deterministic_algorithms"],
    }


# ------------------------------------------------------------- entry point


def validate_run_directory(run_directory, manifest_row=None):
    run_directory = Path(run_directory)
    result_path = _single(
        run_directory.glob("federated_*.json"), "federated result JSON")
    with result_path.open() as handle:
        result = json.load(handle)

    commit = result.get("commit")
    _require(
        isinstance(commit, str) and commit != "unknown"
        and not commit.endswith("-dirty")
        and not commit.endswith("-unknown-worktree"),
        "result does not have clean Git provenance")

    num_rounds = int(result["num_rounds"])
    state_paths = sorted(run_directory.glob("states_*.pt"))
    _require(
        len(state_paths) == num_rounds,
        f"expected {num_rounds} state files, found {len(state_paths)}")

    contract = result.get("method_contract") or {}
    if result["lora_mode"] == "frozen-a":
        # An implicit 'unit' default silently reintroduces the ~1.73x B->dW
        # rescale the frozen-A coordinate axis exists to isolate.
        _require(
            contract.get("frozen_a_row_scale_specified") is True,
            "frozen-A run's row scale was not specified explicitly, so the "
            f"recorded scale {contract.get('frozen_a_row_scale')!r} is an "
            "implicit default")

    launched = None
    data_fingerprints = None
    resources = None
    if manifest_row is not None:
        launched, data_fingerprints = _validate_manifest_row(
            result, manifest_row, run_directory.name)
        resources = _validate_resource_record(run_directory, result)
    elif result["lora_mode"] == "frozen-a":
        provenance = result.get("provenance") or {}
        recorded_fingerprints = provenance.get("data_sha256")
        if isinstance(recorded_fingerprints, dict):
            _validate_configuration_hash(
                result, result.get("args") or {}, recorded_fingerprints)

    worst_ratio = 0.0
    worst_round = None
    fallback_rounds = 0
    applied_rounds = 0
    first_broadcast = None
    final_global = None
    scale_audit = {}
    direction_audit = {}
    for round_number in range(1, num_rounds + 1):
        round_label = f"round_{round_number}"
        state_path = _single(
            run_directory.glob(f"states_*_round{round_number}.pt"),
            f"{round_label} state file")
        payload = torch.load(
            state_path, map_location="cpu", weights_only=True)
        _require(
            state_dict_sha256(payload["broadcast"])
            == payload["broadcast_state_sha256"],
            f"{round_label}: persisted broadcast hash is invalid")
        _require(
            state_dict_sha256(payload["global"])
            == payload["global_state_sha256"],
            f"{round_label}: persisted global hash is invalid")

        _validate_finite_states(payload, round_label)
        _validate_lora_shapes(result, payload, round_label)
        _validate_scheme_round(result, round_label)
        _validate_client_delta_norms(result, round_label, result["slices"])
        if result["lora_mode"] == "frozen-a":
            _validate_fixed_a(payload, round_label)
            diagnostic = None
            if result.get("weight_by_canonical") == "normmaxmin":
                diagnostic = (result.get("fedspan_diagnostics") or {}).get(
                    round_label)
            _, _, _, scale_details = _derive_frozen_a_geometry_scales(
                result, payload, diagnostic, round_label)
            scale_audit[round_label] = {
                "module_count": len(scale_details),
                "max_recorded_vs_derived_scale_error": max(
                    abs(values["recorded_geometry_scale"]
                        - values["derived_geometry_scale"])
                    for values in scale_details.values()),
                "max_stored_a_diagonal_error": max(
                    values["stored_a_diagonal_error"]
                    for values in scale_details.values()),
                "max_stored_a_off_diagonal_error": max(
                    values["stored_a_off_diagonal_error"]
                    for values in scale_details.values()),
            }
        recomputation = _validate_recomputed_global(
            result, payload, round_label)
        if result.get("weight_by_canonical") == "normmaxmin":
            step = _validate_fedspan_round(result, payload, round_label)
            direction_audit[round_label] = step.get("direction_residuals")
            if step["fallback"] is not None:
                fallback_rounds += 1
            elif step["applied_step_norm"] > 0.0:
                applied_rounds += 1
        if first_broadcast is None:
            first_broadcast = payload["broadcast"]
        final_global = payload["global"]
        if recomputation["max_tolerance_ratio"] > worst_ratio:
            worst_ratio = recomputation["max_tolerance_ratio"]
            worst_round = round_label

    if result.get("weight_by_canonical") == "normmaxmin":
        # A normmaxmin arm that no-ops for its whole life reports the frozen
        # baseline's retention under the FedSpan label, and does so at the
        # healthiest possible recomputation headroom, because nothing was
        # aggregated to deviate.
        _require(
            applied_rounds > 0,
            f"the normmaxmin arm never applied a nonzero update: all "
            f"{fallback_rounds} of {num_rounds} rounds fell back to a zero "
            "step, so this run carries the frozen baseline's numbers")
        _require(
            not _states_are_identical(first_broadcast, final_global),
            "the normmaxmin arm's final global is bit-identical to the "
            "round-1 broadcast, so no server update survived the campaign")

    report = {
        "result_path": str(result_path),
        "commit": commit,
        "rounds_validated": num_rounds,
        "lora_mode": result["lora_mode"],
        "weight_by_canonical": result.get("weight_by_canonical"),
        "aggregate_recomputation_worst_tolerance_ratio": worst_ratio,
        "aggregate_recomputation_worst_round": worst_round,
        "fedspan_fallback_rounds": fallback_rounds,
        "fedspan_applied_rounds": applied_rounds,
        "manifest_verified": manifest_row is not None,
        "dataset_content_verified": data_fingerprints is not None,
        "recorded_fingerprint_cross_binding_verified": (
            manifest_row is not None),
        "frozen_a_scale_audit": scale_audit,
        "fedspan_direction_residuals": direction_audit,
    }
    if launched is not None:
        report["launched"] = launched
    if resources is not None:
        report["resources"] = resources
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    parser.add_argument("--manifest",
                        help="frozen E0 manifest that launched this run")
    parser.add_argument("--run_id",
                        help="manifest run id (default: directory name)")
    parser.add_argument(
        "--allow_missing_manifest", action="store_true",
        help="validate contracts only; the report records that the run was "
             "not checked against the manifest that launched it")
    args = parser.parse_args()

    run_id = args.run_id or Path(args.run_directory).name
    manifest_row = None
    if args.manifest:
        manifest_row = _load_manifest_row(args.manifest, run_id)
    elif not args.allow_missing_manifest:
        parser.error(
            "--manifest is required; pass --allow_missing_manifest to "
            "validate a run whose launch manifest is unavailable")

    report = validate_run_directory(args.run_directory, manifest_row)
    if not report["manifest_verified"]:
        print("WARNING: run was not checked against a launch manifest",
              file=sys.stderr)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
