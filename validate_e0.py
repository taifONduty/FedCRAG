"""Validate persisted E0 result/state contracts without evaluating metrics.

The aggregate recomputation below is written inline on purpose. Calling the
production aggregation functions would only re-run the code under test, which
cannot detect a persisted global that the production code never produced.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch

from aggregation_schemes import state_dict_sha256

_LORA_KEY = re.compile(r"(.*)\.lora_(A|B)\.weight$")

# Server aggregates are stored as float32, so an independent recomputation in
# float64 agrees only to float32 resolution. Tolerance is relative to the
# largest magnitude in the expected tensor, floored at 1 so near-zero blocks
# keep an absolute floor. Measured worst deviation/tolerance ratio across the
# five E0 aggregation paths on four-client rounds: 1.0e-02 (trainable-ab), and
# exactly 0 on every frozen-A path. A global rescaled by 1.5x scores 5.0e+04.
_AGGREGATE_RTOL = 1e-5

_SUPPORTED_ARMS = (None, "rawmaxmin", "normmaxmin", "examples",
                   "qffl", "afl", "fednova")
# Recorded scheme weights are compared against a recomputation from the
# scheme's own persisted inputs. FedNova's inputs are integers, so it is held
# to float precision; q-FFL and AFL depend on broadcast-point losses that the
# driver persists rounded to five decimals, so their tolerance is loosened to
# what that rounding can move (relative 1e-5 per loss, compounded over at most
# a few tens of rounds for AFL's chain).
_EXACT_COEFFICIENT_RTOL = 1e-9
_ROUNDED_LOSS_COEFFICIENT_RTOL = 1e-4
_DIRECTION_POLICIES = ("minnorm", "maxmin-lp", "exact", "fixed")

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


def _reference_delta_application(broadcast_state, client_states, v):
    """w^{t+1} = w^t + sum_k v_k (w_k - w^t) on complete states, in float64.

    The delta-space arms (q-FFL, FedNova) produce coefficients that need not
    sum to one, so the simplex average is the wrong reference for them.
    """
    out = {}
    for key in broadcast_state:
        base = broadcast_state[key].detach().cpu().double()
        accumulator = base.clone()
        for coefficient, state in zip(v, client_states):
            if coefficient == 0.0:
                continue
            accumulator += coefficient * (
                state[key].detach().cpu().double() - base)
        out[key] = accumulator.float()
    return out


def _recorded_scheme_weights(result, arm, round_label, num_clients):
    """The weights the run says it applied, at full precision."""
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
    weights = [float(value) for value in weights]
    _require(all(_finite(value) for value in weights),
             f"{round_label}: recorded {arm} weights contain a nonfinite value")
    return weights


def _require_coefficients_agree(recorded, recomputed, rtol, arm, round_label):
    """Relative to the LARGEST recomputed coefficient, not to 1: delta-space
    coefficients can be tiny (q-FFL with L = 1/lr gives ~1e-6), and a floor at
    1 would let a forged input pass unnoticed."""
    scale = max(abs(value) for value in recomputed)
    _require(scale > 0, f"{round_label}: recomputed {arm} weights are all zero")
    for index, (have, want) in enumerate(zip(recorded, recomputed)):
        tolerance = rtol * scale
        _require(
            abs(have - want) <= tolerance,
            f"{round_label}: recorded {arm} weight for client {index} is "
            f"{have:.10g} but the recomputation from the persisted inputs "
            f"gives {want:.10g} (tolerance {tolerance:.3g})")


def _client_stats(result, round_label, field):
    stats = (result.get("clients") or {}).get(round_label)
    _require(isinstance(stats, dict),
             f"{round_label}: round has no per-client stats")
    values = []
    for name in result["slices"]:
        value = (stats.get(name) or {}).get(field)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and value >= 0 and value == value,
                 f"{round_label}: no persisted {field} for '{name}'")
        values.append(float(value))
    return values


def _round_losses(result, round_label):
    losses = (result.get("client_losses") or {}).get(round_label)
    _require(isinstance(losses, dict),
             f"{round_label}: run records no broadcast-point client losses")
    values = []
    for name in result["slices"]:
        value = losses.get(name)
        _require(_finite(value),
                 f"{round_label}: no finite persisted loss for '{name}'")
        values.append(float(value))
    return values


def _reference_fednova_coefficients(result, round_label):
    """FedNova (2007.07481): p_k = n_k / n, tau_eff = sum p_k tau_k,
    v_k = tau_eff p_k / tau_k; clients with tau_k = 0 are masked out."""
    counts = _client_stats(result, round_label, "num_examples")
    steps = _client_stats(result, round_label, "num_steps")
    active = [tau > 0 for tau in steps]
    _require(any(active),
             f"{round_label}: FedNova round in which no client trained")
    mass = sum(n for n, on in zip(counts, active) if on)
    _require(mass > 0, f"{round_label}: FedNova active example mass is zero")
    p = [(n / mass if on else 0.0) for n, on in zip(counts, active)]
    tau_eff = sum(pk * tau for pk, tau, on in zip(p, steps, active) if on)
    return [(tau_eff * pk / tau if on else 0.0)
            for pk, tau, on in zip(p, steps, active)]


def _reference_afl_coefficients(result, round_label):
    """AFL (1902.00146) mixture ascent replayed from the uniform start:
    lambda_k <- lambda_k exp(eta (F_k - max F)), renormalised, one step per
    round, using every persisted loss vector up to this round."""
    eta = (result.get("args") or {}).get("afl_eta")
    _require(_finite(eta) and float(eta) > 0,
             f"{round_label}: AFL run records no positive afl_eta")
    eta = float(eta)
    num_clients = len(result["slices"])
    current = int(round_label.split("_")[1])
    lam = [1.0 / num_clients] * num_clients
    for index in range(1, current + 1):
        losses = _round_losses(result, f"round_{index}")
        shift = max(losses)
        new = [value * math.exp(eta * (loss - shift))
               for value, loss in zip(lam, losses)]
        total = sum(new)
        _require(_finite(total) and total > 0,
                 f"round_{index}: AFL weight mass is nonfinite or nonpositive")
        lam = [value / total for value in new]
    return lam


def _trainable_update_sq_norms(payload, result, round_label):
    """||sigma (B_k A_k - B_g A_g)||_F^2 per client from the persisted states."""
    scale = (result.get("provenance") or {}).get("module_scales")
    _require(_finite(scale) and float(scale) > 0,
             f"{round_label}: q-FFL recomputation needs the run's scalar "
             "PEFT scale in provenance.module_scales")
    sigma = float(scale)
    broadcast = payload["broadcast"]
    modules = _lora_modules(broadcast, "broadcast")
    base = {name: broadcast[factors["B"]].detach().cpu().double()
            @ broadcast[factors["A"]].detach().cpu().double()
            for name, factors in modules.items()}
    out = []
    for name in result["slices"]:
        state = payload["clients"][name]
        total = 0.0
        for module, factors in modules.items():
            product = (state[factors["B"]].detach().cpu().double()
                       @ state[factors["A"]].detach().cpu().double())
            total += float(torch.sum((sigma * (product - base[module])) ** 2))
        out.append(total)
    return out


def _reference_qffl_coefficients(result, payload, round_label):
    """q-FedAvg (1905.10497) delta-space weights, full participation:
    h_k = q F_k^{q-1} L^2 ||w_k - w^t||^2 + L F_k^q, v_k = L F_k^q / sum_j h_j,
    with L = 1/lr and F_k the persisted broadcast-point losses."""
    recorded_args = result.get("args") or {}
    lr = recorded_args.get("lr")
    q = recorded_args.get("qffl_q")
    _require(_finite(lr) and float(lr) > 0,
             f"{round_label}: q-FFL run records no positive learning rate")
    _require(_finite(q) and float(q) >= 0,
             f"{round_label}: q-FFL run records no valid qffl_q")
    L = 1.0 / float(lr)
    q = float(q)
    losses = [max(value, 1e-8) for value in _round_losses(result, round_label)]
    d2 = _trainable_update_sq_norms(payload, result, round_label)
    fq = [1.0 if q == 0 else f ** q for f in losses]
    fqm1 = [0.0 if q == 0 else f ** (q - 1.0) for f in losses]
    h = [q * a * (L ** 2) * b + L * c for a, b, c in zip(fqm1, d2, fq)]
    total = sum(h)
    _require(_finite(total) and total > 0,
             f"{round_label}: q-FFL h-sum is nonfinite or nonpositive")
    return [L * c / total for c in fq]


def _simplex_from_recorded(weights, label):
    weights = [float(value) for value in weights]
    _require(all(_finite(value) for value in weights),
             f"{label}: recorded weights contain a nonfinite value")
    _require(min(weights) >= 0.0,
             f"{label}: recorded simplex weights contain a negative value")
    total = sum(weights)
    _require(total > 0.0, f"{label}: recorded simplex weights sum to {total}")
    return [value / total for value in weights]


def _round_coefficients(result, round_label, num_clients, payload=None):
    """(kind, coefficients) the record says were applied for this round.

    ``kind`` names the application the reference must reproduce: ``fedavg``
    (simplex average of complete states), ``delta`` (delta-space coefficients
    on complete states) or ``frozen-b-delta`` (coefficients on raw-B deltas
    with A copied). ``payload`` is needed only by arms whose coefficients
    depend on the persisted states themselves (q-FFL's update norms).
    """
    arm = result.get("weight_by_canonical")
    frozen = result["lora_mode"] == "frozen-a"
    kind = "frozen-b-delta" if frozen else "fedavg"
    _require(arm in _SUPPORTED_ARMS,
             f"{round_label}: no recomputation reference for arm '{arm}'")

    if arm == "fednova":
        recorded = _recorded_scheme_weights(result, arm, round_label,
                                            num_clients)
        _require_coefficients_agree(
            recorded, _reference_fednova_coefficients(result, round_label),
            _EXACT_COEFFICIENT_RTOL, arm, round_label)
        return ("frozen-b-delta" if frozen else "delta"), recorded

    if arm == "afl":
        recorded = _recorded_scheme_weights(result, arm, round_label,
                                            num_clients)
        _require_coefficients_agree(
            recorded, _reference_afl_coefficients(result, round_label),
            _ROUNDED_LOSS_COEFFICIENT_RTOL, arm, round_label)
        return kind, _simplex_from_recorded(recorded, round_label)

    if arm == "qffl":
        _require(not frozen,
                 f"{round_label}: the q-FFL recomputation reference covers the "
                 "trainable coordinate only")
        _require(payload is not None,
                 f"{round_label}: q-FFL recomputation needs the round states")
        recorded = _recorded_scheme_weights(result, arm, round_label,
                                            num_clients)
        _require_coefficients_agree(
            recorded,
            _reference_qffl_coefficients(result, payload, round_label),
            _ROUNDED_LOSS_COEFFICIENT_RTOL, arm, round_label)
        return "delta", recorded

    if arm is None:
        _require(not result.get("weighted"),
                 f"{round_label}: weighted run records no weighting arm")
        return kind, [1.0 / num_clients] * num_clients

    if arm == "examples":
        # n_k weighting: w_k = num_examples_k / sum_j num_examples_j, derived
        # from the PERSISTED per-round client stats so the recomputation
        # checks the coefficients the run says it earned, not an assumption.
        # Legal in either coordinate: FedAvg on factor states (trainable-ab)
        # or the same simplex weights on raw-B deltas (frozen-a).
        stats = (result.get("clients") or {}).get(round_label)
        _require(isinstance(stats, dict),
                 f"{round_label}: examples round has no per-client stats")
        counts = []
        for name in result["slices"]:
            count = (stats.get(name) or {}).get("num_examples")
            _require(isinstance(count, (int, float)) and count >= 0,
                     f"{round_label}: no persisted num_examples for '{name}'")
            counts.append(float(count))
        exponent = (result.get("args") or {}).get("weight_pow")
        exponent = 1.0 if exponent is None else float(exponent)
        _require(exponent >= 0 and exponent == exponent,
                 f"{round_label}: invalid weight_pow {exponent!r}")
        counts = [value ** exponent for value in counts]
        total = sum(counts)
        _require(total > 0, f"{round_label}: example counts sum to zero")
        return kind, [value / total for value in counts]

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
    kind, coefficients = _round_coefficients(result, round_label, len(slices),
                                             payload=payload)
    if kind == "fedavg":
        expected = _reference_weighted_average(client_states, coefficients)
    elif kind == "delta":
        expected = _reference_delta_application(
            payload["broadcast"], client_states, coefficients)
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
    policy = diagnostic.get("direction_policy")
    solver = diagnostic.get("min_norm_solver") or {}
    exact = diagnostic.get("exact_solver")

    if policy == "exact":
        # The exact arm is NOT gated on the iterative reference solver. E3's
        # federations are built from near-duplicate clients, so their Gram is
        # singular by construction and away-step Frank-Wolfe stalls on them by
        # design (measured Wolfe gaps 1.1e-06 and 4.8e-07 on a 3-clone
        # federation). Gating the exact arm on that solver would have rejected
        # every E3 clone run while the answer it applied carried a proof.
        # What is required instead is that proof: the face enumeration's own
        # solver-independent Wolfe certificate.
        _require(isinstance(exact, dict),
                 f"{round_label}: the exact arm recorded no exact_solver "
                 "diagnostics, so the applied direction carries no proof")
        _require(exact.get("algorithm") == "face-enumeration-least-norm-argmin/v1",
                 f"{round_label}: unknown exact solver algorithm "
                 f"'{exact.get('algorithm')}'")
        _require(diagnostic.get("min_norm_value_source")
                 == "exact-face-enumeration",
                 f"{round_label}: the exact arm did not source its attainable "
                 f"optimum from the face enumeration "
                 f"(source '{diagnostic.get('min_norm_value_source')}')")
        certificate = exact.get("wolfe_certificate")
        _require(_finite(certificate) and abs(float(certificate)) <= 1e-8,
                 f"{round_label}: the exact solver's optimality certificate "
                 f"is {certificate!r}, not zero: the applied direction is not "
                 "certified optimal")
        return

    if policy == "fixed":
        # A fixed arm solves nothing, so neither solver's convergence is a
        # property of the round. What must be on the record is the declared
        # weight vector and the MEASURED distance from optimal, which is what
        # makes the arm interpretable as a control rather than an attempt.
        weights = diagnostic.get("fixed_weights")
        _require(isinstance(weights, (list, tuple)) and len(weights) > 0
                 and all(_finite(value) for value in weights),
                 f"{round_label}: the fixed arm recorded no finite declared "
                 f"weight vector (got {weights!r})")
        _require(_finite(diagnostic.get("wolfe_certificate")),
                 f"{round_label}: the fixed arm recorded no measured "
                 "optimality gap")
        return

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

    # Independent of everything above: rebuild the step from the state files
    # rather than comparing two numbers the run wrote about itself.
    _validate_median_active_step(result, diagnostic, round_label)
    recomputed_norm = _validate_recomputed_fedspan_step(
        result, payload, diagnostic, round_label)
    return {"fallback": fallback, "applied_step_norm": recomputed_norm}


# ------------------------------------------------------- launch provenance


def _argv_value(argv, flag, required=False):
    if flag not in argv:
        _require(not required, f"manifest command omits {flag}")
        return None
    index = argv.index(flag)
    _require(index + 1 < len(argv), f"manifest command truncates {flag}")
    return argv[index + 1]


def _argv_values(argv, flag):
    if flag not in argv:
        return None
    values = []
    for token in argv[argv.index(flag) + 1:]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def _validate_manifest_row(result, row, run_id):
    argv = list(row["argv"])
    _require(row["run_id"] == run_id,
             f"manifest row is for '{row['run_id']}', not '{run_id}'")
    _require(
        result["commit"] == row["commit"],
        f"run commit {result['commit']} differs from the manifest commit "
        f"{row['commit']}")

    # Two of the grid's axes live only in these flags: e0-frozen-a-uniform-full
    # and e0-frozen-a-unitscale-uniform-full differ in the row scale alone, and
    # the direction policy is a declared part of the normmaxmin method.
    launched = {
        "seed": int(_argv_value(argv, "--seed", required=True)),
        "num_rounds": int(_argv_value(argv, "--num_rounds", required=True)),
        "lora_mode": _argv_value(argv, "--lora_mode", required=True),
        "max_steps_per_round": int(
            _argv_value(argv, "--max_steps_per_round", required=True)),
        "slices": _argv_values(argv, "--slices"),
        "weight_by": (_argv_value(argv, "--weight_by")
                      if "--weighted" in argv else None),
        "frozen_a_row_scale": _argv_value(argv, "--frozen_a_row_scale"),
        "fedspan_direction_policy": _argv_value(
            argv, "--fedspan_direction_policy"),
    }
    _require(launched["slices"], "manifest command omits --slices")
    arguments = result.get("args")
    _require(isinstance(arguments, dict)
             and "max_steps_per_round" in arguments,
             "result records no argument namespace to compare")
    contract = result.get("method_contract") or {}
    recorded = {
        "seed": int(result["seed"]),
        "num_rounds": int(result["num_rounds"]),
        "lora_mode": result["lora_mode"],
        "max_steps_per_round": int(arguments["max_steps_per_round"]),
        "slices": list(result["slices"]),
        "weight_by": result["weight_by"],
        "frozen_a_row_scale": contract.get("frozen_a_row_scale"),
        "fedspan_direction_policy": contract.get("fedspan_direction_policy"),
    }
    for field, expected in sorted(launched.items()):
        _require(
            recorded[field] == expected,
            f"manifest row '{run_id}' launched {field}={expected!r} but the "
            f"result records {recorded[field]!r}")
    _require(row["coordinate"] == recorded["lora_mode"],
             f"manifest row '{run_id}' coordinate differs from --lora_mode")
    expected_arm = None if row["arm"] == "uniform" else row["arm"]
    _require(recorded["weight_by"] == expected_arm,
             f"manifest row '{run_id}' arm '{row['arm']}' differs from the "
             f"recorded weighting {recorded['weight_by']!r}")
    return launched


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
    resources = None
    if manifest_row is not None:
        launched = _validate_manifest_row(
            result, manifest_row, run_directory.name)
        resources = _validate_resource_record(run_directory, result)

    worst_ratio = 0.0
    worst_round = None
    fallback_rounds = 0
    applied_rounds = 0
    first_broadcast = None
    final_global = None
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
        _validate_scheme_round(result, round_label)
        _validate_client_delta_norms(result, round_label, result["slices"])
        if result["lora_mode"] == "frozen-a":
            _validate_fixed_a(payload, round_label)
        recomputation = _validate_recomputed_global(
            result, payload, round_label)
        if result.get("weight_by_canonical") == "normmaxmin":
            step = _validate_fedspan_round(result, payload, round_label)
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
