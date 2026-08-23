"""Validate persisted E0 result/state contracts without evaluating metrics."""
import argparse
import json
import math
from pathlib import Path

import torch

from aggregation_schemes import state_dict_sha256


class E0ValidationError(RuntimeError):
    """A persisted E0 run violates its declared implementation contract."""


def _require(condition, message):
    if not condition:
        raise E0ValidationError(message)


def _single(paths, label):
    paths = list(paths)
    _require(len(paths) == 1, f"expected one {label}, found {len(paths)}")
    return paths[0]


def _a_keys(state):
    return sorted(key for key in state if ".lora_A.weight" in key)


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


def _validate_fedspan_round(result, payload, round_label):
    diagnostic = result["fedspan_diagnostics"][round_label]
    application = diagnostic["application"]
    _require(
        diagnostic["step_policy"]
        == result["method_contract"]["fedspan_step_policy"],
        f"{round_label}: step policy differs from method contract")
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


def validate_run_directory(run_directory):
    run_directory = Path(run_directory)
    result_path = _single(
        run_directory.glob("federated_*.json"), "federated result JSON")
    with result_path.open() as handle:
        result = json.load(handle)

    commit = result.get("commit")
    _require(
        isinstance(commit, str) and commit != "unknown"
        and not commit.endswith("-dirty"),
        "result does not have clean Git provenance")

    num_rounds = int(result["num_rounds"])
    state_paths = sorted(run_directory.glob("states_*.pt"))
    _require(
        len(state_paths) == num_rounds,
        f"expected {num_rounds} state files, found {len(state_paths)}")

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

        if result["lora_mode"] == "frozen-a":
            _validate_fixed_a(payload, round_label)
        if result.get("weight_by_canonical") == "normmaxmin":
            _validate_fedspan_round(result, payload, round_label)

    return {
        "result_path": str(result_path),
        "commit": commit,
        "rounds_validated": num_rounds,
        "lora_mode": result["lora_mode"],
        "weight_by_canonical": result.get("weight_by_canonical"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    args = parser.parse_args()
    report = validate_run_directory(args.run_directory)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

