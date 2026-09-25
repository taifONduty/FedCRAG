"""Independent checks of a continual run directory: the manifest, the round chain, the
memory budget, the split boundaries and the acquisition references."""
import argparse
import json
from pathlib import Path

import torch

import acceptance
import experiences
from aggregation_schemes import state_dict_sha256
from continual_driver import REPLAY_ARMS, _digest
from federated_forgetting import _sha256_file, fedavg


class ContinualValidationError(Exception):
    pass


def _require(condition, message):
    if not condition:
        raise ContinualValidationError(message)


def _single(paths, what):
    paths = list(paths)
    _require(len(paths) == 1, f"expected one {what}, found {len(paths)}")
    return paths[0]


def _check_memory(record, client, index, manifest_client, budget, position):
    memory = record["memory"][client]
    cells = manifest_client["experiences"]
    held_out = {q for cell in cells.values() for q in cell["guard"] + cell["test"]}
    test_ids = {q for cell in cells.values() for q in cell["test"]}
    replay, reserved = set(memory["replay"]), set(memory["reserved"])
    _require(not (replay & held_out) and not (reserved & test_ids),
             f"round {index}: client {client} retains a test or guard query")
    earlier = [cells[str(manifest_client["order"][s])] for s in range(position)]
    _require(replay <= {q for cell in earlier for q in cell["train"]},
             f"round {index}: client {client} replays a query outside its earlier experiences")
    _require(reserved <= {q for cell in earlier for q in cell["guard"]},
             f"round {index}: client {client} guards a query outside its earlier experiences")
    _require(memory["budget"] == budget and memory["used"] <= budget
             and memory["used"] == len(replay) + len(reserved),
             f"round {index}: client {client}'s memory record breaks its budget")
    current = cells[str(manifest_client["order"][position])]["train"]
    _require(record["training_ids_sha256"][client] == _digest(sorted(set(current) | replay)),
             f"round {index}: client {client} trained on queries other than its experience "
             "and its replay")


def validate_run(run_directory, manifest_path=None):
    run_directory = Path(run_directory)
    result_path = _single(run_directory.glob("continual_*.json"), "continual result JSON")
    with result_path.open() as handle:
        result = json.load(handle)
    commit = result.get("commit")
    _require(isinstance(commit, str) and commit != "unknown" and not commit.endswith("-dirty")
             and not commit.endswith("-unknown-worktree"),
             "result does not have clean Git provenance")
    manifest_path = Path(manifest_path or result["manifest_path"])
    _require(_sha256_file(manifest_path) == result["manifest_sha256"],
             "manifest digest differs from the record")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    experiences.verify_manifest(manifest)

    arm, clients = result["arm"], result["clients"]
    T, R = manifest["experiences_per_client"], result["rounds_per_experience"]
    _require(result["order"] == {c: manifest["clients"][c]["order"] for c in clients},
             "experience order differs from the manifest")
    rounds = result["rounds"]
    expected = 0 if arm == "frozen" else T * R
    _require(len(rounds) == expected, f"expected {expected} rounds, found {len(rounds)}")
    local = arm == "local"
    budget = int(result["args"]["memory_budget"]) if arm in REPLAY_ARMS else 0
    initial = result["initial_state_sha256"]
    previous = {c: initial for c in clients} if local else initial
    reference_hashes = {}
    for index, record in enumerate(rounds):
        position, r = divmod(index, R)
        _require((record["position"], record["round"]) == (position, r),
                 f"round {index}: out of sequence")
        _require(record["experience"] == {c: manifest["clients"][c]["order"][position]
                                          for c in clients},
                 f"round {index}: experience differs from the manifest order")
        payload = torch.load(run_directory / record["state_file"], weights_only=True)
        for c in clients:
            _require(record["hashes"]["clients"][c] == state_dict_sha256(payload["clients"][c]),
                     f"round {index}: client {c}'s recorded hash differs from its state")
            _check_memory(record, c, index, manifest["clients"][c], budget, position)
        if local:
            for c in clients:
                _require(state_dict_sha256(payload["clients_before"][c]) == previous[c],
                         f"round {index}: client {c}'s start state does not connect to its "
                         "previous state")
                previous[c] = state_dict_sha256(payload["clients"][c])
        else:
            _require(state_dict_sha256(payload["broadcast"]) == previous,
                     f"round {index}: the broadcast does not connect to the previous "
                     "global state")
            recomputed = fedavg([payload["clients"][c] for c in clients])
            checked = arm == "fedavg-replay-accept" and position > 0
            _require(("acceptance" in record) == checked,
                     f"round {index}: an acceptance record where none belongs, or none where "
                     "one does")
            if checked:
                choice = record["acceptance"]
                _require(acceptance.rule_holds(choice["step"], choice["tried"]),
                         f"round {index}: the kept step does not follow the acceptance rule")
                _require(all(len(record["memory"][c]["reserved"]) == acceptance.GUARD_SLOTS
                             for c in clients),
                         f"round {index}: a client does not hold {acceptance.GUARD_SLOTS} "
                         "guard queries")
                recomputed = acceptance.step_state(payload["broadcast"], recomputed,
                                                   choice["step"])
            for key, tensor in recomputed.items():
                _require(torch.allclose(tensor, payload["global"][key].float(),
                                        rtol=1e-6, atol=1e-7),
                         f"round {index}: the persisted global is not the kept step toward the "
                         "uniform average of the persisted client states")
            previous = state_dict_sha256(payload["global"])
        if r == R - 1:
            for c in clients:
                reference_hashes[(c, str(manifest["clients"][c]["order"][position]))] = (
                    previous[c] if local else previous)
    for c in clients:
        for e, reference in result["references"][c].items():
            expected_hash = initial if arm == "frozen" else reference_hashes.get((c, e))
            _require(reference["sha256"] == expected_hash,
                     f"client {c} experience {e}: the acquisition reference hash does not "
                     "match the persisted state")
    for t in range(T):
        for c in clients:
            _require(set(result["matrix"][str(t)][c])
                     == {str(manifest["clients"][c]["order"][s]) for s in range(t + 1)},
                     f"position {t}: client {c}'s evaluation matrix is incomplete")
    return {"result_path": str(result_path), "arm": arm, "rounds_validated": len(rounds),
            "clients": clients, "experiences": T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_directory")
    ap.add_argument("--manifest")
    args = ap.parse_args()
    print(json.dumps(validate_run(args.run_directory, args.manifest), indent=2))


if __name__ == "__main__":
    main()
