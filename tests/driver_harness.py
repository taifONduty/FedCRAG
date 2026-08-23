"""Run the real driver end to end on CPU with data, model and eval mocked.

Everything the scientific contract depends on — argument legality, the
aggregation dispatch, the diagnostics, the persisted states and hashes — is
the production code path. Only the parts that need a GPU and a corpus are
replaced.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import federated_forgetting as driver  # noqa: E402

MODULE = "encoder.layer0.query"
A_KEY = f"{MODULE}.lora_A.weight"
B_KEY = f"{MODULE}.lora_B.weight"
MODULE_SCALE = 2.0
SLICES = ("c0", "c1", "c2")
CLEAN_COMMIT = "abc123def456"

# Three client directions whose cosine Gram is well conditioned and whose
# max-min simplex solution is far from uniform, so an aggregation that quietly
# reverts to uniform weights changes the persisted global.
CLIENT_B_BLOCKS = {
    "c0": [[1.0, 0.2], [0.1, 0.3], [0.2, 0.5]],
    "c1": [[0.2, 0.8], [0.4, 0.1], [0.7, 0.2]],
    "c2": [[-0.6, 0.3], [0.9, -0.2], [0.1, 0.8]],
}


def broadcast_state():
    """Shared row-orthonormal A and a zero B, as frozen-A initialization."""
    return {
        A_KEY: torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        B_KEY: torch.zeros(3, 2),
    }


def client_states():
    base = broadcast_state()
    return {
        name: {A_KEY: base[A_KEY].clone(),
               B_KEY: torch.tensor(block)}
        for name, block in CLIENT_B_BLOCKS.items()
    }


def effective_updates(states, broadcast):
    """Dense sigma * (B_k A_k - B_g A_g) per client, flattened to float64."""
    vectors = []
    for state in states:
        update = (state[B_KEY].double() @ state[A_KEY].double()
                  - broadcast[B_KEY].double() @ broadcast[A_KEY].double())
        vectors.append(MODULE_SCALE * update.reshape(-1))
    return torch.stack(vectors).numpy()


def cosine_gram(states, broadcast):
    stacked = effective_updates(states, broadcast)
    norms = np.linalg.norm(stacked, axis=1)
    return (stacked @ stacked.T) / np.outer(norms, norms), norms


def install_mocks(monkeypatch, commit=CLEAN_COMMIT, clients=None):
    clients = clients or client_states()
    base = broadcast_state()

    monkeypatch.setattr(driver, "get_git_commit", lambda: commit)
    monkeypatch.setattr(
        driver, "load_slice_with_train",
        lambda name, root: {
            "corpus": {"d0": {"text": name}}, "train_q": {}, "train_qrels": {},
            "eval_q": {}, "eval_qrels": {}, "split_fallback": False})
    monkeypatch.setattr(
        driver, "resolve_local", lambda name: ("fake-model", "", "", False))
    monkeypatch.setattr(
        driver, "new_model",
        lambda *args, **kwargs: (object(), {MODULE: MODULE_SCALE}))
    monkeypatch.setattr(
        driver, "get_adapter_state",
        lambda model: {key: value.clone() for key, value in base.items()})
    monkeypatch.setattr(
        driver, "_runtime_provenance", lambda *args, **kwargs: {"test": True})
    monkeypatch.setattr(
        driver, "client_train",
        lambda model, global_state, data, q_prefix, d_prefix, epochs,
               batch_size, lr, name, max_steps=0:
            ({key: value.clone() for key, value in clients[name].items()},
             10, 1))
    monkeypatch.setattr(
        driver, "eval_global",
        lambda model, state, data, slices, q_prefix, d_prefix, metrics,
               batch_size: {
                   name: {metric: 0.5 for metric in metrics}
                   for name in slices})
    monkeypatch.setattr(driver.torch.cuda, "empty_cache", lambda: None)


def build_argv(out_directory, lora_mode, arm, num_rounds=1,
               direction_policy="minnorm", extra=()):
    argv = [
        "federated_forgetting.py",
        "--slices", *SLICES,
        "--metrics", "ndcg@10",
        "--num_rounds", str(num_rounds),
        "--lora_mode", lora_mode,
        "--save_states",
        "--out", str(out_directory),
    ]
    if arm != "uniform":
        argv.extend(["--weighted", "--weight_by", arm])
    if arm == "normmaxmin":
        argv.extend(["--fedspan_step_policy", "median-active",
                     "--fedspan_direction_policy", direction_policy])
    argv.extend(extra)
    return argv


def run_driver(monkeypatch, out_directory, lora_mode, arm, num_rounds=1,
               direction_policy="minnorm", commit=CLEAN_COMMIT, extra=(),
               clients=None):
    """Run one driver invocation; returns (result dict, result path)."""
    install_mocks(monkeypatch, commit=commit, clients=clients)
    monkeypatch.setattr(sys, "argv", build_argv(
        out_directory, lora_mode, arm, num_rounds=num_rounds,
        direction_policy=direction_policy, extra=extra))
    driver.main()

    paths = list(Path(out_directory).glob("federated_*.json"))
    if len(paths) != 1:
        raise AssertionError(f"expected one result JSON, found {len(paths)}")
    with paths[0].open() as handle:
        return json.load(handle), paths[0]


def load_round_states(out_directory, round_number=1):
    paths = list(Path(out_directory).glob(f"states_*_round{round_number}.pt"))
    if len(paths) != 1:
        raise AssertionError(f"expected one state file, found {len(paths)}")
    return torch.load(paths[0], map_location="cpu", weights_only=True), paths[0]


def rewrite_states(path, payload):
    torch.save(payload, path)


def rewrite_result(path, result):
    with Path(path).open("w") as handle:
        json.dump(result, handle)
