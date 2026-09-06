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
from aggregation_schemes import ModuleScales  # noqa: E402

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


def broadcast_state(row_scale_c=1.0):
    """Shared A with ``A A^T = c^2 I`` and a zero B, as frozen-A init.

    ``row_scale_c`` defaults to 1 for the historical fixtures. Any value other
    than 1 is what makes the geometry scale ``sigma*c`` numerically distinct
    from the bare PEFT scale ``sigma``.
    """
    return {
        A_KEY: float(row_scale_c) * torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        B_KEY: torch.zeros(3, 2),
    }


def client_states(row_scale_c=1.0):
    base = broadcast_state(row_scale_c)
    return {
        name: {A_KEY: base[A_KEY].clone(),
               B_KEY: torch.tensor(block)}
        for name, block in CLIENT_B_BLOCKS.items()
    }


def module_scales(lora_mode, row_scale_c=1.0):
    """What ``new_model`` hands back: geometry scales for frozen-A, sigma else.

    Mirrors ``configure_frozen_lora_a``: the mapping is ``sigma*c`` and the
    per-module records carry the bare ``sigma`` that materialized spaces need.
    """
    if lora_mode != "frozen-a":
        return MODULE_SCALE
    c = float(row_scale_c)
    scales = ModuleScales({MODULE: MODULE_SCALE * c})
    scales.records[MODULE] = {
        "peft_scale": MODULE_SCALE,
        "row_scale_mode": "constant",
        "row_scale_c": c,
        "measured_init_row_rms": c,
        "geometry_scale": MODULE_SCALE * c,
    }
    return scales


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


def install_mocks(monkeypatch, commit=CLEAN_COMMIT, clients=None,
                  row_scale_c=1.0, example_counts=None, step_counts=None,
                  losses=None):
    """``step_counts`` and ``losses`` map client name -> value; both default
    to the historical fixtures (one step each; no loss estimate) so existing
    tests keep their persisted records bit-for-bit."""
    clients = clients or client_states(row_scale_c)
    example_counts = example_counts or {}
    step_counts = step_counts or {}
    base = broadcast_state(row_scale_c)
    if losses is not None:
        monkeypatch.setattr(
            driver, "estimate_client_losses",
            lambda model, global_state, data_by_slice, slices, q_prefix,
                   d_prefix, sample, batch_size, rng_seed:
                [float(losses[name]) for name in slices])

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
        lambda *args, **kwargs: (
            object(),
            module_scales(kwargs.get("lora_mode", "trainable-ab"),
                          row_scale_c)))
    monkeypatch.setattr(
        driver, "get_adapter_state",
        lambda model: {key: value.clone() for key, value in base.items()})
    # Production persists the model's PEFT scale(s) in provenance; the q-FFL
    # recomputation reads the scalar trainable-A+B scale from there.
    monkeypatch.setattr(
        driver, "_runtime_provenance",
        lambda *args, **kwargs: {
            "test": True,
            "module_scales": (
                MODULE_SCALE if kwargs.get("module_scales") is None
                or isinstance(kwargs.get("module_scales"), float)
                else "frozen-a-mapping")})
    monkeypatch.setattr(
        driver, "client_train",
        lambda model, global_state, data, q_prefix, d_prefix, epochs,
               batch_size, lr, name, max_steps=0:
            ({key: value.clone() for key, value in clients[name].items()},
             example_counts.get(name, 10), step_counts.get(name, 1)))
    monkeypatch.setattr(
        driver, "eval_global",
        lambda model, state, data, slices, q_prefix, d_prefix, metrics,
               batch_size: {
                   name: {metric: 0.5 for metric in metrics}
                   for name in slices})
    monkeypatch.setattr(driver.torch.cuda, "empty_cache", lambda: None)


def build_argv(out_directory, lora_mode, arm, num_rounds=1,
               direction_policy="minnorm", extra=(), row_scale="unit"):
    argv = [
        "federated_forgetting.py",
        "--slices", *SLICES,
        "--metrics", "ndcg@10",
        "--num_rounds", str(num_rounds),
        "--lora_mode", lora_mode,
        "--save_states",
        "--out", str(out_directory),
    ]
    if lora_mode == "frozen-a" and row_scale is not None:
        argv.extend(["--frozen_a_row_scale", row_scale])
    if arm != "uniform":
        argv.extend(["--weighted", "--weight_by", arm])
    if arm == "normmaxmin":
        argv.extend(["--fedspan_step_policy", "median-active",
                     "--fedspan_direction_policy", direction_policy])
    argv.extend(extra)
    return argv


def run_driver(monkeypatch, out_directory, lora_mode, arm, num_rounds=1,
               direction_policy="minnorm", commit=CLEAN_COMMIT, extra=(),
               clients=None, row_scale_c=1.0, row_scale="unit",
               example_counts=None, step_counts=None, losses=None):
    """Run one driver invocation; returns (result dict, result path)."""
    install_mocks(monkeypatch, commit=commit, clients=clients,
                  row_scale_c=row_scale_c, example_counts=example_counts,
                  step_counts=step_counts, losses=losses)
    monkeypatch.setattr(sys, "argv", build_argv(
        out_directory, lora_mode, arm, num_rounds=num_rounds,
        direction_policy=direction_policy, extra=extra,
        row_scale=row_scale))
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
