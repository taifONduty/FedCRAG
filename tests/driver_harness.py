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


def archived_slice_payload(name):
    return {
        "corpus": {"d0": {"text": name, "title": None}},
        "train_q": {"q_train": f"train {name}"},
        "train_qrels": {"q_train": {"d0": 1}},
        "eval_q": {"q_eval": f"eval {name}"},
        "eval_qrels": {"q_eval": {"d0": 1}},
        "split_fallback": False,
    }


def write_archived_data(root, slices=SLICES):
    """Create the tiny local-only BEIR archive used by validator tests."""
    root = Path(root)
    for name in slices:
        directory = root / name
        (directory / "qrels").mkdir(parents=True, exist_ok=True)
        payload = archived_slice_payload(name)
        with (directory / "corpus.jsonl").open("w") as handle:
            handle.write(json.dumps({
                "_id": "d0", "text": name, "title": None,
            }) + "\n")
        with (directory / "queries.jsonl").open("w") as handle:
            handle.write(json.dumps({
                "_id": "q_train", "text": payload["train_q"]["q_train"],
            }) + "\n")
            handle.write(json.dumps({
                "_id": "q_eval", "text": payload["eval_q"]["q_eval"],
            }) + "\n")
        for split, query_id in (("train", "q_train"),
                                ("test", "q_eval")):
            with (directory / "qrels" / f"{split}.tsv").open("w") as handle:
                handle.write("query-id\tcorpus-id\tscore\n")
                handle.write(f"{query_id}\td0\t1\n")
    return root


def broadcast_state(row_scale_c=1.0):
    """Shared A with ``A A^T = c^2 I`` and a zero B, as frozen-A init.

    ``row_scale_c`` defaults to 1 for the historical fixtures. Any value other
    than 1 is what makes the geometry scale ``sigma*c`` numerically distinct
    from the bare PEFT scale ``sigma``.
    """
    return {
        A_KEY: float(row_scale_c) * torch.eye(16),
        B_KEY: torch.zeros(3, 16),
    }


def client_states(row_scale_c=1.0):
    base = broadcast_state(row_scale_c)
    states = {}
    for name, block in CLIENT_B_BLOCKS.items():
        b_value = base[B_KEY].clone()
        b_value[:, :2] = torch.tensor(block)
        states[name] = {A_KEY: base[A_KEY].clone(), B_KEY: b_value}
    return states


def module_scales(lora_mode, row_scale_c=1.0, row_scale_mode="unit"):
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
        "row_scale_mode": row_scale_mode,
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
                  row_scale_c=1.0, row_scale_mode="unit", broadcast=None,
                  scale_override=None):
    clients = clients or client_states(row_scale_c)
    base = broadcast or broadcast_state(row_scale_c)

    monkeypatch.setattr(driver, "get_git_commit", lambda: commit)
    monkeypatch.setattr(
        driver, "load_slice_with_train",
        lambda name, root: archived_slice_payload(name))
    monkeypatch.setattr(
        driver, "resolve_local", lambda name: ("fake-model", "", "", False))
    monkeypatch.setattr(
        driver, "new_model",
        lambda *args, **kwargs: (
            object(),
            scale_override if scale_override is not None else module_scales(
                kwargs.get("lora_mode", "trainable-ab"),
                row_scale_c, row_scale_mode)))
    monkeypatch.setattr(
        driver, "get_adapter_state",
        lambda model: {key: value.clone() for key, value in base.items()})
    monkeypatch.setattr(
        driver, "_runtime_provenance",
        lambda commit, requested_model, model_path, model, scales,
               data_root, data_sha256: {
                   "test": True,
                   "data_root": str(Path(data_root).resolve()),
                   "data_sha256": data_sha256,
                   "module_scales": scales,
               })
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
               direction_policy="minnorm", extra=(), row_scale="unit"):
    argv = [
        "federated_forgetting.py",
        "--slices", *SLICES,
        "--metrics", "ndcg@10",
        "--num_rounds", str(num_rounds),
        "--lora_rank", "16",
        "--lora_mode", lora_mode,
        "--data_root", str(Path(out_directory) / "archived_data"),
        "--save_states",
        "--out", str(out_directory),
    ]
    if lora_mode == "frozen-a" and row_scale is not None:
        argv.extend(["--frozen_a_row_scale", row_scale])
    if arm != "uniform":
        argv.extend(["--weighted", "--weight_by", arm])
    if arm == "normmaxmin":
        argv.extend(["--fedspan_step_policy", "median-active",
                     "--fedspan_direction_policy", direction_policy,
                     "--fedspan_active_abs_tol", "1e-12",
                     "--fedspan_active_rel_tol", "1e-8",
                     "--fedspan_mixture_norm_tol", "1e-6"])
    argv.extend(extra)
    return argv


def run_driver(monkeypatch, out_directory, lora_mode, arm, num_rounds=1,
               direction_policy="minnorm", commit=CLEAN_COMMIT, extra=(),
               clients=None, row_scale_c=1.0, row_scale="unit",
               broadcast=None, scale_override=None):
    """Run one driver invocation; returns (result dict, result path)."""
    write_archived_data(Path(out_directory) / "archived_data")
    install_mocks(monkeypatch, commit=commit, clients=clients,
                  row_scale_c=row_scale_c, row_scale_mode=row_scale,
                  broadcast=broadcast, scale_override=scale_override)
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
