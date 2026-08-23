"""CPU-only integration tests for FedSpan driver provenance and persistence."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import federated_forgetting as driver  # noqa: E402
from federated_forgetting import (  # noqa: E402
    _data_fingerprints,
    _frozen_run_configuration_sha256,
    dump_torch,
)


def frozen_args(**overrides):
    values = {
        "slices": ["nfcorpus", "fiqa"],
        "metrics": ["ndcg@10"],
        "num_rounds": 2,
        "local_epochs": 1,
        "batch_size": 8,
        "lr": 2e-5,
        "lora_rank": 4,
        "lora_mode": "frozen-a",
        "weighted": True,
        "weight_by": "normmaxmin",
        "qffl_q": 1.0,
        "afl_eta": 0.1,
        "loss_sample": 64,
        "max_steps_per_round": 10,
        "fedspan_step_norm": 0.1,
        "fedspan_active_abs_tol": 1e-12,
        "fedspan_active_rel_tol": 1e-8,
        "fedspan_mixture_norm_tol": 1e-6,
        "fedspan_max_abs_delta_weight": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_frozen_configuration_hash_is_stable_and_collision_sensitive():
    data_hashes = {"nfcorpus": "a", "fiqa": "b"}
    first = _frozen_run_configuration_sha256(
        frozen_args(), data_sha256=data_hashes)
    second = _frozen_run_configuration_sha256(
        frozen_args(), data_sha256=data_hashes)
    capped = _frozen_run_configuration_sha256(
        frozen_args(max_steps_per_round=11), data_sha256=data_hashes)
    different_step = _frozen_run_configuration_sha256(
        frozen_args(fedspan_step_norm=0.2), data_sha256=data_hashes)
    different_data = _frozen_run_configuration_sha256(
        frozen_args(), data_sha256={"nfcorpus": "changed", "fiqa": "b"})
    assert len(first) == 64
    assert first == second
    assert first != capped
    assert first != different_step
    assert first != different_data


def test_data_fingerprint_is_order_independent_and_content_sensitive():
    left = {
        "s": {
            "corpus": {"d2": {"text": "b"}, "d1": {"text": "a"}},
            "train_q": {"q1": "query"},
            "train_qrels": {"q1": {"d1": 1}},
            "eval_q": {},
            "eval_qrels": {},
        }
    }
    reordered = {
        "s": {
            "eval_qrels": {},
            "eval_q": {},
            "train_qrels": {"q1": {"d1": 1}},
            "train_q": {"q1": "query"},
            "corpus": {"d1": {"text": "a"}, "d2": {"text": "b"}},
        }
    }
    changed = {
        "s": {**left["s"],
              "corpus": {"d1": {"text": "changed"},
                         "d2": {"text": "b"}}}
    }
    assert _data_fingerprints(left) == _data_fingerprints(reordered)
    assert _data_fingerprints(left) != _data_fingerprints(changed)


def test_dump_torch_is_atomic_and_roundtrips(tmp_path):
    destination = tmp_path / "state.pt"
    payload = {"x": torch.arange(5), "label": "round"}
    dump_torch(payload, str(destination))
    assert destination.is_file()
    assert not (tmp_path / "state.pt.tmp").exists()
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    assert torch.equal(loaded["x"], payload["x"])
    assert loaded["label"] == "round"


def test_driver_normmaxmin_dispatch_persists_exact_round_record(
        monkeypatch, tmp_path):
    module = "encoder.layer0.query"
    akey = f"{module}.lora_A.weight"
    bkey = f"{module}.lora_B.weight"
    broadcast = {
        akey: torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]]),
        bkey: torch.zeros(3, 2),
    }
    clients = {
        "c0": {akey: broadcast[akey].clone(),
               bkey: torch.tensor([[1.0, 0.2], [0.1, 0.3], [0.2, 0.5]])},
        "c1": {akey: broadcast[akey].clone(),
               bkey: torch.tensor([[0.2, 0.8], [0.4, 0.1], [0.7, 0.2]])},
    }

    monkeypatch.setattr(driver, "get_git_commit", lambda: "abc123def456")
    monkeypatch.setattr(
        driver, "load_slice_with_train",
        lambda name, root: {
            "corpus": {}, "train_q": {}, "train_qrels": {},
            "eval_q": {}, "eval_qrels": {}, "split_fallback": False})
    monkeypatch.setattr(
        driver, "resolve_local", lambda name: ("fake-model", "", "", False))
    monkeypatch.setattr(
        driver, "new_model", lambda *args, **kwargs: (object(), {module: 2.0}))
    monkeypatch.setattr(
        driver, "get_adapter_state",
        lambda model: {key: value.clone() for key, value in broadcast.items()})
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

    monkeypatch.setattr(sys, "argv", [
        "federated_forgetting.py",
        "--slices", "c0", "c1",
        "--metrics", "ndcg@10",
        "--num_rounds", "1",
        "--weighted",
        "--weight_by", "normmaxmin",
        "--lora_mode", "frozen-a",
        "--fedspan_step_norm", "0.1",
        "--save_states",
        "--out", str(tmp_path),
    ])
    driver.main()

    result_paths = list(tmp_path.glob("federated_*.json"))
    state_paths = list(tmp_path.glob("states_*.pt"))
    assert len(result_paths) == 1
    assert len(state_paths) == 1
    with open(result_paths[0]) as handle:
        result = json.load(handle)
    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert result["lora_mode"] == "frozen-a"
    assert result["weight_by_canonical"] == "normmaxmin"
    assert diagnostic["status"] == "optimal"
    assert diagnostic["application"]["applied_step_norm"] == pytest.approx(
        0.1, abs=2e-6)
    saved = torch.load(state_paths[0], map_location="cpu", weights_only=True)
    assert torch.equal(saved["broadcast"][akey], broadcast[akey])
    assert torch.equal(saved["global"][akey], broadcast[akey])
