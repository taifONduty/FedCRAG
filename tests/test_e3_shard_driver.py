"""The single driver site where sharding is applied, exercised end to end."""
import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import driver_harness  # noqa: E402
import federated_forgetting as driver  # noqa: E402
from e3_fixtures import nfcorpus_shaped_payload  # noqa: E402

BATCH_SIZE = 4
PARENT_CAP = 6


def payload_for(name):
    if name == "clone":
        return nfcorpus_shaped_payload(seed=11, n_train=60, n_eval=180,
                                       n_docs=64)
    return nfcorpus_shaped_payload(seed=12, n_train=40, n_eval=60, n_docs=48)


def install(monkeypatch, calls=None):
    driver_harness.install_mocks(monkeypatch)
    monkeypatch.setattr(driver, "load_slice_with_train",
                        lambda name, root: payload_for(name))

    base = driver_harness.broadcast_state()

    def fake_client_train(model, global_state, data, q_prefix, d_prefix,
                          epochs, batch_size, lr, name, max_steps=0):
        if calls is not None:
            calls.append({"name": name, "max_steps": max_steps,
                          "n_eval": len(data["eval_q"])})
        seed = abs(hash(name)) % 97 + 1
        state = {driver_harness.A_KEY: base[driver_harness.A_KEY].clone(),
                 driver_harness.B_KEY: torch.full((3, 2), seed / 100.0)}
        return state, 10, 1

    monkeypatch.setattr(driver, "client_train", fake_client_train)


def build_argv(out_directory, extra=()):
    return ["federated_forgetting.py",
            "--slices", "clone", "single",
            "--metrics", "ndcg@10",
            "--num_rounds", "1",
            "--batch_size", str(BATCH_SIZE),
            "--max_steps_per_round", str(PARENT_CAP),
            "--out", str(out_directory), *extra]


def run(monkeypatch, out_directory, extra=(), calls=None):
    install(monkeypatch, calls=calls)
    monkeypatch.setattr(sys, "argv", build_argv(out_directory, extra))
    driver.main()
    paths = list(Path(out_directory).glob("federated_*.json"))
    assert len(paths) == 1
    with paths[0].open() as handle:
        return json.load(handle), paths[0]


SHARD_ARGS = ("--shard_spec", "clone:3", "--shard_seed", "42",
              "--conserve_shard_steps")


def test_expands_slices_and_conserves_steps(tmp_path, monkeypatch):
    calls = []
    result, _ = run(monkeypatch, tmp_path, extra=SHARD_ARGS, calls=calls)
    assert result["slices"] == ["clone-s0", "clone-s1", "clone-s2", "single"]
    assert [call["name"] for call in calls] == result["slices"]
    assert [call["max_steps"] for call in calls] == [2, 2, 2, PARENT_CAP]
    assert sum(call["max_steps"] for call in calls[:3]) == PARENT_CAP
    assert set(result["clients"]["round_1"]) == set(result["slices"])


def test_unsharded_run_passes_the_parent_cap(tmp_path, monkeypatch):
    calls = []
    result, _ = run(monkeypatch, tmp_path, calls=calls)
    assert result["slices"] == ["clone", "single"]
    assert [call["max_steps"] for call in calls] == [PARENT_CAP, PARENT_CAP]
    assert "shard_manifest" not in result
    assert not list(Path(tmp_path).glob("shard_manifest_*.json"))


def test_manifest_embedded_and_written(tmp_path, monkeypatch):
    result, _ = run(monkeypatch, tmp_path, extra=SHARD_ARGS)
    manifest = result["shard_manifest"]
    assert manifest["expanded_slices"] == result["slices"]
    assert manifest["shard_seed"] == 42
    assert manifest["batch_size"] == BATCH_SIZE
    assert manifest["parent_max_steps_per_round"] == PARENT_CAP
    written = Path(result["shard_manifest_path"])
    assert written.exists()
    assert json.loads(written.read_text()) == manifest


def test_manifest_written_before_the_first_gradient_step(tmp_path,
                                                         monkeypatch):
    seen = {}
    install(monkeypatch)
    real = driver.client_train

    def watching(*args, **kwargs):
        seen.setdefault(
            "manifests", list(Path(tmp_path).glob("shard_manifest_*.json")))
        return real(*args, **kwargs)

    monkeypatch.setattr(driver, "client_train", watching)
    monkeypatch.setattr(sys, "argv", build_argv(tmp_path, SHARD_ARGS))
    driver.main()
    assert len(seen["manifests"]) == 1


def test_data_fingerprints_cover_the_shards(tmp_path, monkeypatch):
    result, _ = run(monkeypatch, tmp_path, extra=SHARD_ARGS)
    recorded = {shard["client"]: shard["client_data_sha256"]
                for entry in result["shard_manifest"]["parents"]
                for shard in entry["shards"]}
    assert set(recorded) == set(result["slices"])
    assert len({recorded[name] for name in result["slices"]
                if name.startswith("clone")}) == 3


def test_shard_eval_queries_are_disjoint(tmp_path, monkeypatch):
    calls = []
    run(monkeypatch, tmp_path, extra=SHARD_ARGS, calls=calls)
    clone_evals = [call["n_eval"] for call in calls if "clone" in call["name"]]
    assert sum(clone_evals) == 180
    assert max(clone_evals) - min(clone_evals) <= 1


# --- configuration identity ------------------------------------------------

def _frozen_args(**overrides):
    fields = {
        "slices": ["a", "b"], "metrics": ["ndcg@10"], "num_rounds": 1,
        "local_epochs": 1, "batch_size": 32, "eval_batch_size": 128,
        "lr": 2e-5, "lora_rank": 16, "lora_mode": "frozen-a",
        "weighted": False, "weight_by": "examples", "qffl_q": 1.0,
        "afl_eta": 0.1, "loss_sample": 2048, "max_steps_per_round": 0,
        "fedspan_step_policy": None, "fedspan_step_norm": None,
        "fedspan_active_abs_tol": 1e-12, "fedspan_active_rel_tol": 1e-8,
        "fedspan_mixture_norm_tol": 1e-6,
        "fedspan_max_abs_delta_weight": None,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def test_config_hash_unchanged_when_sharding_is_absent():
    legacy = driver._frozen_run_configuration_sha256(
        _frozen_args(), data_sha256={"a": "x"}, row_scale="unit")
    with_fields = driver._frozen_run_configuration_sha256(
        _frozen_args(shard_spec=None, shard_seed=42,
                     conserve_shard_steps=False, shard_manifest_sha256=None),
        data_sha256={"a": "x"}, row_scale="unit")
    assert legacy == with_fields


def test_config_hash_separates_shard_seeds():
    left = driver._frozen_run_configuration_sha256(
        _frozen_args(shard_spec=["a:3"], shard_seed=42,
                     conserve_shard_steps=True,
                     shard_manifest_sha256="aaaa"),
        data_sha256={"a": "x"}, row_scale="unit")
    right = driver._frozen_run_configuration_sha256(
        _frozen_args(shard_spec=["a:3"], shard_seed=123,
                     conserve_shard_steps=True,
                     shard_manifest_sha256="bbbb"),
        data_sha256={"a": "x"}, row_scale="unit")
    assert left != right


# --- argument guards -------------------------------------------------------

def _expect_exit(tmp_path, monkeypatch, extra):
    install(monkeypatch)
    monkeypatch.setattr(sys, "argv", build_argv(tmp_path, extra))
    with pytest.raises(SystemExit) as excinfo:
        driver.main()
    assert excinfo.value.code == 2


def test_naive_shard_cap_is_rejected(tmp_path, monkeypatch, capsys):
    _expect_exit(tmp_path, monkeypatch, ("--shard_spec", "clone:3"))
    assert "--conserve_shard_steps" in capsys.readouterr().err
    assert not list(Path(tmp_path).glob("federated_*.json"))


def test_shard_seed_requires_a_shard_spec(tmp_path, monkeypatch, capsys):
    _expect_exit(tmp_path, monkeypatch, ("--shard_seed", "7"))
    assert "--shard_spec" in capsys.readouterr().err


def test_conserve_flag_requires_a_shard_spec(tmp_path, monkeypatch, capsys):
    _expect_exit(tmp_path, monkeypatch, ("--conserve_shard_steps",))
    assert "--shard_spec" in capsys.readouterr().err


def test_unknown_shard_parent_rejected(tmp_path, monkeypatch, capsys):
    _expect_exit(tmp_path, monkeypatch,
                 ("--shard_spec", "absent:3", "--conserve_shard_steps"))
    assert "not in --slices" in capsys.readouterr().err


def test_malformed_shard_spec_rejected(tmp_path, monkeypatch, capsys):
    _expect_exit(tmp_path, monkeypatch,
                 ("--shard_spec", "clone", "--conserve_shard_steps"))
    assert "PARENT:N" in capsys.readouterr().err


def test_failed_shard_assertion_writes_no_result(tmp_path, monkeypatch):
    install(monkeypatch)
    monkeypatch.setattr(sys, "argv", build_argv(
        tmp_path, ("--shard_spec", "clone:9", "--conserve_shard_steps")))
    with pytest.raises(SystemExit) as excinfo:
        driver.main()
    assert excinfo.value.code == 2
    assert not list(Path(tmp_path).glob("federated_*.json"))
    assert not list(Path(tmp_path).glob("shard_manifest_*.json"))


def test_fixed_weights_are_counted_per_client(tmp_path, monkeypatch, capsys):
    install(monkeypatch)
    argv = build_argv(tmp_path, (
        "--lora_mode", "frozen-a", "--frozen_a_row_scale", "unit",
        "--weighted", "--weight_by", "normmaxmin", "--save_states",
        "--fedspan_step_policy", "median-active",
        "--fedspan_direction_policy", "fixed",
        "--fedspan_fixed_weights", "0.5", "0.5",
        "--shard_spec", "clone:3", "--conserve_shard_steps"))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        driver.main()
    assert excinfo.value.code == 2
    assert "one weight per slice (4)" in capsys.readouterr().err


# --- filenames and manifest placement --------------------------------------

def _frozen_shard_argv(out_directory, shard_seed, manifest_out=None):
    extra = ["--lora_mode", "frozen-a", "--frozen_a_row_scale", "unit",
             "--shard_spec", "clone:3", "--shard_seed", str(shard_seed),
             "--conserve_shard_steps"]
    if manifest_out is not None:
        extra += ["--shard_manifest_out", str(manifest_out)]
    return build_argv(out_directory, extra)


def test_shard_seeds_do_not_overwrite_each_other(tmp_path, monkeypatch):
    for seed in (42, 123):
        install(monkeypatch)
        monkeypatch.setattr(sys, "argv", _frozen_shard_argv(tmp_path, seed))
        driver.main()
    results = sorted(p.name for p in Path(tmp_path).glob("federated_*.json"))
    manifests = sorted(Path(tmp_path).glob("shard_manifest_*.json"))
    assert len(results) == 2, results
    assert len(manifests) == 2, manifests


def test_explicit_manifest_path_is_used(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "manifest.json"
    target.parent.mkdir()
    install(monkeypatch)
    monkeypatch.setattr(sys, "argv",
                        _frozen_shard_argv(tmp_path, 42, manifest_out=target))
    driver.main()
    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["shard_seed"] == 42
    assert not list(Path(tmp_path).glob("shard_manifest_*.json"))


def test_sharded_run_cannot_overwrite_an_unsharded_one(tmp_path,
                                                       monkeypatch):
    run(monkeypatch, tmp_path)
    install(monkeypatch)
    monkeypatch.setattr(sys, "argv", build_argv(tmp_path, SHARD_ARGS))
    driver.main()
    names = sorted(p.name for p in Path(tmp_path).glob("federated_*.json"))
    assert len(names) == 2, names
    assert any("-shard" in name for name in names)
    assert any("-shard" not in name for name in names)
