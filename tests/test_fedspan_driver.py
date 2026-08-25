"""CPU-only integration tests for FedSpan driver provenance and persistence."""
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import federated_forgetting as driver  # noqa: E402
from federated_forgetting import (  # noqa: E402
    _data_fingerprints,
    _frozen_run_configuration_sha256,
    dump_torch,
)
import driver_harness  # noqa: E402
from reference_solvers import maximin_reference, min_norm_reference  # noqa: E402


def test_e0_markers_bind_every_complete_round_to_the_telemetry_run_id(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("FEDCRAG_E0_RUN_ID", "e0-test-row")

    driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "uniform", num_rounds=2)

    output = capsys.readouterr().out
    expected = [
        "E0_ROUND_START e0-test-row 1/2",
        "E0_ROUND_END e0-test-row 1/2",
        "E0_ROUND_START e0-test-row 2/2",
        "E0_ROUND_END e0-test-row 2/2",
    ]
    positions = [output.index(marker) for marker in expected]
    assert positions == sorted(positions)
    assert all(output.count(marker) == 1 for marker in expected)


@pytest.mark.parametrize("run_id", ["", "UPPER", "has space", "../row",
                                     "e0_row", "-leading", "trailing-"])
def test_e0_marker_rejects_a_malformed_launcher_identity_before_data_work(
        monkeypatch, capsys, tmp_path, run_id):
    touched_data = False

    def unexpected_data_work(*args, **kwargs):
        nonlocal touched_data
        touched_data = True
        raise AssertionError("malformed telemetry must fail before data work")

    monkeypatch.setenv("FEDCRAG_E0_RUN_ID", run_id)
    monkeypatch.setattr(driver, "load_slice_with_train", unexpected_data_work)
    monkeypatch.setattr(driver, "get_git_commit", lambda: "abc123def456")
    monkeypatch.setattr(sys, "argv", driver_harness.build_argv(
        tmp_path, "frozen-a", "uniform"))

    with pytest.raises(SystemExit, match="2"):
        driver.main()

    assert "FEDCRAG_E0_RUN_ID" in capsys.readouterr().err
    assert touched_data is False


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
        "fedspan_step_policy": "fixed",
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
    different_policy = _frozen_run_configuration_sha256(
        frozen_args(fedspan_step_policy="median-active",
                    fedspan_step_norm=None), data_sha256=data_hashes)
    different_data = _frozen_run_configuration_sha256(
        frozen_args(), data_sha256={"nfcorpus": "changed", "fiqa": "b"})
    assert len(first) == 64
    assert first == second
    assert first != capped
    assert first != different_step
    assert first != different_policy
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


@pytest.mark.parametrize("step_policy", ["fixed", "median-active"])
def test_driver_normmaxmin_dispatch_persists_exact_round_record(
        monkeypatch, tmp_path, step_policy):
    module = "encoder.layer0.query"
    akey = f"{module}.lora_A.weight"
    bkey = f"{module}.lora_B.weight"
    broadcast = {
        akey: torch.eye(16),
        bkey: torch.zeros(3, 16),
    }
    c0_b = broadcast[bkey].clone()
    c1_b = broadcast[bkey].clone()
    c0_b[:, :2] = torch.tensor(
        [[1.0, 0.2], [0.1, 0.3], [0.2, 0.5]])
    c1_b[:, :2] = torch.tensor(
        [[0.2, 0.8], [0.4, 0.1], [0.7, 0.2]])
    clients = {
        "c0": {akey: broadcast[akey].clone(),
               bkey: c0_b},
        "c1": {akey: broadcast[akey].clone(),
               bkey: c1_b},
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
        driver, "new_model", lambda *args, **kwargs: (
            object(), driver_harness.module_scales("frozen-a")))
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

    argv = [
        "federated_forgetting.py",
        "--slices", "c0", "c1",
        "--metrics", "ndcg@10",
        "--num_rounds", "1",
        "--weighted",
        "--weight_by", "normmaxmin",
        "--lora_mode", "frozen-a",
        "--frozen_a_row_scale", "unit",
        "--fedspan_step_policy", step_policy,
        "--fedspan_direction_policy", "minnorm",
        "--save_states",
        "--out", str(tmp_path),
    ]
    if step_policy == "fixed":
        argv.extend(["--fedspan_step_norm", "0.1"])
    monkeypatch.setattr(sys, "argv", argv)
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
    assert result["method_contract"]["fedspan_step_policy"] == step_policy
    assert diagnostic["step_policy"] == step_policy
    assert diagnostic["status"] == "optimal"
    expected_step = 0.1
    if step_policy == "median-active":
        client_norms = [
            2.0 * torch.linalg.vector_norm(clients[name][bkey]).item()
            for name in ("c0", "c1")
        ]
        expected_step = sum(client_norms) / 2.0
        assert "smedian-active" in result_paths[0].name
    else:
        assert "s0p1" in result_paths[0].name
    assert diagnostic["resolved_step_norm"] == pytest.approx(
        expected_step, abs=2e-6)
    assert diagnostic["application"]["applied_step_norm"] == pytest.approx(
        expected_step, abs=2e-6)
    saved = torch.load(state_paths[0], map_location="cpu", weights_only=True)
    assert torch.equal(saved["broadcast"][akey], broadcast[akey])
    assert torch.equal(saved["global"][akey], broadcast[akey])
    from validate_e0 import validate_run_directory
    validation = validate_run_directory(tmp_path)
    assert validation["rounds_validated"] == 1
    assert validation["commit"] == "abc123def456"


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        ([], "requires --fedspan_step_policy"),
        (["--fedspan_step_policy", "fixed"],
         "fixed policy requires a positive finite --fedspan_step_norm"),
        (["--fedspan_step_policy", "median-active",
          "--fedspan_step_norm", "0.1"],
         "median-active policy rejects --fedspan_step_norm"),
    ],
)
def test_driver_rejects_illegal_normmaxmin_step_policy_before_data_work(
        monkeypatch, capsys, extra_args, message):
    touched_data = False

    def unexpected_data_work(*args, **kwargs):
        nonlocal touched_data
        touched_data = True
        raise AssertionError("data loading must not occur for illegal CLI")

    monkeypatch.setattr(driver, "load_slice_with_train", unexpected_data_work)
    monkeypatch.setattr(driver, "get_git_commit", lambda: "abc123def456")
    monkeypatch.setattr(sys, "argv", [
        "federated_forgetting.py",
        "--weighted",
        "--weight_by", "normmaxmin",
        "--lora_mode", "frozen-a",
        "--fedspan_direction_policy", "minnorm",
        "--save_states",
        *extra_args,
    ])

    with pytest.raises(SystemExit, match="2"):
        driver.main()

    assert message in capsys.readouterr().err
    assert touched_data is False


def test_driver_rejects_fedspan_policy_for_non_normmaxmin_arm(
        monkeypatch, capsys):
    touched_data = False

    def unexpected_data_work(*args, **kwargs):
        nonlocal touched_data
        touched_data = True
        raise AssertionError("data loading must not occur for illegal CLI")

    monkeypatch.setattr(driver, "load_slice_with_train", unexpected_data_work)
    monkeypatch.setattr(driver, "get_git_commit", lambda: "abc123def456")
    monkeypatch.setattr(sys, "argv", [
        "federated_forgetting.py",
        "--weighted",
        "--weight_by", "examples",
        "--fedspan_step_policy", "fixed",
        "--fedspan_step_norm", "0.1",
    ])

    with pytest.raises(SystemExit, match="2"):
        driver.main()

    assert "FedSpan step options are legal only with --weight_by normmaxmin" \
        in capsys.readouterr().err
    assert touched_data is False


# ------------------------------------- the real E0 coordinate x arm product


E0_CELLS = [
    ("trainable-ab", "uniform"),
    ("trainable-ab", "rawmaxmin"),
    ("frozen-a", "uniform"),
    ("frozen-a", "rawmaxmin"),
    ("frozen-a", "normmaxmin"),
]


def _independent_simplex_weights(states, broadcast, arm):
    """Weights the recorded arm implies, from an oracle rather than the run."""
    gram, norms = driver_harness.cosine_gram(states, broadcast)
    if arm == "uniform":
        return [1.0 / len(states)] * len(states), gram, norms
    _, weights = maximin_reference(gram)
    return list(weights), gram, norms


@pytest.mark.parametrize(("lora_mode", "arm"), E0_CELLS)
def test_driver_e0_cross_product_applies_a_hand_computed_aggregate(
        monkeypatch, tmp_path, lora_mode, arm):
    """Every legal E0 cell must persist the global its recorded arm implies.

    The expected global is rebuilt here from the client states and weights
    derived by the independent oracle, never from the driver's own record, so
    an aggregation that silently reverts to uniform weights fails.
    """
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, lora_mode, arm)
    payload, _ = driver_harness.load_round_states(tmp_path)

    broadcast = payload["broadcast"]
    states = [payload["clients"][name] for name in driver_harness.SLICES]
    expected_weights, gram, _ = _independent_simplex_weights(
        states, broadcast, arm)

    assert result["lora_mode"] == lora_mode
    assert result["weight_by_canonical"] == (None if arm == "uniform" else arm)

    if arm == "normmaxmin":
        diagnostic = result["fedspan_diagnostics"]["round_1"]
        optimum, min_norm_w = min_norm_reference(gram)
        assert diagnostic["direction_policy"] == "minnorm"
        assert diagnostic["direction_policy_specified"] is True
        assert diagnostic["achieved_min_direction_cosine"] == pytest.approx(
            math.sqrt(optimum), abs=1e-6)
        assert diagnostic["direction_solver_shortfall"] == pytest.approx(
            0.0, abs=1e-6)
        assert diagnostic["simplex_weights"] == pytest.approx(
            list(min_norm_w), abs=1e-6)
        # Non-simplex true-step coefficients, applied to raw B deltas only.
        coefficients = diagnostic["delta_weights"]
        expected_b = broadcast[driver_harness.B_KEY].double().clone()
        for coefficient, state in zip(coefficients, states):
            expected_b += coefficient * (
                state[driver_harness.B_KEY].double()
                - broadcast[driver_harness.B_KEY].double())
        assert torch.allclose(
            payload["global"][driver_harness.B_KEY].double(),
            expected_b, atol=1e-6)
        assert torch.equal(payload["global"][driver_harness.A_KEY],
                           broadcast[driver_harness.A_KEY])
        return

    if arm == "rawmaxmin":
        recorded = result["scheme_diagnostics"]["round_1"]
        assert recorded["scheme"] == "rawmaxmin"
        assert recorded["fallback"] is None
        assert recorded["weights"] == pytest.approx(expected_weights, abs=1e-6)
        # The fixture must actually discriminate: uniform weights would give a
        # materially different aggregate.
        assert max(abs(value - 1.0 / len(states))
                   for value in expected_weights) > 0.05

    if lora_mode == "frozen-a":
        expected_b = broadcast[driver_harness.B_KEY].double().clone()
        for weight, state in zip(expected_weights, states):
            expected_b += weight * (
                state[driver_harness.B_KEY].double()
                - broadcast[driver_harness.B_KEY].double())
        assert torch.allclose(
            payload["global"][driver_harness.B_KEY].double(),
            expected_b, atol=1e-6)
        assert torch.equal(payload["global"][driver_harness.A_KEY],
                           broadcast[driver_harness.A_KEY])
    else:
        for key in broadcast:
            expected = sum(
                weight * state[key].double()
                for weight, state in zip(expected_weights, states))
            assert torch.allclose(
                payload["global"][key].double(), expected, atol=1e-6)

    assert payload["global_state_sha256"] == driver.state_dict_sha256(
        payload["global"])


@pytest.mark.parametrize(("lora_mode", "arm"), E0_CELLS)
def test_driver_records_per_client_delta_norms_for_every_e0_cell(
        monkeypatch, tmp_path, lora_mode, arm):
    """D2a audit trail: the effective step magnitudes must exist everywhere."""
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, lora_mode, arm)
    payload, _ = driver_harness.load_round_states(tmp_path)

    states = [payload["clients"][name] for name in driver_harness.SLICES]
    _, expected_norms = driver_harness.cosine_gram(states, payload["broadcast"])

    recorded = result["client_delta_norms"]["round_1"]
    assert sorted(recorded) == sorted(driver_harness.SLICES)
    for name, expected in zip(driver_harness.SLICES, expected_norms):
        assert recorded[name] == pytest.approx(float(expected), rel=1e-6)


@pytest.mark.parametrize("row_scale_c", [1.0, 0.5773502691896258, 3.0])
def test_driver_records_true_effective_step_norms_at_any_frozen_row_scale(
        monkeypatch, tmp_path, row_scale_c):
    """The frozen-A row scale c must be counted exactly once, not twice.

    ``client_delta_norms`` and ``fedspan_diagnostics.client_norms`` are two
    records of one quantity written in the same round. They agree only when
    the materialized-space record is scaled by the bare PEFT sigma, since the
    materialized update already contains A and therefore already contains c.
    """
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        row_scale_c=row_scale_c)
    payload, _ = driver_harness.load_round_states(tmp_path)

    states = [payload["clients"][name] for name in driver_harness.SLICES]
    _, truth = driver_harness.cosine_gram(states, payload["broadcast"])

    recorded = result["client_delta_norms"]["round_1"]
    for name, expected in zip(driver_harness.SLICES, truth):
        assert recorded[name] == pytest.approx(float(expected), rel=1e-9)

    # The raw-B record reaches the same quantity through the float64 geometry
    # scale sigma*c rather than through the float32 A the state carries, so it
    # agrees only to float32 resolution in c (measured gap 1.8e-08 at
    # c = 0.5773502691896258).
    solver_norms = result["fedspan_diagnostics"]["round_1"]["client_norms"]
    for reported, expected in zip(solver_norms, truth):
        assert reported == pytest.approx(float(expected), rel=1e-6)


def test_driver_rejects_normmaxmin_without_a_direction_policy(
        monkeypatch, capsys, tmp_path):
    driver_harness.install_mocks(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "federated_forgetting.py",
        "--weighted", "--weight_by", "normmaxmin",
        "--lora_mode", "frozen-a", "--save_states",
        "--fedspan_step_policy", "median-active",
        "--out", str(tmp_path),
    ])
    with pytest.raises(SystemExit, match="2"):
        driver.main()
    assert "requires --fedspan_direction_policy" in capsys.readouterr().err


def test_driver_rejects_direction_policy_on_a_non_normmaxmin_arm(
        monkeypatch, capsys, tmp_path):
    driver_harness.install_mocks(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "federated_forgetting.py",
        "--weighted", "--weight_by", "rawmaxmin",
        "--lora_mode", "frozen-a",
        "--fedspan_direction_policy", "minnorm",
        "--out", str(tmp_path),
    ])
    with pytest.raises(SystemExit, match="2"):
        driver.main()
    assert "--fedspan_direction_policy is legal only with" in \
        capsys.readouterr().err


@pytest.mark.parametrize("policy", ["minnorm", "maxmin-lp"])
def test_driver_separates_direction_policies_in_filename_and_hash(
        monkeypatch, tmp_path, policy):
    result, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy=policy)
    assert f"-dir{policy}" in path.name
    assert result["method_contract"]["fedspan_direction_policy"] == policy
    assert result["fedspan_diagnostics"]["round_1"]["direction_policy"] == \
        policy


def test_driver_direction_policy_changes_the_configuration_hash(
        monkeypatch, tmp_path):
    left, _ = driver_harness.run_driver(
        monkeypatch, tmp_path / "a", "frozen-a", "normmaxmin",
        direction_policy="minnorm")
    right, _ = driver_harness.run_driver(
        monkeypatch, tmp_path / "b", "frozen-a", "normmaxmin",
        direction_policy="maxmin-lp")
    assert (left["method_contract"]["run_configuration_sha256"]
            != right["method_contract"]["run_configuration_sha256"])


# ------------------------------------------------------ provenance refusal


@pytest.mark.parametrize("commit", ["abc123def456-dirty", "unknown",
                                    "abc123def456-unknown-worktree"])
def test_normmaxmin_refuses_unclean_source_provenance(
        monkeypatch, capsys, tmp_path, commit):
    driver_harness.install_mocks(monkeypatch, commit=commit)
    monkeypatch.setattr(sys, "argv", driver_harness.build_argv(
        tmp_path, "frozen-a", "normmaxmin"))

    with pytest.raises(SystemExit, match="2"):
        driver.main()

    assert "refuses unknown/dirty source provenance" in capsys.readouterr().err
    assert not list(tmp_path.glob("federated_*.json"))


@pytest.mark.parametrize("commit", ["abc123def456-dirty", "unknown",
                                    "abc123def456-unknown-worktree"])
def test_explicit_override_permits_unclean_provenance_and_records_it(
        monkeypatch, tmp_path, commit):
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", commit=commit,
        extra=["--allow_dirty_provenance"])

    assert result["commit"] == commit
    assert result["method_contract"]["dirty_provenance_override"] is True


def test_unclean_provenance_is_permitted_for_non_normmaxmin_arms(
        monkeypatch, tmp_path):
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "rawmaxmin",
        commit="abc123def456-dirty")
    assert result["commit"] == "abc123def456-dirty"
    assert result["method_contract"]["dirty_provenance_override"] is False


# ------------------------------------------- explicit frozen-A row scale


def test_driver_requires_an_explicit_row_scale_for_frozen_a(
        monkeypatch, capsys, tmp_path):
    """Defaulting to 'unit' silently reintroduces the sqrt(3) confound."""
    driver_harness.install_mocks(monkeypatch)
    monkeypatch.setattr(sys, "argv", driver_harness.build_argv(
        tmp_path, "frozen-a", "uniform", row_scale=None))

    with pytest.raises(SystemExit, match="2"):
        driver.main()

    assert "--frozen_a_row_scale is required" in capsys.readouterr().err
    assert not list(tmp_path.glob("federated_*.json"))


@pytest.mark.parametrize("row_scale", ["unit", "peft-init"])
def test_driver_records_the_row_scale_it_was_given(
        monkeypatch, tmp_path, row_scale):
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "uniform", row_scale=row_scale)
    contract = result["method_contract"]

    assert contract["frozen_a_row_scale"] == row_scale
    assert contract["frozen_a_row_scale_specified"] is True
    assert result["args"]["frozen_a_row_scale"] == row_scale


# --------------------------------------------------- fallback visibility


def test_driver_warns_when_fedspan_falls_back_to_a_zero_update(
        monkeypatch, capsys, tmp_path):
    """Every other arm prints on fallback; a silent FedSpan no-op arm would
    report the frozen baseline's retention under the FedSpan label."""
    idle = {name: driver_harness.broadcast_state()
            for name in driver_harness.SLICES}
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin", clients=idle)

    diagnostic = result["fedspan_diagnostics"]["round_1"]
    assert diagnostic["fallback"] == "zero_update"
    assert "WARNING: normmaxmin fell back to zero_update" in \
        capsys.readouterr().out
