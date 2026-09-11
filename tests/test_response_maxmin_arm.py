"""The response-maxmin arm through the real driver (design note 2026-09-08)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import driver_harness  # noqa: E402
import federated_forgetting as driver  # noqa: E402
from validate_e0 import E0ValidationError, validate_run_directory  # noqa: E402

COUNTS = {"c0": 1000, "c1": 250, "c2": 40}
STEPS = {"c0": 31, "c1": 8, "c2": 2}
ARM_FLAGS = ("--response_dev_fraction", "0.25", "--response_dev_min", "2",
             "--response_lattice_step", "0.5", "--response_scales", "1.0",
             "--response_verify", "2")


def test_harness_dev_data_has_forty_queries_with_one_positive_each():
    loader = driver_harness.dev_mock_data()
    data = loader("c0", "/nowhere")
    assert len(data["train_q"]) == 40 and len(data["train_qrels"]) == 40
    assert all(len(v) == 1 for v in data["train_qrels"].values())
    assert data["corpus"]["c0-d3"]["text"] == "c0-d3"
    emb = driver_harness.fake_response_encoder(
        None, driver_harness.broadcast_state(), ["c0-d3", "c0-q3"], 8)
    assert emb.shape == (2, 8) and np.allclose(np.linalg.norm(emb, axis=1), 1.0)


def run_arm(monkeypatch, tmp_path, extra=()):
    return driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "response-maxmin",
        example_counts=COUNTS, step_counts=STEPS,
        dev_data=driver_harness.dev_mock_data(),
        fake_encoder=driver_harness.fake_response_encoder,
        extra=ARM_FLAGS + tuple(extra))


def test_arm_records_split_grid_shortlist_and_applied_weights(monkeypatch, tmp_path):
    result, path = run_arm(monkeypatch, tmp_path)
    split = result["dev_split"]
    assert split["fraction"] == 0.25 and split["per_client"]["c0"]["n_dev_queries"] == 10
    assert split["per_client"]["c0"]["n_train_queries"] == 30
    assert len(split["per_client"]["c0"]["dev_sha256"]) == 64
    rec = result["scheme_diagnostics"]["round_1"]
    assert rec["scheme"] == "response-maxmin"
    assert set(rec["candidates"]) >= {"uniform", "examples", "fednova",
                                      "solo_c0", "solo_c1", "solo_c2"}
    assert rec["candidates"]["examples"]["v"] == pytest.approx(
        [1000 / 1290, 250 / 1290, 40 / 1290])
    # the shortlist holds at most --response_verify floor-feasible candidates; with the
    # mock encoder only some candidates clear the frozen floor on prediction
    assert 1 <= len(rec["shortlist"]) <= 2 and rec["chosen"] in rec["verified"]
    assert rec["chosen"] in rec["shortlist"] and rec["status"] == "optimal"
    assert rec["grid_size"] == len(rec["candidates"]) and rec["encodes_per_client"] >= 5
    assert rec["weights"] == rec["verified"][rec["chosen"]]["v"]
    assert result["weight_space"] == "delta"
    assert result["round_weights"]["round_1"] == [round(w, 5) for w in rec["weights"]]
    assert "-rmm" in path.name


def test_filename_tag_changes_with_the_arm_configuration(monkeypatch, tmp_path):
    _, p1 = run_arm(monkeypatch, tmp_path / "a")
    _, p2 = run_arm(monkeypatch, tmp_path / "b", extra=("--response_verify", "3"))
    assert p1.name != p2.name and "-rmm" in p2.name


def test_dev_split_is_reproducible_across_runs(monkeypatch, tmp_path):
    r1, _ = run_arm(monkeypatch, tmp_path / "a")
    r2, _ = run_arm(monkeypatch, tmp_path / "b")
    assert r1["dev_split"] == r2["dev_split"]
    assert r1["dev_split"]["per_client"]["c1"]["n_train_queries"] == 30


def test_unreachable_floor_applies_zero_step_and_keeps_the_broadcast(monkeypatch, tmp_path):
    result, _ = run_arm(monkeypatch, tmp_path, extra=("--response_floor_delta", "-1.0"))
    rec = result["scheme_diagnostics"]["round_1"]
    assert rec["status"] == "zero_step" and rec["weights"] == [0.0, 0.0, 0.0]
    payload, _ = driver_harness.load_round_states(tmp_path)
    for key in payload["broadcast"]:
        assert driver.torch.equal(payload["global"][key], payload["broadcast"][key])


@pytest.mark.parametrize("bad", [
    ("--lora_mode", "frozen-a"),                       # wrong coordinate
    ("--response_verify", "0"),
    ("--response_lattice_step", "0.3"),
    ("--response_dev_fraction", "0.7"),
    ("--response_scales", "1.0,-1.0"),
])
def test_illegal_configurations_are_refused(monkeypatch, tmp_path, bad):
    lora_mode = "frozen-a" if bad[0] == "--lora_mode" else "trainable-ab"
    extra = ARM_FLAGS if bad[0] == "--lora_mode" else ARM_FLAGS + bad
    with pytest.raises(SystemExit):
        driver_harness.run_driver(
            monkeypatch, tmp_path, lora_mode, "response-maxmin",
            example_counts=COUNTS, step_counts=STEPS,
            dev_data=driver_harness.dev_mock_data(),
            fake_encoder=driver_harness.fake_response_encoder, extra=extra)


def test_arm_without_weighted_is_refused(monkeypatch, tmp_path):
    driver_harness.install_mocks(monkeypatch, dev_data=driver_harness.dev_mock_data(),
                                 fake_encoder=driver_harness.fake_response_encoder)
    argv = driver_harness.build_argv(tmp_path, "trainable-ab", "uniform")
    argv += ["--weight_by", "response-maxmin", *ARM_FLAGS]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        driver.main()


def _rewrite(path, result):
    driver_harness.rewrite_result(path, result)


def test_validator_accepts_a_genuine_run(monkeypatch, tmp_path):
    run_arm(monkeypatch, tmp_path)
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1


def test_validator_refuses_forged_weights(monkeypatch, tmp_path):
    result, path = run_arm(monkeypatch, tmp_path)
    rec = result["scheme_diagnostics"]["round_1"]
    rec["weights"] = [w * 1.1 + 0.01 for w in rec["weights"]]
    _rewrite(path, result)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_validator_refuses_a_choice_the_measurements_do_not_support(monkeypatch, tmp_path):
    result, path = run_arm(monkeypatch, tmp_path, extra=("--response_floor", "none"))
    rec = result["scheme_diagnostics"]["round_1"]
    assert len(rec["shortlist"]) == 2                  # no floor: both leaders verified
    other = [n for n in rec["shortlist"] if n != rec["chosen"]][0]
    rec["chosen"] = other
    rec["weights"] = rec["verified"][other]["v"]
    _rewrite(path, result)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_validator_refuses_a_shortlist_the_predictions_do_not_support(monkeypatch, tmp_path):
    result, path = run_arm(monkeypatch, tmp_path)
    rec = result["scheme_diagnostics"]["round_1"]
    loser = [n for n in rec["candidates"] if n not in rec["shortlist"]][0]
    rec["candidates"][loser]["pred"] = [1.0] * 3      # would have won the shortlist
    _rewrite(path, result)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_validator_accepts_the_zero_step_outcome(monkeypatch, tmp_path):
    run_arm(monkeypatch, tmp_path, extra=("--response_floor_delta", "-1.0"))
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1


V2_FLAGS = ("--response_candidates", "compact", "--response_select", "pessimistic",
            "--response_eq_scales", "0.5,1.0,2.0", "--response_game_scales", "1.0,2.0",
            "--response_model_pick")


def test_v2_arm_runs_validates_and_has_its_own_tag(monkeypatch, tmp_path):
    result, path = run_arm(monkeypatch, tmp_path, extra=V2_FLAGS)
    rec = result["scheme_diagnostics"]["round_1"]
    assert rec["candidate_mode"] == "compact" and rec["select"] == "pessimistic"
    # default --response_eq_top 2,3 with three clients: m = 2 is valid, m = 3 is not
    assert set(rec["families"]) == {"eq_x0.5", "eq_x1", "eq_x2", "eq2_x0.5", "eq2_x1", "eq2_x2",
                                    "game_x1", "game_x2"}
    assert {"greedy", "subset", "model"} <= set(rec["picks"])
    assert rec["chosen"] in rec["contenders"]
    assert len(rec["geometry"]["norms"]) == 3 and sum(rec["applied_magnitude_shares"]) == pytest.approx(1.0)
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1
    _, v1_path = run_arm(monkeypatch, tmp_path / "v1")
    assert path.name != v1_path.name


def test_validator_refuses_a_forged_family_candidate(monkeypatch, tmp_path):
    result, path = run_arm(monkeypatch, tmp_path, extra=V2_FLAGS)
    rec = result["scheme_diagnostics"]["round_1"]
    rec["candidates"]["eq_x1"]["v"] = [x * 1.5 for x in rec["candidates"]["eq_x1"]["v"]]
    _rewrite(path, result)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_validator_refuses_a_subset_pick_the_statistics_do_not_support(monkeypatch, tmp_path):
    result, path = run_arm(monkeypatch, tmp_path, extra=V2_FLAGS)
    rec = result["scheme_diagnostics"]["round_1"]
    source = rec["picks"]["subset"]["source"]
    other = [n for n in rec["candidates"] if n.startswith("sub_") and n != source][0]
    rec["candidates"][other]["stat"] = [1.0] * 3       # would have been the best subset
    _rewrite(path, result)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_validator_refuses_a_pessimistic_choice_the_statistics_do_not_support(monkeypatch, tmp_path):
    result, path = run_arm(monkeypatch, tmp_path, extra=V2_FLAGS + ("--response_floor", "none"))
    rec = result["scheme_diagnostics"]["round_1"]
    assert len(rec["shortlist"]) == 2
    other = [n for n in rec["shortlist"] if n != rec["chosen"]][0]
    rec["verified"][other]["stat"] = [1.0] * 3          # forged statistic for the loser
    _rewrite(path, result)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_model_pick_is_refused_outside_compact_mode(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_arm(monkeypatch, tmp_path, extra=("--response_model_pick",))


def test_pareto_arm_runs_and_validates(monkeypatch, tmp_path):
    flags = ("--response_candidates", "compact", "--response_select", "pareto",
             "--response_eq_scales", "1.0", "--response_game_scales", "1.0",
             "--response_eq_top", "2")
    result, path = run_arm(monkeypatch, tmp_path, extra=flags)
    rec = result["scheme_diagnostics"]["round_1"]
    assert rec["select"] == "pareto" and "eq2_x1" in rec["families"]
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1
    rec["candidates"]["eq2_x1"]["v"] = [x * 2 for x in rec["candidates"]["eq2_x1"]["v"]]
    _rewrite(path, result)
    with pytest.raises(E0ValidationError):
        validate_run_directory(tmp_path)


def test_dev_holdout_baseline_trains_on_the_arm_split_and_records_it(monkeypatch, tmp_path):
    result, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "trainable-ab", "uniform",
        example_counts=COUNTS, step_counts=STEPS,
        dev_data=driver_harness.dev_mock_data(),
        extra=("--dev_holdout", "--response_dev_fraction", "0.25", "--response_dev_min", "2"))
    split = result["dev_split"]
    assert split["holdout_only"] is True and split["fraction"] == 0.25
    assert split["per_client"]["c0"] == {"n_train_queries": 30, "n_dev_queries": 10,
                                         "dev_sha256": split["per_client"]["c0"]["dev_sha256"]}
    assert "-devholdout0p25" in path.name and "scheme_diagnostics" not in result
    assert validate_run_directory(tmp_path)["rounds_validated"] == 1
    # the same seed and slice give the response arm the identical held-out queries
    arm_result, _ = run_arm(monkeypatch, tmp_path / "arm")
    assert arm_result["dev_split"]["per_client"]["c0"]["dev_sha256"] == split["per_client"]["c0"]["dev_sha256"]


def test_dev_holdout_is_refused_with_the_response_arm(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_arm(monkeypatch, tmp_path, extra=("--dev_holdout",))
