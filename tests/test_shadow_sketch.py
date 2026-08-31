"""Shadow sketch telemetry: what an m-dim Gaussian sketch WOULD have done.

Behavior-neutral by contract: the applied weights must be bit-identical with
telemetry on and off (pinned below with ==, not allclose). The sketch is the
feasibility instrument for a secure-aggregation-compatible Gram (supervisor
note 2026-08-31): every client projects its effective update through the SAME
per-round Gaussian matrix, and the server solves the direction on the sketched
Gram. Here we only RECORD that counterfactual, next to the true one.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregation_schemes import fedspan_delta_weights  # noqa: E402
from fedspan_fixtures import (EFFECTIVE_DIM,  # noqa: E402
                              federation_from_unit_directions,
                              federation_with_cosine_gram)

GRAM = np.array([
    [1.00, 0.55, 0.35, -0.10],
    [0.55, 1.00, 0.40, 0.05],
    [0.35, 0.40, 1.00, 0.20],
    [-0.10, 0.05, 0.20, 1.00],
])
RADII = [1.9, 2.0, 2.1, 0.2]
SHADOW = {"sizes": (256, 512), "seed": 20260831}


def solve(policy="exact", shadow=None, **kwargs):
    broadcast, clients, scales = federation_with_cosine_gram(GRAM, RADII)
    return fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_policy="median-active",
        direction_policy=policy, shadow_sketch=shadow, **kwargs)


def test_shadow_sketch_absent_by_default():
    assert "shadow_sketch" not in solve()


@pytest.mark.parametrize("policy,extra", [
    ("exact", {}), ("minnorm", {}), ("maxmin-lp", {}),
    ("fixed", {"fixed_weights": [0.25, 0.25, 0.25, 0.25]}),
])
def test_applied_weights_bit_identical_with_and_without(policy, extra):
    """The telemetry must not move ANY applied quantity, on any arm, at all."""
    plain = solve(policy, shadow=None, **extra)
    shadowed = solve(policy, shadow=SHADOW, **extra)
    assert shadowed["delta_weights"] == plain["delta_weights"]          # bitwise
    assert shadowed["simplex_weights"] == plain["simplex_weights"]      # bitwise
    assert shadowed["resolved_step_norm"] == plain["resolved_step_norm"]
    assert shadowed["min_norm_value"] == plain["min_norm_value"]
    assert "shadow_sketch" in shadowed and "shadow_sketch" not in plain


def test_shadow_block_well_formed():
    record = solve(shadow=SHADOW)["shadow_sketch"]
    assert record["sizes"] == [256, 512]
    assert record["seed"] == SHADOW["seed"]
    for m in ("256", "512"):
        entry = record["per_size"][m]
        assert "failed" not in entry
        w = np.asarray(entry["weights"])
        assert w.shape == (4,) and w.min() >= -1e-12
        assert abs(float(w.sum()) - 1.0) < 1e-9
        C_hat = np.asarray(entry["sketched_cosine_gram"])
        assert C_hat.shape == (4, 4)
        np.testing.assert_allclose(C_hat, C_hat.T, atol=0)
        np.testing.assert_allclose(np.diag(C_hat), 1.0, atol=1e-12)
        assert all(v > 0 for v in entry["sketched_norms"])
        # The sketched direction cannot beat the exact optimum in the TRUE
        # geometry -- that is the definition of the optimum.
        assert entry["shortfall_vs_exact"] >= -1e-9
        assert math.isfinite(entry["gamma_true_of_sketched_direction"])
    # More dimensions must not hurt (seeded, so this is deterministic).
    assert (record["per_size"]["512"]["gram_max_abs_err"]
            <= record["per_size"]["256"]["gram_max_abs_err"] * 1.6)


def test_shadow_deterministic_in_seed_and_sensitive_to_it():
    a = solve(shadow=SHADOW)["shadow_sketch"]
    b = solve(shadow=SHADOW)["shadow_sketch"]
    assert a["per_size"] == b["per_size"]
    c = solve(shadow={"sizes": (256, 512), "seed": 7})["shadow_sketch"]
    assert (a["per_size"]["256"]["sketched_cosine_gram"]
            != c["per_size"]["256"]["sketched_cosine_gram"])


def test_all_clients_share_one_sketch_matrix():
    """Exact clones must sketch to cosine EXACTLY 1.

    Proportional vectors stay proportional under any shared linear map, so
    their sketched cosine is 1 to float precision. Under per-client sketch
    matrices the two projections would be independent Gaussians with cosine
    O(1/sqrt(m)) -- catastrophically wrong for the clone federation E3 is
    built to study. This test is what pins the shared-S property.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=EFFECTIVE_DIM)
    other = rng.normal(size=EFFECTIVE_DIM)
    directions = np.stack([base, base, other])   # clients 0,1 exact clones
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    broadcast, clients, scales = federation_from_unit_directions(
        directions, [1.5, 2.5, 1.0])
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_policy="median-active",
        direction_policy="exact", shadow_sketch={"sizes": (256,), "seed": 11})
    C_hat = np.asarray(
        result["shadow_sketch"]["per_size"]["256"]["sketched_cosine_gram"])
    assert C_hat[0, 1] > 1.0 - 1e-6, (
        f"clone sketched cosine {C_hat[0, 1]}: the sketch matrix is not "
        "shared across clients")


def test_shadow_failure_is_recorded_not_raised(monkeypatch):
    """A diagnostics failure must not abort a pre-registered run.

    Fail-closed is the contract of the APPLIED path; the shadow record is a
    counterfactual, and a telemetry crash that killed a $55 E3 run would
    create pressure to strip telemetry mid-experiment -- worse for integrity
    than a loudly-recorded gap.
    """
    import aggregation_schemes as A
    real = A.minnorm_exact_weights
    calls = {"n": 0}

    def explode_on_shadow_calls(cosine, *a, **k):
        calls["n"] += 1
        if calls["n"] > 2:     # let the applied + measurement solves through
            raise A.DirectionSolverError("synthetic shadow failure")
        return real(cosine, *a, **k)

    monkeypatch.setattr(A, "minnorm_exact_weights", explode_on_shadow_calls)
    result = solve("exact", shadow=SHADOW)
    assert result["status"] == "optimal"                 # applied path intact
    assert all(math.isfinite(v) for v in result["delta_weights"])
    per = result["shadow_sketch"]["per_size"]
    assert any("failed" in per[m] for m in per)


def test_shadow_sketch_config_is_validated():
    with pytest.raises(ValueError, match="shadow_sketch"):
        solve(shadow={"sizes": (0, 256), "seed": 1})
    with pytest.raises(ValueError, match="shadow_sketch"):
        solve(shadow={"sizes": (512, 256), "seed": 1})
    with pytest.raises(ValueError, match="shadow_sketch"):
        solve(shadow={"sizes": (256,)})


# --- driver level ------------------------------------------------------------

import json  # noqa: E402

import driver_harness  # noqa: E402
from validate_e0 import validate_run_directory  # noqa: E402


def test_driver_records_shadow_every_round_and_validator_accepts(
        monkeypatch, tmp_path):
    result, _ = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy="exact", num_rounds=2, row_scale="peft-init",
        extra=("--fedspan_shadow_sketch", "64", "128"))
    diagnostics = result["fedspan_diagnostics"]
    assert set(diagnostics) == {"round_1", "round_2"}
    seeds = set()
    for label, record in diagnostics.items():
        shadow = record["shadow_sketch"]
        assert shadow["sizes"] == [64, 128]
        assert set(shadow["per_size"]) == {"64", "128"}
        for entry in shadow["per_size"].values():
            assert "failed" not in entry, entry
        seeds.add(shadow["seed"])
    assert len(seeds) == 2, "per-round sketch seeds must differ"
    # The acceptance gate run_e0.sh applies to every run must accept this.
    assert validate_run_directory(tmp_path)["rounds_validated"] == 2


def test_driver_without_flag_is_unchanged(monkeypatch, tmp_path):
    result, path = driver_harness.run_driver(
        monkeypatch, tmp_path, "frozen-a", "normmaxmin",
        direction_policy="exact", row_scale="peft-init")
    for record in result["fedspan_diagnostics"].values():
        assert "shadow_sketch" not in record
    # Recorded args keep the flag at null (provenance is verbatim); what must
    # NOT change is the configuration hash, which adds the field only when
    # supplied -- so legacy runs keep their filenames.
    assert result["args"].get("fedspan_shadow_sketch") is None


def test_driver_rejects_shadow_outside_the_geometry_pipeline(
        monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        driver_harness.run_driver(
            monkeypatch, tmp_path, "trainable-ab", "uniform",
            extra=("--fedspan_shadow_sketch", "64"))
    assert "legal only with --weight_by normmaxmin" in capsys.readouterr().err
