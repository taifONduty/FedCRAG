"""The 33 registered E3 runs must match the signed registration exactly."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e3_manifest import (MAIN_SHARD_SEED, SEEDS,  # noqa: E402
                         p0_gate, p0_probe, registered_runs)


RUNS = registered_runs()
BY_NAME = {run["name"]: run for run in RUNS}


def flags(run):
    return " ".join(run["args"])


def test_exactly_33_registered_runs_with_unique_names():
    assert len(RUNS) == 33
    assert len(BY_NAME) == 33


def test_registered_composition_matches_the_signed_budget():
    main = [n for n in BY_NAME if n.startswith("e3-") and "-k2-" not in n
            and "partition" not in n and "msweep" not in n
            and "noise-floor" not in n]
    k2 = [n for n in BY_NAME if "-k2-" in n]
    robust = [n for n in BY_NAME if "partition" in n]
    msweep = [n for n in BY_NAME if "msweep" in n]
    floor = [n for n in BY_NAME if "noise-floor" in n]
    assert (len(main), len(k2), len(robust), len(msweep), len(floor)) == \
        (18, 6, 4, 4, 1)


def test_main_grid_is_paired_on_one_partition():
    """Every main-grid and m-sweep run shards with shard_seed 42; ONLY the
    four robustness runs differ — otherwise cross-arm contrasts are not
    paired on identical shards and the primary analysis is invalid."""
    for name, run in BY_NAME.items():
        text = flags(run)
        if "--shard_seed" not in text:
            assert "-k2-" in name, f"{name}: unsharded but not the K=2 control"
            continue
        seed = run["args"][run["args"].index("--shard_seed") + 1]
        if "partition" in name:
            assert seed != str(MAIN_SHARD_SEED), name
        else:
            assert seed == str(MAIN_SHARD_SEED), name


def test_every_sharded_run_conserves_steps():
    for name, run in BY_NAME.items():
        if "--shard_spec" in flags(run):
            assert "--conserve_shard_steps" in flags(run), name


def test_fixed_weight_arities_match_client_counts():
    """K clients = m shards + singletons; a wrong arity is refused by the
    driver AFTER data loading, i.e. after money is spent."""
    for name, run in BY_NAME.items():
        text = run["args"]
        if "--fedspan_fixed_weights" not in text:
            continue
        i = text.index("--fedspan_fixed_weights") + 1
        weights = []
        while i < len(text) and not text[i].startswith("--"):
            weights.append(float(text[i])); i += 1
        if "-k2-" in name:
            expected = 2
        elif "msweep-m2" in name:
            expected = 4
        elif "msweep-m4" in name:
            expected = 6
        else:
            expected = 5
        assert len(weights) == expected, (name, weights)
        assert abs(sum(weights) - 1.0) < 1e-9, (name, sum(weights))


def test_over_distributions_puts_a_third_on_each_distribution():
    run = BY_NAME["e3-uniform-over-distributions-s42"]
    text = run["args"]
    i = text.index("--fedspan_fixed_weights") + 1
    w = [float(text[j]) for j in range(i, i + 5)]
    assert w[3] == pytest.approx(1 / 3) and w[4] == pytest.approx(1 / 3)
    assert sum(w[:3]) == pytest.approx(1 / 3)


def test_shadow_sketch_rides_every_geometry_pipeline_run():
    for name, run in BY_NAME.items():
        text = flags(run)
        if "--weight_by normmaxmin" in text:
            assert "--fedspan_shadow_sketch 1024 4096" in text, name
        else:
            assert "--fedspan_shadow_sketch" not in text, name


def test_qffl_q_is_the_registered_fallback():
    for seed in SEEDS:
        assert "--qffl_q 1.0" in flags(BY_NAME[f"e3-qffl-s{seed}"])


def test_noise_floor_duplicates_the_first_fedspan_cell_exactly():
    a = flags(BY_NAME["e3-fedspan-exact-s42"])
    b = flags(BY_NAME["e3-noise-floor-fedspan-exact-s42"])
    assert a == b, "the floor must repeat the cell bit-for-bit (args level)"


def test_p0_probe_is_one_round_and_not_registered():
    probe = p0_probe()
    assert "--num_rounds 1" in flags(probe)
    assert probe["name"] not in BY_NAME


def _gate_result(block, cross):
    C = [[1.0] * 5 for _ in range(5)]
    for i in range(3):
        for j in range(i + 1, 3):
            C[i][j] = C[j][i] = block
        for j in (3, 4):
            C[i][j] = C[j][i] = cross
    return {"fedspan_diagnostics": {"round_1": {"cosine_gram_active": C}}}


def test_p0_gate_passes_clone_geometry_and_fails_non_clone():
    ok, report = p0_gate(_gate_result(block=0.25, cross=0.05))
    assert ok, report
    ok, report = p0_gate(_gate_result(block=0.10, cross=0.05))
    assert not ok and "FAIL" in report          # shards not clone-like
    ok, report = p0_gate(_gate_result(block=0.30, cross=0.35))
    assert not ok                               # singletons as close as clones
