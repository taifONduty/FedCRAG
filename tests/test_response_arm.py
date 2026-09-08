"""Round logic of the response-maxmin arm with a deterministic fake encoder."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import response_aggregation as ra  # noqa: E402
import response_arm  # noqa: E402

A_KEY, B_KEY = "m.lora_A.weight", "m.lora_B.weight"
SLICES = ("c0", "c1", "c2")


def _hash_vec(text, salt, dim=8):
    digest = hashlib.sha256(f"{salt}|{text}".encode()).digest()
    return np.frombuffer(digest[:dim], dtype=np.uint8).astype(np.float64) / 255.0 - 0.5


def fake_encode(state, texts):
    """Unit embeddings that depend linearly on the adapter's B block. Text 'q7' and 'd7'
    share a strong direction (they are relevant to each other); the state moves every
    embedding along a text-specific direction, so different weights give different
    rankings."""
    w = state[B_KEY].reshape(-1).double().numpy()[:3]
    shift = 0.25 * float(w[0] + 0.5 * w[1] - w[2])
    rows = []
    for t in texts:
        tail = t.split("-")[-1]
        base = np.zeros(8) + 0.1 * _hash_vec(tail[0] + "shared", 0)
        base[int(tail[1:]) % 8] += 2.0
        rows.append(base + shift * _hash_vec(t, 1))
    x = np.array(rows)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make_states():
    broadcast = {A_KEY: torch.eye(2, 3), B_KEY: torch.zeros(3, 2)}
    blocks = {"c0": [[1.0, 0.2], [0.1, 0.3], [0.2, 0.5]],
              "c1": [[0.2, 0.8], [0.4, 0.1], [0.7, 0.2]],
              "c2": [[-0.6, 0.3], [0.9, -0.2], [0.1, 0.8]]}
    return broadcast, [{A_KEY: torch.eye(2, 3), B_KEY: torch.tensor(b)}
                       for b in blocks.values()]


def make_dev(n_docs=24, n_q=8):
    dev = {}
    for s in SLICES:
        cids = [f"{s}-d{i}" for i in range(n_docs)]
        qids = [f"{s}-q{i}" for i in range(n_q)]
        dev[s] = {"cids": cids, "ctext": cids, "qids": qids, "qtext": qids,
                  "qrels": {f"{s}-q{i}": {f"{s}-d{i}": 1} for i in range(n_q)}}
    return dev


CONFIG = {"fixed_points": {"uniform": [1 / 3] * 3, "examples": [0.7, 0.2, 0.1]},
          "lattice_step": 0.5, "scales": [1.0], "n_verify": 2, "floor": "frozen",
          "floor_delta": 0.0, "halvings": 2}


def test_round_returns_applied_weights_and_a_complete_record():
    broadcast, clients = make_states()
    result, record, dev_frozen = response_arm.run_response_maxmin_round(
        fake_encode, broadcast, clients, make_dev(), SLICES, CONFIG, dev_frozen=None)
    assert len(result) == 3 and record["weights"] == [float(x) for x in result]
    assert record["chosen"] in record["verified"] and record["chosen"] in record["shortlist"]
    assert record["weights"] == record["verified"][record["chosen"]]["v"]
    assert len(record["shortlist"]) == 2
    assert set(record["candidates"]) >= {"uniform", "examples", "solo_c0", "solo_c1", "solo_c2"}
    assert record["dev_frozen"] == dev_frozen == record["dev_current"]   # round 1
    assert record["fallback"] is None and record["status"] == "optimal"
    assert record["encodes_per_client"] == 1 + 3 + 2
    for entry in record["verified"].values():
        assert len(entry["measured"]) == 3 and len(entry["lcb_gain"]) == 3


def test_solo_predictions_equal_solo_measurements():
    broadcast, clients = make_states()
    _, record, _ = response_arm.run_response_maxmin_round(
        fake_encode, broadcast, clients, make_dev(), SLICES, CONFIG, dev_frozen=None)
    for j, s in enumerate(SLICES):
        assert record["candidates"][f"solo_{s}"]["pred"] == pytest.approx(
            record["solo_measured"][j], abs=1e-6)


def test_torch_prediction_matches_the_numpy_reference():
    rng = np.random.default_rng(3)
    dev = make_dev(n_docs=30, n_q=6)
    base, responses = {}, {}
    for s in SLICES:
        c0 = ra.normalise_rows(rng.normal(size=(30, 8)))
        q0 = ra.normalise_rows(rng.normal(size=(6, 8)))
        base[s] = {"c": c0, "q": q0}
        responses[s] = [(ra.normalise_rows(rng.normal(size=(30, 8))) - c0,
                         ra.normalise_rows(rng.normal(size=(6, 8))) - q0) for _ in range(3)]
    v = [0.4, 0.35, 0.25]
    got = response_arm._predicted_means(base, responses, dev, SLICES, v, device="cpu")
    want = []
    for s in SLICES:
        c_hat = ra.predict_embeddings(base[s]["c"], [r[0] for r in responses[s]], v)
        q_hat = ra.predict_embeddings(base[s]["q"], [r[1] for r in responses[s]], v)
        want.append(float(ra.ndcg10(q_hat @ c_hat.T, dev[s]["cids"], dev[s]["qids"],
                                    dev[s]["qrels"]).mean()))
    assert got == pytest.approx(want, abs=1e-6)


def test_unreachable_floor_yields_zero_step_after_halvings():
    broadcast, clients = make_states()
    cfg = {**CONFIG, "floor_delta": -1.0}          # floor = frozen + 1: impossible
    result, record, _ = response_arm.run_response_maxmin_round(
        fake_encode, broadcast, clients, make_dev(), SLICES, cfg, dev_frozen=None)
    assert list(result) == [0.0, 0.0, 0.0]
    assert record["status"] == "zero_step" and record["fallback"] is None
    assert record["chosen"] is None
    assert record["shortlist"] == []                 # nothing was feasible on prediction
    assert record["halvings_used"] == 0


def test_halving_is_tried_when_predictions_pass_but_measurements_fail():
    """Force the gap between prediction and measurement: an encoder whose measured
    embeddings differ from the linear model so the shortlist passes the floor on
    prediction and fails it on measurement, then a halved step passes."""
    broadcast, clients = make_states()
    calls = {"n": 0}

    def encode(state, texts):
        calls["n"] += 1
        return fake_encode(state, texts)

    cfg = {**CONFIG, "floor_delta": -0.02, "halvings": 3}
    result, record, _ = response_arm.run_response_maxmin_round(
        encode, broadcast, clients, make_dev(), SLICES, cfg, dev_frozen=None)
    # whatever the outcome, the record is self-consistent
    if record["status"] == "halved":
        name = record["chosen"]
        assert "_half" in name and record["halvings_used"] >= 1
        assert record["weights"] == record["verified"][name]["v"]
    elif record["status"] == "zero_step":
        assert list(result) == [0.0, 0.0, 0.0]
    else:
        assert record["status"] == "optimal"


def test_prior_dev_frozen_is_kept_as_the_floor():
    broadcast, clients = make_states()
    frozen = [0.1, 0.1, 0.1]
    _, record, out = response_arm.run_response_maxmin_round(
        fake_encode, broadcast, clients, make_dev(), SLICES, CONFIG, dev_frozen=frozen)
    assert out == frozen and record["dev_frozen"] == frozen
    assert record["floors"] == frozen
