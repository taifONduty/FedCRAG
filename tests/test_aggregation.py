"""Unit tests for aggregation_schemes.py — the E2 external-baseline weighting
schemes (q-FedAvg, AFL, MGDA, FedNova) plus the shared Gram/delta machinery.

Pure CPU; no sentence-transformers dependency. Fake LoRA states use the same
key format federated_forgetting.py produces (``<module>.lora_{A,B}.weight``).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregation_schemes import (  # noqa: E402
    afl_update,
    apply_delta_weights,
    fednova_delta_weights,
    maxmin_weights,
    mgda_weights,
    qffl_delta_weights,
    update_gram,
)

D_OUT, D_IN, RANK = 8, 6, 2
MODULES = ("layer0.attn.q", "layer1.ffn.up")


def fake_state(rng, scale=1.0):
    st = {}
    for m in MODULES:
        st[f"{m}.lora_A.weight"] = torch.tensor(
            rng.normal(size=(RANK, D_IN)) * scale, dtype=torch.float32)
        st[f"{m}.lora_B.weight"] = torch.tensor(
            rng.normal(size=(D_OUT, RANK)) * scale, dtype=torch.float32)
    return st


def dense_update(state, broadcast):
    """Materialized weight-space update sum_m (B A - B_g A_g), per module."""
    upd = {}
    for m in MODULES:
        BA = state[f"{m}.lora_B.weight"] @ state[f"{m}.lora_A.weight"]
        BgAg = (broadcast[f"{m}.lora_B.weight"]
                @ broadcast[f"{m}.lora_A.weight"])
        upd[m] = BA - BgAg
    return upd


def dense_ip(u1, u2):
    return sum(torch.sum(u1[m] * u2[m]).item() for m in MODULES)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def federation(rng):
    broadcast = fake_state(rng)
    clients = [fake_state(rng) for _ in range(4)]
    return broadcast, clients


# ---------------------------------------------------------------- update_gram

def test_update_gram_matches_dense(federation):
    broadcast, clients = federation
    G = update_gram(clients, broadcast)
    dense = [dense_update(c, broadcast) for c in clients]
    scale = float(np.mean(np.diag(G)))          # typical magnitude
    for i in range(4):
        for j in range(4):
            # float32 matmul: absolute tolerance scaled to the Gram magnitude
            # (off-diagonals can near-cancel, so pure rel is ill-posed)
            assert G[i, j] == pytest.approx(dense_ip(dense[i], dense[j]),
                                            rel=1e-3, abs=1e-4 * scale)
            assert G[i, j] == pytest.approx(G[j, i], rel=1e-9)


def test_update_gram_cosine_normalized(federation):
    broadcast, clients = federation
    Gc = update_gram(clients, broadcast, normalize=True)
    assert np.allclose(np.diag(Gc), 1.0, atol=1e-5)
    assert np.all(np.abs(Gc) <= 1.0 + 1e-6)


# ------------------------------------------------------------- maxmin (moved)

def test_maxmin_certificate_beats_uniform(federation):
    broadcast, clients = federation
    w = maxmin_weights(clients, broadcast)
    assert w is not None and len(w) == 4
    assert sum(w) == pytest.approx(1.0, abs=1e-6)
    assert all(x >= -1e-9 for x in w)
    Gc = update_gram(clients, broadcast, normalize=True)
    gamma_star = np.min(Gc @ np.array(w))
    gamma_unif = np.min(Gc @ (np.ones(4) / 4))
    assert gamma_star >= gamma_unif - 1e-7


# ----------------------------------------------------------------------- MGDA

def test_mgda_minnorm_matches_grid_search(rng):
    broadcast = fake_state(rng)
    clients = [fake_state(rng) for _ in range(3)]
    w = np.array(mgda_weights(clients, broadcast))
    assert w.shape == (3,) and w.sum() == pytest.approx(1.0, abs=1e-6)
    G = update_gram(clients, broadcast)
    best = np.inf
    grid = np.arange(0.0, 1.0001, 0.02)
    for a in grid:
        for b in grid:
            if a + b <= 1.0 + 1e-12:
                v = np.array([a, b, 1.0 - a - b])
                best = min(best, float(v @ G @ v))
    val = float(w @ G @ w)
    assert val <= best * (1.0 + 1e-3) + 1e-9


def test_mgda_downweights_dominant_norm_client(rng):
    broadcast = fake_state(rng)
    small = fake_state(rng, scale=0.05)
    huge = {k: broadcast[k] + 40.0 * (small[k] - broadcast[k])
            for k in broadcast}
    other = fake_state(rng, scale=0.05)
    w = mgda_weights([huge, small, other], broadcast)
    # min-norm point leans away from the large-norm update
    assert w[0] < min(w[1], w[2])


# ------------------------------------------------------------------- q-FedAvg

def test_qffl_q0_is_uniform_fedavg():
    v = qffl_delta_weights([0.5, 2.0, 3.0], [1.0, 4.0, 9.0], q=0.0, L=50.0)
    assert v == pytest.approx([1 / 3] * 3, abs=1e-9)


def test_qffl_higher_loss_gets_more_weight():
    v = qffl_delta_weights([0.1, 1.0, 2.5], [1.0, 1.0, 1.0], q=1.0, L=10.0)
    assert v[0] < v[1] < v[2]


def test_qffl_matches_hand_formula():
    losses, d2, q, L = [1.0, 3.0], [0.5, 2.0], 2.0, 10.0
    h = [q * f ** (q - 1) * L ** 2 * d + L * f ** q
         for f, d in zip(losses, d2)]
    expect = [L * f ** q / sum(h) for f in losses]
    v = qffl_delta_weights(losses, d2, q=q, L=L)
    assert v == pytest.approx(expect, rel=1e-9)


# -------------------------------------------------------------------- FedNova

def test_fednova_equal_steps_reduces_to_nk():
    n = [100, 50, 50]
    v = fednova_delta_weights(n, [10, 10, 10])
    assert v == pytest.approx([0.5, 0.25, 0.25], abs=1e-9)
    assert sum(v) == pytest.approx(1.0, abs=1e-9)


def test_fednova_hand_formula():
    n, tau = [80, 20], [8, 2]
    p = [x / 100 for x in n]
    tau_eff = sum(pi * ti for pi, ti in zip(p, tau))
    expect = [tau_eff * pi / ti for pi, ti in zip(p, tau)]
    v = fednova_delta_weights(n, tau)
    assert v == pytest.approx(expect, rel=1e-9)


def test_fednova_zero_step_client_masked():
    v = fednova_delta_weights([100, 50, 10], [10, 5, 0])
    assert v[2] == 0.0
    assert all(x > 0 for x in v[:2])
    assert math.isfinite(sum(v))


# ------------------------------------------------------------------------ AFL

def test_afl_uniform_losses_is_fixed_point():
    lam = [0.25] * 4
    new = afl_update(lam, [1.7] * 4, eta=0.5)
    assert new == pytest.approx(lam, abs=1e-9)


def test_afl_ascends_toward_high_loss_and_is_immutable():
    lam = [0.25, 0.25, 0.25, 0.25]
    lam_before = list(lam)
    new = afl_update(lam, [0.1, 0.1, 0.1, 5.0], eta=0.3)
    assert lam == lam_before            # no in-place mutation
    assert new[3] > 0.25 and all(new[i] < 0.25 for i in range(3))
    assert sum(new) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- delta apply

def test_apply_delta_weights_selects_and_preserves(federation):
    broadcast, clients = federation
    out = apply_delta_weights(broadcast, clients, [1.0, 0.0, 0.0, 0.0])
    for k in broadcast:
        assert torch.allclose(out[k], clients[0][k].float(), atol=1e-6)
    out0 = apply_delta_weights(broadcast, clients, [0.0] * 4)
    for k in broadcast:
        assert torch.allclose(out0[k], broadcast[k].float(), atol=1e-6)


def test_apply_delta_weights_formula_and_no_mutation(federation):
    broadcast, clients = federation
    snap = {k: v.clone() for k, v in broadcast.items()}
    v = [0.3, -0.1, 0.5, 0.2]
    out = apply_delta_weights(broadcast, clients, v)
    for k in broadcast:
        manual = broadcast[k].float().clone()
        for vi, st in zip(v, clients):
            manual += vi * (st[k].float() - broadcast[k].float())
        assert torch.allclose(out[k], manual, atol=1e-5)
        assert torch.equal(broadcast[k], snap[k])   # input untouched
