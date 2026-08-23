"""Unit tests for aggregation_schemes.py — the E2 external-baseline weighting
schemes (q-FedAvg, AFL, MGDA, FedNova) plus the shared Gram/delta machinery.

Pure CPU; no sentence-transformers dependency. Fake LoRA states use the same
key format federated_forgetting.py produces (``<module>.lora_{A,B}.weight``).
"""
import math
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregation_schemes import (  # noqa: E402
    FedSpanContractError,
    afl_update,
    apply_delta_weights,
    apply_fedspan_update,
    apply_frozen_b_delta_weights,
    configure_frozen_lora_a,
    fednova_delta_weights,
    fedspan_delta_weights,
    frozen_a_state_diagnostics,
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


def frozen_a_federation(rng, client_deltas, module_scales=None):
    """Shared row-orthonormal A; clients differ only in raw PEFT B."""
    module_scales = module_scales or {MODULES[0]: 0.25, MODULES[1]: 2.75}
    broadcast = {}
    for m in MODULES:
        q, _ = np.linalg.qr(rng.normal(size=(D_IN, RANK)))
        broadcast[f"{m}.lora_A.weight"] = torch.tensor(
            q[:, :RANK].T, dtype=torch.float32)
        broadcast[f"{m}.lora_B.weight"] = torch.tensor(
            rng.normal(size=(D_OUT, RANK)), dtype=torch.float32)
    clients = []
    for blocks in client_deltas:
        st = {k: v.clone() for k, v in broadcast.items()}
        for m, delta in zip(MODULES, blocks):
            st[f"{m}.lora_B.weight"] += torch.tensor(
                delta, dtype=torch.float32)
        clients.append(st)
    return broadcast, clients, module_scales


def effective_b_delta(state, broadcast, module_scales):
    return torch.cat([
        module_scales[m]
        * (state[f"{m}.lora_B.weight"]
           - broadcast[f"{m}.lora_B.weight"]).double().reshape(-1)
        for m in MODULES
    ])


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


def test_update_gram_honors_unequal_peft_module_scales(rng):
    deltas = [[rng.normal(size=(D_OUT, RANK)) for _ in MODULES]
              for _ in range(3)]
    broadcast, clients, scales = frozen_a_federation(rng, deltas)
    gram = update_gram(
        clients, broadcast, normalize=False, module_scales=scales)
    effective_dense = []
    for state in clients:
        effective_dense.append({
            m: scales[m]
            * ((state[f"{m}.lora_B.weight"]
                - broadcast[f"{m}.lora_B.weight"])
               @ broadcast[f"{m}.lora_A.weight"])
            for m in MODULES
        })
    for i in range(3):
        for j in range(3):
            assert gram[i, j] == pytest.approx(
                dense_ip(effective_dense[i], effective_dense[j]),
                rel=2e-5, abs=2e-5)


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


# --------------------------------------------------- norm-consistent FedSpan

def test_fedspan_exact_true_step_with_unequal_norms_and_module_scales(rng):
    deltas = [
        [rng.normal(size=(D_OUT, RANK)) * 0.02,
         rng.normal(size=(D_OUT, RANK)) * 3.0],
        [rng.normal(size=(D_OUT, RANK)) * 4.0,
         rng.normal(size=(D_OUT, RANK)) * 0.03],
        [rng.normal(size=(D_OUT, RANK)) * 0.4,
         rng.normal(size=(D_OUT, RANK)) * 0.6],
    ]
    broadcast, clients, scales = frozen_a_federation(rng, deltas)
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.413)
    applied, diag = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)

    assert result["status"] == "optimal"
    assert len(result["delta_weights"]) == len(clients)
    assert sum(result["simplex_weights"]) == pytest.approx(1.0, abs=1e-8)
    assert sum(result["delta_weights"]) != pytest.approx(1.0, abs=1e-3)
    actual = effective_b_delta(applied, broadcast, scales)
    unit_dirs = [effective_b_delta(st, broadcast, scales)
                 / torch.linalg.vector_norm(
                     effective_b_delta(st, broadcast, scales))
                 for st in clients]
    solved = sum(w * u for w, u in zip(result["simplex_weights"], unit_dirs))
    expected = 0.413 * solved / torch.linalg.vector_norm(solved)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.linalg.vector_norm(actual).item() == pytest.approx(
        0.413, abs=2e-6)
    assert diag["applied_step_norm"] == pytest.approx(0.413, abs=2e-6)
    assert diag["max_effective_block_error"] < 2e-6
    assert result["solver_simplex_residual"] < 1e-7
    assert result["solver_constraint_violation"] < 1e-7
    assert result["proposed_delta_weights"] == result["delta_weights"]
    assert diag["applied_delta_weights"] == result["delta_weights"]
    assert diag["applied_min_active_cosine"] is not None
    assert diag["applied_min_active_cosine"] == pytest.approx(
        result["certified_min_direction_cosine"], abs=2e-6)
    json.dumps({"solve": result, "application": diag})
    assert len(diag["broadcast_state_sha256"]) == 64
    assert len(diag["applied_state_sha256"]) == 64
    for m in MODULES:
        key = f"{m}.lora_A.weight"
        assert torch.equal(applied[key], broadcast[key])


def test_fedspan_zero_and_tiny_clients_are_inactive(rng):
    z = np.zeros((D_OUT, RANK))
    tiny = np.full((D_OUT, RANK), 1e-15)
    broadcast, clients, scales = frozen_a_federation(
        rng, [[z, z], [tiny, tiny]])
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.2,
        active_abs_tol=1e-12, active_rel_tol=1e-8)
    applied, diag = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)
    assert result["status"] == "no_active"
    assert result["active_mask"] == [False, False]
    assert result["delta_weights"] == [0.0, 0.0]
    assert diag["applied_step_norm"] == 0.0
    for key in broadcast:
        assert torch.equal(applied[key], broadcast[key])


def test_fedspan_nonfinite_client_excluded_and_singleton_is_exact(rng):
    good = [np.ones((D_OUT, RANK)), np.ones((D_OUT, RANK)) * 0.5]
    bad = [np.ones((D_OUT, RANK)), np.ones((D_OUT, RANK))]
    bad[0][0, 0] = np.nan
    broadcast, clients, scales = frozen_a_federation(rng, [good, bad])
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.17)
    applied, diag = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)
    assert result["status"] == "singleton"
    assert result["active_mask"] == [True, False]
    assert result["inactive_reasons"][1] == "nonfinite_delta"
    assert result["simplex_weights"] == [1.0, 0.0]
    assert diag["applied_step_norm"] == pytest.approx(0.17, abs=2e-6)


def test_fedspan_opposite_directions_fail_closed_on_cancellation(rng):
    d0 = np.ones((D_OUT, RANK))
    d1 = np.zeros((D_OUT, RANK))
    broadcast, clients, scales = frozen_a_federation(
        rng, [[d0, d1], [-d0, -d1]],
        module_scales={MODULES[0]: 1.0, MODULES[1]: 1.0})
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.2)
    applied, diag = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)
    assert result["status"] == "near_cancellation"
    assert result["fallback"] == "zero_update"
    assert result["delta_weights"] == [0.0, 0.0]
    assert diag["applied_step_norm"] == 0.0
    for key in broadcast:
        assert torch.equal(applied[key], broadcast[key])


def test_fedspan_solver_error_fails_closed(monkeypatch, rng):
    deltas = [[rng.normal(size=(D_OUT, RANK)) for _ in MODULES]
              for _ in range(3)]
    broadcast, clients, scales = frozen_a_federation(rng, deltas)

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr("scipy.optimize.linprog", explode)
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.2)
    assert result["status"] == "solver_error"
    assert result["fallback"] == "zero_update"
    assert result["delta_weights"] == [0.0] * 3
    assert "synthetic solver failure" in result["solver_message"]


def test_fedspan_declared_coefficient_limit_fails_closed(rng):
    deltas = [[rng.normal(size=(D_OUT, RANK)) for _ in MODULES]
              for _ in range(3)]
    broadcast, clients, scales = frozen_a_federation(rng, deltas)
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.2,
        max_abs_delta_weight=1e-12)
    applied, diag = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)
    assert result["status"] == "coefficient_limit"
    assert result["fallback"] == "zero_update"
    assert result["delta_weights"] == [0.0] * 3
    assert any(abs(value) > 0 for value in result["proposed_delta_weights"])
    assert result["proposed_max_abs_delta_weight"] > 1e-12
    assert result["delta_weight_limit"] == 1e-12
    assert diag["applied_step_norm"] == 0.0
    for key in broadcast:
        assert torch.equal(applied[key], broadcast[key])


def test_fedspan_rejects_non_shared_a_and_missing_factor(rng):
    deltas = [[rng.normal(size=(D_OUT, RANK)) for _ in MODULES]
              for _ in range(2)]
    broadcast, clients, scales = frozen_a_federation(rng, deltas)
    clients[0][f"{MODULES[0]}.lora_A.weight"][0, 0] += 1e-4
    with pytest.raises(FedSpanContractError, match="shared frozen A"):
        fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.2)

    broadcast, clients, scales = frozen_a_federation(rng, deltas)
    del clients[1][f"{MODULES[1]}.lora_B.weight"]
    with pytest.raises(FedSpanContractError, match="factor keys"):
        fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.2)


@pytest.mark.parametrize("bad_step", [0.0, -1.0, float("nan"), float("inf")])
def test_fedspan_requires_positive_finite_declared_step(rng, bad_step):
    deltas = [[rng.normal(size=(D_OUT, RANK)) for _ in MODULES]
              for _ in range(2)]
    broadcast, clients, scales = frozen_a_federation(rng, deltas)
    with pytest.raises(ValueError, match="step_norm"):
        fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=bad_step)


def test_configure_frozen_lora_a_is_orthonormal_and_freezes_only_a():
    class ToyLoraLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict({
                "default": nn.Linear(D_IN, RANK, bias=False)})
            self.lora_B = nn.ModuleDict({
                "default": nn.Linear(RANK, D_OUT, bias=False)})
            self.scaling = {"default": 2.75}

    model = nn.Module()
    model.block = ToyLoraLayer()
    with torch.no_grad():
        model.block.lora_B["default"].weight.zero_()
    scales = configure_frozen_lora_a(model, adapter_name="default")
    A = model.block.lora_A["default"].weight
    B = model.block.lora_B["default"].weight
    assert torch.allclose(A @ A.T, torch.eye(RANK), atol=1e-6)
    assert A.requires_grad is False
    assert B.requires_grad is True
    assert scales == {"block": 2.75}
    diagnostic = frozen_a_state_diagnostics({
        "block.lora_A.weight": A.detach().clone(),
        "block.lora_B.weight": B.detach().clone(),
    }, scales)
    assert diagnostic["block"]["a_row_orthonormal_max_abs_error"] < 1e-6


def test_configure_frozen_lora_a_refuses_nonzero_initial_b():
    class ToyLoraLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict({
                "default": nn.Linear(D_IN, RANK, bias=False)})
            self.lora_B = nn.ModuleDict({
                "default": nn.Linear(RANK, D_OUT, bias=False)})
            self.scaling = {"default": 2.0}

    model = nn.Module()
    model.block = ToyLoraLayer()
    with pytest.raises(FedSpanContractError, match="nonzero LoRA B"):
        configure_frozen_lora_a(model)


def test_fedspan_contract_with_real_peft_state_keys():
    from peft import (LoraConfig, TaskType, get_peft_model,
                      get_peft_model_state_dict)
    from transformers import BertConfig, BertModel

    base = BertModel(BertConfig(
        hidden_size=12, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=3, vocab_size=32))
    peft_model = get_peft_model(base, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=2, lora_alpha=6,
        lora_dropout=0.0, target_modules=["query", "value"]))
    scales = configure_frozen_lora_a(peft_model)
    with pytest.warns(UserWarning, match="Could not find a config file"):
        peft_state = get_peft_model_state_dict(peft_model)
    broadcast = {k: v.detach().cpu().clone()
                 for k, v in peft_state.items()}
    clients = []
    for multiplier in (0.25, 1.75):
        state = {k: v.clone() for k, v in broadcast.items()}
        for key in state:
            if ".lora_B.weight" in key:
                state[key] += multiplier
        clients.append(state)
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.09)
    applied, diag = apply_fedspan_update(
        broadcast, clients, result, module_scales=scales)
    assert result["status"] in ("optimal", "singleton")
    assert diag["applied_step_norm"] == pytest.approx(0.09, abs=2e-6)
    for key in broadcast:
        if ".lora_A.weight" in key:
            assert torch.equal(applied[key], broadcast[key])


def test_fedspan_randomized_dense_oracle_scale_and_permutation_invariance():
    """Stress the exact dense step under varied norms/scales/orderings."""
    for seed in range(40):
        local_rng = np.random.default_rng(1000 + seed)
        client_deltas = []
        for _ in range(4):
            magnitude = 10.0 ** local_rng.uniform(-4.0, 4.0)
            client_deltas.append([
                magnitude * local_rng.normal(size=(D_OUT, RANK))
                for _ in MODULES
            ])
        scales = {m: 10.0 ** local_rng.uniform(-1.0, 1.0)
                  for m in MODULES}
        broadcast, clients, scales = frozen_a_federation(
            local_rng, client_deltas, scales)
        step = 10.0 ** local_rng.uniform(-3.0, 0.0)

        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=step,
            active_rel_tol=0.0)
        assert result["status"] == "optimal"
        applied, _ = apply_fedspan_update(
            broadcast, clients, result, module_scales=scales)

        dense_blocks = []
        for m in MODULES:
            dense_blocks.append(
                scales[m]
                * (applied[f"{m}.lora_B.weight"]
                   - broadcast[f"{m}.lora_B.weight"])
                @ broadcast[f"{m}.lora_A.weight"])
        dense_norm = math.sqrt(sum(torch.sum(block.double() ** 2).item()
                                   for block in dense_blocks))
        assert dense_norm == pytest.approx(step, rel=3e-5, abs=3e-6)

        permutation = local_rng.permutation(len(clients)).tolist()
        permuted = [clients[index] for index in permutation]
        perm_result = fedspan_delta_weights(
            permuted, broadcast, module_scales=scales, step_norm=step,
            active_rel_tol=0.0)
        perm_applied, _ = apply_fedspan_update(
            broadcast, permuted, perm_result, module_scales=scales)
        for m in MODULES:
            key = f"{m}.lora_B.weight"
            assert torch.allclose(
                applied[key], perm_applied[key], atol=5e-5, rtol=5e-5)

        # Positive rescaling of each client's entire update must not change
        # the normalized game direction or the applied true-norm step.
        rescaled = []
        for index, state in enumerate(clients):
            factor = 10.0 ** local_rng.uniform(-0.3, 0.3)
            scaled_state = {key: value.clone()
                            for key, value in broadcast.items()}
            for m in MODULES:
                bkey = f"{m}.lora_B.weight"
                scaled_state[bkey] += factor * (
                    state[bkey] - broadcast[bkey])
            rescaled.append(scaled_state)
        scaled_result = fedspan_delta_weights(
            rescaled, broadcast, module_scales=scales, step_norm=step,
            active_rel_tol=0.0)
        scaled_applied, _ = apply_fedspan_update(
            broadcast, rescaled, scaled_result, module_scales=scales)
        for m in MODULES:
            key = f"{m}.lora_B.weight"
            assert torch.allclose(
                applied[key], scaled_applied[key], atol=7e-5, rtol=7e-5)


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


def test_delta_application_preserves_shared_frozen_a_bitwise(rng):
    deltas = [[rng.normal(size=(D_OUT, RANK)) for _ in MODULES]
              for _ in range(3)]
    broadcast, clients, _ = frozen_a_federation(rng, deltas)
    for m in MODULES:
        key = f"{m}.lora_A.weight"
        broadcast[key] = broadcast[key].half()
        for state in clients:
            state[key] = broadcast[key].clone()
    out = apply_frozen_b_delta_weights(
        broadcast, clients, [1 / 3, 1 / 3, 1 / 3],
        module_scales={m: 1.0 for m in MODULES})
    for m in MODULES:
        key = f"{m}.lora_A.weight"
        assert torch.equal(out[key], broadcast[key])
        assert out[key].dtype == broadcast[key].dtype
