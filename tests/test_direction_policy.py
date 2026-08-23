"""The FedSpan direction game, pinned against independent oracles.

Every expected value here comes from ``reference_solvers`` (exact enumeration,
no production code) or from a closed form written out by hand. Nothing asserts
that the implementation agrees with itself.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregation_schemes import (  # noqa: E402
    apply_fedspan_update,
    fedspan_delta_weights,
)
from fedspan_fixtures import (  # noqa: E402
    effective_cosine_gram,
    federation_from_unit_directions,
    federation_with_cosine_gram,
)
from reference_solvers import (  # noqa: E402
    achieved_worst_case_cosine,
    maximin_reference,
    min_norm_reference,
)

POLICIES = ("minnorm", "maxmin-lp")

# The three-client exhibit behind supervisor decision D1: the max-min LP
# attains a worst-case cosine of 0.4116 where 0.5484 is attainable. The two
# off-diagonals were solved for so both figures are exact; the test below
# re-derives them from the Gram with the independent oracles before asserting
# anything about the implementation, so the fixture cannot drift unnoticed.
EXHIBIT_GRAM = np.array([
    [1.0, -0.39851488, 0.65],
    [-0.39851488, 1.0, -0.047965849427836278],
    [0.65, -0.047965849427836278, 1.0],
])
EXHIBIT_LP_ACHIEVED = 0.4116
EXHIBIT_OPTIMUM = 0.5484
EXHIBIT_SHORTFALL = 0.1368


def active_weights(result):
    return np.asarray(
        [result["simplex_weights"][index]
         for index in result["active_indices"]],
        dtype=np.float64)


def recorded_gram(result):
    return np.asarray(result["cosine_gram_active"], dtype=np.float64)


def random_unit_directions(rng, count, dimension):
    directions = rng.normal(size=(count, dimension))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


# ------------------------------------------------ the solve, against oracles


@pytest.mark.parametrize("policy", POLICIES)
def test_recomputed_gamma_equals_the_solvers_own_objective(policy):
    """gamma, the mixture norm and the achieved cosine must describe one w."""
    rng = np.random.default_rng(3)
    for trial in range(12):
        count = int(rng.integers(2, 6))
        directions = random_unit_directions(rng, count, count + 2)
        radii = 10.0 ** rng.uniform(-1.5, 1.5, size=count)
        broadcast, clients, scales = federation_from_unit_directions(
            directions, radii, seed=100 + trial)

        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.37,
            direction_policy=policy)
        assert result["status"] in ("optimal", "singleton")

        gram = recorded_gram(result)
        w = active_weights(result)
        payoffs = gram @ w
        assert result["gamma"] == pytest.approx(float(np.min(payoffs)),
                                                abs=1e-9)
        assert result["solver_objective_gamma"] == pytest.approx(
            result["gamma"], abs=1e-9)
        assert result["mixture_norm"] == pytest.approx(
            math.sqrt(float(w @ gram @ w)), abs=1e-9)
        assert result["achieved_min_direction_cosine"] == pytest.approx(
            float(np.min(payoffs)) / math.sqrt(float(w @ gram @ w)), abs=1e-9)


@pytest.mark.parametrize("policy", POLICIES)
def test_solved_weights_beat_uniform_on_the_returned_gram(policy):
    """Each policy must not be beaten by uniform on its own objective."""
    rng = np.random.default_rng(5)
    improvements = []
    federations = [
        federation_with_cosine_gram(EXHIBIT_GRAM, [0.7, 1.0, 3.1], seed=71)]
    for trial in range(12):
        count = int(rng.integers(3, 7))
        directions = random_unit_directions(rng, count, count + 3)
        radii = 10.0 ** rng.uniform(-1.0, 1.0, size=count)
        federations.append(federation_from_unit_directions(
            directions, radii, seed=200 + trial))
    for broadcast, clients, scales in federations:
        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.2,
            direction_policy=policy)
        gram = recorded_gram(result)
        w = active_weights(result)
        uniform = np.ones(len(w), dtype=np.float64) / len(w)

        if policy == "minnorm":
            assert float(w @ gram @ w) <= float(uniform @ gram @ uniform) + 1e-9
            improvements.append(
                float(uniform @ gram @ uniform) - float(w @ gram @ w))
            # min-norm maximizes the applied quantity exactly, so it can never
            # be beaten by uniform on it either.
            assert achieved_worst_case_cosine(gram, w) >= (
                achieved_worst_case_cosine(gram, uniform) - 1e-9)
        else:
            assert (float(np.min(gram @ w))
                    >= float(np.min(gram @ uniform)) - 1e-9)
            improvements.append(
                float(np.min(gram @ w)) - float(np.min(gram @ uniform)))
    # A solver that simply returned uniform would satisfy the inequalities
    # above vacuously; at least one geometry must be strictly improved. The
    # exhibit alone clears this by 0.116 (LP) and 0.078 (min-norm).
    assert max(improvements) > 0.05


def test_maxmin_lp_can_lose_to_uniform_on_the_quantity_it_applies():
    """Why D1 requires per-round shortfall logging rather than an assumption.

    The LP maximizes min_i (C w)_i, but the applied direction is normalized,
    so its achieved worst-case cosine can fall below what plain uniform
    weights attain. Pinning one such geometry keeps that failure mode from
    being quietly reintroduced as an optimality claim.
    """
    rng = np.random.default_rng(5)
    losses = []
    for trial in range(12):
        count = int(rng.integers(3, 7))
        directions = random_unit_directions(rng, count, count + 3)
        radii = 10.0 ** rng.uniform(-1.0, 1.0, size=count)
        broadcast, clients, scales = federation_from_unit_directions(
            directions, radii, seed=200 + trial)
        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.2,
            direction_policy="maxmin-lp")
        gram = recorded_gram(result)
        w = active_weights(result)
        uniform = np.ones(len(w), dtype=np.float64) / len(w)
        losses.append(achieved_worst_case_cosine(gram, uniform)
                      - achieved_worst_case_cosine(gram, w))
        # The min-norm optimum is never below either of them.
        optimum, _ = min_norm_reference(gram)
        assert math.sqrt(optimum) >= achieved_worst_case_cosine(gram, uniform) - 1e-9
    assert max(losses) > 1e-3


@pytest.mark.parametrize("policy", POLICIES)
def test_solver_matches_exact_enumeration_on_random_geometries(policy):
    """The returned objective must equal the exactly enumerated optimum."""
    rng = np.random.default_rng(17)
    for trial in range(15):
        count = int(rng.integers(2, 6))
        directions = random_unit_directions(rng, count, count + 1)
        radii = 10.0 ** rng.uniform(-1.0, 1.0, size=count)
        broadcast, clients, scales = federation_from_unit_directions(
            directions, radii, seed=300 + trial)

        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.11,
            direction_policy=policy)
        if result["fallback"] is not None:
            continue
        gram = recorded_gram(result)
        w = active_weights(result)

        optimum, _ = min_norm_reference(gram)
        assert result["min_norm_value"] == pytest.approx(
            math.sqrt(optimum), abs=1e-9)
        if policy == "minnorm":
            assert float(w @ gram @ w) == pytest.approx(optimum, abs=1e-9)
        else:
            reference_t, _ = maximin_reference(gram)
            assert float(np.min(gram @ w)) == pytest.approx(
                reference_t, abs=1e-9)


def test_achieved_cosine_never_exceeds_the_attainable_optimum():
    """min_i (C w)_i / sqrt(w^T C w) <= sqrt(min_w w^T C w), for both policies."""
    rng = np.random.default_rng(29)
    for trial in range(15):
        count = int(rng.integers(2, 7))
        directions = random_unit_directions(rng, count, count + 1)
        radii = 10.0 ** rng.uniform(-1.0, 1.0, size=count)
        broadcast, clients, scales = federation_from_unit_directions(
            directions, radii, seed=400 + trial)
        for policy in POLICIES:
            result = fedspan_delta_weights(
                clients, broadcast, module_scales=scales, step_norm=0.5,
                direction_policy=policy)
            if result["fallback"] is not None:
                continue
            optimum, _ = min_norm_reference(recorded_gram(result))
            achieved = result["achieved_min_direction_cosine"]
            assert achieved <= math.sqrt(optimum) + 1e-9
            assert result["direction_solver_shortfall"] == pytest.approx(
                result["min_norm_value"] - achieved, abs=1e-12)
            assert result["direction_solver_shortfall"] >= -1e-9


# --------------------------------------------------------------- the exhibit


def test_exhibit_gram_reference_values_are_the_published_d1_figures():
    """The fixture itself is checked before it is used to judge the solver."""
    reference_t, lp_w = maximin_reference(EXHIBIT_GRAM)
    optimum, min_norm_w = min_norm_reference(EXHIBIT_GRAM)

    assert achieved_worst_case_cosine(EXHIBIT_GRAM, lp_w) == pytest.approx(
        EXHIBIT_LP_ACHIEVED, abs=1e-9)
    assert math.sqrt(optimum) == pytest.approx(EXHIBIT_OPTIMUM, abs=1e-9)
    assert reference_t == pytest.approx(
        float(np.min(EXHIBIT_GRAM @ lp_w)), abs=1e-12)
    # The two policies genuinely disagree here: the LP puts no weight on the
    # first client, the min-norm point splits between the first two.
    assert min_norm_w == pytest.approx([0.5, 0.5, 0.0], abs=1e-9)
    assert lp_w[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("policy", "expected_achieved", "expected_shortfall"),
    [("maxmin-lp", EXHIBIT_LP_ACHIEVED, EXHIBIT_SHORTFALL),
     ("minnorm", EXHIBIT_OPTIMUM, 0.0)],
)
def test_exhibit_pins_both_direction_policies(
        policy, expected_achieved, expected_shortfall):
    broadcast, clients, scales = federation_with_cosine_gram(
        EXHIBIT_GRAM, radii=[0.7, 1.0, 3.1], seed=71)

    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.3,
        direction_policy=policy)

    assert result["status"] == "optimal"
    assert result["direction_policy"] == policy
    assert result["direction_policy_specified"] is True
    assert result["achieved_min_direction_cosine"] == pytest.approx(
        expected_achieved, abs=1e-6)
    assert result["min_norm_value"] == pytest.approx(
        EXHIBIT_OPTIMUM, abs=1e-6)
    assert result["direction_solver_shortfall"] == pytest.approx(
        expected_shortfall, abs=1e-6)


def test_exhibit_shortfall_is_visible_in_what_is_actually_applied():
    """The logged cosine must describe the applied step, not solver internals."""
    broadcast, clients, scales = federation_with_cosine_gram(
        EXHIBIT_GRAM, radii=[0.7, 1.0, 3.1], seed=71)
    applied_cosines = {}
    for policy in POLICIES:
        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.3,
            direction_policy=policy)
        _, diagnostic = apply_fedspan_update(
            broadcast, clients, result, module_scales=scales)
        assert diagnostic["applied_min_active_cosine"] == pytest.approx(
            result["achieved_min_direction_cosine"], abs=1e-8)
        applied_cosines[policy] = diagnostic["applied_min_active_cosine"]
    assert applied_cosines["minnorm"] - applied_cosines["maxmin-lp"] == (
        pytest.approx(EXHIBIT_SHORTFALL, abs=1e-6))


# ------------------------------------------------- hand-computed closed forms


def test_orthogonal_clients_have_a_hand_computed_closed_form():
    """K mutually orthogonal unit directions: C = I.

    By hand, for C = I: w^T C w = sum w_k^2 is minimized on the simplex at
    w = 1/K with value 1/K, and min_i (C w)_i = min_i w_i is maximized at the
    same point with value 1/K. So both policies return uniform weights,
    gamma = 1/K, the mixture norm is 1/sqrt(K), and the achieved worst-case
    cosine is (1/K)/(1/sqrt(K)) = 1/sqrt(K). The raw-B coefficient is then
    c_k = s w_k / (r_k * mixture_norm) = s / (sqrt(K) r_k).
    """
    count = 3
    radii = [1.0, 2.0, 4.0]
    step = 0.25
    broadcast, clients, scales = federation_with_cosine_gram(
        np.eye(count), radii, seed=41)

    for policy in POLICIES:
        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=step,
            direction_policy=policy)
        assert result["simplex_weights"] == pytest.approx(
            [1.0 / count] * count, abs=1e-9)
        assert result["gamma"] == pytest.approx(1.0 / count, abs=1e-9)
        assert result["mixture_norm"] == pytest.approx(
            1.0 / math.sqrt(count), abs=1e-9)
        assert result["achieved_min_direction_cosine"] == pytest.approx(
            1.0 / math.sqrt(count), abs=1e-9)
        assert result["min_norm_value"] == pytest.approx(
            1.0 / math.sqrt(count), abs=1e-9)
        assert result["client_norms"] == pytest.approx(radii, rel=1e-12)
        assert result["delta_weights"] == pytest.approx(
            [step / (math.sqrt(count) * radius) for radius in radii],
            rel=1e-9)

        # The applied B is stored as float32, so the applied norm agrees with
        # the closed form only to float32 resolution.
        _, diagnostic = apply_fedspan_update(
            broadcast, clients, result, module_scales=scales)
        assert diagnostic["applied_step_norm"] == pytest.approx(step, rel=1e-6)


@pytest.mark.parametrize("cosine", [-0.4, 0.0, 0.3, 0.85])
def test_two_client_pair_has_a_hand_computed_closed_form(cosine):
    """C = [[1, c], [c, 1]].

    By hand: both objectives are symmetric under swapping the two clients and
    the simplex is one-dimensional, so w = (1/2, 1/2). Then
    (C w)_i = (1 + c)/2 for both i, so gamma = (1 + c)/2; the mixture norm is
    sqrt(w^T C w) = sqrt((1 + c)/2); and the achieved worst-case cosine is
    gamma / mixture_norm = sqrt((1 + c)/2), which also equals the attainable
    optimum, so the shortfall is exactly zero for both policies.
    """
    radii = [0.5, 3.0]
    step = 0.8
    gram = np.array([[1.0, cosine], [cosine, 1.0]])
    broadcast, clients, scales = federation_with_cosine_gram(
        gram, radii, seed=53)
    expected_cosine = math.sqrt((1.0 + cosine) / 2.0)

    for policy in POLICIES:
        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=step,
            direction_policy=policy)
        assert result["simplex_weights"] == pytest.approx([0.5, 0.5], abs=1e-9)
        assert result["gamma"] == pytest.approx((1.0 + cosine) / 2.0, abs=1e-9)
        assert result["mixture_norm"] == pytest.approx(
            expected_cosine, abs=1e-9)
        assert result["achieved_min_direction_cosine"] == pytest.approx(
            expected_cosine, abs=1e-9)
        assert result["min_norm_value"] == pytest.approx(
            expected_cosine, abs=1e-9)
        assert result["direction_solver_shortfall"] == pytest.approx(
            0.0, abs=1e-9)
        assert result["delta_weights"] == pytest.approx(
            [step * 0.5 / (radius * expected_cosine) for radius in radii],
            rel=1e-9)


# --------------------------------------------- solver diagnostics, recomputed


@pytest.mark.parametrize("policy", POLICIES)
def test_solver_diagnostics_are_recomputable_not_merely_small(policy):
    """Every reported solver number is re-derived from the recorded solve."""
    rng = np.random.default_rng(61)
    for trial in range(10):
        count = int(rng.integers(3, 6))
        directions = random_unit_directions(rng, count, count + 2)
        radii = 10.0 ** rng.uniform(-1.0, 1.0, size=count)
        broadcast, clients, scales = federation_from_unit_directions(
            directions, radii, seed=500 + trial)

        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.19,
            direction_policy=policy)
        if result["fallback"] is not None:
            continue
        gram = recorded_gram(result)
        w = active_weights(result)

        assert result["solver_simplex_residual"] == pytest.approx(
            abs(float(w.sum()) - 1.0), abs=1e-12)
        expected_violation = max(
            0.0, float(np.max(result["solver_objective_gamma"] - gram @ w)))
        assert result["solver_constraint_violation"] == pytest.approx(
            expected_violation, abs=1e-12)

        # min_norm_solver claims convergence of a specific quantity; recompute
        # that quantity exactly rather than trusting the reported gap.
        optimum, optimal_w = min_norm_reference(gram)
        assert result["min_norm_value"] == pytest.approx(
            math.sqrt(optimum), abs=1e-9)
        solver = result["min_norm_solver"]
        assert solver["converged"] is True
        assert solver["gap"] <= solver["tol"]
        assert solver["iterations"] >= 1
        reference_gap = (float(optimal_w @ gram @ optimal_w)
                         - float(np.min(gram @ optimal_w)))
        assert reference_gap == pytest.approx(0.0, abs=1e-9)

        # At the min-norm point the KKT conditions force min_i (C w)_i to
        # equal w^T C w, so gamma must equal the squared optimum there.
        if policy == "minnorm":
            assert result["gamma"] == pytest.approx(optimum, abs=1e-9)
            assert result["gamma"] == pytest.approx(
                result["min_norm_value"] ** 2, abs=1e-9)


def test_recorded_gram_matches_an_independent_dense_materialization():
    """The Gram the solver reports must be the geometry of the real updates."""
    broadcast, clients, scales = federation_with_cosine_gram(
        EXHIBIT_GRAM, radii=[0.7, 1.0, 3.1], seed=71)
    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.3,
        direction_policy="minnorm")

    dense_gram, dense_norms = effective_cosine_gram(clients, broadcast, scales)
    assert recorded_gram(result) == pytest.approx(dense_gram, abs=1e-9)
    assert recorded_gram(result) == pytest.approx(EXHIBIT_GRAM, abs=1e-9)
    assert result["client_norms"] == pytest.approx(dense_norms, rel=1e-9)


def test_unspecified_direction_policy_is_recorded_as_implicit():
    """An omitted policy must stay machine-detectable, not silently default."""
    broadcast, clients, scales = federation_with_cosine_gram(
        EXHIBIT_GRAM, radii=[0.7, 1.0, 3.1], seed=71)
    implicit = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.3)
    explicit = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_norm=0.3,
        direction_policy="maxmin-lp")

    assert implicit["direction_policy_specified"] is False
    assert explicit["direction_policy_specified"] is True
    assert implicit["direction_policy"] == "maxmin-lp"
    assert implicit["delta_weights"] == pytest.approx(
        explicit["delta_weights"], rel=1e-12)


def test_unknown_direction_policy_is_rejected():
    broadcast, clients, scales = federation_with_cosine_gram(
        np.eye(2), radii=[1.0, 1.0], seed=13)
    with pytest.raises(ValueError, match="direction_policy"):
        fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.1,
            direction_policy="lp")


def test_opposed_clients_fail_closed_under_both_policies():
    """Exact cancellation must produce a zero update, not a normalized NaN."""
    directions = np.array([[1.0, 0.0], [-1.0, 0.0]])
    broadcast, clients, scales = federation_from_unit_directions(
        directions, [1.0, 1.0], seed=83)
    for policy in POLICIES:
        result = fedspan_delta_weights(
            clients, broadcast, module_scales=scales, step_norm=0.2,
            direction_policy=policy)
        applied, diagnostic = apply_fedspan_update(
            broadcast, clients, result, module_scales=scales)
        assert result["status"] == "near_cancellation"
        assert result["fallback"] == "zero_update"
        assert result["delta_weights"] == [0.0, 0.0]
        assert result["min_norm_value"] == pytest.approx(0.0, abs=1e-9)
        assert diagnostic["applied_step_norm"] == 0.0
        for key in broadcast:
            assert torch.equal(applied[key], broadcast[key])
