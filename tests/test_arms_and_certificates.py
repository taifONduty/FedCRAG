"""The exact direction arm, the per-round optimality certificate, and the
fixed-weight arms that make the geometry claim attributable.

FedSpan differs from plain uniform averaging in three ways at once: the 1/r_k
coefficient rule equalises client update norms, the Gram-solved simplex weight
chooses a direction, and the median-active policy sets the step length. A
fixed-weight arm keeps the first and the third and replaces only the second, so
a contrast against it isolates the weight vector alone. These tests pin that
the fixed arms really are step- and norm-matched.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregation_schemes import (  # noqa: E402
    apply_fedspan_update,
    fedspan_delta_weights,
    minnorm_exact_weights,
    wolfe_certificate,
)
from fedspan_fixtures import federation_with_cosine_gram  # noqa: E402

ALL_POLICIES = ("minnorm", "maxmin-lp", "exact", "fixed")

# Three near-clones and a singleton: the geometry E3 is built to create.
CLONE_GRAM = np.array([
    [1.00, 0.75, 0.75, -0.065],
    [0.75, 1.00, 0.75, -0.065],
    [0.75, 0.75, 1.00, -0.065],
    [-0.065, -0.065, -0.065, 1.00],
])
CLONE_RADII = [1.9, 2.0, 2.1, 0.2]
UNIFORM_OVER_CLIENTS = [0.25, 0.25, 0.25, 0.25]
UNIFORM_OVER_DISTRIBUTIONS = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5]


def clone_federation(seed=41):
    return federation_with_cosine_gram(CLONE_GRAM, CLONE_RADII, seed=seed)


def active_weights(result):
    return np.asarray([result["simplex_weights"][index]
                       for index in result["active_indices"]],
                      dtype=np.float64)


def recorded_gram(result):
    return np.asarray(result["cosine_gram_active"], dtype=np.float64)


def solve(policy, fixed_weights=None, step_policy="median-active",
          step_norm=None, clients=None):
    broadcast, states, scales = clients or clone_federation()
    kwargs = {}
    if fixed_weights is not None:
        kwargs["fixed_weights"] = fixed_weights
    return fedspan_delta_weights(
        states, broadcast, module_scales=scales, step_norm=step_norm,
        step_policy=step_policy, direction_policy=policy, **kwargs)


# -------------------------------------------------------- the exact arm


def test_exact_policy_applies_the_exact_face_enumeration_argmin():
    result = solve("exact")
    gram = recorded_gram(result)
    expected_w, expected_value = minnorm_exact_weights(gram)
    assert result["status"] == "optimal"
    assert np.max(np.abs(active_weights(result) - expected_w)) < 1e-12
    assert result["mixture_norm"] == pytest.approx(expected_value, abs=1e-12)


def test_exact_policy_splits_clone_mass_evenly_where_frank_wolfe_need_not():
    """The clone block is symmetric, so its three weights must be equal."""
    w = active_weights(solve("exact"))
    assert abs(float(w[0] - w[1])) < 1e-9
    assert abs(float(w[1] - w[2])) < 1e-9
    assert float(w[:3].sum()) > float(w[3])


def test_exact_policy_is_recorded_in_the_result():
    result = solve("exact")
    assert result["direction_policy"] == "exact"
    assert result["direction_policy_specified"] is True
    assert result["min_norm_value_source"] == "exact-face-enumeration"
    assert result["exact_solver"]["value"] == pytest.approx(
        result["min_norm_value"], abs=1e-15)
    assert result["exact_solver"]["wolfe_certificate"] <= 1e-9


def test_exact_policy_gives_up_nothing_against_the_attainable_optimum():
    result = solve("exact")
    assert result["direction_solver_shortfall"] == pytest.approx(0.0,
                                                                abs=1e-12)


def test_frank_wolfe_policies_do_not_apply_the_exact_solver():
    """The exact optimum is MEASURED every round; it is APPLIED only by the
    exact arm.

    These are different claims and the diagnostics must not let one be read as
    the other -- reading "the exact solver ran" as "the exact solver was
    deployed" is exactly the error the supervisor had to correct on
    2026-08-28. So: the recorded optimum may legitimately come from the face
    enumeration on any arm, while the APPLIED direction on these two arms
    still comes from Frank-Wolfe and the LP respectively.
    """
    for policy, applied in (("minnorm", "away-step-frank-wolfe"),
                            ("maxmin-lp", "scipy-linprog")):
        result = solve(policy)
        assert result["exact_solver"] is None, (
            "exact_solver diagnostics belong to the arm that APPLIES it")
        assert result["applied_direction_solver"] == applied
        # The optimum the arm's shortfall is measured against is exact.
        assert result["min_norm_value_source"] == "exact-face-enumeration"
        assert result["frank_wolfe_value"] is not None, (
            "the iterative solver's own value must stay on the record")


def test_exact_and_minnorm_agree_on_the_optimal_value():
    exact = solve("exact")
    frank_wolfe = solve("minnorm")
    assert exact["min_norm_value"] <= frank_wolfe["min_norm_value"] + 1e-9
    assert exact["mixture_norm"] <= frank_wolfe["mixture_norm"] + 1e-9


# ------------------------------------------------- the Wolfe certificate


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_wolfe_certificate_is_measured_for_every_direction_policy(policy):
    fixed = UNIFORM_OVER_CLIENTS if policy == "fixed" else None
    result = solve(policy, fixed_weights=fixed)
    certificate = result["wolfe_certificate"]
    assert isinstance(certificate, float)
    assert certificate >= 0.0
    assert certificate == pytest.approx(
        wolfe_certificate(recorded_gram(result), active_weights(result)),
        abs=1e-12)


def test_optimal_policies_carry_a_zero_certificate():
    for policy in ("minnorm", "exact"):
        assert solve(policy)["wolfe_certificate"] <= 1e-9


def test_uniform_weights_carry_a_positive_certificate_on_the_clone_gram():
    """Uniform is not the min-norm point here, and the gap is measured."""
    result = solve("fixed", fixed_weights=UNIFORM_OVER_CLIENTS)
    assert result["wolfe_certificate"] > 1e-3


def test_min_norm_value_is_persisted_alongside_the_certificate():
    for policy in ALL_POLICIES:
        fixed = UNIFORM_OVER_CLIENTS if policy == "fixed" else None
        result = solve(policy, fixed_weights=fixed)
        assert isinstance(result["min_norm_value"], float)
        assert result["min_norm_value"] > 0.0
        assert result["direction_solver_shortfall"] is not None


# ------------------------------------------------------ the fixed arms


def test_fixed_policy_applies_the_supplied_weights():
    result = solve("fixed", fixed_weights=UNIFORM_OVER_DISTRIBUTIONS)
    assert result["status"] == "fixed"
    assert np.max(np.abs(active_weights(result)
                         - np.asarray(UNIFORM_OVER_DISTRIBUTIONS))) < 1e-15
    assert result["fixed_weights"] == UNIFORM_OVER_DISTRIBUTIONS


def test_norm_equalised_uniform_is_step_and_norm_matched_to_fedspan():
    """Only the weight vector may differ between the two arms."""
    broadcast, states, scales = clone_federation()
    fedspan = fedspan_delta_weights(
        states, broadcast, module_scales=scales, step_policy="median-active",
        direction_policy="exact")
    uniform = fedspan_delta_weights(
        states, broadcast, module_scales=scales, step_policy="median-active",
        direction_policy="fixed", fixed_weights=UNIFORM_OVER_CLIENTS)

    assert uniform["resolved_step_norm"] == fedspan["resolved_step_norm"]
    assert uniform["client_norms"] == fedspan["client_norms"]
    assert uniform["active_indices"] == fedspan["active_indices"]
    assert uniform["activity_threshold"] == fedspan["activity_threshold"]
    assert np.allclose(recorded_gram(uniform), recorded_gram(fedspan))

    for result in (fedspan, uniform):
        _, applied = apply_fedspan_update(broadcast, states, result,
                                          module_scales=scales)
        # Adapters are aggregated in float32, so the applied norm matches the
        # requested one to the cast, which is what the production check uses.
        assert applied["applied_step_norm"] == pytest.approx(
            result["resolved_step_norm"], rel=1e-6)
    assert not np.allclose(active_weights(uniform), active_weights(fedspan))


def test_fixed_coefficients_follow_the_same_one_over_r_rule():
    result = solve("fixed", fixed_weights=UNIFORM_OVER_DISTRIBUTIONS)
    gram = recorded_gram(result)
    w = active_weights(result)
    mixture = math.sqrt(float(w @ gram @ w))
    step = result["resolved_step_norm"]
    for local, index in enumerate(result["active_indices"]):
        expected = (step * w[local]
                    / (result["client_norms"][index] * mixture))
        assert result["delta_weights"][index] == pytest.approx(expected,
                                                              abs=1e-12)


def test_fixed_weights_are_renormalised_over_the_active_clients():
    """A gated-out client's share is redistributed, not silently applied."""
    broadcast, states, scales = federation_with_cosine_gram(
        CLONE_GRAM, [1.0, 1.0, 1.0, 0.0], seed=44)
    result = fedspan_delta_weights(
        states, broadcast, module_scales=scales, step_policy="median-active",
        direction_policy="fixed",
        fixed_weights=UNIFORM_OVER_DISTRIBUTIONS)
    assert result["active_mask"] == [True, True, True, False]
    assert np.max(np.abs(active_weights(result) - 1.0 / 3.0)) < 1e-12
    assert result["simplex_weights"][3] == 0.0


def test_fixed_weights_with_no_active_mass_fail_closed():
    broadcast, states, scales = federation_with_cosine_gram(
        CLONE_GRAM, [1.0, 1.0, 1.0, 0.0], seed=45)
    result = fedspan_delta_weights(
        states, broadcast, module_scales=scales, step_policy="median-active",
        direction_policy="fixed", fixed_weights=[0.0, 0.0, 0.0, 1.0])
    assert result["fallback"] == "zero_update"
    assert result["status"] == "fixed_weights_inactive"
    assert result["delta_weights"] == [0.0] * 4


@pytest.mark.parametrize("weights", [
    [0.5, 0.5, 0.0],                       # wrong length
    [0.5, 0.6, -0.1, 0.0],                 # negative
    [0.5, float("nan"), 0.25, 0.25],       # nonfinite
    [0.0, 0.0, 0.0, 0.0],                  # no mass at all
])
def test_fixed_policy_rejects_malformed_weight_vectors(weights):
    with pytest.raises(ValueError, match="fixed_weights"):
        solve("fixed", fixed_weights=weights)


def test_fixed_policy_requires_weights():
    with pytest.raises(ValueError, match="fixed_weights"):
        solve("fixed")


def test_other_policies_reject_fixed_weights():
    for policy in ("minnorm", "maxmin-lp", "exact"):
        with pytest.raises(ValueError, match="fixed_weights"):
            solve(policy, fixed_weights=UNIFORM_OVER_CLIENTS)


def test_unknown_direction_policy_still_rejected():
    with pytest.raises(ValueError, match="direction_policy"):
        solve("min-norm")


def test_uniform_over_distributions_beats_uniform_over_clients_here():
    """The oracle grouping recovers part of the clone discount, as designed."""
    over_clients = solve("fixed", fixed_weights=UNIFORM_OVER_CLIENTS)
    over_distributions = solve("fixed",
                               fixed_weights=UNIFORM_OVER_DISTRIBUTIONS)
    exact = solve("exact")
    assert (over_clients["achieved_min_direction_cosine"]
            < over_distributions["achieved_min_direction_cosine"]
            < exact["achieved_min_direction_cosine"] + 1e-12)


# --- Regressions from the pre-E3 verification pass (2026-08-31) -------------
#
# Each test below pins a defect that the 357-test suite did NOT catch. They are
# kept together because they share one lesson: a green suite is evidence only
# about what it actually constrains.


def test_fixed_arm_maps_weights_by_client_index_not_by_position():
    """The declared weight follows the CLIENT, across an inactive gap.

    This is the line the whole fixed-arm contrast rests on: a fixed arm is only
    a control if w lands on the clients it names. When a fail-closed gate drops
    a client, ``active`` stops being ``range(K)``, and mapping by position
    silently re-points every later weight onto the wrong client -- the
    singleton's protective mass onto a clone, in E3's case. A mutant doing
    exactly that survived the full suite before this test existed.
    """
    # Client 1 contributes nothing, so the activity gate drops it and the
    # active set is the non-contiguous [0, 2, 3].
    broadcast, clients, scales = federation_with_cosine_gram(
        CLONE_GRAM, [1.9, 0.0, 2.1, 0.2])
    declared = [0.1, 0.2, 0.3, 0.4]

    result = fedspan_delta_weights(
        clients, broadcast, module_scales=scales, step_policy="median-active",
        direction_policy="fixed", fixed_weights=declared)

    active = [index for index, live in enumerate(result["active_mask"]) if live]
    assert active == [0, 2, 3], "the zero-update client must be gated out"

    # By client index: [0.1, 0.3, 0.4] -> [0.125, 0.375, 0.5].
    # By position:     [0.1, 0.2, 0.3] -> [1/6,   1/3,   0.5].  Distinguishable.
    applied = np.asarray([result["simplex_weights"][index] for index in active])
    by_index = np.asarray([0.1, 0.3, 0.4]); by_index /= by_index.sum()
    by_position = np.asarray([0.1, 0.2, 0.3]); by_position /= by_position.sum()

    assert not np.allclose(by_index, by_position), "the test cannot discriminate"
    np.testing.assert_allclose(applied, by_index, atol=1e-12)


def test_exact_solver_survives_ill_conditioned_faces():
    """One numerically bad sub-face must not lose a solvable problem.

    The KKT identity mu = 2 w^T C w holds only up to |w| times the solve
    residual, so a degenerate face can violate an unscaled tolerance while the
    global problem is perfectly well posed. Aborting the enumeration there
    failed ~7% of well-formed Grams. Rejecting the face cannot return a wrong
    answer unnoticed, because the winner is still checked against the global
    Wolfe certificate -- which this test re-checks independently.
    """
    rng = np.random.default_rng(20260831)
    failures = []
    for trial in range(400):
        K = int(rng.integers(2, 9))
        rank = int(rng.integers(1, K + 2))
        directions = rng.normal(size=(K, rank))
        if trial % 2 == 0 and K >= 3:            # inject a near-duplicate row
            directions[1] = directions[0] + 1e-7 * rng.normal(size=rank)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if float(norms.min()) < 1e-9:
            continue
        C = (directions / norms) @ (directions / norms).T
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 1.0)
        try:
            w, value = minnorm_exact_weights(C)
        except Exception as exc:                 # noqa: BLE001 - reported below
            failures.append((trial, type(exc).__name__, str(exc)[:90]))
            continue
        assert w.min() >= -1e-12 and abs(float(w.sum()) - 1.0) < 1e-9
        assert wolfe_certificate(C, w) <= 1e-8, (
            f"trial {trial}: returned answer is not certified optimal")
        # Compared in the squared space the solver actually minimises: near a
        # cancelling geometry (w^T C w ~ 1e-18, i.e. the origin is inside the
        # conic hull) the square root amplifies a 1e-18 disagreement to 1e-9,
        # which says nothing about the solve. Such rounds are refused by the
        # driver's own gamma* > 1e-6 near-cancellation gate.
        squared = float(w @ C @ w)
        assert abs(value ** 2 - squared) <= 1e-9 * max(1.0, abs(squared))
    assert not failures, (
        f"{len(failures)}/400 well-formed Grams were refused: {failures[:3]}")


def test_minnorm_status_does_not_claim_optimal_when_frank_wolfe_stalls():
    """A stalled iterate must not be labelled with the word 'optimal'.

    The exact arm keeps the label on the same geometry, because there the
    claim is earned by the face enumeration's certificate rather than by an
    iteration count.
    """
    eps = 1e-4
    near_clone = np.array([
        [1.0, 1 - eps, 1 - eps, -0.065],
        [1 - eps, 1.0, 1 - eps, -0.065],
        [1 - eps, 1 - eps, 1.0, -0.065],
        [-0.065, -0.065, -0.065, 1.0],
    ])
    broadcast, clients, scales = federation_with_cosine_gram(
        near_clone, CLONE_RADII)

    def solve_near(policy, **kwargs):
        return fedspan_delta_weights(
            clients, broadcast, module_scales=scales,
            step_policy="median-active", direction_policy=policy, **kwargs)

    stalled = solve_near("minnorm")
    assert stalled["frank_wolfe_converged"] is False, (
        "the fixture no longer reproduces a Frank-Wolfe stall")
    assert stalled["status"] == "stalled"
    assert stalled["direction_solver_shortfall"] > 1e-7, (
        "a stalled round must report the distance it actually fell short by")

    exact = solve_near("exact")
    assert exact["status"] == "optimal"
    assert exact["exact_solver"]["wolfe_certificate"] <= 1e-8
    # The exact measurement, not the stalled iterate, defines the target.
    assert exact["min_norm_value_source"] == "exact-face-enumeration"
    assert stalled["min_norm_value"] == pytest.approx(exact["min_norm_value"])
