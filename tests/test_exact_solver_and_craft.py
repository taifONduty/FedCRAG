"""Exact face-enumeration direction solver + CRAFT-style equality projection.

The solver is the E2-onward primary: at deployment scale (K<=10) enumerating
the 2^K-1 faces with an augmented-KKT solve is exact, so no iterative
convergence caveat survives. The C_S w = 1 shortcut is deliberately NOT used:
it misses faces whose restricted Gram is singular.

CRAFT (arXiv:2605.21317) is an unwired reference implementation, not a runnable
arm and not one of the pre-registered baselines: it PRESCRIBES the alignment
profile (rho proportional to data) and projects a reference onto the equality
constraints, where we choose the profile endogenously. Its equality constraint
is unsatisfiable whenever duplicated clients are given different targets, which
is why it now refuses rather than relaxing.
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from aggregation_schemes import (CraftInfeasibleError,
                                 craft_delta_coefficients,
                                 minnorm_exact_weights, wolfe_certificate)


def unit_rows(rng, K, dim, spread=1.0, anchor=None):
    V = rng.normal(size=(K, dim))
    if anchor is not None:
        V = anchor[None, :] + spread * V
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def reference_min_value(C):
    """Independent SLSQP solve of min_{w in simplex} w^T C w."""
    K = C.shape[0]
    best = np.inf
    for start in range(8):
        w0 = (np.ones(K) / K if start == 0
              else np.random.default_rng(start).dirichlet(np.ones(K)))
        r = minimize(lambda w: w @ C @ w, w0, jac=lambda w: 2 * C @ w,
                     bounds=[(0, 1)] * K,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                   "jac": lambda w: np.ones_like(w)}],
                     method="SLSQP", options={"maxiter": 800, "ftol": 1e-14})
        if r.success:
            best = min(best, float(r.fun))
    return best


# ------------------------------------------------------- exact solver

def test_exact_matches_independent_solver_on_random_grams():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(60):
        K = int(rng.integers(2, 8))
        U = unit_rows(rng, K, int(rng.integers(K, 20)))
        C = U @ U.T
        w, value = minnorm_exact_weights(C)
        assert w.shape == (K,)
        assert w.min() >= -1e-12 and abs(w.sum() - 1) < 1e-9
        worst = max(worst, abs(value ** 2 - reference_min_value(C)))
    assert worst < 1e-8, worst


def test_exact_is_never_worse_than_frank_wolfe():
    from aggregation_schemes import _min_norm_simplex_weights
    rng = np.random.default_rng(1)
    for _ in range(40):
        K = int(rng.integers(3, 9))
        C = (lambda U: U @ U.T)(unit_rows(rng, K, int(rng.integers(K, 16))))
        _, exact = minnorm_exact_weights(C)
        w_fw, _ = _min_norm_simplex_weights(C)
        assert exact <= float(np.sqrt(max(w_fw @ C @ w_fw, 0.0))) + 1e-9


@pytest.mark.parametrize("case", ["duplicate", "antipodal", "singleton",
                                  "rank_deficient", "orthogonal"])
def test_exact_handles_degenerate_geometry(case):
    if case == "duplicate":
        u = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elif case == "antipodal":
        u = np.array([[1.0, 0.0], [-1.0, 0.0]])
    elif case == "singleton":
        u = np.array([[1.0, 0.0]])
    elif case == "rank_deficient":
        u = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.8, 0.6, 0.0]])
    else:
        u = np.eye(4)
    C = u @ u.T
    w, value = minnorm_exact_weights(C)
    assert np.all(np.isfinite(w)) and abs(w.sum() - 1) < 1e-9
    assert value >= -1e-12
    if case == "antipodal":
        assert value < 1e-9          # cancellation: no protective direction
    if case == "orthogonal":
        assert abs(value - np.sqrt(1 / 4)) < 1e-9   # tight spectral bound
    if case == "singleton":
        assert abs(value - 1.0) < 1e-12


def test_wolfe_certificate_holds_at_the_exact_optimum():
    rng = np.random.default_rng(2)
    for _ in range(30):
        K = int(rng.integers(2, 7))
        C = (lambda U: U @ U.T)(unit_rows(rng, K, int(rng.integers(K, 14))))
        w, value = minnorm_exact_weights(C)
        violation = wolfe_certificate(C, w)
        assert violation <= 1e-9, violation


def test_exact_solution_equalizes_alignments_on_its_support():
    """KKT complementary slackness: supported clients attain gamma* exactly."""
    rng = np.random.default_rng(3)
    C = (lambda U: U @ U.T)(unit_rows(rng, 5, 9))
    w, value = minnorm_exact_weights(C)
    payoffs = C @ w
    support = w > 1e-9
    assert np.allclose(payoffs[support], value ** 2, atol=1e-9)
    assert np.all(payoffs[~support] >= value ** 2 - 1e-9)


# ------------------------------------------------------------- CRAFT

def test_craft_uniform_targets_recover_equal_alignments():
    rng = np.random.default_rng(4)
    U = unit_rows(rng, 4, 10, spread=0.6,
                  anchor=np.array([1.0] + [0.0] * 9))
    C = U @ U.T
    v = craft_delta_coefficients(C, np.ones(4) / 4)
    align = C @ v
    assert np.allclose(align, align[0], atol=1e-8)


def test_craft_data_proportional_targets_starve_the_minority():
    """CRAFT's default profile is prescribed, so extreme skew is inherited.

    Both quantities are worst-case cosines of the NORMALISED direction each
    rule applies: ``min_j (C v)_j / sqrt(v^T C v)`` for CRAFT's coefficient
    vector, ``sqrt(min_w w^T C w)`` for the min-norm point. Comparing CRAFT's
    raw ``min_j (C v)_j`` against the latter would mix units, since ``v`` is
    not a unit direction and is not a simplex point.
    """
    rng = np.random.default_rng(5)
    U = unit_rows(rng, 4, 12, spread=0.7,
                  anchor=np.array([1.0] + [0.0] * 11))
    C = U @ U.T
    counts = np.array([110000.0, 14000.0, 900.0, 700.0])
    rho = counts / counts.sum()
    v_craft = craft_delta_coefficients(C, rho)
    craft_norm = float(np.sqrt(max(float(v_craft @ C @ v_craft), 0.0)))
    worst_craft = float(np.min(C @ v_craft)) / craft_norm
    _, gamma = minnorm_exact_weights(C)
    assert worst_craft >= -1e-9              # positivity is guaranteed
    assert worst_craft < gamma               # but magnitude is not
    assert worst_craft < 0.05 * gamma        # measured on this fixture: 0.9 %


def test_craft_is_marked_unwired():
    """It has no command line and is not one of the pre-registered arms."""
    doc = craft_delta_coefficients.__doc__
    assert "UNWIRED" in doc
    assert "not a runnable arm" in doc


def test_craft_raises_when_its_equality_constraint_is_infeasible():
    """Exact clones force equal alignments, so an unequal profile is not met.

    With ``u1 = u2 = u3`` the constraint ``(C v)_1 = (C v)_2 = (C v)_3`` holds
    identically, so a target that asks the three sub-silos for different
    alignments is outside ``range(C)`` and the closed form silently returns a
    least-squares relaxation of a program it claims to solve exactly.
    """
    rows = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    C = rows @ rows.T
    counts = np.array([0.9, 0.09, 0.01, 0.0])
    rho = np.append(counts[:3], 0.2)
    with pytest.raises(CraftInfeasibleError, match="range"):
        craft_delta_coefficients(C, rho)


def test_craft_accepts_a_feasible_profile_on_the_same_clone_gram():
    """Equal targets for the clones are in range(C), so the solve stands."""
    rows = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    C = rows @ rows.T
    rho = np.array([0.3, 0.3, 0.3, 0.1])
    v = craft_delta_coefficients(C, rho)
    assert np.allclose(C @ v, rho, atol=1e-8)


def test_craft_pseudoinverse_cutoff_is_an_explicit_argument():
    """The cutoff moves the answer by orders of magnitude."""
    epsilon = 1e-5
    rows = np.array([[1.0, 0.0, 0.0],
                     [1.0, epsilon, 0.0],
                     [1.0, 0.0, epsilon],
                     [0.0, 0.0, 1.0]])
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    C = rows @ rows.T
    rho = np.array([0.34, 0.33, 0.33, 0.05])
    permissive = craft_delta_coefficients(C, rho, rcond=1e-15,
                                          feasibility_tol=None)
    truncating = craft_delta_coefficients(C, rho, rcond=1e-8,
                                          feasibility_tol=None)
    assert np.linalg.norm(permissive) > 100.0 * np.linalg.norm(truncating)
    default = craft_delta_coefficients(C, rho, feasibility_tol=None)
    assert np.allclose(default, permissive)


def test_craft_satisfies_its_equality_constraints():
    rng = np.random.default_rng(6)
    C = (lambda U: U @ U.T)(unit_rows(rng, 5, 11))
    rho = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    v = craft_delta_coefficients(C, rho)
    assert np.allclose(C @ v, rho, atol=1e-8)


def test_craft_is_least_norm_correction_from_its_reference():
    rng = np.random.default_rng(7)
    C = (lambda U: U @ U.T)(unit_rows(rng, 4, 9))
    rho = np.array([0.3, 0.3, 0.2, 0.2])
    a = np.ones(4) / 4
    v = craft_delta_coefficients(C, rho, reference=a)
    # any feasible alternative must be at least as far from the reference
    for _ in range(20):
        z = rng.normal(size=4)
        z = z - np.linalg.pinv(C) @ (C @ z)       # keep feasibility
        alt = v + z
        if np.allclose(C @ alt, rho, atol=1e-8):
            d_v = (v - a) @ C @ (v - a)
            d_alt = (alt - a) @ C @ (alt - a)
            assert d_v <= d_alt + 1e-9
