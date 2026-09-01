"""The exact solver's ARGMIN, not just its optimal value.

In degenerate geometry -- which a clone federation guarantees -- the minimiser
of ``w^T C w`` over the simplex is not unique. The optimal set is
``{w in simplex : C w = C w*}``, an affine slice of the simplex, so a solver
has to declare which of its points it returns. This module pins that choice:
among all optima the returned point is the one of least Euclidean norm.

That matters because the headline statistic of the clone experiment is the
per-client weight vector, and a tie-break inherited from a linear-algebra
backend makes it an artifact of the backend rather than a property of the
problem.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregation_schemes import (minnorm_exact_weights,  # noqa: E402
                                 wolfe_certificate)
from reference_solvers import (clone_block_argmin,  # noqa: E402
                               min_norm_argmin_scipy, min_norm_reference)

CLONE_CASES = [(1.0, -0.065), (1.0, 0.0), (1.0, 0.08),
               (0.999999, 0.0), (0.99, -0.065), (0.75, 0.0), (0.23, 0.08)]


def clone_gram(rho, cross, n_clone=3):
    C = np.full((n_clone + 1, n_clone + 1), float(cross), dtype=np.float64)
    C[:n_clone, :n_clone] = float(rho)
    np.fill_diagonal(C, 1.0)
    return C


def random_degenerate_gram(rng, size, rank):
    basis = rng.normal(size=(rank, 12))
    mixing = rng.normal(size=(size, rank))
    rows = mixing @ basis
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    C = rows @ rows.T
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


@pytest.mark.parametrize("rho,cross", CLONE_CASES)
def test_clone_gram_returns_the_symmetric_minimum_norm_argmin(rho, cross):
    """Exact clones are interchangeable, so their weights must be equal."""
    expected_value, expected_w = clone_block_argmin(rho, cross)
    w, value = minnorm_exact_weights(clone_gram(rho, cross))
    assert abs(value - expected_value) < 1e-12
    assert np.max(np.abs(w - expected_w)) < 1e-9, (w, expected_w)


def test_tie_break_is_exactly_the_minimum_norm_optimum():
    """Every other optimum of a degenerate Gram has a strictly larger norm."""
    C = clone_gram(1.0, -0.065)
    w, _ = minnorm_exact_weights(C)
    reference_value, _ = min_norm_reference(C)
    assert abs(float(w @ C @ w) - reference_value) < 1e-12
    rng = np.random.default_rng(11)
    mass = float(w[:3].sum())
    for _ in range(400):
        split = rng.dirichlet(np.ones(3)) * mass
        alternative = np.append(split, 1.0 - mass)
        assert abs(float(alternative @ C @ alternative)
                   - reference_value) < 1e-12
        assert float(w @ w) <= float(alternative @ alternative) + 1e-15


def test_argmin_matches_an_independent_scipy_argmin_on_degenerate_grams():
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(12):
        size = int(rng.integers(3, 6))
        C = random_degenerate_gram(rng, size, int(rng.integers(1, size)))
        w, value = minnorm_exact_weights(C)
        reference_value, reference_w = min_norm_argmin_scipy(C)
        assert abs(value - reference_value) < 1e-6
        worst = max(worst, float(np.max(np.abs(w - reference_w))))
    assert worst < 1e-5, worst


def test_argmin_is_permutation_equivariant():
    rng = np.random.default_rng(7)
    C = clone_gram(1.0, 0.05)
    base_w, base_value = minnorm_exact_weights(C)
    for _ in range(12):
        order = rng.permutation(C.shape[0])
        permuted = C[np.ix_(order, order)]
        w, value = minnorm_exact_weights(permuted)
        assert abs(value - base_value) < 1e-12
        assert np.max(np.abs(w - base_w[order])) < 1e-12


def test_exact_duplicates_split_their_mass_evenly():
    """Three copies of one direction plus two others; K=5, rank 3."""
    rows = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    C = rows @ rows.T
    w, value = minnorm_exact_weights(C)
    assert abs(value - np.sqrt(1.0 / 3.0)) < 1e-12
    assert np.max(np.abs(w - np.array([1, 1, 1, 3, 3]) / 9.0)) < 1e-12


def test_returned_weights_are_a_simplex_point_to_machine_precision():
    rng = np.random.default_rng(13)
    for rho, cross in CLONE_CASES:
        w, _ = minnorm_exact_weights(clone_gram(rho, cross))
        assert abs(float(w.sum()) - 1.0) <= 1e-15, abs(float(w.sum()) - 1.0)
        assert w.min() >= 0.0
    for _ in range(20):
        size = int(rng.integers(2, 7))
        C = random_degenerate_gram(rng, size, int(rng.integers(1, size + 1)))
        w, _ = minnorm_exact_weights(C)
        assert abs(float(w.sum()) - 1.0) <= 1e-15, abs(float(w.sum()) - 1.0)
        assert w.min() >= 0.0


def test_returned_weights_carry_a_zero_wolfe_certificate():
    rng = np.random.default_rng(17)
    for _ in range(25):
        size = int(rng.integers(2, 7))
        C = random_degenerate_gram(rng, size, int(rng.integers(1, size + 1)))
        w, _ = minnorm_exact_weights(C)
        assert wolfe_certificate(C, w) <= 1e-12


def test_clone_mass_is_preserved_while_the_split_is_pinned():
    """Mass is identifiable, the individual weights are pinned by the rule."""
    for rho, cross in CLONE_CASES:
        w, _ = minnorm_exact_weights(clone_gram(rho, cross))
        _, expected = clone_block_argmin(rho, cross)
        assert abs(float(w[:3].sum()) - float(expected[:3].sum())) < 1e-9
        # A near-clone block leaves the face systems ill conditioned, so the
        # split is symmetric to solve precision rather than to the last bit.
        assert abs(float(w[0] - w[1])) < 1e-8
        assert abs(float(w[1] - w[2])) < 1e-8


@pytest.mark.parametrize("size", [2, 3, 4, 5, 6])
def test_identical_clients_receive_equal_weight(size):
    """The all-ones Gram: every client is the same direction."""
    C = np.ones((size, size), dtype=np.float64)
    w, value = minnorm_exact_weights(C)
    assert abs(value - 1.0) < 1e-12
    assert np.max(np.abs(w - 1.0 / size)) < 1e-12


def test_near_identical_clients_answer_like_identical_ones():
    """A clone block a few ulps from cosine 1 must not flip the split.

    Off-diagonal 1 - 1e-14 leaves a Gram whose minimiser is formally unique
    but numerically unresolvable; the declared tie-break resolves it the same
    way as the exactly singular Gram instead of returning backend noise.
    """
    offsets = np.array([[0.00, 1.05, 2.46, 1.13],
                        [1.05, 0.00, 2.14, 0.78],
                        [2.46, 2.14, 0.00, 1.55],
                        [1.13, 0.78, 1.55, 0.00]])
    C = 1.0 - 1e-14 * offsets
    np.fill_diagonal(C, 1.0)
    w, value = minnorm_exact_weights(C)
    assert abs(value - 1.0) < 1e-12
    assert np.max(np.abs(w - 0.25)) < 1e-9, w
