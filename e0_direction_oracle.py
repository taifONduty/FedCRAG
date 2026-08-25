"""Audit-only exact enumerators for the E0 simplex direction games."""
import itertools
import math

import numpy as np


_FEASIBILITY_TOL = 1e-10
_RESIDUAL_TOL = 1e-10
_SYMMETRY_TOL = 1e-12
_TIE_TOL = 1e-12


def _validated_gram(gram):
    try:
        matrix = np.asarray(gram, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("gram must contain numeric values") from error
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0):
        raise ValueError("gram must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("gram must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_SYMMETRY_TOL):
        raise ValueError("gram must be symmetric")
    return np.asarray((matrix + matrix.T) * 0.5, dtype=np.float64)


def _all_supports(size):
    for count in range(1, size + 1):
        yield from itertools.combinations(range(size), count)


def _linear_solution(matrix, rhs, allow_rank_deficient):
    if np.linalg.matrix_rank(matrix) == matrix.shape[0]:
        solved = np.linalg.solve(matrix, rhs)
    elif allow_rank_deficient:
        solved, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    else:
        return None
    residual = float(np.linalg.norm(matrix @ solved - rhs, ord=np.inf))
    scale = (1.0 + float(np.linalg.norm(matrix, ord=np.inf))
             * float(np.linalg.norm(solved, ord=np.inf))
             + float(np.linalg.norm(rhs, ord=np.inf)))
    if residual > _RESIDUAL_TOL * scale:
        return None
    return solved


def _simplex_weights(candidate):
    if float(np.min(candidate)) < -_FEASIBILITY_TOL:
        return None
    if abs(float(np.sum(candidate)) - 1.0) > _FEASIBILITY_TOL:
        return None
    weights = np.asarray(candidate, dtype=np.float64).copy()
    weights[weights < 0.0] = 0.0
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    return weights / total


def _lexicographically_better(weights, incumbent):
    return tuple(float(value) for value in weights) < tuple(
        float(value) for value in incumbent)


def min_norm_simplex_oracle(gram):
    """Return the exact minimum of ``w.T @ gram @ w`` on the simplex."""
    matrix = _validated_gram(gram)
    size = matrix.shape[0]
    best_value = None
    best_weights = None

    for support in _all_supports(size):
        count = len(support)
        block = matrix[np.ix_(support, support)]
        kkt = np.zeros((count + 1, count + 1), dtype=np.float64)
        kkt[:count, :count] = block
        kkt[:count, count] = 1.0
        kkt[count, :count] = 1.0
        rhs = np.zeros(count + 1, dtype=np.float64)
        rhs[count] = 1.0
        solved = _linear_solution(kkt, rhs, allow_rank_deficient=True)
        if solved is None:
            continue

        candidate = np.zeros(size, dtype=np.float64)
        candidate[list(support)] = solved[:count]
        weights = _simplex_weights(candidate)
        if weights is None:
            continue
        value = float(weights @ matrix @ weights)
        if (best_value is None or value < best_value - _TIE_TOL
                or (math.isclose(value, best_value, rel_tol=0.0,
                                 abs_tol=_TIE_TOL)
                    and _lexicographically_better(weights, best_weights))):
            best_value = value
            best_weights = weights

    if best_weights is None:
        raise ValueError("gram has no feasible min-norm candidate")
    objective = float(best_weights @ matrix @ best_weights)
    return {
        "weights": best_weights,
        "objective": objective,
        "simplex_residual": abs(float(np.sum(best_weights)) - 1.0),
        "constraint_violation": max(0.0, -float(np.min(best_weights))),
    }


def maximin_simplex_oracle(gram):
    """Return the exact maximum of ``min(gram @ w)`` on the simplex."""
    matrix = _validated_gram(gram)
    size = matrix.shape[0]
    constraints = ([('tight', index) for index in range(size)]
                   + [('zero', index) for index in range(size)])
    best_value = None
    best_weights = None

    for combo in itertools.combinations(range(2 * size), size):
        equations = np.zeros((size + 1, size + 1), dtype=np.float64)
        equations[0, :size] = 1.0
        rhs = np.zeros(size + 1, dtype=np.float64)
        rhs[0] = 1.0
        for row_number, position in enumerate(combo, start=1):
            kind, index = constraints[position]
            if kind == 'tight':
                equations[row_number, :size] = matrix[index]
                equations[row_number, size] = -1.0
            else:
                equations[row_number, index] = 1.0
        solved = _linear_solution(
            equations, rhs, allow_rank_deficient=False)
        if solved is None:
            continue

        weights = _simplex_weights(solved[:size])
        if weights is None:
            continue
        vertex_t = float(solved[size])
        payoffs = matrix @ weights
        if float(np.min(payoffs)) < vertex_t - _FEASIBILITY_TOL:
            continue
        value = float(np.min(payoffs))
        if (best_value is None or value > best_value + _TIE_TOL
                or (math.isclose(value, best_value, rel_tol=0.0,
                                 abs_tol=_TIE_TOL)
                    and _lexicographically_better(weights, best_weights))):
            best_value = value
            best_weights = weights

    if best_weights is None:
        raise ValueError("gram has no feasible maximin vertex")
    payoffs = matrix @ best_weights
    objective = float(np.min(payoffs))
    return {
        "weights": best_weights,
        "objective": objective,
        "simplex_residual": abs(float(np.sum(best_weights)) - 1.0),
        "constraint_violation": max(
            0.0,
            -float(np.min(best_weights)),
            float(np.max(objective - payoffs)),
        ),
    }
