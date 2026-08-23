"""Independent oracles for the FedSpan direction game, for use by tests.

Nothing here calls the production solvers. Both problems are solved by exact
enumeration plus dense linear algebra, so a test can state what the answer
*is* rather than what the implementation happened to return.

Definitions, on a symmetric cosine Gram ``C`` of unit client directions:

  maximin LP     max over the simplex of  min_i (C w)_i
  min-norm point min over the simplex of  w^T C w

The applied server direction is the NORMALIZED mixture, so the worst-case
cosine a solution actually attains is ``min_i (C w)_i / sqrt(w^T C w)``. By
minimax duality its largest attainable value is ``sqrt(min_w w^T C w)``.
"""
import itertools

import numpy as np

_FEASIBILITY_TOL = 1e-10


def _candidate_supports(size):
    for count in range(1, size + 1):
        yield from itertools.combinations(range(size), count)


def min_norm_reference(gram):
    """Exact ``(value, w)`` for ``min_w w^T C w`` over the simplex.

    Enumerates every support. On a fixed support the equality-constrained
    minimizer is the linear solve ``w_S = C_SS^{-1} 1 / (1^T C_SS^{-1} 1)``;
    every such point that is nonnegative is simplex-feasible, so its value
    bounds the optimum from above, and the true minimizer's own support is
    among those enumerated. The minimum over feasible candidates is therefore
    the exact optimum, with no optimality test required.
    """
    C = np.asarray(gram, dtype=np.float64)
    size = C.shape[0]
    best_value = None
    best_w = None
    for support in _candidate_supports(size):
        block = C[np.ix_(support, support)]
        try:
            solved = np.linalg.solve(block, np.ones(len(support)))
        except np.linalg.LinAlgError:
            continue
        total = float(solved.sum())
        if abs(total) < 1e-14:
            continue
        weights = solved / total
        if weights.min() < -_FEASIBILITY_TOL:
            continue
        w = np.zeros(size, dtype=np.float64)
        w[list(support)] = np.clip(weights, 0.0, None)
        w = w / w.sum()
        value = float(w @ C @ w)
        if best_value is None or value < best_value:
            best_value, best_w = value, w
    if best_value is None:
        raise AssertionError("no feasible min-norm candidate")
    return best_value, best_w


def maximin_reference(gram):
    """Exact ``(t, w)`` for ``max_w min_i (C w)_i`` over the simplex.

    Vertex enumeration of ``{(w, t): 1^T w = 1, w >= 0, C w >= t 1}``. With
    K + 1 unknowns and one equality, a vertex activates K of the 2K
    inequalities; every such solve that is feasible gives a lower bound on
    the optimum, and the optimum is attained at one of them.
    """
    C = np.asarray(gram, dtype=np.float64)
    size = C.shape[0]
    constraints = ([("tight", index) for index in range(size)]
                   + [("zero", index) for index in range(size)])
    best_t = None
    best_w = None
    for combo in itertools.combinations(range(2 * size), size):
        rows = [[1.0] * size + [0.0]]
        rhs = [1.0]
        for position in combo:
            kind, index = constraints[position]
            if kind == "tight":
                rows.append(list(C[index]) + [-1.0])
            else:
                row = [0.0] * (size + 1)
                row[index] = 1.0
                rows.append(row)
            rhs.append(0.0)
        try:
            solved = np.linalg.solve(np.asarray(rows, dtype=np.float64),
                                     np.asarray(rhs, dtype=np.float64))
        except np.linalg.LinAlgError:
            continue
        w, t = solved[:size], float(solved[size])
        if w.min() < -_FEASIBILITY_TOL:
            continue
        if abs(w.sum() - 1.0) > _FEASIBILITY_TOL:
            continue
        if float(np.min(C @ w)) < t - _FEASIBILITY_TOL:
            continue
        if best_t is None or t > best_t:
            best_t = t
            best_w = np.clip(w, 0.0, None) / float(np.clip(w, 0.0, None).sum())
    if best_t is None:
        raise AssertionError("no feasible maximin vertex")
    return best_t, best_w


def achieved_worst_case_cosine(gram, weights):
    """``min_i (C w)_i / ||sum_i w_i u_i||`` — what a mixture actually attains."""
    C = np.asarray(gram, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mixture_norm = float(np.sqrt(max(float(w @ C @ w), 0.0)))
    if mixture_norm <= 0.0:
        return None
    return float(np.min(C @ w)) / mixture_norm
