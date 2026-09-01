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


def clone_block_argmin(rho, cross, n_clone=3):
    """Analytic ``(value, w)`` for the clone-block + singleton Gram family.

    The Gram is ``n_clone`` mutually-``rho`` clients plus one singleton at
    cosine ``cross`` to each of them. ``w^T C w`` is convex and the Gram is
    invariant under permuting the clone block, so averaging any optimum over
    that permutation group yields another optimum with strictly smaller norm
    unless it is already symmetric: the minimum-norm optimum therefore splits
    its clone mass ``m`` evenly. Substituting ``w = (m/n, ..., m/n, 1 - m)``
    leaves a one-dimensional quadratic in ``m`` whose stationary point is

        m = (1 - cross) / ((1 + (n - 1) rho) / n - 2 cross + 1),

    which is the returned optimum whenever it lies strictly inside (0, 1).
    """
    n = int(n_clone)
    denominator = (1.0 + (n - 1) * rho) / n - 2.0 * cross + 1.0
    mass = (1.0 - cross) / denominator
    if not 0.0 < mass < 1.0:
        raise AssertionError("clone-block oracle requires an interior optimum")
    w = np.full(n + 1, mass / n, dtype=np.float64)
    w[n] = 1.0 - mass
    C = np.full((n + 1, n + 1), cross, dtype=np.float64)
    C[:n, :n] = rho
    np.fill_diagonal(C, 1.0)
    return float(np.sqrt(max(float(w @ C @ w), 0.0))), w


def min_norm_argmin_scipy(gram, value_restarts=12, epsilon=1e-11):
    """``min ||w||_2`` over the optima of ``min_w w^T C w`` on the simplex.

    Two SLSQP stages, sharing no code with the production face enumeration:
    stage one finds the optimal value from many starts, stage two minimises
    the squared Euclidean norm over the convex sublevel set
    ``{w in simplex : w^T C w <= value + epsilon}``. The optimal set of a
    convex program is convex, so the second stage has a unique solution.
    """
    from scipy.optimize import minimize

    C = np.asarray(gram, dtype=np.float64)
    size = C.shape[0]
    simplex = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                "jac": lambda w: np.ones_like(w)}]
    bounds = [(0.0, 1.0)] * size
    best = np.inf
    for start in range(value_restarts):
        w0 = (np.ones(size) / size if start == 0
              else np.random.default_rng(start).dirichlet(np.ones(size)))
        solved = minimize(lambda w: w @ C @ w, w0, jac=lambda w: 2 * C @ w,
                          bounds=bounds, constraints=simplex, method="SLSQP",
                          options={"maxiter": 1000, "ftol": 1e-16})
        if solved.success:
            best = min(best, float(solved.fun))
    if not np.isfinite(best):
        raise AssertionError("scipy could not solve the min-norm value")
    optimal = simplex + [{
        "type": "ineq",
        "fun": lambda w: best + epsilon - float(w @ C @ w),
        "jac": lambda w: -2 * C @ w,
    }]
    champion = None
    for start in range(value_restarts):
        w0 = (np.ones(size) / size if start == 0
              else np.random.default_rng(100 + start).dirichlet(np.ones(size)))
        solved = minimize(lambda w: float(w @ w), w0, jac=lambda w: 2 * w,
                          bounds=bounds, constraints=optimal, method="SLSQP",
                          options={"maxiter": 1000, "ftol": 1e-16})
        if not solved.success:
            continue
        w = np.clip(np.asarray(solved.x, dtype=np.float64), 0.0, None)
        w = w / w.sum()
        if float(w @ C @ w) > best + 1e-8:
            continue
        if champion is None or float(w @ w) < float(champion @ champion):
            champion = w
    if champion is None:
        raise AssertionError("scipy could not solve the min-norm argmin")
    return float(np.sqrt(max(best, 0.0))), champion
