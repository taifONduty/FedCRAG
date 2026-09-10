"""Pure pieces of the response-maxmin aggregation arm.

Design: research_workspace/supervisor/2026-09-08_solution_design_response_aggregation.md.
Each client measures how its own held-out ranking metric responds to every other client's
update; the server ranks candidate weight vectors by the worst client's predicted gain
under a floor and applies the verified best. Nothing here touches torch or the driver.
"""
import hashlib
import itertools
import json
import zlib

import numpy as np


def dev_split(qids, fraction, seed, slice_name, min_dev):
    """Deterministic (train, dev) partition of one client's training query ids.

    The dev size is max(min_dev, round(fraction * n)) but never more than half of n; a
    client with fewer than two queries keeps them all for training. The permutation is
    seeded by the run seed and the slice name, so the same run reproduces the same split
    whatever order the ids arrive in.
    """
    ordered = sorted(qids)
    n = len(ordered)
    if n < 2:
        return ordered, []
    n_dev = min(max(int(min_dev), int(round(fraction * n))), n // 2)
    rng = np.random.default_rng([int(seed), zlib.crc32(slice_name.encode("utf-8"))])
    chosen = set(ordered[i] for i in rng.permutation(n)[:n_dev])
    return [q for q in ordered if q not in chosen], sorted(chosen)


def simplex_lattice(K, step):
    """Every v >= 0 with sum 1 on the grid of resolution ``step`` (1/step integer).

    Stars and bars: choose K-1 bar positions among m+K-1 slots; part sizes are the gaps.
    """
    m = int(round(1.0 / step))
    assert abs(m * step - 1.0) < 1e-9, "1/step must be an integer"
    points = []
    for bars in itertools.combinations(range(m + K - 1), K - 1):
        parts, prev = [], -1
        for b in bars:
            parts.append(b - prev - 1)
            prev = b
        parts.append(m + K - 2 - prev)
        points.append(tuple(p / m for p in parts))
    return points


def candidate_grid(K, fixed_points, lattice_step, scales):
    """Ordered name -> weight vector. Fixed points first (their order kept), then every
    lattice point at every scale as ``lat{i}_x{scale}``. A vector equal (to 1e-9) to an
    earlier one is dropped, so the baselines keep their own names."""
    grid, seen = {}, []

    def add(name, v):
        v = [float(x) for x in v]
        assert len(v) == K and all(np.isfinite(v)) and min(v) >= 0.0, name
        if any(max(abs(a - b) for a, b in zip(u, v)) < 1e-9 for u in seen):
            return
        seen.append(v)
        grid[name] = v

    for name, v in fixed_points.items():
        add(name, v)
    for i, p in enumerate(simplex_lattice(K, lattice_step)):
        for s in scales:
            add(f"lat{i}_x{s:g}", [s * x for x in p])
    return grid


def normalise_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def predict_embeddings(base, responses, v):
    """First-order response model (design note, section 4.1).

    ``base`` holds unit-norm embeddings under the broadcast adapter, ``responses[j]`` the
    embeddings under the single-client aggregate j minus ``base``. The prediction for
    weights ``v`` is the renormalised sum; it is exact at v = 0 and at every vertex e_j and
    first-order accurate in between.
    """
    out = np.array(base, dtype=np.float64, copy=True)
    for vj, u in zip(v, responses):
        if vj != 0.0:
            out += float(vj) * u
    return normalise_rows(out)


def ndcg10_from_top(top_idx, cids, qids, qrels, k=10):
    """nDCG@k per query from the row-wise indices of the highest-scoring documents
    (``top_idx`` has at least k + 1 columns, sorted by descending score). trec_eval's
    ndcg_cut conventions: gain is the graded relevance, discount 1/log2(rank + 1), the
    ideal ranking uses the query's own qrels cut at k. A document whose id equals the
    query id is skipped, as the driver's evaluation does for ArguAna. Queries without
    relevant documents score 0."""
    cid_arr = np.asarray(cids)
    out = np.zeros(len(qids), dtype=np.float64)
    for i, q in enumerate(qids):
        rel = qrels.get(q) or {}
        if not rel:
            continue
        dcg, rank = 0.0, 0
        for j in top_idx[i]:
            if cid_arr[j] == q:
                continue
            rank += 1
            if rank > k:
                break
            gain = rel.get(str(cid_arr[j]), 0)
            if gain > 0:
                dcg += gain / np.log2(rank + 1)
        ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        out[i] = dcg / idcg if idcg > 0 else 0.0
    return out


def top_indices(sims, take):
    """Row-wise indices of the ``take`` largest scores, sorted by descending score."""
    sims = np.asarray(sims)
    take = min(int(take), sims.shape[1])
    part = np.argpartition(-sims, take - 1, axis=1)[:, :take]
    order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def ndcg10(sims, cids, qids, qrels, k=10):
    """Per-query nDCG@k from a full score matrix (queries by documents)."""
    return ndcg10_from_top(top_indices(sims, k + 1), cids, qids, qrels, k=k)


def _feasible(means, floors):
    if floors is None:
        return True
    return all(f is None or m >= f for m, f in zip(means, floors))


def rank_candidates(pred_means, current, floors, n_top, scores=None):
    """Names of the ``n_top`` floor-feasible candidates ordered by the worst client's
    selection statistic (descending), then its mean, then name. The statistic is the mean
    gain ``means - current`` unless ``scores`` maps a name to its per-client statistic (the
    pessimistic gain of method v2). The floor always applies to the means. ``floors`` is a
    per-client list (None entries mean no floor for that client) or None for no floor."""
    rows = []
    for name, means in pred_means.items():
        if not _feasible(means, floors):
            continue
        stat = (list(scores[name]) if scores is not None
                else [m - c for m, c in zip(means, current)])
        rows.append((-min(stat), -float(np.mean(stat)), name))
    rows.sort()
    return [name for _, _, name in rows[:n_top]]


def choose_applied(measured_means, current, floors, scores=None):
    """The same rule on measured values: (name, worst-client statistic) of the best
    feasible candidate, or (None, None) when no candidate satisfies the floor."""
    best = rank_candidates(measured_means, current, floors, 1, scores)
    if not best:
        return None, None
    stat = (list(scores[best[0]]) if scores is not None
            else [m - c for m, c in zip(measured_means[best[0]], current)])
    return best[0], float(min(stat))


def pessimistic_gain(diffs):
    """Mean of paired per-query gains minus one standard error: the selection statistic of
    method v2 (research_loop/2026-09-11_solution_loop.md, section 4), which stops a client
    with few dev queries from winning the max-min through noise. NaN for an empty input;
    the single value for one query."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return float("nan")
    if d.size == 1:
        return float(d[0])
    return float(d.mean() - d.std(ddof=1) / np.sqrt(d.size))


def magnitude_candidates(norms, game_weights, eq_scales, game_scales, rel_floor=1e-6):
    """Weight vectors that act on update magnitudes (loop document, sections 3 and 6).

    With product-space update norms r_k and the unit-direction game weights w*:
      eq_x{s}:   v_k = s * rbar / (n_active * r_k)   equal magnitude shares, total s * rbar;
      game_x{s}: v_k = s * w*_k * rbar / r_k          unit directions mixed by w*.
    rbar is the mean active norm, so eq_x1 has the total magnitude of uniform weights with
    equal shares. A client whose norm is at most ``rel_floor`` times the largest moved
    nowhere and receives weight 0.
    """
    r = np.asarray(norms, dtype=np.float64)
    K = r.size
    largest = float(r.max()) if K and bool(np.all(np.isfinite(r))) else 0.0
    if largest <= 0.0:
        return {}
    active = r > rel_floor * largest
    rbar = float(r[active].mean())
    inverse = np.where(active, rbar / np.where(active, r, 1.0), 0.0)
    n_active = int(active.sum())
    out = {}
    for s in eq_scales:
        out[f"eq_x{s:g}"] = [float(s * inverse[k] / n_active) for k in range(K)]
    w = np.asarray(game_weights, dtype=np.float64)
    for s in game_scales:
        out[f"game_x{s:g}"] = [float(s * w[k] * inverse[k]) for k in range(K)]
    return out


def uniform_subsets(K):
    """Every nonempty subset of clients averaged uniformly, named ``sub_<indices>``."""
    out = {}
    for mask in range(1, 2 ** K):
        idx = [k for k in range(K) if mask >> k & 1]
        out["sub_" + "".join(str(k) for k in idx)] = [
            1.0 / len(idx) if k in idx else 0.0 for k in range(K)]
    return out


def greedy_soup(order, gains_of):
    """Per-client greedy soup over the vertices (Model soups' greedy rule applied to every
    client at once): walk the vertices in ``order`` and keep the uniform soup of the
    vertices kept so far, adding the next one only if every client's gain improves. The
    first vertex always enters. ``gains_of(v)`` returns per-client gains for weights v.
    Returns (weights, kept indices)."""
    K = len(order)
    soup, current = [], None
    for k in order:
        trial = soup + [int(k)]
        v = [1.0 / len(trial) if j in trial else 0.0 for j in range(K)]
        gains = [float(g) for g in gains_of(v)]
        if current is None or all(a > b for a, b in zip(gains, current)):
            soup, current = trial, gains
    return [1.0 / len(soup) if j in soup else 0.0 for j in range(K)], soup


def paired_lower_bound(diffs, alpha=0.05, n_boot=2000, seed=0):
    """One-sided bootstrap lower confidence bound at level 1 - alpha on the mean of paired
    per-query differences (design note, section 4.4). NaN for an empty input."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    return float(np.quantile(d[idx].mean(axis=1), alpha))


def response_config_tag(config):
    """Eight hex characters binding the arm's configuration into the output filename, so
    two settings never overwrite each other."""
    keys = ("dev_fraction", "dev_min", "lattice_step", "scales", "n_verify", "floor",
            "floor_delta", "halvings")
    values = [config[k] for k in keys]
    # Method v2 keys enter the tag only when set, so the registered v1 tag is unchanged.
    for key in ("candidates", "select", "eq_scales", "game_scales", "model_pick"):
        if key in config:
            values.append([key, config[key]])
    payload = json.dumps(values, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
