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
