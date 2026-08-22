"""Server-side aggregation weighting schemes for federated LoRA.

Pure math over adapter state dicts — no model, dataset, or GPU dependency.
Every scheme consumes only quantities a C1-compliant server legitimately
sees: client adapter states, the broadcast state, example counts, local
step counts, and client-reported scalar losses.

Two families, matching how federated_forgetting.py applies them:
  * simplex weights (sum to 1; feed ``fedavg``):
      maxmin_weights   — FedSpan repair: max-min LP over the cosine Gram
      mgda_weights     — MGDA min-norm point (Sener & Koltun, 1810.04650)
      afl_update       — AFL multiplicative-weights ascent (1902.00146)
  * delta-space weights (need NOT sum to 1; feed ``apply_delta_weights``,
    which rescales the update  w^{t+1} = w^t + sum_k v_k (w_k - w^t)):
      qffl_delta_weights    — q-FedAvg with the h_k normalization (1905.10497)
      fednova_delta_weights — tau-normalized averaging (2007.07481)
"""
import re

import numpy as np
import torch

_LORA_KEY = re.compile(r"(.*)\.lora_(A|B)\.weight$")


def _module_pairs(state):
    mods = {}
    for k, v in state.items():
        m = _LORA_KEY.match(k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.float()
    return {n: (ab["A"], ab["B"]) for n, ab in mods.items()
            if "A" in ab and "B" in ab}


def _stack_ip(s1, s2):
    # <sum ci Bi Ai, sum cj Bj Aj>_F via the trace identity — r x r ops only,
    # never materializing d_out x d_in products:
    #   <B1 A1, B2 A2>_F = sum((B1^T B2) * (A1 A2^T))
    # NOTE 2026-08-22: the campaign-era copies (federated_forgetting.py,
    # mechanism_suite.py) used (A2 @ A1.T) here, which computes the
    # cross-paired <B1 A2, B2 A1> instead. Diagonals are identical; off-
    # diagonals coincide only when clients share an (almost) common A — the
    # measured regime (A-residual ~1e-4), which is why every cross-check
    # passed. Fixed here; test_update_gram_matches_dense pins the identity.
    tot = 0.0
    for c1, B1, A1 in s1:
        for c2, B2, A2 in s2:
            tot += c1 * c2 * torch.sum((B1.T @ B2) * (A1 @ A2.T)).item()
    return tot


def _update_stacks(client_states, broadcast_state):
    prev = _module_pairs(broadcast_state)
    stacks = []
    for st in client_states:
        mp = _module_pairs(st)
        stacks.append({n: [(1.0, B, A)]
                       + ([(-1.0, prev[n][1], prev[n][0])] if n in prev else [])
                       for n, (A, B) in mp.items()})
    return stacks


def update_gram(client_states, broadcast_state, normalize=False):
    """K x K Gram of weight-space client updates dW_k = B_k A_k - B_g A_g.

    ``normalize=True`` returns the cosine Gram (unit diagonal); raw Frobenius
    inner products otherwise. Exact in factor space via the trace identity.
    """
    stacks = _update_stacks(client_states, broadcast_state)
    K = len(stacks)
    G = np.zeros((K, K))
    for a in range(K):
        for b in range(a, K):
            ip = sum(_stack_ip(stacks[a][n], stacks[b][n])
                     for n in stacks[a] if n in stacks[b])
            G[a, b] = G[b, a] = ip
    if not normalize:
        return G
    norms = np.sqrt(np.clip(np.diag(G), 1e-24, None))
    return G / np.outer(norms, norms)


def maxmin_weights(client_states, broadcast_state):
    """FedSpan repair: w* = argmax_{w in simplex} min_i (G_c w)_i  (LP).

    Returns None (caller falls back to uniform) if the LP fails.
    """
    from scipy.optimize import linprog
    Gc = update_gram(client_states, broadcast_state, normalize=True)
    K = Gc.shape[0]
    c = np.zeros(K + 1); c[-1] = -1.0
    res = linprog(c, A_ub=np.hstack([-Gc, np.ones((K, 1))]), b_ub=np.zeros(K),
                  A_eq=[[1.0] * K + [0.0]], b_eq=[1.0],
                  bounds=[(0, 1)] * K + [(None, None)], method="highs")
    if not res.success:
        print("  WARNING: max-min LP failed; falling back to uniform weights")
        return None
    return [float(x) for x in res.x[:K]]


def mgda_weights(client_states, broadcast_state, iters=500, tol=1e-9):
    """MGDA min-norm convex combination over the RAW update Gram.

    Frank-Wolfe on  min_{w in simplex} w^T G w  (Sener & Koltun's MinNormSolver
    specialized to our exact factor-space Gram). Raw — not cosine — inner
    products: MGDA operates on unnormalized task gradients by definition,
    which is exactly the property the paper contrasts with the max-min LP.
    """
    G = update_gram(client_states, broadcast_state, normalize=False)
    K = G.shape[0]
    if not np.all(np.isfinite(G)) or np.trace(G) <= 0:
        print("  WARNING: degenerate Gram; MGDA falling back to uniform")
        return [1.0 / K] * K
    w = np.ones(K) / K
    for _ in range(iters):
        grad = G @ w
        t = int(np.argmin(grad))
        d = np.eye(K)[t] - w
        denom = float(d @ G @ d)
        if denom <= tol:
            break
        gamma = float(np.clip(-(d @ G @ w) / denom, 0.0, 1.0))
        if gamma * np.abs(d).max() < tol:
            break
        w = w + gamma * d
    return [float(x) for x in w]


def qffl_delta_weights(losses, sq_update_norms, q, L):
    """q-FedAvg (q-FFL) delta-space weights, full-participation form.

    Li et al. (1905.10497), Alg. 2:  Delta_k = L (w^t - w_k),
      h_k = q F_k^{q-1} ||Delta_k||^2 + L F_k^q,
      w^{t+1} = w^t - sum_k F_k^q Delta_k / sum_k h_k
    which in delta space is v_k = L F_k^q / sum_j h_j  applied to
    (w_k - w^t). ``sq_update_norms`` are ||w_k - w^t||_F^2 (raw Gram diag);
    the L^2 factor for ||Delta_k||^2 is applied here. L = 1/lr per the paper.
    q = 0 reduces to uniform FedAvg under full participation.
    """
    f = np.clip(np.asarray(losses, dtype=np.float64), 1e-8, None)
    d2 = np.clip(np.asarray(sq_update_norms, dtype=np.float64), 0.0, None)
    fq = np.ones_like(f) if q == 0 else f ** q
    fqm1 = np.zeros_like(f) if q == 0 else f ** (q - 1.0)
    h = q * fqm1 * (L ** 2) * d2 + L * fq
    total = float(np.sum(h))
    if not np.isfinite(total) or total <= 0:
        print("  WARNING: q-FedAvg h-sum degenerate; falling back to uniform")
        return [1.0 / len(losses)] * len(losses)
    return [float(x) for x in (L * fq) / total]


def fednova_delta_weights(n_examples, local_steps):
    """FedNova (2007.07481) tau-normalized delta weights:
        p_k = n_k / n,  tau_eff = sum_k p_k tau_k,  v_k = tau_eff p_k / tau_k.
    Clients with tau_k = 0 trained nothing (their delta is zero); they are
    masked out of both p and tau_eff and receive v_k = 0.
    """
    n = np.asarray(n_examples, dtype=np.float64)
    tau = np.asarray(local_steps, dtype=np.float64)
    active = tau > 0
    if not np.any(active):
        print("  WARNING: no client trained; FedNova returns zero weights")
        return [0.0] * len(n)
    p = np.where(active, n, 0.0)
    p = p / max(p.sum(), 1e-12)
    tau_eff = float(np.sum(p[active] * tau[active]))
    v = np.zeros_like(n)
    v[active] = tau_eff * p[active] / tau[active]
    return [float(x) for x in v]


def afl_update(lam, losses, eta):
    """AFL (1902.00146) mixture-weight ascent: one multiplicative-weights /
    exponentiated-gradient step toward the worst-off client,
        lam_k <- lam_k * exp(eta * F_k),  renormalized to the simplex.
    Returns a NEW list; the input is not mutated. Equal losses are a fixed
    point (weights cancel in the normalization).
    """
    f = np.asarray(losses, dtype=np.float64)
    f = f - f.max()          # shift-invariant; avoids exp overflow
    new = np.asarray(lam, dtype=np.float64) * np.exp(eta * f)
    total = new.sum()
    if not np.isfinite(total) or total <= 0:
        print("  WARNING: AFL update degenerate; resetting to uniform")
        return [1.0 / len(lam)] * len(lam)
    return [float(x) for x in new / total]


def apply_delta_weights(broadcast_state, client_states, v):
    """w^{t+1} = w^t + sum_k v_k (w_k - w^t), as a NEW state dict.

    For simplex v this equals fedavg(states, v); delta-space schemes
    (q-FedAvg, FedNova) need it because their v need not sum to 1.
    """
    out = {}
    for key in broadcast_state:
        base = broadcast_state[key].float()
        acc = base.clone()
        for vi, st in zip(v, client_states):
            acc += vi * (st[key].float() - base)
        out[key] = acc
    return out
