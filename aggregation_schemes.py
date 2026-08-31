"""Server-side aggregation weighting schemes for federated LoRA.

Pure math over adapter state dicts — no model, dataset, or GPU dependency.
Every scheme consumes only quantities a C1-compliant server legitimately
sees: client adapter states, the broadcast state, example counts, local
step counts, and client-reported scalar losses.

Three families, matching how federated_forgetting.py applies them:
  * simplex weights (sum to 1; feed ``fedavg``):
      maxmin_weights   — historical raw-maxmin control over the cosine Gram
      mgda_weights     — MGDA min-norm point (Sener & Koltun, 1810.04650)
      afl_update       — AFL multiplicative-weights ascent (1902.00146)
  * delta-space weights (need NOT sum to 1; feed ``apply_delta_weights``,
    which rescales the update  w^{t+1} = w^t + sum_k v_k (w_k - w^t)):
      qffl_delta_weights    — q-FedAvg with the h_k normalization (1905.10497)
      fednova_delta_weights — tau-normalized averaging (2007.07481)
  * frozen-A exact direction + application:
      fedspan_delta_weights / apply_fedspan_update — PEFT-scale-aware,
      true-step-normalized norm-maxmin with fail-closed edge cases, under a
      selectable direction policy ("minnorm" = FedMGDA+ (2006.11489) at
      epsilon = 1 on the cosine Gram, "maxmin-lp" = the LP ablation)

Every scheme returns a ``SchemeResult``/record carrying the solver status, so
a caller can persist why a set of weights was produced, not only which.
"""
import hashlib
import math
import re

import numpy as np
import torch

_LORA_KEY = re.compile(r"(.*)\.lora_(A|B)\.weight$")


class FedSpanContractError(ValueError):
    """The communicated adapter states cannot represent frozen-A FedSpan."""


class ModuleScales(dict):
    """Per-module geometry scales plus the measurements that produced them.

    Compares and serializes as the plain ``{module: scale}`` mapping every
    consumer already expects; ``records`` carries the PEFT scale, the row
    constant ``c``, and the measured pre-orthogonalization row RMS.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = {}


class SchemeResult(list):
    """Aggregation weights carrying the solver status that produced them.

    Behaves as the plain weight list every call site already consumes, so a
    scheme can report *why* it produced a value without changing what is
    applied. ``fallback`` is None only when the scheme's own solution was
    used; a non-None ``fallback`` means the listed weights are a substitute.
    """

    def __new__(cls, weights, **_):
        return super().__new__(cls, weights)

    def __init__(self, weights, status="ok", solver_status=None,
                 solver_message="", fallback=None):
        super().__init__(weights)
        self.status = status
        self.solver_status = solver_status
        self.solver_message = solver_message
        self.fallback = fallback

    def record(self):
        """A fresh JSON-serializable trace of this solve."""
        return {
            "weights": [float(value) for value in self],
            "status": self.status,
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
            "fallback": self.fallback,
        }


def _min_norm_simplex_weights(gram, max_iters=20000, tol=1e-14):
    """Away-step Frank-Wolfe for  min_{w in simplex} w^T C w.

    The minimizer is the min-norm point of the convex hull of the client
    directions whose Gram is ``gram``; its value ``sqrt(w^T C w)`` is, by
    minimax duality, the largest worst-case cosine any normalized mixture can
    attain. Away steps are required rather than cosmetic: plain Frank-Wolfe
    zig-zags between vertices and stalls short of the optimum, and the value
    itself is reported.

    Returns ``(w, info)`` with ``info["gap"]`` the Frank-Wolfe duality gap
    ``w^T C w - min_i (C w)_i``, which upper-bounds the suboptimality, and
    ``info["converged"]``. ``sqrt(w^T C w)`` is an UPPER bound on the
    attainable worst-case cosine and equals it only once converged, so the
    flag has to travel with the value.
    """
    C = np.asarray(gram, dtype=np.float64)
    M = C.shape[0]
    w = np.zeros(M, dtype=np.float64)
    w[int(np.argmin(np.diag(C)))] = 1.0
    gap = float("inf")
    iterations = 0
    for iterations in range(1, int(max_iters) + 1):
        g = C @ w
        value = float(w @ g)
        toward = int(np.argmin(g))
        gap = value - float(g[toward])
        if gap <= tol:
            break
        support = np.flatnonzero(w > 0.0)
        away = int(support[int(np.argmax(g[support]))])
        gap_away = float(g[away]) - value
        if gap >= gap_away:
            direction = -w.copy()
            direction[toward] += 1.0
            max_step = 1.0
        else:
            direction = w.copy()
            direction[away] -= 1.0
            max_step = (w[away] / (1.0 - w[away]) if w[away] < 1.0
                        else float("inf"))
        denominator = float(direction @ C @ direction)
        numerator = -float(direction @ g)
        if denominator <= 0.0:
            step = max_step
        else:
            step = min(max(numerator / denominator, 0.0), max_step)
        if not math.isfinite(step) or step <= 0.0:
            break
        w = np.clip(w + step * direction, 0.0, None)
        total = float(w.sum())
        if total <= 0.0:
            break
        w = w / total
    return w, {"gap": float(gap), "iterations": int(iterations),
               "converged": bool(gap <= tol), "tol": float(tol)}


def configure_frozen_lora_a(model, adapter_name="default", row_scale="unit"):
    """Row-orthogonalize and freeze every PEFT LoRA A for one adapter.

    The operation is performed once, immediately after ``add_adapter`` and
    while LoRA B is still zero, so it does not change the initialized model
    function. QR is computed on CPU in float64 for deterministic, accurate
    rows and copied back to the original device/dtype.

    ``row_scale`` selects the constant ``c`` in ``A A^T = c^2 I``:
      * ``"unit"``      — ``c = 1``.
      * ``"peft-init"`` — ``c`` is the module's own pre-orthogonalization row
        RMS, so the frozen rows keep the magnitude PEFT's kaiming init gave
        them. Measured over 24 real PEFT LoRA modules (BERT query/key/value,
        hidden 128-768, r 4-16) that RMS is 0.5773 +/- 0.0067, so unit rows
        are a factor 1.7323 larger than the init PEFT would have used.
      * a positive float — that constant for every module.

    Returns per-module geometry scales ``sigma * c`` (``sigma = alpha/r``), so
    ``scale * ||B||_F == ||sigma * B A||_F`` exactly and all downstream
    geometry stays in true weight space. The returned mapping also carries a
    ``.records`` breakdown of the measured quantities per module.
    """
    if isinstance(row_scale, str):
        if row_scale not in ("unit", "peft-init"):
            raise ValueError(
                "row_scale must be 'unit', 'peft-init', or a positive float")
    else:
        row_scale = float(row_scale)
        if not math.isfinite(row_scale) or row_scale <= 0:
            raise ValueError("numeric row_scale must be positive and finite")
    scales = ModuleScales()
    for module_name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        scaling = getattr(module, "scaling", None)
        if lora_a is None or lora_b is None or adapter_name not in lora_a:
            continue
        if adapter_name not in lora_b or scaling is None:
            raise FedSpanContractError(
                f"LoRA module '{module_name}' has incomplete adapter metadata")
        linear_a = lora_a[adapter_name]
        linear_b = lora_b[adapter_name]
        if torch.count_nonzero(linear_b.weight.detach()).item() != 0:
            raise FedSpanContractError(
                f"LoRA module '{module_name}' has nonzero LoRA B at frozen-A "
                "initialization; changing A would change the model function")
        weight = linear_a.weight
        if weight.ndim != 2 or weight.shape[0] > weight.shape[1]:
            raise FedSpanContractError(
                f"LoRA A for '{module_name}' cannot have orthonormal rows: "
                f"shape={tuple(weight.shape)}")
        source = weight.detach().cpu().double().T
        row_rms = float(torch.sqrt(
            torch.mean(torch.sum(source ** 2, dim=0))).item())
        if row_scale == "unit":
            c = 1.0
        elif row_scale == "peft-init":
            c = row_rms
        else:
            c = float(row_scale)
        if not math.isfinite(c) or c <= 0:
            raise FedSpanContractError(
                f"row scale for '{module_name}' is not positive and finite: {c}")
        q, r = torch.linalg.qr(source, mode="reduced")
        # Canonicalize QR column signs so the same initial matrix produces the
        # same A independently of QR's otherwise arbitrary sign convention.
        diagonal = torch.diagonal(r)
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal),
                            torch.ones_like(diagonal))
        orthonormal = (c * (q * signs).T).to(
            device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            weight.copy_(orthonormal)
        weight.requires_grad_(False)
        gram = weight.detach().float() @ weight.detach().float().T
        identity = (c ** 2) * torch.eye(weight.shape[0], device=gram.device,
                                        dtype=gram.dtype)
        if not torch.allclose(gram, identity, atol=2e-5 * max(1.0, c ** 2),
                              rtol=2e-5):
            raise FedSpanContractError(
                f"row-orthogonalization verification failed for '{module_name}'")
        try:
            scale = float(scaling[adapter_name])
        except Exception as exc:
            raise FedSpanContractError(
                f"missing PEFT scale for '{module_name}'/{adapter_name}") from exc
        if not math.isfinite(scale) or scale <= 0:
            raise FedSpanContractError(
                f"invalid PEFT scale for '{module_name}': {scale}")
        scales[module_name] = scale * c
        scales.records[module_name] = {
            "peft_scale": scale,
            "row_scale_mode": (row_scale if isinstance(row_scale, str)
                               else "constant"),
            "row_scale_c": c,
            "measured_init_row_rms": row_rms,
            "geometry_scale": scale * c,
        }
    if not scales:
        raise FedSpanContractError(
            f"no LoRA modules found for adapter '{adapter_name}'")
    return scales


def peft_scales(module_scales):
    """The bare PEFT scales ``sigma`` behind a frozen-A geometry mapping.

    ``configure_frozen_lora_a`` returns ``sigma * c`` because raw-B space has
    to supply the row constant itself. A materialized update
    ``B A - B_g A_g`` contracts the real A and therefore already carries c, so
    it must be scaled by ``sigma`` alone; applying ``sigma * c`` there counts c
    twice. Mappings with no measurement records carry no row constant — a
    scalar ``alpha/r``, or a caller-supplied dict — and are returned unchanged.
    """
    records = getattr(module_scales, "records", None)
    if not records:
        return module_scales
    bare = {}
    for name in module_scales:
        scale = (records.get(name) or {}).get("peft_scale")
        try:
            value = float(scale)
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or value <= 0:
            raise FedSpanContractError(
                f"module '{name}' has a geometry scale but no usable PEFT "
                f"scale ({scale!r}), so the frozen-A row constant cannot be "
                "removed")
        bare[name] = value
    return bare


def _module_pairs(state, dtype=torch.float32):
    mods = {}
    for k, v in state.items():
        m = _LORA_KEY.match(k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.to(dtype)
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


def _update_stacks(client_states, broadcast_state, dtype=torch.float32):
    prev = _module_pairs(broadcast_state, dtype=dtype)
    stacks = []
    for st in client_states:
        mp = _module_pairs(st, dtype=dtype)
        stacks.append({n: [(1.0, B, A)]
                       + ([(-1.0, prev[n][1], prev[n][0])] if n in prev else [])
                       for n, (A, B) in mp.items()})
    return stacks


def update_gram(client_states, broadcast_state, normalize=False,
                module_scales=None, dtype=torch.float32):
    """K x K Gram of weight-space client updates dW_k = B_k A_k - B_g A_g.

    ``normalize=True`` returns the cosine Gram (unit diagonal); raw Frobenius
    inner products otherwise. Exact in factor space via the trace identity.
    ``dtype`` is the accumulation precision; float64 is required whenever the
    client delta is small next to the broadcast it is subtracted from.

    CONTRACT: ``module_scales`` must be bare PEFT scales ``sigma``, NEVER the
    frozen-A geometry scales ``sigma * c`` that ``configure_frozen_lora_a``
    returns. The trace identity contracts the real A tensors, so the update
    this function scales already carries the row constant c; passing
    ``sigma * c`` reports ``c`` times the truth and reweights the cosine Gram
    by ``c_l^2`` per module. Use ``peft_scales()`` on a frozen-A mapping.
    """
    stacks = _update_stacks(client_states, broadcast_state, dtype=dtype)
    module_names = sorted(set().union(*(set(stack) for stack in stacks)))
    scales = ({name: 1.0 for name in module_names}
              if module_scales is None
              else _resolve_module_scales(module_names, module_scales))
    K = len(stacks)
    G = np.zeros((K, K))
    for a in range(K):
        for b in range(a, K):
            ip = sum((scales[n] ** 2)
                     * _stack_ip(stacks[a][n], stacks[b][n])
                     for n in stacks[a] if n in stacks[b])
            G[a, b] = G[b, a] = ip
    if not normalize:
        return G
    norms = np.sqrt(np.clip(np.diag(G), 1e-24, None))
    return G / np.outer(norms, norms)


def client_update_norms(client_states, broadcast_state, module_scales=None):
    """Per-client effective weight-space delta norms ||dW_k||_F.

    Arm-agnostic: it uses the full B A - B_g A_g update, so it is defined for
    trainable-A arms as well as frozen-A ones, which is what makes per-round
    step magnitudes comparable across arms.

    ``module_scales`` are bare PEFT scales; see ``update_gram``'s contract.
    """
    gram = update_gram(client_states, broadcast_state, normalize=False,
                       module_scales=module_scales, dtype=torch.float64)
    diagonal = np.diag(gram)
    return [(float(math.sqrt(value)) if math.isfinite(value) and value >= 0
             else None) for value in diagonal]


def maxmin_weights(client_states, broadcast_state, module_scales=None,
                   min_rel_diagonal=1e-12):
    """Historical raw-maxmin control: solve the cosine-game simplex LP.

    These simplex weights do NOT include client-norm/true-step conversion and
    therefore are not corrected FedSpan. A degenerate Gram or a failed LP is
    rejected and reported; the returned weights are then uniform, which is
    what the historical ``None`` return caused callers to apply.

    A client that did not move has a diagonal that is zero only in exact
    arithmetic — ``||BA||^2 - 2<BA,B_g A_g> + ||B_g A_g||^2`` cancels to a
    residue of either sign — so the guard is relative to the largest
    diagonal, not a bare sign test. Normalizing such a row would divide the
    cosine Gram by the square root of that residue.

    ``module_scales`` are bare PEFT scales; see ``update_gram``'s contract.
    """
    from scipy.optimize import linprog
    G = update_gram(client_states, broadcast_state, normalize=False,
                    module_scales=module_scales, dtype=torch.float64)
    K = G.shape[0]
    uniform = [1.0 / K] * K
    diagonal = np.diag(G)
    largest = float(np.max(diagonal)) if np.all(np.isfinite(diagonal)) else 0.0
    floor = max(0.0, float(min_rel_diagonal) * largest)
    if not np.all(np.isfinite(G)) or largest <= 0 or np.min(diagonal) <= floor:
        degenerate = [index for index, value in enumerate(diagonal)
                      if not math.isfinite(value) or value <= floor]
        print("  WARNING: degenerate update Gram; max-min falling back to "
              "uniform weights")
        return SchemeResult(
            uniform, status="degenerate_gram", fallback="uniform",
            solver_message=(
                "Gram is nonfinite or has a vanishing diagonal for clients "
                f"{degenerate}"))
    norms = np.sqrt(diagonal)
    Gc = G / np.outer(norms, norms)
    c = np.zeros(K + 1); c[-1] = -1.0
    res = linprog(c, A_ub=np.hstack([-Gc, np.ones((K, 1))]), b_ub=np.zeros(K),
                  A_eq=[[1.0] * K + [0.0]], b_eq=[1.0],
                  bounds=[(0, 1)] * K + [(None, None)], method="highs")
    if not res.success:
        print("  WARNING: max-min LP failed; falling back to uniform weights")
        return SchemeResult(
            uniform, status="solver_failure", fallback="uniform",
            solver_status=int(res.status), solver_message=str(res.message))
    return SchemeResult([float(x) for x in res.x[:K]],
                        status="optimal", solver_status=int(res.status),
                        solver_message=str(res.message))


def mgda_weights(client_states, broadcast_state, iters=500, tol=1e-9,
                 module_scales=None):
    """MGDA min-norm convex combination over the RAW update Gram.

    Frank-Wolfe on  min_{w in simplex} w^T G w  (Sener & Koltun's MinNormSolver
    specialized to our exact factor-space Gram). Raw — not cosine — inner
    products: MGDA operates on unnormalized task gradients by definition,
    which is exactly the property the paper contrasts with the max-min LP.

    ``module_scales`` are bare PEFT scales; see ``update_gram``'s contract.
    """
    G = update_gram(client_states, broadcast_state, normalize=False,
                    module_scales=module_scales)
    K = G.shape[0]
    if not np.all(np.isfinite(G)) or np.trace(G) <= 0:
        print("  WARNING: degenerate Gram; MGDA falling back to uniform")
        return SchemeResult(
            [1.0 / K] * K, status="degenerate_gram", fallback="uniform",
            solver_message="Gram is nonfinite or has a nonpositive trace")
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
    return SchemeResult([float(x) for x in w], status="optimal")


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
        return SchemeResult(
            [1.0 / len(losses)] * len(losses), status="degenerate_h_sum",
            fallback="uniform",
            solver_message="q-FedAvg h-sum is nonfinite or nonpositive")
    return SchemeResult([float(x) for x in (L * fq) / total],
                        status="optimal")


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
        return SchemeResult([0.0] * len(n), status="no_active",
                            fallback="zero_update",
                            solver_message="every client reported tau_k = 0")
    p = np.where(active, n, 0.0)
    p = p / max(p.sum(), 1e-12)
    tau_eff = float(np.sum(p[active] * tau[active]))
    v = np.zeros_like(n)
    v[active] = tau_eff * p[active] / tau[active]
    return SchemeResult([float(x) for x in v], status="optimal")


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
        return SchemeResult(
            [1.0 / len(lam)] * len(lam), status="degenerate_update",
            fallback="uniform",
            solver_message="AFL weight mass is nonfinite or nonpositive")
    return SchemeResult([float(x) for x in new / total], status="optimal")


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


# --------------------------------------------------- norm-consistent FedSpan

def state_dict_sha256(state):
    """Stable content hash over tensor names, shapes, dtypes, and raw bytes."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _factor_entries(state, label):
    modules = {}
    for key, value in state.items():
        match = _LORA_KEY.match(key)
        if match:
            modules.setdefault(match.group(1), {})[match.group(2)] = (key, value)
    if not modules:
        raise FedSpanContractError(f"{label} contains no LoRA A/B factor keys")
    incomplete = sorted(name for name, factors in modules.items()
                        if set(factors) != {"A", "B"})
    if incomplete:
        raise FedSpanContractError(
            f"{label} has incomplete LoRA factor keys for modules: {incomplete}")
    for name, factors in modules.items():
        A = factors["A"][1]
        B = factors["B"][1]
        if A.ndim != 2 or B.ndim != 2 or B.shape[1] != A.shape[0]:
            raise FedSpanContractError(
                f"{label} has incompatible A/B shapes for module '{name}': "
                f"A={tuple(A.shape)}, B={tuple(B.shape)}")
    return modules


def _resolve_module_scales(module_names, module_scales):
    if isinstance(module_scales, (int, float)):
        scale = float(module_scales)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("module_scales must be positive and finite")
        return {name: scale for name in module_names}
    if not isinstance(module_scales, dict):
        raise TypeError("module_scales must be a positive scalar or a dict")
    resolved = {}
    for name in module_names:
        if name in module_scales:
            value = module_scales[name]
        else:
            # PEFT state keys and named-module paths can differ by a stable
            # base-model prefix. Permit a unique suffix match, never a guess.
            matches = [value for key, value in module_scales.items()
                       if name.endswith(key) or key.endswith(name)]
            if len(matches) != 1:
                raise FedSpanContractError(
                    f"no unique PEFT alpha/r scale for module '{name}'")
            value = matches[0]
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"module scale for '{name}' must be positive and finite")
        resolved[name] = value
    return resolved


def _fedspan_blocks(client_states, broadcast_state, module_scales):
    """Validate fixed-A state and return PEFT-scaled raw-B delta blocks."""
    if not client_states:
        raise FedSpanContractError("FedSpan requires at least one client state")
    broadcast = _factor_entries(broadcast_state, "broadcast_state")
    names = sorted(broadcast)
    scales = _resolve_module_scales(names, module_scales)
    for name in names:
        A = broadcast[name]["A"][1]
        B = broadcast[name]["B"][1]
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise FedSpanContractError(
                f"broadcast_state has nonfinite factor values in '{name}'")

    blocks = []
    for client_index, state in enumerate(client_states):
        client = _factor_entries(state, f"client_states[{client_index}]")
        if set(client) != set(names):
            missing = sorted(set(names) - set(client))
            extra = sorted(set(client) - set(names))
            raise FedSpanContractError(
                f"client_states[{client_index}] factor keys differ from the "
                f"broadcast (missing={missing}, extra={extra})")
        client_blocks = {}
        for name in names:
            A0 = broadcast[name]["A"][1]
            B0 = broadcast[name]["B"][1]
            A = client[name]["A"][1]
            B = client[name]["B"][1]
            if A.shape != A0.shape or B.shape != B0.shape:
                raise FedSpanContractError(
                    f"client_states[{client_index}] shape mismatch in '{name}'")
            # This is deliberately exact: set_adapter_state followed by frozen
            # local training must not alter a shared A by even one bit.
            if not torch.equal(A.detach().cpu(), A0.detach().cpu()):
                raise FedSpanContractError(
                    f"client_states[{client_index}] violates shared frozen A "
                    f"for module '{name}'")
            client_blocks[name] = scales[name] * (
                B.detach().cpu().double() - B0.detach().cpu().double())
        blocks.append(client_blocks)
    return broadcast, names, scales, blocks


def validate_frozen_a_states(client_states, broadcast_state, module_scales):
    """Fail if clients do not share the broadcast A bit-for-bit.

    Returns the resolved per-module PEFT scales. This is useful for all
    frozen-A comparator arms, not only normmaxmin.
    """
    _, _, scales, _ = _fedspan_blocks(
        client_states, broadcast_state, module_scales)
    return scales


def frozen_a_state_diagnostics(state, module_scales):
    """Record per-module A row-orthogonality error and geometry scale.

    The row constant ``c`` in ``A A^T = c^2 I`` is measured from the state
    itself rather than assumed, so the diagnostic is valid for both the
    unit-row and the PEFT-init-scaled frozen A.
    """
    modules = _factor_entries(state, "adapter_state")
    names = sorted(modules)
    scales = _resolve_module_scales(names, module_scales)
    records = {}
    for name in names:
        A = modules[name]["A"][1].detach().cpu().double()
        gram = A @ A.T
        c_squared = float(torch.mean(torch.diagonal(gram)).item())
        identity = torch.eye(A.shape[0], dtype=torch.float64)
        error = torch.max(torch.abs(gram - c_squared * identity)).item()
        records[name] = {
            "geometry_scale": scales[name],
            "a_row_scale_c": math.sqrt(max(c_squared, 0.0)),
            "a_row_orthonormal_max_abs_error": float(error),
        }
    return records


def apply_frozen_b_delta_weights(broadcast_state, client_states, coefficients,
                                 module_scales):
    """Apply arbitrary finite coefficients only to raw B deltas.

    Unlike generic state averaging, this copies the shared frozen A and its
    dtype bit-for-bit from the broadcast. The call also revalidates the full
    frozen-A factor contract before applying anything.
    """
    broadcast, names, _, blocks = _fedspan_blocks(
        client_states, broadcast_state, module_scales)
    coefficients = [float(value) for value in coefficients]
    if (len(coefficients) != len(client_states)
            or not all(math.isfinite(value) for value in coefficients)):
        raise FedSpanContractError(
            "frozen-B delta coefficients must contain one finite value per client")
    out = {key: value.detach().cpu().clone()
           for key, value in broadcast_state.items()}
    client_entries = [
        _factor_entries(state, f"client_states[{index}]")
        for index, state in enumerate(client_states)
    ]
    for name in names:
        b_key, base_b = broadcast[name]["B"]
        # Accumulate in float64. Coefficients can be inversely proportional to
        # very unequal client norms; float32 multiply-adds otherwise lose the
        # declared small true step even when all communicated factors are f32.
        base_b_cpu = base_b.detach().cpu()
        acc = base_b_cpu.double().clone()
        for index, coefficient in enumerate(coefficients):
            if coefficient == 0.0:
                continue
            if not torch.isfinite(blocks[index][name]).all():
                raise FedSpanContractError(
                    "a nonfinite client received a nonzero delta coefficient")
            client_b = client_entries[index][name]["B"][1]
            acc += coefficient * (
                client_b.detach().cpu().double() - base_b_cpu.double())
        # PEFT adapters are server-aggregated in float32 in this codebase.
        # The subsequent independent verification measures the cast error.
        out[b_key] = acc.float()
        a_key, base_a = broadcast[name]["A"]
        out[a_key] = base_a.detach().cpu().clone()
    return out


def _zero_fedspan_result(status, client_norms, active_mask,
                         inactive_reasons, step_norm, threshold,
                         step_policy="fixed", declared_step_norm=None,
                         solver_status=None, solver_message="",
                         fallback="zero_update", active_indices=None,
                         cosine_gram_active=None, simplex_weights=None,
                         gamma=None, mixture_norm=None, module_scales=None,
                         solver_objective_gamma=None,
                         solver_simplex_residual=None,
                         solver_constraint_violation=None,
                         proposed_delta_weights=None,
                         proposed_max_abs_delta_weight=None,
                         delta_weight_limit=None,
                         step_reconstruction_error=None,
                         direction_policy=None,
                         direction_policy_specified=None,
                         min_norm_value=None, min_norm_solver=None,
                         min_norm_value_source=None, exact_solver=None,
                         fixed_weights=None, wolfe_certificate_value=None,
                         achieved_min_direction_cosine=None,
                         frank_wolfe_value=None, frank_wolfe_converged=None,
                         exact_measurement_note=None,
                         applied_direction_solver=None,
                         shadow_sketch=None):
    K = len(client_norms)
    shortfall = (None if (min_norm_value is None
                          or achieved_min_direction_cosine is None)
                 else float(min_norm_value - achieved_min_direction_cosine))
    return {
        "status": status,
        "fallback": fallback,
        "direction_policy": direction_policy,
        "direction_policy_specified": direction_policy_specified,
        "fixed_weights": fixed_weights,
        "solver_status": solver_status,
        "solver_message": solver_message,
        "applied_direction_solver": applied_direction_solver,
        "frank_wolfe_value": frank_wolfe_value,
        "frank_wolfe_converged": frank_wolfe_converged,
        "exact_measurement_note": exact_measurement_note,
        **({"shadow_sketch": shadow_sketch}
           if shadow_sketch is not None else {}),
        "solver_objective_gamma": solver_objective_gamma,
        "solver_simplex_residual": solver_simplex_residual,
        "solver_constraint_violation": solver_constraint_violation,
        "step_policy": step_policy,
        "declared_step_norm": declared_step_norm,
        "resolved_step_norm": (None if step_norm is None
                               else float(step_norm)),
        # Compatibility alias retained for existing diagnostics consumers.
        "requested_step_norm": (None if step_norm is None
                                else float(step_norm)),
        "activity_threshold": float(threshold),
        "client_norms": client_norms,
        "active_mask": active_mask,
        "active_indices": active_indices or [],
        "inactive_reasons": inactive_reasons,
        "module_scales": module_scales,
        "cosine_gram_active": cosine_gram_active,
        "simplex_weights": simplex_weights or [0.0] * K,
        "delta_weights": [0.0] * K,
        "proposed_delta_weights": proposed_delta_weights,
        "gamma": gamma,
        "mixture_norm": mixture_norm,
        "achieved_min_direction_cosine": achieved_min_direction_cosine,
        "min_norm_value": min_norm_value,
        "min_norm_value_source": min_norm_value_source,
        "min_norm_solver": min_norm_solver,
        "exact_solver": exact_solver,
        "wolfe_certificate": wolfe_certificate_value,
        "direction_solver_shortfall": shortfall,
        # Deprecated alias for achieved_min_direction_cosine; the value was
        # never a certificate of optimality, only of what was applied.
        "certified_min_direction_cosine": achieved_min_direction_cosine,
        "max_abs_delta_weight": 0.0,
        "proposed_max_abs_delta_weight": proposed_max_abs_delta_weight,
        "delta_weight_limit": delta_weight_limit,
        "step_reconstruction_error": step_reconstruction_error,
        "solved_effective_step_sha256": None,
    }


def _effective_vector_sha256(blocks):
    digest = hashlib.sha256()
    for name in sorted(blocks):
        tensor = blocks[name].detach().cpu().double().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


# Ill-conditioned faces discarded by the most recent exact enumeration, so a
# round can record how much of the face lattice was numerically unusable
# rather than presenting a silently reduced search as a clean one.
_LAST_EXACT_REJECTED_FACES = []


# Above this active-client count the 2^K - 1 face enumeration stops being
# free. Every federation in this project is far below it (K = 4 in E0/E1,
# K = 4..5 in E3); the cap exists so a future large-K run degrades to the
# iterative value with a recorded note rather than hanging.
_EXACT_MEASUREMENT_MAX_CLIENTS = 12


class DirectionSolverError(RuntimeError):
    """The exact direction solver could not certify its own answer.

    Raised instead of returning an uncertified direction, so a caller either
    gets a solution that passed every stated invariant or fails closed.
    """


def _face_stationary_point(C, support, feasibility_tol, rejected=None):
    """Stationary point of ``w^T C w`` on one face, or ``None``.

    The stationarity and normalisation conditions are solved together as the
    augmented KKT system

        [ 2 C_S  -1 ] [ w_S ]   [ 0 ]
        [   1^T   0 ] [  mu ] = [ 1 ]

    in a least-squares sense, which stays correct when ``C_S`` is singular.
    The common ``C_S w = 1`` shortcut is deliberately avoided: it silently
    drops faces whose restricted Gram has a zero eigenvalue, exactly the clone
    and near-cancellation geometries this project cares about.

    Two identities of that system are enforced rather than assumed. Taking the
    inner product of the stationarity rows with ``w_S`` gives
    ``mu = 2 w_S^T C_S w_S``, and the last row gives ``1^T w_S = 1``; a
    solution that violates either is not a solution of the system this
    function documents, so it fails closed.
    """
    size = len(support)
    Cs = C[np.ix_(support, support)]
    system = np.zeros((size + 1, size + 1), dtype=np.float64)
    system[:size, :size] = 2.0 * Cs
    system[:size, size] = -1.0
    system[size, :size] = 1.0
    rhs = np.zeros(size + 1, dtype=np.float64)
    rhs[size] = 1.0
    solution, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    if not np.all(np.isfinite(solution)):
        return None
    residual = float(np.linalg.norm(system @ solution - rhs))
    if residual > feasibility_tol:
        return None                       # no stationary point on this face
    w_s = solution[:size]
    multiplier = float(solution[size])
    stationary_value = float(w_s @ Cs @ w_s)
    # mu = 2 w^T C_S w follows from the stationarity rows only up to
    # |w| * residual, since the identity is obtained by taking the inner
    # product of those rows with w. A degenerate face can return a
    # large-norm least-squares solution whose residual is inside tolerance
    # while |w| * residual is not, so the bound is scaled by |w| rather
    # than held at an unscaled constant.
    identity_slack = max(1e-6 * max(1.0, abs(multiplier)),
                         4.0 * residual * float(np.linalg.norm(w_s)))
    if abs(multiplier - 2.0 * stationary_value) > identity_slack:
        # Reject THIS FACE, do not abort the enumeration. Discarding a face
        # can only make the returned answer worse, never wrong-and-unnoticed:
        # whatever survives is checked against the global, solver-independent
        # Wolfe certificate below, which raises if the winner is not in fact
        # the simplex minimiser. Raising here instead lost globally solvable
        # problems to a single ill-conditioned sub-face (measured on ~7% of
        # random well-formed unit-diagonal Grams).
        if rejected is not None:
            rejected.append(
                f"support={support}: KKT multiplier {multiplier} contradicts "
                f"2 w^T C w = {2.0 * stationary_value} "
                f"(residual {residual:.3e}, |w| {np.linalg.norm(w_s):.3e})")
        return None
    if w_s.min() < -feasibility_tol:
        return None                       # leaves the simplex; a sub-face wins
    w_s = np.clip(w_s, 0.0, None)
    total = float(w_s.sum())
    if total <= 0.0:
        return None
    if abs(total - 1.0) > max(2.0, size) * feasibility_tol:
        raise DirectionSolverError(
            f"stationary point carries simplex mass {total}")
    w_s = w_s / total
    if abs(float(w_s.sum()) - 1.0) > 1e-12:
        raise DirectionSolverError(
            f"normalised face weights sum to {float(w_s.sum())}")
    return w_s, float(w_s @ Cs @ w_s)


def _least_norm_optimum(C, target, optimal_value, feasibility_tol,
                        argmin_rcond):
    """Least-Euclidean-norm point of ``{w in simplex : C w = target}``.

    ``C`` is positive semidefinite, so two simplex points share a value and a
    gradient exactly when ``C(w - w') = 0``: the optimal set of
    ``min_w w^T C w`` is the affine slice of the simplex through any one
    optimum with ``C w`` held fixed. Minimising ``||w||_2`` over that convex
    set has a unique solution, and it is attained on the face equal to its own
    support, where the nonnegativity constraints are inactive and the
    least-norm least-squares solve is therefore the constrained minimiser.
    Enumerating supports and keeping the nonnegative solves is exact.
    """
    K = C.shape[0]
    scale = max(1.0, float(np.linalg.norm(target)))
    value_slack = 1e-9 * max(1.0, abs(optimal_value))
    best_w, best_norm = None, math.inf
    rhs = np.concatenate([np.asarray(target, dtype=np.float64), [1.0]])
    for mask in range(1, 1 << K):
        support = [index for index in range(K) if (mask >> index) & 1]
        block = np.vstack([C[:, support],
                           np.ones((1, len(support)), dtype=np.float64)])
        w_s, *_ = np.linalg.lstsq(block, rhs, rcond=argmin_rcond)
        if not np.all(np.isfinite(w_s)):
            continue
        if np.linalg.norm(block @ w_s - rhs) > feasibility_tol * scale:
            continue
        if w_s.min() < -feasibility_tol:
            continue
        w_s = np.clip(w_s, 0.0, None)
        total = float(w_s.sum())
        if total <= 0.0:
            continue
        w_s = w_s / total
        candidate = np.zeros(K, dtype=np.float64)
        candidate[support] = w_s
        if float(candidate @ C @ candidate) > optimal_value + value_slack:
            continue
        norm2 = float(candidate @ candidate)
        if norm2 < best_norm:
            best_norm, best_w = norm2, candidate
    if best_w is None:
        raise DirectionSolverError(
            "no simplex point reproduces the optimal alignment profile")
    return best_w


def minnorm_exact_weights(cosine, feasibility_tol=1e-9, argmin_rcond=1e-10):
    """Exact minimiser of w^T C w over the simplex, by face enumeration.

    At deployment scale (K <= 10) the 2^K - 1 faces are cheap to enumerate, so
    the direction can be solved exactly and no iterative-convergence caveat
    survives. Each face is solved through the augmented KKT system described
    in ``_face_stationary_point``.

    TIE-BREAK, part of the contract. When the Gram is singular -- three
    near-duplicate clients make it so -- the minimiser is not unique, and the
    whole optimal set gives the same value. This function returns the optimum
    of least Euclidean norm, which is unique because the optimal set is convex
    and the norm is strictly convex. Concretely, for exactly duplicated
    clients the shared mass is split evenly between them, so the returned
    weight vector is a property of the problem rather than of the linear
    algebra backend: without the rule, the tie-break falls to the backend's
    minimum-norm least-squares solution of the augmented system, which
    minimises the joint norm of ``(w, mu)`` including the Lagrange multiplier.
    The multiplier is recovered after the fact and is not part of the choice.

    ``argmin_rcond`` is the explicit singular-value cutoff, relative to the
    largest singular value, at which the tie-break treats two client
    directions as the same direction. It is named and defaulted rather than
    left to the backend because the tie-break is otherwise discontinuous: a
    Gram a few units in the last place away from exactly singular has a unique
    minimiser that no float64 solve can locate, and the cutoff is what makes a
    clone block at cosine 1 - 1e-14 give the same answer as one at cosine 1.

    SCOPE OF THAT LAST GUARANTEE, measured. It holds for a Gram whose clone
    off-diagonals are exactly equal -- which is what an EXACT clone federation
    produces, E3's construction. It does NOT hold entrywise for a Gram formed
    from perturbed vectors, where the clone off-diagonals differ from each
    other in the last places: there the optimal set is genuinely a single
    point rather than a face, individual weights inside the clone block can
    move at the 1e-7 level between representations, and the cutoff cannot
    restore a symmetry the problem no longer has. The block TOTAL and the
    optimal value are stable in both regimes; E3's headline statistic is the
    block total, so it is inside the guaranteed regime, but a per-client
    weight read off a near-clone Gram is not.
    It only takes effect on singular values between machine precision and the
    cutoff; it never changes the optimal value, which is checked afterwards.

    Returns ``(w, value)`` with ``value = sqrt(min_w w^T C w)`` -- the
    attainable worst-case cosine of the normalised mixture (min-norm duality).
    Raises ``DirectionSolverError`` rather than returning an uncertified
    answer.
    """
    C = np.asarray(cosine, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("cosine must be a square matrix")
    K = C.shape[0]
    rejected_faces = []
    witness, best_val = None, math.inf
    for mask in range(1, 1 << K):
        support = [index for index in range(K) if (mask >> index) & 1]
        solved = _face_stationary_point(C, support, feasibility_tol,
                                        rejected=rejected_faces)
        if solved is None:
            continue
        w_s, value = solved
        if value < best_val:
            candidate = np.zeros(K, dtype=np.float64)
            candidate[support] = w_s
            best_val, witness = value, candidate
    if witness is None:
        raise DirectionSolverError(
            "no face of the simplex carried a stationary point; the Gram is "
            "not a well-scaled positive semidefinite matrix"
            + (f" ({len(rejected_faces)} faces rejected: "
               f"{rejected_faces[0]})" if rejected_faces else ""))
    w = _least_norm_optimum(C, C @ witness, best_val, feasibility_tol,
                            argmin_rcond)
    value = float(w @ C @ w)
    if abs(value - best_val) > 1e-9 * max(1.0, abs(best_val)):
        raise DirectionSolverError(
            f"least-norm optimum has value {value}, not {best_val}")
    if wolfe_certificate(C, w) > 1e-9 * max(1.0, abs(best_val)):
        raise DirectionSolverError(
            "least-norm optimum fails its own optimality certificate"
            + (f"; {len(rejected_faces)} ill-conditioned faces were rejected "
               f"during enumeration, first: {rejected_faces[0]}"
               if rejected_faces else ""))
    _LAST_EXACT_REJECTED_FACES.clear()
    _LAST_EXACT_REJECTED_FACES.extend(rejected_faces)
    return w, float(math.sqrt(max(best_val, 0.0)))


def wolfe_certificate(cosine, weights):
    """Max violation of the optimality condition (C w)_j >= w^T C w.

    Zero (to numerical tolerance) certifies that ``weights`` minimises
    w^T C w over the simplex, independently of which solver produced it.
    Behaviour-neutral, and recorded by ``fedspan_delta_weights`` for every
    round and every direction policy, so an iterative or fixed-weight round
    carries a measured optimality gap rather than an assumed one.
    """
    C = np.asarray(cosine, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    value = float(w @ C @ w)
    return float(max(0.0, np.max(value - C @ w)))


class CraftInfeasibleError(ValueError):
    """A CRAFT alignment profile that lies outside ``range(C)``."""


def craft_delta_coefficients(cosine, targets, reference=None,
                             rcond=1e-15, feasibility_tol=1e-8):
    """CRAFT-style equality projection (arXiv:2605.21317), in coefficient space.

    UNWIRED, EXPERIMENTAL. This is not a runnable arm: it has no command-line
    value, no call site in the driver, and is not one of the pre-registered
    baselines. It also returns a raw coefficient vector, not a simplex point —
    on real Grams its entries sum to well under one and individual entries can
    be negative — so wiring it would additionally require a normalisation and
    sign convention that is not defined anywhere in this project. Read it as a
    reference implementation of the closed form, nothing more.

    CRAFT prescribes the alignment profile and projects a reference direction
    onto the equality constraints:

        min ||g - g_hat||^2  s.t.  U g = rho        (their Eq. 4.3)
        g = g_hat + U^+ (rho - U g_hat)             (their Eq. 4.5)

    Writing g = sum_k v_k u_k and g_hat = sum_k a_k u_k, both U g = C v and
    U^+ act in coefficient space through the pseudo-inverse of the Gram, so
    the closed form becomes v = a + C^+ (rho - C a).

    DEVIATION, recorded deliberately: their reference (Eq. 4.6) is the previous
    round's normalised global update, which is not expressible in the current
    round's client span; we use the uniform mixture of the current directions
    unless a reference coefficient vector is supplied. The differentiator this
    arm exists to measure — a prescribed data-proportional profile versus an
    endogenously maximised one — does not depend on that choice.

    ``C v = rho`` is solvable only when ``rho`` lies in ``range(C)``, because
    ``C C^+`` is the projector onto that range. Duplicated clients force
    ``(C v)_i = (C v)_j``, so any profile assigning them different targets is
    unsatisfiable by construction and the closed form quietly returns the
    least-squares relaxation instead. ``feasibility_tol`` is measured against
    ``||C v - rho|| / max(1, ||rho||)`` and a violation raises
    ``CraftInfeasibleError``; pass ``None`` to accept the relaxation
    deliberately.

    ``rcond`` is the pseudo-inverse's singular-value cutoff, relative to the
    largest singular value. It is a named argument because it is not a
    detail: on a near-clone Gram the returned coefficients scale by orders of
    magnitude across the plausible range of cutoffs. The default reproduces
    NumPy's own ``pinv`` default.
    """
    C = np.asarray(cosine, dtype=np.float64)
    K = C.shape[0]
    rho = np.asarray(targets, dtype=np.float64)
    if rho.shape != (K,):
        raise ValueError("targets must supply one alignment per client")
    a = (np.ones(K, dtype=np.float64) / K if reference is None
         else np.asarray(reference, dtype=np.float64))
    if a.shape != (K,):
        raise ValueError("reference must supply one coefficient per client")
    correction = np.linalg.pinv(C, rcond=rcond) @ (rho - C @ a)
    v = a + correction
    if feasibility_tol is not None:
        scale = max(1.0, float(np.linalg.norm(rho)))
        residual = float(np.linalg.norm(C @ v - rho)) / scale
        if residual > float(feasibility_tol):
            raise CraftInfeasibleError(
                f"alignment profile is not in range(C): relative residual "
                f"{residual:.3e} exceeds {float(feasibility_tol):.3e}; the "
                "closed form would return a least-squares relaxation of the "
                "equality-constrained program")
    return v


DIRECTION_POLICIES = ("minnorm", "maxmin-lp", "exact", "fixed")


def _validate_fixed_weights(fixed_weights, direction_policy, client_count):
    """Normalise and check caller-supplied simplex weights, or ``None``.

    Supplied weights are a declared experimental arm, so a malformed vector is
    a specification error and raises rather than degrading to something the
    run would then report under the arm's name.
    """
    if direction_policy != "fixed":
        if fixed_weights is not None:
            raise ValueError(
                "fixed_weights is legal only with direction_policy 'fixed'")
        return None
    if fixed_weights is None:
        raise ValueError(
            "direction_policy 'fixed' requires fixed_weights")
    values = [float(value) for value in fixed_weights]
    if len(values) != client_count:
        raise ValueError(
            f"fixed_weights must supply one weight per client "
            f"({client_count}), got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("fixed_weights must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("fixed_weights must be nonnegative")
    if sum(values) <= 0.0:
        raise ValueError("fixed_weights must carry positive mass")
    return values


def _validate_shadow_sketch_config(shadow_sketch):
    """Fail-fast config check; the caller contract, not runtime diagnostics."""
    if not isinstance(shadow_sketch, dict) or "seed" not in shadow_sketch:
        raise ValueError("shadow_sketch requires {'sizes': ..., 'seed': int}")
    sizes = list(shadow_sketch.get("sizes") or [])
    if not sizes or any((not isinstance(m, (int, np.integer)) or m < 8
                         or m > (1 << 20)) for m in sizes):
        raise ValueError(
            "shadow_sketch sizes must be ints in [8, 1048576]")
    if sorted(sizes) != sizes or len(set(sizes)) != len(sizes):
        raise ValueError("shadow_sketch sizes must be strictly ascending")
    return [int(m) for m in sizes], int(shadow_sketch["seed"])


def _shadow_sketch_record(blocks, names, active, active_norms, cosine,
                          exact_optimum, sizes, seed, chunk_columns=65536):
    """What an m-dimensional Gaussian sketch WOULD have done this round.

    Feasibility instrument for a secure-aggregation-compatible Gram
    (supervisor note 2026-08-31): every client projects its effective update
    through the SAME per-round Gaussian matrix -- pinned by test: exact clones
    must sketch to cosine exactly 1 -- and the direction is solved on the
    sketched Gram. Only RECORDED here; the applied weights never touch it,
    and the bit-identity of the applied path with telemetry on and off is
    itself under test.

    Never raises. The applied path is fail-closed; this is a counterfactual
    diagnostic, and a telemetry crash that aborted a pre-registered run would
    create pressure to strip telemetry mid-experiment -- worse for integrity
    than a loudly-recorded gap. Failures land in the record and on stdout.

    The nested-prefix trick makes all sizes one pass: rows of a Gaussian
    sketch are iid, so the first m coordinates of the m_max-dim sketch ARE an
    m-dim Gaussian sketch.
    """
    import time
    started = time.perf_counter()
    record = {"sizes": [int(m) for m in sizes], "seed": int(seed),
              "numpy_version": np.__version__, "per_size": {}}
    try:
        m_max = max(sizes)
        vectors = []
        for index in active:
            parts = [np.asarray(blocks[index][name].detach().cpu(),
                                dtype=np.float32).ravel() for name in names]
            vectors.append(np.concatenate(parts))
        X = np.stack(vectors)                          # (M, D) float32
        M, D = X.shape
        rng = np.random.default_rng(seed)
        Y = np.zeros((M, m_max), dtype=np.float32)
        for start in range(0, D, chunk_columns):
            stop = min(start + chunk_columns, D)
            S = rng.standard_normal((m_max, stop - start), dtype=np.float32)
            Y += X[:, start:stop] @ S.T
        C_true = np.asarray(cosine, dtype=np.float64)
        norms_true = np.asarray(active_norms, dtype=np.float64)
        for m in sizes:
            entry = {}
            try:
                Ys = Y[:, :m].astype(np.float64) / math.sqrt(m)
                r_hat = np.linalg.norm(Ys, axis=1)
                if r_hat.min() <= 0.0:
                    raise DirectionSolverError(
                        "sketched norm collapsed to zero")
                C_hat = (Ys @ Ys.T) / np.outer(r_hat, r_hat)
                C_hat = 0.5 * (C_hat + C_hat.T)
                np.fill_diagonal(C_hat, 1.0)
                w_hat, value_hat = minnorm_exact_weights(C_hat)
                mix = float(w_hat @ C_true @ w_hat)
                if mix <= 0.0:
                    raise DirectionSolverError(
                        "sketched direction cancels in the true geometry")
                gamma_true = float(np.min(C_true @ w_hat)) / math.sqrt(mix)
                entry = {
                    "sketched_norms": [float(v) for v in r_hat],
                    "sketched_cosine_gram": C_hat.tolist(),
                    "weights": [float(v) for v in w_hat],
                    "value_sketched": float(value_hat),
                    "gamma_true_of_sketched_direction": gamma_true,
                    "shortfall_vs_exact": (
                        None if exact_optimum is None
                        else float(exact_optimum - gamma_true)),
                    "gram_max_abs_err": float(np.max(np.abs(C_hat - C_true))),
                    "norm_max_rel_err": float(np.max(
                        np.abs(r_hat - norms_true) / norms_true)),
                }
            except (DirectionSolverError, np.linalg.LinAlgError,
                    ValueError) as exc:
                entry = {"failed": f"{type(exc).__name__}: {exc}"}
                print(f"  WARNING shadow sketch m={m} failed (recorded, "
                      f"run continues): {exc}")
            record["per_size"][str(m)] = entry
    except Exception as exc:                # noqa: BLE001 - diagnostics only
        record["failed"] = f"{type(exc).__name__}: {exc}"
        print(f"  WARNING shadow sketch failed entirely (recorded, run "
              f"continues): {exc}")
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def fedspan_delta_weights(client_states, broadcast_state, module_scales,
                          step_norm=None, step_policy="fixed",
                          direction_policy=None, fixed_weights=None,
                          active_abs_tol=1e-12,
                          active_rel_tol=1e-8, mixture_norm_tol=1e-6,
                          max_abs_delta_weight=None, shadow_sketch=None):
    """Frozen-A, norm-consistent worst-case-cosine aggregation coefficients.

    Geometry is formed from concatenated effective PEFT blocks
    ``sigma_l * (Braw_k,l - Braw_global,l)``. The direction is solved only
    over finite, non-tiny clients. If ``w`` is the simplex solution and
    ``r_k`` the effective client norm, the raw-B delta coefficient is

      c_k = resolved_step_norm * w_k / (r_k * sqrt(w.T @ H @ w)),

    which is degree-0 homogeneous in ``w``, so every direction policy flows
    through identical downstream machinery.

    ``direction_policy``:
      * ``"minnorm"``   — maximize the worst-case cosine of the APPLIED
        (normalized) direction, i.e. minimize ``w^T C w`` over the simplex.
        This is FedMGDA+ (arXiv:2006.11489) at epsilon = 1, run on the cosine
        Gram of the effective client directions.
      * ``"maxmin-lp"`` — maximize ``min_i (C w)_i`` over the simplex. The
        applied direction is normalized, so this LP optimizes a quantity that
        is not the one applied; it is retained as a recorded ablation.
      * ``"exact"`` — the same objective as ``"minnorm"``, solved by face
        enumeration with a declared minimum-norm tie-break instead of
        iteratively. It carries no convergence caveat and, unlike Frank-Wolfe,
        returns a weight vector that is a property of the Gram rather than of
        the iterate path, which matters when the Gram is degenerate.
      * ``"fixed"`` — skip the direction solve and use the caller's
        ``fixed_weights``, restricted to the active clients and renormalized.
        Every downstream stage is shared with the solved policies: the same
        ``1/r_k`` coefficient rule, the same step policy, the same fail-closed
        gates. A fixed arm is therefore step-matched and norm-matched to a
        solved arm and differs from it only in ``w``, which is what makes a
        contrast between them attribute an effect to the weight vector alone.
    ``min_norm_value`` is the attainable optimum, measured every round
    regardless of policy (exact when ``min_norm_value_source`` is the face
    enumeration or when ``min_norm_solver["converged"]``, an upper bound
    otherwise), ``direction_solver_shortfall`` is what the chosen policy gave
    up against it, and ``wolfe_certificate`` is the largest violation of
    ``(C w)_j >= w^T C w`` by the weights actually applied — zero exactly when
    they minimize ``w^T C w``, whatever produced them.

    Solver errors, invalid solutions, near cancellation, and an optional
    coefficient-limit violation fail closed to a zero server update. Contract
    violations such as trainable/non-shared A or malformed factor states raise
    ``FedSpanContractError`` and invalidate the run rather than falling back.
    """
    direction_policy_specified = direction_policy is not None
    if not direction_policy_specified:
        direction_policy = "maxmin-lp"
    if direction_policy not in DIRECTION_POLICIES:
        raise ValueError(
            "direction_policy must be one of "
            + ", ".join(repr(name) for name in DIRECTION_POLICIES))
    fixed_weights = _validate_fixed_weights(
        fixed_weights, direction_policy, len(client_states))
    if step_policy not in ("fixed", "median-active"):
        raise ValueError(
            "step_policy must be 'fixed' or 'median-active'")
    shadow_sizes = shadow_seed = None
    if shadow_sketch is not None:
        shadow_sizes, shadow_seed = _validate_shadow_sketch_config(
            shadow_sketch)
    if step_policy == "fixed":
        if (step_norm is None or not math.isfinite(float(step_norm))
                or float(step_norm) <= 0):
            raise ValueError(
                "fixed step policy requires a positive finite step_norm")
        declared_step_norm = float(step_norm)
    else:
        if step_norm is not None:
            raise ValueError("median-active step policy rejects step_norm")
        declared_step_norm = None
    for name, value in (("active_abs_tol", active_abs_tol),
                        ("active_rel_tol", active_rel_tol),
                        ("mixture_norm_tol", mixture_norm_tol)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (max_abs_delta_weight is not None
            and (not math.isfinite(float(max_abs_delta_weight))
                 or float(max_abs_delta_weight) <= 0)):
        raise ValueError(
            "max_abs_delta_weight must be positive and finite when set")

    _, names, scales, blocks = _fedspan_blocks(
        client_states, broadcast_state, module_scales)
    K = len(blocks)
    client_norms = []
    finite = []
    for client_blocks in blocks:
        is_finite = all(torch.isfinite(client_blocks[name]).all().item()
                        for name in names)
        finite.append(is_finite)
        if is_finite:
            norm2 = sum(torch.sum(client_blocks[name] ** 2).item()
                        for name in names)
            client_norms.append(float(math.sqrt(max(norm2, 0.0))))
        else:
            client_norms.append(None)
    largest = max((value for value in client_norms if value is not None),
                  default=0.0)
    threshold = max(float(active_abs_tol), float(active_rel_tol) * largest)
    active_mask = []
    inactive_reasons = []
    for is_finite, norm in zip(finite, client_norms):
        if not is_finite:
            active_mask.append(False)
            inactive_reasons.append("nonfinite_delta")
        elif norm <= threshold:
            active_mask.append(False)
            inactive_reasons.append("zero_or_tiny_delta")
        else:
            active_mask.append(True)
            inactive_reasons.append(None)
    active = [index for index, flag in enumerate(active_mask) if flag]
    if not active:
        return _zero_fedspan_result(
            "no_active", client_norms, active_mask, inactive_reasons,
            declared_step_norm, threshold, step_policy=step_policy,
            direction_policy=direction_policy,
            direction_policy_specified=direction_policy_specified,
            declared_step_norm=declared_step_norm,
            active_indices=[], module_scales=scales,
            delta_weight_limit=max_abs_delta_weight)

    resolved_step_norm = (
        declared_step_norm
        if step_policy == "fixed"
        else float(np.median([client_norms[index] for index in active])))
    if (not math.isfinite(resolved_step_norm)
            or resolved_step_norm <= 0):
        return _zero_fedspan_result(
            "invalid_step_norm", client_norms, active_mask,
            inactive_reasons, resolved_step_norm, threshold,
            step_policy=step_policy,
            direction_policy=direction_policy,
            direction_policy_specified=direction_policy_specified,
            declared_step_norm=declared_step_norm,
            active_indices=active, module_scales=scales,
            solver_message="resolved step norm is nonpositive or nonfinite",
            delta_weight_limit=max_abs_delta_weight)
    step_norm = resolved_step_norm

    M = len(active)
    gram = np.zeros((M, M), dtype=np.float64)
    for ai, i in enumerate(active):
        for aj in range(ai, M):
            j = active[aj]
            ip = sum(torch.sum(blocks[i][name] * blocks[j][name]).item()
                     for name in names)
            gram[ai, aj] = gram[aj, ai] = ip
    active_norms = np.asarray([client_norms[index] for index in active],
                              dtype=np.float64)
    cosine = gram / np.outer(active_norms, active_norms)
    cosine = 0.5 * (cosine + cosine.T)
    np.fill_diagonal(cosine, 1.0)
    cosine_list = cosine.tolist()

    # The attainable optimum is measured for EVERY policy every round, so the
    # shortfall of whichever policy is applied is always on the record.
    min_norm_w, min_norm_info = _min_norm_simplex_weights(cosine)
    min_norm_value = math.sqrt(max(
        float(min_norm_w @ cosine @ min_norm_w), 0.0))
    exact_solver_info = None
    min_norm_value_source = "away-step-frank-wolfe"

    # The attainable optimum is what EVERY arm's shortfall is measured
    # against, so it must not itself come from a solver that stalled. At
    # deployment scale the exact enumeration costs 2^K - 1 tiny linear solves
    # (15 at K=4) against a full round of GPU training, so it is measured
    # every round for every policy -- including the arms that do not apply it.
    # Without this, a fixed arm's recorded distance from optimal on a clone
    # federation was its distance from a STALLED Frank-Wolfe iterate, and
    # E3's clone Grams are singular by construction, i.e. exactly where the
    # iterative solver stalls. The applied direction is untouched: this
    # changes what is recorded, never what is sent.
    measured_optimum = None
    if M <= _EXACT_MEASUREMENT_MAX_CLIENTS:
        try:
            measured_w, measured_optimum = minnorm_exact_weights(cosine)
        except (DirectionSolverError, np.linalg.LinAlgError) as exc:
            measured_optimum = None
            exact_measurement_note = f"exact measurement unavailable: {exc}"
        else:
            exact_measurement_note = None
            min_norm_value = float(measured_optimum)
            min_norm_value_source = "exact-face-enumeration"
    else:
        exact_measurement_note = (
            f"exact measurement skipped: {M} clients exceeds the "
            f"{_EXACT_MEASUREMENT_MAX_CLIENTS}-client enumeration cap")

    if direction_policy == "exact":
        try:
            exact_w, exact_value = minnorm_exact_weights(cosine)
        except (DirectionSolverError, np.linalg.LinAlgError) as exc:
            return _zero_fedspan_result(
                "solver_error", client_norms, active_mask, inactive_reasons,
                step_norm, threshold, solver_status="exact",
                step_policy=step_policy,
                direction_policy=direction_policy,
                direction_policy_specified=direction_policy_specified,
                declared_step_norm=declared_step_norm,
                solver_message=f"{type(exc).__name__}: {exc}",
                active_indices=active, cosine_gram_active=cosine_list,
                module_scales=scales, min_norm_value=min_norm_value,
                min_norm_solver=min_norm_info,
                min_norm_value_source=min_norm_value_source,
                fixed_weights=fixed_weights,
                delta_weight_limit=max_abs_delta_weight)
        exact_solver_info = {
            "algorithm": "face-enumeration-least-norm-argmin/v1",
            "value": float(exact_value),
            "weights": [float(value) for value in exact_w],
            "wolfe_certificate": float(wolfe_certificate(cosine, exact_w)),
            # How much of the 2^K - 1 face lattice was numerically unusable
            # this round. The answer is still certified optimal by the Wolfe
            # gap above; this records that the search was reduced, so a run
            # cannot present a shrunken enumeration as a full one.
            "rejected_faces": len(_LAST_EXACT_REJECTED_FACES),
            "first_rejected_face": (_LAST_EXACT_REJECTED_FACES[0]
                                    if _LAST_EXACT_REJECTED_FACES else None),
        }
        # The face enumeration is exact, so it, not the iterative solver,
        # defines the attainable optimum whenever it has been run.
        min_norm_value = float(exact_value)
        min_norm_value_source = "exact-face-enumeration"
    direction_telemetry = {
        "min_norm_value": min_norm_value,
        "min_norm_solver": min_norm_info,
        "min_norm_value_source": min_norm_value_source,
        "exact_solver": exact_solver_info,
        "fixed_weights": fixed_weights,
        # The iterative solver's own record is kept alongside the exact
        # measurement rather than replaced by it, so a round says both what
        # the optimum is and whether Frank-Wolfe found it. E1 ran before this
        # measurement existed and its files record the Frank-Wolfe value.
        "frank_wolfe_value": float(math.sqrt(max(
            float(np.asarray(min_norm_w) @ cosine @ np.asarray(min_norm_w)),
            0.0))),
        "frank_wolfe_converged": bool(min_norm_info.get("converged")),
        "exact_measurement_note": exact_measurement_note,
        # WHICH SOLVER PRODUCED THE WEIGHTS THAT WERE ACTUALLY SENT. Kept
        # distinct from min_norm_value_source, which describes only where the
        # recorded attainable optimum came from. The two answer different
        # questions, and reading one as the other is precisely how a solver
        # that was implemented but unreachable got reported as deployed
        # (supervisor correction, 2026-08-28). Carried on every return path,
        # including every fail-closed one.
        "applied_direction_solver": {
            "exact": "face-enumeration-least-norm-argmin/v1",
            "minnorm": "away-step-frank-wolfe",
            "maxmin-lp": "scipy-linprog",
            "fixed": "declared-weights-no-solve",
        }[direction_policy],
    }

    if shadow_sizes is not None:
        direction_telemetry["shadow_sketch"] = _shadow_sketch_record(
            blocks, names, active, active_norms, cosine, min_norm_value,
            sizes=shadow_sizes, seed=shadow_seed)

    if direction_policy == "fixed":
        active_w = np.asarray([fixed_weights[index] for index in active],
                              dtype=np.float64)
        supplied_mass = float(active_w.sum())
        if supplied_mass <= 0.0:
            return _zero_fedspan_result(
                "fixed_weights_inactive", client_norms, active_mask,
                inactive_reasons, step_norm, threshold,
                solver_status="fixed", step_policy=step_policy,
                direction_policy=direction_policy,
                direction_policy_specified=direction_policy_specified,
                declared_step_norm=declared_step_norm,
                solver_message=(
                    "supplied fixed weights put no mass on any active client"),
                active_indices=active, cosine_gram_active=cosine_list,
                module_scales=scales, **direction_telemetry,
                delta_weight_limit=max_abs_delta_weight)
        active_w = active_w / supplied_mass
        solver_status = 0
        solver_message = (
            f"caller-supplied direction weights, {supplied_mass:.12g} of the "
            f"declared mass on the {M} active clients")
        solver_objective_gamma = float(np.min(cosine @ active_w))
        solver_simplex_residual = float(abs(active_w.sum() - 1.0))
        solver_constraint_violation = float(max(
            0.0, np.max(solver_objective_gamma - cosine @ active_w)))
        status = "fixed"
    elif M == 1:
        active_w = np.ones(1, dtype=np.float64)
        solver_status = 0
        solver_message = "singleton active set; no direction solve required"
        status = "singleton"
        solver_objective_gamma = 1.0
        solver_simplex_residual = 0.0
        solver_constraint_violation = 0.0
    elif direction_policy == "exact":
        active_w = np.asarray(exact_solver_info["weights"], dtype=np.float64)
        solver_status = 0
        solver_message = (
            f"exact face enumeration over {2 ** M - 1} faces, least-norm "
            f"tie-break, Wolfe violation "
            f"{exact_solver_info['wolfe_certificate']:.3e}")
        solver_objective_gamma = float(np.min(cosine @ active_w))
        solver_simplex_residual = float(abs(active_w.sum() - 1.0))
        solver_constraint_violation = float(max(
            0.0, np.max(solver_objective_gamma - cosine @ active_w)))
        status = "optimal"
    elif direction_policy == "minnorm":
        active_w = np.asarray(min_norm_w, dtype=np.float64)
        if (not np.all(np.isfinite(active_w)) or active_w.min() < -1e-9
                or abs(active_w.sum() - 1.0) > 1e-9):
            return _zero_fedspan_result(
                "solver_invalid", client_norms, active_mask, inactive_reasons,
                step_norm, threshold, solver_status="minnorm",
                step_policy=step_policy,
                direction_policy=direction_policy,
                direction_policy_specified=direction_policy_specified,
                declared_step_norm=declared_step_norm,
                solver_message=(
                    "min-norm Frank-Wolfe returned a nonsimplex point"),
                active_indices=active, cosine_gram_active=cosine_list,
                module_scales=scales, **direction_telemetry,
                delta_weight_limit=max_abs_delta_weight)
        active_w = np.clip(active_w, 0.0, 1.0)
        active_w = active_w / active_w.sum()
        solver_status = 0
        solver_message = (
            f"away-step Frank-Wolfe "
            f"{'converged' if min_norm_info['converged'] else 'STALLED'} at "
            f"duality gap {min_norm_info['gap']:.3e} after "
            f"{min_norm_info['iterations']} iterations")
        solver_objective_gamma = float(np.min(cosine @ active_w))
        solver_simplex_residual = float(abs(active_w.sum() - 1.0))
        solver_constraint_violation = float(max(
            0.0, np.max(solver_objective_gamma - cosine @ active_w)))
        # "optimal" is a claim about the APPLIED weights. On this arm they are
        # the Frank-Wolfe iterate, so the claim is only as good as that
        # solver's convergence -- and a near-clone Gram, which is what E3 is
        # built from, is exactly where away-step FW stalls. Labelling a
        # stalled iterate "optimal" would put an unearned word next to a
        # measured 1.6e-06 shortfall.
        status = ("optimal" if min_norm_info.get("converged") else "stalled")
    else:
        from scipy.optimize import linprog
        objective = np.zeros(M + 1, dtype=np.float64)
        objective[-1] = -1.0
        try:
            solved = linprog(
                objective,
                A_ub=np.hstack([-cosine, np.ones((M, 1))]),
                b_ub=np.zeros(M),
                A_eq=[([1.0] * M) + [0.0]], b_eq=[1.0],
                bounds=[(0.0, 1.0)] * M + [(None, None)], method="highs")
        except Exception as exc:
            return _zero_fedspan_result(
                "solver_error", client_norms, active_mask, inactive_reasons,
                step_norm, threshold, solver_status="exception",
                step_policy=step_policy,
                direction_policy=direction_policy,
                direction_policy_specified=direction_policy_specified,
                declared_step_norm=declared_step_norm,
                solver_message=f"{type(exc).__name__}: {exc}",
                active_indices=active, cosine_gram_active=cosine_list,
                **direction_telemetry,
                module_scales=scales,
                delta_weight_limit=max_abs_delta_weight)
        if not solved.success:
            return _zero_fedspan_result(
                "solver_failure", client_norms, active_mask, inactive_reasons,
                step_norm, threshold, solver_status=int(solved.status),
                step_policy=step_policy,
                direction_policy=direction_policy,
                direction_policy_specified=direction_policy_specified,
                declared_step_norm=declared_step_norm,
                solver_message=str(solved.message), active_indices=active,
                cosine_gram_active=cosine_list, module_scales=scales,
                **direction_telemetry,
                delta_weight_limit=max_abs_delta_weight)
        raw_w = np.asarray(solved.x[:M], dtype=np.float64)
        raw_gamma = float(solved.x[-1])
        if (not np.all(np.isfinite(raw_w)) or not math.isfinite(raw_gamma)
                or raw_w.min() < -1e-7
                or raw_w.max() > 1.0 + 1e-7
                or abs(raw_w.sum() - 1.0) > 1e-7):
            return _zero_fedspan_result(
                "solver_invalid", client_norms, active_mask, inactive_reasons,
                step_norm, threshold, solver_status=int(solved.status),
                step_policy=step_policy,
                direction_policy=direction_policy,
                direction_policy_specified=direction_policy_specified,
                declared_step_norm=declared_step_norm,
                solver_message="LP returned an infeasible/nonfinite simplex point",
                active_indices=active, cosine_gram_active=cosine_list,
                **direction_telemetry,
                module_scales=scales,
                delta_weight_limit=max_abs_delta_weight)
        solver_objective_gamma = raw_gamma
        solver_simplex_residual = float(abs(raw_w.sum() - 1.0))
        solver_constraint_violation = float(max(
            0.0, np.max(raw_gamma - cosine @ raw_w)))
        active_w = np.clip(raw_w, 0.0, 1.0)
        active_w /= active_w.sum()
        solver_status = int(solved.status)
        solver_message = str(solved.message)
        status = "optimal"

    # Measured for every policy: zero certifies that the weights actually
    # applied minimize w^T C w, whatever produced them, so a Frank-Wolfe or
    # fixed round carries a measured optimality gap rather than an assumption.
    applied_certificate = float(wolfe_certificate(cosine, active_w))
    direction_telemetry["wolfe_certificate_value"] = applied_certificate

    payoffs = cosine @ active_w
    gamma = float(np.min(payoffs))
    mixture_sq = float(active_w @ cosine @ active_w)
    mixture_norm = math.sqrt(max(mixture_sq, 0.0))
    achieved_cosine = (None if mixture_norm <= 0.0
                       else float(gamma / mixture_norm))
    simplex = [0.0] * K
    for local_index, client_index in enumerate(active):
        simplex[client_index] = float(active_w[local_index])
    if not math.isfinite(mixture_norm) or mixture_norm <= mixture_norm_tol:
        return _zero_fedspan_result(
            "near_cancellation", client_norms, active_mask, inactive_reasons,
            step_norm, threshold, solver_status=solver_status,
            step_policy=step_policy,
            direction_policy=direction_policy,
            direction_policy_specified=direction_policy_specified,
            declared_step_norm=declared_step_norm,
            solver_message=solver_message, active_indices=active,
            cosine_gram_active=cosine_list, simplex_weights=simplex,
            **direction_telemetry,
            gamma=gamma, mixture_norm=mixture_norm, module_scales=scales,
            solver_objective_gamma=solver_objective_gamma,
            solver_simplex_residual=solver_simplex_residual,
            solver_constraint_violation=solver_constraint_violation,
            delta_weight_limit=max_abs_delta_weight)

    coefficients = [0.0] * K
    for local_index, client_index in enumerate(active):
        coefficients[client_index] = float(
            step_norm * active_w[local_index]
            / (active_norms[local_index] * mixture_norm))
    coefficient_max = max(abs(value) for value in coefficients)
    if (max_abs_delta_weight is not None
            and coefficient_max > float(max_abs_delta_weight)):
        return _zero_fedspan_result(
            "coefficient_limit", client_norms, active_mask, inactive_reasons,
            step_norm, threshold, solver_status=solver_status,
            step_policy=step_policy,
            direction_policy=direction_policy,
            direction_policy_specified=direction_policy_specified,
            declared_step_norm=declared_step_norm,
            solver_message=solver_message, active_indices=active,
            cosine_gram_active=cosine_list, simplex_weights=simplex,
            **direction_telemetry,
            gamma=gamma, mixture_norm=mixture_norm, module_scales=scales,
            solver_objective_gamma=solver_objective_gamma,
            solver_simplex_residual=solver_simplex_residual,
            solver_constraint_violation=solver_constraint_violation,
            achieved_min_direction_cosine=achieved_cosine,
            proposed_delta_weights=coefficients,
            proposed_max_abs_delta_weight=coefficient_max,
            delta_weight_limit=max_abs_delta_weight)

    solved_blocks = {
        name: sum(coefficients[index] * blocks[index][name]
                  for index in active)
        for name in names
    }
    solved_norm = math.sqrt(sum(torch.sum(value ** 2).item()
                                for value in solved_blocks.values()))
    if abs(solved_norm - step_norm) > 1e-9 * max(1.0, step_norm):
        return _zero_fedspan_result(
            "reconstruction_failure", client_norms, active_mask,
            inactive_reasons, step_norm, threshold,
            step_policy=step_policy,
            direction_policy=direction_policy,
            direction_policy_specified=direction_policy_specified,
            declared_step_norm=declared_step_norm,
            solver_status=solver_status,
            solver_message=(
                f"coefficient reconstruction produced norm {solved_norm} "
                f"instead of {step_norm}"),
            active_indices=active, cosine_gram_active=cosine_list,
            **direction_telemetry,
            simplex_weights=simplex, gamma=gamma,
            mixture_norm=mixture_norm, module_scales=scales,
            solver_objective_gamma=solver_objective_gamma,
            solver_simplex_residual=solver_simplex_residual,
            solver_constraint_violation=solver_constraint_violation,
            achieved_min_direction_cosine=achieved_cosine,
            proposed_delta_weights=coefficients,
            proposed_max_abs_delta_weight=coefficient_max,
            delta_weight_limit=max_abs_delta_weight,
            step_reconstruction_error=solved_norm - step_norm)
    return {
        "status": status,
        "fallback": None,
        "direction_policy": direction_policy,
        "direction_policy_specified": direction_policy_specified,
        "solver_status": solver_status,
        "solver_message": solver_message,
        "solver_objective_gamma": solver_objective_gamma,
        "solver_simplex_residual": solver_simplex_residual,
        "solver_constraint_violation": solver_constraint_violation,
        "step_policy": step_policy,
        "declared_step_norm": declared_step_norm,
        "resolved_step_norm": step_norm,
        "requested_step_norm": step_norm,
        "activity_threshold": threshold,
        "client_norms": client_norms,
        "active_mask": active_mask,
        "active_indices": active,
        "inactive_reasons": inactive_reasons,
        "module_scales": scales,
        "cosine_gram_active": cosine_list,
        "simplex_weights": simplex,
        "delta_weights": coefficients,
        "proposed_delta_weights": coefficients,
        "gamma": gamma,
        "mixture_norm": mixture_norm,
        "achieved_min_direction_cosine": achieved_cosine,
        # Spread rather than re-listed: this dict and every fail-closed
        # return must carry the SAME direction telemetry, and hand-copying the
        # keys into both is how they drifted apart in the first place.
        **direction_telemetry,
        "wolfe_certificate": applied_certificate,
        "direction_solver_shortfall": (
            None if achieved_cosine is None
            else float(min_norm_value - achieved_cosine)),
        # Deprecated alias for achieved_min_direction_cosine; the value was
        # never a certificate of optimality, only of what was applied.
        "certified_min_direction_cosine": achieved_cosine,
        "max_abs_delta_weight": coefficient_max,
        "proposed_max_abs_delta_weight": coefficient_max,
        "delta_weight_limit": max_abs_delta_weight,
        "step_reconstruction_error": solved_norm - step_norm,
        "solved_effective_step_sha256": _effective_vector_sha256(solved_blocks),
    }


def apply_fedspan_update(broadcast_state, client_states, result,
                         module_scales, verify_atol=5e-6):
    """Apply a ``fedspan_delta_weights`` result to raw B deltas only.

    Shared A is copied bitwise from the broadcast. The applied effective step
    is reconstructed independently and compared with the solved coefficient
    mixture before the new state is returned.
    """
    broadcast, names, scales, blocks = _fedspan_blocks(
        client_states, broadcast_state, module_scales)
    K = len(client_states)
    coefficients = [float(value) for value in result.get("delta_weights", [])]
    if len(coefficients) != K or not all(math.isfinite(x) for x in coefficients):
        raise FedSpanContractError(
            "FedSpan delta_weights must contain one finite value per client")
    expected_scales = result.get("module_scales")
    if expected_scales is not None:
        for name in names:
            if float(expected_scales[name]) != scales[name]:
                raise FedSpanContractError(
                    f"module scale changed between solve and apply for '{name}'")

    out = apply_frozen_b_delta_weights(
        broadcast_state, client_states, coefficients, module_scales)

    expected = {
        name: sum(coefficients[index] * blocks[index][name]
                  for index in range(K) if coefficients[index] != 0.0)
        for name in names
    }
    actual = {
        name: scales[name] * (
            out[broadcast[name]["B"][0]].double()
            - broadcast[name]["B"][1].detach().cpu().double())
        for name in names
    }
    max_error = max((torch.max(torch.abs(actual[name] - expected[name])).item()
                     if actual[name].numel() else 0.0)
                    for name in names)
    applied_norm = math.sqrt(sum(torch.sum(value ** 2).item()
                                 for value in actual.values()))
    resolved_step_norm = result.get(
        "resolved_step_norm", result.get("requested_step_norm"))
    tolerance = float(verify_atol) * max(
        1.0, float(resolved_step_norm or 0.0))
    if max_error > tolerance:
        raise RuntimeError(
            f"applied FedSpan update differs from solved update: {max_error}")
    if result.get("fallback") is None:
        if resolved_step_norm is None:
            raise FedSpanContractError(
                "non-fallback FedSpan result lacks a resolved step norm")
        requested = float(resolved_step_norm)
        if abs(applied_norm - requested) > tolerance:
            raise RuntimeError(
                f"applied FedSpan norm {applied_norm} != requested {requested}")
    elif any(coefficients):
        raise FedSpanContractError("fallback result must apply a zero update")

    direction_cosines = [None] * K
    if applied_norm > 0:
        for index in result.get("active_indices", []):
            client_norm = result["client_norms"][index]
            dot = sum(torch.sum(blocks[index][name] * actual[name]).item()
                      for name in names)
            direction_cosines[index] = float(
                dot / (client_norm * applied_norm))
    active_cosines = [value for value in direction_cosines
                      if value is not None]

    return out, {
        "applied_step_norm": applied_norm,
        "max_effective_block_error": max_error,
        "applied_delta_weights": coefficients,
        "applied_direction_cosines": direction_cosines,
        "applied_min_active_cosine": (min(active_cosines)
                                      if active_cosines else None),
        "applied_effective_step_sha256": _effective_vector_sha256(actual),
        "broadcast_state_sha256": state_dict_sha256(broadcast_state),
        "client_state_sha256": [state_dict_sha256(state)
                                for state in client_states],
        "applied_state_sha256": state_dict_sha256(out),
    }
