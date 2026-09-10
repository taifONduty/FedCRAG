"""Round orchestration of the response-maxmin aggregation arm.

Design: research_workspace/supervisor/2026-09-08_solution_design_response_aggregation.md,
sections 3 and 4. The encoder is injected as ``encode(state, texts) -> unit rows`` so the
logic runs under test without a model; the driver passes a closure over the real model.
The candidate predictions run in torch (GPU when available) because a round predicts
several hundred weight vectors per client; the numpy functions in response_aggregation
are the reference the torch path is tested against.
"""
import numpy as np
import torch

from aggregation_schemes import (SchemeResult, apply_delta_weights, maxmin_weights,
                                 update_gram)
from response_aggregation import (candidate_grid, choose_applied, greedy_soup,
                                  magnitude_candidates, ndcg10, ndcg10_from_top,
                                  paired_lower_bound, pessimistic_gain, rank_candidates,
                                  simplex_lattice, uniform_subsets)

TOP_K = 10


def unit_vector(K, j):
    v = [0.0] * K
    v[j] = 1.0
    return v


def measure_dev(encode, state, dev, slices):
    """Encode every client's corpus and held-out queries under ``state`` and score them.
    Returns per slice: c (documents), q (queries), per_query nDCG@10, mean."""
    out = {}
    for s in slices:
        d = dev[s]
        c = np.asarray(encode(state, d["ctext"]), dtype=np.float64)
        q = np.asarray(encode(state, d["qtext"]), dtype=np.float64)
        per_query = ndcg10(q @ c.T, d["cids"], d["qids"], d["qrels"], k=TOP_K)
        out[s] = {"c": c, "q": q, "per_query": per_query,
                  "mean": float(per_query.mean()) if len(per_query) else float("nan")}
    return out


class _ClientTensors:
    """One client's base embeddings and stacked responses, resident on the device once."""

    def __init__(self, base_c, base_q, responses, device):
        dtype = torch.float64 if device == "cpu" else torch.float32
        self.c0 = torch.as_tensor(np.asarray(base_c), dtype=dtype, device=device)
        self.q0 = torch.as_tensor(np.asarray(base_q), dtype=dtype, device=device)
        self.rc = torch.stack([torch.as_tensor(np.asarray(r[0]), dtype=dtype, device=device)
                               for r in responses])
        self.rq = torch.stack([torch.as_tensor(np.asarray(r[1]), dtype=dtype, device=device)
                               for r in responses])

    def top_indices(self, v, take):
        w = torch.as_tensor(v, dtype=self.c0.dtype, device=self.c0.device)
        c_hat = self.c0 + torch.einsum("k,knd->nd", w, self.rc)
        q_hat = self.q0 + torch.einsum("k,knd->nd", w, self.rq)
        c_hat = c_hat / c_hat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        q_hat = q_hat / q_hat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sims = q_hat @ c_hat.T
        take = min(int(take), sims.shape[1])
        return torch.topk(sims, k=take, dim=1).indices.cpu().numpy()


def _client_tensors(base, responses, slices, device):
    return {s: _ClientTensors(base[s]["c"], base[s]["q"], responses[s], device)
            for s in slices}


def _predicted_per_query(base, responses, dev, slices, v, device="cpu", tensors=None):
    """Predicted per-query dev nDCG@10 per client at weights ``v`` (first-order model)."""
    tensors = tensors or _client_tensors(base, responses, slices, device)
    out = []
    for s in slices:
        top = tensors[s].top_indices(v, TOP_K + 1)
        d = dev[s]
        out.append(ndcg10_from_top(top, d["cids"], d["qids"], d["qrels"], k=TOP_K))
    return out


def _predicted_means(base, responses, dev, slices, v, device="cpu", tensors=None):
    """Predicted mean dev nDCG@10 per client at weights ``v`` (first-order model)."""
    return [float(pq.mean()) for pq in
            _predicted_per_query(base, responses, dev, slices, v, device, tensors)]


def update_geometry(round_broadcast, client_states):
    """Product-space geometry of the round's updates (loop document, section 3): the
    norms r_k of dW_k = B_k A_k - B_g A_g, their cosine Gram, and the unit-direction game
    (max-min cosine over the simplex) weights and value. The Gram is exact in factor space
    (``update_gram``); the game is the existing LP solver."""
    G = update_gram(client_states, round_broadcast, normalize=False, dtype=torch.float64)
    K = G.shape[0]
    norms = np.sqrt(np.clip(np.diag(G), 0.0, None))
    largest = float(norms.max()) if K else 0.0
    active = norms > 1e-6 * largest if largest > 0 else np.zeros(K, dtype=bool)
    cosine = np.eye(K)
    if active.any():
        safe = np.where(active, norms, 1.0)
        cosine = G / np.outer(safe, safe)
        cosine[~active, :] = 0.0
        cosine[:, ~active] = 0.0
        np.fill_diagonal(cosine, 1.0)
    game = maxmin_weights(client_states, round_broadcast)
    weights = [float(x) for x in game]
    value = float(min(cosine @ np.asarray(weights))) if active.any() else float("nan")
    return {"norms": [float(x) for x in norms], "cosine_gram": cosine.tolist(),
            "game_weights": weights, "game_value": value,
            "game_status": game.status, "game_fallback": game.fallback}


def _stat(select, per_query_gains):
    return (pessimistic_gain(per_query_gains) if select == "pessimistic"
            else float(np.mean(per_query_gains)))


def run_response_maxmin_round(encode, round_broadcast, client_states, dev, slices,
                              config, dev_frozen):
    """One round of the arm. Returns (SchemeResult of applied delta weights, record,
    dev_frozen). ``dev_frozen`` is None on the first round, when the broadcast adapter is
    the untrained one and its dev scores define the floor for the rest of the run.

    Candidate modes (config["candidates"]): "lattice" is the registered v1 grid (fixed
    points, vertices, the simplex lattice at every scale); "compact" is method v2: fixed
    points, vertices, the magnitude-equalised and game families, the per-client greedy
    soup, the best uniform subset and, with config["model_pick"], the best lattice point,
    the last three chosen by the response model. Selection (config["select"]): "mean"
    ranks by the worst client's mean predicted gain; "pessimistic" by the worst client's
    mean gain minus one standard error. The floor always applies to means.
    """
    K = len(slices)
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    mode = config.get("candidates", "lattice")
    select = config.get("select", "mean")
    base = measure_dev(encode, round_broadcast, dev, slices)
    current = [base[s]["mean"] for s in slices]
    if dev_frozen is None:
        dev_frozen = list(current)
    floors = ([f - float(config["floor_delta"]) for f in dev_frozen]
              if config["floor"] == "frozen" else None)

    responses = {s: [] for s in slices}
    solo_measured, solo_per_query = [], []
    for j in range(K):
        solo = measure_dev(encode, apply_delta_weights(
            round_broadcast, client_states, unit_vector(K, j)), dev, slices)
        for s in slices:
            responses[s].append((solo[s]["c"] - base[s]["c"], solo[s]["q"] - base[s]["q"]))
        solo_measured.append([solo[s]["mean"] for s in slices])
        solo_per_query.append([solo[s]["per_query"] for s in slices])

    geometry = update_geometry(round_broadcast, client_states)
    fixed = dict(config["fixed_points"])
    for j, s in enumerate(slices):
        fixed[f"solo_{s}"] = unit_vector(K, j)
    families = {}
    if mode == "compact":
        families = magnitude_candidates(geometry["norms"], geometry["game_weights"],
                                        config.get("eq_scales", []),
                                        config.get("game_scales", []))
    if mode == "lattice":
        pool = candidate_grid(K, fixed, config["lattice_step"], config["scales"])
        contenders = list(pool)
    else:
        pool = candidate_grid(K, {**fixed, **families}, 1.0, [1.0])
        contenders = list(pool)
        pool = {**pool, **{name: v for name, v in uniform_subsets(K).items() if name not in pool}}
        if config.get("model_pick"):
            for i, point in enumerate(simplex_lattice(K, config["lattice_step"])):
                for scale in config["scales"]:
                    pool.setdefault(f"lat{i}_x{scale:g}", [scale * x for x in point])
    tensors = _client_tensors(base, responses, slices, device)
    pred_pq = {name: _predicted_per_query(base, responses, dev, slices, v, device, tensors)
               for name, v in pool.items()}
    del tensors
    pred_means = {name: [float(pq.mean()) for pq in pqs] for name, pqs in pred_pq.items()}
    pred_stat = {name: [_stat(select, pq - base[s]["per_query"])
                        for pq, s in zip(pqs, slices)] for name, pqs in pred_pq.items()}

    picks = {}
    if mode == "compact":
        # greedy soup over the vertices in order of pooled measured dev gain
        order = sorted(range(K), key=lambda j: -float(np.mean(
            [solo_measured[j][i] - current[i] for i in range(K)])))
        tensors = _client_tensors(base, responses, slices, device)

        def gains_of(v):
            pqs = _predicted_per_query(base, responses, dev, slices, v, device, tensors)
            return [float((pq - base[s]["per_query"]).mean()) for pq, s in zip(pqs, slices)]
        soup_v, soup_idx = greedy_soup(order, gains_of)
        del tensors
        picks["greedy"] = {"v": soup_v, "kept": soup_idx, "order": order}
        subset_names = [n for n in pool if n.startswith("sub_")]
        best_sub = rank_candidates({n: pred_means[n] for n in subset_names}, current,
                                  None, 1, {n: pred_stat[n] for n in subset_names})
        picks["subset"] = {"v": pool[best_sub[0]], "source": best_sub[0]}
        if config.get("model_pick"):
            lat_names = [n for n in pool if n.startswith("lat")]
            best_lat = rank_candidates({n: pred_means[n] for n in lat_names}, current,
                                      None, 1, {n: pred_stat[n] for n in lat_names})
            picks["model"] = {"v": pool[best_lat[0]], "source": best_lat[0]}
        for name, pick in picks.items():
            v = [float(x) for x in pick["v"]]
            if any(max(abs(a - b) for a, b in zip(pool[n], v)) < 1e-9 for n in contenders):
                continue  # identical to a fixed contender; keep that name
            pool[name] = v
            contenders.append(name)
            tensors = _client_tensors(base, responses, slices, device)
            pred_pq[name] = _predicted_per_query(base, responses, dev, slices, v, device, tensors)
            del tensors
            pred_means[name] = [float(pq.mean()) for pq in pred_pq[name]]
            pred_stat[name] = [_stat(select, pq - base[s]["per_query"])
                               for pq, s in zip(pred_pq[name], slices)]

    scores = pred_stat if select == "pessimistic" else None
    shortlist = rank_candidates({n: pred_means[n] for n in contenders}, current, floors,
                                int(config["n_verify"]),
                                {n: scores[n] for n in contenders} if scores else None)
    verified = {}

    def verify(name, v, halvings):
        measured = measure_dev(encode, apply_delta_weights(
            round_broadcast, client_states, v), dev, slices)
        verified[name] = {
            "v": [float(x) for x in v],
            "measured": [measured[s]["mean"] for s in slices],
            "measured_gain": [measured[s]["mean"] - base[s]["mean"] for s in slices],
            "stat": [_stat(select, measured[s]["per_query"] - base[s]["per_query"])
                     for s in slices],
            "lcb_gain": [paired_lower_bound(measured[s]["per_query"] - base[s]["per_query"])
                         for s in slices],
            "halvings": halvings}
        return verified[name]["measured"]

    measured_means = {name: verify(name, pool[name], 0) for name in shortlist}
    measured_scores = ({n: verified[n]["stat"] for n in measured_means}
                       if select == "pessimistic" else None)
    chosen, min_gain = choose_applied(measured_means, current, floors, measured_scores)
    status, halvings_used = "optimal", 0
    if chosen is None and shortlist:
        v = list(pool[shortlist[0]])
        for h in range(1, int(config["halvings"]) + 1):
            v = [x / 2.0 for x in v]
            name = f"{shortlist[0]}_half{h}"
            means = {name: verify(name, v, h)}
            chosen, min_gain = choose_applied(
                means, current, floors,
                {name: verified[name]["stat"]} if select == "pessimistic" else None)
            if chosen is not None:
                status, halvings_used = "halved", h
                break
    if chosen is None:
        applied, status = [0.0] * K, "zero_step"
        message = "no verified candidate satisfied the floor; the broadcast is kept"
    else:
        applied = verified[chosen]["v"]
        message = f"applied {chosen}: measured worst-client statistic {min_gain:+.4f}"
    result = SchemeResult(applied, status=status,
                          solver_status="ok" if chosen is not None else "infeasible",
                          solver_message=message, fallback=None)
    record = result.record()
    norms = np.asarray(geometry["norms"], dtype=np.float64)
    magnitude = np.asarray(applied, dtype=np.float64) * norms
    shares = (magnitude / magnitude.sum()).tolist() if magnitude.sum() > 0 else [0.0] * K
    record.update({
        "chosen": chosen, "shortlist": shortlist, "grid_size": len(pool),
        "candidates": {name: {"v": v, "pred": pred_means[name], "stat": pred_stat[name]}
                       for name, v in pool.items()},
        "contenders": contenders, "families": list(families), "picks": picks,
        "candidate_mode": mode, "select": select,
        "verified": verified, "dev_current": current, "dev_frozen": list(dev_frozen),
        "floors": floors, "solo_measured": solo_measured, "halvings_used": halvings_used,
        "geometry": geometry, "applied_magnitude_shares": shares,
        "n_dev_queries": [len(dev[s]["qids"]) for s in slices],
        "encodes_per_client": 1 + K + len(verified), "device": device})
    return result, record, list(dev_frozen)
