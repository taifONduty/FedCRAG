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

from aggregation_schemes import SchemeResult, apply_delta_weights
from response_aggregation import (candidate_grid, choose_applied, ndcg10,
                                  ndcg10_from_top, paired_lower_bound,
                                  rank_candidates)

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


def _predicted_means(base, responses, dev, slices, v, device="cpu", tensors=None):
    """Predicted mean dev nDCG@10 per client at weights ``v`` (first-order model)."""
    tensors = tensors or _client_tensors(base, responses, slices, device)
    means = []
    for s in slices:
        top = tensors[s].top_indices(v, TOP_K + 1)
        d = dev[s]
        means.append(float(ndcg10_from_top(top, d["cids"], d["qids"], d["qrels"],
                                           k=TOP_K).mean()))
    return means


def run_response_maxmin_round(encode, round_broadcast, client_states, dev, slices,
                              config, dev_frozen):
    """One round of the arm. Returns (SchemeResult of applied delta weights, record,
    dev_frozen). ``dev_frozen`` is None on the first round, when the broadcast adapter is
    the untrained one and its dev scores define the floor for the rest of the run."""
    K = len(slices)
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    base = measure_dev(encode, round_broadcast, dev, slices)
    current = [base[s]["mean"] for s in slices]
    if dev_frozen is None:
        dev_frozen = list(current)
    floors = ([f - float(config["floor_delta"]) for f in dev_frozen]
              if config["floor"] == "frozen" else None)

    responses = {s: [] for s in slices}
    solo_measured = []
    for j in range(K):
        solo = measure_dev(encode, apply_delta_weights(
            round_broadcast, client_states, unit_vector(K, j)), dev, slices)
        for s in slices:
            responses[s].append((solo[s]["c"] - base[s]["c"], solo[s]["q"] - base[s]["q"]))
        solo_measured.append([solo[s]["mean"] for s in slices])

    fixed = dict(config["fixed_points"])
    for j, s in enumerate(slices):
        fixed[f"solo_{s}"] = unit_vector(K, j)
    grid = candidate_grid(K, fixed, config["lattice_step"], config["scales"])
    tensors = _client_tensors(base, responses, slices, device)
    pred_means = {name: _predicted_means(base, responses, dev, slices, v, device, tensors)
                  for name, v in grid.items()}
    del tensors

    shortlist = rank_candidates(pred_means, current, floors, int(config["n_verify"]))
    verified = {}

    def verify(name, v, halvings):
        measured = measure_dev(encode, apply_delta_weights(
            round_broadcast, client_states, v), dev, slices)
        verified[name] = {
            "v": [float(x) for x in v],
            "measured": [measured[s]["mean"] for s in slices],
            "measured_gain": [measured[s]["mean"] - base[s]["mean"] for s in slices],
            "lcb_gain": [paired_lower_bound(measured[s]["per_query"] - base[s]["per_query"])
                         for s in slices],
            "halvings": halvings}
        return verified[name]["measured"]

    measured_means = {name: verify(name, grid[name], 0) for name in shortlist}
    chosen, min_gain = choose_applied(measured_means, current, floors)
    status, halvings_used = "optimal", 0
    if chosen is None and shortlist:
        v = list(grid[shortlist[0]])
        for h in range(1, int(config["halvings"]) + 1):
            v = [x / 2.0 for x in v]
            name = f"{shortlist[0]}_half{h}"
            chosen, min_gain = choose_applied({name: verify(name, v, h)}, current, floors)
            if chosen is not None:
                status, halvings_used = "halved", h
                break
    if chosen is None:
        applied, status = [0.0] * K, "zero_step"
        message = "no verified candidate satisfied the floor; the broadcast is kept"
    else:
        applied = verified[chosen]["v"]
        message = f"applied {chosen}: measured worst-client gain {min_gain:+.4f}"
    result = SchemeResult(applied, status=status,
                          solver_status="ok" if chosen is not None else "infeasible",
                          solver_message=message, fallback=None)
    record = result.record()
    record.update({
        "chosen": chosen, "shortlist": shortlist, "grid_size": len(grid),
        "candidates": {name: {"v": v, "pred": pred_means[name]} for name, v in grid.items()},
        "verified": verified, "dev_current": current, "dev_frozen": list(dev_frozen),
        "floors": floors, "solo_measured": solo_measured, "halvings_used": halvings_used,
        "n_dev_queries": [len(dev[s]["qids"]) for s in slices],
        "encodes_per_client": 1 + K + len(verified), "device": device})
    return result, record, list(dev_frozen)
