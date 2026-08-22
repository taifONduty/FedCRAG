"""Quantify the _stack_ip cross-pairing bug on real campaign states.

For each available round t>=2 of the two decision-critical runs (maxmin r15,
n_k r15), rebuild the update Gram under the OLD (A2@A1.T, cross-paired) and
NEW (A1@A2.T, correct) formulas with broadcast = previous round's global.
Report: off-diagonal relative deviation, maxmin weights + gamma* under both.
CPU-only; safe to run beside training.
"""
import glob
import json
import re

import numpy as np
import torch
from scipy.optimize import linprog

STATE_GLOB = "results/states_contriever_seed42_weighted-{arm}_r15_round{r}.pt"
LORA_KEY = re.compile(r"(.*)\.lora_(A|B)\.weight$")


def module_pairs(state):
    mods = {}
    for k, v in state.items():
        m = LORA_KEY.match(k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.float()
    return {n: (ab["A"], ab["B"]) for n, ab in mods.items()
            if "A" in ab and "B" in ab}


def stack_ip(s1, s2, old):
    tot = 0.0
    for c1, B1, A1 in s1:
        for c2, B2, A2 in s2:
            AA = (A2 @ A1.T) if old else (A1 @ A2.T)
            tot += c1 * c2 * torch.sum((B1.T @ B2) * AA).item()
    return tot


def gram(client_states, broadcast, old):
    prev = module_pairs(broadcast)
    stacks = []
    for st in client_states:
        mp = module_pairs(st)
        stacks.append({n: [(1.0, B, A)]
                       + ([(-1.0, prev[n][1], prev[n][0])] if n in prev else [])
                       for n, (A, B) in mp.items()})
    K = len(stacks)
    G = np.zeros((K, K))
    for a in range(K):
        for b in range(a, K):
            ip = sum(stack_ip(stacks[a][n], stacks[b][n], old)
                     for n in stacks[a] if n in stacks[b])
            G[a, b] = G[b, a] = ip
    return G


def cosine(G):
    n = np.sqrt(np.clip(np.diag(G), 1e-24, None))
    return G / np.outer(n, n)


def maxmin(Gc):
    K = Gc.shape[0]
    c = np.zeros(K + 1); c[-1] = -1.0
    res = linprog(c, A_ub=np.hstack([-Gc, np.ones((K, 1))]), b_ub=np.zeros(K),
                  A_eq=[[1.0] * K + [0.0]], b_eq=[1.0],
                  bounds=[(0, 1)] * K + [(None, None)], method="highs")
    return (res.x[:K], -res.fun) if res.success else (None, None)


report = {}
for arm in ("maxmin", "examples"):
    rows = []
    for r in range(2, 16):
        cur_p = STATE_GLOB.format(arm=arm, r=r)
        prev_p = STATE_GLOB.format(arm=arm, r=r - 1)
        if not (glob.glob(cur_p) and glob.glob(prev_p)):
            continue
        cur = torch.load(cur_p, map_location="cpu", weights_only=False)
        prev = torch.load(prev_p, map_location="cpu", weights_only=False)
        clients = list(cur["clients"].values())
        names = list(cur["clients"].keys())
        broadcast = prev["global"]
        Go = gram(clients, broadcast, old=True)
        Gn = gram(clients, broadcast, old=False)
        off = ~np.eye(Go.shape[0], dtype=bool)
        dev = float(np.linalg.norm((Go - Gn)[off]) / (np.linalg.norm(Gn[off]) + 1e-30))
        diag_dev = float(np.max(np.abs(np.diag(Go) - np.diag(Gn))
                                / np.clip(np.diag(Gn), 1e-30, None)))
        wo, go_ = maxmin(cosine(Go))
        wn, gn_ = maxmin(cosine(Gn))
        wdiff = float(np.max(np.abs(wo - wn))) if wo is not None else None
        rows.append({"round": r, "offdiag_rel_dev": round(dev, 5),
                     "diag_rel_dev": round(diag_dev, 8),
                     "gamma_old": round(float(go_), 4),
                     "gamma_new": round(float(gn_), 4),
                     "max_weight_shift": round(wdiff, 5),
                     "w_old": [round(float(x), 4) for x in wo],
                     "w_new": [round(float(x), 4) for x in wn]})
    report[arm] = {"clients": names if rows else [], "rows": rows}

print(json.dumps(report, indent=1))
with open("results/gram_bug_audit.json", "w") as f:
    json.dump(report, f, indent=1)
