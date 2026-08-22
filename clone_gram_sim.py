"""E3 pre-registration input: CPU-only clone-federation Gram simulation.

Question: when a federation contains near-duplicate silos (3x NFCorpus
sub-silos) plus a singleton minority (ArguAna), does the max-min LP discount
the redundant block — assigning the clone TRIO less total mass than
uniform-over-clients does (0.75) — while loss-symmetric rules cannot?

Clone proxies: nfcorpus updates from the three seeds {42,123,2024} of the
capped UNIFORM r15 arm (same data distribution, independent SGD noise).
At round 1 every seed shares the IDENTICAL broadcast (LoRA B=0 init), so the
round-1 Gram is an exact common-broadcast clone federation; rounds 2+ mix
per-seed broadcasts and are reported as approximate corroboration.
Uses the CORRECTED trace identity (A1 @ A2.T). Runs beside training (nice).
"""
import json
import os
import re

import numpy as np
import torch
from scipy.optimize import linprog

SEEDS = [42, 123, 2024]
ARM = "unweighted_r15"
SINGLETON_SEED = 42
LORA_KEY = re.compile(r"(.*)\.lora_(A|B)\.weight$")


def module_pairs(state):
    mods = {}
    for k, v in state.items():
        m = LORA_KEY.match(k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.float()
    return {n: (ab["A"], ab["B"]) for n, ab in mods.items()
            if "A" in ab and "B" in ab}


def update_stack(client_sd, global_prev):
    mp = module_pairs(client_sd)
    prev = module_pairs(global_prev) if global_prev is not None else {}
    return {n: [(1.0, B, A)]
            + ([(-1.0, prev[n][1], prev[n][0])] if n in prev else [])
            for n, (A, B) in mp.items()}


def stack_ip(s1, s2):
    tot = 0.0
    for c1, B1, A1 in s1:
        for c2, B2, A2 in s2:
            tot += c1 * c2 * torch.sum((B1.T @ B2) * (A1 @ A2.T)).item()
    return tot


def gram(stacks):
    K = len(stacks)
    G = np.zeros((K, K))
    for a in range(K):
        for b in range(a, K):
            ip = sum(stack_ip(stacks[a][n], stacks[b][n])
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


def mgda_fw(G, iters=500, tol=1e-9):
    K = G.shape[0]
    w = np.ones(K) / K
    for _ in range(iters):
        t = int(np.argmin(G @ w))
        d = np.eye(K)[t] - w
        den = float(d @ G @ d)
        if den <= tol:
            break
        g = float(np.clip(-(d @ G @ w) / den, 0.0, 1.0))
        if g * np.abs(d).max() < tol:
            break
        w = w + g * d
    return w


def spath(seed, rnd):
    return f"results/states_contriever_seed{seed}_{ARM}_round{rnd}.pt"


rows = []
for rnd in range(1, 16):
    if not all(os.path.exists(spath(s, rnd)) for s in SEEDS):
        continue
    prev = {}
    ok = True
    for s in SEEDS:
        if rnd == 1:
            prev[s] = None                      # exact: shared zero-B broadcast
        elif os.path.exists(spath(s, rnd - 1)):
            prev[s] = torch.load(spath(s, rnd - 1), map_location="cpu",
                                 weights_only=False)["global"]
        else:
            ok = False
    if not ok:
        continue
    blobs = {s: torch.load(spath(s, rnd), map_location="cpu",
                           weights_only=False) for s in SEEDS}
    stacks = [update_stack(blobs[s]["clients"]["nfcorpus"], prev[s])
              for s in SEEDS]
    stacks.append(update_stack(blobs[SINGLETON_SEED]["clients"]["arguana"],
                               prev[SINGLETON_SEED]))
    G = gram(stacks)
    Gc = cosine(G)
    w_mm, gamma_star = maxmin(Gc)
    w_uni = np.ones(4) / 4
    w_mgda = mgda_fw(G)
    rows.append({
        "round": rnd,
        "exact_broadcast": rnd == 1,
        "clone_block_cosines": [round(float(Gc[i, j]), 4)
                                for i in range(3) for j in range(i + 1, 3)],
        "clone_singleton_cosines": [round(float(Gc[i, 3]), 4)
                                    for i in range(3)],
        "maxmin_w": [round(float(x), 4) for x in w_mm],
        "maxmin_clone_mass": round(float(np.sum(w_mm[:3])), 4),
        "uniform_clone_mass": 0.75,
        "gamma_maxmin": round(float(gamma_star), 4),
        "gamma_uniform": round(float(np.min(Gc @ w_uni)), 4),
        "singleton_alignment_uniform": round(float((Gc @ w_uni)[3]), 4),
        "singleton_alignment_maxmin": round(float((Gc @ np.asarray(w_mm))[3]), 4),
        "mgda_w_rawgram": [round(float(x), 4) for x in w_mgda],
    })

summary = {
    "design": "3x nfcorpus cross-seed clone proxies + arguana singleton, "
              f"arm={ARM}, corrected trace identity",
    "rows": rows,
}
print(json.dumps(summary, indent=1))
with open("results/clone_gram_sim.json", "w") as f:
    json.dump(summary, f, indent=1)
print("\nsaved results/clone_gram_sim.json")
