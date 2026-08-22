# Mechanism diagnostics over saved per-round adapter states (federated arms).
# All inner products are computed in LoRA FACTOR SPACE — never materializing
# d_out x d_in products: <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)), r x r ops.
# (v1 materialized full products: ~340MB/client/round -> swap-death on 8GB.)
#
# Per round, from states_*.pt ({"clients": {slice: peft_sd}, "global": peft_sd})
# and the paired run JSON:
#   1. cosine Gram of client weight-space updates dW_k = B_k A_k - B_g A_g
#   2. per-client alignment with the aggregate direction (run weights + uniform)
#   3. relative bilinear factor-aggregation residual (run weights)
#   4. right-vs-left shared-subspace diagnostic (FedAS-LoRA question)
#   5. max-min Gram weights (gamma*, w*) — the conflict-aware program
#   6. erosion linkage: corr(alignment_r, ndcg-delta_{r+1}) per client
# Usage:
#   python mechanism_suite.py --states_dir DIR --run_json results/federated_X.json
import argparse
import glob
import json
import os
import re

import numpy as np
import torch


def load_round(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob["clients"], blob["global"]


def module_pairs(state):
    mods = {}
    for k, v in state.items():
        m = re.match(r"(.*)\.lora_(A|B)\.weight$", k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.float()
    return {n: (ab["A"], ab["B"]) for n, ab in mods.items()
            if "A" in ab and "B" in ab}


# ---- factor-space inner products ------------------------------------------
# A "stack" is a list of (coef, B, A) low-rank terms; it represents
# sum_i coef_i * B_i A_i. Client update at round r = [(1, B_k, A_k), (-1, B_g, A_g)].

def stack_ip_module(s1, s2):
    tot = 0.0
    for c1, B1, A1 in s1:
        for c2, B2, A2 in s2:
            # corrected 2026-08-22: (A1 @ A2.T), not (A2 @ A1.T) — the latter
            # computes the cross-paired <B1 A2, B2 A1>; see
            # tests/test_aggregation.py::test_update_gram_matches_dense and
            # results/gram_bug_audit.json (measured effect on shipped
            # weights: <= 0.0025, below seed noise)
            tot += c1 * c2 * torch.sum((B1.T @ B2) * (A1 @ A2.T)).item()
    return tot


def stacks_ip(stacks1, stacks2):
    # stacks: dict module -> stack; inner product summed over common modules
    return sum(stack_ip_module(stacks1[n], stacks2[n])
               for n in stacks1 if n in stacks2)


def client_update_stacks(client_sd, global_prev):
    mp = module_pairs(client_sd)
    out = {}
    for n, (A, B) in mp.items():
        stack = [(1.0, B, A)]
        if global_prev is not None and n in global_prev:
            Ag, Bg = global_prev[n]
            stack.append((-1.0, Bg, Ag))
        out[n] = stack
    return out


def maxmin_weights(G):
    from scipy.optimize import linprog
    K = G.shape[0]
    c = np.zeros(K + 1); c[-1] = -1.0
    A_ub = np.hstack([-G, np.ones((K, 1))])
    A_eq = np.zeros((1, K + 1)); A_eq[0, :K] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(K), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, 1)] * K + [(None, None)], method="highs")
    return (float(res.x[-1]), [float(x) for x in res.x[:K]]) if res.success else (None, None)


def orth_basis(M):
    Q, _ = np.linalg.qr(M.numpy().T)
    return Q


def subspace_residual(child, parent):
    proj = parent @ (parent.T @ child)
    return float(1.0 - (np.linalg.norm(proj) ** 2) / max(np.linalg.norm(child) ** 2, 1e-12))


def side_diagnostic(client_sds, agg_sd):
    agg = module_pairs(agg_sd)
    outs = {"A_residual": [], "B_residual": []}
    for n, (Ag, Bg) in agg.items():
        bAg, bBg = orth_basis(Ag), orth_basis(Bg.T)
        for sd in client_sds:
            mp = module_pairs(sd)
            if n not in mp:
                continue
            Ak, Bk = mp[n]
            outs["A_residual"].append(subspace_residual(orth_basis(Ak), bAg))
            outs["B_residual"].append(subspace_residual(orth_basis(Bk.T), bBg))
    return {k: round(float(np.mean(v)), 4) for k, v in outs.items() if v}


def bilinear_residual_rel(mods_list, weights):
    # || sum w_k B_k A_k - (sum w_k B_k)(sum w_k A_k) ||_F / || sum w_k B_k A_k ||_F
    num2, den2 = 0.0, 0.0
    names = set.intersection(*[set(m) for m in mods_list])
    for n in names:
        terms = [(w, m[n][1], m[n][0]) for w, m in zip(weights, mods_list)]
        Bbar = sum(w * m[n][1] for w, m in zip(weights, mods_list))
        Abar = sum(w * m[n][0] for w, m in zip(weights, mods_list))
        diff = terms + [(-1.0, Bbar, Abar)]
        num2 += stack_ip_module(diff, diff)
        den2 += stack_ip_module(terms, terms)
    return float((max(num2, 0) ** 0.5) / max(den2 ** 0.5, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True)
    ap.add_argument("--run_json", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = json.load(open(args.run_json))
    slices = run["slices"]
    tag = os.path.basename(args.run_json).replace("federated_", "").replace(".json", "")
    paths = sorted(glob.glob(os.path.join(args.states_dir, f"states_{tag}_round*.pt")),
                   key=lambda p: int(re.search(r"round(\d+)\.pt$", p).group(1)))
    if not paths:
        raise SystemExit(f"no states matching tag '{tag}' in {args.states_dir}")

    def run_weights(rnd):
        if not run.get("weighted"):
            return [1.0 / len(slices)] * len(slices)
        if run.get("weight_by") == "corpus":
            raise SystemExit("corpus weights not stored per round; use n_k/uniform runs")
        cl = run["clients"][f"round_{rnd}"]
        tot = sum(cl[s]["num_examples"] for s in slices)
        return [cl[s]["num_examples"] / tot for s in slices]

    n_rounds = len(paths)
    nd = {s: [run["R_matrix"][f"round_{r}"][s]["ndcg@10"]
              for r in range(1, n_rounds + 1)] for s in slices}

    global_prev = None  # round-1 broadcast: B=0 => zero product; A term irrelevant
    out = {"tag": tag, "slices": slices, "rounds": []}
    for i, p in enumerate(paths, start=1):
        clients, agg = load_round(p)
        stacks = [client_update_stacks(clients[s], global_prev) for s in slices]
        K = len(slices)
        Graw = np.zeros((K, K))
        for a in range(K):
            for b in range(a, K):
                Graw[a, b] = Graw[b, a] = stacks_ip(stacks[a], stacks[b])
        norms = np.sqrt(np.clip(np.diag(Graw), 1e-24, None))
        Gcos = Graw / np.outer(norms, norms)
        w_run = np.array(run_weights(i))
        w_uni = np.full(K, 1.0 / K)

        def align(w):
            dv = Graw @ w
            dn = float(np.sqrt(max(w @ Graw @ w, 1e-24)))
            return [float(dv[a] / (norms[a] * dn)) for a in range(K)]

        gamma, w_star = maxmin_weights(Gcos)
        rec = {
            "round": i,
            "cosine_gram": [[round(float(x), 4) for x in row] for row in Gcos],
            "update_norms": [round(float(n), 3) for n in norms],
            "weights_run": [round(float(w), 4) for w in w_run],
            "align_to_run_aggregate": [round(a, 4) for a in align(w_run)],
            "align_to_uniform_aggregate": [round(a, 4) for a in align(w_uni)],
            "gamma_star": None if gamma is None else round(gamma, 4),
            "maxmin_weights": w_star and [round(w, 4) for w in w_star],
            "bilinear_residual_rel_run": round(bilinear_residual_rel(
                [module_pairs(clients[s]) for s in slices], list(w_run)), 4),
            "side_diagnostic": side_diagnostic([clients[s] for s in slices], agg),
            "ndcg10": {s: round(nd[s][i - 1], 4) for s in slices},
        }
        out["rounds"].append(rec)
        global_prev = module_pairs(agg)
        print(f"[round {i:2d}] gamma*={rec['gamma_star']} "
              f"align_run={rec['align_to_run_aggregate']} "
              f"resid={rec['bilinear_residual_rel_run']} side={rec['side_diagnostic']}",
              flush=True)

    link = {}
    for j, s in enumerate(slices):
        al = [r["align_to_run_aggregate"][j] for r in out["rounds"][:-1]]
        dn = [nd[s][r + 1] - nd[s][r] for r in range(n_rounds - 1)]
        if len(al) > 2 and np.std(al) > 0 and np.std(dn) > 0:
            link[s] = round(float(np.corrcoef(al, dn)[0, 1]), 3)
    out["alignment_vs_next_delta_corr"] = link
    print("alignment->next-round-delta correlations:", link, flush=True)

    dst = args.out or f"mech_{tag}.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", dst)


if __name__ == "__main__":
    main()
