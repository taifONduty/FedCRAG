"""Query-level margin replay from saved round states (registration §9.5, M1).

For every saved round of a run, load the global adapter state, re-encode each
client's evaluation queries and corpus EXACTLY as the driver's eval does, and
record per query: nDCG@10, the ranks of its relevant documents, the top-20
ranked ids, and the score margins at ranking cutoffs {1, 3, 5, 10, 20, 100}.
Also records each client's leave-one-out alignment with the rest of the
federation, computed from the persisted client states.

Fidelity gate: the mean of the replayed per-query nDCG@10 must reproduce the
aggregate the driver recorded in the run JSON for that round and client. A
replay that cannot reproduce the number it claims to explain is refused.

Registered prediction (§9.5): relevant-document exits from the top 10
concentrate in queries in the lowest pre-round margin quartile and are more
frequent in rounds where the client's leave-one-out alignment is negative.
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import torch

CUTOFFS = (1, 3, 5, 10, 20, 100)
DEPTH = 100
TOP_IDS_KEPT = 20
FIDELITY_TOL = 1e-3


# ----------------------------------------------------------------- pure parts

def margins_at_cutoffs(sorted_scores, cutoffs=CUTOFFS):
    """Score gap between rank k and rank k+1 (1-indexed) for each cutoff.

    ``sorted_scores`` is descending. A cutoff at or beyond the list end has no
    successor and yields None rather than a fabricated zero.
    """
    out = {}
    for k in cutoffs:
        if k < len(sorted_scores):
            out[str(k)] = float(sorted_scores[k - 1] - sorted_scores[k])
        else:
            out[str(k)] = None
    return out


def relevant_ranks(ranked_ids, relevant):
    """1-indexed rank of every relevant id present in the ranking."""
    return {doc: rank + 1 for rank, doc in enumerate(ranked_ids)
            if doc in relevant}


def rank_query(sims_row, cids, qid, ignore_identical_ids=True,
               depth=DEPTH + 1):
    """(ranked_ids, sorted_scores) for one query, mirroring the driver.

    The driver drops a candidate whose corpus id equals the query id (BEIR
    convention, matters on ArguAna) and evaluates the top ``depth`` of what
    remains; we keep one extra so the margin at rank 100 has a successor.
    """
    order = np.argsort(-sims_row)
    ids, scores = [], []
    for j in order:
        if ignore_identical_ids and cids[j] == qid:
            continue
        ids.append(cids[j])
        scores.append(float(sims_row[j]))
        if len(ids) >= depth:
            break
    return ids, scores


def loo_alignment_from_gram(gram, weights, k):
    """cos(dW_k, sum_{j != k} w_j dW_j) from a Gram of raw (unnormalised) deltas."""
    gram = np.asarray(gram, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).copy()
    w[k] = 0.0
    rest_sq = float(w @ gram @ w)
    self_sq = float(gram[k, k])
    if rest_sq <= 0.0 or self_sq <= 0.0:
        return None
    return float((gram[k] @ w) / np.sqrt(self_sq * rest_sq))


def effective_delta_gram(client_states, broadcast_state, sigma):
    """Gram of effective weight-space deltas sigma*(B_k A_k - B_g A_g).

    Accumulated module by module so nothing the size of a full weight matrix
    per client is ever held at once.
    """
    names = sorted({key.rsplit(".lora_", 1)[0] for key in broadcast_state
                    if key.endswith(".lora_A.weight")})
    K = len(client_states)
    gram = np.zeros((K, K), dtype=np.float64)
    for name in names:
        a_key, b_key = f"{name}.lora_A.weight", f"{name}.lora_B.weight"
        base = (broadcast_state[b_key].double()
                @ broadcast_state[a_key].double())
        deltas = [sigma * (st[b_key].double() @ st[a_key].double() - base)
                  for st in client_states]
        for i in range(K):
            for j in range(i, K):
                v = float(torch.sum(deltas[i] * deltas[j]))
                gram[i, j] = gram[j, i] = gram[i, j] + v
    return gram


# --------------------------------------------------------------- GPU replay

def replay_run(run_dir, data_root, out_path, eval_batch_size=256):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pytrec_eval
    import federated_forgetting as drv
    from fedcrag_common import doc_text, load_slice_with_train, resolve_local

    result_paths = glob.glob(os.path.join(run_dir, "federated_*.json"))
    assert len(result_paths) == 1, f"expected one result JSON in {run_dir}"
    result = json.load(open(result_paths[0]))
    args = result["args"]
    slices = result["slices"]
    rounds_recorded = sorted(
        (k for k in result["R_matrix"] if k.startswith("round_")),
        key=lambda s: int(s.split("_")[1]))
    assert len(rounds_recorded) == result["num_rounds"], "partial run JSON"

    model_path, q_prefix, d_prefix, fp16 = resolve_local(args["model"])
    model, _ = drv.new_model(args["model"], model_path, args["lora_rank"],
                             fp16, lora_mode=args["lora_mode"])
    sigma = 2.0                                    # lora_alpha = 2r in new_model
    data = {s: load_slice_with_train(s, data_root) for s in slices}

    # corpus embeddings depend on the adapter state, so they are re-encoded
    # per state; ids and qrels are fixed
    fixed = {}
    for s in slices:
        corpus = data[s]["corpus"]
        cids = list(corpus.keys())
        qids = [q for q in data[s]["eval_q"]
                if q in data[s]["eval_qrels"] and data[s]["eval_qrels"][q]]
        fixed[s] = dict(
            cids=cids, qids=qids,
            ctext=[d_prefix + doc_text(corpus[c]) for c in cids],
            qtext=[q_prefix + data[s]["eval_q"][q] for q in qids],
            qrels={q: {d: int(v) for d, v in data[s]["eval_qrels"][q].items()}
                   for q in qids},
            relevant={q: {d for d, v in data[s]["eval_qrels"][q].items() if v > 0}
                      for q in qids})

    state_files = {}
    for label in rounds_recorded:
        r = int(label.split("_")[1])
        matches = glob.glob(os.path.join(run_dir, f"states_*_round{r}.pt"))
        assert len(matches) == 1, f"no unique state file for {label}"
        state_files[label] = matches[0]

    output = {"run": os.path.basename(run_dir.rstrip("/")),
              "result_json": os.path.basename(result_paths[0]),
              "weight_by": result.get("weight_by"),
              "cutoffs": list(CUTOFFS), "rounds": {}, "loo_alignment": {},
              "fidelity": {}}

    def evaluate_state(state, label, recorded):
        drv.set_adapter_state(model, state)
        per_client = {}
        worst = 0.0
        for s in slices:
            fx = fixed[s]
            if not fx["qids"]:
                continue
            c_emb = model.encode(fx["ctext"], batch_size=eval_batch_size,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
            q_emb = model.encode(fx["qtext"], batch_size=eval_batch_size,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
            sims = q_emb @ c_emb.T
            run_dict, records = {}, {}
            for i, qid in enumerate(fx["qids"]):
                ids, scores = rank_query(sims[i], fx["cids"], qid)
                run_dict[qid] = dict(zip(ids[:DEPTH], scores[:DEPTH]))
                records[qid] = {
                    "margins": margins_at_cutoffs(scores),
                    "rel_ranks": relevant_ranks(ids[:DEPTH], fx["relevant"][qid]),
                    "top_ids": ids[:TOP_IDS_KEPT],
                }
            evaluator = pytrec_eval.RelevanceEvaluator(fx["qrels"], {"ndcg_cut.10"})
            per_q = evaluator.evaluate(run_dict)
            for qid in fx["qids"]:
                records[qid]["ndcg10"] = float(per_q[qid]["ndcg_cut_10"])
            mean_ndcg = float(np.mean([records[q]["ndcg10"] for q in fx["qids"]]))
            if recorded is not None:
                diff = abs(mean_ndcg - recorded[s]["ndcg@10"])
                worst = max(worst, diff)
                output["fidelity"][f"{label}/{s}"] = {
                    "replayed": mean_ndcg, "recorded": recorded[s]["ndcg@10"],
                    "abs_diff": diff}
            per_client[s] = records
        return per_client, worst

    # frozen: the round-1 broadcast IS the untrained adapter the driver scored
    first = torch.load(state_files[rounds_recorded[0]], map_location="cpu",
                       weights_only=True)
    frozen_records, w0 = evaluate_state(first["broadcast"], "frozen",
                                        result["R_matrix"]["frozen"])
    output["rounds"]["frozen"] = frozen_records
    worst_overall = w0
    print(f"  frozen replayed; worst |diff| vs recorded {w0:.2e}", flush=True)

    for label in rounds_recorded:
        blob = torch.load(state_files[label], map_location="cpu",
                          weights_only=True)
        recs, w = evaluate_state(blob["global"], label, result["R_matrix"][label])
        output["rounds"][label] = recs
        worst_overall = max(worst_overall, w)
        # leave-one-out alignment from the persisted client states
        client_states = [blob["clients"][s] for s in slices]
        gram = effective_delta_gram(client_states, blob["broadcast"], sigma)
        if result.get("weighted") and result.get("weight_by") == "examples":
            counts = [result["clients"][label][s]["num_examples"] for s in slices]
            weights = np.asarray(counts, dtype=np.float64) / float(sum(counts))
        else:
            weights = np.full(len(slices), 1.0 / len(slices))
        output["loo_alignment"][label] = {
            s: loo_alignment_from_gram(gram, weights, k)
            for k, s in enumerate(slices)}
        output["loo_alignment"][label]["_weights"] = [float(x) for x in weights]
        print(f"  {label} replayed; worst |diff| {w:.2e}; LOO "
              + " ".join(f"{s}={output['loo_alignment'][label][s]:+.3f}"
                         for s in slices), flush=True)

    output["fidelity"]["worst_abs_diff"] = worst_overall
    output["fidelity"]["passed"] = bool(worst_overall <= FIDELITY_TOL)
    with gzip.open(out_path, "wt") as fh:
        json.dump(output, fh)
    if not output["fidelity"]["passed"]:
        raise SystemExit(
            f"FIDELITY FAIL: replayed nDCG@10 deviates from the recorded "
            f"aggregate by {worst_overall:.3e} > {FIDELITY_TOL}")
    print(f"wrote {out_path}; fidelity worst |diff| {worst_overall:.2e} PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    a = ap.parse_args()
    replay_run(a.run_dir, a.data_root, a.out, a.eval_batch_size)


if __name__ == "__main__":
    main()
