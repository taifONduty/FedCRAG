"""Per-query retrieval scores and the retention measures of the continual pipeline."""
import numpy as np
import pytrec_eval

MEASURES = {"ndcg@10": "ndcg_cut.10", "recall@100": "recall.100"}


def per_query_scores(cids, c_emb, qids, q_emb, qrels, measures=MEASURES):
    """{qid: {metric: value}} for every judged query; unjudged queries are left out."""
    judged = [i for i, q in enumerate(qids) if qrels.get(q)]
    if not judged:
        return {}
    depth = max(int(key.split(".")[1]) for key in measures.values())
    sims = q_emb[judged] @ c_emb.T
    top = np.argsort(-sims, axis=1)[:, :min(depth, sims.shape[1])]
    run = {qids[i]: {cids[j]: float(sims[row, j]) for j in top[row]}
           for row, i in enumerate(judged)}
    evaluator = pytrec_eval.RelevanceEvaluator(
        {qids[i]: qrels[qids[i]] for i in judged}, set(measures.values()))
    raw = evaluator.evaluate(run)
    return {q: {name: raw[q][key.replace(".", "_")] for name, key in measures.items()}
            for q in raw}


def positive_regression(reference, current):
    """Mean over the reference's queries of the score lost since the reference, per query,
    floored at zero: a gain on one query cannot offset a loss on another."""
    return float(np.mean([max(reference[q] - current[q], 0.0) for q in reference]))


def mean_difference(current, reference):
    """Signed mean of current minus reference over the reference's queries: acquisition
    against the frozen backbone, or backward transfer against an acquisition reference."""
    return float(np.mean([current[q] - reference[q] for q in reference]))


def peak_forgetting(history):
    """Mean over queries of the drop from each query's best score in ``history`` to its
    final score."""
    final = history[-1]
    return float(np.mean([max(h[q] for h in history) - final[q] for q in final]))


def cell_summary(cells, threshold):
    """Mean, worst cell and the fraction of cells at or above ``threshold``."""
    worst = max(cells, key=cells.get)
    values = list(cells.values())
    return {"mean": float(np.mean(values)),
            "worst": {"cell": list(worst), "value": cells[worst]},
            "fraction_over_threshold": float(np.mean([v >= threshold for v in values])),
            "threshold": threshold}
