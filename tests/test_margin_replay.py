"""margin_replay's pure parts must mirror the driver's evaluation exactly.

The GPU replay is only as good as its agreement with the numbers it explains,
so the ranking/filtering logic is pinned here against fedcrag_common's own
evaluate_metrics on random embeddings, including the ArguAna-style
identical-id drop.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedcrag_common import evaluate_metrics  # noqa: E402
from margin_replay import (effective_delta_gram, loo_alignment_from_gram,  # noqa: E402
                           margins_at_cutoffs, rank_query, relevant_ranks)


def test_margins_are_successor_gaps_and_none_past_the_end():
    scores = [0.9, 0.8, 0.5, 0.45]
    m = margins_at_cutoffs(scores, cutoffs=(1, 2, 3, 4, 10))
    assert m["1"] == pytest.approx(0.1)
    assert m["2"] == pytest.approx(0.3)
    assert m["3"] == pytest.approx(0.05)
    assert m["4"] is None and m["10"] is None


def test_relevant_ranks_are_one_indexed_and_skip_absent():
    assert relevant_ranks(["a", "b", "c"], {"b", "z"}) == {"b": 2}


def test_rank_query_drops_identical_id_like_the_driver():
    cids = ["q7", "d1", "d2"]
    sims = np.array([0.99, 0.5, 0.4])
    ids, scores = rank_query(sims, cids, "q7")
    assert ids == ["d1", "d2"] and scores == [0.5, 0.4]
    ids2, _ = rank_query(sims, cids, "q7", ignore_identical_ids=False)
    assert ids2[0] == "q7"


def test_replay_ndcg_reproduces_evaluate_metrics_on_random_data():
    """Mean of per-query nDCG@10 from our ranking must equal the driver's
    aggregate, identical-id filtering included."""
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = np.random.default_rng(0)
    n_docs, n_q, dim = 60, 12, 8
    cids = [f"d{i}" for i in range(n_docs)]
    qids = [f"d{i}" for i in range(n_q)]      # ids collide with docs on purpose
    c = rng.normal(size=(n_docs, dim)); c /= np.linalg.norm(c, axis=1, keepdims=True)
    q = rng.normal(size=(n_q, dim)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    qrels = {qid: {cids[(i * 7 + 3) % n_docs]: 1, cids[(i * 11 + 5) % n_docs]: 2}
             for i, qid in enumerate(qids)}
    reference = evaluate_metrics(cids, c, qids, q, qrels, ["ndcg@10"])["ndcg@10"]

    sims = q @ c.T
    run = {}
    for i, qid in enumerate(qids):
        ids, scores = rank_query(sims[i], cids, qid)
        run[qid] = dict(zip(ids[:100], scores[:100]))
    per_q = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(run)
    replayed = float(np.mean([per_q[qid]["ndcg_cut_10"] for qid in qids]))
    assert replayed == pytest.approx(reference, abs=1e-12)


def test_loo_alignment_matches_direct_computation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 50))
    gram = X @ X.T
    w = np.array([0.7, 0.1, 0.1, 0.1])
    for k in range(4):
        rest = sum(w[j] * X[j] for j in range(4) if j != k)
        direct = float(X[k] @ rest / (np.linalg.norm(X[k]) * np.linalg.norm(rest)))
        assert loo_alignment_from_gram(gram, w, k) == pytest.approx(direct, rel=1e-12)


def test_effective_delta_gram_matches_dense_reconstruction():
    torch.manual_seed(0)
    names = ["enc.l0.q", "enc.l1.v"]
    def state(scale):
        st = {}
        for n in names:
            st[f"{n}.lora_A.weight"] = torch.randn(4, 12, dtype=torch.float64)
            st[f"{n}.lora_B.weight"] = scale * torch.randn(12, 4, dtype=torch.float64)
        return st
    broadcast = state(1.0)
    clients = [state(1.5), state(0.5), state(1.0)]
    sigma = 2.0
    gram = effective_delta_gram(clients, broadcast, sigma)
    dense = []
    for st in clients:
        parts = []
        for n in names:
            d = sigma * (st[f"{n}.lora_B.weight"] @ st[f"{n}.lora_A.weight"]
                         - broadcast[f"{n}.lora_B.weight"] @ broadcast[f"{n}.lora_A.weight"])
            parts.append(d.flatten())
        dense.append(torch.cat(parts).numpy())
    dense = np.stack(dense)
    np.testing.assert_allclose(gram, dense @ dense.T, rtol=1e-10)
