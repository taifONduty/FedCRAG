# Unit tests for fedcrag_common.evaluate_metrics (pytrec_eval semantics,
# mrr@k cutoff, ignore_identical_ids/ArguAna convention, recall denominator).
# Run: .venv/bin/python -m pytest tests/ -q
import numpy as np
import pytest

from fedcrag_common import evaluate_metrics


# One query "q1" against four candidates. Cosine scores are controlled exactly
# via raw dot products: ranking is q1-doc (.9) > d2 (.8) > d1 (.5) > d3 (.2).
# The corpus id "q1" simulates ArguAna's self-retrieval (query's own document).
CIDS = ["d1", "d2", "d3", "q1"]
C_EMB = np.array([[0.5, 0.0], [0.8, 0.0], [0.2, 0.0], [0.9, 0.0]], dtype=np.float32)
QIDS = ["q1"]
Q_EMB = np.array([[1.0, 0.0]], dtype=np.float32)


def ev(qrels, metrics, ignore):
    return evaluate_metrics(CIDS, C_EMB, QIDS, Q_EMB, qrels, metrics,
                            ignore_identical_ids=ignore)


def test_self_hit_excluded_by_default_promotes_relevant_doc():
    qrels = {"q1": {"d2": 1}}
    out = ev(qrels, ["ndcg@10", "recall@10", "mrr@10"], ignore=True)
    # self-doc dropped -> d2 is rank 1
    assert out["ndcg@10"] == pytest.approx(1.0)
    assert out["recall@10"] == pytest.approx(1.0)
    assert out["mrr@10"] == pytest.approx(1.0)


def test_self_hit_kept_when_disabled_demotes_relevant_doc():
    qrels = {"q1": {"d2": 1}}
    out = ev(qrels, ["ndcg@10", "mrr@10"], ignore=False)
    # self-doc (non-relevant) at rank 1, d2 at rank 2
    assert out["ndcg@10"] == pytest.approx(1.0 / np.log2(3.0), rel=1e-6)
    assert out["mrr@10"] == pytest.approx(0.5)


def test_mrr_cutoff_zeroes_gold_beyond_k():
    qrels = {"q1": {"d3": 1}}  # with self-hit dropped: d2, d1, d3 -> rank 3
    out = ev(qrels, ["mrr@2", "mrr@10"], ignore=True)
    assert out["mrr@2"] == pytest.approx(0.0)
    assert out["mrr@10"] == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_recall_denominator_is_total_relevant():
    qrels = {"q1": {"d2": 1, "d3": 1}}  # top-2 (ignore on) = d2, d1 -> 1 of 2 found
    out = ev(qrels, ["recall@2"], ignore=True)
    assert out["recall@2"] == pytest.approx(0.5)


def test_ndcg_uses_ideal_normalization_with_multiple_relevant():
    qrels = {"q1": {"d2": 1, "d3": 1}}  # ranks (ignore on): d2=1, d3=3
    out = ev(qrels, ["ndcg@10"], ignore=True)
    dcg = 1.0 + 1.0 / np.log2(4.0)          # ranks 1 and 3
    idcg = 1.0 + 1.0 / np.log2(3.0)         # ideal: ranks 1 and 2
    assert out["ndcg@10"] == pytest.approx(dcg / idcg, rel=1e-6)
