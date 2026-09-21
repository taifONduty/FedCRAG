"""regression.py: per-query scores and the retention measures of the continual pipeline."""
import numpy as np
import pytest

import regression

# Four candidates, one query; cosine scores are exact dot products.
CIDS = ["d1", "d2", "d3", "d4"]
C_EMB = np.array([[0.5, 0.0], [0.8, 0.0], [0.2, 0.0], [0.9, 0.0]], dtype=np.float32)
Q_EMB = np.array([[1.0, 0.0]], dtype=np.float32)


def test_per_query_scores_follow_the_ranking():
    scores = regression.per_query_scores(CIDS, C_EMB, ["q"], Q_EMB, {"q": {"d2": 1}})
    assert scores["q"]["ndcg@10"] == pytest.approx(1.0 / np.log2(3.0), rel=1e-6)
    assert scores["q"]["recall@100"] == pytest.approx(1.0)


def test_queries_without_judgements_are_left_out():
    scores = regression.per_query_scores(CIDS, C_EMB, ["q", "unjudged"],
                                         np.vstack([Q_EMB, Q_EMB]), {"q": {"d4": 1}})
    assert set(scores) == {"q"}


def test_positive_regression_ignores_gains():
    reference = {"a": 0.5, "b": 0.5}
    current = {"a": 0.6, "b": 0.3}
    assert regression.positive_regression(reference, current) == pytest.approx(0.1)
    assert regression.mean_difference(current, reference) == pytest.approx(-0.05)


def test_peak_forgetting_is_the_drop_from_each_query_peak():
    history = [{"a": 0.2, "b": 0.6}, {"a": 0.5, "b": 0.4}, {"a": 0.4, "b": 0.5}]
    assert regression.peak_forgetting(history) == pytest.approx((0.1 + 0.1) / 2)


def test_cell_summary_reports_worst_cell_and_fraction_over_threshold():
    cells = {("c0", 0, 1): 0.004, ("c0", 0, 2): 0.030, ("c1", 0, 1): 0.012}
    summary = regression.cell_summary(cells, threshold=0.010)
    assert summary["worst"] == {"cell": ["c0", 0, 2], "value": 0.030}
    assert summary["fraction_over_threshold"] == pytest.approx(2 / 3)
    assert summary["mean"] == pytest.approx((0.004 + 0.030 + 0.012) / 3)
