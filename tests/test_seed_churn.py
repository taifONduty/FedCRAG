"""seed_churn.py: per-query differences between two seeds at the same stage."""
import copy

import pytest

import seed_churn


def record(scores):
    cell = {"test": {"per_query": {"ndcg@10": scores}}}
    return {"clients": ["0"], "order": {"0": [0]},
            "references": {"0": {"0": {"scores": cell}}}}


def test_churn_is_half_the_absolute_difference_and_half_the_share_over_the_loss():
    a = record({"q1": 0.5, "q2": 0.4, "q3": 0.3, "q4": 0.2})
    assert seed_churn.churn(a, copy.deepcopy(a)) == (0.0, 0.0)
    b = record({"q1": 0.5, "q2": 0.42, "q3": 0.3, "q4": 0.195})
    mean, share = seed_churn.churn(a, b)
    assert mean == pytest.approx((0.02 + 0.005) / 4 / 2) and share == pytest.approx(1 / 4 / 2)
